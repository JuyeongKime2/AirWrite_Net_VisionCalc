"""허공 글씨 획 처리 (numpy + cv2, mediapipe/torch 불필요).

획(stroke) 표현: (n, 3) 배열 [t, x, y]
    x는 화면 가로/세로 비를 곱한 값(가로세로 같은 축척), y는 아래로 갈수록 커짐.
한 글자 = 획 리스트.

여기서 하는 일
    AirWriter         프레임마다 손 관절 -> 손 모양 판별 -> 획 쌓기 -> '완료' 시 글자 하나 반환
    render_image      획 -> 28x28 이미지 (CNN 입력, MNIST와 같은 형식)
    seq_features      획 -> 64x6 시퀀스 (GRU 입력: 위치, 진행 방향, 획 시작 표시, 진행률)
    augment           학습용 변형 (회전, 기울임, 떨림 ...)
    synth_operator    합성 연산자 (MNIST에 연산자가 없어서 보충)
"""

import cv2
import numpy as np

from .config import (
    CLASSES,
    DOT_LENGTH,
    FIST_HOLD_SECONDS,
    GESTURE_STABLE_FRAMES,
    IMAGE_INNER,
    IMAGE_SIZE,
    RENDER_SCALE,
    SEQ_POINTS,
    STROKE_THICKNESS,
    TIP_SMOOTHING,
)
from .hand_tracking import INDEX_TIP, classify_gesture, finger_states

FEATURE_VERSION = 1


# ============================================================
# 실시간 획 쌓기 (상태 기계)
# ============================================================
class AirWriter:
    """update()가 돌려주는 이벤트
        None
        ("confirm", strokes)   손바닥을 펴서 글자 완료
        ("clear", had_strokes) 주먹을 잠깐 유지. had_strokes=False면 '마지막 입력 지우기' 의미
    """

    def __init__(self):
        self.strokes = []
        self.current = None
        self.gesture = "none"
        self._candidate, self._candidate_count = "none", 0
        self._tip = None
        self._confirm_latched = False
        self._fist_start, self._fist_fired = None, False

    # ---------------- 내부 ----------------
    def _end_stroke(self):
        if self.current:
            # 손 모양이 바뀌는 동안(안정화 프레임) 찍힌 꼬리를 잘라낸다
            trim = max(0, GESTURE_STABLE_FRAMES - 1)
            points = self.current[:-trim] if len(self.current) > trim + 1 else self.current[:1]
            self.strokes.append(np.asarray(points, np.float32))
        self.current = None

    def _stable_gesture(self, raw):
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = raw, 1
        if self._candidate_count >= GESTURE_STABLE_FRAMES:
            self.gesture = self._candidate
        return self.gesture

    # ---------------- 공개 ----------------
    def update(self, t, hand, aspect):
        if hand is None:
            self._end_stroke()
            self.gesture, self._candidate, self._candidate_count = "none", "none", 0
            self._tip = None
            return None

        gesture = self._stable_gesture(classify_gesture(finger_states(hand, aspect)))
        tip = np.array([hand[INDEX_TIP, 0] * aspect, hand[INDEX_TIP, 1]], np.float32)
        self._tip = tip if self._tip is None else TIP_SMOOTHING * tip + (1 - TIP_SMOOTHING) * self._tip

        if gesture == "draw":
            if self.current is None:
                self.current = []
            self.current.append((t, float(self._tip[0]), float(self._tip[1])))
        else:
            self._end_stroke()

        event = None
        if gesture == "confirm":
            if not self._confirm_latched:
                self._confirm_latched = True
                if self.strokes:
                    event = ("confirm", self.strokes)
                    self.strokes = []
        else:
            self._confirm_latched = False

        if gesture == "fist":
            if self._fist_start is None:
                self._fist_start = t
            elif not self._fist_fired and t - self._fist_start >= FIST_HOLD_SECONDS:
                self._fist_fired = True
                event = ("clear", bool(self.strokes))
                self.strokes = []
        else:
            self._fist_start, self._fist_fired = None, False
        return event

    def force_confirm(self):
        """키보드(스페이스)로 완료할 때."""
        self._end_stroke()
        strokes, self.strokes = self.strokes, []
        return strokes

    def clear(self):
        self.strokes, self.current = [], None

    def visible_strokes(self):
        """화면에 그릴 획 (쓰는 중인 획 포함)."""
        out = list(self.strokes)
        if self.current:
            out.append(np.asarray(self.current, np.float32))
        return out

    @property
    def tip(self):
        return self._tip


# ============================================================
# 저장 / 불러오기
# ============================================================
def save_sample(path, strokes, label, aspect):
    points = np.concatenate(strokes, axis=0).astype(np.float32)
    stroke_id = np.concatenate([np.full(len(s), i, np.int16) for i, s in enumerate(strokes)])
    np.savez_compressed(path, points=points, stroke_id=stroke_id, label=np.array(label), aspect=np.float32(aspect))


def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        points, stroke_id = data["points"].astype(np.float32), data["stroke_id"]
        label = str(data["label"])
    if label not in CLASSES:
        raise ValueError(f"알 수 없는 라벨 {label}")
    strokes = [points[stroke_id == i] for i in np.unique(stroke_id)]
    if not strokes or sum(len(s) for s in strokes) < 2:
        raise ValueError("점이 너무 적음")
    return strokes, label


# ============================================================
# 정규화 / 변환
# ============================================================
def _xy(strokes):
    return [np.asarray(s, np.float32)[:, -2:] for s in strokes if len(s)]


def normalize(strokes):
    """글자 전체의 가운데를 원점, 긴 변 절반을 1로. 가로세로 비율은 유지 ('−'은 납작한 그대로)."""
    xy = _xy(strokes)
    allp = np.concatenate(xy, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    center, scale = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 1e-6)
    return [(s - center) / scale for s in xy]


def _length(s):
    return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0


def render_image(strokes, size=IMAGE_SIZE, inner=IMAGE_INNER):
    """MNIST 형식 이미지 (검은 배경, 흰 글씨, 0~1, 무게중심을 가운데로)."""
    norm = normalize(strokes)
    big, half = size * RENDER_SCALE, inner * RENDER_SCALE / 2
    canvas = np.zeros((big, big), np.uint8)
    for s in norm:
        px = np.round(s * half + big / 2).astype(np.int32)
        if _length(s) < DOT_LENGTH:
            c = px.mean(axis=0).astype(int)
            cv2.circle(canvas, (int(c[0]), int(c[1])), STROKE_THICKNESS // 2 + 1, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [px.reshape(-1, 1, 2)], False, 255, STROKE_THICKNESS, cv2.LINE_AA)
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    total = img.sum()
    if total > 0:  # MNIST처럼 무게중심을 정가운데로
        ys, xs = np.mgrid[0:size, 0:size]
        cy, cx = (ys * img).sum() / total, (xs * img).sum() / total
        shift = np.float32([[1, 0, (size - 1) / 2 - cx], [0, 1, (size - 1) / 2 - cy]])
        img = cv2.warpAffine(img, shift, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.clip(img, 0, 1)


def resample_strokes(strokes, n=SEQ_POINTS):
    """획 길이에 비례해 점을 나눠 배정하고, 각 획을 길이 기준으로 균등하게 다시 뽑는다.
    -> 천천히 쓰든 빨리 쓰든 같은 결과."""
    norm = normalize(strokes)
    lengths = np.array([max(_length(s), 0.02) for s in norm])
    alloc = np.maximum(1, np.round(n * lengths / lengths.sum()).astype(int))
    while alloc.sum() > n:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n:
        alloc[np.argmax(lengths / alloc)] += 1

    points, starts = [], []
    for s, k in zip(norm, alloc):
        if len(s) == 1 or _length(s) == 0:
            pts = np.repeat(s.mean(axis=0, keepdims=True), k, axis=0)
        else:
            seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
            cum = np.r_[0, np.cumsum(seg)]
            target = np.linspace(0, cum[-1], k) if k > 1 else np.array([cum[-1] / 2])
            pts = np.stack([np.interp(target, cum, s[:, 0]), np.interp(target, cum, s[:, 1])], axis=1)
        points.append(pts)
        starts.append(np.r_[1.0, np.zeros(k - 1)])
    return np.concatenate(points).astype(np.float32), np.concatenate(starts).astype(np.float32)


def seq_features(strokes, n=SEQ_POINTS):
    """(n, 6): x, y, 진행 방향 단위벡터(dx, dy), 획 시작 표시, 전체 진행률."""
    pts, starts = resample_strokes(strokes, n)
    direction = np.zeros_like(pts)
    nxt = pts[1:] - pts[:-1]
    same_stroke = starts[1:] == 0
    norm = np.linalg.norm(nxt, axis=1, keepdims=True)
    direction[:-1][same_stroke] = (nxt / np.maximum(norm, 1e-6))[same_stroke]
    progress = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return np.concatenate([pts, direction, starts[:, None], progress], axis=1).astype(np.float32)


# ============================================================
# 증강 (학습 때만)
# ============================================================
def augment(strokes, rng, rot_deg=12.0, shear=0.15, aspect=(0.85, 1.15), jitter=0.012, stroke_shift=0.04):
    """회전은 ±12도로 제한 (+와 ×가 45도 회전 관계라 크게 돌리면 서로 바뀜)."""
    norm = normalize(strokes)
    theta = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    sh = np.array([[1, rng.uniform(-shear, shear)], [0, 1]], np.float32)
    sc = np.diag([rng.uniform(*aspect), 1.0]).astype(np.float32)
    mat = rot @ sh @ sc
    out = []
    for s in norm:
        s = s @ mat.T
        if len(norm) > 1:
            s = s + rng.uniform(-stroke_shift, stroke_shift, 2)       # 획끼리 위치가 조금씩 어긋나게
        s = s + rng.normal(0, jitter, s.shape)
        if len(s) > 6 and rng.random() < 0.3:                         # 획 시작/끝 몇 점 잘림
            a, b = rng.integers(0, 3), rng.integers(0, 3)
            s = s[a:len(s) - b] if len(s) - a - b >= 2 else s
        out.append(s.astype(np.float32))
    return out


# ============================================================
# 합성 연산자 (학습 fold에만 섞음)
# ============================================================
def _line(p0, p1, rng, n=None):
    n = n or int(rng.integers(15, 36))
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) * np.asarray(p0, np.float32) + t * np.asarray(p1, np.float32)
    wobble = np.cumsum(rng.normal(0, 0.012, (n, 2)), axis=0)               # 손 떨림처럼 부드러운 흔들림
    wobble -= np.linspace(0, 1, n)[:, None] * wobble[-1]
    return (pts + wobble + rng.normal(0, 0.006, (n, 2))).astype(np.float32)


def _dot(center, rng):
    n = int(rng.integers(2, 6))
    return (np.asarray(center, np.float32) + rng.normal(0, 0.015, (n, 2))).astype(np.float32)


def synth_operator(name, rng):
    j = lambda: rng.uniform(-0.12, 0.12)  # noqa: E731
    if name == "plus":
        h = _line((-1 + j(), j()), (1 + j(), j()), rng)
        v = _line((j(), -1 + j()), (j(), 1 + j()), rng)
        strokes = [h, v] if rng.random() < 0.8 else [v, h]
    elif name == "minus":
        strokes = [_line((-1, j() * 0.5), (1, j() * 0.5), rng)]
    elif name == "times":
        strokes = [_line((-1 + j(), -1 + j()), (1 + j(), 1 + j()), rng),
                   _line((1 + j(), -1 + j()), (-1 + j(), 1 + j()), rng)]
    elif name == "divide":
        gap = rng.uniform(0.45, 0.8)
        line = _line((-1, j() * 0.3), (1, j() * 0.3), rng)
        dots = [_dot((j(), -gap), rng), _dot((j(), gap), rng)]
        strokes = [line] + dots if rng.random() < 0.7 else [dots[0], line, dots[1]]
    elif name == "equals":
        gap = rng.uniform(0.25, 0.55)
        strokes = [_line((-1 + j(), -gap), (1 + j(), -gap + j() * 0.3), rng),
                   _line((-1 + j(), gap), (1 + j(), gap + j() * 0.3), rng)]
    else:
        raise ValueError(name)
    strokes = [s[::-1] if rng.random() < 0.1 else s for s in strokes]      # 가끔 반대 방향으로 긋기
    t0 = 0.0
    out = []
    for s in strokes:                                                       # [t, x, y] 형식으로
        t = t0 + np.arange(len(s)) / 30.0
        out.append(np.column_stack([t, s]).astype(np.float32))
        t0 = t[-1] + 0.3
    return out
