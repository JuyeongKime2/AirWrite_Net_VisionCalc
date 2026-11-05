"""조기 예측용 데이터 처리, 기준선(비딥러닝), 평가 지표 (numpy만 사용).

조기 예측 = 글자를 다 쓰기 전에
    ① 무슨 기호인지 (분류)
    ② 나머지가 어떻게 그려질지 (궤적 좌표)
를 예측하고, 실제로 끝까지 그린 궤적과 비교한다.

용어
    쓴 비율 f   지금까지 그린 잉크 길이 / 완성된 글자의 전체 잉크 길이 (획 사이 이동 제외)
    앞부분      f만큼 그린 부분 (prefix)
    abs 좌표    앞부분의 첫 점을 원점으로, 화면 세로 ABS_SCALE을 1로 하는 좌표.
                글자 크기 정보가 남아 있어서 '얼마나 더 그릴지'까지 예측할 수 있다.
"""

import numpy as np

from .config import ABS_SCALE, COMPLETE_POINTS, EARLY_POINTS, FRAME_HEIGHT

INPUT_DIM = 8


# ============================================================
# 기본 연산
# ============================================================
def xy_strokes(strokes):
    """[t, x, y] 또는 [x, y] 획 목록 -> [x, y] 획 목록."""
    return [np.asarray(s, np.float32)[:, -2:] for s in strokes if len(s)]


def _seg_lengths(s):
    return np.linalg.norm(np.diff(s, axis=0), axis=1) if len(s) > 1 else np.zeros(0, np.float32)


def ink_length(strokes):
    return float(sum(_seg_lengths(s).sum() for s in xy_strokes(strokes)))


def symbol_size(strokes):
    pts = np.concatenate(xy_strokes(strokes))
    return max(float((pts.max(0) - pts.min(0)).max()), 1e-4)


def cut_prefix(strokes, frac):
    """잉크 길이 기준으로 앞에서 frac만큼만 남긴다. 마지막 점은 보간해서 정확히 frac에서 끊는다."""
    xy = xy_strokes(strokes)
    if frac >= 1.0:
        return xy
    total = sum(_seg_lengths(s).sum() for s in xy)
    if total <= 1e-9:
        return [xy[0][:1]]
    target, acc, out = frac * total, 0.0, []
    for s in xy:
        seg = _seg_lengths(s)
        length = float(seg.sum())
        if acc + length <= target:
            out.append(s)
            acc += length
            continue
        remain = target - acc
        if remain <= 1e-9:
            break
        cum = np.r_[0.0, np.cumsum(seg)]
        k = int(np.searchsorted(cum, remain))           # cum[k-1] < remain <= cum[k]
        ratio = (remain - cum[k - 1]) / max(seg[k - 1], 1e-9)
        end = s[k - 1] + ratio * (s[k] - s[k - 1])
        out.append(np.vstack([s[:k], end[None]]).astype(np.float32))
        break
    return out or [xy[0][:1]]


def resample_path(strokes, n):
    """여러 획을 잉크 길이 비율대로 n개 점으로 다시 뽑는다 (정규화 없이 원래 좌표 그대로).
    반환: 점 (n, 2), 획 시작 표시 (n,)"""
    xy = xy_strokes(strokes)
    lengths = np.array([max(_seg_lengths(s).sum(), 1e-4) for s in xy])
    alloc = np.maximum(1, np.round(n * lengths / lengths.sum()).astype(int))
    while alloc.sum() > n:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n:
        alloc[np.argmax(lengths / alloc)] += 1
    pts, starts = [], []
    for s, k in zip(xy, alloc):
        seg = _seg_lengths(s)
        if len(s) == 1 or seg.sum() <= 1e-9:
            p = np.repeat(s.mean(axis=0, keepdims=True), k, axis=0)
        else:
            cum = np.r_[0.0, np.cumsum(seg)]
            target = np.linspace(0, cum[-1], k) if k > 1 else np.array([cum[-1] / 2])
            p = np.stack([np.interp(target, cum, s[:, 0]), np.interp(target, cum, s[:, 1])], axis=1)
        pts.append(p)
        starts.append(np.r_[1.0, np.zeros(k - 1)])
    return np.concatenate(pts).astype(np.float32), np.concatenate(starts).astype(np.float32)


def split_strokes(points, starts):
    """점 목록 + 획 시작 표시 -> 획 목록."""
    idx = np.flatnonzero(np.asarray(starts) > 0.5).tolist()
    if not idx or idx[0] != 0:
        idx = [0] + idx
    bounds = idx + [len(points)]
    return [points[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


# ============================================================
# 모델 입력 / 목표
# ============================================================
def origin_of(prefix):
    return xy_strokes(prefix)[0][0].copy()


def early_features(prefix, n=EARLY_POINTS):
    """(n, 8): 앞부분 기준 정규화 좌표 2, abs 좌표 2, 진행 방향 2, 획 시작 표시 1, 몇 번째 획인지 1."""
    pts, starts = resample_path(prefix, n)
    allp = np.concatenate(xy_strokes(prefix))
    lo, hi = allp.min(0), allp.max(0)
    norm = (pts - (lo + hi) / 2) / max(float((hi - lo).max()) / 2, 1e-3)
    absf = (pts - origin_of(prefix)) / ABS_SCALE
    direction = np.zeros_like(pts)
    step = pts[1:] - pts[:-1]
    same = starts[1:] == 0
    direction[:-1][same] = (step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-6))[same]
    stroke_idx = (np.cumsum(starts) - 1) / 3.0
    return np.concatenate([norm, absf, direction, starts[:, None], stroke_idx[:, None]], axis=1).astype(np.float32)


def completion_target(full, origin, m=COMPLETE_POINTS):
    """완성된 글자를 m개 점으로 뽑아 abs 좌표로. (m, 3): x, y, 획 시작 표시."""
    pts, starts = resample_path(full, m)
    return np.concatenate([(pts - origin) / ABS_SCALE, starts[:, None]], axis=1).astype(np.float32)


def decode_completion(pred, origin):
    """모델 출력 (m, 3) -> 화면 좌표 획 목록."""
    pred = np.asarray(pred, np.float32)
    pts = pred[:, :2] * ABS_SCALE + origin
    return split_strokes(pts, pred[:, 2])


# ============================================================
# 학습용 증강 (크기 정보를 유지하는 버전)
# ============================================================
def augment_keep_scale(strokes, rng, rot_deg=12.0, shear=0.15, aspect=(0.85, 1.15), size=(0.8, 1.25), jitter=0.012):
    xy = xy_strokes(strokes)
    allp = np.concatenate(xy)
    center = (allp.min(0) + allp.max(0)) / 2
    half = max(float((allp.max(0) - allp.min(0)).max()) / 2, 1e-3)
    theta = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    mat = rot @ np.array([[1, rng.uniform(-shear, shear)], [0, 1]], np.float32) \
        @ np.diag([rng.uniform(*aspect), 1.0]).astype(np.float32) * rng.uniform(*size)
    out = []
    for s in xy:
        s = (s - center) @ mat.T + center
        s = s + rng.normal(0, jitter * half, s.shape)
        out.append(s.astype(np.float32))
    return out


def to_screen(strokes, rng, center=(0.6, 0.5), half_size=(0.05, 0.18)):
    """합성 연산자(대략 -1~1 범위)를 실제 화면 크기로 바꾼다."""
    scale = rng.uniform(*half_size)
    c = np.asarray(center, np.float32) + rng.uniform(-0.1, 0.1, 2)
    return [(np.asarray(s, np.float32)[:, -2:] * scale + c).astype(np.float32) for s in strokes]


def sample_fraction(rng):
    return 1.0 if rng.random() < 0.25 else float(rng.uniform(0.15, 1.0))


# ============================================================
# 기준선 1: 직선 연장 (마지막 방향으로 남은 길이만큼 쭉)
# ============================================================
def linear_extrapolation(prefix, frac, m=COMPLETE_POINTS):
    """쓴 비율 frac을 알고 있다고 가정(유리한 조건)하고, 마지막 진행 방향으로 남은 잉크 길이만큼 직선을 긋는다."""
    xy = xy_strokes(prefix)
    last = xy[-1]
    tail = last[-6:] if len(last) >= 2 else np.concatenate(xy)[-6:]
    d = tail[-1] - tail[0] if len(tail) >= 2 else np.array([1.0, 0.0], np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-6)
    remaining = ink_length(prefix) * (1.0 / max(frac, 1e-3) - 1.0)
    end = last[-1] + d * remaining
    ext = np.vstack([last, end[None]]).astype(np.float32) if remaining > 0 else last
    full = xy[:-1] + [ext]
    return completion_target(full, origin_of(prefix), m)


# ============================================================
# 기준선 2: kNN 템플릿 매칭 (학습 글씨 중 앞부분이 가장 비슷한 것들의 나머지를 빌려옴)
# ============================================================
class KNNTemplate:
    """분류(다수결)와 궤적(이웃 평균) 모두 하는 고전적 방법. 딥러닝 비교 기준."""

    def __init__(self, strokes_list, labels, k=5, n=EARLY_POINTS, m=COMPLETE_POINTS):
        self.full = [xy_strokes(s) for s in strokes_list]
        self.labels = np.asarray(labels)
        self.k, self.n, self.m = k, n, m
        self._bank = {}

    def _bank_for(self, frac):
        key = round(float(frac), 3)
        if key not in self._bank:
            feats, targets, counts = [], [], []
            for s in self.full:
                p = cut_prefix(s, frac)
                pts, _ = resample_path(p, self.n)
                feats.append(((pts - origin_of(p)) / ABS_SCALE).ravel())
                targets.append(completion_target(s, origin_of(p), self.m))
                counts.append(len(p))
            self._bank[key] = (np.stack(feats), np.stack(targets), np.asarray(counts))
        return self._bank[key]

    def predict(self, prefix, frac, num_classes):
        feats, targets, counts = self._bank_for(frac)
        pts, _ = resample_path(prefix, self.n)
        q = ((pts - origin_of(prefix)) / ABS_SCALE).ravel()
        dist = np.sqrt(((feats - q) ** 2).mean(axis=1)) + 0.5 * np.abs(counts - len(prefix))
        nn = np.argsort(dist)[: self.k]
        w = 1.0 / (dist[nn] + 1e-3)
        probs = np.zeros(num_classes)
        np.add.at(probs, self.labels[nn], w)
        probs /= probs.sum()
        top = self.labels[nn][self.labels[nn] == probs.argmax()]
        same = nn[np.isin(self.labels[nn], top)]
        completion = targets[same].mean(axis=0)
        return probs, completion


# ============================================================
# 평가 지표
# ============================================================
def completion_errors(pred, true, frac, size):
    """예측 완성 궤적 vs 실제 완성 궤적. 모두 '글자 크기 대비 비율'과 '화면 픽셀'로.

    ade  남은 부분 평균 거리 (Average Displacement Error)
    fde  마지막 점 거리 (Final Displacement Error) = 글자를 어디서 끝낼지
    chamfer  전체 모양 거리 (점 순서와 무관, 서로 가장 가까운 점끼리 평균)
    """
    p, t = np.asarray(pred)[:, :2], np.asarray(true)[:, :2]
    m = len(t)
    rest = np.arange(m) / (m - 1) > frac
    if not rest.any():
        rest[-1] = True
    d = np.linalg.norm(p - t, axis=1) * ABS_SCALE
    pair = np.linalg.norm(p[:, None] - t[None], axis=2) * ABS_SCALE
    chamfer = (pair.min(axis=1).mean() + pair.min(axis=0).mean()) / 2
    ade, fde = float(d[rest].mean()), float(d[-1])
    return {
        "ade_rel": ade / size, "fde_rel": fde / size, "chamfer_rel": float(chamfer) / size,
        "ade_px": ade * FRAME_HEIGHT, "fde_px": fde * FRAME_HEIGHT, "chamfer_px": float(chamfer) * FRAME_HEIGHT,
        "stroke_count_ok": int(round(np.asarray(pred)[:, 2].clip(0, 1).round().sum())) == int(np.asarray(true)[:, 2].sum()),
    }


def merge_prefix_and_rest(prefix, pred, frac):
    """실제로 그린 앞부분 + 모델이 예측한 나머지 -> 완성 글자 (화면 좌표 획 목록).
    예측 완성 궤적 중 '아직 안 그린 부분'(진행률 > frac)만 이어 붙인다."""
    xy = [s.copy() for s in xy_strokes(prefix)]
    pred = np.asarray(pred, np.float32)
    m = len(pred)
    rest = np.arange(m) / (m - 1) > frac
    if frac >= 1.0 or not rest.any():
        return xy
    pts = pred[rest, :2] * ABS_SCALE + origin_of(prefix)
    starts = pred[rest, 2]
    if starts[0] <= 0.5:                                    # 마지막 획을 이어서 그림
        cut = np.flatnonzero(starts > 0.5)
        first_end = cut[0] if len(cut) else len(pts)
        xy[-1] = np.vstack([xy[-1], pts[:first_end]])
        pts, starts = pts[first_end:], starts[first_end:]
    if len(pts):
        xy += split_strokes(pts, np.r_[1.0, starts[1:]])
    return xy


def rest_strokes(prefix, pred, frac):
    """예측 완성 궤적 중 '아직 안 그린 부분'만 (그림용). 마지막 획을 이어 그리면 펜 끝에서 시작."""
    pred = np.asarray(pred, np.float32)
    m = len(pred)
    rest = np.arange(m) / (m - 1) > frac
    if frac >= 1.0 or not rest.any():
        return []
    pts = pred[rest, :2] * ABS_SCALE + origin_of(prefix)
    starts = pred[rest, 2]
    pieces = []
    if starts[0] <= 0.5:
        cut = np.flatnonzero(starts > 0.5)
        first_end = cut[0] if len(cut) else len(pts)
        pieces.append(np.vstack([xy_strokes(prefix)[-1][-1:], pts[:first_end]]))
        pts, starts = pts[first_end:], starts[first_end:]
    if len(pts):
        pieces += split_strokes(pts, np.r_[1.0, starts[1:]])
    return pieces


def earliest_stable(correct_by_frac, fractions):
    """이 비율 이후로 쭉 맞힌 가장 이른 쓴 비율. 끝까지 못 맞히면 nan."""
    earliest = float("nan")
    for f, ok in sorted(zip(fractions, correct_by_frac), reverse=True):
        if not ok:
            break
        earliest = f
    return earliest


INPUT_FEATURES = ["norm_x", "norm_y", "abs_x", "abs_y", "dir_x", "dir_y", "stroke_start", "stroke_index"]
assert len(INPUT_FEATURES) == INPUT_DIM