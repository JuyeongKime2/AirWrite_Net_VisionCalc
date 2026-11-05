"""화면 그리기: 얼굴 자동 모자이크, 손 골격, 쓰는 중인 글씨, 확인용 미리보기 이미지 (cv2만 사용)."""

import cv2
import numpy as np

from .strokes import normalize, render_image

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
GESTURE_COLOR = {"draw": (60, 200, 60), "hover": (255, 170, 0), "confirm": (0, 200, 255),
                 "fist": (60, 60, 230), "other": (180, 180, 180), "none": (120, 120, 120)}
GESTURE_TEXT = {"draw": "WRITE", "hover": "MOVE", "confirm": "DONE", "fist": "ERASE", "other": "-", "none": "NO HAND"}

_face_cascade = None


# ============================================================
# 얼굴 모자이크 (손만 추적하므로 OpenCV 얼굴 검출 사용)
# ============================================================
def face_boxes(frame_bgr):
    global _face_cascade
    if _face_cascade is None:
        try:
            _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except AttributeError:
            _face_cascade = False
    if _face_cascade is False or _face_cascade.empty():
        return []
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (w // 2, h // 2))          # 속도를 위해 절반 크기에서 검출
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    boxes = []
    for x, y, fw, fh in _face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(24, 24)):
        x, y, fw, fh = x * 2, y * 2, fw * 2, fh * 2
        px, py = int(fw * 0.3), int(fh * 0.4)
        boxes.append((max(0, x - px), max(0, y - py), min(w, x + fw + px), min(h, y + fh + py)))
    return boxes


def mosaic(frame_bgr, boxes, blocks=8):
    for x0, y0, x1, y1 in boxes:
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        small = cv2.resize(roi, (blocks, max(1, int(blocks * roi.shape[0] / roi.shape[1]))))
        frame_bgr[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return frame_bgr


class FaceMosaic:
    """얼굴 검출은 몇 프레임마다 하고, 사이 프레임은 마지막 위치를 재사용 (속도)."""

    def __init__(self, every=3, keep=8):
        self.every, self.keep = every, keep
        self.count, self.boxes, self.age = 0, [], 0

    def __call__(self, frame_bgr):
        if self.count % self.every == 0:
            found = face_boxes(frame_bgr)
            if found:
                self.boxes, self.age = found, 0
            else:
                self.age += self.every
                if self.age > self.keep:
                    self.boxes = []
        self.count += 1
        return mosaic(frame_bgr, self.boxes)


# ============================================================
# 실시간 화면 그리기
# ============================================================
def draw_hand(img, hand, gesture):
    if hand is None:
        return img
    h, w = img.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in hand]
    color = GESTURE_COLOR.get(gesture, (200, 200, 200))
    for a, b in HAND_EDGES:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    cv2.circle(img, pts[8], 7, color, -1, cv2.LINE_AA)       # 검지 끝 = 펜
    return img


def draw_strokes(img, strokes, aspect, color=(40, 230, 255)):
    h, w = img.shape[:2]
    for s in strokes:
        if len(s) == 0:
            continue
        px = np.column_stack([s[:, -2] / aspect * w, s[:, -1] * h]).astype(np.int32)
        if len(px) == 1:
            cv2.circle(img, tuple(px[0]), 5, color, -1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [px.reshape(-1, 1, 2)], False, color, 5, cv2.LINE_AA)
    return img


def draw_status_bar(img, gesture, text_right=""):
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 40), (25, 25, 25), -1)
    color = GESTURE_COLOR.get(gesture, (200, 200, 200))
    cv2.putText(img, GESTURE_TEXT.get(gesture, gesture), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if text_right:
        (tw, _), _ = cv2.getTextSize(text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(img, text_right, (w - tw - 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ============================================================
# 확인용 미리보기: [쓴 궤적(획 순서별 색, 시작점 표시)] + [CNN이 보는 28x28]
# ============================================================
STROKE_COLORS = [(214, 112, 47), (60, 160, 60), (40, 40, 220), (180, 80, 180), (0, 160, 220)]


def trajectory_preview(strokes, size=160):
    canvas = np.full((size, size, 3), 250, np.uint8)
    norm = normalize(strokes)
    half = size * 0.4
    for i, s in enumerate(norm):
        color = STROKE_COLORS[i % len(STROKE_COLORS)]
        px = np.round(s * half + size / 2).astype(np.int32)
        if len(px) == 1 or np.linalg.norm(np.diff(s, axis=0), axis=1).sum() < 0.08:
            cv2.circle(canvas, tuple(px.mean(axis=0).astype(int)), 5, color, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [px.reshape(-1, 1, 2)], False, color, 3, cv2.LINE_AA)
            cv2.circle(canvas, tuple(px[0]), 6, color, 2, cv2.LINE_AA)               # 시작점 = 빈 원
            cv2.putText(canvas, str(i + 1), (int(px[0][0]) + 7, int(px[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cnn = (render_image(strokes) * 255).astype(np.uint8)
    cnn = cv2.cvtColor(cv2.resize(cnn, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    divider = np.full((size, 4, 3), 200, np.uint8)
    return cv2.hconcat([canvas, divider, cnn])
