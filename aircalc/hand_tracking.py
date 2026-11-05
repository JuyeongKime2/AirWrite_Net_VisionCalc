"""손 추적(MediaPipe Hands)과 손 모양 판별(규칙), 카메라 열기.

MediaPipe Hands는 사전학습된 딥러닝 모델(손 검출 CNN + 관절 21개 회귀)이다.
손 모양(쓰기/이동/완료/지우기)은 손가락 마디 각도로 판별하는 규칙이라 학습이 필요 없다.
한 손만 추적하며, 왼손/오른손 어느 쪽이든 동작한다.
"""

import platform

import cv2
import numpy as np

from .config import FINGER_EXTENDED_ANGLE, FRAME_HEIGHT, FRAME_WIDTH

# 손가락별 (MCP, PIP, DIP, TIP) 관절 번호
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
INDEX_TIP = 8


def finger_states(hand, aspect):
    """손가락이 펴졌는지 판별. 엄지는 쓰지 않는다 (사람마다 습관이 달라 불안정).

    펴짐 = 가운데 마디(PIP) 각도가 충분히 곧고, 손끝이 가운데 마디보다 손목에서 멀다.
    각도를 쓰므로 손을 기울여도 판별이 유지된다.
    """
    pts = np.asarray(hand, dtype=np.float32).copy()
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect          # 깊이(z)도 같이 써서 손이 기울어도 각도가 덜 왜곡됨
    wrist = pts[0]
    states = {}
    for name, (mcp, pip, _dip, tip) in FINGERS.items():
        v1, v2 = pts[mcp] - pts[pip], pts[tip] - pts[pip]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        farther = np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist)
        states[name] = bool(angle > FINGER_EXTENDED_ANGLE and farther)
    return states


def classify_gesture(states):
    """'draw' | 'hover' | 'confirm' | 'fist' | 'other'."""
    idx, mid, ring, pinky = states["index"], states["middle"], states["ring"], states["pinky"]
    if idx and not mid and not ring and not pinky:
        return "draw"
    if idx and mid and not ring and not pinky:
        return "hover"
    if idx and mid and ring and pinky:
        return "confirm"
    if not (idx or mid or ring or pinky):
        return "fist"
    return "other"


class HandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover
            raise ImportError("mediapipe가 설치되지 않았습니다: pip install -r requirements-dev.txt") from exc
        if not hasattr(mp, "solutions"):
            raise ImportError(f"설치된 mediapipe {getattr(mp, '__version__', '?')}에는 mp.solutions가 없습니다. "
                              "pip install mediapipe==0.10.21")
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=1,
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """손 21개 관절 (21, 3) 정규화 좌표, 손이 없으면 None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lms = result.multi_hand_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lms], np.float32)

    def close(self):
        if getattr(self, "hands", None) is not None:
            self.hands.close()
            self.hands = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def open_camera(index=0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, (frame.shape[1], frame.shape[0])
        cap.release()
    raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결, 장치 번호(--camera), 권한을 확인하세요.")
