"""허공 손글씨 계산기 - 공통 설정."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 프로젝트 맨 위 폴더 (data/, models/, outputs/ 위치)
DATA_DIR = ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"      # 학습용 세션
HOLDOUT_DIR = DATA_DIR / "holdout"        # 마지막 확인용 세션 (학습에 절대 안 씀)
PRETRAIN_DIR = DATA_DIR / "pretrained"    # MNIST 원본과 사전학습 가중치 캐시
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# ------------------------------------------------------------
# 클래스: 숫자 10개 + 연산자 5개, 모두 허공에 써서 입력
# ------------------------------------------------------------
DIGITS = [str(i) for i in range(10)]
OPERATORS = ["plus", "minus", "times", "divide", "equals"]
CLASSES = DIGITS + OPERATORS
NUM_CLASSES = len(CLASSES)
SYMBOL = {**{d: d for d in DIGITS}, "plus": "+", "minus": "−", "times": "×", "divide": "÷", "equals": "="}
ASCII = {**{d: d for d in DIGITS}, "plus": "+", "minus": "-", "times": "x", "divide": "/", "equals": "="}

# 수집할 때 화면에 보여줄 쓰는 법 (획 순서)
WRITING_GUIDE = {
    "plus": "가로 → (V 손으로 이동) → 세로",
    "minus": "가로 한 획",
    "times": "↘ 대각선 → (V 손으로 이동) → ↙ 대각선",
    "divide": "가로 → 위에 점 콕 → 아래에 점 콕",
    "equals": "위 가로 → (V 손으로 이동) → 아래 가로",
}

# ------------------------------------------------------------
# 카메라 / 손 모양 (규칙 기반)
#   검지만 펴기      = 쓰기 (펜 내림)
#   검지+중지 (V)    = 이동 (펜 들기, 획 사이 이동)
#   손바닥 펴기      = 한 글자 완료 -> 인식
#   주먹 (잠깐 유지) = 쓰던 글자 지우기 / 쓰던 게 없으면 마지막 입력 지우기
# ------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GESTURE_STABLE_FRAMES = 3       # 손 모양이 이 프레임 수만큼 유지돼야 바뀐 것으로 인정
FINGER_EXTENDED_ANGLE = 150.0   # 손가락 가운데 마디 각도가 이보다 크면 '편 손가락'
FIST_HOLD_SECONDS = 0.6
TIP_SMOOTHING = 0.5             # 검지 끝 위치 떨림 보정 (0~1, 클수록 원래 위치에 가까움)
DOT_LENGTH = 0.08               # 정규화 후 이보다 짧은 획은 '점'으로 그림 (÷의 점)

# ------------------------------------------------------------
# 모델 입력
# ------------------------------------------------------------
IMAGE_SIZE = 28                 # CNN 입력 (MNIST와 같은 28x28)
IMAGE_INNER = 20                # 글씨가 들어가는 영역 (MNIST와 같은 20x20 + 여백)
RENDER_SCALE = 4                # 4배 크게 그린 뒤 줄여서 부드럽게
STROKE_THICKNESS = 9            # 4배 캔버스 기준 선 굵기 (28px 기준 약 2.2px, MNIST와 비슷)
SEQ_POINTS = 64                 # GRU 입력: 획 길이를 따라 균등하게 뽑는 점 수 (쓰는 속도와 무관)

# ------------------------------------------------------------
# 수집
# ------------------------------------------------------------
ROUNDS_PER_SESSION = 4          # 한 세션 = 15개 기호 x 4바퀴 = 60개 (순서는 바퀴마다 섞임)

# ------------------------------------------------------------
# 학습
# ------------------------------------------------------------
CNN_HP = {"dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4}
SEQ_HP = {"hidden": 64, "dropout": 0.3, "lr": 2e-3, "weight_decay": 1e-4}
EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.05
RANDOM_SEED = 42
CV_FOLDS = 5
MIN_SUBJECTS_FOR_SUBJECT_SPLIT = 3
MIN_SAMPLES_PER_CLASS = 8
MNIST_PRETRAIN_EPOCHS = 3
SYNTHETIC_PER_OPERATOR = 120    # 학습 fold에만 섞는 합성 연산자 수 (연산자는 MNIST에 없어서 보충)
ENSEMBLE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # CNN 가중치 후보 (나머지는 GRU)

# ------------------------------------------------------------
# 앱
# ------------------------------------------------------------
MIN_CONFIDENCE = 0.50           # 이보다 낮으면 입력하지 않고 "다시 써 주세요"
CAMERA_CONSTRAINTS = {"width": {"ideal": FRAME_WIDTH}, "height": {"ideal": FRAME_HEIGHT}, "frameRate": {"ideal": 30}}
