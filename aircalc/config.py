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
GESTURE_STABLE_FRAMES = 3       # 쓰기 '시작': 손 모양이 이 프레임 수만큼 유지돼야 인정
GESTURE_RELEASE_FRAMES = 2      # 쓰기 '멈춤': 검지만 편 모양이 아닌 프레임이 이만큼이면 바로 획을 끊음
FINGER_EXTENDED_ANGLE = 140.0   # 손가락 가운데 마디 각도가 이보다 크면 '편 손가락'
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
# 조기 예측 (다 쓰기 전에 어떤 기호인지 맞히기)
#   '쓴 비율' = 잉크(획 길이) 기준. 50% = 전체 획 길이의 앞쪽 절반만 쓴 상태
# ------------------------------------------------------------
PREFIX_TRAIN_PROB = 0.3         # 학습 때 이 확률로 궤적의 앞부분만 보여줌 (부분만 보고도 맞히도록)
PREFIX_MIN = 0.25               # 앞부분만 보여줄 때 최소 비율
EARLY_FRACTIONS = [0.25, 0.5, 0.75, 1.0]   # 평가할 '쓴 비율'
EARLY_AUTO_CONFIDENCE = 0.90    # 앱에서 이 확신도 이상이면 손바닥 없이 자동 입력 (켰을 때만)
EARLY_STABLE_PREDICTIONS = 3    # 자동 입력: 같은 예측이 이만큼 연속으로 나와야 함
EARLY_NO_AUTO = ["minus", "1"]  # 다른 기호의 첫 획과 모양이 같아 자동 입력하지 않음 (− 는 + = ÷ 의 첫 획)
EARLY_MIN_POINTS = 8            # 이보다 점이 적으면 예측하지 않음
EARLY_INTERVAL = 0.15           # 쓰는 중 예측 간격 (초)
EARLY_AUTO_DEFAULT = False      # 앱 시작 시 자동 입력 켜짐 여부

# ------------------------------------------------------------
# 앱
# ------------------------------------------------------------
MIN_CONFIDENCE = 0.50           # 이보다 낮으면 입력하지 않고 "다시 써 주세요"
CAMERA_CONSTRAINTS = {"width": {"ideal": FRAME_WIDTH}, "height": {"ideal": FRAME_HEIGHT}, "frameRate": {"ideal": 30}}

# ------------------------------------------------------------
# 조기 예측 (글자를 다 쓰기 전에 기호와 나머지 궤적을 예측)
# ------------------------------------------------------------
EARLY_POINTS = 32
COMPLETE_POINTS = 48
ABS_SCALE = 0.25
EVAL_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
EARLY_ENCODERS = ["gru", "lstm", "tcn", "transformer"]
EARLY_HP = {"hidden": 96, "dropout": 0.2, "lr": 2e-3, "weight_decay": 1e-4,
            "coord_weight": 1.0, "start_weight": 0.2}
EARLY_EPOCHS = 40
EARLY_LIVE_EVERY = 4
EARLY_AUTO_CONFIRM = False
EARLY_AUTO_THRESHOLD = 0.92
EARLY_AUTO_STABLE = 6
