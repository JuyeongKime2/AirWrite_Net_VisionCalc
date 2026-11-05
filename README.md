# ✍️ 허공 손글씨 계산기

웹캠 앞에서 검지로 허공에 숫자와 연산자(+ − × ÷ =)를 쓰면 인식해서 계산하는 프로젝트입니다.
데이터 수집과 학습은 **내 PC(VS Code)**에서, 완성된 앱은 **Streamlit Community Cloud**에 배포합니다.

https://airwritenetvisioncalc-ke2w9waykhj8lbusbjrdzp.streamlit.app/
---

## 0. 어떻게 동작하나요? (하이브리드 딥러닝)

```
웹캠
 ↓
MediaPipe Hands ── 사전학습 딥러닝: 손 관절 21개 위치
 ↓
손 모양 규칙 ───── ☝️쓰기 / ✌️이동 / 🖐️완료 / ✊지우기  (학습 불필요)
 ↓ 검지 끝 궤적
 ├─→ 28×28 그림 → CNN  (MNIST 7만 장으로 사전학습 → 내 글씨로 미세조정)  "모양"
 └─→ 점 64개 순서 → GRU (양방향 + 어텐션)                                 "쓰는 순서"
 ↓
앙상블 (CNN과 GRU 확률의 가중 평균, 비중은 검증 데이터로 자동 결정)
 ↓
계산기 (× ÷ 먼저 계산, 분수로 정확하게)
```

- **사전학습 + 전이학습**: 숫자는 MNIST로 먼저 배운 CNN을 가져와서 적은 데이터로도 잘 됩니다.
- **합성 데이터**: 연산자는 MNIST에 없어서, 프로그램으로 만든 연산자 궤적을 학습에 섞습니다.
- **앙상블**: 모양만 보는 CNN과 순서만 보는 GRU가 서로의 약점을 보완합니다.
- 학습 결과에서 CNN 단독 / GRU 단독 / 앙상블 점수를 비교해 볼 수 있습니다.
- **조기 예측**: 다 쓰기 전에 어떤 기호인지 맞힙니다. 학습 때 궤적의 앞부분만 잘라서도 보여주고,
  평가 때 25% · 50% · 75% · 100%까지만 쓴 궤적으로 채점해 '얼마나 썼을 때 얼마나 맞히는지' 곡선을 그립니다.
  앱에서는 쓰는 도중 예상 기호가 뜨고, 켜면 확신도가 높을 때 손바닥 없이 자동 입력됩니다.

---

## 1. 처음 한 번: 설치

준비물: **Python 3.11**, **VS Code**(+ Python 확장), **Git**, **GitHub 계정**

VS Code에서 이 폴더를 열고 `터미널 > 새 터미널`:

**Windows**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
**macOS**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
`Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv` 선택.

확인: `python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"` → `0.10.21 True`

---

## 2. 폴더 구성

```
AirWrite_Net_VisionCalc/
├── app.py                 계산기 앱 (Streamlit 시작 파일, 배포용)
├── requirements.txt       앱 실행용 라이브러리 (Streamlit Cloud가 읽음)
├── requirements-dev.txt   + 수집·학습용 라이브러리 (내 PC)
├── packages.txt           Streamlit Cloud용 리눅스 패키지
│
├── scripts/               내 PC에서 실행
│   ├── collect_data.py    ① 데이터 수집 (화면이 쓸 기호를 알려줌)
│   ├── train.py           ② 학습 + 교차검증 + 결과 페이지
│   └── evaluate_holdout.py ③ 확인용 세션 최종 점수
│
├── aircalc/               핵심 코드 (직접 실행하지 않음)
│   ├── config.py          모든 설정
│   ├── hand_tracking.py   손 추적, 손 모양 판별 규칙
│   ├── strokes.py         획 쌓기, 그림/시퀀스 변환, 증강, 합성 연산자
│   ├── models.py          CNN, GRU 모델
│   ├── recognizer.py      인식 + 계산기 연결
│   ├── calculator.py      식 계산
│   ├── drawing.py         얼굴 모자이크, 화면 그리기, 궤적 미리보기
│   └── report.py          시험 결과 HTML
│
├── models/                학습된 모델 (GitHub에 올림)
├── data/                  수집 데이터 (자동 생성, GitHub에 안 올림)
└── outputs/               결과 리포트 (자동 생성, GitHub에 안 올림)
```

명령은 항상 **프로젝트 맨 위 폴더에서** 실행하세요 (예: `python scripts/train.py`).
VS Code 왼쪽 **실행 및 디버그**(`Ctrl+Shift+D`)에서 1~4번 메뉴를 골라 ▶ 로 실행할 수 있습니다.

---

## 3. 손 모양

| 손 모양 | 동작 |
|---|---|
| ☝️ 검지만 펴기 | **쓰기** (검지 끝이 펜) |
| ✌️ 검지 + 중지 | **이동** (펜 들기, 획 사이 이동) |
| 🖐️ 손바닥 펴기 | **한 글자 완료** → 저장 또는 인식 |
| ✊ 주먹 0.6초 | **지우기** (쓰던 글자, 없으면 마지막 입력) |

연산자 쓰는 법 (획이 여러 개면 사이에 ✌️로 이동):

| 기호 | 쓰는 법 |
|---|---|
| + | 가로 → ✌️ → 세로 |
| − | 가로 한 획 |
| × | ↘ → ✌️ → ↙ |
| ÷ | 가로 → ✌️ → 위에 점 콕 → ✌️ → 아래에 점 콕 |
| = | 위 가로 → ✌️ → 아래 가로 |

한 손으로만 하면 됩니다. 왼손, 오른손 모두 동작하지만 **수집할 때와 같은 손**을 쓰세요.

---

## 4. 데이터 수집 계획 (혼자)

### 한 세션 = 60개, 약 7~8분
화면 오른쪽 위에 **써야 할 기호**가 나옵니다. 15개 기호를 섞인 순서로 4바퀴 씁니다.
그 기호를 쓰고 🖐️ 하면 저장되고 다음 기호로 넘어갑니다. **라벨은 화면이 정해 주므로 신경 쓸 필요가 없습니다.**

| 키 | 동작 |
|---|---|
| `space` | 완료 (🖐️ 대신) |
| `c` | 쓰던 글자 지우기 |
| `r` | 방금 저장한 것 삭제 → 다시 쓰기 |
| `s` | 이 기호 건너뛰기 |
| `q` | 종료 |

### 세션 9개 (학습용 8 + 확인용 1)

| 세션 | 조건 | 메뉴 |
|---|---|---|
| 1 | 책상 앞에 앉아서, 밝은 조명, 보통 크기 | 1번 |
| 2 | 크게 쓰기 (팔 전체로) | 1번 |
| 3 | 작게 쓰기 (손목 위주) | 1번 |
| 4 | 서서, 카메라에서 1.5m | 1번 |
| 5 | 어두운 조명 또는 창문 등지고 | 1번 |
| 6 | 빨리 쓰기 | 1번 |
| 7 | 천천히 또박또박 | 1번 |
| 8 | 다른 날, 다른 장소 | 1번 |
| 9 | 위 조건을 새로 섞어서 | **1-1번 (확인용)** |

- 학습용 8세션 × 기호당 4개 = **기호당 32개, 총 480개**
- 하루에 다 하지 말고 2~3일에 나눠 찍으면 글씨가 자연스럽게 달라져서 더 좋습니다.
- 4, 7처럼 원래 획 수가 사람마다 다른 숫자는 **평소 쓰는 방식 그대로** 쓰세요.
- 저장된 `.png`로 궤적을 바로 확인할 수 있습니다 (색 = 획 순서, 빈 원 = 시작점).

---

## 5. 학습과 결과 확인

```bash
python scripts/train.py              # 처음 실행 시 MNIST를 내려받아 CNN 사전학습 (몇 분, 한 번만)
python scripts/train.py --ablation   # MNIST 사전학습 효과 비교 (발표 자료용)
python scripts/train.py --no-prefix  # 앞부분 잘라 학습하기 없이 (조기 예측 효과 비교용)
python scripts/evaluate_holdout.py   # 확인용 세션 최종 점수
```

| 결과 (`outputs/`) | 내용 |
|---|---|
| `test_samples.html` | **브라우저로 열기.** 시험에 쓰인 글씨마다 궤적, CNN 입력 그림, 정답, CNN·GRU·앙상블 예측. "틀린 것만", "의견 다른 것만" 보기 |
| `holdout_samples.html` | 확인용 세션 결과 (가장 믿을 만한 점수) |
| `model_comparison.png/csv` | CNN vs GRU vs 앙상블 |
| `confusion_matrix.png` | 어떤 기호끼리 헷갈리는지 |
| `early_prediction.png/csv` | **조기 예측**: 25~100% 썼을 때 모델별 정확도 (전체 / 숫자 / 연산자), 자동 입력 비율과 그 정확도 |
| `holdout_early_prediction.png/csv` | 확인용 세션의 조기 예측 곡선 |

학습된 모델은 `models/cnn.pt`, `models/gru.pt`, `models/ensemble.json`에 저장됩니다.

**조기 예측 읽는 법**: '쓴 비율'은 획 길이 기준입니다 (50% = 전체 획 길이의 앞쪽 절반).
연산자는 앞부분만 보면 구별이 원래 불가능한 경우가 많습니다 (+ · = · ÷ 의 첫 획은 모두 − 와 같음).
그래서 연산자 곡선은 낮게 시작하는 것이 정상이고, 앱의 자동 입력도 − 와 1 은 하지 않습니다 (`config.py`의 `EARLY_NO_AUTO`).

---

## 6. GitHub에 올리기

`data/`(수집 데이터, MNIST)와 `outputs/`는 자동으로 빠집니다. **`models/`는 올라가야 앱이 동작합니다.**

**VS Code 버튼**: 소스 제어(`Ctrl+Shift+G`) → **Publish to GitHub** → Public 선택

**터미널** (GitHub에서 빈 저장소를 먼저 만든 뒤):
```bash
git init
git add .
git status          # data/, outputs/ 가 없는지 확인
git commit -m "허공 손글씨 계산기 초기 버전"
git branch -M main
git remote add origin https://github.com/<내아이디>/<저장소>.git
git push -u origin main
```
모델을 다시 학습하면 `git add models/ && git commit -m "모델 업데이트" && git push`

---

## 7. Streamlit Community Cloud 배포

1. [share.streamlit.io](https://share.streamlit.io) → GitHub 계정으로 로그인
2. **Create app** → 저장소, 브랜치 `main`, Main file `app.py`
3. **Advanced settings** → Python **3.11**
4. **Deploy** (첫 설치는 몇 분 걸림) → 접속해서 **START** → 카메라 권한 허용

회사·학교망에서 카메라 연결이 안 되면 `.streamlit/secrets.toml.example`을 참고해 앱 설정의 **Secrets**에 Cloudflare TURN 키를 넣으세요.

---

## 8. 문제 해결

| 증상 | 해결 |
|---|---|
| `mp.solutions가 없습니다` | `pip install mediapipe==0.10.21` |
| `'MessageFactory' object has no attribute 'GetPrototype'` | `pip install "protobuf>=4.25.3,<5"` |
| numpy 관련 오류 | `pip install "numpy<2"` |
| 손 모양이 자주 바뀌어 획이 끊김 | 손바닥이 카메라를 향하게, 손가락을 확실히 접기. `config.py`의 `GESTURE_STABLE_FRAMES`를 4~5로 |
| 획 끝에 꼬리가 생김 | `GESTURE_STABLE_FRAMES`를 늘리거나, 완료 전에 ✌️로 바꾼 뒤 🖐️ |
| MNIST 다운로드 실패 | 인터넷 확인. 안 되면 `python scripts/train.py --no-pretrain` (CNN을 처음부터 학습) |
| 배포 앱 "모델을 불러오지 못했습니다" | `models/cnn.pt`, `gru.pt`, `ensemble.json`을 커밋·푸시 |
| 배포 앱 `libGL.so.1` 오류 | `packages.txt`가 저장소 맨 위에 있는지 확인 |
| 내 PC에선 되는데 배포 설치 실패 | `pip show mediapipe streamlit streamlit-webrtc torch`로 버전 확인 후 `requirements.txt`에 `==`로 고정 |

## 개인정보

- 저장하는 것은 **검지 끝 궤적 좌표**와 궤적 그림뿐입니다. 카메라 영상이나 사진은 저장하지 않습니다.
- 화면의 얼굴은 OpenCV 얼굴 검출로 자동 모자이크됩니다 (옆모습은 놓칠 수 있음).
