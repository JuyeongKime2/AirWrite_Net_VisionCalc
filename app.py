"""허공 손글씨 계산기 - Streamlit 앱 (배포용).

로컬 실행:  streamlit run app.py
배포:      Streamlit Community Cloud에서 Main file을 app.py로 지정

카메라는 접속한 사람의 브라우저 카메라를 쓴다 (streamlit-webrtc).
"""

import json
import os
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    VideoProcessorBase = object

from aircalc.config import (
    CAMERA_CONSTRAINTS,
    EARLY_AUTO_CONFIDENCE,
    EARLY_AUTO_DEFAULT,
    EARLY_NO_AUTO,
    MODEL_DIR,
    SYMBOL,
    WRITING_GUIDE,
)
from aircalc.drawing import FaceMosaic, draw_hand, draw_status_bar, draw_strokes
from aircalc.hand_tracking import HandTracker
from aircalc.recognizer import AirCalculatorSession, SymbolRecognizer

TO_ASCII = str.maketrans({"−": "-", "×": "x", "÷": "/"})

st.set_page_config(page_title="허공 손글씨 계산기", page_icon="✍️", layout="wide")
st.title("✍️ 허공 손글씨 계산기")
st.caption("검지로 허공에 숫자와 연산자를 쓰면 인식해서 계산합니다. 얼굴은 자동 모자이크되며 영상은 저장하지 않습니다.")


@st.cache_resource
def get_recognizer():
    return SymbolRecognizer(MODEL_DIR)


try:
    recognizer = get_recognizer()
except Exception as exc:
    st.error(f"모델을 불러오지 못했습니다: {exc}\n\n내 PC에서 `python scripts/train.py`로 학습한 뒤 "
             "`models/` 폴더(cnn.pt, gru.pt, ensemble.json)를 GitHub에 올려 주세요.")
    st.stop()
if not recognizer.version_ok:
    st.warning("모델이 다른 버전의 전처리로 학습됐습니다. train.py로 다시 학습하세요.")


# ============================================================
# TURN 설정 (회사/학교망 대비)
# ============================================================
def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def rtc_configuration():
    key_id, token = _secret("CLOUDFLARE_TURN_KEY_ID"), _secret("CLOUDFLARE_TURN_KEY_API_TOKEN")
    if key_id and token:
        os.environ["CLOUDFLARE_TURN_KEY_ID"] = str(key_id)
        os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = str(token)
        return None
    ice = _secret("ICE_SERVERS")
    if ice:
        return {"iceServers": json.loads(ice) if isinstance(ice, str) else list(ice)}
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ============================================================
# 영상 처리 (별도 스레드)
# ============================================================
class AirCalcProcessor(VideoProcessorBase):
    def __init__(self, recognizer):
        self.tracker = HandTracker()            # 접속자마다 따로
        self.session = AirCalculatorSession(recognizer)
        self.face = FaceMosaic()
        self._lock = threading.Lock()

    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)      # 거울 모드 (수집과 동일)
        aspect = img.shape[1] / img.shape[0]
        hand = self.tracker.process(img)
        with self._lock:
            self.session.update(time.perf_counter(), hand, aspect)
            strokes = self.session.writer.visible_strokes()
            gesture = self.session.writer.gesture
            display = self.session.calc.display().translate(TO_ASCII)
            live = self.session.live
        view = self.face(img)
        draw_strokes(view, strokes, aspect)
        draw_hand(view, hand, gesture)
        draw_status_bar(view, gesture, display[-28:])
        if live:                                        # 조기 예측: 쓰는 중 예상 기호
            text = f"guess: {SYMBOL[live['label']].translate(TO_ASCII)} {live['confidence']:.0%}"
            cv2.putText(view, text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 230, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(view, format="bgr24")

    def snapshot(self):
        with self._lock:
            return self.session.snapshot()

    def set_auto(self, on):
        with self._lock:
            self.session.auto_commit = bool(on)

    def backspace(self):
        with self._lock:
            self.session.writer.clear()
            self.session.calc.backspace()
            self.session.notice = "마지막 입력을 지웠습니다"

    def clear_all(self):
        with self._lock:
            self.session.writer.clear()
            self.session.calc.clear()
            self.session.notice = "모두 지웠습니다"

    def __del__(self):
        try:
            self.tracker.close()
        except Exception:
            pass


# ============================================================
# 화면
# ============================================================
left, right = st.columns([3, 2])
with left:
    ctx = webrtc_streamer(
        key="air-calc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration(),
        media_stream_constraints={"video": CAMERA_CONSTRAINTS, "audio": False},
        video_processor_factory=lambda: AirCalcProcessor(recognizer),
        async_processing=True,
    )
    with st.expander("사용법", expanded=not ctx.state.playing):
        st.markdown(
            "| 손 모양 | 동작 |\n|---|---|\n"
            "| ☝️ 검지만 펴기 | 쓰기 |\n| ✌️ 검지+중지 | 이동 (획 사이) |\n"
            "| 🖐️ 손바닥 펴기 | 한 글자 완료 → 인식 |\n| ✊ 주먹 잠깐 | 쓰던 글자 지우기 / 없으면 마지막 입력 지우기 |\n\n"
            "**연산자 쓰는 법**\n\n" + "\n".join(f"- **{SYMBOL[k]}** : {v}" for k, v in WRITING_GUIDE.items())
            + "\n\n한 글자씩 쓰고 손바닥을 펴세요. 예: `1` → 🖐️ → `2` → 🖐️ → `+` → 🖐️ → `3` → 🖐️ → `=` → 🖐️"
        )

with right:
    auto = st.toggle("다 쓰기 전에 자동 입력 (조기 예측)", value=EARLY_AUTO_DEFAULT,
                     help=f"쓰는 도중 확신도가 {EARLY_AUTO_CONFIDENCE:.0%} 이상으로 안정되면 손바닥 없이 바로 입력합니다. "
                          f"{', '.join(SYMBOL[c] for c in EARLY_NO_AUTO)} 는 다른 기호의 첫 획과 같아서 끝까지 써야 합니다. "
                          "자동 입력 뒤에는 손 모양을 바꾼 다음 다음 글자를 쓰세요.")
    expr_area = st.empty()
    live_area = st.empty()
    notice_area = st.empty()
    last_area = st.empty()
    b1, b2 = st.columns(2)
    if b1.button("⌫ 마지막 지우기") and ctx.video_processor:
        ctx.video_processor.backspace()
    if b2.button("🗑️ 전체 지우기") and ctx.video_processor:
        ctx.video_processor.clear_all()
    history_area = st.empty()


def render(snap):
    expr_area.markdown(f"<div style='font-size:2.6rem;font-weight:700;text-align:right;padding:12px 16px;"
                       f"border:1px solid rgba(128,128,128,.35);border-radius:12px;word-break:break-all'>"
                       f"{snap['display']}</div>", unsafe_allow_html=True)
    live = snap.get("live")
    if live:
        live_area.markdown(f"✍️ 쓰는 중 예상: **{SYMBOL[live['label']]}** ({live['confidence']:.0%})")
    else:
        live_area.empty()
    notice_area.info(snap["notice"])
    last = snap["last"]
    if last:
        cands = " · ".join(f"{SYMBOL[c]} {p:.0%}" for c, p in last["top3"])
        last_area.caption(f"마지막 인식: **{SYMBOL[last['label']]}** ({last['confidence']:.0%}) | 후보 {cands} | "
                          f"CNN {SYMBOL[last['cnn']]} · GRU {SYMBOL[last['gru']]}")
    if snap["history"]:
        history_area.markdown("**계산 기록**\n\n" + "\n".join(f"- {e} = **{r}**" for e, r in reversed(snap["history"])))


if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor is not None:
            ctx.video_processor.set_auto(auto)
            render(ctx.video_processor.snapshot())
        time.sleep(0.15)
else:
    expr_area.markdown("<div style='font-size:2.6rem;text-align:right;padding:12px 16px;opacity:.5'>0</div>",
                       unsafe_allow_html=True)
    notice_area.info("왼쪽의 START를 누르고 카메라 권한을 허용하세요.")
