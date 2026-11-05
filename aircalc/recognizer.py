"""학습된 CNN + GRU로 기호를 인식하고, 계산기와 연결한다 (Streamlit, 카메라와 무관하게 테스트 가능)."""

import json

import numpy as np
import torch

from .calculator import Calculator
from .config import (
    CLASSES,
    EARLY_AUTO_CONFIDENCE,
    EARLY_AUTO_DEFAULT,
    EARLY_INTERVAL,
    EARLY_MIN_POINTS,
    EARLY_NO_AUTO,
    EARLY_STABLE_PREDICTIONS,
    MIN_CONFIDENCE,
    MODEL_DIR,
    SYMBOL,
)
from .models import ensemble, load_checkpoint
from .strokes import FEATURE_VERSION, AirWriter, render_image, seq_features


class SymbolRecognizer:
    def __init__(self, model_dir=MODEL_DIR, device="cpu"):
        info_path = model_dir / "ensemble.json"
        if not info_path.exists():
            raise FileNotFoundError("models/ensemble.json이 없습니다. 먼저 python scripts/train.py로 학습하세요.")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.cnn, self.cnn_meta = load_checkpoint(model_dir / self.info["cnn"], device)
        self.gru, self.gru_meta = load_checkpoint(model_dir / self.info["gru"], device)
        self.weight = float(self.info["cnn_weight"])
        self.classes = self.info["classes"]
        if self.classes != CLASSES:
            raise ValueError("모델의 클래스 구성이 config.py와 다릅니다. 다시 학습하세요.")
        self.version_ok = self.info.get("feature_version") == FEATURE_VERSION

    @torch.no_grad()
    def predict(self, strokes):
        img = torch.from_numpy(render_image(strokes))[None, None]
        seq = torch.from_numpy(seq_features(strokes))[None]
        p_cnn = torch.softmax(self.cnn(img), dim=1)[0].numpy()
        p_gru = torch.softmax(self.gru(seq), dim=1)[0].numpy()
        p = ensemble(p_cnn, p_gru, self.weight)
        order = np.argsort(p)[::-1]
        return {
            "label": self.classes[order[0]],
            "confidence": float(p[order[0]]),
            "top3": [(self.classes[i], float(p[i])) for i in order[:3]],
            "cnn": self.classes[int(p_cnn.argmax())],
            "gru": self.classes[int(p_gru.argmax())],
        }


class AirCalculatorSession:
    """프레임마다 update()를 부르면 글씨 쓰기 -> 인식 -> 계산까지 처리한다.

    조기 예측: 쓰는 도중에도 EARLY_INTERVAL초마다 지금까지 쓴 궤적으로 예측해 live에 담는다.
    auto_commit을 켜면, 쓰는 중(검지만 편 상태)에 같은 예측이 EARLY_STABLE_PREDICTIONS번 연속,
    확신도 EARLY_AUTO_CONFIDENCE 이상으로 나올 때 손바닥 없이 바로 입력한다.
    """

    def __init__(self, recognizer, min_confidence=MIN_CONFIDENCE, auto_commit=EARLY_AUTO_DEFAULT):
        self.recognizer = recognizer
        self.writer = AirWriter()
        self.calc = Calculator()
        self.min_confidence = min_confidence
        self.auto_commit = auto_commit
        self.last = None            # 마지막 인식 결과 dict
        self.live = None            # 쓰는 중 예측 결과 dict (조기 예측)
        self.notice = "검지만 펴서 숫자를 써 보세요"
        self._reset_live()

    def _reset_live(self):
        self.live = None
        self._live_t, self._live_points = float("-inf"), 0
        self._live_label, self._live_count = None, 0

    def _early(self, t):
        strokes = self.writer.visible_strokes()
        n_points = sum(len(s) for s in strokes)
        if n_points < EARLY_MIN_POINTS:
            self._reset_live()
            return
        if t - self._live_t < EARLY_INTERVAL or n_points == self._live_points:   # 새로 쓴 게 없으면 그대로
            return
        self._live_t, self._live_points = t, n_points
        result = self.recognizer.predict(strokes)
        self.live = result
        if result["label"] == self._live_label:
            self._live_count += 1
        else:
            self._live_label, self._live_count = result["label"], 1
        if (self.auto_commit and self.writer.gesture == "draw"
                and result["confidence"] >= EARLY_AUTO_CONFIDENCE
                and result["label"] not in EARLY_NO_AUTO
                and self._live_count >= EARLY_STABLE_PREDICTIONS):
            self.writer.suspend()           # 남은 획은 손 모양을 바꿀 때까지 무시
            self._reset_live()
            self._commit(result, auto=True)

    def update(self, t, hand, aspect):
        event = self.writer.update(t, hand, aspect)
        if event is None:
            self._early(t)
            return
        self._reset_live()
        kind, payload = event
        if kind == "confirm":
            self.handle_strokes(payload)
        elif kind == "clear":
            if payload:
                self.notice = "쓰던 글자를 지웠습니다"
            else:
                self.calc.backspace()
                self.notice = "마지막 입력을 지웠습니다"

    def handle_strokes(self, strokes):
        result = self.recognizer.predict(strokes)
        self.last = result
        if result["confidence"] < self.min_confidence:
            cands = ", ".join(SYMBOL[c] for c, _ in result["top3"])
            self.notice = f"잘 모르겠어요 (후보: {cands}). 다시 써 주세요"
            return
        self._commit(result)

    def _commit(self, result, auto=False):
        self.last = result
        message = self.calc.push(result["label"])
        how = "자동 입력 (손 모양을 바꾸고 다음 글자)" if auto else "입력"
        self.notice = message or f"'{SYMBOL[result['label']]}' {how} ({result['confidence']:.0%})"

    def snapshot(self):
        return {
            "display": self.calc.display(),
            "notice": self.notice,
            "last": self.last,
            "live": self.live,
            "auto_commit": self.auto_commit,
            "history": list(self.calc.history[-5:]),
            "gesture": self.writer.gesture,
        }
