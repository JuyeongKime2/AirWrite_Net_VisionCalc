"""학습된 CNN + GRU로 기호를 인식하고, 계산기와 연결한다 (Streamlit, 카메라와 무관하게 테스트 가능)."""

import json

import numpy as np
import torch

from .calculator import Calculator
from .config import CLASSES, MIN_CONFIDENCE, MODEL_DIR, SYMBOL
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
    """프레임마다 update()를 부르면 글씨 쓰기 -> 인식 -> 계산까지 처리한다."""

    def __init__(self, recognizer, min_confidence=MIN_CONFIDENCE):
        self.recognizer = recognizer
        self.writer = AirWriter()
        self.calc = Calculator()
        self.min_confidence = min_confidence
        self.last = None            # 마지막 인식 결과 dict
        self.notice = "검지만 펴서 숫자를 써 보세요"

    def update(self, t, hand, aspect):
        event = self.writer.update(t, hand, aspect)
        if event is None:
            return
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
        message = self.calc.push(result["label"])
        self.notice = message or f"'{SYMBOL[result['label']]}' 입력 ({result['confidence']:.0%})"

    def snapshot(self):
        return {
            "display": self.calc.display(),
            "notice": self.notice,
            "last": self.last,
            "history": list(self.calc.history[-5:]),
            "gesture": self.writer.gesture,
        }
