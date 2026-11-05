"""딥러닝 모델: 모양을 보는 CNN + 쓰는 순서를 보는 GRU, 그리고 저장/불러오기.

하이브리드 구조
    MediaPipe Hands (사전학습 딥러닝, 손 관절 추출)
      -> 규칙 (손 모양으로 쓰기/이동/완료/지우기)
      -> CNN  (궤적을 28x28 그림으로, MNIST로 사전학습 후 미세조정)
      -> GRU  (궤적을 점 순서대로)
      -> 가중 평균 앙상블
"""

import json

import numpy as np
import torch
import torch.nn as nn


class StrokeCNN(nn.Module):
    """MNIST용으로 흔히 쓰는 작은 CNN. 28x28 1채널 입력."""

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64))          # 28 -> 14 -> 7
        self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                   nn.Dropout(dropout))
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):                         # x: (B, 1, 28, 28)
        return self.head(self.embed(self.features(x)))

    def load_pretrained(self, state_dict, keep_rows):
        """MNIST(10클래스)로 학습한 가중치를 불러온다.
        특징 추출부는 그대로, 마지막 층은 숫자 클래스 줄만 복사하고 연산자 줄은 새로 시작."""
        own = self.state_dict()
        for k, v in state_dict.items():
            if k.startswith("head."):
                continue
            if k in own and own[k].shape == v.shape:
                own[k] = v
        for new_row, old_row in keep_rows.items():
            own["head.weight"][new_row] = state_dict["head.weight"][old_row]
            own["head.bias"][new_row] = state_dict["head.bias"][old_row]
        self.load_state_dict(own)


class AttentionPool(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.score = nn.Linear(size, 1)

    def forward(self, h):
        w = torch.softmax(self.score(h), dim=1)
        return (w * h).sum(dim=1)


class StrokeGRU(nn.Module):
    """(B, 64, 6) 점 순서 입력. 양방향 GRU + 어텐션 풀링."""

    def __init__(self, num_classes, input_size=6, hidden=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(hidden * 2)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.pool(h))


def build_model(kind, num_classes, **kwargs):
    if kind == "cnn":
        return StrokeCNN(num_classes, dropout=kwargs.get("dropout", 0.3))
    if kind == "gru":
        return StrokeGRU(num_classes, hidden=kwargs.get("hidden", 64), dropout=kwargs.get("dropout", 0.3))
    raise ValueError(kind)


def save_checkpoint(path, model, meta):
    clean = json.loads(json.dumps(meta, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    torch.save({"state_dict": model.state_dict(), "meta": clean}, path)


def load_checkpoint(path, device="cpu"):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    meta = ckpt["meta"]
    model = build_model(meta["kind"], meta["num_classes"], **meta.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta


def ensemble(p_cnn, p_gru, weight):
    """CNN 확률과 GRU 확률의 가중 평균. weight = CNN 비중 (0~1)."""
    return weight * np.asarray(p_cnn) + (1 - weight) * np.asarray(p_gru)
