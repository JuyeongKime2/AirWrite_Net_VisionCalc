"""조기 예측 딥러닝 모델: 앞부분 궤적 -> (기호 분류, 완성 궤적 좌표) 동시 예측 (멀티태스크).

    앞부분 궤적 (32점 x 8특징)
            │
      인코더 (GRU | LSTM | TCN(1D-CNN) | Transformer)   ← 이걸 바꿔 가며 비교
            │ h
     ┌──────┴───────────┐
  분류 머리            궤적 머리 (h + 분류 확률 -> 완성 궤적 48점 x [x, y, 획 시작])
  (15개 기호)

두 작업을 같이 배우면 '어떤 기호인지'가 '어떻게 마저 그릴지'를 돕고, 그 반대도 성립한다.
"""

import json
import math

import torch
import torch.nn as nn

from .early import INPUT_DIM
from .models import AttentionPool


class _TCN(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers, cin = [], in_dim
        for dilation in (1, 2, 4):
            layers += [nn.Conv1d(cin, hidden, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            cin = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):                                   # (B, T, F) -> (B, T, H)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class _Transformer(nn.Module):
    def __init__(self, in_dim, hidden, dropout, length, layers=2, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, length, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        return self.encoder(self.proj(x) + self.pos[:, : x.size(1)])


class EarlyNet(nn.Module):
    def __init__(self, encoder, num_classes, length, out_points, hidden=96, dropout=0.2):
        super().__init__()
        self.encoder_type, self.out_points = encoder, out_points
        if encoder in ("gru", "lstm"):
            rnn = nn.GRU if encoder == "gru" else nn.LSTM
            self.encoder = rnn(INPUT_DIM, hidden, batch_first=True, bidirectional=True)
            size = hidden * 2
        elif encoder == "tcn":
            self.encoder = _TCN(INPUT_DIM, hidden, dropout)
            size = hidden
        elif encoder == "transformer":
            hidden = int(math.ceil(hidden / 4) * 4)          # head 수(4)로 나눠떨어지게
            self.encoder = _Transformer(INPUT_DIM, hidden, dropout, length)
            size = hidden
        else:
            raise ValueError(encoder)
        self.pool = AttentionPool(size)
        self.cls = nn.Sequential(nn.LayerNorm(size), nn.Dropout(dropout), nn.Linear(size, num_classes))
        self.traj = nn.Sequential(nn.Linear(size + num_classes, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_points * 3))

    def forward(self, x):
        h = self.encoder(x)
        if isinstance(h, tuple):                            # GRU/LSTM은 (출력, 상태)
            h = h[0]
        h = self.pool(h)
        logits = self.cls(h)
        traj = self.traj(torch.cat([h, torch.softmax(logits, dim=1).detach()], dim=1))
        return logits, traj.view(-1, self.out_points, 3)   # [..., :2] 좌표, [..., 2] 획 시작 logit


def save_early(path, model, meta):
    clean = json.loads(json.dumps(meta, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    torch.save({"state_dict": model.state_dict(), "meta": clean}, path)


def load_early(path, device="cpu"):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    meta = ckpt["meta"]
    model = EarlyNet(meta["encoder"], meta["num_classes"], meta["length"], meta["out_points"],
                     hidden=meta["hidden"], dropout=meta["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta