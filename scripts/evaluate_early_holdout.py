"""확인용(홀드아웃) 세션으로 조기 예측 최종 분석.

train_early.py로 저장한 모델(models/early.json)을 학습에 안 쓴 data/holdout 세션으로 평가하고,
같은 형식의 리포트를 outputs/early_holdout/early_report.html 로 만든다.

실행:  python scripts/evaluate_early_holdout.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 프로젝트 맨 위를 import 경로에 추가

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train as T  # noqa: E402
from aircalc.config import EVAL_FRACTIONS, HOLDOUT_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: E402
from aircalc.early import KNNTemplate  # noqa: E402
from aircalc.early_models import load_early  # noqa: E402
from aircalc.early_report import build_early_report  # noqa: E402
from aircalc.models import load_checkpoint  # noqa: E402
from train_early import evaluate_test  # noqa: E402


def main():
    info_path = MODEL_DIR / "early.json"
    if not info_path.exists():
        raise SystemExit("models/early.json이 없습니다. 먼저 python scripts/train_early.py를 실행하세요.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    seq, _ = load_early(MODEL_DIR / info["seq"])
    enc = info["encoder"]
    cnn = load_checkpoint(MODEL_DIR / info["cnn"])[0] if info.get("cnn") else None
    weight = info.get("cnn_weight", 0.0)

    train_strokes, train_y, *_ = T.load_dataset()               # kNN 기준선은 학습 데이터를 템플릿으로 사용
    knn = KNNTemplate(train_strokes, train_y)
    strokes, y, _, sessions, paths = T.load_dataset(HOLDOUT_DIR)
    print(f"확인용 글씨 {len(y)}개, 세션 {len(set(sessions.tolist()))}개")

    rows, gallery, evolution = evaluate_test(np.arange(len(y)), strokes, y, {enc: seq}, cnn, weight, enc, knn, 1,
                                             EVAL_FRACTIONS)
    df = pd.DataFrame(rows)
    df["session"] = [str(sessions[i]) for i in df["index"]]
    df["path"] = [paths[i] for i in df["index"]]
    out_dir = OUTPUT_DIR / "early_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "early_predictions.csv", index=False, encoding="utf-8-sig")

    classifiers = ["knn"] + (["cnn"] if cnn is not None else []) + [enc] + (["ensemble"] if cnn is not None else [])
    page = build_early_report(out_dir, df, classifiers, ["linear", "knn", enc], [enc], enc, enc, gallery, evolution,
                              "holdout", [{"fold": 1, "test": sorted(set(sessions.tolist())), "weight": weight}],
                              title="허공 손글씨 조기 예측 · 확인용 세션")
    best = "ensemble" if cnn is not None else enc
    for f in (0.3, 0.5, 0.7, 1.0):
        d = df[df.frac == f]
        print(f"{int(f * 100):3d}% 썼을 때 정확도: {(d[f'{best}_pred'] == d['true']).mean():.3f}"
              + (f" | 남은 궤적 오차 {d[f'{enc}_ade_rel'].mean() * 100:.1f}% ({d[f'{enc}_ade_px'].mean():.0f}px)"
                 if f < 1 else ""))
    print(f"리포트: {page}")


if __name__ == "__main__":
    main()
