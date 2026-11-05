"""확인용(홀드아웃) 세션으로 최종 성능 확인.

학습에 전혀 쓰지 않은 data/holdout 세션으로 저장된 모델(models/)을 채점한다.

실행:   python scripts/evaluate_holdout.py
결과:   outputs/holdout_samples.html, outputs/holdout_confusion_matrix.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 프로젝트 맨 위를 import 경로에 추가

import json

import numpy as np

from aircalc.config import CLASSES, HOLDOUT_DIR, MODEL_DIR, OUTPUT_DIR
from aircalc.models import ensemble
from aircalc.recognizer import SymbolRecognizer
from aircalc.report import build_report
from train import compute_metrics, load_dataset, plot_confusion, precompute, predict_proba
from sklearn.metrics import confusion_matrix


def main():
    rec = SymbolRecognizer(MODEL_DIR)
    if not rec.version_ok:
        print("[주의] 모델이 다른 버전의 전처리로 학습됐습니다. python scripts/train.py로 다시 학습하세요.")
    strokes, y, _, sessions, paths = load_dataset(HOLDOUT_DIR)
    print(f"확인용 글씨 {len(y)}개, 세션 {len(set(sessions.tolist()))}개")

    p_cnn = predict_proba(rec.cnn, precompute("cnn", strokes))
    p_gru = predict_proba(rec.gru, precompute("gru", strokes))
    p_ens = ensemble(p_cnn, p_gru, rec.weight)

    summary, metrics = {}, {}
    for name, p in (("cnn", p_cnn), ("gru", p_gru), ("ensemble", p_ens)):
        m = compute_metrics(y, p.argmax(1))
        metrics[name] = m
        summary[name] = {f"{k}_mean": v for k, v in m.items()}
        summary[name].update({f"{k}_std": 0.0 for k in m})

    pred = p_ens.argmax(1)
    records = []
    for i in range(len(y)):
        order = np.argsort(p_ens[i])[::-1][:3]
        records.append({"index": i, "path": paths[i], "session": str(sessions[i]), "fold": 1,
                        "true": int(y[i]), "pred": int(pred[i]),
                        "top3": [(int(c), float(p_ens[i][c])) for c in order],
                        "cnn": int(p_cnn[i].argmax()), "cnn_conf": float(p_cnn[i].max()),
                        "gru": int(p_gru[i].argmax()), "gru_conf": float(p_gru[i].max())})
    fold = {"fold": 1, "test_groups": sorted(set(sessions.tolist())),
            "train_groups": [f"data/sessions 전체 ({rec.info.get('trained_on_samples', '?')}개)"],
            "val_groups": [], "metrics": metrics["ensemble"], "weight": rec.weight}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page = build_report(OUTPUT_DIR / "holdout_samples.html", "확인용 세션", "holdout", [fold], records, strokes, summary)
    plot_confusion(confusion_matrix(y, pred, labels=list(range(len(CLASSES)))), "Ensemble (holdout)",
                   OUTPUT_DIR / "holdout_confusion_matrix.png")
    (OUTPUT_DIR / "holdout_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cv = rec.info.get("cv_summary", {}).get("ensemble", {})
    for name, m in metrics.items():
        print(f"{name:9s} macro F1 {m['macro_f1']:.3f} | 숫자 {m['digit_accuracy']:.3f} | 연산자 {m['operator_accuracy']:.3f}")
    if cv:
        gap = cv["macro_f1_mean"] - metrics["ensemble"]["macro_f1"]
        print(f"교차검증 앙상블 macro F1 {cv['macro_f1_mean']:.3f} -> 확인용 {metrics['ensemble']['macro_f1']:.3f}")
        if gap > 0.10:
            print("[참고] 확인용 점수가 0.1 이상 낮습니다. 확인용 세션의 조건(조명, 거리, 글씨 크기 등)이 "
                  "학습 세션에 없었을 수 있습니다. 그 조건으로 학습 세션을 더 찍어 보세요.")
    print(f"확인 페이지: {page}")


if __name__ == "__main__":
    main()
