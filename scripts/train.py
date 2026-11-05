"""허공 손글씨 인식 모델 학습 + 교차검증.

실행:
    python scripts/train.py                 # MNIST 사전학습 CNN + GRU + 앙상블
    python scripts/train.py --ablation      # 'MNIST 사전학습 없는 CNN'도 같이 학습해 사전학습 효과 비교
    python scripts/train.py --no-synthetic  # 합성 연산자 없이 (효과 비교용)
    python scripts/train.py --verbose

순서
    1. MNIST(손글씨 숫자 7만 장)로 CNN 사전학습 (처음 한 번, data/pretrained에 저장해 재사용)
    2. 세션 단위 교차검증: fold마다
         학습 세션(+합성 연산자)으로 CNN 미세조정, GRU 학습
         검증 세션으로 학습 중단 시점과 앙상블 비중(CNN : GRU) 결정
         시험 세션으로 CNN / GRU / 앙상블 각각 채점
    3. 전체 데이터로 최종 모델 학습 -> models/cnn.pt, models/gru.pt, models/ensemble.json
    4. outputs/test_samples.html: 시험에 쓰인 글씨마다 궤적, 정답, 세 모델의 예측
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 프로젝트 맨 위를 import 경로에 추가

import argparse
import copy
import json
import random
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from aircalc.config import (  # noqa: E402
    ASCII,
    BATCH_SIZE,
    CLASSES,
    CNN_HP,
    CV_FOLDS,
    DIGITS,
    ENSEMBLE_WEIGHTS,
    EPOCHS,
    LABEL_SMOOTHING,
    MIN_SAMPLES_PER_CLASS,
    MIN_SUBJECTS_FOR_SUBJECT_SPLIT,
    MNIST_PRETRAIN_EPOCHS,
    MODEL_DIR,
    NUM_CLASSES,
    OPERATORS,
    OUTPUT_DIR,
    PATIENCE,
    PRETRAIN_DIR,
    RANDOM_SEED,
    SEQ_HP,
    SESSIONS_DIR,
    SYNTHETIC_PER_OPERATOR,
)
from aircalc.models import build_model, ensemble, save_checkpoint  # noqa: E402
from aircalc.report import build_report  # noqa: E402
from aircalc.strokes import FEATURE_VERSION, augment, load_sample, render_image, seq_features, synth_operator  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
DIGIT_IDX = [CLASS_INDEX[d] for d in DIGITS]
OP_IDX = [CLASS_INDEX[o] for o in OPERATORS]
HP = {"cnn": CNN_HP, "gru": SEQ_HP}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else (obj.tolist() if isinstance(obj, np.ndarray) else str(obj))


# ============================================================
# 데이터
# ============================================================
def load_dataset(root=None):
    root = SESSIONS_DIR if root is None else root
    if not root.exists():
        raise RuntimeError(f"데이터 폴더가 없습니다: {root}\n먼저 python scripts/collect_data.py 로 수집하세요.")
    strokes, labels, subjects, sessions, paths = [], [], [], [], []
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        info_path = session_dir / "session_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        subject = str(info.get("subject") or f"unknown_{session_dir.name}")
        for cls in CLASSES:
            for path in sorted((session_dir / cls).glob("*.npz")):
                try:
                    sample, label = load_sample(path)
                except Exception as exc:
                    print(f"[제외] {path}: {exc}")
                    continue
                strokes.append(sample)
                labels.append(CLASS_INDEX[label])
                subjects.append(subject)
                sessions.append(session_dir.name)
                paths.append(str(path))
    if not strokes:
        raise RuntimeError(f"샘플이 없습니다: {root}")
    return strokes, np.asarray(labels, np.int64), np.asarray(subjects), np.asarray(sessions), paths


def print_summary(y, sessions):
    table = pd.crosstab(pd.Series(sessions, name="session"), pd.Series([ASCII[CLASSES[v]] for v in y], name="class"))
    table = table.reindex(columns=[ASCII[c] for c in CLASSES], fill_value=0)
    print(f"전체 {len(y)}개, 세션 {table.shape[0]}개")
    print(table.to_string())


def check_counts(y):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    short = [f"{ASCII[CLASSES[i]]}={counts[i]}" for i in range(NUM_CLASSES) if counts[i] < MIN_SAMPLES_PER_CLASS]
    if short:
        raise RuntimeError(f"기호별 최소 {MIN_SAMPLES_PER_CLASS}개가 필요합니다. 부족: {', '.join(short)}")


def choose_groups(subjects, sessions):
    if len(np.unique(subjects)) >= MIN_SUBJECTS_FOR_SUBJECT_SPLIT:
        return subjects, "subject"
    if len(np.unique(sessions)) >= 2:
        print("[안내] 피험자가 적어 세션 단위로 나눠 평가합니다 (혼자 수집하면 정상).")
        return sessions, "session"
    print("[경고] 세션이 1개뿐이라 무작위로 나눕니다. 점수가 크게 부풀려집니다. 세션을 더 찍으세요.")
    return None, "random"


def make_folds(y, groups, n_folds, seed):
    if groups is None:
        k = max(2, min(n_folds, int(np.bincount(y).min())))
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    k = min(n_folds, len(np.unique(groups)))
    if k < 2:
        raise RuntimeError("교차검증에 세션이 2개 이상 필요합니다.")
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def split_validation(train_idx, y, groups, seed, ratio=0.2):
    rng = np.random.default_rng(seed)
    needed = set(np.unique(y[train_idx]).tolist())
    if groups is None:
        for _ in range(200):
            mask = rng.random(len(train_idx)) < ratio
            if set(y[train_idx[mask]].tolist()) == needed and (~mask).any():
                return train_idx[~mask], train_idx[mask]
        return train_idx, None
    unique = np.unique(groups[train_idx])
    if len(unique) < 3:
        return train_idx, None
    n_val = max(1, int(round(len(unique) * ratio)))
    for _ in range(200):
        picked = rng.choice(unique, n_val, replace=False)
        in_val = np.isin(groups[train_idx], picked)
        if set(y[train_idx[in_val]].tolist()) == needed:
            return train_idx[~in_val], train_idx[in_val]
    return train_idx, None


def make_synthetic(n_per_op, rng):
    strokes, labels = [], []
    for op in OPERATORS:
        for _ in range(n_per_op):
            strokes.append(synth_operator(op, rng))
            labels.append(CLASS_INDEX[op])
    return strokes, np.asarray(labels, np.int64)


# ============================================================
# 지표
# ============================================================
def compute_metrics(y_true, y_pred):
    present = np.unique(y_true)
    digit_mask, op_mask = np.isin(y_true, DIGIT_IDX), np.isin(y_true, OP_IDX)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "digit_accuracy": float(np.mean(y_true[digit_mask] == y_pred[digit_mask])) if digit_mask.any() else float("nan"),
        "operator_accuracy": float(np.mean(y_true[op_mask] == y_pred[op_mask])) if op_mask.any() else float("nan"),
    }


# ============================================================
# 모델 입력
# ============================================================
def to_input(kind, strokes):
    if kind == "cnn":
        return render_image(strokes)[None]           # (1, 28, 28)
    return seq_features(strokes)                      # (64, 6)


def precompute(kind, strokes_list):
    return np.stack([to_input(kind, s) for s in strokes_list]).astype(np.float32)


class StrokeDataset(Dataset):
    def __init__(self, strokes_list, labels, kind, seed):
        self.strokes, self.labels, self.kind = strokes_list, labels, kind
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = to_input(self.kind, augment(self.strokes[i], self.rng))
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.labels[i])


# ============================================================
# MNIST 사전학습
# ============================================================
def mnist_pretrained(epochs=MNIST_PRETRAIN_EPOCHS, seed=RANDOM_SEED):
    """MNIST로 학습한 CNN 가중치 (처음 한 번만 학습하고 저장). 실패하면 None."""
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    cache = PRETRAIN_DIR / "cnn_mnist.pt"
    if cache.exists():
        ckpt = torch.load(cache, map_location="cpu", weights_only=True)
        print(f"MNIST 사전학습 가중치 사용 (MNIST 시험 정확도 {ckpt['test_acc']:.4f})")
        return ckpt["state_dict"]
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("[주의] torchvision이 없어 MNIST 사전학습을 건너뜁니다: pip install -r requirements-dev.txt")
        return None
    try:
        aug = transforms.Compose([transforms.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                                  transforms.ToTensor()])
        train_set = datasets.MNIST(str(PRETRAIN_DIR), train=True, download=True, transform=aug)
        test_set = datasets.MNIST(str(PRETRAIN_DIR), train=False, download=True, transform=transforms.ToTensor())
    except Exception as exc:
        print(f"[주의] MNIST를 내려받지 못해 사전학습을 건너뜁니다 ({exc})")
        return None

    print(f"MNIST로 CNN 사전학습 중 ({epochs} epoch, 처음 한 번만)...")
    set_seed(seed)
    model = build_model("cnn", 10, dropout=CNN_HP["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in DataLoader(test_set, batch_size=512):
                correct += (model(xb.to(DEVICE)).argmax(1).cpu() == yb).sum().item()
        acc = correct / len(test_set)
        print(f"  epoch {epoch}: MNIST 시험 정확도 {acc:.4f}")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "test_acc": acc}, cache)
    return state


# ============================================================
# 학습
# ============================================================
def predict_proba(model, X, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out)


def fit(kind, strokes, labels, val_X, val_y, epochs, seed, pretrained=None, verbose=False):
    set_seed(seed)
    hp = HP[kind]
    kwargs = {k: hp[k] for k in ("hidden", "dropout") if k in hp}
    model = build_model(kind, NUM_CLASSES, **kwargs)
    if kind == "cnn" and pretrained is not None:
        model.load_pretrained(pretrained, keep_rows={CLASS_INDEX[d]: int(d) for d in DIGITS})
    model.to(DEVICE)

    ds = StrokeDataset(strokes, labels, kind, seed)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed),
                        drop_last=(len(ds) % BATCH_SIZE == 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=hp["lr"] * 0.05)

    best, best_state, best_epoch, patience = (-1.0, float("inf")), None, epochs, PATIENCE
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        if val_X is None:
            continue
        probs = predict_proba(model, val_X)
        f1 = f1_score(val_y, probs.argmax(1), labels=np.unique(val_y), average="macro", zero_division=0)
        nll = float(-np.mean(np.log(probs[np.arange(len(val_y)), val_y] + 1e-9)))
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"      [{kind}] epoch {epoch:3d} loss={np.mean(losses):.3f} val_f1={f1:.3f} val_nll={nll:.3f}")
        if f1 > best[0] + 1e-6 or (abs(f1 - best[0]) <= 1e-6 and nll < best[1]):
            best, best_epoch, patience = (f1, nll), epoch, PATIENCE
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def choose_weight(p_cnn, p_gru, y):
    """검증 세트에서 macro F1이 가장 높은 CNN 비중. 동점이면 0.5에 가까운 쪽."""
    scored = []
    for w in ENSEMBLE_WEIGHTS:
        pred = ensemble(p_cnn, p_gru, w).argmax(1)
        f1 = f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)
        scored.append((f1, -abs(w - 0.5), w))
    return max(scored)[2]


def measure_latency_ms(cnn, gru, strokes, runs=30):
    cnn, gru = copy.deepcopy(cnn).cpu().eval(), copy.deepcopy(gru).cpu().eval()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            cnn(torch.from_numpy(render_image(strokes))[None, None])
            gru(torch.from_numpy(seq_features(strokes))[None])
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


# ============================================================
# 시각화
# ============================================================
def plot_confusion(cm, title, path):
    labels = [ASCII[c] for c in CLASSES]
    totals = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(NUM_CLASSES), labels)
    ax.set_yticks(range(NUM_CLASSES), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(summary, protocol, path):
    names = list(summary)
    means = [summary[n]["macro_f1_mean"] for n in names]
    stds = [summary[n]["macro_f1_std"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#7b8db3", "#7fb38d", "#d18b5f", "#b0b0b0"]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[:len(names)])
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, min(m + 0.03, 1.02), f"{m:.3f}", ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro F1 (CV mean ± std)")
    ax.set_title(f"CNN vs GRU vs Ensemble ({protocol} CV)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="허공 손글씨 인식 학습 + 교차검증")
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ablation", action="store_true", help="MNIST 사전학습 없는 CNN도 학습해 비교")
    parser.add_argument("--no-synthetic", action="store_true", help="합성 연산자를 섞지 않음")
    parser.add_argument("--no-pretrain", action="store_true", help="MNIST 사전학습을 쓰지 않음")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    strokes, y, subjects, sessions, paths = load_dataset()
    print_summary(y, sessions)
    check_counts(y)
    groups, protocol = choose_groups(subjects, sessions)
    pretrained = None if args.no_pretrain else mnist_pretrained(seed=args.seed)
    n_synth = 0 if args.no_synthetic else SYNTHETIC_PER_OPERATOR

    model_names = ["cnn", "gru", "ensemble"] + (["cnn_scratch"] if args.ablation else [])
    folds = make_folds(y, groups, args.folds, args.seed)
    fold_rows = {n: [] for n in model_names}
    best_epochs = {"cnn": [], "gru": []}
    weights, records, fold_info = [], [], []
    pooled_true, pooled_pred = [], []

    for k, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n=== Fold {k}/{len(folds)}: 시험 = "
              f"{', '.join(sorted(set(groups[test_idx].tolist()))) if groups is not None else f'무작위 {len(test_idx)}개'}")
        fit_idx, val_idx = split_validation(train_idx, y, groups, args.seed + k)
        rng = np.random.default_rng(args.seed + k)
        syn_strokes, syn_y = make_synthetic(n_synth, rng) if n_synth else ([], np.zeros(0, np.int64))
        train_strokes = [strokes[i] for i in fit_idx] + syn_strokes
        train_y = np.concatenate([y[fit_idx], syn_y])

        test_strokes = [strokes[i] for i in test_idx]
        val_strokes = [strokes[i] for i in val_idx] if val_idx is not None else None
        probs, val_probs = {}, {}
        for kind, pre, name in [("cnn", pretrained, "cnn"), ("gru", None, "gru")] + \
                               ([("cnn", None, "cnn_scratch")] if args.ablation else []):
            val_X = precompute(kind, val_strokes) if val_strokes else None
            model, best_epoch = fit(kind, train_strokes, train_y, val_X, y[val_idx] if val_idx is not None else None,
                                    args.epochs, args.seed + k, pretrained=pre, verbose=args.verbose)
            if name in best_epochs:
                best_epochs[name].append(best_epoch if val_idx is not None else args.epochs)
            probs[name] = predict_proba(model, precompute(kind, test_strokes))
            if val_X is not None:
                val_probs[name] = predict_proba(model, val_X)

        w = choose_weight(val_probs["cnn"], val_probs["gru"], y[val_idx]) if val_idx is not None else 0.5
        weights.append(w)
        probs["ensemble"] = ensemble(probs["cnn"], probs["gru"], w)

        metrics = {}
        for name in model_names:
            m = compute_metrics(y[test_idx], probs[name].argmax(1))
            m["fold"] = k
            fold_rows[name].append(m)
            metrics[name] = m
        print("  " + " | ".join(f"{n} macroF1={metrics[n]['macro_f1']:.3f}" for n in model_names)
              + f" | 앙상블 비중 CNN:GRU = {w:.2f}:{1 - w:.2f}")

        pred_e = probs["ensemble"].argmax(1)
        pooled_true.append(y[test_idx])
        pooled_pred.append(pred_e)
        for j, i in enumerate(test_idx):
            pe = probs["ensemble"][j]
            order = np.argsort(pe)[::-1][:3]
            records.append({
                "index": int(i), "path": paths[i], "session": str(sessions[i]), "fold": k,
                "true": int(y[i]), "pred": int(pred_e[j]),
                "top3": [(int(c), float(pe[c])) for c in order],
                "cnn": int(probs["cnn"][j].argmax()), "cnn_conf": float(probs["cnn"][j].max()),
                "gru": int(probs["gru"][j].argmax()), "gru_conf": float(probs["gru"][j].max()),
            })

        def names_of(idx):
            if idx is None:
                return []
            if groups is None:
                return [f"무작위 {len(idx)}개"]
            return sorted(set(groups[idx].tolist()))

        fold_info.append({"fold": k, "test_groups": names_of(test_idx), "train_groups": names_of(fit_idx),
                          "val_groups": names_of(val_idx), "metrics": metrics["ensemble"], "weight": w})

    # ---------------- 요약 ----------------
    summary = {}
    for name in model_names:
        df = pd.DataFrame(fold_rows[name])
        summary[name] = {f"{c}_{s}": float(getattr(np, f"nan{s}")(df[c]))
                         for c in ("macro_f1", "accuracy", "digit_accuracy", "operator_accuracy") for s in ("mean", "std")}
    y_true, y_pred = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plot_confusion(cm, f"Ensemble (pooled {protocol} CV)", OUTPUT_DIR / "confusion_matrix.png")
    plot_comparison(summary, protocol, OUTPUT_DIR / "model_comparison.png")
    pd.DataFrame([{"model": n, **summary[n]} for n in model_names]).to_csv(
        OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---------------- 최종 모델 (전체 데이터) ----------------
    final_w = float(np.median(weights))
    rng = np.random.default_rng(args.seed)
    syn_strokes, syn_y = make_synthetic(n_synth, rng) if n_synth else ([], np.zeros(0, np.int64))
    all_strokes, all_y = strokes + syn_strokes, np.concatenate([y, syn_y])
    finals = {}
    for kind, pre in (("cnn", pretrained), ("gru", None)):
        epochs = max(10, int(np.median(best_epochs[kind])))
        print(f"\n최종 {kind.upper()} 학습: 실제 {len(y)}개 + 합성 {len(syn_y)}개, {epochs} epoch")
        finals[kind], _ = fit(kind, all_strokes, all_y, None, None, epochs, args.seed, pretrained=pre)
        hp = HP[kind]
        save_checkpoint(MODEL_DIR / f"{kind}.pt", finals[kind], {
            "kind": kind, "num_classes": NUM_CLASSES, "classes": CLASSES,
            "model_kwargs": {k: hp[k] for k in ("hidden", "dropout") if k in hp},
            "feature_version": FEATURE_VERSION, "epochs": epochs,
            "mnist_pretrained": bool(kind == "cnn" and pretrained is not None),
        })
    latency = measure_latency_ms(finals["cnn"], finals["gru"], strokes[0])
    info = {
        "cnn": "cnn.pt", "gru": "gru.pt", "cnn_weight": final_w, "classes": CLASSES,
        "feature_version": FEATURE_VERSION, "evaluation_protocol": protocol, "cv_folds": len(folds),
        "cv_summary": summary, "fold_weights": weights, "mnist_pretrained": pretrained is not None,
        "synthetic_per_operator": n_synth, "trained_on_samples": int(len(y)),
        "cpu_latency_ms_per_symbol": latency, "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "ensemble.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=_json_default),
                                             encoding="utf-8")

    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=[ASCII[c] for c in CLASSES], output_dict=True, zero_division=0)
    (OUTPUT_DIR / "cv_report.json").write_text(json.dumps(
        {**info, "folds": fold_info, "per_class": report, "confusion_matrix": cm.tolist()},
        ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    page = build_report(OUTPUT_DIR / "test_samples.html", "허공 손글씨 계산기", protocol, fold_info, records,
                        strokes, summary)

    print("\n" + "=" * 70)
    print(f"교차검증 결과 ({protocol} 단위, {len(folds)} fold)")
    print("=" * 70)
    for name in model_names:
        s = summary[name]
        print(f"{name:12s} macro F1 {s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} | "
              f"숫자 {s['digit_accuracy_mean']:.3f} | 연산자 {s['operator_accuracy_mean']:.3f}")
    print(f"앙상블 비중 CNN:GRU = {final_w:.2f}:{1 - final_w:.2f} | 글자당 CPU {latency:.1f}ms")
    print(f"\n시험 샘플 확인 페이지: {page}")
    print(f"모델 저장: {MODEL_DIR}/cnn.pt, gru.pt, ensemble.json")


if __name__ == "__main__":
    main()
