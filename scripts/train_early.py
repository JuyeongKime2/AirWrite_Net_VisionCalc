"""조기 예측 학습 + 교차검증 분석.

글자를 다 쓰기 전(쓴 비율 20%~100%)에
    ① 무슨 기호인지 맞히는지 (분류 예측력)
    ② 나머지 궤적을 실제와 얼마나 비슷하게 그리는지 (좌표 오차)
를 여러 모델로 비교한다.

비교 모델
    분류   kNN(비딥러닝) | CNN(부분 그림, MNIST 사전학습) | GRU | LSTM | TCN(1D-CNN) | Transformer | 앙상블(CNN+최고 순서모델)
    궤적   직선 연장(비딥러닝) | kNN 템플릿(비딥러닝) | GRU | LSTM | TCN | Transformer

실행:
    python scripts/train_early.py                    # 전체 (CPU로 30분 안팎)
    python scripts/train_early.py --quick            # 빠르게 (3 fold, GRU+TCN만, 20 epoch)
    python scripts/train_early.py --encoders gru,transformer --folds 4

결과:  outputs/early/early_report.html (그래프, 표, 실제 vs 예측 궤적 갤러리)
모델:  models/early_<인코더>.pt, models/early_cnn.pt, models/early.json  (앱의 실시간 조기 예측에 사용)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 프로젝트 맨 위를 import 경로에 추가

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import train as T  # noqa: E402  (같은 scripts 폴더의 데이터 로드/분할/사전학습 함수 재사용)
from aircalc.config import (  # noqa: E402
    BATCH_SIZE,
    CLASSES,
    CNN_HP,
    COMPLETE_POINTS,
    CV_FOLDS,
    EARLY_ENCODERS,
    EARLY_EPOCHS,
    EARLY_HP,
    EARLY_POINTS,
    ENSEMBLE_WEIGHTS,
    EVAL_FRACTIONS,
    LABEL_SMOOTHING,
    MODEL_DIR,
    NUM_CLASSES,
    OPERATORS,
    OUTPUT_DIR,
    PATIENCE,
    RANDOM_SEED,
    SYNTHETIC_PER_OPERATOR,
)
from aircalc.early import (  # noqa: E402
    KNNTemplate,
    augment_keep_scale,
    completion_errors,
    completion_target,
    cut_prefix,
    early_features,
    linear_extrapolation,
    merge_prefix_and_rest,
    origin_of,
    rest_strokes,
    sample_fraction,
    symbol_size,
    to_screen,
    xy_strokes,
)
from aircalc.early_models import EarlyNet, save_early  # noqa: E402
from aircalc.early_report import build_early_report  # noqa: E402
from aircalc.models import build_model, save_checkpoint  # noqa: E402
from aircalc.strokes import FEATURE_VERSION, render_image, synth_operator  # noqa: E402

DEVICE = T.DEVICE
VAL_FRACTIONS = (0.3, 0.5, 0.7, 1.0)
GALLERY_FRACTIONS = (0.3, 0.5, 0.7)
EARLY_PHASE = 0.7      # '일찍 맞히기' 점수: 쓴 비율 0.7 이하에서의 평균 정확도


# ============================================================
# 데이터셋 (매번 증강 -> 무작위 비율로 자르기)
# ============================================================
class EarlyDataset(Dataset):
    def __init__(self, strokes_list, labels, kind, seed):
        self.strokes, self.labels, self.kind = strokes_list, labels, kind
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        full = augment_keep_scale(self.strokes[i], self.rng)
        prefix = cut_prefix(full, sample_fraction(self.rng))
        y = int(self.labels[i])
        if self.kind == "cnn":
            return torch.from_numpy(render_image(prefix)[None].astype(np.float32)), y
        x = torch.from_numpy(early_features(prefix))
        target = torch.from_numpy(completion_target(full, origin_of(prefix)))
        return x, y, target


def inputs_at(kind, strokes_list, frac):
    prefixes = [cut_prefix(s, frac) for s in strokes_list]
    if kind == "cnn":
        X = np.stack([render_image(p)[None] for p in prefixes]).astype(np.float32)
    else:
        X = np.stack([early_features(p) for p in prefixes])
    return prefixes, X


@torch.no_grad()
def predict_seq(model, X, batch=256):
    model.eval()
    probs, trajs = [], []
    for i in range(0, len(X), batch):
        logits, traj = model(torch.from_numpy(X[i:i + batch]).to(DEVICE))
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        t = traj.cpu().numpy()
        t[..., 2] = 1 / (1 + np.exp(-t[..., 2]))           # 획 시작 logit -> 확률
        trajs.append(t)
    return np.concatenate(probs), np.concatenate(trajs)


@torch.no_grad()
def predict_cnn(model, X, batch=256):
    model.eval()
    return np.concatenate([torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(DEVICE)), 1).cpu().numpy()
                           for i in range(0, len(X), batch)])


# ============================================================
# 학습
# ============================================================
def _loader(dataset, seed):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed),
                      drop_last=(len(dataset) % BATCH_SIZE == 1))


def train_seq(encoder, strokes, labels, val, epochs, seed, verbose=False):
    """val: None 또는 {frac: (X, y, targets)}"""
    T.set_seed(seed)
    hp = EARLY_HP
    model = EarlyNet(encoder, NUM_CLASSES, EARLY_POINTS, COMPLETE_POINTS, hidden=hp["hidden"],
                     dropout=hp["dropout"]).to(DEVICE)
    loader = _loader(EarlyDataset(strokes, labels, "seq", seed), seed)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    coord_loss, start_loss = nn.SmoothL1Loss(), nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=hp["lr"] * 0.05)

    best, best_state, best_epoch, patience = (-1.0, float("inf")), None, epochs, PATIENCE
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, traj = model(xb)
            loss = ce(logits, yb) + hp["coord_weight"] * coord_loss(traj[..., :2], tb[..., :2]) \
                + hp["start_weight"] * start_loss(traj[..., 2], tb[..., 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        if val is None:
            continue
        accs, ades = [], []
        for X, y, tgt in val.values():
            p, tr = predict_seq(model, X)
            accs.append(float(np.mean(p.argmax(1) == y)))
            ades.append(float(np.linalg.norm(tr[..., :2] - tgt[..., :2], axis=2).mean()))
        score = (float(np.mean(accs)), float(np.mean(ades)))
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"      [{encoder}] epoch {epoch:3d} val_acc={score[0]:.3f} val_coord_err={score[1]:.3f}")
        if score[0] > best[0] + 1e-6 or (abs(score[0] - best[0]) <= 1e-6 and score[1] < best[1]):
            best, best_epoch, patience = score, epoch, PATIENCE
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best_epoch


def train_cnn(strokes, labels, val, epochs, seed, pretrained, verbose=False):
    T.set_seed(seed)
    model = build_model("cnn", NUM_CLASSES, dropout=CNN_HP["dropout"])
    if pretrained is not None:
        model.load_pretrained(pretrained, keep_rows={T.CLASS_INDEX[d]: int(d) for d in T.DIGITS})
    model.to(DEVICE)
    loader = _loader(EarlyDataset(strokes, labels, "cnn", seed), seed)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=CNN_HP["lr"], weight_decay=CNN_HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=CNN_HP["lr"] * 0.05)
    best, best_state, best_epoch, patience = -1.0, None, epochs, PATIENCE
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            ce(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        sched.step()
        if val is None:
            continue
        acc = float(np.mean([np.mean(predict_cnn(model, X).argmax(1) == y) for X, y in val.values()]))
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"      [cnn] epoch {epoch:3d} val_acc={acc:.3f}")
        if acc > best + 1e-6:
            best, best_epoch, patience = acc, epoch, PATIENCE
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best_epoch


def synthetic_screen(n_per_op, rng):
    strokes, labels = [], []
    for op in OPERATORS:
        for _ in range(n_per_op):
            strokes.append(to_screen(synth_operator(op, rng), rng))
            labels.append(T.CLASS_INDEX[op])
    return strokes, np.asarray(labels, np.int64)


# ============================================================
# 한 fold 평가
# ============================================================
def evaluate_test(test_idx, strokes, y, models, cnn, cnn_weight, main_enc, knn, fold, fractions):
    rows, gallery, evolution = [], {}, {}
    test_strokes = [strokes[i] for i in test_idx]
    fulls = [xy_strokes(s) for s in test_strokes]
    sizes = [symbol_size(s) for s in test_strokes]
    for frac in fractions:
        prefixes, Xseq = inputs_at("seq", test_strokes, frac)
        out = {enc: predict_seq(m, Xseq) for enc, m in models.items()}
        p_cnn = predict_cnn(cnn, inputs_at("cnn", test_strokes, frac)[1]) if cnn is not None else None
        for j, i in enumerate(test_idx):
            prefix, true_label = prefixes[j], int(y[i])
            origin = origin_of(prefix)
            target = completion_target(fulls[j], origin)
            row = {"fold": fold, "index": int(i), "frac": frac, "true": true_label}
            probs = {enc: out[enc][0][j] for enc in models}
            knn_probs, knn_traj = knn.predict(prefix, frac, NUM_CLASSES)
            probs["knn"] = knn_probs
            if p_cnn is not None:
                probs["cnn"] = p_cnn[j]
                probs["ensemble"] = cnn_weight * p_cnn[j] + (1 - cnn_weight) * out[main_enc][0][j]
            for name, p in probs.items():
                row[f"{name}_pred"] = int(p.argmax())
                row[f"{name}_conf"] = float(p.max())
                row[f"{name}_ptrue"] = float(p[true_label])
            if frac < 1.0:
                trajs = {enc: out[enc][1][j] for enc in models}
                trajs["knn"] = knn_traj
                trajs["linear"] = linear_extrapolation(prefix, frac)
                for name, tr in trajs.items():
                    for k, v in completion_errors(tr, target, frac, sizes[j]).items():
                        row[f"{name}_{k}"] = v
                if cnn is not None:            # 예측해서 완성한 글자를 CNN이 원래 기호로 읽는가?
                    for enc in models:
                        merged = merge_prefix_and_rest(prefix, out[enc][1][j], frac)
                        img = render_image(merged)[None, None].astype(np.float32)
                        row[f"{enc}_consistent"] = int(predict_cnn(cnn, img)[0].argmax() == true_label)
                if frac in GALLERY_FRACTIONS:
                    gallery[(int(i), frac)] = {
                        "prefix": prefix, "full": fulls[j],
                        "pred": rest_strokes(prefix, out[main_enc][1][j], frac),
                        "knn": rest_strokes(prefix, knn_traj, frac),
                        "linear": rest_strokes(prefix, trajs["linear"], frac),
                        "pred_label": int(probs[main_enc].argmax()), "conf": float(probs[main_enc].max()),
                    }
            evolution.setdefault(int(i), {})[frac] = {k: float(v[true_label]) for k, v in probs.items()}
            rows.append(row)
    return rows, gallery, evolution


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="조기 예측 학습 + 교차검증 분석")
    parser.add_argument("--encoders", default=",".join(EARLY_ENCODERS), help="비교할 인코더 (쉼표로 구분)")
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--epochs", type=int, default=EARLY_EPOCHS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-cnn", action="store_true", help="부분 그림 CNN과 앙상블 생략 (빠름)")
    parser.add_argument("--quick", action="store_true", help="3 fold, GRU+TCN, 20 epoch")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.folds, args.encoders, args.epochs = 3, "gru,tcn", 20
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    out_dir = OUTPUT_DIR / "early"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    strokes, y, subjects, sessions, paths = T.load_dataset()
    T.print_summary(y, sessions)
    T.check_counts(y)
    groups, protocol = T.choose_groups(subjects, sessions)
    pretrained = None if args.no_cnn else T.mnist_pretrained(seed=args.seed)
    folds = T.make_folds(y, groups, args.folds, args.seed)
    main_enc = "gru" if "gru" in encoders else encoders[0]

    all_rows, gallery, evolution, weights = [], {}, {}, []
    best_epochs = {e: [] for e in encoders + ["cnn"]}
    fold_info = []
    for k, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n=== Fold {k}/{len(folds)}")
        fit_idx, val_idx = T.split_validation(train_idx, y, groups, args.seed + k)
        rng = np.random.default_rng(args.seed + k)
        syn_s, syn_y = synthetic_screen(SYNTHETIC_PER_OPERATOR, rng)
        train_s = [strokes[i] for i in fit_idx] + syn_s
        train_y = np.concatenate([y[fit_idx], syn_y])

        val_seq = val_cnn = None
        if val_idx is not None:
            val_strokes = [strokes[i] for i in val_idx]
            val_seq, val_cnn = {}, {}
            for f in VAL_FRACTIONS:
                prefixes, X = inputs_at("seq", val_strokes, f)
                tgt = np.stack([completion_target(s, origin_of(p)) for s, p in zip(val_strokes, prefixes)])
                val_seq[f] = (X, y[val_idx], tgt)
                val_cnn[f] = (inputs_at("cnn", val_strokes, f)[1], y[val_idx])

        models = {}
        for enc in encoders:
            t0 = time.time()
            models[enc], ep = train_seq(enc, train_s, train_y, val_seq, args.epochs, args.seed + k, args.verbose)
            best_epochs[enc].append(ep if val_idx is not None else args.epochs)
            print(f"  {enc:12s} 학습 완료 ({time.time() - t0:.0f}s, best epoch {ep})")
        cnn, w = None, None
        if not args.no_cnn:
            t0 = time.time()
            cnn, ep = train_cnn(train_s, train_y, val_cnn, args.epochs, args.seed + k, pretrained, args.verbose)
            best_epochs["cnn"].append(ep if val_idx is not None else args.epochs)
            print(f"  {'cnn':12s} 학습 완료 ({time.time() - t0:.0f}s, best epoch {ep})")
            w = 0.5
            if val_idx is not None:   # 앙상블 비중: 검증 세트 비율별 평균 정확도가 가장 높은 값
                scores = []
                for cand in ENSEMBLE_WEIGHTS:
                    accs = [np.mean((cand * predict_cnn(cnn, val_cnn[f][0])
                                     + (1 - cand) * predict_seq(models[main_enc], val_seq[f][0])[0]).argmax(1)
                                    == y[val_idx]) for f in VAL_FRACTIONS]
                    scores.append((float(np.mean(accs)), -abs(cand - 0.5), cand))
                w = max(scores)[2]
            weights.append(w)

        knn = KNNTemplate(train_s, train_y)
        rows, gal, evo = evaluate_test(test_idx, strokes, y, models, cnn, w, main_enc, knn, k, EVAL_FRACTIONS)
        all_rows += rows
        gallery.update(gal)
        evolution.update(evo)
        df_fold = pd.DataFrame(rows)
        acc50 = {n: float((df_fold[df_fold.frac == 0.5][f"{n}_pred"] == df_fold[df_fold.frac == 0.5]["true"]).mean())
                 for n in (["knn"] + (["cnn", "ensemble"] if cnn is not None else []) + encoders)}
        print("  50%만 썼을 때 정확도: " + " | ".join(f"{n} {v:.2f}" for n, v in acc50.items()))
        fold_info.append({"fold": k, "test": sorted(set(groups[test_idx].tolist())) if groups is not None else ["random"],
                          "weight": w})

    df = pd.DataFrame(all_rows)
    df["session"] = [str(sessions[i]) for i in df["index"]]
    df["path"] = [paths[i] for i in df["index"]]
    df.to_csv(out_dir / "early_predictions.csv", index=False, encoding="utf-8-sig")

    classifiers = ["knn"] + (["cnn"] if not args.no_cnn else []) + encoders + (["ensemble"] if not args.no_cnn else [])
    completers = ["linear", "knn"] + encoders

    # '일찍 맞히기' 점수로 앱에 쓸 순서 모델 선택 (같으면 궤적 오차가 작은 쪽)
    early_scores = {}
    for enc in encoders:
        d = df[df.frac <= EARLY_PHASE]
        early_scores[enc] = (float((d[f"{enc}_pred"] == d["true"]).mean()),
                             -float(df[df.frac < 1][f"{enc}_ade_rel"].mean()))
    chosen = max(encoders, key=lambda e: early_scores[e])
    print(f"\n앱에 쓸 순서 모델: {chosen} (쓴 비율 {int(EARLY_PHASE * 100)}% 이하 평균 정확도 {early_scores[chosen][0]:.3f})")

    # ---------------- 최종 모델 (전체 데이터) ----------------
    rng = np.random.default_rng(args.seed)
    syn_s, syn_y = synthetic_screen(SYNTHETIC_PER_OPERATOR, rng)
    all_s, all_y = strokes + syn_s, np.concatenate([y, syn_y])
    epochs = max(10, int(np.median(best_epochs[chosen])))
    print(f"최종 {chosen} 학습 ({epochs} epoch)")
    final_seq, _ = train_seq(chosen, all_s, all_y, None, epochs, args.seed)
    hp = EARLY_HP
    save_early(MODEL_DIR / f"early_{chosen}.pt", final_seq, {
        "encoder": chosen, "num_classes": NUM_CLASSES, "classes": CLASSES, "length": EARLY_POINTS,
        "out_points": COMPLETE_POINTS, "hidden": hp["hidden"], "dropout": hp["dropout"],
        "feature_version": FEATURE_VERSION})
    info = {"seq": f"early_{chosen}.pt", "encoder": chosen, "cnn": None, "cnn_weight": 0.0, "classes": CLASSES,
            "feature_version": FEATURE_VERSION, "evaluation_protocol": protocol, "cv_folds": len(folds),
            "trained_at": datetime.now().isoformat(timespec="seconds")}
    if not args.no_cnn:
        cnn_epochs = max(10, int(np.median(best_epochs["cnn"])))
        print(f"최종 CNN 학습 ({cnn_epochs} epoch)")
        final_cnn, _ = train_cnn(all_s, all_y, None, cnn_epochs, args.seed, pretrained)
        save_checkpoint(MODEL_DIR / "early_cnn.pt", final_cnn, {
            "kind": "cnn", "num_classes": NUM_CLASSES, "classes": CLASSES,
            "model_kwargs": {"dropout": CNN_HP["dropout"]}, "feature_version": FEATURE_VERSION})
        info.update({"cnn": "early_cnn.pt", "cnn_weight": float(np.median(weights)),
                     "ensemble_seq_encoder_in_cv": main_enc})

    page = build_early_report(out_dir, df, classifiers, completers, encoders, main_enc, chosen, gallery, evolution,
                              protocol, fold_info, title="허공 손글씨 조기 예측 분석")
    info["cv_summary_path"] = str((out_dir / "early_summary.json").relative_to(OUTPUT_DIR.parent))
    (MODEL_DIR / "early.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    table = pd.DataFrame({n: [(df[df.frac == f][f"{n}_pred"] == df[df.frac == f]["true"]).mean() for f in EVAL_FRACTIONS]
                          for n in classifiers}, index=[f"{int(f * 100)}%" for f in EVAL_FRACTIONS]).T
    print("쓴 비율별 정확도 (모든 fold 합산)")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\n분석 리포트: {page}")
    print(f"모델 저장: {MODEL_DIR}/early.json  (총 {time.time() - start_time:.0f}초)")


if __name__ == "__main__":
    main()