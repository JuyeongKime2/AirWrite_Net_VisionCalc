"""조기 예측 분석 리포트: 그래프 + 표 + 실제 vs 예측 궤적 갤러리 -> HTML.

그래프 안 글자는 영어 (matplotlib 기본 글꼴에 한글이 없어 깨짐), 설명은 HTML에 한국어로.
"""

import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ASCII, CLASSES, EVAL_FRACTIONS, NUM_CLASSES  # noqa: E402
from .early import earliest_stable  # noqa: E402

NAMES = {"knn": "kNN template", "cnn": "CNN (partial image)", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN (1D-CNN)",
         "transformer": "Transformer", "ensemble": "Ensemble (CNN+seq)", "linear": "Linear extrapolation"}
KO = {"knn": "kNN 템플릿 (비딥러닝)", "cnn": "CNN (부분 그림)", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN (1D-CNN)",
      "transformer": "Transformer", "ensemble": "앙상블 (CNN + 순서 모델)", "linear": "직선 연장 (비딥러닝)"}
STYLE = {"knn": ("#9a9a9a", "--"), "linear": ("#c9a227", ":"), "cnn": ("#4c72b0", "-"), "gru": ("#dd8452", "-"),
         "lstm": ("#55a868", "-"), "tcn": ("#8172b3", "-"), "transformer": ("#c44e52", "-"), "ensemble": ("#111111", "-")}


def _style(name):
    return STYLE.get(name, ("#777777", "-"))


def _per_fold(df, fn):
    """fold별로 계산한 값의 평균과 표준편차."""
    vals = [fn(g) for _, g in df.groupby("fold")]
    vals = [v for v in vals if v == v]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def _acc(g, name):
    return float((g[f"{name}_pred"] == g["true"]).mean())


def _fmt(mean, std, pct=True):
    if mean != mean:
        return "-"
    return f"{mean * 100:.1f}% ± {std * 100:.1f}" if pct else f"{mean:.1f} ± {std:.1f}"


# ============================================================
# 그래프
# ============================================================
def _line_plot(path, series, xlabel, ylabel, title, ylim=None, percent=True):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, (xs, means, stds) in series.items():
        color, ls = _style(name)
        means, stds = np.asarray(means), np.asarray(stds)
        ax.plot(xs, means, ls, color=color, lw=2.4 if name == "ensemble" else 1.8, marker="o", ms=4,
                label=NAMES.get(name, name))
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(EVAL_FRACTIONS), [f"{int(f * 100)}%" for f in EVAL_FRACTIONS])
    if ylim:
        ax.set_ylim(*ylim)
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_strokes(ax, strokes, **kw):
    for s in strokes:
        s = np.asarray(s)[:, -2:]
        if len(s) == 1 or np.linalg.norm(np.diff(s, axis=0), axis=1).sum() < 1e-3:
            ax.plot(s[:, 0].mean(), s[:, 1].mean(), "o", color=kw.get("color"), ms=kw.get("ms", 5))
        else:
            ax.plot(s[:, 0], s[:, 1], **{k: v for k, v in kw.items() if k != "ms"})


def _setup_cell(ax, full):
    pts = np.concatenate([np.asarray(s)[:, -2:] for s in full])
    lo, hi = pts.min(0), pts.max(0)
    pad = max((hi - lo).max() * 0.35, 0.02)
    c = (lo + hi) / 2
    half = (hi - lo).max() / 2 + pad
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] + half, c[1] - half)                 # 화면 좌표는 y가 아래로 증가
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _gallery(path, items, fractions, title):
    if not items:
        return False
    fig, axes = plt.subplots(len(items), len(fractions), figsize=(2.6 * len(fractions), 2.6 * len(items)),
                             squeeze=False)
    for r, (label, cells) in enumerate(items):
        for c, frac in enumerate(fractions):
            ax = axes[r][c]
            g = cells[frac]
            _setup_cell(ax, g["full"])
            _draw_strokes(ax, g["full"], color="#cfcfcf", lw=6, solid_capstyle="round", ms=9)
            _draw_strokes(ax, g["pred"], color="#d62728", lw=1.8, ls="--", ms=5)
            _draw_strokes(ax, g["prefix"], color="#1f4e9c", lw=3.2, ms=7)
            ok = g["pred_label"] == label
            ax.set_title(f"{int(frac * 100)}% -> {ASCII[CLASSES[g['pred_label']]]} ({g['conf']:.0%})",
                         fontsize=9, color="#1a7f37" if ok else "#c0392b")
            if c == 0:
                ax.set_ylabel(f"true: {ASCII[CLASSES[label]]}", fontsize=10)
    fig.suptitle(title + "\n(gray = actual full, blue = written so far, red dashed = predicted rest)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / (2.6 * len(items))))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _compare_gallery(path, items, frac):
    if not items:
        return False
    methods = [("linear", "Linear extrapolation"), ("knn", "kNN template"), ("pred", "Deep model")]
    fig, axes = plt.subplots(len(items), 3, figsize=(7.8, 2.6 * len(items)), squeeze=False)
    for r, (label, g) in enumerate(items):
        for c, (key, name) in enumerate(methods):
            ax = axes[r][c]
            _setup_cell(ax, g["full"])
            _draw_strokes(ax, g["full"], color="#cfcfcf", lw=6, solid_capstyle="round", ms=9)
            _draw_strokes(ax, g[key], color="#d62728", lw=1.8, ls="--", ms=5)
            _draw_strokes(ax, g["prefix"], color="#1f4e9c", lw=3.2, ms=7)
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"true: {ASCII[CLASSES[label]]}", fontsize=10)
    fig.suptitle(f"Predicted rest of the symbol at {int(frac * 100)}% written", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (2.6 * len(items))))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ============================================================
# 리포트
# ============================================================
def build_early_report(out_dir, df, classifiers, completers, encoders, main_enc, chosen, gallery, evolution,
                       protocol, fold_info, title="조기 예측 분석"):
    fracs = list(EVAL_FRACTIONS)
    part = [f for f in fracs if f < 1.0]
    summary = {"protocol": protocol, "folds": fold_info, "n_samples": int(df["index"].nunique()),
               "accuracy": {}, "trajectory": {}, "consistency": {}}

    # 1. 쓴 비율별 정확도
    acc_series = {}
    for name in classifiers:
        stats = [_per_fold(df[df.frac == f], lambda g, n=name: _acc(g, n)) for f in fracs]
        acc_series[name] = (fracs, [m for m, _ in stats], [s for _, s in stats])
        summary["accuracy"][name] = {f"{int(f * 100)}%": {"mean": m, "std": s} for f, (m, s) in zip(fracs, stats)}
    _line_plot(out_dir / "accuracy_vs_fraction.png", acc_series, "Fraction of the symbol written",
               "Accuracy (CV mean ± std)", "Early classification: how early can each model tell the symbol?",
               ylim=(0, 1.02))

    # 2. 궤적 오차
    traj_series, cham_series = {}, {}
    for name in completers:
        ade = [_per_fold(df[df.frac == f], lambda g, n=name: float(g[f"{n}_ade_rel"].mean())) for f in part]
        cham = [_per_fold(df[df.frac == f], lambda g, n=name: float(g[f"{n}_chamfer_rel"].mean())) for f in part]
        px = [float(df[df.frac == f][f"{name}_ade_px"].mean()) for f in part]
        strokes_ok = [float(df[df.frac == f][f"{name}_stroke_count_ok"].mean()) for f in part]
        traj_series[name] = (part, [m for m, _ in ade], [s for _, s in ade])
        cham_series[name] = (part, [m for m, _ in cham], [s for _, s in cham])
        summary["trajectory"][name] = {f"{int(f * 100)}%": {"ade_rel": a[0], "ade_rel_std": a[1], "chamfer_rel": c[0],
                                                             "ade_px": p, "stroke_count_ok": so}
                                       for f, a, c, p, so in zip(part, ade, cham, px, strokes_ok)}
    _line_plot(out_dir / "trajectory_error_vs_fraction.png", traj_series, "Fraction of the symbol written",
               "Mean distance of remaining points\n(% of symbol size, lower is better)",
               "Trajectory prediction: predicted rest vs actually drawn rest")
    _line_plot(out_dir / "shape_error_vs_fraction.png", cham_series, "Fraction of the symbol written",
               "Chamfer distance (% of symbol size)", "Completed shape vs actual shape (order-free)")

    # 3. 예측해서 완성한 글자가 원래 기호로 읽히는가
    cons_cols = [e for e in encoders if f"{e}_consistent" in df.columns]
    if cons_cols:
        series = {}
        for enc in cons_cols:
            stats = [_per_fold(df[df.frac == f], lambda g, e=enc: float(g[f"{e}_consistent"].mean())) for f in part]
            series[enc] = (part, [m for m, _ in stats], [s for _, s in stats])
            summary["consistency"][enc] = {f"{int(f * 100)}%": m for f, (m, _) in zip(part, stats)}
        _line_plot(out_dir / "consistency_vs_fraction.png", series, "Fraction of the symbol written",
                   "Completed symbol read as the true class", "Does the predicted completion look like the right symbol?",
                   ylim=(0, 1.02))

    # 4. 기호별 '이 비율부터 쭉 맞힘'
    best_clf = "ensemble" if "ensemble" in classifiers else chosen
    earliest = []
    for idx, g in df.groupby("index"):
        g = g.sort_values("frac")
        e = earliest_stable((g[f"{best_clf}_pred"] == g["true"]).tolist(), g["frac"].tolist())
        earliest.append({"index": idx, "true": int(g["true"].iloc[0]), "earliest": e})
    ear = pd.DataFrame(earliest)
    by_class = ear.groupby("true")["earliest"].agg(["mean", "count", lambda s: s.isna().mean()])
    by_class.columns = ["mean", "count", "never"]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [ASCII[CLASSES[c]] for c in by_class.index]
    ax.bar(labels, by_class["mean"], color=["#4c72b0" if CLASSES[c].isdigit() else "#dd8452" for c in by_class.index])
    for i, (m, nv) in enumerate(zip(by_class["mean"], by_class["never"])):
        ax.text(i, (m if m == m else 0) + 0.02, f"{m:.0%}" if m == m else "-", ha="center", fontsize=8)
        if nv > 0:
            ax.text(i, 0.04, f"miss {nv:.0%}", ha="center", fontsize=7, color="white", rotation=90)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Written fraction from which\nthe prediction stays correct")
    ax.set_title(f"How early each symbol is recognized ({NAMES.get(best_clf, best_clf)}, lower = earlier)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "earliest_by_class.png", dpi=150)
    plt.close(fig)
    summary["earliest_by_class"] = {ASCII[CLASSES[c]]: {"mean": float(r["mean"]), "never": float(r["never"])}
                                    for c, r in by_class.iterrows()}
    summary["earliest_overall"] = float(ear["earliest"].mean())

    # 5. 확신도 변화 (예시 6개)
    picks, seen = [], set()
    for idx in sorted(evolution):
        label = int(df[df["index"] == idx]["true"].iloc[0])
        if label not in seen:
            picks.append((idx, label))
            seen.add(label)
        if len(picks) == 6:
            break
    if picks:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
        for ax, (idx, label) in zip(axes.ravel(), picks):
            evo = evolution[idx]
            for name in classifiers:
                ys = [evo[f].get(name, np.nan) for f in fracs]
                color, ls = _style(name)
                ax.plot(fracs, ys, ls, color=color, lw=2.2 if name == "ensemble" else 1.4, label=NAMES.get(name, name))
            ax.set_title(f"true: {ASCII[CLASSES[label]]}")
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0.2, 0.5, 0.8, 1.0], ["20%", "50%", "80%", "100%"])
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("P(true symbol)")
        axes[1][0].set_ylabel("P(true symbol)")
        handles, lbls = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=8)
        fig.suptitle("Confidence in the true symbol while it is being written")
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        fig.savefig(out_dir / "confidence_evolution.png", dpi=150)
        plt.close(fig)

    # 6. 50%만 썼을 때 혼동행렬
    d50 = df[df.frac == 0.5]
    cm = pd.crosstab(d50["true"], d50[f"{best_clf}_pred"]).reindex(index=range(NUM_CLASSES), columns=range(NUM_CLASSES),
                                                                    fill_value=0).values
    totals = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
    ticks = [ASCII[c] for c in CLASSES]
    ax.set_xticks(range(NUM_CLASSES), ticks)
    ax.set_yticks(range(NUM_CLASSES), ticks)
    ax.set_xlabel("Predicted at 50% written")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion at 50% written ({NAMES.get(best_clf, best_clf)})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_at_50.png", dpi=150)
    plt.close(fig)

    # 7. 갤러리
    gfr = sorted({f for _, f in gallery})
    per_label = {}
    for (idx, frac), g in sorted(gallery.items()):
        label = int(df[df["index"] == idx]["true"].iloc[0])
        per_label.setdefault(label, {}).setdefault(idx, {})[frac] = g
    chosen_items = []
    for label in sorted(per_label):
        for idx, cells in per_label[label].items():
            if all(f in cells for f in gfr):
                chosen_items.append((label, cells))
                break
    digits = [it for it in chosen_items if CLASSES[it[0]].isdigit()]
    ops = [it for it in chosen_items if not CLASSES[it[0]].isdigit()]
    has_g1 = _gallery(out_dir / "gallery_digits.png", digits, gfr, f"Digits: trajectory prediction ({chosen.upper()})")
    has_g2 = _gallery(out_dir / "gallery_operators.png", ops, gfr, f"Operators: trajectory prediction ({chosen.upper()})")
    cmp_items = [(label, cells[0.5]) for label, cells in chosen_items if 0.5 in cells][::2][:6]
    has_cmp = _compare_gallery(out_dir / "compare_methods_50.png", cmp_items, 0.5)

    (out_dir / "early_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- HTML ----------------
    def acc_table():
        head = "".join(f"<th>{int(f * 100)}%</th>" for f in fracs)
        rows = ""
        for name in classifiers:
            cells = ""
            for f in fracs:
                cell = summary["accuracy"][name][f"{int(f * 100)}%"]
                cells += f"<td>{_fmt(cell['mean'], cell['std'])}</td>"
            rows += f"<tr><td>{html.escape(KO.get(name, name))}</td>{cells}</tr>"
        return f"<table><tr><th>모델</th>{head}</tr>{rows}</table>"

    def traj_table():
        head = "".join(f"<th>{int(f * 100)}%</th>" for f in part)
        rows = ""
        for name in completers:
            cells = ""
            for f in part:
                t = summary["trajectory"][name][f"{int(f * 100)}%"]
                cells += (f"<td>{t['ade_rel'] * 100:.1f}%<br><small>{t['ade_px']:.0f}px · 획수 "
                          f"{t['stroke_count_ok'] * 100:.0f}%</small></td>")
            rows += f"<tr><td>{html.escape(KO.get(name, name))}</td>{cells}</tr>"
        return f"<table><tr><th>방법</th>{head}</tr>{rows}</table>"

    def img(name, alt):
        return f'<figure><img src="{name}" alt="{html.escape(alt)}"></figure>' if (out_dir / name).exists() else ""

    best_note = KO.get(best_clf, best_clf)
    note = {"subject": "사람 단위", "session": "세션 단위", "random": "무작위", "holdout": "확인용 세션"}.get(protocol, protocol)
    acc50 = summary["accuracy"][best_clf]["50%"]["mean"] * 100
    acc100 = summary["accuracy"][best_clf]["100%"]["mean"] * 100
    ade50 = summary["trajectory"][chosen]["50%"]["ade_rel"] * 100
    ens_note = " + CNN 앙상블" if "ensemble" in classifiers else ""
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--mute:#677085;--card:#fff;--line:#e2e5eb;--acc:#2f6fd6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171d;--fg:#e6e9ef;--mute:#98a1b3;--card:#1d222b;--line:#2c3340;--acc:#6ea1ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 system-ui,-apple-system,"Malgun Gothic",sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px 16px 60px}}h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:19px;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}}
p,li{{color:var(--fg)}}.mute{{color:var(--mute)}}
table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);font-size:13px;display:block;overflow-x:auto}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--mute);font-weight:600}}small{{color:var(--mute)}}
figure{{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}}figure img{{width:100%;display:block}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}}
.kpi div{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}.kpi b{{font-size:22px;display:block}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="mute">{note}로 나눈 {len(fold_info)}-fold 평가 · 시험 글씨 {summary['n_samples']}개 · 쓴 비율 = 지금까지 그린 잉크 길이 ÷ 완성 글자 잉크 길이</p>
<div class="kpi">
<div><b>{acc50:.1f}%</b>절반만 썼을 때 정확도 ({html.escape(best_note)})</div>
<div><b>{acc100:.1f}%</b>다 썼을 때 정확도</div>
<div><b>{summary['earliest_overall'] * 100:.0f}%</b>평균적으로 이만큼 쓰면 이후 계속 맞힘</div>
<div><b>{ade50:.1f}%</b>50%일 때 남은 궤적 오차 (글자 크기 대비, {chosen.upper()})</div>
</div>

<h2>1. 얼마나 일찍 맞히나 (분류 예측력)</h2>
<p>글자를 20%, 30%, ... 100% 썼을 때 각 모델이 최종 기호를 맞힌 비율입니다. 선이 왼쪽 위로 빨리 올라갈수록 일찍 맞히는 모델입니다. 음영은 fold 간 표준편차입니다.</p>
{img("accuracy_vs_fraction.png", "accuracy")}
{acc_table()}

<h2>2. 남은 궤적을 얼마나 비슷하게 예측하나 (좌표 오차)</h2>
<p>앞부분만 보고 예측한 나머지 궤적과 실제로 마저 그린 궤적을, 같은 진행 위치의 점끼리 비교한 평균 거리입니다.
글자 크기 대비 %(작을수록 좋음)와 화면 픽셀(480px 기준), 획 수까지 맞힌 비율을 함께 보여줍니다.
비딥러닝 기준선인 <b>직선 연장</b>(쓴 비율을 미리 알려주는 유리한 조건)과 <b>kNN 템플릿</b>과 비교합니다.</p>
{img("trajectory_error_vs_fraction.png", "trajectory error")}
{traj_table()}
{img("shape_error_vs_fraction.png", "shape error")}
<p class="mute">Chamfer 거리: 점 순서와 상관없이 두 모양이 얼마나 겹치는지. 순서는 틀려도 모양이 맞으면 작게 나옵니다.</p>

<h2>3. 실제 궤적 vs 예측 궤적</h2>
<p>회색 = 실제로 끝까지 그린 글자, 파랑 = 지금까지 그린 부분, 빨간 점선 = 모델이 예측한 나머지. 제목은 그 시점의 예측 기호와 확신도입니다 (초록 = 정답).</p>
{img("gallery_digits.png", "digits gallery") if has_g1 else ""}
{img("gallery_operators.png", "operators gallery") if has_g2 else ""}
{"<p>같은 글씨를 50% 썼을 때 세 방법이 예측한 나머지 궤적입니다.</p>" + img("compare_methods_50.png", "methods") if has_cmp else ""}

<h2>4. 예측해서 완성한 글자가 진짜 그 기호로 보이나</h2>
<p>실제로 그린 앞부분에 모델이 예측한 나머지를 이어 붙여 완성한 그림을 CNN에게 보여주고, 원래 기호로 읽는 비율입니다. 높을수록 예측 궤적이 '그럴듯한' 글씨라는 뜻입니다.</p>
{img("consistency_vs_fraction.png", "consistency")}

<h2>5. 기호별로 언제부터 확신하나</h2>
<p>각 글씨에서 '이 비율부터는 끝까지 계속 맞힌' 가장 이른 시점의 평균입니다. 낮을수록 빨리 알아보는 기호입니다 (파랑 = 숫자, 주황 = 연산자). miss는 끝까지 안정적으로 맞히지 못한 비율입니다.</p>
{img("earliest_by_class.png", "earliest")}
{img("confidence_evolution.png", "confidence")}
<p class="mute">위: 글씨 하나를 쓰는 동안 각 모델이 정답 기호에 준 확률의 변화.</p>

<h2>6. 절반만 썼을 때 헷갈리는 기호</h2>
{img("confusion_at_50.png", "confusion")}

<h2>파일</h2>
<ul><li><code>early_predictions.csv</code>: 글씨 × 쓴 비율마다 모든 모델의 예측, 확신도, 궤적 오차</li>
<li><code>early_summary.json</code>: 이 페이지의 표 숫자</li>
<li>앱에서 쓰는 모델: <b>{chosen.upper()}</b>{ens_note} (쓴 비율 70% 이하 정확도가 가장 높은 순서 모델)</li></ul>
</main></body></html>"""
    out = out_dir / "early_report.html"
    out.write_text(page, encoding="utf-8")
    return out
