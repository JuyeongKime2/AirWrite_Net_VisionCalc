"""시험에 쓰인 글씨를 하나하나 보여주는 HTML 리포트.

각 카드: [쓴 궤적 (획 순서별 색, 빈 원 = 시작점)] [CNN이 실제로 본 28x28 그림]
         정답, 앙상블 예측과 상위 3개 점수, CNN 단독 / GRU 단독 예측
필터: 앙상블이 틀린 것만, CNN과 GRU 의견이 다른 것만, 정답 기호별, fold별
"""

import html
import os
from pathlib import Path

import cv2

from .config import CLASSES, SYMBOL
from .drawing import trajectory_preview

NOTES = {
    "subject": "사람 단위로 나눴습니다. 각 fold의 시험 데이터는 그 fold 모델이 학습 때 보지 못한 사람입니다.",
    "session": "세션 단위로 나눴습니다. 각 fold의 시험 세션은 그 fold 모델이 학습 때 보지 못한 세션입니다. "
               "같은 사람이 쓴 글씨라서 다른 사람에게는 점수가 낮을 수 있습니다.",
    "random": "샘플을 무작위로 나눴습니다. 같은 세션의 글씨가 학습과 시험에 섞여 점수가 부풀려집니다.",
    "holdout": "확인용 세션입니다. 학습, 검증, 모델 선택 어디에도 쓰지 않은 데이터라 가장 믿을 만한 점수입니다.",
}
MODEL_LABEL = {"cnn": "CNN (모양)", "gru": "GRU (쓰는 순서)", "ensemble": "앙상블", "cnn_scratch": "CNN (사전학습 없음)"}


def _sym(i):
    return html.escape(SYMBOL[CLASSES[i]])


def build_report(out_path, title, protocol, folds, records, strokes, summary):
    out_path = Path(out_path)
    img_dir = out_path.parent / (out_path.stem + "_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    blocks = []
    for fold in folds:
        recs = sorted((r for r in records if r["fold"] == fold["fold"]),
                      key=lambda r: (r["true"] == r["pred"], r["cnn"] == r["gru"], r["true"]))
        cards = []
        for r in recs:
            img_name = f"{r['fold']}_{r['index']}.png"
            cv2.imwrite(str(img_dir / img_name), trajectory_preview(strokes[r["index"]]))
            src = os.path.relpath(img_dir / img_name, out_path.parent).replace(os.sep, "/")
            ok = r["true"] == r["pred"]
            agree = r["cnn"] == r["gru"]
            disagree_badge = "" if agree else '<span class="warn">의견 다름</span>'
            cnn_cls = "m-ok" if r["cnn"] == r["true"] else "m-bad"
            gru_cls = "m-ok" if r["gru"] == r["true"] else "m-bad"
            bars = "".join(
                f'<div class="bar{" truth" if c == r["true"] else ""}"><span class="bl">{_sym(c)}</span>'
                f'<span class="track"><span class="fill" style="width:{p * 100:.0f}%"></span></span>'
                f'<span class="bv">{p:.2f}</span></div>' for c, p in r["top3"])
            cards.append(
                f'<article class="card {"ok" if ok else "bad"}" data-ok="{int(ok)}" data-agree="{int(agree)}" '
                f'data-true="{html.escape(CLASSES[r["true"]])}" data-fold="{fold["fold"]}">'
                f'<img loading="lazy" src="{html.escape(src)}" alt="">'
                f'<div class="meta"><div class="head"><span class="big">{_sym(r["true"])}</span>'
                f'<span class="arrow">→</span><span class="big pred">{_sym(r["pred"])}</span>'
                f'<span class="verdict">{"맞음" if ok else "틀림"}</span></div>'
                f'<div class="bars">{bars}</div>'
                f'<div class="models"><span class="{cnn_cls}">CNN {_sym(r["cnn"])} {r["cnn_conf"]:.2f}</span>'
                f'<span class="{gru_cls}">GRU {_sym(r["gru"])} {r["gru_conf"]:.2f}</span>{disagree_badge}</div>'
                f'<div class="file">{html.escape(r["session"])} · {html.escape(Path(r["path"]).name)}</div></div></article>')
        m = fold["metrics"]
        weight = fold.get("weight")
        weight_text = "" if weight is None else f" · CNN:GRU = {weight:.2f}:{1 - weight:.2f}"
        n_bad = sum(r["true"] != r["pred"] for r in recs)
        blocks.append(
            f'<section class="fold" data-fold="{fold["fold"]}"><h2>Fold {fold["fold"]} <small>시험 {len(recs)}개 · 틀림 '
            f'{n_bad}개 · 앙상블 macro F1 {m["macro_f1"]:.3f}{weight_text}</small></h2>'
            f'<dl class="split"><dt>시험 (처음 보는 데이터)</dt><dd>{html.escape(", ".join(fold["test_groups"]))}</dd>'
            f'<dt>학습</dt><dd>{html.escape(", ".join(fold["train_groups"]))}</dd>'
            f'<dt>검증 (학습 중단·앙상블 비중 결정)</dt><dd>{html.escape(", ".join(fold["val_groups"]) or "없음")}</dd></dl>'
            f'<div class="grid">{"".join(cards)}</div></section>')

    rows = "".join(
        f'<tr><td>{html.escape(MODEL_LABEL.get(n, n))}</td><td>{s["macro_f1_mean"]:.3f} ± {s["macro_f1_std"]:.3f}</td>'
        f'<td>{s["digit_accuracy_mean"]:.3f}</td><td>{s["operator_accuracy_mean"]:.3f}</td></tr>'
        for n, s in summary.items())
    class_buttons = "".join(f'<button data-cls="{html.escape(c)}">{html.escape(SYMBOL[c])}</button>' for c in CLASSES)
    fold_buttons = "".join(f'<button data-f="{f["fold"]}">Fold {f["fold"]}</button>' for f in folds)
    n_wrong = sum(r["true"] != r["pred"] for r in records)
    n_disagree = sum(r["cnn"] != r["gru"] for r in records)

    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} 시험 결과</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--mute:#677085;--card:#fff;--line:#e2e5eb;--ok:#1f8a4c;--bad:#c4322b;--acc:#2f6fd6;--warn:#b7791f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171d;--fg:#e6e9ef;--mute:#98a1b3;--card:#1d222b;--line:#2c3340;--ok:#4cc27e;--bad:#ef6b62;--acc:#6ea1ff;--warn:#e0a84a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Malgun Gothic",sans-serif}}
header,main{{max-width:1280px;margin:auto;padding:16px}}h1{{margin:8px 0 4px;font-size:22px}}.note{{color:var(--mute);margin:0 0 12px}}
table{{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:8px 14px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--mute);font-weight:600;font-size:12px}}
.counts{{margin:10px 0 0;color:var(--mute)}}
nav{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}}
nav .row{{max-width:1280px;margin:auto;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}nav span{{color:var(--mute);font-size:12px;margin:0 2px 0 10px}}
button{{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:4px 11px;cursor:pointer;font:inherit}}
button.on{{background:var(--acc);border-color:var(--acc);color:#fff}}
.fold h2{{font-size:18px;margin:24px 0 6px}}.fold h2 small{{color:var(--mute);font-weight:400;font-size:13px;margin-left:8px}}
.split{{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:0 0 12px;font-size:13px;color:var(--mute)}}.split dd{{margin:0;color:var(--fg)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,330px),1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--ok);border-radius:10px;overflow:hidden}}.card.bad{{border-left-color:var(--bad)}}
.card img{{width:100%;display:block;background:#fafafa}}.meta{{padding:10px 12px}}
.head{{display:flex;align-items:center;gap:8px}}.big{{font-size:26px;font-weight:700;min-width:22px;text-align:center}}
.arrow{{color:var(--mute)}}.verdict{{margin-left:auto;font-weight:700;color:var(--ok)}}.bad .verdict,.bad .pred{{color:var(--bad)}}
.bars{{display:grid;gap:3px;margin:8px 0}}.bar{{display:grid;grid-template-columns:24px 1fr 36px;gap:8px;align-items:center;font-size:12px;color:var(--mute)}}
.track{{height:8px;background:var(--line);border-radius:4px;overflow:hidden}}.fill{{display:block;height:100%;background:var(--acc)}}
.bar.truth .bl{{color:var(--fg);font-weight:700}}
.models{{display:flex;flex-wrap:wrap;gap:6px;font-size:12px}}.models span{{border:1px solid var(--line);border-radius:6px;padding:1px 7px}}
.m-ok{{color:var(--ok)}}.m-bad{{color:var(--bad)}}.warn{{color:var(--warn);border-color:var(--warn)!important}}
.file{{margin-top:6px;color:var(--mute);font-size:11px;word-break:break-all}}.hidden{{display:none!important}}
</style></head><body>
<header><h1>{html.escape(title)} · 시험 데이터 확인</h1>
<p class="note">{html.escape(NOTES.get(protocol, ""))} 왼쪽 그림은 실제로 쓴 궤적(색 = 획 순서, 빈 원 = 시작점),
오른쪽은 CNN이 본 28×28 입력입니다. 틀린 것과 두 모델 의견이 다른 것이 먼저 나옵니다.</p>
<table><tr><th>모델</th><th>macro F1 (fold 평균 ± 표준편차)</th><th>숫자 정확도</th><th>연산자 정확도</th></tr>{rows}</table>
<p class="counts">시험한 글씨 {len(records)}개 · 앙상블 틀림 {n_wrong}개 · CNN과 GRU 의견 다름 {n_disagree}개</p></header>
<nav><div class="row"><span>보기</span><button class="on" data-v="all">전체</button><button data-v="wrong">틀린 것만</button>
<button data-v="disagree">의견 다른 것만</button><span>정답</span><button class="on" data-cls="all">전체</button>{class_buttons}
<span>Fold</span><button class="on" data-f="all">전체</button>{fold_buttons}</div></nav>
<main>{"".join(blocks)}</main>
<script>
const s={{v:"all",cls:"all",f:"all"}};
function apply(){{
 document.querySelectorAll(".card").forEach(c=>{{
  const v=s.v==="all"||(s.v==="wrong"&&c.dataset.ok==="0")||(s.v==="disagree"&&c.dataset.agree==="0");
  const show=v&&(s.cls==="all"||c.dataset.true===s.cls)&&(s.f==="all"||c.dataset.fold===s.f);
  c.classList.toggle("hidden",!show);}});
 document.querySelectorAll(".fold").forEach(x=>x.classList.toggle("hidden",!x.querySelector(".card:not(.hidden)")));
}}
document.querySelectorAll("nav button").forEach(b=>b.addEventListener("click",()=>{{
 const key=b.dataset.v!==undefined?"v":b.dataset.cls!==undefined?"cls":"f";
 s[key]=b.dataset[key];
 document.querySelectorAll(`nav button[data-${{key}}]`).forEach(x=>x.classList.toggle("on",x===b));apply();}}));
</script></body></html>"""
    out_path.write_text(page, encoding="utf-8")
    return out_path
