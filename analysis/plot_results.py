"""metrics_raw.csv → 결과 figure(PNG) — 프레임워크 안전 중심 (위험률·EMERGENCY누락·OOD일반화·트레이드오프)."""

import argparse, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_avail = {f.name for f in fm.fontManager.ttflist}
for _kf in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _kf in _avail:
        matplotlib.rcParams["font.family"] = _kf
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 스타일 ────────────────────────────────────────────────────────────────────
BOX_COLORS = ["#1D5DB8", "#E63946", "#2A9D8F", "#F4A261", "#6C757D"]
TYPE_COLORS = {"FIRE_RISK": "#E63946", "LIQUID_LEAK": "#1D5DB8", "AGV_STOPPED": "#F4A261",
               "SMOKE_DETECTED": "#6C757D", "GAS_LEAK": "#2A9D8F", "ELECTRICAL_FAULT": "#9B5DE5",
               "STRUCTURAL_DAMAGE": "#8D6E63"}
COND_COLORS = {"Fixed (SOP)": "#6C757D", "LLM-Full": "#1D5DB8", "LLM+HiTL": "#E63946"}
MODEL_COLORS = {"haiku": "#1D5DB8", "sonnet": "#E63946", "gpt4o": "#2A9D8F", "llama": "#F4A261"}
MODEL_LABELS = {"haiku": "Haiku", "sonnet": "Sonnet", "gpt4o": "GPT-4o mini", "llama": "Llama"}
ACTION_ORDER = ["APPROACH", "WAIT_AND_OBSERVE", "KEEP_DISTANCE"]
TIER2_ORDER = ["EMERGENCY", "MAINTENANCE", "CONTINUE", "FALSE_ALARM"]
OOD_TYPES = {"GAS_LEAK", "ELECTRICAL_FAULT", "STRUCTURAL_DAMAGE"}
T1_LVL = {"APPROACH": 0, "WAIT_AND_OBSERVE": 1, "KEEP_DISTANCE": 2}
T2_RANK = {"FALSE_ALARM": 0, "CONTINUE": 1, "MAINTENANCE": 2, "EMERGENCY": 3}
CONDS = {"Fixed (SOP)": ("fixed_tier1", "fixed_tier2"), "LLM-Full": ("llm_full_tier1", "llm_full_tier2"),
         "LLM+HiTL": ("hitl_tier1", "hitl_tier2_final")}
DPI = 200
BOOL_COLS = ["escalation_gt", "fixed_tier1_correct", "fixed_tier2_correct", "llm_full_tier1_correct",
             "llm_full_tier2_correct", "hitl_escalated", "hitl_tier1_correct", "hitl_escalation_correct",
             "hitl_tier2_correct", "is_ood"]


def _coerce(df):
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    if "is_ood" not in df.columns:
        df["is_ood"] = df["anomaly_type"].isin(OOD_TYPES)
    return df


def _mlabel(m):
    return MODEL_LABELS.get(m, m)


def _save(fig, out_dir, name, show):
    out = Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[OK] Saved → {out}")
    if show:
        plt.show()
    plt.close(fig)


def t2_danger(df, t2col):
    if len(df) == 0: return float("nan")
    pr = df[t2col].map(T2_RANK); gr = df["tier2_gt"].map(T2_RANK)
    haz = df["tier2_gt"].isin(["EMERGENCY", "MAINTENANCE"])
    return ((pr < gr) & haz).mean() * 100

def missed_emerg(df, t2col):
    e = df[df["tier2_gt"] == "EMERGENCY"]
    return (e[t2col] != "EMERGENCY").mean() * 100 if len(e) else float("nan")

def t1_unsafe(df, t1col):
    if len(df) == 0: return float("nan")
    return (df[t1col].map(T1_LVL) < df["tier1_gt"].map(T1_LVL)).mean() * 100


def _grouped(ax, xlabels, series, colors, ylabel, ylim=(0, 105), annotate=True):
    keys = list(series); width = 0.8/len(keys); x = np.arange(len(xlabels))
    for i, k in enumerate(keys):
        offs = (i - (len(keys)-1)/2) * width
        bars = ax.bar(x + offs, series[k], width, label=k,
                      color=colors[i] if isinstance(colors, list) else colors.get(k, "#888"),
                      alpha=0.88, edgecolor="#222", linewidth=0.7)
        if annotate:
            for b, v in zip(bars, series[k]):
                if v == v:
                    ax.text(b.get_x()+b.get_width()/2, v+1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14); ax.set_ylim(*ylim); ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.35); ax.set_axisbelow(True); ax.legend(fontsize=11, framealpha=0.95)


# ── 1) Tier-1 scatter: GT(target) + 모델별 ──
def _scatter_ax(ax, sub, col, title):
    a2y = {a: i for i, a in enumerate(ACTION_ORDER)}; rng = np.random.default_rng(42)
    y = sub[col].map(a2y).to_numpy() + rng.uniform(-0.32, 0.32, len(sub))
    xj = sub["severity_score"].to_numpy() + rng.uniform(-0.006, 0.006, len(sub))
    for i, c in enumerate(["#d4edda", "#fff3cd", "#f8d7da"]):
        ax.axhspan(i-0.5, i+0.5, alpha=0.25, color=c, zorder=0)
    for t in sorted(sub["anomaly_type"].unique()):
        m = (sub["anomaly_type"] == t).to_numpy()
        lbl = t + (" (OOD)" if t in OOD_TYPES else "")
        ax.scatter(xj[m], y[m], c=TYPE_COLORS.get(t, "black"), label=lbl, s=110,
                   alpha=0.78, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["APPROACH", "WAIT AND\nOBSERVE", "KEEP\nDISTANCE"], fontsize=11)
    ax.set_xlabel("Severity Score", fontsize=13); ax.set_xlim(0, 1); ax.set_ylim(-0.5, 2.5)
    ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True); ax.set_title(title, fontsize=13)


def scatter_per_source(df, out_dir, show):
    models = sorted(df["model"].unique())
    gt_df = df.drop_duplicates(subset=["lap", "zone_id"])
    sources = [("Ground Truth (target)", gt_df, "tier1_gt", "scatter_tier1_GT.png")]
    for m in models:
        sources.append((f"{_mlabel(m)} (LLM)", df[df.model == m], "hitl_tier1", f"scatter_tier1_{m}.png"))
    for title, sub, col, fname in sources:
        fig, ax = plt.subplots(figsize=(8, 5.5)); _scatter_ax(ax, sub, col, title)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=8); fig.tight_layout()
        _save(fig, out_dir, "scatter/" + fname, show)
    ncol = 3; nrow = int(np.ceil(len(sources)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 5*nrow), squeeze=False)
    for idx, (title, sub, col, _) in enumerate(sources):
        ax = axes[idx//ncol][idx % ncol]; _scatter_ax(ax, sub, col, title)
        if idx == 0: ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    for idx in range(len(sources), nrow*ncol):
        axes[idx//ncol][idx % ncol].axis("off")
    fig.suptitle("Tier-1 Action vs. Severity (GT target + each model)", fontsize=16, y=1.0)
    fig.tight_layout(); _save(fig, out_dir, "fig1_tier1_scatter_grid.png", show)


# ── 2) HEADLINE: 조건별 안전 위험 지표 (낮을수록 좋음) ──
def headline_fig(df, out_dir, show, fname=None):
    conds = list(CONDS)
    series = {"Tier2 위험%": [t2_danger(df, CONDS[c][1]) for c in conds],
              "EMERGENCY 누락%": [missed_emerg(df, CONDS[c][1]) for c in conds],
              "Tier1 위험%": [t1_unsafe(df, CONDS[c][0]) for c in conds]}
    fig, ax = plt.subplots(figsize=(9, 6))
    _grouped(ax, conds, series, ["#E63946", "#B5179E", "#F4A261"], "Risk metric (%, 낮을수록 안전)")
    ax.set_title("프레임워크 집계 — 안전 위험 지표 (전 모델)", fontsize=14)
    fig.tight_layout(); _save(fig, out_dir, fname or "fig2_headline_safety.png", show)


# ── 3) 일반화: In-dist vs OOD (EMERGENCY 누락률) ──
def generalization_fig(df, out_dir, show, fname=None):
    conds = list(CONDS)
    series = {"In-distribution": [missed_emerg(df[~df.is_ood], CONDS[c][1]) for c in conds],
              "OOD (신규 타입)": [missed_emerg(df[df.is_ood], CONDS[c][1]) for c in conds]}
    fig, ax = plt.subplots(figsize=(9, 6))
    _grouped(ax, conds, series, {"In-distribution": "#2A9D8F", "OOD (신규 타입)": "#E63946"},
             "EMERGENCY 누락률 (%, 낮을수록 안전)")
    ax.set_title("일반화 — Fixed는 신규 타입에서 붕괴, LLM+법은 일반화", fontsize=13)
    fig.tight_layout(); _save(fig, out_dir, fname or "fig3_generalization.png", show)


# ── 4) 자동화 vs 안전 트레이드오프 (모델 + baseline) ──
def tradeoff_fig(df, out_dir, show):
    models = sorted(df["model"].unique()); crit = df[df.tier2_gt == "EMERGENCY"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter([0], [100], marker="*", s=420, color="#444", zorder=5, label="Always-escalate")
    fx_safe = (crit.fixed_tier2 == "EMERGENCY").mean()*100
    ax.scatter([100], [fx_safe], marker="X", s=240, color="#6C757D", zorder=5, label="Fixed SOP")
    for m in models:
        g = df[df.model == m]; gc = crit[crit.model == m]
        auto = (1 - g.hitl_escalated.mean())*100
        hitl_safe = (gc.hitl_tier2_final == "EMERGENCY").mean()*100
        full_safe = (gc.llm_full_tier2 == "EMERGENCY").mean()*100
        col = MODEL_COLORS.get(m, "#888")
        ax.scatter([100], [full_safe], marker="s", s=110, facecolors="none", edgecolors=col, linewidths=2, zorder=4)
        ax.scatter([auto], [hitl_safe], marker="o", s=200, color=col, edgecolors="white", linewidths=1.2, zorder=6, label=_mlabel(m))
        ax.annotate("", xy=(auto, hitl_safe), xytext=(100, full_safe),
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.5, lw=1.5), zorder=3)
    ax.set_xlabel("Automation Rate (1 - escalation rate, %)", fontsize=14)
    ax.set_ylabel("Safety: EMERGENCY Tier-2 Accuracy (%)", fontsize=14)
    ax.set_xlim(-5, 110); ax.set_ylim(0, 105); ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.35); ax.set_axisbelow(True)
    ax.text(102, 2, "□ LLM-Full  ○ LLM+HiTL  ✕ Fixed", ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    ax.legend(loc="lower center", fontsize=10, framealpha=0.95, ncol=2)
    ax.set_title("자동화율 vs 안전 트레이드오프 (우상단=이상적)", fontsize=14)
    fig.tight_layout(); _save(fig, out_dir, "fig4_tradeoff.png", show)


# ── 5) 에스컬레이션 P/R/F1 ──
def escalation_pr_fig(df, out_dir, show):
    models = sorted(df["model"].unique()); metrics = {"Precision": [], "Recall": [], "F1": []}
    for m in models:
        g = df[df.model == m]
        tp = (g.hitl_escalated & g.escalation_gt).sum(); fp = (g.hitl_escalated & ~g.escalation_gt).sum()
        fn = (~g.hitl_escalated & g.escalation_gt).sum()
        prec = tp/(tp+fp) if (tp+fp) else 0.0; rec = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        metrics["Precision"].append(prec*100); metrics["Recall"].append(rec*100); metrics["F1"].append(f1*100)
    fig, ax = plt.subplots(figsize=(max(7, 1.9*len(models)+3), 6))
    _grouped(ax, [_mlabel(m) for m in models], metrics, BOX_COLORS, "Escalation Decision Quality (%)")
    fig.tight_layout(); _save(fig, out_dir, "fig5_escalation_pr.png", show)


# ── 6) 자율 실력: 탈출율 vs 자율 정확도 ──
def autonomy_fig(df, out_dir, show):
    models = sorted(df["model"].unique()); gt_esc = df.drop_duplicates(["lap", "zone_id"]).escalation_gt.mean()*100
    esc, acc = [], []
    for m in models:
        g = df[df.model == m]; esc.append(g.hitl_escalated.mean()*100)
        non = g[~g.hitl_escalated]; acc.append(non.hitl_tier2_correct.mean()*100 if len(non) else 0.0)
    fig, ax = plt.subplots(figsize=(max(8, 1.9*len(models)+3), 6))
    _grouped(ax, [_mlabel(m) for m in models],
             {"Escalation rate": esc, "Autonomous Tier-2 acc (non-esc)": acc}, ["#E63946", "#2A9D8F"], "Percentage (%)")
    ax.axhline(gt_esc, color="#444", linestyle="--", linewidth=1.5)
    ax.text(len(models)-0.5, gt_esc+1.5, f"GT 탈출율 {gt_esc:.0f}%", ha="right", fontsize=10, color="#444")
    ax.set_title("탈출율(과잉위임?) vs 자율 정확도(오라클 도움 0)", fontsize=13)
    fig.tight_layout(); _save(fig, out_dir, "fig6_autonomy.png", show)


# ── 7) Tier-2 혼동행렬 (모델별 × Fixed/Full/HiTL) ──
def tier2_confusion_fig(df, out_dir, show):
    models = sorted(df["model"].unique())
    conds = [("Fixed", "fixed_tier2"), ("LLM-Full", "llm_full_tier2"), ("LLM+HiTL", "hitl_tier2_final")]
    fig, axes = plt.subplots(len(models), len(conds), figsize=(4.4*len(conds), 4.0*len(models)), squeeze=False)
    for r, m in enumerate(models):
        g = df[df.model == m]
        for c, (label, col) in enumerate(conds):
            ax = axes[r][c]
            ct = pd.crosstab(g["tier2_gt"], g[col]).reindex(index=TIER2_ORDER, columns=TIER2_ORDER, fill_value=0)
            counts = ct.to_numpy(); rs = counts.sum(axis=1, keepdims=True); rs[rs == 0] = 1
            norm = counts/rs; ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "#222", fontsize=10)
            ax.set_xticks(range(4)); ax.set_xticklabels([t[:5] for t in TIER2_ORDER], fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(4)); ax.set_yticklabels([t[:5] for t in TIER2_ORDER], fontsize=8)
            ax.set_title(f"{_mlabel(m)} — {label}", fontsize=11)
            if c == 0: ax.set_ylabel("GT", fontsize=10)
            ax.set_xlabel("Pred", fontsize=10)
    fig.suptitle("Tier-2 Confusion (cell=count, color=row-normalized)", fontsize=15, y=1.0)
    fig.tight_layout(); _save(fig, out_dir, "fig7_tier2_confusion.png", show)


# ── 8) anomaly type별 Tier2 위험률 (OOD 표시) ──
def per_type_fig(df, out_dir, show):
    types = sorted(df["anomaly_type"].unique(), key=lambda t: (t in OOD_TYPES, t))
    series = {"LLM-Full": [t2_danger(df[df.anomaly_type == t], "llm_full_tier2") for t in types],
              "LLM+HiTL": [t2_danger(df[df.anomaly_type == t], "hitl_tier2_final") for t in types]}
    fig, ax = plt.subplots(figsize=(max(9, 1.5*len(types)+3), 6))
    labels = [t + ("\n(OOD)" if t in OOD_TYPES else "") for t in types]
    _grouped(ax, labels, series, {"LLM-Full": "#1D5DB8", "LLM+HiTL": "#E63946"}, "Tier-2 위험 과소대응률 (%)")
    for i, t in enumerate(types):
        if t in OOD_TYPES:
            ax.axvspan(i-0.45, i+0.45, color="#ffe8e8", alpha=0.5, zorder=0)
    ax.set_title("Anomaly Type별 Tier-2 위험률 (분홍=OOD 신규 타입)", fontsize=13)
    fig.tight_layout(); _save(fig, out_dir, "fig8_per_type_danger.png", show)


# ── 9) 결정 시간 ──
def decision_time_fig(df, out_dir, show):
    models = sorted(df["model"].unique())
    arrays = [(df[df.model == m]["hitl_decision_time_ms"]/1000.0).tolist() for m in models]
    fig, ax = plt.subplots(figsize=(max(7, 1.6*len(models)+3), 6.5))
    kw = dict(patch_artist=True, showmeans=True, showfliers=False, widths=0.55,
              meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=7, markeredgewidth=1.2),
              medianprops=dict(color="black", linewidth=2.0), whiskerprops=dict(color="#333", linewidth=1.2),
              capprops=dict(color="#333", linewidth=1.2))
    try:
        bp = ax.boxplot(arrays, tick_labels=[_mlabel(m) for m in models], **kw)
    except TypeError:
        bp = ax.boxplot(arrays, labels=[_mlabel(m) for m in models], **kw)
    for patch, color in zip(bp["boxes"], BOX_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor("#222"); patch.set_linewidth(1.2)
    ax.set_ylabel("Decision Time (s)", fontsize=14); ax.tick_params(axis="x", labelsize=13); ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.35); ax.set_axisbelow(True)
    fig.tight_layout(); _save(fig, out_dir, "fig9_decision_time.png", show)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/metrics_raw.csv")
    p.add_argument("--output-dir", default="results/figures")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()
    show = not a.no_show
    df = _coerce(pd.read_csv(a.input))
    print(f"[INFO] Loaded {len(df)} rows, models: {sorted(df['model'].unique())}, OOD rows: {int(df['is_ood'].sum())}")
    scatter_per_source(df, a.output_dir, show)
    headline_fig(df, a.output_dir, show)
    generalization_fig(df, a.output_dir, show)
    escalation_pr_fig(df, a.output_dir, show)
    tier2_confusion_fig(df, a.output_dir, show)
    per_type_fig(df, a.output_dir, show)
    decision_time_fig(df, a.output_dir, show)
    # tradeoff_fig / autonomy_fig: automation-rate 중심 → 본 연구 지표 아님, 미사용


if __name__ == "__main__":
    main()
