"""논문용 최종 figure를 한 파일에서 모두 생성 (자립형: 계산·축·라벨·틱·스타일·PNG 저장 전부 포함).

입력 : results/metrics_raw.csv  (Isaac Sim 재실행 불필요)
출력 : --output-dir (기본 results/paper_figures)
  headline_safety.png                 조건별 안전률 (Fixed SOP / LLM-Full / Proposed)
  generalization.png                  In-dist vs OOD emergency detection
  escalation_curve_emergency.png      escalation rate vs safety (emergency miss)
  escalation_curve_underresponse.png  escalation rate vs safety (under-response)
  min_help.png                        목표 안전 달성에 필요한 최소 escalation rate (막대)
  reliability.png                     LLM confidence vs 실제 정답률 (과신 진단)

용어는 프로젝트 전체와 동일하게 escalation으로 통일.
그래프 양식(축/라벨/틱/색/제목)은 모두 이 파일 안에서 수정한다.
"""

import argparse, sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
_avail = {f.name for f in fm.fontManager.ttflist}
for _kf in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _kf in _avail:
        matplotlib.rcParams["font.family"] = _kf; break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 공통 상수 ─────────────────────────────────────────────────────────────────
DPI = 200
OOD_TYPES = {"GAS_LEAK", "ELECTRICAL_FAULT", "STRUCTURAL_DAMAGE"}
HIGH_RISK_TYPES = {"FIRE_RISK", "SMOKE_DETECTED", "GAS_LEAK", "ELECTRICAL_FAULT"}
T2_RANK = {"FALSE_ALARM": 0, "CONTINUE": 1, "MAINTENANCE": 2, "EMERGENCY": 3}
T1_LVL = {"APPROACH": 0, "WAIT_AND_OBSERVE": 1, "KEEP_DISTANCE": 2}
# 조건 → (tier1 col, tier2 col).  키 = 통일된 방법 명칭(전 figure 공통)
CONDS = {"Fixed SOP": ("fixed_tier1", "fixed_tier2"),
         "LLM-Full": ("llm_full_tier1", "llm_full_tier2"),
         "Proposed (LLM+HiTL)": ("hitl_tier1", "hitl_tier2_final")}
BOOL_COLS = ["escalation_gt", "fixed_tier1_correct", "fixed_tier2_correct", "llm_full_tier1_correct",
             "llm_full_tier2_correct", "hitl_escalated", "hitl_tier1_correct", "hitl_escalation_correct",
             "hitl_tier2_correct", "is_ood"]
# 위임 정책 곡선/포인트 스타일 (명칭 통일)
STYLE = {
    "LLM-Full":             dict(color="#888", marker="v", ms=11),
    "Always-escalate":      dict(color="#444", marker="*", ms=18),
    "Random":               dict(color="#bbb", ls="--", lw=2),
    "Severity Threshold":   dict(color="#F4A261", marker="o", ms=5, lw=2, drawstyle="steps-post"),
    "Fixed SOP":            dict(color="#6C757D", marker="X", ms=13),
    "Oracle (optimal)":     dict(color="#2A9D8F", marker="P", ms=14),
    "Proposed (LLM+HiTL)":  dict(color="#1D5DB8", marker="o", ms=5, lw=2.5, drawstyle="steps-post"),
}


# ── 데이터 로드 / 공통 헬퍼 ───────────────────────────────────────────────────
def coerce(df):
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    if "is_ood" not in df.columns:
        df["is_ood"] = df["anomaly_type"].isin(OOD_TYPES)
    return df


def save(fig, out_dir, name, show):
    out = Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[OK] Saved → {out}")
    if show:
        plt.show()
    plt.close(fig)


def grouped_bar(ax, xlabels, series, colors, ylabel, ylim=(0, 105), annotate=True):
    keys = list(series); width = 0.8 / len(keys); x = np.arange(len(xlabels))
    for i, k in enumerate(keys):
        offs = (i - (len(keys) - 1) / 2) * width
        vals = [v if v == v else 0 for v in series[k]]
        bars = ax.bar(x + offs, vals, width, label=k,
                      color=colors[i] if isinstance(colors, list) else colors.get(k, "#888"),
                      alpha=0.88, edgecolor="#222", linewidth=0.7)
        if annotate:
            for b, v in zip(bars, series[k]):
                ax.text(b.get_x() + b.get_width() / 2, (v if v == v else 0) + 1.2,
                        (f"{v:.0f}" if v == v else "N/A"), ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold"); ax.set_ylim(*ylim); ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.35); ax.set_axisbelow(True); ax.legend(fontsize=12, framealpha=0.95)


# ── 안전 지표 계산 ────────────────────────────────────────────────────────────
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


# ── 에스컬레이션 정책 계산 ────────────────────────────────────────────────────
def _final_t2(df, e):
    return np.where(e, df["tier2_gt"].to_numpy(), df["llm_full_tier2"].to_numpy())

def help_rate(e):
    return float(np.mean(e)) * 100

def _missed_rate(df, e):
    em = (df["tier2_gt"] == "EMERGENCY").to_numpy()
    if em.sum() == 0: return np.nan
    return (_final_t2(df, e)[em] != "EMERGENCY").mean() * 100

def _danger_rate(df, e):
    haz = df["tier2_gt"].isin(["EMERGENCY", "MAINTENANCE"]).to_numpy()
    ft = _final_t2(df, e)
    pr = pd.Series(ft).map(T2_RANK).to_numpy(); gr = df["tier2_gt"].map(T2_RANK).to_numpy()
    return ((pr < gr) & haz).mean() * 100

def safety(df, e, metric):
    return 100 - (_missed_rate(df, e) if metric == "missed" else _danger_rate(df, e))


def policy_points(df, metric):
    """정책별 (help%, safety%) 좌표. metric: 'missed' | 'danger'."""
    N = len(df); sev = df["severity_score"].to_numpy(); out = {}
    out["LLM-Full"] = [(0.0, safety(df, np.zeros(N, bool), metric))]
    out["Always-escalate"] = [(100.0, safety(df, np.ones(N, bool), metric))]
    err0 = 100 - out["LLM-Full"][0][1]
    out["Random"] = [(p, 100 - (1 - p / 100) * err0) for p in np.linspace(0, 100, 21)]
    pts = []
    for tau in sorted(set(sev.tolist())) + [sev.max() + 1e-6]:
        e = sev >= tau; pts.append((help_rate(e), safety(df, e, metric)))
    out["Severity Threshold"] = sorted(pts)
    e = df["anomaly_type"].isin(HIGH_RISK_TYPES).to_numpy()
    out["Fixed SOP"] = [(help_rate(e), safety(df, e, metric))]
    ft0 = _final_t2(df, np.zeros(N, bool))
    if metric == "missed":
        e = (df["tier2_gt"] == "EMERGENCY").to_numpy() & (ft0 != "EMERGENCY")
    else:
        haz = df["tier2_gt"].isin(["EMERGENCY", "MAINTENANCE"]).to_numpy()
        pr = pd.Series(ft0).map(T2_RANK).to_numpy(); gr = df["tier2_gt"].map(T2_RANK).to_numpy()
        e = haz & (pr < gr)
    out["Oracle (optimal)"] = [(help_rate(e), safety(df, e, metric))]
    if "tier2_confidence" in df.columns:
        conf = df["tier2_confidence"].to_numpy(); pts = []
        for tau in [-0.01] + sorted(set(conf.tolist())) + [1.01]:
            e = conf < tau; pts.append((help_rate(e), safety(df, e, metric)))
        out["Proposed (LLM+HiTL)"] = sorted(set(pts))
    return out


def _min_help(curve, target):
    ok = [h for h, s in curve if s >= target - 1e-9]
    return min(ok) if ok else np.nan

def _min_help_random(df, metric, target):
    err0 = 100 - safety(df, np.zeros(len(df), bool), metric)
    if err0 <= (100 - target): return 0.0
    return max(0.0, (1 - (100 - target) / err0)) * 100


# ═════════════════════════════════════════════════════════════════════════════
#  Figures
# ═════════════════════════════════════════════════════════════════════════════

def fig_headline_safety(df, out_dir, show):
    # 보완값(높을수록 안전): 100 - 위험률
    conds = list(CONDS)
    series = {"Correct response (Tier-2)":  [100 - t2_danger(df, CONDS[c][1]) for c in conds],
              "Emergency detection":        [100 - missed_emerg(df, CONDS[c][1]) for c in conds],
              "Adequate caution (Tier-1)":  [100 - t1_unsafe(df, CONDS[c][0]) for c in conds]}
    fig, ax = plt.subplots(figsize=(9, 6))
    grouped_bar(ax, conds, series, ["#2A9D8F", "#1D5DB8", "#F4A261"], "Safety rate (%)")
    fig.tight_layout(); save(fig, out_dir, "headline_safety.png", show)


def fig_generalization(df, out_dir, show):
    # 보완값(높을수록 안전): EMERGENCY 포착률 = 100 - 누락률
    conds = list(CONDS)
    series = {"In-distribution": [100 - missed_emerg(df[~df.is_ood], CONDS[c][1]) for c in conds],
              "OOD": [100 - missed_emerg(df[df.is_ood], CONDS[c][1]) for c in conds]}
    fig, ax = plt.subplots(figsize=(9, 6))
    grouped_bar(ax, conds, series, {"In-distribution": "#2A9D8F", "OOD": "#1D5DB8"},
                "Emergency detection (%)")
    fig.tight_layout(); save(fig, out_dir, "generalization.png", show)


def _escalation_curve_single(df, metric, ylabel, fname, out_dir, show):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pol = policy_points(df, metric)
    for name in ["Random", "Severity Threshold"] + (["Proposed (LLM+HiTL)"] if "Proposed (LLM+HiTL)" in pol else []):
        xs = [p[0] for p in pol[name]]; ys = [p[1] for p in pol[name]]
        ax.plot(xs, ys, label=name, **STYLE[name])
    for name in ["LLM-Full", "Fixed SOP"]:
        x, y = pol[name][0]
        ax.plot([x], [y], label=name, linestyle="none", **STYLE[name])
    ax.set_xlabel("Escalation rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.set_xlim(-3, 103); ax.set_ylim(90, 101)
    ax.grid(True, alpha=0.35); ax.set_axisbelow(True); ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.95)
    fig.tight_layout(); save(fig, out_dir, fname, show)


def fig_escalation_curve(df, out_dir, show):
    _escalation_curve_single(df, "missed", "Safety (%)",
                             "escalation_curve_emergency.png", out_dir, show)
    _escalation_curve_single(df, "danger", "Safety (%)",
                             "escalation_curve_underresponse.png", out_dir, show)


def fig_min_help(df, out_dir, show, metric="missed"):
    pol = policy_points(df, metric)
    if "Proposed (LLM+HiTL)" not in pol:
        print("[skip] min_help: no tier2_confidence column"); return
    targets = [98, 99, 100]
    series = {"Proposed (LLM+HiTL)": [_min_help(pol["Proposed (LLM+HiTL)"], t) for t in targets],
              "Severity Threshold": [_min_help(pol["Severity Threshold"], t) for t in targets],
              "Random": [_min_help_random(df, metric, t) for t in targets]}
    colors = {"Proposed (LLM+HiTL)": "#1D5DB8", "Severity Threshold": "#F4A261", "Random": "#bbb"}
    fig, ax = plt.subplots(figsize=(9, 6))
    keys = list(series); width = 0.8 / len(keys); x = np.arange(len(targets))
    for i, k in enumerate(keys):
        offs = (i - (len(keys) - 1) / 2) * width
        vals = [v if v == v else 0 for v in series[k]]
        bars = ax.bar(x + offs, vals, width, label=k, color=colors[k], alpha=0.9, edgecolor="#222", linewidth=0.7)
        for b, v in zip(bars, series[k]):
            ax.text(b.get_x() + b.get_width() / 2, (v if v == v else 0) + 1.2,
                    (f"{v:.0f}" if v == v else "N/A"), ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([f">={t}%" for t in targets], fontsize=15)
    ax.set_xlabel("Target emergency-detection level", fontsize=16, fontweight="bold")
    ax.set_ylabel("Min. escalation rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 105); ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.35); ax.set_axisbelow(True); ax.legend(fontsize=12, framealpha=0.95)
    fig.tight_layout(); save(fig, out_dir, "min_help.png", show)


def fig_reliability(p_raw, y, out_dir, show, nbins=8):
    """LLM confidence vs 실제 정답률 (raw 값 그대로 — 과신 진단)."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    edges = np.linspace(0, 1, nbins + 1); xs, ys, ws = [], [], []
    for i in range(nbins):
        m = (p_raw > edges[i]) & (p_raw <= edges[i + 1]) if i > 0 else (p_raw >= edges[i]) & (p_raw <= edges[i + 1])
        if m.sum() == 0: continue
        xs.append(p_raw[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.5, label="Ideal")
    ax.scatter(xs, ys, s=60 + 600 * np.array(ws) / max(ws), color="#E63946", alpha=0.8,
               edgecolors="#222", zorder=3, label="LLM confidence")
    ax.plot(xs, ys, color="#E63946", alpha=0.5, zorder=2)
    ax.set_xlabel("Stated confidence", fontsize=16, fontweight="bold")
    ax.set_ylabel("Actual accuracy", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.35); ax.set_axisbelow(True)
    ax.tick_params(labelsize=12); ax.legend(fontsize=12, loc="upper left")
    fig.tight_layout(); save(fig, out_dir, "reliability.png", show)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/metrics_raw.csv")
    p.add_argument("--output-dir", default="results/paper_figures")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()
    show = not a.no_show
    out = a.output_dir

    df = coerce(pd.read_csv(a.input))
    print(f"[INFO] Loaded {len(df)} rows from {a.input}")

    fig_headline_safety(df, out, show)
    fig_generalization(df, out, show)
    fig_escalation_curve(df, out, show)
    fig_min_help(df, out, show)

    # confidence 기반 reliability (raw 값만)
    if "tier2_confidence" not in df.columns:
        print("[WARN] no tier2_confidence column -> skipping reliability (re-run sim with confidence output)")
        return
    p_raw = df["tier2_confidence"].to_numpy()
    y = df["llm_full_tier2_correct"].to_numpy().astype(float)
    fig_reliability(p_raw, y, out, show)

    print(f"\n[DONE] figures → {out}/")


if __name__ == "__main__":
    main()
