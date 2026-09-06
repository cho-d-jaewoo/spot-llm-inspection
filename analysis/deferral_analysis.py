"""위임(escalation) 정책 비교 — KnowNo 스타일 safety vs human-help 곡선 + 에스컬레이션 판별력(ROC) + 과/과소의존.

핵심: '위임 많이=좋음'(사소한 단조축)이 아니라, *같은 help budget에서 올바른 케이스를 위임하는가*를 평가.
정책: No-help / Random / Severity-threshold / Type(SOP) / LLM-reasoned(제안) / Oracle-router / Always.
"""

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
        matplotlib.rcParams["font.family"] = _kf; break
matplotlib.rcParams["axes.unicode_minus"] = False

OOD_TYPES = {"GAS_LEAK", "ELECTRICAL_FAULT", "STRUCTURAL_DAMAGE"}
T2_RANK = {"FALSE_ALARM": 0, "CONTINUE": 1, "MAINTENANCE": 2, "EMERGENCY": 3}
HIGH_RISK_TYPES = {"FIRE_RISK", "SMOKE_DETECTED", "GAS_LEAK", "ELECTRICAL_FAULT"}
BOOL_COLS = ["escalation_gt", "hitl_escalated", "llm_full_tier2_correct", "hitl_tier2_correct", "is_ood"]
DPI = 200


def _coerce(df):
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    if "is_ood" not in df.columns:
        df["is_ood"] = df["anomaly_type"].isin(OOD_TYPES)
    return df


# ── 위임 마스크 e → 최종 Tier2 (위임=오라클=GT, 아니면 LLM 자율) → 안전 지표 ──
def _final_t2(df, e):
    return np.where(e, df["tier2_gt"].to_numpy(), df["llm_full_tier2"].to_numpy())

def missed_emerg_rate(df, e):
    em = (df["tier2_gt"] == "EMERGENCY").to_numpy()
    if em.sum() == 0: return np.nan
    ft = _final_t2(df, e)
    return (ft[em] != "EMERGENCY").mean() * 100

def danger_rate(df, e):
    haz = df["tier2_gt"].isin(["EMERGENCY", "MAINTENANCE"]).to_numpy()
    ft = _final_t2(df, e)
    pr = pd.Series(ft).map(T2_RANK).to_numpy(); gr = df["tier2_gt"].map(T2_RANK).to_numpy()
    return ((pr < gr) & haz).mean() * 100

def help_rate(e):
    return float(np.mean(e)) * 100


def _safety(df, e, metric):
    return 100 - (missed_emerg_rate(df, e) if metric == "missed" else danger_rate(df, e))


# ── 정책별 (help%, safety%) 산출 ──────────────────────────────────────────────
def policy_points(df, metric):
    N = len(df); sev = df["severity_score"].to_numpy()
    out = {}
    # 양 끝점
    out["No-help (LLM only)"] = [(0.0, _safety(df, np.zeros(N, bool), metric))]
    out["Always-escalate"] = [(100.0, _safety(df, np.ones(N, bool), metric))]
    # Random: 해석적 직선 (위임=정답이므로 기대 오류 = (1-p)*LLM단독오류)
    err0 = 100 - out["No-help (LLM only)"][0][1]
    out["Random"] = [(p, 100 - (1 - p/100) * err0) for p in np.linspace(0, 100, 21)]
    # Severity-threshold 스윕
    pts = []
    for tau in sorted(set(sev.tolist())) + [sev.max() + 1e-6]:
        e = sev >= tau
        pts.append((help_rate(e), _safety(df, e, metric)))
    out["Severity-threshold"] = sorted(pts)
    # Type / SOP heuristic (고위험 타입 위임)
    e = df["anomaly_type"].isin(HIGH_RISK_TYPES).to_numpy()
    out["Type-based (SOP)"] = [(help_rate(e), _safety(df, e, metric))]
    # Oracle router (LLM이 틀릴 케이스만 위임 = 이론적 최적 frontier 점)
    ft0 = _final_t2(df, np.zeros(N, bool))
    if metric == "missed":
        e = (df["tier2_gt"] == "EMERGENCY").to_numpy() & (ft0 != "EMERGENCY")
    else:
        haz = df["tier2_gt"].isin(["EMERGENCY", "MAINTENANCE"]).to_numpy()
        pr = pd.Series(ft0).map(T2_RANK).to_numpy(); gr = df["tier2_gt"].map(T2_RANK).to_numpy()
        e = haz & (pr < gr)
    out["Oracle-router (optimal)"] = [(help_rate(e), _safety(df, e, metric))]
    # LLM-reasoned (제안 기법) — 실제 LLM 이진 위임 결정 (운영점)
    e = df["hitl_escalated"].to_numpy()
    out["LLM-reasoned (ours)"] = [(help_rate(e), _safety(df, e, metric))]
    # LLM-confidence(τ) 스윕 — confidence 낮을수록 위임 (KnowNo식 LLM 곡선)
    if "tier2_confidence" in df.columns:
        conf = df["tier2_confidence"].to_numpy()
        pts = []
        for tau in [-0.01] + sorted(set(conf.tolist())) + [1.01]:
            e = conf < tau   # confidence < τ 이면 위임
            pts.append((help_rate(e), _safety(df, e, metric)))
        out["LLM-confidence(τ)"] = sorted(set(pts))
    return out


STYLE = {
    "No-help (LLM only)":     dict(color="#888", marker="v", ms=11),
    "Always-escalate":        dict(color="#444", marker="*", ms=18),
    "Random":                 dict(color="#bbb", ls="--", lw=2),
    "Severity-threshold":     dict(color="#F4A261", marker="o", ms=5, lw=2, drawstyle="steps-post"),
    "Type-based (SOP)":       dict(color="#6C757D", marker="X", ms=13),
    "Oracle-router (optimal)": dict(color="#2A9D8F", marker="P", ms=14),
    "LLM-reasoned (ours)":    dict(color="#E63946", marker="*", ms=22),
    "LLM-confidence(τ)":      dict(color="#1D5DB8", marker="o", ms=5, lw=2.5, drawstyle="steps-post"),
}


def deferral_curve_fig(df, out_dir, show, fname=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, metric, title in zip(axes, ["missed", "danger"],
                                 ["Safety = 1 − EMERGENCY 누락", "Safety = 1 − Tier2 위험오류"]):
        pol = policy_points(df, metric)
        curve_names = ["Random", "Severity-threshold"] + (["LLM-confidence(τ)"] if "LLM-confidence(τ)" in pol else [])
        for name in curve_names:
            xs = [p[0] for p in pol[name]]; ys = [p[1] for p in pol[name]]
            ax.plot(xs, ys, label=name, **STYLE[name])
        for name in ["No-help (LLM only)", "Type-based (SOP)", "Oracle-router (optimal)",
                     "Always-escalate", "LLM-reasoned (ours)"]:
            x, y = pol[name][0]
            ax.plot([x], [y], label=name, linestyle="none", **STYLE[name])
        ax.set_xlabel("Human-help rate (escalation %)", fontsize=13)
        ax.set_ylabel("Safety (%)", fontsize=13)
        ax.set_title(title, fontsize=13); ax.set_xlim(-3, 103); ax.set_ylim(84, 101)
        ax.grid(True, alpha=0.35); ax.set_axisbelow(True); ax.tick_params(labelsize=11)
    axes[0].legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.suptitle("위임 정책 비교 — 같은 help에서 더 안전한가 (좌상단=효율적, KnowNo 스타일)", fontsize=15, y=1.0)
    fig.tight_layout()
    out = Path(out_dir) / (fname or "fig_deferral_curve.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); print(f"[OK] Saved → {out}")
    if show: plt.show()
    plt.close(fig)


# ── 에스컬레이션 판별력: severity ROC + 과/과소의존 ───────────────────────────
def _roc(score, label):
    order = np.argsort(-score); y = label[order]
    P = y.sum(); Nn = (~y).sum()
    tpr = np.concatenate([[0], np.cumsum(y) / P]) if P else np.zeros(len(y)+1)
    fpr = np.concatenate([[0], np.cumsum(~y) / Nn]) if Nn else np.zeros(len(y)+1)
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc


def deferral_split_fig(df, out_dir, show, fname=None):
    """In-dist vs OOD 분리. OOD에선 severity/SOP는 사전 규칙이 없어 배포 불가(점선·주석)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (mask, title) in zip(axes, [(~df.is_ood, "In-distribution (기존 타입)"),
                                          (df.is_ood, "OOD (신규 타입 — 사전 severity/SOP 규칙 없음)")]):
        sub = df[mask]
        if len(sub) == 0:
            ax.set_title(title + " — 데이터 없음"); continue
        pol = policy_points(sub, "missed")
        xs = [p[0] for p in pol["Random"]]; ys = [p[1] for p in pol["Random"]]
        ax.plot(xs, ys, label="Random", **STYLE["Random"])
        is_ood_panel = bool(sub.is_ood.iloc[0])
        # severity-threshold: OOD에선 '배포 불가'로 회색 점선 처리
        xs = [p[0] for p in pol["Severity-threshold"]]; ys = [p[1] for p in pol["Severity-threshold"]]
        if is_ood_panel:
            ax.plot(xs, ys, color="#ccc", ls=":", lw=2, label="Severity-threshold (배포불가: 신규타입 공식 없음)")
        else:
            ax.plot(xs, ys, label="Severity-threshold", **STYLE["Severity-threshold"])
        for name in ["No-help (LLM only)", "Oracle-router (optimal)", "Always-escalate", "LLM-reasoned (ours)"]:
            x, y = pol[name][0]
            ax.plot([x], [y], label=name, linestyle="none", **STYLE[name])
        ax.set_xlabel("Human-help rate (escalation %)", fontsize=13); ax.set_ylabel("Safety = 1 − EMERGENCY 누락 (%)", fontsize=12)
        ax.set_title(title, fontsize=12); ax.set_xlim(-3, 103); ax.set_ylim(84, 101)
        ax.grid(True, alpha=0.35); ax.set_axisbelow(True); ax.tick_params(labelsize=11)
    axes[0].legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.suptitle("위임 정책 — In-dist vs OOD. 신규 타입에선 규칙 기반 위임이 불가, LLM만 일반화", fontsize=14, y=1.0)
    fig.tight_layout()
    out = Path(out_dir) / (fname or "fig_deferral_split.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); print(f"[OK] Saved → {out}")
    if show: plt.show()
    plt.close(fig)


def escalation_roc_fig(df, out_dir, show):
    label = df["escalation_gt"].to_numpy()
    sev = df["severity_score"].to_numpy()
    fpr, tpr, auc = _roc(sev, label)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(fpr, tpr, color="#F4A261", lw=2.5, label=f"Severity score (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1.5, label="Random (AUC=0.5)")
    # LLM-reasoned 운영점 (FPR=과의존, TPR=recall)
    e = df["hitl_escalated"].to_numpy()
    tp = (e & label).sum(); fp = (e & ~label).sum(); fn = (~e & label).sum(); tn = (~e & ~label).sum()
    rec = tp/(tp+fn) if (tp+fn) else 0; fpr_pt = fp/(fp+tn) if (fp+tn) else 0
    ax.plot([fpr_pt], [rec], marker="*", ms=24, color="#E63946", linestyle="none",
            label=f"LLM-reasoned (ours): recall={rec:.2f}, FPR={fpr_pt:.2f}")
    ax.set_xlabel("False Positive Rate (과의존: 불필요 위임)", fontsize=13)
    ax.set_ylabel("True Positive Rate (recall: 위험 포착)", fontsize=13)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.35); ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="lower right"); ax.set_title("에스컬레이션 판별력 (위험/비위험 분별)", fontsize=14)
    fig.tight_layout()
    out = Path(out_dir) / "fig_escalation_roc.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); print(f"[OK] Saved → {out}")
    if show: plt.show()
    plt.close(fig)
    return auc


def print_tables(df):
    print("\n" + "=" * 78)
    print("  위임 정책 비교 (전 모델 풀, 실제 데이터)")
    print("=" * 78)
    for metric, lbl in [("missed", "EMERGENCY 누락"), ("danger", "Tier2 위험오류")]:
        pol = policy_points(df, metric)
        ours_help, ours_safe = pol["LLM-reasoned (ours)"][0]
        # 동일 help budget에서 random/threshold 안전 비교
        rnd = 100 - (1 - ours_help/100) * (100 - pol["No-help (LLM only)"][0][1])
        thr = min(pol["Severity-threshold"], key=lambda p: abs(p[0] - ours_help))
        print(f"\n[Safety = 1 − {lbl}]  (help가 같을 때 누가 더 안전한가)")
        print(f"  No-help(LLM단독)       : help=  0%  safety={pol['No-help (LLM only)'][0][1]:5.1f}%")
        print(f"  Oracle-router(최적)    : help={pol['Oracle-router (optimal)'][0][0]:4.0f}%  safety={pol['Oracle-router (optimal)'][0][1]:5.1f}%  (최소 help로 100%)")
        print(f"  --- 우리 help={ours_help:.0f}% 에서 ---")
        print(f"  Random                 : safety={rnd:5.1f}%")
        print(f"  Severity-threshold(근사): help={thr[0]:4.0f}%  safety={thr[1]:5.1f}%")
        print(f"  LLM-reasoned (ours)    : safety={ours_safe:5.1f}%   ← 같은 help에서 random 대비 +{ours_safe-rnd:.1f}%p")
    print("=" * 78 + "\n")


def _min_help(curve, target):
    ok = [h for h, s in curve if s >= target - 1e-9]
    return min(ok) if ok else np.nan

def _min_help_random(df, metric, target):
    err0 = 100 - _safety(df, np.zeros(len(df), bool), metric)
    if err0 <= (100 - target):
        return 0.0
    return max(0.0, (1 - (100 - target) / err0)) * 100


def min_help_fig(df, out_dir, show, metric="missed", fname=None):
    """목표 안전 달성에 필요한 최소 human-help (막대) — 낮을수록 효율적."""
    pol = policy_points(df, metric)
    if "LLM-confidence(τ)" not in pol:
        print("[skip] min_help_fig: confidence 없음"); return
    targets = [98, 99, 100]
    series = {"LLM-confidence (ours)": [_min_help(pol["LLM-confidence(τ)"], t) for t in targets],
              "Severity-threshold": [_min_help(pol["Severity-threshold"], t) for t in targets],
              "Random": [_min_help_random(df, metric, t) for t in targets]}
    colors = {"LLM-confidence (ours)": "#1D5DB8", "Severity-threshold": "#F4A261", "Random": "#bbb"}
    fig, ax = plt.subplots(figsize=(9, 6))
    keys = list(series); width = 0.8 / len(keys); x = np.arange(len(targets))
    for i, k in enumerate(keys):
        offs = (i - (len(keys) - 1) / 2) * width
        vals = [v if v == v else 0 for v in series[k]]
        bars = ax.bar(x + offs, vals, width, label=k, color=colors[k], alpha=0.9, edgecolor="#222", linewidth=0.7)
        for b, v in zip(bars, series[k]):
            ax.text(b.get_x() + b.get_width() / 2, (v if v == v else 0) + 1.2,
                    f"{v:.0f}" if v == v else "N/A", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"≥{t}%" for t in targets], fontsize=13)
    ax.set_xlabel("Target EMERGENCY safety", fontsize=14)
    ax.set_ylabel("최소 human-help rate (%)  ↓ 낮을수록 효율", fontsize=13)
    ax.set_ylim(0, 105); ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.35); ax.set_axisbelow(True); ax.legend(fontsize=11, framealpha=0.95)
    ax.set_title("목표 안전 달성에 필요한 인간 개입량 (적을수록 우수)", fontsize=14)
    fig.tight_layout()
    out = Path(out_dir) / (fname or "fig_min_help.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); print(f"[OK] Saved → {out}")
    if show: plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/metrics_raw.csv")
    p.add_argument("--output-dir", default="results/figures")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()
    show = not a.no_show
    df = _coerce(pd.read_csv(a.input))
    print(f"[INFO] {len(df)} rows, models {sorted(df['model'].unique())}")
    print_tables(df)
    deferral_curve_fig(df, a.output_dir, show)
    min_help_fig(df, a.output_dir, show)
    deferral_split_fig(df, a.output_dir, show)
    auc = escalation_roc_fig(df, a.output_dir, show)
    # OOD에서 LLM vs 규칙불가 정량
    ood = df[df.is_ood]
    if len(ood):
        e = ood["hitl_escalated"].to_numpy()
        print(f"[OOD] LLM-reasoned: help={help_rate(e):.0f}%  EMERGENCY누락={missed_emerg_rate(ood, e):.1f}%  "
              f"(severity/SOP 규칙은 신규타입에 부재)")
    print(f"[INFO] Severity escalation AUC = {auc:.3f}")


if __name__ == "__main__":
    main()
