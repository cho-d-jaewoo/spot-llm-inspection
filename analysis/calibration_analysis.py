"""LLM confidence 신뢰도 보정 — reliability diagram(과신 시각화) + temperature scaling + 보정 전/후 위임 곡선."""

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

sys.path.insert(0, str(Path(__file__).parent))
from deferral_analysis import _coerce, _safety, help_rate

DPI = 200
EPS = 1e-6


# ── temperature scaling (1-param, NLL 최소화 grid scan) ──────────────────────
def fit_temperature(p, y):
    logit = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    best_T, best_nll = 1.0, 1e18
    for T in np.linspace(0.3, 15.0, 600):
        pc = 1.0 / (1.0 + np.exp(-logit / T))
        pc = np.clip(pc, EPS, 1 - EPS)
        nll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


def apply_temperature(p, T):
    logit = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    return 1.0 / (1.0 + np.exp(-logit / T))


def ece(p, y, nbins=8):
    """Expected Calibration Error (%)."""
    edges = np.linspace(0, 1, nbins + 1); e = 0.0
    for i in range(nbins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i > 0 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(p)
    return e * 100


# ── reliability diagram (raw vs calibrated) ──────────────────────────────────
def reliability_fig(p_raw, p_cal, y, T, out_dir, show, nbins=8, fname=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, p, title in zip(axes, [p_raw, p_cal],
                            [f"Raw confidence (ECE={ece(p_raw,y):.1f}%)",
                             f"Temperature-scaled (T={T:.2f}, ECE={ece(p_cal,y):.1f}%)"]):
        edges = np.linspace(0, 1, nbins + 1); xs, ys, ws = [], [], []
        for i in range(nbins):
            m = (p > edges[i]) & (p <= edges[i + 1]) if i > 0 else (p >= edges[i]) & (p <= edges[i + 1])
            if m.sum() == 0:
                continue
            xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
        ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.5, label="Perfect calibration")
        sizes = 60 + 600 * np.array(ws) / max(ws)
        ax.scatter(xs, ys, s=sizes, color="#E63946", alpha=0.8, edgecolors="#222", zorder=3, label="LLM (bin)")
        ax.plot(xs, ys, color="#E63946", alpha=0.5, zorder=2)
        ax.set_xlabel("Mean confidence", fontsize=13); ax.set_ylabel("Actual accuracy", fontsize=13)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.35); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=13); ax.legend(fontsize=10, loc="upper left")
        ax.text(0.97, 0.05, "점 위=과소확신\n점 아래=과신", ha="right", va="bottom",
                fontsize=9, color="#555", transform=ax.transAxes)
    fig.suptitle("신뢰도 보정 (reliability) — 대각선 위=잘맞음, 아래=과신", fontsize=15, y=1.0)
    fig.tight_layout()
    out = Path(out_dir) / (fname or "fig_reliability.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); print(f"[OK] Saved → {out}")
    if show: plt.show()
    plt.close(fig)


# ── 보정 전/후 위임 곡선 (단조 보정이라 곡선은 거의 동일함을 입증) ───────────
def _conf_curve(df, score, metric):
    pts = []
    for tau in [-0.01] + sorted(set(np.round(score, 6).tolist())) + [1.01]:
        e = score < tau
        pts.append((help_rate(e), _safety(df, e, metric)))
    return sorted(set(pts))

def _sev_curve(df, metric):
    sev = df["severity_score"].to_numpy(); pts = []
    for tau in sorted(set(sev.tolist())) + [sev.max() + 1e-6]:
        e = sev >= tau; pts.append((help_rate(e), _safety(df, e, metric)))
    return sorted(pts)


def deferral_calibrated_fig(df, p_raw, p_cal, out_dir, show, fname=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, metric, title in zip(axes, ["missed", "danger"],
                                 ["Safety = 1 − EMERGENCY 누락", "Safety = 1 − Tier2 위험오류"]):
        err0 = 100 - _safety(df, np.zeros(len(df), bool), metric)
        ax.plot([0, 100], [100 - err0, 100], ls="--", color="#bbb", lw=2, label="Random")
        sc = _sev_curve(df, metric)
        ax.plot([p[0] for p in sc], [p[1] for p in sc], color="#F4A261", marker="o", ms=5, lw=2, label="Severity-threshold")
        cr = _conf_curve(df, p_raw, metric)
        ax.plot([p[0] for p in cr], [p[1] for p in cr], color="#1D5DB8", marker="o", ms=5, lw=2.5, label="LLM-confidence (raw)")
        cc = _conf_curve(df, p_cal, metric)
        ax.plot([p[0] for p in cc], [p[1] for p in cc], color="#2A9D8F", ls=":", lw=2.5, label="LLM-confidence (calibrated)")
        ax.set_xlabel("Human-help rate (escalation %)", fontsize=13); ax.set_ylabel("Safety (%)", fontsize=13)
        ax.set_title(title, fontsize=13); ax.set_xlim(-3, 103); ax.set_ylim(85, 101)
        ax.grid(True, alpha=0.35); ax.set_axisbelow(True); ax.tick_params(labelsize=11)
    axes[0].legend(fontsize=10, loc="lower right", framealpha=0.95)
    fig.suptitle("보정 전/후 위임 곡선 — 단조 보정은 순위 불변 → 곡선 일치 (보정은 '임계 해석'을 줌)", fontsize=14, y=1.0)
    fig.tight_layout()
    out = Path(out_dir) / (fname or "fig_deferral_calibrated.png"); out.parent.mkdir(parents=True, exist_ok=True)
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
    if "tier2_confidence" not in df.columns:
        print("[ERR] tier2_confidence 컬럼 없음 — confidence 출력하는 새 코드로 재실행 필요."); return
    df["llm_full_tier2_correct"] = df["llm_full_tier2_correct"].astype(str).str.lower().isin(["true", "1", "yes"])

    p_raw = df["tier2_confidence"].to_numpy()
    y = df["llm_full_tier2_correct"].to_numpy().astype(float)
    T = fit_temperature(p_raw, y)
    p_cal = apply_temperature(p_raw, T)
    print(f"[INFO] n={len(df)}  base accuracy={y.mean()*100:.1f}%")
    print(f"[INFO] Temperature T={T:.3f}  (T>1 → 과신 완화)")
    print(f"[INFO] ECE raw={ece(p_raw,y):.1f}%  →  calibrated={ece(p_cal,y):.1f}%")
    print(f"[INFO] mean conf raw={p_raw.mean():.3f} → cal={p_cal.mean():.3f}  (실제 정답률 {y.mean():.3f})")

    reliability_fig(p_raw, p_cal, y, T, a.output_dir, show)
    deferral_calibrated_fig(df, p_raw, p_cal, a.output_dir, show)

    # 목표 안전을 달성하는 최소 help (confidence 순위 기반, 보정 불변) — KnowNo "minimal help"
    cr = _conf_curve(df, p_raw, "missed")
    print("\n[목표 안전 달성에 필요한 최소 human-help (confidence 순위 기반)]")
    for target in [98.0, 99.0, 100.0]:
        ok = [h for h, s in cr if s >= target - 1e-9]
        print(f"  EMERGENCY 안전 ≥{target:.0f}%  →  최소 help = {min(ok):.0f}%" if ok else
              f"  EMERGENCY 안전 ≥{target:.0f}%  →  미달성")
    print("  (참고: 보정은 단조 변환이라 이 곡선/최소help는 불변. 보정은 임계값을 '해석 가능'하게 만들 뿐.)")


if __name__ == "__main__":
    main()
