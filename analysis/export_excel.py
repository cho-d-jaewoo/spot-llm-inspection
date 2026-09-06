"""metrics_raw.csv → Excel 변환 (Raw + Tier1/Tier2 정확도 + 에스컬레이션 + 안전 + 결정시간 시트)."""

import argparse, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import pandas as pd

BOOL_COLS = ["escalation_gt", "fixed_tier1_correct", "fixed_tier2_correct",
             "llm_full_tier1_correct", "llm_full_tier2_correct", "hitl_escalated",
             "hitl_tier1_correct", "hitl_escalation_correct", "hitl_tier2_correct", "is_ood"]
OOD_TYPES = {"GAS_LEAK", "ELECTRICAL_FAULT", "STRUCTURAL_DAMAGE"}


def _coerce_bool(df):
    """CSV의 'True'/'False' 문자열 → 실제 bool."""
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    if "is_ood" not in df.columns:
        df["is_ood"] = df["anomaly_type"].isin(OOD_TYPES)
    return df


def tier_accuracy(df, tier):
    """모델별 Fixed/LLM-Full/LLM+HiTL의 tier(1/2) 정확도 표."""
    fixed_col = f"fixed_tier{tier}_correct"
    full_col  = f"llm_full_tier{tier}_correct"
    hitl_col  = f"hitl_tier{tier}_correct"
    rows = []
    for model, g in df.groupby("model"):
        rows.append({"model": model,
                     "Fixed (%)":    round(g[fixed_col].mean() * 100, 1),
                     "LLM-Full (%)": round(g[full_col].mean() * 100, 1),
                     "LLM+HiTL (%)": round(g[hitl_col].mean() * 100, 1),
                     "N": len(g)})
    return pd.DataFrame(rows)


def escalation_analysis(df):
    """모델별 에스컬레이션 Precision/Recall/F1 + GT True/False 케이스별 Tier2 정확도."""
    rows = []
    for model, g in df.groupby("model"):
        tp = ((g["hitl_escalated"]) & (g["escalation_gt"])).sum()
        fp = ((g["hitl_escalated"]) & (~g["escalation_gt"])).sum()
        fn = ((~g["hitl_escalated"]) & (g["escalation_gt"])).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        gt_true  = g[g["escalation_gt"]]
        gt_false = g[~g["escalation_gt"]]
        rows.append({
            "model": model,
            "Precision": round(prec, 3), "Recall": round(rec, 3), "F1": round(f1, 3),
            "EscAcc (%)": round(g["hitl_escalation_correct"].mean() * 100, 1),
            "Tier2Acc | GT=True (%)":  round(gt_true["hitl_tier2_correct"].mean() * 100, 1) if len(gt_true) else None,
            "Tier2Acc | GT=False (%)": round(gt_false["hitl_tier2_correct"].mean() * 100, 1) if len(gt_false) else None,
        })
    return pd.DataFrame(rows)


def safety_critical(df):
    """tier2_gt == EMERGENCY 케이스의 모델별 × 조건별 Tier2 정확도 (가장 중요한 안전 지표)."""
    crit = df[df["tier2_gt"] == "EMERGENCY"]
    rows = []
    for model, g in crit.groupby("model"):
        rows.append({"model": model,
                     "Fixed (%)":    round(g["fixed_tier2_correct"].mean() * 100, 1),
                     "LLM-Full (%)": round(g["llm_full_tier2_correct"].mean() * 100, 1),
                     "LLM+HiTL (%)": round(g["hitl_tier2_correct"].mean() * 100, 1),
                     "N_emergency": len(g)})
    return pd.DataFrame(rows)


def decision_time(df):
    """모델별 결정 시간 mean/min/max/std (ms) — HiTL 기준."""
    rows = []
    for model, g in df.groupby("model"):
        t = g["hitl_decision_time_ms"]
        rows.append({"model": model, "mean_ms": round(t.mean(), 1), "min_ms": round(t.min(), 1),
                     "max_ms": round(t.max(), 1), "std_ms": round(t.std(), 1), "N": len(g)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/metrics_raw.csv")
    p.add_argument("--output", default="results/inspection_v2_results.xlsx")
    a = p.parse_args()

    df = _coerce_bool(pd.read_csv(a.input))
    print(f"[INFO] Loaded {len(df)} rows, models: {sorted(df['model'].unique())}")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Raw Data", index=False)
        tier_accuracy(df, 1).to_excel(xl, sheet_name="Tier1 Accuracy", index=False)
        tier_accuracy(df, 2).to_excel(xl, sheet_name="Tier2 Accuracy", index=False)
        escalation_analysis(df).to_excel(xl, sheet_name="Escalation Analysis", index=False)
        safety_critical(df).to_excel(xl, sheet_name="Safety Critical", index=False)
        decision_time(df).to_excel(xl, sheet_name="Decision Time", index=False)
    print(f"[OK] Excel saved → {out}")


if __name__ == "__main__":
    main()
