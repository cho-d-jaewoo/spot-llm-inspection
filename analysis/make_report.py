"""metrics_raw.csv → 논문용 결과 표 (markdown) — 프레임워크 실용성 중심 (안전 ordinal, 위험오류 분류, OOD 일반화)."""

import argparse, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import pandas as pd

MODEL_LABELS = {"haiku": "Haiku", "sonnet": "Sonnet", "gpt4o": "GPT-4o mini", "llama": "Llama"}
OOD_TYPES = {"GAS_LEAK", "ELECTRICAL_FAULT", "STRUCTURAL_DAMAGE"}
T1_LVL = {"APPROACH": 0, "WAIT_AND_OBSERVE": 1, "KEEP_DISTANCE": 2}
T2_RANK = {"FALSE_ALARM": 0, "CONTINUE": 1, "MAINTENANCE": 2, "EMERGENCY": 3}
TIER2_ORDER = ["EMERGENCY", "MAINTENANCE", "CONTINUE", "FALSE_ALARM"]
BOOL_COLS = ["escalation_gt", "fixed_tier1_correct", "fixed_tier2_correct",
             "llm_full_tier1_correct", "llm_full_tier2_correct", "hitl_escalated",
             "hitl_tier1_correct", "hitl_escalation_correct", "hitl_tier2_correct", "is_ood"]
# 조건 → (tier1 col, tier2 col)
CONDS = {"Fixed (SOP)": ("fixed_tier1", "fixed_tier2"),
         "LLM-Full": ("llm_full_tier1", "llm_full_tier2"),
         "LLM+HiTL": ("hitl_tier1", "hitl_tier2_final")}


def _coerce(df):
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    if "is_ood" not in df.columns:
        df["is_ood"] = df["anomaly_type"].isin(OOD_TYPES)
    return df


def _mlabel(m):
    return MODEL_LABELS.get(m, m)


def md_table(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def cond_metrics(df, t1col, t2col):
    """한 조건의 안전 중심 지표 (%). 비어있으면 None."""
    if len(df) == 0:
        return None
    n = len(df)
    p1, g1 = df[t1col], df["tier1_gt"]
    p2, g2 = df[t2col], df["tier2_gt"]
    pr2 = p2.map(T2_RANK); gr2 = g2.map(T2_RANK)
    g2_haz = g2.isin(["EMERGENCY", "MAINTENANCE"])
    emerg = df[g2 == "EMERGENCY"]
    return {
        "exact_t2": (p2 == g2).mean() * 100,
        "t1_unsafe": (p1.map(T1_LVL) < g1.map(T1_LVL)).mean() * 100,      # 덜 보수적 = 위험
        "t2_danger": ((pr2 < gr2) & g2_haz).mean() * 100,                  # 위험 과소대응
        "missed_emerg": ((emerg[t2col] != "EMERGENCY").mean() * 100) if len(emerg) else float("nan"),
        "t2_over": (pr2 > gr2).mean() * 100,                               # 과대응(안전하나 비효율)
        "n": n,
    }


def esc_metrics(g):
    tp = int((g.hitl_escalated & g.escalation_gt).sum())
    fp = int((g.hitl_escalated & ~g.escalation_gt).sum())
    fn = int((~g.hitl_escalated & g.escalation_gt).sum())
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    non = g[~g.hitl_escalated]
    return {"esc_rate": g.hitl_escalated.mean()*100, "auto_rate": (1-g.hitl_escalated.mean())*100,
            "prec": prec, "rec": rec, "f1": f1, "esc_acc": g.hitl_escalation_correct.mean()*100,
            "auto_acc": (non.hitl_tier2_correct.mean()*100) if len(non) else float("nan"), "auto_n": len(non)}


def _f(x):
    return "-" if x != x else f"{x:.1f}"   # nan 처리


def build_report(df):
    models = sorted(df["model"].unique())
    ev = df.drop_duplicates(["lap", "zone_id"])
    gt_esc = ev.escalation_gt.mean()*100
    sec = []
    sec.append("# SPOT 2-Tier Inspection — 결과 리포트 (v3: 프레임워크 실용성)\n")
    sec.append(f"- 모델: {', '.join(_mlabel(m) for m in models)} (강건성 확인용 부차 변수)")
    sec.append(f"- 총 {len(df)}행, laps={df['lap'].nunique()}, In-dist {(~ev.is_ood).sum()} / OOD {ev.is_ood.sum()} 이벤트")
    sec.append(f"- GT 탈출율 {gt_esc:.1f}%\n")
    sec.append("## 해석 가이드\n")
    sec.append("- **Fixed = 현행 공장 SOP** (타입 단위 고정 반응, 속성 무시). 이상적 GT가 아니라 **현실 baseline**.")
    sec.append("- **LLM**은 법 원칙(프롬프트 제공)으로 속성/맥락을 통합. **HiTL**은 고위험·불확실을 인간 위임.")
    sec.append("- **OOD**(GAS_LEAK·ELECTRICAL_FAULT·STRUCTURAL_DAMAGE) = 사전 SOP가 모르는 신규 타입 → 일반화 테스트.")
    sec.append("- exact 정확도 대신 **안전 중심 지표**: Tier1 위험률(덜 보수적), Tier2 **위험 과소대응률** / **EMERGENCY 누락률**, 과대응률.")
    sec.append("- HiTL의 raw 정확도는 오라클 의존이 있으므로, 핵심은 **위험 누락 ↓ @ 자동화율** + 에스컬레이션 품질.\n")

    # ── Table 1 (HEADLINE): 프레임워크 집계 (전 모델 풀) ──
    rows = []
    for cond, (c1, c2) in CONDS.items():
        m = cond_metrics(df, c1, c2)
        auto = (1 - df.hitl_escalated.mean()) * 100 if cond == "LLM+HiTL" else 100.0
        rows.append([cond, _f(m["exact_t2"]), _f(m["t1_unsafe"]), f"**{_f(m['t2_danger'])}**",
                     _f(m["missed_emerg"]), _f(m["t2_over"]), f"{auto:.1f}"])
    sec.append("## Table 1 (HEADLINE). 프레임워크 집계 — 안전 중심 지표 (%, 전 모델)\n")
    sec.append("> 핵심: **Tier2 위험률**과 **EMERGENCY 누락률**이 낮을수록 안전. 자동화율은 인간 부담 역수.\n")
    sec.append(md_table(["조건", "Tier2 exact%", "Tier1 위험%", "Tier2 위험%", "EMERGENCY 누락%", "과대응%", "자동화%"], rows) + "\n")

    # ── Table 2: 일반화 (In-dist vs OOD) ──
    rows = []
    for cond, (c1, c2) in CONDS.items():
        mi = cond_metrics(df[~df.is_ood], c1, c2)
        mo = cond_metrics(df[df.is_ood], c1, c2)
        rows.append([cond, _f(mi["t2_danger"]), _f(mo["t2_danger"]),
                     _f(mi["missed_emerg"]), _f(mo["missed_emerg"])])
    sec.append("## Table 2. 일반화 — In-distribution vs OOD (신규 타입)\n")
    sec.append("> Fixed는 신규 타입(OOD)에 규칙이 없어 추락. LLM+법은 zero-shot 일반화, HiTL이 위험 잔여 차단.\n")
    sec.append(md_table(["조건", "Tier2위험% (In)", "Tier2위험% (OOD)", "EMERG누락% (In)", "EMERG누락% (OOD)"], rows) + "\n")

    # ── Table 3: 자동화 vs 안전 트레이드오프 좌표 (모델 + baseline) ──
    rows = []
    for m in models:
        g = df[df.model == m]; e = esc_metrics(g)
        emerg = g[g.tier2_gt == "EMERGENCY"]
        safe = (emerg.hitl_tier2_final == "EMERGENCY").mean()*100 if len(emerg) else float("nan")
        rows.append([_mlabel(m), f"{e['auto_rate']:.1f}", _f(safe)])
    fullsafe = df[df.tier2_gt == "EMERGENCY"]
    rows.append(["[baseline] Always-escalate", "0.0", "100.0"])
    rows.append(["[baseline] Fixed SOP",
                 "100.0", _f((fullsafe.fixed_tier2 == "EMERGENCY").mean()*100)])
    rows.append(["[baseline] LLM-Full (never-esc)",
                 "100.0", _f((fullsafe.llm_full_tier2 == "EMERGENCY").mean()*100)])
    sec.append("## Table 3. 자동화율 vs 안전(EMERGENCY 정확도) 트레이드오프\n")
    sec.append(md_table(["System", "자동화%", "EMERGENCY 안전%"], rows) + "\n")

    # ── Table 4: 에스컬레이션 품질 + 자율 실력 (모델별) ──
    rows = []
    for m in models:
        e = esc_metrics(df[df.model == m])
        rows.append([_mlabel(m), f"{e['esc_rate']:.1f}", f"{e['prec']:.3f}", f"{e['rec']:.3f}",
                     f"{e['f1']:.3f}", f"{e['esc_acc']:.1f}", f"{_f(e['auto_acc'])} (n={e['auto_n']})"])
    sec.append("## Table 4. 에스컬레이션 품질 & 자율 실력 (모델별)\n")
    sec.append("> 자율 정확도 = 탈출 안 한 케이스의 Tier2 정확도(오라클 도움 0).\n")
    sec.append(md_table(["Model", "탈출율%", "Precision", "Recall", "F1", "EscAcc%", "자율정확도%"], rows) + "\n")

    # ── Table 5: 모델별 위험률 (강건성) ──
    rows = []
    for m in models:
        g = df[df.model == m]
        full = cond_metrics(g, "llm_full_tier1", "llm_full_tier2")
        hitl = cond_metrics(g, "hitl_tier1", "hitl_tier2_final")
        rows.append([_mlabel(m), _f(full["t2_danger"]), _f(hitl["t2_danger"]),
                     _f(full["missed_emerg"]), _f(hitl["missed_emerg"])])
    sec.append("## Table 5. 모델별 위험률 (강건성 확인)\n")
    sec.append(md_table(["Model", "Tier2위험% Full", "Tier2위험% HiTL", "EMERG누락% Full", "EMERG누락% HiTL"], rows) + "\n")

    # ── Table 6: 결정 시간 ──
    rows = []
    for m in models:
        t = df[df.model == m]["hitl_decision_time_ms"]
        rows.append([_mlabel(m), f"{t.mean():.0f}", f"{t.min():.0f}", f"{t.max():.0f}", f"{t.std():.0f}"])
    sec.append("## Table 6. 결정 지연시간 (ms)\n")
    sec.append(md_table(["Model", "mean", "min", "max", "std"], rows) + "\n")

    # ── Table 7: Tier2 혼동행렬 (전 모델 합산) ──
    sec.append("## Table 7. Tier-2 혼동행렬 (전 모델 합산, GT → 예측)\n")
    for label, col in [("Fixed", "fixed_tier2"), ("LLM-Full", "llm_full_tier2"), ("LLM+HiTL", "hitl_tier2_final")]:
        ct = pd.crosstab(df["tier2_gt"], df[col]).reindex(index=TIER2_ORDER, columns=TIER2_ORDER, fill_value=0)
        rows = [[gt] + [int(ct.loc[gt, pr]) for pr in TIER2_ORDER] for gt in TIER2_ORDER]
        sec.append(f"\n**{label}** (행=GT, 열=예측)\n")
        sec.append(md_table(["GT \\ Pred"] + TIER2_ORDER, rows) + "\n")

    return "\n".join(sec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/metrics_raw.csv")
    p.add_argument("--output", default="results/report.md")
    a = p.parse_args()
    df = _coerce(pd.read_csv(a.input))
    report = build_report(df)
    print(report)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n[OK] Markdown report saved → {out}")


if __name__ == "__main__":
    main()
