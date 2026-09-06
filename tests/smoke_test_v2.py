"""Isaac Sim 비의존 스모크 테스트 — v3 config 검증 + 합성 CSV 생성 (export/plot/report 입력용)."""

import sys, os, random, itertools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_config import (ANOMALY_CONFIG, ANOMALY_TYPES, OOD_TYPES, compute_severity,
                            get_ground_truth, get_fixed_decision, sample_attributes, build_prompt, is_ood)
from oracle import oracle_tier2
from metrics_collector import MetricsCollector, DecisionRecord

TIER1 = ["APPROACH", "WAIT_AND_OBSERVE", "KEEP_DISTANCE"]
TIER2 = ["EMERGENCY", "MAINTENANCE", "CONTINUE", "FALSE_ALARM"]
T1_LVL = {a: i for i, a in enumerate(TIER1)}

# ── 1) 전 속성조합 무결성 검증 ────────────────────────────────────────────────
total = 0
for atype, cfg in ANOMALY_CONFIG.items():
    pool = cfg["attribute_pool"]; keys = list(pool)
    for combo in itertools.product(*[pool[k] for k in keys]):
        attrs = dict(zip(keys, combo))
        gt = get_ground_truth(atype, attrs)
        assert gt["tier1"] in TIER1 and gt["tier2"] in TIER2, (atype, attrs, gt)
        assert 0.0 <= compute_severity(atype, attrs) <= 1.0
        assert oracle_tier2(atype, attrs) == gt["tier2"]
        assert "{" not in build_prompt(atype, "Z", attrs)
        total += 1
print(f"[OK] {total} combos validated, types={len(ANOMALY_TYPES)} (OOD={OOD_TYPES})")

# ── 2) 합성 데이터 → CSV ──────────────────────────────────────────────────────
rng = random.Random(0)
collector = MetricsCollector()
ZONES = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E"]
models = ["haiku", "sonnet", "gpt4o", "llama"]
# 합성 LLM: in-dist는 똑똑(80%), OOD는 약간 덜(65%); 위험할수록 탈출 경향
for lap in range(1, 11):
    for zone in ZONES:
        atype = rng.choice(ANOMALY_TYPES)
        attrs = sample_attributes(atype, rng)
        gt = get_ground_truth(atype, attrs); fx = get_fixed_decision(atype, attrs)
        sev = compute_severity(atype, attrs)
        for model in models:
            p_ok = 0.65 if is_ood(atype) else 0.8
            llm_t1 = gt["tier1"] if rng.random() < p_ok else rng.choice(TIER1)
            # 탈출 경향: GT가 탈출이면 자주 탈출 + 무작위
            llm_esc = gt["escalation"] if rng.random() < 0.72 else (not gt["escalation"])
            llm_t2 = gt["tier2"] if rng.random() < (p_ok - 0.1) else rng.choice(TIER2)
            hitl_t2 = oracle_tier2(atype, attrs) if llm_esc else llm_t2
            # 합성 confidence: 정답이면 높게, 오답이면 낮게 (대략 보정됨) + 노이즈
            conf = (0.82 if llm_t2 == gt["tier2"] else 0.45) + rng.uniform(-0.12, 0.12)
            conf = max(0.0, min(1.0, conf))
            ms = rng.uniform(400, 1800)
            collector.add(DecisionRecord(
                lap=lap, zone_id=zone, anomaly_type=atype, is_ood=is_ood(atype),
                attributes=attrs, severity_score=sev,
                tier1_gt=gt["tier1"], tier2_gt=gt["tier2"], escalation_gt=gt["escalation"],
                fixed_tier1=fx["tier1"], fixed_tier2=fx["tier2"],
                fixed_tier1_correct=(fx["tier1"] == gt["tier1"]), fixed_tier2_correct=(fx["tier2"] == gt["tier2"]),
                model=model, llm_full_tier1=llm_t1, llm_full_tier2=llm_t2,
                llm_full_tier1_correct=(llm_t1 == gt["tier1"]), llm_full_tier2_correct=(llm_t2 == gt["tier2"]),
                llm_full_decision_time_ms=round(ms, 1),
                hitl_tier1=llm_t1, hitl_escalated=llm_esc, tier2_confidence=conf, hitl_tier2_final=hitl_t2,
                hitl_tier1_correct=(llm_t1 == gt["tier1"]),
                hitl_escalation_correct=(llm_esc == gt["escalation"]),
                hitl_tier2_correct=(hitl_t2 == gt["tier2"]),
                hitl_decision_time_ms=round(ms, 1), hitl_reasoning="synthetic."))

out = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_synthetic.csv")
collector.save_csv(out)
collector.print_summary()
print(f"[OK] synthetic CSV → {out} ({len(collector.records)} rows)")
