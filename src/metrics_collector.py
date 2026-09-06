"""결정 이벤트 수집 + CSV 저장 + 요약 출력 (Fixed / LLM-Full / LLM+HiTL 3개 mode 동시 기록)."""

import csv, json
from dataclasses import dataclass, field, asdict


@dataclass
class DecisionRecord:
    # ── 식별자 ──
    lap: int
    zone_id: str
    anomaly_type: str
    is_ood: bool
    attributes: dict
    severity_score: float
    # ── Ground Truth ──
    tier1_gt: str
    tier2_gt: str
    escalation_gt: bool
    # ── Fixed ──
    fixed_tier1: str
    fixed_tier2: str
    fixed_tier1_correct: bool
    fixed_tier2_correct: bool
    # ── 모델 ──
    model: str
    # ── LLM-Full ──
    llm_full_tier1: str
    llm_full_tier2: str
    llm_full_tier1_correct: bool
    llm_full_tier2_correct: bool
    llm_full_decision_time_ms: float
    # ── LLM+HiTL ──
    hitl_tier1: str
    hitl_escalated: bool
    tier2_confidence: float
    hitl_tier2_final: str
    hitl_tier1_correct: bool
    hitl_escalation_correct: bool
    hitl_tier2_correct: bool
    hitl_decision_time_ms: float
    hitl_reasoning: str


# CSV 컬럼 순서 (attributes는 JSON 문자열 + attr_* 평탄화 둘 다 저장)
_BASE_FIELDS = [
    "lap", "zone_id", "anomaly_type", "is_ood", "severity_score",
    "tier1_gt", "tier2_gt", "escalation_gt",
    "fixed_tier1", "fixed_tier2", "fixed_tier1_correct", "fixed_tier2_correct",
    "model",
    "llm_full_tier1", "llm_full_tier2", "llm_full_tier1_correct", "llm_full_tier2_correct",
    "llm_full_decision_time_ms",
    "hitl_tier1", "hitl_escalated", "tier2_confidence", "hitl_tier2_final",
    "hitl_tier1_correct", "hitl_escalation_correct", "hitl_tier2_correct",
    "hitl_decision_time_ms", "hitl_reasoning",
    "attributes",
]


class MetricsCollector:
    def __init__(self):
        self.records: list[DecisionRecord] = []

    def add(self, record: DecisionRecord):
        self.records.append(record)

    def save_csv(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        attr_keys = sorted({k for r in self.records for k in r.attributes.keys()})
        fieldnames = _BASE_FIELDS + [f"attr_{k}" for k in attr_keys]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                d = asdict(r)
                row = {k: d[k] for k in _BASE_FIELDS if k != "attributes"}
                row["severity_score"] = round(r.severity_score, 3)
                row["attributes"] = json.dumps(r.attributes, ensure_ascii=False)
                for k in attr_keys:
                    row[f"attr_{k}"] = r.attributes.get(k, "")
                writer.writerow(row)
        print(f"[METRICS] CSV saved → {path}  ({len(self.records)} rows)")

    def print_summary(self):
        if not self.records:
            print("\n[METRICS] no decisions recorded."); return
        n = len(self.records)
        def acc(getter):
            return sum(1 for r in self.records if getter(r)) / n * 100

        models = sorted({r.model for r in self.records})
        print("\n" + "=" * 70)
        print(f"  METRICS SUMMARY — models: {', '.join(models)} — {n} decisions")
        print("=" * 70)
        print("\n[Tier1 Accuracy]")
        print(f"  Fixed     : {acc(lambda r: r.fixed_tier1_correct):5.1f}%")
        print(f"  LLM-Full  : {acc(lambda r: r.llm_full_tier1_correct):5.1f}%")
        print(f"  LLM+HiTL  : {acc(lambda r: r.hitl_tier1_correct):5.1f}%")
        print("\n[Tier2 Accuracy]")
        print(f"  Fixed     : {acc(lambda r: r.fixed_tier2_correct):5.1f}%")
        print(f"  LLM-Full  : {acc(lambda r: r.llm_full_tier2_correct):5.1f}%")
        print(f"  LLM+HiTL  : {acc(lambda r: r.hitl_tier2_correct):5.1f}%")

        # 에스컬레이션 P/R/F1 (escalation_gt 기준)
        tp = sum(1 for r in self.records if r.hitl_escalated and r.escalation_gt)
        fp = sum(1 for r in self.records if r.hitl_escalated and not r.escalation_gt)
        fn = sum(1 for r in self.records if not r.hitl_escalated and r.escalation_gt)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print("\n[Escalation (LLM+HiTL)]")
        print(f"  Precision : {prec:.3f}   Recall : {rec:.3f}   F1 : {f1:.3f}")
        print(f"  Escalation accuracy : {acc(lambda r: r.hitl_escalation_correct):5.1f}%")

        # 안전 critical: tier2_gt == EMERGENCY
        crit = [r for r in self.records if r.tier2_gt == "EMERGENCY"]
        if crit:
            m = len(crit)
            cf = sum(1 for r in crit if r.fixed_tier2_correct) / m * 100
            cl = sum(1 for r in crit if r.llm_full_tier2_correct) / m * 100
            ch = sum(1 for r in crit if r.hitl_tier2_correct) / m * 100
            print(f"\n[Safety Critical — EMERGENCY cases, n={m}]")
            print(f"  Fixed {cf:5.1f}%   LLM-Full {cl:5.1f}%   LLM+HiTL {ch:5.1f}%")

        times = [r.hitl_decision_time_ms for r in self.records]
        print(f"\n[Decision Time] avg/min/max : "
              f"{sum(times)/len(times):.0f} / {min(times):.0f} / {max(times):.0f} ms")
        print("=" * 70 + "\n")
