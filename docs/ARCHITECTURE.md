# Architecture — two-tier LLM–human inspection decisions

Design spec. For how to run it and what the results were, see the
[README](../README.md).

Everything runs in Isaac Sim 5.0; no physical robot is involved. The entry
point is `src/spot_factory_v2.py`.

---

## Decision structure

```
patrol → anomaly detected
  ↓
[Tier 1 — always LLM or Fixed]
observation stance: APPROACH / WAIT_AND_OBSERVE / KEEP_DISTANCE
  ↓
SPOT executes the stance (ActionExecutor)
  ↓
[Tier 2 — depends on mode]
  ├─ fixed      rule-based Tier 2
  ├─ llm_full   LLM decides Tier 2 directly, never escalates
  └─ llm_hitl   LLM escalates based on its own confidence
        ├─ escalate=True  → oracle (GT) decides Tier 2
        └─ escalate=False → LLM decides Tier 2
  ↓
record all three modes → resume patrol
```

**A single Isaac Sim session collects decisions for all three modes at every
anomaly event**, so the comparison across conditions is paired on identical
events. SPOT's physical motion follows the `llm_hitl` decision.

### Tier 1 — observation stance

| Action | Meaning | Standoff |
|---|---|---|
| `APPROACH` | close observation | 1.5 m |
| `WAIT_AND_OBSERVE` | mid-range observation | 2.5 m |
| `KEEP_DISTANCE` | keep a safe distance | 4.0 m |

The robot performs an arc observation at the chosen radius. Tier 1 is
evaluated, but Tier 2 carries the safety-critical metrics.

### Tier 2 — resolution

| Action | Korean | Meaning |
|---|---|---|
| `EMERGENCY` | 긴급 대응 호출 | immediate fire/safety team call |
| `MAINTENANCE` | 정비팀 파견 | non-urgent specialist inspection |
| `CONTINUE` | 모니터링 유지 | resume patrol, keep watching |
| `FALSE_ALARM` | 오탐 처리 | dismiss, resume normal patrol |

---

## Comparison modes

| mode | Tier 1 | Escalation | Tier 2 |
|---|---|---|---|
| `fixed` | rule | none | rule |
| `llm_full` | LLM | none | LLM |
| `llm_hitl` | LLM | LLM decides | LLM or oracle |

---

## LLM output spec (tool calling)

`INSPECTION_TOOL` / `OPENAI_TOOL` in `src/llm_decision.py`.

```python
{
    "tier1_action":     str,    # APPROACH | WAIT_AND_OBSERVE | KEEP_DISTANCE
    "tier2_escalate":   bool,   # True = hand to human, False = decide autonomously
    "tier2_action":     str,    # EMERGENCY | MAINTENANCE | CONTINUE | FALSE_ALARM
                                #   (only valid when tier2_escalate=False)
    "tier2_confidence": float,  # 0.0-1.0, calibrated P(tier2_action is correct)
    "severity":         str,    # LOW | MEDIUM | HIGH
    "reasoning":        str,    # 2-3 sentences
}
```

When `tier2_escalate=True` the LLM's `tier2_action` is discarded and overwritten
with the oracle's ground-truth value.

> **In deployment**, do not use the binary `tier2_escalate` flag directly — it
> fires on 85 % of events. Use the `tier2_confidence` ranking with a calibrated
> threshold instead.

Model aliases are defined in `MODEL_ALIASES` (`haiku`, `sonnet`, `gpt4o`,
`llama`).

---

## Anomalies

Ground-truth rules live in `src/anomaly_config.py`; this is the design summary.
Four in-distribution types plus three OOD types with no prior rules
(`GAS_LEAK`, `ELECTRICAL_FAULT`, `STRUCTURAL_DAMAGE`) held out to test
generalization.

### Escalation ground truth

- `escalation_gt = True` ↔ Tier2_GT is EMERGENCY, or MAINTENANCE with a safety
  hazard attached.
- `escalation_gt = False` ↔ Tier2_GT is CONTINUE / FALSE_ALARM, or purely
  operational MAINTENANCE.

### Legal grounding

Injected into the prompt as *principles*, not as lookup rules — this is what
enables generalization to unseen anomaly types.

- 화재예방법 §40 (duty to report a fire immediately)
- 위험물안전관리법 §2 (hazardous material classification and reporting)
- 소방기본법 §20 (duty to call 119 on visible flame)
- 전기안전관리법 §22 (electrical installation safety)
- 산업안전보건기준에 관한 규칙 §241, §98
- KOSHA GUIDE E-184, C-15, M-171

### Severity formulas (in-distribution types)

**FIRE_RISK** — `temperature` {75, 90, 110, 150} °C · `workers_nearby` {0, 1, 3} · `flammables_adjacent` {T, F}

```
severity = clip((temp - 75) / (150 - 75) * 0.5 + workers * 0.1 + flammables * 0.2, 0, 1)
```

**LIQUID_LEAK** — `area` {0.3, 0.8, 2.5} m² · `substance` {water, coolant, lubricant, chemical} · `near_electrical` {T, F}

```
substance_score = {water: 0, coolant: 0.2, lubricant: 0.3, chemical: 0.6}
severity = clip(area / 2.5 * 0.2 + substance_score + near_electrical * 0.3, 0, 1)
```

**SMOKE_DETECTED** — `density` {light, moderate, dense} · `smell` {odorless, burning, chemical} · `visible_flame` {T, F}

```
density_score = {light: 0.1, moderate: 0.3, dense: 0.5}
smell_score   = {odorless: 0, burning: 0.2, chemical: 0.3}
severity = clip(density_score + smell_score + visible_flame * 0.4, 0, 1)
```

**AGV_STOPPED** — `stopped_duration` {2, 15, 60} min · `blocking_main_path` {T, F} · `load_status` {empty, partial, full}

```
load_score = {empty: 0, partial: 0.1, full: 0.2}
severity = clip(min(duration / 60, 1) * 0.4 + blocking * 0.3 + load_score, 0, 1)
```

The same anomaly type means different things depending on its attributes, which
is the core difference from a rule-based SOP.

---

## Simulation environment

- Isaac Sim 5.0, `SpotFlatTerrainPolicy` (pretrained locomotion).
  `PHYSICS_DT = RENDERING_DT = 0.002` (500 Hz).
- Environment USD: `Simple_Warehouse/warehouse_multiple_shelves.usd`
  (ceiling and pillars hidden, tiled floor).
- Five inspection zones: Assembly Line A (−7, −5) · Storage Area B (−2.5, 5) ·
  AGV Workspace (−4.6, 15.6) · Packaging Area (7, 11) · Loading Dock (2.5, −1.8).
- Patrol route: 8 waypoints around the perimeter, home at (0, −7). Detection
  radius 2.5 m.
- Event generation: each lap assigns a random anomaly type to each zone and
  samples attributes from its pool, then renders a natural-language report.
  Reproducible via `--seed` (default 42).
- Scale: 4 models × 10 laps = 49 events per model, 196 decisions total.

---

## Collected metrics

`DecisionRecord` in `src/metrics_collector.py`, written to
`results/metrics_raw.csv`, which every script in `analysis/` reads.

```
lap, zone_id, anomaly_type, attributes(dict), severity_score,
tier1_gt, tier2_gt, escalation_gt,

# Fixed
fixed_tier1, fixed_tier2, fixed_tier1_correct, fixed_tier2_correct,

# LLM-Full (per model)
llm_full_{model}_tier1, llm_full_{model}_tier2,
llm_full_{model}_tier1_correct, llm_full_{model}_tier2_correct,
llm_full_{model}_decision_time_ms,

# LLM+HiTL (per model)
hitl_{model}_tier1, hitl_{model}_escalated, hitl_{model}_tier2_final,
hitl_{model}_tier1_correct, hitl_{model}_escalation_correct,
hitl_{model}_tier2_correct, hitl_{model}_decision_time_ms,
hitl_{model}_reasoning
```
