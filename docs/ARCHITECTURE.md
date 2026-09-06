# Architecture — 2계층 LLM-인간 협력 점검 의사결정

이 문서는 프레임워크의 설계 명세다. 실행 방법과 결과 요약은
[README](../README.md), 논문용 상세 수치는
[`PAPER_DOSSIER.md`](PAPER_DOSSIER.md)를 참조.

Isaac Sim 5.0 시뮬레이션만으로 완결되며, 실로봇은 사용하지 않는다.
`legacy/spot_factory_inspection.py`(v1)를 뼈대로 삼아 2계층 의사결정 구조로
확장한 것이 현재의 `src/spot_factory_v2.py`(v2)다.

---

## 핵심 설계: 2계층 의사결정 구조

```
SPOT 순찰 → Anomaly 감지
  ↓
[Tier 1 — 항상 LLM 또는 Fixed]
관찰 자세 결정: APPROACH / WAIT_AND_OBSERVE / KEEP_DISTANCE
  ↓
SPOT 자세 실행 (ActionExecutor)
  ↓
[Tier 2 — mode에 따라 다름]
  ├─ Fixed mode:      규칙 기반 Tier2 결정
  ├─ LLM-Full mode:   LLM이 Tier2 직접 결정 (에스컬레이션 없음)
  └─ LLM+HiTL mode:   LLM이 confidence에 따라 에스컬레이션 여부 결정
        ├─ escalate=True  → Oracle(GT)이 Tier2 결정
        └─ escalate=False → LLM이 Tier2 직접 결정
  ↓
결과 기록 (3개 mode 동시) → 순찰 재개
```

**하나의 Isaac Sim 세션에서 매 anomaly 이벤트마다 3개 mode 모두에 대해 결정을
동시에 수집한다.** 따라서 세 조건의 비교는 동일 이벤트 위에서의 paired
비교다. SPOT의 물리적 행동은 `llm_hitl` 기준으로 실행한다.

### Tier 1 행동 집합 — 관찰 자세

| 코드 | 의미 | 접근 거리 |
|------|------|-----------|
| `APPROACH` | 근접 관찰 | 반경 1.5 m |
| `WAIT_AND_OBSERVE` | 중거리 대기 관찰 | 반경 2.5 m |
| `KEEP_DISTANCE` | 안전 거리 유지 | 반경 4.0 m |

선택된 자세에서 호(arc) 관찰을 수행한다. Tier1도 평가 대상이지만, 안전축의
핵심 지표는 Tier2다.

### Tier 2 행동 집합 — 대응 (4종)

| 코드 | 한국어 | 의미 |
|------|--------|------|
| `EMERGENCY` | 긴급 대응 호출 | 즉각 소방/안전팀 신고 |
| `MAINTENANCE` | 정비팀 파견 | 비긴급 전문 점검 요청 |
| `CONTINUE` | 모니터링 유지 | SPOT 재순찰, 자율 감시 |
| `FALSE_ALARM` | 오탐 처리 | 즉시 정상 순찰 재개 |

---

## 비교 모드 3개

| mode | Tier1 | 에스컬레이션 | Tier2 |
|------|-------|-------------|-------|
| `fixed` | 규칙 | 없음 | 규칙 |
| `llm_full` | LLM | 없음 (항상 LLM) | LLM |
| `llm_hitl` | LLM | LLM이 결정 | LLM 또는 Oracle |

---

## LLM 출력 스펙 (tool calling)

`src/llm_decision.py`의 `INSPECTION_TOOL` / `OPENAI_TOOL` 스키마.

```python
{
    "tier1_action":     str,    # APPROACH | WAIT_AND_OBSERVE | KEEP_DISTANCE
    "tier2_escalate":   bool,   # True=인간에게 이관, False=LLM이 직접 결정
    "tier2_action":     str,    # EMERGENCY | MAINTENANCE | CONTINUE | FALSE_ALARM
                                #   (tier2_escalate=False일 때만 유효)
    "tier2_confidence": float,  # 0.0-1.0, tier2_action이 정답일 보정된 확률
    "severity":         str,    # LOW | MEDIUM | HIGH
    "reasoning":        str,    # 판단 근거 (영문, 2~3문장)
}
```

`tier2_escalate=True`인 경우 LLM이 출력한 `tier2_action`은 무시하고 oracle(GT)
값으로 덮어쓴다.

> **운영 시 주의.** LLM의 이진 `tier2_escalate` 플래그는 과잉 발동한다(85 %).
> 실제 운영에서는 이 플래그를 직접 쓰지 말고 `tier2_confidence` **순위 +
> 보정된 임계값**으로 위임 여부를 결정할 것. 근거는
> [`PAPER_SUMMARY.md`](PAPER_SUMMARY.md) §4.

모델 별칭은 `MODEL_ALIASES`에 정의되어 있다 (`haiku`, `sonnet`, `gpt4o`,
`llama`).

---

## Anomaly 구성

상세 GT 규칙은 `src/anomaly_config.py`에 구현. 아래는 설계 요약이다.
In-distribution 4종에 더해 사전 규칙이 없는 OOD 3종(`GAS_LEAK`,
`ELECTRICAL_FAULT`, `STRUCTURAL_DAMAGE`)을 일반화 평가용으로 둔다.

### GT 판단 기준 (에스컬레이션)

- `escalation_gt = True` ↔ Tier2_GT = EMERGENCY, 또는 안전 위험 동반 MAINTENANCE
- `escalation_gt = False` ↔ Tier2_GT = CONTINUE / FALSE_ALARM, 또는 순수 운영 MAINTENANCE

### 참조 법령

프롬프트에 *원칙*으로 주입되며, 룩업 규칙으로 주어지지 않는다. 이것이 신규
이상 유형에 대한 일반화를 가능하게 한다.

- 화재예방법 제40조 (화재 즉각 신고 의무)
- 위험물안전관리법 제2조 (위험물 분류 및 신고)
- 소방기본법 제20조 (화염 확인 시 119 신고 의무)
- 전기안전관리법 제22조 (전기 설비 안전관리)
- 산업안전보건기준에 관한 규칙 제241조 / 제98조
- KOSHA GUIDE E-184, C-15, M-171

### In-distribution 4종의 severity 공식

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

**AGV_STOPPED** — `stopped_duration` {2, 15, 60} 분 · `blocking_main_path` {T, F} · `load_status` {empty, partial, full}

```
load_score = {empty: 0, partial: 0.1, full: 0.2}
severity = clip(min(duration / 60, 1) * 0.4 + blocking * 0.3 + load_score, 0, 1)
```

같은 유형이라도 속성 조합에 따라 맥락이 달라진다는 점이 규칙 기반 SOP 대비
핵심 차이다.

---

## 시뮬레이션 환경

- **플랫폼:** Isaac Sim 5.0, `SpotFlatTerrainPolicy` (pretrained locomotion).
  `PHYSICS_DT = RENDERING_DT = 0.002` (500 Hz).
- **환경 USD:** `Simple_Warehouse/warehouse_multiple_shelves.usd`
  (천장·기둥 숨김, 바닥 타일).
- **점검 구역 5개:** Assembly Line A (−7, −5) · Storage Area B (−2.5, 5) ·
  AGV Workspace (−4.6, 15.6) · Packaging Area (7, 11) · Loading Dock (2.5, −1.8).
- **순찰 경로:** 외곽 직사각형 8 waypoint, home (0, −7). 이상 감지 반경 2.5 m.
- **이벤트 생성:** lap마다 각 ZONE에 anomaly 유형을 랜덤 배정하고 속성 풀에서
  랜덤 추출해 자연어 리포트를 만든다. `--seed`로 재현 (기본 42).
- **규모:** 4개 모델 × 10 lap = 모델당 49 이벤트, 총 196 결정.

---

## 수집 메트릭 (결정 이벤트 1건당)

`src/metrics_collector.py`의 `DecisionRecord`. `results/metrics_raw.csv`로
저장되며 `analysis/`의 모든 스크립트가 이 파일을 입력으로 받는다.

```
lap, zone_id, anomaly_type, attributes(dict), severity_score,
tier1_gt, tier2_gt, escalation_gt,

# Fixed
fixed_tier1, fixed_tier2, fixed_tier1_correct, fixed_tier2_correct,

# LLM-Full (모델별)
llm_full_{model}_tier1, llm_full_{model}_tier2,
llm_full_{model}_tier1_correct, llm_full_{model}_tier2_correct,
llm_full_{model}_decision_time_ms,

# LLM+HiTL (모델별)
hitl_{model}_tier1, hitl_{model}_escalated, hitl_{model}_tier2_final,
hitl_{model}_tier1_correct, hitl_{model}_escalation_correct,
hitl_{model}_tier2_correct, hitl_{model}_decision_time_ms,
hitl_{model}_reasoning
```

---

## v1 → v2 변경 사항

`legacy/spot_factory_inspection.py`에서 **유지한 부분**:

- `SpotPatrolController` 클래스 구조 (waypoint 순찰 로직)
- `ActionExecutor` (APPROACH / WAIT / KEEP_DISTANCE 실행, arc 관찰)
- Isaac Sim 환경 초기화 (physics dt, warehouse USD 로드)
- Anomaly zone 시각화 (색상 구체 표시)
- LLM 호출 기본 구조 (Anthropic tool_use / OpenAI function calling / Ollama)

**변경한 부분**:

- Anomaly 정의를 새 7종(In-dist 4 + OOD 3) × 새 속성 풀로 교체
- LLM 출력 스펙에 `tier2_escalate`, `tier2_action`, `tier2_confidence` 추가
- Fixed 정책을 Tier2까지 포함하는 규칙으로 확장
- 메트릭 수집에 Tier2·에스컬레이션 필드 추가
- 실행 구조를 3개 mode 동시 수집으로 변경
