# 논문 수정 지시 — LLM 출력 구조 및 escalation 메커니즘 정합화

이 문서는 구현된 시스템의 실제 동작을 기준으로, 논문 초안에서 이와 어긋나는 서술을 수정하도록 지시한다.
아래 "실제 구현" 내용을 정답으로 삼고, "수정 지시"에 따라 본문·그림 캡션·수식·표현을 고쳐라.

---

## 1. 실제 구현 — LLM 출력 구조

LLM은 anomaly 1건마다 **단일 tool call (`submit_inspection_decision`)** 로 아래 6개 필드를 동시에 출력한다.

| 필드 | 타입 | 의미 |
|---|---|---|
| `tier1_action` | enum {APPROACH, WAIT_AND_OBSERVE, KEEP_DISTANCE} | **Tier 1 — 관찰 자세(observation posture)** |
| `tier2_escalate` | bool | Tier 2를 인간에게 위임할지 여부 (게이트) |
| `tier2_action` | enum {EMERGENCY, MAINTENANCE, CONTINUE, FALSE_ALARM} | **Tier 2 — 대응 행동(response-level decision)**, `tier2_escalate=False`일 때만 사용 |
| `tier2_confidence` | float 0–1 | `tier2_action`이 정답일 것이라는 LLM의 확신도 |
| `severity` | enum {LOW, MEDIUM, HIGH} | 부가 |
| `reasoning` | string | 근거 |

### 계층 구조 (중요)
- 시스템은 **2계층**이다. Tier 3는 없다.
- `tier2_escalate`는 별도 계층이 아니라, **Tier 2의 실행 주체를 가르는 라우팅 게이트**다:
  - `tier2_escalate=True` → 인간(오라클)이 Tier 2 행동을 결정 (LLM의 `tier2_action`은 사용하지 않음)
  - `tier2_escalate=False` → LLM의 `tier2_action`을 그대로 사용
- 따라서 escalation은 **"Tier 2에 대한 선택적 위임(selective human-in-the-loop) 게이트"**로 서술해야 하며, 별도의 단계/계층으로 서술하면 안 된다.

```
Tier 1 ── 관찰 자세 (항상 LLM, 즉시 실행)
Tier 2 ── 대응 행동
            └─ tier2_escalate ?
                 ├─ True  → 인간(오라클) 결정
                 └─ False → LLM 결정 (tier2_action)
```

---

## 2. 실제 구현 — escalation은 어떻게 결정되었나

- 실험에서 escalate 여부를 결정한 것은 **`tier2_escalate` (binary)** 하나다.
- 이 binary는 LLM이 system prompt의 규칙에 따라 출력한다:
  > escalate=True ⇐ 법적 신고 의무가 있거나, 비가역적 안전 결과가 있거나, 결정이 불확실할 때.
- 즉 escalation 기준은 "EMERGENCY로 분류하면 escalate"도 아니고 "confidence 임계"도 아니다.
  LLM이 위험성과 불확실성을 종합해 내리는 **단일 binary 판단**이다.

---

## 3. 실제 구현 — `tier2_confidence`의 위치

- `tier2_confidence`는 **파이프라인의 escalation 결정에 사용되지 않았다.**
- LLM 출력 스키마에 포함되어 **기록만 되었고, 사후(post-hoc) 분석 용도로만** 쓰인다.
- 본 연구의 운영(실행) 메커니즘은 confidence가 아니라 §2의 binary `tier2_escalate`다.

---

## 4. 수정 지시 (논문 초안에서 고칠 것)

1. **계층 표현**: escalation을 "Tier 3" 또는 별도 단계로 적은 부분이 있으면 전부 삭제하고,
   "Tier 2에 대한 selective escalation gate"로 통일하라. 시스템은 2계층임을 명확히 하라.

2. **운영 escalation 기준**: 본문에서 escalation의 작동 기준을 기술할 때, 운영상 실제로 쓰인 것은
   **LLM이 위험성+불확실성을 종합해 출력한 binary `tier2_escalate`**임을 명시하라.
   - "confidence 임계로 escalate한다", "confidence 기반으로 인간에게 위임한다" 같은 서술이 있으면 삭제·수정하라.
     그런 메커니즘은 실행되지 않았다.

3. **confidence 서술**: `tier2_confidence`는 escalation 결정에 사용되지 않았고 사후 분석용으로만 출력되었음을
   분명히 하라. confidence를 시스템의 작동 구성요소(operational component)로 기술하지 마라.

4. **보정(calibration) 관련 서술**: temperature scaling 등 confidence 보정을 시스템에 적용했다는 식의 서술이
   있으면 삭제하라. 실험 파이프라인에서 보정은 수행되지 않았다. confidence의 신뢰도(과신) 관찰은
   한계/향후 과제 수준으로만 다루어라.

5. **그림 캡션 정합화**:
   - `headline_safety`, `generalization`: 운영 정책(binary `tier2_escalate` → 위임 시 오라클) 기준 결과로 서술하라.
   - `escalation_curve`, `min_help`: 실행 정책이 아니라, **기록된 점수로 위임 대상을 사후에 골랐을 때의
     비교 분석**임을 캡션에 명시하라. 두 후보 신호를 비교한다:
     - Severity Threshold = 규칙 기반 점수 `severity_score`(속성으로 계산, LLM 아님) 기준 위임
     - Proposed (LLM+HiTL) = LLM `tier2_confidence` 기준 위임
     이 곡선들은 "어떤 신호가 위임 대상을 잘 고르는가"를 비교하는 post-hoc 분석이며, 운영 시 실행한 정책과는
     구분된다고 적어라.
   - `reliability`: confidence vs 실제 정답률(raw) 진단 그림으로만 서술하라. 보정 결과 그림이 아니다.

6. **용어**: 그림·본문 모두 "deferral"이 아니라 프로젝트 전체와 동일하게 **"escalation"**으로 통일하라.

---

## 5. 참고 — 비교 조건/명칭 (그림과 일치시킬 것)
- Fixed SOP / LLM-Full / Proposed (LLM+HiTL) / Severity Threshold / Random / Always-escalate / Oracle (optimal)
- 코드 위치: 출력 스키마·프롬프트 = `src/llm_decision.py`, 실행 흐름 = `src/spot_factory_v2.py`,
  severity_score 정의 = `src/anomaly_config.py`, 그림 = `analysis/make_paper_figures.py`.
