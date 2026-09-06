# 결과 분석 섹션 작성 가이드 (프롬프트)

이 문서를 입력으로, 논문의 "실험 결과 및 분석" 섹션을 작성하라.
모든 수치는 Isaac Sim 실측(196 결정, 4 LLM × 10 lap) 결과이며 그대로 사용한다.
용어·표현 규칙은 §0을 먼저 준수하고, §1 구조에 따라 §2의 figure/캡션을 배치해 서술하라.

---

## 0. 작성 규칙 (필수)

- 비교 조건/방법 명칭은 그림과 동일하게: **Fixed SOP / LLM-Full / Proposed (LLM+HiTL) / Severity Threshold / Random**.
- 위임 동작은 일관되게 **escalation**으로 칭한다("deferral" 금지).
- escalation은 별도 계층이 아니라 **Tier-2에 대한 선택적 위임 게이트**다(Tier-3 아님). 시스템은 2계층.
- **운영(실행) escalation 기준 = LLM이 위험성+불확실성을 종합해 출력한 binary 플래그**(`tier2_escalate`). escalate 시 인간(오라클)이 Tier-2 결정. 운영 escalation 비율은 약 85%.
- `tier2_confidence`는 **운영 escalation에 사용되지 않았고**, 출력·기록만 되어 **사후(post-hoc) 분석**에만 쓰였다. confidence를 시스템 운영 구성요소로 서술하지 마라.
- 따라서 결과를 **운영 결과(§1.2–1.3)** 와 **사후 분석(§1.4–1.5)** 로 명확히 구분해 서술한다.
- temperature scaling 등 confidence 보정은 적용하지 않았다. confidence의 신뢰도 문제는 한계/향후 과제로만 다룬다.
- 평가지표 방향: 안전률·검출률은 높을수록, escalation rate·위험률은 낮을수록 양호(본문에서 자연스럽게 서술하되 "낮을수록 좋다" 식 사족은 피한다).

---

## 1. 섹션 구조 및 서술 포인트

### 1.1 실험 설정 (간단)
- 4개 LLM × 10 lap = 196 결정 이벤트. anomaly 7종 = In-distribution 4종 + OOD(신규) 3종.
- 3개 조건(Fixed SOP / LLM-Full / Proposed)과 평가지표(Tier-1 적정 경계, Tier-2 정확 대응, EMERGENCY 검출, escalation rate) 정의.
- 운영 escalation 메커니즘(§0) 명시.

### 1.2 전체 안전 성능 — Fig. headline_safety
- 주장: Fixed SOP의 안전 공백을 LLM 추론이 메우고, Proposed가 모든 안전 지표에서 최고.
- 수치: EMERGENCY 검출 24%(Fixed) → 93%(LLM-Full) → **100%(Proposed)**; Tier-2 정확 대응 55 → 94 → **98%**; Tier-1 적정 경계 69 → 79 → 79%.

### 1.3 신규 유형 일반화 — Fig. generalization
- 주장: 사전 규칙이 없는 OOD에서 Fixed SOP는 붕괴, LLM 계열은 유지.
- 수치: OOD EMERGENCY 검출 **0%(Fixed)** vs 91%(LLM-Full) vs **100%(Proposed)**; In-distribution은 35 / 94 / 100%.
- 메시지: 규칙은 열거되지 않은 상황에 일반화하지 못하고, 법 원칙 기반 LLM 추론은 신규 유형에도 적용된다.

### 1.4 escalation 신호 분석 (사후) — Fig. escalation_curve_emergency, escalation_curve_underresponse
- **운영 정책과 별개의 사후 분석임을 도입부와 캡션에 명시.** 기록된 점수로 escalation 대상을 골랐을 때의 성능을 비교한다.
- 비교: Severity Threshold(규칙 점수 `severity_score` 기반, LLM 아님) vs Proposed(LLM `tier2_confidence` 기반) vs Random. 참조점으로 LLM-Full(escalation 0), Fixed SOP를 함께 표시.
- 주장: LLM confidence로 escalation 대상을 고르면 동일 escalation rate에서 더 높은 안전을 얻는다(곡선이 Severity·Random 위).
- 보강: severity 점수는 OOD 신규 유형에 정의되지 않아 적용 불가, confidence는 자연어 기반이라 OOD에도 작동.

### 1.5 confidence 신뢰도 진단 (한계) — Fig. reliability
- 주장: LLM의 출력 confidence(raw)는 과신 경향 — 평균 0.85를 보고하나 실제 정확도는 0.57.
- 메시지: confidence의 절대값은 그대로 신뢰하기 어렵다. 운영 escalation은 confidence 절대값에 의존하지 않으며, confidence 활용 시 보정이 필요하다는 점을 향후 과제로 제시.

### 논리 흐름
운영 안전 향상(1.2) → 신규 유형 일반화(1.3) → [그 escalation을 어떤 신호로 고를까] 사후 분석(1.4) → [그 신호의 한계] 신뢰도 진단(1.5).

---

## 2. Figure 캡션 (한글 / 영어)

### Fig. headline_safety
- KR: **그림 N. 조건별 안전 지표 비교(전 모델 집계).** Tier-2 정확 대응률, EMERGENCY 검출률, Tier-1 적정 경계율을 Fixed SOP·LLM-Full·Proposed(LLM+HiTL)에 대해 나타낸다. Proposed가 세 지표 모두에서 가장 높으며, 특히 EMERGENCY 검출률이 100%에 도달한다.
- EN: **Fig. N. Aggregate safety metrics by condition (all models).** Correct Tier-2 response, emergency detection, and adequate Tier-1 caution for Fixed SOP, LLM-Full, and Proposed (LLM+HiTL). Proposed attains the highest value on all three metrics, reaching 100% emergency detection.

### Fig. generalization
- KR: **그림 N. 신규(OOD) 이상 유형에 대한 일반화.** In-distribution과 OOD 유형에서의 EMERGENCY 검출률. Fixed SOP는 사전 규칙이 없는 OOD에서 0%로 붕괴하는 반면, 법 원칙 기반 LLM 조건은 검출률을 유지한다(Proposed 100%).
- EN: **Fig. N. Generalization to novel (OOD) anomaly types.** Emergency detection on in-distribution vs. OOD types. Fixed SOP collapses to 0% on OOD types it has no rule for, whereas the law-grounded LLM conditions retain detection (Proposed: 100%).

### Fig. escalation_curve_emergency
- KR: **그림 N. escalation 신호별 안전–escalation 곡선(EMERGENCY 기준, 사후 분석).** 기록된 점수로 escalation 대상을 선택했을 때, escalation rate에 따른 안전(=1−EMERGENCY 누락)을 비교한다. LLM confidence 기반 선택(Proposed)이 규칙 점수 기반(Severity Threshold) 및 Random보다 동일 escalation rate에서 높은 안전을 보인다. 본 곡선은 운영 정책과 구분되는 사후 비교다.
- EN: **Fig. N. Safety–escalation curves by escalation signal (emergency criterion, post-hoc).** Safety (= 1 − emergency miss rate) as a function of escalation rate when escalation targets are chosen by each recorded score. Confidence-based selection (Proposed) yields higher safety than the rule-based Severity Threshold and Random at the same escalation rate. This is a post-hoc comparison, distinct from the deployed policy.

### Fig. escalation_curve_underresponse
- KR: **그림 N. escalation 신호별 안전–escalation 곡선(Tier-2 과소대응 기준, 사후 분석).** 안전을 1−Tier-2 과소대응률로 정의했을 때의 동일 비교. confidence 기반 선택이 전 구간에서 우위를 유지한다.
- EN: **Fig. N. Safety–escalation curves by escalation signal (Tier-2 under-response criterion, post-hoc).** The same comparison with safety defined as 1 − Tier-2 under-response rate. Confidence-based selection remains superior across the operating range.

### Fig. min_help
- KR: **그림 N. 목표 안전 달성에 필요한 최소 escalation rate(사후 분석).** 목표 EMERGENCY 검출 수준(≥98/99/100%)을 달성하기 위해 각 신호가 요구하는 최소 escalation rate. confidence 기반(Proposed)이 ≥98%를 25%의 escalation으로 달성하여 Severity Threshold(41%)·Random(71%)보다 적은 인간 개입을 요구한다.
- EN: **Fig. N. Minimum escalation rate to reach a target safety level (post-hoc).** Minimum escalation rate each signal needs to reach a target emergency-detection level (≥98/99/100%). Confidence-based selection (Proposed) reaches ≥98% with 25% escalation, requiring less human intervention than Severity Threshold (41%) and Random (71%).

### Fig. reliability
- KR: **그림 N. LLM confidence의 신뢰도.** 출력 confidence 구간별 평균값과 실제 정확도. 점 크기는 해당 구간의 결정 수에 비례한다. 대각선 아래에 위치한 다수의 점은 과신을 의미하며, 평균 confidence 0.85에 비해 실제 정확도는 0.57이다.
- EN: **Fig. N. Reliability of LLM confidence.** Mean stated confidence vs. actual accuracy per confidence bin; marker size is proportional to the number of decisions in the bin. Points below the diagonal indicate over-confidence; mean stated confidence is 0.85 against an actual accuracy of 0.57.

---

## 3. 그림 파일 매핑
- headline_safety.png, generalization.png, escalation_curve_emergency.png, escalation_curve_underresponse.png, min_help.png, reliability.png  (모두 `results/paper_figures/`)
- 본문 그림 번호는 배치에 맞게 매기되, 캡션의 "그림 N / Fig. N"을 실제 번호로 치환하라.
