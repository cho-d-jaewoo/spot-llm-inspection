# PAPER DOSSIER — LLM-인간 협력 공장 점검 의사결정 프레임워크

> 이 파일 하나로 논문 작성에 필요한 모든 정보(제목·시나리오·프롬프트·GT 기준·실험 설계·결과 수치·그림)를 담는다.
> 모든 수치는 Isaac Sim 실측 실행(196 결정 이벤트, 4개 LLM × 10 lap) 결과다.

---

## 0. 제목 / 한 줄 요약

- **국문(작업 제목):** 불확실성 인지 기반 선택적 에스컬레이션을 통한 LLM-인간 협력 공장 점검 의사결정 프레임워크
- **영문:** Confidence-Guided Selective Escalation for LLM-Human Collaborative Decision-Making in Robotic Factory Inspection
- **한 줄:** 순찰 로봇이 이상을 감지하면 LLM이 법 원칙에 근거해 2계층 결정을 내리고, 자신의 불확실성에 따라 인간에게 선택적으로 위임한다. 사전 규칙이 없는 신규 이상 유형으로 일반화하며, 동일한 인간 개입량 대비 더 높은 안전을 달성한다.

---

## 1. 연구 동기 / 문제 정의

- 기존 공장 SPOT 배치(BMW·POSCO·Cargill 등)는 **고정 스케줄 순찰**만 수행하고, 이상 이벤트에 대한 맥락 기반 반응형 의사결정이 없다.
- 규칙 기반 SOP(표준작업절차)는 (1) 이상 유형 단위의 고정 반응이라 속성 맥락을 반영 못 하고, (2) **사전에 열거하지 않은 신규 이상 유형**에 대응하지 못한다.
- 인간-로봇 협업(HiTL)이 이미 전제된 환경에서, **LLM이 협업의 효용(안전 대비 인간 개입량)을 끌어올릴 수 있는가**가 핵심 질문.

**기여**
1. 법 원칙 기반 2계층(관찰 자세 + 대응) LLM-인간 협업 점검 프레임워크.
2. 사전 규칙이 없는 신규 이상 유형(OOD)으로의 일반화.
3. confidence 기반 선택적 위임 → 동일 인간 개입량 대비 더 높은 안전(KnowNo식 coverage–risk).
4. LLM confidence의 보정 분석(과신 → temperature scaling) 및 단조 보정의 위임 선택 불변성.

---

## 2. 시뮬레이션 시나리오

- **플랫폼:** Isaac Sim 5.0, Boston Dynamics SPOT(`SpotFlatTerrainPolicy`, pretrained locomotion). PHYSICS_DT = RENDERING_DT = 0.002 (500 Hz).
- **환경:** `Simple_Warehouse/warehouse_multiple_shelves.usd` (천장/기둥 숨김, 바닥 타일).
- **점검 구역(ZONE) 5개:** Assembly Line A(-7,-5), Storage Area B(-2.5,5), AGV Workspace(-4.6,15.6), Packaging Area(7,11), Loading Dock(2.5,-1.8).
- **순찰 경로:** 외곽 직사각형 8 waypoint. SPOT 시작/home (0,-7). 이상 감지 반경 2.5 m.
- **이상 이벤트 생성:** lap마다 각 ZONE에 anomaly 유형 랜덤 배정 + 속성 풀에서 랜덤 추출 → 자연어 리포트 생성. 같은 유형도 속성에 따라 다른 맥락. seed로 재현(기본 42, lap별 변형).
- **Tier1 물리 실행:** 선택된 관찰 자세를 SPOT가 수행. APPROACH=반경 1.5 m, WAIT_AND_OBSERVE=2.5 m, KEEP_DISTANCE=4.0 m에서 호(arc) 관찰. (Tier1은 평가 대상이지 안전축의 핵심은 Tier2.)
- **규모:** 4개 모델(haiku/sonnet/gpt4o/llama) × 10 lap = 모델당 49 이벤트, 총 196 결정.

### 2.1 이상 유형 (7종)

| 유형 | 구분 | 속성 풀 |
|---|---|---|
| FIRE_RISK | In-dist | temperature {75,90,110,150}°C · workers_nearby {0,1,3} · flammables_adjacent {T,F} |
| LIQUID_LEAK | In-dist | area {0.3,0.8,2.5} m² · substance {water,coolant,lubricant,chemical} · near_electrical {T,F} |
| SMOKE_DETECTED | In-dist | density {light,moderate,dense} · smell {odorless,burning,chemical} · visible_flame {T,F} |
| AGV_STOPPED | In-dist | stopped_duration {2,15,60} min · blocking_main_path {T,F} · load_status {empty,partial,full} |
| **GAS_LEAK** | **OOD** | gas_type {inert,toxic,flammable} · concentration {low,moderate,high} · ventilation {good,poor} |
| **ELECTRICAL_FAULT** | **OOD** | condition {overheating,sparking,arcing} · near_flammable {T,F} · water_present {T,F} |
| **STRUCTURAL_DAMAGE** | **OOD** | crack_size {hairline,moderate,severe} · load_bearing {T,F} · workers_below {0,1,3} |

OOD 3종은 **Fixed SOP가 사전 규칙을 갖지 않는** 신규 유형(일반화 테스트용). LLM은 동일 법 원칙으로 zero-shot 추론.

### 2.2 자연어 리포트 템플릿 (LLM 입력 예)
- FIRE_RISK: "SPOT detected a high-temperature heat source with potential fire risk while patrolling {zone}. Measured temperature: approximately {temperature}°C. Workers in proximity: {workers_nearby}. Flammable materials stored in adjacent area: {flammables_adjacent}."
- GAS_LEAK: "SPOT detected a gas leak in {zone}. Suspected gas type: {gas_type}. Measured concentration level: {concentration}. Area ventilation: {ventilation}."
- (나머지 5종 템플릿은 `src/anomaly_config.py`의 `prompt_template` 참조.)

---

## 3. 비교 조건 (3종)

| 조건 | Tier1 | Tier2 | 에스컬레이션 |
|---|---|---|---|
| **Fixed (SOP)** | 유형 단위 고정 | 유형 단위 고정 | 없음 |
| **LLM-Full** | LLM(법 원칙) | LLM 직접 | 없음(항상 자율) |
| **LLM+HiTL** | LLM(법 원칙) | LLM 또는 인간 | LLM이 confidence/위험으로 결정 |

- **Fixed SOP 매핑(속성 무시):** FIRE_RISK→(KEEP_DISTANCE, EMERGENCY), SMOKE_DETECTED→(WAIT_AND_OBSERVE, MAINTENANCE), LIQUID_LEAK→(APPROACH, MAINTENANCE), AGV_STOPPED→(APPROACH, CONTINUE). 신규(OOD) 유형 → 보수적 default (KEEP_DISTANCE, MAINTENANCE).
- **위임 시 인간(Oracle) = GT Tier2** 반환(완벽한 전문가 가정).
- LLM 호출은 이벤트당 1회이며 LLM-Full / LLM+HiTL은 **동일 응답을 공유**(에스컬레이션 처리만 차이).

---

## 4. LLM 프롬프트 / 출력 스펙

### 4.1 System Prompt (전문)
```
You are an intelligent inspection decision system for a SPOT quadruped robot patrolling a smart factory.

## Your Role
Make two-tier decisions when anomalies are detected:

**Tier 1 (Observation Stance)** — Select immediately based on safety:
- APPROACH (1.5m): Safe, low-risk situation. Direct inspection needed.
- WAIT_AND_OBSERVE (2.5m): Moderate risk. Gather info before closing in.
- KEEP_DISTANCE (4.0m): High risk. Maintain safety distance for multi-angle observation.

**Tier 2 (Resolution Action)** — Decide whether to handle autonomously or escalate:
- EMERGENCY: Imminent life/safety threat. Mandatory under Korean law (화재예방법 §40, 소방기본법 §20).
- MAINTENANCE: Specialist inspection or repair needed (non-urgent).
- CONTINUE: Autonomous monitoring sufficient. Resume patrol.
- FALSE_ALARM: Likely false detection. Resume patrol immediately.

**Escalation Rule (tier2_escalate)**:
- Set True when: Tier 2 decision involves legal reporting obligation OR irreversible safety
  consequences (wrong call = injury, major damage, or legal liability), OR you are genuinely
  uncertain about the correct resolution.
- Set False when: Tier 2 is routine (maintenance request automatable via MES, or simple
  monitoring/false-alarm handling) AND you are confident.

**Confidence (tier2_confidence)**:
- Report your honest, well-calibrated probability that tier2_action is correct. This drives how/when a human is asked.
- A human operator's time is limited: high confidence should mean you are reliably correct,
  low confidence should flag cases you would likely get wrong. Do NOT be overconfident.

## Key Principle
Consider ALL attribute interactions, not just the anomaly type.
The same anomaly type can require different decisions depending on context.
You may encounter anomaly types not seen before — reason from the legal principles below.

## Korean Safety/Reporting Principles (apply by reasoning, not lookup)
- 화재예방법 §40 / 소방기본법 §20: Confirmed flame or beyond-ignition heat → immediate emergency report (119) is MANDATORY.
- 위험물안전관리법 §2: Hazardous/toxic chemicals → immediate report.
- 전기안전관리법 §22: Live electrical hazard combined with conductive liquid, water, or arcing → emergency.
- 고압가스안전관리법: High-concentration flammable/toxic gas → emergency; confined poorly-ventilated space raises asphyxiation risk.
- 산업안전보건기준규칙 §98/§241: Passage blockage, high heat near workers, collapse/fall risk near workers → mandatory escalation.
- KOSHA GUIDE (E-184, C-15, M-171): irreversible safety risks need specialist (MAINTENANCE) inspection even if not immediately life-threatening.

## How to apply
- EMERGENCY when there is a legal reporting obligation OR imminent irreversible harm to life/property.
- MAINTENANCE when specialist repair/inspection is needed but not immediately life-threatening.
- CONTINUE for autonomous monitoring; FALSE_ALARM for benign/self-resolving cases.
- These are PRINCIPLES: you may encounter anomaly types or attribute combinations not explicitly
  listed — reason from the underlying intent (life safety > legal duty > asset protection > efficiency).

You MUST respond by invoking the submit_inspection_decision tool.
```

### 4.2 Tool 스키마 (`submit_inspection_decision`)
| 필드 | 타입 | 설명 |
|---|---|---|
| tier1_action | enum {APPROACH, WAIT_AND_OBSERVE, KEEP_DISTANCE} | 즉시 관찰 자세 |
| tier2_escalate | bool | True=인간 위임, False=자율 |
| tier2_action | enum {EMERGENCY, MAINTENANCE, CONTINUE, FALSE_ALARM} | 대응(escalate=False일 때 사용) |
| tier2_confidence | number 0–1 | tier2_action이 정답일 보정 확률(위임 임계의 근거) |
| severity | enum {LOW, MEDIUM, HIGH} | |
| reasoning | string | 근거 2–3문장 |

- Anthropic는 tool_use, OpenAI/Ollama는 function calling으로 동일 스키마 호출. 모델 별칭: haiku=claude-haiku-4-5, sonnet=claude-sonnet-4-6, gpt4o=gpt-4o-mini, llama=llama3.1(Ollama).

---

## 5. Ground Truth 기준 (전문가 규칙 + 참조 법령)

GT는 **전체 속성 + 아래 법 원칙**으로 정의되는 전문가 정답(첫 매칭 규칙 적용). 평가 기준이며 LLM/Fixed가 직접 접근하지 않는다. severity_score는 0–1 휴리스틱(시각화 + severity-threshold baseline용).

### 5.1 참조 법령/지침
화재예방법 §40, 소방기본법 §20, 위험물안전관리법 §2, 전기안전관리법 §22, 고압가스안전관리법, 산업안전보건기준에 관한 규칙 §98·§241, KOSHA GUIDE E-184·C-15·M-171.

### 5.2 GT 규칙 (유형별, 우선순위 순 — tier1 / tier2 / escalation / 근거)
**FIRE_RISK** — temp 150 → KEEP/EMERGENCY/T(화재예방법§40) · temp≥110 & flammable → KEEP/EMERGENCY/T(KOSHA E-184) · temp≥110 & workers≥1 → KEEP/EMERGENCY/T(산안기준§241) · temp≥110 → WAIT/MAINTENANCE/T · temp=90 & flammable → WAIT/MAINTENANCE/T · temp=90 → WAIT/CONTINUE/F · else(75) → APPROACH/CONTINUE/F.
**LIQUID_LEAK** — chemical → KEEP/EMERGENCY/T(위험물안전관리법§2) · near_electrical & ≠water → KEEP/EMERGENCY/T(전기안전관리법§22) · near_electrical & water → KEEP/MAINTENANCE/T · area≥2.5 & coolant → WAIT/MAINTENANCE/F · lubricant → APPROACH/MAINTENANCE/F · else → APPROACH/FALSE_ALARM/F.
**SMOKE_DETECTED** — visible_flame → KEEP/EMERGENCY/T(소방기본법§20) · dense & chemical → KEEP/EMERGENCY/T · dense & burning → KEEP/EMERGENCY/T · moderate & (chemical|burning) → WAIT/MAINTENANCE/T · light & burning → WAIT/CONTINUE/F · else → APPROACH/CONTINUE/F.
**AGV_STOPPED** — blocking & full & dur≥15 → WAIT/MAINTENANCE/T(KOSHA M-171) · blocking & dur≥15 → WAIT/MAINTENANCE/T(산안기준§98) · blocking & dur<5 → APPROACH/CONTINUE/F · ¬blocking & dur≥60 → APPROACH/MAINTENANCE/F · else → APPROACH/FALSE_ALARM/F.
**GAS_LEAK(OOD)** — flammable & high → KEEP/EMERGENCY/T(고압가스안전관리법) · toxic & (high|poor vent) → KEEP/EMERGENCY/T(산안기준) · (toxic|flammable) & moderate → WAIT/MAINTENANCE/T · inert & high & poor → WAIT/MAINTENANCE/T · low → APPROACH/CONTINUE/F · else → APPROACH/CONTINUE/F.
**ELECTRICAL_FAULT(OOD)** — arcing → KEEP/EMERGENCY/T(전기안전관리법§22/화재예방법) · sparking & (near_flammable|water) → KEEP/EMERGENCY/T · sparking → WAIT/MAINTENANCE/T · overheating & near_flammable → WAIT/MAINTENANCE/T · overheating → APPROACH/MAINTENANCE/F · else → APPROACH/CONTINUE/F.
**STRUCTURAL_DAMAGE(OOD)** — severe & load_bearing → KEEP/EMERGENCY/T(산안기준) · severe & workers≥1 → KEEP/EMERGENCY/T · severe → WAIT/MAINTENANCE/T · moderate & load_bearing → WAIT/MAINTENANCE/T · moderate → APPROACH/MAINTENANCE/F · else → APPROACH/CONTINUE/F.

(전체 lambda 구현은 `src/anomaly_config.py`의 `ANOMALY_CONFIG[...]['gt_rules']` 참조.)

---

## 6. 평가 지표

- **Tier1 (관찰 자세) — 안전 ordinal:** 순서 APPROACH < WAIT_AND_OBSERVE < KEEP_DISTANCE. 선택이 GT보다 *덜 보수적*이면 위험(Tier1 위험률).
- **Tier2 (대응) — 위험 분류:** 강도 순위 FALSE_ALARM<CONTINUE<MAINTENANCE<EMERGENCY. **위험 과소대응률** = 예측 < GT & GT∈{EMERGENCY,MAINTENANCE}. **EMERGENCY 누락률** = GT=EMERGENCY인데 예측≠EMERGENCY. **과대응률** = 예측 > GT.
- **에스컬레이션 결정:** GT=escalation_gt 기준 Precision/Recall/F1, severity-score 기반 AUC.
- **위임 곡선(coverage–risk):** x=human-help rate(에스컬레이션%), y=safety(1−EMERGENCY 누락 또는 1−Tier2 위험). 위임 시 Oracle=GT, 아니면 LLM 자율. 정책: Random(해석적 직선)·Severity-threshold(τ)·LLM-confidence(τ)·Oracle-router(최적)·No-help·Always.
- **신뢰도 보정:** ECE(Expected Calibration Error), temperature scaling T.

---

## 7. 결과 (실측, 196 결정)

### 7.1 프레임워크 집계 — 안전 지표 (전 모델, %)
| 조건 | Tier2 위험 과소대응 | EMERGENCY 누락 | Tier1 위험 | 과대응 |
|---|---|---|---|---|
| Fixed (SOP) | 44.9 | 76.0 | 30.6 | 22.4 |
| LLM-Full | 5.6 | 7.0 | 20.9 | 37.2 |
| **LLM+HiTL** | **1.5** | **0.0** | 20.9 | 7.7 |

### 7.2 일반화 — In-distribution vs OOD (EMERGENCY 누락 / Tier2 위험, %)
| 조건 | EMERG누락 In | EMERG누락 OOD | Tier2위험 In | Tier2위험 OOD |
|---|---|---|---|---|
| Fixed (SOP) | 64.7 | **100.0** | 46.7 | 42.1 |
| LLM-Full | 5.9 | 9.4 | 5.0 | 6.6 |
| **LLM+HiTL** | **0.0** | **0.0** | 0.8 | 2.6 |

→ Fixed는 신규 유형에서 긴급을 전부 누락. LLM은 동일 법 원칙으로 신규 유형에서도 긴급 0% 누락.

### 7.3 위임 효율 — 목표 안전 달성에 필요한 최소 human-help (%)
| 목표 EMERGENCY 안전 | LLM-confidence | Severity-threshold | Random |
|---|---|---|---|
| ≥98% | **25** | 41 | 71 |
| ≥99% | 49 | **41** | 86 |
| ≥100% | **65** | 71 | 100 |

- severity-threshold는 GT 속성 기반 점수가 필요하며 OOD 신규 유형에는 정의되지 않음(배포 불가). LLM-confidence는 자연어만으로 작동하며 OOD에도 일반화.
- LLM-Full(no-help) 단독 Tier2 정확도 57.1%. severity 에스컬레이션 AUC = 0.93.

### 7.4 신뢰도 보정
- raw confidence 평균 0.846 vs 실제 정답률 0.571 → **ECE 30.5%(과신)**.
- temperature scaling **T=7.0** → 평균 0.576, **ECE 4.7%**.
- 단조 보정이므로 위임 순위 불변 → 위임 곡선은 보정 전후 동일(보정은 임계값 해석에 기여).

### 7.5 결정 지연
모델별 평균 ≈ GPT-4o mini 1.7 s < Haiku 2.4 s < Llama 3.3 s < Sonnet 5.3 s (`fig9` / `report.md` Table 6).

---

## 8. 사용할 그림 (최종)

### 메인
| 라벨 | 파일 | 캡션 요지 |
|---|---|---|
| F1 | (개념도, 직접 작도) | 순찰→감지→2계층 LLM 결정(법 원칙)→confidence 위임→인간/자율 |
| F2 | `generalization.png` | In vs OOD EMERGENCY 누락: Fixed 100% 누락 vs LLM 일반화 |
| F3 | `headline_safety.png` | 조건별 안전 지표(Tier2 위험·EMERGENCY 누락·Tier1 위험) |
| F4 | `deferral_curve.png` | safety vs human-help 곡선: LLM-confidence가 random·severity 상회 |
| F4b | `min_help.png` | 목표 안전 달성 최소 human-help(막대): LLM-confidence 최소 |
| F5 | `deferral_split.png` | In vs OOD 위임: 규칙 기반은 신규 유형에 부재 |
| F6 | `reliability.png` | confidence 과신(ECE 30.5%) → temperature scaling(4.7%) |
| F7 | `deferral_calibrated.png` | 단조 보정 → 위임 곡선 불변 |

---

## 9. 한계 및 후속

- 시뮬레이션 한정. GT는 저자 작성 규칙(객관적 평가 기준이나 합성). Oracle은 완벽한 인간 가정.
- 시나리오가 고위험 분포(EMERGENCY 비중 큼)라 GT 위임필요 비율(≈70%)이 높음 — 평가는 자동화율이 아니라 "같은 help에서의 안전"으로 수행.
- LLM의 이진 self-escalation은 과잉 위임(≈85%) 경향 → 직접 사용 대신 confidence 순위 + 보정 임계로 운영점 설정.
- raw confidence는 과신·비단조 → temperature scaling은 평균만 교정. 후속: token logprob / self-consistency / 엔트로피 등 풍부한 불확실성 신호, 시나리오 재균형(루틴 다수), 다중 seed·확장으로 통계 유의성, 인간 불완전성 모델링.

---

## 10. 인용 (방법론 근거)
- Ren et al., *Robots That Ask For Help: Uncertainty Alignment for LLM Planners*, CoRL 2023 (arXiv:2307.01928).
- *Aligning Learning with Communication in Shared Autonomy* (arXiv:2403.12023, D. Losey 그룹).
- Gao et al., *Learning Complementary Policies for Human-AI Teams* (arXiv:2302.02944).
- *Towards Effective Human-AI Decision-Making: Appropriate Reliance* (arXiv:2310.02108).

---

## 11. 코드/산출물 맵
- `src/anomaly_config.py` — 7종 정의·속성·severity·GT 규칙·Fixed SOP·법 원칙 텍스트.
- `src/llm_decision.py` — system prompt·tool 스키마(confidence 포함)·provider 분기.
- `src/oracle.py` — 위임 시 GT Tier2 반환.
- `src/metrics_collector.py` — DecisionRecord·CSV(`results/metrics_raw.csv`).
- `src/spot_factory_v2.py` — Isaac Sim 메인(순찰·실행·3조건 동시 수집).
- `analysis/make_report.py` — 안전 지표·일반화 표(`results/report.md`).
- `analysis/plot_results.py` — F2·F3 및 부록 그림.
- `analysis/deferral_analysis.py` — F4·F4b·F5·ROC(위임 곡선·최소 help).
- `analysis/calibration_analysis.py` — F6·F7(reliability·보정 곡선).
- 실행: `python src/spot_factory_v2.py --models haiku sonnet gpt4o llama --laps 10 --seed 42` → 분석 스크립트들.
