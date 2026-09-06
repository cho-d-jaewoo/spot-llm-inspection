# 실험 재설계 (v3) — "왜 LLM+HiTL인가"를 증명하는 구조

## 문제 (v2)
Fixed를 GT 규칙과 동일하게 구현 → Fixed=100% → LLM 쓸 이유가 사라짐.
작은 이산 속성공간에 결정론적 GT라, 규칙이 정의상 최적이라 LLM은 못 이긴다.

## 해결 원칙
**법/지침은 '원칙'이지 '룩업 테이블'이 아니다.** 규칙이 100%가 되려면
(1) 모든 속성을 완벽 구조화 관측 (2) 모든 상황을 사전 열거 — 둘 다 현실에선 불가.
따라서 평가 환경을 "규칙이 안 통하는 영역"으로 옮긴다.

## 3개 조건 재정의
| 조건 | 정의 |
|---|---|
| **GT** | 전체 잠재 속성 + 법 원칙으로 계산한 전문가 정답 (시뮬레이터만 앎, 객관적 평가 기준) |
| **Fixed (coarse SOP)** | 현행 공장 관행 = anomaly **타입 단위** 고정 반응 (속성 무시, 에스컬레이션 없음). 신규 타입엔 보수적 default. |
| **LLM-Full** | 자연어 리포트 + **법 원칙(system prompt)** 받아 Tier1/Tier2 직접 결정 |
| **LLM+HiTL** | LLM이 에스컬레이션 판단 → 고위험/불확실은 Oracle(인간) 위임 |

→ Fixed는 속성 무시라 50~70%대 (현실적 baseline). LLM은 속성 통합으로 그 위. HiTL이 최고+안전.

## 일반화(OOD) 테스트 — 핵심 논증
- **In-distribution**: FIRE_RISK, LIQUID_LEAK, SMOKE_DETECTED, AGV_STOPPED (기존 4종)
- **OOD (신규 타입)**: GAS_LEAK, ELECTRICAL_FAULT, STRUCTURAL_DAMAGE (사전 SOP가 모름)
  - Fixed: 규칙 없음 → 보수적 default(KEEP_DISTANCE/MAINTENANCE)로 추락
  - GT: 법 원칙 기반 규칙으로 정답 정의
  - LLM: 같은 법 원칙으로 zero-shot 추론 → 일반화
- "법 줬으니 Fixed면 되잖아"에 대한 반박: **사전에 본 적 없는 anomaly엔 규칙을 못 짠다. LLM+법은 일반화하고, 잔여 위험은 HiTL이 잡는다.**

## 지표 개혁 (exact-match 탈피)
- **Tier1 (관찰 자세)** — 안전 ordinal (APPROACH<WAIT<KEEP):
  - 위험(under-cautious): 선택 stance가 GT보다 **덜 보수적** → 안전 위반 (핵심 지표)
  - 과보수(over-cautious): 더 보수적 → 허용(경미한 비용)
- **Tier2 (대응)** — 오류 분류 (대응강도 EMERGENCY>MAINTENANCE>CONTINUE>FALSE_ALARM):
  - **위험 오류(dangerous)**: 예측이 GT보다 약함 & GT∈{EMERGENCY,MAINTENANCE} (특히 EMERGENCY 놓침)
  - **과대응(conservative)**: 예측이 GT보다 강함 (안전하나 비효율)
  - **효율 오류**: FALSE_ALARM↔CONTINUE 혼동 (위험 아님)
- **프레임워크 헤드라인**: "HiTL **위험누락률 ≈ 0** @ **자동화율 X%**" — exact pipeline % 아님

## 프레이밍
"어떤 LLM이 best"가 아니라 **프레임워크 실용성**. 모델은 강건성 확인용 부차 변수
(여러 모델에서 일관되게 위험누락↓). 메인 결과는 모델 집계.

---

# v4 — 위임(deferral) 정책 평가 (KnowNo 스타일) + 정직한 발견

## 방법론 (문헌 근거)
"automation rate"는 지표가 아님(인간 0명=100%). 이 분야 표준은 **safety vs human-help 곡선**에서
*같은 help budget에서 올바른 케이스를 위임하는가*를 평가:
- KnowNo (Ren et al., CoRL 2023, arXiv:2307.01928): success vs help-rate, "minimal help".
- Losey shared autonomy (arXiv:2403.12023): 인간 노력/workload 최소화, 신뢰도 기반 제어 양도.
- Learning-to-defer / 인간-AI 상보성(CTP, arXiv:2302.02944), appropriate reliance(arXiv:2310.02108):
  과의존(over-escalation)/과소의존(위험누락) 분해.
정책: No-help(LLM단독) / Random / Severity-threshold(τ스윕) / Type(SOP) / **LLM-reasoned(제안)** /
Oracle-router(최적) / Always-escalate. 분석: analysis/deferral_analysis.py.

## 실제 데이터 발견 (192행, 4모델, 정직하게)
- **In-distribution**: LLM-reasoned는 help 84%로 **과잉 위임**하면서도 random과 동률/약간 하회,
  severity-threshold(83% help→100%)에 패배. Oracle-router는 **5% help로 100%**. → 현 이진 escalate는
  **볼륨 미보정 + 자기오류 비표적**. "단순 규칙보다 똑똑한 위임"은 in-dist에서 **미지지**.
- **단, severity-threshold는 GT 속성 기반 공식**(배포 시 구조화 추출 필요, 신규타입엔 부재) = 반칙 baseline.
- **OOD(신규 타입)**: LLM-Full(no-help)이 **이미 0% 긴급누락** → 신규 타입 일반화는 **LLM 추론**의 공이지
  HiTL 에스컬레이션의 공이 아님. 규칙 기반 위임(severity/SOP)은 신규 타입에 **존재 불가**.
- 종합: **(A) LLM의 법-기반 추론은 신규 타입에 일반화(강함, 지지됨). (B) 현재 HiTL 에스컬레이션은
  추가 가치 미입증**(과잉 위임 + 이미 LLM이 맞히는 곳을 위임).

## HiTL 가치를 입증하려면 (다음 단계)
- LLM이 **이진 escalate 대신 graded confidence/uncertainty** 출력 → 임계 보정(KnowNo conformal).
- 에스컬레이션을 **자기 오류 영역(낮은 confidence)** 에 표적화 → 곡선 추적해 severity-threshold와
  *같은 help에서* 비교 + OOD 일반화. (시뮬 재실행 필요: LLM confidence 수집)
