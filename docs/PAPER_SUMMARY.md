# 최종 결과 정리 — LLM-HiTL 공장 점검 협업 프레임워크

실험: Isaac Sim 5.0, SPOT 순찰, 4 LLM(Haiku/Sonnet/GPT-4o mini/Llama) × 10 lap, 196 결정 이벤트.
anomaly 7종 = In-distribution 4종(FIRE/LIQUID/SMOKE/AGV) + **OOD 신규 3종**(GAS_LEAK/ELECTRICAL_FAULT/STRUCTURAL_DAMAGE).
조건: **Fixed**(현행 타입단위 SOP) · **LLM-Full**(법원칙+자율) · **LLM+HiTL**(불확실/고위험 인간 위임).

---

## 1. 기여 (Contributions)

1. **2계층 LLM-인간 협업 점검 의사결정 프레임워크.** Tier1(관찰 자세: APPROACH/WAIT/KEEP) + Tier2(대응:
   EMERGENCY/MAINTENANCE/CONTINUE/FALSE_ALARM), 법 원칙(화재예방법·소방기본법 등)을 *프롬프트 원칙*으로 주입,
   **confidence 기반 에스컬레이션**으로 인간에게 위임. 실로봇 없이 물리 시뮬로 완결.

2. **신규 이상 유형에 대한 일반화.** 사전 규칙(SOP)이 없는 OOD 유형에서 규칙 기반은 붕괴(긴급 100% 누락),
   **LLM+법은 zero-shot 일반화**(긴급 0% 누락). "규칙은 열거 못 한 것을 못 푼다, LLM+법은 원칙으로 일반화"를 입증.

3. **confidence-guided 선택적 위임이 인간-로봇 협업 효율을 높임 (KnowNo 방식).** 목표 안전을 *더 적은 인간 부담*으로
   달성 — 긴급 0 누락 @ **65% help**, 98% 안전 @ **25% help**. 구조화 속성이 필요한 severity-rule보다 효율적이며
   **자연어만으로** 작동 + OOD 일반화.

4. **정직한 신뢰도 분석.** LLM raw confidence는 심한 과신(ECE 30%) → temperature scaling으로 교정(4.7%).
   방법론적으로, **단조 보정은 위임 선택 곡선을 불변**으로 둔다(보정은 임계 *해석*을 줄 뿐, 선택을 개선하지 않음).
   더 풍부한 불확실성 신호(엔트로피·self-consistency)를 후속 과제로 제시.

---

## 2. 핵심 수치 (전 모델 집계, 196건)

| 지표 | Fixed (SOP) | LLM-Full | LLM+HiTL |
|---|---|---|---|
| Tier2 위험 과소대응률 | 44.9% | 5.6% | **1.5%** |
| EMERGENCY 누락률 | 76.0% | 7.0% | **0.0%** |
| Tier1 위험률(덜 보수적) | 30.6% | 20.9% | 20.9% |

**일반화 (EMERGENCY 누락, In → OOD):** Fixed 64.7% → **100%(붕괴)**, LLM-Full 5.9% → 9.4%, LLM+HiTL 0.0% → **0.0%**.

**위임 효율 (confidence 순위, 보정 불변):** 안전 ≥98% @ help 25%, ≥99% @ 49%, ≥100% @ 65%.
severity 에스컬레이션 AUC 0.93. LLM-Full Tier2 base accuracy 57.1%.

**신뢰도:** raw conf 평균 0.846 vs 실제 0.571 (ECE 30.5%) → T=7.0 보정 → 평균 0.576, ECE 4.7%.

---

## 3. Figure 구성 (논문 배치 권장)

### 메인 (스토리 순서)
| # | 파일 | 보여주는 것 | 기여 |
|---|---|---|---|
| F1 | (개념도, 직접 작도) | 2계층 + 법원칙 + confidence 위임 구조 | 1 |
| F2 | `fig3_generalization.png` | Fixed는 OOD 긴급 100% 누락, LLM 일반화 | 2 |
| F3 | `fig2_headline_safety.png` | 조건별 안전 위험 지표(Tier2 위험/EMERGENCY 누락/Tier1) | 2 |
| F4 | `fig_deferral_curve.png` | **safety vs human-help — confidence 위임이 random·severity 압도** | 3 |
| F5 | `fig_deferral_split.png` | In-dist vs OOD 위임 — 규칙은 신규 타입에 부재 | 2+3 |
| F6 | `fig_reliability.png` | raw confidence 과신 + temperature scaling 교정 | 4 |
| F7 | `fig_deferral_calibrated.png` | 단조 보정 ⇒ 위임 곡선 불변(정교한 방법론 포인트) | 4 |

### 보조 / 부록
- `fig1_tier1_scatter_grid.png` (+ `scatter/`) — severity↔Tier1 매핑, context-sensitivity, GT 대비 모델별.
- `fig_escalation_roc.png` — 에스컬레이션 판별력(AUC) + 운영점.
- `fig7_tier2_confusion.png` — Fixed의 과대/과소대응 오류 구조.
- `fig8_per_type_danger.png` — anomaly type별 위험률(OOD 표시).
- `fig5_escalation_pr.png`, `fig9_decision_time.png` — 에스컬레이션 P/R/F1, 지연시간.

### ⚠️ 사용 지양 (deprecated — automation rate 중심, 본 연구 지표 아님)
- `fig4_tradeoff.png`, `fig6_autonomy.png` — 자동화율 프레이밍. 본문에서 빼거나 "자동화율은 본 연구 지표 아님" 각주로만.

---

## 4. 한계 (정직하게 명시)

- **시뮬레이션 한정.** GT는 저자 작성 규칙(객관적이나 합성). 실제 전문가 라벨 아님.
- **고위험 시나리오 편향.** 이벤트의 ~50%가 EMERGENCY, GT 위임필요 ~71% → 자동화 여지 구조적 제약.
  단 *자동화율은 본 연구 지표가 아님*(인간 0명=100%는 무의미) — 평가는 "같은 help에서 더 안전한가".
- **LLM 이진 self-escalation은 과잉(85%)** → 직접 쓰지 말고 **confidence 순위 + 보정 임계**로 운영.
- **raw confidence 과신 + 비단조** → T-scaling은 평균만 교정, 형태(중간대 부정확)는 못 고침.
- **소표본(N=196).** 정책 간 마진이 모듈러 → 다중 seed/확장으로 유의성 보강 필요.
- **Oracle=완벽한 인간(GT).** 실제 인간은 불완전 → 상한 가정.

---

## 5. 방법론 근거 (인용)

- KnowNo: Ren et al., *Robots That Ask For Help: Uncertainty Alignment for LLM Planners*, CoRL 2023 (arXiv:2307.01928). — safety vs help, "minimal help".
- Losey (VT Collab): *Aligning Learning with Communication in Shared Autonomy* (arXiv:2403.12023). — 인간 노력 최소화, 신뢰도 기반 제어 양도.
- Gao et al., *Learning Complementary Policies for Human-AI Teams* (arXiv:2302.02944). — CTP, deferral routing.
- *Appropriate Reliance on AI Advice* (arXiv:2310.02108). — 과의존/과소의존.

---

## 6. 후속 과제

1. **불확실성 신호 고도화**: token logprob / self-consistency(N-sample 분산) / 엔트로피 → confidence보다 강한 판별력으로 위임 곡선 추가 개선.
2. **시나리오 재균형**: 루틴 다수 + 긴급 소수로 위임 효율 동적 범위 확대.
3. **다중 seed × 확장**으로 통계적 유의성.
4. **인간 불완전성 모델링**(oracle 노이즈) — 현실적 협업 가정.
