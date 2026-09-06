"""Anomaly 정의 + Severity + GT(전체속성·법원칙) vs Fixed(거친 타입SOP) + OOD 신규타입 + 법 원칙 텍스트."""

import random

# ─────────────────────────────────────────────────────────────────────────────
#  score 테이블 (severity 계산용)
# ─────────────────────────────────────────────────────────────────────────────
SUBSTANCE_SCORE = {"water": 0.0, "coolant": 0.2, "lubricant": 0.3, "chemical": 0.6}
DENSITY_SCORE   = {"light": 0.1, "moderate": 0.3, "dense": 0.5}
SMELL_SCORE     = {"odorless": 0.0, "burning": 0.2, "chemical": 0.3}
LOAD_SCORE      = {"empty": 0.0, "partial": 0.1, "full": 0.2}
GAS_SCORE       = {"inert": 0.1, "toxic": 0.5, "flammable": 0.5}
CONC_SCORE      = {"low": 0.1, "moderate": 0.3, "high": 0.5}
ELEC_SCORE      = {"overheating": 0.2, "sparking": 0.4, "arcing": 0.6}
CRACK_SCORE     = {"hairline": 0.1, "moderate": 0.3, "severe": 0.6}


def _sev_fire(a):
    return min((a["temperature"] - 75) / 75 * 0.5 + min(a["workers_nearby"] * 0.1, 0.3)
               + (0.2 if a["flammables_adjacent"] else 0.0), 1.0)

def _sev_leak(a):
    return min(a["area"] / 2.5 * 0.2 + SUBSTANCE_SCORE[a["substance"]]
               + (0.3 if a["near_electrical"] else 0.0), 1.0)

def _sev_smoke(a):
    return min(DENSITY_SCORE[a["density"]] + SMELL_SCORE[a["smell"]]
               + (0.4 if a["visible_flame"] else 0.0), 1.0)

def _sev_agv(a):
    return min(min(a["stopped_duration"] / 60, 1.0) * 0.4 + (0.3 if a["blocking_main_path"] else 0.0)
               + LOAD_SCORE[a["load_status"]], 1.0)

def _sev_gas(a):
    return min(GAS_SCORE[a["gas_type"]] + CONC_SCORE[a["concentration"]]
               + (0.2 if a["ventilation"] == "poor" else 0.0), 1.0)

def _sev_elec(a):
    return min(ELEC_SCORE[a["condition"]] + (0.25 if a["near_flammable"] else 0.0)
               + (0.25 if a["water_present"] else 0.0), 1.0)

def _sev_struct(a):
    return min(CRACK_SCORE[a["crack_size"]] + (0.25 if a["load_bearing"] else 0.0)
               + min(a["workers_below"] * 0.1, 0.2), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly 정의.  ood=True 이면 사전 SOP(Fixed)가 모르는 신규 타입.
#  gt_rules: (조건, tier1, tier2, escalation, 법령) — 첫 매칭 적용.
# ─────────────────────────────────────────────────────────────────────────────
ANOMALY_CONFIG = {
    # ════════ In-distribution (Fixed SOP가 아는 4종) ════════
    "FIRE_RISK": {
        "ood": False, "color": (1.0, 0.1, 0.1), "severity_fn": _sev_fire,
        "attribute_pool": {"temperature": [75, 90, 110, 150], "workers_nearby": [0, 1, 3],
                           "flammables_adjacent": [True, False]},
        "prompt_template": ("SPOT detected a high-temperature heat source with potential fire risk while patrolling {zone_name}. "
                            "Measured temperature: approximately {temperature}°C. Workers in proximity: {workers_nearby}. "
                            "Flammable materials stored in adjacent area: {flammables_adjacent}."),
        "gt_rules": [
            (lambda a: a["temperature"] == 150, "KEEP_DISTANCE", "EMERGENCY", True, "화재예방법 제40조: 발화점 초과 즉각 신고"),
            (lambda a: a["temperature"] >= 110 and a["flammables_adjacent"], "KEEP_DISTANCE", "EMERGENCY", True, "KOSHA E-184: 발화점 근접 + 가연물"),
            (lambda a: a["temperature"] >= 110 and a["workers_nearby"] >= 1, "KEEP_DISTANCE", "EMERGENCY", True, "산안기준규칙 제241조: 고온 + 인접 작업자"),
            (lambda a: a["temperature"] >= 110, "WAIT_AND_OBSERVE", "MAINTENANCE", True, "KOSHA E-184: 고온 이상, 전문 점검 (비가역)"),
            (lambda a: a["temperature"] == 90 and a["flammables_adjacent"], "WAIT_AND_OBSERVE", "MAINTENANCE", True, "발화 가능물질 인접 고온"),
            (lambda a: a["temperature"] == 90, "WAIT_AND_OBSERVE", "CONTINUE", False, "이상 온도이나 즉각 위협 없음"),
            (lambda a: True, "APPROACH", "CONTINUE", False, "정상 범위 근접"),
        ]},
    "LIQUID_LEAK": {
        "ood": False, "color": (0.1, 0.3, 1.0), "severity_fn": _sev_leak,
        "attribute_pool": {"area": [0.3, 0.8, 2.5], "substance": ["water", "coolant", "lubricant", "chemical"],
                           "near_electrical": [True, False]},
        "prompt_template": ("SPOT detected a liquid leak on the floor of {zone_name}. Estimated affected area: {area} m². "
                            "Suspected substance: {substance}. Adjacent electrical equipment: {near_electrical}."),
        "gt_rules": [
            (lambda a: a["substance"] == "chemical", "KEEP_DISTANCE", "EMERGENCY", True, "위험물안전관리법 제2조: 유해화학물질 즉각 신고"),
            (lambda a: a["near_electrical"] and a["substance"] != "water", "KEEP_DISTANCE", "EMERGENCY", True, "전기안전관리법 제22조: 전도성 액체 + 전기설비"),
            (lambda a: a["near_electrical"] and a["substance"] == "water", "KEEP_DISTANCE", "MAINTENANCE", True, "물 + 전기설비: 감전 위험 (비가역)"),
            (lambda a: a["area"] >= 2.5 and a["substance"] == "coolant", "WAIT_AND_OBSERVE", "MAINTENANCE", False, "대면적 냉각수: MES 자동 요청 가능"),
            (lambda a: a["substance"] == "lubricant", "APPROACH", "MAINTENANCE", False, "윤활유: 슬립 위험, 순수 운영 이슈"),
            (lambda a: True, "APPROACH", "FALSE_ALARM", False, "비위험 물질: 자체 처리 가능"),
        ]},
    "SMOKE_DETECTED": {
        "ood": False, "color": (0.6, 0.6, 0.6), "severity_fn": _sev_smoke,
        "attribute_pool": {"density": ["light", "moderate", "dense"], "smell": ["odorless", "burning", "chemical"],
                           "visible_flame": [True, False]},
        "prompt_template": ("SPOT detected smoke near the ceiling of {zone_name}. Density: {density}. "
                            "Odor characteristics: {smell}. Visible flame observed: {visible_flame}."),
        "gt_rules": [
            (lambda a: a["visible_flame"], "KEEP_DISTANCE", "EMERGENCY", True, "소방기본법 제20조: 화염 확인 시 즉각 119"),
            (lambda a: a["density"] == "dense" and a["smell"] == "chemical", "KEEP_DISTANCE", "EMERGENCY", True, "화재예방법: 유독가스 가능성"),
            (lambda a: a["density"] == "dense" and a["smell"] == "burning", "KEEP_DISTANCE", "EMERGENCY", True, "화재예방법: 대량 연기 + 연소 냄새"),
            (lambda a: a["density"] == "moderate" and a["smell"] in ["chemical", "burning"], "WAIT_AND_OBSERVE", "MAINTENANCE", True, "유해물질/초기화재 의심 (비가역)"),
            (lambda a: a["density"] == "light" and a["smell"] == "burning", "WAIT_AND_OBSERVE", "CONTINUE", False, "미약한 연기: 자율 모니터링"),
            (lambda a: True, "APPROACH", "CONTINUE", False, "수증기·먼지 오탐 가능성"),
        ]},
    "AGV_STOPPED": {
        "ood": False, "color": (1.0, 0.9, 0.1), "severity_fn": _sev_agv,
        "attribute_pool": {"stopped_duration": [2, 15, 60], "blocking_main_path": [True, False],
                           "load_status": ["empty", "partial", "full"]},
        "prompt_template": ("SPOT detected a stopped AGV (automated guided vehicle) in {zone_name}. "
                            "Idle duration: approximately {stopped_duration} minutes. Blocking main passage: {blocking_main_path}. "
                            "AGV load status: {load_status}."),
        "gt_rules": [
            (lambda a: a["blocking_main_path"] and a["load_status"] == "full" and a["stopped_duration"] >= 15, "WAIT_AND_OBSERVE", "MAINTENANCE", True, "KOSHA M-171: 통로 차단 + 전적재"),
            (lambda a: a["blocking_main_path"] and a["stopped_duration"] >= 15, "WAIT_AND_OBSERVE", "MAINTENANCE", True, "산안기준규칙 제98조: 통로 차단 지속"),
            (lambda a: a["blocking_main_path"] and a["stopped_duration"] < 5, "APPROACH", "CONTINUE", False, "일시 정지 가능성: 자율 대기"),
            (lambda a: not a["blocking_main_path"] and a["stopped_duration"] >= 60, "APPROACH", "MAINTENANCE", False, "장시간 정지: MES 자동 요청 가능"),
            (lambda a: True, "APPROACH", "FALSE_ALARM", False, "정상 대기 구간"),
        ]},

    # ════════ OOD — 사전 SOP가 모르는 신규 타입 3종 (일반화 테스트) ════════
    "GAS_LEAK": {
        "ood": True, "color": (0.5, 1.0, 0.4), "severity_fn": _sev_gas,
        "attribute_pool": {"gas_type": ["inert", "toxic", "flammable"], "concentration": ["low", "moderate", "high"],
                           "ventilation": ["good", "poor"]},
        "prompt_template": ("SPOT detected a gas leak in {zone_name}. Suspected gas type: {gas_type}. "
                            "Measured concentration level: {concentration}. Area ventilation: {ventilation}."),
        "gt_rules": [
            (lambda a: a["gas_type"] == "flammable" and a["concentration"] == "high", "KEEP_DISTANCE", "EMERGENCY", True, "고압가스안전관리법: 가연성 고농도 즉각 신고"),
            (lambda a: a["gas_type"] == "toxic" and (a["concentration"] == "high" or a["ventilation"] == "poor"), "KEEP_DISTANCE", "EMERGENCY", True, "산안기준규칙: 중독·질식 위험 즉각 대피"),
            (lambda a: a["gas_type"] in ("toxic", "flammable") and a["concentration"] == "moderate", "WAIT_AND_OBSERVE", "MAINTENANCE", True, "유해·가연 가스 중농도: 전문 점검 (비가역)"),
            (lambda a: a["gas_type"] == "inert" and a["concentration"] == "high" and a["ventilation"] == "poor", "WAIT_AND_OBSERVE", "MAINTENANCE", True, "밀폐공간 질식 위험"),
            (lambda a: a["concentration"] == "low", "APPROACH", "CONTINUE", False, "저농도: 근접 확인 후 모니터링"),
            (lambda a: True, "APPROACH", "CONTINUE", False, "경미한 누출: 자율 모니터링"),
        ]},
    "ELECTRICAL_FAULT": {
        "ood": True, "color": (1.0, 0.6, 0.0), "severity_fn": _sev_elec,
        "attribute_pool": {"condition": ["overheating", "sparking", "arcing"], "near_flammable": [True, False],
                           "water_present": [True, False]},
        "prompt_template": ("SPOT detected an electrical fault on a control panel in {zone_name}. Observed condition: {condition}. "
                            "Flammable material nearby: {near_flammable}. Water/moisture present: {water_present}."),
        "gt_rules": [
            (lambda a: a["condition"] == "arcing", "KEEP_DISTANCE", "EMERGENCY", True, "전기안전관리법 제22조 / 화재예방법: 아크 = 화재·감전 즉각 신고"),
            (lambda a: a["condition"] == "sparking" and (a["near_flammable"] or a["water_present"]), "KEEP_DISTANCE", "EMERGENCY", True, "스파크 + 가연물/물: 발화·감전 위험"),
            (lambda a: a["condition"] == "sparking", "WAIT_AND_OBSERVE", "MAINTENANCE", True, "스파크: 전문 점검 필요 (비가역)"),
            (lambda a: a["condition"] == "overheating" and a["near_flammable"], "WAIT_AND_OBSERVE", "MAINTENANCE", True, "과열 + 가연물: 발화 우려"),
            (lambda a: a["condition"] == "overheating", "APPROACH", "MAINTENANCE", False, "과열: 정비 필요, MES 자동 요청 가능"),
            (lambda a: True, "APPROACH", "CONTINUE", False, "경미: 자율 모니터링"),
        ]},
    "STRUCTURAL_DAMAGE": {
        "ood": True, "color": (0.6, 0.4, 0.2), "severity_fn": _sev_struct,
        "attribute_pool": {"crack_size": ["hairline", "moderate", "severe"], "load_bearing": [True, False],
                           "workers_below": [0, 1, 3]},
        "prompt_template": ("SPOT detected structural damage (crack/deformation) in {zone_name}. Crack severity: {crack_size}. "
                            "Component is load-bearing: {load_bearing}. Workers below/near: {workers_below}."),
        "gt_rules": [
            (lambda a: a["crack_size"] == "severe" and a["load_bearing"], "KEEP_DISTANCE", "EMERGENCY", True, "산안기준규칙: 붕괴 위험 즉각 대피·통제"),
            (lambda a: a["crack_size"] == "severe" and a["workers_below"] >= 1, "KEEP_DISTANCE", "EMERGENCY", True, "산안기준규칙: 낙하·붕괴 + 작업자"),
            (lambda a: a["crack_size"] == "severe", "WAIT_AND_OBSERVE", "MAINTENANCE", True, "심각 균열: 구조 점검 필요 (비가역)"),
            (lambda a: a["crack_size"] == "moderate" and a["load_bearing"], "WAIT_AND_OBSERVE", "MAINTENANCE", True, "내력부재 중간 균열: 전문 점검"),
            (lambda a: a["crack_size"] == "moderate", "APPROACH", "MAINTENANCE", False, "비내력 균열: 정비 요청"),
            (lambda a: True, "APPROACH", "CONTINUE", False, "미세 균열: 자율 모니터링"),
        ]},
}

ANOMALY_TYPES = list(ANOMALY_CONFIG.keys())
IN_DIST_TYPES = [t for t, c in ANOMALY_CONFIG.items() if not c["ood"]]
OOD_TYPES     = [t for t, c in ANOMALY_CONFIG.items() if c["ood"]]


# ─────────────────────────────────────────────────────────────────────────────
#  Fixed (coarse SOP) — 현행 공장 관행. 타입 단위 고정 반응, 속성 무시, 에스컬레이션 없음.
#  신규(OOD) 타입은 사전 규칙 없음 → 보수적 default. (이것이 일반화 실패를 드러냄)
# ─────────────────────────────────────────────────────────────────────────────
FIXED_SOP = {
    "FIRE_RISK":      {"tier1": "KEEP_DISTANCE",    "tier2": "EMERGENCY"},
    "SMOKE_DETECTED": {"tier1": "WAIT_AND_OBSERVE", "tier2": "MAINTENANCE"},
    "LIQUID_LEAK":    {"tier1": "APPROACH",         "tier2": "MAINTENANCE"},
    "AGV_STOPPED":    {"tier1": "APPROACH",         "tier2": "CONTINUE"},
}
FIXED_DEFAULT = {"tier1": "KEEP_DISTANCE", "tier2": "MAINTENANCE"}  # 미지 타입 보수적 default


# ─────────────────────────────────────────────────────────────────────────────
#  법 원칙 텍스트 (LLM system prompt 주입용) — 룩업 테이블이 아닌 '원칙'
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_PRINCIPLES = """## Korean Safety/Reporting Principles (apply by reasoning, not lookup)
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
- These are PRINCIPLES: you may encounter anomaly types or attribute combinations not explicitly listed — reason from the underlying intent (life safety > legal duty > asset protection > efficiency)."""


# ─────────────────────────────────────────────────────────────────────────────
#  공개 API
# ─────────────────────────────────────────────────────────────────────────────
def is_ood(anomaly_type: str) -> bool:
    return ANOMALY_CONFIG[anomaly_type]["ood"]


def compute_severity(anomaly_type: str, attrs: dict) -> float:
    return float(ANOMALY_CONFIG[anomaly_type]["severity_fn"](attrs))


def get_ground_truth(anomaly_type: str, attrs: dict) -> dict:
    """전체 속성 + 법 원칙 기반 전문가 정답 (객관적 평가 기준)."""
    for cond, t1, t2, esc, ref in ANOMALY_CONFIG[anomaly_type]["gt_rules"]:
        if cond(attrs):
            return {"tier1": t1, "tier2": t2, "escalation": esc, "reference": ref}
    return {"tier1": "WAIT_AND_OBSERVE", "tier2": "CONTINUE", "escalation": False, "reference": "default"}


def get_fixed_decision(anomaly_type: str, attrs: dict) -> dict:
    """거친 타입 단위 SOP (속성 무시). 신규 타입은 보수적 default. 에스컬레이션 없음."""
    sop = FIXED_SOP.get(anomaly_type, FIXED_DEFAULT)
    return {"tier1": sop["tier1"], "tier2": sop["tier2"], "escalated": False}


def sample_attributes(anomaly_type: str, rng: random.Random) -> dict:
    pool = ANOMALY_CONFIG[anomaly_type]["attribute_pool"]
    return {k: rng.choice(v) for k, v in pool.items()}


def _fmt_attr(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def build_prompt(anomaly_type: str, zone_name: str, attrs: dict) -> str:
    fmt = {"zone_name": zone_name}
    for k, v in attrs.items():
        fmt[k] = _fmt_attr(v)
    return ANOMALY_CONFIG[anomaly_type]["prompt_template"].format(**fmt)
