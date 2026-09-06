"""
spot_factory_inspection.py  —  Phase 4 (multi-LLM + local + recording)
=======================================================================
Isaac Sim 5.0  |  SPOT 자율 점검 시뮬레이션

Action set (3개):
  APPROACH         : 1.5m 접근 후 arcing 관찰 (저위험)
  WAIT_AND_OBSERVE : 현 위치에서 arcing 관찰 (중위험)
  KEEP_DISTANCE    : 4m 거리 유지하며 arcing 관찰 (고위험)

CLI 옵션:
  --use-llm ALIAS  : LLM 모델 별칭 (생략 시 rule-based)
                     Claude : haiku / sonnet
                     OpenAI : gpt-5-mini / gpt-5-nano / gpt-4o
                     Local  : llama3.3
                              (Ollama 서버 로컬 실행 필요)
  --laps N         : 총 순찰 lap 수 (기본 10). 각 lap마다 anomaly 재스폰.
  --metrics        : 모든 lap 완료 후 결과 지표 출력하고 자동 종료
  --seed N         : random anomaly 생성용 시드 (재현용)
  --follow-cam     : SPOT 추적 카메라 (기본: 부감 시점)
  --speed N        : 시뮬레이션 속도 배율 (1~4, 기본 1)
  --record         : viewport 영상 녹화 (./recordings/[timestamp]/simulation.mp4)
  --record-fps N   : 녹화 FPS (기본 20)
  --headless       : GUI 없이 실행

Rule-based fixed mapping (LLM 미사용 시):
  FIRE_RISK      → KEEP_DISTANCE
  SMOKE_DETECTED → WAIT_AND_OBSERVE
  AGV_STOPPED    → APPROACH
  LIQUID_LEAK    → APPROACH

환경 변수:
  ANTHROPIC_API_KEY  (Claude 별칭 사용 시)
  OPENAI_API_KEY     (OpenAI 별칭 사용 시)

로컬 LLM (Ollama):
  pip install ollama 또는 openai
  ollama serve                  # 서버 실행 (기본 localhost:11434)
  ollama pull llama3.3          # 모델 다운로드
  python ... --use-llm llama3.3

녹화 의존성:
  ffmpeg (PATH에 있으면 자동 MP4 변환, 없으면 PNG 시퀀스 저장)

실행 예시:
  python spot_factory_inspection.py                                   # rule-based
  python spot_factory_inspection.py --use-llm haiku --metrics         # Claude Haiku
  python spot_factory_inspection.py --use-llm gpt-4o-mini --metrics   # GPT-4o-mini
  python spot_factory_inspection.py --use-llm llama3.3 --metrics      # 로컬 Llama
  python spot_factory_inspection.py --use-llm sonnet --record --laps 5
"""

import argparse
import json
import os
import random
import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless",   action="store_true",
                    help="GUI 없이 실행")
parser.add_argument("--use-llm",    type=str, default=None, metavar="ALIAS",
                    help="LLM 별칭: haiku/sonnet (Claude), gpt-5-mini/gpt-5-nano/gpt-4o (OpenAI), "
                         "llama3.3 (Ollama). 생략 시 rule-based")
parser.add_argument("--record",     action="store_true",
                    help="viewport 영상 녹화 (./recordings/[timestamp]/ 저장)")
parser.add_argument("--record-fps", type=int, default=20,
                    help="녹화 FPS (기본 20)")
parser.add_argument("--metrics",    action="store_true",
                    help="모든 anomaly 처리 후 결과 지표 출력 + 자동 종료")
parser.add_argument("--seed",       type=int, default=None,
                    help="anomaly 랜덤 생성 시드 (재현용)")
parser.add_argument("--follow-cam", action="store_true",
                    help="SPOT 추적 카메라 (기본: 부감 시점 고정)")
parser.add_argument("--speed",      type=int, default=1, choices=[1, 2, 3, 4],
                    help="시뮬레이션 속도 배율 (1~4, 한 render당 physics step 수)")
parser.add_argument("--laps",       type=int, default=10,
                    help="총 순찰 lap 수 (각 lap 끝나면 anomaly 재스폰, 기본 10)")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless, "anti_aliasing": 0})

# ── Core imports (SimulationApp 이후) ─────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy

# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════

# 공장 환경 USD — warehouse_multiple_shelves (선반 많은 작업 창고)
ENV_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)

SPOT_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.0/Isaac/Robots/BostonDynamics/spot/spot.usd"
)

# Anomaly 아이콘 텍스처 경로
# ── Repo-relative asset paths ────────────────────────────────────────────────
# Resolved from this file's location so a fresh clone works anywhere. Set
# SPOT_REPO_ROOT to point elsewhere if you keep large assets outside the repo.
from pathlib import Path as _Path
_REPO_ROOT = _Path(os.environ.get("SPOT_REPO_ROOT") or _Path(__file__).resolve().parents[1])

ICON_DIR = str(_REPO_ROOT / "assets" / "icons")
ICON_FILES = {
    "FIRE_RISK":      "icon_fire.png",
    "LIQUID_LEAK":    "icon_leak.png",
    "AGV_STOPPED":    "icon_agv.png",
    "SMOKE_DETECTED": "icon_smoke.png",
}

POLICY_PATH = str(_REPO_ROOT / "policies" / "spot_policy.pt")
ENV_YAML    = str(_REPO_ROOT / "policies" / "spot_env.yaml")

PHYSICS_DT   = 0.002
RENDERING_DT = 0.002


# ── LLM 설정 ──────────────────────────────────────────────────────────────────
# ── 멀티 LLM 설정 ─────────────────────────────────────────────────────────
# 별칭 → (provider, model_id)
#  claude : ANTHROPIC_API_KEY 필요  https://console.anthropic.com
#  openai : OPENAI_API_KEY 필요     https://platform.openai.com
#  local  : Ollama 서버 필요        https://ollama.ai  (ollama serve)
# 별칭 → (provider, model_id)
#  claude : ANTHROPIC_API_KEY 필요  https://console.anthropic.com
#  openai : OPENAI_API_KEY 필요     https://platform.openai.com
#  local  : Ollama 서버 필요        https://ollama.ai  (ollama serve)
MODEL_ALIASES = {
    # Anthropic Claude
    "haiku":       ("claude", "claude-haiku-4-5"),     # 저가/고속
    "sonnet":      ("claude", "claude-sonnet-4-6"),    # 균형
    # OpenAI GPT
    "gpt-4o-mini":  ("openai", "gpt-4o-mini"),

    # 로컬 LLM (Ollama) — 오프라인 baseline
    "llama3.1":    ("local", "llama3.1"),
}

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Action set — SPOT의 sensing mode 결정 (3개)
# 모든 action 종료 후 자동으로 PATROL 복귀 → REPORT/REQUEST는 별도 action 아님
ACTIONS = [
    "APPROACH",          # 저위험 — 직접 접근
    "WAIT_AND_OBSERVE",  # 중위험 — 제자리 관찰
    "KEEP_DISTANCE",     # 고위험 — 거리 유지 + 호 그리며 다각도 관찰
]

# Action 설명 (LLM system prompt에 포함)
ACTION_DESC = """- APPROACH: Move within 1.5m of the anomaly for close-up inspection. Use when the situation appears LOW risk and direct examination is safe.
- WAIT_AND_OBSERVE: Stay in place and observe from current position. Use for MEDIUM risk situations where moving closer or further is unnecessary.
- KEEP_DISTANCE: Maintain a 4m safety distance, face the anomaly, and arc around it for multi-angle observation. Use for HIGH risk situations where direct approach would be dangerous."""

# 공통 JSON schema
_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string", "enum": ACTIONS,
            "description": "The action SPOT should perform.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief justification (1-2 sentences in English).",
        },
        "severity": {
            "type": "string", "enum": ["LOW", "MEDIUM", "HIGH"],
            "description": "Severity level of the situation.",
        },
    },
    "required": ["action", "reasoning", "severity"],
}

# Anthropic tool schema
ANTHROPIC_TOOL = {
    "name": "decide_inspection_action",
    "description": "Decide SPOT robot's inspection action in a smart factory scenario.",
    "input_schema": _TOOL_PARAMS,
}

# OpenAI / Ollama function-calling schema
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_inspection_action",
        "description": "Decide SPOT robot's inspection action in a smart factory scenario.",
        "parameters": _TOOL_PARAMS,
    },
}

SYSTEM_PROMPT = f"""You are the decision-making module of SPOT, a quadruped inspection robot operating in a smart factory.

SPOT cannot directly resolve issues — it can only choose HOW to observe an anomaly. 
After observation, the situation is either deemed safe (patrol resumes) or a human operator is called. 
Your task is to choose the appropriate SENSING MODE based on the perceived risk level.

[Available sensing modes]
{ACTION_DESC}

[Severity levels]
- LOW    : Minor concern, safe to approach for direct inspection.
- MEDIUM : Moderate concern, in-place observation appropriate.
- HIGH   : Significant risk, maintain distance and observe from afar.

[Severity-action alignment (expected pattern)]
- LOW    → APPROACH
- MEDIUM → WAIT_AND_OBSERVE
- HIGH   → KEEP_DISTANCE

[Decision criteria]
1. Look carefully at the SPECIFIC attributes provided (e.g., temperature value, number of workers,
   substance type, electrical proximity, smoke density). The same anomaly type can have very different
   severity depending on these details.
2. Consider attribute INTERACTIONS — a moderate temperature combined with nearby workers and
   adjacent flammables is much more dangerous than a high temperature alone in an empty area.
3. SPOT's own safety also matters — do not approach extreme heat, dense chemical smoke,
   or live electrical hazards.
4. Provide reasoning that explicitly references the specific attribute values you used.

You MUST respond by invoking the decide_inspection_action tool."""


# ── Anthropic 클라이언트 (지연 import — 패키지 없어도 시뮬 자체는 실행 가능) ──
class LLMConfig:
    """LLM client + provider + model_id 묶음."""
    def __init__(self, provider, model_id, client):
        self.provider = provider   # "claude" | "openai" | "local"
        self.model_id = model_id
        self.client   = client


def init_llm_client(alias: str = None):
    """LLM 클라이언트 초기화.
       alias 예: haiku, sonnet, gpt-4o-mini, llama3.3 ...
       None → rule-based 모드 (None 반환)."""
    if alias is None:
        print("[DECISION] Rule-based mode  (use --use-llm <alias> to enable LLM)")
        return None

    if alias not in MODEL_ALIASES:
        print(f"[DECISION] Unknown alias '{alias}'. Available: {', '.join(MODEL_ALIASES)}")
        import sys; sys.exit(1)

    provider, model_id = MODEL_ALIASES[alias]

    # ── Anthropic Claude ─────────────────────────────────────────────────
    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            print(f"[DECISION] ANTHROPIC_API_KEY 없음 → cannot use '{alias}'"); import sys; sys.exit(1)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print(f"[DECISION] LLM mode — Claude '{alias}'  ({model_id})")
            return LLMConfig(provider, model_id, client)
        except ImportError:
            print("[DECISION] pip install anthropic"); import sys; sys.exit(1)

    # ── OpenAI GPT ───────────────────────────────────────────────────────
    if provider == "openai":
        if not OPENAI_API_KEY:
            print(f"[DECISION] OPENAI_API_KEY 없음 → cannot use '{alias}'"); import sys; sys.exit(1)
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print(f"[DECISION] LLM mode — OpenAI '{alias}'  ({model_id})")
            return LLMConfig(provider, model_id, client)
        except ImportError:
            print("[DECISION] pip install openai"); import sys; sys.exit(1)

    # ── 로컬 LLM (Ollama) ─────────────────────────────────────────────────
    # Ollama는 OpenAI-compatible REST API 제공 → openai SDK 재사용
    # 서버 실행: ollama serve  |  모델 다운: ollama pull <model>
    if provider == "local":
        try:
            import openai
            client = openai.OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",          # Ollama는 키 불필요, 아무 값이나 가능
            )
            print(f"[DECISION] LLM mode — Local (Ollama) '{alias}'  ({model_id})")
            print(f"           Ollama URL: {OLLAMA_BASE_URL}")
            print(f"           모델 없으면: ollama pull {model_id}")
            return LLMConfig(provider, model_id, client)
        except ImportError:
            print("[DECISION] pip install openai  (Ollama도 openai SDK 사용)")
            import sys; sys.exit(1)

    import sys; sys.exit(1)


# ── 점검 구역 정의 ───────────────────────────────────────────────────────────
# ZONE 정의 (시각화 + 자연어 prompt용)
# 위치는 PATROL_WAYPOINTS 경로 위에 있도록 배치
ZONES = {
    "ZONE_A": {"name": "Assembly Line A",   "center": (-7.0, -5.0)},  # 좌측 라인
    "ZONE_B": {"name": "Storage Area B",    "center": (-2.5,  5.0)},  # 좌측 라인
    "ZONE_C": {"name": "AGV Workspace",     "center": ( -4.6, 15.6)},  # 상단 라인
    "ZONE_D": {"name": "Packaging Area",    "center": ( 7.0, 11.0)},  # 우측 라인
    "ZONE_E": {"name": "Loading Dock",      "center": ( 2.5, -1.8)},  # 우측 라인
}

# 순찰 경로 — 공장 외곽 직사각형 한 바퀴 (8개 waypoint)
# ZONE 위치와 무관하게 독립적으로 정의, 경로가 모든 ZONE을 통과하도록 설계
PATROL_WAYPOINTS = [
    (-5.0, -8.0),   # 1. 좌하
    (-5.0,  0.0),   # 2. 좌중  
    (-5.0,  8.0),   # 3. 좌상 
    (-5.0, 14.0),   # 4. 좌상끝
    ( 5.0, 14.0),   # 5. 우상끝 
    ( 5.0,  8.0),   # 6. 우상 
    ( 5.0,  0.0),   # 7. 우중
    ( 5.0, -8.0),   # 8. 우하
]


# ── 이상 객체 정의 (시작 시점부터 모두 배치) ─────────────────────────────────
# 이상 상황 type 풀 — 실행마다 ZONE에 랜덤 배정 + attributes도 랜덤 생성
# attr_pool에서 랜덤 추출하여 같은 type이라도 매번 다른 맥락 생성 (LLM 효용)
ANOMALY_TYPES = [
    {
        "type": "FIRE_RISK",
        "color": (1.0, 0.1, 0.1),
        "attr_pool": {
            "temperature": [75, 90, 110, 150],            # °C
            "workers_nearby": [0, 1, 3],                  # count
            "flammables_adjacent": [True, False],
        },
        "prompt_template": (
            "SPOT detected a high-temperature heat source with potential fire risk while patrolling {zone_name}. "
            "Measured temperature: approximately {temperature}°C. "
            "Workers in proximity: {workers_nearby}. "
            "Flammable materials stored in adjacent area: {flammables_adjacent}."
        ),
    },
    {
        "type": "LIQUID_LEAK",
        "color": (0.1, 0.3, 1.0),
        "attr_pool": {
            "area": [0.3, 0.8, 2.5],                       # m²
            "substance_suspected": ["coolant", "lubricant oil",
                                    "suspected chemical", "water"],
            "near_electrical": [True, False],
        },
        "prompt_template": (
            "SPOT detected a liquid leak on the floor of {zone_name}. "
            "Estimated affected area: {area} m². "
            "Suspected substance: {substance_suspected}. "
            "Adjacent electrical equipment: {near_electrical}."
        ),
    },
    {
        "type": "AGV_STOPPED",
        "color": (1.0, 0.9, 0.1),
        "attr_pool": {
            "stopped_duration_min": [2, 15, 60],            # minutes
            "blocking_path": [True, False],
            "load_status": ["empty", "fully loaded", "loading in progress"],
        },
        "prompt_template": (
            "SPOT detected a malfunctioning AGV in {zone_name}. "
            "Idle duration: approximately {stopped_duration_min} minutes. "
            "Blocking main passage: {blocking_path}. "
            "AGV load status: {load_status}."
        ),
    },
    {
        "type": "SMOKE_DETECTED",
        "color": (0.6, 0.6, 0.6),
        "attr_pool": {
            "density": ["light", "moderate", "dense"],
            "smell": ["odorless", "burnt smell", "chemical smell"],
            "visible_flame": [True, False],
        },
        "prompt_template": (
            "SPOT detected smoke near the ceiling of {zone_name}. "
            "Density: {density}. "
            "Odor characteristics: {smell}. "
            "Visible flame observed: {visible_flame}."
        ),
    },
]


def _fmt_attr(v):
    """Convert attribute value to natural language for prompt."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def compute_severity_score(anomaly_type: str, attributes: dict) -> float:
    """Anomaly attributes 기반 0~1 severity score 계산.

    [용도]
      시각적 표현 전용 — anomaly zone의 빨강 진하기를 결정.
      낮은 점수 → 옅은 핑크빨강 (관측만 해도 안전해 보임)
      높은 점수 → 진한 빨강    (위험 상황으로 보임)

    [주의]
      이 score는 의사결정에 사용되지 않음.
      Rule-based baseline은 type 단독 매핑만 사용 (이 함수와 무관).
      LLM은 raw attributes를 직접 보고 판단 (이 함수와 무관).

    [식]
      score = mean( score_table[type][attribute][value] for attr in attrs )
      각 attribute 값별 0~1 점수는 임의로 부여한 시각적 강도 매핑.
      엄밀한 위험성평가 모델이 아니며, 단순히 zone 색상 그라데이션을 위한 휴리스틱.
    """
    SCORING = {
        "FIRE_RISK": {
            "temperature":          {75: 0.2, 90: 0.4, 110: 0.7, 150: 1.0},
            "workers_nearby":       {0: 0.0, 1: 0.3, 3: 0.7},
            "flammables_adjacent":  {True: 0.7, False: 0.0},
        },
        "LIQUID_LEAK": {
            "area":                 {0.3: 0.1, 0.8: 0.4, 2.5: 0.8},
            "substance_suspected":  {
                "water": 0.1, "coolant": 0.3,
                "lubricant oil": 0.5, "suspected chemical": 0.9,
            },
            "near_electrical":      {True: 0.7, False: 0.0},
        },
        "AGV_STOPPED": {
            "stopped_duration_min": {2: 0.1, 15: 0.4, 60: 0.8},
            "blocking_path":        {True: 0.5, False: 0.1},
            "load_status":          {
                "empty": 0.1, "loading in progress": 0.4, "fully loaded": 0.5,
            },
        },
        "SMOKE_DETECTED": {
            "density":              {"light": 0.2, "moderate": 0.5, "dense": 0.9},
            "smell":                {
                "odorless": 0.1, "burnt smell": 0.6, "chemical smell": 0.9,
            },
            "visible_flame":        {True: 1.0, False: 0.0},
        },
    }

    table = SCORING.get(anomaly_type, {})
    if not table:
        return 0.5

    scores = []
    for k, v in attributes.items():
        if k in table:
            scores.append(table[k].get(v, 0.5))
    return sum(scores) / len(scores) if scores else 0.5


def severity_to_red(score: float) -> np.ndarray:
    """0~1 score를 빨강 톤 그라데이션으로 매핑 (불투명해 보이게 채도 진하게).
       score=0.0 → (0.95, 0.55, 0.55)  핑크빨강    (LOW)
       score=0.5 → (0.78, 0.30, 0.30)  중간 빨강   (MEDIUM)
       score=1.0 → (0.60, 0.05, 0.05)  진한 빨강   (HIGH)"""
    score = max(0.0, min(1.0, score))
    r = 0.95 - 0.35 * score
    g = 0.55 - 0.50 * score
    b = 0.55 - 0.50 * score
    return np.array([r, g, b])


def generate_anomalies(seed=None):
    """매 실행마다 ZONE-Type 매칭 + 각 anomaly의 attribute를 랜덤 생성.
       LLM의 의사결정 효용을 위해: 같은 type이라도 attribute에 따라 prompt가 달라짐."""
    rng = random.Random(seed)
    anomalies = []
    for zone_id, zone_info in ZONES.items():
        atype = rng.choice(ANOMALY_TYPES)

        # attribute 랜덤 추출
        attrs = {k: rng.choice(v) for k, v in atype["attr_pool"].items()}

        # 한국어 prompt 생성
        fmt = {"zone_name": zone_info["name"]}
        for k, v in attrs.items():
            fmt[k] = _fmt_attr(v)
        prompt = atype["prompt_template"].format(**fmt)

        anomalies.append({
            "id":       f"{atype['type']}_{zone_id}",
            "type":     atype["type"],
            "zone":     zone_id,
            "zone_name": zone_info["name"],
            "position": zone_info["center"],
            "radius":   2.5,
            "color":    atype["color"],
            "attributes": attrs,
            "prompt":   prompt,
            "severity_score": compute_severity_score(atype["type"], attrs),
        })
    return anomalies


# ═════════════════════════════════════════════════════════════════════════════
#  LLM 호출 (Claude API + rule-based fallback)
# ═════════════════════════════════════════════════════════════════════════════

def query_llm(llm_cfg, prompt: str) -> dict:
    """결정 함수.  llm_cfg=None → rule-based, 있으면 provider별 호출."""
    if llm_cfg is None:
        return _fallback_rule_based(prompt)
    if llm_cfg.provider == "claude":
        return _query_claude(llm_cfg, prompt)
    # openai / local(Ollama) 모두 같은 SDK 사용
    return _query_openai_compat(llm_cfg, prompt)


def _query_claude(cfg, prompt: str) -> dict:
    """Anthropic Claude — tool_use API."""
    try:
        resp = cfg.client.messages.create(
            model=cfg.model_id, max_tokens=400,
            system=SYSTEM_PROMPT,
            tools=[ANTHROPIC_TOOL],
            tool_choice={"type": "tool", "name": "decide_inspection_action"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return {"action": block.input["action"],
                        "reasoning": block.input["reasoning"],
                        "severity": block.input["severity"],
                        "source": "LLM"}
        _llm_exit("no tool_use block", cfg, str(resp.content))
    except Exception as e:
        _llm_exit(str(e), cfg)


def _query_openai_compat(cfg, prompt: str) -> dict:
    """OpenAI GPT 및 Ollama(local) — function-calling API.
       - GPT-5 시리즈: reasoning 토큰 필요 → max_completion_tokens 크게 + reasoning_effort=minimal
       - GPT-4 시리즈: 일반 chat — 작은 토큰으로 충분
       - 로컬 모델: function calling 불안정 → JSON prompt fallback 포함
    """
    # GPT-5 reasoning model 감지 (hidden reasoning tokens 사용)
    is_gpt5 = cfg.model_id.startswith("gpt-5")

    kwargs = {
        "model": cfg.model_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",   "content": prompt}],
        "tools": [OPENAI_TOOL],
        "tool_choice": {"type": "function",
                        "function": {"name": "decide_inspection_action"}},
        # GPT-5는 hidden reasoning tokens 사용 → 충분한 여유 필요
        "max_completion_tokens": 2500 if is_gpt5 else 400,
    }
    if is_gpt5:
        # reasoning_effort: "minimal" / "low" / "medium" / "high"
        # 본 과제는 단순 결정이라 minimal로 충분 + 응답 속도 빠름
        kwargs["extra_body"] = {"reasoning_effort": "minimal"}

    # 1차 시도: function calling
    try:
        resp = cfg.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            args = json.loads(msg.tool_calls[0].function.arguments)
            return {"action": args["action"], "reasoning": args["reasoning"],
                    "severity": args["severity"], "source": "LLM"}
    except Exception as e1:
        if cfg.provider == "local":
            return _query_local_json_fallback(cfg, prompt, str(e1))
        _llm_exit(str(e1), cfg)

    # tool_calls 없음 → 디버깅 정보 출력
    if cfg.provider == "local":
        return _query_local_json_fallback(cfg, prompt, "no tool_calls")

    finish = resp.choices[0].finish_reason
    content_preview = (msg.content or "")[:200]
    extra = f"finish_reason={finish}, content={content_preview!r}"
    if finish == "length":
        extra += "  → reasoning이 토큰 한도 초과. max_completion_tokens 더 크게 시도"
    _llm_exit("no tool_calls in response", cfg, extra)


def _query_local_json_fallback(cfg, prompt: str, reason: str) -> dict:
    """로컬 모델 function calling 실패 시 JSON prompt로 재시도.
       function calling 미지원 모델(일부 Llama/Gemma)도 처리 가능."""
    json_prompt = (
        prompt + "\n\n"
        "Respond ONLY with a JSON object (no markdown, no explanation):\n"
        '{"action": "APPROACH|WAIT_AND_OBSERVE|KEEP_DISTANCE", '
        '"reasoning": "...", "severity": "LOW|MEDIUM|HIGH"}'
    )
    try:
        resp = cfg.client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": json_prompt}],
            max_completion_tokens=300,
        )
        text = resp.choices[0].message.content or ""
        # JSON 파싱 (```json ``` 블록 제거)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        args = json.loads(text)
        if args.get("action") in ACTIONS:
            return {"action": args["action"], "reasoning": args.get("reasoning",""),
                    "severity": args.get("severity","MEDIUM"), "source": "LLM"}
    except Exception as e2:
        pass
    # 최종 실패
    _llm_exit(f"function calling failed ({reason}), JSON fallback also failed", cfg)


def _llm_exit(reason: str, cfg, extra: str = ""):
    """에러 진단 출력 + 종료."""
    print("\n" + "!" * 68)
    print(f"  [LLM ERROR] {cfg.provider.upper()} 호출 실패")
    print(f"  Alias    : {cfg.model_id}")
    print(f"  Reason   : {reason}")
    if extra: print(f"  Detail   : {extra[:200]}")
    print()
    err = reason.lower()
    if "401" in err or "auth" in err or "api key" in err:
        env = "ANTHROPIC_API_KEY" if cfg.provider=="claude" else "OPENAI_API_KEY"
        print(f"  진단: API 키 인증 실패 → echo %{env}% 확인")
    elif "404" in err or "not found" in err or "does not exist" in err:
        if cfg.provider == "local":
            print(f"  진단: Ollama 모델 없음 → ollama pull {cfg.model_id}")
            print(f"        Ollama 서버 실행 중인지 확인 → ollama serve")
        else:
            print(f"  진단: 모델 ID 없음. MODEL_ALIASES 확인")
    elif "connect" in err or "connection" in err:
        print(f"  진단: Ollama 서버 미실행 → ollama serve")
    elif "credit" in err or "402" in err or "billing" in err:
        print(f"  진단: 크레딧 부족 → Console → Plans & Billing")
    elif "429" in err or "rate" in err:
        print(f"  진단: 요청 속도 초과 → 잠시 후 재시도")
    print("!" * 68)
    import sys; sys.exit(1)


def _fallback_rule_based(prompt: str) -> dict:
    """Rule-based fixed mapping — type만 보고 action 결정.
       Attribute는 일절 고려하지 않음 (LLM의 차별점을 명확히 하기 위함).

    [Type → Action 매핑]
      FIRE_RISK      → KEEP_DISTANCE   (화재는 무조건 거리 유지)
      SMOKE_DETECTED → WAIT_AND_OBSERVE (연기는 제자리 관찰)
      AGV_STOPPED    → APPROACH         (AGV 정지는 가까이 점검)
      LIQUID_LEAK    → APPROACH         (액체 누출은 가까이 점검)
    """
    p = prompt.lower()
    if "heat source" in p:
        return {"action": "KEEP_DISTANCE",
                "reasoning": "Fire risk - keep safe distance (fixed rule).",
                "severity": "HIGH", "source": "RULE"}
    if "smoke" in p:
        return {"action": "WAIT_AND_OBSERVE",
                "reasoning": "Smoke detected - observe in place (fixed rule).",
                "severity": "MEDIUM", "source": "RULE"}
    if "stopped agv" in p:
        return {"action": "APPROACH",
                "reasoning": "Stopped AGV - approach for direct inspection (fixed rule).",
                "severity": "LOW", "source": "RULE"}
    if "liquid leak" in p:
        return {"action": "APPROACH",
                "reasoning": "Liquid leak - approach for close inspection (fixed rule).",
                "severity": "MEDIUM", "source": "RULE"}
    return {"action": "WAIT_AND_OBSERVE",
            "reasoning": "Unknown anomaly - default observe (fixed rule).",
            "severity": "LOW", "source": "RULE"}




# 제어 파라미터
WP_REACH_DIST  = 1.0   # 마커 크기와 일치
INSPECT_TIME   = 5.0
SPOT_SPEED     = 2.8   # 학습 범위 vx max=3.0 내, 거의 한계까지
SPOT_TURN_GAIN = 3.0   # 더 민첩하게 방향 전환


# ═════════════════════════════════════════════════════════════════════════════
#  시나리오 시각화
# ═════════════════════════════════════════════════════════════════════════════

def hide_overhead_prims(root_path: str = "/World/Factory"):
    """천장, 조명 메쉬, 기둥, 보, 벽 등 부감 시점을 가리는 prim 숨김.
       단, 실제 광원(UsdLux Light)은 보존하여 조명 유지."""
    from pxr import Usd, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[ENV] hide_overhead_prims: {root_path} not found")
        return

    keywords = ["ceiling", "roof", "lamp", "beam", "pillar", "truss",
                "rafter", "girder", "bracket", "slot", "window"]
    hidden = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        # 실제 광원은 절대 숨기지 않음
        if "Light" in prim.GetTypeName():
            continue
        if any(kw in name for kw in keywords):
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
            except Exception:
                pass
    print(f"[ENV] Hidden {hidden} overhead/wall prims")


def deactivate_prims(root_path: str, keywords: list):
    """특정 키워드 prim 완전 비활성화 (visual + collision 모두).
       hide_overhead_prims는 visual만 끄지만 collision은 남음.
       이 함수는 SetActive(False)로 prim 자체를 비활성화하여 충돌도 제거."""
    from pxr import Usd
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    deactivated = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if any(kw in name for kw in keywords):
            prim.SetActive(False)
            deactivated += 1
    print(f"[ENV] Deactivated {deactivated} prims (keywords: {keywords})")


def inspect_warehouse_prims(root_path: str = "/World/Factory"):
    """USD 내부의 shelf/rack/wall 등 주요 prim 위치를 콘솔에 출력.
       이 출력을 참고하여 ZONE 좌표를 실제 배치에 맞게 조정."""
    from pxr import Usd, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    keywords = ["shelf", "rack", "pallet", "wall", "floor",
                "ceiling", "aisle"]

    print("\n" + "="*68)
    print("  Warehouse USD Prim Inspection — 좌표 분석")
    print("="*68)

    counts = {}
    samples = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        name_lc = name.lower()
        for kw in keywords:
            if kw in name_lc:
                counts[kw] = counts.get(kw, 0) + 1
                samples.setdefault(kw, [])
                if len(samples[kw]) < 10:
                    try:
                        xform = UsdGeom.Xformable(prim)
                        m = xform.ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default())
                        t = m.ExtractTranslation()
                        samples[kw].append(
                            (name, round(float(t[0]), 2),
                             round(float(t[1]), 2),
                             round(float(t[2]), 2)))
                    except Exception:
                        pass
                break

    for kw in keywords:
        if kw in counts:
            print(f"\n[{kw}] 총 {counts[kw]}개  (처음 {min(10, counts[kw])}개)")
            for s in samples[kw]:
                print(f"  {s[0]:35s}  pos=({s[1]:7.2f}, {s[2]:7.2f}, {s[3]:6.2f})")
    print("\n" + "="*68 + "\n")


def spawn_factory_floor(x_range=(-50, 50), y_range=(-35, 35),
                         tile_size=5.0, z=0.03):
    """공장 콘크리트 바닥 타일.
       - 광범위 (팩토리 안팎 전부 커버)
       - 진한 회색 (흰색 묻힘 방지)
       - z=0.03, 두께=0.01 (SPOT 발이 잠기지 않게 얇게)
       - scale 1.0 (타일 사이 간격 없이 빈틈 없게)
       - /World/FloorTiles/ Xform 하위에 그룹핑 (stage 트리 정리)"""
    # 부모 Xform 생성
    from pxr import UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/FloorTiles")

    tile_count = 0
    nx = int((x_range[1] - x_range[0]) / tile_size)
    ny = int((y_range[1] - y_range[0]) / tile_size)

    # 진한 콘크리트 회색
    light_color = np.array([0.30, 0.30, 0.32])
    dark_color  = np.array([0.24, 0.24, 0.26])

    for i in range(nx):
        for j in range(ny):
            x = x_range[0] + (i + 0.5) * tile_size
            y = y_range[0] + (j + 0.5) * tile_size
            c = light_color if (i + j) % 2 == 0 else dark_color

            VisualCuboid(
                prim_path=f"/World/FloorTiles/Tile_{i}_{j}",
                name=f"tile_{i}_{j}",
                position=np.array([x, y, z]),
                scale=np.array([tile_size, tile_size, 0.01]),
                color=c,
            )
            tile_count += 1
    print(f"[FLOOR] {tile_count} tiles under /World/FloorTiles  "
          f"({nx}×{ny}, tile={tile_size}m)")


def create_textured_plane(prim_path: str, position, scale_xy, texture_path: str,
                           rotation_axis: str = None):
    """텍스처 매핑된 양면 plane (billboard).

    Args:
      prim_path: USD prim path
      position : (x, y, z) world position
      scale_xy : (width, height) plane 크기
      texture_path: PNG 파일 절대경로
      rotation_axis: None=수평(xy plane), 'x'=yz로 회전, 'y'=xz로 회전
    """
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    # ── 1) Mesh 생성 (4 정점 + 1 face) ────────────────────────────────────
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    sx, sy = scale_xy[0]/2, scale_xy[1]/2

    if rotation_axis is None:
        # 수평 — xy 평면 (위에서 보임)
        points = [Gf.Vec3f(-sx, -sy, 0), Gf.Vec3f( sx, -sy, 0),
                  Gf.Vec3f( sx,  sy, 0), Gf.Vec3f(-sx,  sy, 0)]
    elif rotation_axis == 'x':
        # x축 normal — yz 평면 (좌우에서 보임)
        points = [Gf.Vec3f(0, -sx, -sy), Gf.Vec3f(0,  sx, -sy),
                  Gf.Vec3f(0,  sx,  sy), Gf.Vec3f(0, -sx,  sy)]
    elif rotation_axis == 'y':
        # y축 normal — xz 평면 (앞뒤에서 보임)
        points = [Gf.Vec3f(-sx, 0, -sy), Gf.Vec3f( sx, 0, -sy),
                  Gf.Vec3f( sx, 0,  sy), Gf.Vec3f(-sx, 0,  sy)]

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)   # 뒤에서도 보이도록

    # UV (texture coordinates)
    uvs = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        interpolation=UsdGeom.Tokens.faceVarying,
    )
    uvs.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
             Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    # Position
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*position))

    # ── 2) Material + Texture ────────────────────────────────────────────
    mat_path = f"{prim_path}_Mat"
    material = UsdShade.Material.Define(stage, mat_path)

    # UsdPreviewSurface (PBR)
    surface = UsdShade.Shader.Define(stage, f"{mat_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface.CreateInput("metallic",  Sdf.ValueTypeNames.Float).Set(0.0)

    # Texture sampler
    texture = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture_path))

    # UV reader
    st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    # Connect: UV → Texture → Surface
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface")

    # Bind material to mesh
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)


def despawn_anomaly_markers(anomalies: list):
    """기존 anomaly 시각 객체 (zone, pole, icon) 모두 삭제 — 다음 lap 시 재스폰용."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for a in anomalies:
        for prefix in ("AnomalyZone", "AnomalyPole", "AnomalyIcon"):
            path = f"/World/{prefix}_{a['id']}"
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)


def spawn_anomaly_markers(anomalies: list, world):
    """이상 객체 시각화:
       - 바닥 감지반경 원판: 빨간색 (severity score에 따라 진하기 차등)
         · 옅은 핑크빨강 = 낮은 위험 (APPROACH 적합)
         · 진한 빨강     = 높은 위험 (KEEP_DISTANCE 적합)
         · 처리 완료 시 → 연두색으로 자동 변경
       - 얇은 회색 수직 선: anomaly 위치 마커
       - 꼭대기 type 아이콘: 텍스처 매핑된 plane 1장 (xz 평면)

    Type별 아이콘 파일 (ICON_DIR):
      FIRE_RISK        → icon_fire.png
      LIQUID_LEAK      → icon_leak.png
      AGV_STOPPED      → icon_agv.png
      SMOKE_DETECTED   → icon_smoke.png
    """
    POLE_GRAY = np.array([0.35, 0.35, 0.38])

    for a in anomalies:
        x, y = a["position"]
        icon_color = np.array(a["color"])

        # 1) 바닥 감지반경 원판 — severity score에 따라 빨강 진하기 조절
        #    낮은 위험 → 옅은 핑크빨강 (APPROACH 후보)
        #    높은 위험 → 진한 빨강    (KEEP_DISTANCE 후보)
        zone_color = severity_to_red(a["severity_score"])
        VisualCylinder(
            prim_path=f"/World/AnomalyZone_{a['id']}",
            name=f"zone_{a['id']}",
            position=np.array([x, y, 0.05]),
            radius=a["radius"],
            height=0.02,
            color=zone_color,
        )
        # 명시적으로 opacity=1.0 설정 (USD displayOpacity)
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(f"/World/AnomalyZone_{a['id']}")
            if prim.IsValid():
                UsdGeom.Gprim(prim).CreateDisplayOpacityAttr().Set([1.0])
        except Exception:
            pass

        # 2) 얇은 수직 선 — 폭 0.06m, 높이 2.2m
        VisualCuboid(
            prim_path=f"/World/AnomalyPole_{a['id']}",
            name=f"pole_{a['id']}",
            position=np.array([x, y, 1.1]),
            scale=np.array([0.06, 0.06, 2.2]),
            color=POLE_GRAY,
        )

        # 3) 꼭대기 아이콘 — 텍스처 매핑된 plane 1장 (xz 평면)
        icon_file = ICON_FILES.get(a["type"])
        tex_path  = os.path.join(ICON_DIR, icon_file) if icon_file else None

        if tex_path and os.path.exists(tex_path):
            tex_path_usd = tex_path.replace("\\", "/")
            create_textured_plane(
                prim_path=f"/World/AnomalyIcon_{a['id']}",
                position=(x, y, 2.55),
                scale_xy=(1.0, 1.0),
                texture_path=tex_path_usd,
                rotation_axis='y',        # xz 평면 (정면을 보고 세워짐)
            )
        else:
            # 파일 없음 → fallback: 색 큐브
            if tex_path:
                print(f"[WARN] icon file not found: {tex_path}")
            VisualCuboid(
                prim_path=f"/World/AnomalyIcon_{a['id']}",
                name=f"icon_{a['id']}",
                position=np.array([x, y, 2.45]),
                scale=np.array([0.5, 0.5, 0.5]),
                color=icon_color,
            )

        print(f"[ANOMALY] {a['zone']:7s} → {a['type']:15s} at ({x:5.1f},{y:5.1f})")


def spawn_waypoint_markers():
    """순찰 waypoint를 작은 초록 디스크로 생성 (존재하되 invisible)."""
    from pxr import UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    for i, (x, y) in enumerate(PATROL_WAYPOINTS):
        VisualCuboid(
            prim_path=f"/World/Waypoint_{i}",
            name=f"wp_{i}",
            position=np.array([x, y, 0.06]),
            scale=np.array([2.0, 2.0, 0.05]),
            color=np.array([0.0, 0.85, 0.65]),
        )
        # invisible 처리 (logic은 유지, 시각만 숨김)
        prim = stage.GetPrimAtPath(f"/World/Waypoint_{i}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


# ═════════════════════════════════════════════════════════════════════════════
#  Action Executor (선택된 action을 실제 SPOT 행동으로 변환)
# ═════════════════════════════════════════════════════════════════════════════

class ActionExecutor:
    """LLM이 선택한 action을 sub-state machine으로 수행.

    모든 action은 동일한 구조:
      INIT → POSITIONING (radius 확보)
           → ARCING (arc_neg → arc_pos)
           → WAITING (arc_pos 위치에서 3초 정지)
           → DONE
    Action별로 radius와 arc 범위만 다름.

    update(dt)는 매 step마다 호출되며 (vx, vy, wz) command를 반환.
    is_done()이 True가 되면 patrol controller가 PATROLLING 상태로 복귀.
    """

    # Action별 sensing 파라미터 (radius, arc 범위)
    ACTION_PARAMS = {
        "APPROACH":         {"radius": 1.5, "arc_neg_deg": -45, "arc_pos_deg":  90},
        "WAIT_AND_OBSERVE": {"radius": 2.5, "arc_neg_deg": -30, "arc_pos_deg":  60},
        "KEEP_DISTANCE":    {"radius": 4.0, "arc_neg_deg": -60, "arc_pos_deg": 120},
    }
    DIST_TOLERANCE = 0.3
    ARC_SPEED      = 1.2     # rad/s
    POST_WAIT_TIME = 3.0     # ARCING 후 중앙 복귀 + 정지 대기 시간 (초)

    def __init__(self, action: str, anomaly: dict, spot_ctrl):
        self.action     = action
        self.anomaly    = anomaly
        self.spot_ctrl  = spot_ctrl
        self.phase      = "INIT"
        self.arc_angle  = 0.0

    def is_done(self) -> bool:
        return self.phase == "DONE"

    def update(self, dt: float) -> np.ndarray:
        """모든 action이 동일한 arc-observation 로직 사용."""
        if self.action not in self.ACTION_PARAMS:
            # 안전 fallback
            self.phase = "DONE"
            return np.array([0.0, 0.0, 0.0])
        return self._arc_observe(dt)

    def _arc_observe(self, dt: float) -> np.ndarray:
        """공통 로직:
           POSITIONING: 목표 radius로 거리 확보 (뒷걸음/직진, yaw는 anomaly 향함)
           ARCING     : arc_neg → arc_pos (음수 끝 → 양수 끝)
           WAITING    : arc_pos 위치에서 3초 정지 (대응 대기, yaw만 anomaly 응시)
        """
        params = self.ACTION_PARAMS[self.action]
        radius  = params["radius"]
        arc_neg = np.deg2rad(params["arc_neg_deg"])
        arc_pos = np.deg2rad(params["arc_pos_deg"])

        pos, yaw = self.spot_ctrl.get_pose()
        ap   = np.array(self.anomaly["position"])
        diff = ap - pos
        dist = float(np.linalg.norm(diff))

        # ── 0) 첫 진입 로그 ─────────────────────────────────────────────────
        if self.phase == "INIT":
            print(f"  [ACTION] {self.action} → radius {radius}m "
                  f"(arc {params['arc_neg_deg']}° ~ +{params['arc_pos_deg']}°)")
            self.phase = "POSITIONING"

        # ── 1) POSITIONING: 목표 radius까지 거리 확보 ──────────────────────
        if self.phase == "POSITIONING":
            tgt_yaw = float(np.arctan2(diff[1], diff[0]))
            yaw_err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(3.0 * yaw_err, -2.0, 2.0))

            # yaw 정렬 부족하면 먼저 제자리 회전
            if abs(yaw_err) > 0.4:
                return np.array([0.0, 0.0, wz])

            if dist < radius - self.DIST_TOLERANCE:
                return np.array([-1.5, 0.0, wz])    # 뒷걸음
            if dist > radius + self.DIST_TOLERANCE:
                return np.array([ 1.5, 0.0, wz])    # 정면 직진

            # 거리 OK → arcing 시작
            self.arc_start      = float(np.arctan2(-diff[1], -diff[0]))
            self.arc_angle      = self.arc_start
            self.arc_target_neg = self.arc_start + arc_neg
            self.arc_target_pos = self.arc_start + arc_pos
            self.arc_sub        = "TO_NEG"
            print(f"  [ACTION] {self.action} → 거리 {dist:.1f}m 확보, oscillation 시작")
            self.phase = "ARCING"
            return np.array([0.0, 0.0, 0.0])

        # ── 2) ARCING: arc_neg → arc_pos → 그 자리에서 WAITING ─────────────
        if self.phase == "ARCING":
            # 호 각도 진행
            if self.arc_sub == "TO_NEG":
                self.arc_angle -= dt * self.ARC_SPEED
                if self.arc_angle <= self.arc_target_neg:
                    self.arc_sub = "TO_POS"
            elif self.arc_sub == "TO_POS":
                self.arc_angle += dt * self.ARC_SPEED
                if self.arc_angle >= self.arc_target_pos:
                    print(f"  [ACTION] {self.action} → arc_pos 도달, 대처 대기")
                    self.phase = "WAITING"
                    self.wait_timer = 0.0
                    return np.array([0.0, 0.0, 0.0])

            # 호 위 목표점 (anomaly 중심에서 정확히 radius)
            next_pt = np.array([
                ap[0] + radius * np.cos(self.arc_angle),
                ap[1] + radius * np.sin(self.arc_angle),
            ])

            # 거리 보정 (매 스텝마다 radius와의 오차 보상)
            dist_err   = radius - dist
            radial_dir = -diff / (dist + 1e-6)
            move_dir   = (next_pt - pos) + radial_dir * dist_err * 1.5
            move_norm  = float(np.linalg.norm(move_dir))

            # yaw 보정 (정면 = anomaly)
            tgt_yaw = float(np.arctan2(diff[1], diff[0]))
            yaw_err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(3.0 * yaw_err, -2.0, 2.0))

            if move_norm < 1e-3:
                return np.array([0.0, 0.0, wz])

            move_dir /= move_norm
            spot_forward = np.array([np.cos(yaw), np.sin(yaw)])
            spot_left    = np.array([-np.sin(yaw), np.cos(yaw)])

            # 학습 범위 한계: vx [-2,3], vy [-1,1]
            vx = float(np.clip(np.dot(move_dir, spot_forward) * 2.0, -1.5, 1.5))
            vy = float(np.clip(np.dot(move_dir, spot_left)    * 2.5, -1.0, 1.0))
            return np.array([vx, vy, wz])

        # ── 3) WAITING: 중앙 복귀 후 대처 대기 (3초) ──────────────────────
        if self.phase == "WAITING":
            # yaw만 anomaly 정면 유지 (위치는 정지)
            tgt_yaw = float(np.arctan2(diff[1], diff[0]))
            yaw_err = (tgt_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(3.0 * yaw_err, -2.0, 2.0))

            self.wait_timer += dt
            if self.wait_timer >= self.POST_WAIT_TIME:
                print(f"  [ACTION] {self.action} 완료")
                self.phase = "DONE"
                return np.array([0.0, 0.0, 0.0])

            return np.array([0.0, 0.0, wz])

        return np.array([0.0, 0.0, 0.0])




# ═════════════════════════════════════════════════════════════════════════════
#  SPOT 순찰 컨트롤러 (상태머신)
# ═════════════════════════════════════════════════════════════════════════════

class SpotPatrolController:
    """
    PATROLLING : waypoint 추종, 이상 감지 시 INSPECTING으로 전이
    INSPECTING : 정지 후 INSPECT_TIME초 대기 (실제 점검 시뮬레이션)
                 완료 후 해당 이상 객체를 processed에 추가, PATROLLING 복귀
    """

    def __init__(self, spot: SpotFlatTerrainPolicy, anomalies: list,
                 llm_client=None):
        self.spot           = spot
        self.anomalies      = anomalies
        self.llm_client     = llm_client
        self.wp_idx         = 0
        self.current_lap    = 1               # 현재 lap (1부터 시작)
        # 상태머신: PATROLLING → LLM_QUERY → EXECUTING → PATROLLING
        self.state          = "PATROLLING"
        self.current        = None
        self.executor       = None
        self.processed      = set()
        # metrics 수집: 각 anomaly별 decision 기록
        self.decision_log   = []

    # ─ 위치/방향 조회 ────────────────────────────────────────────────────────
    def get_pose(self):
        pos, q = self.spot.robot.get_world_pose()
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw = float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
        return np.array([float(pos[0]), float(pos[1])]), yaw

    # ─ 새 lap 시작 시 anomaly 교체 + 처리 상태 초기화 ────────────────────────
    def reset_for_new_lap(self, new_anomalies: list):
        """새 lap 시작: anomaly 리스트 교체 + processed 초기화.
           wp_idx, decision_log는 유지 (전체 lap 통계용)."""
        self.anomalies = new_anomalies
        self.processed = set()
        self.current   = None
        self.executor  = None
        self.state     = "PATROLLING"

    # ─ 목표 방향 명령 계산 ────────────────────────────────────────────────────
    def compute_command(self, target_xy):
        pos, yaw = self.get_pose()
        diff = np.array(target_xy) - pos
        dist = float(np.linalg.norm(diff))

        if dist < WP_REACH_DIST:
            return np.array([0.0, 0.0, 0.0]), True

        tgt_yaw = float(np.arctan2(diff[1], diff[0]))
        err     = (tgt_yaw - yaw + np.pi) % (2*np.pi) - np.pi
        wz      = float(np.clip(SPOT_TURN_GAIN * err, -2.0, 2.0))
        vx      = float(SPOT_SPEED * max(0.0, 1.0 - abs(err)/1.2))
        return np.array([vx, 0.0, wz]), False

    # ─ 이상 감지 ─────────────────────────────────────────────────────────────
    def detect_anomaly(self):
        if self.state != "PATROLLING":
            return None
        pos, _ = self.get_pose()
        for a in self.anomalies:
            if a["id"] in self.processed:
                continue
            d = float(np.linalg.norm(pos - np.array(a["position"])))
            if d < a["radius"]:
                return a
        return None

    # ─ 감지 시 의사결정 호출 + ActionExecutor 생성 ─────────────────────────
    def on_detect(self, anomaly):
        # 의사결정 호출 (LLM 또는 RULE)
        import time
        t0 = time.time()
        decision = query_llm(self.llm_client, anomaly["prompt"])  # llm_client는 LLMConfig 또는 None
        decision_time = time.time() - t0

        # 모드별 다른 출력 템플릿
        if decision["source"] == "LLM":
            self._print_llm_decision(anomaly, decision, decision_time)
        else:
            self._print_rule_decision(anomaly, decision, decision_time)

        # metrics 수집
        self.decision_log.append({
            "lap":        self.current_lap,
            "anomaly_id": anomaly["id"],
            "zone":       anomaly["zone"],
            "zone_name":  anomaly["zone_name"],
            "type":       anomaly["type"],
            "severity_score": anomaly["severity_score"],
            "attributes": dict(anomaly["attributes"]),
            "prompt":     anomaly["prompt"],
            "action":     decision["action"],
            "severity":   decision["severity"],
            "reasoning":  decision["reasoning"],
            "source":     decision["source"],
            "decision_time_ms": round(decision_time * 1000, 1),
        })

        self.current  = anomaly
        self.executor = ActionExecutor(decision["action"], anomaly, self)
        self.state    = "EXECUTING"

    def _print_llm_decision(self, anomaly, decision, t_sec):
        """LLM 모드 출력 — prompt + LLM 응답 전체."""
        print("\n" + "━"*68)
        print(f"  🚨  ANOMALY DETECTED  —  {anomaly['id']}")
        print(f"      Zone     : {anomaly['zone']} ({anomaly['zone_name']})")
        print(f"      Type     : {anomaly['type']}")
        print(f"      Attributes: {anomaly['attributes']}")
        print()
        print(f"  [LLM INPUT — natural language prompt]")
        print(f"  {anomaly['prompt']}")
        print()
        print(f"  [LLM OUTPUT]  ({t_sec*1000:.0f}ms API)")
        print(f"    action    : {decision['action']}")
        print(f"    severity  : {decision['severity']}")
        print(f"    reasoning : {decision['reasoning']}")
        print("━"*68)

    def _print_rule_decision(self, anomaly, decision, t_sec):
        """Rule-based 모드 출력 — attribute 직접 표시 + 적용 규칙."""
        print("\n" + "─"*68)
        print(f"  [ANOMALY]  {anomaly['id']}")
        print(f"    Zone       : {anomaly['zone']} ({anomaly['zone_name']})")
        print(f"    Type       : {anomaly['type']}")
        print(f"    Attributes : {anomaly['attributes']}")
        print(f"    Severity   : {anomaly['severity_score']:.2f}  (rule-computed)")
        print(f"  [RULE DECISION]  ({t_sec*1000:.0f}ms, fixed type→action mapping)")
        print(f"    action  → {decision['action']}")
        print(f"    rule    : {decision['reasoning']}")
        print("─"*68)

    # ─ Action 완료 ──────────────────────────────────────────────────────────
    def on_execution_done(self):
        a = self.current
        print(f"[SPOT] Action 완료: {a['id']} → patrol 재개\n")

        # 처리 완료된 zone의 색을 연두색으로 변경 (시각적 피드백)
        self._mark_zone_processed(a)

        self.processed.add(a["id"])
        self.current  = None
        self.executor = None
        self.state    = "PATROLLING"

    def _mark_zone_processed(self, anomaly):
        """처리 완료된 anomaly의 바닥 원판 색을 빨강 → 청록(waypoint 색)으로 변경."""
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim_path = f"/World/AnomalyZone_{anomaly['id']}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                gprim = UsdGeom.Gprim(prim)
                # waypoint 마커와 동일한 청록색
                gprim.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.85, 0.65)])
        except Exception as e:
            print(f"[WARN] zone 색상 변경 실패: {e}")

    # ─ 매 스텝 업데이트 ──────────────────────────────────────────────────────
    def update(self, dt):
        if self.state == "PATROLLING":
            detected = self.detect_anomaly()
            if detected is not None:
                self.on_detect(detected)
                # on_detect 직후 첫 스텝은 정지
                return np.array([0.0, 0.0, 0.0])

            target = PATROL_WAYPOINTS[self.wp_idx]
            cmd, reached = self.compute_command(target)
            if reached:
                self.wp_idx = (self.wp_idx + 1) % len(PATROL_WAYPOINTS)
            return cmd

        elif self.state == "EXECUTING":
            cmd = self.executor.update(dt)
            if self.executor.is_done():
                self.on_execution_done()
            return cmd

        return np.array([0.0, 0.0, 0.0])


# ═════════════════════════════════════════════════════════════════════════════
#  Metrics 분석 + 출력
# ═════════════════════════════════════════════════════════════════════════════

def print_metrics(decision_log: list, mode: str):
    """모든 anomaly 처리 후 결과 지표 출력 + JSON/CSV 파일 저장.

    지표:
      [1] Case Study Table
      [2] Context Sensitivity (같은 type, 다른 action 비율)
      [3] Severity Distribution (LOW/MEDIUM/HIGH)
      [4] Severity-Action Consistency (LOW→APPROACH 등 매핑 일관성)
      [5] Reasoning Quality (attribute 인용 비율)
      [6] Decision Time
    """
    if not decision_log:
        print("\n[METRICS] no decisions recorded.")
        return

    from collections import defaultdict

    print("\n" + "="*78)
    print(f"  METRICS REPORT  —  mode: {mode}  —  {len(decision_log)} decisions")
    print("="*78)

    # ── [1] Case Study Table ──────────────────────────────────────────────
    print("\n[1] Case Study — per-anomaly decisions")
    print("-"*78)
    print(f"  {'Zone':<8} {'Type':<18} {'Score':<7} {'Action':<18} {'Sev':<7} {'Src':<6}")
    print("-"*78)
    for d in decision_log:
        print(f"  {d['zone']:<8} {d['type']:<18} "
              f"{d['severity_score']:<7.2f} "
              f"{d['action']:<18} {d['severity']:<7} {d['source']:<6}")

    # ── [2] Context Sensitivity ───────────────────────────────────────────
    type_to_actions = defaultdict(list)
    for d in decision_log:
        type_to_actions[d["type"]].append(d["action"])

    print(f"\n[2] Context Sensitivity (same type → different action?)")
    cs_scores = []
    for t, actions in type_to_actions.items():
        if len(actions) <= 1:
            continue
        unique = len(set(actions))
        score = (unique - 1) / (len(actions) - 1) * 100
        cs_scores.append(score)
        print(f"      {t:<18} → {len(actions)} times, "
              f"{unique} unique actions  ({score:.0f}%)")
    if cs_scores:
        print(f"      Avg sensitivity     : {sum(cs_scores)/len(cs_scores):.1f}%")
    else:
        print(f"      (need ≥2 same-type cases; run more seeds)")

    # ── [3] Severity Distribution ─────────────────────────────────────────
    sev_counts = defaultdict(int)
    for d in decision_log:
        sev_counts[d["severity"]] += 1
    print(f"\n[3] Severity Distribution")
    for sev in ["LOW", "MEDIUM", "HIGH"]:
        n = sev_counts[sev]
        bar = "█" * n
        print(f"      {sev:<8} : {n}  {bar}")

    # ── [4] Severity-Action Consistency ───────────────────────────────────
    # Expected: LOW→APPROACH, MEDIUM→WAIT_AND_OBSERVE, HIGH→KEEP_DISTANCE
    expected_map = {
        "LOW":    "APPROACH",
        "MEDIUM": "WAIT_AND_OBSERVE",
        "HIGH":   "KEEP_DISTANCE",
    }
    consistent = 0
    mismatches = []
    for d in decision_log:
        exp = expected_map.get(d["severity"])
        if exp == d["action"]:
            consistent += 1
        else:
            mismatches.append(
                f"{d['zone']}: severity={d['severity']} but action={d['action']}")
    total = len(decision_log)
    print(f"\n[4] Severity-Action Consistency "
          f"(LOW→APPROACH / MED→WAIT / HIGH→KEEP)")
    print(f"      Consistent : {consistent}/{total}  "
          f"({consistent/total*100:.0f}%)")
    if mismatches:
        print(f"      Mismatches :")
        for m in mismatches:
            print(f"        {m}")

    # ── [5] Reasoning Quality ─────────────────────────────────────────────
    mention_count = 0
    for d in decision_log:
        for v in d["attributes"].values():
            if str(v).lower() in d["reasoning"].lower():
                mention_count += 1
                break
    print(f"\n[5] Reasoning Quality")
    print(f"      Reasoning mentions ≥1 attribute: "
          f"{mention_count}/{total}  ({mention_count/total*100:.0f}%)")

    # ── [6] Decision Time ─────────────────────────────────────────────────
    times = [d["decision_time_ms"] for d in decision_log]
    print(f"\n[6] Decision Time")
    print(f"      avg / min / max : "
          f"{sum(times)/len(times):.0f} / {min(times):.0f} / {max(times):.0f}  ms")

    # ── 파일 저장 (JSON + CSV) ────────────────────────────────────────────
    import datetime
    import os
    OUTPUT_DIR = str(_REPO_ROOT / "results")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUTPUT_DIR, f"metrics_{mode}_{ts}")

    # JSON: 전체 raw (중첩 attributes, prompt, reasoning 포함)
    try:
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(decision_log, f, ensure_ascii=False, indent=2)
        print(f"\n[METRICS] JSON saved → {base}.json")
    except Exception as e:
        print(f"\n[METRICS] JSON save failed: {e}")

    # CSV: 평탄화 — attributes는 별도 컬럼화 (pandas/matplotlib용)
    try:
        import csv
        # 모든 attribute key 수집 (행마다 다를 수 있음)
        attr_keys = sorted({k for d in decision_log for k in d["attributes"].keys()})

        fieldnames = [
            "lap", "anomaly_id", "zone", "zone_name", "type",
            "severity_score", "action", "severity",
            "source", "decision_time_ms", "reasoning",
        ] + [f"attr_{k}" for k in attr_keys]

        with open(f"{base}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in decision_log:
                row = {
                    "lap":              d.get("lap", 1),
                    "anomaly_id":       d["anomaly_id"],
                    "zone":             d["zone"],
                    "zone_name":        d["zone_name"],
                    "type":             d["type"],
                    "severity_score":   round(d["severity_score"], 3),
                    "action":           d["action"],
                    "severity":         d["severity"],
                    "source":           d["source"],
                    "decision_time_ms": d["decision_time_ms"],
                    "reasoning":        d["reasoning"],
                }
                for k in attr_keys:
                    row[f"attr_{k}"] = d["attributes"].get(k, "")
                writer.writerow(row)
        print(f"[METRICS] CSV saved → {base}.csv  ({len(decision_log)} rows)")
    except Exception as e:
        print(f"[METRICS] CSV save failed: {e}")

    print("="*78 + "\n")



# ═════════════════════════════════════════════════════════════════════════════
#  Viewport 영상 녹화
#  --record 옵션 사용 시 활성화.
#  매 render 후 PNG 캡처 → 종료 시 ffmpeg로 MP4 합성.
#  ffmpeg 없으면 PNG 시퀀스만 보존 후 수동 변환 명령 안내.
# ═════════════════════════════════════════════════════════════════════════════

class FrameRecorder:
    """Isaac Sim viewport를 PNG 시퀀스로 캡처 → ffmpeg로 MP4 합성."""

    def __init__(self, base_dir: str = "./recordings", fps: int = 20):
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, ts)
        self.frames_dir  = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.fps = fps
        self.frame_idx = 0
        self._capture_fn = None
        self._viewport   = None
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            self._viewport   = get_active_viewport()
            self._capture_fn = capture_viewport_to_file
            print(f"[REC] Recording enabled → {self.session_dir}  ({fps} FPS)")
        except Exception as e:
            print(f"[REC] WARNING: viewport API 불가 ({e}) — 녹화 비활성")

    def capture(self):
        if self._capture_fn is None:
            return
        path = os.path.join(self.frames_dir, f"frame_{self.frame_idx:06d}.png")
        try:
            self._capture_fn(self._viewport, file_path=path)
            self.frame_idx += 1
        except Exception as e:
            if self.frame_idx == 0:
                print(f"[REC] WARNING: capture failed ({e})")

    def finalize(self):
        if self.frame_idx == 0:
            print("[REC] No frames captured."); return
        out = os.path.join(self.session_dir, "simulation.mp4")
        pattern = os.path.join(self.frames_dir, "frame_%06d.png")
        import subprocess, shutil
        try:
            subprocess.run(["ffmpeg", "-y", "-framerate", str(self.fps),
                            "-i", pattern, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-loglevel", "error", out],
                           check=True)
            print(f"[REC] Video saved → {out}  ({self.frame_idx} frames)")
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        except FileNotFoundError:
            print(f"[REC] ffmpeg not found. PNG preserved: {self.frames_dir}")
            print(f"      Convert: ffmpeg -framerate {self.fps} -i \"{pattern}\" \"{out}\"")
        except subprocess.CalledProcessError as e:
            print(f"[REC] ffmpeg error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════

def main():
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
    )

    # ── 공장 환경 ─────────────────────────────────────────────────────────────
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=ENV_USD, prim_path="/World/Factory")
    print(f"[ENV] warehouse_multiple_shelves loaded")

    # 천장/조명메쉬/기둥/보/슬롯/브래킷 숨기기 (visual만 끔, collision 유지)
    hide_overhead_prims("/World/Factory")

    # PalletBin 완전 비활성화 (visual + collision 모두) — patrol path 위에 있어서 충돌함
    deactivate_prims("/World/Factory", ["palletbin"])

    # USD 내부 prim 좌표 분석 (콘솔 출력 → 디버깅용)
    inspect_warehouse_prims("/World/Factory")

    # 체크무늬 바닥 타일
    spawn_factory_floor()

    # 카메라: 전체 공장 부감 시점
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([0.0, -22.0, 30.0]),
                    target=np.array([0.0, 4.0, 0.0]))

    # ── 시나리오 마커 ─────────────────────────────────────────────────────────
    # 매 실행마다 ZONE-Type 매칭 + attributes 랜덤 생성
    # --seed N으로 재현 가능, 미지정 시 매번 다른 결과
    anomalies = generate_anomalies(seed=args.seed)

    spawn_waypoint_markers()
    spawn_anomaly_markers(anomalies, world)

    # ── SPOT ──────────────────────────────────────────────────────────────────
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/SPOT",
        name="spot",
        usd_path=SPOT_USD,
        position=np.array([0.0, -7.0, 0.75]),  # 창고 안쪽 가운데 (외벽/ZONE 모두 안전)
    )
    spot.load_policy(POLICY_PATH, ENV_YAML)

    world.reset()

    # 안정화
    spot.initialize()
    spot.post_reset()
    spot.robot.set_joint_positions(spot.default_pos)
    spot.robot.set_joint_velocities(spot.default_vel)
    for _ in range(200):
        world.step(render=False)

    # 의사결정 모드 — --use-llm 옵션에 따라 분기
    llm_client = init_llm_client(alias=args.use_llm)
    decision_mode = f"LLM-{args.use_llm}" if llm_client else "RULE"

    patrol = SpotPatrolController(spot, anomalies, llm_client)

    if llm_client:
        mode_label = (f"LLM via {llm_client.provider.upper()} "
                      f"(alias: {args.use_llm}, model: {llm_client.model_id})")
    else:
        mode_label = "Rule-based (fixed type→action mapping)"
    print("\n" + "="*68)
    print("  SPOT Factory Inspection  —  Phase 3")
    print(f"  Zones        : {len(ZONES)}")
    print(f"  Anomalies    : {len(anomalies)} (random type + attrs, seed={args.seed})")
    print(f"  Waypoints    : {len(PATROL_WAYPOINTS)}")
    print(f"  Decision     : {mode_label}")
    print(f"  Action set   : {', '.join(ACTIONS)}")
    print(f"  Camera       : {'Follow SPOT' if args.follow_cam else 'Top-down (fixed)'}")
    print(f"  Recording    : {'ON (' + str(args.record_fps) + ' FPS)' if args.record else 'OFF'}")
    print(f"  Speed mult.  : ×{args.speed}")
    print(f"  Total laps   : {args.laps}  (anomalies re-spawn each lap)")
    print(f"  Metrics mode : "
          f"{'ON (auto-exit after all laps)' if args.metrics else 'OFF'}")
    print(f"  Initial      : SPOT at (0.0, -7.0) → patrol 시작")
    print(f"  Lap 1 anomaly assignment:")
    for a in anomalies:
        print(f"    {a['zone']:7s} → {a['type']:18s} "
              f"severity={a['severity_score']:.2f}  attrs={a['attributes']}")
    print("="*68 + "\n")

    # follow-cam용 import
    if args.follow_cam:
        from isaacsim.core.utils.viewports import set_camera_view as _set_cam

    # 녹화기 초기화
    recorder = None
    if args.record:
        recorder = FrameRecorder(base_dir="./recordings", fps=args.record_fps)

    step = 0
    all_processed_announced = False
    lap_completed_announced = False
    home_return_announced   = False
    prev_wp_idx = patrol.wp_idx
    visited_wps = set()
    SPOT_HOME   = np.array([0.0, -7.0])
    HOME_RADIUS = 1.5

    print(f"\n[SIM] Starting Lap 1 / {args.laps}\n")

    while simulation_app.is_running():
        # ── 한 frame에 args.speed번 physics step → SPOT이 args.speed배 빠르게 ──
        for sub in range(args.speed):
            cmd = patrol.update(PHYSICS_DT)
            spot.forward(dt=PHYSICS_DT, command=cmd)
            world.step(render=(sub == args.speed - 1))
            step += 1

        # ── 영상 녹화 (render 직후 1회) ───────────────────────────────────
        if recorder is not None:
            recorder.capture()

        # ── waypoint 한 바퀴 완주 감지 ──
        if patrol.state == "PATROLLING":
            visited_wps.add(patrol.wp_idx)
            if (patrol.wp_idx != prev_wp_idx
                    and patrol.wp_idx == 0
                    and len(visited_wps) >= len(PATROL_WAYPOINTS)):
                if not lap_completed_announced:
                    print(f"\n[SIM] Lap {patrol.current_lap} waypoints complete "
                          f"({len(PATROL_WAYPOINTS)} visited). "
                          f"Returning to start...\n")
                    lap_completed_announced = True
        prev_wp_idx = patrol.wp_idx

        # ── SPOT 추적 카메라 (옵션) ──
        if args.follow_cam:
            pos, yaw = patrol.get_pose()
            backward = np.array([np.cos(yaw + np.pi), np.sin(yaw + np.pi)])
            eye = (float(pos[0] + backward[0] * 4.0),
                   float(pos[1] + backward[1] * 7.0),
                   8.0)
            target = (float(pos[0]), float(pos[1]), 2)
            _set_cam(eye=np.array(eye), target=np.array(target))

        # ── 진행 로그 ──
        if step % 500 == 0:
            pos, _ = patrol.get_pose()
            extra = ""
            if patrol.state == "EXECUTING" and patrol.executor:
                extra = f"  action={patrol.executor.action}/{patrol.executor.phase}"
            #print(f"[t={step*PHYSICS_DT:6.1f}s]  "
            #      f"lap={patrol.current_lap}/{args.laps}  "
            #      f"SPOT=({pos[0]:5.1f},{pos[1]:5.1f})  "
            #      f"state={patrol.state:11s}  "
            #      f"wp={patrol.wp_idx}/{len(PATROL_WAYPOINTS)}  "
            #      f"processed={len(patrol.processed)}/{len(anomalies)}{extra}")

        # ── 모든 anomaly 처리 완료 (정보 로그만) ──
        if (len(patrol.processed) == len(anomalies)
                and patrol.state == "PATROLLING"
                and not all_processed_announced):
            print(f"\n[SIM] Lap {patrol.current_lap}: all {len(anomalies)} anomalies "
                  f"processed (continuing patrol until lap complete).\n")
            all_processed_announced = True

        # ── Lap 종료: 한 바퀴 완주 + 시작 위치 복귀 ──
        if lap_completed_announced and patrol.state == "PATROLLING":
            pos, _ = patrol.get_pose()
            dist_home = float(np.linalg.norm(pos - SPOT_HOME))
            if dist_home < HOME_RADIUS:
                if not home_return_announced:
                    print(f"\n[SIM] SPOT returned to home ({pos[0]:.1f}, "
                          f"{pos[1]:.1f}).  Lap {patrol.current_lap} complete.\n")
                    home_return_announced = True

                # 마지막 lap이면 → 종료
                if patrol.current_lap >= args.laps:
                    print(f"[SIM] All {args.laps} laps completed.\n")
                    if args.metrics:
                        print_metrics(patrol.decision_log, mode=decision_mode)
                        print("[SIM] --metrics mode → metrics saved.\n")
                    print("[SIM] Auto-exit.\n")
                    break   # metrics 옵션과 무관하게 종료
                else:
                    # 다음 lap 시작 — anomaly 재스폰
                    next_lap = patrol.current_lap + 1
                    print(f"[SIM] Re-spawning anomalies for Lap {next_lap}...")

                    # 이전 anomaly 시각객체 제거
                    despawn_anomaly_markers(anomalies)

                    # 새 anomaly 생성 (seed가 있으면 lap별 다른 결과를 위해 변형)
                    new_seed = (args.seed + next_lap) if args.seed is not None else None
                    anomalies = generate_anomalies(seed=new_seed)
                    spawn_anomaly_markers(anomalies, world)

                    # controller 상태 갱신
                    patrol.current_lap = next_lap
                    patrol.reset_for_new_lap(anomalies)

                    # lap 추적 변수 초기화
                    visited_wps             = set()
                    lap_completed_announced = False
                    home_return_announced   = False
                    all_processed_announced = False

                    print(f"\n[SIM] Starting Lap {next_lap} / {args.laps}\n")
                    for a in anomalies:
                        print(f"    {a['zone']:7s} → {a['type']:18s} "
                              f"severity={a['severity_score']:.2f}")
                    print()

    # 시뮬 루프 종료 후 녹화 마무리
    if recorder is not None:
        recorder.finalize()


if __name__ == "__main__":
    main()
    simulation_app.close()
