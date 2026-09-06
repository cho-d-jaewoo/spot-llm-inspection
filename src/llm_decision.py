"""2계층(Tier1+Tier2+Escalation) LLM 의사결정 인터페이스 — Anthropic/OpenAI/Ollama 통합 호출."""

import json, os, time
from anomaly_config import LEGAL_PRINCIPLES

# 별칭 → (provider, model_id).  provider: "claude" | "openai" | "local"
MODEL_ALIASES = {
    "haiku":  ("claude", "claude-haiku-4-5"),
    "sonnet": ("claude", "claude-sonnet-4-6"),
    "gpt4o":  ("openai", "gpt-4o-mini"),
    "llama":  ("local",  "llama3.1"),
}

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

TIER1_ACTIONS = ["APPROACH", "WAIT_AND_OBSERVE", "KEEP_DISTANCE"]
TIER2_ACTIONS = ["EMERGENCY", "MAINTENANCE", "CONTINUE", "FALSE_ALARM"]

# ─────────────────────────────────────────────────────────────────────────────
#  Tool / function-calling schema
# ─────────────────────────────────────────────────────────────────────────────
_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "tier1_action": {"type": "string", "enum": TIER1_ACTIONS,
                         "description": "Observation stance to take immediately."},
        "tier2_escalate": {"type": "boolean",
                           "description": "True=escalate Tier2 to human operator, False=decide autonomously."},
        "tier2_action": {"type": "string", "enum": TIER2_ACTIONS,
                         "description": "Resolution action (used only when tier2_escalate=False)."},
        "tier2_confidence": {"type": "number",
                             "description": "Your calibrated probability (0.0-1.0) that tier2_action is the CORRECT resolution. "
                                            "Be honest and well-calibrated: report LOW confidence when the situation is ambiguous, "
                                            "novel/unfamiliar, or high-stakes-and-uncertain; HIGH only when the correct action is clear."},
        "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "reasoning": {"type": "string",
                      "description": "Decision rationale in English, 2-3 sentences."},
    },
    "required": ["tier1_action", "tier2_escalate", "tier2_action", "tier2_confidence", "severity", "reasoning"],
}

INSPECTION_TOOL = {
    "name": "submit_inspection_decision",
    "description": "Submit the two-tier inspection decision for the detected anomaly.",
    "input_schema": _TOOL_PARAMS,
}

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_inspection_decision",
        "description": "Submit the two-tier inspection decision for the detected anomaly.",
        "parameters": _TOOL_PARAMS,
    },
}

SYSTEM_PROMPT = """You are an intelligent inspection decision system for a SPOT quadruped robot patrolling a smart factory.

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
- Set True when: Tier 2 decision involves legal reporting obligation OR irreversible safety consequences (wrong call = injury, major damage, or legal liability), OR you are genuinely uncertain about the correct resolution.
- Set False when: Tier 2 is routine (maintenance request automatable via MES, or simple monitoring/false-alarm handling) AND you are confident.

**Confidence (tier2_confidence)**:
- Report your honest, well-calibrated probability that tier2_action is correct. This drives how/when a human is asked.
- A human operator's time is limited: high confidence should mean you are reliably correct, low confidence should flag cases you would likely get wrong. Do NOT be overconfident.

## Key Principle
Consider ALL attribute interactions, not just the anomaly type.
The same anomaly type can require different decisions depending on context.
You may encounter anomaly types not seen before — reason from the legal principles below.

__LEGAL_PRINCIPLES__

You MUST respond by invoking the submit_inspection_decision tool."""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("__LEGAL_PRINCIPLES__", LEGAL_PRINCIPLES)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM 클라이언트
# ─────────────────────────────────────────────────────────────────────────────
class LLMDecisionClient:
    """Anthropic / OpenAI / Ollama 를 동일 인터페이스(decide)로 호출."""

    def __init__(self, model: str):
        if model not in MODEL_ALIASES:
            raise ValueError(f"Unknown model '{model}'. Available: {', '.join(MODEL_ALIASES)}")
        self.model = model
        self.provider, self.model_id = MODEL_ALIASES[model]
        self.client = self._init_client()

    def _init_client(self):
        if self.provider == "claude":
            if not ANTHROPIC_API_KEY:
                raise RuntimeError(f"ANTHROPIC_API_KEY 없음 → cannot use '{self.model}'")
            import anthropic
            print(f"[LLM] Anthropic '{self.model}' ({self.model_id})")
            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError(f"OPENAI_API_KEY 없음 → cannot use '{self.model}'")
            import openai
            print(f"[LLM] OpenAI '{self.model}' ({self.model_id})")
            return openai.OpenAI(api_key=OPENAI_API_KEY)
        if self.provider == "local":
            import openai
            print(f"[LLM] Local Ollama '{self.model}' ({self.model_id}) @ {OLLAMA_BASE_URL}")
            return openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        raise ValueError(f"Unknown provider '{self.provider}'")

    def decide(self, prompt: str) -> tuple[dict, float]:
        """반환: (decision_dict, elapsed_ms).
           decision_dict = {tier1_action, tier2_escalate, tier2_action, severity, reasoning}."""
        t0 = time.time()
        if self.provider == "claude":
            decision = self._decide_claude(prompt)
        else:
            decision = self._decide_openai_compat(prompt)
        elapsed_ms = (time.time() - t0) * 1000.0
        return decision, elapsed_ms

    # ── Anthropic tool_use ───────────────────────────────────────────────────
    def _decide_claude(self, prompt: str) -> dict:
        resp = self.client.messages.create(
            model=self.model_id, max_tokens=600, system=SYSTEM_PROMPT,
            tools=[INSPECTION_TOOL],
            tool_choice={"type": "tool", "name": "submit_inspection_decision"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return _normalize(block.input)
        raise RuntimeError(f"[LLM] no tool_use block: {resp.content}")

    # ── OpenAI / Ollama function-calling ─────────────────────────────────────
    def _decide_openai_compat(self, prompt: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                tools=[OPENAI_TOOL],
                tool_choice={"type": "function", "function": {"name": "submit_inspection_decision"}},
                max_completion_tokens=600,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                return _normalize(json.loads(msg.tool_calls[0].function.arguments))
        except Exception as e:
            if self.provider == "local":
                return self._decide_local_json(prompt, str(e))
            raise RuntimeError(f"[LLM] {self.provider} call failed: {e}")
        if self.provider == "local":
            return self._decide_local_json(prompt, "no tool_calls")
        raise RuntimeError(f"[LLM] no tool_calls (finish={resp.choices[0].finish_reason})")

    # ── Ollama JSON fallback (function calling 미지원 모델 대비) ──────────────
    def _decide_local_json(self, prompt: str, reason: str) -> dict:
        json_prompt = (
            prompt + "\n\nRespond ONLY with a JSON object (no markdown, no explanation):\n"
            '{"tier1_action": "APPROACH|WAIT_AND_OBSERVE|KEEP_DISTANCE", '
            '"tier2_escalate": true|false, '
            '"tier2_action": "EMERGENCY|MAINTENANCE|CONTINUE|FALSE_ALARM", '
            '"severity": "LOW|MEDIUM|HIGH", "reasoning": "..."}'
        )
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": json_prompt}],
            max_completion_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _normalize(json.loads(text))


def _normalize(d: dict) -> dict:
    """LLM 출력 정규화 — 누락/형 오류 방어, enum 보정."""
    tier1 = d.get("tier1_action", "WAIT_AND_OBSERVE")
    if tier1 not in TIER1_ACTIONS: tier1 = "WAIT_AND_OBSERVE"
    tier2 = d.get("tier2_action", "CONTINUE")
    if tier2 not in TIER2_ACTIONS: tier2 = "CONTINUE"
    esc = d.get("tier2_escalate", False)
    if isinstance(esc, str): esc = esc.strip().lower() in ("true", "1", "yes")
    sev = str(d.get("severity", "MEDIUM")).upper()
    if sev not in ("LOW", "MEDIUM", "HIGH"): sev = "MEDIUM"
    try:
        conf = float(d.get("tier2_confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return {"tier1_action": tier1, "tier2_escalate": bool(esc),
            "tier2_action": tier2, "tier2_confidence": conf, "severity": sev,
            "reasoning": str(d.get("reasoning", ""))}
