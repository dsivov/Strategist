"""Local llama.cpp engine plugin — the agent side runs on a local model.

Two engines against a llama.cpp `llama-server` (or any OpenAI-compatible
endpoint), needing NO cloud key:

  - `llamacpp` — single-call actor: persona + goal framing + anchors become the
    system prompt; the model's reply is the agent's message. The local analog
    of the `baseline` arm.
  - `llamacpp-strategist` — two-stage Strategist-STYLE pipeline: stage 1
    (supervisor) picks a strategy directive as JSON from a fixed palette;
    stage 2 (actor) renders that directive into the customer-facing message.
    The local analog of the directive→render architecture the real (vendor-
    infrastructure-bound) `strategist` arm uses.

Notes for reasoning-tuned models (e.g. Gemma "thinking" builds):
  - `chat_template_kwargs.enable_thinking=false` is sent by default so the
    model answers directly instead of spending the token budget on a thought
    block. Toggle with the `thinking` param (single-call engine only).
  - If the server still returns an empty `content` with a populated
    `reasoning_content` (thinking-only truncation), we salvage the last line
    of the reasoning as a degraded fallback and flag it in `meta`.

The customer simulator on the other side of the conversation is part of the
benchmark harness and still uses its own model.
"""
from __future__ import annotations

import json
import os
import re
import time

DEFAULT_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:8081")
DEFAULT_MAX_TOKENS = 220          # WhatsApp-style short agent messages
_TIMEOUT = 120.0                  # local 12B on CPU/GPU can be slow; be patient

# Strategy palette for the local strategist's stage-1 supervisor. Cialdini's
# six + the practical sales moves the reference arms reason in.
STRATEGIES = (
    "reciprocity", "social_proof", "authority", "scarcity", "commitment",
    "liking", "value_anchor", "price_reframe", "objection_reframe",
    "clarify_needs", "urgency", "close_attempt", "graceful_close",
)
TONES = ("warm", "professional", "playful", "direct", "empathetic")


def _persona_block(opp_meta: dict) -> str:
    bits = []
    for k in ("primary_motivator", "decision_logic", "trust_level",
              "communication_style", "objection_pattern", "primary_resistance",
              "budget_sensitivity", "purchase_urgency"):
        v = opp_meta.get(k)
        if v:
            bits.append(f"- {k.replace('_', ' ')}: {v}")
    return "\n".join(bits)


def _anchor_block(opp_meta: dict) -> str:
    anchors = opp_meta.get("anchors") or {}
    return "\n".join(f"- {k}: {v}" for k, v in anchors.items()
                     if isinstance(v, (int, float, str)))


def _system_prompt(opp_meta: dict, business_rules: str) -> str:
    company = opp_meta.get("company") or "the company"
    opp_type = opp_meta.get("opp_type") or "a sales conversation"
    parts = [
        f"You are a skilled, honest sales agent for {company}, handling: {opp_type}.",
        "You are chatting with a customer in a messaging app. Reply with the "
        "agent's next message ONLY — no preamble, no quotes, no markdown. Keep "
        "it under 35 words, natural and specific to what the customer just said. "
        "Advance the conversation toward a close without being pushy, and never "
        "invent prices or facts beyond the reference data below.",
    ]
    persona = _persona_block(opp_meta)
    anchors = _anchor_block(opp_meta)
    if persona:
        parts.append("Customer profile:\n" + persona)
    if anchors:
        parts.append("Reference data (pricing / limits):\n" + anchors)
    if business_rules:
        parts.append(f"Business rules you must follow:\n{business_rules}")
    return "\n\n".join(parts)


def _dialog_messages(dialog_history: list) -> list[dict]:
    out = []
    for m in dialog_history or []:
        role = "assistant" if m.get("role") == "agent" else "user"
        txt = (m.get("text") or "").strip()
        if txt:
            out.append({"role": role, "content": txt})
    if out and out[-1]["role"] == "assistant":
        # Two agent turns in a row (e.g. cold open): nudge for a follow-up.
        out.append({"role": "user",
                    "content": "(the customer has not replied yet — "
                               "send a natural follow-up message)"})
    return out


async def _chat(base_url: str, model: str, messages: list[dict], *,
                temperature: float, max_tokens: int,
                thinking: bool = False) -> tuple[str, dict]:
    """One chat completion. Returns (text, call_meta); salvages thinking-only
    output and strips leaked thought-channel markup."""
    import httpx

    payload = {
        "model": model or "default",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens if not thinking else max(max_tokens, 2048),
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    t0 = time.time()
    async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
        r = await http.post(f"{base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    salvaged = False
    if not text and reasoning:
        lines = [ln.strip(" *-") for ln in reasoning.splitlines() if ln.strip()]
        text = lines[-1] if lines else ""
        salvaged = True
    text = re.sub(r"<\|?channel\|?>\w*", "", text).strip()
    call_meta = {
        "model": data.get("model") or model,
        "latency_ms": int((time.time() - t0) * 1000),
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "salvaged_from_reasoning": salvaged,
    }
    return text, call_meta


class LlamaCppEngine:
    """Single-call `produce()` engine backed by an OpenAI-compatible server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "",
                 temperature: float = 0.6, thinking: bool = False):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or "default"
        self.temperature = float(temperature)
        self.thinking = bool(thinking)

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        messages = ([{"role": "system",
                      "content": _system_prompt(opp_meta or {}, business_rules or "")}]
                    + _dialog_messages(dialog_history))
        text, call_meta = await _chat(
            self.base_url, self.model, messages,
            temperature=self.temperature, max_tokens=DEFAULT_MAX_TOKENS,
            thinking=self.thinking)
        return text, {"engine": "llamacpp", "base_url": self.base_url,
                      "thinking": self.thinking, **call_meta}


class LlamaCppStrategistEngine:
    """Two-stage local pipeline: supervisor directive (JSON) → actor render.

    Stage 1 sees the persona, anchors, and dialog and picks a strategy from a
    fixed palette, returning {strategy, tone, move, rationale}. Stage 2 renders
    that directive into the ≤35-word customer-facing message. Both stages run
    on the same local server; the directive is recorded verbatim in `meta`
    (same shape of telemetry the reference arms emit: strategy / tone / gates).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "",
                 temperature: float = 0.5):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or "default"
        self.temperature = float(temperature)

    async def _stage1_directive(self, opp_meta, dialog_history) -> tuple[dict, dict]:
        company = opp_meta.get("company") or "the company"
        opp_type = opp_meta.get("opp_type") or "a sales conversation"
        transcript = "\n".join(
            f"{m.get('role', '?').upper()}: {m.get('text', '')}"
            for m in (dialog_history or [])[-12:] if (m.get("text") or "").strip())
        sys_prompt = (
            f"You are a sales strategy supervisor for {company} ({opp_type}). "
            "Given the customer profile and the conversation so far, choose the "
            "single best next strategy for the agent.\n\n"
            f"Customer profile:\n{_persona_block(opp_meta) or '- unknown'}\n\n"
            f"Reference data:\n{_anchor_block(opp_meta) or '- none'}\n\n"
            f"Allowed strategies: {', '.join(STRATEGIES)}\n"
            f"Allowed tones: {', '.join(TONES)}\n\n"
            "Answer with ONLY a JSON object, no other text:\n"
            '{"strategy": "...", "tone": "...", '
            '"move": "one concrete instruction for the agent, <=20 words", '
            '"rationale": "<=15 words"}'
        )
        raw, call_meta = await _chat(
            self.base_url, self.model,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": f"Conversation so far:\n{transcript or '(agent opens)'}"}],
            temperature=0.3, max_tokens=200)
        directive = None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                directive = json.loads(m.group(0))
            except Exception:
                directive = None
        if not isinstance(directive, dict):
            directive = {"strategy": "clarify_needs", "tone": "warm",
                         "move": "ask what matters most to the customer",
                         "rationale": "fallback: unparseable directive",
                         "_parse_failed": True}
        if directive.get("strategy") not in STRATEGIES:
            directive["_off_palette"] = directive.get("strategy")
            directive["strategy"] = "clarify_needs"
        return directive, call_meta

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        opp_meta = opp_meta or {}
        t0 = time.time()
        directive, sup_meta = await self._stage1_directive(opp_meta, dialog_history)

        render_sys = (
            _system_prompt(opp_meta, business_rules or "")
            + "\n\nYour supervisor's directive for THIS reply — follow it:\n"
            + f"- strategy: {directive.get('strategy')}\n"
            + f"- tone: {directive.get('tone')}\n"
            + f"- move: {directive.get('move')}"
        )
        messages = ([{"role": "system", "content": render_sys}]
                    + _dialog_messages(dialog_history))
        text, act_meta = await _chat(
            self.base_url, self.model, messages,
            temperature=self.temperature, max_tokens=DEFAULT_MAX_TOKENS)

        meta = {
            "engine": "llamacpp-strategist",
            "base_url": self.base_url,
            "model": act_meta.get("model"),
            "strategy": directive.get("strategy"),
            "tone": directive.get("tone"),
            "directive": directive,
            "supervisor_latency_ms": sup_meta.get("latency_ms"),
            "actor_latency_ms": act_meta.get("latency_ms"),
            "latency_ms": int((time.time() - t0) * 1000),
            "salvaged_from_reasoning": (sup_meta.get("salvaged_from_reasoning")
                                        or act_meta.get("salvaged_from_reasoning")),
        }
        return text, meta


class LlamaCppPlannerEngine:
    """SOP-graph planner on a local model — the Planner architecture, all-local.

    Loads the bundled won-deal SOP graph for the scenario's (tenant, opp_type)
    (poc/planner/data/sop_graph/*.json). Stage 1: the local model estimates the
    customer's current state (graph node) and picks the next agent action from
    the edges allowed out of that state. Stage 2: renders the action into the
    customer-facing message. Directive + allowed-action set go into `meta`.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "",
                 temperature: float = 0.5):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or "default"
        self.temperature = float(temperature)
        self._graphs: dict = {}

    def _graph(self, tenant: str, opp_type: str) -> dict | None:
        key = (tenant, opp_type)
        if key not in self._graphs:
            g = None
            try:
                import glob
                import poc.planner as _pl
                base = os.path.join(os.path.dirname(_pl.__file__), "data", "sop_graph")
                slug = re.sub(r"[^a-z0-9]+", "_", (opp_type or "").lower()).strip("_")
                cands = (glob.glob(os.path.join(base, f"{tenant}__{slug}*.json"))
                         or glob.glob(os.path.join(base, f"{tenant}__*.json")))
                if cands:
                    with open(cands[0]) as f:
                        g = json.load(f)
            except Exception:
                g = None
            self._graphs[key] = g
        return self._graphs[key]

    @staticmethod
    def _allowed(g: dict, state: str) -> list[str]:
        """Agent actions reachable from a user state per the SOP edges."""
        acts = set()
        for e in g.get("edges", []):
            d = e.get("dir")
            if e.get("src") == state and d in ("fwd", "bi") and e.get("dst") in g["agent_actions"]:
                acts.add(e["dst"])
            if e.get("dst") == state and d in ("back", "bi") and e.get("src") in g["agent_actions"]:
                acts.add(e["src"])
        return sorted(acts) or list(g.get("agent_actions") or [])

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        opp_meta = opp_meta or {}
        t0 = time.time()
        g = self._graph(opp_meta.get("company") or "", opp_meta.get("opp_type") or "")
        states = (g or {}).get("user_states") or ["engaged", "price_objection", "near_close"]
        transcript = "\n".join(
            f"{m.get('role', '?').upper()}: {m.get('text', '')}"
            for m in (dialog_history or [])[-12:] if (m.get("text") or "").strip())

        sys1 = (
            "You are a sales planner using a state graph mined from won deals.\n"
            f"Customer states: {', '.join(states)}\n\n"
            "First identify the customer's CURRENT state from the transcript, "
            "then pick the next agent action.\n"
            "Answer with ONLY JSON: {\"state\": \"...\", \"action\": \"...\", "
            "\"focus\": \"one concrete instruction, <=15 words\"}"
        )
        raw, sup_meta = await _chat(
            self.base_url, self.model,
            [{"role": "system", "content": sys1},
             {"role": "user", "content": f"Transcript:\n{transcript or '(agent opens)'}"}],
            temperature=0.3, max_tokens=160)
        plan = None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                plan = json.loads(m.group(0))
            except Exception:
                plan = None
        if not isinstance(plan, dict):
            plan = {"state": "engaged", "action": "information",
                    "focus": "answer helpfully and move forward", "_parse_failed": True}
        state = plan.get("state") if plan.get("state") in states else "engaged"
        allowed = self._allowed(g, state) if g else []
        action = plan.get("action")
        if allowed and action not in allowed:
            plan["_off_graph"] = action
            action = allowed[0]

        render_sys = (
            _system_prompt(opp_meta, business_rules or "")
            + "\n\nPlanner directive for THIS reply — follow it:\n"
            + f"- customer state: {state}\n"
            + f"- action to take: {action}\n"
            + f"- focus: {plan.get('focus')}"
        )
        text, act_meta = await _chat(
            self.base_url, self.model,
            [{"role": "system", "content": render_sys}] + _dialog_messages(dialog_history),
            temperature=self.temperature, max_tokens=DEFAULT_MAX_TOKENS)

        return text, {
            "engine": "llamacpp-planner",
            "model": act_meta.get("model"),
            "state": state, "action": action, "plan": plan,
            "allowed_actions": allowed,
            "sop_graph": bool(g),
            "supervisor_latency_ms": sup_meta.get("latency_ms"),
            "actor_latency_ms": act_meta.get("latency_ms"),
            "latency_ms": int((time.time() - t0) * 1000),
        }


def get_engines():
    try:
        from poc import EngineSpec, ParamSpec
    except Exception:  # host put the package dir (not its parent) on sys.path
        from registry import EngineSpec, ParamSpec  # type: ignore

    common_params = (
        ParamSpec(name="base_url", label="Server URL", type="string",
                  default=DEFAULT_BASE_URL,
                  help="llama-server address (OpenAI-compatible)."),
        ParamSpec(name="model", label="Model", type="string", default="",
                  help="Model name; llama.cpp serves its loaded model "
                       "regardless, so usually leave empty."),
    )
    return [
        EngineSpec(
            id="llamacpp",
            name="Local LLM (llama.cpp)",
            description="Agent runs on a local llama.cpp server (OpenAI-compatible "
                        "API) — e.g. Gemma. No cloud key needed for this panel.",
            runnable=True,
            params=common_params + (
                ParamSpec(name="temperature", label="Temperature", type="string",
                          default="0.6", help="Sampling temperature."),
                ParamSpec(name="thinking", label="Thinking", type="bool",
                          default=False,
                          help="Let reasoning-tuned models think before replying "
                               "(slower; needs a large token budget)."),
            ),
            factory=lambda base_url=DEFAULT_BASE_URL, model="", temperature="0.6",
                           thinking=False: LlamaCppEngine(
                base_url=base_url, model=model,
                temperature=float(temperature or 0.6),
                thinking=(thinking in (True, "true", "1", "on", "yes"))),
        ),
        EngineSpec(
            id="llamacpp-planner",
            name="Local Planner (llama.cpp)",
            description="SOP-graph planner on a local llama.cpp server: state "
                        "estimate + action pick constrained by the bundled "
                        "won-deal graph, then local render. No cloud key needed.",
            runnable=True,
            params=common_params + (
                ParamSpec(name="temperature", label="Actor temp", type="string",
                          default="0.5", help="Sampling temperature for the "
                                              "render stage (planner runs at 0.3)."),
            ),
            factory=lambda base_url=DEFAULT_BASE_URL, model="",
                           temperature="0.5": LlamaCppPlannerEngine(
                base_url=base_url, model=model,
                temperature=float(temperature or 0.5)),
        ),
        EngineSpec(
            id="llamacpp-strategist",
            name="Local Strategist (llama.cpp)",
            description="Two-stage local pipeline on a llama.cpp server: a "
                        "supervisor call picks a strategy directive (Cialdini "
                        "palette), an actor call renders it. Local analog of "
                        "the Strategist architecture — no cloud key needed.",
            runnable=True,
            params=common_params + (
                ParamSpec(name="temperature", label="Actor temp", type="string",
                          default="0.5", help="Sampling temperature for the "
                                              "render stage (supervisor runs at 0.3)."),
            ),
            factory=lambda base_url=DEFAULT_BASE_URL, model="",
                           temperature="0.5": LlamaCppStrategistEngine(
                base_url=base_url, model=model,
                temperature=float(temperature or 0.5)),
        ),
    ]
