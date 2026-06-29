"""CoT+SOP online planner — PCA §5.3.

One LLM call per turn: infer the current user_state, then choose the next
agent_action constrained by the SOP graph's valid successors (controllability)
with an explicit proactive escape hatch (PCA proactivity). Emits a directive;
the shared actor renders it (single-variable invariant — actor is harness
substrate, not Strategist code).

v1 SOP guidance = children(inferred_user_state) (the current-node constraint).
Full edit-distance-over-subpaths (PCA §5.3 exact) is a v1.1 refinement and is
intentionally deferred.

Independent: imports only planner.sop + anthropic + stdlib.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import sop

_ROOT = Path(__file__).resolve().parent.parent
_env_loaded = False


def _load_env() -> None:
    global _env_loaded
    if _env_loaded:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:
        pass
    _env_loaded = True


def _last_customer(dialogue: list[dict]) -> str:
    for m in reversed(dialogue or []):
        if m.get("role") == "customer":
            return (m.get("text") or "").strip()
    return ""


def _transcript(dialogue: list[dict], k: int = 10) -> str:
    out = []
    for m in (dialogue or [])[-k:]:
        r = "Customer" if m.get("role") == "customer" else "Agent"
        t = (m.get("text") or "").replace("\n", " ").strip()[:240]
        if t:
            out.append(f"{r}: {t}")
    return "\n".join(out) or "(no turns yet)"


def _sop_adjacency_block(g: dict) -> str:
    lines = []
    for s in g.get("user_states", []):
        ch = sop.children(g, s)
        if ch:
            lines.append(f"  {s} → {', '.join(ch)}")
    return "\n".join(lines)


def _anchor_brief(opp_meta: dict) -> str:
    # Replayer T-81 enrichment stashes the pack on opp_meta["anchors"]
    # (replayer.py:2091). Read that; keep "_anchors" as a defensive alias.
    a = ((opp_meta or {}).get("anchors")
         or (opp_meta or {}).get("_anchors") or {})
    if not a:
        return "(no economic anchors)"
    keys = ("last_year_price_usd", "current_quoted_price_usd",
            "market_avg_for_segment_usd", "profile_appropriate_opening_usd",
            "max_discount_pct_internal")
    return "; ".join(f"{k}={a[k]}" for k in keys if a.get(k) is not None) or "(sparse)"


_PRICE_RE = re.compile(r"(\d[\d,]{2,})\s*USD", re.IGNORECASE)
_EARNED_RE = re.compile(
    r"competitor|another (company|quote|offer)|cheaper|got a quote|"
    r"\b\d[\d,]{2,}\b.*?(elsewhere|other)", re.IGNORECASE)

# Difficulty gate: any customer-side price pressure. Deliberately broad and
# evaluated over the WHOLE dialogue incl. the seed/opening, so the gate
# fires on the FIRST signal — never only after a concession has happened.
_PRESSURE_RE = re.compile(
    r"\b(expensive|too (much|high|expensive)|so high|cheaper|lower (it|the price)?"
    r"|discount|reduce|too pricey|can you do better|better price|"
    r"competitor|another (company|quote|offer)|got a quote|"
    r"why (so much|did it (go up|increase))|increase|out of (my )?budget|"
    r"can'?t afford|match (the|that)|beat (the|that))\b", re.IGNORECASE)


def _price_pressure(dialogue: list[dict]) -> bool:
    """True once the customer has exerted any price pressure anywhere in the
    conversation so far (seed included). Monotone: once true it stays true,
    because past turns remain in the dialogue."""
    for m in dialogue or []:
        if m.get("role") != "customer":
            continue
        t = m.get("text") or ""
        if _PRESSURE_RE.search(t):
            return True
        # a bare price number from the customer = an implicit counter/ask
        if _PRICE_RE.search(t):
            return True
    return False


def _envelope_mode(state: dict) -> str:
    """Normalize the envelope mode. Back-compatible: truthy/`always` →
    'always'; 'auto' → difficulty-gated; anything else → 'off'."""
    v = state.get("planner_envelope")
    if v is True:
        return "always"
    s = str(v).strip().lower() if v is not None else "off"
    if s in ("always", "true", "1", "on", "yes"):
        return "always"
    if s == "auto":
        return "auto"
    return "off"


def _negotiation_envelope(state: dict, opp: dict) -> dict | None:
    """Topology-B economic constraints, PCA-native (constraint values fed
    into the prompt — NO Strategist gate code). Floor + cumulative-drop +
    earned-signal expressed as planning constraints; the Planner reasons
    within them instead of being corrected after."""
    a = (opp or {}).get("anchors") or (opp or {}).get("_anchors") or {}
    cur = a.get("current_quoted_price_usd")
    maxd = a.get("max_discount_pct_internal")
    if not cur or maxd is None:
        return None
    floor = round(float(cur) * (1.0 - float(maxd) / 100.0))
    agent_prices, cust_signal = [], False
    for m in state.get("dialogue") or []:
        txt = m.get("text") or ""
        if m.get("role") == "agent":
            agent_prices += [int(x.replace(",", "")) for x in _PRICE_RE.findall(txt)]
        elif m.get("role") == "customer" and _EARNED_RE.search(txt):
            cust_signal = True
    first_offer = max(agent_prices) if agent_prices else int(cur)
    latest = agent_prices[-1] if agent_prices else int(cur)
    drop_pct = round(100.0 * (first_offer - latest) / first_offer, 1) if first_offer else 0.0
    return {
        "active": True, "floor_usd": floor, "max_discount_pct": float(maxd),
        "first_offer_usd": first_offer, "latest_offer_usd": latest,
        "cum_drop_pct": drop_pct, "earned_signal": cust_signal,
    }


def _envelope_block(env: dict) -> str:
    return f"""NEGOTIATION ENVELOPE (hard economic constraints — obey):
- FLOOR: never offer below {env['floor_usd']} USD (max authorized discount \
{env['max_discount_pct']}% off {env['first_offer_usd']}). Below-floor = invalid.
- ANTI-STAIRCASE: you have already moved {env['first_offer_usd']}→\
{env['latest_offer_usd']} ({env['cum_drop_pct']}% total drop). Do NOT make \
repeated small concessions; if you concede it must be one meaningful move.
- ANTI-CAPITULATION: earned_signal={env['earned_signal']}. If false, do NOT \
concede toward the floor — hold the price and probe for a competitor number \
or commitment first. A discount is only justified after an earned signal."""


def _build_prompt(state: dict, g: dict, env_block: str = "",
                   playbook_block: str = "") -> str:
    opp = state.get("opp_meta") or {}
    profile = "; ".join(
        f"{k}={opp.get(k)}" for k in
        ("primary_motivator", "decision_logic", "trust_level",
         "objection_pattern", "purchase_urgency") if opp.get(k))
    return f"""You are the online PLANNER for a {opp.get('company','?')} \
{opp.get('opp_type','renewal')} sales conversation, in the PCA formalism \
(arXiv:2407.03884, §5.3 CoT+SOP).

CUSTOMER PROFILE: {profile or '(unknown)'}
ECONOMIC ANCHORS: {_anchor_brief(opp)}
{env_block}
BUSINESS RULES: {(state.get('business_rules') or '(none)')[:600]}

{playbook_block}

CONVERSATION (last 10 turns):
{_transcript(state.get('dialogue'))}

SOP GRAPH — valid agent_action successors per user_state (controllability \
constraint):
{_sop_adjacency_block(g)}

AGENT ACTIONS: {sop.AGENT_ACTIONS}
USER STATES:   {sop.USER_STATES}

Reason step by step (Chain-of-Thought):
1. INFER the customer's current user_state from their last message + \
trajectory (choose exactly one from USER STATES).
2. From the SOP-valid successors of that user_state, CHOOSE the best next \
agent_action. You MAY choose an action NOT in the SOP successors only if it \
is clearly superior here — set "off_sop": true and justify (PCA proactivity).
3. Decide tone and a concrete must_say / must_not_say for the actor.

Output ONLY a fenced ```json block:
{{"user_state": <one of USER STATES>,
  "agent_action": <one of AGENT ACTIONS>,
  "off_sop": <bool>,
  "tone": "professional|friendly|empathetic|direct",
  "must_say": ["..."],
  "must_not_say": ["..."],
  "rationale": "<=30 words",
  "confidence": <0.0-1.0>}}"""


def plan(state: dict, model: str = "claude-sonnet-4-6") -> dict:
    """Return a directive dict (no agent_text — the shared actor renders)."""
    _load_env()
    opp = state.get("opp_meta") or {}
    tenant = opp.get("company") or "Insurance"
    opp_type = opp.get("opp_type") or "renewal"
    # Resolve to a SOP artifact: exact (tenant, opp_type) → tenant canonical
    # flow → Insurance/renewal global fallback. Ecommerce cart variants ("Abandoned
    # Cart", "Abandoned Cart US No Consent", …) all share one recovery SOP.
    g = sop.load(tenant, opp_type)
    if g is None:
        _default = {"Ecommerce": "abandoned_cart"}.get(tenant, "renewal")
        g = sop.load(tenant, _default) or sop.load("Insurance", "renewal")
    if g is None:
        raise RuntimeError(f"no SOP graph for {tenant}/{opp_type}")

    # Topology B: economic envelope as in-prompt planning constraints.
    #   off    → pure Planner (unchanged control)
    #   always → envelope every turn
    #   auto   → envelope only once price pressure is detected (difficulty
    #            gate; fires on the FIRST signal incl. the seed)
    mode = _envelope_mode(state)
    pressure = _price_pressure(state.get("dialogue") or [])
    env = None
    env_block = ""
    applied = False
    if mode != "off":
        env = _negotiation_envelope(state, opp)
        if env and (mode == "always" or (mode == "auto" and pressure)):
            env_block = _envelope_block(env)
            applied = True

    # Positive-direction mined playbooks — shared substrate (raw YAML data,
    # not Strategist code). Compact block surfacing high-lift agent moves
    # mined from won deals at this tenant; the LLM picks within the SOP graph.
    from . import playbook_reader
    _pbs = playbook_reader.load_tenant_playbooks(tenant)
    playbook_block = playbook_reader.format_for_prompt(_pbs, state)
    _pb_ids = []
    for pb in _pbs[:3]:
        _pb_ids.append(pb.get("script_id") or "?")

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model, max_tokens=700,
        messages=[{"role": "user",
                   "content": _build_prompt(state, g, env_block,
                                            playbook_block)}])
    text = "".join(b.text for b in resp.content
                    if getattr(b, "type", "") == "text")
    raw = text
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    d = json.loads(raw.strip())

    action = d.get("agent_action")
    if action not in sop.AGENT_ACTIONS:
        action = "objection_handling"  # safe SOP-valid default
    user_state = d.get("user_state")
    if user_state not in sop.USER_STATES:
        user_state = "engaged"
    sop_valid = action in sop.children(g, user_state)

    directive = {
        "engine": "planner",
        "strategy": {"primary": action,
                     "tone": d.get("tone") or "professional"},
        "primary_strategy": action,
        "tone": d.get("tone") or "professional",
        "must_say": d.get("must_say") or [],
        "must_not_say": d.get("must_not_say") or [],
        "rationale": d.get("rationale") or "",
        # Conform to the harness directive contract (_summarize_directive +
        # actor): confidence is a dict; rationale also under audit.
        "confidence": {"overall": float(d.get("confidence") or 0.0)},
        "audit": {"rationale_summary": d.get("rationale") or ""},
    }
    arch = {
        "off": "Planner — PCA CoT+SOP (§5.3)",
        "always": "Planner+Envelope[always] — PCA CoT+SOP+econ (§5.3)",
        "auto": f"Planner+Envelope[auto:{'engaged' if applied else 'idle'}] "
                f"— PCA CoT+SOP+econ (§5.3)",
    }[mode]
    meta = {
        "architecture": arch,
        "engine": "planner",
        "tier": "cot_sop_envelope" if applied else "cot_sop",
        "primary_strategy": action,
        "user_state": user_state,
        "sop_valid": sop_valid,
        "off_sop": bool(d.get("off_sop")) or not sop_valid,
        "confidence": directive["confidence"],
        "envelope_mode": mode,
        "envelope_gate_fired": pressure,   # price pressure detected this far
        "envelope_applied": applied,        # constraints actually in-prompt
        "envelope": env if applied else None,
        # positive-direction mined library exposure
        "playbooks_loaded": len(_pbs),
        "playbook_ids": _pb_ids,
        "playbook_block_present": bool(playbook_block),
    }
    return {"directive": directive, "directive_meta": meta}
