"""Tier 2 strategy enum — concrete sales moves (Phase 1 of strategy-enum-extension).

Per `research-notes/2026-05-03-PROPOSAL-strategy-enum-extension.md`:

  Tier 1 (UNCHANGED): abstract influence primitives (objection_handling /
    reciprocity / authority / etc.) — 11 entries, supervised by Mode 1b's
    existing system prompt. Captures the *psychological mechanism*.

  Tier 2 (THIS MODULE): concrete sales moves — what the agent actually DOES.
    Each move = (name, composes_primitive, parameter_schema, when_to_use,
    when_NOT_to_use, execution_template). Selected by Mode 1b when there's
    a concrete decision to make; OPTIONAL on simple turns.

Phase 1 ships the 10-move cross-tenant base set. Per-tenant catalogs (Libra
extras, Heavys extras, HoneyBook extras) are Phase 2-3.

Design note: each move's `execution_template` is a string fragment injected
into prompt_build_answer's user context (via chain_executor.build_context_blocks).
The template tells build_answer LITERALLY what to do with what parameters.
This bypasses the contradiction problem we hit in Day-4 case 3 / session
30986515 — the supervisor's strategy is now self-explaining, no must_not_say
patches needed."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConcreteMove:
    """Tier 2 concrete sales move. Each instance is one menu entry the
    supervisor can pick in `directive.strategy.concrete_move`."""
    name: str
    composes_primitive: str    # which Tier 1 primitive this is the concrete form of
    description: str            # one-line for supervisor's prompt
    parameters: dict[str, str]  # parameter_name → "type — what it means"
    when_to_use: str            # supervisor reads this to decide
    when_not_to_use: str        # negative criterion — when NOT to pick this move
    execution_template: str     # injected into build_answer user prompt


# ── Cross-tenant base set (Phase 1 — 10 moves) ──────────────────────────────

MOVES: list[ConcreteMove] = [

    ConcreteMove(
        name="match_competitor_offer",
        composes_primitive="objection_handling",
        description="Match a price the customer EXPLICITLY disclosed from a competitor.",
        parameters={
            "target_price": "number — the matching price (in tenant's native currency)",
            "magnitude_cap": "number — the maximum drop from agent's prior best you're willing to authorize (sanity bound)",
            "rationale": "string — one sentence: why matching is the right call here",
        },
        when_to_use=(
            "Customer disclosed a SPECIFIC competitor price in the last 1-3 turns "
            "AND coverage parity is established or probed. The disclosure must be "
            "concrete — 'they offered me 2200' not 'their price is lower'."
        ),
        when_not_to_use=(
            "Customer hasn't disclosed a number; or the disclosed number is wildly "
            "outside our authorization (more than magnitude_cap below prior best); "
            "or coverage hasn't been clarified (use apples_to_apples_probe first)."
        ),
        execution_template=(
            "Make the price match concrete and immediate. State the new price "
            "({target_price}), confirm coverage parity, propose proceeding. "
            "Do NOT preamble with 'let me check what I can do' — the supervisor "
            "has already authorized the match. Reason: {rationale}"
        ),
    ),

    ConcreteMove(
        name="selective_component_reduction",
        composes_primitive="objection_handling",
        description="Reduce ONE pricing component while holding others fixed.",
        parameters={
            "component": "string — which component to drop (e.g., 'third_party', 'comprehensive', 'discount_code')",
            "new_amount": "number — the reduced amount for that component",
            "hold_components": "list[string] — components that must stay at current level",
            "rationale": "string — why this surgical move beats a flat percentage discount",
        },
        when_to_use=(
            "The customer has a specific competitor offer AND the agent's pricing "
            "decomposes into multiple components (insurance: third_party + mandatory; "
            "headphones: bundle + accessories). Allows match without flat percentage drop."
        ),
        when_not_to_use=(
            "Single-component pricing; or competitor offer is undisclosed; or the "
            "drop violates anchor pack max_discount_pct_internal."
        ),
        execution_template=(
            "Reduce {component} to {new_amount}, hold {hold_components} at current "
            "levels. State the new total. Reason: {rationale}"
        ),
    ),

    ConcreteMove(
        name="escalate_for_match_authorization",
        composes_primitive="authority",
        description="Ask the customer for proof, frame approval as a special request.",
        parameters={
            "request_artifact": "string — what to ask for ('screenshot', 'policy pdf', 'quote email')",
            "implied_outcome": "string — what we'll do with it ('request special approval', 'check with underwriting')",
            "rationale": "string — why escalation is the right move vs immediate match",
        },
        when_to_use=(
            "Customer claims a significant competitor advantage but hasn't provided "
            "evidence; OR the disclosed price is below what we can authorize without "
            "review. Buys time + creates legitimacy frame."
        ),
        when_not_to_use=(
            "Customer's offer is small enough to match directly (use match_competitor_offer); "
            "or customer has already provided evidence; or this would feel like a stall."
        ),
        execution_template=(
            "Ask for {request_artifact} so you can {implied_outcome}. Be specific "
            "about what you need. Reason: {rationale}"
        ),
    ),

    ConcreteMove(
        name="conditional_close_on_delay",
        composes_primitive="commitment",
        description="Respect the delay AND propose a conditional close.",
        parameters={
            "delay_acknowledged": "string — what the customer said they wanted to do",
            "conditional_offer": "string — what we'd give if they say yes ('small additional discount', 'free express shipping')",
            "close_question": "string — the literal question to ask",
        },
        when_to_use=(
            "Customer explicitly requested time / market check / spousal consult / "
            "future contact AND the agent has at least one concession card to play. "
            "The conditional creates a soft commit hook while respecting the delay."
        ),
        when_not_to_use=(
            "Customer explicitly declined; or the customer is firmly objecting; "
            "or no concession card available."
        ),
        execution_template=(
            "Acknowledge {delay_acknowledged}. Then propose a soft conditional close: "
            "'{close_question}' — making the conditional offer ({conditional_offer}) "
            "explicit but contingent on their yes."
        ),
    ),

    ConcreteMove(
        name="apples_to_apples_probe",
        composes_primitive="objection_handling",
        description="Surface coverage / scope dimensions for direct comparison before pricing.",
        parameters={
            "dimensions_to_probe": "list[string] — what to ask about (deductible, warranty, included_coverage, return_policy, scope)",
            "tenant_advantage_anchor": "string — one concrete differentiator to plant before/after the probe",
        },
        when_to_use=(
            "Customer cites a competitor offer without coverage details. Probing "
            "frames the gap as informational, not adversarial."
        ),
        when_not_to_use=(
            "Coverage parity already established; or the customer has clearly "
            "rejected the comparison frame and just wants a price."
        ),
        execution_template=(
            "Ask the customer about {dimensions_to_probe}. Then plant your anchor: "
            "{tenant_advantage_anchor}. Make the probe non-adversarial — frame as "
            "'making sure you're comparing the right things'."
        ),
    ),

    ConcreteMove(
        name="feature_differentiation",
        composes_primitive="authority",
        description="Anchor specific product/service features the competitor doesn't have.",
        parameters={
            "features_to_anchor": "list[string] — concrete features pulled from the anchor pack",
            "competitor_implicit_gaps": "list[string] — what we KNOW the competitor lacks (only if certain)",
        },
        when_to_use=(
            "Customer cites a price gap but coverage / feature details haven't been "
            "discussed. Differentiation reframes the comparison as quality + value, "
            "not just price."
        ),
        when_not_to_use=(
            "Customer has already validated coverage parity (their offer is "
            "apples-to-apples). Then matching is the right move, not differentiation."
        ),
        execution_template=(
            "Cite these features concretely: {features_to_anchor}. If certain about "
            "competitor gaps, mention specifically what they lack: "
            "{competitor_implicit_gaps}. Avoid disparaging — frame as 'unique to us'."
        ),
    ),

    ConcreteMove(
        name="value_stack_with_anchors",
        composes_primitive="reciprocity",
        description="Bundle non-price value (perks, accessories, warranty) instead of dropping price.",
        parameters={
            "items_to_stack": "list[string] — concrete items from the anchor pack ('14-day return', 'free shipping', 'artist shell upgrade')",
            "rationale": "string — why this is more valuable than a price drop",
        },
        when_to_use=(
            "Customer is engaged but price-sensitive AND we're at or near the "
            "max_discount_pct_internal cap. Stacking reframes value without further "
            "price erosion."
        ),
        when_not_to_use=(
            "Customer has explicitly rejected non-price value; or no concrete "
            "stackable items in the anchor pack."
        ),
        execution_template=(
            "Don't drop price further. Stack these instead: {items_to_stack}. "
            "Frame as 'I can't go lower on price but I can add this'. Reason: "
            "{rationale}"
        ),
    ),

    ConcreteMove(
        name="risk_reversal_offer",
        composes_primitive="reciprocity",
        description="Lower the customer's perceived risk via a try-it / refund / guarantee.",
        parameters={
            "mechanism": "string — '14-day return', 'money-back guarantee', 'cancel anytime'",
            "scope": "string — what's covered by the reversal",
        },
        when_to_use=(
            "Customer's hesitation is fit/quality-driven, not price-driven. Risk "
            "reversal flips the decision risk asymmetry."
        ),
        when_not_to_use=(
            "Customer's objection is purely price; or our policy doesn't actually "
            "support the reversal mechanism."
        ),
        execution_template=(
            "Offer the {mechanism} explicitly, scoped to {scope}. Make it the lead "
            "of the message — 'try it without commitment'."
        ),
    ),

    ConcreteMove(
        name="relationship_preserve",
        composes_primitive="empathy",
        description="Soft-retention close — accept the no, leave the door open.",
        parameters={
            "continuity_signal": "string — the literal phrase ('we'll be here', 'reach out anytime')",
            "specific_followup_window": "string — optional ('next renewal', 'in 6 months') if customer signaled timing",
        },
        when_to_use=(
            "Customer has firmly declined AND coverage / pricing options are "
            "exhausted. The 'back off' move §9.1 #4 of the research paper called for."
        ),
        when_not_to_use=(
            "Customer is still engaged or the conversation has options not yet "
            "tried. Premature retreat is worse than continued engagement."
        ),
        execution_template=(
            "Accept the no warmly. Use {continuity_signal}. If known, reference "
            "{specific_followup_window}. Do NOT make another offer — the move is "
            "explicit retreat."
        ),
    ),

    ConcreteMove(
        name="await_customer_return",
        composes_primitive="empathy",
        description="Explicit no-further-outreach commitment — gives the customer space.",
        parameters={
            "followup_window": "string — when (if at all) we'd reach out ('next renewal cycle', 'in 6 months', 'never unless they reach out')",
        },
        when_to_use=(
            "Customer explicitly asked for space / 'I'll get back to you' / 'don't "
            "follow up'. Honoring the request preserves brand trust."
        ),
        when_not_to_use=(
            "Customer is still asking questions or negotiating; or there's a "
            "natural near-term followup the customer would expect."
        ),
        execution_template=(
            "State explicitly that we won't follow up until {followup_window} (or "
            "until they reach out). Make the boundary clear and brief."
        ),
    ),
]


# ── Lookup helpers ──────────────────────────────────────────────────────────

_MOVES_BY_NAME: dict[str, ConcreteMove] = {m.name: m for m in MOVES}


def get_move(name: str) -> ConcreteMove | None:
    """Lookup a move by name. Returns None if name is unknown."""
    return _MOVES_BY_NAME.get(name)


def all_move_names() -> list[str]:
    """List all available move names. Used by Mode 1b's system prompt to
    enumerate the supervisor's choices."""
    return [m.name for m in MOVES]


# ── Supervisor-prompt rendering ─────────────────────────────────────────────

def render_moves_for_supervisor_prompt() -> str:
    """Format the move catalog as a markdown section for Mode 1b's system
    prompt. The supervisor reads this when deciding `strategy.concrete_move`.

    Output is verbose (one move = ~6-8 lines × 10 moves = 60-80 lines of
    prompt text). This is intentional: the supervisor needs to see when_to_use
    and when_not_to_use explicitly to discriminate."""
    lines = [
        "## CONCRETE MOVES (Tier 2 — what to actually DO)",
        "",
        "After picking strategy.primary (Tier 1 abstract primitive), OPTIONALLY pick",
        "a concrete_move from the list below. The concrete_move tells the conversation",
        "agent the LITERAL ACTION to take, with specific parameters.",
        "",
        "Pick a concrete_move when there's a *concrete decision to make* — a price,",
        "a scope, an escalation, a conditional close. Leave it null on simple turns",
        "(small talk, plain commitment confirmation, etc.).",
        "",
        "If you pick one, you MUST populate ALL its parameters with grounded values",
        "(numbers from dialog or anchor pack; strings that quote evidence). Don't",
        "invent parameters — if you can't ground them, don't pick the move.",
        "",
        "Available moves:",
        "",
    ]
    for m in MOVES:
        lines.append(f"### `{m.name}` (composes: {m.composes_primitive})")
        lines.append(f"  {m.description}")
        lines.append(f"  parameters:")
        for pname, pdesc in m.parameters.items():
            lines.append(f"    - {pname}: {pdesc}")
        lines.append(f"  use when: {m.when_to_use}")
        lines.append(f"  do NOT use when: {m.when_not_to_use}")
        lines.append("")
    lines.append(
        "Output the chosen move as `strategy.concrete_move`:"
    )
    lines.append('```json')
    lines.append('{')
    lines.append('  "strategy": {')
    lines.append('    "primary": "objection_handling",')
    lines.append('    "concrete_move": {')
    lines.append('      "name": "match_competitor_offer",')
    lines.append('      "parameters": {')
    lines.append('        "target_price": 2200,')
    lines.append('        "magnitude_cap": 350,')
    lines.append('        "rationale": "customer disclosed Wesure 2200 with same coverage"')
    lines.append('      }')
    lines.append('    },')
    lines.append('    ...')
    lines.append('  }')
    lines.append('}')
    lines.append('```')
    lines.append('')
    lines.append('OR set `concrete_move: null` if no concrete action is needed this turn.')
    return "\n".join(lines)


# ── Build_answer rendering ──────────────────────────────────────────────────

def render_move_for_build_answer(directive_strategy: dict | None) -> str:
    """When the directive includes a concrete_move, format it as an
    "## Execute this concrete move" block for prompt_build_answer's user
    context. Returns empty string if no move was picked.

    Called by chain_executor.build_context_blocks during build_answer's
    context construction."""
    if not directive_strategy:
        return ""
    cm = directive_strategy.get("concrete_move")
    if not isinstance(cm, dict) or not cm.get("name"):
        return ""
    name = cm.get("name")
    params = cm.get("parameters") or {}
    move = get_move(name)
    if move is None:
        # Unknown move name — surface it but don't synthesize
        return (
            f"## Execute this concrete move\n"
            f"Move name: `{name}` (UNKNOWN — supervisor picked an unregistered move; "
            f"fall back to your default judgment)\n"
            f"Parameters: {json.dumps(params, indent=2)}"
        )
    # Render the move's execution_template with the supervisor's parameters
    try:
        rendered = move.execution_template.format(**params)
    except (KeyError, IndexError) as e:
        rendered = (
            f"(template render failed: {e}; raw parameters below)\n"
            f"{json.dumps(params, indent=2)}"
        )
    # 2026-05-05 — When a parameter is a list with multiple items, surface
    # only the FIRST one so the actor doesn't faithfully recite all 4-5 items
    # as bullets in one message. Empirical: real won-conversations have median
    # 12-word agent messages; multi-item bullet lists blow past that 5×.
    def _truncate_list_param(v):
        if isinstance(v, list) and len(v) > 1:
            return f"{v[0]} (one item per turn — save the rest for follow-up turns: {v[1:]})"
        return v
    lines = [
        "## Execute this concrete move",
        f"**Move:** `{move.name}` (composes Tier-1 primitive: {move.composes_primitive})",
        f"**Parameters (supervisor-grounded):**",
    ]
    for k, v in params.items():
        lines.append(f"  - {k}: {_truncate_list_param(v)}")
    lines.append("")
    lines.append(f"**Execution instruction:** {rendered}")
    lines.append("")
    lines.append(
        "This move was chosen by the supervisor with full context. Execute it "
        "directly — don't preamble with 'let me check' or 'I understand'. The "
        "supervisor has already done the strategic reasoning."
    )
    # 2026-05-05 — HARD LENGTH RULE reminder. Without this, multi-parameter
    # moves (apples_to_apples_probe with list dimensions; payment_plan_offer
    # with multiple installment options) spawn paragraph-level messages that
    # violate the 35-word target from real won-conversation data.
    lines.append("")
    lines.append(
        "**HARD LENGTH RULE:** ≤35 words, ≤2 sentences. Real won-deal agents "
        "send 1 short message per dimension (median 12 words, p90 37). If "
        "the move has multiple parameters or list items, address only the "
        "MOST IMPORTANT one this turn — save the others for follow-up "
        "turns. NO bullet lists. NO paragraph-style multi-fact responses. "
        "One thing per message."
    )
    return "\n".join(lines)


# ── Validation (Phase 2 will extend with grounding-check gate) ──────────────

def validate_move_invocation(directive_strategy: dict | None) -> tuple[bool, list[str]]:
    """Lightweight schema validation of a Tier 2 move invocation. Returns
    (is_valid, list_of_issues). Phase 1 only checks: name exists in catalog
    and required parameters are present + non-empty. Phase 2 adds grounding
    validation (e.g., target_price must appear in dialog)."""
    if not directive_strategy:
        return True, []
    cm = directive_strategy.get("concrete_move")
    if cm is None:
        return True, []  # null is valid (move is optional)
    if not isinstance(cm, dict):
        return False, [f"concrete_move must be dict or null, got {type(cm).__name__}"]
    name = cm.get("name")
    if not name:
        return False, ["concrete_move.name is required when concrete_move is non-null"]
    move = get_move(name)
    if move is None:
        return False, [f"unknown concrete_move name: '{name}' (catalog: {all_move_names()})"]
    params = cm.get("parameters") or {}
    issues = []
    for pname in move.parameters:
        if pname not in params:
            issues.append(f"missing parameter: {pname}")
        elif params[pname] in (None, "", []):
            issues.append(f"empty parameter: {pname}")
    return (len(issues) == 0), issues
