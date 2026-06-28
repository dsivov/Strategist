"""T-83 anti-staircase mechanical gate (2026-05-02).

Empirical motivation: in 2026-05-02 sessions on opp a3f517d7, the supervised
agent staircase-priced (2,685 → 2,470 → 2,300 NIS) DESPITE the explicit
ANCHOR-AWARE PRICING RULE in the supervisor's system prompt. Prompt-text
rules don't bind; only mechanical/programmatic gates do. Customer's
empirically-validated reaction: "Wait, you started at 2,685 and now 2,300?
Why didn't you just offer this from the start?" → trust collapse.

This module provides a tenant-aware extractor + a panel-state tracker that
detects when the agent's next message would be a worsening concession from
its prior best offer. Caller (replayer._live_turn) uses the verdict to
reject + regenerate the agent message.

Tenant-aware:
  Libra (NIS, full-price quoting):
    - Failure: agent lowers quoted full premium repeatedly
    - Detect via NIS regex + agent-vs-reference context classifier
    - Reject if new agent-offered price < min(prior agent offers)
  Heavys (USD, discount-code offering):
    - Failure: agent escalates discount magnitude
    - Detect via $X off regex + distinct VIP code count
    - Reject if new offered discount > max(prior offered discounts)

Conservative policy: when ambiguous, default to "not staircase" — false
negatives are safer than false positives (we don't want to reject good
agent text just because regex was uncertain).
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


# ── Libra (NIS) extraction ──────────────────────────────────────────────────

LIBRA_NIS_PATTERN = re.compile(
    # Match either comma-grouped (1,272 / 1,887) or plain (2685 / 1100) numbers.
    # The (?<!\d) lookbehind prevents matching the tail of a longer number.
    r'(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*NIS',
    re.IGNORECASE)
LIBRA_HEBREW_NIS_PATTERN = re.compile(
    r'(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:₪|ש"ח|שח)')

# Markers indicating the price refers to someone OTHER than us (competitor,
# customer's prior policy, market reference). When present in the ±60 char
# context window around a price, classify as 'reference' not 'agent_offer'.
LIBRA_COMPARISON_MARKERS = re.compile(
    r'\b(they|their|another|other|elsewhere|competitor|you got|'
    r'you were quoted|I got|I was quoted|got an offer|received an offer|'
    r'the other|other company|comparing|compare|against|versus|vs\b|'
    r'than (?:what|the))\b',
    re.IGNORECASE)


def extract_libra_offered_prices(text: str) -> list[float]:
    """Return list of agent-offered NIS prices found in `text`.
    Filters out competitor/reference/market prices via context classifier."""
    if not text:
        return []
    offers = []
    # English NIS
    for m in LIBRA_NIS_PATTERN.finditer(text):
        amount = _parse_amount(m.group(1))
        if amount is None or amount < 100 or amount > 100000:
            continue  # plausibility filter
        ctx_start = max(0, m.start() - 60)
        ctx_end = min(len(text), m.end() + 30)
        ctx = text[ctx_start:ctx_end]
        if not LIBRA_COMPARISON_MARKERS.search(ctx):
            offers.append(amount)
    # Hebrew NIS (same context filter applies; Hebrew comparison markers
    # would need their own dictionary — for now just trust agent's bilingual
    # text won't mix reference + offer in same Hebrew clause)
    for m in LIBRA_HEBREW_NIS_PATTERN.finditer(text):
        amount = _parse_amount(m.group(1))
        if amount is None or amount < 100 or amount > 100000:
            continue
        offers.append(amount)
    return offers


# ── Heavys (USD) extraction ─────────────────────────────────────────────────

HEAVYS_DISCOUNT_PATTERN = re.compile(r'\$(\d+(?:\.\d+)?)\s*off', re.IGNORECASE)
HEAVYS_VIP_CODE_PATTERN = re.compile(r'\bVIP(\d+)[A-Z]+\b', re.IGNORECASE)


def extract_heavys_offered_discounts(text: str) -> dict[str, Any]:
    """Return dict with keys:
      'discount_amounts': list of $X off amounts
      'codes': list of unique VIP{N}{SUFFIX} codes (each code may bundle
               its own discount magnitude in N)"""
    if not text:
        return {"discount_amounts": [], "codes": []}
    amounts = []
    for m in HEAVYS_DISCOUNT_PATTERN.finditer(text):
        amount = _parse_amount(m.group(1))
        if amount is not None and 1 <= amount <= 1000:
            amounts.append(amount)
    codes = list(set(m.group(0).upper() for m in HEAVYS_VIP_CODE_PATTERN.finditer(text)))
    return {"discount_amounts": amounts, "codes": codes}


# ── Unified panel-state contract ────────────────────────────────────────────

def extract_customer_disclosed_competitor_price(
    dialog: list[dict] | None, lookback_customer_msgs: int = 3,
) -> float | None:
    """Scan the customer's recent messages for an explicit competitor-quoted
    price. Used to whitelist legitimate competitive matching from the
    anti-staircase gate.

    Patterns matched (Libra NIS, all extracted to NIS amounts):
      - "I got quoted 2200 [from X]"
      - "X has it for 2200"
      - "I'm getting 2200 elsewhere"
      - "another company [quoted/offered] 2200"
      - bare "2,200 NIS" alongside a competitor-context word

    Returns the most recent customer-disclosed competitor price, or None.
    """
    if not dialog:
        return None
    customer_msgs = [m for m in dialog if m.get("role") == "customer"]
    if not customer_msgs:
        return None
    recent = customer_msgs[-lookback_customer_msgs:]

    competitor_context = re.compile(
        r"\b(?:quote[ds]?|offer(?:ed|ing)?|got|getting|elsewhere|"
        r"another\s+(?:company|insurer|insurance|place)|"
        r"competitor|wesure|phoenix|clal|harel|ayalon|menora)\b",
        re.IGNORECASE,
    )
    nis_pattern = re.compile(
        r"(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{3,5}(?:\.\d+)?)\s*"
        r"(?:nis|shekels?|₪)?",
        re.IGNORECASE,
    )

    # Walk most-recent-first; first match wins
    for msg in reversed(recent):
        text = msg.get("text") or ""
        if not competitor_context.search(text):
            continue
        # Find numeric prices in this competitor-context message
        for m in nis_pattern.finditer(text):
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            # Sanity bounds for car-insurance NIS prices
            if 500 <= val <= 15000:
                return val
    return None


# 2026-05-06 — Context-aware cumulative-concession threshold (Option D).
# Empirical finding: 23% of real won-deal Libra agents drop >15% from anchor.
# A static 15% gate over-blocks. Loosen DYNAMICALLY based on conversation
# context: explicit competitor quote → already whitelisted (existing logic);
# late-phase stall → 30%; customer-stated target → match exactly.
DEFAULT_CUMULATIVE_THRESHOLD = 0.15        # baseline (current)
LATE_PHASE_STALL_THRESHOLD = 0.30          # 5+ turns, low engagement
CUSTOMER_TARGET_THRESHOLD = 1.00           # explicit "I'd renew at X" → match exactly

# 2026-05-10 — Critical-violation threshold. When cumulative drop exceeds
# this, the customer simulator (and real customers — see UI manual session
# 2026-05-10 reporting "you started with an inflated baseline and only moved
# when I pushed back") perceives the pattern as overt manipulation. On
# CRITICAL violation + retry exhaustion, the chain runner replaces the
# violating draft with a safe fallback message instead of passing the
# violation through.
CRITICAL_CUMULATIVE_THRESHOLD = 0.25

# Regex: customer states a specific target price they'll close at
# Matches "I'd renew at 2,600", "if you can do 2200", "match 2,500 and I'm in",
# "round to 3000", "anything below 3500", etc.
import re as _re_target
_CUSTOMER_TARGET_RE = _re_target.compile(
    r"\b(?:if you (?:can|could) (?:do|match|hit|drop|get to)|"
    r"(?:i'?d|i would) (?:renew|do it|take it)\s*(?:at|for)|"
    r"match\s+(?:my|that|the)?\s*(?:price|offer|quote)?\s*(?:of|at)?|"
    r"(?:round|drop|do) (?:it )?(?:to|down to|at)|"
    r"\bany\s*(?:thing)?\s*below)\s*(\d[\d,]*)\b",
    _re_target.IGNORECASE,
)


# R35 (2026-05-06) — Heavys customer-target extractor. Parallel to Libra
# extract_customer_stated_target above but tuned for e-commerce $X off / X%
# patterns instead of NIS prices. Returns either a dollar-discount target
# ({"kind": "dollar_off", "amount": 80}) or a percent target
# ({"kind": "percent_off", "amount": 25}). None when no target found.
#
# Empirical motivation (R34 doc 2026-05-06): in 43 won-deal Heavys pushback
# conversations, customer-target reveals were 6 of 43 (~14%); when they
# happened, real agents went ABOVE the stated target ("you can do $42? Would
# about $55 be useful?"), not matched-exactly. So the detected target is
# the FLOOR for the agent's counter-offer, not the exact price to match.
_HEAVYS_DOLLAR_TARGET_RE = _re_target.compile(
    r"\$\s*(\d{1,4}(?:\.\d+)?)\s*(?:off|discount|deal|coupon)\b|"
    r"(?:i'?d|i would) (?:buy|take|order|pay)\s*(?:at|for)\s*\$?\s*(\d{1,5})|"
    r"\bholding out (?:for|until)\s*(?:the\s*)?\$?\s*(\d{1,4})\s*(?:off)?|"
    r"\bwaiting for\s*(?:the\s*)?\$?\s*(\d{1,4})\s*(?:off)?|"
    r"\b(?:match|hit|do)\s+(?:my|that|the)?\s*(?:offer|deal|price|coupon)?\s*(?:of|at)?\s*\$\s*(\d{1,4})",
    _re_target.IGNORECASE,
)
_HEAVYS_PERCENT_TARGET_RE = _re_target.compile(
    r"\b(\d{1,2})\s*%\s*(?:off|discount|deal)\b",
    _re_target.IGNORECASE,
)


def extract_customer_stated_target_heavys(
    dialog: list[dict] | None,
) -> dict | None:
    """For e-commerce (Heavys-family) tenants, scan the customer's recent
    messages for an explicit dollar-off or percent-off target. Returns a
    dict {kind: 'dollar_off' | 'percent_off', amount: int} or None.

    Looks at the LAST 3 customer messages (matches Libra version's window).
    Prefers dollar-off (more specific) over percent-off when both match.
    """
    if not dialog:
        return None
    customer_msgs = [m for m in dialog if m.get("role") == "customer"][-3:]
    for m in customer_msgs:
        text = m.get("text") or ""
        # Try dollar-off first
        for match in _HEAVYS_DOLLAR_TARGET_RE.finditer(text):
            for group in match.groups():
                if group:
                    try:
                        v = int(float(group))
                        if 5 <= v <= 500:  # plausible $ off range
                            return {"kind": "dollar_off", "amount": v}
                    except (ValueError, TypeError):
                        pass
        # Fall back to percent-off
        for match in _HEAVYS_PERCENT_TARGET_RE.finditer(text):
            try:
                v = int(match.group(1))
                if 5 <= v <= 80:  # plausible % off range
                    return {"kind": "percent_off", "amount": v}
            except (ValueError, TypeError):
                pass
    return None


# Phase A (2026-05-10) — generic-numeric extractor for use after a semantic
# classifier has fired. Scans for any plausible NIS-amount in the text.
# Works with Hebrew text too because it doesn't require an English currency
# token next to the number — the classifier already established this is
# target-statement context.
_GENERIC_TARGET_NUMERIC_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{3,5})(?!\d)"
)


def extract_customer_stated_target(
    dialog: list[dict] | None,
    cfg: dict | None = None,
) -> int | None:
    """If the customer's recent messages explicitly state a target price
    they'll close at, return that target. Used to relax the cumulative-
    concession gate so the agent can match the exact stated number —
    legitimate close, not a staircase concession.

    Phase A (2026-05-10): when `cfg` is provided AND defines a
    `classifiers.customer_target` semantic block, use the classifier engine
    to detect target-statement context (paraphrase-robust, multilingual);
    then extract the numeric value with a generic regex. Otherwise falls
    back to the legacy regex `_CUSTOMER_TARGET_RE` (English-only, narrow
    paraphrase coverage).

    Args:
      dialog: list of {role, text} dicts
      cfg:    optional tenant staircase config (see staircase_config.get_config)
    """
    if not dialog:
        return None
    # Look at the LAST 3 customer messages
    customer_msgs = [m for m in dialog if m.get("role") == "customer"][-3:]

    # Phase A — try semantic classifier first when configured
    try:
        from classifier_engine import is_classifier_configured, classifier_match
    except ImportError:
        is_classifier_configured = lambda *args, **kwargs: False  # type: ignore[assignment]
        classifier_match = None  # type: ignore[assignment]

    if cfg and is_classifier_configured("customer_target", cfg):
        # Walk most-recent-first: the latest target-statement wins
        for msg in reversed(customer_msgs):
            text = msg.get("text") or ""
            if not text:
                continue
            result = classifier_match("customer_target", text, cfg)
            if not result.get("match"):
                continue
            # Classifier matched — pull the numeric target.
            # First try the legacy regex (it captures the explicit "I'd renew at X")
            m = _CUSTOMER_TARGET_RE.search(text)
            if m:
                try:
                    v = int(m.group(1).replace(",", ""))
                    if 500 <= v <= 50000:
                        return v
                except (ValueError, TypeError):
                    pass
            # Fallback: any plausible NIS amount in the message — picks the
            # smallest plausible amount (most-likely-target — customers
            # usually state a lower number than the agent's quote).
            candidates: list[int] = []
            for nm in _GENERIC_TARGET_NUMERIC_RE.finditer(text):
                try:
                    v = int(nm.group(1).replace(",", ""))
                    if 500 <= v <= 50000:
                        candidates.append(v)
                except (ValueError, TypeError):
                    continue
            if candidates:
                return min(candidates)
            # Classifier matched but no number extractable — log and continue
            log.info("classifier customer_target matched (score=%s) but no "
                     "numeric extractable in text=%r",
                     result.get("score"), text[:80])
        return None

    # Legacy path: regex-only (preserved for tenants that don't define the
    # classifier block — backward compatible with pre-2026-05-10 behavior)
    for msg in customer_msgs:
        text = msg.get("text") or ""
        match = _CUSTOMER_TARGET_RE.search(text)
        if match:
            try:
                v = int(match.group(1).replace(",", ""))
                if 500 <= v <= 50000:
                    return v
            except (ValueError, TypeError):
                pass
    return None


def detect_late_phase_stall(dialog: list[dict] | None,
                             panel_concessions: list[dict]) -> bool:
    """Return True if conversation is in a late-phase stall: 5+ turns with
    customer engagement low. Looser staircase threshold applies because the
    agent has invested time and a real human salesperson would now drop the
    price more aggressively to close.

    Heuristic: 5+ agent turns AND customer's last 2 messages contain repeated
    objection language (similar lengths, no commitment cues like 'sounds
    good', 'let me think', 'maybe').
    """
    if not dialog:
        return False
    agent_turns = sum(1 for m in dialog if m.get("role") == "agent")
    if agent_turns < 5:
        return False
    customer_msgs = [m for m in dialog if m.get("role") == "customer"][-2:]
    if len(customer_msgs) < 2:
        return False
    objection_kw = ["too high", "too expensive", "not interested", "no thanks",
                    "above my budget", "still high", "won't work", "can't justify",
                    "out of range"]
    customer_text = " ".join((m.get("text") or "").lower() for m in customer_msgs)
    if any(kw in customer_text for kw in objection_kw):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# R33 (2026-05-06) — Manager-escalation framing
#
# Empirical motivation: real-agent analysis (200 won-deal Libra Insurance
# Renewal × Price/Analytical conversations) found that 23% of won deals drop
# >15% from anchor. Production agents have a `max_discount_pct_internal`
# (typically 15%) — that's their DISCRETIONARY budget. For drops beyond that,
# they ESCALATE to a retention manager ("Let me check with my supervisor…
# OK, I got special approval for…"). This makes the larger concession trust-
# preserving instead of staircase-coded.
#
# Our context-aware gate (Option D) ALLOWS up to 30% in late-phase stall,
# but without escalation framing the actor LLM produces silent staircase
# language. The customer then sees: "first you said 2,685, then 2,300, now
# 2,150 — why didn't you offer this from the start?"
#
# This module decides when to inject manager-escalation framing into the
# build_answer prompt context. The framing is CONDITIONAL — actor only
# uses it IF its directive supports a larger-than-typical concession. Idle
# turns don't fire spurious "let me check with manager" messages.
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that indicate the agent has ALREADY used manager-escalation
# language in a prior turn. Used to suppress duplicate escalations within
# the same conversation (one escalation per conversation; subsequent turns
# treat the post-escalation price as final).
_ESCALATION_PHRASES_RE = _re_target.compile(
    r"\b(?:check(?:ing|ed)? with (?:my |the )?(?:retention |sales |line )?(?:manager|supervisor|team lead)|"
    # Approval-verb broadened: real winning sessions used "was able to get approval"
    # and "received special approval from my supervisor" — neither matched
    # past-tense-only "got". 2026-05-06.
    r"(?:got|get|getting|received|obtained|secured|(?:was |been )?able to (?:get|secure)) (?:special )?approval|"
    r"(?:special )?approval (?:from|by) (?:my |the )?(?:manager|supervisor|retention|line)|"
    r"pushed for (?:an |the )?exception|"
    r"one[-\s]time exception|"
    r"(?:beyond|outside) my (?:normal |usual )?authority|"
    r"(?:my |the )?(?:manager|supervisor) (?:approved|authorized|signed off)|"
    r"(?:let me|i'?ll|i will) (?:talk|speak|check) (?:to|with) (?:my )?(?:manager|supervisor))\b",
    _re_target.IGNORECASE,
)


def has_escalated_already(dialog: list[dict] | None) -> bool:
    """Return True if the agent has already used manager-escalation language
    in any prior turn. Once an escalation has been performed, the customer's
    trust model treats the post-escalation price as final — reusing the
    framing would feel manipulative."""
    if not dialog:
        return False
    for m in dialog:
        if m.get("role") != "agent":
            continue
        text = m.get("text") or ""
        if _ESCALATION_PHRASES_RE.search(text):
            return True
    return False


def should_inject_escalation_framing(
    *, dialog: list[dict] | None,
    panel_concessions: list[dict],
    opp_meta: dict | None = None,
) -> tuple[bool, dict]:
    """Decide whether prompt_build_answer should receive manager-escalation
    framing guidance. Returns (active, meta) where meta describes the
    triggering signal for telemetry.

    Active when ALL of:
      - Tenant is Libra (price-quoted tenant; Heavys uses VIP codes via a
        separate mechanic)
      - At least one prior agent price offer exists (something to escalate
        beyond — first-quote turns don't need it)
      - Agent has not ALREADY escalated earlier in this conversation
        (one escalation per conversation; built-in regex detector)
      - Late-phase stall is detected (5+ agent turns with customer in
        repeated objection language) — same trigger as the gate's stall-
        relax — OR the customer has stated an explicit target the agent
        cannot reach within typical authority

    The framing instruction is appended to build_answer's user context. The
    actor LLM evaluates conditionally: if it's about to offer a price drop
    larger than default authority, frame it as supervisor approval.
    """
    meta = {"active": False, "trigger": None, "prior_min": None,
            "stall_active": False, "customer_target": None,
            "already_escalated": False}

    tenant = (opp_meta or {}).get("company") if opp_meta else None
    if not tenant or tenant.lower() != "libra":
        return False, meta

    prior_prices = [c["amount"] for c in (panel_concessions or [])
                     if c.get("type") == "price"]
    if not prior_prices:
        return False, meta
    meta["prior_min"] = min(prior_prices)

    # One escalation per conversation — suppress further injection
    if has_escalated_already(dialog):
        meta["already_escalated"] = True
        return False, meta

    stall_active = detect_late_phase_stall(dialog, panel_concessions)
    meta["stall_active"] = stall_active

    customer_target = extract_customer_stated_target(dialog)
    meta["customer_target"] = customer_target

    if stall_active:
        meta["active"] = True
        meta["trigger"] = "late_phase_stall"
        return True, meta

    # Customer-stated target below the agent's prior min by >15% is also a
    # legitimate escalation moment (need supervisor approval to match the
    # target). Below this threshold, the existing customer-target match
    # whitelist handles it without escalation.
    if customer_target is not None and prior_prices:
        prior_max = max(prior_prices)
        if customer_target < prior_max * (1 - DEFAULT_CUMULATIVE_THRESHOLD):
            meta["active"] = True
            meta["trigger"] = "customer_target_beyond_authority"
            return True, meta

    return False, meta


def render_escalation_framing_block(meta: dict, anchors: dict | None = None) -> str:
    """Render the manager-escalation framing block for build_answer's context.
    The block tells the actor LLM: IF you make a larger-than-default
    concession this turn, FRAME it as supervisor approval, not as a silent
    drop. Below default authority, no framing is needed.
    """
    max_disc = None
    if anchors:
        max_disc = anchors.get("max_discount_pct_internal")
    if max_disc is None:
        max_disc = 15

    trigger = meta.get("trigger") or "stall"
    trigger_phrase = {
        "late_phase_stall": (
            "Conversation is in late-phase stall (5+ turns, customer is "
            "still pushing back on price). Real agents in this state often "
            "escalate to a retention manager for an exception."),
        "customer_target_beyond_authority": (
            f"Customer has stated a specific target price (≈{meta.get('customer_target')} NIS) "
            f"that is beyond your default discretionary authority "
            f"(max_discount={max_disc}%). Real agents call their retention "
            f"manager in this situation."),
    }.get(trigger, "Late-phase concession context.")

    return (
        "## manager_escalation_framing (R33 — empirical: 23% of won deals exceed default authority)\n"
        f"- Context: {trigger_phrase}\n"
        f"- Your default discretionary discount authority is {max_disc}%.\n"
        "- DECISION RULE for this turn:\n"
        "  - If your directive points to a price concession WITHIN default "
        f"authority ({max_disc}% from anchor or shallower): hold price OR "
        "concede normally (no escalation framing needed).\n"
        "  - If your directive points to a price concession BEYOND default "
        "authority: FRAME it as supervisor approval. Do not silently drop.\n"
        "- Acceptable escalation phrasings (English; pick ONE that fits naturally, "
        "do not stack):\n"
        "  - \"Let me check with my retention manager — one moment.\"\n"
        "  - \"I just got special approval from my supervisor on this one.\"\n"
        "  - \"Given how long we've been working on this, I escalated to my "
        "manager and got a one-time exception for you.\"\n"
        "  - \"This is beyond my normal authority, but I pushed for an "
        "exception — here's what I can do.\"\n"
        "- The escalation language must PRECEDE the new price in the same "
        "message. Do NOT use this language without making a meaningful "
        "concession (it would feel manipulative without the follow-through).\n"
        "- Do NOT escalate twice in one conversation — once a manager "
        "exception is granted, treat the price as final."
    )


def check_staircase(
    *, panel_concessions: list[dict],
    new_text: str,
    tenant: str,
    dialog: list[dict] | None = None,
) -> tuple[bool, dict]:
    """Determine whether `new_text` would be a staircase concession given
    the panel's prior concession history.

    Args:
      panel_concessions: list of prior {turn, type, amount, ...} dicts
        accumulated on this panel during the live A/B phase
      new_text: the agent's just-generated message text
      tenant: opp_meta.company — 'Libra' / 'Heavys' / other
      dialog: optional list of {role, text} dicts. When provided, the gate
        whitelists legitimate competitor-matching (agent matching a price
        the customer explicitly disclosed) so it doesn't get falsely
        flagged as staircase. Closes the Day-4-style false-positive where
        the customer says "I got 2200 from Wesure" and the agent's
        legitimate match to 2,200 was being blocked.

    Returns:
      (is_staircase: bool, meta: dict)
        meta = {detected_type, new_amount, prior_best, reason, new_concessions,
                competitor_match: bool}
        new_concessions is the list of new entries the caller should append
        to panel_concessions after acceptance (or discard on rejection).

    2026-05-10 refactor: thresholds, whitelist tolerances, locale, and
    extractor dispatch are now driven by per-tenant YAML config in
    `data/anti_staircase/<Tenant>.yaml`. See `staircase_config.get_config`.
    Hard-coded module constants (DEFAULT_CUMULATIVE_THRESHOLD, etc.) remain
    as fallback if config load fails — preserves backward compat.
    """
    meta = {
        "detected_type": None,
        "new_amount": None,
        "prior_best": None,
        "reason": None,
        "new_concessions": [],
        "competitor_match": False,
    }
    if not new_text or not tenant:
        return False, meta

    # Load per-tenant config (cached). Falls back to hard-coded defaults
    # for known tenants if YAML system is unreachable.
    from staircase_config import get_config
    cfg = get_config(tenant) or {}
    extractor_module = (cfg.get("extractor_module") or tenant.strip().lower())
    thresholds = cfg.get("thresholds") or {}
    whitelist = cfg.get("whitelist") or {}
    cumulative_default = float(thresholds.get("cumulative_default", DEFAULT_CUMULATIVE_THRESHOLD))
    cumulative_late_stall = float(thresholds.get("cumulative_late_stall", LATE_PHASE_STALL_THRESHOLD))
    critical_threshold = float(thresholds.get("critical", CRITICAL_CUMULATIVE_THRESHOLD))
    competitor_tolerance = float(whitelist.get("competitor_match_tolerance", 200))
    customer_target_tolerance = float(whitelist.get("customer_target_tolerance", 200))
    meta["_config_used"] = bool(cfg)
    meta["_extractor_module"] = extractor_module

    if extractor_module == "libra":
        offers = extract_libra_offered_prices(new_text)
        if not offers:
            return False, meta
        new_min = min(offers)
        meta["new_amount"] = new_min
        meta["detected_type"] = "price"
        prior_prices = [c["amount"] for c in panel_concessions
                         if c.get("type") == "price"]
        if prior_prices:
            prior_best = min(prior_prices)  # lowest = best (most concessive)
            prior_max = max(prior_prices)   # highest = original anchor
            meta["prior_best"] = prior_best
            meta["prior_max"] = prior_max

            # 2026-05-05 → 2026-05-06 — Context-aware cumulative-concession
            # check (Option D). Default 15% threshold matches typical
            # production behavior, but loosens dynamically when:
            #   (a) customer disclosed competitor quote → match exactly
            #   (b) customer stated a specific target → match exactly
            #   (c) conversation in late-phase stall (5+ turns, repeated
            #       objection) → loosen to 30% (matches what real human
            #       agents do per empirical analysis 2026-05-06)
            # Empirical: 23% of real Libra won-deal agents drop >15%;
            # most of those happen in late-phase / target-match contexts.

            # Pre-check: customer-stated target (highest priority — match exactly)
            # Phase A: pass cfg so the classifier path activates when configured
            customer_target = extract_customer_stated_target(dialog, cfg=cfg)
            if customer_target is not None and abs(new_min - customer_target) <= customer_target_tolerance:
                meta["reason"] = (
                    f"libra customer-target match: customer stated target "
                    f"{customer_target} NIS; agent offered {new_min} NIS within "
                    f"±{customer_target_tolerance:.0f}. Legitimate close, not staircase.")
                meta["competitor_match"] = False
                meta["customer_target_match"] = True
                meta["new_concessions"] = [
                    {"type": "price", "amount": p, "currency": "NIS"} for p in offers]
                return False, meta

            # Determine the active threshold for THIS turn (config-driven)
            active_threshold = cumulative_default
            stall_active = detect_late_phase_stall(dialog, panel_concessions)
            if stall_active:
                active_threshold = cumulative_late_stall
            cumulative_floor = prior_max * (1 - active_threshold)
            meta["active_threshold"] = active_threshold
            meta["stall_relaxed"] = stall_active

            if new_min < cumulative_floor:
                # Existing competitor whitelist (still applies)
                competitor_quote = extract_customer_disclosed_competitor_price(dialog)
                if competitor_quote is not None and abs(new_min - competitor_quote) <= competitor_tolerance:
                    meta["reason"] = (
                        f"libra competitive match (whitelisted, cumulative-OK): "
                        f"agent offered {new_min} matching customer-disclosed "
                        f"competitor {competitor_quote}")
                    meta["competitor_match"] = True
                    meta["new_concessions"] = [
                        {"type": "price", "amount": p, "currency": "NIS"} for p in offers]
                    return False, meta
                # Block — over the (context-adjusted) threshold
                threshold_pct = int(active_threshold * 100)
                # Compute observed drop fraction for criticality classification
                observed_drop = (prior_max - new_min) / prior_max if prior_max > 0 else 0.0
                is_critical = observed_drop >= critical_threshold
                meta["observed_drop_pct"] = round(observed_drop * 100, 1)
                meta["is_critical"] = is_critical
                meta["reason"] = (
                    f"libra cumulative concession: agent-offered {new_min} NIS "
                    f"is >{threshold_pct}% below max-ever {prior_max} NIS "
                    f"(floor={cumulative_floor:.0f}, "
                    f"observed drop={meta['observed_drop_pct']}%, "
                    f"active threshold={threshold_pct}% "
                    f"{'(stall-relaxed)' if stall_active else '(default)'}). "
                    f"Customer will perceive a {prior_max - new_min:.0f} NIS drop "
                    f"and lose trust. Hold price or surface real justification."
                    + (f" [CRITICAL: ≥{int(critical_threshold*100)}% drop — "
                       f"will trigger safe-fallback on retry exhaustion]"
                       if is_critical else ""))
                return True, meta

            if new_min < prior_best:
                # Whitelist: customer explicitly disclosed a competitor's
                # quoted price. Agent matching that disclosed price is NOT
                # staircase — it's a legitimate competitive response.
                competitor_quote = extract_customer_disclosed_competitor_price(dialog)
                if competitor_quote is not None and abs(new_min - competitor_quote) <= competitor_tolerance:
                    meta["reason"] = (
                        f"libra competitive match (whitelisted): agent offered {new_min} "
                        f"NIS within ±{competitor_tolerance:.0f} of customer-disclosed competitor "
                        f"{competitor_quote} NIS")
                    meta["competitor_match"] = True
                    meta["new_concessions"] = [
                        {"type": "price", "amount": p, "currency": "NIS"} for p in offers]
                    return False, meta
                meta["reason"] = (
                    f"libra staircase: agent-offered {new_min} NIS is below "
                    f"prior best {prior_best} NIS")
                return True, meta
        # Not a staircase → record new offers for panel state
        meta["new_concessions"] = [
            {"type": "price", "amount": p, "currency": "NIS"} for p in offers]
        return False, meta

    if extractor_module == "heavys":
        d = extract_heavys_offered_discounts(new_text)
        amounts = d["discount_amounts"]
        codes = d["codes"]
        if not amounts and not codes:
            return False, meta
        # Compare discount magnitudes
        if amounts:
            new_max = max(amounts)
            meta["new_amount"] = new_max
            meta["detected_type"] = "discount"
            prior_discs = [c["amount"] for c in panel_concessions
                            if c.get("type") == "discount"]
            if prior_discs:
                prior_best = max(prior_discs)
                meta["prior_best"] = prior_best
                if new_max > prior_best:
                    meta["reason"] = (
                        f"heavys staircase: agent escalated discount to "
                        f"${new_max} off (prior best was ${prior_best})")
                    return True, meta
        # Code-count check: if 2+ distinct VIP codes already offered AND
        # this turn introduces a NEW one, that's also staircase
        prior_codes = set()
        for c in panel_concessions:
            if c.get("type") == "code" and c.get("code"):
                prior_codes.add(c["code"].upper())
        new_codes = set(codes) - prior_codes
        if len(prior_codes) >= 1 and new_codes:
            meta["detected_type"] = "code"
            meta["reason"] = (
                f"heavys staircase: introducing additional discount code "
                f"{sorted(new_codes)} on top of prior {sorted(prior_codes)}")
            return True, meta
        # Not a staircase → record
        new_concs = []
        for amt in amounts:
            new_concs.append({"type": "discount", "amount": amt, "currency": "USD"})
        for code in codes:
            new_concs.append({"type": "code", "code": code})
        meta["new_concessions"] = new_concs
        return False, meta

    return False, meta


def build_correction_prompt(meta: dict) -> str:
    """Generate a correction string to inject into the agent prompt on retry.
    Caller appends this to the existing system prompt before regenerating."""
    detected = meta.get("detected_type") or "concession"
    new_amount = meta.get("new_amount")
    prior_best = meta.get("prior_best")
    reason = meta.get("reason") or "staircase concession"

    if detected == "price":
        return (
            f"\n\n# ⚠ STAIRCASE DETECTED — REGENERATE\n"
            f"Your previous draft would have lowered the price to "
            f"{new_amount} NIS, below your prior best offer of {prior_best} NIS. "
            f"Multiple price drops in one conversation are a known sales "
            f"failure mode — customer reaction is 'why didn't you offer this "
            f"from the start?' followed by trust collapse.\n"
            f"REGENERATE this turn WITHOUT lowering the price. Address the "
            f"objection via coverage value, deductibles, loyalty perks, "
            f"installment terms, cashback, or relationship — anything except "
            f"another price drop. Hold the line at {prior_best} NIS."
        )
    if detected == "discount":
        return (
            f"\n\n# ⚠ STAIRCASE DETECTED — REGENERATE\n"
            f"Your previous draft would have escalated the discount to "
            f"${new_amount} off, larger than your prior best of ${prior_best}. "
            f"Repeatedly escalating discounts damages trust — customer "
            f"reaction is 'why wasn't this the offer from the start?'\n"
            f"REGENERATE this turn WITHOUT increasing the discount. Address "
            f"the objection via product value, complementary perks, "
            f"installment options, or relationship. Hold at ${prior_best} off."
        )
    if detected == "code":
        return (
            f"\n\n# ⚠ STAIRCASE DETECTED — REGENERATE\n"
            f"Your previous draft would have introduced an additional discount "
            f"code on top of the one already offered. Stacking discount codes "
            f"reads as desperation. REGENERATE without adding a new code; "
            f"address the objection via product value or relationship."
        )
    return f"\n\n# ⚠ {reason} — REGENERATE without further concession.\n"


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_amount(s: str) -> float | None:
    """Parse a numeric string like '2,987' or '3,480.50' to float. Returns
    None on any error — caller handles."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None
