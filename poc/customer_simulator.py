"""Customer simulator — hybrid Option C (v1) + reference-aware (v2).

Plays the role of the historical customer when the live agent diverges from the
historical agent path. Used by both panels independently but shares the
similarity-cache and randomness-seed for parallelism.

v1 algorithm per turn (default):
  1. Get the live agent message just emitted
  2. Compare to the historical agent message at the same turn-index
     (semantic similarity, embedding-based)
  3. If similarity > 0.7 ("path matches history"): rephrase historical
     customer reply (preserve intent, vary wording 30-50%)
  4. If similarity ≤ 0.7 ("path diverged"): generate a fresh persona-conditioned
     reply using customer profile + agent message + conversation history

v2 algorithm per turn (opt-in via POC_SIM_V2_REFERENCE=on):
  Single unified path. Always uses the persona-generate prompt, but with the
  historical customer reply AT THE SAME TURN-INDEX surfaced explicitly as a
  reference block. Character-grounding historical is sliced to "history up to
  and including the current historical agent message" — same-turn-only
  guardrail, no future leakage. The LLM blends reference posture/tone with
  on-topic response to whatever the live agent actually said.

Detects closing-decline phrases (final "no thanks", "going elsewhere") to signal
panel-side Lost outcome.
"""
from __future__ import annotations

import logging
import os
import re

import anthropic
import numpy as np

log = logging.getLogger(__name__)

SIMULATOR_MODEL = "claude-sonnet-4-5-20250929"
EMBED_MODEL = "voyage-3-lite"  # fallback to simple word-overlap if voyage not available
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
SIMILARITY_THRESHOLD = 0.7

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


# R10 — Hard-customer mode (2026-05-05). Layered on top of the normal persona
# block to make the simulator more skeptical and harder to close. Used to
# stress-test supervisor wins so batch numbers aren't inflated by an
# over-cooperative simulator. Toggle via POC_HARD_CUSTOMER env var.
HARD_CUSTOMER_OVERLAY = """
## HARD-CUSTOMER MODE — applied on top of your normal persona traits

You are a STRICTER version of this customer. You're not hostile, but you've
been in negotiations before and you're well-defended. Apply these rules in
addition to your normal traits:

1. RESIST VALUE-STACK REFRAMES. If the agent lists features ("8-driver
   system, 50-hour battery, free shipping," or "free coverage included,
   3% cashback") without first matching your price target, your reaction
   is: "I appreciate the specs, but they don't change the price." Feature
   lists alone do NOT raise your engagement.

2. SLOW THE COMMITMENT LADDER. Even when getting closer to yes, never jump
   more than +1 commitment level per turn. Save the explicit "yes, do it"
   reply for a moment when the agent has actually solved your stated
   problem. If the agent merely improved the offer, you say "I'd consider
   it" — not "let's do it."

3. REQUIRE EXPLICIT MATCH BEFORE CLOSING. You only respond with closing
   language ("yes, let's do it" / "go ahead") if BOTH:
   (a) the agent's offer exactly matches your stated target number (or is
       below it), AND
   (b) the agent has addressed the SPECIFIC objection you raised most
       recently.
   A "shall we proceed?" question without both is a "let me think about it."

4. PROBE INCONSISTENCIES. If the agent quoted price X earlier and later
   offers X minus a meaningful amount without explaining what changed,
   ask: "Why didn't you offer this from the start? What's the catch?"
   This is a ONE-TURN concern. After the agent answers (even partially),
   move on — do NOT escalate it into multi-year-guarantee demands or
   fairness grievances. Real customers raise this once and then either
   close or shift to logistics.

5. CHALLENGE VAGUE NUMBERS. If the agent says "3% cashback," "up to $50
   off," "interest-free installments," ask for the specific dollar amount
   on YOUR purchase. Don't accept percentages without converting them.

6. STAY POLITE. You're hard, not rude. No name-calling, no profanity. You
   exit by saying "I'll think about it" or "thanks anyway, not today" —
   never by being insulting.

Reply 1-2 sentences typical."""


# 2026-05-10 — Post-match calibration overlay. Empirical libra_c0 distribution
# (n=1,814 won-side post-match pairs, see research-notes/POST-MATCH-CUSTOMER-
# DISTRIBUTION.md): 65% close_intent, 29% short ack, 6% new_objection,
# 0.1% trust_grievance, 0.1% counter_offer.
# Without this overlay the simulator (especially in hard mode) over-produces
# trust_grievance / multi-year-guarantee fixation by ~500x vs reality.
POST_MATCH_OVERLAY = """
## ⚠⚠⚠ POST-MATCH STATE — THIS RULE OVERRIDES YOUR PERSONA THIS TURN

The agent has now matched your target price (or come within touching
distance). You are at the moment of decision. Empirical libra_c0 won-deal
data (n=1,814 real customers in your exact profile cohort: Skeptical +
Analytical + Price/savings) shows what real people do at this moment:

  - 65% CLOSE IMMEDIATELY → "ok let's do it" / "send installments" / "great,
    proceed with signature" / "yes 10 installments, last 4 are 1234"
  - 29% short non-committal ack → "ok thanks, let me confirm briefly" — and
    close in their NEXT turn
  - 6% raise ONE concrete logistics objection (coverage detail, paperwork
    timing, payment schedule) — and close after the agent answers
  - **0.1% trust_grievance** ("what guarantee about next year" / "you only
    give fair prices when threatened" / "what's stopping you from raising
    it again"). ONE in ONE THOUSAND customers. You are NOT that customer.

YOU MUST NOT:
  ✗ Ask "what guarantee do I have you won't raise it next year"
  ✗ Ask "what's stopping you from raising it again after I renew"
  ✗ Ask "why didn't you offer this from the start" MORE THAN ONCE in the
    whole conversation. If the agent already answered it (with ANY answer),
    accept and pivot to close or logistics. Do NOT re-ask.
  ✗ Cancel the deal over fairness grievances after getting your target price
  ✗ Extrapolate "agent could lower 8K → 5K now" into "agent has been
    overcharging me, I can't trust them" → walking away. Real customers in
    your cohort do not do this.
  ✗ Demand multi-year price guarantees as a condition of renewing now

YOU SHOULD (pick one, weighted by the empirical distribution above):
  ✓ Close: "ok let's do it, 10 installments" or similar (65%)
  ✓ Brief ack: "ok thanks, give me a moment" (29%)
  ✓ One concrete logistics question: "what about mandatory insurance?",
    "when does it start?", "is signature electronic?" (6%)

This overrides your normal HARD-CUSTOMER stubbornness for this single turn.
Hard mode is SUPPRESSED when post-match is detected. The empirical
distribution wins."""


def _post_match_detected(live_agent_msg: str, dialog_history: list[dict]) -> bool:
    """Heuristic: does the agent's most recent message look like a price-match
    delivery? Combines (a) NIS price present + (b) close-pivot phrase + (c)
    a recent customer turn that mentioned competitor/match.
    """
    if not live_agent_msg:
        return False
    text = live_agent_msg.lower()
    # (a) NIS price token present
    has_price = bool(re.search(r"\d[\d,\.]{2,5}\s*(ש[״\"]ח|שח|₪|nis)",
                                live_agent_msg, re.IGNORECASE))
    if not has_price:
        return False
    # (b) close-pivot phrase
    pivot_en = ["shall we proceed", "let's lock", "let's close",
                "how many installments", "to confirm", "great, agreed",
                "send for signature", "send the policy", "go ahead",
                "ready to renew", "ready to close", "match approved",
                "let's do it"]
    pivot_he = ["מעולה", "סבבה", "מנהל", "אישור", "סגור", "תשלומים",
                "כרטיס", "לחדש", "תוכל לחדש", "להתקדם ולסגור"]
    has_pivot = any(p.lower() in text for p in pivot_en) or any(p in live_agent_msg for p in pivot_he)
    if not has_pivot:
        return False
    # (c) earlier in dialog the customer signaled a competing offer.
    # 2026-05-13 — Semantic check via mined customer_post_match anchors first
    # (catches "I already have these from Bose", "I went with another company",
    # "got a quote from AIG" etc.). Falls back to keyword scan for tenant where
    # mining had insufficient data.
    EN_KEYS = ["competitor", "match", "beat", "cheaper", "elsewhere",
               "another company", "another insurer", "another offer",
               "got a quote", "got a price", "they offered", "their offer",
               "somewhere else", "another place", "different company"]
    HE_KEYS = ["מתחר", "להשוות", "זול", "במקום אחר", "חברה אחרת",
               "הצעה אחרת", "הצעה כוללת"]
    PRICE_RE_LOCAL = re.compile(r"\d[\d,\.]{2,5}\s*(ש[״\"]ח|שח|₪|nis)", re.IGNORECASE)
    try:
        from intent_classifier import intent_score, is_available
        has_classifier = is_available()
    except Exception:
        has_classifier = False
    for m in dialog_history[-12:]:
        if m.get("role") != "customer":
            continue
        cust_t = (m.get("text") or "")
        cust_lower = cust_t.lower()
        # Semantic check — but vetoed on question-marks. A customer asking a
        # question is still engaged, not post-match. Without this veto, mining
        # picks up price-mentioning patterns that semantically overlap with
        # "What's the price for X?" — false positive.
        if has_classifier and "?" not in cust_t:
            try:
                decision, _s, _a = intent_score(cust_t, "customer_post_match")
                if decision:
                    return True
            except Exception:
                pass
        if any(k in cust_lower for k in EN_KEYS):
            return True
        if any(k in cust_t for k in HE_KEYS):
            return True
        if PRICE_RE_LOCAL.search(cust_t):
            return True
    return False


def _hard_customer_enabled(opp_meta: dict | None = None) -> bool:
    """Read flag at call time. Per-session override (set via /api/run body
    parameter, propagated onto opp_meta as `_hard_customer`) takes precedence
    over the POC_HARD_CUSTOMER env var."""
    if opp_meta is not None and opp_meta.get("_hard_customer"):
        return True
    return os.getenv("POC_HARD_CUSTOMER", "0").strip().lower() in ("1", "true", "yes")


def render_opp_coherence_constraint(opp_meta: dict) -> str:
    """Build a coherence constraint based on opp_type so the simulator's
    GENERATE path doesn't produce responses that contradict the customer's
    actual buy-intent state.

    Empirical motivation (2026-05-01): Karl-class customer (Heavys c3, opp_type=
    'Abandoned Cart') generated a response saying 'I wasn't even shopping for
    these' — which literally contradicts having added the product to cart 5
    minutes earlier. The persona was internally consistent but the opp-state
    coherence was broken.
    """
    opp_type = (opp_meta.get("opp_type") or "").lower()
    tenant = opp_meta.get("company") or ""

    # Map opp_type to coherence rules
    if any(k in opp_type for k in ("abandoned cart", "cart abandon", "abandon cart")):
        return f"""## Opportunity coherence constraint
You ({tenant} customer) are in an ABANDONED CART scenario. You explicitly
added a product to your cart and then walked away from checkout. Your
responses MUST be consistent with that fact:

- You CAN say "I lost interest", "I changed my mind", "the price gave me pause",
  "I wanted to think about it more", "I got distracted", "second-thoughts kicked in"
- You CANNOT say "I wasn't shopping for this", "I never wanted this", "I never
  considered this", "this isn't something I'd buy" — those contradict the cart-add
- You can claim the price was too high, the timing was wrong, you found a
  competitor, you found doubt — but you CANNOT deny that you were shopping
- It IS valid to say "my current product works fine, that's why I walked away"
  — that's a buyer's-remorse-like reason, consistent with cart abandonment
"""

    if "renewal" in opp_type or "renew" in opp_type:
        return f"""## Opportunity coherence constraint
You ({tenant} customer) are an EXISTING customer at RENEWAL stage. You have an
active relationship with this product/service. Your responses MUST be consistent:

- You CAN say "I'm comparing offers", "another company quoted me less",
  "I want to review the terms first", "my needs have changed", "I'm thinking
  about switching"
- You CANNOT say "I never used this", "this isn't my product", "I'm not a
  customer" — those contradict the renewal context
- You CAN say "I'm not sure I'll renew" or "I might cancel" — those are valid
  walk-away reasons consistent with renewal
- You CAN reference your past experience with the product (good or bad)
"""

    if "upsell" in opp_type or "cross" in opp_type or "expansion" in opp_type:
        return f"""## Opportunity coherence constraint
You ({tenant} customer) are an EXISTING customer being offered an UPSELL.
You already have the base product/service. Your responses MUST be consistent:

- You CAN say "my current setup is enough", "I don't see the value-add",
  "I'd need to see ROI", "let me think about it", "not interested in upgrading
  right now"
- You CANNOT say "I'm not your customer" or "I never bought from you" —
  those contradict the upsell context
- You CAN reference using the base product
"""

    if "trial" in opp_type:
        return f"""## Opportunity coherence constraint
You ({tenant} customer) are in a TRIAL → PAID conversion scenario. You signed
up for a trial of this product. Your responses MUST be consistent:

- You CAN say "I'm still evaluating", "I haven't used it enough", "I'm not
  sure if it's worth the price", "the trial wasn't enough time"
- You CANNOT say "I never signed up" or "I don't know what this is" —
  those contradict the trial context
"""

    if "purchasing assistance" in opp_type or "search_catalog" in opp_type or "browse" in opp_type:
        return f"""## Opportunity coherence constraint
You ({tenant} customer) ENGAGED Luna voluntarily — you reached out for help
selecting a product. Your responses can be:

- Buy-intent ("I'm looking for X") OR browse-intent ("just exploring") —
  both valid since you initiated the conversation
- You CANNOT claim "I never wanted to talk to you" or "stop messaging me out
  of nowhere" — you started this conversation
"""

    if "review" in opp_type or "leave review" in opp_type:
        return f"""## Opportunity coherence constraint
You ({tenant} customer) are POST-PURCHASE — this is a review-request context,
not a sales conversation. Your responses should be:

- Reflect on your actual experience with the product
- You CAN decline to leave a review or say "I don't have time"
- You CANNOT pretend you never bought the product — you did
"""

    # Default — generic; no specific constraints
    return ""


# Closing-decline phrases — when the customer simulator emits one, the panel
# is marked Lost.
#
# IMPORTANT (2026-04-30): regex must distinguish HARD-DECLINE from BARGAINING.
# Found in Libra c5 session 10a45705: "I'm not interested in being just a
# hundred or two below the competitor" was killing the supervised panel as
# decline, but the customer was bargaining (the conditional was on price gap,
# not on the deal). Negative-lookahead added to "not interested" excludes
# common bargaining contexts ("interested in being [adverb]", "in just X",
# "in a [token/small/sub-par] discount", etc.). Stand-alone "not interested"
# (sentence-final or followed by basic decline contexts) still fires.
DECLINE_PHRASES_EN = [
    # "not interested" UNLESS followed by a bargaining/conditional clause:
    #   - "in being [adverb] [adj]" (e.g., "in being just a hundred below")
    #   - "in just/only/merely/maybe X"
    #   - "in a [small/sub-par/tiny/token/marginal/minor/slight/petty] discount"
    #   - "in paying/getting/having more/less/under/over X" (bargaining)
    r"\bnot interested\b(?!\s+in\s+(?:being|just|only|merely|maybe|a\s+(?:sub-?par|small|tiny|token|marginal|minor|slight|petty)|paying\s+(?:more|less|over|under)|getting\s+(?:less|under|over)|having\s+(?:less|just)))",
    r"\bgoing elsewhere\b", r"\bsigned with\b",
    r"\bclosed with another\b", r"\bdone here\b", r"\bdon[''']t contact\b",
    r"\bstop messaging\b", r"\bstop contact(?:ing)?\b", r"\bplease stop\b",
    r"\bgoodbye\b", r"\bgood\s*bye\b", r"\bbye[!.\s]*$", r"\bleave me alone\b",
    r"\bremove me\b", r"\bunsubscribe\b", r"\btake me off\b",
    # NEW (v0.3): customer confirms they went with competitor — unrecoverable
    r"\bdid it with another\b", r"\bi went with another\b", r"\bwent with another\b",
    r"\bbought (?:it )?from\b", r"\bcheaper with all\b", r"\bcheaper with the coverage\b",
    r"\bi['' ]?ve (?:already )?signed\b", r"\bi paid (?:already|the)\b",
    r"\balready (?:signed|paid|renewed) with\b", r"\bgot (?:it|the policy) from\b",
]
DECLINE_PHRASES_HE = [
    "לא מעוניין", "לא מעוניינת", "סגרתי עם", "לא רלוונטי",
    "תודה לא", "לא תודה", "אל תפנו אלי", "תפסיקו",
    "להסיר אותי", "תסירו אותי", "להפסיק",
    # NEW: competitor-confirmed in Hebrew
    "סגרתי עם אחר", "עברתי לחברה אחרת", "כבר חידשתי", "כבר שילמתי",
]


# Agent-side graceful-close patterns — when the agent emits a final warm close
# AFTER the customer has effectively declined. Used to end the panel cleanly
# instead of looping into "thanks / good luck" pleasantries.
AGENT_GRACEFUL_CLOSE_PATTERNS_EN = [
    r"\ball the best\b", r"\btake care\b", r"\bbe well\b",
    r"\bwishing you (?:a )?(?:safe|great|wonderful)\b",
    r"\bif (?:anything|things) change[,. ]?\b",
    r"\bbest wishes\b", r"\bhave a (?:great|wonderful|good) (?:day|year)\b",
    r"\bglad you found\b", r"\bcoverage that works for you\b",
    r"\bgood luck\b", r"\bwe[''']?re here\b.{0,30}\bif\b",
]
AGENT_GRACEFUL_CLOSE_PATTERNS_HE = [
    "כל טוב", "בהצלחה", "שתהיה לך שנה", "להתראות",
    "אם משהו ישתנה", "אנחנו כאן", "תזכור אותנו",
    "יום טוב", "שיהיה לך",
]


def detect_agent_graceful_close(text: str) -> bool:
    """The agent has emitted a final warm closure (no pitch).
    Combined with low commitment history, this signals end of panel."""
    if not text:
        return False
    low = text.lower()
    if any(re.search(p, low) for p in AGENT_GRACEFUL_CLOSE_PATTERNS_EN):
        return True
    if any(p in text for p in AGENT_GRACEFUL_CLOSE_PATTERNS_HE):
        return True
    return False


def detect_language(texts: list[str]) -> str:
    """Detect predominant language ('en' / 'he' / 'unknown') from a list of message texts.
    Cheap heuristic: count Hebrew characters vs Latin characters."""
    if not texts:
        return "unknown"
    he_chars = 0
    en_chars = 0
    for t in texts:
        if not t:
            continue
        for c in t:
            if "֐" <= c <= "׿":  # Hebrew block
                he_chars += 1
            elif c.isalpha() and c.isascii():
                en_chars += 1
    if he_chars > en_chars * 1.5:
        return "he"
    if en_chars > he_chars * 1.5:
        return "en"
    return "unknown"


# Agent self-refusal patterns — when the actor LLM emits meta-commentary
# instead of a real customer-facing message. We treat these as "agent refused
# to continue" → end the panel with status=lost+reason=agent_refused.
AGENT_REFUSAL_PATTERNS = [
    r"\*no message\*", r"\*no reply\*",
    r"\bi should not (?:reply|send|continue|message)\b",
    r"\bi cannot (?:send|reply|continue)\b",
    r"violat(?:e|es|ing)\s+(?:their|the)?\s*(?:clear\s+)?boundary",
    r"the conversation (?:must|should)\s+remain\s+closed",
    r"\bi['' ]?ll (?:not|stop)\s+(?:reply|send|message)",
    r"\bend(?:ing)?\s+the\s+conversation\s+here\b",
    r"explicitly\s+asked\s+(?:to\s+)?stop",
]


def detect_agent_refusal(text: str) -> bool:
    """Detect when the actor LLM emitted self-aware refusal/meta-commentary
    instead of a real customer-facing message."""
    if not text:
        return True  # empty agent message = refusal
    low = text.lower()
    for p in AGENT_REFUSAL_PATTERNS:
        if re.search(p, low):
            return True
    # Heuristic: actor output that's mostly explanation in asterisks or brackets
    # rather than a real message
    stripped = re.sub(r"\*[^*]+\*|\[[^\]]+\]", "", text).strip()
    if not stripped or len(stripped) < 8:
        return True
    return False

# Customer-side soft farewell — neither decline nor commitment, just polite
# wind-down. Distinct from DECLINE (hard "not interested") and WIN (active
# commit). When these fire AFTER engagement, the conversation is effectively
# over — keeping it alive only produces low-information pleasantry loops that
# dilute the persuasion-score curve.
#
# Patterns tolerate optional agent-name suffix (e.g. "Thanks Sarah", "Take
# care Sarah, have a great day") because customer-simulator personas often
# echo the agent name in farewell messages. Length cap (~80 chars) prevents
# false-positives on substantive messages that happen to start with "thanks".
CUSTOMER_FAREWELL_PATTERNS_EN = [
    # standalone affirmations / brief closes (anchored)
    r"^\s*(?:tx|thx|thanks?(?:\s+(?:again|so much|a lot))?|thank\s+you(?:\s+again)?)\s*[!.?]*\s*$",
    r"^\s*(?:later|laters|cheers|peace)\s*[!.?]*\s*$",
    r"^\s*sounds?\s+good\s*[!.?]*\s*$",
    r"^\s*(?:cool|nice|sweet)\s*[!.?]*\s*$",
    r"^\s*(?:ok|okay|alright|alrighty)\s*[!.?]*\s*$",
    r"^\s*tx\s+ill\s+check\s*[!.?]*\s*$",
    r"^\s*(?:appreciate\s+it|appreciated)\s*[!.?]*\s*$",
    # parting verbs (anywhere in short message)
    r"\bcatch\s+you\s+later\b", r"\btalk\s+(?:to\s+you\s+)?(?:later|soon)\b",
    r"\bchat\s+(?:soon|later)\b", r"\bsee\s+(?:ya|you)(?:\s+later)?\b",
    r"\bgotta\s+(?:run|go)\b", r"\bhave\s+to\s+(?:run|go|leave)\b",
    # take-care variants (with optional name suffix)
    r"\btake\s+care\b",
    # day/weekend wishes (clear closing signals)
    r"\bhave\s+a\s+(?:great|good|nice|wonderful|lovely)\s+(?:day|night|weekend|year|evening|one)\b",
    # thanks + name pattern: "Thanks <NAME>" where the message is short/closing
    # 2026-05-11 — Bug fix: matcher runs against a LOWERCASED string, so
    # the previous [A-Z][a-z]+ requirement for capitalization could never
    # match. Replaced with a name-like token that ISN'T a common continuation
    # word (for/again/so/a/the/etc.) — guards against "thanks for the info"
    # firing as farewell.
    r"^\s*(?:tx|thx|thanks?|thank\s+you)\s+(?!for\b|again\b|so\b|a\s|the\b|that\b|this\b|though\b|but\b)[a-z]+\b",
    # appreciate + (it|that|name)
    r"\bi\s+appreciate\s+(?:it|that|you|your)\b",
    # 2026-05-11 — Common farewell sign-offs missed by the previous pattern set.
    # These appear frequently in winding-down conversations: customer is
    # signaling "no more agent response needed, I'll re-initiate if I need to."
    r"\bi(?:'ll| will)\s+wait\s+to\s+hear\b",         # "I'll wait to hear from you"
    r"\bi(?:'ll| will)\s+be\s+in\s+touch\b",          # "I'll be in touch"
    r"\bi(?:'ll| will)\s+reach\s+out\s+if\b",         # "I'll reach out if anything changes"
    r"\bi(?:'ll| will)\s+let\s+you\s+know\b",         # "I'll let you know"
    r"\bi(?:'ll| will)\s+get\s+back\s+to\s+you\b",    # "I'll get back to you"
    r"\bi(?:'ll| will)\s+think\s+about\s+it\b",       # "I'll think about it" (terminal stall)
    r"\bif\s+(?:i|things)\s+change\b",                # "if anything changes on my end"
    r"\bbefore\s+(?:the\s+)?(?:holidays|new\s+year|end\s+of)\b",  # temporal farewell
]

# Length cap — messages over this are considered substantive (could carry
# new info) so farewell patterns don't fire on them.
# 2026-05-11 — Bumped from 80 to 140 because the new patterns above (e.g.,
# "I'll be in touch if I decide to move forward before then") routinely exceed
# the prior cap and are still clearly farewell-class.
_FAREWELL_MAX_CHARS = 140


def detect_customer_farewell(text: str, tenant: str | None = None) -> bool:
    """Detect soft conversational close: 'thanks again', 'catch you later',
    'take care Sarah', 'have a great day', 'I appreciate it'. Customer is
    winding down without declining or committing.

    2026-05-11 — Primary path is now semantic (model2vec via
    server.intent_classifier). Regex path retained as fallback for when the
    model is unavailable.
    2026-05-13 — Tenant parameter wires through to per-tenant mined farewell
    anchors (data/intent_anchors/farewell__<Tenant>.json).

    A question mark anywhere in the message is a hard universal veto: a
    customer asking something is not signing off, regardless of detector.
    """
    if not text:
        return False
    stripped = text.strip()
    if "?" in stripped:
        return False
    # Primary: semantic classifier
    try:
        from intent_classifier import intent_score, is_available
        if is_available():
            decision, _score, _anchor = intent_score(
                stripped, "farewell", tenant=tenant)
            return decision
    except Exception:
        pass
    # Fallback (model unavailable): regex path preserved verbatim
    if len(stripped) > _FAREWELL_MAX_CHARS:
        strong = [
            r"\btake\s+care\b",
            r"\bhave\s+a\s+(?:great|good|nice|wonderful|lovely)\s+(?:day|night|weekend|year|evening|one)\b",
            r"\bcatch\s+you\s+later\b", r"\btalk\s+(?:to\s+you\s+)?(?:later|soon)\b",
        ]
        low = stripped.lower()
        return any(re.search(p, low) for p in strong)
    low = stripped.lower()
    if any(re.search(p, low) for p in CUSTOMER_FAREWELL_PATTERNS_EN):
        return True
    return False


# Saturation signal-word regex — engagement vocabulary. If ANY of the last
# 3 customer messages contains one of these tokens, the customer is engaged
# and the panel should NOT be marked saturated.
_SATURATION_SIGNAL_RE = re.compile(
    r"\?|\$|nis|shekel|price|cost|tax|ship|deliver|warrant|return|"
    r"discount|coupon|code|payment|installment|when|how|why|what|"
    # price-objection vocabulary
    r"expensive|expense|cheap(?:er)?|afford|budget|"
    r"too\s+(?:high|low|much)|too\s+expensive|"
    # Hebrew engagement
    r"יקר|זול|תקציב|מחיר",
    re.I,
)


def detect_saturation(recent_customer_msgs: list[str],
                       tenant: str | None = None) -> bool:
    """Three consecutive short post-close-pleasantry customer messages —
    the conversation has plateaued. End the panel rather than keep emitting
    pleasantries.

    Trigger requirements (ALL must hold):
      1. ≥ 3 customer messages available
      2. Each of the last 3 is ≤ 60 chars
      3. NONE of the last 3 contains an engagement signal word (price /
         discount / expensive / יקר / etc.) — cheap regex first-pass
      4. ALL of the last 3 score ≥ 0.42 on the mined
         `customer_chitchat_acknowledgment` intent — semantic confirm

    Layered defense: regex blocks any cheap engagement signal; semantic
    requires positive chitchat evidence. Both must agree to declare
    saturation. False-positive cost (declaring chitchat when engaged) is
    bounded by the regex pass; false-negative cost (declaring engaged
    when actually chitchat) is bounded by the semantic pass.

    Fallback: when the model is unavailable, falls back to the regex-only
    behavior (pre-vector logic).

    2026-05-13 — Migrated from regex-only to layered regex+vector. The
    chitchat intent was mined from last 2 customer turns of WON deals.
    """
    if len(recent_customer_msgs) < 3:
        return False
    last3 = recent_customer_msgs[-3:]
    # (2) structural: all 3 short
    for m in last3:
        if not m or len(m.strip()) > 60:
            return False
    # (3) regex first-pass: any signal word → NOT saturated. Cheap, catches
    # English + Hebrew price-objection vocabulary.
    for m in last3:
        if _SATURATION_SIGNAL_RE.search(m):
            return False

    # (4) semantic confirm: ALL 3 must score as chitchat. Without this,
    # the regex-only path would over-saturate on messages like "sounds
    # reasonable" / "let me think" that contain no signal word but are
    # also not pure pleasantries.
    try:
        from intent_classifier import intent_score, is_available
        if is_available():
            for m in last3:
                decision, _s, _a = intent_score(
                    m.strip(), "customer_chitchat_acknowledgment", tenant=tenant)
                if not decision:
                    return False
            return True
    except Exception:
        pass

    # Model unavailable — regex passed, fall back to historical behavior.
    return True


# Closing-WIN phrases — trigger "the customer effectively closed"
WIN_PHRASES_EN = [
    r"\byes(?: please)?,?\s*renew\b", r"\blet'?s renew\b", r"\bgo ahead\b",
    r"\b(?:i'?ll|we'?ll) take it\b", r"\b(?:i'?m|i am) in\b",
    r"\bsounds good[,!.]\s*let",
    # Extended 2026-05-05 — close phrases the original regex missed
    # (e.g. "Perfect, let's do it. Renew for 2,600 and we're good.")
    r"\b(?:i'?ll|we'?ll) do (?:it|this)\b",
    r"\blet'?s do (?:it|this)\b",
    r"\blet'?s go\b",
    r"\bperfect[,!.\s]+let'?s\b",
    r"\bsign me up\b",
    r"\bdone deal\b",
    r"\bsold[!.\s]*$",
    r"\bwe'?re good\b",
    r"\b(?:ok|okay)[,!.\s]+(?:do|send|process) it\b",
    r"\bsend (?:me )?the link\b",
    r"\bcc[\s:]*[\d\s\-]{4}\b",
    # 2026-05-05 — customer-self-initiated payment close. Customer providing
    # CC last-4 digits or phone-confirm is a DEFINITIVE close regardless
    # of whether agent explicitly asked. Triggered case: a897d59b turn 31
    # where customer said "last 4 digits are 7834" after agent made a
    # concession statement (not an explicit close offer). G3 was rejecting
    # the close because agent_just_offered_close=False; this regex makes
    # it a G2-passing close-signal so G3-failure becomes irrelevant.
    r"\blast\s+(?:4|four)\s+(?:digits?|nums?)\s*(?:are\s+|is\s+|:\s*)?\d{4}\b",
    r"\bcc(?:[\s:#-]+\d){3,4}\b",                  # "CC: 1234 5678..."
    r"\bcredit\s+card\s+(?:is\s+|number\s+is\s+|:\s*)?\d{4}\b",
    r"\b(?:my )?card[\s:]+ending\s+(?:in\s+)?\d{4}\b",  # "card ending in 7834"
]
WIN_PHRASES_HE = [
    "כן תחדש", "כן בבקשה", "סגור", "מאשר", "מאשרת",
    "תחדש לי", "אני רוצה לחדש",
]

# 2026-05-05 — counter-offer detector. A customer message containing
# negotiation-conditional language ("round to X", "if you can do Y", "match
# their offer") + an implied price is NOT acceptance even when the persuasion
# scorer rates it commitment=5. The deal isn't closed until the agent agrees
# to the customer's number AND the customer confirms. Without this, Gemini's
# "this is a conditional closing statement" reads were firing won prematurely.
# Triggered case: 9b552922 turn 36 — customer said "Can we round it to 3000
# and move forward?" — agent never agreed to 3000, but won fired.
COUNTER_OFFER_PATTERNS_EN = [
    r"\bround (?:it )?to\b",                              # "round it to 3000"
    r"\bif you can (?:do|match|hit|drop|get|go to)\b",    # "if you can do 2600"
    r"\bwhat (?:if|about)\b[\s\S]{0,40}\d",               # "what if we did 2500"
    r"\bhow about\b[\s\S]{0,30}\d",                       # "how about 3000"
    r"\bcould (?:we|you) (?:do|drop|hit|match|go)\b",     # "could you do 2600"
    r"\bwould you (?:do|match|drop|hit|consider|go to)\b", # "would you match 2600"
    r"\bany chance\b[\s\S]{0,40}(?:\d|cheaper|less|drop)",
    r"\bmeet (?:me|us) (?:at|halfway)\b",
    r"\bsplit the difference\b",
    r"\bi'?d (?:do|take) it (?:for|at|if)\b",             # "I'd do it for 3000"
    r"\bif (?:you|we) (?:can|could) (?:get|drop|do|hit)\b[\s\S]{0,30}\d",
]
COUNTER_OFFER_PATTERNS_HE = [
    "אם תוכל",      # if you can
    "אם אפשר",      # if possible
    "מה דעתך",       # what do you think
    "אולי נסכם על",  # maybe we can settle on
    "תוריד עוד",     # drop a bit more
    "תעשה לי",       # do for me
]
_COUNTER_OFFER_REGEX_EN = re.compile(
    "|".join(COUNTER_OFFER_PATTERNS_EN), re.IGNORECASE
)


def detect_counter_offer(text: str, tenant: str | None = None) -> bool:
    """True if customer's message is a negotiation counter-offer (not
    acceptance). Used by the close-guard's G2 to block premature `won` on
    conditional/negotiation language.

    Primary: model2vec semantic match against mined counter-offer anchors
    (mid-commit customer turns with conditional language + digits, from won
    deals). Regex path retained as fallback when model is unavailable.
    """
    if not text:
        return False
    try:
        from intent_classifier import intent_score, is_available
        if is_available():
            decision, _s, _a = intent_score(text, "counter_offer", tenant=tenant)
            if decision:
                return True
    except Exception:
        pass
    if _COUNTER_OFFER_REGEX_EN.search(text):
        return True
    for he in COUNTER_OFFER_PATTERNS_HE:
        if he in text:
            return True
    return False


def detect_decline(text: str, tenant: str | None = None) -> bool:
    if not text:
        return False
    try:
        from intent_classifier import intent_score, is_available
        if is_available():
            decision, _s, _a = intent_score(text, "decline", tenant=tenant)
            if decision:
                return True
    except Exception:
        pass
    low = text.lower()
    if any(re.search(p, low) for p in DECLINE_PHRASES_EN):
        return True
    if any(p in text for p in DECLINE_PHRASES_HE):
        return True
    return False


def detect_close_signal(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if any(re.search(p, low) for p in WIN_PHRASES_EN):
        return True
    if any(p in text for p in WIN_PHRASES_HE):
        return True
    return False


def _word_overlap_similarity(a: str, b: str) -> float:
    """Cheap fallback similarity if no embedding model. Token Jaccard."""
    if not a or not b:
        return 0.0
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def similarity(a: str, b: str) -> float:
    """For POC we use cheap token-Jaccard as proxy. Good enough for the
    'did the agent say roughly the same thing as history?' check.

    A future iteration can swap to voyage-3-lite or text-embedding-3-large
    if we want sharper paraphrase detection."""
    return _word_overlap_similarity(a, b)


# ── Rephrase prompt (path matches history) ─────────────────────────────────
REPHRASE_SYSTEM = """You are paraphrasing a customer's reply in a sales conversation.

You will be given:
- The historical customer reply (verbatim from a real conversation that happened)
- The agent's just-emitted message (which closely matches the historical agent message)

Your task: rephrase the historical customer reply so it preserves the same intent
and same key information, but varies the wording naturally (30-50% different
words).

CRITICAL LANGUAGE RULE: Preserve the EXACT language of the historical reply.
- If the historical reply is in English, respond in English.
- If the historical reply is in Hebrew, respond in Hebrew.
- DO NOT translate. DO NOT switch languages.

Output: just the rephrased customer message. No quotes, no preamble, no JSON."""


async def rephrase_customer_reply(historical_reply: str, live_agent_msg: str,
                                    forced_language: str | None = None) -> str:
    lang_instruction = ""
    if forced_language == "en":
        lang_instruction = "\n\nFORCED LANGUAGE: respond in English ONLY. Do not use Hebrew."
    elif forced_language == "he":
        lang_instruction = "\n\nFORCED LANGUAGE: respond in Hebrew ONLY."

    user = f"""Historical customer reply (the customer actually said this):
{historical_reply}

Agent just said:
{live_agent_msg}

Rephrase the historical reply naturally.{lang_instruction}"""
    import time as _time
    _t0 = _time.monotonic()
    msg = await _client.messages.create(
        model=SIMULATOR_MODEL,
        max_tokens=300,
        system=REPHRASE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    response_text = msg.content[0].text.strip() if msg.content else historical_reply
    # T-86 trace
    try:
        from trace_logger import TraceLogger
        trace = TraceLogger.current()
        if trace:
            trace.llm(
                stage="simulator.rephrase",
                provider="anthropic",
                model=SIMULATOR_MODEL,
                system=REPHRASE_SYSTEM,
                user=user,
                response=response_text,
                latency_ms=int((_time.monotonic() - _t0) * 1000),
                input_tokens=getattr(msg.usage, "input_tokens", 0) if hasattr(msg, "usage") else 0,
                output_tokens=getattr(msg.usage, "output_tokens", 0) if hasattr(msg, "usage") else 0,
            )
    except Exception:
        pass
    return response_text


# ── Generate prompt (path diverged) ────────────────────────────────────────
# 2026-05-11 — Scope-constraint overlay for the simulator. Without this, the
# GENERATE path produces dramatic out-of-scope crises (eviction threats,
# unauthorized-charge disputes, fraud accusations) that are NOT representative
# of real sales conversations and make POC demos confusing. Constrains
# customer behavior to SALES SCENARIO TERRITORY:
#   • In-scope: price objections, feature questions, durability concerns,
#     competitor comparisons, decision delays, "I need to think", "I want
#     a better discount", etc.
#   • Out-of-scope (suppressed): refund disputes, eviction threats, fraud
#     claims, account-locked scenarios, shipping problems, bug reports.
#
# Real customers DO sometimes raise these, but the rate is ~1-2% per session
# (we'd need separate mining of NotRelevant outcomes to quantify). For demo
# purposes, simulating these scenarios moves the conversation outside what
# the sales agent can address — not useful for showing sales-supervisor
# capability.
#
# This is a SIMULATOR CALIBRATION CONSTRAINT, not a sales-agent rule. The
# real product's escalation router (stage_escalation_router consuming
# prompt_co_pilot_escalation) handles out-of-scope cases when they DO occur
# in production. For the POC simulator, we keep conversations in sales lane.
SCOPE_CONSTRAINT_OVERLAY = """
## SIMULATOR SCOPE CONSTRAINT — stay in sales-scenario territory

This is a SALES conversation. The agent is here to help you decide whether
to buy / renew / not — that's the entire scope. The agent does NOT have
access to billing systems, refund tools, account-management, shipping
operations, or support escalation paths.

You MUST stay within sales-scenario behaviors. In-scope concerns you may
raise (these are interesting and realistic):
  • Price too high / wanting a better discount
  • Feature questions (specs, comparisons, fit for your use case)
  • Durability / build quality concerns
  • Competitor comparison ("I got a quote of X elsewhere")
  • Decision delays ("let me think", "I need to check with my partner")
  • Coverage scope, warranty terms, return policy
  • Skepticism about marketing claims

OUT-OF-SCOPE behaviors you must NOT raise (these are real but not what
sales agents can address, and not interesting for this demo):
  • Refund disputes for charges you didn't authorize
  • Eviction threats / financial crisis framing tied to the company's error
  • Account-locked / can't-log-in problems
  • Order-shipping problems on an already-completed purchase
  • Bug reports about the website / checkout
  • Demands for a phone number to a specific human / supervisor / manager
    (this triggers the agent's escalation rule and ends the session,
    which is a valid production behavior but not what we're testing)
  • Generic anger at the company unrelated to the sales decision

If your historical persona contains hints of such out-of-scope concerns,
keep them present but DO NOT escalate them. One brief mention is fine;
do not turn the conversation into a refund/billing/eviction crisis.

Your character can be skeptical, frustrated, demanding, walking away — all
within the SALES SCENARIO. The conversation should end with one of:
  • You buy / renew / agree to close
  • You decline politely / walk away from the offer
  • You stall ("let me think") — neither buy nor walk

NOT with: customer-service escalation, refund demand, fraud accusation,
or threat-of-eviction framing.
"""


GENERATE_SYSTEM = """You play the role of a SPECIFIC, REAL customer in an ongoing sales conversation.

You will be given:
  (a) The customer's psychographic profile (motivator, decision_logic, etc.)
  (b) An "Opportunity coherence constraint" describing the customer's actual
      buy-state context (cart-abandonment, renewal, etc.)
  (c) The FULL ORIGINAL CONVERSATION this real customer had with the agent —
      use this as your CHARACTER REFERENCE. It tells you how this customer
      actually speaks, what they actually pushed back on, what they revealed
      about themselves, what their real concerns were.
  (d) The LIVE conversation so far — which MAY have diverged from the original
      because the live agent is making different moves than the historical agent.
  (e) The agent's most recent LIVE message — this is what you respond to.

Your job is to respond to (e) AS THIS SPECIFIC CUSTOMER. Use (c) for character
grounding — match how this person actually talked, what they actually cared
about, how they actually reasoned. Stay consistent with (b) — your buy-state
context. Apply (a) — the persona axes.

DO NOT just copy the historical response. The live agent is making different
moves; respond to those, not to what the historical agent said.

CRITICAL LANGUAGE RULE: Use the SAME language as the most recent messages in
the LIVE conversation. Look at the agent's last live message and the prior live
turns:
- If they are in English, respond in English.
- If they are in Hebrew, respond in Hebrew.
- DO NOT switch languages mid-conversation. DO NOT translate.

Stay in character. Use ALL the profile axes:
- Skeptical trust → don't accept claims at face value; ask for proof
- Terse communication style → reply 1 sentence
- Prevention regulatory focus → frame concerns around what could go wrong / loss
- High budget sensitivity → push back on price even when it's reasonable
- High purchase urgency → ask about timing; low urgency → defer
- Primary resistance → return to that resistance often
- Objection pattern → reflect the typical objection style for this persona

Don't be overly cooperative if the profile says skeptical. Don't be overly long
if the style is terse. Reply 1-2 sentences typical."""


def _render_full_historical(historical_messages: list[dict] | None,
                              max_chars: int = 6000) -> str:
    """Format the full original conversation for character grounding.

    Empirical motivation: with only the last 8 live turns and a profile, the
    simulator was producing characterful but opp-incoherent responses (Karl
    saying 'I wasn't even shopping' for a cart_abandonment opp). The full
    historical conversation grounds the simulator in the REAL customer's
    voice — what they actually cared about, what they actually pushed back on.

    Bug fix (2026-05-01, T-78): when conversation exceeds max_chars, keep the
    LATER turns (closer to divergence) rather than the earliest greetings.
    Long Libra Renewal conversations (n_msgs 26-36) were losing their
    bargaining/objection tail — the simulator was character-grounded in
    greetings + opening quote only, missing the price-pushback evidence.
    """
    if not historical_messages:
        return "(no historical conversation provided)"
    # Iterate REVERSED so we accumulate the most-recent turns first; reverse
    # at end to restore chronological order.
    rev_lines = []
    total = 0
    truncated = False
    for m in reversed(historical_messages):
        text = (m.get("text") or "").strip()
        if not text:
            continue
        role = "Customer" if m.get("direction") == "inbound" else "Agent"
        line = f"{role}: {text}"
        if total + len(line) > max_chars:
            truncated = True
            break
        rev_lines.append(line)
        total += len(line)
    lines = list(reversed(rev_lines))
    if truncated and lines:
        lines.insert(0, "[... earlier greetings/intro turns truncated ...]")
    if not lines:
        return "(historical conversation empty)"
    return "\n".join(lines)


def _slice_historical_to_turn(historical_messages: list[dict] | None,
                                agent_turn_index: int) -> list[dict] | None:
    """v2 same-turn-only guardrail.

    Return historical messages from start up to (and INCLUDING) the
    agent_turn_index'th historical agent message — but NOT the customer reply
    that came after it. That reply is surfaced separately as the "real
    same-turn reply" reference; keeping it out of the character-grounding
    block prevents the simulator from seeing how the original conversation
    eventually resolved beyond this turn.
    """
    if not historical_messages:
        return historical_messages
    out: list[dict] = []
    agent_seen = 0
    target = agent_turn_index + 1  # we want exactly this many agent messages
    for m in historical_messages:
        text = (m.get("text") or "").strip()
        if not text:
            # preserve whitespace-only entries' positions by skipping the same
            # way _render_full_historical does
            continue
        if m.get("direction") == "outbound":
            if agent_seen >= target:
                break
            out.append(m)
            agent_seen += 1
            if agent_seen >= target:
                # stop AFTER the target agent message, before any subsequent
                # customer reply
                break
        else:
            if agent_seen < target:
                out.append(m)
            else:
                break
    return out


async def generate_persona_reply(opp_meta: dict, dialog_history: list[dict],
                                  live_agent_msg: str,
                                  forced_language: str | None = None,
                                  historical_messages: list[dict] | None = None,
                                  agent_turn_index: int | None = None,
                                  same_turn_real_reply: str | None = None) -> str:
    # 10-axis persona — extends the original 6 axes with the 4 fields already
    # fetched by db.py:fetch_opp_meta but previously unused. Widens persona
    # vocabulary without changing the scenario pool.
    profile_summary = (
        f"- Motivator: {opp_meta.get('primary_motivator')}\n"
        f"- Decision logic: {opp_meta.get('decision_logic')}\n"
        f"- Trust level: {opp_meta.get('trust_level')}\n"
        f"- Communication style: {opp_meta.get('communication_style')}\n"
        f"- Objection pattern (typical for this customer): {opp_meta.get('objection_pattern') or 'none'}\n"
        f"- Emotional volatility: {opp_meta.get('emotional_volatility') or 'unknown'}\n"
        f"- Regulatory focus: {opp_meta.get('regulatory_focus') or 'unknown'} "
        f"(prevention=loss-averse / promotion=gain-seeking)\n"
        f"- Budget sensitivity: {opp_meta.get('budget_sensitivity') or 'unknown'}\n"
        f"- Purchase urgency: {opp_meta.get('purchase_urgency') or 'unknown'}\n"
        f"- Primary resistance: {opp_meta.get('primary_resistance') or 'none'}"
    )

    # T-81: Anchor enrichment — closes the unanchored-simulator loop. Without
    # economic reference points (last year's price, market avg, competitor
    # offers), the simulator generates training-distribution-driven adversarial
    # pushback regardless of whether the agent's offer is reasonable. With
    # anchors, the simulator negotiates from a real reference frame: accept
    # if at-or-below market; pushback if above; recognize match-to-competitor
    # as a successful close move (not as suspicious price-drop).
    anchor_block = ""
    anchors = opp_meta.get("anchors") or {}
    if anchors:
        a_lines = ["", "## YOUR economic reference frame as this customer (use it; you genuinely know these numbers)"]
        if anchors.get("last_year_price_nis"):
            a_lines.append(f"- Last year you paid: {anchors['last_year_price_nis']} NIS for similar coverage.")
        if anchors.get("market_avg_for_segment_nis"):
            a_lines.append(f"- You've heard market average for your vehicle is around {anchors['market_avg_for_segment_nis']} NIS.")
        if anchors.get("actual_market_yoy_change_pct") is not None:
            a_lines.append(f"- You suspect prices haven't actually risen much this year ({anchors['actual_market_yoy_change_pct']:+.1f}% real change in the broader market). Any agent claim of a large hike (e.g. 'prices went up 48%') sounds inflated.")
        a_lines.append("")
        a_lines.append("## How these anchors shape your reactions")
        a_lines.append("- If the agent's offered price is ABOVE your market_avg by >10%, push back hard.")
        a_lines.append("- If the agent quotes a DIFFERENT price each turn (staircase pricing — 2,850 → 2,565 → 2,350), you become SUSPICIOUS of the original quote, not pleased: 'why didn't you offer this from the start? what's the catch?'")
        a_lines.append("- If the agent matches a competitor offer or hits market_avg, accept naturally — that's a fair deal, not a victory to extract more from.")
        a_lines.append("- If the agent gracefully retreats ('let me know when you're ready'), feel respected — your engagement stays warm even if you don't sign immediately.")
        a_lines.append("- Don't generate adversarial pushback when the offer is at-or-below your reference frame. A real customer accepts a fair price.")
        anchor_block = "\n" + "\n".join(a_lines) + "\n"

    last_turns = dialog_history[-8:]
    dialog_text = "\n".join(
        f"{('Customer' if m.get('role')=='customer' else 'Agent')}: {(m.get('text') or '')[:300]}"
        for m in last_turns if m.get("text")
    )

    lang_instruction = ""
    if forced_language == "en":
        lang_instruction = "\n\nFORCED LANGUAGE: respond in English ONLY. Do NOT use Hebrew."
    elif forced_language == "he":
        lang_instruction = "\n\nFORCED LANGUAGE: respond in Hebrew ONLY."

    coherence_constraint = render_opp_coherence_constraint(opp_meta)
    # v2 same-turn-only guardrail: when an explicit same-turn real reply has
    # been passed in (POC_SIM_V2_REFERENCE=on path), slice the character-
    # grounding historical so it includes the historical conversation only up
    # to (and including) the agent message at this turn-index — never the
    # future. The customer reply at this turn is surfaced separately as the
    # explicit reference block below, so it's not duplicated here.
    if same_turn_real_reply is not None and agent_turn_index is not None:
        sliced_historical = _slice_historical_to_turn(historical_messages, agent_turn_index)
    else:
        sliced_historical = historical_messages
    historical_block = _render_full_historical(sliced_historical)

    # v2 reference block — the historical customer reply at this same turn-
    # index, as guidance for posture/tone/priorities. Adapted (not copied)
    # whenever the live agent has diverged from the original move.
    reference_block = ""
    if same_turn_real_reply:
        reference_block = (
            "\n## REAL same-turn customer reply (REFERENCE — what the real customer "
            "ACTUALLY said in response to a similar agent move at this same point in "
            "the original conversation)\n"
            f"\"{same_turn_real_reply}\"\n\n"
            "## How to use the reference\n"
            "- Lean on it for POSTURE (warm / skeptical / cold), TONE (terse / "
            "elaborate / emotional), and PRIORITIES (what the customer chose to "
            "push on, accept, or ignore).\n"
            "- ADAPT the content to the live agent's actual most-recent message — if "
            "the live agent has diverged (different number, different framing, "
            "different question), your reply must respond on-topic to what was "
            "ACTUALLY said live, not to the original agent move.\n"
            "- If the live agent's move is very close to the original, your reply "
            "should be very close to the reference (rephrased into your voice).\n"
            "- If the live agent's move has diverged significantly, keep the "
            "reference's POSTURE while responding on-topic to the live message.\n"
            "- Never invent information that neither the reference nor the live "
            "agent's message has put on the table.\n"
        )
    # T-84: voice profile extracted from historical messages — gives the
    # simulator the customer's actual phrasing/register/decisiveness markers.
    from voice_profile import render_voice_block
    voice_block = render_voice_block(opp_meta.get("voice_profile"))

    is_post_match = _post_match_detected(live_agent_msg, dialog_history)
    # 2026-05-10 — Post-match calibration. When the agent's most recent message
    # looks like a price-match delivery, append the empirical-distribution
    # overlay so the simulator doesn't over-produce trust_grievance / multi-year
    # guarantee fixation (real rate is 0.1%; without this overlay hard-mode
    # produces it as the modal response). Hard mode is SUPPRESSED when post-match
    # is detected (post-match empirical distribution overrides general
    # stubbornness — real customers don't dig in further once their target is met).
    if is_post_match:
        log.info("simulator: POST_MATCH_OVERLAY injected (hard_customer=%s, suppressed)",
                 _hard_customer_enabled(opp_meta))
        hard_block = ""
        post_match_block = POST_MATCH_OVERLAY
    else:
        hard_block = HARD_CUSTOMER_OVERLAY if _hard_customer_enabled(opp_meta) else ""
        post_match_block = ""

    user = f"""## Customer profile
{profile_summary}{anchor_block}{voice_block}
{coherence_constraint}
## Original conversation (CHARACTER REFERENCE — this is who you really are; how you actually talk and what you actually care about)
{historical_block}
{reference_block}
## Live conversation so far (the agent has been making different moves than the original; this is what's actually happened in this run)
{dialog_text}

## Agent's most recent LIVE message — respond to THIS, not to the historical agent's message
{live_agent_msg}{lang_instruction}{hard_block}{post_match_block}{SCOPE_CONSTRAINT_OVERLAY}

Your reply (as this customer, in their voice from the historical reference, responding to the LIVE agent's message above):"""
    # T-84: per-turn temperature jitter — varies sampling temperature within
    # [0.5, 0.9] reproducibly per (opp_id, turn). Real customers exhibit
    # within-conversation variability; fixed temperature 0.7 collapsed this
    # and was contributing to over-uniform simulator responses.
    from voice_profile import stable_temperature
    turn_idx = len(dialog_history)
    temp = stable_temperature(opp_meta.get("id") or "unknown", turn_idx)
    import time as _time
    _t0 = _time.monotonic()
    msg = await _client.messages.create(
        model=SIMULATOR_MODEL,
        max_tokens=300,
        temperature=temp,
        system=GENERATE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    response_text = msg.content[0].text.strip() if msg.content else "ok"
    # T-86 trace
    try:
        from trace_logger import TraceLogger
        trace = TraceLogger.current()
        if trace:
            trace.llm(
                stage="simulator.persona_reply",
                provider="anthropic",
                model=SIMULATOR_MODEL,
                system=GENERATE_SYSTEM,
                user=user,
                response=response_text,
                latency_ms=int((_time.monotonic() - _t0) * 1000),
                input_tokens=getattr(msg.usage, "input_tokens", 0) if hasattr(msg, "usage") else 0,
                output_tokens=getattr(msg.usage, "output_tokens", 0) if hasattr(msg, "usage") else 0,
                extra={"temperature": temp, "turn_idx": turn_idx,
                       "forced_language": forced_language,
                       "hard_customer": _hard_customer_enabled(opp_meta),
                       "v2_reference_used": same_turn_real_reply is not None,
                       "agent_turn_index": agent_turn_index},
            )
    except Exception:
        pass
    return response_text


# ── Public API ─────────────────────────────────────────────────────────────
class CustomerSimulator:
    """One simulator instance per session, shared between left + right panels.
    Caches similarity decisions per turn-index for parallel-call consistency."""

    def __init__(self, opp_meta: dict, historical_messages: list[dict]):
        self.opp_meta = opp_meta
        # Keep the FULL historical conversation for character-grounding the
        # GENERATE path. (Empirically: opp-coherence + persona alone weren't
        # enough; the simulator was making up character-traits inconsistent
        # with the real customer's actual voice.)
        self.historical_messages = historical_messages
        # Pre-extract per-turn historical structure
        self.historical_agent_at_turn: dict[int, str] = {}
        self.historical_customer_at_turn: dict[int, str] = {}
        agent_idx = 0
        customer_idx = 0
        for m in historical_messages:
            text = (m.get("text") or "").strip()
            if not text:
                continue
            if m.get("direction") == "outbound":
                self.historical_agent_at_turn[agent_idx] = text
                agent_idx += 1
            else:
                self.historical_customer_at_turn[customer_idx] = text
                customer_idx += 1
        self.n_historical_agent = agent_idx
        self.n_historical_customer = customer_idx
        # Detect language of seed/historical conversation (first 6 messages = "early conversation")
        seed_texts = [m.get("text") or "" for m in historical_messages[:6]]
        self.seed_language = detect_language(seed_texts)

    async def reply(self, dialog_history: list[dict], live_agent_msg: str,
                    agent_turn_index: int) -> tuple[str, str]:
        """Generate the customer's reply.

        v1 dispatch (default): hybrid Option C — rephrase vs generate by
        agent-similarity threshold.
        v2 dispatch (POC_SIM_V2_REFERENCE=on): unified reference-aware generate;
        always surfaces the same-turn historical reply as explicit reference,
        with same-turn-only character grounding (no future leakage).

        Args:
          dialog_history: full session dialog so far (both panels share this view per panel)
          live_agent_msg: the agent message just emitted
          agent_turn_index: which agent turn this corresponds to (0-indexed in this session)

        Returns:
          (customer_reply_text, mode) where mode in {"rephrase", "generate",
          "generate_v2_ref", ...}
        """
        # Determine forced language: clamp to seed language if it was clearly one or the other
        forced_lang = self.seed_language if self.seed_language in ("en", "he") else None

        v2_ref_on = os.getenv("POC_SIM_V2_REFERENCE", "off").lower() == "on"

        if v2_ref_on:
            # v2: always reference-aware generate. Pass same-turn historical
            # customer reply (may be None if we've run off the end of historical;
            # then the path degrades gracefully to plain persona-generate).
            same_turn = self.historical_customer_at_turn.get(agent_turn_index)
            text = await generate_persona_reply(
                self.opp_meta, dialog_history, live_agent_msg,
                forced_language=forced_lang,
                historical_messages=self.historical_messages,
                agent_turn_index=agent_turn_index,
                same_turn_real_reply=same_turn,
            )
            mode = "generate_v2_ref" if same_turn else "generate_v2_no_ref"
            return text, mode

        historical_agent = self.historical_agent_at_turn.get(agent_turn_index)
        if historical_agent is None:
            text = await generate_persona_reply(
                self.opp_meta, dialog_history, live_agent_msg,
                forced_language=forced_lang,
                historical_messages=self.historical_messages)
            return text, "generate"

        sim = await similarity(live_agent_msg, historical_agent)
        if sim >= SIMILARITY_THRESHOLD:
            historical_customer = self.historical_customer_at_turn.get(agent_turn_index)
            if historical_customer:
                # If historical reply is in different language than seed, fall through to generate
                hist_lang = detect_language([historical_customer])
                if forced_lang and hist_lang != "unknown" and hist_lang != forced_lang:
                    text = await generate_persona_reply(
                        self.opp_meta, dialog_history, live_agent_msg,
                        forced_language=forced_lang,
                        historical_messages=self.historical_messages)
                    return text, "generate (lang-mismatch fallback)"
                text = await rephrase_customer_reply(
                    historical_customer, live_agent_msg, forced_language=forced_lang)
                return text, "rephrase"
        text = await generate_persona_reply(
            self.opp_meta, dialog_history, live_agent_msg,
            forced_language=forced_lang,
            historical_messages=self.historical_messages)
        return text, "generate"


# ── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def smoke():
        opp_meta = {
            "company": "Libra",
            "primary_motivator": "Price/savings",
            "decision_logic": "Analytical",
            "trust_level": "Skeptical",
            "communication_style": "Terse",
            "objection_pattern": "Comparing competitor prices",
            "emotional_volatility": "Low",
        }
        # Mock historical messages
        historical = [
            {"direction": "outbound", "text": "Hey, this is Nofar from Libra. Your renewal: 4,238 NIS for the year."},
            {"direction": "inbound", "text": "Other insurer offered me 3,800 NIS."},
            {"direction": "outbound", "text": "Want me to lock in the rate?"},
            {"direction": "inbound", "text": "I'll think about it."},
        ]
        sim = CustomerSimulator(opp_meta, historical)

        # Path matches: agent says basically the same thing as history's first agent msg
        dialog = [{"role":"agent","text":"Hi, this is Nofar from Libra. Your renewal price: 4,238 NIS for the year.","sequence_number":0}]
        reply, mode = await sim.reply(dialog, dialog[0]["text"], agent_turn_index=0)
        print(f"=== TURN 0 (path-match expected) — mode={mode} ===")
        print(f"  customer reply: {reply}")
        print()

        # Path diverged: agent asks for breakdown (different from history)
        dialog2 = [
            {"role":"agent","text":"Hey, this is Nofar from Libra. Your renewal: 4,238 NIS.","sequence_number":0},
            {"role":"customer","text":"Other insurer offered 3,800.","sequence_number":1},
            {"role":"agent","text":"Got it — could you share the breakdown of their offer (mandatory vs comprehensive)?","sequence_number":2},
        ]
        reply, mode = await sim.reply(dialog2, dialog2[2]["text"], agent_turn_index=1)
        print(f"=== TURN 1 (path-diverged expected) — mode={mode} ===")
        print(f"  customer reply: {reply}")

    asyncio.run(smoke())
