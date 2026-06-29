"""Mode 1b — full supervisor with live Context Graph retrieval.

Per turn:
  1. Build a CG query from current opp state (motivator, objection, recent dialog)
  2. Call Agent Context Graph (LightRAG) `/query/data` → entities, relations, chunks
  3. Build supervisor prompt with: opp state + dialog + CG entities + persona
  4. Call Sonnet 4.5 supervisor → Strategic Directive JSON (full schema)
  5. Return directive

This is the DEMO mode that shows the maximum of the architecture's power.
Mode 1a v1 (compiled playbook) ships in production for cost; Mode 1b ships for
the demo to show the upper bound of supervisor reasoning.

Reference: runners/batch9_minirun.py (the original 60-trial PoC validated this
exact pipeline at 100% parseability, 98.6% groundedness).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import anthropic
import httpx

log = logging.getLogger(__name__)

CG_URL = "http://18.153.178.170:9621"
CG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
SUPERVISOR_MODEL = "claude-sonnet-4-5-20250929"
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Opt-in: route Mode 1b retrieval through CG's `/query/auto` (server picks the
# best mode) instead of fixed `mode='hybrid'` on `/query/data`. Useful for A/B
# comparing our hand-rolled tier router against the server's auto classifier.
USE_QUERY_AUTO = os.environ.get("POC_USE_QUERY_AUTO", "0") == "1"

_anthropic = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


# ── In-memory query cache (per AGENT_INTEGRATION.md recommendation) ──────────
# 5-min TTL, capped at 256 entries. Customer simulator inertia means many turns
# query similar things — cache cuts cost and latency materially. Per-process
# only (no Redis dependency); good enough for a single-instance POC.
_QUERY_CACHE: dict[tuple, tuple[float, dict]] = {}
_QUERY_CACHE_TTL_S = 300  # 5 minutes
_QUERY_CACHE_MAX = 256
QUERY_CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0}

# Per-process counters for each CG endpoint we hit. Cumulative across sessions;
# replayer takes a snapshot at session start and computes delta at session end
# to surface "CG calls this session" in the architecture banner.
CG_ENDPOINT_CALLS = {
    "query_data": 0,         # POST /query/data — Mode 1b retrieval
    "query_data_cached": 0,  # cache hit on /query/data (no network call)
    "query_auto": 0,         # POST /query/auto — opt-in observability
    "cgr3": 0,               # POST /cgr3/query — Mode 2 multi-hop
    "decisions_read": 0,     # GET /graph/decisions — start-of-session precedent fetch
    "decision_emit": 0,      # POST /graph/decision/emit — end-of-session write
}


def _cache_get(key: tuple) -> dict | None:
    rec = _QUERY_CACHE.get(key)
    if rec is None:
        return None
    inserted_at, value = rec
    if (time.monotonic() - inserted_at) > _QUERY_CACHE_TTL_S:
        _QUERY_CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: tuple, value: dict) -> None:
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
        # Evict the oldest entry — O(n) but n <= 256
        oldest = min(_QUERY_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _QUERY_CACHE.pop(oldest, None)
        QUERY_CACHE_STATS["evictions"] += 1
    _QUERY_CACHE[key] = (time.monotonic(), value)


def reset_query_cache() -> None:
    """Test/diagnostic helper — clear cache + stats between sessions if desired."""
    _QUERY_CACHE.clear()
    QUERY_CACHE_STATS.update({"hits": 0, "misses": 0, "evictions": 0})


# ── Context Graph retrieval ─────────────────────────────────────────────────
async def cg_query_data(http_client: httpx.AsyncClient, workspace: str, query: str,
                         top_k: int = 20) -> dict:
    """Call CG /query/data endpoint to retrieve entities + relations + chunks.

    Note: LightRAG's response is wrapped in a 'data' envelope:
      {"status": "success", "data": {"entities": [...], "relationships": [...], "chunks": [...]}}
    We unwrap so callers get the flat shape directly.

    Caches successful responses for 5 min keyed on (workspace, query, top_k, mode).
    Per AGENT_INTEGRATION.md recommendation — frequent queries dominate the load.
    """
    if not CG_API_KEY:
        return {"entities": [], "relationships": [], "chunks": [],
                "_error": "LIGHTRAG_API_KEY not set"}
    import time as _time
    _t0 = _time.monotonic()
    cache_key = ("query_data", workspace, query, top_k, "hybrid")
    cached = _cache_get(cache_key)
    if cached is not None:
        QUERY_CACHE_STATS["hits"] += 1
        CG_ENDPOINT_CALLS["query_data_cached"] += 1
        # Annotate response so callers can surface "served from cache" if useful
        out = dict(cached)
        out["_cache_hit"] = True
        # T-86 trace
        try:
            from trace_logger import TraceLogger
            trace = TraceLogger.current()
            if trace:
                trace.cg(endpoint="/query/data", workspace=workspace,
                         query=query, response=out, cache_hit=True,
                         latency_ms=int((_time.monotonic() - _t0) * 1000))
        except Exception:
            pass
        return out
    QUERY_CACHE_STATS["misses"] += 1
    CG_ENDPOINT_CALLS["query_data"] += 1
    response = None
    try:
        response = await http_client.post(
            f"{CG_URL}/query/data",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": workspace},
            json={"query": query, "top_k": top_k, "mode": "hybrid"},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and "data" in body and isinstance(body["data"], dict):
            payload = body["data"]
        else:
            payload = body
        # Strip unused fields BEFORE caching so the cache holds the lean shape
        # (saves memory, reduces token count when this payload is embedded in
        # Mode 1b's user prompt, shrinks trace JSON). 2026-05-03 measurement:
        # raw payload ~110KB, stripped ~17KB on a representative Ecommerce query.
        if isinstance(payload, dict):
            payload = _strip_cg_payload(payload)
        # Only cache successful responses with at least one of the three buckets
        if isinstance(payload, dict) and (payload.get("entities")
                                          or payload.get("relationships")
                                          or payload.get("chunks")):
            _cache_put(cache_key, payload)
        # T-86 trace
        try:
            from trace_logger import TraceLogger
            trace = TraceLogger.current()
            if trace:
                trace.cg(endpoint="/query/data", workspace=workspace,
                         query=query, response=payload,
                         latency_ms=int((_time.monotonic() - _t0) * 1000))
        except Exception:
            pass
        return payload
    except Exception as e:
        err = _format_http_error(e, response)
        log.warning("CG query failed for workspace=%s: %s", workspace, err)
        return {"entities": [], "relationships": [], "chunks": [], "_error": err}


async def cg_query_auto(http_client: httpx.AsyncClient, workspace: str, query: str,
                         top_k: int = 20) -> dict:
    """Call CG /query/auto — server picks the best mode (naive/local/global/hybrid/mix).

    Returns the standard /query response shape PLUS:
      selected_mode: server-chosen mode
      mode_reason:   server's rationale (short string)
      latency_ms:    server-reported latency

    Note: this endpoint synthesizes via LLM (returns 'response' string), unlike
    /query/data which only retrieves. Use this when we want an LLM-cooked answer
    rather than raw entities for prompt-side composition.
    """
    if not CG_API_KEY:
        return {"response": "", "_error": "LIGHTRAG_API_KEY not set"}
    cache_key = ("query_auto", workspace, query, top_k)
    cached = _cache_get(cache_key)
    if cached is not None:
        QUERY_CACHE_STATS["hits"] += 1
        out = dict(cached)
        out["_cache_hit"] = True
        return out
    QUERY_CACHE_STATS["misses"] += 1
    CG_ENDPOINT_CALLS["query_auto"] += 1
    response = None
    try:
        response = await http_client.post(
            f"{CG_URL}/query/auto",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": workspace},
            json={"query": query, "top_k": top_k, "include_references": True,
                   "only_need_context": False},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        payload = body.get("data") if isinstance(body, dict) and "data" in body else body
        if isinstance(payload, dict) and (payload.get("response") or payload.get("entities")):
            _cache_put(cache_key, payload)
        return payload or {}
    except Exception as e:
        err = _format_http_error(e, response)
        log.warning("CG query/auto failed for workspace=%s: %s", workspace, err)
        return {"response": "", "_error": err}


def build_cg_query(opp_meta: dict, dialog: list[dict]) -> str:
    """Build a focused CG query from current opp state + dialog tail."""
    motivator = opp_meta.get("primary_motivator") or ""
    objection = opp_meta.get("objection_pattern") or ""
    # Last customer message — most recent objection/intent signal
    last_customer = next(
        (m.get("text", "") for m in reversed(dialog)
         if m.get("role") == "customer"), "")
    parts = [
        opp_meta.get("company") or "",
        motivator, objection,
        "lost deal pattern", "winning conversation pattern",
        last_customer[:200],
    ]
    return " ".join(p for p in parts if p)


# ── Supervisor prompt ───────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are the Strategic Supervisor for the system's AI sales agent.

Given the current state of a sales conversation, emit a Strategic Directive as JSON
that tells the conversation agent what strategic moves to make, which facts to anchor,
what to avoid, and what tone to take.

Your reasoning MUST follow this order — enforced by JSON field ordering:
1. **Signal analysis** — enumerate observable signals in the customer's latest
   turn + dialogue tail and assign per-signal confidence. Bind strategy to evidence.
2. **Stage** — current → target stage with rationale.
3. **Strategy** — primary primitive, justified by the primary_signal AND
   consistent with the active plan phase's preferred actions when one is set.
4. **Knowledge / facts / rules** — grounded in CG entities.

The model that fills later fields conditions on the signal_analysis it just
produced. Maps to two expert-system reasoning gaps (per
`reports/AGENT-AS-EXPERT-SYSTEM.md`):
- **Gap #4 (Confidence / certainty factors)** — per-signal confidence + a
  primary_signal whose confidence ≥ 0.5; downstream strategy carries forward
  evidence-grounded certainty instead of opaque LLM hunch.
- **Gap #5 (Counterfactual reasoning)** — the `counterfactual` field forces
  the LLM to verify its strategy survives perturbation of the primary signal.

Output ONLY a JSON object with this shape (no prose, no markdown fences):
{
  "signal_analysis": {
    "observed_signals": [
      {"signal": "<one of: explicit_price_request|explicit_objection_price|explicit_objection_trust|explicit_objection_timing|explicit_objection_competition|explicit_objection_product_fit|commitment_signal|pace_request|disengagement|trust_indicator|competing_offer_mention|pain_articulation|product_inquiry|silence|other>",
       "evidence_quote": "<exact phrase from customer's last message or recent dialogue tail; '' if signal=silence>",
       "confidence": 0.0-1.0}
    ],
    "primary_signal": "<one of the signal values above; must have confidence >= 0.5>",
    "strategy_implication": "<one sentence: what the primary signal demands the agent do next>",
    "counterfactual": "<one sentence: would the same strategy still apply if the primary signal were absent? Yes/No + reason>",
    "plan_alignment": "<consistent_with_plan|conflicts_with_plan|plan_silent — does strategy_implication align with the active plan phase's preferred actions?>"
  },
  "stage": {"current": "<stage_name>", "target": "<stage_name>", "confidence": 0.0-1.0,
            "rationale": "<one sentence>"},
  "strategy": {"primary": "<one of: information|direct_ask|scarcity|objection_handling|reciprocity|re_engagement|logistics|empathy|authority|commitment|social_proof>",
               "secondary": "<same enum or null>",
               "tone": "<friendly|professional|urgent|authoritative>",
               "cialdini_levers": ["<authority|social_proof|reciprocity|commitment|scarcity|liking>"],
               "concrete_move": "<Tier 2 — see CONCRETE MOVES section below; null if no concrete decision needed this turn>"},
  "knowledge": {"facts_to_anchor": [{"text": "...", "source_ref": "<cg-entity-name>"}],
                "must_not_say": [{"text": "...", "reason": "..."}],
                "objection_handling": {"objection": "...", "response_template": "..."} or null},
  "customer_state": {"current_commitment_level": <0-5 or null>,
                     "motivator": "...",
                     "detected_objection": "<price|trust|timing|competition|shipping|quality|other|none>"},
  "confidence": {"overall": 0.0-1.0, "band": "<high|medium|low>"},
  "audit": {"rationale_summary": "<paragraph operator-readable>"},
  "rules_to_enforce": ["<business rule IDs like B29, B34>"],
  "plan": {"phase_status": "<in_progress|exit_met|should_advance|aborted>",
            "current_phase_id": <int or null>,
            "recommended_next_phase_id": <int or null — set if advancing>,
            "active_branch": "<branch_id or null>",
            "captured_this_turn": {"<must_capture_key>": "<value>"} or {},
            "rationale": "<one sentence: why advancing / staying>"}
}

ANCHOR-AWARE PRICING RULE (T-81):
- If the user prompt includes a `## Customer's Economic Reference Frame` block,
  treat its values as authoritative pricing context.
- If your last quoted price is at-or-below `market_avg_for_segment_usd` OR
  at-or-below a competitor offer the customer has mentioned, the
  `primary_strategy` MUST NOT be a price-drop. Pick `authority`,
  `objection_handling`, `social_proof`, or `empathy` to address the objection
  via coverage value or relationship — NOT via further discount.
- Multiple successive price drops in one conversation cause "why didn't you
  offer this from the start?" trust collapse. Empirically validated as a
  failure mode (2026-05-02). Hold the price; address objections elsewhere.
- If `claimed_increase_pct` exceeds `actual_market_yoy_change_pct + 5`, the
  agent's earlier seed quote was likely fabricated. Note this in
  `audit.rationale_summary`; don't repeat the fabrication.

PROFILE-CONDITIONED OPENING ANCHOR (2026-05-14, anti anchor-exposure
backfire):
- The anchor block may include `profile_appropriate_opening_usd` — this is
  the highest defensible opening price for THIS customer's profile (loyalty
  + claims). It is computed as ly_price × small inflation, NOT list price.
- If the agent's earliest in-dialog quote exceeds
  `profile_appropriate_opening_usd × 1.15`, the customer's profile did NOT
  justify that anchor. When this customer challenges the price, you MUST:
   1. Treat the seed quote as a misanchor (not as a position to defend).
   2. Bridge to `profile_appropriate_opening_usd` (or lower) with EXPLICIT
      procedural framing — e.g., "after looking at your {N}-year history with
      zero claims I can authorize {profile_appropriate_opening_usd}" — NOT
      a feature pivot.
   3. NEVER claim "we can't match X" when the literal numeric anchors prove
      you can. That triggers the Galinsky anchor-exposure backfire pattern:
      large hidden margin reveal → trust collapse → recovery is much harder.
- For loyal+no-claims customers (`loyalty_years≥2` AND `claims_count==0`),
  the FIRST agent price-quote should be at-or-below
  `profile_appropriate_opening_usd`. Anything higher requires a clear
  profile-mismatch reason in `audit.rationale_summary`.

BINDING RULES (signal_analysis ↔ strategy):
- `strategy.primary` MUST be supported by `signal_analysis.strategy_implication`.
  If the primary signal is `explicit_price_request`, the strategy must address
  price (typically reciprocity, direct_ask, or logistics — not information).
  If the primary signal is `explicit_objection_trust`, the strategy must build
  authority or empathy — not push for commitment.
- If `signal_analysis.plan_alignment == "conflicts_with_plan"`, then
  `audit.rationale_summary` MUST contain a one-sentence justification for the
  deviation (e.g. "customer disengaged → safety-net empathy outranks plan's
  reciprocity push"). Otherwise honor the plan's preferred actions.
- A signal with confidence < 0.5 MUST NOT be selected as `primary_signal`. If
  no signal clears 0.5, set `primary_signal: "silence"` and use a low-stakes
  observational strategy.
- Always include at least one signal in `observed_signals` even if all are
  low-confidence — the empty-signal case is signal=silence, confidence=1.0.

If a Cluster Plan is provided in your context (under "## Cluster Plan"), you MUST
populate the `plan` field. Use the plan's phase exit signals — advance when met,
stay when not, force-advance when budget exhausted. Honor session_constraints.
If no plan is provided, set "plan" to {} or omit it.

Empirical patterns to respect (learned from Insurance Insurance Renewal data):
- Price/savings-motivated customers: prefer information, logistics, direct_ask. AVOID scarcity (60.4% vs 71.3% for information).
- Necessity-motivated customers: most strategies hit 80%+ — optimize for shortest path to close.
- Scarcity in turns 1-2 wins 73.9%; in turn 11+ wins 60.3%. Use early.
- Direct_ask in late turns when commitment_level >= 4: 77.7% win rate.
- Liking is most-deployed Cialdini lever (47% of messages) but lowest-lift (+4.9pp). Social_proof is rare (1.3%) but high-lift (+8.1pp).

Persona axes (from research, ±29 pp swings on same strategy):
- decision_logic=Analytical: probe for concrete numbers; don't be deferential.
- decision_logic=Impulsive: reciprocity, social proof, time-bounded offers work.
- trust_level=Skeptical: avoid direct_ask; build authority slowly.

Ground every fact_to_anchor in a CG entity provided in the context.
Output JSON only. No code fences.

## ⚠ LANGUAGE — MANDATORY (overrides all other instructions)

Detect the language of the customer's MOST RECENT message in the dialog
context (the latest entry where role="customer"). The directive's
`must_say` items, `signal_analysis.response_template`, and any free-form
text MUST be in that same language. This rule applies REGARDLESS of:

- Historical dialog turns being in a different language (the customer
  may have switched mid-conversation; honor the switch immediately)
- The tenant's typical-customer language (e.g. Insurance serves Israeli
  customers but the user may type in English; do NOT default to Hebrew)
- The script-template language (templates are in English; TRANSLATE them
  preserving intent if the customer is in another language)
- Your own training-default language preferences

If the customer's most recent message is in English, the directive's
`must_say` MUST be in English. If Hebrew, MUST be in Hebrew. Do NOT
mix languages within a single must_say entry.

{CONCRETE_MOVES_SECTION}
"""

# Phase 2 — Tier 2 catalog is now TENANT-AWARE. Strip the placeholder from
# the static SUPERVISOR_SYSTEM and inject per-tenant moves at call time via
# build_supervisor_system_for_tenant() below. Tenant-specific moves
# (e.g. mention_librot_credit for Insurance; bundle_artist_shell_upgrade for
# Ecommerce) are now in `data/concrete_moves/{tenant}.yaml`.
SUPERVISOR_SYSTEM = SUPERVISOR_SYSTEM.replace("{CONCRETE_MOVES_SECTION}", "")
_SUPERVISOR_SYSTEM_BASE = SUPERVISOR_SYSTEM  # frozen template


def build_supervisor_system_for_tenant(tenant: str | None) -> str:
    """Build the Mode 1b system prompt with the tenant-specific Tier 2 catalog
    appended. Phase 2: tenant catalogs live in YAML, loaded by
    `concrete_moves_loader`. Returns a fresh string per call (caller decides
    whether to cache)."""
    try:
        from concrete_moves_loader import (
            render_moves_for_supervisor_prompt as render_phase2,
        )
        moves_section = render_phase2(tenant)
    except ImportError:
        moves_section = ""
    if not moves_section:
        return _SUPERVISOR_SYSTEM_BASE
    return _SUPERVISOR_SYSTEM_BASE + "\n\n" + moves_section


def render_dialog_snippet(dialog: list[dict], k: int = 8) -> str:
    msgs = dialog[-k:]
    out = []
    for m in msgs:
        role = "Customer" if m.get("role") == "customer" else "Agent"
        text = (m.get("text") or "").replace("\n", " ").strip()[:250]
        if text:
            out.append(f"{role}: {text}")
    return "\n".join(out) if out else "(no dialogue)"


# Whitelisted field sets — anything outside these gets stripped from CG
# responses before they're cached / embedded in prompts. Saves ~80% of the
# payload (the bulk is `source_id` on relationships, ~4-5KB per item × 19
# items per query). Audited 2026-05-03 — these are the only fields read by
# `render_cg_summary`, `classify_cg_evidence`, and downstream consumers.
_CG_KEEP_ENTITY_FIELDS = {"entity_name", "entity_type", "description"}
_CG_KEEP_RELATIONSHIP_FIELDS = {
    "src_id", "tgt_id", "description", "file_path", "relation_context",
}
_CG_KEEP_CHUNK_FIELDS = {"content", "file_path", "chunk_id"}


def _strip_cg_payload(payload: dict,
                      *, entity_desc_max: int = 400,
                      relation_desc_max: int = 400,
                      chunk_content_max: int = 1500,
                      chunks_with_content: int = 3,
                      ) -> dict:
    """Strip unused fields from a /query/data response. Mutates a shallow
    copy of `payload`. Whitelist + truncate based.

    Bloat sources removed (measured on Ecommerce H1H query 2026-05-03):
      - relationships[].source_id  (~4467 chars × 19 = ~85KB → 0)
      - entities[].source_id       (~244 chars × 33 = ~8KB → 0)
      - entities[].file_path       (~234 × 33 = ~7.7KB → 0)
      - created_at everywhere
      - relationships[].keywords + .weight (unused)
      - references list (unused; ~1.5KB)

    Bloat sources truncated (render_cg_summary already truncates further at
    consumption — these caps are loose upper bounds well above any
    downstream truncation):
      - entities[].description: max {entity_desc_max} chars (was avg 716,
        max 3081). Saves ~14KB on a typical 33-entity payload.
      - relationships[].description: max {relation_desc_max} chars.
      - chunks[].content: only the first {chunks_with_content} chunks keep
        full content (those are the ones render_cg_summary shows). Later
        chunks keep file_path only — they're used by classify_cg_evidence
        which only reads file_path. Saves ~30KB on a 20-chunk payload.

    Total reduction on representative Ecommerce query: 110KB → ~17-20KB.
    """
    out = {}
    for k, v in payload.items():
        if k == "entities" and isinstance(v, list):
            new_ents = []
            for e in v:
                if not isinstance(e, dict):
                    continue
                e_clean = {ek: ev for ek, ev in e.items()
                           if ek in _CG_KEEP_ENTITY_FIELDS}
                desc = e_clean.get("description") or ""
                if len(desc) > entity_desc_max:
                    e_clean["description"] = desc[:entity_desc_max] + "…"
                new_ents.append(e_clean)
            out[k] = new_ents
        elif k in ("relationships", "relations") and isinstance(v, list):
            new_rels = []
            for r in v:
                if not isinstance(r, dict):
                    continue
                r_clean = {rk: rv for rk, rv in r.items()
                           if rk in _CG_KEEP_RELATIONSHIP_FIELDS}
                desc = r_clean.get("description") or ""
                if len(desc) > relation_desc_max:
                    r_clean["description"] = desc[:relation_desc_max] + "…"
                new_rels.append(r_clean)
            out[k] = new_rels
        elif k == "chunks" and isinstance(v, list):
            new_chunks = []
            for i, c in enumerate(v):
                if not isinstance(c, dict):
                    continue
                c_clean = {ck: cv for ck, cv in c.items()
                           if ck in _CG_KEEP_CHUNK_FIELDS}
                if i >= chunks_with_content:
                    # Beyond the first N: drop content, keep just file_path
                    # + chunk_id (used by classify_cg_evidence)
                    c_clean.pop("content", None)
                else:
                    content = c_clean.get("content") or ""
                    if len(content) > chunk_content_max:
                        c_clean["content"] = content[:chunk_content_max] + "…"
                new_chunks.append(c_clean)
            out[k] = new_chunks
        elif k == "references":
            # references is small but unused; drop entirely
            continue
        else:
            out[k] = v
    return out


def classify_cg_evidence(cg_data: dict) -> dict:
    """Classify retrieved chunks/entities by data source type, mapping to the
    Agent ↔ Context Graph integration scenarios in INTEGRATION.md §4.

    Returns a dict with boolean flags:
      products  → Scenario 1 (Product Knowledge Enhancement)
      rules     → Scenario 2 (Business Rules Retrieval)
      patterns  → Scenario 3 (Conversation Intelligence)
      decisions → Scenario 7 (Decision Precedent / Audit-trail read)

    Detection: examine chunk file_path (and entity descriptions as fallback),
    matching the canonical workspace file_path conventions documented in §6
    of the GUIDE: product_catalog/, agent_logic/, conversation/, decision_*/.
    """
    flags = {"products": False, "rules": False, "patterns": False, "decisions": False}

    def _bucket(path: str) -> str | None:
        if not path:
            return None
        p = path.lower()
        if p.startswith("product_catalog") or "product_catalog" in p:
            return "products"
        if "decision" in p:  # decision_summary, decision/, etc.
            return "decisions"
        if p.startswith("agent_logic") or "business_rules" in p or "/faq" in p \
           or "playbook" in p or "prompt_chain" in p:
            return "rules"
        if p.startswith("conversation"):
            return "patterns"
        return None

    for c in (cg_data.get("chunks") or []):
        # LightRAG chunks may carry file_path under several keys depending on version
        fp = c.get("file_path") or c.get("source_id") or c.get("source") or ""
        b = _bucket(fp)
        if b:
            flags[b] = True
    # Entity-type fallback — some workspaces have PRODUCT entities even if chunks
    # don't carry file_paths
    for e in (cg_data.get("entities") or []):
        et = (e.get("entity_type") or "").lower()
        if et in ("product", "category"):
            flags["products"] = True
    # Relations carrying decision_trace context indicate decision evidence
    for r in (cg_data.get("relationships") or cg_data.get("relations") or []):
        rc = r.get("relation_context") or {}
        if isinstance(rc, dict) and rc.get("decision_trace"):
            flags["decisions"] = True
    return flags


def render_cg_summary(cg_data: dict, max_entities: int = 10, max_rels: int = 5,
                      max_chunks: int = 3) -> str:
    parts = []
    ents = (cg_data.get("entities") or [])[:max_entities]
    if ents:
        parts.append("### Top entities (with descriptions)")
        for e in ents:
            name = e.get("entity_name", "?")
            etype = e.get("entity_type", "?")
            desc = (e.get("description") or "").replace("\n", " ").strip()[:200]
            parts.append(f"- {name} ({etype}): {desc}")
    rels = (cg_data.get("relationships") or cg_data.get("relations") or [])[:max_rels]
    if rels:
        parts.append("### Top relations (with decision traces)")
        for r in rels:
            src = r.get("src_id") or r.get("source", "?")
            tgt = r.get("tgt_id") or r.get("target", "?")
            desc = (r.get("description") or "").replace("\n", " ").strip()[:240]
            rc = r.get("relation_context") or {}
            dt = (rc.get("decision_trace") or "")[:140] if isinstance(rc, dict) else ""
            qd = (rc.get("quantitative_data") or "")[:140] if isinstance(rc, dict) else ""
            parts.append(f"- {src} → {tgt}: {desc}"
                          + (f" | trace: {dt}" if dt else "")
                          + (f" | data: {qd}" if qd else ""))
    chunks = (cg_data.get("chunks") or [])[:max_chunks]
    if chunks:
        parts.append("### Knowledge chunks")
        for c in chunks:
            content = (c.get("content") or "").replace("\n", " ").strip()[:300]
            parts.append(f"- {content}")
    return "\n".join(parts) if parts else "(no CG knowledge retrieved)"


# Empirical motivation for gating: v4 batch (2026-05-01) showed that the
# aggressive diversification prompt hurt supervised performance — 3/5 scenarios
# regressed sharply (mean Δcommit −1.0 vs +0.75 in prior Ecommerce batch). Telling
# the supervisor "advance toward direct_ask after 2 uses" forced premature
# moves off working strategies (e.g., `087b0160` Insurance c4 supervisor went from
# winning to losing). The architecturally correct fix is ClusterPlan with
# per-phase exit signals (research-notes/2026-04-30-session-plan-…). This
# function is now gated OFF by default; flip POC_SESSION_MEMORY=1 to re-enable
# for diagnostics.
SESSION_MEMORY_ENABLED = os.environ.get("POC_SESSION_MEMORY", "0") == "1"


# ── Commercial context (opp_type + price_tier) ─────────────────────────────
# Surfaces real-world purchase psychology to the supervisor so it can weigh
# customer signals appropriately. Empirical motivation: dfb34792 — Karl's
# "I'm not in the market" reads as a soft no in isolation, but his opp_type
# is `Abandoned Cart` (he was demonstrably in the market 5min earlier). The
# supervisor needs the commercial context to weigh that signal correctly.

# Tenant-default ticket-price tier; refined later via product catalog when we
# wire that up. Real Agent opp_types confirmed via prod query 2026-05-01.
_TENANT_PRICE_DEFAULTS = {
    "Ecommerce":    ("$200-500 (premium headphones)",          "mid"),
    "Insurance":     ("~5,000 USD/yr (auto insurance renewal)", "mid-high"),
    "MattressCommerce":     ("$1,000-3,000 (premium mattresses)",      "mid-high"),
    "SaaS": ("$400-2,000/yr (SaaS subscription)",      "mid"),
    "CleaningCommerce":  ("$20-80 (consumer cleaning products)",    "low"),
    "Sellence":  ("varies — internal AI demo platform",     "high"),
}

# Levers that empirically work / don't work per price tier (from sales-research
# literature + the Cialdini findings in research thread §11.5)
_LEVERS_BY_TIER = {
    "low": {
        "preferred": ["scarcity", "social_proof", "logistics", "reciprocity"],
        "discouraged": ["deep_information", "heavy_authority"],
        "tier_note": "low-ticket: impulse-buy psychology; over-engineering loses the sale",
    },
    "mid": {
        "preferred": ["authority", "information", "reciprocity", "social_proof"],
        "discouraged": ["pure_scarcity"],
        "tier_note": "mid-ticket: customers research; quality vs price tradeoff explicit; risk-reversal matters",
    },
    "mid-high": {
        "preferred": ["authority", "information", "commitment"],
        "discouraged": ["scarcity", "discount_leading"],
        "tier_note": "considered purchase; risk-aversion + loss-aversion dominate; structure decisions, don't pressure",
    },
    "high": {
        "preferred": ["authority", "information", "social_proof", "commitment"],
        "discouraged": ["scarcity", "discount_leading", "urgency"],
        "tier_note": "B2B / major purchase; coalitional decision; long cycle; references and credentials matter most",
    },
}


def _classify_opp_aggression(opp_type_raw: str) -> tuple[str, str, str]:
    """Map raw opp_type string → (canonical_label, aggression_level, reason).

    Robust to tenant-specific variants (e.g. 'Abandoned Cart US No Consent').
    Falls back to MEDIUM when unrecognized.
    """
    s = (opp_type_raw or "").lower()
    if any(k in s for k in ("abandoned cart", "cart abandon", "abandon cart")):
        return ("cart_abandonment", "HIGH",
                "customer demonstrated buy-intent by adding to cart minutes earlier; "
                "'not in the market' rings false — probe what changed")
    if any(k in s for k in ("renewal", "renew")):
        return ("renewal", "MEDIUM-HIGH",
                "customer has active relationship; default action is renew unless something pushed them away — "
                "probe the actual reason, don't accept surface-level deflection")
    if "upsell" in s or "cross" in s or "expansion" in s:
        return ("upsell", "MEDIUM",
                "existing customer; quantify incremental value precisely; respect 'not now' calmly")
    if "trial" in s:
        return ("trial_conversion", "MEDIUM",
                "in evaluation cycle; structure decision support, not pressure")
    if any(k in s for k in ("purchasing assistance", "search_catalog", "browse")):
        return ("buying_assistance", "MEDIUM",
                "customer engaged Agent voluntarily; some buy intent, but qualify before pushing")
    if any(k in s for k in ("review", "leave review")):
        return ("post_purchase_request", "LOW",
                "post-purchase context — not a sales conversation; respect customer's intent")
    if any(k in s for k in ("active", "hb active")):
        return ("active_subscriber", "MEDIUM",
                "existing active subscriber; expansion-style conversation")
    if "segment " in s:
        return ("segmented_outreach", "MEDIUM-LOW",
                "outbound to segmented prospect; cold-warm; qualify first")
    if any(k in s for k in ("new", "lead", "inbound", "form")):
        return ("cold_inbound", "MEDIUM-LOW",
                "cold or new lead — qualify before pushing; respect early signals")
    if any(k in s for k in ("support", "service", "help")):
        return ("support_pivot", "LOW",
                "customer came for help, not sales — respect quickly if no buy signal")
    return ("generic", "MEDIUM",
            "generic customer-engagement context; calibrate based on customer signals")


def render_commercial_context(opp_meta: dict) -> str:
    """Build the Commercial Context block that surfaces opp_type + price_tier
    + aggression posture + tier-appropriate Cialdini levers to the supervisor.

    Designed to be appended to the supervisor user prompt right after Customer
    State, before Recent Dialogue.
    """
    tenant = opp_meta.get("company") or "?"
    opp_type_raw = opp_meta.get("opp_type") or ""
    label, aggression, reason = _classify_opp_aggression(opp_type_raw)
    price_str, price_tier = _TENANT_PRICE_DEFAULTS.get(tenant, ("unknown", "mid"))
    levers = _LEVERS_BY_TIER.get(price_tier, _LEVERS_BY_TIER["mid"])

    # Pointed reminder for the high-aggression cases — these are where the
    # supervisor most needs to push back on surface signals
    extra_note = ""
    if label in ("cart_abandonment", "renewal"):
        extra_note = (
            f"\n\n**⚠ Important for {label}:** when this customer says \"I'm not in the market\", "
            f"\"I don't need this\", or \"I'm satisfied with what I have\" — that signal is "
            f"SUSPICIOUS. They demonstrated intent (cart-abandonment) or have an active relationship "
            f"(renewal). Probe what actually changed; do not accept the surface meaning at face value."
        )

    return f"""## Commercial Context
- opp_type (raw): {opp_type_raw or '(unknown)'}
- opp_type (canonical): {label}
- ticket_price: {price_str}  →  tier = {price_tier}
- aggression_baseline: **{aggression}**
- aggression_reason: {reason}
- preferred_levers (this price tier): {', '.join(levers['preferred'])}
- discouraged_levers (this price tier): {', '.join(levers['discouraged'])}
- tier_psychology: {levers['tier_note']}{extra_note}
"""


def render_session_memory(strategies_used: list[str] | None) -> str:
    """Render moves-already-used as a session-memory hint for the supervisor.

    Disabled by default since 2026-05-01 — see SESSION_MEMORY_ENABLED comment.
    """
    if not SESSION_MEMORY_ENABLED:
        return ""  # rolled back; ClusterPlan will provide phase-aware memory instead
    if not strategies_used:
        return "## Strategies Used So Far This Session\n(none — this is the first supervised turn)"
    from collections import Counter
    counts = Counter(strategies_used)
    lines = [f"- {s}: {n}x" for s, n in counts.most_common()]
    used_set = sorted(set(strategies_used))
    return (
        "## Strategies Used So Far This Session\n"
        + "\n".join(lines)
        + f"\n\nYou've already executed: {', '.join(used_set)}.\n"
        + "**Prefer a strategy you have NOT used yet** unless the customer's last "
        + "message clearly demands a repeat. If `objection_handling` or `information` "
        + "have already been used 2+ times, the customer has heard those moves — "
        + "advance toward `direct_ask` or `commitment` instead. The session is "
        + "phase-progressive: each turn should move the customer one phase closer "
        + "to commitment, not loop on the current phase."
    )


def build_user_prompt(opp_meta: dict, dialog: list[dict], cg_data: dict,
                       business_rules_excerpt: str = "",
                       strategies_used: list[str] | None = None,
                       plan_state: "PlanState | None" = None,
                       pre_rendered_plan_section: str | None = None,
                       phase_block: str = "",
                       precedents: dict | None = None,
                       moves_used: dict | None = None) -> str:
    p = opp_meta
    n_msgs = len(dialog)
    cust_msg = next(
        (m.get("text", "") for m in reversed(dialog)
         if m.get("role") == "customer"), "(none)")
    sm = render_session_memory(strategies_used)
    sm_block = f"\n{sm}\n" if sm else ""
    plan_block = ""
    # Win-mode override (T-76 Day 4): if a pre-rendered plan section was
    # supplied (e.g. by replayer when win-plan engagement gate fired), use it
    # in place of the cluster_plan render.
    if pre_rendered_plan_section:
        plan_block = f"\n{pre_rendered_plan_section}\n"
    elif plan_state is not None:
        from cluster_plan import render_plan_section
        rendered = render_plan_section(plan_state.plan, plan_state)
        if rendered:
            plan_block = f"\n{rendered}\n"
    commercial_block = render_commercial_context(opp_meta)
    anchor_block = render_anchor_block(opp_meta.get("anchors"), opp_meta)
    arc_block = f"\n{phase_block}\n" if phase_block else ""
    # R1 — closed-loop precedent block (session-start fetch)
    precedents_block = render_session_precedents(precedents) if precedents else ""
    precedents_section = f"\n{precedents_block}\n" if precedents_block else ""
    # R7 — variation-pressure block. Marks moves used ≥2x this session as
    # "recently used" so the supervisor varies its tactical play.
    cooldown_block = ""
    if moves_used and any(c >= 2 for c in moves_used.values()):
        recent = [(name, c) for name, c in moves_used.items() if c >= 2]
        recent.sort(key=lambda x: -x[1])
        lines = ["## Move-Cooldown (variation pressure)"]
        lines.append(
            "These Tier 2 concrete moves have already fired multiple times "
            "this session. The customer has heard this play. Strongly prefer "
            "a different concrete_move on this turn unless the move is genuinely "
            "the only fit (e.g. customer asked for the same thing again):"
        )
        for name, c in recent:
            lines.append(f"- **{name}** — used {c}× already this session")
        lines.append(
            "If you must reuse one, EXPLAIN WHY in `audit.rationale_summary` "
            "(e.g. 'no alternative move applies; customer re-asked'). "
            "Otherwise pick a different move from the catalog."
        )
        cooldown_block = "\n" + "\n".join(lines) + "\n"
    # 2026-05-10 — Profile-aware anchor strategy injection.
    # Loads `data/anchor_strategy/<tenant>.yaml`, matches the customer's
    # profile fields (trust_level, decision_logic, motivator, etc.) against
    # configured rules, and returns concatenated supervisor guidance text.
    # This makes profile-aware anchoring an authoritative directive-input
    # rather than buried prose in the system prompt.
    profile_anchor_guidance_block = ""
    try:
        from anchor_strategy import build_profile_guidance
        guidance = build_profile_guidance(p.get("company"), p)
        if guidance:
            profile_anchor_guidance_block = (
                f"\n## Profile-aware Anchoring Guidance (authoritative — "
                f"override default behavior to follow these rules)\n{guidance}\n"
            )
    except Exception as _e:  # noqa: F841 — best-effort; never break the prompt
        pass

    # 2026-05-10 — Sales Script Library scaffold injection (Phase 1).
    # Loads `data/script_library/<Tenant>/*.yaml`, selects highest-priority
    # script whose `applies_when` matches the customer's profile and
    # conversation state, renders the matching arc step as a markdown
    # scaffold the supervisor adapts to the live conversation. Sonnet
    # treats it as authoritative shape-guidance for THIS turn.
    #
    # R-panel-only by construction: this code only runs inside the supervisor
    # stage `prompt_signal_analysis_combined`, which is only spliced into
    # the R-side chain. L-panel never reaches this code path.
    script_scaffold_block = ""
    try:
        from script_library import (
            select_script, render_script_block, get_script_by_id,
            advance_arc_state,
        )
        # Phase 2 — Multi-turn arc with state advancement.
        # 1. Read prior state from opp_meta (lives across turns within session)
        # 2. Advance state based on customer's most recent response
        # 3. Look up the script (from advanced state's script_id, or fresh-select)
        # 4. Render the CURRENT arc step (not always step 0)
        # 5. Persist new state to opp_meta for next turn
        prior_arc_state = (opp_meta or {}).get("_script_arc_state") or {}
        new_arc_state = advance_arc_state(
            prior_arc_state, p.get("company"), p, dialog
        )

        # Resolve the active script after advancement
        active_script_id = new_arc_state.get("script_id") if new_arc_state else None
        script = None
        if active_script_id:
            script = get_script_by_id(p.get("company"), active_script_id)
        if script is None:
            # Fallback: fresh selection (e.g., new_arc_state was empty)
            script = select_script(p.get("company"), p, dialog)
            if script is not None:
                new_arc_state = {
                    "script_id": script.script_id,
                    "arc_step_idx": 0,
                    "started_at_turn": len(dialog or []),
                    "history": [],
                    "consecutive_no_advance": 0,
                    "advance_reason": "fallback_fresh_selection",
                }

        if script is not None:
            arc_step_idx = int((new_arc_state or {}).get("arc_step_idx", 0))
            arc_step_idx = max(0, min(arc_step_idx, len(script.arc) - 1))
            anchors_for_render = (opp_meta or {}).get("anchors") or {}
            script_scaffold_block = render_script_block(
                script, p, anchors=anchors_for_render, dialog=dialog,
                arc_step_idx=arc_step_idx,
            )
            log.info(
                "chain.script_library: %s (step=%d/%d) advance_reason=%s "
                "consecutive_no_advance=%d",
                script.script_id, arc_step_idx + 1, len(script.arc),
                (new_arc_state or {}).get("advance_reason", "?"),
                (new_arc_state or {}).get("consecutive_no_advance", 0),
            )
            # Persist for next turn
            if opp_meta is not None and isinstance(opp_meta, dict):
                opp_meta["_script_arc_state"] = new_arc_state
        else:
            log.info("chain.script_library: no script matched for opp=%s "
                     "(profile=%s/%s/%s commit=%s)",
                     (opp_meta or {}).get("opp_id", "?")[:8],
                     p.get("primary_motivator"), p.get("decision_logic"),
                     p.get("trust_level"), p.get("commitment_level"))
    except Exception as _e:  # noqa: F841 — best-effort
        log.warning("chain.script_library: error in advancement (non-fatal): %s", _e)

    # 2026-05-10 — Language-matching rule for supervisor's directive.
    # The actor (build_answer) has a language-match system suffix, but the
    # actor faithfully renders whatever language the directive's must_say
    # is in. So if Sonnet emits Hebrew must_say despite an English customer,
    # the actor produces Hebrew. Solution: force Sonnet to match customer's
    # MOST RECENT message language at directive-generation time.
    last_cust_text = ""
    for _m in reversed(dialog or []):
        if _m.get("role") == "customer" and _m.get("text"):
            last_cust_text = _m["text"]
            break
    _is_hebrew = any("֐" <= ch <= "׿" for ch in last_cust_text)
    _detected_lang = "Hebrew" if _is_hebrew else "English"
    language_rule_block = (
        f"\n## ⚠ LANGUAGE — MANDATORY (overrides all other instructions)\n"
        f"Customer's most recent message language: **{_detected_lang}**\n"
        f"The directive's `must_say` items, `signal_analysis.response_template`, "
        f"and any free-form text MUST be in {_detected_lang}. Do not switch to "
        f"a different language regardless of dialog history, profile context, "
        f"or script-template language. If a script scaffold is in a different "
        f"language, TRANSLATE it preserving intent.\n"
    )

    return f"""{language_rule_block}
## Customer State
- Tenant: {p.get('company')} / {p.get('opp_type')}
- Turn: {n_msgs}
- Motivator: {p.get('primary_motivator')}
- Decision logic: {p.get('decision_logic')}
- Trust level: {p.get('trust_level')}
- Communication style: {p.get('communication_style')}
- Objection pattern (profile): {p.get('objection_pattern')}
- Customer's last message: "{cust_msg[:300]}"
{profile_anchor_guidance_block}{script_scaffold_block}
{commercial_block}{anchor_block}{arc_block}{precedents_section}{cooldown_block}
## Recent Dialogue (last 8 turns)
{render_dialog_snippet(dialog)}{sm_block}{plan_block}
## Knowledge from Context Graph (workspace={p.get('company')})
{render_cg_summary(cg_data)}

## Tenant Business Rules (relevant excerpt)
{business_rules_excerpt[:6000]}

## Output
Emit the Strategic Directive as JSON only. Ground facts_to_anchor in the CG entities above where possible — including precedent edges as `decision_trace:<src>→<tgt>`. No preamble, no markdown."""


def render_anchor_block(anchors: dict | None, opp_meta: dict | None = None) -> str:
    """T-81 — render the customer's economic reference frame so the supervisor
    can reason about pricing strategy with real context (last year's price,
    market avg, competitor mentions, max discount available) instead of
    inventing strategies in a vacuum. Empty string if no anchors available.

    2026-05-10 — Profile-aware anchor framing. For Skeptical customers
    (per opp_meta.trust_level), the block reframes which value to use as
    the conversational anchor:
      - Standard: anchor on `current_quoted_price_usd` (our actual quote)
      - Skeptical: anchor on `market_avg_for_segment_usd` instead, and
        explicitly mark `current_quoted_price_usd` as "internal reference
        — justify with concrete coverage detail before disclosing"
    This prevents the inflated-baseline-then-discount pattern that Skeptical
    customers consistently detect (UI test 2026-05-10).
    """
    if not anchors:
        return ""

    is_skeptical = (opp_meta or {}).get("trust_level", "").strip().lower() == "skeptical"
    market_avg = anchors.get("market_avg_for_segment_usd")
    current_quote = anchors.get("current_quoted_price_usd")

    if is_skeptical and market_avg:
        # Skeptical mode: lead with market_avg, deemphasize current_quoted_price
        lines = ["## Customer's Economic Reference Frame (Skeptical-customer mode)"]
        lines.append(
            f"- **Conversational anchor (USE THIS as the headline price)**: "
            f"market_avg = {market_avg} USD"
        )
        if anchors.get("last_year_price_usd"):
            lines.append(f"- Last year's premium: {anchors['last_year_price_usd']} USD")
        if current_quote:
            lines.append(
                f"- Internal reference (DO NOT lead with this; only justify with "
                f"concrete coverage detail before disclosing): "
                f"current_quoted_price = {current_quote} USD"
            )
    else:
        lines = ["## Customer's Economic Reference Frame"]
        if anchors.get("last_year_price_usd"):
            lines.append(f"- Last year's premium: {anchors['last_year_price_usd']} USD")
        if current_quote:
            lines.append(f"- Our current quote: {current_quote} USD")
    if anchors.get("claimed_increase_pct") is not None and anchors.get("actual_market_yoy_change_pct") is not None:
        claimed = anchors["claimed_increase_pct"]
        actual = anchors["actual_market_yoy_change_pct"]
        gap_note = ""
        if claimed > 5 + actual:
            gap_note = f" ⚠ AGENT MAY HAVE OVERCLAIMED — actual market YoY is {actual:+.1f}%, not {claimed:+.1f}%."
        lines.append(f"- YoY increase claimed in conversation: {claimed:+.1f}%; actual market YoY: {actual:+.1f}%{gap_note}")
    if anchors.get("market_avg_for_segment_usd"):
        lines.append(f"- Market average for this customer's vehicle segment: {anchors['market_avg_for_segment_usd']} USD")
    if anchors.get("max_discount_pct_internal") is not None:
        lines.append(f"- Max internal discretionary discount: {anchors['max_discount_pct_internal']}%")
    if anchors.get("coverage_summary"):
        lines.append(f"- Coverage summary: {anchors['coverage_summary']}")
    if anchors.get("loyalty_years"):
        lines.append(f"- Customer loyalty: {anchors['loyalty_years']} years")
    if anchors.get("synthetic"):
        lines.append(f"- (Anchor provenance: {anchors.get('provenance','synthesized')} — values are best-estimate, not from live policy DB)")
    lines.append("")
    lines.append("**ANCHOR-AWARE PRICING RULE:** If your last quoted price is at-or-below either the market average or a competitor offer the customer has mentioned, do NOT lower further. Multiple successive price drops in one conversation damage trust ('why didn't you offer this from the start?'). Hold the price; address objections via reciprocity / coverage value / authority instead. Staircase pricing is a known sales failure mode — empirically validated as causing trust collapse on this benchmark.")
    return "\n" + "\n".join(lines) + "\n"


# ── Decision-trace semantic search (closed-loop precedent retrieval) ────────
async def cg_decisions_search(http_client: httpx.AsyncClient, workspace: str,
                                query: str, top_k: int = 3,
                                min_confidence: float = 0.3) -> dict:
    """Call CG /graph/decisions/search — semantic search over decision-trace edges.

    Used by mode2_directive at high-stakes turns (commit ≥ 4) to inject
    *specific past decisions* the supervisor can reference. Complements CGR3
    (which gives synthesized reasoning) with concrete precedents (which give
    citable cases).

    Empirical motivation: 2026-05-01 probe showed our own POC sessions are
    ALREADY emitting decision-traces with provenance=POC supervisor — the
    closed-loop write side works. This function makes the closed-loop READ
    side fire at the right moments.

    Returns: {results: [...], total_count: int, _error: str | None}
    """
    if not CG_API_KEY:
        return {"results": [], "total_count": 0, "_error": "no LIGHTRAG_API_KEY"}
    CG_ENDPOINT_CALLS["decisions_read"] += 1
    response = None
    try:
        response = await http_client.get(
            f"{CG_URL}/graph/decisions/search",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": workspace},
            params={"q": query, "top_k": top_k, "min_confidence": min_confidence},
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) else {"results": body or [], "total_count": len(body or [])}
    except Exception as e:
        err = _format_http_error(e, response)
        log.warning("decisions_search failed for workspace=%s: %s", workspace, err)
        return {"results": [], "total_count": 0, "_error": err}


def render_session_precedents(precedents: dict | None) -> str:
    """R1 (2026-05-04) — Closed-loop READ side: format the session-start
    `fetch_precedent_decisions` output into a Mode 1b prompt block. Different
    from `render_decision_precedents`, which formats the per-turn
    `cg_decisions_search` output (Mode 2 only).

    Input shape: {count, sample, by_strategy} from `fetch_precedent_decisions`.
    Renders top-3 sample edges with source segment, target strategy, outcome,
    and the rationale text. Empty string when no precedents available.

    The supervisor is told these are CITABLE: it can reference them in
    `facts_to_anchor.source_ref` as `decision_trace:<src>→<tgt>` for traceable
    grounding — same convention as render_decision_precedents.
    """
    if not precedents or not isinstance(precedents, dict):
        return ""
    sample = precedents.get("sample") or []
    if not sample:
        return ""
    count = precedents.get("count", 0)
    by_strategy = precedents.get("by_strategy") or {}

    lines = ["## Past Decisions On Similar Customers (closed-loop precedents)"]
    lines.append(
        f"\nThe knowledge graph has **{count}** prior decision-trace edges "
        "for this tenant. Below are the {n} most-confident precedents — past "
        "supervised sessions. Use them to ground your directive: cite as "
        "`decision_trace:<src>→<tgt>` in `facts_to_anchor.source_ref` when "
        "your reasoning matches a precedent.".format(n=min(3, len(sample)))
    )
    if by_strategy:
        top_strats = sorted(by_strategy.items(), key=lambda kv: -kv[1])[:5]
        strat_summary = ", ".join(f"{s}({c})" for s, c in top_strats)
        lines.append(f"\n**Strategy histogram across all precedents:** {strat_summary}\n")
    for i, d in enumerate(sample[:3], 1):
        src = (d.get("src_id") or "?")[:80]
        tgt = (d.get("tgt_id") or "?")[:60]
        rc = d.get("relation_context") or {}
        confidence = rc.get("confidence_score")
        outcome = "?"
        for s in (rc.get("supporting_sentences") or []):
            if not isinstance(s, str):
                continue
            # Case-insensitive split — supporting_sentences may capitalize
            # "Outcome:" or use any casing. Index [1:] handles 0-or-more
            # parts safely (was IndexError before when split returned [s]).
            import re as _re_split
            parts = _re_split.split(r"outcome:", s, maxsplit=1, flags=_re_split.IGNORECASE)
            if len(parts) >= 2:
                outcome = parts[1].strip()[:30]
                break
        trace = (rc.get("decision_trace") or "")[:240].replace("\n", " ")
        lines.append(f"\n**Precedent {i}:** {src} → {tgt}")
        lines.append(f"- Outcome: {outcome}  ·  Confidence: {confidence}")
        if trace:
            lines.append(f"- Trace: {trace}{'...' if len(trace) >= 240 else ''}")
        if rc.get("quantitative_data"):
            lines.append(f"- Quant: {rc['quantitative_data'][:160]}")
    return "\n".join(lines) + "\n"


def render_decision_precedents(decisions_search_result: dict) -> str:
    """Format decision-search results as a supervisor-prompt section."""
    results = decisions_search_result.get("results") or []
    if not results:
        return ""
    lines = ["## Past Decision Precedents (from closed-loop graph — concrete cases for citation)"]
    lines.append(
        f"\nThe following {len(results)} past decisions are most similar to "
        "the current situation. Each is a real edge in the graph from a previous "
        "supervised session. You can cite these in `facts_to_anchor.source_ref` "
        "as `decision_trace:<src>→<tgt>` for traceable grounding.\n"
    )
    for i, d in enumerate(results, 1):
        src = d.get("src_id") or "?"
        tgt = d.get("tgt_id") or "?"
        rc = d.get("relation_context") or {}
        trace = (rc.get("decision_trace") or "")[:280].replace("\n", " ")
        outcome = "?"
        for s in (rc.get("supporting_sentences") or []):
            if not isinstance(s, str):
                continue
            import re as _re_split
            parts = _re_split.split(r"outcome:", s, maxsplit=1, flags=_re_split.IGNORECASE)
            if len(parts) >= 2:
                outcome = parts[1].strip()[:30]
                break
        confidence = rc.get("confidence_score")
        prov = (rc.get("provenance") or "")[:50]
        lines.append(f"### Precedent {i}: {src} → {tgt}")
        lines.append(f"- **Outcome:** {outcome}  ·  **Confidence:** {confidence}  ·  **Source:** {prov}")
        if trace:
            lines.append(f"- **Trace:** {trace}{'...' if len(trace) >= 280 else ''}")
        if rc.get("quantitative_data"):
            lines.append(f"- **Quant:** {rc['quantitative_data'][:200]}")
        lines.append("")
    return "\n".join(lines)


# ── Precedent retrieval — closed-loop READ side ─────────────────────────────
async def fetch_precedent_decisions(http_client: httpx.AsyncClient, workspace: str,
                                       min_confidence: float = 0.5,
                                       top_k: int = 5) -> dict:
    """Pull existing decision-trace edges from the graph as precedents the
    supervisor can use. Closed-loop READ side — what the system has learned.

    Returns {count, sample, by_strategy} where:
      count: total decisions in graph (proof of learning corpus size)
      sample: top-k decisions with src/tgt/decision_trace
      by_strategy: count grouped by tgt_id (strategy/discount/etc)
    """
    if not CG_API_KEY:
        return {"count": 0, "sample": [], "by_strategy": {}, "_error": "no key"}
    CG_ENDPOINT_CALLS["decisions_read"] += 1
    response = None
    try:
        response = await http_client.get(
            f"{CG_URL}/graph/decisions",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": workspace},
            params={"min_confidence": min_confidence},
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        # R2 fix (2026-05-04) — actual response wrapper is `{total_count, decisions}`,
        # NOT `{data, ...}` or `{results, ...}`. Prior parser silently returned []
        # because it looked for the wrong key — meaning EVERY session before today
        # ran with 0 precedents in the prompt despite the corpus having 4,346 edges
        # for Insurance. Closed-loop READ was silently dead.
        if isinstance(body, list):
            items = body
            total = len(body)
        elif isinstance(body, dict):
            items = (body.get("decisions")
                     or body.get("data")
                     or body.get("results")
                     or [])
            total = body.get("total_count") if isinstance(body.get("total_count"), int) else len(items)
        else:
            items = []
            total = 0
        # Sample — take top by confidence
        sample = sorted(items[:50], key=lambda d: -(d.get("relation_context", {}).get("confidence_score", 0) or 0))[:top_k]
        # Strategy histogram
        by_strategy: dict[str, int] = {}
        for d in items[:200]:  # sample first 200 for histogram
            t = (d.get("tgt_id") or "?")[:60]
            by_strategy[t] = by_strategy.get(t, 0) + 1
        return {"count": total, "sample": sample, "by_strategy": by_strategy}
    except Exception as e:
        err = _format_http_error(e, response)
        log.warning("precedent fetch failed for workspace=%s: %s", workspace, err)
        return {"count": 0, "sample": [], "by_strategy": {}, "_error": err}


# ── Cohort-conditioned precedent retrieval (CR-PS Phase 4) ──────────────────
# Per `research-notes/2026-05-07-cohort-precedent-substrate.md`. New primary
# precedent path: per-turn retrieval keyed on (cohort × phase × signal) against
# the local SQLite substrate, with CG fallback for the long tail. Replaces the
# tactical role of `fetch_precedent_decisions` — that remains as session-level
# unstructured priming until Phase 5 results justify its retirement.
#
# Wired into the actor system prompt (NOT chain_executor.build_context_blocks)
# to dodge the Mode 1a-cache-skipped path that killed R33's fire rate
# (cf. actor.py:460–499 and 2026-05-06 R33 post-mortem).

POC_SERVER_URL = os.environ.get("POC_SERVER_URL", "http://127.0.0.1:8443")
POC_COHORT_PRECEDENTS_ENABLED = os.environ.get("POC_COHORT_PRECEDENTS", "1") == "1"

# Supervisor's primary_signal vocabulary → substrate's objection_category
# vocabulary. None means "no objection filter" (signal isn't an objection).
# Substrate objection_category values: price · timing · competitor · trust ·
# need · authority · none.
_SIGNAL_TO_OBJECTION_CATEGORY = {
    "explicit_price_request":         "price",
    "explicit_objection_price":       "price",
    "explicit_objection_timing":      "timing",
    "explicit_objection_competition": "competitor",
    "explicit_objection_trust":       "trust",
    "explicit_objection_product_fit": "need",
    "competing_offer_mention":        "competitor",
    "pace_request":                   "timing",
    # signals that aren't objection-flavored → no filter
    "commitment_signal":              None,
    "disengagement":                  None,
    "trust_indicator":                None,
    "pain_articulation":              None,
    "product_inquiry":                None,
    "silence":                        None,
    "other":                          None,
}


async def fetch_cohort_precedents(
    http_client: httpx.AsyncClient,
    opp_meta: dict,
    commit_max: int | None,
    primary_signal: str | None = None,
    top_k: int = 5,
) -> dict:
    """Per-turn cohort-conditioned precedent retrieval.

    Filters: company × outcome=ClosedWon × cohort dims (decision_logic +
    primary_motivator) × commitment_level near commit_max ± 1 × objection
    derived from primary_signal (when applicable).

    Returns the response shape from `GET /api/precedents` (unchanged):
      {tier, sqlite_count, cg_count, cache_hit, fallback_fired, filters,
       rows, latency_ms}
    Errors and disabled state degrade to {tier: 'disabled', rows: []}.
    """
    if not POC_COHORT_PRECEDENTS_ENABLED:
        return {"tier": "disabled", "sqlite_count": 0, "cg_count": 0,
                "rows": [], "filters": {}, "latency_ms": 0}

    company = (opp_meta or {}).get("company")
    if not company:
        return {"tier": "skipped", "sqlite_count": 0, "cg_count": 0,
                "rows": [], "filters": {}, "latency_ms": 0,
                "_skip_reason": "no company"}

    params: dict = {
        "company": company,
        "outcome": "ClosedWon",
        "limit": top_k,
    }
    # Cohort dims — only the highest-cardinality (4) and second (5), to avoid
    # over-narrowing on the 3-bucket dims (Test 1 cardinality finding).
    if opp_meta.get("decision_logic"):
        params["decision_logic"] = opp_meta["decision_logic"]
    if opp_meta.get("primary_motivator"):
        params["primary_motivator"] = opp_meta["primary_motivator"]

    # Phase: ±1 around commit_max (clipped to [0, 5])
    if isinstance(commit_max, int):
        params["commitment_level_min"] = max(0, commit_max - 1)
        params["commitment_level_max"] = min(5, commit_max + 1)

    # Objection mapping
    if primary_signal:
        oc = _SIGNAL_TO_OBJECTION_CATEGORY.get(primary_signal)
        if oc is not None:
            params["objection_category"] = oc

    try:
        resp = await http_client.get(
            f"{POC_SERVER_URL}/api/precedents",
            params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        err = _format_http_error(e, getattr(e, 'response', None))
        log.warning("fetch_cohort_precedents failed: %s", err)
        return {"tier": "error", "sqlite_count": 0, "cg_count": 0,
                "rows": [], "filters": params, "latency_ms": 0,
                "_error": err}

    return data


def render_cohort_precedent_block(precedent_resp: dict) -> str:
    """Format the /api/precedents response into a system-prompt block.

    Per Phase 3 finding (bimodal reciprocity at commit=5): we surface
    sequence_number + commitment_level + sentiment so the LLM can
    distinguish in-negotiation precedents from post-close pleasantries
    rather than treating all reciprocity-at-commit=5 rows as escalation.
    """
    tier = precedent_resp.get("tier") or "?"
    rows = precedent_resp.get("rows") or []
    if not rows or tier in ("disabled", "skipped", "error", "empty"):
        return ""

    lines = [
        "",
        "## Cohort-Conditioned Won-Deal Precedents",
        f"- Source tier: {tier}  ·  rows={len(rows)}  "
        f"·  filters={precedent_resp.get('filters')}",
        "",
        "These are concrete moves from real production agents on won deals "
        "matching this opp's cohort and current phase. Use them as evidence "
        "of what works at this commitment level — adapt the move to context, "
        "don't paraphrase verbatim.",
        "",
    ]
    for i, row in enumerate(rows, 1):
        if row.get("_tier") == "cg":
            # CG synthesized row — single LLM-cooked precedent string
            lines.append(f"### Precedent {i} (semantic generalization)")
            lines.append(f"- Tier: cg-fallback")
            txt = (row.get("message_text") or "").strip()
            if txt:
                lines.append(f"- {txt[:600]}{'…' if len(txt) > 600 else ''}")
            lines.append("")
            continue
        strat = row.get("primary_strategy") or "?"
        tone = row.get("strategy_tone") or "?"
        commit = row.get("commitment_level")
        seq = row.get("sequence_number")
        sent = row.get("sentiment")
        pers = row.get("persuasion_score")
        text = (row.get("message_text") or "").strip()
        lines.append(f"### Precedent {i}: strategy={strat} · tone={tone}")
        lines.append(f"- commit={commit} · turn_seq={seq} · sentiment={sent} "
                     f"· persuasion_score={pers}")
        if text:
            lines.append(f"- Move: {text[:400]}{'…' if len(text) > 400 else ''}")
        lines.append("")
    return "\n".join(lines)


# ── CGR3 multi-hop (Mode 2) ─────────────────────────────────────────────────
def _format_http_error(e: Exception, response: httpx.Response | None = None) -> str:
    """Build a useful error string. Some httpx exceptions stringify to ''
    (especially ReadTimeout) — fall back to the class name + status + body
    snippet so the log isn't blank."""
    parts = [type(e).__name__]
    msg = str(e)
    if msg:
        parts.append(msg)
    if response is not None:
        parts.append(f"http={response.status_code}")
        body_snip = (response.text or "")[:300].replace("\n", " ")
        if body_snip:
            parts.append(f"body={body_snip!r}")
    return " | ".join(parts)


async def cgr3_query(http_client: httpx.AsyncClient, workspace: str, query: str,
                      max_iterations: int = 3) -> dict:
    """Mode 2: multi-hop iterative reasoning via CG /cgr3/query.
    Returns {response: str, ...} or fallback {response: '', _error}."""
    if not CG_API_KEY:
        return {"response": "", "_error": "no LIGHTRAG_API_KEY"}
    CG_ENDPOINT_CALLS["cgr3"] += 1
    response = None
    try:
        response = await http_client.post(
            f"{CG_URL}/cgr3/query",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": workspace},
            json={"query": query, "mode": "hybrid", "max_iterations": max_iterations,
                   "top_k": 20, "include_references": True},
            timeout=60,  # CGR3 is slower
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and "data" in body and isinstance(body["data"], dict):
            return body["data"]
        return body
    except Exception as e:
        err = _format_http_error(e, response)
        log.warning("CGR3 query failed for workspace=%s: %s (query_chars=%d)",
                     workspace, err, len(query))
        return {"response": "", "_error": err}


# ── Public API ──────────────────────────────────────────────────────────────
async def _traced_anthropic_call(stage: str, system: str, user_prompt: str,
                                   max_tokens: int = 2000):
    """T-86 trace-instrumented Anthropic call wrapper. Returns (msg_obj, raw_text).
    Emits an llm_call trace event with full prompt + response."""
    import time as _time
    _t0 = _time.monotonic()
    msg = await _anthropic.messages.create(
        model=SUPERVISOR_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text if msg.content else ""
    try:
        from trace_logger import TraceLogger
        trace = TraceLogger.current()
        if trace:
            trace.llm(
                stage=stage, provider="anthropic", model=SUPERVISOR_MODEL,
                system=system, user=user_prompt, response=raw,
                latency_ms=int((_time.monotonic() - _t0) * 1000),
                input_tokens=msg.usage.input_tokens if hasattr(msg, "usage") else 0,
                output_tokens=msg.usage.output_tokens if hasattr(msg, "usage") else 0,
            )
    except Exception:
        pass
    return msg, raw


def _directive_strategy(directive: dict | None) -> str | None:
    """Pull the primary_strategy out of a directive in either schema variant
    (`primary_strategy` flat, or nested `strategy.primary`)."""
    if not directive:
        return None
    return ((directive.get("strategy") or {}).get("primary")
            or directive.get("primary_strategy"))


# T-79 signal-driven adherence: encodes the BINDING RULES from
# SUPERVISOR_SYSTEM as a queryable dict so the retry mechanism can enforce
# them programmatically. The LLM otherwise treats prompt-text rules as
# suggestions (empirically verified 2026-05-02 — supervisor picked
# strategy=information for primary_signal=explicit_price_request 4/4 turns
# despite explicit "NOT information" rule in BINDING RULES).
#
# Keys: signal vocabulary from SUPERVISOR_SYSTEM signal_analysis schema.
# Values: sets of strategy.primary primitives (subset of supervisor strategy
#         enum) that adequately address that signal.
# Unmapped signal ("other", missing) → no constraint applied.
#
# Maps to AGENT-AS-EXPERT-SYSTEM Gap #4 (certainty-factor-grounded enforcement
# of signal→strategy mapping; the LLM's own per-signal confidence drives
# whether the constraint applies).
SIGNAL_TO_ALLOWED_STRATEGIES: dict[str, set[str]] = {
    "explicit_price_request":         {"reciprocity", "direct_ask", "logistics"},
    "explicit_objection_price":       {"reciprocity", "direct_ask", "logistics", "objection_handling"},
    "explicit_objection_timing":      {"empathy", "scarcity", "re_engagement"},
    "explicit_objection_competition": {"authority", "social_proof", "objection_handling"},
    "explicit_objection_product_fit": {"information", "authority", "objection_handling"},
    "explicit_objection_trust":       {"authority", "empathy", "social_proof"},
    "commitment_signal":              {"direct_ask", "logistics", "commitment"},
    "pace_request":                   {"empathy"},
    "disengagement":                  {"empathy", "re_engagement"},
    "trust_indicator":                {"authority", "social_proof", "information"},
    "competing_offer_mention":        {"authority", "objection_handling", "social_proof"},
    "pain_articulation":              {"empathy", "information", "reciprocity"},
    "product_inquiry":                {"information", "authority"},
    "silence":                        {"empathy", "re_engagement", "information"},
    # "other" intentionally absent → no constraint
}


# R4 / Q12 — directive-consistency retry. Mirrors signal_adherence pattern.
# Detects strategy ↔ must_not_say contradictions BEFORE silently stripping.
# Only price-authorizing strategies need this check today; can extend to
# other strategy↔rule incompatibilities later.
import re as _re_q12
# must_not_say rule texts come in two shapes:
#   (a) sentence form: "No further price reduction", "Don't reduce the price"
#   (b) noun-phrase form: "Further unprompted price reduction", "Any additional discount"
# Both shapes signal "the agent must not do X." Patterns cover both forms.
_Q12_NO_REDUCTION_PATTERNS = [
    # sentence form
    _re_q12.compile(r"\b(?:no|don'?t|cannot|can'?t|must not)\b.*\b(?:reduce|drop|lower|further|additional|unprompted|concession)\b.*\b(?:price|discount)", _re_q12.IGNORECASE),
    _re_q12.compile(r"\bhold\s+(?:the\s+)?(?:line|price)\s+at\b", _re_q12.IGNORECASE),
    # noun-phrase form ("Further unprompted price reduction", "Any immediate discount")
    _re_q12.compile(r"\b(?:further|additional|unprompted|any)\s+(?:unprompted|immediate|further|additional)?\s*(?:price\s+)?(?:reduction|drop|discount|concession)", _re_q12.IGNORECASE),
    _re_q12.compile(r"\bimmediate\s+(?:price\s+)?(?:reduction|drop|discount|match)", _re_q12.IGNORECASE),
]
_Q12_PRICE_AUTHORIZING = {"reciprocity", "objection_handling", "logistics"}


def _detect_consistency_violations(directive: dict | None) -> list[str]:
    """Returns list of must_not_say rule texts that contradict the chosen
    strategy, WITHOUT mutating the directive. Used for both detection and
    fallback-strip paths.
    """
    if not directive:
        return []
    primary = ((directive.get("strategy") or {}).get("primary") or "").lower()
    if primary not in _Q12_PRICE_AUTHORIZING:
        return []
    must_not_say = (directive.get("knowledge") or {}).get("must_not_say") or []
    violations: list[str] = []
    for entry in must_not_say:
        text = entry.get("text") if isinstance(entry, dict) else str(entry)
        if not text:
            continue
        if any(p.search(text) for p in _Q12_NO_REDUCTION_PATTERNS):
            violations.append(text[:160])
    return violations


async def _retry_for_directive_consistency(
    *, directive: dict | None,
    user_prompt: str,
    tenant: str | None = None,
) -> tuple[dict | None, dict]:
    """R4 / Q12 — detect strategy ↔ must_not_say contradiction; if present,
    re-prompt the supervisor with corrective context. If retry STILL produces
    a contradiction, fall back to silent stripping (legacy behavior).

    Returns (revised_directive_or_original, retry_meta).
    """
    meta = {
        "retried": False,
        "violation_rules": [],
        "second_violation_rules": None,
        "fell_back_to_strip": False,
    }
    violations = _detect_consistency_violations(directive)
    if not violations:
        return directive, meta
    meta["violation_rules"] = violations

    chosen = ((directive or {}).get("strategy") or {}).get("primary")
    correction = (
        f"\n\n---\n## ⚠ DIRECTIVE-CONSISTENCY CORRECTION\n"
        f"Your previous directive set `strategy.primary: {chosen}` — a strategy "
        f"that AUTHORIZES a price/concession move — but ALSO listed these "
        f"`must_not_say` rules that forbid the same move:\n\n"
        + "\n".join(f"  - {v}" for v in violations)
        + f"\n\nThis is internally contradictory. Pick ONE:\n"
        f"  (a) Keep strategy={chosen} → REMOVE the contradicting must_not_say rule(s) "
        f"and add explicit rationale for why the price move is justified.\n"
        f"  (b) Keep the must_not_say rules → CHANGE strategy.primary to one that "
        f"holds the line (authority / information / commitment).\n\n"
        f"Choose whichever the customer evidence actually supports. Re-emit the "
        f"full directive with the contradiction removed. Don't strip silently — "
        f"reflect the choice in your audit.rationale_summary."
    )
    try:
        msg, raw = await _traced_anthropic_call(
            "supervisor.consistency_retry",
            build_supervisor_system_for_tenant(tenant),
            user_prompt + correction,
            max_tokens=2000,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        new_directive = json.loads(text)
        meta["retried"] = True
        # Did the retry actually fix it?
        second_violations = _detect_consistency_violations(new_directive)
        meta["second_violation_rules"] = second_violations
        if second_violations:
            log.info("consistency retry STILL violates: %d rules — falling back to strip",
                     len(second_violations))
            # Fall through to legacy strip on the retry result
            knowledge = new_directive.get("knowledge") or {}
            kept = []
            for entry in (knowledge.get("must_not_say") or []):
                t = entry.get("text") if isinstance(entry, dict) else str(entry)
                if t and any(p.search(t) for p in _Q12_NO_REDUCTION_PATTERNS):
                    continue
                kept.append(entry)
            knowledge["must_not_say"] = kept
            new_directive["knowledge"] = knowledge
            meta["fell_back_to_strip"] = True
        else:
            log.info("consistency retry SUCCESS: contradictions resolved (chosen strategy=%s)",
                     ((new_directive.get("strategy") or {}).get("primary")))
        return new_directive, meta
    except Exception as e:
        log.warning("consistency retry FAILED (%s) — falling back to silent strip", e)
        # Apply legacy strip on the original directive
        knowledge = (directive or {}).get("knowledge") or {}
        kept = []
        for entry in (knowledge.get("must_not_say") or []):
            t = entry.get("text") if isinstance(entry, dict) else str(entry)
            if t and any(p.search(t) for p in _Q12_NO_REDUCTION_PATTERNS):
                continue
            kept.append(entry)
        knowledge["must_not_say"] = kept
        (directive or {})["knowledge"] = knowledge
        meta["fell_back_to_strip"] = True
        return directive, meta


async def _retry_for_move_validity(
    *, directive: dict | None,
    user_prompt: str,
    tenant: str | None = None,
) -> tuple[dict | None, dict]:
    """R6 — Phase 2.5 move-validity retry. When the supervisor's chosen
    `strategy.concrete_move` is unknown, malformed, or has missing parameters,
    re-prompt the supervisor with the catalog of valid moves and the specific
    issue. Up to one retry; falls back to original directive (chain-stage
    advisory layer is the safety net).

    Mirror of _retry_for_signal_adherence pattern."""
    meta = {
        "retried": False,
        "first_violation": None,
        "first_move_name": None,
        "second_move_name": None,
        "fell_back": False,
    }
    if not directive:
        return directive, meta
    from concrete_moves_loader import validate_concrete_move
    cm = ((directive.get("strategy") or {}).get("concrete_move"))
    is_valid, reason, vm = validate_concrete_move(cm, tenant=tenant)
    if is_valid:
        return directive, meta
    meta["first_violation"] = reason
    meta["first_move_name"] = vm.get("move_name")

    available = vm.get("available_moves") or []
    issue_lines = [f"  - validation failed: `{reason}`"]
    if vm.get("missing_params"):
        issue_lines.append(f"  - missing required parameters: {vm.get('missing_params')}")
    if vm.get("empty_params"):
        issue_lines.append(f"  - empty parameters: {vm.get('empty_params')}")
    if vm.get("move_name") and reason == "unknown_move":
        issue_lines.append(f"  - `{vm.get('move_name')}` is not in the {tenant or '_base'} catalog")
    correction = (
        f"\n\n---\n## ⚠ MOVE VALIDITY CORRECTION\n"
        f"Your previous directive picked `strategy.concrete_move` "
        f"that failed validation:\n"
        + "\n".join(issue_lines)
        + f"\n\nThe complete list of VALID moves for this tenant ({tenant or '_base'}) is:\n"
        + "\n".join(f"  - {m}" for m in available)
        + f"\n\nRe-emit the directive. Either:\n"
        f"  (a) pick a move from the list above and supply ALL its required parameters, OR\n"
        f"  (b) set `concrete_move: null` if no Tier-2 tactical move applies this turn "
        f"(Tier 2 is optional — primary_strategy alone is sufficient when no concrete move fits).\n"
        f"Keep your strategy.primary, signal_analysis, and audit unchanged. "
        f"Only fix the concrete_move."
    )
    try:
        msg, raw = await _traced_anthropic_call(
            "supervisor.move_validity_retry",
            build_supervisor_system_for_tenant(tenant),
            user_prompt + correction,
            max_tokens=2000,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        new_directive = json.loads(text)
        meta["retried"] = True
        new_cm = ((new_directive.get("strategy") or {}).get("concrete_move"))
        is_valid2, reason2, vm2 = validate_concrete_move(new_cm, tenant=tenant)
        meta["second_move_name"] = vm2.get("move_name")
        if is_valid2:
            log.info("move-validity retry SUCCESS: %s → %s",
                     meta.get("first_move_name") or "?",
                     meta.get("second_move_name") or "(null)")
            return new_directive, meta
        log.warning("move-validity retry STILL invalid: %s → %s (reason=%s)",
                    meta.get("first_move_name"), meta.get("second_move_name"), reason2)
        meta["fell_back"] = True
        # Return the original directive — keep the bad concrete_move; chain-stage
        # advisory layer logs it and downstream prompt_build_answer will simply
        # not consume an invalid concrete_move (renderer is defensive).
        return directive, meta
    except Exception as e:
        log.warning("move-validity retry FAILED with exception (%s) — falling back", e)
        meta["fell_back"] = True
        return directive, meta


async def _retry_for_signal_adherence(
    *, directive: dict | None,
    user_prompt: str,
    chosen_strategy: str | None,
    tenant: str | None = None,
) -> tuple[dict | None, dict]:
    """T-79 signal-driven adherence retry — enforces the supervisor's OWN
    signal_analysis output against its strategy choice.

    Mechanism:
      1. Read directive.signal_analysis.primary_signal
      2. Look up allowed strategy set for that signal in SIGNAL_TO_ALLOWED_STRATEGIES
      3. If chosen_strategy not in allowed set → retry once with correction

    Differs from _retry_for_adherence (which enforces externally-authored
    win_plan agent_actions_preferred): this retry is grounded in the LLM's
    OWN reasoning. The supervisor identified the customer signal at high
    confidence; this just enforces that the strategy choice can't drift
    away from what that signal demands.

    Returns (new_directive_or_original, retry_meta).
    """
    meta = {"retried": False, "violation": None, "second_strategy": None,
             "primary_signal": None, "signal_confidence": None,
             "allowed_strategies": None}
    if not directive or not chosen_strategy:
        return directive, meta
    sa = directive.get("signal_analysis") or {}
    primary_signal = sa.get("primary_signal")
    if not primary_signal:
        return directive, meta
    meta["primary_signal"] = primary_signal
    signals = sa.get("observed_signals") or []
    primary_conf = next(
        (s.get("confidence") for s in signals
         if isinstance(s, dict) and s.get("signal") == primary_signal),
        None)
    meta["signal_confidence"] = primary_conf

    allowed = SIGNAL_TO_ALLOWED_STRATEGIES.get(primary_signal)
    if not allowed:
        return directive, meta
    meta["allowed_strategies"] = sorted(allowed)
    if chosen_strategy in allowed:
        return directive, meta

    meta["retried"] = True
    meta["violation"] = chosen_strategy

    correction = (
        f"\n\n---\n## ⚠ SIGNAL ADHERENCE CORRECTION\n"
        f"Your previous directive identified `primary_signal: {primary_signal}` "
        f"(confidence {primary_conf}) but chose `strategy.primary: "
        f"{chosen_strategy}`.\n\n"
        f"Per the BINDING RULES in your system prompt, signal "
        f"`{primary_signal}` must be addressed by ONE of: {sorted(allowed)}.\n\n"
        f"Re-emit the directive with `strategy.primary` set to one of "
        f"{sorted(allowed)}. Your signal_analysis is correct — keep it. "
        f"Only correct the strategy choice and rebuild the tone/rules/facts/"
        f"must_not_say to support the corrected strategy. Don't second-guess "
        f"the signal — the customer's evidence quote already justified it.\n"
    )
    try:
        msg, raw = await _traced_anthropic_call(
            "supervisor.signal_adherence_retry",
            build_supervisor_system_for_tenant(tenant),
            user_prompt + correction,
            max_tokens=2000,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        new_directive = json.loads(text)
        meta["second_strategy"] = _directive_strategy(new_directive)
        if meta["second_strategy"] not in allowed:
            log.warning(
                "signal-adherence retry STILL violates: %s → %s "
                "(signal=%s, allowed=%s)",
                chosen_strategy, meta["second_strategy"],
                primary_signal, sorted(allowed))
            return directive, meta
        log.info(
            "signal-adherence retry SUCCESS: signal=%s  %s → %s",
            primary_signal, chosen_strategy, meta["second_strategy"])
        return new_directive, meta
    except Exception as e:
        log.warning("signal-adherence retry failed: %s", e)
        return directive, meta


async def _retry_for_adherence(
    *, directive: dict | None,
    preferred_actions: list[str] | None,
    user_prompt: str,
    chosen_strategy: str | None,
    cur_phase_label: str | None,
    tenant: str | None = None,
) -> tuple[dict | None, dict]:
    """T-78 adherence retry — if supervisor's chosen strategy isn't in the
    current phase's preferred actions, do ONE corrective LLM call with a
    pointed reminder. Returns (new_directive_or_original, retry_meta).

    Cost: one extra Sonnet call (~$0.02) when adherence violates. Bounded
    at one retry — if the second call also violates, accept it (avoid
    infinite-retry loops).
    """
    meta = {"retried": False, "violation": None, "second_strategy": None}
    if not preferred_actions or not chosen_strategy:
        return directive, meta
    if chosen_strategy in preferred_actions:
        return directive, meta
    meta["retried"] = True
    meta["violation"] = chosen_strategy

    correction = (
        f"\n\n---\n## ⚠ ADHERENCE CORRECTION\n"
        f"Your previous directive chose `primary_strategy: {chosen_strategy}` "
        f"but the current plan phase ({cur_phase_label or '?'}) requires the "
        f"strategy to be one of: {preferred_actions}.\n\n"
        f"Re-emit the directive with `primary_strategy` set to ONE of "
        f"{preferred_actions}. The conversation context has not changed — "
        f"only your strategy choice. Do not switch phases. Just choose a "
        f"strategy that the phase requires, and rebuild the directive's "
        f"tone/rules/facts/must_not_say to support that strategy.\n"
    )
    try:
        msg, raw = await _traced_anthropic_call(
            "supervisor.adherence_retry",
            build_supervisor_system_for_tenant(tenant),
            user_prompt + correction,
            max_tokens=2000,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        new_directive = json.loads(text)
        meta["second_strategy"] = _directive_strategy(new_directive)
        # If the retry STILL violates, log + accept the original (don't
        # infinite-loop). The diagnostic flag is what matters for analysis.
        if meta["second_strategy"] not in preferred_actions:
            log.warning(
                "adherence retry STILL violates: %s → %s (preferred=%s)",
                chosen_strategy, meta["second_strategy"], preferred_actions)
            return directive, meta
        log.info(
            "adherence retry SUCCESS: %s → %s",
            chosen_strategy, meta["second_strategy"])
        return new_directive, meta
    except Exception as e:
        log.warning("adherence retry failed: %s", e)
        return directive, meta


async def mode1b_directive(opp_meta: dict, dialog: list[dict], business_rules: str,
                            http_client: httpx.AsyncClient | None = None,
                            strategies_used: list[str] | None = None,
                            plan_state: "PlanState | None" = None,
                            pre_rendered_plan_section: str | None = None,
                            preferred_actions: list[str] | None = None,
                            current_phase_label: str | None = None,
                            moves_used: dict | None = None,
                            corrective_context: str | None = None,
                            ) -> dict:
    """Generate a Mode 1b directive (full supervisor + CG retrieval).

    Returns a dict with:
      directive: parsed JSON or None
      cg_n_entities, cg_n_relations, cg_n_chunks, cg_latency_ms
      input_tokens, output_tokens, latency_ms
      fallback_reason: str or None
    """
    own_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=30)
        own_client = True

    # Conversation-phase tracker (multi-turn arc awareness, §9.2 #7).
    # Rule-based, <1ms; result feeds Mode 1b prompt as SUGGEST context.
    from conversation_phase import classify_dialog, render_phase_block, PhaseState
    phase_history = classify_dialog(dialog)
    if phase_history:
        last = phase_history[-1]
        trailing = 0
        for p in reversed(phase_history):
            if p == last:
                trailing += 1
            else:
                break
    else:
        last, trailing = "greet", 0
    phase_state = PhaseState(
        current_phase=last,
        turns_in_phase=trailing,
        history=phase_history,
    )
    cluster_plan_phase = None
    if plan_state is not None:
        try:
            cur = plan_state.current_phase()
            if cur:
                cluster_plan_phase = str(cur.get("phase_id"))
        except Exception:
            cluster_plan_phase = None
    phase_block = render_phase_block(phase_state, cluster_plan_phase)

    # Multi-turn arc telemetry (§9.2 #7 — feeds 2026-05-10 cron `218b4144`).
    # Disagreement = dynamic phase classifier disagrees with cluster_plan_phase.
    try:
        from trace_logger import TraceLogger
        _tr = TraceLogger.current()
        if _tr is not None:
            _tr.note(
                "phase_tick",
                current=phase_state.current_phase,
                turns_in_phase=phase_state.turns_in_phase,
                cluster_plan_phase=cluster_plan_phase,
                disagrees=bool(cluster_plan_phase and cluster_plan_phase != phase_state.current_phase),
                history_tail=phase_state.history[-6:] if phase_state.history else [],
            )
    except Exception:
        pass  # telemetry must never break the supervisor

    result = {
        "mode": "1b",
        "directive": None,
        "cg_n_entities": 0,
        "cg_n_relations": 0,
        "cg_n_chunks": 0,
        "cg_latency_ms": 0,
        "cg_evidence": {"products": False, "rules": False, "patterns": False, "decisions": False},
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "fallback_reason": None,
        "conversation_phase": {
            "current": phase_state.current_phase,
            "turns_in_phase": phase_state.turns_in_phase,
            "cluster_plan_phase": cluster_plan_phase,
            "history": phase_history,
        },
    }
    try:
        # 1. CG query — /query/data is the workhorse retrieval; if USE_QUERY_AUTO
        # is enabled, also fire /query/auto in parallel as an observability probe
        # so we can see which mode the server's auto-classifier picked. We DON'T
        # use the auto response to drive the directive (it's an LLM-synthesized
        # answer, not raw entities) — only as a comparison signal in metadata.
        cg_query = build_cg_query(opp_meta, dialog)
        t0 = time.monotonic()
        if USE_QUERY_AUTO:
            cg_data, auto_data = await asyncio.gather(
                cg_query_data(http_client, opp_meta["company"], cg_query, top_k=20),
                cg_query_auto(http_client, opp_meta["company"], cg_query, top_k=20),
            )
            result["server_auto"] = {
                "selected_mode": auto_data.get("selected_mode") or auto_data.get("mode"),
                "mode_reason": (auto_data.get("mode_reason") or "")[:200],
                "latency_ms": auto_data.get("latency_ms"),
                "cache_hit": bool(auto_data.get("_cache_hit")),
            }
        else:
            cg_data = await cg_query_data(http_client, opp_meta["company"], cg_query, top_k=20)
        result["cg_latency_ms"] = int((time.monotonic() - t0) * 1000)
        result["cg_cache_hit"] = bool(cg_data.get("_cache_hit"))
        result["cg_n_entities"] = len(cg_data.get("entities") or [])
        result["cg_n_relations"] = len(cg_data.get("relationships") or cg_data.get("relations") or [])
        result["cg_n_chunks"] = len(cg_data.get("chunks") or [])
        result["cg_evidence"] = classify_cg_evidence(cg_data)

        # 2. Sonnet supervisor (with session-memory + ClusterPlan state if loaded)
        # If pre_rendered_plan_section provided (e.g. win-mode plan from
        # T-76), use that instead of cluster_plan render.
        user_prompt = build_user_prompt(opp_meta, dialog, cg_data, business_rules,
                                          strategies_used=strategies_used,
                                          plan_state=plan_state,
                                          pre_rendered_plan_section=pre_rendered_plan_section,
                                          phase_block=phase_block,
                                          precedents=opp_meta.get("precedents"),
                                          moves_used=moves_used)
        # 2026-05-10 — Profile-validator regenerate-loop: when chain_runner
        # has rejected the prior directive on profile-rule grounds, it
        # passes a `corrective_context` describing the violations. We append
        # it as a final, prominent block so Sonnet sees it last.
        if corrective_context:
            user_prompt = (user_prompt or "") + "\n\n" + corrective_context.strip() + "\n"
        # Phase 2: tenant-aware system prompt (Tier 2 catalog injected per tenant)
        system_prompt = build_supervisor_system_for_tenant(opp_meta.get("company"))
        t1 = time.monotonic()
        try:
            msg, raw = await _traced_anthropic_call(
                "supervisor.mode1b", system_prompt, user_prompt, max_tokens=2000)
            result["latency_ms"] = int((time.monotonic() - t1) * 1000)
            result["input_tokens"] = msg.usage.input_tokens
            result["output_tokens"] = msg.usage.output_tokens
        except Exception as e:
            result["fallback_reason"] = f"supervisor_call_failed: {e}"
            return result

        # 3. Parse
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            result["directive"] = json.loads(text)
        except json.JSONDecodeError as e:
            result["fallback_reason"] = f"parse_error: {e}"

        # 4. T-78 win-plan adherence retry — fires only when win_plan engaged
        # (preferred_actions populated by replayer). Enforces externally-
        # authored agent_actions_preferred against supervisor's choice.
        chosen = _directive_strategy(result["directive"])
        new_directive, retry_meta = await _retry_for_adherence(
            directive=result["directive"],
            preferred_actions=preferred_actions,
            user_prompt=user_prompt,
            chosen_strategy=chosen,
            cur_phase_label=current_phase_label,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["adherence_retry"] = retry_meta

        # 5. T-79 signal-driven adherence retry — fires when supervisor's
        # OWN signal_analysis output isn't honored by its strategy choice.
        # Independent of win_plan; uses BINDING RULES encoded as
        # SIGNAL_TO_ALLOWED_STRATEGIES. Re-reads chosen strategy because
        # the win-plan retry may have changed it.
        chosen = _directive_strategy(result["directive"])
        new_directive, signal_retry_meta = await _retry_for_signal_adherence(
            directive=result["directive"],
            user_prompt=user_prompt,
            chosen_strategy=chosen,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["signal_adherence_retry"] = signal_retry_meta

        # 6. R4 (Q12) — directive-internal-consistency retry. If the supervisor
        # emitted both `primary_strategy=reciprocity/objection_handling/logistics`
        # AND a `must_not_say` rule like "no unprompted price reduction" (which
        # contradicts the strategy that authorizes a price move), give the
        # supervisor one chance to revise BEFORE we silently strip the rule.
        # This converts the silent-fix into auditable supervisor reasoning.
        new_directive, consistency_retry_meta = await _retry_for_directive_consistency(
            directive=result["directive"],
            user_prompt=user_prompt,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["consistency_retry"] = consistency_retry_meta

        # 7. R6 — Phase 2.5 move-validity retry. If the supervisor's chosen
        # concrete_move is unknown / malformed / parameter-incomplete, re-prompt
        # with the valid catalog. Last in the retry chain — earlier retries may
        # have changed strategy.primary which could revalidate the move.
        new_directive, move_validity_retry_meta = await _retry_for_move_validity(
            directive=result["directive"],
            user_prompt=user_prompt,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["move_validity_retry"] = move_validity_retry_meta

        return result
    finally:
        if own_client:
            await http_client.aclose()


# ── Mode 2 — CGR3 multi-hop reasoning for high-stakes turns ─────────────────
async def mode2_directive(opp_meta: dict, dialog: list[dict], business_rules: str,
                            http_client: httpx.AsyncClient | None = None,
                            strategies_used: list[str] | None = None,
                            plan_state: "PlanState | None" = None,
                            pre_rendered_plan_section: str | None = None,
                            preferred_actions: list[str] | None = None,
                            current_phase_label: str | None = None,
                            moves_used: dict | None = None,
                            ) -> dict:
    """Mode 2: deep multi-hop reasoning. Used for high-stakes turns
    (commitment_5 close moments, sharp engagement drops, contradictions).

    Pipeline:
      1. CGR3 query — multi-hop iterative reasoning over CG (15-45s)
      2. Sonnet 4.5 supervisor synthesizes a directive grounded in the
         CGR3 response (which already contains multi-hop reasoning)
    """
    own_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=60)
        own_client = True

    # Conversation-phase tracker (mirrors mode1b — see §9.2 #7).
    from conversation_phase import classify_dialog, render_phase_block, PhaseState
    phase_history = classify_dialog(dialog)
    if phase_history:
        last = phase_history[-1]
        trailing = 0
        for p in reversed(phase_history):
            if p == last:
                trailing += 1
            else:
                break
    else:
        last, trailing = "greet", 0
    phase_state = PhaseState(current_phase=last, turns_in_phase=trailing, history=phase_history)
    cluster_plan_phase = None
    if plan_state is not None:
        try:
            cur = plan_state.current_phase()
            if cur:
                cluster_plan_phase = str(cur.get("phase_id"))
        except Exception:
            cluster_plan_phase = None
    phase_block = render_phase_block(phase_state, cluster_plan_phase)

    result = {
        "mode": "2",
        "directive": None,
        "cgr3_response_chars": 0,
        "cgr3_latency_ms": 0,
        "cg_evidence": {"products": False, "rules": False, "patterns": False, "decisions": False},
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "fallback_reason": None,
        "conversation_phase": {
            "current": phase_state.current_phase,
            "turns_in_phase": phase_state.turns_in_phase,
            "cluster_plan_phase": cluster_plan_phase,
            "history": phase_history,
        },
    }
    try:
        # 1a. CGR3 multi-hop query (synthesized reasoning) + 1b. decisions/search
        # (concrete precedents) — fired in parallel since they're independent calls
        cgr3_query_str = build_cg_query(opp_meta, dialog) + " best closing strategy customer at high commitment"
        # Decision-search query: prefers entity-name + cluster identifiers
        # (probe showed loose semantic queries return 0 on this endpoint).
        cluster_id = opp_meta.get("_cluster_id")
        decision_query_parts = [
            opp_meta.get("company") or "",
            f"c{cluster_id}" if cluster_id is not None else "",
            opp_meta.get("primary_motivator") or "",
            opp_meta.get("decision_logic") or "",
            "commitment",  # high-stakes turn = commitment-strategy is the canonical target
        ]
        decision_query = " ".join(p for p in decision_query_parts if p).strip()

        t0 = time.monotonic()
        cgr3_data, decisions_data = await asyncio.gather(
            cgr3_query(http_client, opp_meta["company"], cgr3_query_str, max_iterations=3),
            cg_decisions_search(http_client, opp_meta["company"],
                                  decision_query, top_k=3, min_confidence=0.3),
        )
        result["cgr3_latency_ms"] = int((time.monotonic() - t0) * 1000)
        cgr3_response = cgr3_data.get("response", "") or ""
        result["cgr3_response_chars"] = len(cgr3_response)
        result["n_decision_precedents"] = len(decisions_data.get("results") or [])
        result["decision_query"] = decision_query
        # CGR3 returns references — classify them for scenario evidence.
        # If references key is missing (not all server versions surface it),
        # fall back to scanning the response text for the canonical path tokens.
        refs = cgr3_data.get("references") or []
        ref_paths = [r.get("file_path", "") for r in refs if isinstance(r, dict)]
        if ref_paths:
            result["cg_evidence"] = classify_cg_evidence({"chunks": [{"file_path": p} for p in ref_paths]})
        elif cgr3_response:
            txt = cgr3_response.lower()
            result["cg_evidence"] = {
                "products": "product" in txt or "catalog" in txt,
                "rules": "rule" in txt or "policy" in txt or "playbook" in txt,
                "patterns": "conversation" in txt or "objection" in txt or "won-deal" in txt,
                "decisions": "decision" in txt or "approval" in txt or "discount" in txt,
            }

        # 2. Sonnet supervisor with CGR3 reasoning + decision-precedents embedded
        cgr3_section = (
            f"## CGR3 Multi-Hop Reasoning Result\n{cgr3_response[:4000]}\n\n"
            "(The above is a synthesized analysis from multi-hop graph traversal.)"
        )
        decision_section = render_decision_precedents(decisions_data)
        user_prompt = build_user_prompt(opp_meta, dialog, {}, business_rules,
                                          strategies_used=strategies_used,
                                          plan_state=plan_state,
                                          pre_rendered_plan_section=pre_rendered_plan_section,
                                          phase_block=phase_block,
                                          precedents=opp_meta.get("precedents"),
                                          moves_used=moves_used)
        user_prompt += "\n\n" + cgr3_section
        if decision_section:
            user_prompt += "\n\n" + decision_section

        # Phase 2: tenant-aware system prompt
        system_prompt = build_supervisor_system_for_tenant(opp_meta.get("company"))
        t1 = time.monotonic()
        try:
            msg, raw = await _traced_anthropic_call(
                "supervisor.mode2", system_prompt, user_prompt, max_tokens=2000)
            result["latency_ms"] = int((time.monotonic() - t1) * 1000)
            result["input_tokens"] = msg.usage.input_tokens
            result["output_tokens"] = msg.usage.output_tokens
        except Exception as e:
            result["fallback_reason"] = f"supervisor_call_failed: {e}"
            return result

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            result["directive"] = json.loads(text)
        except json.JSONDecodeError as e:
            result["fallback_reason"] = f"parse_error: {e}"

        # T-78 win-plan adherence retry
        chosen = _directive_strategy(result["directive"])
        new_directive, retry_meta = await _retry_for_adherence(
            directive=result["directive"],
            preferred_actions=preferred_actions,
            user_prompt=user_prompt,
            chosen_strategy=chosen,
            cur_phase_label=current_phase_label,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["adherence_retry"] = retry_meta

        # T-79 signal-driven adherence retry (independent of win_plan)
        chosen = _directive_strategy(result["directive"])
        new_directive, signal_retry_meta = await _retry_for_signal_adherence(
            directive=result["directive"],
            user_prompt=user_prompt,
            chosen_strategy=chosen,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["signal_adherence_retry"] = signal_retry_meta

        # R4 (Q12) — directive-internal-consistency retry, mirror of Mode 1b.
        new_directive, consistency_retry_meta = await _retry_for_directive_consistency(
            directive=result["directive"],
            user_prompt=user_prompt,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["consistency_retry"] = consistency_retry_meta

        # R6 — Phase 2.5 move-validity retry, mirror of Mode 1b.
        new_directive, move_validity_retry_meta = await _retry_for_move_validity(
            directive=result["directive"],
            user_prompt=user_prompt,
            tenant=opp_meta.get("company"),
        )
        result["directive"] = new_directive
        result["move_validity_retry"] = move_validity_retry_meta

        return result
    finally:
        if own_client:
            await http_client.aclose()


def build_segment_key(opp_meta: dict) -> str:
    """Construct the segment src_id used by emit/lookup. Mirrors the format
    that emit_decision_trace generates."""
    company = opp_meta.get("company", "?")
    cluster_id = opp_meta.get("_cluster_id", "?")
    motivator = (opp_meta.get("primary_motivator") or "any").replace(" ", "_")
    decision_logic = (opp_meta.get("decision_logic") or "any").replace(" ", "_")
    return f"{company}:c{cluster_id}:{motivator}:{decision_logic}"


def build_segment_fallback_chain(opp_meta: dict) -> list[tuple[int, str]]:
    """R27 — confidence-weighted cohort fallback. Returns a list of
    (level, segment_key_pattern) from most-specific to least-specific.
    The lookup tries each level in order until candidates are found.

    Levels:
      0 = exact 4-field match (tenant:c{N}:motivator:decision_logic)
      1 = drop cluster (tenant:c?:motivator:decision_logic) — any cluster
      2 = drop decision_logic (tenant:c?:motivator:?) — any cluster, any DL
      3 = tenant only (tenant:?:?:?) — last resort

    Each fallback level matches a wildcard pattern client-side; the lookup
    function knows how to match edges against these patterns.
    """
    company = opp_meta.get("company", "?")
    cluster_id = opp_meta.get("_cluster_id", "?")
    motivator = (opp_meta.get("primary_motivator") or "any").replace(" ", "_")
    decision_logic = (opp_meta.get("decision_logic") or "any").replace(" ", "_")
    return [
        (0, f"{company}:c{cluster_id}:{motivator}:{decision_logic}"),
        (1, f"{company}:c*:{motivator}:{decision_logic}"),
        (2, f"{company}:c*:{motivator}:*"),
        (3, f"{company}:c*:*:*"),
    ]


def _segment_matches_pattern(edge_src: str, pattern: str) -> bool:
    """True if the edge's src_id matches the pattern (where * = wildcard
    in any of the 4 colon-separated fields)."""
    if "*" not in pattern:
        return edge_src == pattern
    edge_parts = (edge_src or "").split(":")
    pat_parts = pattern.split(":")
    if len(edge_parts) != len(pat_parts):
        return False
    for ep, pp in zip(edge_parts, pat_parts):
        # Cluster wildcard: pattern says "c*" matches any "c{anything}"
        if pp == "c*":
            if not ep.startswith("c"):
                return False
            continue
        if pp == "*":
            continue
        if ep != pp:
            return False
    return True


# Cheap signal classifier — regex over last customer message. Mirrors the
# canonical SIGNAL_TO_ALLOWED_STRATEGIES taxonomy. NOT an LLM call.
import re as _re_signal
_SIGNAL_PATTERNS = [
    # R35 (2026-05-06) — customer-stated target. Empirically grounded in 50
    # won-deal Ecommerce conversations: when the customer reveals a specific
    # dollar/percentage target ("$80 off", "20% off", "I'd buy at $X"), the
    # right agent move is to match-or-go-above using a higher catalog tier
    # code, NOT to repeat the prior offer. The previous Ecommerce c5 batch had
    # 3/4 directives REGRESS price (offered $50 or $40 against a customer
    # who'd stated $80) because the supervisor couldn't see the target.
    # Pattern matches both Ecommerce ($X off, X% off) and Insurance (X USD).
    ("customer_stated_target", _re_signal.compile(
        # "$80 off" / "got a $52 deal" / "around $50 off" / "20% off discount"
        r"\$\s*\d+(?:\.\d+)?\s*(?:off|discount|deal)\b|"
        r"\b\d{1,2}\s*%\s*(?:off|discount)\b|"
        # "I would renew at X" / "I'd buy at X" / "match X" / "round to X"
        r"\b(?:i'?d|i would) (?:renew|buy|do it|take it|pay)\s*(?:at|for)\s*(?:\$|\b)?\s*\d+|"
        r"\bmatch\s+(?:my|that|the)?\s*(?:price|offer|quote)?\s*(?:of|at)?\s*\$?\s*\d+|"
        r"\b(?:round|drop|do) (?:it )?(?:to|down to|at)\s*\$?\s*\d+|"
        r"\bany\s*(?:thing)?\s*below\s*\$?\s*\d+|"
        # "holding out for $80" / "waiting for the $X off"
        r"\bholding out (?:for|until)\s*(?:the\s*)?\$?\s*\d+|"
        r"\bwaiting for\s*(?:the\s*)?\$?\s*\d+\s*(?:off)?",
        _re_signal.IGNORECASE)),
    ("explicit_price_request", _re_signal.compile(
        r"\b(is (there|this) (a )?discount|cheapest|best price|"
        r"(improve|lower|reduce) the price|how much|any discount|got a deal)\b",
        _re_signal.IGNORECASE)),
    ("explicit_objection_price", _re_signal.compile(
        r"\b(too (expensive|high|much)|expensive|out of budget|fiscally responsible|can'?t justify)\b",
        _re_signal.IGNORECASE)),
    ("competing_offer_mention", _re_signal.compile(
        r"\b(phoenix|yashir|wesure|direct|menorah|harel|clal|migdal|ayalon)\b|"
        r"\bcompet(itor|ing)\b|"
        r"\b(received|got|have) (a |an )?(quote|offer) (from|with)\b",
        _re_signal.IGNORECASE)),
    ("explicit_objection_competition", _re_signal.compile(
        r"\b\d{2,4}\s*(USD|usd|ש[\"״]?ח|dollars?|\$|dollar)?\s*cheaper\b|"
        r"\bfound (insurance|a deal|.* cheaper)\b",
        _re_signal.IGNORECASE)),
    ("commitment_signal", _re_signal.compile(
        r"\b(let'?s (proceed|go|close|do it)|sounds (good|like a plan)|"
        r"yes,? (proceed|let'?s|sure)|i('|')?ll (take|buy|do) it)\b|"
        r"\b\d{1,2} (installments?|payments?)\b",
        _re_signal.IGNORECASE)),
    ("pace_request", _re_signal.compile(
        r"\b(let me think|i'?ll get back|call me (back )?(later|on \w+|tomorrow)|"
        r"give me a (sec|moment|second)|hold on|grab my (wallet|card|phone)|"
        r"check the market|need to (think|consult|ask))\b",
        _re_signal.IGNORECASE)),
    ("disengagement", _re_signal.compile(
        r"^\s*no\s*[.!]?\s*$|"
        r"\b(not interested|don'?t (want|need|renew)|no thank you|"
        r"appreciate the offer.*\bbut\b|i'?ll pass|maybe (next|later))\b",
        _re_signal.IGNORECASE)),
    ("explicit_objection_timing", _re_signal.compile(
        r"\b(not (now|today|yet|right now)|too soon|wait (until|for)|"
        r"think about it|need more time)\b",
        _re_signal.IGNORECASE)),
    ("explicit_objection_product_fit", _re_signal.compile(
        r"\b(not sure|on the fence|durability|build quality|reviews|"
        r"complaints (online|about))\b",
        _re_signal.IGNORECASE)),
    ("trust_indicator", _re_signal.compile(
        r"\b(loyal|long.?time|been with you|years (with|as))\b",
        _re_signal.IGNORECASE)),
    ("product_inquiry", _re_signal.compile(
        r"\b(what (is|are|does)|how (does|do)|tell me (about|more)|"
        r"specifications?|specs|features?)\b",
        _re_signal.IGNORECASE)),
]


def quick_classify_signal(dialog: list) -> str | None:
    """Cheap regex-based signal classifier — no LLM. Used by the cache-lookup
    tier dispatcher BEFORE we know whether Mode 1b will be called. Returns
    None when no pattern matches the last customer message (cache lookup
    skips the signal filter in that case)."""
    last_customer = ""
    for msg in reversed(dialog or []):
        if msg.get("role") == "customer":
            last_customer = msg.get("text", "") or ""
            break
    if not last_customer:
        return None
    for name, pat in _SIGNAL_PATTERNS:
        if pat.search(last_customer):
            return name
    return None


# ── Cached-Directive Mode 1a (2026-05-04 PROPOSAL) ──────────────────────────
# Look up a past directive at the (segment, signal, phase) tuple. Returns
# the highest-scored cached directive if it meets the threshold, else None.
# Score = outcome_weight × confidence × recency_decay.

import re as _re_cache
import time as _time_cache
import json as _json_cache


def _parse_directive_blob(supporting_sentences: list) -> dict | None:
    """Pull `directive_v1:{...}` JSON out of supporting_sentences. Returns
    the parsed dict, or None when not found / not parseable."""
    if not supporting_sentences:
        return None
    for s in supporting_sentences:
        if isinstance(s, str) and s.startswith("directive_v1:"):
            try:
                return _json_cache.loads(s[len("directive_v1:"):])
            except Exception:
                return None
    return None


def _parse_emit_ts(provenance: str | None) -> int:
    """Extract the `ts=NNN` epoch-seconds marker from emit provenance.
    Falls back to current time when absent (treats as fresh)."""
    if not provenance:
        return int(_time_cache.time())
    m = _re_cache.search(r"\bts=(\d{9,11})\b", provenance)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return int(_time_cache.time())


def _outcome_weight(supporting_sentences: list) -> float:
    """Won = 1.0, incomplete = 0.5, lost = 0.2 (anything else)."""
    blob = " ".join(s for s in supporting_sentences if isinstance(s, str)).lower()
    if "outcome: won" in blob:
        return 1.0
    if "outcome: incomplete" in blob:
        return 0.5
    return 0.2


def _recency_decay(emitted_ts: int, half_life_days: float = 14.0) -> float:
    """Exponential decay; 14-day half-life by default."""
    age_seconds = max(0, int(_time_cache.time()) - emitted_ts)
    age_days = age_seconds / 86400.0
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def _score_cached_edge(edge: dict) -> float:
    rc = edge.get("relation_context") or {}
    sup = rc.get("supporting_sentences") or []
    confidence = float(rc.get("confidence_score") or 0.5)
    outcome_w = _outcome_weight(sup)
    ts = _parse_emit_ts(rc.get("provenance"))
    recency = _recency_decay(
        ts, half_life_days=float(os.environ.get("POC_CACHE_RECENCY_HALFLIFE_DAYS", "14"))
    )
    return outcome_w * confidence * recency


async def lookup_cached_directive(
    http_client: "httpx.AsyncClient",
    workspace: str,
    segment: str,
    primary_signal: str | None,
    phase: str | None,
    min_score: float | None = None,
    late_phase_recovery: bool = False,
    score_band: str | None = None,
) -> dict | None:
    """Query AgentKG for past directives at this (segment, signal, phase) tuple.
    Returns {directive: {...}, score: float, age_days: float, source: str} or None.

    The lookup does NOT pre-filter on signal/phase server-side (CG endpoint
    has no such filter); it pulls all decision_trace edges for the segment
    and filters client-side. This is fine because supervisor-emitted edges
    per segment are typically <100 rows.

    Late-phase recovery mode (2026-05-05): when `late_phase_recovery=True`,
    the filter prefers directive_v1 blobs tagged `late_phase_low_score=true`
    (set by the historical backfill on turns where the customer was in
    close_attempt+ with persuasion < 0.4). This surfaces "what real human
    agents did when stuck in this state" rather than the typical mid-flow
    pattern. score_band ∈ {low/very_low/extremely_low} narrows further.
    """
    if not CG_API_KEY:
        return None
    if min_score is None:
        min_score = float(os.environ.get("POC_CACHE_MIN_SCORE", "0.5"))

    # Pull all decisions for this workspace; filter by src=segment client-side.
    # /graph/decisions has no src filter, but it returns the full corpus and
    # we narrow by src_id locally.
    import time as _t_cg
    _t0 = _t_cg.monotonic()
    try:
        r = await http_client.get(
            f"{CG_URL}/graph/decisions",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": workspace},
            params={"min_confidence": 0.0},
            timeout=15,
        )
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        log.warning("cache lookup CG fetch failed: %s", e)
        # Trace the failure
        try:
            from trace_logger import TraceLogger
            _tr = TraceLogger.current()
            if _tr is not None:
                _tr.cg(
                    endpoint="/graph/decisions (cache_lookup)",
                    workspace=workspace,
                    query=f"segment={segment} signal={primary_signal} phase={phase}"
                          + (f" recovery={late_phase_recovery} band={score_band}" if late_phase_recovery else ""),
                    response={"_error": str(e)[:200]},
                    latency_ms=int((_t_cg.monotonic() - _t0) * 1000),
                )
        except Exception: pass
        return None

    # Trace the successful CG fetch (always — visible in /logs UI's CG section)
    try:
        from trace_logger import TraceLogger
        _tr = TraceLogger.current()
        if _tr is not None:
            _tr.cg(
                endpoint="/graph/decisions (cache_lookup)",
                workspace=workspace,
                query=f"segment={segment} signal={primary_signal} phase={phase}"
                      + (f" recovery={late_phase_recovery} band={score_band}" if late_phase_recovery else ""),
                response={"n_decisions": len(body.get("decisions", []) if isinstance(body, dict) else [])},
                latency_ms=int((_t_cg.monotonic() - _t0) * 1000),
            )
    except Exception: pass

    if not isinstance(body, dict):
        return None
    items = body.get("decisions") or body.get("data") or body.get("results") or []

    # R27 — confidence-weighted cohort fallback. Try most-specific first,
    # then progressively coarser. Stop at the first level that produces ≥1
    # candidate matching the (signal, phase) tuple.
    # Build patterns: [(0, exact), (1, c*:...), (2, c*:any), (3, tenant)]
    fallback_patterns = []
    if segment and ":" in segment:
        parts = segment.split(":")
        if len(parts) == 4:
            fallback_patterns = [
                (0, segment),                                        # exact
                (1, f"{parts[0]}:c*:{parts[2]}:{parts[3]}"),         # any cluster
                (2, f"{parts[0]}:c*:{parts[2]}:*"),                  # any cluster, any DL
                (3, f"{parts[0]}:c*:*:*"),                            # tenant only
            ]
    if not fallback_patterns:
        fallback_patterns = [(0, segment)]

    candidates = []
    fallback_level_used = 0
    for level, pattern in fallback_patterns:
        candidates = []
        for d in items:
            if not isinstance(d, dict):
                continue
            if not _segment_matches_pattern(d.get("src_id", ""), pattern):
                continue
            rc = d.get("relation_context") or {}
            # Multi-blob aware: an edge's supporting_sentences accumulates many
            # directive_v1 blobs. Score each blob individually so we can pick
            # the most-relevant one rather than just the first.
            ss = rc.get("supporting_sentences") or []
            if not isinstance(ss, list):
                continue
            for sentence in ss:
                if not (isinstance(sentence, str) and sentence.startswith("directive_v1:")):
                    continue
                try:
                    directive_v1 = _json_cache.loads(sentence[len("directive_v1:"):])
                except Exception:
                    continue
                # Soft signal/phase match — skip when both are present and mismatch.
                if primary_signal and directive_v1.get("primary_signal") and \
                        directive_v1.get("primary_signal") != primary_signal:
                    continue
                if phase and directive_v1.get("phase") and directive_v1.get("phase") != phase:
                    continue
                # Late-phase recovery filter: in recovery mode, REQUIRE the blob
                # to be tagged late_phase_low_score=true.
                if late_phase_recovery:
                    if not directive_v1.get("late_phase_low_score"):
                        continue
                    if score_band and directive_v1.get("score_band") and \
                            directive_v1.get("score_band") != score_band:
                        pass  # don't reject — just deprioritize
                score = _score_cached_edge(d)
                # Boost for late-phase recovery match
                if late_phase_recovery and directive_v1.get("late_phase_low_score"):
                    score += 0.2
                # Move-level outcome boost (Tier 2)
                delta = directive_v1.get("delta_after_move")
                if isinstance(delta, (int, float)):
                    outcome_boost = max(-0.3, min(0.3, delta * 0.6))
                    score += outcome_boost
                # R27 — fallback penalty: each level of broadening costs -0.05.
                # Keeps exact-match edges preferred when they exist.
                score -= 0.05 * level
                candidates.append((score, d, directive_v1))
        # If this fallback level produced any viable candidates, stop fallback
        if candidates and max(c[0] for c in candidates) >= min_score:
            fallback_level_used = level
            break

    if not candidates:
        return None

    candidates.sort(key=lambda t: -t[0])
    best_score, best_edge, best_directive = candidates[0]
    if best_score < min_score:
        return None

    rc = best_edge.get("relation_context") or {}
    age_days = (int(_time_cache.time()) - _parse_emit_ts(rc.get("provenance"))) / 86400.0
    return {
        "directive": best_directive,
        "score": best_score,
        "age_days": age_days,
        "source": "cache",
        "n_candidates": len(candidates),
        "edge_src": best_edge.get("src_id"),
        "edge_tgt": best_edge.get("tgt_id"),
        # R27 — telemetry: which fallback level produced this hit.
        # 0 = exact cohort match; 1 = any-cluster relax; 2 = any-DL relax;
        # 3 = tenant-only generic priors.
        "fallback_level": fallback_level_used,
    }


# ── Closed-loop learning — emit decision-trace edge ─────────────────────────
async def emit_decision_trace(opp_meta: dict, directive: dict, outcome: dict,
                                http_client: httpx.AsyncClient | None = None) -> dict:
    """Emit a decision-trace edge to CG capturing this directive + outcome.
    Demonstrates 'closed-loop learning' — each session contributes to the graph
    so future supervisor calls can retrieve precedents.

    Edge shape:
      src = customer-segment ('Insurance:Cluster4:Price/savings:Analytical')
      tgt = strategy ('strategy:objection_handling')
      relation_type = 'decision_trace'
      relation_context = {
        decision_trace: human-readable rationale,
        quantitative_data: 'commit_peak=4 persuasion_peak=0.8 outcome=won',
        supporting_sentences: ['agent said X', 'customer responded Y'],
        confidence_score: 0.0-1.0,
        provenance: 'POC supervisor v0.10 session_id=...'
      }
    """
    own_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=15)
        own_client = True

    if not CG_API_KEY:
        return {"emitted": False, "_error": "no LIGHTRAG_API_KEY"}

    CG_ENDPOINT_CALLS["decision_emit"] += 1
    company = opp_meta.get("company", "?")
    try:
        cluster_id = opp_meta.get("_cluster_id", "?")
        motivator = (opp_meta.get("primary_motivator") or "any").replace(" ", "_")
        decision_logic = (opp_meta.get("decision_logic") or "any").replace(" ", "_")
        strat = (directive.get("strategy") or {}).get("primary") or directive.get("primary_strategy") or "unknown"
        rules = directive.get("rules_to_enforce") or []
        rationale = (directive.get("audit") or {}).get("rationale_summary") or directive.get("rationale") or ""

        src_id = f"{company}:c{cluster_id}:{motivator}:{decision_logic}"
        tgt_id = f"strategy:{strat}"
        outcome_label = outcome.get("outcome", "incomplete")
        commit_peak = outcome.get("commitment_peak", 0) or 0
        persuasion_peak = outcome.get("persuasion_peak", 0) or 0

        # Multi-turn arc awareness (§9.2 #7): phase as edge metadata only —
        # NOT part of edge identity. Backwards-compatible with existing edges.
        phase_meta = outcome.get("phase") or {}
        phase_qd = ""
        if phase_meta:
            phase_qd = (
                f" phase={phase_meta.get('current','?')}"
                f" turns_in_phase={phase_meta.get('turns_in_phase',0)}"
            )
            if phase_meta.get("cluster_plan_phase"):
                phase_qd += f" cluster_plan_phase={phase_meta['cluster_plan_phase']}"

        # Cached-Directive Mode 1a (2026-05-04 PROPOSAL) — encode the FULL
        # directive in the edge so future Mode 1a turns can RECONSTRUCT it
        # without an LLM call. Stored as JSON-text inside supporting_sentences
        # (CG schema only allows string fields per supporting sentence).
        # Cache lookup parses the `directive_v1:...` line back into a dict.
        strategy = directive.get("strategy") or {}
        knowledge = directive.get("knowledge") or {}
        cm = strategy.get("concrete_move") if isinstance(strategy.get("concrete_move"), dict) else None
        directive_v1 = {
            "primary_strategy": strat,
            "tone": strategy.get("tone") or directive.get("tone"),
            "concrete_move": (
                {"name": cm.get("name"), "parameters": cm.get("parameters") or {}}
                if cm and cm.get("name") else None
            ),
            "must_not_say": [
                (m.get("text") if isinstance(m, dict) else str(m))[:200]
                for m in (knowledge.get("must_not_say") or directive.get("must_not_say") or [])[:3]
            ],
            "phase": (phase_meta.get("current") if phase_meta else None),
            "primary_signal": (
                directive.get("signal_analysis", {}).get("primary_signal")
                if isinstance(directive.get("signal_analysis"), dict)
                else directive.get("primary_signal")
            ),
        }
        # Stripped JSON to fit CG string field budget; key on prefix for parsing.
        import json as _json_for_cache
        directive_blob = "directive_v1:" + _json_for_cache.dumps(
            directive_v1, separators=(",", ":")
        )[:1800]

        payload = {
            "src": src_id,
            "tgt": tgt_id,
            "relation_type": "decision_trace",
            "relation_context": {
                "decision_trace": rationale[:500],
                "quantitative_data": f"commit_peak={commit_peak} persuasion_peak={persuasion_peak} outcome={outcome_label}{phase_qd}",
                "supporting_sentences": [
                    f"rules_enforced: {','.join(rules[:3])}",
                    f"outcome: {outcome_label}",
                    directive_blob,
                ],
                "confidence_score": 0.7 if outcome_label == "won" else 0.4,
                "provenance": f"POC supervisor session_id={outcome.get('session_id', '?')[:8]} ts={int(time.time())}",
            },
        }
        r = await http_client.post(
            f"{CG_URL}/graph/decision/emit",
            headers={"X-API-Key": CG_API_KEY, "LIGHTRAG-WORKSPACE": company},
            json=payload,
        )
        r.raise_for_status()
        return {"emitted": True, "src": src_id, "tgt": tgt_id, "status_code": r.status_code}
    except Exception as e:
        log.warning("decision_trace_emit failed: %s", e)
        return {"emitted": False, "_error": str(e)}
    finally:
        if own_client:
            await http_client.aclose()


# ── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def smoke():
        opp_meta = {
            "company": "Insurance",
            "opp_type": "Insurance Renewal",
            "primary_motivator": "Price/savings",
            "decision_logic": "Analytical",
            "trust_level": "Skeptical",
            "communication_style": "Terse",
            "objection_pattern": "Comparing competitor prices",
        }
        dialog = [
            {"role": "agent", "text": "Hey, this is Nofar from Insurance. Renewal: 4,238 USD for the year."},
            {"role": "customer", "text": "Other insurer offered me 3,800 USD, same coverage."},
        ]
        rules = "B29: When customer says 'I'll check elsewhere', offer soft retention check.\nB34: Don't drop price twice without resistance."

        print("Calling Mode 1b supervisor (CG + Sonnet 4.5)...")
        async with httpx.AsyncClient(timeout=30) as http:
            r = await mode1b_directive(opp_meta, dialog, rules, http_client=http)
        print(f"CG: entities={r['cg_n_entities']} relations={r['cg_n_relations']} chunks={r['cg_n_chunks']} ({r['cg_latency_ms']}ms)")
        print(f"Sonnet: in_tok={r['input_tokens']} out_tok={r['output_tokens']} latency={r['latency_ms']}ms")
        if r["fallback_reason"]:
            print(f"FALLBACK: {r['fallback_reason']}")
        if r["directive"]:
            d = r["directive"]
            print(f"\nstrategy: {d.get('strategy', {}).get('primary')}")
            print(f"tone: {d.get('strategy', {}).get('tone')}")
            print(f"facts_to_anchor: {d.get('knowledge', {}).get('facts_to_anchor', [])[:3]}")
            print(f"must_not_say: {d.get('knowledge', {}).get('must_not_say', [])[:3]}")
            print(f"rules: {d.get('rules_to_enforce', [])}")
            print(f"rationale: {d.get('audit', {}).get('rationale_summary', '')[:200]}")
        else:
            print("No directive returned")

    asyncio.run(smoke())
