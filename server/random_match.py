"""Random Conversation demo mode — sources clean-loss opps from DB by
audience-defined criteria, ranks by win-likelihood heuristic, returns top-K.

Architectural reasoning: research-notes/2026-04-30-… (POC architecture)
Demo rationale: pre-curated scenarios are vulnerable to "you cherry-picked these"
suspicion. This module lets the audience define criteria → system finds
candidates from prod data → picks the most-recoverable → demo runs that one.

Public API:
    build_candidate_cache() -> dict[opp_id → CandidateMeta]
        Run once at server startup. ~3 min cold (392 opps × 1 embedding each
        for substitute-content classification).
    load_or_build_candidate_cache() -> dict
        Disk-cached wrapper. Persists to data/random_match_cache.json so
        server restarts don't pay the 3-min embedding cost again.
        Auto-rebuilds if the source CSV (clean_loss_assignments.csv) is
        newer than the cache.
    find_best_match(criteria, cache, k=50, prefer_with_plan=True) -> dict
        Per-request matcher with hierarchical relaxation on empty result.
    heuristic_win_likelihood(features) -> float
        Score 0-10 for how recoverable a clean-loss opp looks.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_LOSS_BASE = os.environ.get(
    "POC_CLEAN_LOSS_BASE",
    "/home/dev/development_team_agent/research/ai-agent-sales-improvement/playbooks_v1",
)
PLANS_DIR_ENV = os.environ.get(
    "POC_PLANS_DIR",
    os.path.join(_PROJECT_ROOT, "data", "cluster_plans"),
)


# Legacy keyword-based phrase list — KEPT for fallback / fast-path.
# When embedding-based detection (preferred) is unavailable or returns
# uncertain similarity, regex fallback matches the high-precision phrases.
# Path B Day 1 (2026-05-01) replaced primary detection with semantic
# matching — see anchor sets below.
ADEQUATE_SUBSTITUTE_PHRASES = [
    "works fine", "works great", "working fine", "working great",
    "works perfectly", "happy with", "satisfied with", "no problem with",
    "no issues with", "have already", "i already have", "already got",
    "already own", "excellent headphone", "great headphone",
    "no need to upgrade", "don't need a new", "don't need new",
    "love my current", "love my", "love them",
    "עובד מצוין", "עובד טוב", "אני מרוצה", "מסתפק", "אין בעיה",
    "כבר יש לי", "אין צורך", "לא צריך",
]


# ── Path B Day 1: semantic Karl-detector via Gemini embeddings ──────────────
# Anchor sets — natural-language examples of two opposing classes.
# Curated 2026-05-01 from observed customer messages across our test_logs.
# Keep these to ~12-15 each; tuning + threshold matter more than length.

ANCHOR_KARL_CLASS = [
    # Karl-class = customer has adequate substitute (typically a competitor's
    # product or no felt need); empirically uncloseable regardless of effort
    "my current product works fine, I don't need to switch",
    "I'm satisfied with what I have right now",
    "what I have already does the job for me",
    "I don't see why I'd spend money on something I don't need",
    "no problem to solve, my current setup is adequate",
    "I'm not in the market right now",
    "my old setup is still working perfectly fine",
    "I have an excellent product already, no reason to look",
    "I just bought one recently, no plans to replace it",
    "my Sony works great, I'm not looking",
    "the Audio-Technica I have does everything I need",
    "I love my current pair, why would I switch",
    "I deal with problems when they actually happen, not hypothetically",
]

ANCHOR_RICHARD_CLASS = [
    # Richard-class = power-user of OUR product; substitute language is
    # actually validation; recoverable (often wants a SECOND of our product)
    "I have your product and want a second one",
    "I'm a long-time customer, want to upgrade",
    "I've had my Ecommerce for six months and use them daily",
    "love your H1H, time for a backup pair",
    "had them for months, want another for travel",
    "my current Ecommerce are great but I want a dedicated pair for travel",
    "I'm an existing customer, considering an additional purchase",
    "your headphones changed my listening, want a second set",
    "Insurance has been my insurer for years, time to renew",
    "long-time policyholder, want to discuss my options",
]


def _embed_safe(texts: list[str]):
    """Wrapper around embed() that swallows exceptions and falls back to None
    so detection can fall back to regex if embeddings unavailable."""
    try:
        from embeddings import embed
        return embed(texts)
    except Exception as e:
        log.warning("embedding unavailable, falling back to regex: %s", e)
        return None


def _build_class_centroids(rebuild: bool = False) -> tuple:
    """Build (or load) Karl + Richard centroids. Returns (karl_centroid, richard_centroid)
    or (None, None) if embeddings unavailable."""
    try:
        from embeddings import get_or_build_centroid
        karl = get_or_build_centroid("karl_class", ANCHOR_KARL_CLASS, rebuild=rebuild)
        richard = get_or_build_centroid("richard_class", ANCHOR_RICHARD_CLASS, rebuild=rebuild)
        return karl, richard
    except Exception as e:
        log.warning("centroid build failed: %s", e)
        return None, None


# Lazy-init centroids on first call
_KARL_CENTROID = None
_RICHARD_CENTROID = None
_CENTROIDS_INITIALIZED = False
KARL_DETECTION_THRESHOLD = 0.55  # cosine similarity above this → Karl-class
RICHARD_OVERRIDE_THRESHOLD = 0.50  # if Richard sim above this → power-user override


# Bug fix 2026-05-01 (Richard/7f555fce WIN finding): the previous adequate-
# substitute detector incorrectly penalized POWER-USERS who own OUR product
# already and want a SECOND pair. Richard said "had them for about six months"
# — Ecommerce H1H — and converted to commit=5. Same surface phrase as Karl, but
# opposite buy-intent. Differentiate: if the historical mentions OUR brand or
# product names alongside the "have/own/use" language, it's a power-user
# (recoverable, no penalty). If a competitor's name (Odioone, Sony, Bose, etc.)
# OR no brand reference at all, it's Karl-class (penalty).
OUR_PRODUCT_NAMES_BY_TENANT = {
    "Ecommerce": ["ecommerce", "h1h", "h1 ", "h1p", "shells", "heavy headphones"],
    "Insurance": ["insurance"],
    "MattressCommerce": ["mattresscommerce"],
    "SaaS": ["SaaS"],
    "CleaningCommerce": ["cleaningcommerce"],
}

# Known competitor / alternative brands per tenant — when customer mentions
# these AND has substitute language, it's Karl-class (uncloseable).
COMPETITOR_BRANDS_BY_TENANT = {
    "Ecommerce": [
        "sony", "bose", "audio-technica", "audio technica", "marshall", "marley",
        "beats", "apple", "airpods", "samsung", "skullcandy", "sennheiser",
        "shure", "akg", "audeze", "hifiman", "philips", "jbl", "razer",
        "steelseries", "logitech", "odioone",  # the canonical Karl brand
    ],
    "Insurance": [
        "menora", "aig", "phoenix", "shlomo", "shirbit", "harel", "ayalon",
        "clal", "migdal", "passport-card", "passport card",
    ],
    "MattressCommerce": ["ikea", "sealy", "tempur", "purple", "casper", "leesa", "saatva"],
    "SaaS": ["dubsado", "studio ninja", "17hats", "tave", "hellobonsai"],
}


def _detect_adequate_substitute(early_customer_msgs: list[str],
                                  tenant: str | None = None) -> tuple[bool, bool]:
    """Detect Karl-class adequate-substitute pattern using semantic matching
    (Path B Day 1, 2026-05-01) with regex fallback if embeddings unavailable.

    Returns:
        (has_adequate_substitute, has_our_product_already)
        - has_adequate_substitute: triggers heuristic_win_likelihood penalty.
          True iff customer's text is semantically Karl-class AND not Richard-class
          AND no explicit OUR-product brand mention.
        - has_our_product_already: True if Richard-class semantic OR explicit
          OUR-brand mention (power-user signal; diagnostic + future features).

    Algorithm:
        1. Lazy-load Karl + Richard centroids (built once from anchor sets).
        2. Embed customer's early-message text.
        3. Compute karl_sim, richard_sim.
        4. Classify:
           - has_our = (richard_sim > 0.50) OR explicit our-brand keyword
           - has_substitute = (karl_sim > 0.55) AND NOT has_our
        5. Fallback: if embeddings fail, use regex (legacy behavior).
    """
    global _KARL_CENTROID, _RICHARD_CENTROID, _CENTROIDS_INITIALIZED
    if not early_customer_msgs:
        return False, False

    text = " ".join(early_customer_msgs)[:2000]
    text_lower = text.lower()
    our_names = OUR_PRODUCT_NAMES_BY_TENANT.get(tenant, []) if tenant else []
    has_explicit_our_brand = any(b in text_lower for b in our_names)

    # Lazy-init centroids once per process
    if not _CENTROIDS_INITIALIZED:
        _KARL_CENTROID, _RICHARD_CENTROID = _build_class_centroids()
        _CENTROIDS_INITIALIZED = True

    # Path 1: combined competitor-brand detection + semantic delta
    # Karl ↔ Richard centroids share 87% similarity (sales-context ownership
    # language clusters tightly), so absolute similarity is unreliable.
    # Empirical tuning 2026-05-01: use brand detection as primary signal,
    # delta as secondary disambiguator.
    competitor_brands = COMPETITOR_BRANDS_BY_TENANT.get(tenant, []) if tenant else []
    has_competitor_brand = any(b in text_lower for b in competitor_brands)

    if _KARL_CENTROID is not None and _RICHARD_CENTROID is not None:
        from embeddings import cosine_similarity
        msg_embed = _embed_safe([text])
        if msg_embed is not None and len(msg_embed) > 0:
            karl_sim = cosine_similarity(msg_embed[0], _KARL_CENTROID)
            richard_sim = cosine_similarity(msg_embed[0], _RICHARD_CENTROID)
            sim_delta = karl_sim - richard_sim
            log.debug("karl_detector: karl_sim=%.3f richard_sim=%.3f delta=%+.3f "
                      "competitor_brand=%s our_brand=%s",
                      karl_sim, richard_sim, sim_delta,
                      has_competitor_brand, has_explicit_our_brand)

            # Decision rules (priority order):
            # 1. Explicit our-brand → power-user (Richard-class)
            if has_explicit_our_brand:
                return False, True
            # 2. Competitor brand + Karl-leaning semantic OR clear semantic Karl
            #    → Karl-class (uncloseable)
            if has_competitor_brand and sim_delta > -0.05:
                # Competitor mentioned + not strongly Richard-leaning
                return True, False
            if sim_delta > 0.07:
                # Clearly Karl-leaning regardless of brand
                return True, False
            # 3. Clear Richard-leaning semantic → power-user
            if sim_delta < -0.05:
                return False, True
            # 4. Ambiguous → no flag either way
            return False, False

    # Path 2: regex fallback (embeddings unavailable)
    has_substitute_phrase = any(p in text_lower for p in ADEQUATE_SUBSTITUTE_PHRASES)
    if has_substitute_phrase and has_explicit_our_brand:
        return False, True
    return has_substitute_phrase, has_explicit_our_brand


# ── Heuristic scorer ────────────────────────────────────────────────────────
def heuristic_win_likelihood(features: dict) -> float:
    """Score 0-10 for how recoverable a clean-loss opp looks.

    Positive signals: customer reached engagement (max_p_conv, max_commit,
    had_high_pconv); fell off after climbing (p_conv_drop); responded to
    specific moves (used_*).
    Negative signals: signed elsewhere (customer_announced_competitor);
    hard ghost-pattern (b41_triggered); ADEQUATE-SUBSTITUTE language in
    early customer messages (Karl-class — empirically uncloseable).
    """
    s = 0.0
    s += 1.5 if features.get("had_high_pconv") else 0.0
    s += 0.6 * float(features.get("max_commit", 0) or 0)         # 0..3
    s += 1.0 * float(features.get("max_p_conv", 0) or 0)         # 0..1
    s += 0.5 * min(1.0, float(features.get("n_msgs", 0) or 0) / 20)  # 0..0.5
    p_drop = float(features.get("p_conv_drop", 0) or 0)
    if p_drop > 0.3:
        s += 0.5  # post-engagement drop = recoverable
    if features.get("used_objection_handling"):
        s += 0.3
    if features.get("used_direct_ask"):
        s += 0.3
    # Negatives
    if features.get("customer_announced_competitor"):
        s -= 2.5
    if features.get("b41_triggered"):
        s -= 1.5
    if features.get("customer_b29_trigger"):
        s -= 0.5
    # 2026-05-01 fix: heavy penalty for customers with adequate substitute.
    # Empirical evidence: Karl (0/11+ tests) had this signature; was scoring
    # 5.75/10 but in reality is genuinely uncloseable.
    if features.get("has_adequate_substitute"):
        s -= 3.0
    return max(0.0, min(10.0, s))


# ── Plan availability lookup ───────────────────────────────────────────────
def _plan_filename(tenant: str, cluster_id: int, motivator: str,
                    decision_logic: str) -> str:
    mot = (motivator or "").replace("/", "_")
    dl = (decision_logic or "").replace("/", "_")
    return f"{tenant}__c{cluster_id}__{mot}__{dl}.json"


def _check_plan_available(tenant: str, cluster_id: int | None,
                           motivator: str | None, decision_logic: str | None,
                           plans_dir: str = PLANS_DIR_ENV) -> tuple[bool, str | None]:
    """Returns (has_plan, plan_id_or_none) without loading the JSON content."""
    if cluster_id is None or not motivator or not decision_logic:
        return False, None
    fname = _plan_filename(tenant, cluster_id, motivator, decision_logic)
    full = os.path.join(plans_dir, fname)
    if os.path.isfile(full):
        return True, f"{tenant}:c{cluster_id}:{motivator}:{decision_logic}"
    return False, None


# ── Cache builder ──────────────────────────────────────────────────────────
def build_candidate_cache() -> dict:
    """One-shot startup cache.

    Sources:
      1. clean_loss_assignments.csv per tenant — pre-classified into clusters
      2. opportunity + research_profile_flash — fills opp_type, trust_level
      3. research_turn_state_flash aggregates — fills heuristic features
      4. cluster_plans/ — fills has_plan + plan_id

    Returns: dict {opp_id: {tenant, opp_type, motivator, decision_logic,
                              trust_level, cluster_id, n_msgs, max_commit,
                              max_p_conv, p_conv_drop, had_high_pconv,
                              customer_announced_competitor, b41_triggered,
                              customer_b29_trigger, used_objection_handling,
                              used_direct_ask, has_plan, plan_id, score}}
    """
    # Lazy-import db so this module can be imported without DB conn
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from db import open_conn

    cache: dict = {}

    # Stage 1 — load pre-classified opps from CSV
    pre_classified = {}  # opp_id → (tenant, cluster_id, motivator, decision_logic)
    for tenant in ("Insurance", "Ecommerce"):
        path = f"{CLEAN_LOSS_BASE}/{tenant}/clean_loss_assignments.csv"
        if not os.path.isfile(path):
            log.warning("clean_loss_assignments not found for %s", tenant)
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                opp_id = (row.get("opp_id") or "").strip()
                if not opp_id:
                    continue
                cluster_raw = (row.get("cluster") or "").strip()
                cluster_id = int(cluster_raw) if cluster_raw and cluster_raw.lstrip("-").isdigit() else None
                motivator = (row.get("motivator") or "").strip() or None
                decision_logic = (row.get("decision_logic") or "").strip() or None
                pre_classified[opp_id] = (tenant, cluster_id, motivator, decision_logic)

    log.info("loaded %d pre-classified opps from clean_loss_assignments", len(pre_classified))

    if not pre_classified:
        return cache

    # Stage 2 — bulk fetch DB metadata (opp_type, trust_level) + turn-state aggregates
    opp_ids = list(pre_classified.keys())
    placeholders = ",".join(["%s"] * len(opp_ids))
    conn = open_conn()
    try:
        with conn.cursor() as cur:
            # Opportunity + profile
            cur.execute(f"""
                SELECT o.id AS opp_id, o.type AS opp_type,
                       o.total_inbounds + o.total_outbounds AS n_msgs,
                       p.trust_level
                FROM opportunity o
                LEFT JOIN research_profile_flash p ON p.opportunity_id = o.id
                WHERE o.id IN ({placeholders})
            """, opp_ids)
            opp_meta = {r["opp_id"]: r for r in cur.fetchall()}

            # Turn-state aggregates per opp
            cur.execute(f"""
                SELECT opportunity_id AS opp_id,
                       MAX(commitment_level) AS max_commit,
                       MAX(p_conv) AS max_p_conv,
                       (MAX(p_conv) - MIN(CASE WHEN sequence_number > (
                           SELECT sequence_number FROM research_turn_state_flash t2
                           WHERE t2.opportunity_id = t.opportunity_id
                             AND t2.p_conv = (SELECT MAX(p_conv) FROM research_turn_state_flash t3
                                              WHERE t3.opportunity_id = t.opportunity_id)
                           LIMIT 1
                       ) THEN p_conv END)) AS p_conv_drop_unsafe,
                       SUM(CASE WHEN p_conv >= 0.6 THEN 1 ELSE 0 END) > 0 AS had_high_pconv
                FROM research_turn_state_flash t
                WHERE opportunity_id IN ({placeholders})
                GROUP BY opportunity_id
            """, opp_ids)
            ts_agg = {r["opp_id"]: r for r in cur.fetchall()}

            # First 5 inbound (customer) messages per opp — for adequate-substitute detection
            cur.execute(f"""
                SELECT me.opportunity_id AS opp_id, me.timestamp,
                       COALESCE(
                         (SELECT m.text FROM message m
                          WHERE m.opportunity_id=me.opportunity_id AND m.message_id=me.message_id
                            AND m.lang='en_US' LIMIT 1),
                         (SELECT m.text FROM message m
                          WHERE m.opportunity_id=me.opportunity_id AND m.message_id=me.message_id
                            AND m.lang='he' LIMIT 1),
                         (SELECT m.text FROM message m
                          WHERE m.opportunity_id=me.opportunity_id AND m.message_id=me.message_id
                          LIMIT 1)
                       ) AS text
                FROM message_event me
                WHERE me.opportunity_id IN ({placeholders})
                  AND me.type = 'inbound'
                  AND (me.is_deleted IS NULL OR me.is_deleted = 0)
                ORDER BY me.opportunity_id, me.timestamp ASC
            """, opp_ids)
            early_customer_msgs: dict[str, list[str]] = {}
            for r in cur.fetchall():
                lst = early_customer_msgs.setdefault(r["opp_id"], [])
                if len(lst) < 5 and r["text"]:
                    lst.append(r["text"])
    finally:
        conn.close()

    # Stage 3 — assemble cache
    for opp_id, (tenant, cluster_id, motivator, decision_logic) in pre_classified.items():
        meta = opp_meta.get(opp_id, {})
        ts = ts_agg.get(opp_id, {})
        has_plan, plan_id = _check_plan_available(tenant, cluster_id, motivator, decision_logic)

        # p_conv_drop is approximate without per-turn ordering; just use max - final or 0
        max_p = float(ts.get("max_p_conv") or 0)
        n_msgs_val = int(meta.get("n_msgs") or 0)

        early_msgs = early_customer_msgs.get(opp_id, [])
        has_adequate, has_our_product = _detect_adequate_substitute(early_msgs, tenant)

        # Pre-compute feature dict for the heuristic scorer
        features = {
            "had_high_pconv": bool(ts.get("had_high_pconv")),
            "max_commit": int(ts.get("max_commit") or 0),
            "max_p_conv": max_p,
            "p_conv_drop": max_p,  # approximation: peak persuasion = potential to recover from
            "n_msgs": n_msgs_val,
            "customer_announced_competitor": False,  # we don't have this signal pre-computed; default false
            "b41_triggered": False,
            "customer_b29_trigger": False,
            "used_objection_handling": False,
            "used_direct_ask": False,
            "has_adequate_substitute": has_adequate,  # NEW — Karl-class detector
        }
        score = heuristic_win_likelihood(features)

        cache[opp_id] = {
            "opp_id": opp_id,
            "tenant": tenant,
            "cluster_id": cluster_id,
            "motivator": motivator,
            "decision_logic": decision_logic,
            "opp_type": meta.get("opp_type"),
            "trust_level": meta.get("trust_level"),
            "n_msgs": n_msgs_val,
            "max_commit": features["max_commit"],
            "max_p_conv": max_p,
            "had_high_pconv": features["had_high_pconv"],
            "has_adequate_substitute": has_adequate,
            "has_our_product_already": has_our_product,
            "has_plan": has_plan,
            "plan_id": plan_id,
            "score": round(score, 2),
        }

    log.info("candidate cache built: %d opps total", len(cache))
    log.info("  with plan: %d", sum(1 for c in cache.values() if c["has_plan"]))
    log.info("  by tenant: Insurance=%d Ecommerce=%d",
              sum(1 for c in cache.values() if c["tenant"] == "Insurance"),
              sum(1 for c in cache.values() if c["tenant"] == "Ecommerce"))
    return cache


# ── Matcher with hierarchical relaxation ───────────────────────────────────
def _matches_criteria(c: dict, criteria: dict, relaxed: set[str]) -> bool:
    """Apply criteria with selectively-relaxed axes."""
    for key in ("tenant", "opp_type", "motivator", "decision_logic", "trust_level"):
        if key in relaxed:
            continue
        want = criteria.get(key)
        if not want or want == "any":
            continue
        if c.get(key) != want:
            return False
    return True


# Hierarchical relaxation order — drop most-specific axes first
_RELAXATION_ORDER = ["trust_level", "decision_logic", "opp_type", "motivator"]


def find_best_match(criteria: dict, cache: dict, *, k: int = 50,
                     prefer_with_plan: bool = True,
                     deterministic: bool = False,
                     top_pool_size: int = 15) -> dict:
    """Find the best clean-loss candidate matching audience criteria.

    Algorithm:
      1. Filter cache by exact criteria. If <k results, relax axes one at a time.
      2. If prefer_with_plan, partition into has-plan / no-plan. Use has-plan
         pool first; fall back to no-plan only if has-plan is empty.
      3. Sort by score desc, take top_pool_size as the "high-quality pool".
      4. Weighted-random sample 5 from that pool (probability proportional to
         score). Same criteria → different result each call (variety).
         Pass deterministic=True to disable randomization (returns deterministic
         top-5 like the original behavior — useful for tests).

    Bug fix 2026-05-01: previously sorted+sliced top-5, so identical criteria
    always returned identical results. Now samples within top-15 to give the
    presenter variety while still preferring high-scored candidates.
    """
    criteria = {k: v for k, v in criteria.items() if v}  # drop empty
    relaxed: set[str] = set()
    candidates: list[dict] = []

    # Stage 1: exact match → relaxation hierarchy
    while True:
        candidates = [c for c in cache.values() if _matches_criteria(c, criteria, relaxed)]
        if len(candidates) >= 5:
            break
        # Find next axis to relax that's actually constrained
        next_to_relax = None
        for axis in _RELAXATION_ORDER:
            if axis in criteria and axis not in relaxed:
                next_to_relax = axis
                break
        if next_to_relax is None:
            break  # nothing left to relax
        relaxed.add(next_to_relax)
        log.info("random_match: relaxing axis '%s' (had %d candidates, need 5)",
                  next_to_relax, len(candidates))

    n_total = len(candidates)

    # Stage 2: partition by plan availability
    if prefer_with_plan:
        with_plan = [c for c in candidates if c["has_plan"]]
        no_plan = [c for c in candidates if not c["has_plan"]]
        if with_plan:
            chosen_pool = with_plan
            pool_label = "with_plan"
        else:
            chosen_pool = no_plan
            pool_label = "no_plan_fallback"
    else:
        chosen_pool = candidates
        pool_label = "all"

    # Stage 3: rank by score desc, take top_pool_size as the high-quality pool
    chosen_pool.sort(key=lambda c: -c["score"])
    high_quality_pool = chosen_pool[:top_pool_size]

    if deterministic or len(high_quality_pool) <= 5:
        top_5 = high_quality_pool[:5]
    else:
        # Weighted-random sample 5 from the pool (without replacement).
        # Weight = score + 0.1 (floor to handle 0-scored candidates).
        # This guarantees variety on repeated calls while preferring quality.
        available = list(high_quality_pool)
        weights = [c["score"] + 0.1 for c in available]
        sampled = []
        for _ in range(min(5, len(available))):
            if not available:
                break
            idx = random.choices(range(len(available)), weights=weights, k=1)[0]
            sampled.append(available.pop(idx))
            weights.pop(idx)
        top_5 = sampled

    return {
        "n_total_candidates": n_total,
        "n_after_plan_filter": len(chosen_pool),
        "high_quality_pool_size": len(high_quality_pool),
        "pool_used": pool_label,
        "relaxations_applied": sorted(relaxed),
        "criteria_received": criteria,
        "top_5": top_5,
        "best_match_opp_id": top_5[0]["opp_id"] if top_5 else None,
        "sampling_mode": "deterministic" if deterministic else "weighted_random_top_15",
    }


# ── Disk-cache wrapper ─────────────────────────────────────────────────────

_CACHE_FILE = os.environ.get(
    "POC_RANDOM_MATCH_CACHE",
    os.path.join(_PROJECT_ROOT, "data", "random_match_cache.json"),
)


def _csv_paths_for_freshness_check() -> list[str]:
    """The source CSVs that should invalidate the cache when newer."""
    return [
        f"{CLEAN_LOSS_BASE}/Insurance/clean_loss_assignments.csv",
        f"{CLEAN_LOSS_BASE}/Ecommerce/clean_loss_assignments.csv",
    ]


def _cache_is_fresh(cache_path: str) -> bool:
    """Cache is fresh if it exists and its mtime is >= every source CSV mtime."""
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for csv_path in _csv_paths_for_freshness_check():
        if os.path.exists(csv_path) and os.path.getmtime(csv_path) > cache_mtime:
            log.info("random_match cache stale: source %s newer than cache", csv_path)
            return False
    return True


def load_or_build_candidate_cache() -> dict:
    """Load the candidate cache from disk if fresh; else rebuild and persist.

    Cold rebuild costs ~3 min (392 opps × 1 embedding each). Disk-load is
    sub-second. The cache invalidates only when the source clean_loss CSVs
    change mtime, which happens rarely (manual data refresh)."""
    if _cache_is_fresh(_CACHE_FILE):
        try:
            t0 = time.time()
            with open(_CACHE_FILE) as f:
                cache = json.load(f)
            log.info("random_match cache: loaded %d opps from disk in %dms (path=%s)",
                     len(cache), int((time.time() - t0) * 1000), _CACHE_FILE)
            return cache
        except Exception as e:
            log.warning("random_match cache disk-load failed (%s); rebuilding", e)

    log.info("random_match cache: building (~3 min, embeds 392 opps)")
    t0 = time.time()
    cache = build_candidate_cache()
    elapsed = int(time.time() - t0)

    # Persist to disk so subsequent restarts skip the rebuild
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, default=str)
        log.info("random_match cache: built %d opps in %ds, persisted to %s",
                 len(cache), elapsed, _CACHE_FILE)
    except Exception as e:
        log.warning("random_match cache disk-persist failed: %s (cache still in memory)", e)
    return cache
