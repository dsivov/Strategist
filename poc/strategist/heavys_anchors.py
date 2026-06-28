"""T-86 — Heavys product anchors (parallel of T-81 Libra economic-anchor).

Source: Context Graph workspace `Heavys` (mixed-case; the all-caps `HEAVYS`
workspace is empty — verified 2026-05-03).

Output: a structured dict the supervisor / actor can inject into its
`facts_to_anchor` pool to enable concrete value-stacking, feature
differentiation, and risk-reversal moves on Heavys cart-abandon scenarios.

Design choices:
- Process-level cache: Heavys product data is tenant-wide (not per-opp), so
  one CG round-trip per server lifetime is sufficient. Refresh on TTL or
  server restart.
- Multi-query extraction: ONE big query produces summary text but loses
  structure. Several focused queries each return a clean answer for one
  anchor field.
- Defensive: any single query failing leaves the field empty rather than
  bombing out the whole anchor load.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


CG_URL = "http://18.153.178.170:9621"
WORKSPACE = "Heavys"  # mixed-case — verified to have data
CACHE_TTL_S = 60 * 60 * 4  # 4 hours; product catalog rarely changes
_CACHE: dict[str, Any] | None = None
_CACHE_TS: float = 0


# Query plan: each item produces one anchor field.
# Keep queries focused — LightRAG does best with concrete questions.
_ANCHOR_QUERIES: list[tuple[str, str, str]] = [
    # (anchor_field, query, mode)
    ("products_summary",
     "List the Heavys headphone products available for sale, including the H1H bundle and any gaming or artist editions. For each, give the model name and 3-5 most important features.",
     "hybrid"),
    ("bundle_components",
     "What is included in the Heavys H1H Headphones Bundle? List every item that ships in the box.",
     "local"),
    ("included_features",
     "What audio features and technologies does the Heavys H1H have? Include: speaker arrangement, microphones, noise cancellation, battery, connectivity, app support.",
     "local"),
    ("warranty_and_returns",
     "What is Heavys's warranty policy and return policy? Include any time limits and conditions.",
     "local"),
    ("payment_options",
     "What payment methods and installment plans does Heavys accept?",
     "local"),
    ("shipping_terms",
     "What are Heavys's shipping terms? Include free shipping thresholds, international shipping, delivery time.",
     "local"),
    ("competitor_differentiators",
     "What makes Heavys headphones different from competitor headphones like Sony, Bose, Audio Technica, Marshall, or AirPods? Focus on metal-music-specific features and design choices.",
     "hybrid"),
    ("social_proof",
     "What metal artists, bands, or notable customers endorse, partner with, or use Heavys headphones? Include user counts or community size if mentioned.",
     "hybrid"),
    ("current_promotions",
     "What discount codes, coupons, or promotional offers are currently active for Heavys? Include any priority order, time limits, or stacking rules.",
     "hybrid"),
]


async def _cg_query(http: httpx.AsyncClient, query: str, mode: str) -> str:
    api_key = os.environ.get("LIGHTRAG_API_KEY", "")
    if not api_key:
        log.warning("heavys_anchors: LIGHTRAG_API_KEY not set; cannot query CG")
        return ""
    try:
        r = await http.post(
            f"{CG_URL}/query",
            json={"query": query, "mode": mode},
            headers={"X-API-Key": api_key, "LIGHTRAG-WORKSPACE": WORKSPACE,
                     "Content-Type": "application/json"},
            timeout=20.0,
        )
        r.raise_for_status()
        body = r.json()
        return (body.get("response") or "").strip()
    except Exception as e:
        log.warning("heavys_anchors: query failed (%s): %s", query[:60], e)
        return ""


async def _fetch_all_anchors() -> dict[str, str]:
    """Run all anchor queries concurrently. Returns {field: response_text}."""
    async with httpx.AsyncClient() as http:
        tasks = [_cg_query(http, q, m) for _, q, m in _ANCHOR_QUERIES]
        responses = await asyncio.gather(*tasks, return_exceptions=False)
    return {field: resp for (field, _, _), resp in zip(_ANCHOR_QUERIES, responses)}


# Local-policy anchors that don't come from CG — Heavys business rules + T-86 internal.
_INTERNAL_POLICY_ANCHORS: dict[str, Any] = {
    # Hard cap: supervisor must never exceed this discount magnitude
    # without escalation. Set conservatively; tighten/loosen as we learn.
    "max_authorized_discount_pct_internal": 25,
    # When the customer presents a competitor offer, the supervisor must
    # probe these dimensions before any price move:
    "apples_to_apples_probe_axes": [
        "battery_life_hours",
        "noise_cancellation_type",
        "driver_count_or_arrangement",
        "warranty_length",
        "return_window_days",
        "included_accessories",
    ],
    # Strategy primitive recommendations per inflection signal:
    "preferred_strategies": {
        "explicit_objection_price": ["objection_handling", "authority", "social_proof"],
        "competing_offer_mention": ["objection_handling", "authority"],
        "explicit_objection_trust": ["authority", "social_proof", "empathy"],
        "explicit_objection_product_fit": ["information", "authority"],
        "commitment_signal": ["logistics", "direct_ask"],
    },
    # Things the agent must never say (Heavys-specific):
    "must_not_say": [
        "I'll check with my manager and get back to you",  # non-existent process
        "We can do a custom payment plan",  # only stated methods allowed
        "I'll match any competitor's price",  # not policy
    ],
}


async def fetch_heavys_anchors(force_refresh: bool = False) -> dict:
    """Public entry. Returns the cached or freshly-fetched Heavys anchor pack.
    Process-level cache with TTL — Heavys product data is tenant-wide so one
    fetch per server lifetime is enough."""
    global _CACHE, _CACHE_TS
    now = time.time()
    if not force_refresh and _CACHE and (now - _CACHE_TS) < CACHE_TTL_S:
        return _CACHE

    log.info("heavys_anchors: fetching from CG workspace=%s", WORKSPACE)
    t0 = time.time()
    try:
        cg_responses = await _fetch_all_anchors()
    except Exception as e:
        log.warning("heavys_anchors: full fetch failed: %s", e)
        cg_responses = {}
    elapsed_ms = int((time.time() - t0) * 1000)

    # Count how many queries returned non-empty content — if zero, the cache
    # is essentially unusable; let downstream decide whether to fall back.
    non_empty = sum(1 for v in cg_responses.values() if v)

    anchors: dict[str, Any] = {
        "_source": "CG workspace=Heavys (T-86)",
        "_fetched_at": now,
        "_cg_queries_total": len(_ANCHOR_QUERIES),
        "_cg_queries_returned_content": non_empty,
        "_fetch_ms": elapsed_ms,
        **cg_responses,
        **_INTERNAL_POLICY_ANCHORS,
    }
    _CACHE = anchors
    _CACHE_TS = now
    log.info("heavys_anchors: fetched %d/%d non-empty fields in %dms",
             non_empty, len(_ANCHOR_QUERIES), elapsed_ms)
    return anchors


def render_anchor_block(anchors: dict, max_per_field: int = 800) -> str:
    """Format the anchor pack as a markdown block for prompt injection.
    Used by the chain_executor's build_context_blocks — mirrors the T-81
    Libra rendering pattern but with Heavys-specific section ordering."""
    if not anchors:
        return ""
    lines = ["## anchors (Heavys product reference frame — T-86)"]
    # Order matters — most load-bearing first (so even truncation preserves them)
    field_order = [
        "products_summary",
        "competitor_differentiators",
        "warranty_and_returns",
        "bundle_components",
        "included_features",
        "shipping_terms",
        "payment_options",
        "social_proof",
        "current_promotions",
    ]
    for field in field_order:
        val = anchors.get(field)
        if not val or not isinstance(val, str):
            continue
        snippet = val[:max_per_field]
        lines.append(f"### {field}\n{snippet}")
    # Internal policy anchors — surface as a compact block
    if "max_authorized_discount_pct_internal" in anchors:
        lines.append(
            f"### internal_policy\n"
            f"- max_discount_pct: {anchors['max_authorized_discount_pct_internal']}%\n"
            f"- preferred_strategies (per inflection): "
            f"{json.dumps(anchors.get('preferred_strategies') or {}, indent=2)}\n"
            f"- must_not_say (Heavys-specific): "
            f"{json.dumps(anchors.get('must_not_say') or [], indent=2)}"
        )
    return "\n\n".join(lines)


# ── CLI for quick validation ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Dotenv load (parity with main.py)
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    async def _main():
        a = await fetch_heavys_anchors()
        print(render_anchor_block(a))
        print("\n─── Fetch summary ───")
        print(f"  total queries: {a.get('_cg_queries_total')}")
        print(f"  non-empty:     {a.get('_cg_queries_returned_content')}")
        print(f"  fetch time:    {a.get('_fetch_ms')}ms")
    asyncio.run(_main())
