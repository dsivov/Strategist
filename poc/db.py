"""Read-only DB access for the POC.

Connects via the existing SSH tunnel (mysql-prod-tunnel) using credentials
from ~/.my.cnf. Pure-SQL helpers — no Luna imports, no transitive deps.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import pymysql

log = logging.getLogger(__name__)


def _read_cnf(path: str = os.environ.get("POC_MY_CNF", os.path.expanduser("~/.my.cnf"))) -> dict[str, str]:
    raw = open(path).read()
    out = {}
    for k in ("host", "port", "user", "password", "database"):
        m = re.search(rf"^\s*{k}\s*=\s*(.*?)\s*$", raw, re.MULTILINE)
        if m:
            v = m.group(1).strip()
            if v[:1] in '"\'' and v[-1:] == v[:1]:
                v = v[1:-1]
            out[k] = v
    return out


def open_conn():
    cfg = _read_cnf()
    # READ-ONLY enforced at handshake via init_command — runs before any
    # user query can execute, even on connection bring-up.
    conn = pymysql.connect(
        host=cfg["host"], port=int(cfg["port"]),
        user=cfg["user"], password=cfg["password"],
        database=cfg["database"],
        cursorclass=pymysql.cursors.DictCursor, autocommit=False,
        init_command="SET SESSION TRANSACTION READ ONLY",
    )
    with conn.cursor() as cur:
        cur.execute("SET SESSION sql_safe_updates=1")
        cur.execute("SET SESSION max_execution_time=60000")
    return conn


def fetch_opp_meta(conn, opp_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT o.id, o.company, o.type AS opp_type, o.status,
                   o.created_at, o.status_update_timestamp,
                   o.total_inbounds, o.total_outbounds, o.total_reminders,
                   p.primary_motivator, p.objection_pattern, p.decision_logic,
                   p.trust_level, p.regulatory_focus, p.budget_sensitivity,
                   p.purchase_urgency, p.primary_resistance,
                   p.communication_style, p.emotional_volatility,
                   p.tone, p.gender
            FROM luna.opportunity o
            LEFT JOIN luna.research_profile_flash p ON p.opportunity_id = o.id
            WHERE o.id = %s
        """, (opp_id,))
        return cur.fetchone()


def fetch_messages(conn, opp_id: str) -> list[dict]:
    """Returns full conversation transcript with timestamps + direction.
    Prefers en_US text, falls back to he, then any."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT me.message_id, me.type AS direction, me.timestamp,
                   me.is_reminder, me.is_followup, me.automatic_response,
                   COALESCE(
                     (SELECT m.text FROM luna.message m
                      WHERE m.opportunity_id=me.opportunity_id AND m.message_id=me.message_id
                        AND m.lang='en_US' LIMIT 1),
                     (SELECT m.text FROM luna.message m
                      WHERE m.opportunity_id=me.opportunity_id AND m.message_id=me.message_id
                        AND m.lang='he' LIMIT 1),
                     (SELECT m.text FROM luna.message m
                      WHERE m.opportunity_id=me.opportunity_id AND m.message_id=me.message_id
                      LIMIT 1)
                   ) AS text
            FROM luna.message_event me
            WHERE me.opportunity_id = %s
              AND (me.is_deleted IS NULL OR me.is_deleted = 0)
            ORDER BY me.timestamp ASC
        """, (opp_id,))
        rows = cur.fetchall()
    return [r for r in rows if (r.get("text") or "").strip()]


def fetch_turn_states(conn, opp_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sequence_number, persuasion_score, commitment_level, sentiment,
                   objection_category, p_conv, message_id
            FROM luna.research_turn_state_flash
            WHERE opportunity_id = %s
            ORDER BY sequence_number ASC
        """, (opp_id,))
        return list(cur.fetchall())


def fetch_persuasive_scores(conn, opp_id: str) -> dict[str, dict]:
    """Returns {message_id: {score, reason}} from prod persuasive_score table."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_message_id, score, reason
            FROM luna.persuasive_score
            WHERE opportunity_id = %s
        """, (opp_id,))
        return {r["user_message_id"]: r for r in cur.fetchall()}


def fetch_business_rules(conn, company: str) -> str:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT info_data FROM luna.company_business_info
            WHERE company = %s AND info_type = 'business_rules' AND is_draft = 0
            ORDER BY ver_num DESC LIMIT 1
        """, (company,))
        row = cur.fetchone()
    return row["info_data"] if row else ""


# T-81 anchor enrichment (2026-05-02) — fixes the impoverished customer-state
# model. Production Luna agent CAN reach `libra_stage_data` for any active
# Libra renewal opp (ly_price, current quoted price, max_discount, coverage
# detail, claim history, market segment) — but currently doesn't. This
# fetches those anchors so both the supervisor AND the customer simulator
# can negotiate from a real economic reference frame instead of inventing
# adversarial pushback (simulator) or staircase-pricing (supervised agent).
#
# Closes a structural gap empirically confirmed 2026-05-02: a3f517d7's seed
# dialog quotes 1,272+1,887=3,159 NIS justified by a "48% market hike",
# but the actual prod data shows YoY change is +0.24% (effectively flat)
# and market median for that car cohort is ~2,200 NIS. The conversation
# was operating on a fabricated premium with no grounding in real data.

# Synthetic anchors for our 2024-2025 test scenarios (libra_stage_data
# only retains current cycle, so historical opps need synthesis from
# seed-dialog signals + cohort medians). Tagged synthetic=True so the
# prompts can be honest about provenance.
SYNTHETIC_ANCHORS_BY_OPP: dict[str, dict] = {
    "a3f517d7-b16b-4dc6-9671-83ac8aba56a7": {
        # Older Nissan, "competed-away" cluster. Seed dialog quotes
        # 1,272+1,887=3,159 with claimed 48% increase justification.
        # Real YoY change in prod data: +0.24% — quote is inflated.
        # Customer in dialog mentions competitor at ~2,400 NIS.
        "last_year_price_nis": 2150,           # back-calculated assuming
                                                # roughly flat YoY
        "current_quoted_price_nis": 3159,
        "claimed_increase_pct": 48.0,
        "actual_market_yoy_change_pct": 0.24,
        "market_avg_for_segment_nis": 2200,    # 2014-2017 Nissan cohort
        "max_discount_pct_internal": 15.0,     # "approval to match" implies
                                                # they have ~15% discretion
        "coverage_summary": "Third-party + mandatory; deductible 990 NIS arrangement / 1,500 NIS private; headlights/mirrors included",
        "loyalty_years": 3,                    # plausible from "renewal" context
        "vehicle": "older Nissan",
        "synthetic": True,
        "provenance": "seed_dialog + cohort_median_2014-2017 + observed_yoy_from_prod",
    },
    "5db6fda9-a5a1-403a-8c01-d1b62bbf14ba": {
        # Same cluster as a3f517d7 (competed-away)
        "last_year_price_nis": 2050,
        "current_quoted_price_nis": 2987,
        "claimed_increase_pct": 46.0,
        "actual_market_yoy_change_pct": 0.24,
        "market_avg_for_segment_nis": 2100,
        "max_discount_pct_internal": 15.0,
        "coverage_summary": "Third-party + mandatory",
        "loyalty_years": 4,
        "vehicle": "older private car",
        "synthetic": True,
        "provenance": "seed_dialog + cohort_median + observed_yoy_from_prod",
    },
    "bc2d3bd1-3f35-47b5-8bca-5cf9a64f5c12": {
        # Variance-run target opp (median difficulty)
        "last_year_price_nis": 1950,
        "current_quoted_price_nis": 2700,
        "claimed_increase_pct": 38.0,
        "actual_market_yoy_change_pct": 0.24,
        "market_avg_for_segment_nis": 2100,
        "max_discount_pct_internal": 15.0,
        "coverage_summary": "Third-party + mandatory",
        "loyalty_years": 3,
        "vehicle": "older private car",
        "synthetic": True,
        "provenance": "synthesized cohort defaults",
    },
    # 2026-05-14 — Demo-relevant opps backfilled for testing the
    # profile-conditioned opening anchor. ly_price + loyalty + claims are
    # inferred from the seed dialog (customer mentions "several good years"
    # and "no incidents") and reasonable Kia/Citroën cohort baselines.
    "10a45705-0ba1-4763-b2ca-c35a4db94114": {
        # Roy / Kia comprehensive renewal. Seed quotes 8,053 NIS; customer
        # says "I didn't expect a customer of several good years to receive
        # such an offer". Anchor-exposure backfire case.
        "last_year_price_nis": 5000,
        "current_quoted_price_nis": 8053,
        "claimed_increase_pct": 61.0,
        "actual_market_yoy_change_pct": 0.24,
        "market_avg_for_segment_nis": 5500,
        "max_discount_pct_internal": 35.0,
        "coverage_summary": "Comprehensive incl. headlights/mirrors, towing, windshield",
        "loyalty_years": 4,
        "claims_count": 0,
        "vehicle": "Kia (comprehensive)",
        "synthetic": True,
        "provenance": "seed_dialog_extraction + cohort_baseline 2026-05-14",
    },
    "087b0160-38d4-4854-8475-bf78566fecf2": {
        # Citroën third-party renewal. Customer rejects on price ("Still too
        # expensive for that car"). Older vehicle.
        "last_year_price_nis": 1800,
        "current_quoted_price_nis": 2500,
        "claimed_increase_pct": 39.0,
        "actual_market_yoy_change_pct": 0.24,
        "market_avg_for_segment_nis": 2000,
        "max_discount_pct_internal": 20.0,
        "coverage_summary": "Third-party + mandatory",
        "loyalty_years": 3,
        "claims_count": 0,
        "vehicle": "Citroën (older)",
        "synthetic": True,
        "provenance": "seed_dialog_extraction + cohort_baseline 2026-05-14",
    },
    "59f9696f-4c5a-44b3-b163-f97c42a82b06": {
        # libra_c0 — the primary high-SNR measurement opp (3.06x intervention
        # sensitivity per the Phase 5 power analysis). Was running on the
        # hardcoded global default (libra_stage_data empty in this replica),
        # which means prior libra_c0 deltas were measured against a generic
        # anchor, not this opp's real economic frame. Backfilled 2026-05-17.
        #
        # Seed dialog (Suzuki #5869374, Ofir): agent quotes comprehensive
        # 2,447 NIS; ly comprehensive 2,317 + ly mandatory 1,580; agent's
        # stated floor is third-party 1,200 + mandatory 1,628 = 2,828.
        # NOT an anchor-exposure case — the increase is a modest +5.6%.
        # The failure mode here is the aggressive price-shopper ("I'll do a
        # survey", "lower the price further"); the anchor's value is giving
        # the supervisor the real frame to justify the small increase and
        # hold the floor rather than staircase. loyalty/claims not stated in
        # seed → conservative renewal defaults (2y, 0 claims).
        "last_year_price_nis": 2317,
        "current_quoted_price_nis": 2447,
        "claimed_increase_pct": 5.6,
        "actual_market_yoy_change_pct": 0.24,
        "market_avg_for_segment_nis": 2400,    # small-car Suzuki comprehensive
        "max_discount_pct_internal": 15.0,
        "coverage_summary": "Comprehensive; third-party 1,200 + mandatory 1,628 NIS option available",
        "loyalty_years": 2,                    # conservative — renewal, not stated
        "claims_count": 0,                     # none mentioned in seed
        "vehicle": "Suzuki (comprehensive)",
        "synthetic": True,
        "provenance": "seed_dialog_extraction + cohort_baseline 2026-05-17",
    },
}


def fetch_libra_anchors(conn, opp_id: str, opp_meta: dict | None = None) -> dict:
    """Return anchor data for a Libra Insurance Renewal opp.

    Sources, in priority order:
      1. Synthetic per-opp dict (for our historical 2024-2025 test scenarios
         whose libra_stage_data row was purged)
      2. Live `libra_stage_data` join via opp.external_id (for current opps)
      3. Cohort fallback: market_avg only (computed from libra_stage_data
         filtered by manufacture_year proxy — empty if cohort not derivable)

    Returned dict always carries `synthetic: bool` so prompts can be honest
    about provenance. Returns empty dict if nothing reachable.
    """
    # 1. Synthetic per-opp
    if opp_id in SYNTHETIC_ANCHORS_BY_OPP:
        out = dict(SYNTHETIC_ANCHORS_BY_OPP[opp_id])
        # Compute profile-appropriate opening if not already present
        if "profile_appropriate_opening_nis" not in out:
            opening, reason = _profile_appropriate_opening(
                out.get("last_year_price_nis", 0),
                out.get("current_quoted_price_nis", 0),
                out.get("loyalty_years"),
                int(out.get("claims_count", 0)),
            )
            out["profile_appropriate_opening_nis"] = opening
            out["profile_appropriate_opening_reason"] = reason
            out.setdefault("claims_count", 0)
        return out

    # 2. Live stage_data join (only useful for current-cycle opps)
    if opp_meta is None or not opp_meta.get("external_id"):
        # Need external_id; fetch it
        with conn.cursor() as cur:
            cur.execute(
                "SELECT external_id, company FROM luna.opportunity WHERE id = %s",
                (opp_id,))
            r = cur.fetchone()
            if not r:
                return {}
            opp_meta = r

    if (opp_meta.get("company") or "").lower() != "libra":
        return {}  # only Libra has stage_data
    external_id = opp_meta.get("external_id")
    if not external_id:
        return {}

    with conn.cursor() as cur:
        cur.execute("""
            SELECT ly_price, price, max_discount, insurance_type,
                   loyalty_duration, manufacturer_and_model, manufacture_year,
                   first_claim, second_claim,
                   towing_service, windshield_service, headlights_service,
                   no_deductible
            FROM luna.libra_stage_data
            WHERE external_id = %s
            ORDER BY id DESC
            LIMIT 1
        """, (external_id,))
        sd = cur.fetchone()

    if not sd or sd.get("price") is None or sd.get("ly_price") is None:
        # 3. Cohort fallback — compute market_avg only
        return _compute_cohort_market_avg(conn, opp_meta)

    # Compute market_avg for this opp's manufacture_year cohort
    market_avg = _compute_market_avg_by_year(conn, sd.get("manufacture_year"))

    try:
        ly = float(sd["ly_price"])
        cur_p = float(sd["price"])
        yoy_pct = (100.0 * (cur_p - ly) / ly) if ly > 0 else 0.0
    except (ValueError, TypeError):
        return {}

    loyalty_years = (
        int(sd["loyalty_duration"]) if sd.get("loyalty_duration") and
        str(sd["loyalty_duration"]).isdigit() else None)
    claims_count = (
        (1 if sd.get("first_claim") else 0) +
        (1 if sd.get("second_claim") else 0)
    )
    profile_open, profile_reason = _profile_appropriate_opening(
        ly, cur_p, loyalty_years, claims_count)

    return {
        "last_year_price_nis": round(ly),
        "current_quoted_price_nis": round(cur_p),
        "claimed_increase_pct": round(yoy_pct, 1),
        "actual_market_yoy_change_pct": 0.24,  # computed once from full table
        "market_avg_for_segment_nis": market_avg,
        "max_discount_pct_internal": (
            float(sd["max_discount"]) if sd.get("max_discount") else 15.0),
        "coverage_summary": _summarize_coverage(sd),
        "loyalty_years": loyalty_years,
        "claims_count": claims_count,
        # 2026-05-14 — profile-conditioned opening anchor. Loyal customers
        # with no claims should NEVER see a "list price" anchor; quoting
        # current_quoted_price (8053) to a 4-year loyal no-claims customer
        # triggers trust collapse when the agent later concedes to 5000.
        # The behavioral-economics literature (anchor-exposure backfire,
        # Galinsky & Mussweiler 2001) shows large anchor reveals damage
        # trust below baseline. Open near last-year + market inflation.
        "profile_appropriate_opening_nis": profile_open,
        "profile_appropriate_opening_reason": profile_reason,
        "vehicle": (sd.get("manufacturer_and_model") or "").strip()[:60],
        "synthetic": False,
        "provenance": "libra_stage_data live join",
    }


def _profile_appropriate_opening(ly: float, list_price: float,
                                   loyalty_years: int | None,
                                   claims_count: int) -> tuple[int | None, str]:
    """Compute the highest acceptable opening price for a customer's profile.

    Returns (opening_nis, reason). When the customer profile justifies a
    near-floor opening (loyal + no claims), the opening should approximate
    last_year + actual market inflation (~10%) rather than the list price.

    Tiers:
      1. Loyal (≥2y) + no claims  → ly * 1.10  (best customer profile)
      2. Loyal (≥2y) + 1 claim    → ly * 1.18  (still good, slight risk uplift)
      3. Some loyalty (1y) + no claims → ly * 1.15
      4. New / claims-heavy        → list_price (no constraint)
    """
    if not ly or not list_price:
        return (None, "missing-data")
    loyalty = loyalty_years or 0
    if loyalty >= 2 and claims_count == 0:
        return (round(ly * 1.10),
                f"loyal-{loyalty}y-noclaims: ly*1.10 = {round(ly*1.10)} NIS")
    if loyalty >= 2 and claims_count == 1:
        return (round(ly * 1.18),
                f"loyal-{loyalty}y-1claim: ly*1.18 = {round(ly*1.18)} NIS")
    if loyalty >= 1 and claims_count == 0:
        return (round(ly * 1.15),
                f"loyalty-{loyalty}y-noclaims: ly*1.15 = {round(ly*1.15)} NIS")
    return (round(list_price), f"list-price-acceptable (loyalty={loyalty}y, claims={claims_count})")


def _compute_market_avg_by_year(conn, manufacture_year: Any) -> int | None:
    """Median price (NIS, rounded) for cohort within ±2 years of given year."""
    try:
        year = int(manufacture_year) if manufacture_year else None
    except (ValueError, TypeError):
        return None
    if not year or year < 2000 or year > 2030:
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT AVG(CAST(price AS DECIMAL(10,2))) AS avg_p, COUNT(*) AS n
            FROM luna.libra_stage_data
            WHERE manufacture_year BETWEEN %s AND %s
              AND price IS NOT NULL AND price != ''
        """, (str(year - 2), str(year + 2)))
        r = cur.fetchone()
    if not r or not r.get("avg_p") or (r.get("n") or 0) < 30:
        return None
    return round(float(r["avg_p"]))


def _compute_market_ly_by_year(conn, manufacture_year: Any) -> int | None:
    """Median ly_price (NIS, rounded) for cohort within ±2 years. Parallel
    to _compute_market_avg_by_year but pulls last-year prices."""
    try:
        year = int(manufacture_year) if manufacture_year else None
    except (ValueError, TypeError):
        return None
    if not year or year < 2000 or year > 2030:
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT AVG(CAST(ly_price AS DECIMAL(10,2))) AS avg_ly, COUNT(*) AS n
            FROM luna.libra_stage_data
            WHERE manufacture_year BETWEEN %s AND %s
              AND ly_price IS NOT NULL AND ly_price != ''
        """, (str(year - 2), str(year + 2)))
        r = cur.fetchone()
    if not r or not r.get("avg_ly") or (r.get("n") or 0) < 30:
        return None
    return round(float(r["avg_ly"]))


# Hardcoded global defaults for when libra_stage_data is empty (dev/POC
# scenarios). These are Israeli insurance market plausible mid-cohort values
# circa 2026; only used when ALL DB paths fail. Production would have
# populated stage_data so this branch is never reached.
_GLOBAL_DEFAULT_LY_NIS = 3500
_GLOBAL_DEFAULT_CURRENT_NIS = 4500


def _compute_cohort_market_avg(conn, opp_meta: dict) -> dict:
    """Fallback when per-opp stage_data is missing.

    Returns cohort-median current price + cohort-median last-year price
    + a profile-conditioned opening anchor computed under "default loyal
    renewal" assumptions (loyalty=2y, claims=0). This is the safe-direction
    default: every Libra renewal opp is by definition an existing customer,
    so loyalty≥1 is universal; defaulting to 0 claims biases the opening
    LOW (better trust outcome) rather than HIGH (anchor-exposure backfire).

    Resolution order:
      1. Per-cohort by manufacture_year (≥30 rows in libra_stage_data)
      2. Global aggregate over all libra_stage_data
      3. Hardcoded global default (when libra_stage_data is empty — dev DB)
    """
    year = opp_meta.get("manufacture_year") if opp_meta else None
    cohort_avg = _compute_market_avg_by_year(conn, year)
    cohort_ly = _compute_market_ly_by_year(conn, year)
    provenance_bits = []

    if cohort_avg is not None and cohort_ly is not None:
        provenance_bits.append(f"cohort_by_year_{year}")
    else:
        # Fall to global aggregate when cohort too sparse
        with conn.cursor() as cur:
            cur.execute("""
                SELECT AVG(CAST(price AS DECIMAL(10,2))) AS avg_p,
                       AVG(CAST(ly_price AS DECIMAL(10,2))) AS avg_ly,
                       COUNT(*) AS n
                FROM luna.libra_stage_data
                WHERE price IS NOT NULL AND price != ''
                  AND ly_price IS NOT NULL AND ly_price != ''
            """)
            r = cur.fetchone()
        if r and r.get("avg_p"):
            if cohort_avg is None:
                cohort_avg = round(float(r["avg_p"]))
                provenance_bits.append(f"global_avg_n={r.get('n')}")
            if cohort_ly is None and r.get("avg_ly"):
                cohort_ly = round(float(r["avg_ly"]))
                provenance_bits.append(f"global_ly_n={r.get('n')}")

    # Final fallback — libra_stage_data is completely empty (dev DB).
    # Use hardcoded global defaults so EVERY Libra opp gets some
    # profile-appropriate guidance, even without per-opp or cohort data.
    if cohort_avg is None:
        cohort_avg = _GLOBAL_DEFAULT_CURRENT_NIS
        provenance_bits.append("hardcoded_global_default")
    if cohort_ly is None:
        cohort_ly = _GLOBAL_DEFAULT_LY_NIS
        if "hardcoded_global_default" not in provenance_bits:
            provenance_bits.append("hardcoded_global_default")

    # Default profile — most renewal opps are loyal+no-claims. This is the
    # safe default: assuming a strong profile biases the opening DOWN,
    # which protects against anchor-exposure backfire. Per-opp data (when
    # available via live stage_data join) overrides this.
    DEFAULT_LOYALTY_YEARS = 2
    DEFAULT_CLAIMS = 0
    opening, reason = (None, "missing-data")
    if cohort_ly:
        opening, reason = _profile_appropriate_opening(
            cohort_ly, cohort_avg, DEFAULT_LOYALTY_YEARS, DEFAULT_CLAIMS
        )

    return {
        "market_avg_for_segment_nis": cohort_avg,
        # Note: cohort median, NOT this opp's actual last-year price. Used
        # only for opening-anchor computation when per-opp data is absent.
        "last_year_price_nis": cohort_ly,
        "actual_market_yoy_change_pct": 0.24,
        "loyalty_years": DEFAULT_LOYALTY_YEARS,
        "claims_count": DEFAULT_CLAIMS,
        "profile_appropriate_opening_nis": opening,
        "profile_appropriate_opening_reason": (
            f"cohort-fallback ({reason}); defaults to loyal-2y/claims-0 in "
            f"absence of per-opp data"
        ) if opening else "cohort-fallback: insufficient data",
        "synthetic": False,
        "provenance": "; ".join(provenance_bits) or "cohort_fallback",
        "partial": True,
    }


def _summarize_coverage(sd: dict) -> str:
    parts = []
    insurance_type = (sd.get("insurance_type") or "").strip()
    if insurance_type:
        parts.append(insurance_type)
    if sd.get("no_deductible") and str(sd["no_deductible"]).strip().lower() in ("yes","true","1"):
        parts.append("no deductible")
    if sd.get("towing_service") and str(sd["towing_service"]).strip().lower() in ("yes","true","1"):
        parts.append("towing")
    if sd.get("headlights_service") and str(sd["headlights_service"]).strip().lower() in ("yes","true","1"):
        parts.append("headlights")
    if sd.get("windshield_service") and str(sd["windshield_service"]).strip().lower() in ("yes","true","1"):
        parts.append("windshield")
    return ", ".join(parts) if parts else "(coverage detail unavailable)"


def find_failure_mode_turn_index(messages: list[dict]) -> int | None:
    """Index (in messages list) of the customer's last meaningful inbound —
    i.e., the supervisor's intervention point."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("direction") == "inbound":
            return i
    return None


def find_supervisor_intervention_index(messages: list[dict],
                                         turn_states: list[dict],
                                         persuasive_scores: dict) -> int | None:
    """Pick the customer's PEAK-ENGAGEMENT turn as the supervisor's entry point.

    Why this is right (rather than 'last engaged'): if the customer was already
    in a 2-3 turn decline by the time the conversation actually ended, seeding
    at "last engaged" means supervisor inherits a customer who's already given
    up. We want the supervisor to take over at the moment the customer was MOST
    receptive — that's where strategy differences show their effect.

    Algorithm:
      1. For each inbound message, compute combined engagement = (commit_level * 2)
         + (persuasion_score * 5)
      2. Find the FIRST turn where engagement was at peak (we prefer earliest peak —
         that gives more live-phase room afterward)
      3. Floor: at least 3 messages of seed (need some history); ceil: not the
         very last inbound (need live-phase room)
      4. Fallback: last inbound if no peak detected with the floor"""
    ts_by_seq = {t.get("sequence_number"): t for t in turn_states}
    last_inbound_idx = find_failure_mode_turn_index(messages)
    if last_inbound_idx is None:
        return None

    # Build engagement scores per inbound message
    inbound_scores: list[tuple[int, float]] = []
    for i, m in enumerate(messages):
        if m.get("direction") != "inbound":
            continue
        ts = ts_by_seq.get(i)
        commit = ts.get("commitment_level") if ts else 0
        pscore_row = persuasive_scores.get(m.get("message_id")) if persuasive_scores else None
        pscore = pscore_row.get("score") if pscore_row else 0
        engagement = (commit or 0) * 2 + (pscore or 0) * 5
        inbound_scores.append((i, engagement))

    if not inbound_scores:
        return last_inbound_idx

    # Find peak engagement among inbound turns that ALSO leave at least 3 msgs of room
    # for the live phase (i.e., the picked turn must be at least 3 messages from the end).
    LIVE_PHASE_MIN_TAIL = 3
    cutoff = len(messages) - LIVE_PHASE_MIN_TAIL

    eligible = [(i, s) for i, s in inbound_scores if i <= cutoff and i >= 2]
    if not eligible:
        # Conversation too short OR no early engagement — use the FIRST engaged turn
        for i, s in inbound_scores:
            if s >= 4:  # combined score = at least commit=2 (engagement)
                return i
        # No engagement detected at all — fall back to last inbound
        return last_inbound_idx

    # Pick the EARLIEST inbound at peak engagement — gives the supervisor MORE
    # live-phase room to demonstrate value. Earlier seed-end means more turns
    # for the supervisor to make strategic moves before the conversation ends.
    # Trade-off: customer's full objection/persona may still be revealing,
    # but the supervisor has more room to act.
    peak = max(s for _, s in eligible)
    for i, s in eligible:
        if s == peak:
            return i
    return last_inbound_idx
