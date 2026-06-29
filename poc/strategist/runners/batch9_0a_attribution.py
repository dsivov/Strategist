"""Batch 9.0a — empirical loss-attribution analysis on Insurance ClosedLost opps.

Goal: derive the loss-taxonomy from data, not from a priori category guesses.

Pipeline:
  1. Sample ~500 Insurance ClosedLost opps with rich state (research_*_flash populated)
  2. Compute per-opp features: trajectory, conversation shape, last-agent-message,
     profile, outcome, lightweight rule-compliance signals
  3. Cluster (KMeans + agglomerative) to find natural groupings
  4. Sample 5 opp-ids per cluster for manual transcript review
  5. Output: cluster summary stats CSV + sample-transcript JSON

Read-only on prod. No CG calls. No LLM calls. Pure SQL + sklearn.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import pymysql
import mattresscommerces as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = "/tmp/batch9_0a_v3_1"
SAMPLE_SIZE = 500
N_CLUSTERS_KMEANS = 10


# ── Prod conn ──────────────────────────────────────────────────────────────────
def open_prod_conn():
    raw = open("/home/dev/.my.cnf").read()

    def g(k):
        m = re.search(rf"^\s*{k}\s*=\s*(.*?)\s*$", raw, re.MULTILINE)
        v = m.group(1).strip()
        return v[1:-1] if v[:1] in '"\'' and v[-1:] == v[:1] else v

    conn = pymysql.connect(
        host=g("host"), port=int(g("port")), user=g("user"), password=g("password"),
        database=g("database"), cursorclass=pymysql.cursors.DictCursor, autocommit=False,
    )
    with conn.cursor() as cur:
        cur.execute("SET SESSION TRANSACTION READ ONLY")
        cur.execute("SET SESSION sql_safe_updates=1")
        cur.execute("SET SESSION max_execution_time=60000")
    return conn


# ── Sampling ───────────────────────────────────────────────────────────────────
def sample_opps(conn) -> list[dict]:
    sql = """
    SELECT o.id, o.company, o.type AS opp_type, o.status,
           o.created_at, o.status_update_timestamp, o.expiration_date,
           o.total_inbounds, o.total_outbounds, o.total_reminders, o.client_engaged,
           p.primary_motivator, p.objection_pattern, p.decision_logic,
           p.trust_level, p.regulatory_focus, p.budget_sensitivity,
           p.purchase_urgency, p.primary_resistance,
           (SELECT COUNT(*) FROM research_message_strategy_flash
              WHERE opportunity_id = o.id) AS strategy_msg_count,
           (SELECT COUNT(*) FROM research_turn_state_flash
              WHERE opportunity_id = o.id) AS turn_state_count
    FROM opportunity o
    JOIN research_profile_flash p ON p.opportunity_id = o.id
    WHERE o.company = 'Insurance'
      AND o.type = 'Insurance Renewal'
      AND o.status = 'ClosedLost'
      AND p.primary_motivator IS NOT NULL
    HAVING strategy_msg_count >= 3 AND turn_state_count >= 3
    ORDER BY RAND(7919)
    LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (SAMPLE_SIZE,))
        return list(cur.fetchall())


def fetch_opp_data(conn, opp_id: str) -> dict:
    """Pull all per-opp data we need to compute features."""
    out = {"opp_id": opp_id}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sequence_number, persuasion_score, commitment_level, sentiment,
                   objection_category, p_conv,
                   authority_activated, social_proof_activated, reciprocity_activated,
                   commitment_activated, scarcity_activated, liking_activated,
                   resistance_hardening, message_id
            FROM research_turn_state_flash
            WHERE opportunity_id = %s
            ORDER BY sequence_number ASC
        """, (opp_id,))
        out["turns"] = list(cur.fetchall())

        cur.execute("""
            SELECT sequence_number, message_id, primary_strategy, secondary_strategy, tone
            FROM research_message_strategy_flash
            WHERE opportunity_id = %s
            ORDER BY sequence_number ASC
        """, (opp_id,))
        out["strategies"] = list(cur.fetchall())

        cur.execute("""
            SELECT me.message_id, me.type AS direction, me.timestamp,
                   me.is_reminder, me.is_followup, me.automatic_response,
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
            WHERE me.opportunity_id = %s
              AND (me.is_deleted IS NULL OR me.is_deleted = 0)
            ORDER BY me.timestamp ASC
        """, (opp_id,))
        out["messages"] = [r for r in cur.fetchall() if r["text"]]
    return out


# ── Feature extraction ─────────────────────────────────────────────────────────
B29_HINTS = [
    r"i['']?ll check", r"let me check", r"another insurance", r"another company",
    r"another offer", r"competitor", r"competing offer",
    # Hebrew approximations (for inbound text in Hebrew)
    "אבדוק", "אני אבדוק", "אחזור", "תן לי לחשוב", "צריך לחשוב",
]

PRICE_HINTS = ["price", "discount", "cheaper", "cost", "usd", "pay", "payment",
               "מחיר", "הנחה", "שקל", "תשלום"]

CLOSING_ASK_HINTS = [
    "shall we proceed", "how many installments", "last 4 digits", "credit card",
    "let's renew", "let's close", "renew now", "ready to renew",
    "תרצה", "להמשיך", "תשלומים", "אישורי",
]

# NEW (2026-04-29 — re-run after user feedback):
# (1) Competitor announcement: customer eventually said they chose another company.
# Distinct from B29 ("I'll check") — this is post-decision: "I went with X."
COMPETITOR_NAMES = [
    # Insurance competitors in Israel (Hebrew + English variants)
    "harel", "הראל",
    "phoenix", "hafenix", "פניקס", "הפניקס",
    "aig", "איי איי ג'י",
    "menora", "מנורה",
    "migdal", "מגדל",
    "clal", "כלל",
    "yashir", "ישיר",
    "shomera", "שומרה",
    "ayalon", "איילון",
    "shirbit", "שירביט",
    "ifia", "איפיה",
]

COMPETITOR_DECISION_VERBS = [
    # English
    r"\bgoing with\b", r"\brenewed with\b", r"\bwent with\b",
    r"\bchose\b", r"\bgoing to renew\b", r"\bsigned with\b", r"\bfinalized\b",
    r"\bswitched to\b",
    # Hebrew
    "סגרתי", "עברתי", "סוגר עם", "סגרנו", "בחרתי", "הולך עם", "אני עם",
    "הלכתי עם", "התקדם עם", "עשיתי עם",
]

# (2) B41 trigger: customer indicates they're not the decision-maker.
B41_HINTS = [
    r"\btalk to him\b", r"\btalk to her\b", r"\bcall my\b",
    r"\bsend to him\b", r"\bsend to her\b", r"\bnot me\b", r"\bnot the\b.{0,15}\bowner\b",
    r"\bcall my husband\b", r"\bcall my wife\b",
    # Hebrew — "talk to my husband / wife / son / Dad"
    "תדבר עם", "תדברו עם", "תתקשר ל", "לא איתי", "לא אני", "האחראי",
    "החתום על הפוליסה", "בעל הפוליסה", "בעלי", "אשתי",
    "תפנה ל", "פנו ל", "תעבור ל",
]

QUESTION_RE = re.compile(r"[?؟]")

# ── conversation_won_signal feature (v3.1, 2026-04-29) ────────────────────────
# v3.0 fired on only 52.5% of actual Wons (target ≥85%). Failure modes from the
# 200-Won validation pass:
#   - "I renewed for you in 10 interest-free payments" / "Your policy for a full
#     year from..." / "first payment will be deducted" — agent close patterns
#     not in v3 regex.
#   - "I want you to renew for me please" / "אני רוצה לחדש" — explicit-close in
#     freer word order than v3 expected.
#   - 6-digit CC reply ("390156") — v3 CC regex required exactly 4 isolated digits.
#   - "in one payment" / "single payment" — word-number installment count.
#   - **Phone-handoff-for-CC** (4-5 of 10 missed-Won samples) — customer can't
#     give CC in chat (card not with them, spouse pays, dad pays), redirects to
#     phone. New component.

# (a) Agent emitted a final-confirmation / closure-acknowledgement pattern
AGENT_FINAL_CONFIRMATION_PATTERNS = [
    # English — confirmation of renewal / policy continuation
    r"we['']?ll continue with the renewal",
    r"continuing with (the )?renewal",
    r"your policy continues",
    r"policy will continue",
    r"renewed for (a |the )?(full )?year",
    r"i ren?ewed for you",                          # NEW (v3.1)
    r"renewed for you in \d+",                      # NEW (v3.1)
    r"i['']?ve renewed",                            # NEW
    r"policy for a (full )?year from",              # NEW (v3.1) — "Your policy for a full year from 01/06/2025"
    r"the (entire )?amount will be held",           # NEW (v3.1) — banking authorization
    r"first payment (will be )?deducted",           # NEW (v3.1)
    r"perfect[,!.]?\s*(we|let).{0,40}(renewal|continue|payment|payments|installments)",
    r"splitting (it )?into \d+ payments",
    r"\d+ payments? (without|with no) interest",
    r"\d+ interest[- ]free payments?",              # NEW (v3.1) — "10 interest-free payments"
    r"sending you (the )?(confirmation|policy|details)",
    r"all set[,!.]",
    r"insurance is renewed",                        # NEW (v3.1)
    # Hebrew
    "ממשיכים עם החידוש",
    "סגרנו",
    "סגור עם",
    "החידוש בוצע",
    "הפוליסה ממשיכה",
    "ממשיכים לשנה",
    "מצוין",  # often used at close in Hebrew agent voice
    "תשלומים ללא ריבית",
    "אישור",
    # NEW (v3.1) — Hebrew agent close variants
    "חידשתי לך",
    "חידשנו לך",
    "חידשתי בשבילך",
    "סגרתי לך",
    "סגרתי עבורך",
    "פוליסה לשנה",
    "תקופת ביטוח",
    "החיוב הראשון",
    "החיוב יבוצע",
    "הפוליסה תקפה",
    "הפוליסה שלך תקפה",
    "פוליסה חדשה",
]

# (b) Customer provided CC last-4 OR explicit installment count agreement.
# v3.1 fixes: (1) accept 4-8 digit runs (not just exactly-4), (2) when agent
# recently prompted for CC, also accept digit run inside text with other content
# ("390156, so for 10 🙂"), (3) word-number installments ("one payment").
CC_LAST4_RE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")  # was \d{4} — now 4-8
CC_ANY_4PLUS_RE = re.compile(r"\d{4,8}")            # NEW — for agent-prompted context, no boundary req
CC_ONLY_DIGITS_RE = re.compile(r"^\s*[\d\s\-]{4,16}\s*$")
AGENT_CC_PROMPT_PATTERNS = [
    r"\blast\s*4\s*digits\b",
    r"\bcredit card\b",
    r"\bcard last\b",
    r"\b4 digits of\b",
    r"\bcc last\b",
    r"\bsame card as last year\b",                  # NEW (v3.1) — common Insurance phrasing
    r"\bcard for verification\b",                   # NEW
    "כרטיס אשראי",
    "ארבע ספרות אחרונות",
    "4 ספרות אחרונות",
    "4 הספרות האחרונות",
    "ארבע ספרות",
    "ארבע הספרות",
    "אישור לחיוב",
    "אותו כרטיס",                                    # NEW
    "כרטיס מהשנה",                                   # NEW
]
INSTALLMENT_COUNT_RE = re.compile(
    r"\b(\d{1,2})\s*(payments?|installments?|תשלומים)\b", re.IGNORECASE
)
INSTALLMENT_WORD_RE = re.compile(                   # NEW (v3.1)
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|single|first|second|third|fourth|fifth)\s+(payments?|installments?)\b",
    re.IGNORECASE,
)
INSTALLMENT_HEBREW_PATTERNS = [                     # NEW (v3.1)
    "תשלום אחד",
    "תשלום בודד",
    "בתשלום אחד",
]

# (c) Customer explicit verbal close
CUSTOMER_EXPLICIT_CLOSE_PATTERNS = [
    # English — relaxed for free word order (v3.1)
    r"\byes\s*[,!.]?\s*(let['']?s|please)?\s*(renew|proceed|continue|do it|go ahead)\b",
    r"\bgo ahead\b",
    r"\blet['']?s do it\b",
    r"\bplease renew\b",
    r"\brenew it\b",
    r"\bok\s*[,!.]?\s*(renew|proceed|let['']?s|sounds good)\b",
    r"\bsounds good\b",
    r"\bok let['']?s (do|go|renew|proceed)\b",
    # NEW (v3.1) — free word order
    r"\bi want (you )?to renew\b",
    r"\bi['']?d like (you )?to renew\b",
    r"\bi want to renew\b",
    r"\b(renew|close) (for me|us)\b",
    r"\brenew (for me|please|me)\b",
    r"\bi['']?ll close\b",
    r"\bgoing to close\b",
    r"\bgo for it\b",
    # Hebrew — explicit confirmation / closure (relaxed substring)
    "סגור",       # "closed / done deal"
    "מאשר",       # "I confirm" (m.)
    "מאשרת",      # "I confirm" (f.)
    "תחדש",       # "renew"
    "תחדשי",
    "תחדשו",      # NEW
    "אני מסכים",
    "אני מסכימה",
    "מסכים",
    "מסכימה",
    "כן תחדש",
    "אישור",
    "בוא נעשה",
    # NEW (v3.1) — Hebrew customer close variants in free order
    "אני רוצה לחדש",
    "אני רוצה לסגור",
    "תחדשי לי",
    "תחדש לי",
    "סגור לי",
    "סגרי לי",
    "תסגור",
    "תסגרי",
    "תסגרו",
    "תאשרו",
    "כן סגור",
    "כן תסגור",
    "כן לחדש",
    "כן בבקשה",
    "סגרו לי",
    "מעולה",
]

# (d) NEW (v3.1) — phone-handoff-after-engagement signal
# Customer requests phone callback / provides phone number for CC handling /
# redirects close to phone. Strong Won-flavor signal IFF customer has reached
# commitment≥3 (otherwise it's just routine logistics).
PHONE_HANDOFF_PATTERNS = [
    r"\bcan you call\b",
    r"\bcall (me|him|her|us)\b",
    r"\bcall my (husband|wife|dad|father|mom|mother|son|daughter|partner)\b",
    r"\bphone (number|call)\b",
    r"\b0\d{1,2}-?\d{6,8}\b",                       # Israeli mobile/phone number format
    r"\b05\d-?\d{7}\b",                             # 05x-xxxxxxx (cell)
    r"\b0\d-\d{7}\b",                               # 0x-xxxxxxx (landline)
    r"\bmy (husband|wife|dad|father|mom|mother) will\b",
    r"\bgive (you )?(his|her|my) phone\b",
    r"\bsend (his|her|my) phone\b",
    r"\bcharge (him|her)\b",
    "תתקשר",
    "תתקשרי",
    "תתקשרו",
    "תתקשר אלי",
    "תתקשר אליו",
    "תתקשר אליה",
    "מספר טלפון",
    "טלפון של",
    "אבא ייתן",
    "בעלי ייתן",
    "אשתי תיתן",
    "פנו ל",
    "תפנו ל",
    "תעבור ל",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Pattern list may contain regex or plain Hebrew substrings."""
    if not text:
        return False
    text_lower = text.lower()
    for p in patterns:
        if any(c.isascii() for c in p):
            # ASCII-bearing pattern → regex
            try:
                if re.search(p, text_lower):
                    return True
            except re.error:
                if p in text_lower:
                    return True
        else:
            # Hebrew-only substring
            if p in text:
                return True
    return False



def safe_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def features_for_opp(opp: dict, data: dict) -> dict:
    turns = data["turns"]
    strategies = data["strategies"]
    msgs = data["messages"]

    # Trajectory
    p_conv_seq = [safe_float(t.get("p_conv")) for t in turns if t.get("p_conv") is not None]
    commit_seq = [safe_float(t.get("commitment_level")) for t in turns if t.get("commitment_level") is not None]

    f = {
        "opp_id": opp["id"],
        "motivator": opp.get("primary_motivator"),
        "decision_logic": opp.get("decision_logic"),
        "trust_level": opp.get("trust_level"),
        "regulatory_focus": opp.get("regulatory_focus"),
        "objection_pattern": opp.get("objection_pattern"),
        # raw counts
        "n_msgs": len(msgs),
        "n_inbound_db": int(opp.get("total_inbounds") or 0),
        "n_outbound_db": int(opp.get("total_outbounds") or 0),
        "n_reminders_db": int(opp.get("total_reminders") or 0),
        "n_strategies": len(strategies),
        "n_turn_states": len(turns),
        # trajectory
        "max_commit": max(commit_seq) if commit_seq else 0,
        "final_commit": commit_seq[-1] if commit_seq else 0,
        "max_p_conv": max(p_conv_seq) if p_conv_seq else 0.0,
        "final_p_conv": p_conv_seq[-1] if p_conv_seq else 0.0,
        "p_conv_drop": (max(p_conv_seq) - p_conv_seq[-1]) if len(p_conv_seq) >= 2 else 0.0,
        "had_high_pconv": int(any(x > 0.7 for x in p_conv_seq)) if p_conv_seq else 0,
    }

    # Strategy behaviour
    strategies_seen = [s.get("primary_strategy") for s in strategies if s.get("primary_strategy")]
    f["n_distinct_strategies"] = len(set(strategies_seen))
    f["used_scarcity"] = int("scarcity" in strategies_seen)
    f["used_objection_handling"] = int("objection_handling" in strategies_seen)
    f["used_logistics"] = int("logistics" in strategies_seen)
    f["used_direct_ask"] = int("direct_ask" in strategies_seen)
    f["used_commitment"] = int("commitment" in strategies_seen)
    f["last_strategy"] = strategies_seen[-1] if strategies_seen else None

    # Cialdini activation total
    cialdini_total = 0
    for t in turns:
        for k in ("authority_activated", "social_proof_activated", "reciprocity_activated",
                  "commitment_activated", "scarcity_activated", "liking_activated"):
            if t.get(k):
                cialdini_total += 1
    f["cialdini_total_activations"] = cialdini_total

    # Conversation timing + last-message features
    inbounds = [m for m in msgs if m.get("direction") == "inbound"]
    outbounds = [m for m in msgs if m.get("direction") == "outbound"]
    f["n_inbound_msgs"] = len(inbounds)
    f["n_outbound_msgs"] = len(outbounds)
    f["inbound_ratio"] = len(inbounds) / max(len(msgs), 1)

    if msgs and len(msgs) >= 2:
        # Time gaps
        gaps_h = []
        for i in range(1, len(msgs)):
            t0 = msgs[i - 1]["timestamp"]
            t1 = msgs[i]["timestamp"]
            if t0 and t1:
                gaps_h.append((t1 - t0).total_seconds() / 3600.0)
        f["max_gap_hours"] = max(gaps_h) if gaps_h else 0.0
        f["avg_gap_hours"] = sum(gaps_h) / len(gaps_h) if gaps_h else 0.0
    else:
        f["max_gap_hours"] = 0.0
        f["avg_gap_hours"] = 0.0

    # Last agent message features
    last_outbound = outbounds[-1] if outbounds else None
    if last_outbound:
        ts = last_outbound.get("timestamp")
        text = (last_outbound.get("text") or "").lower()
        f["last_agent_msg_hour"] = ts.hour if ts else None
        f["last_agent_msg_dow"] = ts.weekday() if ts else None
        f["last_agent_msg_len"] = len(last_outbound.get("text") or "")
        f["last_agent_was_question"] = int(bool(QUESTION_RE.search(text)))
        f["last_agent_mentioned_price"] = int(any(h in text for h in PRICE_HINTS))
        f["last_agent_was_closing_ask"] = int(any(h in text for h in CLOSING_ASK_HINTS))
        f["last_agent_was_reminder"] = int(bool(last_outbound.get("is_reminder")))
        f["last_agent_was_followup"] = int(bool(last_outbound.get("is_followup")))
        f["last_agent_automatic"] = int(bool(last_outbound.get("automatic_response")))
    else:
        f["last_agent_msg_hour"] = None
        f["last_agent_msg_dow"] = None
        f["last_agent_msg_len"] = 0
        f["last_agent_was_question"] = 0
        f["last_agent_mentioned_price"] = 0
        f["last_agent_was_closing_ask"] = 0
        f["last_agent_was_reminder"] = 0
        f["last_agent_was_followup"] = 0
        f["last_agent_automatic"] = 0

    # Last customer message features
    last_inbound = inbounds[-1] if inbounds else None
    if last_inbound:
        text = (last_inbound.get("text") or "").lower()
        f["customer_last_text_len"] = len(last_inbound.get("text") or "")
        f["customer_last_was_short"] = int(len((last_inbound.get("text") or "").strip()) <= 5)
        f["customer_b29_trigger"] = int(any(re.search(h, text) for h in B29_HINTS) if any(c.isalpha() for c in text) else any(h in last_inbound.get("text") or "" for h in B29_HINTS))
    else:
        f["customer_last_text_len"] = 0
        f["customer_last_was_short"] = 0
        f["customer_b29_trigger"] = 0

    # B29 trigger anywhere in conversation (any inbound message)
    b29_anywhere = 0
    for m in inbounds:
        text = (m.get("text") or "").lower()
        if any(re.search(h, text) for h in B29_HINTS) or any(h in (m.get("text") or "") for h in B29_HINTS if not h.isascii()):
            b29_anywhere = 1
            break
    f["b29_triggered_anywhere"] = b29_anywhere

    # NEW: competitor-announcement detection (customer said "I went with X" / "chose Y")
    competitor_announced = 0
    competitor_named = None
    for m in inbounds:
        raw_text = m.get("text") or ""
        text = raw_text.lower()
        # Decision-verb match
        verb_hit = any(re.search(p, text) for p in COMPETITOR_DECISION_VERBS if any(c.isascii() for c in p)) \
                   or any(p in raw_text for p in COMPETITOR_DECISION_VERBS if not any(c.isascii() for c in p))
        # Competitor-name match
        name_hit = next((n for n in COMPETITOR_NAMES if (n.isascii() and n in text) or (not n.isascii() and n in raw_text)), None)
        if verb_hit and name_hit:
            competitor_announced = 1
            competitor_named = name_hit
            break
        elif name_hit and not competitor_announced:
            # weaker signal — name mentioned without explicit decision verb
            competitor_announced = max(competitor_announced, 0)  # leave at 0 unless we have verb too
            competitor_named = competitor_named or name_hit
    f["customer_announced_competitor"] = competitor_announced
    f["competitor_named"] = competitor_named

    # NEW: B41 trigger (customer indicates they're not the decision-maker)
    b41_triggered = 0
    for m in inbounds:
        raw_text = m.get("text") or ""
        text = raw_text.lower()
        if any(re.search(p, text) for p in B41_HINTS if any(c.isascii() for c in p)) \
           or any(p in raw_text for p in B41_HINTS if not any(c.isascii() for c in p)):
            b41_triggered = 1
            break
    f["b41_triggered"] = b41_triggered

    # Outcome timing
    if opp.get("created_at") and opp.get("status_update_timestamp"):
        f["days_creation_to_close"] = (opp["status_update_timestamp"] - opp["created_at"]).total_seconds() / 86400.0
    else:
        f["days_creation_to_close"] = None

    if msgs and opp.get("status_update_timestamp"):
        last_inbound_ts = inbounds[-1]["timestamp"] if inbounds else None
        if last_inbound_ts:
            f["days_last_inbound_to_close"] = (opp["status_update_timestamp"] - last_inbound_ts).total_seconds() / 86400.0
        else:
            f["days_last_inbound_to_close"] = None
    else:
        f["days_last_inbound_to_close"] = None

    # NEW: noise-status flag — gap > 90 days suggests retroactive status flip / data artifact
    days_gap = f["days_last_inbound_to_close"]
    f["noise_status_artifact"] = int((days_gap is not None) and (days_gap > 90 or days_gap < -7))

    # NEW v3 (2026-04-29): conversation_won_signal — derived from transcript content.
    # Replaces opportunity.status as the primary outcome label, since 96.6% of
    # Insurance ClosedLost transitions are written by Excel-driven Sync Routine
    # (arq_impl/sync_routine_task.py) and reflect policy-lifecycle state, not
    # conversation outcome.

    # (a) Agent emitted a final-confirmation pattern at least once
    agent_final_confirmation = 0
    for m in outbounds:
        if _matches_any(m.get("text") or "", AGENT_FINAL_CONFIRMATION_PATTERNS):
            agent_final_confirmation = 1
            break
    f["agent_final_confirmation"] = agent_final_confirmation

    # (b) Customer signal: provided CC last-4 OR installment count.
    # v3.1: relaxed CC regex (4-8 digit run) and added word-number installments.
    customer_provided_cc = 0
    customer_named_installments = 0
    agent_recently_asked_cc = False
    for m in msgs:
        text = (m.get("text") or "").strip()
        if m.get("direction") == "outbound":
            agent_recently_asked_cc = _matches_any(text, AGENT_CC_PROMPT_PATTERNS)
            continue
        # inbound branch
        is_only_digits = bool(CC_ONLY_DIGITS_RE.match(text)) and bool(CC_LAST4_RE.search(text))
        if is_only_digits:
            customer_provided_cc = 1
        elif agent_recently_asked_cc and CC_ANY_4PLUS_RE.search(text):
            # v3.1: agent prompted CC → accept any 4-8 digit run anywhere in reply
            customer_provided_cc = 1
        if INSTALLMENT_COUNT_RE.search(text) or INSTALLMENT_WORD_RE.search(text) \
           or any(p in text for p in INSTALLMENT_HEBREW_PATTERNS):
            customer_named_installments = 1
        agent_recently_asked_cc = False
        if customer_provided_cc and customer_named_installments:
            break
    f["customer_provided_cc"] = customer_provided_cc
    f["customer_named_installments"] = customer_named_installments

    # (c) Customer explicit verbal close
    customer_explicit_close = 0
    for m in inbounds:
        if _matches_any(m.get("text") or "", CUSTOMER_EXPLICIT_CLOSE_PATTERNS):
            customer_explicit_close = 1
            break
    f["customer_explicit_close"] = customer_explicit_close

    # (d) Reached commitment_level=5 in research_turn_state
    f["commitment_5_reached"] = int(f["max_commit"] >= 5)

    # (e) NEW (v3.1) — phone-handoff-after-engagement.
    # Fires only if the customer has reached commitment_level ≥ 3 at some turn,
    # i.e. they're already engaged. Below that threshold, "call me" is just
    # routine logistics (or politely declining), not a Won signal.
    phone_handoff = 0
    if f["max_commit"] >= 3:
        # Walk through messages chronologically. Once max_commit≥3 has been seen
        # (we approximate by max_commit aggregate since per-turn commit isn't
        # easily aligned to message timestamps across the two tables), look for
        # phone-handoff patterns in either inbound or outbound text.
        for m in msgs:
            text = m.get("text") or ""
            if _matches_any(text, PHONE_HANDOFF_PATTERNS):
                phone_handoff = 1
                break
    f["phone_handoff_after_engagement"] = phone_handoff

    # Compound conversation_won_signal (v3.1):
    #   5 components: final_conf, cc_or_installments, explicit_close, commit5, phone_handoff
    #   STRONG (=2): ≥3 components fire OR (final_confirmation AND ≥1 other)
    #   WEAK   (=1): ≥2 components fire (but not STRONG)
    #   ABSENT (=0): otherwise
    cc_or_installments = customer_provided_cc or customer_named_installments
    components = [
        agent_final_confirmation,
        int(bool(cc_or_installments)),
        customer_explicit_close,
        f["commitment_5_reached"],
        phone_handoff,
    ]
    signal_count = sum(components)
    strong = bool(
        signal_count >= 3 or
        (agent_final_confirmation and signal_count >= 2)
    )
    weak = (signal_count >= 2) and not strong

    if strong:
        f["conversation_won_signal"] = 2
    elif weak:
        f["conversation_won_signal"] = 1
    else:
        f["conversation_won_signal"] = 0

    f["conversation_won_strong"] = int(strong)
    f["conversation_won_weak_only"] = int(weak)
    f["won_signal_count"] = signal_count

    return f


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Opening prod connection")
    conn = open_prod_conn()

    log.info("Sampling %d Insurance ClosedLost opps", SAMPLE_SIZE)
    opps = sample_opps(conn)
    log.info("Sampled %d opps", len(opps))

    log.info("Fetching state + computing features for each")
    feature_rows: list[dict] = []
    sample_data: dict[str, dict] = {}  # for later transcript inspection
    for i, opp in enumerate(opps, 1):
        if i % 50 == 0:
            log.info("  ...%d/%d", i, len(opps))
        try:
            data = fetch_opp_data(conn, opp["id"])
            features = features_for_opp(opp, data)
            feature_rows.append(features)
            sample_data[opp["id"]] = data
        except Exception as e:
            log.warning("opp %s failed: %s", opp["id"], e)
    conn.close()
    log.info("Computed features for %d opps", len(feature_rows))

    df = pd.DataFrame(feature_rows)
    df.to_csv(f"{OUTPUT_DIR}/features.csv", index=False)
    log.info("Wrote features CSV (%d rows × %d cols)", df.shape[0], df.shape[1])

    # ── Distribution summary ──
    print("\n" + "=" * 60)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 60)
    print("\nMotivator distribution:")
    print(df["motivator"].value_counts())
    print("\nObjection pattern distribution:")
    print(df["objection_pattern"].value_counts().head(10))
    print("\nLast strategy distribution:")
    print(df["last_strategy"].value_counts().head(10))
    print("\nB29 triggered anywhere: {:.1%}".format(df["b29_triggered_anywhere"].mean()))
    print("Used scarcity (anywhere): {:.1%}".format(df["used_scarcity"].mean()))
    print("Used objection_handling (anywhere): {:.1%}".format(df["used_objection_handling"].mean()))

    # ── v3: conversation_won_signal distribution ──
    print("\n" + "=" * 60)
    print("CONVERSATION_WON_SIGNAL DISTRIBUTION (v3)")
    print("=" * 60)
    print("On 500 ClosedLost opps; transcript-derived outcome label:")
    print(f"  Strong (final-confirmation + customer-action) : {(df['conversation_won_strong']==1).sum():3d} "
          f"({(df['conversation_won_strong']==1).mean():.1%})")
    print(f"  Weak (≥2 signals, no final-confirmation+action): {(df['conversation_won_weak_only']==1).sum():3d} "
          f"({(df['conversation_won_weak_only']==1).mean():.1%})")
    print(f"  Absent (true conversation losses)              : {((df['conversation_won_signal']==0)).sum():3d} "
          f"({((df['conversation_won_signal']==0)).mean():.1%})")
    print(f"  Component rates:")
    print(f"    agent_final_confirmation     : {df['agent_final_confirmation'].mean():.1%}")
    print(f"    customer_provided_cc          : {df['customer_provided_cc'].mean():.1%}")
    print(f"    customer_named_installments   : {df['customer_named_installments'].mean():.1%}")
    print(f"    customer_explicit_close       : {df['customer_explicit_close'].mean():.1%}")
    print(f"    commitment_5_reached          : {df['commitment_5_reached'].mean():.1%}")
    print(f"    phone_handoff_after_engagement: {df['phone_handoff_after_engagement'].mean():.1%}")
    print(f"  Cross-tab: conversation_won_signal × noise_status_artifact:")
    print(pd.crosstab(df["conversation_won_signal"], df["noise_status_artifact"],
                       margins=True, margins_name="Total").to_string())
    print("\nMax commit_level distribution:")
    print(df["max_commit"].value_counts().sort_index())
    print("\nMax p_conv quartiles:")
    print(df["max_p_conv"].describe())
    print("\nN msgs distribution:")
    print(df["n_msgs"].describe())
    print("\nLast agent msg time-of-day (hour):")
    print(df["last_agent_msg_hour"].value_counts().sort_index())

    # ── Clustering on numeric features ──
    numeric_feats = [
        "n_msgs", "n_inbound_msgs", "inbound_ratio", "n_distinct_strategies",
        "max_commit", "final_commit", "max_p_conv", "final_p_conv", "p_conv_drop",
        "had_high_pconv", "cialdini_total_activations",
        "max_gap_hours", "avg_gap_hours",
        "last_agent_msg_len", "last_agent_was_question",
        "last_agent_mentioned_price", "last_agent_was_closing_ask",
        "last_agent_was_reminder", "last_agent_was_followup",
        "customer_last_text_len", "customer_last_was_short",
        "customer_b29_trigger", "b29_triggered_anywhere",
        "used_scarcity", "used_objection_handling", "used_logistics",
        "used_direct_ask", "used_commitment",
        # NEW v2 features (refined per user feedback)
        "customer_announced_competitor", "b41_triggered", "noise_status_artifact",
        # v3 — transcript-derived conversation outcome (replaces status as label)
        "agent_final_confirmation", "customer_provided_cc",
        "customer_named_installments", "customer_explicit_close",
        "commitment_5_reached",
        # v3.1 — phone-handoff component
        "phone_handoff_after_engagement",
    ]

    Xnum = df[numeric_feats].fillna(0).astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xnum)

    log.info("Clustering with KMeans(K=%d)", N_CLUSTERS_KMEANS)
    km = KMeans(n_clusters=N_CLUSTERS_KMEANS, random_state=7919, n_init=10)
    df["cluster"] = km.fit_predict(Xs)

    print("\n" + "=" * 60)
    print(f"CLUSTERS (K={N_CLUSTERS_KMEANS})")
    print("=" * 60)
    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())

    # ── Per-cluster summary ──
    print("\n" + "=" * 60)
    print("CLUSTER PROFILES (mean of key features)")
    print("=" * 60)
    profile_feats = [
        "n_msgs", "max_commit", "max_p_conv", "p_conv_drop",
        "n_distinct_strategies", "cialdini_total_activations",
        "max_gap_hours", "avg_gap_hours", "days_last_inbound_to_close",
        "last_agent_msg_len", "last_agent_was_question",
        "last_agent_was_closing_ask", "last_agent_was_reminder",
        "customer_b29_trigger", "b29_triggered_anywhere",
        "customer_announced_competitor", "b41_triggered", "noise_status_artifact",
        "used_scarcity", "used_objection_handling",
        # v3
        "agent_final_confirmation", "customer_provided_cc",
        "customer_explicit_close", "commitment_5_reached",
        "conversation_won_strong",
    ]
    summary = df.groupby("cluster")[profile_feats].mean().round(2)
    summary["count"] = df.groupby("cluster").size()
    # Add count of strong-won-signal opps per cluster (these are likely
    # policy-lifecycle losses, not conversation losses)
    summary["n_won_strong"] = df.groupby("cluster")["conversation_won_strong"].sum()
    print(summary.to_string())

    # ── v3: cluster × conversation_won_signal cross-tab ──
    print("\n=== Cluster × conversation_won_signal ===")
    print(pd.crosstab(df["cluster"], df["conversation_won_signal"],
                       margins=True, margins_name="Total").to_string())

    # categorical distribution per cluster
    print("\n=== Motivator distribution per cluster ===")
    cross = pd.crosstab(df["cluster"], df["motivator"])
    cross_pct = cross.div(cross.sum(axis=1), axis=0).round(2)
    print(cross_pct.to_string())

    print("\n=== Objection pattern (top) per cluster ===")
    for c in sorted(df["cluster"].unique()):
        top = df[df["cluster"] == c]["objection_pattern"].value_counts().head(3)
        print(f"  Cluster {c}: {dict(top)}")

    print("\n=== Last strategy per cluster ===")
    for c in sorted(df["cluster"].unique()):
        top = df[df["cluster"] == c]["last_strategy"].value_counts().head(3)
        print(f"  Cluster {c}: {dict(top)}")

    # ── Sample opp-ids per cluster + transcripts for review ──
    samples = {}
    for c in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == c].head(5)  # first 5 per cluster
        sample_list = []
        for _, row in cluster_df.iterrows():
            opp_id = row["opp_id"]
            opp_data = sample_data.get(opp_id, {})
            transcript = []
            for m in opp_data.get("messages", [])[:30]:  # cap at 30 turns
                role = "Customer" if m.get("direction") == "inbound" else "Agent"
                ts = m.get("timestamp").isoformat() if m.get("timestamp") else None
                text = (m.get("text") or "").strip()[:600]
                transcript.append({"role": role, "ts": ts, "text": text})
            sample_list.append({
                "opp_id": opp_id,
                "features": {k: (v if not isinstance(v, float) else round(v, 3))
                             for k, v in row.to_dict().items() if k != "cluster"},
                "n_msgs": len(opp_data.get("messages", [])),
                "transcript": transcript,
            })
        samples[f"cluster_{c}"] = sample_list

    with open(f"{OUTPUT_DIR}/cluster_samples.json", "w") as f:
        json.dump(samples, f, indent=2, default=str, ensure_ascii=False)
    log.info("Wrote cluster_samples.json with 5 transcripts per cluster")

    # Cluster summary CSV
    summary.to_csv(f"{OUTPUT_DIR}/cluster_summary.csv")
    log.info("Wrote cluster_summary.csv")

    print(f"\nOutputs in {OUTPUT_DIR}/:")
    print("  features.csv          — full feature matrix")
    print("  cluster_summary.csv   — per-cluster mean of key features")
    print("  cluster_samples.json  — 5 sample transcripts per cluster")


if __name__ == "__main__":
    main()
