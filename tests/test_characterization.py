"""Characterization (golden) tests — capture CURRENT behavior of the
domain-coupled surface so the domain-pack extraction (Phase 3) can be proven
behavior-preserving. No LLM calls: every assertion exercises pure rendering /
regex-detection code.

If one of these fails after a refactor, the refactor changed observable
behavior. That is either a bug (fix it) or an intentional change (re-record the
golden value in the same commit, with justification).

Recorded against the sales/insurance reference domain on the initial commit.
"""
from __future__ import annotations
import hashlib
import os
import sys
from pathlib import Path

# Same import bootstrap as test_smoke.py
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "poc"))
os.environ.setdefault("ANTHROPIC_API_KEY", "characterization-placeholder")
os.environ.setdefault("GEMINI_API_KEY_1", "characterization-placeholder")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── actor: per-tenant domain description ────────────────────────────────

def test_domain_desc_golden():
    import actor as la
    assert la._domain_desc("Insurance") == (
        "B2C auto insurance renewal in Israel; conversion = customer agrees "
        "to renew + provides CC last-4 digits")
    assert la._domain_desc("Ecommerce") == (
        "B2C e-commerce (premium headphones); conversion = customer completes "
        "the order via the cart link")
    assert la._domain_desc("SaaS") == (
        "B2B SaaS subscription (creative-business platform); conversion = "
        "trial-to-paid plan")
    assert la._domain_desc("CleaningCommerce") == (
        "B2C e-commerce (cleaning products); conversion = customer completes "
        "the order")
    assert la._domain_desc("MattressCommerce") == (
        "B2C e-commerce (mattresses); conversion = customer completes the order")
    # unknown tenant falls back
    assert la._domain_desc("Unknown") == "B2C messaging"


# ── actor: opp-type behavioral coaching ─────────────────────────────────

def test_opp_type_note_golden():
    import actor as la
    assert la._opp_type_behavioral_note("Insurance Renewal").startswith(
        "\n  → Customer is an EXISTING customer at renewal.")
    assert la._opp_type_behavioral_note("Abandoned Cart").startswith(
        "\n  → Customer added a product to cart and walked away.")
    assert la._opp_type_behavioral_note("Upsell").startswith(
        "\n  → Customer already has a base product;")
    assert la._opp_type_behavioral_note("Trial").startswith(
        "\n  → Customer is mid-trial → paid conversion.")
    assert la._opp_type_behavioral_note("Purchasing Assistance").startswith(
        "\n  → Customer reached out for help selecting.")
    assert la._opp_type_behavioral_note("Review Request").startswith(
        "\n  → Post-purchase review request, not a sales conversation.")
    # unrecognized / empty → no note
    assert la._opp_type_behavioral_note("Something Else") == ""
    assert la._opp_type_behavioral_note("") == ""


# ── actor: economic-anchor section ──────────────────────────────────────

_ANCHORS = {
    "last_year_price_usd": 4238,
    "current_quoted_price_usd": 4581,
    "actual_market_yoy_change_pct": 2.5,
    "market_avg_for_segment_usd": 4400,
    "max_discount_pct_internal": 15,
    "loyalty_years": 3,
}


def test_anchor_section_full_golden():
    import actor as la
    expected = (
        "\n# Customer's economic reference frame (use this — don't fabricate market claims)\n"
        "- Last year's premium: 4238 USD\n"
        "- Current quoted price: 4581 USD\n"
        "- ACTUAL market YoY change in our prod data: +2.5% (NOT a 48% hike — do not fabricate)\n"
        "- Market avg for this customer's vehicle segment: 4400 USD\n"
        "- Max internal discretionary discount: 15%\n"
        "- Customer loyalty: 3 years\n"
        "\n# Anti-staircase rule\n"
        "- Once you have stated a price within 5% of market_avg or below a competitor "
        "offer the customer mentioned, HOLD that price. Do not drop again — multiple "
        "price drops damage trust.\n"
        "- If customer continues to push, address objections via coverage value, "
        "deductibles, or relationship — not further discounts.\n"
    )
    assert la._build_anchor_section(_ANCHORS) == expected


def test_anchor_section_empty_golden():
    import actor as la
    out = la._build_anchor_section({})
    assert "Last year's premium" not in out
    assert "# Anti-staircase rule" in out  # the rule block is unconditional


# ── actor: full actor system prompt (integration guard) ─────────────────

_OPP = {
    "id": "o1", "company": "Insurance", "opp_type": "Insurance Renewal",
    "primary_motivator": "Price/savings", "decision_logic": "Analytical",
    "trust_level": "Skeptical", "communication_style": "Terse",
    "anchors": _ANCHORS,
}


def test_actor_system_prompt_golden():
    """Strongest guard: the assembled customer-facing system prompt. If the
    domain extraction is byte-preserving, this hash is unchanged."""
    import actor as la
    sysp = la.build_actor_system(_OPP, "")
    assert len(sysp) == 4272
    assert _sha(sysp) == (
        "379414da7fcc9c8effe6bd2d53a459954155e6e09238ec36e545d6de9c9295fc")


# ── customer_simulator: opp-type coherence constraint ────────────────────────

def test_coherence_constraint_golden():
    import customer_simulator as cs
    cases = {
        "Insurance Renewal": "e9803e440ed0319a",
        "Abandoned Cart":    "a3f88e9fbc392146",
        "Trial":             "d2effdc176b78e5a",
        "Review Request":    "7144d0d74795c4f4",
        "Other":             "e3b0c44298fc1c14",  # empty string
    }
    for opp_type, want in cases.items():
        opp = dict(_OPP)
        opp["opp_type"] = opp_type
        got = cs.render_opp_coherence_constraint(opp)
        assert _sha(got)[:16] == want, f"coherence drift for {opp_type!r}"


# ── customer_simulator: outcome-signal detectors ─────────────────────────────

def test_close_signal_detector_golden():
    import customer_simulator as cs
    for t in ["yes, renew", "let's renew", "go ahead", "send me the link",
              "last 4 digits are 1234"]:
        assert cs.detect_close_signal(t) is True, t
    for t in ["I need to think about it", "not sure yet", "can you tell me more?"]:
        assert cs.detect_close_signal(t) is False, t


def test_decline_detector_golden():
    import customer_simulator as cs
    for t in ["not interested", "going elsewhere",
              "I already signed with another", "done here"]:
        assert cs.detect_decline(t) is True, t
    for t in ["not interested in paying more", "tell me more"]:
        assert cs.detect_decline(t) is False, t


def test_graceful_close_and_refusal_golden():
    import customer_simulator as cs
    assert cs.detect_agent_graceful_close("No worries, have a great day!") is True
    assert cs.detect_agent_graceful_close("Shall we proceed?") is False
    assert cs.detect_agent_refusal("I cannot help with that") is False


# ── benchmark: outcome check + paired summary math ───────────────────────────

def test_benchmark_outcome_check_golden():
    from poc.benchmark import Benchmark
    assert Benchmark._check_customer_outcome("yes, renew now") == "won"
    assert Benchmark._check_customer_outcome("last four digits are 1234") == "won"
    assert Benchmark._check_customer_outcome("not interested, going elsewhere") == "lost"
    assert Benchmark._check_customer_outcome("I need to think about it") is None
    assert Benchmark._check_customer_outcome("maybe later") is None


def test_paired_summary_golden():
    from poc.benchmark import paired_summary
    ps = paired_summary({
        "a": [{"scenario_id": "s1", "outcome": "won"},
              {"scenario_id": "s2", "outcome": "lost"}],
        "b": [{"scenario_id": "s1", "outcome": "lost"},
              {"scenario_id": "s2", "outcome": "lost"}],
    })
    assert ps["n_scenarios"] == 2
    assert ps["arms"]["a"] == {"n": 2, "won": 1, "win_rate": 0.5}
    assert ps["arms"]["b"] == {"n": 2, "won": 0, "win_rate": 0.0}
    assert ps["pairwise"] == {"a_better": 1, "b_better": 0, "ties": 1}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
