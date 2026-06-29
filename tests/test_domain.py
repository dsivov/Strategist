"""Domain-pack pluggability tests. Verifies that swapping the active domain
pack actually changes the domain-coupled surfaces (the inverse of the golden
tests, which pin the default sales behavior). No LLM calls."""
from __future__ import annotations
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "poc"))
os.environ.setdefault("ANTHROPIC_API_KEY", "domain-test-placeholder")
os.environ.setdefault("GEMINI_API_KEY_1", "domain-test-placeholder")


def test_default_active_is_sales():
    from poc import active_domain
    assert active_domain().name == "sales"


def test_swap_domain_changes_actor_text():
    """Registering + activating a custom pack changes actor's domain text,
    then restoring 'sales' returns the original behavior."""
    from poc import DomainPack, set_active_domain
    import actor as la

    class SupportDomain(DomainPack):
        name = "support-test"
        def describe_tenant(self, tenant):
            return "B2C technical support; resolution = ticket closed"
        def opp_type_note(self, opp_type):
            return "\n  → Support contact, not a sale." if opp_type else ""
        def render_anchor_section(self, anchors):
            return ""  # no pricing in a support domain

    try:
        set_active_domain(SupportDomain())
        assert la._domain_desc("Insurance") == "B2C technical support; resolution = ticket closed"
        assert la._opp_type_behavioral_note("Bug Report") == "\n  → Support contact, not a sale."
        assert la._build_anchor_section({"last_year_price_usd": 4238}) == ""
    finally:
        set_active_domain("sales")

    # restored
    assert la._domain_desc("Insurance").startswith("B2C auto insurance renewal")


def test_outcome_signals_come_from_domain():
    """benchmark outcome detection delegates to the active pack."""
    from poc import DomainPack, set_active_domain
    from poc.benchmark import Benchmark

    class NeverWinDomain(DomainPack):
        name = "neverwin-test"
        def detect_close(self, text):
            return False
        def detect_decline(self, text):
            return True

    try:
        set_active_domain(NeverWinDomain())
        # "yes, renew now" would be a win under sales; under this pack it's a loss
        assert Benchmark._check_customer_outcome("yes, renew now") == "lost"
    finally:
        set_active_domain("sales")

    assert Benchmark._check_customer_outcome("yes, renew now") == "won"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
