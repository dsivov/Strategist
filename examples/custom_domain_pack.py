"""Example: add a new domain without editing the core.

The bundled benchmark data is a sales/insurance corpus, so this pack won't
change *that* dataset's outcomes — it shows the seam. To benchmark a genuinely
different domain you'd pair a pack like this with a matching ScenarioSource
(see examples below / poc.scenario_source).

Run:  python examples/custom_domain_pack.py
"""
from __future__ import annotations

import sys
from pathlib import Path
# Make `import poc` work when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc import DomainPack, set_active_domain, active_domain
import actor


class SupportDomain(DomainPack):
    """A B2C technical-support domain: 'conversion' = resolved ticket."""

    name = "support"
    tenants = ("Acme",)

    def describe_tenant(self, tenant: str) -> str:
        return "B2C technical support; resolution = customer confirms the issue is fixed"

    def opp_type_note(self, opp_type: str) -> str:
        t = (opp_type or "").lower()
        if "bug" in t or "error" in t:
            return ("\n  → Customer hit a defect. Acknowledge, reproduce, and give a "
                    "concrete next step or workaround; don't deflect.")
        if "how" in t or "question" in t:
            return "\n  → How-to question. Answer directly with steps; link docs if useful."
        return ""

    def render_anchor_section(self, anchors: dict) -> str:
        return ""  # no pricing/anchors in a support domain

    def detect_close(self, text: str) -> bool:
        t = (text or "").lower()
        return any(p in t for p in ("that worked", "it's fixed", "resolved", "thanks, solved"))

    def detect_decline(self, text: str) -> bool:
        t = (text or "").lower()
        return any(p in t for p in ("still broken", "didn't work", "cancel my account"))


if __name__ == "__main__":
    print("default domain:", active_domain().name)
    set_active_domain(SupportDomain())
    print("active domain :", active_domain().name)
    # The agent's domain framing now comes from the support pack:
    print("describe_tenant:", actor._domain_desc("Acme"))
    print("opp_type_note  :", actor._opp_type_behavioral_note("Bug Report").strip())
    print("anchor_section :", repr(actor._build_anchor_section({"last_year_price_usd": 4238})))
    set_active_domain("sales")  # restore
    print("restored domain:", active_domain().name)
