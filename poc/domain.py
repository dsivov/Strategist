"""Domain packs — the seam that turns baked-in sales/insurance assumptions into
a pluggable provider.

The benchmark dataset is, and stays, a sales/insurance corpus. What this module
decouples is the *code's* assumptions about that domain: the customer-facing
agent's tenant/opp-type framing, the economic-anchor rendering, and the
won/lost outcome signals. Those used to be inlined across `actor`,
`customer_simulator`, and `benchmark`; now they come from the *active domain
pack*, so the sales corpus is one registered provider rather than a hardcoded
assumption.

`SalesDomainPack` reproduces the original strings/behavior byte-for-byte (the
golden characterization tests pin this). To target a new domain, subclass
`DomainPack`, register it, and select it with `set_active_domain(...)` /
`POC_DOMAIN=<name>` — no edits to the core modules.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

log = logging.getLogger(__name__)


# ── Interface ────────────────────────────────────────────────────────────────

class DomainPack:
    """Base class for a domain. Subclass and override what differs.

    The defaults are deliberately neutral so a minimal pack still works; the
    sales corpus lives in `SalesDomainPack`.
    """

    name: str = "generic"
    #: tenant ids this pack recognizes (informational; selection is by name)
    tenants: tuple[str, ...] = ()

    # — customer-facing agent framing (consumed by actor) —

    def describe_tenant(self, tenant: str) -> str:
        """One-line business/conversion description for the agent prompt."""
        return "B2C messaging"

    def opp_type_note(self, opp_type: str) -> str:
        """Opportunity-type behavioral coaching appended to the agent prompt."""
        return ""

    def render_anchor_section(self, anchors: dict) -> str:
        """The 'economic reference frame' block of the agent system prompt.
        Return '' when the domain has no pricing/anchor notion."""
        return ""

    # — outcome model (consumed by benchmark / scoring) —

    def detect_close(self, text: str) -> bool:
        """True when the customer's text signals a win (conversion)."""
        return False

    def detect_decline(self, text: str) -> bool:
        """True when the customer's text signals a loss (decline)."""
        return False


# ── Sales / insurance reference pack ─────────────────────────────────────────

class SalesDomainPack(DomainPack):
    """The original Insurance/Ecommerce sales+insurance domain, verbatim.

    Behavior here is pinned by tests/test_characterization.py — keep output
    byte-identical when editing.
    """

    name = "sales"
    tenants = ("Insurance", "Ecommerce", "SaaS", "CleaningCommerce", "MattressCommerce")

    _DOMAIN_DESC = {
        "Insurance": "B2C auto insurance renewal in Israel; conversion = customer agrees to renew + provides CC last-4 digits",
        "Ecommerce": "B2C e-commerce (premium headphones); conversion = customer completes the order via the cart link",
        "SaaS": "B2B SaaS subscription (creative-business platform); conversion = trial-to-paid plan",
        "CleaningCommerce": "B2C e-commerce (cleaning products); conversion = customer completes the order",
        "MattressCommerce": "B2C e-commerce (mattresses); conversion = customer completes the order",
    }

    def describe_tenant(self, tenant: str) -> str:
        return self._DOMAIN_DESC.get(tenant, "B2C messaging")

    def opp_type_note(self, opp_type: str) -> str:
        t = (opp_type or "").lower()
        if "renewal" in t or "renew" in t:
            return ("\n  → Customer is an EXISTING customer at renewal. They have a "
                    "prior relationship, prior premium, and likely competitor quotes. "
                    "Match their reference frame; don't pitch as if this is a cold sale.")
        if any(k in t for k in ("abandoned cart", "cart abandon", "abandon cart")):
            return ("\n  → Customer added a product to cart and walked away. They had "
                    "buy intent. Address what changed since cart-add (price hesitation, "
                    "second thoughts, distraction); don't re-pitch the product from scratch.")
        if "upsell" in t or "cross" in t or "expansion" in t:
            return ("\n  → Customer already has a base product; this is an upgrade "
                    "conversation. Anchor on their current setup; surface incremental value.")
        if "trial" in t:
            return ("\n  → Customer is mid-trial → paid conversion. Lean on their "
                    "actual experience with the trial; don't over-explain features.")
        if "purchasing assistance" in t or "search_catalog" in t or "browse" in t:
            return ("\n  → Customer reached out for help selecting. Probe intent "
                    "(buy-ready vs browsing) before pushing a specific product.")
        if "review" in t:
            return ("\n  → Post-purchase review request, not a sales conversation. "
                    "Don't pitch; gather feedback.")
        return ""

    def render_anchor_section(self, anchors: dict) -> str:
        anchors = anchors or {}
        lines = ["", "# Customer's economic reference frame (use this — don't fabricate market claims)"]
        if anchors.get("last_year_price_usd"):
            lines.append(f"- Last year's premium: {anchors['last_year_price_usd']} USD")
        if anchors.get("current_quoted_price_usd"):
            lines.append(f"- Current quoted price: {anchors['current_quoted_price_usd']} USD")
        if anchors.get("actual_market_yoy_change_pct") is not None:
            lines.append(f"- ACTUAL market YoY change in our prod data: {anchors['actual_market_yoy_change_pct']:+.1f}% (NOT a 48% hike — do not fabricate)")
        if anchors.get("market_avg_for_segment_usd"):
            lines.append(f"- Market avg for this customer's vehicle segment: {anchors['market_avg_for_segment_usd']} USD")
        if anchors.get("max_discount_pct_internal") is not None:
            lines.append(f"- Max internal discretionary discount: {anchors['max_discount_pct_internal']}%")
        if anchors.get("loyalty_years"):
            lines.append(f"- Customer loyalty: {anchors['loyalty_years']} years")
        lines.append("")
        lines.append("# Anti-staircase rule")
        lines.append("- Once you have stated a price within 5% of market_avg or below a competitor offer the customer mentioned, HOLD that price. Do not drop again — multiple price drops damage trust.")
        lines.append("- If customer continues to push, address objections via coverage value, deductibles, or relationship — not further discounts.")
        lines.append("")
        return "\n".join(lines)

    def detect_close(self, text: str) -> bool:
        # Delegate to the calibrated detectors in the simulator (Hebrew + English,
        # tenant-aware). Imported lazily to avoid a heavy import at module load.
        from customer_simulator import detect_close_signal
        return detect_close_signal(text)

    def detect_decline(self, text: str) -> bool:
        from customer_simulator import detect_decline
        return detect_decline(text)


# ── Registry + active selection ──────────────────────────────────────────────

_DOMAINS: dict[str, DomainPack] = {}
_ACTIVE: DomainPack | None = None


def register_domain(pack: DomainPack, *, replace: bool = False) -> DomainPack:
    if pack.name in _DOMAINS and not replace:
        raise ValueError(f"domain {pack.name!r} already registered")
    _DOMAINS[pack.name] = pack
    return pack


def get_domain(name: str) -> DomainPack:
    return _DOMAINS[name]


def all_domains() -> list[DomainPack]:
    return sorted(_DOMAINS.values(), key=lambda d: d.name)


def set_active_domain(name_or_pack) -> DomainPack:
    """Select the active domain by name or instance."""
    global _ACTIVE
    if isinstance(name_or_pack, DomainPack):
        register_domain(name_or_pack, replace=True)
        _ACTIVE = name_or_pack
    else:
        _ACTIVE = _DOMAINS[name_or_pack]
    return _ACTIVE


def active_domain() -> DomainPack:
    """The currently selected domain pack (defaults via POC_DOMAIN, else sales)."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _DOMAINS.get(os.environ.get("POC_DOMAIN", "sales")) \
            or _DOMAINS["sales"]
    return _ACTIVE


# Register the reference pack and make it the default.
register_domain(SalesDomainPack())
