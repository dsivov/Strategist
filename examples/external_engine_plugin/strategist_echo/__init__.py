"""A minimal out-of-tree engine plugin.

`EchoEngine` implements the `poc.Engine` protocol — one async `produce()` that
returns (customer_facing_text, telemetry_meta). It just echoes the last
customer message; the point is to demonstrate the plugin wiring, not strategy.

The `strategist.engines` entry point in pyproject.toml points at `get_engines`,
which the registry calls during discovery. We import `EngineSpec`/`ParamSpec`
lazily inside the function so this module imports cleanly regardless of how the
host has the package on sys.path (`poc` package vs. flat module dir).
"""
from __future__ import annotations


class EchoEngine:
    def __init__(self, prefix: str = "Echo"):
        self.prefix = prefix

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        last_customer = ""
        for m in reversed(dialog_history or []):
            if m.get("role") == "customer":
                last_customer = m.get("text") or ""
                break
        text = f"{self.prefix}: {last_customer}" if last_customer else f"{self.prefix}: hello"
        return text, {"engine": "echo", "prefix": self.prefix}


def get_engines():
    try:
        from poc import EngineSpec, ParamSpec
    except Exception:  # host put the package dir (not its parent) on sys.path
        from registry import EngineSpec, ParamSpec  # type: ignore

    return [
        EngineSpec(
            id="echo",
            name="Echo (example plugin)",
            description="Example out-of-tree engine discovered via entry_points; "
                        "echoes the last customer message back with a prefix.",
            runnable=True,
            params=(
                ParamSpec(name="prefix", label="Prefix", type="string",
                          default="Echo", help="Text prepended to the echo."),
            ),
            factory=lambda prefix="Echo": EchoEngine(prefix=prefix),
        ),
    ]
