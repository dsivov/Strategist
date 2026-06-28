"""Session logger — persists each POC run for analysis, bug-finding, improvement.

Per session: writes a JSON file at test_logs/{date}/{session_id}.json with:
  - scenario metadata (opp_id, tenant, cluster, motivator, decision_logic, expected_lift)
  - seed dialog (with prod persuasion scores where available)
  - left panel: full dialog + scores + directive (none) + outcome + errors
  - right panel: full dialog + scores + directive details + outcome + errors
  - timing: per-turn latencies, total session duration
  - simulator info: per-customer-turn mode (rephrase / generate)
  - aggregate: outcomes (won/lost/timeout), persuasion-score deltas, win-rate signal

The logger collects events as the session progresses, then writes once at session_complete.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_BASE = os.environ.get(
    "POC_TEST_LOGS_DIR",
    os.path.join(_PROJECT_ROOT, "test_logs"),
)


class SessionLogger:
    def __init__(self, session_id: str, opp_id: str, scenario: dict | None = None):
        self.session_id = session_id
        self.opp_id = opp_id
        self.scenario = scenario or {}
        self.started_at = time.time()
        self.events: list[dict] = []
        self.errors: list[dict] = []
        self.notes: list[str] = []

    def add_event(self, ev: dict) -> None:
        """Record any WebSocket event being sent."""
        self.events.append({"t": round(time.time() - self.started_at, 2), **ev})

    def add_error(self, where: str, error: str) -> None:
        self.errors.append({"t": round(time.time() - self.started_at, 2),
                             "where": where, "error": error})

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    # ── Aggregation ─────────────────────────────────────────────────────────
    def _split_by_side(self) -> dict:
        left_msgs, right_msgs = [], []
        left_scores, right_scores = [], []
        for e in self.events:
            ev = e.get("event")
            if ev == "left_msg":
                left_msgs.append(e)
            elif ev == "right_msg":
                right_msgs.append(e)
            elif ev == "left_score":
                left_scores.append(e)
            elif ev == "right_score":
                right_scores.append(e)
        return {
            "left_msgs": left_msgs, "right_msgs": right_msgs,
            "left_scores": left_scores, "right_scores": right_scores,
        }

    def _outcome_for_side(self, side: str) -> dict:
        for e in reversed(self.events):
            if e.get("event") == "end" and e.get("side") == side:
                return {"outcome": e.get("outcome"), "reason": e.get("reason")}
        return {"outcome": "incomplete", "reason": None}

    def _persuasion_delta(self, scores: list[dict]) -> dict:
        if not scores:
            return {"start": None, "end": None, "delta": None, "max": None, "n": 0}
        vals = [s.get("score", 0) for s in scores if s.get("score") is not None]
        if not vals:
            return {"start": None, "end": None, "delta": None, "max": None, "n": 0}
        return {
            "start": vals[0],
            "end": vals[-1],
            "delta": round(vals[-1] - vals[0], 3),
            "max": max(vals),
            "n": len(vals),
        }

    # ── Persist ─────────────────────────────────────────────────────────────
    def write(self) -> str:
        ended_at = time.time()
        sides = self._split_by_side()
        left_outcome = self._outcome_for_side("left")
        right_outcome = self._outcome_for_side("right")

        body = {
            "session_id": self.session_id,
            "opp_id": self.opp_id,
            "scenario": self.scenario,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "duration_s": round(ended_at - self.started_at, 1),
            "outcomes": {
                "left": left_outcome,
                "right": right_outcome,
            },
            "persuasion": {
                "left": self._persuasion_delta(sides["left_scores"]),
                "right": self._persuasion_delta(sides["right_scores"]),
            },
            "msg_counts": {
                "left": len(sides["left_msgs"]),
                "right": len(sides["right_msgs"]),
            },
            "directives_used_right": [
                e.get("directive") for e in sides["right_msgs"]
                if e.get("role") == "agent" and e.get("directive")
            ],
            "errors": self.errors,
            "notes": self.notes,
            # full event stream for replay/debugging
            "events": self.events,
        }

        date_dir = datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d")
        out_dir = f"{LOGS_BASE}/{date_dir}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{self.session_id[:8]}_{self.opp_id[:8]}.json"
        with open(out_path, "w") as f:
            json.dump(body, f, indent=2, ensure_ascii=False, default=str)
        log.info("Session log written: %s", out_path)
        return out_path
