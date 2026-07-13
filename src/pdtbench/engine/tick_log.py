"""JSONL tick logger — the implementation of schemas/tick_log.md.

Buffers the calls made during a tick and flushes one record when the clock moves,
because a tick's record has to carry what the agent saw *before* it acted and the
fill that landed *after* it acted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import SCHEMA_VERSION
from .types import Fill, Tool, ToolError


class TickLogger:
    def __init__(self, path: Path, meta: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self.path = path

        self._calls: list[dict] = []
        self._seq = 0
        self._open_tick: dict | None = None

        self.n_calls = 0
        self.n_invalid = 0
        self.n_schema_errors = 0
        self.n_forced_waits = 0
        self.n_prose_nudges = 0
        self.n_read_cap_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._t0 = time.monotonic()

        self._write({"type": "meta", "schema_version": SCHEMA_VERSION, **meta})

    # --- plumbing -----------------------------------------------------------

    def _write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, separators=(",", ":"), default=_jsonable))
        self._fh.write("\n")

    # --- per-tick -----------------------------------------------------------

    def begin_tick(self, t: int, obs: dict, equity_cents: int, decision_point: bool = True,
                   skipped_by: int | None = None) -> None:
        self._open_tick = {
            "type": "tick",
            "t": t,
            "decision_point": decision_point,
            "obs": obs,
            "equity_cents": equity_cents,
        }
        if skipped_by is not None:
            self._open_tick["skipped_by"] = skipped_by
        self._calls = []
        self._seq = 0

    def record_call(
        self,
        tool: Tool | str,
        args: dict,
        ok: bool,
        advanced_time: bool,
        error: ToolError | None = None,
        latency_ms: float | None = None,
        tokens: dict | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "seq": self._seq,
            "tool": str(tool),
            "args": args,
            "ok": ok,
            "advanced_time": advanced_time,
        }
        if error is not None:
            rec["error"] = error.as_dict()
        if latency_ms is not None:
            rec["latency_ms"] = round(latency_ms, 1)
        if tokens:
            rec["tokens"] = tokens
            self.tokens_in += tokens.get("in", 0)
            self.tokens_out += tokens.get("out", 0)

        self._calls.append(rec)
        self._seq += 1

        self.n_calls += 1
        if not ok:
            self.n_invalid += 1
            if error is not None:
                from .types import SCHEMA_CODES, ErrorCode

                if error.code in SCHEMA_CODES:
                    self.n_schema_errors += 1
                if error.code == ErrorCode.READ_CAP_EXCEEDED:
                    self.n_read_cap_hits += 1

    def record_prose_nudge(self) -> None:
        self.n_prose_nudges += 1

    def amend_tick(self, equity_cents: int, obs_patch: dict) -> None:
        """Patch the open tick record. Used only by the terminal liquidation, which
        happens after tick 89's mark has already been taken and must replace it with
        the post-liquidation figure."""
        if self._open_tick is None:
            raise RuntimeError("amend_tick without an open tick")
        self._open_tick["equity_cents"] = equity_cents
        self._open_tick["obs"].update(obs_patch)

    def end_tick(self, action: dict | None, fill: Fill | None) -> None:
        if self._open_tick is None:
            raise RuntimeError("end_tick without begin_tick")
        if action is not None and action.get("forced"):
            self.n_forced_waits += 1

        rec = self._open_tick
        rec["calls"] = self._calls
        rec["action"] = action
        rec["fill"] = fill.as_dict() if fill is not None else None
        rec["invalid_count"] = sum(1 for c in self._calls if not c["ok"])
        rec["reads_count"] = sum(
            1 for c in self._calls if c["ok"] and not c["advanced_time"]
        )
        self._write(rec)
        self._open_tick = None
        self._calls = []

    # --- close --------------------------------------------------------------

    def close(
        self,
        t_final: int,
        terminal: dict,
        equity_series_cents: list[int],
        metrics: dict,
        memory_note_in: str | None = None,
        memory_note_out: str | None = None,
        cost: dict | None = None,
        status: str = "ok",
    ) -> dict:
        wallclock = time.monotonic() - self._t0
        reliability = {
            "n_calls": self.n_calls,
            "n_invalid": self.n_invalid,
            "invalid_rate": (self.n_invalid / self.n_calls) if self.n_calls else 0.0,
            "n_schema_errors": self.n_schema_errors,
            "n_forced_waits": self.n_forced_waits,
            "n_prose_nudges": self.n_prose_nudges,
            "n_read_cap_hits": self.n_read_cap_hits,
            "wallclock_s": round(wallclock, 1),
            "hit_wallclock_cap": status == "wallclock_capped",
        }
        rec = {
            "type": "episode_end",
            "t_final": t_final,
            "terminal": terminal,
            "equity_series_cents": equity_series_cents,
            "metrics": metrics,
            "reliability": reliability,
            "memory_note_in": memory_note_in,
            "memory_note_out": memory_note_out,
            "cost": cost or {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out},
            "status": status,
            "ended_at": _now(),
        }
        self._write(rec)
        self._fh.close()
        return rec


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jsonable(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")
