"""The no-look-ahead tripwire.

Every read of the price series goes through this wrapper, and every read is
checked against an explicit allowance. While the observation for tick `t` is
being built, the allowance is `t`: touching bar `t+1` raises. While a fill is
being priced, the allowance is `t+1`, because `open_{t+1}` *is* the fill price —
and nothing beyond it is reachable.

The point is that look-ahead becomes a structural impossibility rather than a
property someone has to keep remembering. A refactor that starts peeking at
tomorrow's close does not produce a subtly optimistic Sharpe six weeks from now;
it raises on the first tick, in the first test that runs.

Bars are addressed by *tick index*: -200..-1 for warmup, 0..89 for scored bars.
The storage offset is an implementation detail nothing outside this file sees.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


class LookAheadError(AssertionError):
    """A read reached past what the current allowance permits."""


class AuditedBars:
    """A read-tracking view over one window's presented bars."""

    def __init__(self, presented: pd.DataFrame, n_warmup: int, n_scored: int):
        if len(presented) != n_warmup + n_scored:
            raise ValueError(f"expected {n_warmup + n_scored} bars, got {len(presented)}")
        self._arr = np.ascontiguousarray(
            presented[list(FIELDS)].to_numpy(dtype=np.float64)
        )
        self._n_warmup = n_warmup
        self._n_scored = n_scored
        self._allowance: int | None = None
        self.max_index_read: int | None = None
        self.n_reads = 0

    # --- allowance ----------------------------------------------------------

    @contextmanager
    def allow(self, max_tick: int):
        """Permit reads up to and including `max_tick`, and nothing later."""
        prev = self._allowance
        self._allowance = max_tick
        try:
            yield self
        finally:
            self._allowance = prev

    def _check(self, tick: int) -> int:
        if tick < -self._n_warmup or tick >= self._n_scored:
            raise IndexError(f"tick {tick} outside window [-{self._n_warmup}, {self._n_scored - 1}]")
        if self._allowance is None:
            raise LookAheadError(
                f"read of tick {tick} outside any allow() scope -- every price read "
                "must declare what it is allowed to see"
            )
        if tick > self._allowance:
            raise LookAheadError(
                f"look-ahead: read of tick {tick} while only {self._allowance} is visible"
            )
        self.n_reads += 1
        self.max_index_read = tick if self.max_index_read is None else max(self.max_index_read, tick)
        return tick + self._n_warmup

    # --- reads --------------------------------------------------------------

    def field(self, tick: int, name: str) -> float:
        return float(self._arr[self._check(tick), FIELDS.index(name)])

    def open(self, tick: int) -> float:
        return self.field(tick, "open")

    def close(self, tick: int) -> float:
        return self.field(tick, "close")

    def bar(self, tick: int) -> dict[str, float]:
        row = self._arr[self._check(tick)]
        return {k: float(v) for k, v in zip(FIELDS, row)}

    def closes_through(self, tick: int, lookback: int) -> np.ndarray:
        """Closes over the `lookback` bars ending at `tick`, oldest first."""
        self._check(tick)
        lo = max(tick - lookback + 1, -self._n_warmup)
        self._check(lo)
        return self._arr[lo + self._n_warmup : tick + self._n_warmup + 1, FIELDS.index("close")]

    def hi_lo_through(self, tick: int, from_tick: int) -> tuple[float, float]:
        """Highest high and lowest low over [from_tick, tick]."""
        self._check(tick)
        self._check(from_tick)
        a = self._arr[from_tick + self._n_warmup : tick + self._n_warmup + 1]
        return float(a[:, FIELDS.index("high")].max()), float(a[:, FIELDS.index("low")].min())

    def bars_through(self, tick: int, lookback: int) -> list[dict]:
        """The `lookback` bars ending at `tick`, oldest first, tagged with tick index."""
        self._check(tick)
        lo = max(tick - lookback + 1, -self._n_warmup)
        self._check(lo)
        out = []
        for i in range(lo, tick + 1):
            row = self._arr[i + self._n_warmup]
            out.append({"t": i, **{k: float(v) for k, v in zip(FIELDS, row)}})
        return out

    # --- unaudited, for the harness only ------------------------------------

    def raw_close_series(self) -> np.ndarray:
        """The full close series, bypassing the audit.

        For the *harness* — metrics, replay verification, the vol-floor reference.
        Never reachable from anything the agent can call. Kept deliberately ugly
        so it is obvious in review when something reaches for it.
        """
        return self._arr[:, FIELDS.index("close")].copy()

    def raw_open(self, tick: int) -> float:
        return float(self._arr[tick + self._n_warmup, FIELDS.index("open")])
