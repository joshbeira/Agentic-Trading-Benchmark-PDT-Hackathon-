"""The execution-reliability scoreboard — deliberately a separate table.

Trading skill and interface competence are different abilities, and merging them would
make both unreadable. A model that trades well but fumbles a JSON schema should show up
as exactly that: a good trader with a high invalid-call rate. If malformed calls cost P&L,
a tool-calling stumble would silently contaminate the trading result and we would end up
ranking models on their JSON hygiene while claiming to rank them on their trading.

So: malformed calls carry **no direct P&L penalty**. The structured error plus a retry is
the feedback. The only consequence is lost time — the clock does not move on an invalid
call, and an agent that burns its reads gets the turn taken away — and every bit of that
is counted here (D13).

The `n_calls` denominator counts *every* call the agent made, valid and invalid, which is
what makes `invalid_rate` comparable across agents that differ wildly in how much they
poke at the interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .loader import Episode, Run

ERROR_CODES = (
    "SCHEMA_ERROR", "UNKNOWN_TOOL", "NONPOSITIVE_QTY", "NAN_QTY",
    "INSUFFICIENT_CASH", "INSUFFICIENT_SHARES", "WAIT_OUT_OF_RANGE",
    "READ_CAP_EXCEEDED", "EPISODE_OVER",
)


@dataclass
class ReliabilityRow:
    agent_id: str
    kind: str
    n_episodes: int
    n_calls: int
    n_invalid: int
    invalid_rate: float
    schema_error_rate: float
    forced_waits_per_episode: float
    prose_nudges_per_episode: float
    read_cap_hits_per_episode: float
    median_wallclock_s: float
    n_capped: int
    n_agent_errors: int
    error_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return self.n_invalid == 0 and self.n_capped == 0 and self.n_agent_errors == 0


def _row(agent_id: str, kind: str, eps: list[Episode]) -> ReliabilityRow:
    # Counted from the `calls` array rather than the cached reliability block: `calls` is
    # the primary record, and a mismatch between them is a bug we want to see.
    n_calls = sum(len(e.calls) for e in eps)
    invalid = [c for e in eps for c in e.calls if not c.get("ok", True)]
    codes = Counter(c["error"]["code"] for c in invalid if c.get("error"))

    def per_ep(key: str) -> float:
        return float(np.mean([e.reliability.get(key, 0) for e in eps])) if eps else 0.0

    schema_errors = codes["SCHEMA_ERROR"] + codes["UNKNOWN_TOOL"]
    return ReliabilityRow(
        agent_id=agent_id,
        kind=kind,
        n_episodes=len(eps),
        n_calls=n_calls,
        n_invalid=len(invalid),
        invalid_rate=len(invalid) / n_calls if n_calls else 0.0,
        schema_error_rate=schema_errors / n_calls if n_calls else 0.0,
        forced_waits_per_episode=per_ep("n_forced_waits"),
        prose_nudges_per_episode=per_ep("n_prose_nudges"),
        read_cap_hits_per_episode=per_ep("n_read_cap_hits"),
        median_wallclock_s=(
            float(np.median([e.reliability.get("wallclock_s", 0) for e in eps])) if eps else 0.0
        ),
        n_capped=sum(1 for e in eps if e.status == "wallclock_capped"),
        n_agent_errors=sum(1 for e in eps if e.status == "agent_error"),
        error_breakdown=dict(codes),
    )


def build(run: Run) -> list[ReliabilityRow]:
    """Every episode, including the ones the trading scoreboard excluded. A crashed
    episode is invisible on the trading board by design (a provider failure is not a
    trading result) — this is where it has to show up instead."""
    rows = []
    for a in run.agents:
        eps = run.select(agent_id=a, scorable_only=False)
        if eps:
            rows.append(_row(a, run.agent_kind(a), eps))
    return sorted(rows, key=lambda r: r.invalid_rate)


def render(rows: list[ReliabilityRow]) -> str:
    out = ["EXECUTION RELIABILITY", "=====================", ""]
    out.append(
        f"  {'agent':<14} {'kind':<9} {'calls':>7} {'invalid':>8} {'schema':>7} "
        f"{'forced W':>9} {'prose':>6} {'readcap':>8} {'wall(s)':>8} {'capped':>7} {'errors':>7}"
    )
    out.append("  " + "-" * 108)
    for r in rows:
        out.append(
            f"  {r.agent_id:<14} {r.kind:<9} {r.n_calls:>7,} {r.invalid_rate:>7.1%} "
            f"{r.schema_error_rate:>6.1%} {r.forced_waits_per_episode:>9.2f} "
            f"{r.prose_nudges_per_episode:>6.2f} {r.read_cap_hits_per_episode:>8.2f} "
            f"{r.median_wallclock_s:>8.0f} {r.n_capped:>7} {r.n_agent_errors:>7}"
        )

    breakdown = Counter()
    for r in rows:
        breakdown.update(r.error_breakdown)
    if breakdown:
        out.append("")
        out.append("  error codes: " + ", ".join(
            f"{c}={n}" for c, n in breakdown.most_common()
        ))

    out.append("")
    out.append("  No P&L penalty attaches to any of this. A malformed call returns a")
    out.append("  structured error and the clock does not move; the only cost is a lost")
    out.append("  decision. Fumbling the interface never contaminates the trading result.")
    return "\n".join(out)
