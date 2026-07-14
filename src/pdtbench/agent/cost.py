"""What an episode cost.

The engine cannot answer this: it has no price table, and the API splits input across
three buckets that price differently. That is why `close_log(cost=)` exists and why the
runner owns it.

The arithmetic is here rather than inline in the loop because `cost.usd` in the tick log
has to be *auditable*: every number on the scoreboard is regenerable from the logs, and
`usd` is only regenerable if the log carries all four buckets and the price table is a
declared constant rather than a literal buried in a call site.
"""

from __future__ import annotations

from dataclasses import dataclass

#: USD per million tokens, by model id. From the published price list; stamped into the
#: run manifest so the analysis reads prices from the run rather than from whatever this
#: table says on the day someone re-runs it.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}

#: A cache read costs a tenth of base input. This multiplier is the entire budget: at par
#: the 120-episode run costs ~6x D5's ceiling.
CACHE_READ_MULTIPLIER = 0.10
#: A 5-minute-TTL cache write costs 1.25x base input.
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    """Token counts accumulated over an episode, in the four buckets that price apart."""

    tokens_in: int = 0  # uncached input only -- NOT the prompt size
    tokens_out: int = 0  # includes thinking tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage) -> "Usage":
        """Accumulate one API response's `usage`.

        The cache fields are absent or None on an uncached response, which must not
        poison the sum -- an episode's first turn always looks like that.
        """
        self.tokens_in += usage.input_tokens or 0
        self.tokens_out += usage.output_tokens or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        return self

    def usd(self, model: str) -> float:
        """Cost in dollars. Raises on an unknown model rather than pricing it at zero --
        a silent 0.0 would look like a free run."""
        p = PRICES[model]
        return (
            self.tokens_in * p["input"]
            + self.cache_read_tokens * p["input"] * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * p["input"] * CACHE_WRITE_MULTIPLIER
            + self.tokens_out * p["output"]
        ) / 1_000_000

    def as_cost_block(self, model: str) -> dict:
        """The `episode_end.cost` block (schema v1.3.0)."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "usd": round(self.usd(model), 6),
        }
