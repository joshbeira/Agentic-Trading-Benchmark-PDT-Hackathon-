"""The agent runner -- driven entirely by a fake client. No network.

The runner is the only thing that can know what an episode cost, so these tests are
where that arithmetic is pinned. Everything else in the pipeline recomputes its numbers
from the logs; `usd` is the one figure that cannot be, unless the log carries every
bucket that went into it.
"""

from __future__ import annotations

import pytest

from pdtbench.agent import cost as C


def test_usd_prices_each_bucket_at_its_own_rate():
    """A million tokens through each bucket, so the arithmetic is readable:
    5.00 input + 0.50 cache read (0.1x) + 6.25 cache write (1.25x) + 25.00 output."""
    u = C.Usage(tokens_in=1_000_000, tokens_out=1_000_000,
                cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert u.usd("claude-opus-4-8") == pytest.approx(36.75)


def test_a_cached_run_costs_a_tenth_of_an_uncached_one():
    """The whole budget rests on this. If cache reads ever priced at par, the run costs
    ~6x D5's ceiling and nothing else in the suite would notice."""
    uncached = C.Usage(tokens_in=1_000_000)
    cached = C.Usage(cache_read_tokens=1_000_000)
    assert cached.usd("claude-opus-4-8") == pytest.approx(uncached.usd("claude-opus-4-8") / 10)


def test_the_cost_block_can_be_recomputed_from_what_it_logs():
    """`usd` must be regenerable from the log alone -- the claim the replay verifier
    exists to defend. If a bucket is missing from the block, it is not."""
    u = C.Usage(tokens_in=1234, tokens_out=567,
                cache_read_tokens=89_012, cache_write_tokens=3_456)
    block = u.as_cost_block("claude-opus-4-8")
    again = C.Usage(
        tokens_in=block["tokens_in"], tokens_out=block["tokens_out"],
        cache_read_tokens=block["cache_read_tokens"],
        cache_write_tokens=block["cache_write_tokens"],
    )
    assert again.usd("claude-opus-4-8") == pytest.approx(block["usd"], abs=1e-6)


def test_add_accumulates_an_api_usage_object():
    class _U:  # the shape the SDK returns
        input_tokens, output_tokens = 10, 20
        cache_read_input_tokens, cache_creation_input_tokens = 30, 40

    u = C.Usage()
    u.add(_U()).add(_U())
    assert (u.tokens_in, u.tokens_out, u.cache_read_tokens, u.cache_write_tokens) == (20, 40, 60, 80)


def test_add_tolerates_a_usage_without_cache_fields():
    """An uncached response may omit them entirely; a None must not poison the sum."""
    class _U:
        input_tokens, output_tokens = 10, 20
        cache_read_input_tokens = None
        cache_creation_input_tokens = None

    u = C.Usage().add(_U())
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_an_unknown_model_is_refused_rather_than_priced_at_zero():
    """A silent 0.0 would look like a free run."""
    with pytest.raises(KeyError):
        C.Usage(tokens_in=1).usd("claude-not-a-model")
