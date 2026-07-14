"""The agent runner -- driven entirely by a fake client. No network.

The runner is the only thing that can know what an episode cost, so these tests are
where that arithmetic is pinned. Everything else in the pipeline recomputes its numbers
from the logs; `usd` is the one figure that cannot be, unless the log carries every
bucket that went into it.
"""

from __future__ import annotations

import pytest

from pdtbench.agent import cost as C
from pdtbench.agent import prompt as P
from pdtbench.agent.tools import TOOL_CHOICE, ServedSurface
from pdtbench.mcp import BASELINES, TOOLS, EpisodeSession


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


def test_the_prompt_discloses_everything_d13_says_it_must():
    """Hiding the metric would test goal inference -- a different experiment."""
    text = P.SYSTEM_PROMPT
    for required in (
        "vol-floored Sharpe",     # the ranking metric (D2)
        "10 bps",                 # the fee schedule (D10)
        "open of the next bar",   # the fill rule (D3)
        "$10,000",                # initial capital (D13)
        "90",                     # the episode length
    ):
        assert required.lower() in text.lower(), required


def test_the_prompt_does_not_lie_about_the_environment():
    """The prompt states the config in English and cannot interpolate it -- that would
    break the cache prefix. So the drift is caught here instead."""
    from pdtbench.config import DEFAULT as D

    assert ("$10,000" in P.SYSTEM_PROMPT) == (D.initial_capital_cents == 1_000_000)
    assert ("10 bps" in P.SYSTEM_PROMPT) == (D.fee_bps + D.slippage_bps == 10)
    assert ("at most 8" in P.SYSTEM_PROMPT) == (D.max_reads_per_tick == 8)
    assert ("Three consecutive" in P.SYSTEM_PROMPT) == (D.max_consecutive_invalid == 3)
    assert ("up to 10 bars" in P.SYSTEM_PROMPT) == (D.max_wait == 10)
    assert ("ticks 0 through 89" in P.SYSTEM_PROMPT) == (D.n_scored == 90)
    # Two separate claims sit in one sentence, and they answer to two different fields.
    # "200 further bars ... precede tick 0" is the warmup the agent can see; "a 200-bar
    # lookback is honorable immediately" is what fetchData will actually serve. Binding
    # both to fetch_lookback_cap would let n_warmup drift while the guard stayed green.
    assert ("200 further bars" in P.SYSTEM_PROMPT) == (D.n_warmup == 200)
    assert ("200-bar lookback" in P.SYSTEM_PROMPT) == (D.fetch_lookback_cap >= 200)
    assert ("0.25 * bh_daily_vol" in P.SYSTEM_PROMPT) == (D.vol_floor_multiple == 0.25)


def test_the_prompt_carries_no_invalidator():
    """Caching is a prefix match and this text is the prefix. A date, a uuid or an
    interpolated id here would silently cost ~6x -- no error, just a bill.

    The sha is the real assertion: it is taken over the text the runner actually sends,
    so it moves if anything varying creeps in."""
    import datetime
    import re

    assert isinstance(P.SYSTEM_PROMPT, str)  # a constant, not a factory
    assert str(datetime.date.today().year) not in P.SYSTEM_PROMPT
    assert not re.search(r"\{[a-z_]+\}", P.SYSTEM_PROMPT)  # no unformatted placeholder
    assert P.system_prompt_sha256() == (
        "b0586a912db9c5221e6214161bbb20f90c61e1d36b32e8437869211bdf353720"
    ), "the frozen prompt changed -- this is the experiment's identity; update deliberately"


def test_the_memory_arms_share_a_byte_identical_frozen_block():
    """D13. The note must sit in its own block *after* the frozen one, or the two arms
    are not running the same experiment."""
    with_note = P.system_blocks("remember: w13 was choppy")
    without = P.system_blocks(None)

    assert without == [with_note[0]]
    assert len(with_note) == 2
    assert with_note[0]["text"] == P.SYSTEM_PROMPT
    assert with_note[0]["cache_control"] == {"type": "ephemeral"}
    assert "w13 was choppy" in with_note[1]["text"]
    assert "cache_control" not in with_note[1]


def test_an_empty_note_is_not_a_block():
    """Episode 0 of a memory lane has no incoming note; it must look exactly like the
    no-memory arm, not like an arm carrying an empty note."""
    assert P.system_blocks("") == P.system_blocks(None)


def _session(windows_dir, tmp_path, window_id="w18", memory="none"):
    return EpisodeSession.start(
        window_id=window_id, track="real",
        agent={"id": "opus_test", "kind": "llm", "memory": memory},
        episode_index=0, run_dir=tmp_path, run_id="test", windows_dir=windows_dir,
    )


def test_the_model_is_offered_exactly_the_engines_tools(windows_dir, tmp_path):
    with ServedSurface(_session(windows_dir, tmp_path)) as surface:
        defs = surface.tools()
    assert sorted(d["name"] for d in defs) == sorted(TOOLS)
    for d in defs:
        assert d["description"], d["name"]
        assert d["input_schema"]["type"] == "object"


def test_the_offered_tools_state_the_fee_schedule(windows_dir, tmp_path):
    """The descriptions reach the model verbatim. This is the surface D13's disclosure
    travels on, alongside the system prompt."""
    with ServedSurface(_session(windows_dir, tmp_path)) as surface:
        by_name = {d["name"]: d["description"] for d in surface.tools()}
    for name in ("Buy", "Sell"):
        assert "10 bps per side" in by_name[name]
        assert "{fill}" not in by_name[name]


def test_parallel_tool_use_is_disabled():
    """Not a preference. Parallel tool use is on by default, so one completion could emit
    Buy and Wait together, the engine would advance twice from one decision, and both
    `action` (one per tick) and `calls[].tokens` ("one completion is one call") would
    become ambiguous."""
    assert TOOL_CHOICE == {"type": "auto", "disable_parallel_tool_use": True}


def test_a_call_through_the_surface_returns_the_agent_payload(windows_dir, tmp_path):
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        out = surface.call("Wait", {"n": 2})
    assert out["ok"] is True
    assert out["observation"]["tick"] == 2


def test_an_invalid_call_is_a_result_not_an_exception(windows_dir, tmp_path):
    """The agent has to see the structured error to correct itself, and the reliability
    scoreboard has to count it."""
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        out = surface.call("Wait", {"n": 99})
    assert out["ok"] is False
    assert out["error"]["code"] == "WAIT_OUT_OF_RANGE"
