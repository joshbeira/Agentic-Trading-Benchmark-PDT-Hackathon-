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


def test_a_hallucinated_tool_name_is_a_result_not_an_exception(windows_dir, tmp_path):
    """FastMCP validates the tool name before the engine ever sees the call and raises
    `ToolError` for an unknown one -- `Wait(n=99)` above never exercises that path, it is
    schema-valid and reaches the engine directly. Across 120 episodes Opus will eventually
    hallucinate a tool name, and today that would kill the episode instead of teaching it.

    The result has to match what the direct path would have produced, and the engine has
    to have counted it -- an uncounted invalid call can never advance the
    `max_consecutive_invalid` ladder."""
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        out = surface.call("Purchase", {})
    assert out["ok"] is False
    assert out["error"]["code"] == "UNKNOWN_TOOL"
    assert session.env.consecutive_invalid == 1


def test_a_schema_invalid_argument_type_is_a_result_not_an_exception(windows_dir, tmp_path):
    """FastMCP validates the argument schema before the engine ever sees the call and
    raises `ToolError` for a mistyped argument -- `n="two"` fails FastMCP's own pydantic
    coercion, unlike `n=99` above, which is schema-valid and merely domain-invalid."""
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        out = surface.call("Wait", {"n": "two"})
    assert out["ok"] is False
    assert out["error"]["code"] == "SCHEMA_ERROR"
    assert session.env.consecutive_invalid == 1


def test_the_invalid_ladder_still_fires_through_the_served_surface(windows_dir, tmp_path):
    """A test that only checks the return value could pass while the fall-through quietly
    swallowed the exception and returned a look-alike payload without ever reaching the
    engine. Proving the ladder still fires -- three consecutive invalid calls forcing a
    Wait, exactly as on the direct path -- proves the engine is the one that saw and
    counted every one of them."""
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        first = surface.call("Purchase", {})            # UNKNOWN_TOOL, invalid #1
        second = surface.call("Purchase", {})            # UNKNOWN_TOOL, invalid #2
        third = surface.call("Wait", {"n": "two"})        # SCHEMA_ERROR, invalid #3 -> ladder

    assert first.get("forced_wait") is not True
    assert second.get("forced_wait") is not True
    assert third["ok"] is False
    assert third["forced_wait"] is True
    assert "consecutive invalid calls" in third["note"]
    assert session.env.consecutive_invalid == 0  # the ladder reset it on firing

    session.finish()
    from pdtbench.engine.replay import load

    _meta, ticks, _end = load(session.env.logger.path)
    assert ticks[0]["action"]["forced"] is True
    assert ticks[0]["action"]["forced_reason"] == "max_consecutive_invalid"
    assert ticks[0]["invalid_count"] == 3


from pdtbench.agent import loop as L
from pdtbench.engine.replay import load, replay
from pdtbench.schema import validate_episode


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    input_tokens, output_tokens = 100, 20
    cache_read_input_tokens, cache_creation_input_tokens = 500, 50


class _Resp:
    def __init__(self, content, stop_reason="tool_use"):
        self.content, self.stop_reason, self.usage = content, stop_reason, _Usage()


class _FakeClient:
    """Returns scripted completions. Records every request for inspection."""

    def __init__(self, script):
        self._script, self.requests = list(script), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        return self._script.pop(0) if self._script else _wait_forever()


def _tool_use(name, args):
    return _Resp([_Block("tool_use", name=name, input=args, id="tu_1")])


def _wait_forever():
    return _tool_use("Wait", {"n": 10})


def _prose(text="I think I should probably wait here."):
    return _Resp([_Block("text", text=text)], stop_reason="end_turn")


class _ExplodingClient(_FakeClient):
    """Serves `after` completions, then raises the way the wire does."""

    def __init__(self, script, after):
        super().__init__(script)
        self._after = after

    def create(self, **kw):
        if len(self.requests) >= self._after:
            raise RuntimeError("the API went away mid-episode")
        return super().create(**kw)


def test_one_completion_is_one_call_is_one_tick(windows_dir, tmp_path):
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_tool_use("Buy", {"fraction": 1.0})])
    L.run_episode(session, client)
    session.finish()

    _meta, ticks, _end = load(session.env.logger.path)
    assert len(ticks) == 90
    assert ticks[0]["action"]["tool"] == "Buy"
    assert len(ticks[0]["calls"]) == 1
    assert ticks[0]["calls"][0]["tokens"] == {"in": 100, "out": 20}
    assert ticks[0]["calls"][0]["latency_ms"] >= 0


def test_every_request_disables_parallel_tool_use(windows_dir, tmp_path):
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([])
    L.run_episode(session, client)
    assert client.requests
    for req in client.requests:
        assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert "temperature" not in req  # rejected with a 400 on Opus 4.8
        assert req["thinking"] == {"type": "adaptive"}


def test_a_prose_reply_is_nudged_once_then_the_turn_is_taken_away(windows_dir, tmp_path):
    """D13. The engine cannot see a reply that made no tool call, so the runner reports
    it -- and the log says so, rather than crediting the agent with a Wait it chose."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_prose(), _prose()])
    L.run_episode(session, client)
    session.finish()

    _meta, ticks, end = load(session.env.logger.path)
    assert ticks[0]["action"]["forced"] is True
    assert ticks[0]["action"]["forced_reason"] == "prose_stall"
    assert end["reliability"]["n_prose_nudges"] == 2
    assert end["reliability"]["n_forced_waits"] >= 1


def test_a_nudged_model_that_recovers_is_not_forced(windows_dir, tmp_path):
    """One nudge, then a tool call. The nudge is counted; the turn is not taken away."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_prose(), _tool_use("Buy", {"fraction": 1.0})])
    L.run_episode(session, client)
    session.finish()

    _meta, ticks, end = load(session.env.logger.path)
    assert ticks[0]["action"]["tool"] == "Buy"
    assert ticks[0]["action"]["forced"] is False
    assert end["reliability"]["n_prose_nudges"] == 1


def test_a_refusal_and_a_truncation_both_land_in_the_prose_ladder(windows_dir, tmp_path):
    """Neither carries a tool call, so neither can advance the clock on its own.

    A refusal can decline before emitting any output, so `content == []` is a real
    response shape -- and echoing it back is a 400 ("all messages must have non-empty
    content except for the optional final assistant message"), which kills the episode
    instead of nudging it. So this asserts on what actually went out: no turn may carry
    empty content. Reading the ladder out of the log alone cannot catch it -- the fake
    client never 400s, so the log looks correct while the real wire would have died.

    Two consecutive `user` turns are expected here and deliberately not asserted against:
    with no reply to echo, the honest conversation is the nudge followed by the forced
    turn. The API merges them, and no assistant reply is lost to the merge. Inventing
    placeholder assistant content to keep roles alternating would put words in the
    model's mouth it never emitted."""
    for stop in ("refusal", "max_tokens"):
        session = _session(windows_dir, tmp_path / stop)
        client = _FakeClient([_Resp([], stop_reason=stop), _Resp([], stop_reason=stop)])
        L.run_episode(session, client)
        session.finish()
        _meta, ticks, _end = load(session.env.logger.path)
        assert ticks[0]["action"]["forced_reason"] == "prose_stall", stop

        # Both empty completions are consumed by request[2]; the ladder fired by then.
        assert len(client.requests) >= 3, (stop, len(client.requests))
        for i, req in enumerate(client.requests):
            for m in req["messages"]:
                assert m["content"], (stop, i, m["role"], req["messages"])


def test_the_usage_of_a_turn_that_made_no_tool_call_is_still_billed(windows_dir, tmp_path):
    """A prose reply costs money. Dropping it would understate the run."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_prose(), _tool_use("Wait", {"n": 10})])
    usage = L.run_episode(session, client)
    assert usage.tokens_out >= 20 * len(client.requests)


def test_the_nudge_arrives_in_a_conversation_that_contains_the_turn_it_nudges_about(
    windows_dir, tmp_path
):
    """The nudge says "You replied without making a tool call" -- so that reply has to be
    on the wire, or the sentence refers to nothing. Dropping the assistant turn also puts
    two `user` turns back to back, which the API silently merges rather than rejecting:
    no error, just a nudge about a reply the model cannot see. The existing prose tests
    cannot catch either, because the fake client ignores `messages` entirely."""
    session = _session(windows_dir, tmp_path)
    reply = _prose()
    client = _FakeClient([reply, _tool_use("Wait", {"n": 1})])
    L.run_episode(session, client)

    sent = client.requests[1]["messages"]
    roles = [m["role"] for m in sent]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    # Identity, not equality -- the echo-unchanged constraint is about *these* blocks.
    assert any(m["role"] == "assistant" and m["content"] is reply.content for m in sent), roles


def test_the_post_force_turn_names_the_real_tick_and_keeps_the_history(windows_dir, tmp_path):
    """R3 is "one nudge, then a forced Wait" -- not "start the episode over". Re-sending
    `_FIRST_TURN` would tell a model at tick 6 it is at tick 0 on its first observation,
    and wiping `messages` would throw away every prior turn."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_tool_use("Wait", {"n": 5}), _prose(), _prose()])
    L.run_episode(session, client)

    # Wait(5) took tick 0 -> 5; the forced Wait(1) took it to 6. The rolling cache
    # breakpoint wraps the newest turn's string content, so read the text back out.
    sent = client.requests[3]["messages"]
    assert sent[-1]["content"][0]["text"] == L._forced_turn(6)
    assert "tick 6 of 89" in sent[-1]["content"][0]["text"]
    # The history stays, and _FIRST_TURN is not re-sent mid-episode.
    assert sent[0]["content"] == L._FIRST_TURN
    assert [m["content"] for m in sent[1:]].count(L._FIRST_TURN) == 0
    # first_turn, assistant, tool_result, prose, nudge, prose, forced_turn.
    assert len(sent) == 7, [m["role"] for m in sent]


def test_a_caller_owned_usage_holds_what_a_raised_episode_spent(windows_dir, tmp_path):
    """Task 10 seals a failed episode's log with the partial cost. Every raise out of
    `run_episode` -- the max_calls guard, an APIStatusError off the wire -- skips the
    return, so a usage the runner owns privately dies with the episode and the tokens
    already billed are unrecoverable."""
    session = _session(windows_dir, tmp_path)
    client = _ExplodingClient([_tool_use("Wait", {"n": 1})] * 2, after=2)
    usage = C.Usage()

    with pytest.raises(RuntimeError):
        L.run_episode(session, client, usage=usage)

    assert len(client.requests) == 2  # two completions were billed before the raise
    assert (usage.tokens_in, usage.tokens_out) == (200, 40)


def test_a_given_usage_is_accumulated_into_and_is_the_object_returned(windows_dir, tmp_path):
    """Accumulated into, not replaced -- the caller's handle must stay live."""
    session = _session(windows_dir, tmp_path)
    mine = C.Usage(tokens_out=7)
    client = _FakeClient([])

    returned = L.run_episode(session, client, usage=mine)

    assert returned is mine
    assert mine.tokens_out == 7 + 20 * len(client.requests)


def test_thinking_blocks_are_echoed_back_unchanged(windows_dir, tmp_path):
    """Required when continuing on the same model. Dropping or editing them breaks the
    turn."""
    session = _session(windows_dir, tmp_path)
    think = _Block("thinking", thinking="hmm")
    client = _FakeClient([
        _Resp([think, _Block("tool_use", name="Wait", input={"n": 1}, id="tu_1")]),
    ])
    L.run_episode(session, client)

    second = client.requests[1]
    assistant = [m for m in second["messages"] if m["role"] == "assistant"][0]
    assert assistant["content"][0] is think


def test_the_memory_arms_send_a_byte_identical_frozen_block(windows_dir, tmp_path):
    """D13, at the wire. If these ever differ the paired comparison is not paired."""
    a = _session(windows_dir, tmp_path / "a", memory="rolling_note")
    ca = _FakeClient([])
    L.run_episode(a, ca, note="w00 was a chop window; I overtraded it.")

    b = _session(windows_dir, tmp_path / "b", memory="none")
    cb = _FakeClient([])
    L.run_episode(b, cb, note=None)

    sa, sb = ca.requests[0]["system"], cb.requests[0]["system"]
    assert sa[0] == sb[0]          # frozen block byte-identical
    assert len(sa) == 2 and len(sb) == 1
    assert "overtraded" in sa[1]["text"]


def test_an_episode_driven_by_a_fake_model_still_replays(windows_dir, tmp_path):
    """The runner cannot produce a log the rest of the pipeline rejects."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([
        _tool_use("fetchData", {"lookback": 50}),
        _tool_use("Buy", {"fraction": 0.5}),
        _tool_use("getStats", {}),
        _tool_use("Sell", {"fraction": 1.0}),
    ])
    usage = L.run_episode(session, client)
    session.finish(cost=usage.as_cost_block(L.MODEL))

    log = session.env.logger.path
    assert validate_episode(log).ok, validate_episode(log).errors[:3]
    res = replay(log, windows_dir)
    assert res.ok, res.failures[:3]
