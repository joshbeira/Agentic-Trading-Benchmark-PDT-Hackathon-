"""The MCP layer: one door into the engine, and the baselines that walk through it.

D14 says the baselines execute "through the same engine API as scripted clients, so fee
parity holds by construction rather than by assertion". Construction is only load-bearing
if something proves the construction is what the code actually does, and this file is that
something: `test_a_baseline_is_logged_identically_...` drives the *same* baseline down the
MCP transport and down the in-process path and demands the two logs agree byte for byte.
If a second path to a fill ever opens — a baseline shortcut, a cheaper execution, a read
that skips the cap — those bytes diverge.

That test is not decoration. The baselines cannot learn, so their per-episode Sharpe *is*
the window-difficulty signal, and every learning curve in the deck is an LLM's trace minus
theirs (D11). A baseline that executed even slightly cheaper than a model would bias every
one of those curves, in the flattering direction, and nothing else in the suite would
notice.

The rest is the Verification section's "baseline sanity" for step 4, one test per clause:
flat scores exactly 0, buy-and-hold pays exactly two frictions, the 10/50 crossover's
trades match hand-computed signals, and every baseline pays the engine's friction.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pdtbench.config import DEFAULT
from pdtbench.data.windows import load_episode
from pdtbench.engine.replay import load, replay
from pdtbench.mcp import BASELINES, TOOLS, EpisodeSession, drive, write_run_manifest
from pdtbench.mcp.server import build_server, session_from_env
from pdtbench.schema import validate_episode

#: One window, fixed, for the tests that hand-compute an expected answer. w18 is chosen,
#: not arbitrary: the crossover changes state seven times on it and is still long at the
#: bell, so a hand-check gets round trips *and* the terminal liquidation. A window it
#: barely trades would let the same test pass while proving almost nothing.
KNOWN = "w18"


# --------------------------------------------------------------------- harness


class _ViaMCP:
    """A session facade that routes every call through the real MCP tool surface.

    Same object underneath — the point is not to fake a second engine, it is to make the
    calls take the long way round (tool dispatch, argument binding, JSON serialization)
    and then show that nothing about the episode changed.
    """

    def __init__(self, session: EpisodeSession):
        self._session = session
        self._server = build_server(session)
        self._loop = asyncio.new_event_loop()

    @property
    def obs(self) -> dict:
        return self._session.obs

    @property
    def done(self) -> bool:
        return self._session.done

    def call(self, tool: str, args: dict | None = None) -> dict:
        blocks = self._loop.run_until_complete(self._server.call_tool(tool, args or {}))
        return json.loads(blocks[0].text)

    def close(self) -> None:
        self._loop.close()


class _Recorder:
    """Wraps a session and keeps every payload the baseline was handed.

    The logs agreeing is necessary but not sufficient: a transport that mangled an
    observation on the way back to the agent would still log identically, because the log
    is written engine-side. Comparing what the *agent saw* closes that gap.
    """

    def __init__(self, inner):
        self._inner = inner
        self.payloads: list[dict] = []

    @property
    def obs(self) -> dict:
        return self._inner.obs

    @property
    def done(self) -> bool:
        return self._inner.done

    def call(self, tool: str, args: dict | None = None) -> dict:
        payload = self._inner.call(tool, args)
        self.payloads.append(payload)
        return payload


def _drive(
    name: str,
    windows_dir: Path,
    run_dir: Path,
    *,
    window_id: str = KNOWN,
    track: str = "real",
    cfg=DEFAULT,
    via_mcp: bool = False,
) -> tuple[EpisodeSession, Path, list[dict]]:
    """Run one baseline to the terminal bar. Returns the session, its log, and every
    payload the baseline received."""
    baseline = BASELINES[name]()
    baseline.on_episode(track, window_id)
    session = EpisodeSession.start(
        window_id=window_id,
        track=track,
        agent=baseline.agent(),
        episode_index=0,
        run_dir=run_dir,
        run_id="test",
        cfg=cfg,
        windows_dir=windows_dir,
    )
    transport = _ViaMCP(session) if via_mcp else None
    recorder = _Recorder(transport or session)
    try:
        drive(recorder, baseline)
        session.finish()
    finally:
        if transport is not None:
            transport.close()
    # The logger names its own file; asking it beats restating the layout rule here.
    return session, session.env.logger.path, recorder.payloads


_VOLATILE = "<volatile>"


def _canonical_log(path: Path) -> str:
    """The log's bytes, with the three fields that *cannot* match blanked: two wall-clock
    timestamps and a duration. Everything else — every call, every argument, every fill,
    every equity mark, the metrics and the reliability counts — has to agree exactly."""
    out = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("type") == "meta":
            rec["started_at"] = _VOLATILE
        elif rec.get("type") == "episode_end":
            rec["ended_at"] = _VOLATILE
            rec["reliability"]["wallclock_s"] = _VOLATILE
        out.append(json.dumps(rec, separators=(",", ":")))
    return "\n".join(out)


def _fills(log: Path) -> list[dict]:
    _meta, ticks, _end = load(log)
    return [t["fill"] for t in ticks if t["fill"]]


# ------------------------------------------------------------- the tool surface


def test_the_served_tools_are_exactly_the_engines_tool_contract(windows_dir, tmp_path):
    """A tool added to the engine and forgotten here is invisible to a model, and a tool
    served that the engine does not know is an UNKNOWN_TOOL the moment it is called."""
    session = EpisodeSession.start(
        window_id=KNOWN, track="real", agent=BASELINES["flat"]().agent(),
        episode_index=0, run_dir=tmp_path, windows_dir=windows_dir,
    )
    served = [t.name for t in asyncio.run(build_server(session).list_tools())]

    assert sorted(served) == sorted(TOOLS)
    assert len(served) == len(set(served)), "a tool was served twice"


def test_every_served_tool_states_the_rules_it_is_bound_by(windows_dir, tmp_path):
    """D13 discloses the fill rule and the fee schedule to the agent. Buy and Sell say so
    in the tool surface itself, and they interpolate `_FILL` to say it once.

    This is here because the interpolation was, briefly, a loop at the foot of
    `build_server` that ran *after* the decorator had already snapshotted `__doc__`. Both
    models duly received a byte-identical tool description reading `... {fill}`, and the
    fee schedule reached them from nowhere. Nothing else in the suite could see it: the
    engine still charged 10 bps, so every fill, every metric and every log was correct.
    """
    session = EpisodeSession.start(
        window_id=KNOWN, track="real", agent=BASELINES["flat"]().agent(),
        episode_index=0, run_dir=tmp_path, windows_dir=windows_dir,
    )
    served = {t.name: (t.description or "") for t in asyncio.run(build_server(session).list_tools())}

    for name, doc in served.items():
        assert "{fill}" not in doc, f"{name} serves an uninterpolated placeholder"
    for name in ("Buy", "Sell"):
        assert "OPEN of the next bar" in served[name], name
        assert "10 bps per side" in served[name], name


def test_the_server_refuses_to_serve_an_unnamed_episode(monkeypatch):
    """One server process is one episode, supplied out of band. A server that quietly
    defaulted to window 0 would write a perfectly valid log for the wrong episode — the
    kind of error that survives every downstream check and lands on a slide."""
    for key in ("PDTBENCH_WINDOW_ID", "PDTBENCH_TRACK", "PDTBENCH_AGENT_ID",
                "PDTBENCH_EPISODE_INDEX"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(SystemExit):
        session_from_env()


# ----------------------------------------------------------- one door, one path


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_baseline_is_logged_identically_through_the_transport_and_in_process(
    name, windows_dir, tmp_path
):
    """The load-bearing one. Same baseline, same window; once through the MCP tool surface
    an LLM would use, once through the in-process call the runner uses. D14's fee parity is
    this assertion — if the two paths could diverge, `sma_10_50` losing money on this
    window set would be a claim about the harness rather than about the strategy."""
    direct, direct_log, direct_payloads = _drive(name, windows_dir, tmp_path / "direct")
    served, served_log, served_payloads = _drive(
        name, windows_dir, tmp_path / "served", via_mcp=True
    )

    assert _canonical_log(direct_log) == _canonical_log(served_log)
    assert direct_payloads == served_payloads

    # ...and it is not two identically empty episodes agreeing with each other.
    assert direct.done and served.done
    assert len(load(direct_log)[1]) == DEFAULT.n_scored
    assert direct_payloads, "the baseline made no calls at all"


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_baseline_pays_the_engines_friction_on_every_fill(name, windows_dir, tmp_path):
    """Fee parity, stated as arithmetic rather than as provenance: 10 bps per side of the
    gross, on every fill a baseline causes, including the terminal liquidation."""
    session, log, _ = _drive(name, windows_dir, tmp_path)
    fills = _fills(log)

    for f in fills:
        expected = int(round(f["gross_notional_cents"] * DEFAULT.friction_rate))
        assert f["friction_cents"] == expected, f["side"]
    assert session.summary["fees_paid_cents"] == sum(f["friction_cents"] for f in fills)


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_baseline_never_fumbles_the_interface(name, windows_dir, tmp_path):
    """A baseline is the control the reliability scoreboard is read against. One that
    generated invalid calls of its own would pollute the board it exists to calibrate."""
    _session, log, _ = _drive(name, windows_dir, tmp_path)
    _meta, _ticks, end = load(log)

    assert end["reliability"]["n_invalid"] == 0
    assert end["reliability"]["n_forced_waits"] == 0
    assert end["status"] == "ok"


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_baseline_episode_validates_and_replays(name, windows_dir, tmp_path):
    """The baselines are 240 of the run's episodes. They meet the same bar as the rest."""
    _session, log, _ = _drive(name, windows_dir, tmp_path)

    report = validate_episode(log)
    assert report.ok, report.errors[:3]
    res = replay(log, windows_dir)
    assert res.ok, res.failures[:3]


# ---------------------------------------------------------- baseline sanity (D14)


def test_flat_scores_exactly_zero(windows_dir, tmp_path):
    """The metric's fixed point (D2). Not approximately zero — exactly. A never-trading
    agent is the reference the vol floor is built to leave alone, and the whole "you
    cannot earn a high Sharpe on capital you never deploy" defense rests on this number
    being 0 and not 0.0003."""
    session, log, _ = _drive("flat", windows_dir, tmp_path)
    m = session.summary

    assert m["sharpe_floored"] == 0.0
    assert m["sharpe_raw"] == 0.0
    assert m["total_return"] == 0.0
    assert m["n_trades"] == 0
    assert m["fees_paid_cents"] == 0
    assert m["time_in_market"] == 0.0
    assert _fills(log) == []

    _meta, _ticks, end = load(log)
    assert set(end["equity_series_cents"]) == {DEFAULT.initial_capital_cents}


def test_buy_and_hold_earns_the_window_net_of_exactly_two_frictions(windows_dir, tmp_path):
    """Buy at `open_1`, ride, liquidate at `close_89`. It touches the market twice, so it
    pays 10 bps twice — the cheapest any strategy in this engine can be. Computed here
    from the raw bars, without asking the engine anything."""
    session, log, _ = _drive("buy_and_hold", windows_dir, tmp_path)
    series, _spec = load_episode(windows_dir, KNOWN, "real")

    close = series["close"].to_numpy(dtype=np.float64)
    open_ = series["open"].to_numpy(dtype=np.float64)
    entry = open_[DEFAULT.n_warmup + 1]              # the fill it never saw coming
    exit_ = close[DEFAULT.n_warmup + DEFAULT.n_scored - 1]  # the forced terminal mark

    net = (1 - DEFAULT.friction_rate) ** 2
    assert session.summary["total_return"] == pytest.approx(net * (exit_ / entry) - 1, abs=1e-5)

    fills = _fills(log)
    assert [(f["fill_tick"], f["side"]) for f in fills] == [(1, "buy"), (89, "liquidation")]
    assert session.summary["n_trades"] == 1  # the liquidation is not a decision

    # The ceiling on exposure, and why it is not 90/90: tick 0 is necessarily flat (the
    # first fill lands at open_1) and tick 89 is necessarily flat (post-liquidation).
    assert session.summary["time_in_market"] == pytest.approx(88 / 90)


def test_the_crossover_trades_match_hand_computed_signals(windows_dir, tmp_path):
    """The 10/50 crossover, recomputed from the window's closes with numpy and compared
    trade for trade. Independent of `getStats` on purpose — the baseline pulls history
    through `fetchData` and averages it itself, so this checks the read path a model
    would use, not a number the engine hands out."""
    _session, log, _ = _drive("sma_10_50", windows_dir, tmp_path)
    series, _spec = load_episode(windows_dir, KNOWN, "real")
    close = series["close"].to_numpy(dtype=np.float64)

    long, expected = False, []
    for t in range(DEFAULT.n_scored - 1):  # 0..88; tick 89 is terminal, no action taken
        window = close[DEFAULT.n_warmup + t - 49 : DEFAULT.n_warmup + t + 1]
        fast, slow = window[-10:].mean(), window.mean()
        if fast > slow and not long:
            expected.append((t + 1, "buy"))  # decided at t, fills at open_{t+1}
            long = True
        elif fast <= slow and long:
            expected.append((t + 1, "sell"))
            long = False
    if long:
        expected.append((DEFAULT.n_scored - 1, "liquidation"))

    assert [(f["fill_tick"], f["side"]) for f in _fills(log)] == expected

    # Not a window it sits out: several round trips, and still long at the bell, so the
    # terminal liquidation is on the hook here too.
    assert len(expected) >= 6, f"{KNOWN} barely trades — pick a window that exercises this"
    assert expected[-1][1] == "liquidation"


# -------------------------------------------------------------- session policy


def test_a_reused_baseline_is_reset_between_episodes(windows_dir, tmp_path):
    """A baseline carries state — a latch, a seeded RNG — and a lane is 30 windows long.

    Reuse one instance without `on_episode` and buy-and-hold buys on the first window and
    never trades again: 29 of 30 episodes score a flat 0. That trace is exactly what every
    learning curve is subtracted from (D11), so the failure would not show up as a broken
    baseline. It would show up as an LLM appearing to learn.
    """
    baseline = BASELINES["buy_and_hold"]()

    trades = []
    for i, window_id in enumerate(("w00", "w01")):
        baseline.on_episode("real", window_id)
        session = EpisodeSession.start(
            window_id=window_id, track="real", agent=baseline.agent(),
            episode_index=i, run_dir=tmp_path / window_id, run_id="test",
            windows_dir=windows_dir,
        )
        drive(session, baseline)
        session.finish()
        trades.append(session.summary["n_trades"])

    assert trades == [1, 1], "the second episode inherited the first's latch"


def test_a_seeded_baseline_is_reproducible_per_episode_not_per_lane(windows_dir, tmp_path):
    """`random_5pct` is a control for turnover, so it has to be the same control for every
    model. Its stream is derived from (seed, track, window) rather than carried across the
    lane, so window 29's coin flips do not depend on how many draws the previous 28
    consumed."""
    first, _log, _p = _drive("random_5pct", windows_dir, tmp_path / "a", window_id="w05")
    again, _log2, _p2 = _drive("random_5pct", windows_dir, tmp_path / "b", window_id="w05")
    other, _log3, _p3 = _drive("random_5pct", windows_dir, tmp_path / "c", window_id="w06")

    assert first.summary["n_trades"] == again.summary["n_trades"]
    assert first.summary["total_return"] == again.summary["total_return"]
    # A different window is a different stream, not the same coins on new prices.
    assert (first.summary["n_trades"], other.summary["n_trades"]) != (0, 0)


def test_the_wallclock_cap_flags_the_episode_but_still_scores_it(windows_dir, tmp_path):
    """D13's hard cap. The remaining ticks auto-Wait and the episode is flagged — but it
    is still a complete, replayable episode that reaches the trading scoreboard. Dithering
    until the clock runs out is a real and bad trading outcome, not a harness failure, and
    quietly dropping the episode would flatter the model that did it."""
    session, log, _ = _drive(
        "buy_and_hold", windows_dir, tmp_path, cfg=replace(DEFAULT, episode_wallclock_cap_s=0)
    )

    assert session.status == "wallclock_capped"
    _meta, ticks, end = load(log)
    assert end["status"] == "wallclock_capped"
    assert end["reliability"]["hit_wallclock_cap"] is True

    # Flagged, not truncated: every tick is present, and it scores.
    assert len(ticks) == DEFAULT.n_scored
    assert end["metrics"]["sharpe_floored"] is not None
    assert validate_episode(log).ok
    assert replay(log, windows_dir).ok

    # The auto-Waits are not `forced` — the schema reserves that for the one trigger that
    # earns it, three consecutive invalid calls. The cap has its own flag.
    assert end["reliability"]["n_forced_waits"] == 0


def test_the_run_manifest_is_written_before_any_episode(windows_dir, tmp_path):
    """A run that dies at episode 3 of 240 should still be interpretable. Write the
    manifest last and the survivors are a directory of orphan logs."""
    agents = [BASELINES[n]().agent() for n in BASELINES]
    manifest = write_run_manifest(tmp_path, agents, windows_dir=windows_dir, run_id="test")

    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "episodes").is_dir() and (tmp_path / "probe").is_dir()
    assert not any((tmp_path / "episodes").iterdir())

    on_disk = json.loads((tmp_path / "run_manifest.json").read_text())
    assert on_disk == manifest
    assert len(on_disk["presentation_order"]) == DEFAULT.n_windows
    assert on_disk["probe"]["chance_level"] == 0.125  # 8-way, read from the run (D9)
    assert on_disk["config"]["config_sha256"]
    assert on_disk["dataset"]["dataset_sha256"]
