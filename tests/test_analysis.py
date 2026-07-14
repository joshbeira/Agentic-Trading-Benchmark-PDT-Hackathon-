"""Scoreboards and learning curves, tested against hand-built fixture logs.

No engine is imported anywhere in this file. The analytics layer reads JSONL and nothing
else, so it must be gradeable without one — otherwise a broken engine could mask a broken
scoreboard, or the reverse.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pdtbench.analysis import learning, reliability, scoreboard
from pdtbench.analysis.loader import load_episode, load_run

from . import fixtures as F

TRADING_DAYS = 252


# ============================================================== the loader recomputes


def test_the_scoreboard_recomputes_sharpe_rather_than_trusting_the_log(tmp_path):
    """The schema says the cached metrics block is a cache, not a source. Prove the loader
    treats it that way: hand it a series whose Sharpe is computable on paper."""
    bh_vol = 0.02
    r = np.full(89, 0.001)  # a perfectly steady +0.1%/day: zero return volatility
    ep = F.make_episode(
        tmp_path / "e.jsonl",
        agent=F.agent_spec("a"), track="real", window_id="w00", regime="bull",
        source_ticker="AAPL", bh_daily_vol=bh_vol, episode_index=0,
        equity_cents=F.equity_from_returns(r), exposure=1.0,
    )
    e = load_episode(ep)

    # std(r) == 0, so the floor is the whole denominator. This is the exploit D2 closes:
    # unfloored, this series divides by ~zero.
    expected = 0.001 / (0.25 * bh_vol) * math.sqrt(TRADING_DAYS)
    assert e.metrics["sharpe_floored"] == pytest.approx(expected, rel=1e-3)
    assert e.metrics["vol_floor_binding"] is True
    assert e.metrics["sharpe_raw"] > 50 * e.metrics["sharpe_floored"]
    assert not e.metric_drift


def test_a_flat_agent_scores_exactly_zero(tmp_path):
    ep = F.make_episode(
        tmp_path / "flat.jsonl",
        agent=F.agent_spec("flat", kind="baseline", memory="none"), track="real",
        window_id="w00", regime="chop", source_ticker="AAPL", bh_daily_vol=0.02,
        episode_index=0, equity_cents=[F.INITIAL_CENTS] * 90, exposure=0.0, n_trades=0,
    )
    e = load_episode(ep)
    assert e.metrics["sharpe_floored"] == 0.0
    assert e.metrics["sharpe_raw"] == 0.0  # 0, not NaN -- the schema is explicit
    assert e.metrics["total_return"] == 0.0
    assert e.metrics["time_in_market"] == 0.0
    assert e.metrics["n_trades"] == 0


def test_a_planted_sharpe_comes_back_out(tmp_path):
    """The fixture standardizes its returns to an exact target, so the scoreboard should
    return the planted number and not merely 'a number'."""
    for target in (-1.5, 0.0, 0.8, 2.4):
        r = F.returns_for_sharpe(target, sigma=0.02, seed=1)
        ep = F.make_episode(
            tmp_path / f"s{target}.jsonl",
            agent=F.agent_spec("a"), track="real", window_id="w00", regime="bull",
            source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
            equity_cents=F.equity_from_returns(r),
        )
        got = load_episode(ep).metrics["sharpe_floored"]
        assert got == pytest.approx(target, abs=0.01), f"planted {target}, recovered {got}"


def test_drift_between_cached_and_recomputed_metrics_is_caught(tmp_path):
    """If the engine and the specification ever disagree, one of them is wrong and we have
    to find out from a test rather than from a judge."""
    r = F.returns_for_sharpe(1.0, sigma=0.02, seed=2)
    ep = F.make_episode(
        tmp_path / "drift.jsonl",
        agent=F.agent_spec("a"), track="real", window_id="w00", regime="bull",
        source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
        equity_cents=F.equity_from_returns(r),
        corrupt_cached_metrics={"sharpe_floored": 9.99, "total_return": 0.5,
                                "fees_paid_cents": 0, "n_trades": 0, "turnover": 0.0,
                                "max_drawdown": 0.0, "time_in_market": 0.0,
                                "sharpe_raw": 9.99, "vol_floor_binding": False,
                                "realized_vol_ann": 0.0},
    )
    e = load_episode(ep)
    assert e.metric_drift
    assert "sharpe_floored" in e.metric_drift
    assert e.metric_drift["sharpe_floored"]["cached"] == 9.99
    assert e.metrics["sharpe_floored"] == pytest.approx(1.0, abs=0.01)  # the truth wins


def test_a_self_contradicting_log_is_refused(tmp_path):
    """episode_end.equity_series_cents must agree with the per-tick marks."""
    r = F.returns_for_sharpe(1.0, sigma=0.02, seed=3)
    path = F.make_episode(
        tmp_path / "bad.jsonl",
        agent=F.agent_spec("a"), track="real", window_id="w00", regime="bull",
        source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
        equity_cents=F.equity_from_returns(r),
    )
    lines = path.read_text().splitlines()
    end = json.loads(lines[-1])
    end["equity_series_cents"][40] += 100_000
    lines[-1] = json.dumps(end)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="contradicts itself"):
        load_episode(path)


# ==================================================================== trading scoreboard


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Two LLMs and four baselines over 30 windows. `flagship` is planted a genuine +0.5
    Sharpe edge over window difficulty; `cheap` is planted nothing."""
    root = tmp_path_factory.mktemp("run")
    agents = [
        F.agent_spec("flagship"),
        F.agent_spec("cheap"),
        F.agent_spec("buy_and_hold", kind="baseline", memory="none"),
        F.agent_spec("flat", kind="baseline", memory="none"),
        F.agent_spec("random_5pct", kind="baseline", memory="none"),
        F.agent_spec("sma_10_50", kind="baseline", memory="none"),
    ]
    windows = F.window_specs()

    def sharpe_fn(agent, track, w, i):
        base = w["difficulty"]  # what a baseline sees: pure window difficulty
        if agent["kind"] == "baseline":
            return base + {"buy_and_hold": 0.1, "flat": 0.0,
                           "random_5pct": -0.1, "sma_10_50": -0.2}[agent["id"]]
        return base + (0.5 if agent["id"] == "flagship" else 0.0)

    return load_run(F.make_run(root, agents=agents, windows=windows, sharpe_fn=sharpe_fn))


def test_the_run_loads_whole(run):
    assert len(run.episodes) == 6 * 2 * 30
    assert set(run.llms) == {"flagship", "cheap"}
    assert set(run.baselines) == {"buy_and_hold", "flat", "random_5pct", "sma_10_50"}
    assert not run.drifted
    assert not run.excluded


def test_the_scoreboard_ranks_by_median_floored_sharpe(run):
    rows = scoreboard.build(run, "real")
    assert [r.agent_id for r in rows][0] == "flagship"  # the only agent with a planted edge
    assert rows == sorted(rows, key=lambda r: r.sharpe.point, reverse=True)

    flagship = next(r for r in rows if r.agent_id == "flagship")
    cheap = next(r for r in rows if r.agent_id == "cheap")
    assert flagship.sharpe.point - cheap.sharpe.point == pytest.approx(0.5, abs=0.25)
    assert flagship.n == 30
    assert flagship.sharpe.lo < flagship.sharpe.point < flagship.sharpe.hi


def test_the_intervals_are_wide_because_the_design_is_power_limited(run):
    """The plan promises to own this rather than be handed it. If the CI ever comes back
    suspiciously tight, something is wrong with the bootstrap, not with the world."""
    rows = scoreboard.build(run, "real")
    assert all(r.sharpe.width > 0.15 for r in rows)


def test_paired_comparison_uses_the_shared_windows(run):
    """Every model sees identical windows, so the comparison is paired (D5). Refusing the
    pairing would throw away most of the power this design has."""
    c = scoreboard.compare(run, "flagship", "cheap", "real")
    assert c.n_pairs == 30
    assert c.median_diff.point == pytest.approx(0.5, abs=0.2)
    assert c.test.pvalue < 0.01  # a planted 0.5 Sharpe edge, paired, is findable at n=30
    assert "flagship ahead" in c.verdict


def test_the_paired_test_is_far_stronger_than_the_unpaired_one(run):
    """The point of the design, made numeric: paired sees a planted edge that an unpaired
    test on the same data misses entirely, because window difficulty swamps it."""
    from pdtbench.analysis import stats as S

    a = [e.metrics["sharpe_floored"] for e in run.select(agent_id="flagship", track="real")]
    b = [e.metrics["sharpe_floored"] for e in run.select(agent_id="cheap", track="real")]

    paired = S.wilcoxon(a, b)
    unpaired = S.mann_whitney(a, b)
    assert paired.pvalue < 0.01
    assert unpaired.pvalue > paired.pvalue * 10


def test_regime_panels_split_ten_and_ten_and_ten(run):
    panels = scoreboard.by_regime(run, "real")
    assert set(panels) == {"bull", "bear", "chop"}
    for regime, rows in panels.items():
        assert all(r.n == 10 for r in rows)
        assert all(r.regime == regime for r in rows)


def test_buy_and_hold_is_only_ever_compared_within_regime(run):
    """Pooling would compare against a buy-and-hold median that our own 10/10/10 sampling
    pins near zero. That is a fact about our sampling, not about the agent (D12)."""
    cmps = scoreboard.vs_buy_and_hold(run, "flagship", "real")
    assert set(cmps) == {"bull", "bear", "chop"}
    assert all(c.n_pairs == 10 for c in cmps.values())


def test_the_scoreboard_renders(run):
    text = scoreboard.render(scoreboard.build(run, "real"), "TRADING SCOREBOARD")
    assert "flagship" in text and "median Sharpe" in text
    assert "88.9%" in text or "ceiling" in text  # the time-in-market ceiling is stated


# ================================================================== excluded episodes


def test_agent_errors_are_excluded_from_trading_but_not_from_reliability(tmp_path):
    """A provider failure is not a trading result. Scoring a crash as 'flat' would reward
    crashing on a bad window (schemas/analysis_artifacts.md Q5)."""
    agents = [F.agent_spec("flaky"), F.agent_spec("flat", kind="baseline", memory="none")]
    windows = F.window_specs(6)

    def status_fn(agent, track, w, i):
        return "agent_error" if (agent["id"] == "flaky" and i == 0 and track == "real") else "ok"

    run = load_run(F.make_run(
        tmp_path / "r", agents=agents, windows=windows,
        sharpe_fn=lambda a, t, w, i: w["difficulty"], status_fn=status_fn,
    ))

    assert len(run.excluded) == 1
    assert len(run.select(agent_id="flaky", track="real")) == 5  # 6 minus the crash

    rows = scoreboard.build(run, "real")
    assert next(r for r in rows if r.agent_id == "flaky").n == 5

    rel = reliability.build(run)
    assert next(r for r in rel if r.agent_id == "flaky").n_agent_errors == 1


def test_wallclock_capped_episodes_still_count(tmp_path):
    """The agent dithered until the clock ran out. That is a real, bad trading outcome and
    it belongs in the distribution -- flagged, not hidden."""
    agents = [F.agent_spec("ditherer")]
    windows = F.window_specs(6)
    run = load_run(F.make_run(
        tmp_path / "r", agents=agents, windows=windows,
        sharpe_fn=lambda a, t, w, i: 0.0,
        status_fn=lambda a, t, w, i: "wallclock_capped" if (i == 0 and t == "real") else "ok",
    ))
    assert not run.excluded  # capped is not excluded ...
    assert scoreboard.build(run, "real")[0].n == 6  # ... it stays in the distribution
    assert reliability.build(run)[0].n_capped == 1  # ... and is flagged over here instead


# ================================================================== reliability board


def test_the_reliability_board_counts_from_the_calls_array(tmp_path):
    """Counted from `calls`, which is the primary record -- not from the cached reliability
    block, so a mismatch between them shows up instead of being inherited."""
    F.make_episode(
        tmp_path / "episodes" / "fumbler__real__w00.jsonl",
        agent=F.agent_spec("fumbler"), track="real", window_id="w00", regime="bull",
        source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
        equity_cents=F.equity_from_returns(F.returns_for_sharpe(1.0, 0.02, seed=4)),
        invalid_calls_per_tick=2, reads_per_tick=1, forced_waits=3,
    )
    run = load_run(tmp_path)
    row = reliability.build(run)[0]

    # 90 ticks: 89 have a Wait, all 90 have 1 read + 2 invalids
    assert row.n_calls == 89 + 90 * 3
    assert row.n_invalid == 180
    assert row.invalid_rate == pytest.approx(180 / (89 + 270))
    assert row.forced_waits_per_episode == 3
    assert row.error_breakdown == {"NONPOSITIVE_QTY": 180}
    assert not row.clean


def test_the_reliability_board_says_it_costs_no_pnl(run):
    text = reliability.render(reliability.build(run))
    assert "No P&L penalty" in text


# ==================================================================== learning curves


def test_a_planted_learning_slope_is_recovered(tmp_path):
    """`learner` improves by exactly +0.04 Sharpe per episode *on top of* window
    difficulty. The excess slope must find it."""
    agents = [
        F.agent_spec("learner"),
        F.agent_spec("static"),
        F.agent_spec("buy_and_hold", kind="baseline", memory="none"),
        F.agent_spec("flat", kind="baseline", memory="none"),
    ]
    windows = F.window_specs()

    def sharpe_fn(agent, track, w, i):
        base = w["difficulty"]
        if agent["kind"] == "baseline":
            return base
        return base + (0.04 * i if agent["id"] == "learner" else 0.0)

    run = load_run(F.make_run(tmp_path / "r", agents=agents, windows=windows,
                              sharpe_fn=sharpe_fn))
    curves = {c.agent_id: c for c in learning.build(run, "real")}

    learner = curves["learner"]
    assert learner.excess_slope.slope == pytest.approx(0.04, abs=0.01)
    assert learner.excess_slope.significant
    assert learner.learned
    assert "learning" in learner.verdict
    assert learner.second_half > learner.first_half

    static = curves["static"]
    assert not static.learned
    assert static.excess_slope.lo < 0 < static.excess_slope.hi
    assert "no detectable learning" in static.verdict


def test_the_baseline_trace_is_what_makes_the_curve_mean_anything(tmp_path):
    """The trap the trace exists to close: an agent whose Sharpe rises purely because the
    later windows are easier. Its RAW slope is strongly positive. Its EXCESS slope must be
    flat -- otherwise we would be reporting window difficulty as machine learning."""
    agents = [
        F.agent_spec("rider"),
        F.agent_spec("buy_and_hold", kind="baseline", memory="none"),
        F.agent_spec("flat", kind="baseline", memory="none"),
    ]
    # difficulty ramps with episode index: every agent, including baselines, does better late
    windows = [dict(w, difficulty=0.05 * i) for i, w in enumerate(F.window_specs())]

    run = load_run(F.make_run(tmp_path / "r", agents=agents, windows=windows,
                              sharpe_fn=lambda a, t, w, i: w["difficulty"]))
    rider = learning.curve(run, "rider", "real")

    assert rider.raw_slope.slope == pytest.approx(0.05, abs=0.01)
    assert rider.raw_slope.significant  # the naive read: "it is learning!"
    assert rider.excess_slope.slope == pytest.approx(0.0, abs=0.01)
    assert not rider.learned  # the honest read: it is not
    assert not rider.excess_slope.significant


def test_a_never_trading_baseline_is_kept_out_of_the_difficulty_trace(tmp_path):
    """A baseline whose score does not depend on the window cannot describe the window.

    `flat` never trades, so its Sharpe is exactly 0 everywhere. Folding a constant into the
    median that DEFINES window difficulty drags the trace off the signal and leaves a
    residual trend in the excess — which then reads as learning. When this was live it
    produced a *statistically significant* +0.004/episode slope for an agent with no
    learning planted at all.
    """
    agents = [
        F.agent_spec("static"),
        F.agent_spec("buy_and_hold", kind="baseline", memory="none"),
        F.agent_spec("random_5pct", kind="baseline", memory="none"),
        F.agent_spec("flat", kind="baseline", memory="none"),  # the constant-zero trap
    ]
    # difficulty drifts upward, so a contaminated trace leaves a slope behind
    windows = [dict(w, difficulty=0.05 * i) for i, w in enumerate(F.window_specs())]

    run = load_run(F.make_run(
        tmp_path / "r", agents=agents, windows=windows,
        sharpe_fn=lambda a, t, w, i: w["difficulty"],
        exposure_fn=lambda a: 0.0 if a["id"] == "flat" else 1.0,
    ))

    assert "flat" in run.baselines
    assert learning.trace_baselines(run, "real") == ["buy_and_hold", "random_5pct"]

    static = learning.curve(run, "static", "real")
    assert static.raw_slope.slope == pytest.approx(0.05, abs=0.01)  # difficulty ramps
    assert static.excess_slope.slope == pytest.approx(0.0, abs=0.005)  # ... and is removed
    assert not static.excess_slope.significant
    assert not static.learned

    text = learning.render(learning.build(run, "real"), "real", run)
    assert "excluded from the trace: ['flat']" in text


def test_a_baseline_has_no_excess_slope_by_construction(tmp_path):
    """A sanity check on the trace itself: baselines are the trace, so their excess must be
    flat. If it is not, the trace is broken."""
    agents = [
        F.agent_spec("buy_and_hold", kind="baseline", memory="none"),
        F.agent_spec("flat", kind="baseline", memory="none"),
    ]
    run = load_run(F.make_run(tmp_path / "r", agents=agents, windows=F.window_specs(),
                              sharpe_fn=lambda a, t, w, i: w["difficulty"]))
    for c in learning.build(run, "real"):
        assert abs(c.excess_slope.slope) < learning.MIN_LEARNING_SLOPE
        assert not c.learned
        assert c.verdict == "no memory (control arm)"


def test_an_effect_can_be_significant_and_still_not_matter(tmp_path):
    """A t-test on a series with almost no residual variance certifies anything.

    Two baselines planted at the identical Sharpe differ only by integer-cent rounding, so
    their excess is ~1e-07 with a trend — and the interval clears zero at p = 0.013. Left
    alone, that reads out as 'learning'. It is seven orders of magnitude below anything
    tradeable, and the effect-size floor is what stops us reporting it.
    """
    agents = [
        F.agent_spec("twinA", kind="baseline", memory="none"),
        F.agent_spec("twinB", kind="baseline", memory="none"),
    ]
    run = load_run(F.make_run(tmp_path / "r", agents=agents, windows=F.window_specs(),
                              sharpe_fn=lambda a, t, w, i: w["difficulty"]))
    c = learning.curve(run, "twinA", "real")

    assert abs(c.excess_slope.slope) < 1e-5  # microscopic ...
    assert not c.learned  # ... and correctly not reported, whatever the p-value says
    assert not c.excess_slope.matters(learning.MIN_LEARNING_SLOPE)


def test_episode_index_orders_the_lane(run):
    lane = run.lane("flagship", "real")
    assert [e.episode_index for e in lane] == list(range(30))
    assert lane[0].memory_note_in is None  # nothing carried into episode 0
    assert lane[1].memory_note_in is not None
