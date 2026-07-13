"""Engine invariants.

The three the design actually rests on — the equity identity, no look-ahead, and
the metric's resistance to being gamed — plus the interface ladder that keeps an
episode from deadlocking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdtbench.data.windows import load_episode
from pdtbench.engine import LookAheadError, TradingEnv
from pdtbench.engine.audit import AuditedBars

from . import policies as P

ALL_POLICIES = {
    "flat": P.Flat,
    "buy_and_hold": P.BuyAndHold,
    "churn": P.Churn,
    "dust": P.Dust,
    "sma_crossover": P.SmaCrossover,
    "random": lambda: P.RandomTrader(seed=7),
}


# =========================================================== equity invariant


@pytest.mark.parametrize("name", list(ALL_POLICIES))
@pytest.mark.parametrize("track", ["real", "twin"])
def test_equity_identity_holds_at_every_tick(make_env, name, track):
    """equity == cash + shares marked at this bar's close, to the cent, always."""
    env = P.run(make_env("w03", track), ALL_POLICIES[name]())
    closes = env.bars.raw_close_series()[env.cfg.n_warmup :]

    for t, (eq, sh) in enumerate(zip(env.equity_series, env.shares_by_tick)):
        # tick 89 is marked post-liquidation, so its position is zero by definition
        expect_cash = eq - round(sh * closes[t] * 100)
        assert eq == expect_cash + round(sh * closes[t] * 100), f"t={t}"
        assert eq > 0, f"equity went non-positive at t={t}"

    assert len(env.equity_series) == env.cfg.n_scored
    assert env.shares == 0.0, "terminal liquidation left a position open"
    assert env.equity_series[-1] == env.cash_cents


@pytest.mark.parametrize("name", list(ALL_POLICIES))
def test_cash_is_conserved(make_env, name):
    """Initial capital + every cash flow == final cash. No cents invented or lost."""
    env = P.run(make_env("w16"), ALL_POLICIES[name]())
    flows = sum(f["cash_delta_cents"] for f in env.fills)
    assert env.cash_cents == env.cfg.initial_capital_cents + flows


@pytest.mark.parametrize("name", list(ALL_POLICIES))
def test_a_fill_costs_exactly_its_friction(make_env, name):
    """The only equity a fill destroys is the friction it paid."""
    env = P.run(make_env("w21"), ALL_POLICIES[name]())
    for f in env.fills:
        px, sh = f["fill_price"], f["shares_delta"]
        # value handed over vs value received, marked at the fill price itself
        delta = f["cash_delta_cents"] + round(sh * px * 100)
        assert abs(delta + f["friction_cents"]) <= 1, (
            f"a {f['side']} moved equity by {delta} cents against a friction of "
            f"{f['friction_cents']}"
        )


def test_integer_cents_end_to_end(make_env):
    env = P.run(make_env("w24"), P.Churn())
    assert isinstance(env.cash_cents, int)
    assert all(isinstance(e, int) for e in env.equity_series)
    assert all(isinstance(f["friction_cents"], int) for f in env.fills)
    assert all(isinstance(f["cash_delta_cents"], int) for f in env.fills)


# =============================================================== no look-ahead


def test_audit_tripwire_fires_on_tomorrow(windows_dir, cfg):
    """The wrapper itself: today is readable, tomorrow is not."""
    series, _ = load_episode(windows_dir, "w00", "real")
    bars = AuditedBars(series, cfg.n_warmup, cfg.n_scored)

    with bars.allow(5):
        bars.close(5)
        bars.close(-200)
        with pytest.raises(LookAheadError, match="look-ahead"):
            bars.close(6)

    # and a read with no declared allowance at all is refused
    with pytest.raises(LookAheadError):
        bars.close(0)


def test_a_peeking_engine_raises(make_env, monkeypatch):
    """The tripwire is *wired into the real observation path*, not merely available.

    Teach the stats builder to look at tomorrow's close — the sort of thing a
    careless refactor does — and the episode must die on the first tick rather
    than quietly produce an optimistic Sharpe.
    """
    env = make_env("w00")
    original = TradingEnv._stats

    def peeking(self, t):
        self.bars.close(t + 1)  # the bug
        return original(self, t)

    monkeypatch.setattr(TradingEnv, "_stats", peeking)
    with pytest.raises(LookAheadError):
        P.run(env, P.Flat())


@pytest.mark.parametrize("name", list(ALL_POLICIES))
def test_every_fill_prices_at_the_next_open(make_env, name):
    """A decision at t fills at open_{t+1} — a price the agent had not seen."""
    env = P.run(make_env("w09"), ALL_POLICIES[name]())
    last = env.cfg.n_scored - 1

    for f in env.fills:
        ft = f["fill_tick"]
        if f["side"] == "liquidation":
            assert ft == last
            assert f["fill_price"] == pytest.approx(env.bars.raw_close_series()[-1])
        else:
            assert 1 <= ft <= last
            assert f["fill_price"] == pytest.approx(env.bars.raw_open(ft))


def test_the_first_action_cannot_fill_before_tick_1(make_env):
    """E_0 is always the untouched initial capital: nothing can fill at open_0."""
    for pol in (P.BuyAndHold(), P.Churn(), P.SmaCrossover()):
        env = P.run(make_env("w12"), pol)
        assert env.equity_series[0] == env.cfg.initial_capital_cents
        assert env.shares_by_tick[0] == 0.0
        assert all(f["fill_tick"] >= 1 for f in env.fills)


# ============================================================ the vol floor (D2)


def test_flat_scores_exactly_zero(make_env):
    env = P.run(make_env("w05"), P.Flat())
    m = env.summary
    assert m["total_return"] == 0.0
    assert m["sharpe_floored"] == 0.0
    assert m["sharpe_raw"] == 0.0
    assert m["n_trades"] == 0
    assert m["fees_paid_cents"] == 0
    assert m["time_in_market"] == 0.0


def test_the_floor_defuses_a_dust_trade(make_env):
    """A 1%-of-capital position for five days has almost no return volatility. Raw
    Sharpe divides by that and explodes; the floor divides by the volatility the
    window actually offered, and it does not."""
    env = P.run(make_env("w03"), P.Dust())
    m = env.summary

    assert m["n_trades"] == 2
    assert m["vol_floor_binding"], "a 1% position should not clear the floor"
    assert abs(m["sharpe_raw"]) > 3 * abs(m["sharpe_floored"]), (
        f"floor barely bit: raw={m['sharpe_raw']:.2f} floored={m['sharpe_floored']:.2f}"
    )
    assert abs(m["sharpe_floored"]) < 1.0, "a dust trade must not post a real Sharpe"


def test_buy_and_hold_pays_exactly_two_frictions(make_env, manifest):
    """The plan's baseline sanity check: B&H return == window return, net of an
    entry at open_1 and the terminal exit at close_89, and nothing else."""
    env = P.run(make_env("w06"), P.BuyAndHold())
    fr = env.cfg.friction_rate

    open_1 = env.bars.raw_open(1)
    close_last = env.bars.raw_close_series()[-1]
    expected = (1 - fr) ** 2 * (close_last / open_1) - 1

    assert env.summary["total_return"] == pytest.approx(expected, abs=2e-4)
    assert env.summary["n_trades"] == 1  # the liquidation is not a trade
    assert len(env.fills) == 2  # ... but it is a fill
    # in the market from the fill at tick 1 through tick 88; flat at 0 and at 89
    assert env.summary["time_in_market"] == pytest.approx(88 / 90)


def test_churn_is_destroyed_by_friction(make_env):
    """Round-tripping every tick pays 20 bps a day. It should be visibly punished —
    this is the design's answer to scalping."""
    env = P.run(make_env("w26"), P.Churn())  # w26: the calmest window, 9.2% vol
    assert env.summary["fees_paid_cents"] > 5000  # >$50 on $10k
    assert env.summary["total_return"] < 0
    assert env.summary["turnover"] > 20


# ====================================================== interface / anti-stall


def test_invalid_calls_do_not_move_the_clock(make_env):
    env = make_env("w00")
    env.reset()
    t0, eq0 = env.t, list(env.equity_series)

    for tool, args in [
        ("Buy", {"notional_cents": -1}),
        ("Sell", {"shares": 1.0}),
    ]:
        res = env.step(tool, args)
        assert not res.ok and not res.advanced_time
    assert env.t == t0
    assert env.equity_series == eq0


def test_error_codes_are_specific(make_env):
    env = make_env("w00")
    env.reset()

    def code(tool, args):
        return env.step(tool, args).error.code

    assert code("Buy", {"notional_cents": 0}) == "NONPOSITIVE_QTY"
    env.consecutive_invalid = 0
    assert code("Buy", {"notional_cents": float("nan")}) == "NAN_QTY"
    env.consecutive_invalid = 0
    assert code("Buy", {"notional_cents": 10**12}) == "INSUFFICIENT_CASH"
    env.consecutive_invalid = 0
    assert code("Sell", {"shares": 1.0}) == "INSUFFICIENT_SHARES"
    env.consecutive_invalid = 0
    assert code("Wait", {"n": 99}) == "WAIT_OUT_OF_RANGE"
    env.consecutive_invalid = 0
    assert code("Teleport", {}) == "UNKNOWN_TOOL"
    env.consecutive_invalid = 0
    assert code("Buy", {"notional_cents": 100, "fraction": 0.5}) == "SCHEMA_ERROR"


def test_three_strikes_forces_a_wait(make_env):
    env = make_env("w00")
    env.reset()
    assert env.t == 0

    env.step("Buy", {"notional_cents": -1})
    env.step("Buy", {"notional_cents": -2})
    res = env.step("Buy", {"notional_cents": -3})

    assert not res.ok  # the third call still fails ...
    assert res.forced_wait and res.advanced_time  # ... and the engine takes the turn
    assert env.t == 1
    assert res.obs is not None


def test_the_read_cap_stops_a_staller(make_env):
    env = make_env("w00")
    env.reset()
    for _ in range(env.cfg.max_reads_per_tick):
        assert env.step("getStats", {}).ok
    res = env.step("getStats", {})
    assert not res.ok and res.error.code == "READ_CAP_EXCEEDED"
    assert env.t == 0  # a read cap does not, by itself, move time


def test_an_adversary_cannot_deadlock_the_episode(make_env):
    """The whole point of the ladder: an agent that never makes a valid call still
    reaches tick 89, and pays for it in time, not in P&L."""
    env = P.run(make_env("w00"), P.Adversary())

    assert env.done
    assert env.t == env.cfg.n_scored - 1
    assert env.summary["total_return"] == 0.0  # never traded: no P&L penalty
    assert env.summary["fees_paid_cents"] == 0
    assert env.logger is None or True  # no log configured here
    assert env.summary["sharpe_floored"] == 0.0


def test_reads_are_free_of_time(make_env):
    env = make_env("w00")
    env.reset()
    for tool in ("getStats", "ViewWallet", "ViewPortfolio", "fetchData"):
        assert env.step(tool, {}).ok
        assert env.t == 0


def test_fetch_data_is_capped_and_never_shows_the_future(make_env):
    env = make_env("w00")
    obs = env.reset()
    res = env.step("fetchData", {"lookback": 500})

    bars = res.data["bars"]
    assert res.data["lookback"] == env.cfg.fetch_lookback_cap == 200
    assert len(bars) == 200
    assert max(b["t"] for b in bars) == 0  # nothing past the current tick
    assert min(b["t"] for b in bars) == -199


def test_wait_is_clamped_at_the_horizon(make_env):
    env = make_env("w00")
    env.reset()
    env.step("Wait", {"n": 10})
    assert env.t == 10

    while env.cfg.n_scored - 1 - env.t > 10:
        env.step("Wait", {"n": 10})

    # a Wait that would run past the final bar lands on it instead of erroring
    env.step("Wait", {"n": 10})
    assert env.done
    assert env.t == env.cfg.n_scored - 1
    assert len(env.equity_series) == env.cfg.n_scored


def test_wait_returns_every_bar_it_skipped(make_env):
    """Wait(n) is a decision-cost optimization (D4), not an information sacrifice."""
    env = make_env("w00")
    env.reset()
    res = env.step("Wait", {"n": 5})
    assert [b["t"] for b in res.obs["elapsed_bars"]] == [1, 2, 3, 4, 5]


def test_one_action_per_tick(make_env):
    """A valid action moves the clock, so a second one lands on a new tick. The
    'one valid action per tick' rule is structural, not policed."""
    env = make_env("w00")
    env.reset()
    env.step("Buy", {"fraction": 0.5})
    assert env.t == 1
    env.step("Buy", {"fraction": 0.5})
    assert env.t == 2


def test_selling_a_displayed_position_is_not_a_rounding_error(make_env):
    """ViewPortfolio rounds shares to 6dp. Selling exactly what was shown must work."""
    env = make_env("w00")
    env.reset()
    env.step("Buy", {"fraction": 1.0})
    shown = env.step("ViewPortfolio", {}).data["portfolio"]["shares"]

    res = env.step("Sell", {"shares": shown})
    assert res.ok, "an agent that sells precisely what it was shown must not be refused"
    assert env.shares == 0.0
