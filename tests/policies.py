"""Scripted policies used to drive the engine in tests.

Deliberately *not* the benchmark baselines — those run through the MCP layer in
step 4. These exist to exercise the engine hard enough that its invariants have
something to be invariant about: a policy that never trades, one that is always
in, one that churns, one that trades dust, and one that abuses the interface.
"""

from __future__ import annotations

import numpy as np

from pdtbench.engine import TradingEnv


def run(env: TradingEnv, policy, max_calls: int = 20_000) -> TradingEnv:
    """Drive an episode to completion. Guards against a policy that never advances."""
    obs = env.reset()
    calls = 0
    while not env.done:
        calls += 1
        if calls > max_calls:
            raise AssertionError("policy never terminated the episode")
        tool, args = policy(obs, env)
        res = env.step(tool, args)
        if res.obs is not None:
            obs = res.obs
    return env


class Flat:
    """Never trades. Must score exactly zero — the metric's fixed point."""

    def __call__(self, obs, env):
        return "Wait", {"n": min(10, obs["ticks_remaining"] or 1)}


class BuyAndHold:
    """All in at the first opportunity, then ride. Exits only via the terminal
    liquidation, so it pays exactly two frictions."""

    def __init__(self):
        self.bought = False

    def __call__(self, obs, env):
        if not self.bought:
            self.bought = True
            return "Buy", {"fraction": 1.0}
        return "Wait", {"n": min(10, obs["ticks_remaining"] or 1)}


class Churn:
    """Buy, sell, buy, sell, every single tick. Stress-tests the fill and cost-basis
    arithmetic, and should be destroyed by friction."""

    def __call__(self, obs, env):
        if obs["portfolio"]["shares"] > 0:
            return "Sell", {"fraction": 1.0}
        return "Buy", {"fraction": 1.0}


class Dust:
    """Puts 1% of capital to work for a few days and then sits in cash.

    The whole reason the Sharpe has a volatility floor (D2). Unfloored, this can
    post an enormous ratio off a near-zero denominator.
    """

    def __init__(self, enter: int = 1, exit_: int = 6):
        self.enter, self.exit_ = enter, exit_

    def __call__(self, obs, env):
        t = obs["tick"]
        if t == self.enter and obs["portfolio"]["shares"] == 0:
            return "Buy", {"notional_cents": env.cfg.initial_capital_cents // 100}
        if t == self.exit_ and obs["portfolio"]["shares"] > 0:
            return "Sell", {"fraction": 1.0}
        return "Wait", {"n": 1}


class SmaCrossover:
    """10/50 crossover, sized all-in / all-out. Trades only on a state change."""

    def __init__(self, fast: int = 10, slow: int = 50):
        self.fast, self.slow = fast, slow

    def __call__(self, obs, env):
        with env.bars.allow(env.t):
            c = env.bars.closes_through(env.t, self.slow)
        fast = float(np.mean(c[-self.fast :]))
        slow = float(np.mean(c))
        long = obs["portfolio"]["shares"] > 0
        if fast > slow and not long:
            return "Buy", {"fraction": 1.0}
        if fast <= slow and long:
            return "Sell", {"fraction": 1.0}
        return "Wait", {"n": 1}


class RandomTrader:
    def __init__(self, seed: int = 0, p: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.p = p

    def __call__(self, obs, env):
        if self.rng.random() < self.p:
            if obs["portfolio"]["shares"] > 0 and self.rng.random() < 0.5:
                return "Sell", {"fraction": 1.0}
            if obs["portfolio"]["cash_cents"] > 0:
                return "Buy", {"fraction": 0.5}
        return "Wait", {"n": 1}


class Adversary:
    """Every way an agent can fumble the interface, on a loop.

    Never issues a valid action. The engine's anti-stall ladder is the only thing
    that can end this episode, which is exactly what it is here to prove.
    """

    BAD = [
        ("Buy", {"notional_cents": -500}),
        ("Buy", {"notional_cents": 0}),
        ("Buy", {"notional_cents": float("nan")}),
        ("Buy", {"notional_cents": 10**12}),
        ("Sell", {"shares": 99999.0}),
        ("Sell", {}),
        ("Buy", {"notional_cents": 100, "fraction": 0.5}),
        ("Wait", {"n": 99}),
        ("Wait", {"n": 0}),
        ("Teleport", {"to": "moon"}),
        ("getStats", {}),  # legal, but reads are capped
    ]

    def __init__(self):
        self.i = 0

    def __call__(self, obs, env):
        call = self.BAD[self.i % len(self.BAD)]
        self.i += 1
        return call
