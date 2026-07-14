"""The MCP server — the surface an LLM agent actually touches.

One server process is **one episode**. The window, track, agent identity and episode
index are supplied out of band (environment variables, set by the runner) and there is
deliberately no tool to change them: a model cannot reset its own episode, re-roll a bad
window, or ask which window it is on. The only thing it can do is trade.

Every tool here is a one-line delegation to `EpisodeSession.call()`. That is the point —
the baselines call the same method in-process, so there is exactly one path to a fill and
fee parity is structural (D14). The tool names are written out longhand rather than
generated from the engine's `Tool` enum — each one needs its own signature and its own
description anyway — so `tests/test_mcp.py` asserts the served surface equals that enum.
A tool added to the engine and forgotten here fails that test rather than going quietly
missing from every model's toolbox.

An invalid call returns a structured error; it does not raise. The agent has to see the
error to correct itself, an invalid call costs time but never money (D13), and the
reliability scoreboard counts every one.

    PDTBENCH_WINDOW_ID=w07 PDTBENCH_TRACK=real PDTBENCH_AGENT_ID=cheap \
    PDTBENCH_EPISODE_INDEX=6 PDTBENCH_RUN_DIR=runs/20260714T0930Z_a1b2c3d4 \
    python -m pdtbench.mcp.server
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..config import DEFAULT
from .session import EpisodeSession

# Stated in every tool description, so the contract reaches the model through the tool
# surface as well as the system prompt. Both models see byte-identical text (D13).
_FILL = (
    "Fills at the OPEN of the next bar — a price you have not seen. Friction is "
    "10 bps per side (2 bps fee + 8 bps slippage), deterministic."
)


def _states_the_fill_rule(fn):
    """Interpolate `_FILL` into a tool's docstring *before* FastMCP registers it.

    Applied under `@mcp.tool`, so it runs first. This used to be a loop at the foot of
    `build_server`, which was too late: the decorator snapshots `__doc__` into the
    registered tool at decoration time, so mutating the function afterwards left every
    model reading a literal `{fill}` where the fee schedule and the fill rule were
    supposed to be. Keeping the interpolation *on* the function is what stops that
    ordering from silently coming apart again.
    """
    fn.__doc__ = (fn.__doc__ or "").format(fill=_FILL)
    return fn


def build_server(session: EpisodeSession, name: str = "pdtbench-trading") -> FastMCP:
    """Wrap one live episode in an MCP tool surface."""
    mcp = FastMCP(name)

    @mcp.tool(name="Buy")
    @_states_the_fill_rule
    def buy(notional_cents: int | None = None, fraction: float | None = None) -> dict:
        """Buy stock. Advances the clock by one bar.

        Give exactly ONE of:
          notional_cents — how much cash to spend, an integer number of cents, <= your cash.
          fraction       — the share of your cash to spend, 0 < f <= 1.

        Buying is denominated in CASH, not shares, so the order is always executable: the
        engine works out the share count at the fill price. Shares bought =
        notional x (1 - friction) / next_open. {fill}
        """
        args = {}
        if notional_cents is not None:
            args["notional_cents"] = notional_cents
        if fraction is not None:
            args["fraction"] = fraction
        return session.call("Buy", args)

    @mcp.tool(name="Sell")
    @_states_the_fill_rule
    def sell(shares: float | None = None, fraction: float | None = None) -> dict:
        """Sell stock. Advances the clock by one bar.

        Give exactly ONE of:
          shares   — how many shares to sell, <= your position. Fractional shares are fine.
          fraction — the share of your position to sell, 0 < f <= 1. Use fraction=1.0 to exit.

        Selling is denominated in SHARES, the thing you own, so the order is always
        executable. Proceeds are net of friction. {fill}
        """
        args = {}
        if shares is not None:
            args["shares"] = shares
        if fraction is not None:
            args["fraction"] = fraction
        return session.call("Sell", args)

    @mcp.tool(name="Wait")
    def wait(n: int = 1) -> dict:
        """Hold your current position for n bars (1 <= n <= 10) and advance the clock.

        The result carries every bar that elapsed, so waiting several bars costs one call,
        not n. A Wait that would run past the final bar is clamped to land on it, not
        rejected.
        """
        return session.call("Wait", {"n": n})

    @mcp.tool(name="fetchData")
    def fetch_data(lookback: int = 50) -> dict:
        """Return the last `lookback` daily bars up to and including today (max 200).

        Does NOT advance the clock. 200 warmup bars precede the scored window, so a full
        200-bar lookback is available from the very first tick.
        """
        return session.call("fetchData", {"lookback": lookback})

    @mcp.tool(name="getStats")
    def get_stats() -> dict:
        """Price, SMA-20, SMA-50, trailing 20-day annualized volatility, and the episode
        high/low. Does NOT advance the clock."""
        return session.call("getStats", {})

    @mcp.tool(name="ViewWallet")
    def view_wallet() -> dict:
        """Your cash balance. Does NOT advance the clock."""
        return session.call("ViewWallet", {})

    @mcp.tool(name="ViewPortfolio")
    def view_portfolio() -> dict:
        """Cash, shares, position value, equity, unrealized P&L and average cost.
        Does NOT advance the clock."""
        return session.call("ViewPortfolio", {})

    # The read cap and the anti-stall ladder are the reason a read is not free:
    # 8 reads in a tick, then READ_CAP_EXCEEDED; three consecutive invalid calls of any
    # kind and the engine takes the turn away and forces a Wait(1).
    return mcp


def session_from_env() -> EpisodeSession:
    """Build the episode this process serves. Missing config is a hard error: a server
    that silently defaults to window 0 would quietly produce a log for the wrong episode."""
    def need(key: str) -> str:
        v = os.environ.get(key)
        if not v:
            raise SystemExit(f"{key} is required (one server process = one episode)")
        return v

    run_dir = os.environ.get("PDTBENCH_RUN_DIR")
    agent: dict = {
        "id": need("PDTBENCH_AGENT_ID"),
        "kind": os.environ.get("PDTBENCH_AGENT_KIND", "llm"),
        "memory": os.environ.get("PDTBENCH_AGENT_MEMORY", "rolling_note"),
    }
    for env_key, field in (
        ("PDTBENCH_PROVIDER", "provider"),
        ("PDTBENCH_MODEL", "model"),
        ("PDTBENCH_SYSTEM_PROMPT_SHA256", "system_prompt_sha256"),
    ):
        if os.environ.get(env_key):
            agent[field] = os.environ[env_key]

    return EpisodeSession.start(
        window_id=need("PDTBENCH_WINDOW_ID"),
        track=need("PDTBENCH_TRACK"),
        agent=agent,
        episode_index=int(need("PDTBENCH_EPISODE_INDEX")),
        run_dir=Path(run_dir) if run_dir else None,
        run_id=os.environ.get("PDTBENCH_RUN_ID"),
        cfg=DEFAULT,
    )


def main() -> None:
    build_server(session_from_env()).run(transport="stdio")


if __name__ == "__main__":
    main()
