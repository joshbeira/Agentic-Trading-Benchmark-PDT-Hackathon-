"""The environment. It is the source of truth, and the agent is a guest in it.

The agent never holds state. It cannot tell us its balance, its position, or its
P&L — it can only ask, and act, and the engine decides what happened. Cash is
integer cents end to end, so no float can quietly manufacture a hundredth of a
dollar.

The two rules everything else hangs off:

**A decision at tick t fills at open_{t+1}.** Never at a price the agent has
already seen. This is enforced by `AuditedBars`, which raises on a read past the
current allowance, so look-ahead is structurally unavailable rather than merely
avoided.

**An invalid call does not move the clock.** It returns a structured error and
the agent tries again. Fumbling the interface costs time, not money (D13) — the
reliability scoreboard counts it, the trading scoreboard never sees it. An agent
that cannot stall forever, though: three consecutive invalid calls, or a ninth
read in one tick, and the engine takes the turn away and forces a Wait.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ENGINE_VERSION, Config
from . import metrics
from .audit import AuditedBars
from .tick_log import TickLogger
from .types import (
    TIME_ADVANCING,
    ErrorCode,
    Fill,
    Tool,
    ToolError,
    ToolResult,
)


class TradingEnv:
    def __init__(
        self,
        series: pd.DataFrame,
        window: dict,
        cfg: Config,
        log_path: Path | None = None,
        meta_extra: dict | None = None,
    ):
        self.cfg = cfg
        self.window = window
        self.bars = AuditedBars(series, cfg.n_warmup, cfg.n_scored)
        self._log_path = log_path
        self._meta_extra = meta_extra or {}

        self.t = 0
        self.done = False
        self.cash_cents = cfg.initial_capital_cents
        self.shares = 0.0
        self.cost_basis_cents = 0
        self.consecutive_invalid = 0
        self.reads_this_tick = 0
        self.equity_series: list[int] = []
        self.shares_by_tick: list[float] = []
        self.fills: list[dict] = []
        self.logger: TickLogger | None = None
        self._last_obs: dict | None = None
        self._elapsed: list[dict] = []
        self.summary: dict | None = None

    # ------------------------------------------------------------------ reset

    def reset(self) -> dict:
        self.t = 0
        self.done = False
        self.cash_cents = self.cfg.initial_capital_cents
        self.shares = 0.0
        self.cost_basis_cents = 0
        self.consecutive_invalid = 0
        self.equity_series = []
        self.shares_by_tick = []
        self.fills = []
        self.summary = None

        if self._log_path is not None:
            self.logger = TickLogger(self._log_path, self._meta())
        self._open_tick(0)
        return self._last_obs  # type: ignore[return-value]

    def _meta(self) -> dict:
        return {
            "episode_id": self._meta_extra.get("episode_id", "unnamed"),
            "run_id": self._meta_extra.get("run_id"),
            "episode_index": self._meta_extra.get("episode_index"),
            "agent": self._meta_extra.get("agent", {"id": "unknown", "kind": "unknown"}),
            "track": self._meta_extra.get("track", "unknown"),
            "window": {
                "window_id": self.window["window_id"],
                "source_ticker": self.window.get("ticker") or self.window.get("source_ticker"),
                "regime": self.window["regime"],
                "n_warmup": self.cfg.n_warmup,
                "n_scored": self.cfg.n_scored,
                "bh_daily_vol": self.window["bh_daily_vol"],
                "series_sha256": self.window["series_sha256"],
            },
            "config": self.cfg.log_block(),
            "dataset": self._meta_extra.get("dataset", {}),
            "seeds": self._meta_extra.get("seeds", {"master_seed": self.cfg.master_seed}),
            "code": {
                "git_commit": self._meta_extra.get("git_commit"),
                "engine_version": ENGINE_VERSION,
            },
            "started_at": _now(),
        }

    # ------------------------------------------------------------------- step

    def step(
        self,
        tool: str,
        args: dict | None = None,
        latency_ms: float | None = None,
        tokens: dict | None = None,
    ) -> ToolResult:
        args = dict(args or {})
        if self.done:
            return ToolResult(
                ok=False,
                tool=tool,
                error=ToolError(ErrorCode.EPISODE_OVER, "the episode has ended"),
            )
        try:
            t_enum = Tool(tool)
        except ValueError:
            return self._invalid(
                tool,
                args,
                ToolError(
                    ErrorCode.UNKNOWN_TOOL,
                    f"unknown tool '{tool}'; valid tools: {', '.join(t.value for t in Tool)}",
                ),
                latency_ms,
                tokens,
            )

        if t_enum in TIME_ADVANCING:
            return self._act(t_enum, args, latency_ms, tokens)
        return self._read(t_enum, args, latency_ms, tokens)

    # -------------------------------------------------------------- read path

    def _read(self, tool: Tool, args: dict, latency_ms, tokens) -> ToolResult:
        if self.reads_this_tick >= self.cfg.max_reads_per_tick:
            return self._invalid(
                tool,
                args,
                ToolError(
                    ErrorCode.READ_CAP_EXCEEDED,
                    f"{self.cfg.max_reads_per_tick} reads already this tick; "
                    "you must Buy, Sell, or Wait to advance the clock",
                ),
                latency_ms,
                tokens,
            )

        if tool is Tool.FETCH_DATA:
            lookback = args.get("lookback", self.cfg.fetch_lookback_default)
            if not _is_int(lookback) or lookback < 1:
                return self._invalid(
                    tool, args,
                    ToolError(ErrorCode.SCHEMA_ERROR, "lookback must be a positive integer"),
                    latency_ms, tokens,
                )
            n = min(int(lookback), self.cfg.fetch_lookback_cap)
            with self.bars.allow(self.t):
                data: dict[str, Any] = {"bars": self.bars.bars_through(self.t, n), "lookback": n}
            if n < lookback:
                data["note"] = f"lookback capped at {self.cfg.fetch_lookback_cap}"
        elif tool is Tool.GET_STATS:
            with self.bars.allow(self.t):
                data = {"stats": self._stats(self.t)}
        elif tool is Tool.VIEW_WALLET:
            data = {"cash_cents": self.cash_cents, "cash_usd": round(self.cash_cents / 100, 2)}
        elif tool is Tool.VIEW_PORTFOLIO:
            with self.bars.allow(self.t):
                data = {"portfolio": self._portfolio(self.bars.close(self.t))}
        else:  # pragma: no cover - TIME_ADVANCING is handled elsewhere
            raise AssertionError(tool)

        self.reads_this_tick += 1
        self.consecutive_invalid = 0
        if self.logger:
            self.logger.record_call(tool, args, ok=True, advanced_time=False,
                                    latency_ms=latency_ms, tokens=tokens)
        return ToolResult(ok=True, tool=tool, data=data)

    # ------------------------------------------------------------- act path

    def _act(self, tool: Tool, args: dict, latency_ms, tokens) -> ToolResult:
        if tool is Tool.BUY:
            parsed, err = self._parse_buy(args)
        elif tool is Tool.SELL:
            parsed, err = self._parse_sell(args)
        else:
            parsed, err = self._parse_wait(args)

        if err is not None:
            return self._invalid(tool, args, err, latency_ms, tokens)

        if self.logger:
            self.logger.record_call(tool, args, ok=True, advanced_time=True,
                                    latency_ms=latency_ms, tokens=tokens)
        self.consecutive_invalid = 0

        action = {"tool": str(tool), "args": args, "forced": False}
        if tool is Tool.WAIT:
            n = parsed["n_effective"]
            action["n_effective"] = n
            self._advance(action, fill=None, n=n, source_tick=self.t)
        else:
            fill_tick = self.t + 1
            with self.bars.allow(fill_tick):
                price = self.bars.open(fill_tick)
            fill = (
                self._buy_fill(price, parsed["notional_cents"], fill_tick)
                if tool is Tool.BUY
                else self._sell_fill(price, parsed["shares"], fill_tick, "sell")
            )
            self._advance(action, fill=fill, n=1)

        return ToolResult(ok=True, tool=tool, advanced_time=True, obs=self._agent_obs())

    def _invalid(self, tool, args, err: ToolError, latency_ms=None, tokens=None) -> ToolResult:
        if self.logger:
            self.logger.record_call(tool, args, ok=False, advanced_time=False,
                                    error=err, latency_ms=latency_ms, tokens=tokens)
        self.consecutive_invalid += 1

        if self.consecutive_invalid >= self.cfg.max_consecutive_invalid:
            # Anti-stall: take the turn away rather than let the episode deadlock.
            # Reachable only at t <= 88 (tick 89 is terminal), so one tick always fits.
            self.consecutive_invalid = 0
            self._advance(
                {"tool": str(Tool.WAIT), "args": {"n": 1}, "forced": True, "n_effective": 1},
                fill=None, n=1, source_tick=self.t,
            )
            return ToolResult(
                ok=False, tool=tool, error=err, advanced_time=True, forced_wait=True,
                obs=self._agent_obs(),
                note=(
                    f"{self.cfg.max_consecutive_invalid} consecutive invalid calls; the engine "
                    "forced Wait(1). No P&L penalty -- but you lost a decision."
                ),
            )
        return ToolResult(ok=False, tool=tool, error=err)

    # ------------------------------------------------------------ validation

    def _parse_buy(self, args: dict) -> tuple[dict, ToolError | None]:
        notional, fraction = args.get("notional_cents"), args.get("fraction")
        if (notional is None) == (fraction is None):
            return {}, ToolError(
                ErrorCode.SCHEMA_ERROR,
                "Buy takes exactly one of notional_cents (int) or fraction (0 < f <= 1 of cash)",
            )
        if fraction is not None:
            if not _is_num(fraction):
                return {}, ToolError(ErrorCode.SCHEMA_ERROR, "fraction must be a number")
            if _is_nan(fraction):
                return {}, ToolError(ErrorCode.NAN_QTY, "fraction is NaN")
            if not 0 < fraction <= 1:
                return {}, ToolError(ErrorCode.NONPOSITIVE_QTY, "fraction must satisfy 0 < f <= 1")
            notional = int(self.cash_cents * fraction)
        else:
            if _is_nan(notional):
                return {}, ToolError(ErrorCode.NAN_QTY, "notional_cents is NaN")
            if not _is_int(notional):
                return {}, ToolError(ErrorCode.SCHEMA_ERROR, "notional_cents must be an integer")
            notional = int(notional)

        if notional <= 0:
            return {}, ToolError(ErrorCode.NONPOSITIVE_QTY, f"notional_cents must be > 0, got {notional}")
        if notional > self.cash_cents:
            return {}, ToolError(
                ErrorCode.INSUFFICIENT_CASH,
                f"notional_cents {notional} exceeds cash {self.cash_cents}",
            )
        return {"notional_cents": notional}, None

    def _parse_sell(self, args: dict) -> tuple[dict, ToolError | None]:
        shares, fraction = args.get("shares"), args.get("fraction")
        if (shares is None) == (fraction is None):
            return {}, ToolError(
                ErrorCode.SCHEMA_ERROR,
                "Sell takes exactly one of shares (float) or fraction (0 < f <= 1 of position)",
            )
        if self.shares <= 0:
            return {}, ToolError(ErrorCode.INSUFFICIENT_SHARES, "no position to sell")

        if fraction is not None:
            if not _is_num(fraction):
                return {}, ToolError(ErrorCode.SCHEMA_ERROR, "fraction must be a number")
            if _is_nan(fraction):
                return {}, ToolError(ErrorCode.NAN_QTY, "fraction is NaN")
            if not 0 < fraction <= 1:
                return {}, ToolError(ErrorCode.NONPOSITIVE_QTY, "fraction must satisfy 0 < f <= 1")
            shares = self.shares * fraction
        else:
            if not _is_num(shares):
                return {}, ToolError(ErrorCode.SCHEMA_ERROR, "shares must be a number")
            if _is_nan(shares):
                return {}, ToolError(ErrorCode.NAN_QTY, "shares is NaN")
            if shares <= 0:
                return {}, ToolError(ErrorCode.NONPOSITIVE_QTY, f"shares must be > 0, got {shares}")
            shares = float(shares)

        # Position is reported to 6dp, so an agent selling "all" of what it was shown
        # can overshoot by a rounding crumb. Absorb that, and only that.
        if shares > self.shares:
            if shares - self.shares <= max(1e-6 * self.shares, 1e-9):
                shares = self.shares
            else:
                return {}, ToolError(
                    ErrorCode.INSUFFICIENT_SHARES,
                    f"shares {shares:.6f} exceeds position {self.shares:.6f}",
                )
        return {"shares": shares}, None

    def _parse_wait(self, args: dict) -> tuple[dict, ToolError | None]:
        n = args.get("n", 1)
        if _is_nan(n):
            return {}, ToolError(ErrorCode.NAN_QTY, "n is NaN")
        if not _is_int(n):
            return {}, ToolError(ErrorCode.SCHEMA_ERROR, "n must be an integer")
        n = int(n)
        if not 1 <= n <= self.cfg.max_wait:
            return {}, ToolError(
                ErrorCode.WAIT_OUT_OF_RANGE, f"n must satisfy 1 <= n <= {self.cfg.max_wait}, got {n}"
            )
        # Clamped, not rejected: waiting past the end is a harmless request, and an
        # error here would burn the agent's last decision on a technicality.
        return {"n_effective": min(n, (self.cfg.n_scored - 1) - self.t)}, None

    # ---------------------------------------------------------------- filling

    def _buy_fill(self, price: float, notional_cents: int, fill_tick: int) -> Fill:
        friction = int(round(notional_cents * self.cfg.friction_rate))
        invested = notional_cents - friction
        return Fill(
            side="buy",
            fill_tick=fill_tick,
            fill_price=price,
            shares_delta=round((invested / 100.0) / price, self.cfg.share_decimals),
            gross_notional_cents=notional_cents,
            friction_cents=friction,
            cash_delta_cents=-notional_cents,
        )

    def _sell_fill(self, price: float, shares: float, fill_tick: int, side: str) -> Fill:
        shares = round(shares, self.cfg.share_decimals)
        gross = int(round(shares * price * 100))
        friction = int(round(gross * self.cfg.friction_rate))
        return Fill(
            side=side,
            fill_tick=fill_tick,
            fill_price=price,
            shares_delta=-shares,
            gross_notional_cents=gross,
            friction_cents=friction,
            cash_delta_cents=gross - friction,
        )

    def _apply(self, f: Fill) -> None:
        self.cash_cents += f.cash_delta_cents
        if f.shares_delta > 0:
            self.shares = round(self.shares + f.shares_delta, self.cfg.share_decimals)
            self.cost_basis_cents += f.gross_notional_cents
        else:
            sold = -f.shares_delta
            before = self.shares
            frac = min(sold / before, 1.0) if before > 0 else 1.0
            self.shares = round(max(before - sold, 0.0), self.cfg.share_decimals)
            self.cost_basis_cents -= int(round(self.cost_basis_cents * frac))
        if self.shares <= 0:
            self.shares = 0.0
            self.cost_basis_cents = 0
        self.fills.append(f.as_dict())

    # ----------------------------------------------------------------- clock

    def _advance(self, action: dict, fill: Fill | None, n: int, source_tick: int | None = None) -> None:
        """Close the current tick, then move `n` ticks, marking equity at each close.

        The fill is computed at decision time (it needs `open_{t+1}`) but *applied*
        only after the clock moves, so the mark at close_t never sees it.
        """
        if self.logger:
            self.logger.end_tick(action, fill)
        self._elapsed = []
        pending = fill

        for k in range(n):
            self.t += 1
            if pending is not None:
                self._apply(pending)
                pending = None

            terminal = self.t >= self.cfg.n_scored - 1
            is_last = k == n - 1
            self._open_tick(
                self.t,
                decision_point=is_last and not terminal,
                skipped_by=None if is_last else source_tick,
            )
            if terminal:
                self._terminate()
                return
            if not is_last and self.logger:
                self.logger.end_tick(action=None, fill=None)

    def _open_tick(self, t: int, decision_point: bool = True, skipped_by: int | None = None) -> None:
        with self.bars.allow(t):
            bar = self.bars.bar(t)
            obs = {
                "tick": t,
                "ticks_remaining": (self.cfg.n_scored - 1) - t,
                "bar": bar,
                "stats": self._stats(t),
                "portfolio": self._portfolio(bar["close"]),
            }
        equity = obs["portfolio"]["equity_cents"]

        self.equity_series.append(equity)
        self.shares_by_tick.append(self.shares)
        self.reads_this_tick = 0
        self._last_obs = obs
        self._elapsed.append({"t": t, **bar})
        if self.logger:
            self.logger.begin_tick(t, obs, equity, decision_point=decision_point,
                                   skipped_by=skipped_by)

    def _terminate(self) -> None:
        """Force-liquidate at close_89, paying the standard friction.

        Every agent's P&L is therefore realized, and nobody earns a hidden discount
        by holding into the final bar to dodge an exit fee. The fill price is one
        the agent has seen, but it has no choice about it, so there is nothing to
        exploit.
        """
        last = self.cfg.n_scored - 1
        with self.bars.allow(last):
            close = self.bars.close(last)

        fill = None
        if self.shares > 0:
            fill = self._sell_fill(close, self.shares, fill_tick=last, side="liquidation")
            self._apply(fill)

        equity = self.cash_cents
        self.equity_series[-1] = equity
        self.shares_by_tick[-1] = self.shares
        portfolio = self._portfolio(close)
        if self._last_obs is not None:
            self._last_obs["portfolio"] = portfolio
            self._last_obs["done"] = True

        if self.logger:
            self.logger.amend_tick(equity_cents=equity, obs_patch={"portfolio": portfolio})
            self.logger.end_tick(action=None, fill=fill)

        self.done = True
        self.summary = self.compute_metrics()

    # --------------------------------------------------------------- readouts

    def compute_metrics(self) -> dict:
        return metrics.compute(
            self.equity_series,
            self.fills,
            self.shares_by_tick,
            self.window["bh_daily_vol"],
            self.cfg,
        )

    def close_log(self, memory_note_in=None, memory_note_out=None, status="ok") -> dict | None:
        if not self.logger:
            return None
        last = self.cfg.n_scored - 1
        liq = next((f for f in self.fills if f["side"] == "liquidation"), None)
        terminal = {
            "liquidated_shares": -liq["shares_delta"] if liq else 0.0,
            "liquidation_price": liq["fill_price"] if liq else None,
            "liquidation_friction_cents": liq["friction_cents"] if liq else 0,
            "final_equity_cents": self.equity_series[-1],
        }
        return self.logger.close(
            t_final=last,
            terminal=terminal,
            equity_series_cents=self.equity_series,
            metrics=self.summary or self.compute_metrics(),
            memory_note_in=memory_note_in,
            memory_note_out=memory_note_out,
            status=status,
        )

    def _agent_obs(self) -> dict:
        obs = dict(self._last_obs or {})
        if len(self._elapsed) > 1:
            obs["elapsed_bars"] = self._elapsed
        obs["done"] = self.done
        if self.done and self.summary:
            obs["episode_summary"] = self.summary
        return obs

    def _portfolio(self, close: float) -> dict:
        # `self.shares` is already held at `share_decimals`, so what is displayed and
        # what is marked are the same number. That is what lets the log reproduce the
        # equity mark exactly, and lets an agent sell precisely what it was shown.
        mv = int(round(self.shares * close * 100))
        return {
            "cash_cents": self.cash_cents,
            "shares": self.shares,
            "position_value_cents": mv,
            "equity_cents": self.cash_cents + mv,
            "unrealized_pnl_cents": mv - self.cost_basis_cents,
            "avg_cost": round(self.cost_basis_cents / 100 / self.shares, 4) if self.shares > 0 else None,
        }

    def _stats(self, t: int) -> dict:
        """Everything an agent needs without pulling raw history (D-interface)."""
        close = self.bars.close(t)
        c50 = self.bars.closes_through(t, 50)
        c20 = self.bars.closes_through(t, 20)
        c21 = self.bars.closes_through(t, 21)
        r = c21[1:] / c21[:-1] - 1.0
        vol = float(np.std(r, ddof=1)) * math.sqrt(self.cfg.trading_days_per_year) if len(r) > 1 else 0.0
        hi, lo = self.bars.hi_lo_through(t, from_tick=0) if t >= 0 else (close, close)
        return {
            "price": close,
            "sma20": round(float(np.mean(c20)), 4),
            "sma50": round(float(np.mean(c50)), 4),
            "vol20_ann": round(vol, 4),
            "ep_high": hi,
            "ep_low": lo,
            "tick": t,
            "ticks_remaining": (self.cfg.n_scored - 1) - t,
        }


# --------------------------------------------------------------------- helpers


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    return isinstance(x, float) and float(x).is_integer()


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
