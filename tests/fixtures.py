"""Hand-built tick logs, written straight from `schemas/tick_log.md`.

Nothing here imports the engine. That is the point twice over:

1. The analytics layer must be testable without running a simulation, so a broken engine
   cannot mask a broken scoreboard (or the reverse).
2. Building a valid log from the schema *alone* is the only real test of whether the schema
   is sufficient. Every field these fixtures needed and could not find became one of the
   six questions in `schemas/analysis_artifacts.md`.

The generator plants **exact** Sharpes. Given a target `S` and a chosen return volatility
`sigma`, the return series is standardized and rescaled to have precisely
`mean = S * sigma / sqrt(252)` and `std = sigma`, so the floored Sharpe comes out to `S`
on the nose (as long as `sigma` clears the floor). That lets a test assert the scoreboard
recovers a planted value rather than asserting it recovers *something*. The only slack is
integer-cent rounding of the equity curve, which perturbs Sharpe at the 1e-4 level.

These logs are internally consistent (`equity == cash + shares x close`) but are not
replay-valid: no parquet backs them, so fill prices are synthetic. Replay is the engine's
test, not the scoreboard's.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Imported, not pinned. A second copy of this literal drifted to "1.2.0" once already,
# silently, because `meta.schema_version` is an unconstrained `Str()` that nothing
# cross-checks against the live schema. One import is what makes that drift impossible
# instead of merely unlikely.
from pdtbench.schema import SCHEMA_VERSION


def stable_seed(*parts) -> int:
    """A seed that does not move between processes.

    Python's `hash()` on strings is salted per interpreter (PYTHONHASHSEED), so seeding an
    RNG with it produces different fixture data on every run — which is not a fixture, it is
    a random number generator with extra steps. This flaked a learning-slope assertion once
    before it was caught.
    """
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:8], 16)

INITIAL_CENTS = 1_000_000
N_WARMUP, N_SCORED = 200, 90
TRADING_DAYS = 252

CONFIG = {
    "config_sha256": "fixture0000000000000000000000000000000000000000000000000000000000",
    "initial_capital_cents": INITIAL_CENTS,
    "fee_bps": 2.0,
    "slippage_bps": 8.0,
    "friction_bps_per_side": 10.0,
    "fill_rule": "next_open",
    "terminal_rule": "liquidate_at_final_close",
    "max_wait": 10,
    "max_reads_per_tick": 8,
    "max_consecutive_invalid": 3,
    "fetch_lookback_default": 50,
    "fetch_lookback_cap": 200,
    "episode_wallclock_cap_s": 1200,
    "vol_floor_multiple": 0.25,
    "trading_days_per_year": TRADING_DAYS,
    "n_warmup": N_WARMUP,
    "n_scored": N_SCORED,
}

FRICTION_RATE = CONFIG["friction_bps_per_side"] / 10_000


# ------------------------------------------------------------------ return synthesis


def returns_for_sharpe(target: float, sigma: float, n: int = N_SCORED - 1, seed: int = 0):
    """A return series with *exactly* the requested Sharpe and volatility."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    r = (r - r.mean()) / r.std(ddof=1)
    return r * sigma + target * sigma / math.sqrt(TRADING_DAYS)


def equity_from_returns(r) -> list[int]:
    e = INITIAL_CENTS * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(r))])
    return [int(round(x)) for x in e]


# --------------------------------------------------------------------- log building


def make_episode(
    path: Path,
    *,
    agent: dict,
    track: str,
    window_id: str,
    regime: str,
    source_ticker: str,
    bh_daily_vol: float,
    episode_index: int,
    equity_cents: list[int],
    exposure: float = 1.0,
    status: str = "ok",
    run_id: str = "fixture_run",
    invalid_calls_per_tick: int = 0,
    reads_per_tick: int = 0,
    forced_waits: int = 0,
    prose_nudges: int = 0,
    memory_note_in: str | None = None,
    memory_note_out: str | None = None,
    corrupt_cached_metrics: dict | None = None,
    n_trades: int = 2,
) -> Path:
    """Write one schema-conforming episode log.

    `exposure` is the fraction of equity held in the asset on every in-market tick, which
    fixes the shares/cash split. Tick 0 and tick 89 are flat by construction — the first
    fill cannot land before `open_1`, and tick 89 is post-liquidation — so
    `time_in_market` tops out at 88/90, exactly as the schema says.
    """
    assert len(equity_cents) == N_SCORED

    # A price path, purely to make obs internally consistent. Anchored at 100 like a
    # masked series, and walking with the equity curve so the numbers look like a market.
    closes = [100.0]
    for i in range(1, N_SCORED):
        step = equity_cents[i] / equity_cents[i - 1] - 1.0
        closes.append(round(closes[-1] * (1 + step / max(exposure, 1e-9)), 2))

    shares, cash = [], []
    for t in range(N_SCORED):
        in_market = exposure > 0 and 1 <= t <= N_SCORED - 2
        s = round(exposure * equity_cents[t] / 100.0 / closes[t], 6) if in_market else 0.0
        shares.append(s)
        cash.append(equity_cents[t] - int(round(s * closes[t] * 100)))

    fills = _synth_fills(shares, closes, n_trades)

    records: list[dict] = [
        {
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "episode_id": f"{agent['id']}__{track}__{window_id}",
            "episode_index": episode_index,
            "agent": agent,
            "track": track,
            "window": {
                "window_id": window_id,
                "source_ticker": source_ticker,
                "regime": regime,
                "n_warmup": N_WARMUP,
                "n_scored": N_SCORED,
                "bh_daily_vol": bh_daily_vol,
                "series_sha256": f"fixture_{track}_{window_id}",
            },
            "config": CONFIG,
            "dataset": {"dataset_sha256": "fixture_dataset", "as_of": "2026-07-14"},
            "seeds": {"master_seed": 20260713, "window_seed": 1, "twin_seed": None},
            "code": {"git_commit": "fixture", "engine_version": "1.0.0"},
            "started_at": "2026-07-14T09:30:00.000Z",
        }
    ]

    is_llm = agent["kind"] == "llm"
    fills_by_tick = {f["_at_tick"]: f for f in fills}
    for t in range(N_SCORED):
        last = t == N_SCORED - 1
        f = fills_by_tick.get(t)
        fill = {k: v for k, v in f.items() if not k.startswith("_")} if f else None
        forced = t < forced_waits

        # The accepted `action` must BE the call that advanced time -- and on a forced Wait
        # there is no such call at all, because a forced Wait is the engine taking the turn
        # away, not something the agent asked for. The fixtures used to get both of these
        # wrong: they logged a Sell as the action while the advancing call was a Wait, and
        # they gave forced waits an advancing call the engine never writes.
        if last:
            action, calls = None, []  # the terminal tick accepts nothing
        elif forced:
            action = {"tool": "Wait", "args": {"n": 1}, "forced": True, "n_effective": 1}
            calls = _synth_calls(invalid_calls_per_tick, reads_per_tick, advancing=None,
                                 with_tokens=is_llm)
        elif fill and fill["side"] != "liquidation":
            tool = "Buy" if fill["shares_delta"] > 0 else "Sell"
            args = {"fraction": 1.0}
            action = {"tool": tool, "args": args, "forced": False}
            calls = _synth_calls(invalid_calls_per_tick, reads_per_tick,
                                 advancing=(tool, args), with_tokens=is_llm)
        else:
            args = {"n": 1}
            action = {"tool": "Wait", "args": args, "forced": False, "n_effective": 1}
            calls = _synth_calls(invalid_calls_per_tick, reads_per_tick,
                                 advancing=("Wait", args), with_tokens=is_llm)

        mv = int(round(shares[t] * closes[t] * 100))
        records.append(
            {
                "type": "tick",
                "t": t,
                "decision_point": not last,
                "obs": {
                    "tick": t,
                    "ticks_remaining": (N_SCORED - 1) - t,
                    # Long keys. The engine has always written these; the schema and these
                    # fixtures said {o,h,l,c,v}, and nothing caught it because the analytics
                    # never read obs.bar. The first consumer that did would have crashed.
                    "bar": {"open": closes[t], "high": closes[t], "low": closes[t],
                            "close": closes[t], "volume": 1.0},
                    "stats": {"price": closes[t], "sma20": closes[t], "sma50": closes[t],
                              "vol20_ann": bh_daily_vol * math.sqrt(TRADING_DAYS),
                              "ep_high": max(closes[: t + 1]), "ep_low": min(closes[: t + 1]),
                              "tick": t, "ticks_remaining": (N_SCORED - 1) - t},
                    "portfolio": {
                        "cash_cents": cash[t],
                        "shares": shares[t],
                        "position_value_cents": mv,
                        "equity_cents": equity_cents[t],
                        "unrealized_pnl_cents": 0,
                        "avg_cost": round(closes[t], 4) if shares[t] > 0 else None,
                    },
                    "done": last,
                },
                "calls": calls,
                "action": action,
                "fill": fill,
                "equity_cents": equity_cents[t],
                "invalid_count": sum(1 for c in calls if not c["ok"]),
                "reads_count": sum(1 for c in calls if c["ok"] and not c["advanced_time"]),
            }
        )

    clean_fills = [{k: v for k, v in f.items() if not k.startswith("_")} for f in fills]
    metrics = _metrics(equity_cents, clean_fills, shares, bh_daily_vol)
    n_calls = sum(len(r["calls"]) for r in records if r["type"] == "tick")
    n_invalid = sum(r["invalid_count"] for r in records if r["type"] == "tick")
    liq = next((f for f in clean_fills if f["side"] == "liquidation"), None)

    records.append(
        {
            "type": "episode_end",
            "t_final": N_SCORED - 1,
            "terminal": {
                "liquidated_shares": -liq["shares_delta"] if liq else 0.0,
                "liquidation_price": liq["fill_price"] if liq else None,
                "liquidation_friction_cents": liq["friction_cents"] if liq else 0,
                "final_equity_cents": equity_cents[-1],
            },
            "equity_series_cents": equity_cents,
            "metrics": corrupt_cached_metrics or metrics,
            "reliability": {
                "n_calls": n_calls,
                "n_invalid": n_invalid,
                "invalid_rate": n_invalid / n_calls if n_calls else 0.0,
                "n_schema_errors": 0,
                "n_forced_waits": forced_waits,
                "n_prose_nudges": prose_nudges,
                "n_read_cap_hits": 0,
                "wallclock_s": 120.0,
                "hit_wallclock_cap": status == "wallclock_capped",
            },
            "memory_note_in": memory_note_in,
            "memory_note_out": memory_note_out,
            "cost": {"tokens_in": 3000 * N_SCORED, "tokens_out": 150 * N_SCORED},
            "status": status,
            "ended_at": "2026-07-14T09:38:00.000Z",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n")
    return path


def _synth_fills(shares: list[float], closes: list[float], n_trades: int) -> list[dict]:
    """Fills consistent with the shares path, plus the terminal liquidation.

    `fill_tick` is `t + 1` on every agent fill and `t` on the liquidation, exactly as the
    schema requires — the fixture would be lying about the design otherwise.
    """
    fills: list[dict] = []
    if max(shares) == 0:
        return fills

    entry_shares = shares[1]
    gross = int(round(entry_shares * closes[1] * 100))
    fills.append({
        "_at_tick": 0, "side": "buy", "fill_tick": 1, "fill_price": closes[1],
        "shares_delta": entry_shares, "gross_notional_cents": gross,
        "friction_cents": int(round(gross * FRICTION_RATE)),
        "cash_delta_cents": -gross,
    })

    # Extra round trips purely so turnover and n_trades are non-degenerate.
    for k in range(1, n_trades):
        t = 10 + k * 7
        if t >= N_SCORED - 3:
            break
        g = int(round(shares[t] * closes[t + 1] * 100))
        fills.append({
            "_at_tick": t, "side": "buy", "fill_tick": t + 1, "fill_price": closes[t + 1],
            "shares_delta": 0.0, "gross_notional_cents": g,
            "friction_cents": int(round(g * FRICTION_RATE)), "cash_delta_cents": 0,
        })

    last = N_SCORED - 1
    liq_shares = shares[last - 1]
    g = int(round(liq_shares * closes[last] * 100))
    fills.append({
        "_at_tick": last, "side": "liquidation", "fill_tick": last, "fill_price": closes[last],
        "shares_delta": -liq_shares, "gross_notional_cents": g,
        "friction_cents": int(round(g * FRICTION_RATE)),
        "cash_delta_cents": g - int(round(g * FRICTION_RATE)),
    })
    return fills


def _synth_calls(n_invalid: int, n_reads: int, advancing: tuple[str, dict] | None,
                 with_tokens: bool = True) -> list[dict]:
    """The calls made during one tick.

    `advancing` is the accepted action, appended last — or None on a forced Wait, where the
    engine writes no advancing call because the agent never made one.

    Tokens live on the **call**, not on the tick. One LLM completion produces one tool call,
    so that is where a token count is unambiguous. A tick can hold several calls (reads, a
    retry after an invalid one), which made the tick-level `tokens` the schema used to show
    ambiguous: a sum, or the last one? The episode total lives in `episode_end.cost`.
    """
    tok = {"in": 3000, "out": 150} if with_tokens else None

    def call(seq, tool, args, ok, advanced, error=None):
        c = {"seq": seq, "tool": tool, "args": args, "ok": ok,
             "advanced_time": advanced, "latency_ms": 800.0}
        if error:
            c["error"] = error
        if tok:
            c["tokens"] = dict(tok)
        return c

    calls, seq = [], 0
    for _ in range(n_reads):
        calls.append(call(seq, "getStats", {}, True, False))
        seq += 1
    for _ in range(n_invalid):
        calls.append(call(seq, "Buy", {"notional_cents": -1}, False, False,
                          {"code": "NONPOSITIVE_QTY", "message": "must be > 0"}))
        seq += 1
    if advancing is not None:
        tool, args = advancing
        calls.append(call(seq, tool, args, True, True))
    return calls


def _metrics(equity: list[int], fills: list[dict], shares: list[float], bh_vol: float) -> dict:
    """Computed here from the schema's definition table, not imported from the engine."""
    e = np.asarray(equity, dtype=float)
    r = e[1:] / e[:-1] - 1.0
    std = float(np.std(r, ddof=1))
    floor = CONFIG["vol_floor_multiple"] * bh_vol
    sq = math.sqrt(TRADING_DAYS)
    return {
        "total_return": float(e[-1] / e[0] - 1),
        "sharpe_floored": float(np.mean(r) / max(std, floor) * sq) if max(std, floor) > 0 else 0.0,
        "sharpe_raw": float(np.mean(r) / std * sq) if std > 0 else 0.0,
        "vol_floor_binding": bool(std < floor),
        "realized_vol_ann": std * sq,
        "max_drawdown": float(np.min(e / np.maximum.accumulate(e) - 1)),
        "turnover": sum(f["gross_notional_cents"] for f in fills) / e[0],
        "fees_paid_cents": int(sum(f["friction_cents"] for f in fills)),
        "time_in_market": float(np.mean([s > 0 for s in shares])),
        "n_trades": sum(1 for f in fills if f["side"] != "liquidation"),
    }


# ------------------------------------------------------------------------ whole runs

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "PG", "NVDA", "KO", "CAT", "DIS", "WMT",
           "BAC", "CVX", "MRK", "PEP", "HD", "INTC", "T", "MCD", "NKE", "IBM",
           "AMD", "ORCL", "CSCO", "TXN", "QCOM", "GS", "MS", "UNH", "LOW", "TGT"]
YEARS = [str(y) for y in range(2005, 2027)]
REGIMES = ["bull", "bear", "chop"]


def window_specs(n: int = 30) -> list[dict]:
    """30 windows, regimes interleaved — the presentation order the design mandates (D11),
    so that a learning curve cannot be an artifact of seeing all ten bulls first."""
    rng = np.random.default_rng(42)
    out = []
    for i in range(n):
        out.append({
            "window_id": f"w{i:02d}",
            "regime": REGIMES[i % 3],
            "source_ticker": TICKERS[i % len(TICKERS)],
            "period": YEARS[3 + (i * 7) % (len(YEARS) - 3)],
            "bh_daily_vol": float(rng.uniform(0.010, 0.030)),
            "difficulty": float(rng.normal(0, 0.6)),  # the signal a baseline trace reveals
        })
    return out


def make_run(
    root: Path,
    *,
    agents: list[dict],
    windows: list[dict],
    sharpe_fn,
    run_id: str = "fixture_run",
    status_fn=lambda a, t, w, i: "ok",
    exposure_fn=lambda a: 1.0,
) -> Path:
    """A whole run directory: manifest + one log per (agent, track, window).

    `sharpe_fn(agent, track, window, episode_index) -> float` is what a test plants.
    `exposure_fn(agent) -> float` sets how much of equity sits in the asset, which is what
    makes a never-trading agent actually look like one (0% time in market, no fees).
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": "2026-07-14T09:30:00.000Z",
        "config": CONFIG,
        "dataset": {"dataset_sha256": "fixture_dataset", "as_of": "2026-07-14"},
        "windows_manifest_sha256": "fixture_windows",
        "agents": agents,
        "tracks": ["real", "twin"],
        "presentation_order": [w["window_id"] for w in windows],
        "windows": [{"window_id": w["window_id"], "regime": w["regime"],
                     "bh_daily_vol": w["bh_daily_vol"]} for w in windows],
        "probe": {"n_options": 8, "n_reps": 5, "period_granularity": "year",
                  "chance_level": 0.125},
        "seeds": {"master_seed": 20260713},
    }, indent=2))

    for agent in agents:
        exposure = exposure_fn(agent)
        for track in ("real", "twin"):
            for i, w in enumerate(windows):
                s = sharpe_fn(agent, track, w, i) if exposure > 0 else 0.0
                sigma = w["bh_daily_vol"]  # clears the floor, so planted Sharpe is exact
                equity = (
                    equity_from_returns(
                        returns_for_sharpe(s, sigma, seed=stable_seed(agent["id"], track, i))
                    )
                    if exposure > 0
                    else [INITIAL_CENTS] * N_SCORED
                )
                make_episode(
                    root / "episodes" / f"{agent['id']}__{track}__{w['window_id']}.jsonl",
                    agent=agent,
                    track=track,
                    window_id=w["window_id"],
                    regime=w["regime"],
                    source_ticker=w["source_ticker"],
                    bh_daily_vol=w["bh_daily_vol"],
                    episode_index=i,
                    equity_cents=equity,
                    exposure=exposure,
                    n_trades=2 if exposure > 0 else 0,
                    run_id=run_id,
                    status=status_fn(agent, track, w, i),
                    memory_note_in=None if i == 0 else "prior lesson",
                    memory_note_out="lesson",
                )
    return root


def agent_spec(agent_id: str, kind: str = "llm", memory: str = "rolling_note") -> dict:
    d = {"id": agent_id, "kind": kind, "memory": memory}
    if kind == "llm":
        d |= {"provider": "anthropic", "model": f"model-{agent_id}", "temperature": None,
              "system_prompt_sha256": "prompt0000"}
    return d
