# Tick-Log Schema (JSONL) — v1.0.0

The tick log is the **only** artifact the scoreboard is allowed to read. Every number we report — median Sharpe, drawdown, turnover, fees, invalid-call rate, learning curve, leakage scatter — must be regenerable from these files and nothing else. The UI reads them too, which is why live mode and replay mode are the same code path (D14).

One episode = one JSONL file. One JSON object per line. Records are typed by a `type` field and appear in strict order:

```
meta            exactly one, first line
tick            one per tick, t = 0..89, in ascending order
episode_end     exactly one, last line
```

Layout on disk:

```
runs/{run_id}/
  run_manifest.json                                  # config, dataset hash, model list, window list
  episodes/{agent_id}__{track}__{window_id}.jsonl    # one per episode
  probe/{agent_id}__{track}__{window_id}.json        # leakage-probe responses (separate harness)
```

Floats are JSON numbers. **Money is always integer cents** (`*_cents`) — never a float, so there is no floating-point exploit and no rounding drift in the equity series. Share counts are floats (fractional shares are permitted). Timestamps are ISO-8601 UTC.

---

## 1. `meta` — first line

Pins *everything* needed to reconstruct the episode: which data, which config, which seeds, which code.

```json
{
  "type": "meta",
  "schema_version": "1.0.0",
  "run_id": "20260713T2240Z_a1b2c3d4",
  "episode_id": "flagship__real__w07",
  "episode_index": 6,

  "agent": {
    "id": "flagship",
    "kind": "llm",
    "provider": "anthropic",
    "model": "claude-...",
    "temperature": null,
    "memory": "rolling_note",
    "system_prompt_sha256": "9f2c..."
  },

  "track": "real",

  "window": {
    "window_id": "w07",
    "source_ticker": "AAPL",
    "regime": "bull",
    "n_warmup": 200,
    "n_scored": 90,
    "bh_daily_vol": 0.01734,
    "series_sha256": "4d51..."
  },

  "config": {
    "config_sha256": "7ae0...",
    "initial_capital_cents": 1000000,
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
    "vol_floor_multiple": 0.25
  },

  "dataset": {
    "dataset_sha256": "e3b0...",
    "parquet_path": "data/processed/prices.parquet",
    "as_of": "2026-07-13"
  },

  "seeds": {
    "master_seed": 20260713,
    "window_seed": 4471,
    "twin_seed": null,
    "agent_seed": null
  },

  "code": { "git_commit": "2ed6419", "engine_version": "1.0.0" },
  "started_at": "2026-07-13T22:40:01.123Z"
}
```

| field | meaning |
|---|---|
| `track` | `"real"` (masked-real window) or `"twin"` (its block-bootstrap twin, D9) |
| `window.window_id` | A twin shares its source window's id — **that is what makes the pair a pair**. |
| `window.series_sha256` | Hash of the exact 290-bar OHLCV array actually used. Pins the data even if the pipeline is re-run. |
| `window.bh_daily_vol` | Precomputed vol-floor reference (D2). A property of the window, identical for every agent. |
| `seeds.twin_seed` | `null` on the real track; set on the twin track. |
| `config.config_sha256` | Hash of the whole config block — one field to compare across runs. |

---

## 2. `tick` — one per tick, `t = 0..89`

The unit of replay. Contains what the agent **saw**, what it **called** (including the calls that failed), what it **did**, what **filled**, and what the position was **worth**.

```json
{
  "type": "tick",
  "t": 12,
  "decision_point": true,

  "obs": {
    "bar": { "o": 103.21, "h": 104.88, "l": 102.90, "c": 104.10, "v": 1.42 },
    "stats": {
      "price": 104.10, "sma20": 101.70, "sma50": 99.40,
      "vol20_ann": 0.2131, "ep_high": 106.20, "ep_low": 97.30
    },
    "portfolio": {
      "cash_cents": 500000,
      "shares": 4.75123,
      "equity_cents": 994593,
      "unrealized_pnl_cents": -5407
    }
  },

  "calls": [
    { "seq": 0, "tool": "getStats", "args": {}, "ok": true,  "advanced_time": false, "latency_ms": 830 },
    { "seq": 1, "tool": "Buy", "args": { "notional_cents": 2000000 }, "ok": false,
      "advanced_time": false, "latency_ms": 910,
      "error": { "code": "INSUFFICIENT_CASH", "message": "notional_cents 2000000 exceeds cash 500000" } },
    { "seq": 2, "tool": "Buy", "args": { "notional_cents": 500000 }, "ok": true,
      "advanced_time": true, "latency_ms": 870 }
  ],

  "action": { "tool": "Buy", "args": { "notional_cents": 500000 }, "forced": false },

  "fill": {
    "side": "buy",
    "fill_tick": 13,
    "fill_price": 104.55,
    "shares_delta": 4.77561,
    "gross_notional_cents": 500000,
    "friction_cents": 500,
    "cash_delta_cents": -500000
  },

  "equity_cents": 994593,
  "invalid_count": 1,
  "reads_count": 1,
  "tokens": { "in": 3120, "out": 190 }
}
```

**Timing — the part that must not be misread.**

- `obs` is the state at the **close of bar `t`**. It contains **nothing from bar `t+1` or later**. This is enforced by a read-audit wrapper in the engine, not by convention: while the observation is being built, a read of any bar index > `t` raises.
- `equity_cents` is the mark at the **close of bar `t`**: `cash_t + shares_t × close_t`. The sequence of `equity_cents` over `t = 0..89` **is** the equity curve that every metric is computed from. It is also mirrored inside `obs.portfolio.equity_cents`; the replay test asserts the two agree.
- `fill` describes the consequence of the action chosen at tick `t`, which lands at **`open_{t+1}`** — so `fill.fill_tick == t + 1` for every agent-initiated fill. It is *not* reflected in this record's `equity_cents`; it shows up in tick `t+1`'s.
- `fill` is `null` on a Wait.
- **One exception, and only one:** the terminal liquidation on tick 89 carries `side: "liquidation"`, `fill_tick: 89`, and prices at `close_89` rather than at an open. It is the single fill that executes at a price the agent has already seen — and it is unconditional, identical for every agent, and not a decision, so there is nothing there to exploit. Replay checks it against `close_89` and checks every other fill against `open_{t+1}`.

**Share precision.** `shares` is held to 6 decimal places — *exactly* the precision at which it is displayed. This is not cosmetic. When the engine briefly kept more precision internally than it published, the log no longer contained enough information to reproduce its own equity mark (`round(shares x close)` differed by a cent whenever the product landed near a rounding boundary), and an agent that sold precisely the position it had been shown was left holding 4.4e-07 shares of dust. What is displayed and what is marked must be the same number.

**Fields.**

| field | meaning |
|---|---|
| `decision_point` | `false` for ticks skipped inside a `Wait(n)` (D4). Those records carry `calls: []`, `action: null`, `skipped_by: <source tick>`, and still carry `obs` + `equity_cents` — the equity curve stays one-entry-per-tick, which is what keeps replay simple. |
| `calls` | **Every** tool call at this tick, in order, including invalid ones. This array is the entire input to the reliability scoreboard. `advanced_time: true` on exactly one call per decision tick (the accepted action) and never on a read or an error. |
| `action` | The call that was accepted. `forced: true` when the engine imposed a `Wait(1)` via the anti-stall ladder (3 consecutive invalids, read cap, or an un-nudgeable prose reply). On a `Wait`, `n_effective` records how many ticks it actually consumed — a `Wait(n)` that would run past the final bar is clamped to land on it rather than rejected, since erroring there would burn an agent's last decision on a technicality. |
| `friction_cents` | The full 10 bps/side, deterministic (D10). Buy: deducted from the notional before shares are computed. Sell: deducted from the proceeds. |
| `invalid_count` / `reads_count` | Counts at this tick — redundant with `calls`, kept because the analytics layer plucks them constantly. Replay asserts they agree with `calls`. |
| `tokens` | `null` for scripted baselines. |

**Error codes** (closed set; anything else is a bug):
`SCHEMA_ERROR`, `UNKNOWN_TOOL`, `NONPOSITIVE_QTY`, `NAN_QTY`, `INSUFFICIENT_CASH`, `INSUFFICIENT_SHARES`, `WAIT_OUT_OF_RANGE`, `READ_CAP_EXCEEDED`, `EPISODE_OVER`.

---

## 3. `episode_end` — last line

```json
{
  "type": "episode_end",
  "t_final": 89,

  "terminal": {
    "liquidated_shares": 4.77561,
    "liquidation_price": 108.20,
    "liquidation_friction_cents": 517,
    "final_equity_cents": 1043210
  },

  "equity_series_cents": [1000000, 1000000, 998412, "... 90 values, E_0..E_89 ..."],

  "metrics": {
    "total_return": 0.04321,
    "sharpe_floored": 1.4412,
    "sharpe_raw": 1.8103,
    "vol_floor_binding": false,
    "max_drawdown": -0.0612,
    "turnover": 2.41,
    "fees_paid_cents": 2040,
    "time_in_market": 0.6180,
    "n_trades": 6
  },

  "reliability": {
    "n_calls": 118, "n_invalid": 4, "invalid_rate": 0.0339,
    "n_schema_errors": 1, "n_forced_waits": 1, "n_prose_nudges": 0,
    "n_read_cap_hits": 0, "wallclock_s": 512, "hit_wallclock_cap": false
  },

  "memory_note_in": "…≤500 tokens, carried in from the previous episode; null on episode 0…",
  "memory_note_out": "…≤500 tokens, written after seeing the performance summary…",

  "cost": { "tokens_in": 284000, "tokens_out": 12800, "usd": 0.41 },
  "status": "ok",
  "ended_at": "2026-07-13T22:48:33.914Z"
}
```

`status` ∈ `ok` | `wallclock_capped` | `agent_error`.

`E_89` (the last element of `equity_series_cents`) is **net of the terminal liquidation friction**, so total return is fully realized and no strategy gets a free exit by holding into the final bar (D-derived mechanics).

**The metrics block is a cache, not a source.** It is written for convenience, and the replay test recomputes every field from `equity_series_cents` + the `fill` records and asserts equality. That assertion *is* the "any number on the scoreboard can be regenerated from the logs" claim — made testable rather than asserted.

Definitions, so the scoreboard is unambiguous:

| metric | definition |
|---|---|
| `total_return` | `E_89 / E_0 − 1` |
| `sharpe_floored` | `mean(r) / max(std(r), 0.25 × bh_daily_vol) × √252`, `r` = daily returns of `equity_series_cents` (89 of them). **The ranking metric** (D2). |
| `sharpe_raw` | Same with no floor. Diagnostic only — degenerate when the agent barely trades, which is the whole reason for the floor. |
| `vol_floor_binding` | `true` when the floor was the active denominator, i.e. the agent under-deployed capital. |
| `max_drawdown` | `min(E_t / cummax(E_t) − 1)` |
| `turnover` | `Σ gross_notional_cents / E_0` over all fills (terminal liquidation included) |
| `fees_paid_cents` | `Σ friction_cents` over all fills + terminal liquidation |
| `time_in_market` | fraction of ticks `0..89` with `shares > 0` |
| `n_trades` | count of fills, terminal liquidation excluded |

---

## Invariants the replay test enforces

1. **Ordering** — exactly one `meta` (first), exactly 90 `tick` records with `t` ascending `0..89`, exactly one `episode_end` (last).
2. **Equity invariant** — for every tick, `equity_cents == cash_cents + round(shares × close_t × 100)`, to the cent.
3. **No look-ahead** — every `fill.fill_tick == t + 1`, and `fill.fill_price == open_{fill_tick}` of the pinned series (`window.series_sha256`), exactly.
4. **Cash conservation** — `Σ cash_delta_cents + Σ friction_cents` reconciles `initial_capital_cents` against terminal cash.
5. **Time advances once** — exactly one call per decision tick has `advanced_time: true`; skipped ticks have none.
6. **Metrics reproduce** — recomputing the whole `metrics` block from `equity_series_cents` + fills gives bit-identical values to the cached block.
