# Tick-Log Schema (JSONL) — v1.2.0

> **This document is not the contract. `src/pdtbench/schema.py` is.**
>
> That module validates every producer's output in the test suite (`tests/test_schema.py`),
> **rejects undeclared fields**, and rejects a float in any `*_cents` field. This page
> exists to explain the format to a human; if the two ever disagree, the module wins and
> this page is a bug.
>
> The separation is not pedantry. Under v1.0.0 the engine emitted `obs.bar` as
> `{open, high, low, close, volume}` while this document — and the fixtures written from
> it — said `{o, h, l, c, v}`. That survived four commits, because the analytics layer
> never reads `obs.bar`. The log viewer would have been the first consumer to touch it, and
> it would have crashed on every real episode. A schema nobody executes is a rumour.

The tick log is the **only** artifact the scoreboard is allowed to read. Every number we
report — median Sharpe, drawdown, turnover, fees, invalid-call rate, learning curve,
leakage scatter — must be regenerable from these files and nothing else. The UI reads them
too, which is why live mode and replay mode are the same code path (D14).

One episode = one JSONL file. One JSON object per line, typed by `type`, in strict order:

```
meta          exactly one, first line
tick          exactly n_scored (90), t ascending 0..89
episode_end   exactly one, last line
```

Run layout (the other two files are specified in `schemas/analysis_artifacts.md`):

```
runs/{run_id}/
  run_manifest.json
  episodes/{agent_id}__{track}__{window_id}.jsonl
  probe/{agent_id}__{track}__{window_id}.json
```

**Money is `int` cents.** A float in a `*_cents` field is a validation error, not a
rounding quirk. Integer cents is the entire reason there is no floating-point exploit in
the equity series, and a schema that shrugged at `1000000.0` would quietly hand that away.
Share counts are floats — fractional shares are permitted — held to **6 decimal places**,
exactly the precision at which they are displayed.

---

## 1. `meta` — first line

```json
{
  "type": "meta",
  "schema_version": "1.2.0",
  "run_id": "20260714T0930Z_a1b2c3d4",
  "episode_id": "flagship__real__w07",
  "episode_index": 6,

  "agent": {
    "id": "flagship",
    "kind": "llm",
    "memory": "rolling_note",
    "provider": "anthropic",
    "model": "claude-...",
    "temperature": null,
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
    "vol_floor_multiple": 0.25,
    "trading_days_per_year": 252,
    "n_warmup": 200,
    "n_scored": 90
  },

  "dataset": { "dataset_sha256": "e3b0...", "parquet_path": "data/processed/prices.parquet",
               "as_of": "2026-07-13" },
  "seeds":   { "master_seed": 20260713, "window_seed": 4471, "twin_seed": null,
               "agent_seed": null },
  "code":    { "git_commit": "263d870", "engine_version": "1.0.0" },
  "started_at": "2026-07-14T09:30:01.123Z"
}
```

| field | type | req | notes |
|---|---|---|---|
| `run_id` | str \| null | ✓ | |
| `episode_id` | str | ✓ | |
| `episode_index` | **int** | ✓ | Index within the `(agent, track)` **memory lane**, `0..29`. The learning curve's x-axis. Every agent walks the same `presentation_order`, so index `i` is the same window for everyone — that is what makes the baseline trace subtractable. The engine **refuses to write a log without it**: a silent default would collapse every lane onto one point. |
| `agent.id` | str | ✓ | scoreboard key |
| `agent.kind` | str | ✓ | **`llm` \| `baseline`**. Nothing else. (`scripted` was in use and was never legal.) |
| `agent.memory` | str | ✓ | **`rolling_note` \| `none`** |
| `agent.provider` / `.model` / `.temperature` / `.system_prompt_sha256` | | opt | LLM agents |
| `agent.seed` | int | opt | seeded baselines (e.g. `random_5pct`) |
| `track` | str | ✓ | **`real` \| `twin`** |
| `window.window_id` | str | ✓ | A twin **shares its source window's id** — that is what makes the pair a pair (D9). |
| `window.source_ticker` | str \| null | ✓ | On a **twin**, this is the *source* window's ticker. A twin has no ticker of its own. |
| `window.regime` | str | ✓ | **`bull` \| `bear` \| `chop`** |
| `window.bh_daily_vol` | num | ✓ | The vol-floor denominator (D2). Frozen property of the window; identical for every agent. |
| `window.series_sha256` | str | ✓ | Hash of the exact 290-bar OHLCV array used. Replay re-prices every fill against it. |
| `config.*` | | ✓ | All 17 keys required. **Both** free parameters of the ranking metric (`vol_floor_multiple`, `trading_days_per_year`) live here, so the scoreboard reads them from the log rather than from whatever the default is on the day someone re-runs the analysis. |
| `dataset.dataset_sha256` | str | ✓ | Engine refuses to write a log without it: a log that cannot name its data cannot be reproduced. |
| `seeds.master_seed` | int | ✓ | `window_seed` / `twin_seed` / `agent_seed` optional, nullable. `twin_seed` is null on the real track. |
| `code.git_commit` | str \| null | ✓ | |

---

## 2. `tick` — one per tick, `t = 0..89`

```json
{
  "type": "tick",
  "t": 12,
  "decision_point": true,

  "obs": {
    "tick": 12,
    "ticks_remaining": 77,
    "bar": { "open": 103.21, "high": 104.88, "low": 102.90, "close": 104.10, "volume": 1.42 },
    "stats": {
      "price": 104.10, "sma20": 101.70, "sma50": 99.40, "vol20_ann": 0.2131,
      "ep_high": 106.20, "ep_low": 97.30, "tick": 12, "ticks_remaining": 77
    },
    "portfolio": {
      "cash_cents": 500000, "shares": 4.751230, "position_value_cents": 494593,
      "equity_cents": 994593, "unrealized_pnl_cents": -5407, "avg_cost": 105.24
    },
    "done": false
  },

  "calls": [
    { "seq": 0, "tool": "getStats", "args": {}, "ok": true, "advanced_time": false,
      "latency_ms": 830, "tokens": { "in": 3120, "out": 190 } },
    { "seq": 1, "tool": "Buy", "args": { "notional_cents": 2000000 }, "ok": false,
      "advanced_time": false, "latency_ms": 910,
      "error": { "code": "INSUFFICIENT_CASH", "message": "notional_cents 2000000 exceeds cash 500000" } },
    { "seq": 2, "tool": "Buy", "args": { "notional_cents": 500000 }, "ok": true,
      "advanced_time": true, "latency_ms": 870 }
  ],

  "action": { "tool": "Buy", "args": { "notional_cents": 500000 }, "forced": false },

  "fill": {
    "side": "buy", "fill_tick": 13, "fill_price": 104.55, "shares_delta": 4.775610,
    "gross_notional_cents": 500000, "friction_cents": 500, "cash_delta_cents": -500000
  },

  "equity_cents": 994593,
  "invalid_count": 1,
  "reads_count": 1
}
```

### Timing — the part that must not be misread

- **`obs` is the state at the close of bar `t`.** It contains nothing from bar `t+1` or
  later. The engine enforces this with a read-audit wrapper that raises on a read past the
  current allowance, so look-ahead is structurally unavailable rather than merely avoided.
- **`equity_cents` is the mark at the close of bar `t`**: `cash_t + round(shares_t × close_t × 100)`.
  The sequence over `t = 0..89` **is** the equity curve every metric is computed from. It is
  mirrored in `obs.portfolio.equity_cents`; the validator requires them equal.
- **`fill` is the consequence of the action chosen at tick `t`, and lands at `open_{t+1}`.**
  So `fill.fill_tick == t + 1` for every agent fill. It is *not* in this record's
  `equity_cents` — it shows up in tick `t+1`'s.
- **One exception, and only one:** the terminal liquidation on tick 89 has
  `side: "liquidation"`, `fill_tick: 89`, and prices at `close_89`. It is the single fill at
  a price the agent has already seen — and it is unconditional, identical for every agent,
  and not a decision, so there is nothing there to exploit. The validator checks it against
  `close_89` and every other fill against `open_{t+1}`.

### The three tick shapes

A consumer must handle all three. `decision_point: false` has **two different causes**, and
`skipped_by` is what tells them apart.

| | `decision_point` | `skipped_by` | `calls` | `action` | `fill` |
|---|---|---|---|---|---|
| **decision tick** (`t ≤ 88`) | `true` | absent | ≥ 1 | the accepted action | fill or `null` |
| **skipped tick** (inside a `Wait(n)`) | `false` | present — the source tick | `[]` | `null` | `null` |
| **terminal tick** (`t == 89`) | `false` | **absent** | `[]` | `null` | liquidation or `null` |

Skipped ticks still carry `obs` and `equity_cents`: the equity curve stays one entry per
tick, which is what keeps replay and the scoreboard simple.

### Fields

| field | type | req | notes |
|---|---|---|---|
| `obs.bar` | obj | ✓ | **`open`, `high`, `low`, `close`, `volume`** — long keys. Not `{o,h,l,c,v}`. These match what `fetchData` returns, so the agent never sees two bar formats. |
| `obs.stats.tick` / `.ticks_remaining` | int | ✓ | Duplicated from `obs.tick` / `obs.ticks_remaining` **on purpose**: `stats` is also the standalone payload of the `getStats` tool, where there is no surrounding `obs`. |
| `obs.portfolio.avg_cost` | num \| null | ✓ | `null` when flat. Average-cost basis. |
| `obs.done` | bool | ✓ | On **every** tick, `false` until the terminal one. (It used to be patched in only at the end, so 89 of 90 observations simply lacked the field.) |
| `calls[]` | array | ✓ | **Every** call at this tick, in order, including invalid ones. The entire input to the reliability scoreboard. May be `[]`. |
| `calls[].tool` | str | ✓ | Not enum-constrained: an `UNKNOWN_TOOL` error has to record what the agent actually said. |
| `calls[].args` | obj | ✓ | Free-form; shape depends on the tool. `Buy`: `notional_cents` **or** `fraction`. `Sell`: `shares` **or** `fraction`. `Wait`: `n`. `fetchData`: `lookback`. |
| `calls[].ok` | bool | ✓ | `false` ⟹ `error` present. `true` ⟹ `error` absent. Enforced. |
| `calls[].error.code` | str | — | Closed set: `SCHEMA_ERROR`, `UNKNOWN_TOOL`, `NONPOSITIVE_QTY`, `NAN_QTY`, `INSUFFICIENT_CASH`, `INSUFFICIENT_SHARES`, `WAIT_OUT_OF_RANGE`, `READ_CAP_EXCEEDED`, `EPISODE_OVER`. |
| `calls[].tokens` | obj \| null | opt | `{in, out}`. **On the call, not the tick** — one LLM completion is one call, so that is where a token count is unambiguous. A tick can hold several calls, which made the old tick-level `tokens` ambiguous (a sum? the last one?). Episode totals live in `episode_end.cost`. Absent for baselines. |
| `calls[].latency_ms` | num | opt | |
| `action` | obj \| null | ✓ | The accepted action. **`action` must BE the call that advanced time** — same `tool`, same `args` — otherwise the log is crediting the agent with something it never did. Enforced. |
| `action.forced` | bool | ✓ | `true` when the engine imposed a `Wait(1)` (3 consecutive invalids, or the read cap). A forced Wait has **no advancing call at all**, because the agent never made one — the engine took the turn away. Enforced. |
| `action.n_effective` | int | opt | `Wait` only. A `Wait(n)` that would run past the final bar is **clamped** to land on it rather than rejected; erroring there would burn an agent's last decision on a technicality. |
| `fill.friction_cents` | int | ✓ | The full 10 bps/side, deterministic (D10). Buy: deducted from the notional before shares are computed. Sell: deducted from the proceeds. |
| `invalid_count` / `reads_count` | int | ✓ | Redundant with `calls` — kept because the analytics plucks them constantly. |

---

## 3. `episode_end` — last line

```json
{
  "type": "episode_end",
  "t_final": 89,
  "terminal": { "liquidated_shares": 4.775610, "liquidation_price": 108.20,
                "liquidation_friction_cents": 517, "final_equity_cents": 1043210 },
  "equity_series_cents": [1000000, 1000000, 998412, "… 90 values, E_0..E_89 …"],
  "metrics": {
    "total_return": 0.04321, "sharpe_floored": 1.4412, "sharpe_raw": 1.8103,
    "vol_floor_binding": false, "realized_vol_ann": 0.1932, "max_drawdown": -0.0612,
    "turnover": 2.41, "fees_paid_cents": 2040, "time_in_market": 0.6180, "n_trades": 6
  },
  "reliability": {
    "n_calls": 118, "n_invalid": 4, "invalid_rate": 0.0339, "n_schema_errors": 1,
    "n_forced_waits": 1, "n_prose_nudges": 0, "n_read_cap_hits": 0,
    "wallclock_s": 512.0, "hit_wallclock_cap": false
  },
  "memory_note_in": null,
  "memory_note_out": "…≤500 tokens…",
  "cost": { "tokens_in": 284000, "tokens_out": 12800, "usd": 0.41 },
  "status": "ok",
  "ended_at": "2026-07-14T09:38:33.914Z"
}
```

`status` ∈ **`ok` | `wallclock_capped` | `agent_error`**. `terminal.liquidation_price` is
`null` when the agent ended flat. `cost.usd` is optional. `memory_note_in` is `null` on
episode 0.

`E_89` — the last element of `equity_series_cents` — is **net of the terminal liquidation
friction**, so total return is fully realized and no strategy gets a free exit by holding
into the final bar.

### The `metrics` block is a cache, not a source

It is written for convenience. The scoreboard **recomputes every field** from
`equity_series_cents` + the `fill` records + `window.bh_daily_vol`, using the
`vol_floor_multiple` and `trading_days_per_year` from *that episode's own config block*, and
the replay test asserts agreement. That assertion *is* the "any number on the scoreboard can
be regenerated from the logs" claim — made testable rather than asserted.

| metric | definition |
|---|---|
| `total_return` | `E_89 / E_0 − 1` |
| `sharpe_floored` | `mean(r) / max(std(r), vol_floor_multiple × bh_daily_vol) × √trading_days_per_year`, `r` = the 89 daily returns of `equity_series_cents`. **The ranking metric** (D2). |
| `sharpe_raw` | Same, no floor. Diagnostic only — degenerate when the agent barely trades, which is the whole reason for the floor. **`0` when `std(r) == 0`, not `NaN`.** |
| `vol_floor_binding` | `true` when the floor was the active denominator, i.e. the agent under-deployed capital. |
| `realized_vol_ann` | `std(r) × √trading_days_per_year` |
| `max_drawdown` | `min(E_t / cummax(E_t) − 1)` |
| `turnover` | `Σ gross_notional_cents / E_0` over all fills, terminal liquidation included |
| `fees_paid_cents` | `Σ friction_cents` over all fills, terminal liquidation included |
| `time_in_market` | Fraction of ticks `0..89` with `shares > 0`. **Ceiling is 88/90 = 0.978, not 1.0** — tick 0 is always flat (the first fill lands at `open_1`) and tick 89 is always flat (post-liquidation). Buy-and-hold sits at the ceiling. |
| `n_trades` | Count of fills, terminal liquidation **excluded**. |

### Which episodes reach the scoreboard

- `wallclock_capped` → **counted.** The agent dithered until the clock ran out and the
  remaining ticks auto-Waited. That is a real (bad) trading outcome and belongs in the
  distribution. It is also flagged on the reliability board.
- `agent_error` → **excluded** from the trading scoreboard, counted on the reliability board.
  A provider or harness failure is not a trading result, and scoring a crash as "flat" would
  reward crashing on a bad window. The scoreboard prints the excluded count on every run, so
  an exclusion is never invisible.

---

## Invariants the validator enforces

Structural (`pdtbench.schema.validate_episode`):

1. **Ordering** — one `meta` first, exactly `n_scored` `tick` records with `t` ascending
   `0..89`, one `episode_end` last.
2. **No undeclared fields, anywhere.** A producer that starts writing a new field must
   declare it in `schema.py` in the same commit. This is the anti-drift mechanism.
3. **Money is `int`.** A float in any `*_cents` field is an error.
4. **Enums** — `track`, `regime`, `agent.kind`, `agent.memory`, `status`, `fill.side`,
   `action.tool`, `error.code`.

Cross-record:

5. **Equity mirror** — `equity_cents == obs.portfolio.equity_cents`, and
   `equity == cash + position_value`, at every tick.
6. **No look-ahead** — `fill.fill_tick == t + 1` on every agent fill; the liquidation is the
   sole exception (`fill_tick == t == 89`) and may occur only on the terminal tick.
7. **Time advances once** — exactly one `advanced_time` call on a decision tick; none on a
   skipped or terminal tick.
8. **The action is the call that moved the clock** — same `tool`, same `args`. A forced Wait
   has *no* advancing call, and must be a `Wait`.
9. **Tick shapes** — a non-decision tick has `skipped_by` **or** is `t == 89`, never neither;
   a decision tick never has `skipped_by`; tick 89 is never a decision point.
10. **`ok`/`error` agree** — `ok: false` ⟹ `error` present; `ok: true` ⟹ `error` absent.
11. **The series reconciles** — the per-tick `equity_cents` sequence equals
    `episode_end.equity_series_cents`, whose last element equals
    `terminal.final_equity_cents`.

Replay (`pdtbench.engine.replay`) additionally re-prices every fill against the pinned
`series_sha256` and recomputes the whole `metrics` block from the log alone.
