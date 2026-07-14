# Analysis Artifacts — `run_manifest.json` and `probe/*.json`

`schemas/tick_log.md` v1.0.0 names two files in the run layout and never specifies either:

```
runs/{run_id}/
  run_manifest.json                             # <- named, never specified
  episodes/{agent_id}__{track}__{window_id}.jsonl
  probe/{agent_id}__{track}__{window_id}.json   # <- named, never specified
```

Step 6 cannot be built without both. This file specifies them, and records the six
questions the tick-log schema left open, with the answer taken for each. **Every
answer below is a proposal, not a fait accompli** — each one is a place where a
different call would change reported numbers, so they are listed rather than buried.

---

## Open questions in `tick_log.md` v1.0.0, and the answers taken

### Q1. What is `episode_index` an index *into*?
The schema shows `"episode_index": 6` and never says. This is the **x-axis of the
learning curve**, so it is load-bearing.

**Answer taken:** the index within the `(agent, track)` **memory lane**, `0..29`.
Memory is scoped per (model, track) and resets between tracks (D11), so a lane is
exactly one continuous learning sequence. Every agent — including baselines — walks
the windows in the same fixed order (`run_manifest.presentation_order`), so
`episode_index = i` denotes the same window for everyone. That is what makes the
baseline trace subtractable.

### Q2. How is a baseline distinguished from an LLM agent?
D11 plots learning curves *against the no-memory baselines' trace*, so the analytics
must be able to tell them apart from the logs alone. The schema's example shows
`"kind": "llm"` and enumerates nothing.

**Answer taken:** pin two fields.
- `agent.kind` ∈ `"llm"` | `"baseline"`
- `agent.memory` ∈ `"rolling_note"` | `"none"`

The **difficulty trace** is `kind == "baseline"` — those four strategies cannot learn,
so their per-episode Sharpe *is* the window-difficulty signal. The optional no-memory
**ablation arm** is `kind == "llm"` with `memory == "none"`, and is deliberately *not*
part of the trace: it is a comparison arm, not a difficulty probe.

### Q3. `trading_days_per_year` is not in the `config` block.
The metric table hardcodes `× √252` in prose, but the annualization factor is not
pinned anywhere in the log, while `vol_floor_multiple` — the other free parameter of
the ranking metric — is. A run under a different convention would be silently
mis-scored and nothing would catch it.

**Answer taken:** add `trading_days_per_year: 252` to the config block (schema v1.1.0).
The scoreboard reads the annualization factor **from the log**, never from a default.

### Q4. The probe's *period* axis has no answer key in the log.
`window` carries `source_ticker`, so the ticker axis is scorable from the log. It
carries no dates, so the period axis is not.

**Answer taken:** the probe's answer key comes from `data/windows/manifest.json`, not
from the run directory. This is a deliberate exception to "the log is the only
artifact the scoreboard may read", and it is a narrow one: the manifest supplies the
*answer key*, never a *reported number*. Every number on every scoreboard still comes
from the logs. Keeping ground-truth dates out of the run directory also keeps them
out of anything an agent's transcript could ever be adjacent to.

*(Alternative, if you prefer strict log-purity: add `date_scored_start` / `date_scored_end`
to `meta.window`. It leaks nothing — the agent never reads its own log — and it would
make the run directory fully self-describing. Say the word and I'll switch.)*

### Q5. Do `agent_error` / `wallclock_capped` episodes enter the trading scoreboard?
Materially changes results, and the schema is silent.

**Answer taken:**
- `wallclock_capped` → **counted.** The agent dithered until the clock ran out; the
  remaining ticks auto-Waited. That is a real (bad) trading outcome and it belongs in
  the distribution. It is *also* flagged on the reliability board.
- `agent_error` → **excluded from the trading scoreboard**, reported on the reliability
  board and in the run summary with a count. A harness or provider failure is not a
  trading result, and silently scoring it as flat would reward crashing on a bad window.

The scoreboard prints the excluded count every time, so an exclusion can never be
invisible.

### Q6. Two definitions that look like bugs and are not.
- **`time_in_market` cannot reach 1.0.** Defined as the fraction of ticks `0..89` with
  `shares > 0`; tick 0 is always flat (the first fill lands at `open_1`) and tick 89 is
  always flat (post-liquidation). The reachable maximum is **88/90 = 0.978** — that is
  buy-and-hold. Reported as-is, and the scoreboard labels the ceiling so it does not
  read as a bug.
- **`sharpe_raw` when `std(r) == 0`** (an agent that never trades) is `0`, not `NaN` or
  a division error. The floored Sharpe is 0 there too, and D2's whole point is that
  these coincide at zero with no discontinuity.

---

## `run_manifest.json`

One per run. Written before the first episode, so a crashed run is still interpretable.

```json
{
  "schema_version": "1.1.0",
  "run_id": "20260714T0930Z_a1b2c3d4",
  "created_at": "2026-07-14T09:30:00.000Z",

  "config": { "...": "the same config block stamped into every episode's meta",
              "config_sha256": "7ae0..." },
  "dataset": { "dataset_sha256": "e3b0...", "as_of": "2026-07-13" },
  "windows_manifest_sha256": "a5f5...",

  "agents": [
    { "id": "flagship", "kind": "llm", "provider": "anthropic", "model": "claude-...",
      "temperature": null, "memory": "rolling_note", "system_prompt_sha256": "9f2c..." },
    { "id": "cheap",    "kind": "llm", "provider": "anthropic", "model": "claude-...",
      "temperature": null, "memory": "rolling_note", "system_prompt_sha256": "9f2c..." },
    { "id": "buy_and_hold", "kind": "baseline", "memory": "none" },
    { "id": "flat",         "kind": "baseline", "memory": "none" },
    { "id": "random_5pct",  "kind": "baseline", "memory": "none", "seed": 11 },
    { "id": "sma_10_50",    "kind": "baseline", "memory": "none" }
  ],

  "tracks": ["real", "twin"],
  "presentation_order": ["w00", "w01", "...", "w29"],
  "windows": [
    { "window_id": "w00", "regime": "bull", "bh_daily_vol": 0.0147 }
  ],

  "probe": { "n_options": 8, "n_reps": 5, "period_granularity": "year",
             "chance_level": 0.125 },
  "seeds": { "master_seed": 20260713 }
}
```

`presentation_order` is the contract that makes `episode_index` comparable across
agents. `windows` is a convenience index; the authoritative per-window properties are
still stamped into each episode's `meta.window`.

---

## `probe/{agent_id}__{track}__{window_id}.json`

The leakage probe (D9). The model is shown a masked series — real or twin, formatted
identically — and asked to name the ticker and the period from a fixed set of options.
Chance is `1 / n_options` = 12.5% on each axis.

**The twin arm is not decoration.** Scoring a *twin's* answer against its *source's*
truth measures how often the model can name a series it has provably never seen — the
false-positive rate from pattern-matching on volatility and texture, which the twin
preserves by construction. Calibrated identifiability is `p_real − p_twin`. Without
that subtraction, "the model said 2008 and it was 2008" proves nothing: 2008 *looks*
like 2008.

```json
{
  "schema_version": "1.1.0",
  "run_id": "20260714T0930Z_a1b2c3d4",
  "agent_id": "flagship",
  "track": "real",
  "window_id": "w07",

  "truth": { "ticker": "AAPL", "period": "2013" },
  "options": {
    "ticker": ["AAPL", "JPM", "XOM", "PG", "NVDA", "KO", "CAT", "DIS"],
    "period": ["2007", "2009", "2013", "2016", "2018", "2020", "2022", "2024"]
  },

  "n_options": 8,
  "chance_level": 0.125,
  "reps": [
    { "rep": 0, "shuffle_seed": 918273,
      "answer": { "ticker": "AAPL", "period": "2016" },
      "correct": { "ticker": true, "period": false },
      "raw_response": "...", "tokens": { "in": 4210, "out": 64 } }
  ],

  "score": {
    "p_correct_ticker": 0.6,
    "p_correct_period": 0.2,
    "p_correct_mean": 0.4,
    "n_reps": 5
  }
}
```

- **The option *set* is identical for a window's real and twin**, seeded off
  `window_id` alone. If the distractors differed across tracks, `p_real − p_twin`
  would be measuring the distractors.
- **The option *order* is shuffled per rep**, seeded off `(window_id, track, rep)`,
  which is what the reps are for: position bias would otherwise masquerade as recall.
- `truth` on a **twin** is its *source window's* truth. A twin has no ticker of its
  own — that is the entire point of scoring against the source's.
- An unparseable or out-of-set answer scores `correct: false` and is counted; it is
  never silently dropped, since dropping failures inflates accuracy.
- `period_granularity: "year"` — the calendar year of the scored window's **midpoint**
  (a 90-bar window can straddle a year boundary). Coarse on purpose: a model that
  recognizes a crash by its volatility signature alone will "identify" the twin too,
  and the twin arm subtracts exactly that away.
