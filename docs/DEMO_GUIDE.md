# Reading the benchmark — a guide to the demo app

    .venv/bin/python -m streamlit run scripts/app.py   →   http://localhost:8501

Every screen in this app exists to answer one question: when a language model makes money
trading a real market window, is that skill — or is it remembering the chart?

This guide explains what each tab shows, what every term means, and which numbers you
should refuse to quote on stage.

---

## The trick the whole thing turns on

A model that trades AAPL well in March 2020 might be a good trader, or might simply recall
what happened next. You cannot tell those apart by looking at the returns.

So every **real** window has a **twin**: a synthetic price path built from the real one,
matched so closely that the *opportunity* is the same and only the *identity* differs.
Same total return (within 3 percentage points), same volatility (within 15% relative), same
regime label — accepted by re-drawing until it matches, a median of 62 draws each. On the
built data: max paired return gap 2.9%, max volatility gap 13% relative, 30/30 regimes
matched.

A twin is **not** a random sample of the market, and is not meant to be. It is a control:
the same trade to be made, against a chart that never existed.

Two measurements follow from that:

- **Edge** — Sharpe on the real window minus Sharpe on its twin. The gap that memorization
  would open up.
- **Identifiability** — how well the model can *name* the window, calibrated by subtracting
  how often it "names" the twin.

The headline plots one against the other:

```
  EXPLOITED MEMORIZATION                GENERALIZATION

  edge                                  edge
   ^                          .          ^
   |                    .  .             |
   |                .                    |    .     .        .
   |            .                        |  .    .     .   .    .
   |      .  .                           |     .     .    .
   |   .                                 |
   +---------------------------> ident   +---------------------------> ident
      the profit lives in the               recognition buys nothing;
      windows it recognises                 the trading skill is real
```

A rising line means the profit lives in the windows the model recognises. A flat line means
recognition buys nothing — the skill stands up.

---

## One episode

An episode is one agent trading one window. Everything on the Episode tab is a view of this.

| | |
|---|---|
| **Window** | 200 warmup bars the agent can look back at, then 90 scored bars (about a trading quarter). |
| **Decisions** | One action per tick, at ticks 0–88 — 89 decisions. Tick 89 shows the final bar and accepts nothing. |
| **Fills** | An action at tick `t` fills at the *next* bar's open — a price the agent has not seen. Look-ahead is impossible by construction, not by good intentions, and a read-audit test enforces it. |
| **Costs** | 10 bps per side (2 bps fee + 8 bps slippage), deterministic. Described as realistic, never tuned to make a result look good. |
| **Capital** | $10,000. Cash earns nothing. |
| **Ending** | Any open position is force-liquidated at the final close, paying the same 10 bps. Nobody dodges an exit fee by holding to the end. |

---

## The five tabs

### Headline — the claim the benchmark exists to make

The scatter above, drawn from your run. Each dot is one window. The big number,
*Edge ~ identifiability*, is the slope of the line through them.

- **Read it:** positive slope → the model profits most where it recognises the chart.
  Flat → recognition doesn't pay, and the trading skill stands up.
- **Also:** probe accuracy for real vs twin. The twin number is the false-positive rate — a
  model that "recognises" a synthetic path is pattern-matching, not recalling. Subtracting
  it is what makes identifiability *calibrated*.
- **⚠ Flag:** the paired **Wilcoxon** test is the honest one. A Mann-Whitney figure also
  appears in the full rendering, deliberately **demoted to description** — it throws away
  the pairing that the twin design exists to create.

### Scoreboard — who traded well, with the uncertainty attached

Agents ranked by median vol-floored Sharpe, pooled and then split by regime. Every Sharpe
carries a bootstrapped interval in brackets.

- **Read it:** the per-regime panels are the real trading result. Ten episodes per cell is a
  small sample and the wide intervals are telling you so — that honesty is the point, not a
  defect.
- **⚠ Flag:** **do not quote the pooled "beat buy-and-hold" line.** The 10/10/10 regime
  balance forces buy-and-hold's pooled median to roughly zero *by construction*, so beating
  it pooled means nothing. Compare within a regime, always.

### Learning — does the rolling note actually teach it anything

An agent carries a short note (≤500 tokens) from one episode to the next, within a lane —
one agent, one track, 30 windows in fixed order. Three lines are drawn: raw Sharpe, the
baseline trace, and excess.

- **Read it:** only **excess** can claim learning. Raw Sharpe rising might just mean the
  later windows were easier. Baselines cannot learn, so their score *is* the window's
  difficulty — subtracting it removes the luck of the running order.
- **Also:** the verdict line refuses to claim a slope that clears zero but is too small to
  matter. A p-value on a rounding error is still a rounding error.

### Reliability — did the agent behave, and what did misbehaving cost it

Counted from the `calls` array — what the agent actually sent — not from the cached summary
block, so a disagreement between the two shows up rather than hides.

- **Read it:** a malformed call carries **no P&L penalty**. The only punishment is lost
  time, and this table is where that shows. An agent burning calls on errors has fewer
  decisions left.
- **⚠ Flag:** `forced waits/ep` above zero means the anti-stall ladder fired — the agent
  stalled and the engine took its turn away.

### Episode — one agent, one window, every move it made

The equity curve and share count over 90 ticks, then all the calls, the fills, and the
memory note carried in and out. **This is the tab to demo** — it is the raw record, not a
summary of one.

- **Read it:** the equity curve *is* the metric input. Every fee and every slippage cost is
  already inside it; nothing is added afterwards.
- **⚠ Flag:** a red *"cached metrics disagree with the replay"* banner means the log's
  stored numbers no longer match what recomputing from the ticks produces. The log wins;
  the cache is stale.

---

## Why the ranking metric looks odd

```
sharpe = mean(r) / max( std(r), 0.25 × bh_daily_vol ) × √252
```

Plain Sharpe on a mostly-cash account is broken. Put 1% of your money in for two lucky days
and your return volatility is almost nothing, so the ratio explodes — a Sharpe of +30 for
doing essentially nothing.

The floor forces the denominator to scale with the opportunity the window actually offered
(`bh_daily_vol` is a fixed, precomputed property of the window, identical for every agent).
An agent that never trades still scores exactly 0, and there is no discontinuity at zero.

The one-line defence: **you cannot earn a high Sharpe on capital you never deploy.**

Raw Sharpe is kept beside it as a diagnostic column.

---

## What you're looking at right now

### Why the headline says "—"

The app opens on `_fixture_demo`, a synthetic run with a **stub probe** — it names the real
series exactly as often as the twin, so identifiability is 0.000 on all 30 windows. A line
through points with no spread has no slope, so the app shows "—" and explains why rather
than printing `nan`. That is the correct answer to a degenerate question, not a bug.

The Headline tab only becomes the real figure once a run has genuine probe data.
**No model has been called yet.**

| Run | Episodes | What it is |
|---|---|---|
| `_fixture_demo` | 360 | Synthetic, with planted effects and a stub probe. Built to prove the analytics work, not to report a result. Its agent names (`cheap`, `flagship`) predate the decision to run a single model in two memory arms. |
| `baselines_dev` | 240 | Real: 4 baselines × 2 tracks × 30 windows. No LLM, so the Headline tab correctly says there is nothing to probe. The Scoreboard and Episode tabs here are genuine. |

---

## Glossary

**real / twin**
The two tracks. *Real* is a masked historical window; *twin* is its matched synthetic
counterfactual. Every agent trades both.

**edge**
Sharpe on the real window minus Sharpe on its twin, per window. The gap memorization would
open up.

**identifiability**
How well the model can name a window, calibrated: accuracy on the real path minus accuracy
on the twin. The probe is 8-way ticker and 8-way period, so chance is 12.5%.

**sharpe (floored)**
The ranking metric — see above. The floor stops idle capital from faking a great
risk-adjusted return.

**floor binding**
How often the volatility floor actually caught this agent. High values mean it was barely
deploying capital.

**bull / bear / chop**
The window's regime, classified on the scored 90 bars: ≥ +8% total return is bull, ≤ −8% is
bear, anything else is chop. The 30 windows are deliberately balanced 10/10/10, which is
*not* how markets actually behave — that's stated openly on the slide, not hidden.

**baseline**
A scripted strategy — `buy_and_hold`, `flat`, `random_5pct`, or `sma_10_50`. They run
through the same engine API and pay the same fees as the models, so fee parity holds by
construction rather than by assertion. They cannot learn, which is exactly why their scores
serve as the difficulty trace. (Worth knowing: the 10/50 momentum baseline *loses money* on
this window set — −0.54 median Sharpe, −3.6% median return. The friction was never tuned to
make that look better.)

**lane / episode_index**
One agent on one track walking all 30 windows in a fixed order, carrying its note forward.
`episode_index` 0–29 is that position, and it is the x-axis of the learning curve. Memory
resets between tracks.

**memory arm**
`rolling_note` carries a note between episodes; `none` is the control that doesn't. Same
model, same tools, same prompt — the note is the only difference.

**prose nudge**
The agent replied with text instead of calling a tool. It gets one nudge; a second stall
costs it the turn.

**forced wait**
The engine imposed a `Wait(1)` because the agent stalled — three consecutive invalid calls
(`max_consecutive_invalid`), or a second prose reply (`prose_stall`). The log records which,
so a forced turn is never mistaken for a decision the agent made.

**read cap**
At most 8 non-time-advancing calls per tick. Look all you like, but not forever — exceeding
it feeds the invalid ladder.

**invalid call**
A malformed or rejected call. Costs no money, only time. That is a deliberate design choice,
and the Reliability tab is where the cost surfaces.

**status**
`ok` is a clean episode. `agent_error` means it died partway and is excluded from the
scoreboard rather than silently scored as a loss.

---

Numbers are recomputed from the tick logs on load, never read from the cached summaries — a
figure here and a figure from `scripts/analyze.py` cannot disagree. Nothing in this app
calls an API, which is why it doubles as the rehearsed replay fallback (D14).
