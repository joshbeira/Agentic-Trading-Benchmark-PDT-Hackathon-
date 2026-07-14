# Agentic Trading Benchmark — Design & Build Plan

*PDT Partners Hackathon. Status: design frozen after a full grilling pass. This document is the contract; the code follows it.*

---

## Context

An **environment-authoritative benchmark** comparing LLM agents on simulated trading. A strict, state-managed simulation where the engine is the absolute source of truth: it owns cash, position, fills, and constraint enforcement, and the agent may only act through tools. Not a reinforcement-learning environment — weights are frozen, no gradient update occurs — but the engine exposes a Gym-style `step`/`reset` API, and cross-episode improvement is handled by an explicit in-context memory mechanism.

The audience is a quantitative hedge fund. Every design choice below is one we expect to defend under scrutiny. A first-pass design was stress-tested and five load-bearing flaws were found and fixed: a gameable headline metric, an order-validity hole at the fill boundary, an unaffordable LLM-call budget, a circular "extractable signal exists" claim, and a leakage experiment that could not separate memorization from generalization. The decisions below (D1–D14) are the result.

Constraint: ~24–48h build, judged on a live demo plus a pitch.

---

## Design decisions

### D1. Scope
The full evaluation is **pre-run and cached**; the stage shows cached results plus one live episode. Every scope call favors "smaller but bulletproof."

### D2. Ranking metric — vol-floored Sharpe
Rank by

```
sharpe = mean(r_t) / max( std(r_t), 0.25 × bh_daily_vol ) × sqrt(252)
```

where `r_t` are the agent's daily equity returns and `bh_daily_vol` is the buy-and-hold daily return volatility **of that same window** (a precomputed property of the window, identical for every agent). Risk-free = 0; cash earns 0.

*Why:* plain episode Sharpe on a mostly-cash equity curve is degenerate. An agent that deploys 1% of capital for two lucky days has near-zero return volatility and posts a Sharpe of +30. The floor makes the denominator scale with the opportunity actually available in the window. A never-trading agent still scores exactly 0, and there is no discontinuity at zero. One-line defense: **you cannot earn a high Sharpe on capital you never deploy.** Raw (unfloored) Sharpe is retained as a diagnostic column.

### D3. Order denomination — each side in the unit that is always executable
- `Buy(notional_cents)` with `notional_cents ≤ cash`. Engine computes `shares = notional × (1 − friction) / open_{t+1}`.
- `Sell(shares)` with `shares ≤ position` (plus a `fraction` convenience argument).
- Fractional shares permitted, so both are exactly satisfiable.

*Why:* an order is submitted at tick `t` but fills at `open_{t+1}`, a price the agent has not seen. If Buy took a **share count**, a gap-up could make a valid-at-submission order unaffordable at fill — forcing either a reject *after* time has advanced (contradicting "invalid actions do not advance time") or a silent partial fill (a poisonous feedback signal for an agent benchmark). Denominating each side in the unit the agent actually owns makes fill-time failure **impossible by construction**.

### D4. Decision cadence — Gym step semantics + `Wait(n)`
Every `Buy`/`Sell`/`Wait` result carries the new observation (elapsed bar(s), stats, position, unrealized P&L), so the hot loop is **~1 LLM call per tick**. `Wait(n ≤ 10)` is allowed and its response includes all `n` elapsed bars. `fetchData` / `getStats` remain available for deeper pulls.

*Why:* a read-then-act loop costs ~300 LLM calls per episode. At 2 models × 2 tracks × 30 episodes that does not finish in a weekend. Step-semantics plus multi-Wait brings an episode to ~40–80 calls.

### D5. Model lineup — 2 models, cheap + flagship, same family
Framing: **does capability buy trading skill, and does it buy faster in-context learning?** Leaderboard = 2 LLMs + 4 baselines. Budget ≈ $30–90; wall-clock ≈ 75–90 min with the four (model × track) lanes run in parallel — episodes stay sequential *within* a lane because memory is sequential.

### D6. Synthetic generator — per-ticker OHLCV bar-block bootstrap
Stationary block bootstrap (expected block length ≈ 20 trading days, geometric) resampling **whole `(O,H,L,C,V) / prev_close` daily tuples** from a single source ticker per path, reconstructed from a base of 100.

*Why bars, not close-to-close returns:* resampling returns forces you to fabricate open/high/low/volume, which produces bars with a different texture from the real track and hands the leakage probe a free tell. Resampling whole normalized bars preserves OHLC internal consistency, overnight gaps, and the volume series **for free**, and makes the two tracks format-identical.

### D7. Signal claim — efficient null + volatility-timing edge
**Dropped:** "friction is calibrated so the momentum baseline is roughly breakeven, which guarantees extractable signal exists." That claim is circular — it calibrates a cost constant against ex-post-selected windows — and this audience will say so.

**Replaced with:** on the synthetic track, trend *continuation* is destroyed by construction — 20-day blocks carry no dependence past their own length, so an agent cannot ride a twin's move by predicting it, only by taking exposure to it. (The twin's *realized* drift is matched to its source, per D9; what is destroyed is the ability to forecast it from the path.) But **volatility clustering survives the block bootstrap**, so *risk* stays predictable even where *direction* does not. Managing exposure against forecastable volatility is therefore a genuine ex-ante edge — and it is exactly the edge a Sharpe-based metric rewards. Friction is fixed at 10 bps/side and described as **realistic**, never as *calibrated*. Baseline net performance per track is reported empirically, not asserted in advance.

**Measured on the built data:** volatility clustering survives, attenuated — short-lag |return| autocorrelation runs 0.100 on the real track against 0.075 on the twins, positive in 27 of 30 twins. Fat tails survive (excess kurtosis 6.1 vs 6.4). And the empirical consequence of *not* rigging the friction: the 10/50 momentum baseline **loses money** on this window set (−0.54 median Sharpe, −3.6% median return). Under the old claim we would now be tuning the cost constant until that number turned green, which is precisely the circularity we removed.

### D8. Warmup — 200 visible pre-episode bars
Every window is **290 bars**: 200 warmup + 90 scored. `fetchData(200)` is honorable from tick 0; SMA-50 and the momentum baseline are defined everywhere; episode high/low and all metrics are computed over **scored bars only**.

*Why:* without warmup, SMA-50 is undefined until tick 50 and the momentum baseline cannot trade for most of the episode.

### D9. Leakage experiment — twins, a calibrated probe, and an edge-vs-identifiability headline
1. **The synthetic track is a set of block-bootstrap *twins* of the same 30 masked-real windows** — same source ticker, same marginal return distribution, same volatility character; differing only in path identity and in structure beyond the block length. *This* is what makes the comparison paired.

   **Amended during the build, after measuring it.** A block bootstrap resamples *with replacement*, so an unmatched twin's realized 90-bar path is a fresh draw: it inherits the source's drift and volatility in expectation, but its realized total return has a standard deviation of roughly 16%. In the first build, the +54% META bull window produced a **−19% bear twin**, and 22 of 30 twins flipped regime. `Sharpe_real − Sharpe_twin` would then have measured *which path happened to trend*, not which path the model remembers — which is the exact confound the twin exists to eliminate. A quant would have spotted it in one glance at the window table.

   Twins are therefore accepted by **rejection sampling**: a draw is kept only if its scored segment realizes the same total return (±3pp), the same volatility (±15% relative), and the same regime label as its source — and its warmup segment matches too, since the agent sees those 200 bars and forms its priors on them. Achieved on the built data: **max paired return gap 2.9%, max volatility gap 13% relative, 30/30 regimes matched**, at a median of 62 draws per twin.

   A twin is consequently **not an unbiased draw from the source's data-generating process, and is not meant to be.** It is a counterfactual path matched on *opportunity*, differing in *identity*. That is what a control is. (Both tracks are conditioned on realized return — the real windows by regime-stratified selection, the twins by matching — so the mild endpoint-conditioning signature shows up symmetrically in both: variance ratio at 60 days is 0.66 real against 0.72 twin.)
2. **The probe is multiple-choice** — 8-way ticker, 8-way period; chance = 12.5% — and is run on **both** the real windows **and** their twins. The twin arm calibrates the false-positive rate: a model that "identifies" a synthetic path is pattern-matching, not recalling.
3. **Headline figure:** per-window **Sharpe edge (real − twin)** against per-window **probe identifiability**. Edge concentrated in the windows a model can name = exploited memorization. Edge flat across identifiability = generalization. The Mann-Whitney U test is demoted to a descriptive statistic.

*Why:* comparing Sharpe on real windows against Sharpe on *unrelated* synthetic paths confounds memorization with a structural difference between the two data-generating processes — the bootstrap destroys multi-week trend, so a trend-following agent scores worse on synthetic **whether or not it has ever seen the real series**. The twin design holds the generating process fixed. Famous-event windows (a COVID-crash shape, say) will be identifiable from shape alone; that is a data point on the scatter, not a bug.

Episode count: **60 per model** (30 paired).

### D10. Slippage — fully deterministic
10 bps per side (2 bps fee + 8 bps slippage), constant. **No RNG anywhere in execution.** Exact pairing across models, trivial replay, nothing to defend. ("Identical slippage RNG seeds" is struck from the design — it contradicted the constant.)

### D11. Memory policy
- Scoped **per (model, track)**; reset between tracks. Protects the paired comparison from cross-track contamination and lets the four lanes run in parallel.
- **Rolling single note, ≤ 500 tokens**, which *replaces* the previous note rather than appending. Context length is therefore constant across episodes — late-episode behavior is not confounded by a growing prompt.
- **Same fixed regime-stratified window order for every model**; twins run in source order.
- Learning curves are plotted **against the no-memory baselines' per-episode trace**. The baselines cannot learn, so their trace *is* the window-difficulty signal; divergence from it is the only honest evidence of learning.
- Optional ablation if time allows: re-run the cheap model with memory off.

### D12. Regimes & window hygiene
Classify on the **scored 90 bars**: total return ≥ +8% → bull, ≤ −8% → bear, else chop. Volatility is *reported* per window, not used to classify. No overlapping `(ticker, time)` windows; at most one window per ticker across the 30.

Per-regime distribution panels are the primary trading result, and **"vs buy-and-hold" is stated only within-regime**. A 10/10/10 regime balance forces the pooled buy-and-hold median to ≈ 0 by construction, so "we beat buy-and-hold on the pooled sample" is never a headline. The slide states plainly that the regime balance is deliberately non-representative of market base rates.

### D13. Run protocol
- **Byte-identical system prompt** for both models, disclosing the scoring metric, the fee schedule, the fill rule, and the tool contract. (Hiding the metric would test goal inference — a different experiment.)
- Provider-default temperature, logged.
- **Anti-stall ladder:** ≤ 8 non-time-advancing calls per tick → structured `READ_CAP_EXCEEDED` error → the existing 3-consecutive-invalid → forced `Wait(1)`. A prose reply with no tool call gets one nudge, then a forced Wait. Hard episode wall-clock cap of ~20 min (remaining ticks auto-Wait; the episode is flagged).
- Language fix: malformed calls carry **no direct P&L penalty; the only consequence is lost time**, and that is logged on the reliability scoreboard.
- Defaults: initial capital **$10,000** (1,000,000 cents); `getStats` realized volatility = trailing 20-day, annualized.

### D14. Demo & ops
- **One UI, reading only from the JSONL stream** — so live mode and replay mode are the same code path, and the rehearsed replay fallback is one keypress away.
- Stage order: cached results first (scoreboards, learning curves, leakage scatter), then one live cheap-model episode.
- Yahoo data fetched, validated, and the Parquet **committed in hour one**. No live API dependency on stage.
- Baselines execute **through the same engine API** as scripted clients, so fee parity holds by construction rather than by assertion.

### Stated openly on the slide (a feature, not a fix)
The standard error of an episode Sharpe estimated from 90 daily bars is ≈ 1.6–1.7 annualized units. **n = 30 paired windows is power-limited.** We report bootstrap confidence intervals, expect them to overlap, and use the paired Wilcoxon signed-rank test for the A-vs-B claim. Owning this preempts the "your error bars are enormous" question — and it is the honest description of what one quarter of daily data can tell you.

### Reporting policy — no claim without an effect size
Every headline claim (learning, memorization) must clear **both** a confidence interval that excludes zero **and** an effect-size floor. This is not pedantry: a t-test on a series with almost no residual variance will certify anything. During the build, the buy-and-hold baseline's excess-Sharpe slope came back "statistically significant" at **−1.0e-07 per episode, p = 0.013** — seven orders of magnitude below anything tradeable, and it would have been reported as a finding. The floors are `0.005` Sharpe/episode for learning (≈ 0.15 Sharpe over a 30-episode lane) and `0.10` Sharpe per unit identifiability for memorization. Significance without magnitude is noise with a certificate, and this is the audience least likely to let that pass.

---

## Derived mechanics (implementation-level consequences of D1–D14)

**Tick semantics.** A window has warmup bars at indices `-200..-1` and scored bars at `0..89`.
- At tick `t` the agent observes bars **up to and including `t`** (the close of day `t`) and submits one action.
- The action fills at **`open_{t+1}`** — a price it has not seen. Look-ahead bias is eliminated by construction, and this is enforced by a read-audit test, not by inspection (see Verification).
- Decision points are `t = 0..88` (**89 decisions**). Tick 89 is terminal: the final bar is shown, no action is accepted.
- **Terminal rule:** the engine force-liquidates any position at `close_89` paying the standard 10 bps. Every agent's P&L is therefore fully realized, and no strategy earns a hidden discount by holding into the final bar to dodge an exit fee.

**Equity series.** `E_t = cash_t + shares_t × close_t` for `t = 0..89`, with `E_0 = initial_capital` (the first action cannot fill before `open_1`) and `E_89` net of the terminal liquidation friction. That yields **89 daily returns** per episode, and the equity curve *is* the metric input — every cost is inside it.

**Share precision.** The share count is held to 6 decimal places, *exactly* the precision at which it is displayed. Keeping more precision internally than the engine publishes meant the tick log could not reproduce its own equity mark (off by a cent whenever `shares × close` landed near a rounding boundary) and left an agent that sold precisely the position it had been shown holding 4.4e-07 shares of dust. The replay verifier caught both.

**Vol-floor reference.** `bh_daily_vol` = std of close-to-close simple returns over the scored bars. A pure property of the window: precomputed, stored in the window manifest, identical for all agents, requiring no baseline run.

---

## Build order

1. **Data pipeline** *(first — Parquet committed in hour one)*: yfinance fetch (`auto_adjust=True`) → OHLC consistency validation → NaN cleaning → Parquet + SHA-256. Candidate window sampler → regime labels (±8%) → hygiene constraints → select 30 → fixed stratified order.
2. **Twin generator**: per-ticker OHLCV bar-block bootstrap, 290 bars per twin, seeded from the source window.
3. **Engine**: integer-cent state; `Buy(notional)` / `Sell(shares|fraction)`; fills at `open_{t+1}`; 10 bps/side; `Wait(n ≤ 10)`; invalid + anti-stall ladders; step results carry observations; JSONL per tick (obs, calls, fills, equity) tagged with dataset hash, config hash, and seeds; replay from logs.
4. **MCP server** wrapping the engine; **baseline clients** (buy-and-hold, flat, random 5%, 10/50 SMA crossover) driving the same API.
5. **Agent runner**: 2 models, identical system prompt, rolling memory note, episodes sequential per (model, track), 4 lanes in parallel.
6. **Analytics**: vol-floored Sharpe + trading scoreboard; reliability scoreboard; learning curve vs the baseline trace; probe harness (multiple-choice, both tracks); leakage scatter.
7. **UI**: JSONL viewer — price with action markers, equity vs buy-and-hold, transcript. Live mode = replay mode.
8. **Full pre-run overnight** → figures cached → rehearse the demo including the replay fallback.

---

## Verification

**Engine invariants (unit tests).**
- **Equity invariant:** `equity ≡ cash + shares × price` after every fill, net of fees, at every tick, to the cent.
- **No look-ahead:** enforced by a **read-audit** wrapper over the bar array. While building the observation for tick `t`, *no read of any index > `t`* is permitted. While executing a fill, index `t+1` is permitted (that is the fill) and nothing beyond. A violation raises, so look-ahead cannot be reintroduced silently by a later refactor.
- **Replay reproducibility:** replaying any episode's JSONL recomputes bit-identical metrics; the metrics cached in the log must equal the metrics recomputed from the logged equity series and fills.

**Twin sanity.** A KS test cannot distinguish a twin's return marginal from its source window's; autocorrelation beyond the block length is destroyed; bars are format-identical to the real track.

**Baseline sanity** *(step 4)*. Flat scores exactly 0. Buy-and-hold's episode return equals the window return net of exactly two frictions. The 10/50 crossover's trades match hand-computed signals on a known window. All baselines pay fees through the engine path.

**Dry run.** One full cheap-model episode end-to-end, with a token and cost check, before committing to the overnight batch.

---

## Deferred / stretch (only if green by Sunday morning)

- λ-parameterized signal-injection track with a ground-truth alpha, reported as **alpha-capture fraction**.
- No-memory ablation arm for the cheap model.

---

## Environment

Python 3.12. Virtualenv at `~/.venvs/pdt` (deliberately outside the repo — the repo lives on a OneDrive-synced mount). Dependencies: numpy, pandas, pyarrow, scipy, yfinance, pytest.
