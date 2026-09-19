# The stage script

Six minutes, five moves. What to click on the left, what to say on the right.

`DEMO_GUIDE.md` explains what the app *means*. This is the running order — what you
actually do, in sequence, and the two sentences that decide whether this pitch lands.

---

## Before you walk on

```bash
.venv/bin/python -m streamlit run scripts/app.py
```

Then, **before anyone is watching**, do these three things:

1. **Switch the sidebar to `baselines_dev`.** The app opens on `_fixture_demo`, and the
   first number a judge sees there is `flagship +1.83`, which is **planted**. Do not let
   that be the opening frame. Land on real data.
2. Click **Episode**, pick `buy_and_hold` / `[00] w00 · bull · ok`. That is your move 3.
3. Have `docs/PLAN.md` open in a second tab. When someone asks "why 10 bps", the answer
   is D10 and you want to be able to point at it.

The app needs ~8 seconds on first load (it revalidates 360 logs). Do that before the
clock starts, not on stage.

---

## The one thing you must not do

**Do not present `_fixture_demo`'s numbers as results.** `flagship +1.83`, the +0.5 skill
edge, the learning slope — all planted, by `scripts/make_fixture_run.py`, on purpose, to
test the analytics. No model has ever been called in this project.

If you show that table and say "our flagship model beat the baselines," you are showing
fabricated data to a room of quants. It is in the repo's own README that no model has run.
Someone will read it.

The fixture is still going on stage — it is move 4 — but it goes on **as a calibration of
the instrument**, announced as synthetic before you show it. Framed that way it is the
strongest thing you have. Framed as a result it is the end of the conversation.

---

## Move 1 — the question (30s, no screen)

> A language model makes 19% trading a real market window. Is that skill, or does it
> remember the chart?
>
> You cannot tell from the returns. That is the whole problem. And it is not a small
> problem — every model here was trained on the entire history of every ticker.

Do not touch the laptop yet. Let the question sit.

---

## Move 2 — the control (60s, no screen — or the window table)

> So every real window has a **twin**. Same source ticker, block-bootstrapped into a path
> that never existed, then re-drawn until it matches: same total return within 3
> percentage points, same volatility within 15%, same regime label. A median of 62 draws
> each to hit that.
>
> A twin is not a random market sample and is not meant to be. It is a control: the same
> trade to make, against a chart nobody has ever seen.
>
> Trade both. If the profit is skill it shows up on both. If it is memory it shows up only
> on the real one — and only where the model can actually name the chart.

**If asked "isn't the twin just easier/harder?":** that is exactly the confound the
matching removes, and it was found by measuring, not by guessing. The first build's twins
flipped regime on 22 of 30 windows — a +54% META bull window produced a −19% bear twin.
That is written up in PLAN.md D9. The rejection sampler is the fix.

---

## Move 3 — the instrument, on real data (90s)

Sidebar already on **`baselines_dev`**. **Episode** tab, `buy_and_hold`, `w00`.

Point at the equity curve.

> This is one agent, one window, 90 trading days. 240 real episodes in this run — four
> scripted baselines across both tracks.
>
> The engine owns everything. The agent gets four tools and cannot touch its own state.
> It submits at the close of day *t*; the fill lands at the open of day *t+1*, a price it
> has never seen. Look-ahead isn't discouraged here, it's **unreachable** — there's a
> read-audit wrapper that raises if the observation builder touches any index past *t*.
>
> Every fee and every slippage cost is already inside this curve. Nothing is added
> afterwards. 10 bps a side, deterministic, no RNG anywhere in execution.

Scroll to the calls and fills.

> And this is the raw record. Not a summary of the record — the record. Every number in
> this app is recomputed from these logs on load; nothing reads a cached summary.

**This is your credibility move.** It is real, it is yours, and it is the part that
cannot be faked.

---

## Move 4 — the calibration (90s) ← *say the label first*

Click **Scoreboard**, still on `baselines_dev`.

> Here's the honest state of the real run. `random_5pct` is top at +0.56 — and its
> interval is [−0.36, +1.48]. It spans zero. Nothing here is significant, and with n=30
> and 90 daily bars it structurally cannot be: the standard error on an episode Sharpe is
> about 1.6 annualized units. We say that on the slide rather than wait to be asked.
>
> Note `sma_10_50` at **−0.54**. The momentum baseline loses money. We never tuned the
> friction to fix that, and PLAN.md says why: calibrating a cost constant until your
> baseline breaks even is circular, and this room would say so.

**Now say the label, then switch:**

> Next is a synthetic run. No model was called — the Sharpes are planted and the probe is
> a stub. It exists for one reason: to prove the analytics recover an effect whose true
> size we chose.

Switch sidebar to **`_fixture_demo`**. Scoreboard.

> We planted a leaker — `flagship`, with a real edge that concentrates in exactly the
> windows it can name. And we planted a hard negative — `cheap`, which "identifies" real
> windows *and* twins alike at 40%, five times chance, because it's reading volatility
> texture rather than recalling a path.
>
> The analysis has to convict the first and clear the second. Clearing `cheap` is the
> harder case, and it is the entire reason the twin arm exists.

That is a unit test for a statistical claim. Most hackathon projects do not have one.

---

## Move 5 — the gap, in your own words (45s)

Click **Headline**, on `_fixture_demo`. It reads *"No slope — the question is degenerate
here."*

> This tab is the figure the benchmark exists to produce: per-window Sharpe edge against
> per-window identifiability. Rising means the profit lives where the model recognises the
> chart. Flat means the skill is real.
>
> It says "no slope" because the fixture's stub probe names the real series exactly as
> often as the twin, so identifiability is 0.000 on all 30 windows and a regression against
> a constant is undefined. The app prints the reason instead of printing `nan`.
>
> **And I'll be straight with you: no model has been called yet.** The engine is built and
> the analytics are calibrated. The runner is written and tested against a fake client.
> The probe and the live batch are what's next.

Then stop. Do not pad it.

**Why this is the right ending:** you are pitching a fund whose entire business is not
fooling itself. A rigorous instrument with an honest null beats a slide of numbers you
cannot defend, and everyone in that room knows which one is harder to build. The failure
mode is not "we have no results." The failure mode is getting caught claiming ones you
don't have.

---

## The questions you will get

**"What's your n?"**
30 paired windows, 90 daily bars each. Power-limited and stated as such — bootstrap CIs,
paired Wilcoxon for A-vs-B. We expect the intervals to overlap.

**"Why that Sharpe formula?"**
`mean(r) / max(std(r), 0.25 × bh_daily_vol) × √252`. Plain Sharpe on a mostly-cash account
is degenerate — deploy 1% for two lucky days, near-zero return vol, Sharpe of +30. The
floor makes the denominator scale with the opportunity the window actually offered.
One line: **you cannot earn a high Sharpe on capital you never deploy.**

**"Did you beat buy-and-hold?"**
Refuse the pooled version. The windows are balanced 10 bull / 10 bear / 10 chop, which
forces buy-and-hold's pooled median to roughly zero *by construction*. Only the
within-regime panels mean anything. Saying this unprompted is worth more than any number
on the board.

**"How do you know the twins aren't just different?"**
Measured: max paired return gap 2.9%, max volatility gap 13% relative, 30/30 regimes
matched. Volatility clustering survives the bootstrap, attenuated — short-lag |return|
autocorrelation 0.100 real vs 0.075 twin, positive in 27 of 30. Fat tails survive: excess
kurtosis 6.1 vs 6.4.

**"Isn't the block bootstrap destroying the signal you're testing for?"**
It destroys *trend continuation* past the block length — deliberately. That's why the
claim is not "momentum works here." Volatility clustering survives, so *risk* stays
forecastable where *direction* does not, and managing exposure against forecastable
volatility is exactly the edge a Sharpe metric rewards.

**"Why is a random baseline beating your momentum baseline?"**
Because we didn't tune the friction to stop it. That's the point. Both intervals span
zero — the honest reading is that neither is distinguishable from noise at n=30.

**"What would convince you the model is memorizing?"**
A positive slope on the Headline scatter that clears both a CI excluding zero and the
0.10 effect-size floor. Significance alone doesn't qualify — during the build a baseline
came back "significant" at −1.0e-07 Sharpe/episode, p=0.013. Seven orders of magnitude
below tradeable. That is why there is a floor.

---

## If it breaks

Nothing in the app calls an API — it reads committed JSONL only, so the venue wifi cannot
take it down. If Streamlit dies: `Ctrl+C`, re-run, 8 seconds. If the port is taken,
`--server.port 8502`. Static figures are in `figures/` as the last resort.
