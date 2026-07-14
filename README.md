# pdtbench — an environment-authoritative agentic trading benchmark

Does a language model that makes money trading a real market window have skill, or is it
remembering the chart?

Every real window in this benchmark has a **twin**: a synthetic price path built from the
real one and matched until the *opportunity* is identical and only the *identity* differs
— same total return (±3pp), same volatility (±15%), same regime. A twin is not a random
market sample. It is a control: the same trade to make, against a chart that never
existed.

Beat the real window by more than its twin and that gap is the **edge**. Separately, ask
the model to *name* the window — that is **identifiability**. Plot one against the other
and the slope is the whole thesis. Rising means the profit lives where the model
recognises the chart. Flat means the skill is real.

---

## Run the demo

Requires **Python 3.12 or newer**. Nothing else — no API key, no network, no config.

```bash
git clone https://github.com/joshbeira/Agentic-Trading-Benchmark-PDT-Hackathon-.git
cd Agentic-Trading-Benchmark-PDT-Hackathon-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
python -m streamlit run scripts/app.py
```

It opens at <http://localhost:8501>. The run data is committed, so there is no build step
and nothing to generate — `git clone` is the setup.

**Read [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) beside it.** It explains what a twin is,
what every tab shows, and — importantly — the two numbers you should refuse to quote.

### Run the tests

```bash
pip install pytest
python -m pytest -q
```

218 tests. The engine, the schema, the replay verifier, the analytics, and the agent
runner. No test touches the network.

---

## What is in the box

| | |
|---|---|
| `src/pdtbench/engine/` | The trading environment. 90 scored bars, one action per tick, fills at the *next* unseen open — look-ahead is impossible by construction, and a read-audit test enforces it. |
| `src/pdtbench/mcp/` | One door into the engine: an in-process MCP server. The baselines and the model go through the same door, so fee parity holds by construction rather than by assertion. |
| `src/pdtbench/agent/` | The episode loop against Claude Opus 4.8 — a frozen system prompt, per-call token attribution, and D13's anti-stall ladder. |
| `src/pdtbench/analysis/` | Scoreboards, learning curves, and the leakage experiment. Recomputes every number from the tick logs; never trusts a cached block. |
| `scripts/app.py` | The demo surface. Reads only from the JSONL, so live mode and replay mode are the same code path. |
| `data/` | The committed Parquet — real windows and their twins. Fetched, validated, and frozen; the demo never depends on Yahoo being up. |
| `runs/` | Two reference runs. `baselines_dev` is real: 4 baselines × 2 tracks × 30 windows. `_fixture_demo` is synthetic, with planted effects and a stub probe — built to prove the analytics work, not to report a result. |
| `docs/PLAN.md` | Every design decision, and the ones that were amended after measuring them. |

**No model has been called yet.** The agent runner is built and tested against a fake
client; the live batch is the next step. The Headline tab therefore reports "no slope" on
the fixture — correctly, because its stub probe names the real series exactly as often as
the twin, and a regression against a constant is undefined. That is the honest answer to
a degenerate question, not a bug.

---

## The metric, and why it looks odd

```
sharpe = mean(r) / max( std(r), 0.25 × bh_daily_vol ) × √252
```

Plain Sharpe on a mostly-cash account is broken: deploy 1% of capital for two lucky days
and the return volatility is almost nothing, so the ratio explodes — a Sharpe of +30 for
doing essentially nothing. The floor makes the denominator scale with the opportunity the
window actually offered. An agent that never trades scores exactly 0.

**You cannot earn a high Sharpe on capital you never deploy.**

---

## Two numbers not to quote

The 30 windows are balanced 10 bull / 10 bear / 10 chop. That is deliberately **not** how
markets behave, and it forces buy-and-hold's *pooled* median to roughly zero by
construction — so "we beat buy-and-hold" on the pooled sample is meaningless. Compare
within a regime.

The leakage analysis also computes a Mann-Whitney U statistic, and **demotes it to
description**. It is unpaired, and it throws away the pairing the twin design exists to
create. The paired Wilcoxon is the honest test.

Friction is fixed at 10 bps/side and described as *realistic*, never as *calibrated*. The
empirical consequence of not rigging it: the 10/50 momentum baseline **loses money** on
this window set (−0.54 median Sharpe). Under a calibrated-friction claim we would now be
tuning the cost constant until that number turned green.
