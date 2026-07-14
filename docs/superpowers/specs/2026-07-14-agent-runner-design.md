# Agent Runner — Design (build order step 5)

*PDT Partners Hackathon. Supersedes D5 and amends D11 and D13 of `docs/PLAN.md`. Written 2026-07-14, after steps 1–4 shipped. This document is the contract for the runner; the code follows it.*

---

## Context

Steps 1–4 are built: the data pipeline, the twin generator, the engine, the MCP server, and the four baselines (240 episodes, 240/240 schema-valid and replaying). Step 6's analytics is also built, but has only ever been driven against `scripts/make_fixture_run.py` — a synthetic run with **planted** effects and a stub probe. **No model has ever been called.** Nothing in the repo imports an LLM provider.

This step closes that gap: it makes the benchmark produce real model output, and it produces the leakage probe D9's headline figure needs.

The plan's frozen design predates the model it will run on, so three of its decisions no longer survive contact with the API. They are amended below, explicitly, rather than quietly worked around.

---

## Amendments to the frozen design

### A1. D5 is superseded — one model, two memory arms

D5 specified "2 models, cheap + flagship, same family" to ask *does capability buy trading skill, and does it buy faster in-context learning?* **The lineup is now Claude Opus 4.8 only.** The model-vs-model comparison is dropped; the paired Wilcoxon A-vs-B claim has no B.

The freed budget buys a second arm on the same model instead: **memory on vs memory off**, promoting D11's optional ablation to the headline.

*Why this is a better experiment than the one it replaces, not merely a cheaper one:* D11 measures learning as "divergence from the no-memory baselines' per-episode trace". But the baselines are *different strategies* — buy-and-hold's per-window Sharpe confounds window difficulty with the fact that it is a different policy generator. A memory-off Opus arm is the same model, the same frozen prompt, the same windows, in the same order, differing in exactly one bit. That is a proper paired control, and it isolates in-context learning in a way the baseline trace cannot. Given the plan's own admission that n = 30 is power-limited and its effect-size floor of 0.005 Sharpe/episode, the learning claim is the one most worth strengthening.

The baseline trace is retained as a diagnostic. It is no longer the only evidence for the learning claim.

**Lineup:** 1 model × 2 arms × 2 tracks × 30 windows = **120 episodes**, in 4 lanes.

| agent id | kind | memory | track lanes |
|---|---|---|---|
| `opus_mem` | `llm` | `rolling_note` | real, twin |
| `opus_nomem` | `llm` | `none` | real, twin |

Both are legal under the existing schema (`AGENT_KINDS = ("llm", "baseline")`, `MEMORY_MODES = ("rolling_note", "none")`) — no schema change is needed to express the arms. Lanes are sequential within and parallel across, exactly as D5 specified, because memory is sequential.

### A2. D5's budget no longer holds

D5's $30–90 assumed a cheap model carried half the run. Two Opus arms break that arithmetic regardless of any other choice. The estimate below is **an extrapolation, not a measurement**, and is superseded by the dry run (see Verification).

At Opus rates ($5/MTok in, $25/MTok out), an 89-decision episode resends a growing history each turn for roughly **0.97M cumulative input tokens**. Uncached, that is ~$4.85/episode — ~$580 for 120 episodes, six times D5's ceiling. **Prompt caching is therefore not an optimisation; it is what makes any budget claim true at all.** With caching the input collapses to roughly $0.60/episode and *output* becomes the dominant, uncertain term:

| thinking config | est. output/episode | est. total, 120 episodes |
|---|---|---|
| off | ~4.5K tok | ~$85 |
| adaptive, `low` | ~40K tok | ~$145–205 |
| adaptive, `high` (default) | ~130K tok | ~$480 |

**Chosen: `thinking={"type": "adaptive"}`, `output_config={"effort": "low"}`.** Adaptive may well decide a hold-or-wait decision needs almost no thinking, in which case the real figure lands near the "off" row; low effort also consolidates tool calls, which serves D4's call-count goal. The dry run settles it. Benchmarking a thinking-disabled Opus was rejected: it tests a deliberately hobbled model, and 4.8 with thinking off leaks reasoning into the visible response, requiring a final-answer-only prompt clause that is itself a confound.

### A3. D13's "provider-default temperature, logged" is now trivial

**Opus 4.8 rejects `temperature`, `top_p`, and `top_k` with a 400.** The parameter does not exist on this model. D13's requirement is satisfied by construction and cannot be varied. It is logged as absent. D13's other clauses — byte-identical system prompt, disclosure of metric/fees/fill rule/tool contract, the anti-stall ladder, the wall-clock cap — stand unchanged.

---

## Runner decisions

### R1. Transport — the in-process MCP surface

The runner holds the `EpisodeSession` and calls `build_server(session)`. Tool definitions handed to the Messages API come from `srv.list_tools()` — name, description, and `inputSchema`, generated by FastMCP from the Python signatures. Dispatch goes through `srv.call_tool(name, args)`.

*Why:* there is exactly one tool contract, generated from one source, so the tools the model sees cannot drift from the tools the engine implements. `tests/test_mcp.py` already proves this path logs byte-identically to the direct in-process path, so this is not a shortcut around MCP — it *is* the MCP surface, minus the stdio hop.

*Why not a real stdio subprocess:* one server process is one episode by design, with deliberately no tool to change it. But `EpisodeSession.finish(memory_note_in=, memory_note_out=, cost=)` must be called by whoever knows the token counts and the outgoing note — the runner. Over stdio the session lives in the subprocess with no channel to pass either in; `close_log(cost=)` exists precisely because "the runner knows and the engine cannot". A stdio runner would need an admin channel that the server's whole design refuses to have, in exchange for a difference the parity test says is unobservable.

`python -m pdtbench.mcp.server` remains real, tested, and the surface an external client (or the live demo) drives.

### R2. The turn — one completion, one call, one tick

`tool_choice={"type": "auto", "disable_parallel_tool_use": True}`.

**This flag is required, not preferred.** Parallel tool use is on by default: one completion could emit `Buy` and `Wait` together, the engine would advance the clock twice from a single decision, and two schema guarantees would break at once — `action` is "the accepted action" (one per tick), and `calls[].tokens` is documented as unambiguous because "one LLM completion is one call". D4's entire cadence rests on this.

Per turn:

1. `client.messages.create(...)` — measure wall-clock latency, read `usage`.
2. Accumulate cost from `usage`, **including turns that produce no tool call** (a prose reply still bills).
3. Find the single `tool_use` block.
4. Dispatch through `srv.call_tool(name, input)`, inside the attribution context of R3(a).
5. Append the assistant content — **thinking blocks echoed back unchanged**, which continuing on the same model requires — and the `tool_result` as a user turn.
6. Repeat until `session.done`.

Request shape:

```
model        = "claude-opus-4-8"
max_tokens   = 16384                      # headroom for adaptive thinking; under the
                                          # non-streaming timeout ceiling
thinking     = {"type": "adaptive"}
output_config= {"effort": "low"}
system       = [ {frozen, cache_control}, {note} ]   # note only on opus_mem
tools        = from srv.list_tools()
tool_choice  = {"type": "auto", "disable_parallel_tool_use": True}
```

### R3. Two additions the runner needs

**(a) `EpisodeSession.attributing(latency_ms=, tokens=)`** — a context manager opened immediately before dispatch.

The MCP tool functions call `session.call(tool, args)` and have no way to pass `latency_ms` or `tokens`, which the schema wants on every `calls[]` entry. Going through the served surface would otherwise silently drop per-call attribution. Because R2 guarantees one completion is exactly one call, the attribution is unambiguous — the property the schema's own comment says it wants.

```python
with session.attributing(latency_ms=lat, tokens={"in": ..., "out": ...}):
    payload = dispatch(name, args)
```

**(b) `TradingEnv.force_wait(reason)` and a second `forced` trigger** — schema `tick_log.md` amended.

D13 says a prose reply with no tool call "gets one nudge, then a forced Wait". The engine cannot observe a prose reply — it only ever receives tool calls — and `tick_log.md` currently states `action.forced` has **exactly one** trigger: `max_consecutive_invalid`. A runner-imposed Wait would therefore log `forced: false`, crediting the agent with a decision it never made.

`forced` gains a second trigger, recorded distinctly:

| trigger | raised by | meaning |
|---|---|---|
| `max_consecutive_invalid` | engine | three consecutive invalid calls |
| `prose_stall` | runner, via `force_wait` | a reply with no tool call, after one nudge |

*Rejected:* synthesising a fake invalid call to ride the existing ladder. `calls[]` is the reliability record and must say what the agent actually sent; putting a tool name in it that the model never emitted is the same class of lie the "`action` must BE the call that advanced time" rule exists to prevent.

`reliability.n_prose_nudges` finally has a partner: forced waits are attributable to their cause.

### R4. Schema v1.3.0 — the cost block must be able to express a cached run

Current: `cost: {tokens_in, tokens_out, usd?}`.

With caching, `usage.input_tokens` is only the **uncached remainder**. The real prompt is `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, and the three buckets price at 1×, 0.1×, and 1.25×. Logged as-is, `tokens_in` would read ~4K where the prompt was ~30K, and `usd` would be an unauditable magic number — unrecomputable from the log. That contradicts the one claim the replay verifier exists to defend: **any number on the scoreboard can be regenerated from the logs.**

Amended:

```
cost: {
  tokens_in,            # int — uncached input only (usage.input_tokens)
  tokens_out,           # int — includes thinking tokens
  cache_read_tokens?,   # int — usage.cache_read_input_tokens
  cache_write_tokens?,  # int — usage.cache_creation_input_tokens
  usd?,                 # num — recomputable from the four counts + the price table
}
```

Additive; the three cache fields are optional and absent for baselines. **The 240 baseline logs already on disk stay valid.**

Price table lives in `agent/cost.py`, keyed by model id, and is stamped into the run manifest so the analysis reads prices from the run rather than from whatever the table says on the day someone re-runs it:

```
usd = (tokens_in·$5.00 + cache_read·$0.50 + cache_write·$6.25 + tokens_out·$25.00) / 1e6
```

### R5. Memory (D11)

- **The note lives in a second system block**, after the frozen one: `system = [{frozen, cache_control}, {note}]` on `opus_mem`, `[{frozen, cache_control}]` on `opus_nomem`. Caching is a prefix match, so ordering it this way means the per-episode note never invalidates anything ahead of it, and the frozen block stays byte-identical across both arms — D13 holds. (It does **not** follow that the arms share a cache entry for the frozen prefix: per R6 that breakpoint sits under the 4096-token minimum and will not fire. The ordering is correct regardless, and pays off if the prompt ever grows past it.)
- **Written by a reflection call at episode end, not by a tool.** A `SaveMemory` tool would have to exist in the `Tool` enum, so `opus_nomem` would see a different tool surface — a different contract and a different cached prefix — confounding the exact comparison the second arm exists to make.
- The reflection call **reuses the episode's own conversation** (a cache read, so nearly free) plus a closing user turn asking for a replacement note. That is the most informed note available at almost no cost.
- **≤500 tokens, enforced by `count_tokens`** on the returned note. Over → one re-ask stating the measured count → still over → hard truncate, and the truncation is reported. The note is logged verbatim as used.
- Scoped per (arm, track); reset between tracks. `memory_note_in` is `null` on episode 0 of each lane. The note *replaces* the previous one, so context length is constant across episodes (D11).
- `opus_nomem` never receives a note and never makes the reflection call. `memory_note_in` and `memory_note_out` stay `null` for its whole lane.

### R6. Caching and cost

Two breakpoints (of the four allowed):

1. On the frozen system block.
2. Rolling, on the last content block of the newest turn.

**The 4096-token minimum bites.** Opus 4.8 will not cache a prefix below 4096 tokens — silently, with no error, just `cache_creation_input_tokens: 0`. Frozen system + tool schemas is ~2,300 tokens, so **breakpoint 1 will not fire** until the prompt grows past the minimum. It is kept because it is harmless and starts working if that happens; it is not load-bearing.

Breakpoint 2 carries the run. It covers system + tools + history, which crosses 4096 around turn ~10. Early turns pay full freight and there is no cross-episode reuse of the frozen prefix (~$0.01/episode — negligible). Three blocks per turn (thinking + tool_use in the assistant message, tool_result in the user message) is comfortably inside the 20-block lookback.

**Silent invalidators, all avoided by construction:** no timestamp or UUID in the frozen prompt or tool defs; tool order is FastMCP registration order (deterministic); observation payloads are FastMCP's own serialization (deterministic).

**Known cost risk, reported not capped:** `fetchData(200)` returns ~6.6K tokens and stays in history permanently, so every later turn resends it. D8 says a 200-bar lookback is honorable from tick 0, so it is not capped. Turnover of the read budget is reported on the reliability scoreboard.

**Budget guard:** a running USD total per lane. If the projected lane total exceeds the ceiling, the lane stops and reports rather than silently burning budget.

### R7. The probe (D9)

Separate from the trading loop: no memory note, no tools. One `Question` carries **both** axes — an 8-way ticker choice and an 8-way period choice answered in a single call — so the cost is 30 windows × 2 tracks × 5 reps = **300 calls**, not 600.

**Almost all of this is already built.** `analysis/leakage.py` shipped with step 6 and already implements the whole harness behind a pluggable `ProbeFn` adapter — deliberately, so the analysis could be validated against a stub whose accuracy we control:

| already exists | behaviour |
|---|---|
| `build_options(window_id, truth, ticker_pool, period_pool, n_options)` | Seeded off `window_id` **alone, never the track** — a window's real series and its twin must be offered an identical option set, or `p_real − p_twin` measures our distractors instead of the model's memory. |
| `build_questions(window_id, track, truth, options, n_reps)` | **Already shuffles option order per rep**, seeded from `_seed(window_id, track, rep)`. Its docstring: "a model with a position bias ('always pick the third option') would post a stable non-chance accuracy and we would read it as recall." |
| `ask(q, bars, probe_fn)` | Runs one question. An unparseable or out-of-set answer scores wrong and is *recorded* — never dropped, because dropping failures inflates accuracy. |
| `score()`, `write_probe_file()` | Scoring and the on-disk artifact. |

The concern that motivated a fix here is real — Opus 4.8 has no `temperature`, so five reps of a byte-identical prompt would be five draws from a near-deterministic function and per-window identifiability would collapse to {0, 1}. **But the shuffling that prevents it predates this spec**, and `leakage.py` already documents the residual quantization honestly: it is measurement error in the regressor, which biases the OLS slope *toward zero*, making the headline a conservative estimate rather than an inflated one.

**So step 5 builds only the adapter and the driver**, not the harness:

- `agent/probe.py` implements `ProbeFn`: `(bars, options, context) -> {"ticker": ..., "period": ..., "raw": ..., "tokens": {...}}`. The returned values must be members of `options[axis]`; `ask()` scores anything else as wrong.
- Answer schema is a **constant `A`–`H` enum** via `output_config.format`, with the shuffled option→letter mapping rotating underneath and the adapter mapping back. A constant schema keeps structured-output compilation cached; putting the shuffled ticker strings in the enum would recompile it every rep.
- Bars go **first, behind a cache breakpoint** (~9.5K tokens for all 290 presented bars — comfortably over the 4096 minimum), options after it. Each (window, track)'s bars are written once and read 4 times across its 5 reps.
- Pools: `ticker_pool` is every ticker in `data/processed/prices.parquet` (the full fetched universe, not just the 30 selected — distractors drawn only from selected windows would leak the selection); `period_pool` is the years spanned by the dataset. `truth = {"ticker": real_spec["ticker"], "period": real_spec["date_scored_start"][:4]}`, built from the **real** window's manifest entry and used for both tracks.
- **Probing a twin asks the same question with the source window's ticker as the "correct" answer.** That is how the twin arm calibrates the false-positive rate (D9): a model that "identifies" a synthetic path is pattern-matching, not recalling.
- Single identity, no memory note. Identifiability is a property of the weights, not of an arm, and both arms are the same Opus 4.8. Probing per arm would pay double to measure the same quantity twice. A note reading "w13 looked like the COVID crash in NVDA" would contaminate the probe and make D9 measure what the agent *wrote down* rather than what the model *recalls*.
- Both arms' Sharpe edge then plots against one shared identifiability x-axis — and the pair answers a bonus question: **does memory make the model exploit identifiability more?**
- Written via the existing `analysis.leakage.write_probe_file()`.

### R8. Failures, budget, resume

| condition | handling |
|---|---|
| `RateLimitError`, 5xx | SDK auto-retries with backoff (`max_retries`). Beyond that, the lane backs off and retries. |
| retries exhausted | episode marked `agent_error` (already a legal schema status); lane continues. |
| `stop_reason == "refusal"` | no valid tool call → the prose ladder (nudge → `force_wait`). Logged. |
| `stop_reason == "max_tokens"` | completion truncated mid-tool-call → no valid tool call → the prose ladder. |
| projected lane cost > ceiling | lane stops and reports. |
| wall-clock > 20 min | already handled by `EpisodeSession` (D13): remaining ticks auto-Wait, `status: wallclock_capped`. |

**Resume:** episodes are files. A lane skips any window whose log already exists and passes `validate_episode` + `replay`. This matters for a 120-episode overnight run — a crash at episode 90 must not cost 90 episodes of budget.

---

## Module layout

A new package, a *client* of the MCP door, sitting exactly where `mcp/baselines.py` sits. The engine, session, and server are untouched but for R3.

```
src/pdtbench/agent/
  __init__.py
  prompt.py    the frozen system prompt (D13) + its sha256 → agent.system_prompt_sha256
  tools.py     Anthropic tool defs from srv.list_tools(); dispatch via srv.call_tool()
  loop.py      one episode: the decision loop, the prose ladder, attribution
  memory.py    the rolling note: reflection call, ≤500 tokens, per-lane state
  cost.py      usage → USD; the price table; the budget guard
  probe.py     the ProbeFn adapter + the driver loop over windows and pools;
               the harness itself already lives in analysis/leakage.py
  lanes.py     4 lanes (arm × track); sequential within, parallel across; resume
scripts/run_agents.py    --dry-run | --lanes | --probe | --resume
```

Touched elsewhere: `engine/env.py` (`force_wait`), `mcp/session.py` (`attributing`), `schema.py` + `schemas/tick_log.md` (v1.3.0 cost block, second `forced` trigger), `pyproject.toml` (`anthropic`).

---

## Verification

**Unit (`tests/test_agent.py`) — a fake Anthropic client returning canned completions. No network.**

- Tool defs built from `srv.list_tools()` are exactly the engine's `Tool` enum (extends the served-surface test).
- `disable_parallel_tool_use` is set — a completion with two tool_use blocks would break tick semantics.
- One completion = one call = one tick; `calls[].tokens` and `latency_ms` are attributed to the right call.
- Prose reply → one nudge → `force_wait(prose_stall)`; `action.forced` is `true`, and `reliability.n_prose_nudges` counts it.
- `opus_mem`'s system is byte-identical to `opus_nomem`'s but for the appended note block (D13).
- Cost: a `usage` carrying all four buckets produces a USD figure equal to a hand-computed one, and is recomputable from the logged counts.
- The note is ≤500 tokens; an over-long note triggers one re-ask, then truncation.
- The probe adapter returns members of `options[axis]` (anything else is scored wrong by `ask()`, which is already tested); the `A`–`H` answer enum is constant across reps while the option→letter mapping rotates.
- The budget guard trips.
- **An episode driven entirely by the fake client still passes `validate_episode` and `replay`** — the runner cannot produce a log the rest of the pipeline rejects.

**Dry run (real API) — the gate before the batch, as the plan's Verification section already requires.**

One full episode, end to end:

1. Assert `cache_read_input_tokens > 0`. **This single assertion is what stands between the run and a silent 4× cost overrun** — a prefix that never caches produces no error, only a bill.
2. Report measured USD, token counts by bucket, call count, and wall-clock.
3. Extrapolate to 120 episodes and compare against the ceiling **before** committing to the batch.
4. Schema-validate and replay the episode.

---

## Open risks

- **The `low`-effort choice is a guess until the dry run.** If adaptive thinking spends more than ~40K output tokens per episode, the run exceeds any reasonable budget and the effort/thinking decision reopens. This is why the dry run is a gate and not a formality.
- **`fetchData(200)` inflates every subsequent turn.** A model that pulls 200 bars repeatedly could double an episode's cost. Reported, not capped, per D8.
- **Analytics has never seen a real agent.** `analysis/learning.py` was written against a fixture whose LLM arms were planted. An `llm`/`memory: none` arm is a trace shape it has not encountered; step 6 may need adjustment once real logs exist.
- **`n = 30` remains power-limited**, exactly as the plan says on the slide. The memory arm improves the control, not the sample size.
