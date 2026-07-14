# Agent Runner Implementation Plan (build order step 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the benchmark call a real model — 120 trading episodes (Opus 4.8 × 2 memory arms × 2 tracks × 30 windows) plus D9's 300-call leakage probe — producing tick logs the existing replay/validate/analysis pipeline already accepts.

**Architecture:** A new `src/pdtbench/agent/` package that is a *client* of the MCP door, sitting exactly where `mcp/baselines.py` sits. The runner holds an `EpisodeSession`, calls `build_server(session)`, hands the model tool definitions generated from `srv.list_tools()`, and dispatches through `srv.call_tool()`. `tests/test_mcp.py` already proves that path logs byte-identically to the direct path. Engine, session, and server are otherwise untouched but for two small additions (Tasks 3 and 4).

**Tech Stack:** Python 3.12, `anthropic` SDK, `mcp` (FastMCP), numpy, pandas, pytest. Venv at `~/.venvs/pdt` (outside the repo — it lives on a OneDrive-synced mount).

**Spec:** `docs/superpowers/specs/2026-07-14-agent-runner-design.md`. Read it before starting; it explains *why* for every decision below.

## Global Constraints

- **Run everything with `~/.venvs/pdt/bin/python`.** There is no in-repo venv.
- **Model is `claude-opus-4-8`, exactly that string.** Never a date suffix.
- **Never send `temperature`, `top_p`, or `top_k`** — Opus 4.8 rejects them with a 400. `agent.temperature` stays absent from the log.
- **Never send `thinking: {type: "enabled", budget_tokens: N}`** — removed on 4.8, returns 400. Only `{"type": "adaptive"}`.
- **Never send a last-assistant-turn prefill** — returns 400 on 4.8.
- **`tool_choice` must always carry `disable_parallel_tool_use: True`.** One completion = one tool call = one tick. Without it the engine can advance twice from one decision.
- **Echo assistant `thinking` blocks back unchanged** on subsequent turns. Do not strip, edit, or reconstruct them.
- **Baseline compatibility is a hard requirement.** The 240 logs in `runs/baselines_dev/` and every existing test must stay green. Every schema change in this plan is additive.
- **No network in unit tests.** `tests/test_agent.py` uses a fake client. Only Task 11 touches the API.
- Existing suite is **178 tests green** on branch `mcp-baselines-and-runner-spec`. Run `~/.venvs/pdt/bin/python -m pytest -q` after every task; it must stay green.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pdtbench/config.py` | **Modify** — `SCHEMA_VERSION` → `"1.3.0"`, single source of truth |
| `src/pdtbench/schema.py` | **Modify** — import `SCHEMA_VERSION` from config; cost block gains cache buckets; `action.forced_reason` |
| `schemas/tick_log.md` | **Modify** — document both |
| `src/pdtbench/engine/env.py` | **Modify** — `force_wait(reason)`; existing forced wait gains `forced_reason` |
| `src/pdtbench/mcp/session.py` | **Modify** — `attributing()` context manager |
| `src/pdtbench/agent/cost.py` | **Create** — price table, `Usage` accumulator, USD |
| `src/pdtbench/agent/prompt.py` | **Create** — the frozen system prompt (D13) + its sha256 |
| `src/pdtbench/agent/tools.py` | **Create** — `ServedSurface`: sync facade over one episode's MCP tool surface |
| `src/pdtbench/agent/loop.py` | **Create** — one episode: the decision loop, the prose ladder, attribution |
| `src/pdtbench/agent/memory.py` | **Create** — the rolling note: reflection call, ≤500 tokens |
| `src/pdtbench/agent/probe.py` | **Create** — the `ProbeFn` adapter + driver (harness already in `analysis/leakage.py`) |
| `src/pdtbench/agent/lanes.py` | **Create** — 4 lanes, resume, budget guard |
| `scripts/run_agents.py` | **Create** — CLI |
| `tests/test_agent.py` | **Create** — the whole loop against a fake client |

---

## Task 1: Schema v1.3.0 — the cost block can express a cached run

**Files:**
- Modify: `src/pdtbench/config.py:16`
- Modify: `src/pdtbench/schema.py:27`, `src/pdtbench/schema.py:252-256`
- Modify: `schemas/tick_log.md`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pdtbench.config.SCHEMA_VERSION == "1.3.0"`; `pdtbench.schema.SCHEMA_VERSION` re-exported from config; `episode_end.cost` accepts optional `cache_read_tokens: int`, `cache_write_tokens: int`.

**Why:** with caching, `usage.input_tokens` is only the *uncached remainder*. The real prompt is `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, priced at 1×, 0.1×, 1.25×. Logged as-is, `usd` would be unrecomputable from the log — contradicting the one claim the replay verifier exists to defend.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_schema.py`. It already has `_valid_log(tmp_path)` (builds a valid log via `fixtures.make_episode`) and `_mutate(path, fn)` (rewrites records in place); use those — the file validates whole files with `validate_episode`, and does not import `validate_record`.

```python
def test_the_cost_block_admits_the_cache_buckets(tmp_path):
    """With caching, `tokens_in` is only the uncached remainder. A log that cannot name
    what it read from cache cannot regenerate its own `usd`. The schema rejects
    undeclared fields, so these have to be declared to be writable at all."""
    log = _mutate(_valid_log(tmp_path), lambda r: r[-1]["cost"].update({
        "cache_read_tokens": 89_012, "cache_write_tokens": 3_456, "usd": 0.123456,
    }))
    rep = validate_episode(log)
    assert rep.ok, rep.errors[:3]


def test_a_cost_block_without_the_cache_buckets_still_validates(tmp_path):
    """The 240 baseline logs on disk have no cache fields, and the fixture writes none.
    This change is additive or it is a breaking one."""
    assert validate_episode(_valid_log(tmp_path)).ok


def test_the_schema_version_has_one_source_of_truth():
    """It was declared in two modules. Bumping one and not the other would write 1.2.0
    into a log while validating it against 1.3.0's rules -- silently."""
    from pdtbench import config, schema
    assert schema.SCHEMA_VERSION is config.SCHEMA_VERSION
    assert config.SCHEMA_VERSION == "1.3.0"
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_schema.py -q -k "cost_block or source_of_truth"
```
Expected: FAIL — `cache_read_tokens` rejected as an unknown key, and `SCHEMA_VERSION == "1.2.0"`.

- [ ] **Step 3: Bump the version in `config.py` and collapse the duplicate**

`src/pdtbench/config.py:16`:

```python
SCHEMA_VERSION = "1.3.0"
```

`src/pdtbench/schema.py` — delete the local `SCHEMA_VERSION = "1.2.0"` at line 27 and re-export instead. Add to the import block:

```python
from .config import SCHEMA_VERSION  # noqa: F401 -- re-exported; one source of truth
```

**Verify no import cycle before proceeding:** `config.py` imports only `.hashing`, so `schema.py → config.py` is acyclic. Confirm with:

```bash
~/.venvs/pdt/bin/python -c "import pdtbench.schema, pdtbench.config; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Widen the cost block**

`src/pdtbench/schema.py`, replace the `cost` entry:

```python
    "cost": Obj({
        "tokens_in": Int(),   # uncached input only -- `usage.input_tokens`
        "tokens_out": Int(),  # includes thinking tokens
        # Absent for baselines, which make no API calls. Present for any agent whose
        # prompt was cached: without them `usd` is a magic number nobody can recompute,
        # because the three input buckets price at 1x, 0.1x and 1.25x.
        "cache_read_tokens": Int(required=False),
        "cache_write_tokens": Int(required=False),
        "usd": Num(required=False),
    }),
```

- [ ] **Step 5: Document it in the schema spec**

In `schemas/tick_log.md`, in the `episode_end` field table, add rows beneath `cost.tokens_out`:

```markdown
| `cost.cache_read_tokens` | int | opt | `usage.cache_read_input_tokens`, summed over the episode. Absent for baselines. |
| `cost.cache_write_tokens` | int | opt | `usage.cache_creation_input_tokens`, summed over the episode. Absent for baselines. |
```

And amend the sentence describing `cost.tokens_in` to say plainly: **`tokens_in` is the uncached remainder, not the prompt size.** The prompt is `tokens_in + cache_read_tokens + cache_write_tokens`. Bump the version header of the document to **1.3.0** and add a changelog line:

```markdown
1.3.0 — cost gains the cache buckets, so usd is recomputable from the log.
```

**Only the cost half.** `action.forced_reason` also lands in 1.3.0, but in Task 3 — and this document's preamble says that when the module and the page disagree, the page is a bug. Naming a field before it exists makes this commit self-contradicting. Task 3 appends its own half of the line.

- [ ] **Step 6: Update the existing test that pins the version**

`tests/test_schema.py::test_the_schema_version_is_stamped` asserts the literal:

```python
    assert meta["schema_version"] == SCHEMA_VERSION == "1.2.0"
```

The literal is deliberate — it stops a bump from sliding through unnoticed. Update it, do not loosen it:

```python
    assert meta["schema_version"] == SCHEMA_VERSION == "1.3.0"
```

- [ ] **Step 7: Run the full suite**

```bash
~/.venvs/pdt/bin/python -m pytest -q
```
Expected: PASS, 181 tests (178 + 3 new). If `test_the_schema_version_is_stamped` fails, Step 6 was skipped.

- [ ] **Step 8: Verify the 240 baseline logs still validate**

```bash
~/.venvs/pdt/bin/python -c "
from pathlib import Path
from pdtbench.schema import validate_episode
import sys; sys.path.insert(0,'src')
logs = sorted(Path('runs/baselines_dev/episodes').glob('*.jsonl'))
bad = [p.name for p in logs if not validate_episode(p).ok]
print(f'{len(logs)-len(bad)}/{len(logs)} valid'); assert not bad, bad[:3]"
```
Expected: `240/240 valid`. (If `runs/baselines_dev` is absent, regenerate with `~/.venvs/pdt/bin/python scripts/run_baselines.py --out runs/baselines_dev --validate` first.)

- [ ] **Step 9: Commit**

```bash
git add src/pdtbench/config.py src/pdtbench/schema.py schemas/tick_log.md tests/test_schema.py
git commit -m "schema: the cost block can express a cached run (v1.3.0)"
```

---

## Task 2: `agent/cost.py` — the price table and USD

**Files:**
- Create: `src/pdtbench/agent/__init__.py`, `src/pdtbench/agent/cost.py`
- Modify: `pyproject.toml`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 1's cost block shape.
- Produces:
  - `PRICES: dict[str, dict[str, float]]` keyed by model id, values `{"input": float, "output": float}` in USD per million tokens.
  - `CACHE_READ_MULTIPLIER = 0.10`, `CACHE_WRITE_MULTIPLIER = 1.25`
  - `class Usage` with fields `tokens_in: int`, `tokens_out: int`, `cache_read_tokens: int`, `cache_write_tokens: int`; methods `add(usage_obj) -> Usage`, `usd(model: str) -> float`, `as_cost_block(model: str) -> dict`.

- [ ] **Step 1: Add the dependency**

`pyproject.toml`, in `dependencies`, keeping the list alphabetical:

```toml
dependencies = [
    "anthropic",
    "mcp",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "yfinance",
]
```

Install:

```bash
~/.venvs/pdt/bin/pip install anthropic
~/.venvs/pdt/bin/python -c "import anthropic; print(anthropic.__version__)"
```
Expected: a version string.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_agent.py`:

```python
"""The agent runner -- driven entirely by a fake client. No network.

The runner is the only thing that can know what an episode cost, so these tests are
where that arithmetic is pinned. Everything else in the pipeline recomputes its numbers
from the logs; `usd` is the one figure that cannot be, unless the log carries every
bucket that went into it.
"""

from __future__ import annotations

import pytest

from pdtbench.agent import cost as C


def test_usd_prices_each_bucket_at_its_own_rate():
    """A million tokens through each bucket, so the arithmetic is readable:
    5.00 input + 0.50 cache read (0.1x) + 6.25 cache write (1.25x) + 25.00 output."""
    u = C.Usage(tokens_in=1_000_000, tokens_out=1_000_000,
                cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert u.usd("claude-opus-4-8") == pytest.approx(36.75)


def test_a_cached_run_costs_a_tenth_of_an_uncached_one():
    """The whole budget rests on this. If cache reads ever priced at par, the run costs
    ~6x D5's ceiling and nothing else in the suite would notice."""
    uncached = C.Usage(tokens_in=1_000_000)
    cached = C.Usage(cache_read_tokens=1_000_000)
    assert cached.usd("claude-opus-4-8") == pytest.approx(uncached.usd("claude-opus-4-8") / 10)


def test_the_cost_block_can_be_recomputed_from_what_it_logs():
    """`usd` must be regenerable from the log alone -- the claim the replay verifier
    exists to defend. If a bucket is missing from the block, it is not."""
    u = C.Usage(tokens_in=1234, tokens_out=567,
                cache_read_tokens=89_012, cache_write_tokens=3_456)
    block = u.as_cost_block("claude-opus-4-8")
    again = C.Usage(
        tokens_in=block["tokens_in"], tokens_out=block["tokens_out"],
        cache_read_tokens=block["cache_read_tokens"],
        cache_write_tokens=block["cache_write_tokens"],
    )
    assert again.usd("claude-opus-4-8") == pytest.approx(block["usd"], abs=1e-6)


def test_add_accumulates_an_api_usage_object():
    class _U:  # the shape the SDK returns
        input_tokens, output_tokens = 10, 20
        cache_read_input_tokens, cache_creation_input_tokens = 30, 40

    u = C.Usage()
    u.add(_U()).add(_U())
    assert (u.tokens_in, u.tokens_out, u.cache_read_tokens, u.cache_write_tokens) == (20, 40, 60, 80)


def test_add_tolerates_a_usage_without_cache_fields():
    """An uncached response may omit them entirely; a None must not poison the sum."""
    class _U:
        input_tokens, output_tokens = 10, 20
        cache_read_input_tokens = None
        cache_creation_input_tokens = None

    u = C.Usage().add(_U())
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_an_unknown_model_is_refused_rather_than_priced_at_zero():
    """A silent 0.0 would look like a free run."""
    with pytest.raises(KeyError):
        C.Usage(tokens_in=1).usd("claude-not-a-model")
```

- [ ] **Step 3: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pdtbench.agent'`

- [ ] **Step 4: Write the implementation**

Create `src/pdtbench/agent/__init__.py`:

```python
"""The agent runner (build order step 5) -- a client of the MCP door.

Sits exactly where `mcp/baselines.py` sits: the baselines drive `EpisodeSession.call()`
in-process, and so does this. The difference is only who decides what to call.

`loop` and `probe` are deliberately not imported here: they are the only modules that
need the Anthropic SDK, and `cost` is useful without it.
"""

from __future__ import annotations

from .cost import PRICES, Usage

__all__ = ["PRICES", "Usage"]
```

Create `src/pdtbench/agent/cost.py`:

```python
"""What an episode cost.

The engine cannot answer this: it has no price table, and the API splits input across
three buckets that price differently. That is why `close_log(cost=)` exists and why the
runner owns it.

The arithmetic is here rather than inline in the loop because `cost.usd` in the tick log
has to be *auditable*: every number on the scoreboard is regenerable from the logs, and
`usd` is only regenerable if the log carries all four buckets and the price table is a
declared constant rather than a literal buried in a call site.
"""

from __future__ import annotations

from dataclasses import dataclass

#: USD per million tokens, by model id. From the published price list; stamped into the
#: run manifest so the analysis reads prices from the run rather than from whatever this
#: table says on the day someone re-runs it.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}

#: A cache read costs a tenth of base input. This multiplier is the entire budget: at par
#: the 120-episode run costs ~6x D5's ceiling.
CACHE_READ_MULTIPLIER = 0.10
#: A 5-minute-TTL cache write costs 1.25x base input.
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    """Token counts accumulated over an episode, in the four buckets that price apart."""

    tokens_in: int = 0  # uncached input only -- NOT the prompt size
    tokens_out: int = 0  # includes thinking tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage) -> "Usage":
        """Accumulate one API response's `usage`.

        The cache fields are absent or None on an uncached response, which must not
        poison the sum -- an episode's first turn always looks like that.
        """
        self.tokens_in += usage.input_tokens or 0
        self.tokens_out += usage.output_tokens or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        return self

    def usd(self, model: str) -> float:
        """Cost in dollars. Raises on an unknown model rather than pricing it at zero --
        a silent 0.0 would look like a free run."""
        p = PRICES[model]
        return (
            self.tokens_in * p["input"]
            + self.cache_read_tokens * p["input"] * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * p["input"] * CACHE_WRITE_MULTIPLIER
            + self.tokens_out * p["output"]
        ) / 1_000_000

    def as_cost_block(self, model: str) -> dict:
        """The `episode_end.cost` block (schema v1.3.0)."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "usd": round(self.usd(model), 6),
        }
```

- [ ] **Step 5: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: PASS, 6 tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/pdtbench/agent/ tests/test_agent.py
git commit -m "agent: the price table and an auditable cost block"
```

---

## Task 3: `force_wait(reason)` — D13's prose ladder can tell the truth

**Files:**
- Modify: `src/pdtbench/engine/env.py:273-295` (`_invalid`), and add `force_wait` near `record_prose_nudge` (~line 568)
- Modify: `src/pdtbench/schema.py` (the `action` object)
- Modify: `schemas/tick_log.md`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pdtbench.schema.FORCED_REASONS = ("max_consecutive_invalid", "prose_stall")`
  - `TradingEnv.force_wait(reason: str) -> ToolResult`
  - `action.forced_reason: str` (optional, enum-constrained) in the tick log.

**Why:** D13 says a prose reply with no tool call "gets one nudge, then a forced Wait". The engine only ever receives tool calls — it cannot observe a prose reply. Today `tick_log.md` states `forced` has exactly one trigger, so a runner-imposed Wait would log `forced: false` and credit the agent with a decision it never made.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_engine.py`:

```python
def test_force_wait_takes_the_turn_away_and_says_why(make_env, tmp_path):
    """The engine cannot see a reply that made no tool call, so the runner reports it.
    The log must record the cause -- a forced Wait attributed to the agent is a lie
    about who made the decision."""
    from pdtbench.engine.replay import load

    log = tmp_path / "prose.jsonl"
    env = make_env("w00", log_path=log)
    env.reset()
    t0 = env.t

    res = env.force_wait("prose_stall")

    assert res.advanced_time and res.forced_wait
    assert env.t == t0 + 1
    env.step("Wait", {"n": 10})  # keep the episode moving
    while not env.done:
        env.step("Wait", {"n": 10})
    env.close_log()

    _meta, ticks, end = load(log)
    tick0 = ticks[0]
    assert tick0["action"]["forced"] is True
    assert tick0["action"]["forced_reason"] == "prose_stall"
    assert tick0["action"]["tool"] == "Wait"
    assert tick0["fill"] is None
    assert end["reliability"]["n_forced_waits"] >= 1


def test_the_invalid_ladder_names_its_own_trigger(make_env, tmp_path):
    """The pre-existing forced Wait must say which trigger fired, now that there are two."""
    from pdtbench.engine.replay import load

    log = tmp_path / "ladder.jsonl"
    env = make_env("w00", log_path=log)
    env.reset()
    for _ in range(env.cfg.max_consecutive_invalid):
        env.step("Teleport", {"to": "moon"})
    while not env.done:
        env.step("Wait", {"n": 10})
    env.close_log()

    _meta, ticks, _end = load(log)
    assert ticks[0]["action"]["forced"] is True
    assert ticks[0]["action"]["forced_reason"] == "max_consecutive_invalid"


def test_force_wait_after_the_episode_is_over_is_refused(make_env):
    env = make_env("w00")
    env.reset()
    while not env.done:
        env.step("Wait", {"n": 10})
    res = env.force_wait("prose_stall")
    assert not res.ok
    assert res.error.code == "EPISODE_OVER"
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_engine.py -q -k "force_wait or names_its_own_trigger"
```
Expected: FAIL — `AttributeError: 'TradingEnv' object has no attribute 'force_wait'`

- [ ] **Step 3: Add the enum and widen the action object**

`src/pdtbench/schema.py`, beside `STATUSES` (~line 37):

```python
#: Why the engine took the turn away. Two triggers, and the log names which -- crediting
#: the agent with a Wait it never chose would misattribute a decision.
FORCED_REASONS = ("max_consecutive_invalid", "prose_stall")
```

In the `action` object spec, add beneath `"forced"`:

```python
        "forced_reason": Str(required=False, enum=FORCED_REASONS),
```

- [ ] **Step 4: Name the trigger on the existing ladder**

`src/pdtbench/engine/env.py`, inside `_invalid`, change the forced-wait action dict:

```python
            self._advance(
                {"tool": str(Tool.WAIT), "args": {"n": 1}, "forced": True,
                 "forced_reason": "max_consecutive_invalid", "n_effective": 1},
                fill=None, n=1, source_tick=self.t,
            )
```

- [ ] **Step 5: Add `force_wait`**

`src/pdtbench/engine/env.py`, directly beneath `record_prose_nudge`:

```python
    def force_wait(self, reason: str) -> ToolResult:
        """Take the turn away for a cause the engine cannot see (D13).

        The engine only ever receives tool calls, so a reply that made none is invisible
        to it -- the runner has to report it. `forced_reason` is what keeps the log
        honest: without it this Wait is indistinguishable from one the agent chose, and
        `reliability.n_prose_nudges` would have no partner on the action side.

        The invalid ladder raises its own forced Wait internally and does not come
        through here.
        """
        if reason not in FORCED_REASONS:
            raise ValueError(f"forced_reason must be one of {list(FORCED_REASONS)}, got {reason!r}")
        if self.done:
            return ToolResult(
                ok=False, tool=Tool.WAIT,
                error=ToolError(ErrorCode.EPISODE_OVER, "the episode has ended"),
            )
        self.consecutive_invalid = 0
        self._advance(
            {"tool": str(Tool.WAIT), "args": {"n": 1}, "forced": True,
             "forced_reason": reason, "n_effective": 1},
            fill=None, n=1, source_tick=self.t,
        )
        return ToolResult(
            ok=True, tool=Tool.WAIT, advanced_time=True, forced_wait=True,
            obs=self._agent_obs(),
            note=f"forced Wait(1): {reason}. No P&L penalty -- but you lost a decision.",
        )
```

Extend the import at the top of `env.py`:

```python
from ..schema import AGENT_KINDS, FORCED_REASONS, MEMORY_MODES
```

- [ ] **Step 6: Update the schema spec**

In `schemas/tick_log.md`, replace the `action.forced` row's claim that there is exactly one trigger. It must now read that there are **two**, and that `forced_reason` names which:

```markdown
| `action.forced` | bool | ✓ | `true` when the engine imposed a `Wait(1)`. **Two** triggers, named by `action.forced_reason`. `max_consecutive_invalid`: `max_consecutive_invalid` consecutive invalid calls — the read cap is not a separate trigger, it chains into this one (D13), so an agent that only reads gets **11** calls in a tick, not 9. `prose_stall`: a reply carrying no tool call at all, after one nudge — the engine cannot see that, so the runner reports it via `force_wait()`. A forced Wait has **no advancing call at all**, because the agent never made one. Enforced. |
| `action.forced_reason` | str | opt | `max_consecutive_invalid` \| `prose_stall`. Present whenever `forced` is `true`. Absent otherwise. |
```

Then extend 1.3.0's changelog line, which Task 1 deliberately left as the cost half only — the field exists as of this commit, so now the page may name it:

```markdown
1.3.0 — cost gains the cache buckets, so usd is recomputable from the log;
        action gains forced_reason, because a forced Wait now has two triggers.
```

- [ ] **Step 7: Run the full suite**

```bash
~/.venvs/pdt/bin/python -m pytest -q
```
Expected: PASS. Existing forced-wait tests in `test_engine.py` and the replay tests must be unaffected — `forced_reason` is additive.

- [ ] **Step 8: Commit**

```bash
git add src/pdtbench/engine/env.py src/pdtbench/schema.py schemas/tick_log.md tests/test_engine.py
git commit -m "engine: a forced Wait names its trigger, and the prose ladder has one"
```

---

## Task 4: `EpisodeSession.attributing()` — per-call latency and tokens

**Files:**
- Modify: `src/pdtbench/mcp/session.py` (`__init__`, `call`, plus the new context manager)
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EpisodeSession.attributing(latency_ms: float | None = None, tokens: dict | None = None)` — a context manager. Within it, any `call()` that does not pass its own `latency_ms`/`tokens` picks up the pending values.

**Why:** the MCP tool functions call `session.call(tool, args)` and have no channel for the metadata the schema wants on `calls[]`. Because one completion is exactly one call (`disable_parallel_tool_use`), the attribution is unambiguous — the property the schema's own comment says it wants.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp.py`:

```python
def test_attributing_reaches_a_call_made_through_the_served_surface(windows_dir, tmp_path):
    """The MCP tool functions take only a tool and args -- there is no parameter for the
    latency and tokens the schema wants on calls[]. Without this the served surface
    silently drops per-call attribution, which is most of the reliability record."""
    baseline = BASELINES["flat"]()
    session = EpisodeSession.start(
        window_id=KNOWN, track="real", agent=baseline.agent(),
        episode_index=0, run_dir=tmp_path, run_id="test", windows_dir=windows_dir,
    )
    transport = _ViaMCP(session)
    try:
        with session.attributing(latency_ms=123.4, tokens={"in": 11, "out": 22}):
            transport.call("getStats", {})
        while not session.done:
            transport.call("Wait", {"n": 10})
    finally:
        transport.close()
    session.finish()

    _meta, ticks, _end = load(session.env.logger.path)
    first = ticks[0]["calls"][0]
    assert first["tool"] == "getStats"
    assert first["latency_ms"] == 123.4
    assert first["tokens"] == {"in": 11, "out": 22}


def test_attribution_does_not_leak_past_its_scope(windows_dir, tmp_path):
    """A baseline makes no API calls and must keep logging no tokens."""
    baseline = BASELINES["flat"]()
    session = EpisodeSession.start(
        window_id=KNOWN, track="real", agent=baseline.agent(),
        episode_index=0, run_dir=tmp_path, run_id="test", windows_dir=windows_dir,
    )
    with session.attributing(latency_ms=1.0, tokens={"in": 1, "out": 1}):
        session.call("getStats", {})
    session.call("ViewWallet", {})
    while not session.done:
        session.call("Wait", {"n": 10})
    session.finish()

    _meta, ticks, _end = load(session.env.logger.path)
    inside, outside = ticks[0]["calls"][0], ticks[0]["calls"][1]
    assert inside["tokens"] == {"in": 1, "out": 1}
    assert "tokens" not in outside
    assert "latency_ms" not in outside
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_mcp.py -q -k attribut
```
Expected: FAIL — `AttributeError: 'EpisodeSession' object has no attribute 'attributing'`

- [ ] **Step 3: Implement**

`src/pdtbench/mcp/session.py` — add to the imports:

```python
from contextlib import contextmanager
```

In `EpisodeSession.__init__`, after `self._status = "ok"`:

```python
        self._pending: dict | None = None
```

Add the context manager beneath the `call` method:

```python
    @contextmanager
    def attributing(self, latency_ms: float | None = None, tokens: dict | None = None):
        """Attach one completion's latency and tokens to the call made inside this scope.

        The MCP tool functions reach `call()` with a tool and args and nothing else, so a
        model driving the served surface would otherwise log no `latency_ms` and no
        `tokens` at all. The runner knows both, and because `disable_parallel_tool_use`
        makes one completion exactly one call, there is no ambiguity about which call
        they belong to -- which is precisely the reason the schema puts tokens on the
        call rather than on the tick.

        Nested scopes restore the outer value, and a baseline that never opens one is
        unaffected.
        """
        prev = self._pending
        self._pending = {"latency_ms": latency_ms, "tokens": tokens}
        try:
            yield self
        finally:
            self._pending = prev
```

Change `call` to consult it — replace the existing signature/body head:

```python
    def call(
        self,
        tool: str,
        args: dict | None = None,
        latency_ms: float | None = None,
        tokens: dict | None = None,
    ) -> dict:
        """Dispatch one tool call. Never raises on a bad call -- an invalid call is a
        *result*, not an exception: the agent has to see the structured error to correct
        itself, and the reliability scoreboard has to count it.

        An explicit `latency_ms`/`tokens` wins over an enclosing `attributing()` scope.
        """
        if self._pending is not None:
            if latency_ms is None:
                latency_ms = self._pending.get("latency_ms")
            if tokens is None:
                tokens = self._pending.get("tokens")
        if not self.env.done and self.elapsed_s > self.cfg.episode_wallclock_cap_s:
            self._cap_out()

        res = self.env.step(tool, args or {}, latency_ms=latency_ms, tokens=tokens)
        if res.obs is not None:
            self._obs = res.obs
        return res.as_agent_payload()
```

- [ ] **Step 4: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_mcp.py -q
```
Expected: PASS, 28 tests (26 + 2). The transport-parity tests must still pass: baselines open no attribution scope, so their logs are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/pdtbench/mcp/session.py tests/test_mcp.py
git commit -m "mcp: attribute a completion's latency and tokens to its call"
```

---

## Task 5: `agent/prompt.py` — the frozen system prompt (D13)

**Files:**
- Create: `src/pdtbench/agent/prompt.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `pdtbench.config.Config`.
- Produces:
  - `SYSTEM_PROMPT: str` — the frozen text, byte-identical for every agent.
  - `system_prompt_sha256() -> str`
  - `system_blocks(note: str | None) -> list[dict]` — the `system` parameter for the Messages API.

**Why:** D13 requires a byte-identical system prompt disclosing the scoring metric, the fee schedule, the fill rule, and the tool contract. Hiding the metric would test goal inference — a different experiment.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from pdtbench.agent import prompt as P


def test_the_prompt_discloses_everything_d13_says_it_must():
    """Hiding the metric would test goal inference -- a different experiment."""
    text = P.SYSTEM_PROMPT
    for required in (
        "vol-floored Sharpe",     # the ranking metric (D2)
        "10 bps",                 # the fee schedule (D10)
        "open of the next bar",   # the fill rule (D3)
        "$10,000",                # initial capital (D13)
        "90",                     # the episode length
    ):
        assert required.lower() in text.lower(), required


def test_the_prompt_does_not_lie_about_the_environment():
    """The prompt states the config in English and cannot interpolate it -- that would
    break the cache prefix. So the drift is caught here instead."""
    from pdtbench.config import DEFAULT as D

    assert ("$10,000" in P.SYSTEM_PROMPT) == (D.initial_capital_cents == 1_000_000)
    assert ("10 bps" in P.SYSTEM_PROMPT) == (D.fee_bps + D.slippage_bps == 10)
    assert ("at most 8" in P.SYSTEM_PROMPT) == (D.max_reads_per_tick == 8)
    assert ("Three consecutive" in P.SYSTEM_PROMPT) == (D.max_consecutive_invalid == 3)
    assert ("up to 10 bars" in P.SYSTEM_PROMPT) == (D.max_wait == 10)
    assert ("ticks 0 through 89" in P.SYSTEM_PROMPT) == (D.n_scored == 90)
    # Two separate claims sit in one sentence, and they answer to two different fields.
    # "200 further bars ... precede tick 0" is the warmup the agent can see; "a 200-bar
    # lookback is honorable immediately" is what fetchData will actually serve. Binding
    # both to fetch_lookback_cap would let n_warmup drift while the guard stayed green.
    assert ("200 further bars" in P.SYSTEM_PROMPT) == (D.n_warmup == 200)
    assert ("200-bar lookback" in P.SYSTEM_PROMPT) == (D.fetch_lookback_cap >= 200)
    assert ("0.25 * bh_daily_vol" in P.SYSTEM_PROMPT) == (D.vol_floor_multiple == 0.25)


def test_the_prompt_carries_no_invalidator():
    """Caching is a prefix match and this text is the prefix. A date, a uuid or an
    interpolated id here would silently cost ~6x -- no error, just a bill.

    The sha is the real assertion: it is taken over the text the runner actually sends,
    so it moves if anything varying creeps in."""
    import datetime
    import re

    assert isinstance(P.SYSTEM_PROMPT, str)  # a constant, not a factory
    assert str(datetime.date.today().year) not in P.SYSTEM_PROMPT
    assert not re.search(r"\{[a-z_]+\}", P.SYSTEM_PROMPT)  # no unformatted placeholder
    assert P.system_prompt_sha256() == (
        "b0586a912db9c5221e6214161bbb20f90c61e1d36b32e8437869211bdf353720"
    ), "the frozen prompt changed -- this is the experiment's identity; update deliberately"


def test_the_memory_arms_share_a_byte_identical_frozen_block():
    """D13. The note must sit in its own block *after* the frozen one, or the two arms
    are not running the same experiment."""
    with_note = P.system_blocks("remember: w13 was choppy")
    without = P.system_blocks(None)

    assert without == [with_note[0]]
    assert len(with_note) == 2
    assert with_note[0]["text"] == P.SYSTEM_PROMPT
    assert with_note[0]["cache_control"] == {"type": "ephemeral"}
    assert "w13 was choppy" in with_note[1]["text"]
    assert "cache_control" not in with_note[1]


def test_an_empty_note_is_not_a_block():
    """Episode 0 of a memory lane has no incoming note; it must look exactly like the
    no-memory arm, not like an arm carrying an empty note."""
    assert P.system_blocks("") == P.system_blocks(None)
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q -k "prompt or arms_share or empty_note"
```
Expected: FAIL — `ImportError: cannot import name 'prompt'`

- [ ] **Step 3: Implement**

Create `src/pdtbench/agent/prompt.py`:

```python
"""The system prompt (D13).

Byte-identical for every agent, disclosing the scoring metric, the fee schedule, the fill
rule, and the tool contract. Hiding the metric would test goal inference, which is a
different experiment than the one this benchmark runs.

It is a module-level constant, not a function of anything. Caching is a prefix match and
this text is the prefix: a date, a uuid, or an f-string interpolation here would
invalidate the cache on every request -- silently, with no error, at roughly six times
the budget. `test_the_prompt_carries_no_invalidator` is the tripwire.

The two memory arms differ *only* in whether `system_blocks` appends a second block. The
frozen block is identical between them, which is what makes the paired comparison a
comparison.
"""

from __future__ import annotations

import hashlib

SYSTEM_PROMPT = """\
You are trading a single stock in a simulated market. The simulation is the source of \
truth: it owns your cash, your position, and every fill. You act only through tools.

THE WINDOW
You trade one 90-day window, ticks 0 through 89. Tick 89 is terminal: the final bar is \
shown to you but no action is accepted. 200 further bars of history precede tick 0 and \
are available to you from the very first tick, so a 200-bar lookback is honorable \
immediately.

HOW ORDERS FILL
You submit an action at tick t. It fills at the OPEN of the next bar -- a price you have \
not seen and cannot see. Friction is 10 bps per side (2 bps fee + 8 bps slippage), \
deterministic, charged on both entry and exit. At tick 89 any position you still hold is \
force-liquidated at the close, paying the same 10 bps. There is no way to avoid the exit \
fee by holding to the end.

WHAT YOU START WITH
$10,000 in cash. Cash earns nothing. You may not borrow and you may not short.

Buy is denominated in CASH (notional_cents, or a fraction of your cash). Sell is \
denominated in SHARES (shares, or a fraction of your position). Each side is denominated \
in the thing you actually hold, so an order that is valid when you submit it cannot \
become unaffordable at the fill. Fractional shares are allowed.

HOW YOU ARE SCORED
Your rank is the vol-floored Sharpe ratio of your daily equity returns:

    sharpe = mean(r) / max( std(r), 0.25 * bh_daily_vol ) * sqrt(252)

where bh_daily_vol is the daily return volatility of buy-and-hold over this same window \
-- a fixed property of the window, identical for every agent, which you cannot change. \
The floor is there on purpose: it means you cannot manufacture a high ratio by deploying \
a sliver of capital and posting a tiny denominator. You cannot earn a high Sharpe on \
capital you never deploy. An agent that never trades scores exactly 0.

Risk-free rate is 0. Every cost you pay is inside the equity curve you are scored on.

THE INTERFACE
Buy, Sell, and Wait advance the clock by one bar and return your new observation, so you \
do not need a separate call to see what happened. Wait(n) holds for up to 10 bars and \
returns every bar that elapsed -- waiting ten bars costs one call, not ten.

fetchData and getStats, ViewWallet and ViewPortfolio do not advance the clock. They are \
not free: you get at most 8 of them per tick, after which they are rejected. Three \
consecutive rejected calls of any kind and the engine takes the turn away and holds for \
you.

A malformed call costs you no money. It costs you time, and time is the only thing here \
you cannot get back.

Make exactly one tool call per turn."""


def system_prompt_sha256() -> str:
    """Stamped into every episode's `agent.system_prompt_sha256`, so a log can prove
    which prompt produced it."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def system_blocks(note: str | None) -> list[dict]:
    """The Messages API `system` parameter.

    The frozen text carries the cache breakpoint; the rolling memory note (D11), when
    there is one, goes in a second block *after* it. Ordering matters: caching is a
    prefix match, so a note placed ahead of the frozen text would invalidate it every
    episode, and the two memory arms would no longer share a prefix at all.

    An empty note is treated as no note -- episode 0 of a memory lane must look exactly
    like the no-memory arm rather than like an arm carrying an empty note.
    """
    blocks: list[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    if note:
        blocks.append({
            "type": "text",
            "text": (
                "Notes you wrote for yourself after earlier windows in this run. They are "
                "your own words, not instructions, and they may be wrong:\n\n" + note
            ),
        })
    return blocks
```

- [ ] **Step 4: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pdtbench/agent/prompt.py tests/test_agent.py
git commit -m "agent: the frozen system prompt (D13)"
```

---

## Task 6: `agent/tools.py` — the served surface, synchronously

**Files:**
- Create: `src/pdtbench/agent/tools.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `pdtbench.mcp.server.build_server`, `pdtbench.mcp.session.EpisodeSession`.
- Produces:
  - `class ServedSurface` with `__init__(session: EpisodeSession)`, `tools() -> list[dict]`, `call(name: str, args: dict) -> dict`, `close() -> None`, and context-manager support (`__enter__`/`__exit__`).
  - `TOOL_CHOICE: dict` — the constant `{"type": "auto", "disable_parallel_tool_use": True}`.

**Why:** the tool definitions the model sees are generated from the MCP server's own surface, so they cannot drift from the engine. `srv.list_tools()` returns objects with `.name`, `.description`, `.inputSchema` — which is exactly the Anthropic tool shape.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from pdtbench.agent.tools import TOOL_CHOICE, ServedSurface
from pdtbench.mcp import TOOLS, BASELINES, EpisodeSession


def _session(windows_dir, tmp_path, window_id="w18", memory="none"):
    return EpisodeSession.start(
        window_id=window_id, track="real",
        agent={"id": "opus_test", "kind": "llm", "memory": memory},
        episode_index=0, run_dir=tmp_path, run_id="test", windows_dir=windows_dir,
    )


def test_the_model_is_offered_exactly_the_engines_tools(windows_dir, tmp_path):
    with ServedSurface(_session(windows_dir, tmp_path)) as surface:
        defs = surface.tools()
    assert sorted(d["name"] for d in defs) == sorted(TOOLS)
    for d in defs:
        assert d["description"], d["name"]
        assert d["input_schema"]["type"] == "object"


def test_the_offered_tools_state_the_fee_schedule(windows_dir, tmp_path):
    """The descriptions reach the model verbatim. This is the surface D13's disclosure
    travels on, alongside the system prompt."""
    with ServedSurface(_session(windows_dir, tmp_path)) as surface:
        by_name = {d["name"]: d["description"] for d in surface.tools()}
    for name in ("Buy", "Sell"):
        assert "10 bps per side" in by_name[name]
        assert "{fill}" not in by_name[name]


def test_parallel_tool_use_is_disabled():
    """Not a preference. Parallel tool use is on by default, so one completion could emit
    Buy and Wait together, the engine would advance twice from one decision, and both
    `action` (one per tick) and `calls[].tokens` ("one completion is one call") would
    become ambiguous."""
    assert TOOL_CHOICE == {"type": "auto", "disable_parallel_tool_use": True}


def test_a_call_through_the_surface_returns_the_agent_payload(windows_dir, tmp_path):
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        out = surface.call("Wait", {"n": 2})
    assert out["ok"] is True
    assert out["observation"]["tick"] == 2


def test_an_invalid_call_is_a_result_not_an_exception(windows_dir, tmp_path):
    """The agent has to see the structured error to correct itself, and the reliability
    scoreboard has to count it."""
    session = _session(windows_dir, tmp_path)
    with ServedSurface(session) as surface:
        out = surface.call("Wait", {"n": 99})
    assert out["ok"] is False
    assert out["error"]["code"] == "WAIT_OUT_OF_RANGE"
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q -k "tools or surface or parallel"
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pdtbench.agent.tools'`

- [ ] **Step 3: Implement**

Create `src/pdtbench/agent/tools.py`:

```python
"""The model's tool surface -- generated from the engine's, never restated.

`build_server(session)` is the same MCP server `python -m pdtbench.mcp.server` runs; the
only thing missing here is the stdio hop. `tests/test_mcp.py` drives a baseline down both
paths and demands byte-identical logs, so this is not a shortcut around MCP -- it *is*
the MCP surface.

Going through it rather than hand-writing tool schemas is what stops the model's tools
from drifting from the engine's. FastMCP builds `name`, `description` and `inputSchema`
by introspecting the Python signatures, and that triple is already the Anthropic tool
shape; there is nothing to translate and nowhere for a third copy of the contract to
live.
"""

from __future__ import annotations

import asyncio
import json

from ..mcp.server import build_server
from ..mcp.session import EpisodeSession

#: One completion, one tool call, one tick.
#:
#: Parallel tool use is **on by default**. Without this flag a single completion could
#: emit `Buy` and `Wait` together; the engine would advance the clock twice from one
#: decision, and two schema guarantees would break at once -- `action` is "the accepted
#: action" (one per tick), and `calls[].tokens` is unambiguous only because "one LLM
#: completion is one call". D4's entire cadence rests on this.
TOOL_CHOICE: dict = {"type": "auto", "disable_parallel_tool_use": True}


class ServedSurface:
    """A synchronous facade over one episode's MCP tool surface.

    FastMCP is async and the runner is not. One private event loop per episode is simpler
    than colouring the whole runner async, and an episode is strictly sequential anyway
    -- there is nothing to overlap.
    """

    def __init__(self, session: EpisodeSession):
        self.session = session
        self._server = build_server(session)
        self._loop = asyncio.new_event_loop()

    def __enter__(self) -> "ServedSurface":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def tools(self) -> list[dict]:
        """The Messages API `tools` parameter, in the server's own registration order.

        Order is deterministic and must stay that way: `tools` renders at position 0 of
        the cached prefix, so a reordering would invalidate every cache entry in the run.
        """
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in self._loop.run_until_complete(self._server.list_tools())
        ]

    def call(self, name: str, args: dict | None = None) -> dict:
        """Dispatch one tool call and return the payload the agent sees.

        Never raises on a bad call: an invalid call is a *result*. Open the session's
        `attributing()` scope around this to record the completion's latency and tokens.
        """
        blocks = self._loop.run_until_complete(self._server.call_tool(name, args or {}))
        return json.loads(blocks[0].text)

    def close(self) -> None:
        self._loop.close()
```

- [ ] **Step 4: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pdtbench/agent/tools.py tests/test_agent.py
git commit -m "agent: the model's tools are generated from the engine's surface"
```

---

## Task 7: `agent/loop.py` — one episode

**Files:**
- Create: `src/pdtbench/agent/loop.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `agent.cost.Usage`, `agent.prompt.system_blocks`, `agent.tools.ServedSurface`/`TOOL_CHOICE`, `EpisodeSession`.
- Produces:
  - `MODEL = "claude-opus-4-8"`, `MAX_TOKENS = 16384`, `THINKING = {"type": "adaptive"}`, `EFFORT = {"effort": "low"}`
  - `def run_episode(session, client, *, note=None, model=MODEL, max_calls=400) -> Usage`
    Drives one episode to the terminal bar and returns the accumulated `Usage`. Does **not** call `session.finish()` — the caller owns memory and cost.

**Fake client contract (used by every test below):** an object with `.messages.create(**kwargs) -> resp`, where `resp` has `.content` (a list of blocks with `.type`, and for `type == "tool_use"` also `.name`, `.input`, `.id`), `.stop_reason`, and `.usage`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from pdtbench.agent import loop as L
from pdtbench.engine.replay import load, replay
from pdtbench.schema import validate_episode


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    input_tokens, output_tokens = 100, 20
    cache_read_input_tokens, cache_creation_input_tokens = 500, 50


class _Resp:
    def __init__(self, content, stop_reason="tool_use"):
        self.content, self.stop_reason, self.usage = content, stop_reason, _Usage()


class _FakeClient:
    """Returns scripted completions. Records every request for inspection."""

    def __init__(self, script):
        self._script, self.requests = list(script), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        return self._script.pop(0) if self._script else _wait_forever()


def _tool_use(name, args):
    return _Resp([_Block("tool_use", name=name, input=args, id="tu_1")])


def _wait_forever():
    return _tool_use("Wait", {"n": 10})


def _prose(text="I think I should probably wait here."):
    return _Resp([_Block("text", text=text)], stop_reason="end_turn")


def test_one_completion_is_one_call_is_one_tick(windows_dir, tmp_path):
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_tool_use("Buy", {"fraction": 1.0})])
    L.run_episode(session, client)
    session.finish()

    _meta, ticks, _end = load(session.env.logger.path)
    assert len(ticks) == 90
    assert ticks[0]["action"]["tool"] == "Buy"
    assert len(ticks[0]["calls"]) == 1
    assert ticks[0]["calls"][0]["tokens"] == {"in": 100, "out": 20}
    assert ticks[0]["calls"][0]["latency_ms"] >= 0


def test_every_request_disables_parallel_tool_use(windows_dir, tmp_path):
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([])
    L.run_episode(session, client)
    assert client.requests
    for req in client.requests:
        assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert "temperature" not in req  # rejected with a 400 on Opus 4.8
        assert req["thinking"] == {"type": "adaptive"}


def test_a_prose_reply_is_nudged_once_then_the_turn_is_taken_away(windows_dir, tmp_path):
    """D13. The engine cannot see a reply that made no tool call, so the runner reports
    it -- and the log says so, rather than crediting the agent with a Wait it chose."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_prose(), _prose()])
    L.run_episode(session, client)
    session.finish()

    _meta, ticks, end = load(session.env.logger.path)
    assert ticks[0]["action"]["forced"] is True
    assert ticks[0]["action"]["forced_reason"] == "prose_stall"
    assert end["reliability"]["n_prose_nudges"] == 2
    assert end["reliability"]["n_forced_waits"] >= 1


def test_a_nudged_model_that_recovers_is_not_forced(windows_dir, tmp_path):
    """One nudge, then a tool call. The nudge is counted; the turn is not taken away."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_prose(), _tool_use("Buy", {"fraction": 1.0})])
    L.run_episode(session, client)
    session.finish()

    _meta, ticks, end = load(session.env.logger.path)
    assert ticks[0]["action"]["tool"] == "Buy"
    assert ticks[0]["action"]["forced"] is False
    assert end["reliability"]["n_prose_nudges"] == 1


def test_a_refusal_and_a_truncation_both_land_in_the_prose_ladder(windows_dir, tmp_path):
    """Neither carries a tool call, so neither can advance the clock on its own."""
    for stop in ("refusal", "max_tokens"):
        session = _session(windows_dir, tmp_path / stop)
        client = _FakeClient([_Resp([], stop_reason=stop), _Resp([], stop_reason=stop)])
        L.run_episode(session, client)
        session.finish()
        _meta, ticks, _end = load(session.env.logger.path)
        assert ticks[0]["action"]["forced_reason"] == "prose_stall", stop


def test_the_usage_of_a_turn_that_made_no_tool_call_is_still_billed(windows_dir, tmp_path):
    """A prose reply costs money. Dropping it would understate the run."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([_prose(), _tool_use("Wait", {"n": 10})])
    usage = L.run_episode(session, client)
    assert usage.tokens_out >= 20 * len(client.requests)


def test_thinking_blocks_are_echoed_back_unchanged(windows_dir, tmp_path):
    """Required when continuing on the same model. Dropping or editing them breaks the
    turn."""
    session = _session(windows_dir, tmp_path)
    think = _Block("thinking", thinking="hmm")
    client = _FakeClient([
        _Resp([think, _Block("tool_use", name="Wait", input={"n": 1}, id="tu_1")]),
    ])
    L.run_episode(session, client)

    second = client.requests[1]
    assistant = [m for m in second["messages"] if m["role"] == "assistant"][0]
    assert assistant["content"][0] is think


def test_the_memory_arms_send_a_byte_identical_frozen_block(windows_dir, tmp_path):
    """D13, at the wire. If these ever differ the paired comparison is not paired."""
    a = _session(windows_dir, tmp_path / "a", memory="rolling_note")
    ca = _FakeClient([])
    L.run_episode(a, ca, note="w00 was a chop window; I overtraded it.")

    b = _session(windows_dir, tmp_path / "b", memory="none")
    cb = _FakeClient([])
    L.run_episode(b, cb, note=None)

    sa, sb = ca.requests[0]["system"], cb.requests[0]["system"]
    assert sa[0] == sb[0]          # frozen block byte-identical
    assert len(sa) == 2 and len(sb) == 1
    assert "overtraded" in sa[1]["text"]


def test_an_episode_driven_by_a_fake_model_still_replays(windows_dir, tmp_path):
    """The runner cannot produce a log the rest of the pipeline rejects."""
    session = _session(windows_dir, tmp_path)
    client = _FakeClient([
        _tool_use("fetchData", {"lookback": 50}),
        _tool_use("Buy", {"fraction": 0.5}),
        _tool_use("getStats", {}),
        _tool_use("Sell", {"fraction": 1.0}),
    ])
    usage = L.run_episode(session, client)
    session.finish(cost=usage.as_cost_block(L.MODEL))

    log = session.env.logger.path
    assert validate_episode(log).ok, validate_episode(log).errors[:3]
    res = replay(log, windows_dir)
    assert res.ok, res.failures[:3]
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q -k "completion or prose or refusal or thinking or replays"
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pdtbench.agent.loop'`

- [ ] **Step 3: Implement**

Create `src/pdtbench/agent/loop.py`:

```python
"""One episode, one model.

The loop is deliberately hand-written rather than handed to the SDK's tool runner. Three
things it has to do that a generic runner will not: attribute each completion's latency
and tokens to the single call it produced, run D13's prose ladder (which the engine
cannot see, because the engine only ever receives tool calls), and bill turns that made
no tool call at all.

`run_episode` does not call `session.finish()`. Memory and cost belong to the caller --
the note is written after the episode ends, and the cost block needs the usage this
returns.
"""

from __future__ import annotations

import time

from ..mcp.session import EpisodeSession
from .cost import Usage
from .prompt import system_blocks
from .tools import TOOL_CHOICE, ServedSurface

MODEL = "claude-opus-4-8"
#: Headroom for adaptive thinking, and under the SDK's non-streaming timeout ceiling.
MAX_TOKENS = 16384
#: The only on-mode on Opus 4.8. Omitting the field runs *without* thinking; the old
#: `{"type": "enabled", "budget_tokens": N}` is removed and returns a 400.
THINKING = {"type": "adaptive"}
#: Settled by the dry run. Low also consolidates tool calls, which serves D4.
EFFORT = {"effort": "low"}

_FIRST_TURN = (
    "You are at tick 0 of 89. This is your first observation. Make exactly one tool call."
)
_NUDGE = (
    "You replied without making a tool call. Only tool calls do anything here. Make "
    "exactly one now -- Buy, Sell or Wait to act, or a read tool to look first."
)


def _tool_use_block(resp):
    """The single tool call in a completion, or None.

    `disable_parallel_tool_use` guarantees at most one, which is what makes the token
    attribution below unambiguous.
    """
    for block in resp.content or []:
        if getattr(block, "type", None) == "tool_use":
            return block
    return None


def run_episode(
    session: EpisodeSession,
    client,
    *,
    note: str | None = None,
    model: str = MODEL,
    max_calls: int = 400,
) -> Usage:
    """Drive one episode to the terminal bar. Returns the accumulated `Usage`.

    `max_calls` guards the overnight batch against a model that never terminates; the
    engine's own wall-clock cap (D13) is the other backstop.
    """
    usage = Usage()
    system = system_blocks(note)
    messages: list[dict] = [{"role": "user", "content": _FIRST_TURN}]
    nudged = False

    with ServedSurface(session) as surface:
        tools = surface.tools()

        while not session.done:
            if max_calls <= 0:
                raise RuntimeError(f"{model} never terminated the episode")
            max_calls -= 1

            t0 = time.monotonic()
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                thinking=THINKING,
                output_config=EFFORT,
                system=system,
                tools=tools,
                tool_choice=TOOL_CHOICE,
                messages=_with_cache_breakpoint(messages),
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            # Billed whether or not it produced a tool call: a prose reply costs money,
            # and dropping it would understate the run.
            usage.add(resp.usage)

            block = _tool_use_block(resp)
            if block is None:
                # D13's prose ladder. A refusal and a max_tokens truncation land here too
                # -- neither carries a tool call, so neither can advance the clock.
                session.record_prose_nudge()
                if nudged:
                    session.env.force_wait("prose_stall")
                    nudged = False
                    messages = [{"role": "user", "content": _FIRST_TURN}]
                else:
                    nudged = True
                    messages.append({"role": "user", "content": _NUDGE})
                continue

            nudged = False
            with session.attributing(
                latency_ms=latency_ms,
                tokens={"in": resp.usage.input_tokens or 0,
                        "out": resp.usage.output_tokens or 0},
            ):
                payload = surface.call(block.name, dict(block.input or {}))

            # Echoed back unchanged -- thinking blocks included, which continuing on the
            # same model requires.
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _render(payload),
            }]})

    return usage


def _render(payload: dict) -> str:
    import json

    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """A rolling breakpoint on the newest turn's last content block.

    This is the breakpoint that carries the run. The one on the system block cannot: the
    frozen prompt plus the tool schemas is ~2,300 tokens and Opus 4.8 will not cache a
    prefix under 4,096 -- silently, with no error, just a bill. This one covers system +
    tools + history, which crosses the minimum around turn ten.

    A copy, because mutating the caller's list would leave a stale breakpoint on every
    prior turn (max 4 per request).
    """
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    last = out[-1]
    content = last["content"]
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content,
                            "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content = [dict(b) if isinstance(b, dict) else b for b in content]
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        last["content"] = content
    return out
```

- [ ] **Step 4: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: PASS, 25 tests. If `test_an_episode_driven_by_a_fake_model_still_replays` fails on `metrics_regenerate`, the bug is in the runner, not the verifier — it means a call reached the engine that the log does not account for.

- [ ] **Step 5: Commit**

```bash
git add src/pdtbench/agent/loop.py tests/test_agent.py
git commit -m "agent: the episode loop, with attribution and D13's prose ladder"
```

---

## Task 8: `agent/memory.py` — the rolling note (D11)

**Files:**
- Create: `src/pdtbench/agent/memory.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `agent.cost.Usage`, `agent.loop.MODEL`.
- Produces:
  - `MAX_NOTE_TOKENS = 500`
  - `def write_note(client, previous, summary, *, model=MODEL, usage=None) -> str` — one reflection call; returns the replacement note, guaranteed ≤ `MAX_NOTE_TOKENS`.
  - `class Lane` with `note: str | None = None`, `def carry(new_note: str) -> None` — the per-(arm, track) memory, reset between tracks by constructing a new `Lane`.

**Why:** a `SaveMemory` tool would have to exist in the `Tool` enum, so the no-memory arm would see a different tool surface — a different contract and a different cached prefix — confounding the exact comparison the second arm exists to make.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from pdtbench.agent import memory as M


class _NoteClient(_FakeClient):
    def __init__(self, notes, token_counts=None):
        super().__init__([_Resp([_Block("text", text=n)], stop_reason="end_turn")
                          for n in notes])
        self._counts = list(token_counts or [])

    def count_tokens(self, **kw):
        class _C:
            input_tokens = self._counts.pop(0) if self._counts else 10
        return _C()


def test_a_note_within_the_limit_is_kept_verbatim():
    client = _NoteClient(["w18 trended hard; SMA crossovers were late."], [42])
    note = M.write_note(client, previous=None, summary={"sharpe_floored": 1.2})
    assert note == "w18 trended hard; SMA crossovers were late."


def test_an_overlong_note_is_re_asked_once_then_truncated():
    """D11 caps the note at 500 tokens so context length is constant across episodes --
    late-episode behavior must not be confounded by a growing prompt."""
    long_note, short_note = "x " * 2000, "brief."
    client = _NoteClient([long_note, short_note], [900, 5])
    note = M.write_note(client, previous=None, summary={})
    assert note == short_note
    assert len(client.requests) == 2


def test_a_note_that_stays_overlong_is_truncated_not_dropped():
    client = _NoteClient(["y " * 2000, "z " * 2000], [900, 900])
    note = M.write_note(client, previous=None, summary={})
    assert note
    assert len(note) < len("z " * 2000)


def test_the_lane_carries_one_note_and_replaces_it():
    """Rolling, not appending (D11). A growing note would grow the prompt."""
    lane = M.Lane()
    assert lane.note is None
    lane.carry("first")
    lane.carry("second")
    assert lane.note == "second"


def test_the_reflection_call_sends_no_tools():
    """It is out of band. Offering tools here would let a note-writing turn trade."""
    client = _NoteClient(["ok"], [5])
    M.write_note(client, previous=None, summary={})
    assert "tools" not in client.requests[0]
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q -k "note or lane"
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pdtbench.agent.memory'`

- [ ] **Step 3: Implement**

Create `src/pdtbench/agent/memory.py`:

```python
"""The rolling memory note (D11) -- and the whole second arm.

Written by a reflection call after the episode ends, not by a tool. A `SaveMemory` tool
would have to live in the engine's `Tool` enum, which means the no-memory arm would see a
different tool surface: a different contract, a different cached prefix, and a confound
sitting in the middle of the one comparison this arm exists to make.

The note *replaces* its predecessor rather than appending to it, so the prompt is the
same length at episode 29 as at episode 0. Otherwise late-episode behaviour would be
confounded by a growing context and "it learned" would be indistinguishable from "it had
more to read".
"""

from __future__ import annotations

import json

from .loop import MODEL

#: D11. Measured with the provider's own tokenizer, never estimated -- a character count
#: would be wrong by enough to matter at this size.
MAX_NOTE_TOKENS = 500

_ASK = """\
That window is over. Here is your scoreboard row:

{summary}

Write a note to yourself for the next window in this run. You will see it, and nothing
else from this episode. It replaces your previous note rather than adding to it, so
carry forward anything still worth knowing and drop what is not.

Hard limit: {limit} tokens. Write only the note -- no preamble."""

_TOO_LONG = """\
That note was {n} tokens; the limit is {limit}. Rewrite it shorter. Only the note."""


class Lane:
    """One (arm, track) memory lane.

    Scoped per lane and reset between tracks by constructing a new one, which is what
    protects the paired comparison from cross-track contamination (D11).
    """

    def __init__(self) -> None:
        self.note: str | None = None

    def carry(self, new_note: str) -> None:
        self.note = new_note


def _text(resp) -> str:
    return "".join(
        b.text for b in (resp.content or []) if getattr(b, "type", None) == "text"
    ).strip()


def _count(client, model: str, text: str) -> int:
    return client.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens


def write_note(client, previous: str | None, summary: dict, *, model: str = MODEL,
               usage=None) -> str:
    """One reflection call. Returns a note guaranteed to be within the limit.

    `usage`, when given, accumulates this call's tokens -- the note costs real money and
    a run that did not count it would understate itself.
    """
    messages = [{"role": "user", "content": _ASK.format(
        summary=json.dumps(summary, indent=2, default=str), limit=MAX_NOTE_TOKENS,
    )}]
    if previous:
        messages.insert(0, {"role": "user", "content": f"Your previous note:\n\n{previous}"})

    for attempt in range(2):
        resp = client.messages.create(
            model=model, max_tokens=2048, thinking={"type": "adaptive"},
            output_config={"effort": "low"}, messages=messages,
        )
        if usage is not None:
            usage.add(resp.usage)
        note = _text(resp)
        n = _count(client, model, note)
        if n <= MAX_NOTE_TOKENS:
            return note
        if attempt == 0:
            messages.append({"role": "assistant", "content": note})
            messages.append({"role": "user", "content": _TOO_LONG.format(
                n=n, limit=MAX_NOTE_TOKENS)})

    # Twice over. Truncate rather than drop: a shortened note is still the agent's own,
    # and dropping it would silently turn this episode into a no-memory one.
    return _truncate(client, model, note)


def _truncate(client, model: str, note: str) -> str:
    """Cut to the limit by bisecting on the real tokenizer."""
    lo, hi = 0, len(note)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count(client, model, note[:mid]) <= MAX_NOTE_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return note[:lo].rstrip()
```

- [ ] **Step 4: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: PASS, 30 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pdtbench/agent/memory.py tests/test_agent.py
git commit -m "agent: the rolling memory note, and the arm it defines"
```

---

## Task 9: `agent/probe.py` — the model adapter for D9

**Files:**
- Create: `src/pdtbench/agent/probe.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `analysis.leakage` (`build_options`, `build_questions`, `ask`, `score`, `write_probe_file`, `AXES`, `DEFAULT_N_OPTIONS`, `DEFAULT_N_REPS`), `data.windows.load_manifest`/`load_episode`.
- Produces:
  - `LETTERS = "ABCDEFGH"`
  - `def pools(processed_dir) -> tuple[list[str], list[str]]` — `(ticker_pool, period_pool)`
  - `def make_probe_fn(client, model=MODEL, usage=None) -> ProbeFn`
  - `def run_probe(run_dir, client, *, windows_dir, processed_dir, run_id, agent_id="opus", n_reps=DEFAULT_N_REPS, usage=None) -> int` — writes one probe file per (track, window); returns the count.

**Why:** `analysis/leakage.py` already implements the whole harness behind the pluggable `ProbeFn` protocol — including per-rep option shuffling seeded from `_seed(window_id, track, rep)`, and distractors seeded off `window_id` alone so a window's real series and its twin are offered an identical option set. **Do not reimplement any of it.** Step 5 supplies only the adapter and the driver.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from pdtbench.agent import probe as PR
from pdtbench.analysis import leakage as LK


def _probe_resp(letters: dict):
    import json
    return _Resp([_Block("text", text=json.dumps(letters))], stop_reason="end_turn")


def test_the_adapter_maps_letters_back_to_the_shuffled_options():
    """The enum stays A-H so structured-output compilation stays cached; the mapping
    rotates underneath. `ask()` scores anything outside the option set as wrong, so a
    mapping bug would look like a model that cannot recognise anything."""
    options = {"ticker": ["AAPL", "MSFT", "NVDA", "UPS", "F", "GE", "KO", "PG"],
               "period": ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]}
    client = _FakeClient([_probe_resp({"ticker": "C", "period": "A"})])
    fn = PR.make_probe_fn(client)
    out = fn([{"t": 0, "close": 100.0}], options, {"window_id": "w00", "rep": 0})
    assert out["ticker"] == options["ticker"][2]
    assert out["period"] == options["period"][0]


def test_an_unparseable_answer_scores_wrong_rather_than_crashing():
    """leakage.ask() records a failure as a wrong answer -- dropping failures inflates
    accuracy. The adapter must not swallow it into a crash first."""
    options = {"ticker": ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"],
               "period": ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]}
    client = _FakeClient([_Resp([_Block("text", text="I refuse")], stop_reason="end_turn")])
    q = LK.build_questions("w00", "real", {"ticker": "A1", "period": "2009"}, options, 1)[0]
    resp = LK.ask(q, [], PR.make_probe_fn(client))
    assert resp.correct == {"ticker": False, "period": False}


def test_the_bars_carry_the_cache_breakpoint_and_the_options_do_not():
    """Each (window, track)'s bars are written once and read across its reps. Options
    shuffle per rep, so anything cached must sit ahead of them."""
    options = {"ticker": list("ABCDEFGH"), "period": [str(2000 + i) for i in range(8)]}
    client = _FakeClient([_probe_resp({"ticker": "A", "period": "A"})])
    PR.make_probe_fn(client)([{"t": 0, "close": 1.0}], options, {"window_id": "w00", "rep": 0})

    content = client.requests[0]["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in content[-1]
    assert "close" in content[0]["text"]        # bars first
    assert "A" in content[-1]["text"]           # options after


def test_the_probe_offers_no_tools_and_no_memory():
    """Identifiability is a property of the weights. A note reading 'w13 looked like the
    COVID crash in NVDA' would make D9 measure what the agent wrote down."""
    options = {"ticker": list("ABCDEFGH"), "period": [str(2000 + i) for i in range(8)]}
    client = _FakeClient([_probe_resp({"ticker": "A", "period": "A"})])
    PR.make_probe_fn(client)([], options, {"window_id": "w00", "rep": 0})
    req = client.requests[0]
    assert "tools" not in req
    assert "system" not in req or req["system"] is None


def test_pools_are_drawn_from_the_whole_fetched_universe(processed_dir):
    """Distractors drawn only from the 30 selected windows would leak the selection."""
    tickers, periods = PR.pools(processed_dir)
    assert len(tickers) >= LK.DEFAULT_N_OPTIONS
    assert len(periods) >= LK.DEFAULT_N_OPTIONS
    assert len(tickers) > 30
    assert all(p.isdigit() and len(p) == 4 for p in periods)


def test_run_probe_writes_one_file_per_window_and_track(windows_dir, processed_dir, tmp_path):
    letters = _probe_resp({"ticker": "A", "period": "A"})
    client = _FakeClient([letters] * 1000)
    n = PR.run_probe(tmp_path, client, windows_dir=windows_dir,
                     processed_dir=processed_dir, run_id="t", n_reps=2)
    files = sorted((tmp_path / "probe").glob("*.json"))
    assert n == len(files) == 60
    scores = LK.load_probe_dir(tmp_path / "probe")
    assert len(scores) == 60
```

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q -k probe
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pdtbench.agent.probe'`

- [ ] **Step 3: Implement**

Create `src/pdtbench/agent/probe.py`:

```python
"""The model adapter for D9's leakage probe.

**The harness is not here.** `analysis/leakage.py` shipped with step 6 and already owns
all of it: `build_options` (seeded off `window_id` alone, never the track -- a window's
real series and its twin must be offered an identical option set, or `p_real - p_twin`
measures our distractors instead of the model's memory), `build_questions` (which already
shuffles option order per rep, defeating position bias), `ask` (which records an
unparseable answer as wrong rather than dropping it, because dropping failures inflates
accuracy), `score`, and `write_probe_file`. The `ProbeFn` protocol exists precisely so
the analysis could be validated against a stub before a model existed.

This module supplies the two things that could not exist before step 5: the adapter that
asks Opus 4.8, and the driver that walks the windows.

One `Question` carries both axes, so a (window, track) costs `n_reps` calls, not
`2 * n_reps`: 30 windows x 2 tracks x 5 reps = 300.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..analysis import leakage as LK
from ..config import DEFAULT, Config
from ..data.windows import load_episode, load_manifest
from .loop import MODEL

#: The answer alphabet. Constant on purpose: the option *order* shuffles every rep, so
#: putting the shuffled ticker strings in the enum would recompile the structured-output
#: schema on every call. The letters stay; the mapping rotates underneath.
LETTERS = "ABCDEFGH"

_ANSWER_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "enum": list(LETTERS)},
            "period": {"type": "string", "enum": list(LETTERS)},
        },
        "required": ["ticker", "period"],
        "additionalProperties": False,
    },
}

_ASK = """\
Which stock and which year is this price series?

Answer with one letter per question. Guess if you are unsure -- an answer is required.

Ticker:
{ticker_options}

Year:
{period_options}"""


def pools(processed_dir: Path) -> tuple[list[str], list[str]]:
    """The distractor pools: every ticker fetched, and every year the dataset spans.

    Drawn from the whole universe rather than the 30 selected windows. Distractors
    restricted to the selection would leak the selection -- the right answer would sit in
    a set that is itself a clue.
    """
    df = pd.read_parquet(Path(processed_dir) / "prices.parquet", columns=["ticker", "date"])
    tickers = sorted(df["ticker"].unique().tolist())
    years = sorted({str(y) for y in df["date"].dt.year.unique().tolist()})
    return tickers, years


def _render_bars(bars: list[dict]) -> str:
    head = "t,open,high,low,close,volume"
    rows = [
        f"{b['t']},{b['open']},{b['high']},{b['low']},{b['close']},{b['volume']}"
        for b in bars
    ]
    return "\n".join([head, *rows])


def make_probe_fn(client, model: str = MODEL, usage=None):
    """A `ProbeFn`: `(bars, options, context) -> {"ticker", "period", "raw", "tokens"}`.

    Returns the option *values*, not the letters -- `ask()` scores anything outside
    `options[axis]` as wrong, so a mapping bug would masquerade as a model that cannot
    recognise anything.
    """

    def probe_fn(bars: list[dict], options: dict[str, list[str]], context: dict) -> dict:
        lettered = {
            axis: "\n".join(f"  {LETTERS[i]}. {opt}" for i, opt in enumerate(opts))
            for axis, opts in options.items()
        }
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "low", "format": _ANSWER_SCHEMA},
            messages=[{"role": "user", "content": [
                # Bars first, behind the breakpoint: ~9.5K tokens for 290 bars, well over
                # the 4096 minimum, written once per (window, track) and read across its
                # reps. The options shuffle every rep, so they must come after.
                {"type": "text", "text": _render_bars(bars),
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": _ASK.format(
                    ticker_options=lettered["ticker"], period_options=lettered["period"])},
            ]}],
        )
        if usage is not None:
            usage.add(resp.usage)

        raw = "".join(
            b.text for b in (resp.content or []) if getattr(b, "type", None) == "text"
        ).strip()
        picked = json.loads(raw)  # a bad payload raises; ask() records it as wrong
        out = {"raw": raw, "tokens": {"in": resp.usage.input_tokens or 0,
                                      "out": resp.usage.output_tokens or 0}}
        for axis in LK.AXES:
            idx = LETTERS.find(picked.get(axis, ""))
            out[axis] = options[axis][idx] if 0 <= idx < len(options[axis]) else None
        return out

    return probe_fn


def run_probe(
    run_dir: Path,
    client,
    *,
    windows_dir: Path,
    processed_dir: Path,
    run_id: str,
    agent_id: str = "opus",
    n_reps: int = LK.DEFAULT_N_REPS,
    cfg: Config = DEFAULT,
    usage=None,
) -> int:
    """Probe every (track, window). Returns the number of files written.

    A single identity, and no memory note: identifiability is a property of the weights,
    and both arms are the same model. Probing per arm would pay double to measure the
    same quantity twice.
    """
    run_dir = Path(run_dir)
    man = load_manifest(windows_dir)
    ticker_pool, period_pool = pools(processed_dir)
    probe_fn = make_probe_fn(client, usage=usage)

    written = 0
    for spec in man["real"]:
        wid = spec["window_id"]
        # Truth comes from the *real* window for both tracks. Scoring a twin against its
        # source's ticker is the whole twin arm: whatever accuracy that yields is the
        # false-positive rate (D9).
        truth = {"ticker": spec["ticker"], "period": spec["date_scored_start"][:4]}
        options = LK.build_options(wid, truth, ticker_pool, period_pool)

        for track in ("real", "twin"):
            series, _spec = load_episode(windows_dir, wid, track)
            # Tick-indexed like the agent sees them: -n_warmup..-1 warmup, 0..89 scored.
            bars = [
                {"t": i - cfg.n_warmup, **{k: float(v) for k, v in row.items()}}
                for i, row in enumerate(
                    series[["open", "high", "low", "close", "volume"]].to_dict("records")
                )
            ]
            responses = [
                LK.ask(q, bars, probe_fn)
                for q in LK.build_questions(wid, track, truth, options, n_reps)
            ]
            s = LK.score(agent_id, responses)
            LK.write_probe_file(
                run_dir / "probe" / f"{agent_id}__{track}__{wid}.json",
                agent_id, run_id, responses, s, n_options=LK.DEFAULT_N_OPTIONS,
            )
            written += 1
    return written
```

- [ ] **Step 4: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
```
Expected: PASS, 36 tests.

- [ ] **Step 5: Commit**

```bash
git add src/pdtbench/agent/probe.py tests/test_agent.py
git commit -m "agent: the probe adapter for D9 (the harness already existed)"
```

---

## Task 10: `agent/lanes.py` + `scripts/run_agents.py` — the run

**Files:**
- Create: `src/pdtbench/agent/lanes.py`, `scripts/run_agents.py`
- Modify: `src/pdtbench/mcp/run.py` (`write_run_manifest` gains `prices`)
- Test: `tests/test_agent.py`

**Spec R4, second half — the price table must be stamped into the run manifest.** `agent/cost.py`'s docstring already asserts this ("stamped into the run manifest so the analysis reads prices from the run rather than from whatever this table says on the day someone re-runs it"), and nothing implements it yet. A `usd` recomputed a year from now against a changed `PRICES` would silently disagree with the log it came from — the same class of unreproducibility the whole `config_sha256`/`dataset_sha256` discipline exists to prevent.

- [ ] **Step 0: Stamp the prices into the manifest**

Write the failing test first, in `tests/test_agent.py`:

```python
def test_the_run_manifest_stamps_the_price_table(windows_dir, tmp_path):
    """`usd` is only auditable if the log names the prices it was computed with. A table
    that lives solely in source drifts the moment anyone edits it."""
    from pdtbench.mcp.run import write_run_manifest

    man = write_run_manifest(tmp_path, list(LN.ARMS), windows_dir=windows_dir,
                             run_id="t", prices=C.PRICES)
    assert man["prices"] == C.PRICES
    assert man["prices"]["claude-opus-4-8"]["input"] == 5.00
    assert json.loads((tmp_path / "run_manifest.json").read_text())["prices"] == C.PRICES


def test_a_baseline_run_manifest_carries_no_prices(windows_dir, tmp_path):
    """Baselines call no API. Additive, exactly like the cost block's cache buckets."""
    from pdtbench.mcp.run import write_run_manifest

    man = write_run_manifest(tmp_path, [], windows_dir=windows_dir, run_id="t")
    assert "prices" not in man
```

Then in `src/pdtbench/mcp/run.py`, add a keyword-only `prices: dict | None = None` to `write_run_manifest`, and after the `seeds` entry:

```python
    if prices is not None:
        # The analysis recomputes `usd` from the logged token buckets; it must read the
        # rates this run actually paid, not whatever `agent/cost.py` says on the day
        # someone re-runs it. Omitted for baseline runs, which call no API.
        manifest["prices"] = dict(prices)
```

`scripts/run_agents.py` (Step 4 below) passes `prices=PRICES`. Note `run_baselines.py` calls `write_run_manifest` without it and must keep working — the 240 baseline logs and their manifest are the regression check.

**Interfaces:**
- Consumes: `agent.loop.run_episode`, `agent.memory.Lane`/`write_note`, `agent.cost.Usage`, `mcp.run.write_run_manifest`, `EpisodeSession`, `schema.validate_episode`, `engine.replay.replay`.
- Produces:
  - `ARMS: tuple[dict, ...]` — the two agent blocks (`opus_mem`, `opus_nomem`).
  - `def agent_block(arm_id: str) -> dict`
  - `def run_lane(run_dir, client, *, arm, track, windows_dir, run_id, budget_usd, usage=None, resume=True) -> Usage`
  - `def already_done(run_dir, agent_id, track, window_id, windows_dir) -> bool`
  - `class BudgetExceeded(RuntimeError)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from pdtbench.agent import lanes as LN


def test_the_two_arms_differ_in_exactly_one_bit():
    """That is the whole design of the second arm. Anything else that differs is a
    confound sitting in the middle of the comparison."""
    mem, nomem = LN.agent_block("opus_mem"), LN.agent_block("opus_nomem")
    assert mem["memory"] == "rolling_note" and nomem["memory"] == "none"
    assert mem["kind"] == nomem["kind"] == "llm"
    assert mem["model"] == nomem["model"] == "claude-opus-4-8"
    assert mem["system_prompt_sha256"] == nomem["system_prompt_sha256"]
    assert "temperature" not in mem  # Opus 4.8 rejects it with a 400
    differing = {k for k in set(mem) | set(nomem) if mem.get(k) != nomem.get(k)}
    assert differing == {"id", "memory"}


def _drive_one(windows_dir, run_dir, arm_id="opus_nomem", window_id="w00"):
    """Write one real, valid episode with a fake model."""
    from pdtbench.config import DEFAULT
    LN.run_episode_for(
        _FakeClient([]), run_dir=run_dir, run_id="t", agent=LN.agent_block(arm_id),
        track="real", window_id=window_id, episode_index=0, note=None,
        windows_dir=windows_dir, cfg=DEFAULT, usage=C.Usage(),
    )


def test_a_finished_episode_is_recognised_and_an_unstarted_one_is_not(windows_dir, tmp_path):
    """A crash at episode 90 of 120 must not cost 90 episodes of budget."""
    _drive_one(windows_dir, tmp_path)
    assert LN.already_done(tmp_path, "opus_nomem", "real", "w00", windows_dir) is True
    assert LN.already_done(tmp_path, "opus_nomem", "real", "w01", windows_dir) is False


def test_a_truncated_log_does_not_count_as_a_finished_episode(windows_dir, tmp_path):
    """A half-written log from a crash is not a shorter episode -- it is an unreadable
    one, and skipping it would put a silent hole in the lane."""
    _drive_one(windows_dir, tmp_path)
    log = tmp_path / "episodes" / "opus_nomem__real__w00.jsonl"
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n")  # lose the episode_end record
    assert LN.already_done(tmp_path, "opus_nomem", "real", "w00", windows_dir) is False


def test_a_resumed_lane_does_not_re_run_what_is_already_done(
    windows_dir, tmp_path, monkeypatch
):
    _drive_one(windows_dir, tmp_path)
    seen: list[str] = []
    real = LN.run_episode_for

    def _spy(client, **kw):
        seen.append(kw["window_id"])
        return real(client, **kw)

    monkeypatch.setattr(LN, "run_episode_for", _spy)
    LN.run_lane(tmp_path, _FakeClient([]), arm=LN.agent_block("opus_nomem"),
                track="real", windows_dir=windows_dir, run_id="t", resume=True)

    assert "w00" not in seen          # already done, skipped
    assert len(seen) == 29            # the other 29 ran


def test_the_budget_guard_stops_the_lane_rather_than_burning_it():
    usage = C.Usage(tokens_out=1_000_000)  # $25 of output
    with pytest.raises(LN.BudgetExceeded):
        LN.check_budget(usage, model="claude-opus-4-8", budget_usd=1.0)
    LN.check_budget(C.Usage(), model="claude-opus-4-8", budget_usd=1.0)  # must not raise
```

**Note for the implementer:** `test_a_resumed_lane_does_not_re_run_what_is_already_done` rebinds the module attribute, so `run_lane` must call `run_episode_for` via the module global (a plain call inside `lanes.py` does exactly that). If you refactor it into a method or a local import, the spy stops working — keep the seam.

- [ ] **Step 2: Run to verify they fail**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q -k "arms or lane or budget"
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pdtbench.agent.lanes'`

- [ ] **Step 3: Implement `lanes.py`**

Create `src/pdtbench/agent/lanes.py`:

```python
"""The run: four lanes.

    (opus_mem, real)  (opus_mem, twin)  (opus_nomem, real)  (opus_nomem, twin)

Sequential *within* a lane, because memory is sequential -- episode i's note is written
by episode i-1. Parallel *across* lanes, because nothing crosses them: memory is scoped
per (arm, track) and reset between tracks (D11), which is what protects the paired
comparison from cross-track contamination.

Every agent walks the manifest's `presentation_order`, so index i is the same window for
everyone and the baseline difficulty trace stays subtractable.
"""

from __future__ import annotations

from pathlib import Path

from ..config import DEFAULT, WINDOWS_DIR, Config
from ..data.windows import load_manifest
from ..engine.replay import replay
from ..mcp.session import EpisodeSession
from ..schema import validate_episode
from .cost import Usage
from .loop import MODEL, run_episode
from .memory import Lane, write_note
from .prompt import system_prompt_sha256

TRACKS = ("real", "twin")


class BudgetExceeded(RuntimeError):
    """The lane stopped rather than silently burning through the ceiling."""


def agent_block(arm_id: str) -> dict:
    """The `meta.agent` block. The two arms differ in exactly two keys -- `id` and
    `memory` -- and that is the entire experiment.

    No `temperature`: Opus 4.8 rejects it with a 400, so the parameter does not exist and
    D13's "provider-default temperature" is satisfied by construction.
    """
    memory = {"opus_mem": "rolling_note", "opus_nomem": "none"}[arm_id]
    return {
        "id": arm_id,
        "kind": "llm",
        "memory": memory,
        "provider": "anthropic",
        "model": MODEL,
        "system_prompt_sha256": system_prompt_sha256(),
    }


ARMS: tuple[dict, ...] = (agent_block("opus_mem"), agent_block("opus_nomem"))


def check_budget(usage: Usage, *, model: str, budget_usd: float) -> None:
    if usage.usd(model) > budget_usd:
        raise BudgetExceeded(
            f"lane spent ${usage.usd(model):.2f}, ceiling ${budget_usd:.2f}"
        )


def already_done(run_dir: Path, agent_id: str, track: str, window_id: str,
                 windows_dir: Path) -> bool:
    """An episode counts as done only if its log both validates and replays.

    A half-written log from a crash is not a shorter episode -- it is an unreadable one,
    and skipping it would silently put a hole in the lane.
    """
    log = Path(run_dir) / "episodes" / f"{agent_id}__{track}__{window_id}.jsonl"
    if not log.exists():
        return False
    try:
        return validate_episode(log).ok and replay(log, windows_dir).ok
    except Exception:  # noqa: BLE001 - an unreadable log is not a finished episode
        return False


def run_episode_for(
    client,
    *,
    run_dir: Path,
    run_id: str,
    agent: dict,
    track: str,
    window_id: str,
    episode_index: int,
    note: str | None,
    windows_dir: Path,
    cfg: Config,
    usage: Usage,
) -> str | None:
    """One episode plus, on the memory arm, its reflection call. Returns the new note."""
    session = EpisodeSession.start(
        window_id=window_id, track=track, agent=agent, episode_index=episode_index,
        run_dir=run_dir, run_id=run_id, cfg=cfg, windows_dir=windows_dir,
    )
    episode_usage = run_episode(session, client, note=note)

    new_note = None
    if agent["memory"] == "rolling_note":
        # Counted against this episode: the note costs real money, and a run that did
        # not bill it would understate itself.
        new_note = write_note(client, note, session.summary or {}, usage=episode_usage)

    session.finish(
        memory_note_in=note,
        memory_note_out=new_note,
        cost=episode_usage.as_cost_block(MODEL),
    )
    _accumulate(usage, episode_usage)
    return new_note


def _accumulate(total: Usage, one: Usage) -> None:
    """Fold one episode's usage into the lane total, after the episode's own cost block
    has been sealed from `one` alone."""
    total.tokens_in += one.tokens_in
    total.tokens_out += one.tokens_out
    total.cache_read_tokens += one.cache_read_tokens
    total.cache_write_tokens += one.cache_write_tokens


def run_lane(
    run_dir: Path,
    client,
    *,
    arm: dict,
    track: str,
    windows_dir: Path = WINDOWS_DIR,
    run_id: str = "run",
    budget_usd: float = 1e9,
    cfg: Config = DEFAULT,
    usage: Usage | None = None,
    resume: bool = True,
) -> Usage:
    """One (arm, track) lane: 30 episodes in presentation order, memory carried forward."""
    usage = usage if usage is not None else Usage()
    lane = Lane()
    order = load_manifest(windows_dir)["presentation_order"]

    for i, window_id in enumerate(order):
        if resume and already_done(run_dir, arm["id"], track, window_id, windows_dir):
            # The note this episode would have written is gone with the crash, so the
            # lane resumes memory-less from here. Recorded rather than papered over:
            # `memory_note_in` in the logs is the ground truth for what each episode saw.
            continue
        check_budget(usage, model=MODEL, budget_usd=budget_usd)
        new_note = run_episode_for(
            client, run_dir=run_dir, run_id=run_id, agent=arm, track=track,
            window_id=window_id, episode_index=i, note=lane.note,
            windows_dir=windows_dir, cfg=cfg, usage=usage,
        )
        if new_note:
            lane.carry(new_note)
    return usage
```

- [ ] **Step 4: Implement the script**

Create `scripts/run_agents.py`:

```python
#!/usr/bin/env python
"""The LLM arm of a run: 2 arms x 2 tracks x 30 windows = 120 episodes, plus D9's probe.

    python scripts/run_agents.py --dry-run              # ONE episode + the cost gate
    python scripts/run_agents.py --lanes --out runs/dev
    python scripts/run_agents.py --probe --out runs/dev

--dry-run is not optional ceremony. Prompt caching is what makes the budget arithmetic
true, and a prefix that never caches produces no error -- only a bill. The gate asserts
the cache actually engaged and extrapolates the measured cost to 120 episodes before you
commit to the batch.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anthropic  # noqa: E402

from pdtbench.agent.cost import PRICES, Usage  # noqa: E402
from pdtbench.agent.lanes import ARMS, TRACKS, run_lane  # noqa: E402
from pdtbench.agent.loop import MODEL  # noqa: E402
from pdtbench.agent.probe import run_probe  # noqa: E402
from pdtbench.config import DEFAULT, PROCESSED_DIR, RUNS_DIR, WINDOWS_DIR  # noqa: E402
from pdtbench.mcp.run import new_run_id, write_run_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lanes", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=200.0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    run_id = new_run_id()
    run_dir = args.out or (RUNS_DIR / run_id)
    # prices= is spec R4: `usd` must be recomputable from the run, not from whatever
    # agent/cost.py happens to say the day someone re-runs the analysis.
    write_run_manifest(run_dir, list(ARMS), DEFAULT, WINDOWS_DIR, run_id=run_id,
                       prices=PRICES)

    if args.dry_run:
        return _dry_run(run_dir, client, run_id)

    total = Usage()
    if args.lanes:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(run_lane, run_dir, client, arm=arm, track=track,
                            run_id=run_id, budget_usd=args.budget_usd / 4,
                            resume=not args.no_resume)
                for arm in ARMS for track in TRACKS
            ]
            for f in futures:
                u = f.result()
                for field in vars(u):
                    setattr(total, field, getattr(total, field) + getattr(u, field))

    if args.probe:
        run_probe(run_dir, client, windows_dir=WINDOWS_DIR,
                  processed_dir=PROCESSED_DIR, run_id=run_id, usage=total)

    print(f"\n{run_dir}\n  ${total.usd(MODEL):.2f}  {total}")
    return 0


def _dry_run(run_dir: Path, client, run_id: str) -> int:
    from pdtbench.agent.lanes import agent_block, run_episode_for

    usage = Usage()
    run_episode_for(
        client, run_dir=run_dir, run_id=run_id, agent=agent_block("opus_mem"),
        track="real", window_id="w00", episode_index=0, note=None,
        windows_dir=WINDOWS_DIR, cfg=DEFAULT, usage=usage,
    )
    usd = usage.usd(MODEL)
    print(f"  tokens_in (uncached) : {usage.tokens_in:>9,}")
    print(f"  cache_read           : {usage.cache_read_tokens:>9,}")
    print(f"  cache_write          : {usage.cache_write_tokens:>9,}")
    print(f"  tokens_out           : {usage.tokens_out:>9,}")
    print(f"  episode              : ${usd:.3f}")
    print(f"  x120 episodes        : ${usd * 120:.2f}")

    if usage.cache_read_tokens == 0:
        print("\n  FAIL: nothing was read from cache. The prefix is not caching -- "
              "silently, with no error. At full price this run costs ~10x. Do not "
              "start the batch.")
        return 1
    print("\n  cache engaged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests**

```bash
~/.venvs/pdt/bin/python -m pytest tests/test_agent.py -q
~/.venvs/pdt/bin/python -m pytest -q
```
Expected: both PASS.

- [ ] **Step 6: Verify the script is at least importable and its help works**

```bash
~/.venvs/pdt/bin/python scripts/run_agents.py --help
```
Expected: the argparse help text, no traceback.

- [ ] **Step 7: Commit**

```bash
git add src/pdtbench/agent/lanes.py scripts/run_agents.py tests/test_agent.py
git commit -m "agent: four lanes, resume, and the budget guard"
```

---

## Task 11: The dry run — the gate before the batch

**Files:** none created. This task runs the real API and reports.

**Interfaces:**
- Consumes: everything above.
- Produces: a go/no-go decision on the 120-episode batch, and the measured number that replaces the spec's extrapolation.

**Why:** the plan's Verification section requires "one full episode end-to-end, with a token and cost check, before committing to the overnight batch". The `effort: "low"` choice is an extrapolation until this runs.

- [ ] **Step 1: Confirm credentials resolve**

```bash
~/.venvs/pdt/bin/python -c "import anthropic; anthropic.Anthropic(); print('client ok')"
```
Expected: `client ok`. If it raises on a missing key, **do not hardcode one** — check `ant auth status` and use an `ANTHROPIC_API_KEY` env var or an `ant auth login` profile. Ask the user rather than guessing.

- [ ] **Step 2: Run one episode against the real API**

```bash
~/.venvs/pdt/bin/python scripts/run_agents.py --dry-run --out runs/dry_run
```
Expected: the token/cost table, then either `cache engaged.` (exit 0) or the `FAIL` banner (exit 1).

- [ ] **Step 3: Act on the cache assertion**

If it reports `FAIL: nothing was read from cache`, **stop and diagnose before spending anything**. The likely causes, in order:
1. The prompt never crosses 4,096 tokens — the rolling breakpoint in `loop._with_cache_breakpoint` is not landing on the last content block.
2. A silent invalidator entered the prefix — check `prompt.SYSTEM_PROMPT` for anything varying, and confirm `surface.tools()` order is stable across turns.
3. The episode ended in under ~10 turns, so the prefix never grew past the minimum. That is expected for a trivial run; re-check with an episode that actually trades.

- [ ] **Step 4: Validate and replay the dry-run episode**

```bash
~/.venvs/pdt/bin/python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from pdtbench.schema import validate_episode
from pdtbench.engine.replay import replay
from pdtbench.config import WINDOWS_DIR
log = next(Path('runs/dry_run/episodes').glob('*.jsonl'))
print('schema:', validate_episode(log).ok)
print('replay:', replay(log, WINDOWS_DIR).ok)
print('cost  :', __import__('json').loads(log.read_text().splitlines()[-1])['cost'])"
```
Expected: `schema: True`, `replay: True`, and a cost block carrying all four buckets and a `usd`.

- [ ] **Step 5: Report and decide**

Report to the user, before starting the batch:
- measured `$/episode` and the ×120 extrapolation;
- the split between output (thinking) and input, since output is the term that decides whether `effort: "low"` survives;
- whether the extrapolation clears the agreed ceiling.

If the extrapolation is over budget, **the effort/thinking decision reopens** — that is what the gate is for. Do not start a 120-episode batch on a failed gate.

- [ ] **Step 6: Commit nothing; report**

There is nothing to commit — `runs/` is gitignored. Hand the numbers back and get a go/no-go.

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| A1 — one model, two arms | 10 (`ARMS`, `agent_block`) |
| A2 — budget, thinking config | 7 (`THINKING`/`EFFORT`), 11 (the gate) |
| A3 — no temperature | 7 (test asserts absent), 10 (`agent_block`) |
| R1 — in-process MCP surface | 6 |
| R2 — the turn, `disable_parallel_tool_use` | 6, 7 |
| R3(a) — `attributing()` | 4 |
| R3(b) — `force_wait`, `prose_stall` | 3 |
| R4 — schema v1.3.0 cost block | 1, 2 |
| R5 — memory | 5 (`system_blocks`), 8 |
| R6 — caching, budget guard | 7 (`_with_cache_breakpoint`), 10 (`check_budget`), 11 |
| R7 — the probe | 9 |
| R8 — failures, resume | 7 (refusal/max_tokens), 10 (`already_done`, `BudgetExceeded`) |
| Verification — fake client, dry run | 2–10, 11 |

**Known gap, deliberately left:** the spec's "Open risks" notes that `analysis/learning.py` has never seen an `llm`/`memory: none` arm. That is step 6's problem and is out of scope here; the logs this plan produces are schema-valid and replayable, which is the contract step 6 consumes. Flag it to the user after Task 11 rather than widening this plan.

**Placeholder scan:** none. Every code step carries real code; every command carries expected output.

**Type consistency:** `Usage` field names (`tokens_in`, `tokens_out`, `cache_read_tokens`, `cache_write_tokens`) are identical in Task 2's dataclass, Task 1's schema block, and Task 10's accumulation loop. `run_episode(session, client, *, note=...)` in Task 7 matches the call in Task 10's `run_episode_for`. `ProbeFn`'s `(bars, options, context) -> dict` in Task 9 matches `leakage.ask()`'s call site. `agent_block()` output keys match the schema's `agent` object from Task 1's file.
