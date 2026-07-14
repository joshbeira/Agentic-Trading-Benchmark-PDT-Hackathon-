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
