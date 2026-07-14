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

from mcp.server.fastmcp.exceptions import ToolError

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
        try:
            blocks = self._loop.run_until_complete(self._server.call_tool(name, args or {}))
        except ToolError:
            # FastMCP validates the name and the arg schema before the engine ever sees
            # them, so an unknown tool or a bad type dies here -- while the engine would
            # have called it UNKNOWN_TOOL / SCHEMA_ERROR, counted it, and advanced the
            # invalid ladder. Hand it to the engine, which is the only authority on what
            # an invalid call is. Keeps the served path logging byte-identically to the
            # direct path.
            return self.session.call(name, args or {})
        # blocks[0] is safe only because every tool in server.py returns a bare `-> dict`,
        # which makes FastMCP skip structured output and return a plain content-block
        # list. Tighten any tool's return annotation there (dict[str, Any], a TypedDict,
        # a pydantic model) and call_tool() starts returning an
        # (unstructured, structured) tuple instead -- blocks[0] would then be that whole
        # tuple, not a content block, and `.text` would raise AttributeError.
        return json.loads(blocks[0].text)

    def close(self) -> None:
        self._loop.close()
