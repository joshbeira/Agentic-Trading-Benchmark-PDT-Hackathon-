"""The agent runner (build order step 5) -- a client of the MCP door.

Sits exactly where `mcp/baselines.py` sits: the baselines drive `EpisodeSession.call()`
in-process, and so does this. The difference is only who decides what to call.

`loop` and `probe` are deliberately not imported here: they are the only modules that
need the Anthropic SDK, and `cost` is useful without it.
"""

from __future__ import annotations

from .cost import PRICES, Usage

__all__ = ["PRICES", "Usage"]
