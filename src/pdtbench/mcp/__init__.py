"""The MCP layer: one door into the engine, for models and baselines alike.

`server` is deliberately **not** imported here. It is the only module that needs the MCP
SDK, and a baseline run — 240 episodes, no model, no transport — should not have to load
a protocol stack to buy a share. Import it explicitly when you want to serve:

    from pdtbench.mcp.server import build_server
"""

from __future__ import annotations

from .baselines import BASELINES, Baseline, drive
from .run import PROBE, new_run_id, run_baseline_episode, run_baselines, write_run_manifest
from .session import ADVANCING, TOOLS, EpisodeSession

__all__ = [
    "ADVANCING",
    "BASELINES",
    "PROBE",
    "TOOLS",
    "Baseline",
    "EpisodeSession",
    "drive",
    "new_run_id",
    "run_baseline_episode",
    "run_baselines",
    "write_run_manifest",
]
