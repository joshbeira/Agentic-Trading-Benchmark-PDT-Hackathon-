"""Analytics: the scoreboards, the learning curves, and the leakage experiment.

Reads tick logs and nothing else (with one narrow, documented exception: the probe's
answer key comes from the window manifest — see schemas/analysis_artifacts.md Q4). Every
module below `loader` is a pure function of loaded episodes, which is what lets the whole
layer be tested against hand-written fixture logs with no engine involved.
"""

from . import leakage, learning, reliability, scoreboard, stats
from .loader import Episode, Run, load_episode, load_run

__all__ = [
    "Episode",
    "Run",
    "load_episode",
    "load_run",
    "scoreboard",
    "reliability",
    "learning",
    "leakage",
    "stats",
]
