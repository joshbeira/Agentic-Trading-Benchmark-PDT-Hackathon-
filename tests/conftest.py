from __future__ import annotations

from pathlib import Path

import pytest

from pdtbench.config import DEFAULT, PROCESSED_DIR, WINDOWS_DIR, Config
from pdtbench.data.windows import load_episode, load_manifest
from pdtbench.engine import TradingEnv


@pytest.fixture(scope="session")
def cfg() -> Config:
    return DEFAULT


@pytest.fixture(scope="session")
def windows_dir() -> Path:
    if not (WINDOWS_DIR / "manifest.json").exists():
        pytest.skip("run scripts/build_dataset.py first")
    return WINDOWS_DIR


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return PROCESSED_DIR


@pytest.fixture(scope="session")
def manifest(windows_dir) -> dict:
    return load_manifest(windows_dir)


@pytest.fixture(scope="session")
def episodes(manifest) -> list[tuple[str, str]]:
    """Every (window_id, track) pair — 30 real + 30 twin."""
    return [(s["window_id"], track) for track in ("real", "twin") for s in manifest[track]]


@pytest.fixture
def make_env(windows_dir, cfg, manifest):
    def _make(window_id: str = "w00", track: str = "real", log_path: Path | None = None,
              **meta_extra) -> TradingEnv:
        series, spec = load_episode(windows_dir, window_id, track)
        # A scripted policy is a memory-less strategy, so it is a `baseline` — "scripted"
        # was never a value the schema allowed, and the engine now refuses to write it.
        defaults = {
            "episode_id": f"test__{track}__{window_id}",
            "episode_index": 0,
            "track": track,
            "agent": {"id": "test", "kind": "baseline", "memory": "none"},
            "dataset": {"dataset_sha256": manifest["dataset_sha256"]},
        }
        return TradingEnv(series, spec, cfg, log_path=log_path, meta_extra=defaults | meta_extra)

    return _make
