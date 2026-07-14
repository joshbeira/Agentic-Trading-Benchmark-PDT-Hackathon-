"""Replay reproducibility.

The claim on the slide is "any number on the scoreboard can be regenerated from
the logs". These tests are that claim: write an episode, throw away the engine,
rebuild the whole scoreboard row from the JSONL alone, and demand agreement.
"""

from __future__ import annotations

import json

import pytest

from pdtbench.engine.replay import load, replay
from pdtbench.schema import SCHEMA_VERSION

from . import policies as P

POLICIES = {
    "flat": P.Flat,
    "buy_and_hold": P.BuyAndHold,
    "churn": P.Churn,
    "dust": P.Dust,
    "sma_crossover": P.SmaCrossover,
    "random": lambda: P.RandomTrader(seed=3),
    "adversary": P.Adversary,
}


@pytest.mark.parametrize("name", list(POLICIES))
@pytest.mark.parametrize("track", ["real", "twin"])
def test_an_episode_replays_from_its_log_alone(make_env, tmp_path, cfg, windows_dir, name, track):
    log = tmp_path / f"{name}__{track}.jsonl"
    env = P.run(make_env("w13", track, log_path=log), POLICIES[name]())
    env.close_log()

    res = replay(log, windows_dir)
    assert res.ok, f"{name}/{track} failed replay: {res.failures}"
    assert all(res.checks.values())

    # and the recomputed metrics are the ones the engine reported live
    for k, v in env.summary.items():
        assert res.recomputed[k] == pytest.approx(v, rel=1e-9), k


def test_replay_checks_are_all_actually_exercised(make_env, tmp_path, cfg, windows_dir):
    """Guard against a vacuous pass — every named check must have run."""
    log = tmp_path / "sma.jsonl"
    env = P.run(make_env("w18", log_path=log), P.SmaCrossover())
    env.close_log()
    res = replay(log, windows_dir)

    assert set(res.checks) == {
        "series_hash_matches_log",
        "tick_count",
        "ticks_ascending",
        "equity_invariant",
        "no_lookahead_fills",
        "time_advances_once",
        "cash_conservation",
        "equity_series_matches_ticks",
        "metrics_regenerate",
        "fees_match_fills",
    }
    assert res.ok


def test_the_log_conforms_to_the_schema(make_env, tmp_path):
    log = tmp_path / "shape.jsonl"
    env = P.run(make_env("w21", log_path=log), P.SmaCrossover())
    end = env.close_log()

    meta, ticks, ended = load(log)

    assert meta["schema_version"] == SCHEMA_VERSION
    for key in ("agent", "track", "window", "config", "seeds", "code", "started_at"):
        assert key in meta, key
    assert meta["window"]["series_sha256"]
    assert meta["config"]["config_sha256"]

    assert len(ticks) == env.cfg.n_scored
    assert [t["t"] for t in ticks] == list(range(env.cfg.n_scored))
    for t in ticks:
        assert {"obs", "calls", "action", "fill", "equity_cents"} <= set(t)
        assert {"bar", "stats", "portfolio"} <= set(t["obs"])

    assert len(ended["equity_series_cents"]) == env.cfg.n_scored
    assert ended["status"] == "ok"
    assert ended["metrics"] and ended["reliability"]
    assert ended["terminal"]["final_equity_cents"] == ended["equity_series_cents"][-1]


def test_replay_catches_a_tampered_log(make_env, tmp_path, cfg, windows_dir):
    """A verifier that cannot fail is not a verifier. Doctor an equity mark and the
    replay must refuse the episode."""
    log = tmp_path / "tampered.jsonl"
    env = P.run(make_env("w00", log_path=log), P.BuyAndHold())
    env.close_log()

    assert replay(log, windows_dir).ok  # clean, first

    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("type") == "tick" and rec["t"] == 40:
            rec["equity_cents"] += 100_000  # a free $1,000
            lines[i] = json.dumps(rec)
            break
    log.write_text("\n".join(lines) + "\n")

    res = replay(log, windows_dir)
    assert not res.ok
    assert not res.checks["equity_invariant"]


def test_replay_catches_a_backdated_fill(make_env, tmp_path, cfg, windows_dir):
    """The look-ahead check has teeth: reprice a fill at today's open — the price the
    agent had already seen — and replay must reject it."""
    log = tmp_path / "backdated.jsonl"
    env = P.run(make_env("w00", log_path=log), P.BuyAndHold())
    env.close_log()

    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("type") == "tick" and rec.get("fill") and rec["fill"]["side"] == "buy":
            rec["fill"]["fill_tick"] = rec["t"]  # filled at a price it had seen
            lines[i] = json.dumps(rec)
            break
    log.write_text("\n".join(lines) + "\n")

    res = replay(log, windows_dir)
    assert not res.ok
    assert not res.checks["no_lookahead_fills"]


def test_every_window_replays(make_env, tmp_path, cfg, windows_dir, episodes):
    """All 60 (window, track) pairs, end to end, under a real strategy."""
    failed = []
    for window_id, track in episodes:
        log = tmp_path / f"{track}__{window_id}.jsonl"
        env = P.run(make_env(window_id, track, log_path=log), P.SmaCrossover())
        env.close_log()
        res = replay(log, windows_dir)
        if not res.ok:
            failed.append((window_id, track, res.failures))

    assert not failed, f"{len(failed)}/60 episodes failed replay: {failed[:3]}"
