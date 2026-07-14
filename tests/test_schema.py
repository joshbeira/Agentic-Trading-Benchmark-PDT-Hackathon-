"""The contract, enforced.

`schemas/tick_log.md` is prose and prose rots. These tests are what stop it: every producer
in the repo has its output validated against `pdtbench.schema`, and the schema rejects
undeclared fields, so a producer that starts writing something new must document it in the
same commit.

This was not hypothetical. The engine emitted `obs.bar` as
`{open, high, low, close, volume}` while the schema and the fixtures both said
`{o, h, l, c, v}` — for four commits — and nothing caught it, because the analytics layer
never reads `obs.bar`. The log viewer would have been the first consumer to touch it, and
it would have crashed on every real episode.
"""

from __future__ import annotations

import json

import pytest

from pdtbench.data.windows import load_episode as load_window
from pdtbench.engine import TradingEnv
from pdtbench.schema import SCHEMA_VERSION, validate_episode

from . import fixtures as F
from . import policies as P

POLICIES = {
    "flat": P.Flat,
    "buy_and_hold": P.BuyAndHold,
    "sma_10_50": P.SmaCrossover,
    "churn": P.Churn,
    "dust": P.Dust,
    "random_5pct": lambda: P.RandomTrader(seed=5),
    "adversary": P.Adversary,  # every error code, plus the forced-wait path
}


# ============================================================ the engine conforms


@pytest.mark.parametrize("name", list(POLICIES))
@pytest.mark.parametrize("track", ["real", "twin"])
def test_engine_logs_conform(make_env, tmp_path, name, track):
    log = tmp_path / f"{name}.jsonl"
    env = P.run(make_env("w00", track, log_path=log), POLICIES[name]())
    env.close_log()

    rep = validate_episode(log)
    assert rep.ok, str(rep)


def test_a_multi_wait_episode_conforms(make_env, tmp_path):
    """Wait(n) produces skipped ticks — `decision_point: false`, `skipped_by`, empty calls.
    They are the shape most likely to be got wrong by a consumer, so they get their own
    test."""
    log = tmp_path / "flat.jsonl"
    env = P.run(make_env("w00", log_path=log), P.Flat())  # Wait(10)
    env.close_log()

    assert validate_episode(log).ok

    recs = [json.loads(x) for x in log.read_text().splitlines()]
    ticks = recs[1:-1]
    skipped = [t for t in ticks if "skipped_by" in t]
    assert len(skipped) > 50
    for t in skipped:
        assert t["decision_point"] is False
        assert t["calls"] == []
        assert t["action"] is None
        assert t["fill"] is None
        assert t["skipped_by"] < t["t"]


# =========================================================== the fixtures conform


@pytest.mark.parametrize("kind,memory", [("llm", "rolling_note"), ("baseline", "none")])
def test_fixture_logs_conform(tmp_path, kind, memory):
    log = F.make_episode(
        tmp_path / "f.jsonl",
        agent=F.agent_spec("a", kind=kind, memory=memory), track="real", window_id="w00",
        regime="bull", source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
        equity_cents=F.equity_from_returns(F.returns_for_sharpe(1.0, 0.02, seed=1)),
        invalid_calls_per_tick=2, reads_per_tick=1, forced_waits=3,
    )
    rep = validate_episode(log)
    assert rep.ok, str(rep)


def test_a_whole_fixture_run_conforms(tmp_path):
    from pdtbench.schema import validate_run

    F.make_run(
        tmp_path / "r",
        agents=[F.agent_spec("m1"), F.agent_spec("bh", kind="baseline", memory="none")],
        windows=F.window_specs(4),
        sharpe_fn=lambda a, t, w, i: w["difficulty"],
        exposure_fn=lambda a: 0.0 if a["id"] == "bh" else 1.0,
    )
    reports = validate_run(tmp_path / "r")
    assert len(reports) == 2 * 2 * 4
    bad = [str(r) for r in reports if not r.ok]
    assert not bad, bad[:2]


# ================================ the two producers agree with each other, not just the spec


def test_the_engine_and_the_fixtures_emit_the_same_shape(make_env, tmp_path):
    """Both conforming is necessary but not sufficient: a field optional in the schema could
    be written by one and not the other, and a consumer would still break. Compare the
    actual key sets."""
    elog = tmp_path / "e.jsonl"
    P.run(make_env("w00", log_path=elog), P.BuyAndHold()).close_log()

    flog = F.make_episode(
        tmp_path / "f.jsonl",
        agent=F.agent_spec("a", kind="baseline", memory="none"), track="real",
        window_id="w00", regime="bull", source_ticker="AAPL", bh_daily_vol=0.02,
        episode_index=0,
        equity_cents=F.equity_from_returns(F.returns_for_sharpe(1.0, 0.02, seed=1)),
    )

    def shape(path):
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        tick = next(t for t in recs[1:-1] if t.get("fill"))
        return {
            "meta": set(recs[0]),
            "tick": set(tick),
            "obs": set(tick["obs"]),
            "bar": set(tick["obs"]["bar"]),
            "stats": set(tick["obs"]["stats"]),
            "portfolio": set(tick["obs"]["portfolio"]),
            "fill": set(tick["fill"]),
            "end": set(recs[-1]),
            "metrics": set(recs[-1]["metrics"]),
        }

    e, f = shape(elog), shape(flog)
    for part in e:
        assert e[part] == f[part], (
            f"{part}: engine has {sorted(e[part] - f[part])}, "
            f"fixture has {sorted(f[part] - e[part])}"
        )

    # the specific drift that started this: long bar keys, not {o,h,l,c,v}
    assert e["bar"] == {"open", "high", "low", "close", "volume"}


# ==================================================== the validator has teeth


def _valid_log(tmp_path):
    return F.make_episode(
        tmp_path / "v.jsonl",
        agent=F.agent_spec("a"), track="real", window_id="w00", regime="bull",
        source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
        equity_cents=F.equity_from_returns(F.returns_for_sharpe(1.0, 0.02, seed=2)),
    )


def _mutate(path, fn):
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    fn(recs)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return path


@pytest.mark.parametrize(
    "name,mutate,expect",
    [
        ("undeclared field",
         lambda r: r[1].update({"surprise": 1}), "undeclared"),
        ("cents as float",
         lambda r: r[1].update({"equity_cents": 1000000.5}), "expected int"),
        ("bad agent kind",
         lambda r: r[0]["agent"].update({"kind": "scripted"}), "not in"),
        ("bad error code",
         lambda r: r[0].__setitem__("track", "pretend"), "not in"),
        ("missing episode_index",
         lambda r: r[0].pop("episode_index"), "required field is missing"),
        ("backdated fill",
         lambda r: _first_fill(r).__setitem__("fill_tick", 0), "fill_tick must be"),
        ("equity mirror broken",
         lambda r: r[5]["obs"]["portfolio"].__setitem__("equity_cents", 1), "disagrees"),
        ("terminal accepts an action",
         lambda r: r[-2].__setitem__("decision_point", True), "terminal tick accepts no"),
    ],
)
def test_the_validator_rejects(tmp_path, name, mutate, expect):
    """A validator that cannot fail is not a validator."""
    log = _mutate(_valid_log(tmp_path), mutate)
    rep = validate_episode(log)
    assert not rep.ok, f"{name}: the validator accepted a broken log"
    assert any(expect in err for err in rep.errors), f"{name}: got {rep.errors[:3]}"


def _first_fill(recs):
    return next(r["fill"] for r in recs if r.get("type") == "tick" and r.get("fill"))


def test_action_must_be_the_call_that_moved_the_clock(tmp_path):
    """The log must not credit the agent with an action it did not take."""
    log = _mutate(_valid_log(tmp_path),
                  lambda r: r[3]["action"].update({"tool": "Sell"}))
    rep = validate_episode(log)
    assert not rep.ok
    assert any("advancing call" in e for e in rep.errors)


def test_a_forced_wait_has_no_advancing_call(tmp_path):
    """A forced Wait is the engine taking the turn away. Giving it an advancing call would
    record an agent action that never happened."""
    log = F.make_episode(
        tmp_path / "forced.jsonl",
        agent=F.agent_spec("a"), track="real", window_id="w00", regime="bull",
        source_ticker="AAPL", bh_daily_vol=0.02, episode_index=0,
        equity_cents=F.equity_from_returns(F.returns_for_sharpe(0.5, 0.02, seed=3)),
        invalid_calls_per_tick=3, forced_waits=2,
    )
    assert validate_episode(log).ok

    recs = [json.loads(x) for x in log.read_text().splitlines()]
    forced = [t for t in recs[1:-1] if t.get("action") and t["action"].get("forced")]
    assert len(forced) == 2
    for t in forced:
        assert not [c for c in t["calls"] if c["advanced_time"]]
        assert t["action"]["tool"] == "Wait"

    broken = _mutate(
        log,
        lambda r: next(t for t in r[1:-1] if t.get("action") and t["action"]["forced"])["calls"]
        .append({"seq": 9, "tool": "Wait", "args": {"n": 1}, "ok": True, "advanced_time": True}),
    )
    assert not validate_episode(broken).ok


def test_the_schema_version_is_stamped(make_env, tmp_path):
    log = tmp_path / "v.jsonl"
    P.run(make_env("w00", log_path=log), P.Flat()).close_log()
    meta = json.loads(log.read_text().splitlines()[0])
    assert meta["schema_version"] == SCHEMA_VERSION == "1.3.0"


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


# =================================== the engine refuses to write an unreproducible log


def test_the_engine_refuses_a_log_it_cannot_pin(windows_dir, cfg, tmp_path, manifest):
    """A log missing its dataset hash, its episode index, or its agent kind is worse than
    useless — it is *plausible*. The engine will not write one."""
    series, spec = load_window(windows_dir, "w00", "real")
    good = {
        "episode_index": 0,
        "track": "real",
        "agent": {"id": "a", "kind": "baseline", "memory": "none"},
        "dataset": {"dataset_sha256": manifest["dataset_sha256"]},
    }

    for drop, expect in [
        ("episode_index", "episode_index is required"),
        ("dataset", "dataset_sha256 is required"),
    ]:
        meta = {k: v for k, v in good.items() if k != drop}
        with pytest.raises(ValueError, match=expect):
            TradingEnv(series, spec, cfg, log_path=tmp_path / "x.jsonl", meta_extra=meta).reset()

    bad_kind = good | {"agent": {"id": "a", "kind": "scripted", "memory": "none"}}
    with pytest.raises(ValueError, match="agent.kind"):
        TradingEnv(series, spec, cfg, log_path=tmp_path / "y.jsonl", meta_extra=bad_kind).reset()
