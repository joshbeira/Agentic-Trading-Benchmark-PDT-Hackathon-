"""The leakage experiment's machinery.

Two tests carry this file, and they are the two that decide whether the headline result
means anything:

  * plant a leak, and the analysis must find it
  * plant nothing, and the analysis must decline to find one

An experiment that cannot come back negative is not an experiment. The stub probe makes
both directions constructible, because its per-window accuracy is prescribed rather than
sampled — so `p_real - p_twin` is exactly what the test asked for and the recovered slope
can be checked against the planted one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pdtbench.analysis import leakage
from pdtbench.analysis.loader import load_run

from . import fixtures as F

N_REPS = 5


class StubProbe:
    """A model whose recall we control.

    Deterministic: on rep `r` it answers correctly iff `r < round(accuracy * n_reps)`. So a
    prescribed accuracy of 0.6 comes back as exactly 0.6, and the arithmetic downstream can
    be checked rather than merely smoke-tested.
    """

    def __init__(self, accuracy: dict[tuple[str, str], float], n_reps: int = N_REPS):
        self.accuracy = accuracy
        self.n_reps = n_reps
        self.calls = 0

    def __call__(self, bars, options, context):
        self.calls += 1
        wid, rep = context["window_id"], context["rep"]
        track = context["track"]
        acc = self.accuracy.get((track, wid), 0.125)
        right = rep < round(acc * self.n_reps)
        truth = context["truth"]
        out = {}
        for axis in ("ticker", "period"):
            if right:
                out[axis] = truth[axis]
            else:
                wrong = [o for o in options[axis] if o != truth[axis]]
                out[axis] = wrong[rep % len(wrong)]
        return out


def run_probe(windows, accuracy, agent_id="flagship", n_reps=N_REPS):
    """Drive the harness over every (window, track) and score it."""
    probe = StubProbe(accuracy, n_reps)
    scores = []
    for w in windows:
        truth = {"ticker": w["source_ticker"], "period": w["period"]}
        options = leakage.build_options(
            w["window_id"], truth, F.TICKERS, F.YEARS, n_options=8
        )
        for track in ("real", "twin"):
            qs = leakage.build_questions(w["window_id"], track, truth, options, n_reps)
            responses = [
                leakage.ask(
                    q,
                    bars=[],
                    probe_fn=lambda b, o, c, _t=track, _tr=truth: probe(
                        b, o, {**c, "track": _t, "truth": _tr}
                    ),
                )
                for q in qs
            ]
            scores.append(leakage.score(agent_id, responses))
    return scores, probe


# ================================================================= the harness itself


def test_a_window_and_its_twin_get_the_identical_option_set():
    """If the distractors differed across tracks, `p_real - p_twin` would be measuring our
    distractors rather than the model's memory, and the whole experiment would be an
    artifact of the harness."""
    truth = {"ticker": "AAPL", "period": "2013"}
    a = leakage.build_options("w07", truth, F.TICKERS, F.YEARS)
    b = leakage.build_options("w07", truth, F.TICKERS, F.YEARS)
    assert a == b  # seeded off window_id alone: nothing about the track enters

    for axis in ("ticker", "period"):
        assert len(a[axis]) == 8
        assert truth[axis] in a[axis]
        assert len(set(a[axis])) == 8

    other = leakage.build_options("w08", truth, F.TICKERS, F.YEARS)
    assert other != a  # ... but different windows get different distractors


def test_option_order_is_shuffled_per_rep():
    """Position bias would otherwise masquerade as recall: a model that always picks the
    third option would post a stable non-chance accuracy."""
    truth = {"ticker": "AAPL", "period": "2013"}
    options = leakage.build_options("w07", truth, F.TICKERS, F.YEARS)
    qs = leakage.build_questions("w07", "real", truth, options, n_reps=5)

    orders = {tuple(q.options["ticker"]) for q in qs}
    assert len(orders) > 1, "the option order never moved across reps"
    for q in qs:
        assert sorted(q.options["ticker"]) == sorted(options["ticker"])  # same set


def test_the_twin_is_scored_against_its_source_s_truth():
    """A twin has no ticker of its own. Scoring it against the source's truth is what makes
    the twin arm a false-positive rate rather than a nonsense question."""
    w = F.window_specs(1)[0]
    scores, _ = run_probe([w], accuracy={("real", w["window_id"]): 1.0,
                                         ("twin", w["window_id"]): 1.0})
    real = next(s for s in scores if s.track == "real")
    twin = next(s for s in scores if s.track == "twin")
    assert real.truth == twin.truth == {"ticker": w["source_ticker"], "period": w["period"]}
    assert twin.p_correct_mean == 1.0  # it "identified" a series it has never seen


def test_an_unparseable_answer_scores_wrong_and_is_counted():
    """Dropping failures inflates accuracy. They score wrong and stay in the denominator."""
    truth = {"ticker": "AAPL", "period": "2013"}
    options = leakage.build_options("w00", truth, F.TICKERS, F.YEARS)
    q = leakage.build_questions("w00", "real", truth, options, n_reps=1)[0]

    for bad in ({"ticker": "NOTATICKER", "period": "1999"}, {}, "garbage"):
        r = leakage.ask(q, [], probe_fn=lambda b, o, c, _b=bad: _b)
        assert r.correct == {"ticker": False, "period": False}
        assert r.answer == {"ticker": None, "period": None}

    boom = leakage.ask(q, [], probe_fn=lambda b, o, c: (_ for _ in ()).throw(RuntimeError("429")))
    assert boom.correct == {"ticker": False, "period": False}
    assert "429" in boom.raw


def test_prescribed_accuracy_comes_back_exactly():
    w = F.window_specs(1)[0]
    scores, _ = run_probe([w], accuracy={("real", w["window_id"]): 0.6,
                                         ("twin", w["window_id"]): 0.2})
    real = next(s for s in scores if s.track == "real")
    twin = next(s for s in scores if s.track == "twin")
    assert real.p_correct_mean == pytest.approx(0.6)
    assert twin.p_correct_mean == pytest.approx(0.2)
    assert real.p_correct["ticker"] == pytest.approx(0.6)


def test_the_probe_file_round_trips(tmp_path):
    w = F.window_specs(1)[0]
    truth = {"ticker": w["source_ticker"], "period": w["period"]}
    options = leakage.build_options(w["window_id"], truth, F.TICKERS, F.YEARS)
    qs = leakage.build_questions(w["window_id"], "real", truth, options, N_REPS)
    probe = StubProbe({("real", w["window_id"]): 0.6})
    responses = [
        leakage.ask(q, [], lambda b, o, c: probe(b, o, {**c, "track": "real", "truth": truth}))
        for q in qs
    ]
    s = leakage.score("flagship", responses)

    path = tmp_path / "probe" / f"flagship__real__{w['window_id']}.json"
    leakage.write_probe_file(path, "flagship", "run1", responses, s, n_options=8)

    d = json.loads(path.read_text())
    assert d["chance_level"] == 0.125
    assert d["truth"] == truth
    assert len(d["reps"]) == N_REPS
    assert d["score"]["p_correct_mean"] == pytest.approx(0.6)

    [reloaded] = leakage.load_probe_dir(path.parent)
    assert reloaded.p_correct_mean == pytest.approx(s.p_correct_mean)
    assert reloaded.window_id == s.window_id


# ============================================== does the analysis find a leak, or not?


def _run_with_edges(tmp_path, edge_fn):
    """A run where the agent's real-vs-twin Sharpe edge is exactly what we prescribe."""
    agents = [F.agent_spec("flagship"), F.agent_spec("flat", kind="baseline", memory="none")]
    windows = F.window_specs()

    def sharpe_fn(agent, track, w, i):
        base = w["difficulty"]
        if agent["kind"] == "baseline":
            return base
        return base + (edge_fn(w, i) if track == "real" else 0.0)

    return load_run(F.make_run(tmp_path, agents=agents, windows=windows,
                               sharpe_fn=sharpe_fn)), windows


def test_a_planted_leak_is_detected(tmp_path):
    """The agent recognizes some windows and *trades better on exactly those*. The analysis
    has to say so, and has to recover the slope it was given."""
    windows = F.window_specs()
    rng = np.random.default_rng(7)

    # a third of the windows are strongly recognizable, the rest are at chance
    ident = {w["window_id"]: (0.8 if i % 3 == 0 else 0.2) for i, w in enumerate(windows)}
    accuracy = {}
    for w in windows:
        accuracy[("real", w["window_id"])] = ident[w["window_id"]]
        accuracy[("twin", w["window_id"])] = 0.2  # the false-positive floor: texture alone

    BETA = 1.5  # Sharpe points of edge per unit of calibrated identifiability

    def edge_fn(w, i):
        return BETA * (ident[w["window_id"]] - 0.2)

    run, _ = _run_with_edges(tmp_path / "leaky", edge_fn)
    probes, _ = run_probe(windows, accuracy)
    res = leakage.analyze(run, "flagship", probes)

    assert len(res.points) == 30
    assert res.accuracy_real.point > res.accuracy_twin.point
    assert res.real_above_chance_p < 0.001
    assert res.calibrated_identifiability.lo > 0
    assert res.recognizes

    # the headline: edge concentrates in the windows it can name, at the planted slope
    assert res.slope.slope == pytest.approx(BETA, abs=0.25)
    assert res.slope.significant
    assert res.exploits
    assert "MEMORIZATION EXPLOITED" in res.verdict
    assert "contaminated" in res.verdict


def test_a_clean_run_is_not_convicted(tmp_path):
    """The negative control, and the more important of the two. The agent has a real edge
    on the real track -- but it is spread evenly, unrelated to which windows it can name.
    That is generalization, and calling it memorization would invalidate the benchmark for
    no reason."""
    windows = F.window_specs()
    rng = np.random.default_rng(11)

    ident = {w["window_id"]: (0.8 if i % 3 == 0 else 0.2) for i, w in enumerate(windows)}
    accuracy = {}
    for w in windows:
        accuracy[("real", w["window_id"])] = ident[w["window_id"]]
        accuracy[("twin", w["window_id"])] = 0.2

    # a flat +0.4 edge everywhere, with noise -- NOT correlated with identifiability
    noise = {w["window_id"]: float(rng.normal(0, 0.15)) for w in windows}
    run, _ = _run_with_edges(
        tmp_path / "clean", lambda w, i: 0.4 + noise[w["window_id"]]
    )
    probes, _ = run_probe(windows, accuracy)
    res = leakage.analyze(run, "flagship", probes)

    assert res.recognizes  # it CAN name the series ...
    assert not res.exploits  # ... and it trades no better for it
    assert not res.slope.significant
    assert res.slope.slope == pytest.approx(0.0, abs=0.5)
    assert "Recognition without exploitation" in res.verdict
    assert res.edge.point == pytest.approx(0.4, abs=0.25)  # the edge is real, just not leaked


def test_a_model_that_cannot_identify_anything_is_cleared(tmp_path):
    """Chance-level identification on both tracks. Nothing to convict."""
    windows = F.window_specs()
    accuracy = {(t, w["window_id"]): 0.2 for w in windows for t in ("real", "twin")}

    run, _ = _run_with_edges(tmp_path / "blind", lambda w, i: 0.0)
    probes, _ = run_probe(windows, accuracy)
    res = leakage.analyze(run, "flagship", probes)

    assert res.calibrated_identifiability.point == pytest.approx(0.0, abs=0.05)
    assert not res.recognizes
    assert "No memorization detected" in res.verdict


def test_the_twin_arm_is_what_prevents_a_false_conviction(tmp_path):
    """The reason the twin arm exists, made into a test.

    A model that reads volatility texture -- "this is a crash, so it is 2008" -- identifies
    the REAL window at 80%. Uncalibrated, that reads as blatant memorization. But it
    identifies the TWIN at 80% too, and the twin is a series it has provably never seen. The
    subtraction takes the accusation back to zero, which is exactly right.
    """
    windows = F.window_specs()
    accuracy = {}
    for w in windows:
        accuracy[("real", w["window_id"])] = 0.8
        accuracy[("twin", w["window_id"])] = 0.8  # texture, not recall

    run, _ = _run_with_edges(tmp_path / "texture", lambda w, i: 0.3)
    probes, _ = run_probe(windows, accuracy)
    res = leakage.analyze(run, "flagship", probes)

    assert res.accuracy_real.point == pytest.approx(0.8, abs=0.02)
    assert res.real_above_chance_p < 1e-6  # naively damning
    assert res.calibrated_identifiability.point == pytest.approx(0.0, abs=0.02)
    assert not res.recognizes  # ... and correctly acquitted
    assert "No memorization detected" in res.verdict

    # Identifiability is the same on every window, so there is no subset for an edge to
    # concentrate in and the regression has nothing to regress against. That is reported
    # as undefined rather than as a NaN masquerading as a number.
    assert not res.slope_defined
    assert "undefined" in leakage.render(res)
    assert "nan" not in leakage.render(res).lower()


def test_mann_whitney_is_computed_and_demoted(tmp_path):
    """The design names it, so it is reported -- as a descriptive statistic, with the paired
    Wilcoxon carrying the actual claim (D9)."""
    windows = F.window_specs()
    accuracy = {(t, w["window_id"]): 0.2 for w in windows for t in ("real", "twin")}
    run, _ = _run_with_edges(tmp_path / "mw", lambda w, i: 0.5)
    probes, _ = run_probe(windows, accuracy)
    res = leakage.analyze(run, "flagship", probes)

    assert res.mann_whitney.name == "mann_whitney"
    assert res.edge_test.name == "wilcoxon"
    # paired sees the planted edge; unpaired is swamped by window difficulty
    assert res.edge_test.pvalue < res.mann_whitney.pvalue
    assert "descriptive only" in leakage.render(res)
