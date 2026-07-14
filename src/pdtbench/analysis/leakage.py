"""The leakage experiment (D9) — the headline science.

The benchmark's weakest assumption is that anonymizing a real price series prevents a
model from recognizing it. Rather than assert that, we measure it, and the measurement has
two halves that only mean something together.

**The probe.** Show a model a masked series and ask it, multiple choice, which ticker and
which year it is. Eight options each, so chance is 12.5%.

**The twin arm, which is the part that matters.** Run the identical probe on the window's
bootstrap twin — a series the model has provably never seen, matched to the real one on
return, volatility, and regime, formatted by the same code. Score the twin's answer against
the *source's* truth. Whatever accuracy that yields is the false-positive rate: how often
the model can name a series purely from its texture. 2008 *looks* like 2008. A crash has a
volatility signature, and the twin preserves it exactly.

So "the model said AAPL and it was AAPL" proves nothing on its own. Calibrated
identifiability is the difference:

    identifiability(w) = p_correct(real, w) - p_correct(twin, w)

**And then the question nobody usually asks: does recognition pay?** Recognizing a series
is not the same as exploiting it. The headline figure is per-window Sharpe edge against
per-window identifiability:

    edge(w) = sharpe(agent, real, w) - sharpe(agent, twin, w)

A positive slope means the agent's advantage on real data is concentrated in exactly the
windows it can name — memorization, converted into P&L. A flat line means it trades real
and synthetic alike, and the benchmark is measuring skill. Either result is publishable;
only the second one lets the rest of the benchmark stand.

The Mann-Whitney U test the original design led with is computed and demoted. The twin
design makes the comparison *paired*, and an unpaired test throws that away.

**Two limits worth stating before someone else does.**

*The identifiability axis is quantized.* Per-window accuracy is a fraction of `n_reps`
trials, so with 5 reps it can only take the values 0, 0.2, 0.4, 0.6, 0.8, 1.0 — and the
two-axis mean lands on a 0.1 grid. That is measurement error in the regressor, and
measurement error in `x` biases an OLS slope **toward zero** (regression dilution). So the
headline slope is a *conservative* estimate: it understates the true edge-vs-recognition
relationship rather than inflating it. Raising `n_reps` is the lever if a tighter estimate
is wanted; the cost is linear in probe calls.

*Identifiability may not vary.* A model that recognizes every window equally — or none of
them — produces a vertical line of points and no slope. That is reported as undefined
rather than as a NaN dressed up as a number, and it is not a failure: it means there is no
subset of windows for an edge to concentrate in, which the calibrated identifiability has
already answered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from . import stats as S
from .loader import Run

DEFAULT_N_OPTIONS = 8
DEFAULT_N_REPS = 5
AXES = ("ticker", "period")

#: Sharpe of edge per unit of calibrated identifiability. Identifiability spans [0, 1], so
#: this is the least edge attributable to recognition that would be worth the accusation.
#: An accusation of memorization is a serious claim about a model; it should not be
#: triggerable by a slope that clears a p-value on near-zero residuals.
MIN_LEAK_SLOPE = 0.10


class ProbeFn(Protocol):
    """The model adapter. Takes the bars and the options, returns a choice per axis.

    Deliberately pluggable: the harness is testable with a stub whose accuracy we control,
    which is the only way to know the analysis can detect a leak that is really there and
    can decline to find one that is not.
    """

    def __call__(self, bars: list[dict], options: dict[str, list[str]],
                 context: dict) -> dict: ...


@dataclass
class Question:
    window_id: str
    track: str
    rep: int
    options: dict[str, list[str]]  # already shuffled for this rep
    truth: dict[str, str]
    shuffle_seed: int


@dataclass
class Response:
    question: Question
    answer: dict[str, str | None]
    correct: dict[str, bool]
    raw: str | None = None
    tokens: dict = field(default_factory=dict)


@dataclass
class ProbeScore:
    agent_id: str
    window_id: str
    track: str
    n_reps: int
    p_correct: dict[str, float]  # per axis
    p_correct_mean: float
    truth: dict[str, str]


def _seed(*parts) -> int:
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def build_options(
    window_id: str,
    truth: dict[str, str],
    ticker_pool: Sequence[str],
    period_pool: Sequence[str],
    n_options: int = DEFAULT_N_OPTIONS,
) -> dict[str, list[str]]:
    """The distractor set for a window.

    Seeded off `window_id` **alone**, never off the track. A window's real series and its
    twin must be offered the identical option set — if the distractors differed across
    tracks, `p_real - p_twin` would be measuring our distractors instead of the model's
    memory, and the entire experiment would be an artifact.
    """
    rng = np.random.default_rng(_seed(window_id, "options"))
    out = {}
    for axis, pool in (("ticker", ticker_pool), ("period", period_pool)):
        true = truth[axis]
        others = sorted(set(pool) - {true})
        if len(others) < n_options - 1:
            raise ValueError(f"{axis} pool too small for {n_options}-way MC")
        picked = rng.choice(others, size=n_options - 1, replace=False).tolist()
        out[axis] = sorted([true, *picked])
    return out


def build_questions(
    window_id: str,
    track: str,
    truth: dict[str, str],
    options: dict[str, list[str]],
    n_reps: int = DEFAULT_N_REPS,
) -> list[Question]:
    """One question per rep, with the option *order* shuffled each time.

    The set is fixed; only the order moves. Without this, a model with a position bias
    ("always pick the third option") would post a stable non-chance accuracy and we would
    read it as recall.
    """
    qs = []
    for rep in range(n_reps):
        seed = _seed(window_id, track, rep)
        rng = np.random.default_rng(seed)
        shuffled = {
            axis: [opts[i] for i in rng.permutation(len(opts))]
            for axis, opts in options.items()
        }
        qs.append(Question(window_id, track, rep, shuffled, dict(truth), seed))
    return qs


def ask(q: Question, bars: list[dict], probe_fn: ProbeFn) -> Response:
    """Run one question. An unparseable or out-of-set answer scores wrong and is counted —
    never silently dropped, because dropping failures inflates accuracy."""
    try:
        raw = probe_fn(bars, q.options, {"window_id": q.window_id, "rep": q.rep})
    except Exception as exc:  # noqa: BLE001 - a provider failure is a wrong answer
        return Response(q, {a: None for a in AXES}, {a: False for a in AXES}, raw=f"error: {exc}")

    answer, correct = {}, {}
    for axis in AXES:
        pick = raw.get(axis) if isinstance(raw, dict) else None
        if pick not in q.options[axis]:
            pick = None  # out of set == wrong, and recorded as such
        answer[axis] = pick
        correct[axis] = pick == q.truth[axis]
    return Response(
        q, answer, correct,
        raw=raw.get("raw") if isinstance(raw, dict) else None,
        tokens=raw.get("tokens", {}) if isinstance(raw, dict) else {},
    )


def score(agent_id: str, responses: list[Response]) -> ProbeScore:
    q = responses[0].question
    p = {
        axis: float(np.mean([r.correct[axis] for r in responses]))
        for axis in AXES
    }
    return ProbeScore(
        agent_id=agent_id,
        window_id=q.window_id,
        track=q.track,
        n_reps=len(responses),
        p_correct=p,
        p_correct_mean=float(np.mean(list(p.values()))),
        truth=q.truth,
    )


def write_probe_file(path: Path, agent_id: str, run_id: str, responses: list[Response],
                     s: ProbeScore, n_options: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q0 = responses[0].question
    payload = {
        "schema_version": "1.1.0",
        "run_id": run_id,
        "agent_id": agent_id,
        "track": q0.track,
        "window_id": q0.window_id,
        "truth": q0.truth,
        "options": {axis: sorted(q0.options[axis]) for axis in AXES},
        "n_options": n_options,
        "chance_level": 1 / n_options,
        "reps": [
            {
                "rep": r.question.rep,
                "shuffle_seed": r.question.shuffle_seed,
                "answer": r.answer,
                "correct": r.correct,
                "raw_response": r.raw,
                "tokens": r.tokens,
            }
            for r in responses
        ],
        "score": {
            "p_correct_ticker": s.p_correct["ticker"],
            "p_correct_period": s.p_correct["period"],
            "p_correct_mean": s.p_correct_mean,
            "n_reps": s.n_reps,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def load_probe_dir(probe_dir: Path) -> list[ProbeScore]:
    out = []
    for p in sorted(Path(probe_dir).glob("*.json")):
        d = json.loads(p.read_text())
        sc = d["score"]
        out.append(
            ProbeScore(
                agent_id=d["agent_id"],
                window_id=d["window_id"],
                track=d["track"],
                n_reps=sc["n_reps"],
                p_correct={"ticker": sc["p_correct_ticker"], "period": sc["p_correct_period"]},
                p_correct_mean=sc["p_correct_mean"],
                truth=d["truth"],
            )
        )
    return out


# ============================================================ the headline analysis


@dataclass
class WindowPoint:
    window_id: str
    regime: str
    sharpe_real: float
    sharpe_twin: float
    edge: float
    p_real: float
    p_twin: float
    identifiability: float  # p_real - p_twin, the calibrated figure


@dataclass
class LeakageResult:
    agent_id: str
    n_options: int
    chance: float
    points: list[WindowPoint]

    accuracy_real: S.Interval
    accuracy_twin: S.Interval
    real_above_chance_p: float
    twin_above_chance_p: float
    calibrated_identifiability: S.Interval

    edge: S.Interval
    edge_test: S.TestResult  # paired Wilcoxon — the right test
    mann_whitney: S.TestResult  # unpaired — descriptive only, demoted (D9)
    slope: S.Slope  # edge ~ identifiability. THE headline.

    @property
    def recognizes(self) -> bool:
        """Above chance on real, *beyond* what it manages on the twin."""
        return self.real_above_chance_p < 0.05 and self.calibrated_identifiability.lo > 0

    @property
    def exploits(self) -> bool:
        """Clears zero *and* is large enough to be worth the accusation (MIN_LEAK_SLOPE)."""
        return self.slope.matters(MIN_LEAK_SLOPE) and self.slope.slope > 0

    @property
    def slope_defined(self) -> bool:
        """The regression needs identifiability to *vary* across windows. A model that
        recognizes every window equally well (or equally badly) gives a vertical line of
        points and no slope — which is not a failure, it is the answer: there is no
        subset of windows where its edge could be concentrating."""
        return bool(np.isfinite(self.slope.slope))

    @property
    def verdict(self) -> str:
        if not self.recognizes:
            return (
                "No memorization detected: identification on real windows is not "
                "distinguishable from the twin false-positive rate."
            )
        if not self.slope_defined:
            return (
                "Recognition is uniform across windows, so there is no subset for an edge "
                "to concentrate in and the edge-vs-identifiability slope is undefined. "
                f"The Sharpe edge itself is {self.edge}."
            )
        if self.exploits:
            return (
                "MEMORIZATION EXPLOITED: the agent identifies real windows above the twin "
                f"baseline, and its edge concentrates in the windows it can name "
                f"(slope {self.slope.slope:+.3f}, p={self.slope.pvalue:.3f}). Results on "
                "the masked-real track are contaminated."
            )
        return (
            "Recognition without exploitation: the agent identifies real windows above the "
            "twin baseline, but its Sharpe edge does not concentrate in the windows it "
            "names. It knows where it is and trades no better for it."
        )


def analyze(run: Run, agent_id: str, probes: list[ProbeScore], seed: int = 0) -> LeakageResult:
    by_key = {(p.track, p.window_id): p for p in probes if p.agent_id == agent_id}
    pairs = run.paired(agent_id)

    points = []
    for real, twin in pairs:
        pr = by_key.get(("real", real.window_id))
        pt = by_key.get(("twin", real.window_id))
        if pr is None or pt is None:
            continue
        points.append(
            WindowPoint(
                window_id=real.window_id,
                regime=real.regime,
                sharpe_real=real.metrics["sharpe_floored"],
                sharpe_twin=twin.metrics["sharpe_floored"],
                edge=real.metrics["sharpe_floored"] - twin.metrics["sharpe_floored"],
                p_real=pr.p_correct_mean,
                p_twin=pt.p_correct_mean,
                identifiability=pr.p_correct_mean - pt.p_correct_mean,
            )
        )

    n_reps = probes[0].n_reps if probes else 1
    n_opts = DEFAULT_N_OPTIONS
    chance = 1 / n_opts

    p_real = [p.p_real for p in points]
    p_twin = [p.p_twin for p in points]
    edges = [p.edge for p in points]
    ident = [p.identifiability for p in points]

    # Accuracy counts: 2 axes x n_reps trials per window.
    trials = len(points) * 2 * n_reps
    k_real = int(round(sum(p_real) * 2 * n_reps))
    k_twin = int(round(sum(p_twin) * 2 * n_reps))

    return LeakageResult(
        agent_id=agent_id,
        n_options=n_opts,
        chance=chance,
        points=points,
        accuracy_real=S.wilson_ci(k_real, trials),
        accuracy_twin=S.wilson_ci(k_twin, trials),
        real_above_chance_p=S.binomial_p(k_real, trials, chance),
        twin_above_chance_p=S.binomial_p(k_twin, trials, chance),
        calibrated_identifiability=S.bootstrap_ci(ident, stat=np.mean, seed=seed),
        edge=S.bootstrap_ci(edges, seed=seed),
        edge_test=S.wilcoxon(
            [p.sharpe_real for p in points], [p.sharpe_twin for p in points]
        ),
        mann_whitney=S.mann_whitney(
            [p.sharpe_real for p in points], [p.sharpe_twin for p in points]
        ),
        slope=S.ols_slope(ident, edges),
    )


def render(res: LeakageResult) -> str:
    out = [f"LEAKAGE PROBE — {res.agent_id}", "=" * (16 + len(res.agent_id)), ""]
    out.append(f"  {res.n_options}-way multiple choice on ticker and year. "
               f"Chance = {res.chance:.1%}.")
    out.append("")
    out.append(f"  identification, real windows  {res.accuracy_real.point:>6.1%} "
               f"[{res.accuracy_real.lo:.1%}, {res.accuracy_real.hi:.1%}]   "
               f"vs chance p={res.real_above_chance_p:.4f}")
    out.append(f"  identification, twins         {res.accuracy_twin.point:>6.1%} "
               f"[{res.accuracy_twin.lo:.1%}, {res.accuracy_twin.hi:.1%}]   "
               f"vs chance p={res.twin_above_chance_p:.4f}")
    out.append(f"  calibrated identifiability    {res.calibrated_identifiability}"
               "   (real - twin; the twin arm is the false-positive rate)")
    out.append("")
    out.append(f"  Sharpe edge (real - twin)     {res.edge}")
    out.append(f"  paired Wilcoxon               {res.edge_test}")
    out.append(f"  Mann-Whitney U                {res.mann_whitney}   [descriptive only: "
               "the twin design is paired, and this test discards that]")
    out.append("")
    if res.slope_defined:
        out.append(f"  HEADLINE  edge ~ identifiability slope = {res.slope}")
    else:
        out.append("  HEADLINE  edge ~ identifiability slope = undefined "
                   "(identifiability does not vary across windows)")
    out.append("")
    out.append(f"  {res.verdict}")
    return "\n".join(out)
