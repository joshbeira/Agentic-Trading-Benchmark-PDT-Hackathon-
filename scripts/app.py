"""The demo surface (D14): a run directory, read and shown.

    .venv/bin/python -m streamlit run scripts/app.py

**This is the one thing in the repo that does not run on `~/.venvs/pdt`.** Streamlit's
dependency chain pins `pyarrow<25` and the benchmark venv is on 25.0.0, so installing it
there would downgrade a declared dependency of the data pipeline underneath the test
suite. The demo is not worth that risk, so it runs from the git-ignored in-repo `.venv`
(streamlit + pandas + numpy + scipy) instead. Everything else -- tests, the engine, the
runner, the batch -- still runs on `~/.venvs/pdt/bin/python`. This module imports no
engine and no `anthropic`, so the split costs nothing: `pdtbench.analysis` needs only
scipy beyond the plotting stack.

This renders what is already in `runs/{run_id}/` and computes nothing the analysis
package does not. Every number on screen comes from `pdtbench.analysis`, which recomputes
from the JSONL rather than trusting the cached blocks -- so a figure here and a figure in
`scripts/analyze.py` cannot disagree. Where a statistic has an interval, the interval is
shown: ten episodes per regime cell is a small sample and the UI should say so rather
than round it away.

The headline tab leads with D9's figure -- per-window Sharpe edge against per-window
identifiability -- because that is the claim the benchmark exists to make. Edge
concentrated in the windows the model can name is exploited memorization; edge flat
across identifiability is generalization.

Nothing here calls the API. It is safe to drive live in front of an audience, and it is
the replay fallback if the network is not.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pdtbench.analysis import leakage, learning, reliability, scoreboard  # noqa: E402
from pdtbench.analysis.loader import Run, load_run  # noqa: E402

RUNS_DIR = REPO / "runs"
TRACKS = ("real", "twin")

st.set_page_config(page_title="pdtbench", layout="wide")


# --------------------------------------------------------------------------- loading


@st.cache_data(show_spinner="Loading run…")
def _load(run_dir: str, stamp: float) -> Run:
    """`stamp` is the directory mtime: it busts the cache when a run is rewritten.

    Both args are part of the cache key and must stay un-underscored -- `cache_data`
    excludes leading-underscore parameters from the hash, so naming this `_stamp` would
    silently pin the first load forever and serve a stale scoreboard off a rewritten run.

    Keyed on the path string rather than the Path so Streamlit can hash it.
    """
    return load_run(Path(run_dir))


def _stamp_of(run_dir: Path) -> float:
    """Newest mtime under the run — a rerun mid-demo should not serve a stale scoreboard."""
    return max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()), default=0.0)


@st.cache_data(show_spinner="Scoring the probe…")
def _leakage(run_dir: str, stamp: float, agent_id: str, seed: int):
    run = _load(run_dir, stamp)
    probe_dir = Path(run_dir) / "probe"
    if not probe_dir.is_dir():
        return None
    probes = leakage.load_probe_dir(probe_dir)
    if not probes:
        return None
    return leakage.analyze(run, agent_id, probes, seed=seed)


def _interval(iv) -> str:
    return str(iv)


# --------------------------------------------------------------------------- sidebar

available = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.is_dir() else []
if not available:
    st.error(f"No run directories under `{RUNS_DIR}`.")
    st.stop()

with st.sidebar:
    st.title("pdtbench")
    run_id = st.selectbox("Run", available, index=0)
    seed = st.number_input(
        "Bootstrap seed", value=0, step=1,
        help="Intervals are bootstrapped. The seed is exposed so a number quoted on "
             "stage can be reproduced exactly.",
    )

run_dir = RUNS_DIR / run_id
stamp = _stamp_of(run_dir)

try:
    run = _load(str(run_dir), stamp)
except Exception as exc:  # a malformed run should name itself, not blank the page
    st.error(f"Could not load `{run_id}`: {exc}")
    st.stop()

with st.sidebar:
    st.caption(
        f"{len(run.episodes)} episodes · {len(run.agents)} agents "
        f"({len(run.llms)} llm, {len(run.baselines)} baseline)"
    )
    unscorable = [e for e in run.episodes if not e.scorable]
    if unscorable:
        st.warning(
            f"{len(unscorable)} episode(s) excluded from the scoreboard "
            f"(status: {', '.join(sorted({e.status for e in unscorable}))})."
        )
    drifted = [e for e in run.episodes if e.metric_drift]
    if drifted:
        st.error(
            f"{len(drifted)} episode(s) whose cached metrics disagree with the replay. "
            "The log is the authority; the cache is stale."
        )

headline, board, curves, rel, inspect = st.tabs(
    ["Headline", "Scoreboard", "Learning", "Reliability", "Episode"]
)


# --------------------------------------------------------------------------- headline

with headline:
    st.header("Does the edge live in the windows the model recognises?")
    st.caption(
        "Each point is one window: how much better the agent did on the real series than "
        "on its twin (edge), against how well it could name the real series (calibrated "
        "identifiability). A positive slope is exploited memorization. A flat line is "
        "generalization."
    )

    if not run.llms:
        st.info("No LLM agents in this run — the probe compares a model against its twin, "
                "and baselines cannot be probed.")
    else:
        agent = st.selectbox("Agent", run.llms, key=f"headline_agent__{run_id}")
        res = _leakage(str(run_dir), stamp, agent, int(seed))
        if res is None:
            st.info(f"No probe files under `{run_dir / 'probe'}` — D9's figure needs them.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "identifiability": p.identifiability,
                        "edge": p.edge,
                        "regime": p.regime,
                        "window": p.window_id,
                    }
                    for p in res.points
                ]
            )

            # A regression against a constant x has no slope. That is not a bug to hide
            # behind a "+nan" tile -- it is the honest answer to a degenerate question,
            # and it should say which question was degenerate.
            defined = not math.isnan(res.slope.slope)

            a, b, c = st.columns(3)
            a.metric(
                "Edge ~ identifiability",
                f"{res.slope.slope:+.3f}" if defined else "—",
                help=str(res.slope),
            )
            b.metric("Sharpe edge (real − twin)", _interval(res.edge))
            c.metric("Identifiability", _interval(res.calibrated_identifiability))

            if not defined:
                spread = {round(p.identifiability, 6) for p in res.points}
                st.warning(
                    f"**No slope: identifiability does not vary.** All {len(res.points)} "
                    f"windows share the same value ({spread.pop():+.3f}), so there is "
                    "nothing to regress the edge against. A probe that names the real "
                    "series exactly as often as its twin does this — a stub probe, or a "
                    "model that recognises nothing. The scatter below is a vertical line."
                )

            st.scatter_chart(df, x="identifiability", y="edge", color="regime", height=420)

            st.markdown(
                f"**Paired test (the right one):** {res.edge_test}  \n"
                f"**Probe accuracy** — real {_interval(res.accuracy_real)} · "
                f"twin {_interval(res.accuracy_twin)} · chance {res.chance:.3f} "
                f"({res.n_options} options)"
            )
            with st.expander("The canonical rendering, as `scripts/analyze.py` prints it"):
                st.text(leakage.render(res))


# ------------------------------------------------------------------------- scoreboard

with board:
    track = st.radio("Track", TRACKS, horizontal=True, key="board_track")
    rows = scoreboard.build(run, track, seed=int(seed))
    if not rows:
        st.info(f"No scorable episodes on the `{track}` track.")
    else:
        st.subheader(f"Ranked by median floored Sharpe — `{track}`")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "agent": r.agent_id,
                        "kind": r.kind,
                        "n": r.n,
                        "sharpe (floored)": _interval(r.sharpe),
                        "sharpe (raw, median)": round(r.sharpe_raw_median, 3),
                        "return": round(r.total_return, 4),
                        "max dd": round(r.max_drawdown, 4),
                        "turnover": round(r.turnover, 2),
                        "fees (¢)": round(r.fees_cents),
                        "in market": round(r.time_in_market, 3),
                        "trades": round(r.n_trades, 1),
                        "floor binding": f"{r.pct_floor_binding:.0%}",
                    }
                    for r in rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The vol floor is what stops a lucky one-percent position posting a Sharpe of "
            "+30 — `floor binding` is how often it caught this agent."
        )

        st.subheader("By regime")
        st.caption("Ten episodes per cell. The intervals are wide because the sample is small.")
        by_regime = scoreboard.by_regime(run, track, seed=int(seed))
        for regime, rrows in by_regime.items():
            if not rrows:
                continue
            with st.expander(f"{regime} — {len(rrows)} agents"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"agent": r.agent_id, "kind": r.kind, "n": r.n,
                             "sharpe (floored)": _interval(r.sharpe)}
                            for r in rrows
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )


# --------------------------------------------------------------------------- learning

with curves:
    track = st.radio("Track", TRACKS, horizontal=True, key="curve_track")
    built = learning.build(run, track)
    if not built:
        st.info(f"No lanes on the `{track}` track.")
    else:
        st.caption(
            "Memory is scoped per (agent, track) and resets between tracks, so one lane is "
            "one continuous learning sequence. Excess Sharpe subtracts the baselines' "
            "trace: baselines cannot learn, so their per-episode score *is* the window's "
            "difficulty. Only the excess slope can claim learning."
        )
        for c in built:
            st.markdown(f"**{c.agent_id}** · memory: `{c.memory}` — {c.verdict}")
            st.line_chart(
                pd.DataFrame(
                    {"sharpe": c.sharpe, "baseline trace": c.baseline_trace, "excess": c.excess},
                    index=pd.Index(c.index, name="episode_index"),
                ),
                height=260,
            )
        with st.expander("The canonical rendering"):
            st.text(learning.render(built, track, run))


# ------------------------------------------------------------------------ reliability

with rel:
    st.caption(
        "Counted from the `calls` array, not the cached reliability block — `calls` is the "
        "primary record, and a disagreement between them is a bug worth seeing."
    )
    rows = reliability.build(run)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "agent": r.agent_id,
                    "kind": r.kind,
                    "episodes": r.n_episodes,
                    "calls": r.n_calls,
                    "invalid": r.n_invalid,
                    "invalid rate": f"{r.invalid_rate:.3%}",
                    "schema errors": f"{r.schema_error_rate:.3%}",
                    "forced waits/ep": round(r.forced_waits_per_episode, 3),
                    "prose nudges/ep": round(r.prose_nudges_per_episode, 3),
                    "read caps/ep": round(r.read_cap_hits_per_episode, 3),
                    "median wallclock (s)": round(r.median_wallclock_s, 1),
                    "capped": r.n_capped,
                    "errors": r.n_agent_errors,
                    "clean": "✓" if r.clean else "",
                }
                for r in rows
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    breakdowns = {r.agent_id: r.error_breakdown for r in rows if r.error_breakdown}
    if breakdowns:
        st.subheader("Error breakdown")
        st.json(breakdowns)
    with st.expander("The canonical rendering"):
        st.text(reliability.render(rows))


# --------------------------------------------------------------------------- episode

with inspect:
    col_a, col_t = st.columns(2)
    agent = col_a.selectbox("Agent", run.agents, key=f"ep_agent__{run_id}")
    track = col_t.radio("Track", TRACKS, horizontal=True, key=f"ep_track__{run_id}")

    lane = run.lane(agent, track)
    if not lane:
        st.info(f"`{agent}` has no episodes on the `{track}` track.")
    else:
        # The key carries the run, agent and track. Without it, switching runs holds the
        # previous run's Episode for one rerun -- `episode_id` collides across runs
        # (`buy_and_hold__real__w00` exists in both), so the widget restores a stale
        # object and renders its metrics under the new run's name. A demo that shows the
        # wrong run's Sharpe while the sidebar says otherwise is worse than one that errors.
        ep = st.selectbox(
            "Episode",
            lane,
            key=f"ep__{run_id}__{agent}__{track}",
            format_func=lambda e: f"[{e.episode_index:02d}] {e.window_id} · {e.regime} · {e.status}",
        )

        m = ep.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe (floored)", f"{m['sharpe_floored']:.3f}")
        c2.metric("Sharpe (raw)", f"{m['sharpe_raw']:.3f}")
        c3.metric("Return", f"{m['total_return']:+.2%}")
        c4.metric("Max drawdown", f"{m['max_drawdown']:.2%}")

        if ep.metric_drift:
            st.error("Cached metrics disagree with the replay — the log is the authority.")
            st.json(ep.metric_drift)

        st.line_chart(
            pd.DataFrame(
                {"equity ($)": [c / 100 for c in ep.equity_cents], "shares": ep.shares_by_tick},
                index=pd.Index(range(len(ep.equity_cents)), name="tick"),
            ),
            height=300,
        )

        st.subheader(f"Calls ({len(ep.calls)})")
        st.caption(
            "The reliability record: what the agent actually sent, including what the "
            "engine refused."
        )
        st.dataframe(pd.DataFrame(ep.calls), hide_index=True, width="stretch", height=280)

        if ep.fills:
            st.subheader(f"Fills ({len(ep.fills)})")
            st.dataframe(pd.DataFrame(ep.fills), hide_index=True, width="stretch")

        note_in, note_out = st.columns(2)
        with note_in:
            st.subheader("Note carried in")
            st.text(ep.memory_note_in or "— none (first episode of the lane, or no-memory arm)")
        with note_out:
            st.subheader("Note carried out")
            st.text(ep.memory_note_out or "— none")

        with st.expander("Config, tokens, raw reliability"):
            st.json({"config": ep.config, "tokens": ep.tokens, "reliability": ep.reliability})
