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

--- On the presentation -------------------------------------------------------------

The read is a tearsheet, not a dashboard: this is operated by people who read numbers for
a living, and the job is to make a result legible enough to be trusted or attacked. So
every figure is monospaced and tabular-aligned, tables are static rather than sortable
widgets (a tearsheet is printed, not queried), and colour is never decoration -- it means
regime, or it means a verdict. Charts are Altair, which ships inside streamlit: no CDN, so
the surface still renders with the venue wifi down.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pdtbench.analysis import leakage, learning, reliability, scoreboard  # noqa: E402
from pdtbench.analysis.loader import Run, load_run  # noqa: E402

RUNS_DIR = REPO / "runs"
TRACKS = ("real", "twin")

# --- tokens (mirrored in .streamlit/config.toml) ------------------------------------
PAPER = "#F5F6F8"
PANEL = "#FFFFFF"
INK = "#0E1419"
INK_2 = "#5A6672"
INK_3 = "#8A96A1"
RULE = "#DDE2E7"
ACCENT = "#17457A"

BULL = "#1F6F54"
BEAR = "#9E3B33"
CHOP = "#6B7684"

POS = "#106B4A"
NEG = "#9E2B25"
WARN = "#8A6212"

SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

st.set_page_config(page_title="pdtbench", layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- chrome

st.markdown(
    f"""
    <style>
      /* Streamlit's own chrome announces the framework, not the work. */
      [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
        display: none !important;
      }}

      [data-testid="stAppViewContainer"] {{ background: {PAPER}; }}
      [data-testid="stMainBlockContainer"] {{
        padding: 2.2rem 2.6rem 5rem;
        max-width: 1320px;
      }}
      /* Set the face on the root and let it inherit. A `*` selector here reaches every
         descendant including Streamlit's Material icon spans, whose glyphs are ligatures
         in an icon font -- overriding their family renders the icon's *name* as text
         ("keyboard_arrow_right" appearing next to an expander). Anything that needs a
         different face below asks for it by name. */
      html, body {{ font-family: {SANS}; }}
      [data-testid="stAppViewContainer"] p,
      [data-testid="stAppViewContainer"] label,
      [data-testid="stAppViewContainer"] span,
      [data-testid="stAppViewContainer"] div {{ font-family: inherit; }}
      [data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {{
        font-family: "Material Symbols Rounded", "Material Icons" !important;
      }}

      /* --- the tearsheet header band --- */
      .sheet-head {{
        border-bottom: 2px solid {INK};
        padding-bottom: 0.7rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1.5rem;
        flex-wrap: wrap;
      }}
      .sheet-title {{
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {INK};
        line-height: 1.1;
      }}
      .sheet-title small {{
        display: block;
        font-family: {MONO};
        font-size: 0.66rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {INK_3};
        margin-bottom: 0.3rem;
      }}
      .sheet-facts {{
        font-family: {MONO};
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: {INK_2};
        text-align: right;
        font-variant-numeric: tabular-nums;
        line-height: 1.7;
      }}
      .sheet-facts b {{ color: {INK}; font-weight: 600; }}

      /* --- the verdict: the app says the answer in words --- */
      .verdict {{
        border-left: 3px solid {ACCENT};
        padding: 0.15rem 0 0.15rem 1rem;
        margin: 0.2rem 0 1.4rem;
      }}
      .verdict .q {{
        font-family: {MONO};
        font-size: 0.66rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {INK_3};
        margin-bottom: 0.35rem;
      }}
      .verdict .a {{
        font-size: 1.16rem;
        line-height: 1.45;
        color: {INK};
        max-width: 62ch;
      }}
      .verdict .a b {{
        font-family: {MONO};
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        color: {ACCENT};
      }}
      .verdict .a em {{ color: {INK_2}; font-style: normal; }}

      /* --- section labels --- */
      .lbl {{
        font-family: {MONO};
        font-size: 0.66rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {INK_3};
        border-bottom: 1px solid {RULE};
        padding-bottom: 0.4rem;
        margin: 0.5rem 0 0.9rem;
      }}
      .note {{ font-size: 0.85rem; color: {INK_2}; line-height: 1.55; max-width: 78ch; }}
      .note code {{
        font-family: {MONO};
        font-size: 0.8em;
        background: {RULE};
        padding: 0.06em 0.3em;
        border-radius: 2px;
        color: {INK};
      }}

      /* --- tabs: a rule with an active mark, not a pill row --- */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 1.6rem;
        border-bottom: 1px solid {RULE};
      }}
      [data-testid="stTabs"] [data-baseweb="tab"] {{
        font-family: {MONO};
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 600;
        color: {INK_3};
        padding: 0.35rem 0 0.6rem;
      }}
      [data-testid="stTabs"] [aria-selected="true"] {{ color: {INK}; }}
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: {ACCENT}; height: 2px; }}
      [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

      /* --- figures: the point estimate leads, the interval stays subordinate --- */
      .stats {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px;
        background: {RULE};
        border: 1px solid {RULE};
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 1.1rem;
      }}
      .stat {{ background: {PANEL}; padding: 0.7rem 0.95rem 0.75rem; }}
      /* The label wraps rather than truncating: on a narrow window an ellipsis turns
         "sharpe edge (real − twin)" into "sharpe edge (real − tw…", which is a headline
         figure whose name the reader cannot finish. min-height reserves the second line
         so the values stay on one baseline across the row whether they wrap or not. */
      .stat-l {{
        font-family: {MONO};
        font-size: 0.6rem;
        line-height: 1.45;
        min-height: 2.9em;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {INK_3};
        text-wrap: balance;
      }}
      .stat-v {{
        font-family: {MONO};
        font-size: 1.42rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: {INK};
        line-height: 1.35;
        letter-spacing: -0.02em;
      }}
      .stat-s {{
        font-family: {MONO};
        font-size: 0.68rem;
        font-variant-numeric: tabular-nums;
        color: {INK_3};
        white-space: nowrap;
      }}
      .stat-v .pos {{ color: {POS}; }}
      .stat-v .neg {{ color: {NEG}; }}
      .stat-v .zero {{ color: {INK_3}; }}

      /* --- tables: printed, not queried ---
         These are hand-rolled HTML, not st.table. Streamlit's table is a canvas-ish
         widget whose internals shift between versions, and styling it through
         data-testid hooks silently hid a real column header (the diff between the
         header row and the body row was one cell, so every label sat over the wrong
         number). Owning the markup makes that class of bug impossible. */
      .tt-wrap {{ overflow-x: auto; margin-bottom: 0.5rem; }}
      table.tt {{
        border-collapse: collapse;
        width: 100%;
        font-size: 0.8rem;
        font-family: {MONO};
        font-variant-numeric: tabular-nums;
      }}
      /* The font and border are restated on the cells themselves, not left to inherit:
         the global reset above matches every descendant with `*`, and a direct match
         beats an inherited value however specific the ancestor rule is. */
      table.tt th {{
        font-family: {MONO};
        font-size: 0.62rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        font-weight: 600;
        color: {INK_3};
        border: none;
        border-bottom: 1px solid {INK};
        padding: 0 0.85rem 0.4rem 0;
        text-align: right;
        white-space: nowrap;
      }}
      table.tt th.l, table.tt td.l {{ text-align: left; }}
      table.tt td {{
        font-family: {MONO};
        font-variant-numeric: tabular-nums;
        border: none;
        border-bottom: 1px solid {RULE};
        padding: 0.42rem 0.85rem 0.42rem 0;
        color: {INK};
        text-align: right;
        white-space: nowrap;
      }}
      table.tt tbody tr:hover td {{ background: rgba(23, 69, 122, 0.05); }}
      table.tt td.name {{ font-weight: 600; letter-spacing: -0.01em; }}
      table.tt td.kind {{ color: {INK_3}; font-size: 0.72rem; }}
      table.tt .ci {{ color: {INK_3}; font-size: 0.72rem; margin-left: 0.35rem; }}
      table.tt .pos {{ color: {POS}; }}
      table.tt .neg {{ color: {NEG}; }}
      table.tt .zero {{ color: {INK_3}; }}
      table.tt tr.lead td {{ background: rgba(23, 69, 122, 0.05); }}

      /* --- sidebar: an instrument panel --- */
      [data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-family: {MONO};
        font-size: 0.63rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: {INK_3};
      }}
      .side-note {{
        font-family: {MONO};
        font-size: 0.68rem;
        line-height: 1.75;
        color: {INK_2};
        font-variant-numeric: tabular-nums;
        border-top: 1px solid {RULE};
        padding-top: 0.7rem;
        margin-top: 0.4rem;
      }}
      .side-note b {{ color: {INK}; font-weight: 600; }}

      /* --- alerts: quieter than the default --- */
      [data-testid="stAlert"] {{ border-radius: 3px; font-size: 0.85rem; }}

      [data-testid="stExpander"] summary p {{
        font-family: {MONO};
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {INK_2};
      }}
      [data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 3px; }}

      @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- loading


@st.cache_data(show_spinner="Reading the run…")
def _load(run_dir: str, stamp: float) -> Run:
    """`stamp` is the directory mtime: it busts the cache when a run is rewritten.

    Both args are part of the cache key and must stay un-underscored -- `cache_data`
    excludes leading-underscore parameters from the hash, so naming this `_stamp` would
    silently pin the first load forever and serve a stale scoreboard off a rewritten run.

    Keyed on the path string rather than the Path so Streamlit can hash it.
    """
    return load_run(Path(run_dir))


@st.cache_data(ttl=5, show_spinner=False)
def _stamp_of(run_dir: str) -> float:
    """Newest mtime under the run — a rerun mid-demo should not serve a stale scoreboard.

    Cached with a short TTL because this walks and stats every file in the run, and the
    repo lives on a OneDrive-synced mount where each stat is a network-ish call. A run has
    360 episode files; without the cache, every widget click paid 360+ stats before it
    could draw anything, which made the whole surface feel broken. Five seconds is well
    inside a demo's attention span and still notices a run rewritten under us.
    """
    return max((p.stat().st_mtime for p in Path(run_dir).rglob("*") if p.is_file()), default=0.0)


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


# ----------------------------------------------------------------------- presentation


def _axis(title: str | None, **kw) -> alt.Axis:
    return alt.Axis(
        title=title,
        titleFont=MONO,
        titleFontSize=10,
        titleColor=INK_3,
        titleFontWeight="normal",
        labelFont=MONO,
        labelFontSize=10,
        labelColor=INK_3,
        domainColor=RULE,
        tickColor=RULE,
        gridColor="#EBEEF1",
        **kw,
    )


def _chart(chart: alt.Chart, height: int) -> None:
    """Draw a styled figure.

    `theme=None` is load-bearing: streamlit's default chart theme is applied *over* the
    spec and would quietly overwrite the palette and fonts configured below with its own.
    """
    st.altair_chart(_style(chart, height), theme=None)


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    """One place for the chart furniture, so no two figures disagree about it."""
    return (
        chart.properties(width="container", height=height)
        .configure_view(strokeWidth=0, fill=PANEL)
        .configure_legend(
            labelFont=MONO,
            labelFontSize=10,
            labelColor=INK_2,
            titleFont=MONO,
            titleFontSize=9,
            titleColor=INK_3,
            symbolStrokeWidth=2,  # a line chart's legend key IS a stroke; zero hides it
            symbolSize=90,
            orient="top-right",
            direction="horizontal",
            title=None,
        )
    )


def _domain(values, min_span: float, center: float | None = None) -> list[float]:
    """A scale domain that refuses to magnify noise into signal.

    Vega fits the axis to the data, which is right until the data has no spread. On a
    degenerate run every window's edge is zero to within float error, and an auto-fitted
    axis blows ±8e-6 of rounding dust up to full height — a scatter that looks like
    structure and is nothing but the last bits of a double. `min_span` is the smallest
    range worth drawing; anything tighter is reported as flat, because it is.
    """
    lo, hi = (min(values), max(values)) if values else (0.0, 0.0)
    mid = center if center is not None else (lo + hi) / 2
    half = max((hi - lo) / 2, min_span / 2)
    pad = half * 0.15
    return [min(lo, mid - half) - pad, max(hi, mid + half) + pad]


def _stats(items: list[tuple[str, str, str]]) -> None:
    """A row of figures: label, value, and the qualifier that keeps the value honest.

    Replaces st.metric because the interval must be visibly subordinate to the point
    estimate rather than crammed into it at the same weight.
    """
    cells = "".join(
        f'<div class="stat"><div class="stat-l">{label}</div>'
        f'<div class="stat-v">{value}</div>'
        f'<div class="stat-s">{sub or "&nbsp;"}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


REGIME_HUE = {"bull": BULL, "bear": BEAR, "chop": CHOP}
REGIME_COLOR = alt.Scale(domain=["bull", "bear", "chop"], range=[BULL, BEAR, CHOP])
# Shape carries the regime too: bull/bear as green/red alone is unreadable to roughly one
# viewer in twelve, and this gets projected.
REGIME_SHAPE = alt.Scale(domain=["bull", "bear", "chop"], range=["circle", "triangle-up", "square"])


def _sheet_head(title: str, sub: str, facts: str) -> None:
    st.markdown(
        f'<div class="sheet-head"><div class="sheet-title"><small>{sub}</small>{title}</div>'
        f'<div class="sheet-facts">{facts}</div></div>',
        unsafe_allow_html=True,
    )


def _verdict(question: str, answer: str) -> None:
    st.markdown(
        f'<div class="verdict"><div class="q">{question}</div><div class="a">{answer}</div></div>',
        unsafe_allow_html=True,
    )


def _lbl(text: str) -> None:
    st.markdown(f'<div class="lbl">{text}</div>', unsafe_allow_html=True)


def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def _signed(value: float, text: str) -> str:
    """A figure coloured by its sign, judged on what the reader can actually see.

    The colour follows the *rendered* number, not the float behind it. An edge of 7e-8
    formats as `+0.00` and is zero to anyone reading the screen; painting it green
    because the last bits of a double happen to be positive is a lie told in colour.
    """
    shown_zero = not any(ch in "123456789" for ch in text)
    cls = "zero" if shown_zero else ("pos" if value > 0 else "neg")
    return f'<span class="{cls}">{text}</span>'


def _sharpe(iv) -> str:
    """The point estimate, with its interval kept subordinate to it.

    The interval is the honest part of the number and must be visible, but a table where
    every cell is `+1.83 [+1.40, +2.20]` is a table nobody reads. Point estimate leads at
    full weight; the interval trails in muted small type on the same line.
    """
    return (f'{_signed(iv.point, f"{iv.point:+.2f}")}'
            f'<span class="ci">[{iv.lo:+.2f}, {iv.hi:+.2f}]</span>')


def _table(cols: list[tuple[str, str]], rows: list[dict], lead: bool = False) -> None:
    """Render a table as HTML. `cols` is [(header, key)]; a key ending in `_l` is
    left-aligned. Cell values are pre-formatted strings and may carry markup.

    `lead` shades the first row -- used where the table is a ranking and the top of it is
    the answer to the question above it.
    """
    head = "".join(
        f'<th class="{"l" if k.endswith("_l") else ""}">{h}</th>' for h, k in cols
    )
    body = []
    for i, r in enumerate(rows):
        tds = "".join(
            f'<td class="{r.get(k + "__cls", "")} {"l" if k.endswith("_l") else ""}">'
            f'{r.get(k, "")}</td>'
            for _, k in cols
        )
        body.append(f'<tr class="{"lead" if lead and i == 0 else ""}">{tds}</tr>')
    st.markdown(
        f'<div class="tt-wrap"><table class="tt"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- sidebar

available = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.is_dir() else []
if not available:
    st.error(f"No run directories under `{RUNS_DIR}`. Nothing to read.")
    st.stop()

with st.sidebar:
    run_id = st.selectbox("Run", available, index=0)
    seed = st.number_input(
        "Bootstrap seed", value=0, step=1,
        help="Intervals are bootstrapped. The seed is exposed so a number quoted on "
             "stage can be reproduced exactly.",
    )

run_dir = RUNS_DIR / run_id
stamp = _stamp_of(str(run_dir))

try:
    run = _load(str(run_dir), stamp)
except Exception as exc:  # a malformed run should name itself, not blank the page
    st.error(f"Could not read `{run_id}`: {exc}")
    st.stop()

unscorable = [e for e in run.episodes if not e.scorable]
drifted = [e for e in run.episodes if e.metric_drift]

with st.sidebar:
    st.markdown(
        f'<div class="side-note">'
        f'episodes &nbsp;<b>{len(run.episodes)}</b><br>'
        f'agents &nbsp;&nbsp;&nbsp;<b>{len(run.llms)}</b> llm &nbsp;<b>{len(run.baselines)}</b> baseline<br>'
        f'windows &nbsp;<b>{len(set(e.window_id for e in run.episodes))}</b><br>'
        f'tracks &nbsp;&nbsp;<b>real</b> · <b>twin</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if unscorable:
        st.warning(
            f"{len(unscorable)} episode(s) held out of the scoreboard — status "
            f"{', '.join(sorted({e.status for e in unscorable}))}."
        )
    if drifted:
        st.error(
            f"{len(drifted)} episode(s) whose cached metrics disagree with the replay. "
            "The log is the authority; the cache is stale."
        )

_sheet_head(
    "Agentic Trading Benchmark",
    "pdtbench · run report",
    f'<b>{run_id}</b><br>{len(run.episodes)} episodes · {len(run.agents)} agents',
)

headline, board, curves, rel, inspect = st.tabs(
    ["Headline", "Scoreboard", "Learning", "Reliability", "Episode"]
)


# --------------------------------------------------------------------------- headline

with headline:
    if not run.llms:
        _verdict(
            "Does the edge live in the windows the model recognises?",
            "<em>No model in this run.</em> The probe asks a model to name a window and "
            "compares that against its twin; scripted baselines have nothing to recognise, "
            "so there is no question to ask here.",
        )
    else:
        agent = st.selectbox("Agent", run.llms, key=f"headline_agent__{run_id}")
        res = _leakage(str(run_dir), stamp, agent, int(seed))

        if res is None:
            _verdict(
                "Does the edge live in the windows the model recognises?",
                f"<em>Not yet answerable.</em> No probe files under "
                f"<code>{run_dir / 'probe'}</code> — D9's figure needs them.",
            )
        else:
            defined = not math.isnan(res.slope.slope)
            significant = bool(getattr(res.slope, "significant", False))

            if not defined:
                shared = {round(p.identifiability, 3) for p in res.points}.pop()
                _verdict(
                    "Does the edge live in the windows the model recognises?",
                    f"<em>No slope — the question is degenerate here.</em> All "
                    f"{len(res.points)} windows share one identifiability "
                    f"(<b>{shared:+.3f}</b>), so there is nothing to regress the edge "
                    f"against. A probe that names the real series exactly as often as its "
                    f"twin does this: a stub, or a model that recognises nothing.",
                )
            elif significant and res.slope.slope > 0:
                _verdict(
                    "Does the edge live in the windows the model recognises?",
                    f"<b>Yes.</b> Edge rises <b>{res.slope.slope:+.3f}</b> Sharpe per unit "
                    f"of identifiability — the profit concentrates in the windows "
                    f"<em>{agent}</em> can name. That is exploited memorization, not skill.",
                )
            elif significant:
                _verdict(
                    "Does the edge live in the windows the model recognises?",
                    f"Edge moves <b>{res.slope.slope:+.3f}</b> Sharpe per unit of "
                    f"identifiability — <em>against</em> recognition. Worth explaining "
                    f"before it is reported.",
                )
            else:
                _verdict(
                    "Does the edge live in the windows the model recognises?",
                    f"<b>No.</b> The slope is <b>{res.slope.slope:+.3f}</b> and does not "
                    f"clear zero — recognising a window buys <em>{agent}</em> nothing. "
                    f"The trading result stands on its own.",
                )

            _stats([
                ("Edge ~ identifiability",
                 f"{res.slope.slope:+.3f}" if defined else "—",
                 "the headline slope" if defined else "identifiability does not vary"),
                ("Sharpe edge (real − twin)",
                 _signed(res.edge.point, f"{res.edge.point:+.2f}"),
                 f"[{res.edge.lo:+.2f}, {res.edge.hi:+.2f}]"),
                ("Identifiability",
                 _signed(res.calibrated_identifiability.point,
                         f"{res.calibrated_identifiability.point:+.3f}"),
                 f"[{res.calibrated_identifiability.lo:+.3f}, "
                 f"{res.calibrated_identifiability.hi:+.3f}]"),
                ("Windows", f"{len(res.points)}", f"chance {res.chance:.3f}"),
            ])

            df = pd.DataFrame(
                [{"identifiability": p.identifiability, "edge": p.edge,
                  "regime": p.regime, "window": p.window_id} for p in res.points]
            )

            # Floors, not fits: MIN_EDGE is about a fifth of a Sharpe and MIN_IDENT a
            # tenth of the probe's range. Below those the axis stops zooming, so a run
            # with no signal draws as a run with no signal.
            xdom = _domain(list(df["identifiability"]), 0.30, center=0.0)
            ydom = _domain(list(df["edge"]), 0.40, center=0.0)

            base = alt.Chart(df)
            zero = (
                alt.Chart(pd.DataFrame({"y": [0.0]}))
                .mark_rule(color=INK_3, strokeDash=[3, 3], strokeWidth=1)
                .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=ydom, nice=False)))
            )
            pts = base.mark_point(filled=True, size=110, opacity=0.85, strokeWidth=0).encode(
                x=alt.X("identifiability:Q",
                        axis=_axis("calibrated identifiability   (p_real − p_twin)",
                                   format="+.2f", tickCount=7),
                        scale=alt.Scale(domain=xdom, nice=False)),
                y=alt.Y("edge:Q", axis=_axis("sharpe edge   (real − twin)", format="+.2f",
                                             tickCount=6),
                        scale=alt.Scale(domain=ydom, nice=False)),
                color=alt.Color("regime:N", scale=REGIME_COLOR),
                shape=alt.Shape("regime:N", scale=REGIME_SHAPE),
                tooltip=[alt.Tooltip("window:N", title="window"),
                         alt.Tooltip("regime:N", title="regime"),
                         alt.Tooltip("identifiability:Q", title="identifiability",
                                     format="+.3f"),
                         alt.Tooltip("edge:Q", title="edge", format="+.3f")],
            )
            layers = [zero, pts]
            if defined:
                layers.append(
                    base.transform_regression("identifiability", "edge")
                    .mark_line(color=ACCENT, strokeWidth=2, strokeDash=[6, 4])
                    .encode(x="identifiability:Q", y="edge:Q")
                )
            _chart(alt.layer(*layers), 360)

            _note(
                f"One mark per window. <b>Probe accuracy</b> — real {res.accuracy_real} · "
                f"twin {res.accuracy_twin} · chance {res.chance:.3f} over {res.n_options} "
                f"options. The twin arm is the false-positive rate: a model that "
                f"“recognises” a synthetic path is pattern-matching, not recalling, and "
                f"subtracting it is what makes identifiability calibrated.<br>"
                f"<b>Paired test</b> — {res.edge_test}. The Mann-Whitney figure in the "
                f"full rendering is deliberately demoted: it discards the pairing the twin "
                f"design exists to create."
            )
            with st.expander("Full rendering, as scripts/analyze.py prints it"):
                st.text(leakage.render(res))


# ------------------------------------------------------------------------- scoreboard

with board:
    track = st.radio("Track", TRACKS, horizontal=True, key="board_track")
    rows = scoreboard.build(run, track, seed=int(seed))

    if not rows:
        st.info(f"No scorable episodes on the {track} track.")
    else:
        top = rows[0]
        _verdict(
            f"Who traded the {track} track best?",
            f"<b>{top.agent_id}</b> leads at <b>{top.sharpe}</b> median floored Sharpe "
            f"over {top.n} episodes. Read the per-regime panels below before believing it: "
            f"<em>ten episodes a cell is a small sample, and the intervals say so.</em>",
        )

        _lbl(f"Pooled · {track} · ranked by median floored Sharpe")
        _table(
            [("agent", "agent_l"), ("", "kind_l"), ("n", "n"), ("sharpe (floored)", "sh_l"),
             ("raw", "raw"), ("return", "ret"), ("max dd", "dd"), ("turnover", "to"),
             ("fees", "fees"), ("in mkt", "mkt"), ("trades", "tr"), ("floor", "fl")],
            [{"agent_l": r.agent_id, "agent_l__cls": "name",
              "kind_l": r.kind, "kind_l__cls": "kind",
              "n": r.n,
              "sh_l": _sharpe(r.sharpe),
              "raw": _signed(r.sharpe_raw_median, f"{r.sharpe_raw_median:+.2f}"),
              "ret": _signed(r.total_return, f"{r.total_return:+.1%}"),
              "dd": f"{r.max_drawdown:.1%}",
              "to": f"{r.turnover:.2f}",
              "fees": f"${r.fees_cents / 100:,.0f}",
              "mkt": f"{r.time_in_market:.0%}",
              "tr": f"{r.n_trades:.1f}",
              "fl": f"{r.pct_floor_binding:.0%}"} for r in rows],
            lead=True,
        )
        _note(
            "The volatility floor is what stops a lucky one-percent position posting a "
            "Sharpe of +30 — <code>floor</code> is how often it caught this agent. "
            "<b>Do not quote a pooled “beat buy-and-hold”:</b> the 10/10/10 regime balance "
            "forces that median to roughly zero by construction. Compare within a regime."
        )

        _lbl("By regime · the primary trading result (D12)")
        by_regime = scoreboard.by_regime(run, track, seed=int(seed))
        live = [(k, v) for k, v in by_regime.items() if v]
        for col, (regime, rrows) in zip(st.columns(len(live) or 1), live):
            with col:
                hue = REGIME_HUE[regime]
                st.markdown(
                    f'<div class="lbl" style="border-color:{hue};color:{hue}">{regime}</div>',
                    unsafe_allow_html=True,
                )
                _table(
                    [("agent", "agent_l"), ("n", "n"), ("sharpe", "sh")],
                    [{"agent_l": r.agent_id, "agent_l__cls": "name", "n": r.n,
                      "sh": _signed(r.sharpe.point, f"{r.sharpe.point:+.2f}")}
                     for r in rrows],
                )


# --------------------------------------------------------------------------- learning

with curves:
    track = st.radio("Track", TRACKS, horizontal=True, key="curve_track")
    built = learning.build(run, track)

    if not built:
        st.info(f"No lanes on the {track} track.")
    else:
        learned = [c for c in built if c.learned]
        _verdict(
            "Does carrying a note between episodes teach it anything?",
            (f"<b>{learned[0].agent_id}</b> improves — excess Sharpe "
             f"<b>{learned[0].excess_slope.slope:+.3f}</b> per episode, "
             f"<em>after subtracting how hard each window was.</em>"
             if learned else
             "<b>No lane shows learning.</b> No excess slope clears zero by enough to "
             "claim. <em>Only the excess slope can say this</em> — raw Sharpe climbing "
             "might just mean the later windows were kinder.")
        )

        def _lane(c) -> None:
            _lbl(f"{c.agent_id} · {c.verdict}")
            # Short keys: a legend that truncates is a legend nobody reads. What each
            # series means is carried by the note under the charts, not by its own name.
            long = pd.DataFrame(
                [{"episode": i, "series": s, "sharpe": v}
                 for s, vals in (("excess", c.excess),
                                 ("agent", c.sharpe),
                                 ("difficulty", c.baseline_trace))
                 for i, v in zip(c.index, vals)]
            )
            order = ["excess", "agent", "difficulty"]
            zero = (
                alt.Chart(pd.DataFrame({"y": [0.0]}))
                .mark_rule(color=RULE, strokeWidth=1)
                .encode(y="y:Q")
            )
            line = (
                alt.Chart(long)
                .mark_line(strokeWidth=1.9, opacity=0.95)
                .encode(
                    x=alt.X("episode:Q",
                            axis=_axis("episode index within the lane", tickCount=8),
                            scale=alt.Scale(domain=[0, max(c.index)], nice=False)),
                    y=alt.Y("sharpe:Q", axis=_axis("floored sharpe", tickCount=5)),
                    color=alt.Color("series:N",
                                    scale=alt.Scale(domain=order, range=[ACCENT, INK, INK_3]),
                                    legend=alt.Legend(orient="top", direction="horizontal",
                                                      title=None, offset=4)),
                    strokeDash=alt.StrokeDash(
                        "series:N",
                        scale=alt.Scale(domain=order, range=[[1, 0], [1, 0], [4, 3]]),
                        legend=None),
                    size=alt.Size("series:N",
                                  scale=alt.Scale(domain=order, range=[2.6, 1.5, 1.2]),
                                  legend=None),
                    tooltip=[alt.Tooltip("episode:Q", title="episode"),
                             alt.Tooltip("series:N", title="series"),
                             alt.Tooltip("sharpe:Q", title="sharpe", format="+.3f")],
                )
            )
            _chart(alt.layer(zero, line), 200)

        llm = [c for c in built if run.agent_kind(c.agent_id) == "llm"]
        for c in llm:
            _lane(c)

        _note(
            "<b>agent</b> is what it scored. <b>difficulty</b> is what the baselines scored "
            "on the same window — they cannot learn, so their score <em>is</em> the "
            "window's difficulty. <b>excess</b> is the difference, and it is the only line "
            "that can claim learning: raw Sharpe drifting up might just mean the later "
            "windows were kinder. The verdict refuses a slope that clears zero but is too "
            "small to matter — a p-value on a rounding error is still a rounding error."
        )

        rest = [c for c in built if c not in llm]
        if rest:
            with st.expander(f"The {len(rest)} baseline lanes — the difficulty trace itself"):
                for c in rest:
                    _lane(c)
        with st.expander("Full rendering"):
            st.text(learning.render(built, track, run))


# ------------------------------------------------------------------------ reliability

with rel:
    rows = reliability.build(run)
    dirty = [r for r in rows if not r.clean]
    _verdict(
        "Did the agents behave?",
        ("<b>Every agent ran clean.</b> No invalid calls, no wall-clock caps, no errors."
         if not dirty else
         f"<b>{len(dirty)} of {len(rows)}</b> did not: "
         + ", ".join(f"<em>{r.agent_id}</em>" for r in dirty)
         + ". A malformed call costs no money — only time, and therefore decisions.")
    )

    _lbl("Counted from calls[] — the primary record, not the cached summary")
    _table(
        [("agent", "agent_l"), ("", "kind_l"), ("eps", "eps"), ("calls", "calls"),
         ("invalid", "inv"), ("invalid %", "invp"), ("schema %", "sch"),
         ("forced/ep", "forced"), ("nudges/ep", "nudge"), ("readcap/ep", "cap"),
         ("wallclock", "wall"), ("capped", "ncap"), ("errors", "err"), ("", "ok")],
        [{"agent_l": r.agent_id, "agent_l__cls": "name",
          "kind_l": r.kind, "kind_l__cls": "kind",
          "eps": r.n_episodes, "calls": f"{r.n_calls:,}",
          "inv": r.n_invalid or "—",
          "invp": f"{r.invalid_rate:.1%}" if r.n_invalid else "—",
          "sch": f"{r.schema_error_rate:.1%}" if r.schema_error_rate else "—",
          "forced": f"{r.forced_waits_per_episode:.2f}" if r.forced_waits_per_episode else "—",
          "nudge": f"{r.prose_nudges_per_episode:.2f}" if r.prose_nudges_per_episode else "—",
          "cap": f"{r.read_cap_hits_per_episode:.2f}" if r.read_cap_hits_per_episode else "—",
          "wall": f"{r.median_wallclock_s:.0f}s",
          "ncap": r.n_capped or "—",
          "err": r.n_agent_errors or "—",
          "ok": '<span class="pos">clean</span>' if r.clean else '<span class="neg">·</span>',
          } for r in rows],
    )
    _note(
        "<code>forced/ep</code> above zero means the anti-stall ladder fired — the agent "
        "stalled and the engine took its turn away. That is logged distinctly from a "
        "decision the agent actually made."
    )
    breakdowns = {r.agent_id: r.error_breakdown for r in rows if r.error_breakdown}
    if breakdowns:
        _lbl("Error breakdown")
        st.json(breakdowns)
    with st.expander("Full rendering"):
        st.text(reliability.render(rows))


# --------------------------------------------------------------------------- episode

with inspect:
    c_a, c_t = st.columns([2, 1])
    agent = c_a.selectbox("Agent", run.agents, key=f"ep_agent__{run_id}")
    track = c_t.radio("Track", TRACKS, horizontal=True, key=f"ep_track__{run_id}")

    lane = run.lane(agent, track)
    if not lane:
        st.info(f"{agent} has no episodes on the {track} track.")
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
            format_func=lambda e: f"[{e.episode_index:02d}]  {e.window_id}  ·  {e.regime}  ·  {e.status}",
        )

        m = ep.metrics
        _verdict(
            f"{ep.window_id} · {ep.regime} · {track}",
            f"<em>{agent}</em> finished at <b>{m['total_return']:+.2%}</b> for a floored "
            f"Sharpe of <b>{m['sharpe_floored']:+.3f}</b>, taking "
            f"<b>{m['n_trades']:.0f}</b> trades and holding a position "
            f"<b>{m['time_in_market']:.0%}</b> of the window.",
        )

        _stats([
            ("Sharpe (floored)", _signed(m["sharpe_floored"], f"{m['sharpe_floored']:+.3f}"),
             "vol floor binding" if m.get("vol_floor_binding") else "floor not binding"),
            ("Sharpe (raw)", _signed(m["sharpe_raw"], f"{m['sharpe_raw']:+.3f}"),
             "unfloored — diagnostic"),
            ("Total return", _signed(m["total_return"], f"{m['total_return']:+.2%}"),
             f"max dd {m['max_drawdown']:.1%}"),
            ("Fees paid", f"${m['fees_paid_cents'] / 100:,.2f}",
             f"{m['n_trades']:.0f} trades · {m['turnover']:.2f}x turnover"),
        ])

        if ep.metric_drift:
            st.error("Cached metrics disagree with the replay — the log is the authority.")
            st.json(ep.metric_drift)

        equity = [c / 100 for c in ep.equity_cents]
        start = equity[0]

        # An area mark carries an implicit baseline at zero, and Vega folds that baseline
        # into the domain -- so `zero=False` alone does not lift the axis, and a curve
        # that moves 20% around $10,000 gets squashed into the top sliver of a $0-based
        # chart. Pin the domain to the data and give the area an explicit floor to sit on.
        lo = min(min(equity), start) * 0.985
        hi = max(max(equity), start) * 1.015
        eq = pd.DataFrame({"tick": range(len(equity)), "equity": equity, "floor": lo})

        _lbl("Equity — every fee and every slippage cost is already inside this curve")
        area = (
            alt.Chart(eq).mark_area(
                line={"color": ACCENT, "strokeWidth": 1.8}, opacity=0.12, color=ACCENT
            ).encode(
                x=alt.X("tick:Q", axis=_axis("tick"), scale=alt.Scale(nice=False, domain=[0, 89])),
                y=alt.Y("equity:Q", axis=_axis("equity", format="$,.0f"),
                        scale=alt.Scale(domain=[lo, hi], nice=False)),
                y2="floor:Q",
                tooltip=[alt.Tooltip("tick:Q", title="tick"),
                         alt.Tooltip("equity:Q", title="equity", format="$,.2f")],
            )
        )
        opening = (
            alt.Chart(pd.DataFrame({"y": [start]}))
            .mark_rule(color=INK_3, strokeDash=[3, 3], strokeWidth=1)
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[lo, hi], nice=False)))
        )
        _chart(alt.layer(area, opening), 260)
        _note(
            f"Dashed line is the opening ${start:,.0f}. The axis is scaled to the curve, "
            f"not to zero — a $0 baseline would flatten the whole episode into a sliver."
        )

        _lbl("Position — the exposure that produced it, in shares held")
        pos = pd.DataFrame({"tick": range(len(ep.shares_by_tick)), "shares": ep.shares_by_tick})
        _chart(
            alt.Chart(pos).mark_line(
                interpolate="step-after", strokeWidth=1.6, color=INK_2
            ).encode(
                x=alt.X("tick:Q", axis=_axis("tick"), scale=alt.Scale(nice=False, domain=[0, 89])),
                # No axis title: at this height it collides with its own tick labels, and
                # the section label above already says these are shares.
                y=alt.Y("shares:Q", axis=_axis(None, tickCount=3)),
                tooltip=[alt.Tooltip("tick:Q", title="tick"),
                         alt.Tooltip("shares:Q", title="shares", format=".4f")],
            ),
            120,
        )

        _lbl(f"Calls — {len(ep.calls)} sent, including what the engine refused")
        st.dataframe(pd.DataFrame(ep.calls), hide_index=True, width="stretch", height=260)

        if ep.fills:
            _lbl(f"Fills — {len(ep.fills)}, each at a price the agent had not seen")
            _table(
                [("side", "side_l"), ("tick", "tick"), ("price", "price"),
                 ("shares Δ", "sh"), ("notional", "not"), ("friction", "fr")],
                [{"side_l": f['side'], "side_l__cls": "name",
                  "tick": f["fill_tick"],
                  "price": f"${f['fill_price']:,.2f}",
                  "sh": _signed(f["shares_delta"], f"{f['shares_delta']:+.4f}"),
                  "not": f"${f['gross_notional_cents'] / 100:,.0f}",
                  "fr": f"${f['friction_cents'] / 100:,.2f}"} for f in ep.fills],
            )

        n_in, n_out = st.columns(2)
        with n_in:
            _lbl("Note carried in")
            _note(ep.memory_note_in or "<em>None — first episode of the lane, or the "
                                       "no-memory arm.</em>")
        with n_out:
            _lbl("Note carried out")
            _note(ep.memory_note_out or "<em>None.</em>")

        with st.expander("Config, tokens, raw reliability"):
            st.json({"config": ep.config, "tokens": ep.tokens, "reliability": ep.reliability})
