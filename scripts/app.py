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

The headline tab leads with D9's claim -- per-window Sharpe edge against per-window
identifiability -- because that is the claim the benchmark exists to make. Edge
concentrated in the windows the model can name is exploited memorization; edge flat
across identifiability is generalization.

Nothing here calls the API. It is safe to drive live in front of an audience, and it is
the replay fallback if the network is not.

--- On the presentation -------------------------------------------------------------

The read is a tearsheet, not a dashboard: this is operated by people who read numbers for
a living, and the job is to make a result legible enough to be trusted or attacked. So
every figure is monospaced and tabular-aligned, tables are static rather than sortable
widgets (a tearsheet is printed, not queried), and colour is never decoration.

The surface is a *two-plate proof*. Every real window in this benchmark has a twin, and
the twin is matched until only the identity differs; two impressions that should register
perfectly, where the fringe between them is the entire finding. So the page prints in one
ink (INK) at three screen densities, and there is exactly one chromatic ink (PROCESS) on
it. PROCESS appears nowhere except where two things that should agree do not: a Sharpe
edge over the twin, a cached metric that disagrees with the replay, a malformed call, a
slope that clears zero. **A clean run prints in black and white.**

Two rules follow from that, and both are load-bearing:

*Hue is not valence.* There is no green and no red. In this benchmark green-is-good is
false -- a large edge concentrated in the windows a model can name is the *incriminating*
result, the thing the twin was built to catch, and painting it green would be a lie told
in colour. Magenta means "look here", never "this is good". Sign is carried by the
`+`/`-` glyph, which cannot be misread.

*Type is epistemology.* A sentence set in the serif is a claim -- something a reader can
argue with. A figure set in the mono is evidence -- a number recomputed from the tick log.
The split holds inside a sentence: the figures inside a verdict are mono, so an assertion
and a measurement are never the same substance.

Charts are Altair, which ships inside streamlit, and both faces are vendored under
`static/`: no CDN anywhere, so the surface still renders with the venue wifi down.
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
# One ink, three densities, and one chromatic ink that means the plates missed.
PROOF = "#EDEFEA"     # the paper
PLATE = "#FBFCF9"     # the specimen field: chart and panel grounds
INK = "#14181B"       # the record: all type, the real series, bull.        100% screen
GRAPHITE = "#69737C"  # the control: the twin, intervals, secondary type, bear.  ~55%
SCREEN = "#929593"    # tertiary: chop, disabled figures.                         ~42%
RULE = "#D5DAD2"      # hairlines, gridlines, table borders
PROCESS = "#C81E65"   # THE GAP. The only chromatic ink on the page.

# GRAPHITE clears 4.5:1 on PROOF; the #8A96A1 it replaces did not, and this is projected.
# SCREEN is under that floor and is therefore never used for type -- only for a mark
# whose meaning is already carried by its shape.

SERIF_FACE = "Spectral"
MONO_FACE = "IBM Plex Mono"
SERIF = f"'{SERIF_FACE}', Georgia, 'Times New Roman', serif"
MONO = f"'{MONO_FACE}', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

#: Every rule that asks for a face must outrank the inheritance reset below, which is an
#: attribute plus an element and therefore beats a bare class. Prefixing with the app
#: container makes the specificity (0,2,1) against its (0,1,1) and the rule lands. This
#: is not decoration: without it `.reg-v` and `.lbl` silently render in the serif, and
#: the split between a claim and a measurement -- the whole point of the pairing -- dies
#: quietly, on a page that still looks fine.
APP = '[data-testid="stAppViewContainer"]'

# `auto`, not `expanded`: streamlit reads the viewport and only forces the panel open on
# a wide one. Pinned `expanded`, the instrument panel is 100% of a 375px screen and 40%
# of a tablet -- the run report becomes a strip behind a control it did not need to show,
# and on the phone the report is not reachable at all. On the projector this is still
# expanded, which is the case that matters.
st.set_page_config(page_title="pdtbench", layout="wide", initial_sidebar_state="auto")


# --------------------------------------------------------------------------- chrome

st.markdown(
    f"""
    <style>
      /* Streamlit's own chrome announces the framework, not the work. */
      [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
        display: none !important;
      }}

      [data-testid="stAppViewContainer"] {{ background: {PROOF}; }}
      [data-testid="stMainBlockContainer"] {{
        padding: 2.2rem 2.6rem 5rem;
        max-width: 1320px;
      }}
      /* Set the face on the root and let it inherit. A `*` selector here reaches every
         descendant including Streamlit's Material icon spans, whose glyphs are ligatures
         in an icon font -- overriding their family renders the icon's *name* as text
         ("keyboard_arrow_right" appearing next to an expander). Anything that needs a
         different face below asks for it by name. */
      html, body {{ font-family: {SERIF}; }}
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
      {APP} .sheet-title {{
        font-family: {SERIF};
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.015em;
        color: {INK};
        line-height: 1.05;
      }}
      {APP} .sheet-title small {{
        display: block;
        font-family: {MONO};
        font-size: 0.64rem;
        font-weight: 400;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: {GRAPHITE};
        margin-bottom: 0.42rem;
      }}
      {APP} .sheet-facts {{
        font-family: {MONO};
        font-size: 0.7rem;
        letter-spacing: 0.03em;
        color: {GRAPHITE};
        text-align: right;
        /* Holds the facts against the right edge once the band wraps. `space-between`
           stops distributing the moment the two children are on separate lines, and a
           shrink-to-fit box has nothing for `text-align` to push against -- so the run
           id drifts into the middle of the page and reads as a caption for the title. */
        margin-left: auto;
        font-variant-numeric: tabular-nums;
        line-height: 1.7;
      }}
      .sheet-facts b {{ color: {INK}; font-weight: 600; }}

      /* --- the verdict: the app says the answer in words ---
         The question is mono (it is the instrument asking); the answer is serif (it is a
         claim). Figures inside the answer stay mono, so an assertion and a measurement
         never read as the same substance. */
      .verdict {{
        border-left: 2px solid {INK};
        padding: 0.1rem 0 0.1rem 1.05rem;
        margin: 0.2rem 0 1.5rem;
      }}
      {APP} .verdict .q {{
        font-family: {MONO};
        font-size: 0.64rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {GRAPHITE};
        margin-bottom: 0.4rem;
      }}
      {APP} .verdict .a {{
        font-family: {SERIF};
        font-size: 1.24rem;
        line-height: 1.5;
        color: {INK};
        max-width: 62ch;
      }}
      {APP} .verdict .a b {{
        font-family: {MONO};
        font-size: 0.92em;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: {INK};
      }}
      .verdict .a em {{ color: {GRAPHITE}; font-style: italic; }}

      /* --- section labels --- */
      {APP} .lbl {{
        font-family: {MONO};
        font-size: 0.64rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {GRAPHITE};
        border-bottom: 1px solid {RULE};
        padding-bottom: 0.4rem;
        margin: 0.5rem 0 0.9rem;
      }}
      {APP} .note {{
        font-family: {SERIF};
        font-size: 0.9rem;
        color: {GRAPHITE};
        line-height: 1.6;
        max-width: 78ch;
      }}
      .note b {{ color: {INK}; font-weight: 600; }}
      {APP} .note code {{
        font-family: {MONO};
        font-size: 0.78em;
        background: {RULE};
        border-radius: 0;
        padding: 0.08em 0.32em;
        color: {INK};
      }}

      /* --- the contact sheet's key --- */
      {APP} .key {{
        display: flex;
        gap: 1.4rem;
        font-family: {MONO};
        font-size: 0.6rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: {GRAPHITE};
        margin: -0.3rem 0 0.7rem;
      }}
      .key span {{ display: inline-flex; align-items: center; gap: 0.42rem; }}
      .key i {{ width: 15px; height: 0; display: inline-block; }}
      .key .k-real {{ border-top: 1.5px solid {INK}; }}
      .key .k-twin {{ border-top: 1.5px dashed {GRAPHITE}; }}
      .key .k-gap {{ height: 8px; background: {PROCESS}; opacity: 0.85; }}

      /* --- tabs: a rule with an active mark, not a pill row ---
         Streamlit 1.59 renders tabs through react-aria: the baseweb hooks this used to
         target do not exist, and styling them was styling nothing. `stTab` is the
         testid, the label is a <p> inside a markdown container, and the moving underline
         is react-aria's own SelectionIndicator. */
      [data-testid="stTabs"] [role="tablist"] {{
        gap: 1.6rem;
        border-bottom: 1px solid {RULE};
      }}
      {APP} [data-testid="stTab"] p {{
        font-family: {MONO};
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 400;
        color: {GRAPHITE};
      }}
      [data-testid="stTab"] {{ padding: 0.35rem 0 0.6rem; }}
      {APP} [data-testid="stTab"][aria-selected="true"] p {{ color: {INK}; font-weight: 600; }}
      .react-aria-SelectionIndicator {{ background: {INK} !important; height: 2px !important; }}

      /* --- the register: figures ruled onto the paper, not cards floated above it ---
         No fill, no radius, no shadow, no vertical rules. Two horizontal rules hold the
         row; the label/value/qualifier hierarchy separates the cells on its own. */
      .reg {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
        gap: 0.9rem 2.2rem;
        border-top: 2px solid {INK};
        border-bottom: 1px solid {RULE};
        padding: 0.75rem 0 0.85rem;
        margin: 0.2rem 0 1.4rem;
      }}
      /* The label wraps rather than truncating: on a narrow window an ellipsis turns
         "sharpe edge (real - twin)" into "sharpe edge (real - tw...", which is a headline
         figure whose name the reader cannot finish. min-height reserves the second line
         so the values stay on one baseline across the row whether they wrap or not. */
      {APP} .reg-l {{
        font-family: {MONO};
        font-size: 0.58rem;
        line-height: 1.5;
        min-height: 2.9em;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: {GRAPHITE};
        text-wrap: balance;
      }}
      {APP} .reg-v {{
        font-family: {MONO};
        font-size: 1.5rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: {INK};
        line-height: 1.3;
        letter-spacing: -0.03em;
      }}
      {APP} .reg-s {{
        font-family: {MONO};
        font-size: 0.66rem;
        font-variant-numeric: tabular-nums;
        color: {GRAPHITE};
        white-space: nowrap;
      }}

      /* The one chromatic ink, and the only rule that grants it. */
      .gap {{ color: {PROCESS}; }}
      .flat {{ color: {GRAPHITE}; }}

      /* --- tables: printed, not queried ---
         These are hand-rolled HTML, not st.table. Streamlit's table is a canvas-ish
         widget whose internals shift between versions, and styling it through
         data-testid hooks silently hid a real column header (the diff between the
         header row and the body row was one cell, so every label sat over the wrong
         number). Owning the markup makes that class of bug impossible.

         There is no row hover. This is a printed record; paper does not light up when
         you point at it, and the highlight was only ever telling the reader that the
         surface had noticed the mouse. */
      .tt-wrap {{ overflow-x: auto; margin-bottom: 0.5rem; }}
      table.tt {{
        border-collapse: collapse;
        width: 100%;
        font-size: 0.78rem;
        font-family: {MONO};
        font-variant-numeric: tabular-nums;
      }}
      /* The font and border are restated on the cells themselves, not left to inherit:
         the global reset above matches every descendant with `*`, and a direct match
         beats an inherited value however specific the ancestor rule is. */
      table.tt th {{
        font-family: {MONO};
        font-size: 0.6rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        font-weight: 400;
        color: {GRAPHITE};
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
      table.tt td.name {{ font-weight: 600; letter-spacing: -0.01em; }}
      table.tt td.kind {{ color: {GRAPHITE}; font-size: 0.7rem; }}
      table.tt .ci {{ color: {GRAPHITE}; font-size: 0.7rem; margin-left: 0.35rem; }}
      table.tt .gap {{ color: {PROCESS}; }}
      table.tt .flat {{ color: {GRAPHITE}; }}
      /* The ranked row is marked with a screen of the ink, not a tint of some other
         hue -- the page has only one ink to shade with. */
      table.tt tr.lead td {{ background: rgba(20, 24, 27, 0.05); }}

      /* --- the contact sheet ---
         30 cells wide by design. It does not reflow: a contact sheet that rewraps is a
         different sheet, and the comparison between neighbours is the whole point. On a
         narrow screen it scrolls, which is what you do with the paper original. */
      .st-key-contact_sheet {{ overflow-x: auto; overflow-y: hidden; }}
      .st-key-contact_sheet [data-testid="stVegaLiteChart"] {{ min-width: 940px; }}

      /* --- sidebar: an instrument panel --- */
      [data-testid="stSidebar"] {{
        background: {PLATE};
        border-right: 1px solid {RULE};
        box-shadow: none !important;
      }}
      {APP} [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-family: {MONO};
        font-size: 0.61rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {GRAPHITE};
      }}
      {APP} .side-note {{
        font-family: {MONO};
        font-size: 0.66rem;
        line-height: 1.8;
        color: {GRAPHITE};
        font-variant-numeric: tabular-nums;
        border-top: 1px solid {RULE};
        padding-top: 0.7rem;
        margin-top: 0.4rem;
      }}
      .side-note b {{ color: {INK}; font-weight: 600; }}

      /* --- alerts: quieter than the default. A disagreement between the log and its
         own cache is a registration failure, so it is the one alert that gets the ink. */
      [data-testid="stAlert"] {{
        border-radius: 0;
        font-size: 0.86rem;
        background: {PLATE};
        border: 1px solid {RULE};
        border-left: 2px solid {GRAPHITE};
        color: {INK};
      }}
      [data-testid="stAlertContentError"] {{ border-left-color: {PROCESS}; }}

      {APP} [data-testid="stExpander"] summary p {{
        font-family: {MONO};
        font-size: 0.66rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {GRAPHITE};
      }}
      [data-testid="stExpander"] details {{ border-color: {RULE}; }}
      [data-testid="stDataFrame"] {{ border: 1px solid {RULE}; }}

      /* --- the framework's own chrome ---
         Streamlit ships a rounded, shadowed, animated widget kit. None of it is wrong;
         all of it belongs to a different page. These are the stable testids -- the
         emotion hashes beside them change between releases, so anything hung on those
         is styled by luck.

         The corner radius goes to zero everywhere. A printed sheet has no rounded
         corners, and 8px on a select box beside a 0px table is the tell that a theme
         was painted over a kit rather than drawn. */
      /* `*`, not a named child: the box that actually carries the radius is an emotion
         div with no stable hook of its own, so naming it would be naming a hash. The
         radio's dot is the one round thing that stays round -- it is a dot. */
      [data-testid="stSelectbox"] *,
      [data-testid="stNumberInput"] *,
      [data-testid="stNumberInputContainer"],
      [data-testid="stStatusWidget"],
      [data-testid="stSkeleton"],
      [data-testid="stExpander"] details,
      [data-testid="stExpander"] summary,
      [data-testid="stDataFrame"],
      [data-testid="stVegaLiteChart"] div,
      [data-testid="stVegaLiteChart"] summary,
      [data-testid="stAlert"],
      [data-testid="stMarkdownContainer"] code {{
        border-radius: 0 !important;
      }}
      [data-testid="stRadioOption"] div {{ border-radius: 50% !important; }}

      /* No shadow anywhere. Nothing on this page floats above the paper. */
      [data-testid="stElementToolbarButtonContainer"],
      [data-testid="stVegaLiteChart"] .vega-actions,
      [data-testid="stVegaLiteChart"] .vega-actions a {{
        box-shadow: none !important;
        border-radius: 0 !important;
      }}

      /* The toolbar that fades in over every chart on hover -- fullscreen, download,
         the Vega "..." menu. This is the scattered hover animation in person: five
         figures on a page means five things that light up when the mouse crosses them,
         on a surface whose whole claim is that it is a printed record. The figures are
         still readable, still tooltipped, still exportable from `scripts/analyze.py`. */
      [data-testid="stElementToolbar"],
      [data-testid="stVegaLiteChart"] .vega-actions,
      [data-testid="stVegaLiteChart"] summary {{
        display: none !important;
      }}

      /* Streamlit's tab strip fades its overflow edge with a gradient. There are five
         tabs and they fit; where they do not, the rule below scrolls them without
         painting a gradient onto a page that has none. */
      [data-testid="stTabs"] [role="tablist"] + button,
      [data-testid="stTabs"] button[class*="e1lncrqy"] {{
        background-image: none !important;
        background: transparent !important;
      }}

      /* Widgets read as instruments: hairline, square, on the plate. */
      [data-testid="stSelectbox"] div[class*="react-aria"],
      [data-testid="stNumberInputContainer"] {{
        border: 1px solid {RULE} !important;
        background: {PLATE} !important;
        transition: none !important;
      }}
      [data-testid="stSelectbox"], [data-testid="stNumberInput"] {{ max-width: 22rem; }}

      /* This page does not animate, so nothing on it needs a duration. There is no load
         sequence, no scroll reveal, no hover that moves: a printed record does not
         perform for the reader. Every easing here is the framework's own -- a select box
         easing its border, an expander rotating its chevron, a status pill fading -- and
         killing them wholesale is both more honest than naming each one and more durable:
         the next streamlit upgrade cannot smuggle a new one in behind a fresh hash. */
      [data-testid="stAppViewContainer"] *,
      [data-testid="stSidebar"] * {{
        transition: none !important;
        animation: none !important;
      }}

      /* Keyboard focus must stay visible: this is the one place a ring is not decoration.
         It is the ink, so it cannot be mistaken for a finding. */
      [data-testid="stAppViewContainer"] :focus-visible {{
        outline: 2px solid {INK};
        outline-offset: 2px;
      }}

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
        titleFont=MONO_FACE,
        titleFontSize=10,
        titleColor=GRAPHITE,
        titleFontWeight="normal",
        labelFont=MONO_FACE,
        labelFontSize=10,
        labelColor=GRAPHITE,
        domainColor=RULE,
        tickColor=RULE,
        gridColor="#E3E6E0",
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
        .configure_view(strokeWidth=0, fill=PLATE)
        .configure_legend(
            labelFont=MONO_FACE,
            labelFontSize=10,
            labelColor=GRAPHITE,
            titleFont=MONO_FACE,
            titleFontSize=9,
            titleColor=GRAPHITE,
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


def _register(items: list[tuple[str, str, str]]) -> None:
    """A row of figures: label, value, and the qualifier that keeps the value honest.

    Replaces st.metric because the interval must be visibly subordinate to the point
    estimate rather than crammed into it at the same weight — and because st.metric is a
    card, and a card is a thing that floats above paper.
    """
    cells = "".join(
        f'<div class="reg-c"><div class="reg-l">{label}</div>'
        f'<div class="reg-v">{value}</div>'
        f'<div class="reg-s">{sub or "&nbsp;"}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="reg">{cells}</div>', unsafe_allow_html=True)


# Regime is encoded by shape first and ink density second, never by hue. Bull/bear as
# green/red is unreadable to roughly one viewer in twelve and this gets projected -- and
# on this page a hue would additionally claim a valence the benchmark does not have.
REGIME_COLOR = alt.Scale(domain=["bull", "bear", "chop"], range=[INK, GRAPHITE, SCREEN])
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


def _gap(value: float, text: str) -> str:
    """A gap, inked only where the reader can actually see one.

    PROCESS is the page's one chromatic ink and it means "two things that should agree,
    don't". Whether they agree is judged on the *rendered* figure, not the float behind
    it: an edge of 7e-8 formats as `+0.00` and is zero to anyone reading the screen.
    Inking that magenta would announce a finding that is a rounding error — a lie told in
    colour, which is the one thing this surface may never do.

    Levels are not gaps. A Sharpe, a return, a share count is set in plain ink however
    large it is; only the difference between two things the benchmark holds equal earns
    the second colour.
    """
    shown_zero = not any(ch in "123456789" for ch in text)
    return f'<span class="{"flat" if shown_zero else "gap"}">{text}</span>'


def _sharpe(iv) -> str:
    """The point estimate, with its interval kept subordinate to it.

    The interval is the honest part of the number and must be visible, but a table where
    every cell is `+1.83 [+1.40, +2.20]` is a table nobody reads. Point estimate leads at
    full weight; the interval trails in muted small type on the same line.
    """
    return f'{iv.point:+.2f}<span class="ci">[{iv.lo:+.2f}, {iv.hi:+.2f}]</span>'


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


# --------------------------------------------------------------- the contact sheet

#: The Sharpe edge at which a window's two impressions count as having missed. This is
#: the same floor the scatter below already draws with (`_domain(..., 0.40)` is ±0.20),
#: and it must stay the same number: a cell inked magenta here and a point sitting inside
#: the flat band there would be one page disagreeing with itself about what an edge is.
SHEET_MIN_EDGE = 0.20

#: The smallest return span a cell will expose for, as a fraction. A window where the
#: agent never took a position moves 0%, and a frame auto-fitted to it would enlarge
#: rounding dust into a pair of wildly diverging curves. Below this the cell draws flat,
#: because the episode was flat.
CELL_MIN_SPAN = 0.04


def _contact_sheet(run: Run, agent: str, res) -> None:
    """Every window printed twice — the real path in ink, its twin in a lighter screen —
    with the magenta reserved for the frames where the two results actually differ.

    **The fringe is not the area between the curves.** That was the first drawing and it
    was a lie: the twin is matched to its real window on *total return*, not path, so the
    two curves wander apart mid-window purely because they are different price series and
    rejoin at the end. On the fixture — whose verdict is "no slope, the question is
    degenerate" — that drawing filled all thirty cells with magenta. A figure that shouts
    while the sentence above it says nothing happened is worse than no figure.

    So the ink of each frame is decided by the *Sharpe edge*, the per-window scalar the
    headline actually regresses: magenta only where |edge| clears SHEET_MIN_EDGE. The
    curves show you the two impressions; the colour tells you whether the result differed.
    A run with no edge prints black and white, which is the promise the palette makes.

    Each frame is exposed for its own subject — normalised to its own return range, with
    a floor — exactly as the frames on a real contact sheet are. That makes the *shape*
    of two impressions comparable and their amplitudes not; the edge beside each window
    id is the number that carries magnitude, and the caption says so.

    Ordered by identifiability where a probe can say what that is, so a model that only
    profits on the windows it recognises pools its magenta at the top left and the thesis
    is visible before a number is read. Never ordered by edge: sorting a sheet by the
    quantity it draws manufactures a gradient out of noise.
    """
    pairs = run.paired(agent)
    if not pairs:
        _note(f"<em>{agent} has no window with both a real and a twin episode — "
              f"nothing to register.</em>")
        return

    ident = {p.window_id: p.identifiability for p in res.points} if res else {}
    ranked = bool(ident) and len({round(v, 6) for v in ident.values()}) > 1

    cells, rows = [], []
    for real, twin in pairs:
        rr = [c / real.equity_cents[0] - 1 for c in real.equity_cents]
        tt = [c / twin.equity_cents[0] - 1 for c in twin.equity_cents]
        n = min(len(rr), len(tt))
        rr, tt = rr[:n], tt[:n]
        edge = real.metrics["sharpe_floored"] - twin.metrics["sharpe_floored"]
        # The frame's own exposure, floored so a flat episode draws flat.
        vals = rr + tt
        mid = (min(vals) + max(vals)) / 2
        half = max((max(vals) - min(vals)) / 2, CELL_MIN_SPAN / 2)
        label = f"{real.window_id}  {edge:+.2f}"
        cells.append({"window": real.window_id, "ident": ident.get(real.window_id),
                      "edge": edge, "label": label})
        for t in range(n):
            rows.append({
                "label": label, "tick": t,
                "real": (rr[t] - mid) / half,
                "twin": (tt[t] - mid) / half,
                "material": abs(edge) >= SHEET_MIN_EDGE,
                "edge": edge, "win": real.window_id,
            })

    cells.sort(key=(lambda c: -c["ident"]) if ranked else (lambda c: c["window"]))
    order = [c["label"] for c in cells]
    n_material = sum(1 for c in cells if abs(c["edge"]) >= SHEET_MIN_EDGE)
    df = pd.DataFrame(rows)

    x = alt.X("tick:Q", axis=None, scale=alt.Scale(domain=[0, 89], nice=False))
    yscale = alt.Scale(domain=[-1.18, 1.18], nice=False)
    tip = [alt.Tooltip("win:N", title="window"),
           alt.Tooltip("edge:Q", title="sharpe edge", format="+.3f")]

    fringe = (
        alt.Chart().mark_area(color=PROCESS, opacity=0.85)
        .transform_filter(alt.datum.material)
        .encode(x=x, y=alt.Y("real:Q", axis=None, scale=yscale), y2="twin:Q", tooltip=tip)
    )
    twin_line = alt.Chart().mark_line(
        color=GRAPHITE, strokeWidth=0.9, strokeDash=[2.5, 2]
    ).encode(x=x, y=alt.Y("twin:Q", axis=None, scale=yscale), tooltip=tip)
    real_line = alt.Chart().mark_line(color=INK, strokeWidth=1.25).encode(
        x=x, y=alt.Y("real:Q", axis=None, scale=yscale), tooltip=tip)

    sheet = (
        alt.layer(fringe, twin_line, real_line, data=df)
        .properties(width=80, height=50)
        .facet(
            facet=alt.Facet(
                "label:N", sort=order, title=None,
                header=alt.Header(labelFont=MONO_FACE, labelFontSize=8.5, labelColor=INK,
                                  labelAnchor="start", labelPadding=1,
                                  labelBaseline="bottom"),
            ),
            columns=10,
            spacing=9,
        )
        .configure_view(strokeWidth=0, fill=PLATE)
    )

    st.markdown(
        f'<div class="key">'
        f'<span><i class="k-real"></i>real</span>'
        f'<span><i class="k-twin"></i>twin</span>'
        f'<span><i class="k-gap"></i>|edge| ≥ {SHEET_MIN_EDGE:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="contact_sheet"):
        st.altair_chart(sheet, theme=None, use_container_width=False)

    _note(
        f"One frame per window, <b>{len(cells)} of them</b>: the real window in ink, its "
        f"twin dashed over it. The figure beside each window id is that window's "
        f"<b>Sharpe edge</b> — real minus twin — and it is the same quantity the slope "
        f"below regresses. <b>A frame is inked magenta only where |edge| ≥ "
        f"{SHEET_MIN_EDGE:.2f}</b>"
        + (f": <b>{n_material} of {len(cells)}</b> here."
           if n_material else
           f", and <b>none of these {len(cells)} clears it</b> — so this sheet prints "
           f"black and white, which is what a run with no edge is supposed to look like.")
        + f" Each frame is exposed for its own window, so the <em>shape</em> of the two "
        f"impressions is comparable across the sheet and their <em>amplitude</em> is not — "
        f"the edge figure carries that.<br>"
        + (f"Ordered by identifiability, hardest-to-name last: if the profit lives where "
           f"<b>{agent}</b> recognises the chart, the magenta pools at the top left."
           if ranked else
           "Ordered by window id. <em>Identifiability cannot rank this sheet — "
           + ("no probe has scored this run" if not ident else
              "every window shares one identifiability, so there is no order to impose")
           + ".</em>")
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
        agent = st.selectbox("Agent", run.agents, key=f"headline_agent__{run_id}")
        _lbl("The contact sheet · every window printed twice, and where they missed")
        _contact_sheet(run, agent, None)
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

        _lbl("The contact sheet · every window printed twice, and where they missed")
        _contact_sheet(run, agent, res)

        if res is not None:
            _register([
                ("Edge ~ identifiability",
                 _gap(res.slope.slope, f"{res.slope.slope:+.3f}") if defined else "—",
                 "the headline slope" if defined else "identifiability does not vary"),
                ("Sharpe edge (real − twin)",
                 _gap(res.edge.point, f"{res.edge.point:+.2f}"),
                 f"[{res.edge.lo:+.2f}, {res.edge.hi:+.2f}]"),
                ("Identifiability",
                 _gap(res.calibrated_identifiability.point,
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

            _lbl("Edge against identifiability · the regression the claim rests on")
            base = alt.Chart(df)
            zero = (
                alt.Chart(pd.DataFrame({"y": [0.0]}))
                .mark_rule(color=RULE, strokeWidth=1)
                .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=ydom, nice=False)))
            )
            pts = base.mark_point(filled=True, size=110, opacity=0.9,
                                  stroke=INK, strokeWidth=0.6).encode(
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
                    .mark_line(color=PROCESS, strokeWidth=2)
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
    real_rows = scoreboard.build(run, "real", seed=int(seed))
    twin_rows = {r.agent_id: r for r in scoreboard.build(run, "twin", seed=int(seed))}

    if not real_rows:
        st.info("No scorable episodes on the real track.")
    else:
        top = real_rows[0]
        _verdict(
            "Who traded the real track best, and did the twin agree?",
            f"<b>{top.agent_id}</b> leads at <b>{top.sharpe}</b> median floored Sharpe "
            f"over {top.n} episodes. Read the per-regime panels below before believing it: "
            f"<em>ten episodes a cell is a small sample, and the intervals say so.</em>",
        )

        # Real and twin sit in one table rather than behind a track toggle: the twin is
        # not another dataset to browse, it is the control this agent's real number is
        # only meaningful against. A radio button asks the reader to hold one column in
        # their head while they fetch the other, which is the comparison the benchmark
        # exists to make and the one thing the page should never make them do.
        _lbl("Pooled · median floored Sharpe, real against its twin · ranked by real")
        _table(
            [("agent", "agent_l"), ("", "kind_l"), ("n", "n"),
             ("real", "real"), ("twin", "twin"), ("edge", "edge"),
             ("return", "ret"), ("max dd", "dd"), ("turnover", "to"),
             ("fees", "fees"), ("in mkt", "mkt"), ("trades", "tr"), ("floor", "fl")],
            [{"agent_l": r.agent_id, "agent_l__cls": "name",
              "kind_l": r.kind, "kind_l__cls": "kind",
              "n": r.n,
              "real": _sharpe(r.sharpe),
              "twin": (f"{twin_rows[r.agent_id].sharpe.point:+.2f}"
                       if r.agent_id in twin_rows else "—"),
              "edge": (_gap(r.sharpe.point - twin_rows[r.agent_id].sharpe.point,
                            f"{r.sharpe.point - twin_rows[r.agent_id].sharpe.point:+.2f}")
                       if r.agent_id in twin_rows else "—"),
              "ret": f"{r.total_return:+.1%}",
              "dd": f"{r.max_drawdown:.1%}",
              "to": f"{r.turnover:.2f}",
              "fees": f"${r.fees_cents / 100:,.0f}",
              "mkt": f"{r.time_in_market:.0%}",
              "tr": f"{r.n_trades:.1f}",
              "fl": f"{r.pct_floor_binding:.0%}"} for r in real_rows],
            lead=True,
        )
        _note(
            "The volatility floor is what stops a lucky one-percent position posting a "
            "Sharpe of +30 — <code>floor</code> is how often it caught this agent. "
            "<b>Do not quote a pooled “beat buy-and-hold”:</b> the 10/10/10 regime balance "
            "forces that median to roughly zero by construction. Compare within a regime. "
            "<code>edge</code> is real minus twin: for a scripted baseline it is the "
            "difference between two charts, not a finding — <em>only a model can "
            "recognise anything.</em>"
        )

        _lbl("By regime · real track · the primary trading result (D12)")
        by_regime = scoreboard.by_regime(run, "real", seed=int(seed))
        live = [(k, v) for k, v in by_regime.items() if v]
        for col, (regime, rrows) in zip(st.columns(len(live) or 1), live):
            with col:
                st.markdown(
                    f'<div class="lbl" style="color:{INK}">{regime}</div>',
                    unsafe_allow_html=True,
                )
                _table(
                    [("agent", "agent_l"), ("n", "n"), ("sharpe", "sh")],
                    [{"agent_l": r.agent_id, "agent_l__cls": "name", "n": r.n,
                      "sh": f"{r.sharpe.point:+.2f}"}
                     for r in rrows],
                )

        twin_by_regime = [(k, v) for k, v in
                          scoreboard.by_regime(run, "twin", seed=int(seed)).items() if v]
        if twin_by_regime:
            with st.expander("The same panels on the twin track — the control"):
                for col, (regime, rrows) in zip(st.columns(len(twin_by_regime)),
                                                twin_by_regime):
                    with col:
                        st.markdown(
                            f'<div class="lbl" style="color:{GRAPHITE}">{regime} · twin</div>',
                            unsafe_allow_html=True,
                        )
                        _table(
                            [("agent", "agent_l"), ("n", "n"), ("sharpe", "sh")],
                            [{"agent_l": r.agent_id, "agent_l__cls": "name", "n": r.n,
                              "sh": f"{r.sharpe.point:+.2f}"}
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
            # `excess` carries the magenta because it is a gap by construction: what the
            # agent scored minus what the window was worth. The two levels it is built
            # from stay in ink.
            line = (
                alt.Chart(long)
                .mark_line(strokeWidth=1.9, opacity=0.95)
                .encode(
                    x=alt.X("episode:Q",
                            axis=_axis("episode index within the lane", tickCount=8),
                            scale=alt.Scale(domain=[0, max(c.index)], nice=False)),
                    y=alt.Y("sharpe:Q", axis=_axis("floored sharpe", tickCount=5)),
                    color=alt.Color("series:N",
                                    scale=alt.Scale(domain=order,
                                                    range=[PROCESS, INK, GRAPHITE]),
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

    # A malformed call is the agent failing to register with the engine, so the counts
    # that mean misbehaviour take the magenta and a clean board prints black on white.
    _lbl("Counted from calls[] — the primary record, not the cached summary")
    _table(
        [("agent", "agent_l"), ("", "kind_l"), ("eps", "eps"), ("calls", "calls"),
         ("invalid", "inv"), ("invalid %", "invp"), ("schema %", "sch"),
         ("forced/ep", "forced"), ("nudges/ep", "nudge"), ("readcap/ep", "cap"),
         ("wallclock", "wall"), ("capped", "ncap"), ("errors", "err"), ("", "ok")],
        [{"agent_l": r.agent_id, "agent_l__cls": "name",
          "kind_l": r.kind, "kind_l__cls": "kind",
          "eps": r.n_episodes, "calls": f"{r.n_calls:,}",
          "inv": _gap(r.n_invalid, str(r.n_invalid)) if r.n_invalid else "—",
          "invp": _gap(r.invalid_rate, f"{r.invalid_rate:.1%}") if r.n_invalid else "—",
          "sch": _gap(r.schema_error_rate, f"{r.schema_error_rate:.1%}")
                 if r.schema_error_rate else "—",
          "forced": _gap(r.forced_waits_per_episode, f"{r.forced_waits_per_episode:.2f}")
                    if r.forced_waits_per_episode else "—",
          "nudge": f"{r.prose_nudges_per_episode:.2f}" if r.prose_nudges_per_episode else "—",
          "cap": f"{r.read_cap_hits_per_episode:.2f}" if r.read_cap_hits_per_episode else "—",
          "wall": f"{r.median_wallclock_s:.0f}s",
          "ncap": _gap(r.n_capped, str(r.n_capped)) if r.n_capped else "—",
          "err": _gap(r.n_agent_errors, str(r.n_agent_errors)) if r.n_agent_errors else "—",
          "ok": '<span class="flat">clean</span>' if r.clean else '<span class="gap">·</span>',
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

        # These are levels, not gaps: they stay in ink however large they are. Only the
        # difference between two things the benchmark holds equal earns the second colour.
        _register([
            ("Sharpe (floored)", f"{m['sharpe_floored']:+.3f}",
             "vol floor binding" if m.get("vol_floor_binding") else "floor not binding"),
            ("Sharpe (raw)", f"{m['sharpe_raw']:+.3f}", "unfloored — diagnostic"),
            ("Total return", f"{m['total_return']:+.2%}",
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
                line={"color": INK, "strokeWidth": 1.8}, opacity=0.1, color=INK
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
            .mark_rule(color=GRAPHITE, strokeDash=[3, 3], strokeWidth=1)
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
                interpolate="step-after", strokeWidth=1.6, color=GRAPHITE
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
                  "sh": f"{f['shares_delta']:+.4f}",
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
