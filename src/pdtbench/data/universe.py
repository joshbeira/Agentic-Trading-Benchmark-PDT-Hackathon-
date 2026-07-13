"""The ticker pool.

50 large-cap US names, spread across sectors, all liquid enough that a daily bar
is trustworthy and a 10 bps slippage assumption is not fantasy. Deep histories
are preferred because window selection needs 290 bars per window plus room to
spread selections across market eras (2008, 2011, 2015, 2018, 2020, 2022).

A handful (META, TSLA, ABBV) list later than 2005; the window sampler only ever
sees bars that exist, so a short history simply yields fewer candidates.
"""

from __future__ import annotations

UNIVERSE: tuple[str, ...] = (
    # Tech / semis
    "AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "TXN", "CSCO", "ORCL", "IBM",
    "ADBE", "CRM",
    # Internet / media
    "AMZN", "GOOGL", "META", "NFLX", "DIS", "CMCSA",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "AXP",
    # Energy
    "XOM", "CVX", "COP",
    # Health care
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "AMGN",
    # Consumer
    "WMT", "TGT", "COST", "HD", "LOW", "PG", "KO", "PEP", "MCD", "NKE",
    # Industrials / telecom / autos
    "CAT", "BA", "HON", "UPS", "LMT", "T", "TSLA",
)

assert len(UNIVERSE) == 50, f"expected 50 tickers, got {len(UNIVERSE)}"
assert len(set(UNIVERSE)) == 50, "duplicate ticker in universe"
