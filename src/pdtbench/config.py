"""Run configuration.

One frozen dataclass holding every constant the benchmark depends on, hashed
into `config_sha256` and stamped on every episode log. If a number matters to a
result, it lives here — not inline in a module somewhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .hashing import hash_json

ENGINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WINDOWS_DIR = DATA_DIR / "windows"
RUNS_DIR = REPO_ROOT / "runs"


@dataclass(frozen=True)
class Config:
    # --- episode shape (D8, derived mechanics) ---
    n_warmup: int = 200  # visible pre-episode bars; SMA-50 defined from tick 0
    n_scored: int = 90  # ~one trading quarter
    initial_capital_cents: int = 1_000_000  # $10,000

    # --- friction (D10): deterministic, no execution RNG ---
    fee_bps: float = 2.0
    slippage_bps: float = 8.0

    # --- execution rules (D3, derived mechanics) ---
    fill_rule: str = "next_open"
    terminal_rule: str = "liquidate_at_final_close"

    # --- interface limits (D4, D13) ---
    max_wait: int = 10
    max_reads_per_tick: int = 8
    max_consecutive_invalid: int = 3
    fetch_lookback_default: int = 50
    fetch_lookback_cap: int = 200
    episode_wallclock_cap_s: int = 1200

    # --- metric (D2) ---
    vol_floor_multiple: float = 0.25
    trading_days_per_year: int = 252

    # --- observation masking (both tracks pass through the same function) ---
    price_base: float = 100.0
    price_decimals: int = 2
    volume_decimals: int = 2
    volume_rolling_window: int = 20

    # The share count is held at exactly the precision it is displayed at. If the
    # engine kept more precision than it showed, the tick log would not contain
    # enough information to reproduce its own equity mark (it was off by a cent
    # whenever shares x close landed near a rounding boundary), and an agent that
    # sold precisely the position it had been shown would be left holding dust.
    share_decimals: int = 6

    # --- window selection (D12) ---
    regime_bull_threshold: float = 0.08
    regime_bear_threshold: float = -0.08
    n_windows_per_regime: int = 10
    candidate_stride: int = 10  # bars between candidate window starts
    max_windows_per_quarter: int = 2  # decorrelates windows across tickers
    max_windows_per_ticker: int = 1

    # --- twin generator (D6, D9) ---
    expected_block_length: int = 20  # stationary bootstrap, geometric
    # Rejection bands. A block bootstrap draws with replacement, so an unmatched
    # twin's realized 90-bar return has a standard deviation of ~16% — enough to
    # turn a bull window's twin into a bear. Matching on realized moments is what
    # makes the twin a control rather than just another window.
    twin_return_tol: float = 0.03  # abs, on the scored segment's total return
    twin_vol_rel_tol: float = 0.15  # relative, on the scored segment's daily vol
    twin_warmup_return_tol: float = 0.10  # the agent sees the warmup and priors on it
    twin_warmup_vol_rel_tol: float = 0.25
    twin_batch: int = 4096
    twin_max_attempts: int = 2_000_000

    # --- seeds ---
    master_seed: int = 20260713

    # --- data ---
    data_start: str = "2005-01-01"
    data_end: str | None = None  # None => today

    @property
    def friction_bps_per_side(self) -> float:
        return self.fee_bps + self.slippage_bps

    @property
    def friction_rate(self) -> float:
        """Fraction of notional charged per side, e.g. 0.0010."""
        return self.friction_bps_per_side / 10_000.0

    @property
    def n_bars(self) -> int:
        return self.n_warmup + self.n_scored

    @property
    def n_windows(self) -> int:
        return self.n_windows_per_regime * 3

    @property
    def last_decision_tick(self) -> int:
        """Tick 89 is terminal (no action accepted): a tick-89 order would need
        an open at tick 90, which is outside the scored window."""
        return self.n_scored - 2

    def log_block(self) -> dict:
        """The `config` block written into every episode's `meta` record."""
        block = {
            "initial_capital_cents": self.initial_capital_cents,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "friction_bps_per_side": self.friction_bps_per_side,
            "fill_rule": self.fill_rule,
            "terminal_rule": self.terminal_rule,
            "max_wait": self.max_wait,
            "max_reads_per_tick": self.max_reads_per_tick,
            "max_consecutive_invalid": self.max_consecutive_invalid,
            "fetch_lookback_default": self.fetch_lookback_default,
            "fetch_lookback_cap": self.fetch_lookback_cap,
            "episode_wallclock_cap_s": self.episode_wallclock_cap_s,
            "vol_floor_multiple": self.vol_floor_multiple,
            "n_warmup": self.n_warmup,
            "n_scored": self.n_scored,
        }
        block["config_sha256"] = hash_json(block)
        return block

    def sha256(self) -> str:
        return hash_json(asdict(self))


DEFAULT = Config()
