"""Tool contract: the closed set of calls, errors, and results.

The error codes are a closed enum on purpose. The reliability scoreboard counts
them, so a stray free-text failure mode would silently become an uncounted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Tool(StrEnum):
    BUY = "Buy"
    SELL = "Sell"
    WAIT = "Wait"
    VIEW_WALLET = "ViewWallet"
    VIEW_PORTFOLIO = "ViewPortfolio"
    FETCH_DATA = "fetchData"
    GET_STATS = "getStats"


#: Tools that move the clock. Everything else is a read and is free of time cost
#: but not free of budget: the read cap (D13) is what stops an agent stalling.
TIME_ADVANCING = frozenset({Tool.BUY, Tool.SELL, Tool.WAIT})


class ErrorCode(StrEnum):
    SCHEMA_ERROR = "SCHEMA_ERROR"  # missing/extra/mistyped argument
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    NONPOSITIVE_QTY = "NONPOSITIVE_QTY"
    NAN_QTY = "NAN_QTY"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_SHARES = "INSUFFICIENT_SHARES"
    WAIT_OUT_OF_RANGE = "WAIT_OUT_OF_RANGE"
    READ_CAP_EXCEEDED = "READ_CAP_EXCEEDED"
    EPISODE_OVER = "EPISODE_OVER"


SCHEMA_CODES = frozenset({ErrorCode.SCHEMA_ERROR, ErrorCode.UNKNOWN_TOOL})


@dataclass(frozen=True)
class ToolError:
    code: ErrorCode
    message: str

    def as_dict(self) -> dict:
        return {"code": str(self.code), "message": self.message}


@dataclass(frozen=True)
class Fill:
    """One execution. Always at `open_{fill_tick}`, always `fill_tick == t + 1`."""

    side: str  # "buy" | "sell" | "liquidation"
    fill_tick: int
    fill_price: float
    shares_delta: float  # signed
    gross_notional_cents: int
    friction_cents: int
    cash_delta_cents: int  # signed, friction included

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "fill_tick": self.fill_tick,
            "fill_price": self.fill_price,
            "shares_delta": self.shares_delta,
            "gross_notional_cents": self.gross_notional_cents,
            "friction_cents": self.friction_cents,
            "cash_delta_cents": self.cash_delta_cents,
        }


@dataclass
class ToolResult:
    """What comes back from every call.

    On a time-advancing call this carries the new observation (D4): the agent does
    not have to spend a second call to find out what happened. That is the whole
    reason an episode costs ~60 LLM calls instead of ~300.
    """

    ok: bool
    tool: Tool | str
    advanced_time: bool = False
    error: ToolError | None = None
    obs: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    forced_wait: bool = False
    note: str | None = None

    def as_agent_payload(self) -> dict:
        """The dict actually handed to the model."""
        out: dict[str, Any] = {"ok": self.ok}
        if self.error is not None:
            out["error"] = self.error.as_dict()
        if self.data is not None:
            out.update(self.data)
        if self.obs is not None:
            out["observation"] = self.obs
        if self.forced_wait:
            out["forced_wait"] = True
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class EpisodeSummary:
    """Handed to the agent at the close of an episode, and the input it reflects on
    when it writes its rolling memory note (D11)."""

    window_id: str
    track: str
    metrics: dict[str, Any]
    reliability: dict[str, Any]
    equity_series_cents: list[int] = field(default_factory=list)
