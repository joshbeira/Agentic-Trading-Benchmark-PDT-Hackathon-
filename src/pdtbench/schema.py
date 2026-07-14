"""The tick-log contract, executable.

`schemas/tick_log.md` describes the format. This module *is* the format: every producer's
output is validated against it in the test suite, so the prose and the bytes cannot drift
apart. They already had — the engine was emitting `obs.bar` as
`{open, high, low, close, volume}` while the schema and the fixtures said `{o, h, l, c, v}`,
and nothing noticed, because the analytics layer never reads `obs.bar`. The first consumer
that did (the log viewer) would have crashed on every real episode.

Two rules make this worth having:

**Unknown keys are errors.** A producer that starts writing a new field must document it
here, in the same commit. Permissiveness is how a schema becomes fiction.

**Money is `int`.** A `*_cents` field carrying a float is rejected. Integer cents is the
whole reason there is no floating-point exploit in the equity series, and a schema that
shrugged at `1000000.0` would quietly give that away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.2.0"

TRACKS = ("real", "twin")
REGIMES = ("bull", "bear", "chop")
AGENT_KINDS = ("llm", "baseline")
MEMORY_MODES = ("rolling_note", "none")
STATUSES = ("ok", "wallclock_capped", "agent_error")
FILL_SIDES = ("buy", "sell", "liquidation")
TOOLS = ("Buy", "Sell", "Wait", "ViewWallet", "ViewPortfolio", "fetchData", "getStats")
ERROR_CODES = (
    "SCHEMA_ERROR", "UNKNOWN_TOOL", "NONPOSITIVE_QTY", "NAN_QTY", "INSUFFICIENT_CASH",
    "INSUFFICIENT_SHARES", "WAIT_OUT_OF_RANGE", "READ_CAP_EXCEEDED", "EPISODE_OVER",
)
BAR_KEYS = ("open", "high", "low", "close", "volume")


# ----------------------------------------------------------------- field specs


@dataclass(frozen=True)
class Spec:
    kind: str  # int | num | str | bool | obj | list | any
    required: bool = True
    nullable: bool = False
    enum: tuple | None = None
    const: Any = None
    fields: dict[str, "Spec"] | None = None  # for obj
    item: "Spec | None" = None  # for list
    free: bool = False  # obj: allow undeclared keys (tool args only)


def Int(**kw) -> Spec: return Spec("int", **kw)
def Num(**kw) -> Spec: return Spec("num", **kw)
def Str(**kw) -> Spec: return Spec("str", **kw)
def Bool(**kw) -> Spec: return Spec("bool", **kw)
def Obj(fields, **kw) -> Spec: return Spec("obj", fields=fields, **kw)
def Arr(item, **kw) -> Spec: return Spec("list", item=item, **kw)
def Free(**kw) -> Spec: return Spec("obj", fields={}, free=True, **kw)


# ------------------------------------------------------------------ the schema

BAR = Obj({k: Num() for k in BAR_KEYS})

STATS = Obj({
    "price": Num(),
    "sma20": Num(),
    "sma50": Num(),
    "vol20_ann": Num(),
    "ep_high": Num(),
    "ep_low": Num(),
    # Duplicated from obs.tick / obs.ticks_remaining on purpose: `stats` is also the
    # standalone payload of the getStats tool, where the agent has no surrounding obs.
    "tick": Int(),
    "ticks_remaining": Int(),
})

PORTFOLIO = Obj({
    "cash_cents": Int(),
    "shares": Num(),
    "position_value_cents": Int(),
    "equity_cents": Int(),
    "unrealized_pnl_cents": Int(),
    "avg_cost": Num(nullable=True),  # null when flat
})

OBS = Obj({
    "tick": Int(),
    "ticks_remaining": Int(),
    "bar": BAR,
    "stats": STATS,
    "portfolio": PORTFOLIO,
    "done": Bool(),
})

CALL = Obj({
    "seq": Int(),
    "tool": Str(),  # not enum'd: an UNKNOWN_TOOL error must record what was actually said
    "args": Free(),
    "ok": Bool(),
    "advanced_time": Bool(),
    "error": Obj({"code": Str(enum=ERROR_CODES), "message": Str()}, required=False),
    "latency_ms": Num(required=False),
    "tokens": Obj({"in": Int(), "out": Int()}, required=False, nullable=True),
})

ACTION = Obj({
    "tool": Str(enum=TOOLS),
    "args": Free(),
    "forced": Bool(),
    "n_effective": Int(required=False),  # Wait only
}, nullable=True)

FILL = Obj({
    "side": Str(enum=FILL_SIDES),
    "fill_tick": Int(),
    "fill_price": Num(),
    "shares_delta": Num(),
    "gross_notional_cents": Int(),
    "friction_cents": Int(),
    "cash_delta_cents": Int(),
}, nullable=True)

CONFIG = Obj({
    "config_sha256": Str(),
    "initial_capital_cents": Int(),
    "fee_bps": Num(),
    "slippage_bps": Num(),
    "friction_bps_per_side": Num(),
    "fill_rule": Str(enum=("next_open",)),
    "terminal_rule": Str(enum=("liquidate_at_final_close",)),
    "max_wait": Int(),
    "max_reads_per_tick": Int(),
    "max_consecutive_invalid": Int(),
    "fetch_lookback_default": Int(),
    "fetch_lookback_cap": Int(),
    "episode_wallclock_cap_s": Int(),
    "vol_floor_multiple": Num(),
    "trading_days_per_year": Int(),
    "n_warmup": Int(),
    "n_scored": Int(),
})

META = {
    "type": Str(const="meta"),
    "schema_version": Str(),
    "run_id": Str(nullable=True),
    "episode_id": Str(),
    "episode_index": Int(),  # index within the (agent, track) memory lane. Required:
                             # it is the learning curve's x-axis, and a silent 0 would
                             # collapse every lane onto one point.
    "agent": Obj({
        "id": Str(),
        "kind": Str(enum=AGENT_KINDS),
        "memory": Str(enum=MEMORY_MODES),
        "provider": Str(required=False),
        "model": Str(required=False),
        "temperature": Num(required=False, nullable=True),
        "system_prompt_sha256": Str(required=False),
        "seed": Int(required=False),
    }),
    "track": Str(enum=TRACKS),
    "window": Obj({
        "window_id": Str(),
        "source_ticker": Str(nullable=True),
        "regime": Str(enum=REGIMES),
        "n_warmup": Int(),
        "n_scored": Int(),
        "bh_daily_vol": Num(),
        "series_sha256": Str(),
    }),
    "config": CONFIG,
    "dataset": Obj({
        "dataset_sha256": Str(),
        "parquet_path": Str(required=False),
        "as_of": Str(required=False),
    }),
    "seeds": Obj({
        "master_seed": Int(),
        "window_seed": Int(required=False, nullable=True),
        "twin_seed": Int(required=False, nullable=True),
        "agent_seed": Int(required=False, nullable=True),
    }),
    "code": Obj({
        "git_commit": Str(nullable=True),
        "engine_version": Str(),
    }),
    "started_at": Str(),
}

TICK = {
    "type": Str(const="tick"),
    "t": Int(),
    "decision_point": Bool(),
    "skipped_by": Int(required=False),  # present iff skipped inside a Wait(n)
    "obs": OBS,
    "calls": Arr(CALL),  # may be empty (skipped ticks, terminal tick)
    "action": ACTION,
    "fill": FILL,
    "equity_cents": Int(),
    "invalid_count": Int(),
    "reads_count": Int(),
}

METRICS = Obj({
    "total_return": Num(),
    "sharpe_floored": Num(),
    "sharpe_raw": Num(),
    "vol_floor_binding": Bool(),
    "realized_vol_ann": Num(),
    "max_drawdown": Num(),
    "turnover": Num(),
    "fees_paid_cents": Int(),
    "time_in_market": Num(),
    "n_trades": Int(),
})

EPISODE_END = {
    "type": Str(const="episode_end"),
    "t_final": Int(),
    "terminal": Obj({
        "liquidated_shares": Num(),
        "liquidation_price": Num(nullable=True),
        "liquidation_friction_cents": Int(),
        "final_equity_cents": Int(),
    }),
    "equity_series_cents": Arr(Int()),
    "metrics": METRICS,
    "reliability": Obj({
        "n_calls": Int(),
        "n_invalid": Int(),
        "invalid_rate": Num(),
        "n_schema_errors": Int(),
        "n_forced_waits": Int(),
        "n_prose_nudges": Int(),
        "n_read_cap_hits": Int(),
        "wallclock_s": Num(),
        "hit_wallclock_cap": Bool(),
    }),
    "memory_note_in": Str(nullable=True),
    "memory_note_out": Str(nullable=True),
    "cost": Obj({
        "tokens_in": Int(),
        "tokens_out": Int(),
        "usd": Num(required=False),
    }),
    "status": Str(enum=STATUSES),
    "ended_at": Str(),
}


# ------------------------------------------------------------------- validation


@dataclass
class Report:
    path: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        if self.ok:
            return f"{self.path}: OK"
        return f"{self.path}: {len(self.errors)} error(s)\n  " + "\n  ".join(self.errors[:12])


def _check(value: Any, spec: Spec, where: str, errs: list[str]) -> None:
    if value is None:
        if not spec.nullable:
            errs.append(f"{where}: null, but the field is not nullable")
        return

    k = spec.kind
    if k == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errs.append(f"{where}: expected int, got {type(value).__name__} ({value!r})")
            return
    elif k == "num":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errs.append(f"{where}: expected number, got {type(value).__name__}")
            return
    elif k == "str":
        if not isinstance(value, str):
            errs.append(f"{where}: expected str, got {type(value).__name__}")
            return
    elif k == "bool":
        if not isinstance(value, bool):
            errs.append(f"{where}: expected bool, got {type(value).__name__}")
            return
    elif k == "obj":
        if not isinstance(value, dict):
            errs.append(f"{where}: expected object, got {type(value).__name__}")
            return
        _check_obj(value, spec, where, errs)
        return
    elif k == "list":
        if not isinstance(value, list):
            errs.append(f"{where}: expected array, got {type(value).__name__}")
            return
        for i, item in enumerate(value):
            _check(item, spec.item, f"{where}[{i}]", errs)
        return

    if spec.const is not None and value != spec.const:
        errs.append(f"{where}: expected {spec.const!r}, got {value!r}")
    if spec.enum is not None and value not in spec.enum:
        errs.append(f"{where}: {value!r} not in {list(spec.enum)}")


def _check_obj(value: dict, spec: Spec, where: str, errs: list[str]) -> None:
    fields = spec.fields or {}
    if spec.free:
        return  # tool args: shape depends on the tool
    for key, fspec in fields.items():
        if key not in value:
            if fspec.required:
                errs.append(f"{where}.{key}: required field is missing")
            continue
        _check(value[key], fspec, f"{where}.{key}", errs)
    for key in value:
        if key not in fields:
            errs.append(f"{where}.{key}: undeclared field -- document it in schema.py "
                        "or stop writing it")


def validate_record(rec: dict, spec: dict, where: str, errs: list[str]) -> None:
    _check_obj(rec, Obj(spec), where, errs)


def validate_episode(path: Path) -> Report:
    """Every structural and cross-record rule the contract makes."""
    rep = Report(path=str(path))
    e = rep.errors

    try:
        recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        e.append(f"not valid JSONL: {exc}")
        return rep
    if len(recs) < 3:
        e.append(f"expected meta + ticks + episode_end, got {len(recs)} records")
        return rep

    meta, end, ticks = recs[0], recs[-1], recs[1:-1]
    if meta.get("type") != "meta":
        e.append("first record must be `meta`")
        return rep
    if end.get("type") != "episode_end":
        e.append("last record must be `episode_end`")
        return rep

    validate_record(meta, META, "meta", e)
    validate_record(end, EPISODE_END, "episode_end", e)

    n = meta.get("window", {}).get("n_scored")
    for i, t in enumerate(ticks):
        if t.get("type") != "tick":
            e.append(f"record {i + 1}: expected type=tick, got {t.get('type')!r}")
            continue
        validate_record(t, TICK, f"tick[{t.get('t', i)}]", e)

    if not isinstance(n, int):
        return rep  # meta already failed; the rest would be noise

    # --- cross-record invariants --------------------------------------------
    if len(ticks) != n:
        e.append(f"expected {n} tick records, got {len(ticks)}")
    if [t.get("t") for t in ticks] != list(range(len(ticks))):
        e.append("tick indices are not 0..n-1 ascending")

    last = n - 1
    equity = []
    for t in ticks:
        if t.get("type") != "tick":
            continue
        ti = t.get("t")
        equity.append(t.get("equity_cents"))

        p = t.get("obs", {}).get("portfolio", {})
        if p.get("equity_cents") != t.get("equity_cents"):
            e.append(f"tick[{ti}]: equity_cents disagrees with obs.portfolio.equity_cents")
        if p.get("cash_cents") is not None and p.get("position_value_cents") is not None:
            if p["cash_cents"] + p["position_value_cents"] != p["equity_cents"]:
                e.append(f"tick[{ti}]: equity != cash + position_value")

        # the terminal tick and a Wait-skipped tick are BOTH decision_point=false;
        # `skipped_by` is what tells them apart, and the terminal tick is t == n-1
        if t.get("decision_point") is False and "skipped_by" not in t and ti != last:
            e.append(f"tick[{ti}]: non-decision tick with no `skipped_by` and not terminal")
        if t.get("decision_point") is True and "skipped_by" in t:
            e.append(f"tick[{ti}]: decision tick must not carry `skipped_by`")
        if ti == last and t.get("decision_point") is not False:
            e.append(f"tick[{last}]: the terminal tick accepts no action; decision_point "
                     "must be false")

        advanced = [c for c in t.get("calls", []) if c.get("advanced_time")]
        action = t.get("action")
        forced = bool(action and action.get("forced"))
        if t.get("decision_point"):
            if forced:
                # A forced Wait is the engine taking the turn away, not a call the agent
                # made. There is no advancing call to point at, and inventing one would
                # credit the agent with an action it never took.
                if advanced:
                    e.append(f"tick[{ti}]: a forced action must have no advancing call")
                if not action or action.get("tool") != "Wait":
                    e.append(f"tick[{ti}]: a forced action must be a Wait")
            elif len(advanced) != 1:
                e.append(f"tick[{ti}]: {len(advanced)} time-advancing calls (expected 1)")
            elif action:
                # The accepted action IS the call that moved the clock. Anything else means
                # the log is attributing an action the agent did not take.
                if action.get("tool") != advanced[0].get("tool"):
                    e.append(f"tick[{ti}]: action.tool={action.get('tool')!r} but the "
                             f"advancing call was {advanced[0].get('tool')!r}")
                if action.get("args") != advanced[0].get("args"):
                    e.append(f"tick[{ti}]: action.args disagree with the advancing call's")
            if action is None:
                e.append(f"tick[{ti}]: a decision tick must record an action")
        else:
            if advanced:
                e.append(f"tick[{ti}]: a non-decision tick has a time-advancing call")
            if action is not None:
                e.append(f"tick[{ti}]: a non-decision tick must not record an action")

        for c in t.get("calls", []):
            if c.get("ok") is False and "error" not in c:
                e.append(f"tick[{ti}] call {c.get('seq')}: ok=false with no `error`")
            if c.get("ok") is True and "error" in c:
                e.append(f"tick[{ti}] call {c.get('seq')}: ok=true carries an `error`")

        f = t.get("fill")
        if f:
            if f["side"] == "liquidation":
                if f["fill_tick"] != ti:
                    e.append(f"tick[{ti}]: liquidation must fill at tick {ti}, not "
                             f"{f['fill_tick']}")
                if ti != last:
                    e.append(f"tick[{ti}]: liquidation may only occur on the terminal tick")
            elif f["fill_tick"] != ti + 1:
                e.append(f"tick[{ti}]: fill_tick must be {ti + 1} (open of the NEXT bar), "
                         f"got {f['fill_tick']}")

    if equity and equity != end.get("equity_series_cents"):
        e.append("episode_end.equity_series_cents disagrees with the per-tick equity marks")
    if equity and end.get("terminal", {}).get("final_equity_cents") != equity[-1]:
        e.append("terminal.final_equity_cents disagrees with the last equity mark")

    return rep


def validate_run(run_dir: Path) -> list[Report]:
    return [validate_episode(p) for p in sorted(Path(run_dir).glob("episodes/*.jsonl"))]
