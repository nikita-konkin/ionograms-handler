"""Forecasts issued and scored as the soundings come: ``muf sim live``.

The soundings of one circuit come one at a time, in time order. On each,
the loop first scores the forecasts issued for its time, then issues new
ones from it. Nothing a forecast is scored on was known when it was issued:
a forecast is made from one sounding, and the loop sees the soundings in
the order they were made. :func:`step` is one turn of the loop, and
:func:`replay` runs an archive through it as though it came live.

The leads are the circuit's next soundings: one and two of its intervals
ahead (:data:`STEPS`), the interval the median of its last few gaps
(:func:`cadence`), so a five-minute circuit is forecast 5 and 10 min ahead
and a ten-minute one 10 and 20. A forecast whose time passes with no
sounding within the tolerance lapses, and so does one whose sounding has no
trace; a sounding with no trace issues nothing.

A pair is scored as a series scores one (:func:`muf.sim.forecast.score`):
its MUF, LOF and mode counts, each mode's delay, C/N0 and top, and the
working frequencies across the sweep. Of those an event keeps each
frequency's two events and its SNR (:func:`compact`), a few kilobytes where
the whole rows are a hundred; the ionograms are not scored, being for the
eye. The pair also says where the sun stood at its control point. The
running figures (:func:`figures`) are plain sums over the pairs scored, by
lead, with no intervals; :func:`as_run` writes the events out as a series'
run, for ``muf sim summary`` to give those.

A pair across a jump of the receiver's clock is scored like any other, with
the verdict beside its values, and set apart from the figures
(:mod:`muf.sim.clock`). Every sounding says where its echo began against
the ground line, so one with no trace can say where its echo went.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import clock, link, series, summary
from . import forecast as fc
from .state import ChannelState

EVENT_SCHEMA = "muf.sim.live.event"
VERSION = 1

#: The leads, in the circuit's intervals: its next two soundings.
STEPS = (1, 2)

#: Gaps the circuit's interval is the median of.
GAPS = 6

#: Seconds a sounding may miss a forecast's time by and still score it.
TOLERANCE_S = series.TOLERANCE_S

#: Seconds of pairs the running figures cover.
SPAN_S = 3600.0

#: A mode's top, censored where either state's sweep cuts it.
TOP = "f_hi_error_mhz"

#: Decimals an event keeps a working frequency's SNR to, dB: far finer than a
#: sounding resolves it.
SNR_DECIMALS = 2

#: The sides of a forecast's every comparison.
SIDES = ("forecast", "truth")


@dataclass(frozen=True)
class Pending:
    """A forecast issued and waiting for its time, and the sounding it came from."""

    source: Path
    forecast: ChannelState

    @property
    def lead_s(self) -> float:
        return float(self.forecast.metrics["lead_s"])

    def to_dict(self) -> dict:
        return {"source": str(self.source), "forecast": self.forecast.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping) -> Pending:
        return cls(Path(data["source"]), ChannelState.from_dict(data["forecast"]))


@dataclass(frozen=True)
class Ledger:
    """What the loop carries from one sounding to the next: the forecasts
    waiting, and when the circuit's last soundings were."""

    pending: tuple[Pending, ...] = ()
    times: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        return {"pending": [p.to_dict() for p in self.pending],
                "times": [float(t) for t in self.times]}

    @classmethod
    def from_dict(cls, data: Mapping) -> Ledger:
        return cls(tuple(Pending.from_dict(p) for p in data.get("pending", ())),
                   tuple(float(t) for t in data.get("times", ())))


def cadence(times: Sequence[float]) -> float | None:
    """The circuit's interval: the median of its last :data:`GAPS` gaps, None
    before its second sounding."""
    gaps = np.diff(np.asarray(times[-(GAPS + 1):], dtype=float))
    return float(np.median(gaps)) if gaps.size else None


def _utc(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _epoch(utc: str) -> float:
    return _dt.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.timezone.utc).timestamp()


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def event(kind: str, at: float, **fields) -> dict:
    """An event of the loop: its ``kind``, at the epoch of the sounding whose
    turn it was, with its ``fields``."""
    return {"schema": EVENT_SCHEMA, "version": VERSION, "event": kind, "at": _utc(at),
            **fields}


def _waited(p: Pending, kind: str, at: float, **fields) -> dict:
    return event(kind, at, **{"from": p.source.name, "lead_s": p.lead_s,
                               "issued_at": _utc(p.forecast.epoch - p.lead_s),
                               "valid_at": _utc(p.forecast.epoch), **fields})


def _settle(p: Pending, session: series.Session, state: ChannelState | None,
            truth: Callable[[Pending], ChannelState]) -> dict:
    """``p`` scored against the sounding of ``session``, or lapsed and why."""
    at = session.epoch
    if state is None:
        return _waited(p, "lapsed", at, reason="the truth's sounding was not observed")
    try:
        found = truth(p)
    except Exception as exc:
        return _waited(p, "lapsed", at, reason=f"truth not observed: {_failure(exc)}")
    if not found.modes:
        return _waited(p, "lapsed", at, reason="no trace in the truth's sounding")
    score = fc.score(p.forecast, found)
    return _waited(p, "scored", at, to=session.path.name,
                   pair={**score.pair, "zenith_deg": series.zenith_deg(p.forecast)},
                   modes=score.modes, carriers=compact(score.carriers),
                   clock=clock.verdict(p.forecast, state))


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def compact(rows: Sequence[Mapping]) -> dict:
    """A pair's working frequencies (:func:`muf.sim.forecast.score`) as an
    event keeps them, in columns: each frequency, its ratio to the truth's
    MUF, its two events on both sides, and its SNR with the error."""
    return {"f0_mhz": [float(r["f0_mhz"]) for r in rows],
            "f_over_muf": [_rounded(r["f_over_muf"], 4) for r in rows],
            **{event: {side: [bool(r["events"][event][side]) for r in rows]
                       for side in SIDES} for event in fc.EVENTS},
            "snr_db": {side: [_rounded(r["figures"]["snr_db"][side], SNR_DECIMALS)
                              for r in rows] for side in (*SIDES, "error")}}


def carrier_rows(e: Mapping) -> list[dict]:
    """A scored event's working frequencies as :mod:`muf.sim.forecast` writes
    them, the figures an event does not keep empty; none for an event from
    before they were kept."""
    kept, pair = e.get("carriers"), e["pair"]
    if not kept:
        return []
    snr = kept["snr_db"]
    return [{"schema": fc.CARRIER_SCHEMA, "version": fc.VERSION,
             "from": pair["from"], "to": pair["to"], "f0_mhz": f0,
             "f_over_muf": kept["f_over_muf"][k],
             "events": {event: {side: kept[event][side][k] for side in SIDES}
                        for event in fc.EVENTS},
             "strongest": dict.fromkeys(SIDES),
             "figures": {name: ({side: snr[side][k] for side in (*SIDES, "error")}
                                if name == "snr_db"
                                else dict.fromkeys((*SIDES, "error")))
                         for name in link.FIGURES},
             "excluded": []}
            for k, f0 in enumerate(kept["f0_mhz"])]


def _line(row: Mapping) -> str:
    return json.dumps(row, allow_nan=False) + "\n"


def as_run(events: Iterable[Mapping], out: str | Path, *, lead_s: float,
           tolerance_s: float = TOLERANCE_S) -> int:
    """The forecasts of ``lead_s`` among ``events`` written to ``out`` as a
    series writes its run (:mod:`muf.sim.series`), for
    :func:`muf.sim.summary.summarise` to sum up as it sums a period of the
    archive.

    A scored forecast is a pair, with its modes, its working frequencies
    (:func:`carrier_rows`) and its clock verdict on the pair, where a series
    puts it; a lapsed one is a pair that could not be scored. A lead within
    ``tolerance_s`` of ``lead_s`` is taken for it, a circuit's interval being
    the median of its gaps. Returns how many pairs.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    params = {"lead_s": float(lead_s), "tolerance_s": float(tolerance_s),
              "bandwidth_hz": link.BANDWIDTH_HZ, "snr_min_db": link.SNR_MIN_DB,
              "carriers": None, "source": "live"}
    (out / "run.json").write_text(json.dumps({"params": params}, indent=1) + "\n",
                                  encoding="utf-8")
    pairs = 0
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as found, \
            (out / "modes.jsonl").open("w", encoding="utf-8") as modes, \
            (out / "carriers.jsonl").open("w", encoding="utf-8") as carriers, \
            (out / "skipped.jsonl").open("w", encoding="utf-8") as skipped:
        for e in events:
            if (e.get("lead_s") is None
                    or abs(float(e["lead_s"]) - lead_s) > tolerance_s):
                continue
            if e["event"] == "scored":
                found.write(_line({**e["pair"], "clock": e.get("clock")}))
                modes.writelines(_line(row) for row in e["modes"])
                carriers.writelines(_line(row) for row in carrier_rows(e))
                pairs += 1
            elif e["event"] == "lapsed":
                skipped.write(_line(series._skipped(e["from"], None, "pair",
                                                    e["reason"])))
    return pairs


def step(ledger: Ledger, session: series.Session, state: ChannelState | None,
         truth: Callable[[Pending], ChannelState], *,
         leads: Sequence[float] | None = None,
         tolerance_s: float = TOLERANCE_S) -> tuple[Ledger, list[dict]]:
    """The loop's turn on the sounding of ``session``.

    ``state`` is the sounding's own, None when it could not be observed;
    ``truth`` gives the state a pending forecast is scored against
    (:func:`muf.sim.series.truth_of`). ``leads``, seconds, replace the
    circuit's intervals. Returns the ledger after the turn, and its events:
    the forecasts due settled, then those issued.
    """
    at = session.epoch
    events, waiting = [], []
    for p in ledger.pending:
        if abs(p.forecast.epoch - at) <= tolerance_s:
            events.append(_settle(p, session, state, truth))
        elif p.forecast.epoch + tolerance_s < at:
            events.append(_waited(p, "lapsed", at, reason=(
                f"no sounding within {tolerance_s:g} s of {_utc(p.forecast.epoch)}")))
        else:
            waiting.append(p)
    times = (*ledger.times, at)[-(GAPS + 1):]
    if state is not None and not state.modes:
        events.append(event("untraced", at, **{
            "from": session.path.name, "echo_offset_us": clock.offset_us(state)}))
    elif state is not None:
        interval = cadence(times)
        ahead = (list(leads) if leads is not None
                 else [] if interval is None else [k * interval for k in STEPS])
        muf = state.metrics.get("muf_mhz", state.muf_mhz)
        muf = float(muf) if muf is not None and math.isfinite(muf) else None
        for lead in ahead:
            p = Pending(session.path, fc.persist(state, float(lead)))
            waiting.append(p)
            events.append(event("issued", at, **{
                "from": session.path.name, "lead_s": float(lead),
                "valid_at": _utc(p.forecast.epoch),
                "muf_mhz": muf,
                "modes": len(state.modes),
                "echo_offset_us": clock.offset_us(state)}))
    return Ledger(tuple(waiting), times), events


def take(ledger: Ledger, session: series.Session, cache: str | Path, *,
         params: Mapping | None = None, stations: Mapping | None = None,
         stations_id: str | None = None, leads: Sequence[float] | None = None,
         tolerance_s: float = TOLERANCE_S) -> tuple[Ledger, list[dict]]:
    """:func:`step` on the sounding of ``session``, read from its file.

    Its state is observed into ``cache`` (:func:`muf.sim.series.state_of`),
    and each forecast due is scored against it as
    :func:`muf.sim.series.truth_of` gives it; a sounding that cannot be
    observed is an ``unobserved`` event. ``params`` are the observing
    parameters (:data:`muf.sim.series.OBSERVE`).
    """
    events = []
    try:
        state = series.state_of(session.path, cache, params, stations=stations,
                                stations_id=stations_id)
    except Exception as exc:
        state = None
        events.append(event("unobserved", session.epoch,
                             **{"from": session.path.name, "reason": _failure(exc)}))
    ledger, turn = step(ledger, session, state,
                        lambda p: series.truth_of(p.source, session.path, cache, params,
                                                  stations=stations),
                        leads=leads, tolerance_s=tolerance_s)
    return ledger, events + turn


def _summed(found: list[tuple[int, float | None, float]], n: int,
            key: str) -> float | None:
    """``key`` of :func:`muf.sim.summary.metrics` over (pair, error, weight)
    triples of ``n`` pairs, with no intervals; None with nothing to sum."""
    if not found:
        return None
    pairs, errors, weights = zip(*found)
    return summary.metrics([np.nan if e is None else e for e in errors], weights,
                           pairs, np.zeros((0, n)), minimum=1,
                           quantiles=False).get(key)


def _uncensored(rows: list[dict], key: str) -> list[tuple[int, float | None, float]]:
    return [(k, r["pair"][key]["error"], 1.0) for k, r in enumerate(rows)
            if not (r["pair"][key]["censored_truth"]
                    or r["pair"][key]["censored_forecast"])]


def _weighed(modes: list[tuple[int, dict]], key: str
             ) -> list[tuple[int, float | None, float]]:
    return [(k, m[key], m["overlap_mhz"] or 0.0) for k, m in modes]


def _figures(rows: list[dict]) -> dict:
    n = len(rows)
    modes = [(k, m) for k, r in enumerate(rows) for m in r["modes"]
             if m["status"] == "matched"]
    tops = [(k, m) for k, m in modes
            if not (m["f_hi_censored_truth"] or m["f_hi_censored_forecast"])]
    return {"n": n, "muf_mae_mhz": _summed(_uncensored(rows, "muf_mhz"), n, "mae"),
            "lof_mae_mhz": _summed(_uncensored(rows, "lof_mhz"), n, "mae"),
            "modes": {k: sum(r["pair"]["modes"][k] for r in rows)
                      for k in ("matched", "missed", "false")},
            "delay_rms_us": _summed(_weighed(modes, "delay_rms_us"), n, "rmse"),
            "cn0_rms_db": _summed(_weighed(modes, "cn0_rms_db"), n, "rmse"),
            "top_median_mhz": _summed(_weighed(tops, TOP), n, "median")}


def figures(events: Iterable[dict], *, since: float | None = None,
            suspect: bool = False) -> dict[float, dict]:
    """The scored pairs' figures from ``since`` on, by lead.

    n; the MUF's and LOF's mean |error| over the pairs censored on neither
    side; the modes matched, missed and false; the RMS of the matched modes'
    delay and C/N0 errors; and the median |error| of their tops, a top
    censored on either side left out. A mode weighs by the band it shares
    with its pair, as in ``muf sim summary``: these are its
    :func:`muf.sim.summary.metrics`, with no intervals.

    The clock-suspect pairs are left out and counted, ``set_apart``;
    ``suspect`` keeps them in, as one pair's own figures want.
    """
    by: dict[float, list[dict]] = {}
    for e in events:
        if e["event"] == "scored" and (since is None or _epoch(e["at"]) >= since):
            by.setdefault(float(e["lead_s"]), []).append(e)
    return {lead: {**_figures([r for r in rows if suspect or not clock.suspect(r)]),
                   "set_apart": 0 if suspect else sum(map(clock.suspect, rows))}
            for lead, rows in sorted(by.items())}


def _ahead(lead_s: float) -> str:
    return f"+{lead_s / 60:g} min"


def _clock(utc: str) -> str:
    return utc[11:19]


def _number(value: float | None, spec: str, unit: str) -> str:
    return "-" if value is None else f"{value:{spec}} {unit}"


def _error(bounded: dict) -> str:
    if bounded["error"] is None:
        return "-"
    censored = bounded["censored_truth"] or bounded["censored_forecast"]
    return f"{bounded['error']:+.2f} MHz{' (a bound)' if censored else ''}"


def format_event(e: dict) -> str:
    """One line for a terminal."""
    at, kind = _clock(e["at"]), e["event"]
    if kind == "issued":
        return (f"{at}  issued  {_ahead(e['lead_s'])} for {_clock(e['valid_at'])}: "
                f"MUF {_number(e['muf_mhz'], '.2f', 'MHz')}, {e['modes']} modes")
    if kind == "scored":
        pair, counts = e["pair"], e["pair"]["modes"]
        delay = _summed(_weighed([(0, m) for m in e["modes"]
                                  if m["status"] == "matched"], "delay_rms_us"),
                        1, "rmse")
        return (f"{at}  scored  {_ahead(e['lead_s'])} from {_clock(e['issued_at'])}: "
                f"MUF {_error(pair['muf_mhz'])}, LOF {_error(pair['lof_mhz'])}, "
                f"modes {counts['matched']} matched, {counts['missed']} missed, "
                f"{counts['false']} false, delay {_number(delay, '.0f', 'us')} RMS"
                + (f" -- clock-suspect: {e['clock']['why']}" if clock.suspect(e)
                   else ""))
    if kind == "lapsed":
        return (f"{at}  lapsed  {_ahead(e['lead_s'])} from {_clock(e['issued_at'])}: "
                f"{e['reason']}")
    if kind == "untraced":
        offset = e.get("echo_offset_us")
        return (f"{at}  no trace in {e['from']}, so nothing issued"
                + ("" if offset is None else
                   f"; its echo begins {offset / 1e3:+.2f} ms from the ground line"))
    return f"{at}  {e['from']} not observed: {e['reason']}"


def format_figures(found: Mapping[float, dict], label: str) -> str:
    """The running figures, a line a lead."""
    if not found:
        return f"          {label}: nothing scored"
    return "\n".join(
        f"          {label} {_ahead(lead)}: n {f['n']}, "
        f"MUF {_number(f['muf_mae_mhz'], '.2f', 'MHz')}, "
        f"LOF {_number(f['lof_mae_mhz'], '.2f', 'MHz')}, "
        f"delay {_number(f['delay_rms_us'], '.0f', 'us')}, "
        f"C/N0 {_number(f['cn0_rms_db'], '.1f', 'dB')}, "
        f"top {_number(f['top_median_mhz'], '.2f', 'MHz')}, modes "
        f"{f['modes']['matched']}/{f['modes']['missed']}/{f['modes']['false']}"
        + (f"; {f['set_apart']} set apart, clock-suspect" if f.get("set_apart")
           else "")
        for lead, f in found.items())


def replay(paths: Iterable, *, circuit: str | None = None,
           leads: Sequence[float] | None = None, tolerance_s: float = TOLERANCE_S,
           pace: float = 0.0, span_s: float = SPAN_S, out: str | Path | None = None,
           params: Mapping | None = None, stations: Mapping | None = None,
           stations_id: str | None = None, emit: Callable[[str], None] = print,
           sleep: Callable[[float], None] = time.sleep) -> dict[float, dict]:
    """Run the soundings under ``paths`` through the loop as though they came live.

    ``pace`` is how much faster than real time: 0 as fast as it goes, 1 at
    the soundings' own pace, 60 an hour a minute. Each event goes to
    ``emit`` as a line, and to ``out``, a JSON Lines file begun afresh, when
    given; after a sounding that scored, the last ``span_s``'s figures
    follow. ``params`` are the observing parameters
    (:data:`muf.sim.series.OBSERVE`). Returns the whole replay's figures.
    """
    found, unreadable = series.sessions(paths, stations=stations)
    for row in unreadable:
        emit(f"skipped {row['from']}: {row['reason']}")
    log: list[dict] = []
    ledger, before = Ledger(), None
    if out is not None:
        Path(out).write_text("", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="muf-live-") as cache:
        for s in series.circuit_of(found, circuit):
            if pace > 0 and before is not None:
                sleep((s.epoch - before) / pace)
            before = s.epoch
            ledger, events = take(ledger, s, cache, params=params, stations=stations,
                                  stations_id=stations_id, leads=leads,
                                  tolerance_s=tolerance_s)
            if out is not None:
                with Path(out).open("a", encoding="utf-8") as handle:
                    handle.writelines(json.dumps(e, allow_nan=False) + "\n"
                                      for e in events)
            for e in events:
                emit(format_event(e))
            log.extend(events)
            if any(e["event"] == "scored" for e in events):
                emit(format_figures(figures(log, since=s.epoch - span_s),
                                    f"last {span_s / 60:g} min"))
    whole = figures(log)
    emit(format_figures(whole, "whole replay"))
    return whole
