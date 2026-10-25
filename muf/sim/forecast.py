"""Forecasting the channel state, and scoring a forecast against the truth.

The first forecast is the naive one, **persistence**: the state a sounding
gives at t is issued as the forecast for t + L. It is the baseline any other
forecast has to beat, and its error is how much the channel changes in L --
for a modem, how often the channel has to be estimated again.

A forecast is a :class:`~muf.sim.state.ChannelState`, as a measurement is,
so every forecast is scored by the same code on the same pairs -- the rule
``services/prediction/scoring.py`` keeps for its baselines. The truth is the
state the sounding at t + L gives, observed the same way. An error is the
forecast less the truth throughout: positive is a forecast too high.

Modes are paired by delay, never by name, since the trace's tracking renames
them from one sounding to the next, and in both directions
(:func:`muf.sim.check.match`): a truth mode no forecast mode pairs with is
**missed** (it appeared), a forecast mode no truth mode pairs with is
**false** (it went). Only paired modes give continuous errors.

A value is scored only where both states measured it, or derived it from
measured values. A persisted prior is the same prior again, its error zero
by construction and meaningless; such fields are counted, not scored.

A MUF at the top of the band is a lower bound and an LOF at its floor an
upper bound (:func:`muf.pipeline.band_edge_mhz`,
:func:`muf.pipeline.lof_edge_mhz`); so is a MUF anywhere in a recording cut
short of its sweep, as the SAO export has it -- the trace can fade before
the recording stops. Either is flagged, and a censored truth also gets the
one-sided error it does bound.
"""

from __future__ import annotations

import copy
import datetime as _dt
import itertools
import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np

from .. import spectro
from . import check, link, observe, sounding
from .state import (
    DERIVED,
    FORECAST,
    MEASURED,
    MODE_FIELDS,
    ChannelState,
    Mode,
    Provenance,
)

PAIR_SCHEMA = "muf.sim.forecast.pair"
MODE_SCHEMA = "muf.sim.forecast.mode"
CARRIER_SCHEMA = "muf.sim.forecast.carrier"
IMAGE_SCHEMA = "muf.sim.forecast.ionogram"
VERSION = 1

#: Share of the truth's range, and of its band, its ionogram and the
#: forecast's must have in common to be compared: an echo window follows the
#: echo, and a recording can stop short of its sweep, so two ionograms can
#: overlap by a sliver.
MIN_COMMON = 0.8

#: How a persistence's note begins; what the value was before follows.
PERSISTENCE = "persistence of "

#: What a persistence's note says of a field that had no provenance.
UNRECORDED = "nothing recorded"

#: Origins a value is scored on. A forecast other than a persistence rests on
#: whatever its model does, and is scored as it stands.
SCORED = (MEASURED, DERIVED, FORECAST)

#: The mode errors each provenance field stands behind.
ERRORS_OF_FIELD = {
    "delay": ("delay_bias_us", "delay_rms_us", "gdd_bias_us_per_mhz",
              "gdd_rms_us_per_mhz", "gdd_near_muf_error_us_per_mhz"),
    "cn0": ("cn0_bias_db", "cn0_rms_db"),
    "band": ("f_lo_error_mhz", "f_hi_error_mhz"),
    "delay_spread": ("delay_spread_error_us",),
}

#: Mode errors read off the metrics each state carries, by mode name: the
#: trace's width (:func:`muf.sim.observe.cross_trace`) and the scatter of the
#: level along it (:func:`muf.sim.observe.level_statistics`). Present in a
#: state only where measured.
METRIC_ERRORS = {"width_error_us": ("cross_trace", "spread_us"),
                 "level_std_error_db": ("levels", "std_db")}

MODE_ERRORS = (*(k for keys in ERRORS_OF_FIELD.values() for k in keys),
               *METRIC_ERRORS)

#: The GDD is also compared at this fraction of the truth mode's top, where
#: it is steep and a wideband signal feels it most.
NEAR_MUF = 0.9

#: The mode fields each carrier figure (:data:`muf.sim.link.FIGURES`) rests
#: on, in every mode the link sees. Not the modes' own delay spreads: a prior
#: one is zero, and the delay spread and coherence bandwidth are then the
#: spread between the modes, which is measured.
CARRIER_FIELDS = {
    "modes": ("cn0",), "first_delay_us": ("delay", "cn0"),
    "multipath_us": ("delay", "cn0"), "delay_spread_us": ("delay", "cn0"),
    "coherence_bandwidth_khz": ("delay", "cn0"), "gdd_us_per_mhz": ("delay", "cn0"),
    "cn0_dbhz": ("cn0",), "snr_db": ("cn0",),
}

#: The same for figures of the strongest mode alone.
STRONGEST_FIELDS = {"coherence_time_s": ("doppler_spread",)}

#: With one mode seen there is no spread between modes, and the delay spread
#: is that mode's own: it rests on its delay spread too, else every such
#: carrier would score a prior of zero against a prior of zero.
ONE_PATH_FIELDS = {"delay_spread_us": ("delay_spread",)}

#: The events a forecast at a carrier is scored on, by the link's flag.
EVENTS = {"mode": "has_mode", "usable": "usable"}


def _carried(item: Provenance | None, source: str, lead_s: float) -> Provenance:
    was = UNRECORDED if item is None else str(item)
    return Provenance(FORECAST, f"{PERSISTENCE}{was} from {source or 'no source'}, "
                                f"lead {lead_s:g} s")


def persist(state: ChannelState, lead_s: float) -> ChannelState:
    """``state`` issued as the forecast for ``lead_s`` seconds after it.

    A deep copy with the same curves and values, the epoch moved on, the lead
    in ``metrics["lead_s"]``, and every field's origin ``forecast`` -- each
    mode's :data:`~muf.sim.state.MODE_FIELDS` and the state's own -- with
    what it was before in the note, where :func:`basis` reads it back.
    """
    if not (math.isfinite(lead_s) and lead_s > 0):
        raise ValueError(f"a lead of {lead_s} s is not ahead of the sounding")
    out = copy.deepcopy(state)
    out.epoch = state.epoch + lead_s
    out.metrics["lead_s"] = float(lead_s)
    for m in out.modes:
        m.provenance = {key: _carried(m.provenance.get(key), state.source, lead_s)
                        for key in dict.fromkeys([*MODE_FIELDS, *m.provenance])}
    out.provenance = {key: _carried(state.provenance.get(key), state.source, lead_s)
                      for key in dict.fromkeys(["epoch", *state.provenance])}
    return out


def basis(item: Provenance | None) -> str:
    """What a value rests on: its origin, or a persistence's origin before.

    A persistence of a persistence is followed to the first. '' when nothing
    was recorded.
    """
    if item is None:
        return ""
    if item.origin != FORECAST or not item.note.startswith(PERSISTENCE):
        return item.origin
    was = item.note[len(PERSISTENCE):].rsplit(" from ", 1)[0]
    return "" if was == UNRECORDED else basis(Provenance.from_dict(was))


@dataclass
class PairScore:
    """One forecast against its truth: the pair's row, and its modes' and carriers'.

    ``pair`` is a line of ``pairs.jsonl``, group A; ``modes`` are lines of
    ``modes.jsonl``, group B -- the paired modes, then the missed and the
    false ones with no errors; ``carriers`` are lines of ``carriers.jsonl``,
    group G, one per working frequency; ``ionogram`` is a line of
    ``ionogram.jsonl``, group D, when the ionograms were compared.
    """

    pair: dict
    modes: list[dict]
    carriers: list[dict] = field(default_factory=list)
    ionogram: dict | None = None

    def to_dict(self) -> dict:
        out = {**self.pair, "mode_pairs": self.modes, "carrier_rows": self.carriers}
        if self.ionogram is not None:
            out["ionogram"] = self.ionogram
        return out


def contingency(outcomes: Iterable[tuple[bool, bool]]) -> dict:
    """(forecast, truth) outcomes of an event as a 2x2 table, and its scores.

    POD = a/(a+c), FAR = b/(a+b), CSI = a/(a+b+c), accuracy = (a+d)/n, with
    a hits, b false alarms, c misses and d correct negatives; and the base
    rate, (a+c)/n, how often the truth had the event, without which POD and
    FAR cannot be read. A score whose denominator is zero is None.
    """
    a = b = c = d = 0
    for forecast_said, truth_had in outcomes:
        if forecast_said and truth_had:
            a += 1
        elif forecast_said:
            b += 1
        elif truth_had:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    return {"hits": a, "false_alarms": b, "misses": c, "correct_negatives": d,
            "pod": a / (a + c) if a + c else None,
            "far": b / (a + b) if a + b else None,
            "csi": a / (a + b + c) if a + b + c else None,
            "accuracy": (a + d) / n if n else None,
            "base_rate": (a + c) / n if n else None}


def _finite(value) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _beyond(value: float | None, state: ChannelState, *, above: bool) -> bool:
    """Whether ``value``, a top (``above``) or a floor, is a bound in
    ``state`` rather than a measurement: past its band's edge or, for a top,
    anywhere in a sweep recorded short, as the SAO export has it."""
    edge = state.metrics.get("band_edge_mhz" if above else "lof_edge_mhz")
    if value is None or edge is None:
        return False
    if above and state.metrics.get("sweep_complete", 1.0) == 0.0:
        return True
    return value >= edge if above else value <= edge


def _bounded(forecast_value, truth_value, forecast: ChannelState,
             truth: ChannelState, *, above: bool) -> dict:
    """A MUF (``above``) or an LOF, censored against each state's own edge.

    A censored truth bounds the error one way only: a MUF at the band's top
    is at least that high, so a forecast under it is wrong by at least the
    difference and one over it may be right.
    """
    f, t = _finite(forecast_value), _finite(truth_value)
    error = None if f is None or t is None else f - t
    censored = _beyond(t, truth, above=above)
    return {"forecast": f, "truth": t, "error": error,
            "censored_truth": censored,
            "censored_forecast": _beyond(f, forecast, above=above),
            "error_bound": (max(0.0, -error if above else error)
                            if censored and error is not None else None)}


def _lof(state: ChannelState) -> float | None:
    return min((m.f_lo for m in state.modes), default=None)


def _assumed(f: Mode, t: Mode) -> list[str]:
    """The mode fields a pair cannot be scored on: assumed in either state."""
    return [key for key in MODE_FIELDS
            if basis(f.provenance.get(key)) not in SCORED
            or basis(t.provenance.get(key)) not in SCORED]


def _metric_error(forecast: ChannelState, truth: ChannelState, f: Mode, t: Mode,
                  table: str, key: str) -> float | None:
    a = (forecast.metrics.get(table) or {}).get(f.name)
    b = (truth.metrics.get(table) or {}).get(t.name)
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return None
    return _finite(float(a[key]) - float(b[key]))


def strength(mode: Mode) -> float:
    """A mode's power over its band: linear C/N0 integrated over frequency.

    What makes a mode the main one. A median would pick a short, strong
    trace near the nose over a long mode that carries more power across
    its band.
    """
    power = 10.0 ** (mode.cn0_dbhz / 10.0)
    return float(np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(mode.freq_mhz)))


def _class(mode: Mode, other: Mode | None, main: Mode | None) -> str:
    """``high``: a high ray on either side; ``main``: else the truth's strongest.

    The high ray first, because the tracking need not label it in both
    soundings: one sounding's strongest mode can be an unlabelled "F" that
    pairs with the other's "1F-high", and a high ray's errors in the main
    class would swamp it.
    """
    if "high" in (mode.branch, other.branch if other is not None else None):
        return "high"
    if mode is main:
        return "main"
    return "other"


def _row(forecast: ChannelState, truth: ChannelState, status: str,
         f: Mode | None, t: Mode | None, cls: str) -> dict:
    return {"schema": MODE_SCHEMA, "version": VERSION,
            "from": forecast.source, "to": truth.source, "status": status,
            "forecast_mode": f.name if f is not None else None,
            "truth_mode": t.name if t is not None else None, "class": cls,
            "overlap_mhz": None, **dict.fromkeys(MODE_ERRORS),
            "f_hi_censored_truth": False, "f_hi_censored_forecast": False,
            "excluded": []}


def _paired(e: check.ModeError, forecast: ChannelState, truth: ChannelState,
            main: Mode | None) -> dict:
    f, t = forecast.mode(e.matched), truth.mode(e.name)
    near = NEAR_MUF * t.f_hi
    errors = {
        "delay_bias_us": e.delay_bias_us, "delay_rms_us": e.delay_rms_us,
        "gdd_bias_us_per_mhz": e.gdd_bias_us_per_mhz,
        "gdd_rms_us_per_mhz": e.gdd_rms_us_per_mhz,
        "gdd_near_muf_error_us_per_mhz": float(f.gdd_at(near) - t.gdd_at(near)),
        "cn0_bias_db": e.cn0_bias_db, "cn0_rms_db": e.cn0_rms_db,
        "f_lo_error_mhz": e.f_lo_error_mhz, "f_hi_error_mhz": e.f_hi_error_mhz,
        "delay_spread_error_us": f.delay_spread_us - t.delay_spread_us,
    }
    excluded = _assumed(f, t)
    for key in excluded:
        for name in ERRORS_OF_FIELD.get(key, ()):
            errors[name] = None
    row = _row(forecast, truth, "matched", f, t, _class(t, f, main))
    row.update({k: _finite(v) for k, v in errors.items()})
    row.update({k: _metric_error(forecast, truth, f, t, *where)
                for k, where in METRIC_ERRORS.items()})
    row.update(overlap_mhz=float(e.overlap_mhz),
               f_hi_censored_truth=_beyond(t.f_hi, truth, above=True),
               f_hi_censored_forecast=_beyond(f.f_hi, forecast, above=True),
               excluded=excluded)
    return row


def _assumed_at(state: ChannelState, figures: dict) -> set[str]:
    """Carrier figures ``state`` assumes: a field they rest on assumed in a mode."""
    seen = [m.mode for m in link.present(state, figures["f0_mhz"])]
    strongest = [state.mode(figures["strongest"])] if figures["strongest"] else []
    alone = seen if len(seen) == 1 else []
    out = set()
    for modes, rests in ((seen, CARRIER_FIELDS), (strongest, STRONGEST_FIELDS),
                         (alone, ONE_PATH_FIELDS)):
        out.update(name for name, keys in rests.items()
                   if any(basis(m.provenance.get(k)) not in SCORED
                          for m in modes for k in keys))
    return out


def _observed(state: ChannelState, f0_mhz: float) -> bool:
    """Whether ``state`` measured the channel at ``f0_mhz``: inside the
    edges its LOF and MUF are measurements within, where it has them -- a
    recording cut short has nothing to say past where it stopped."""
    lo, hi = state.metrics.get("lof_edge_mhz"), state.metrics.get("band_edge_mhz")
    return (lo is None or f0_mhz > lo) and (hi is None or f0_mhz < hi)


def _carrier(forecast: ChannelState, truth: ChannelState, f0_mhz: float,
             muf_mhz: float | None, options: dict) -> dict:
    """One working frequency: both links, the events, and the figures' errors.

    A figure's error needs a mode seen on both sides, and the figure measured
    on both.
    """
    f = link.at(forecast, f0_mhz, **options)
    t = link.at(truth, f0_mhz, **options)
    excluded = _assumed_at(forecast, f) | _assumed_at(truth, t)
    both = f["has_mode"] and t["has_mode"]
    figures = {}
    for name in link.FIGURES:
        a, b = f[name], t[name]
        scored = both and name not in excluded and a is not None and b is not None
        figures[name] = {"forecast": a, "truth": b, "error": a - b if scored else None}
    return {"schema": CARRIER_SCHEMA, "version": VERSION,
            "from": forecast.source, "to": truth.source, "f0_mhz": float(f0_mhz),
            "f_over_muf": f0_mhz / muf_mhz if muf_mhz else None,
            "events": {event: {"forecast": f[flag], "truth": t[flag]}
                       for event, flag in EVENTS.items()},
            "strongest": {"forecast": f["strongest"], "truth": t["strongest"]},
            "figures": figures,
            "excluded": [name for name in link.FIGURES if name in excluded]}


def score(forecast: ChannelState, truth: ChannelState, *,
          name: str = "persistence", carriers: Sequence[float] | None = None,
          bandwidth_hz: float = link.BANDWIDTH_HZ,
          snr_min_db: float = link.SNR_MIN_DB) -> PairScore:
    """Groups A, B and G: ``forecast`` against ``truth``, two states in hand.

    ``name`` is the forecast's kind, as the pair's row records it.
    ``carriers`` are the working frequencies, MHz, by default every
    :data:`muf.sim.link.CARRIER_STEP_MHZ` across the forecast's sweep; the
    link at each reads its SNR in ``bandwidth_hz`` and is usable from
    ``snr_min_db``. A carrier outside either state's band (:func:`_observed`)
    was never measured there, so it is counted and left out.
    """
    sweep = forecast.sweep or truth.sweep
    if carriers is None:
        carriers = ([] if sweep is None
                    else link.carrier_grid(sweep.start_mhz, sweep.stop_mhz))
    seen = [float(f0) for f0 in carriers
            if _observed(forecast, f0) and _observed(truth, f0)]
    muf = _finite(truth.metrics.get("muf_mhz", truth.muf_mhz))
    options = {"bandwidth_hz": bandwidth_hz, "snr_min_db": snr_min_db}
    at_carriers = [_carrier(forecast, truth, f0, muf, options) for f0 in seen]
    forward = check.match(truth, forecast, both_strong=True)
    backward = check.match(forecast, truth, both_strong=True)
    main = max(truth.modes, key=strength, default=None)
    rows = []
    for e in forward:
        if e.matched is None:
            t = truth.mode(e.name)
            rows.append(_row(forecast, truth, "missed", None, t, _class(t, None, main)))
        else:
            rows.append(_paired(e, forecast, truth, main))
    for e in backward:
        if e.matched is None:
            f = forecast.mode(e.name)
            rows.append(_row(forecast, truth, "false", f, None, _class(f, None, None)))
    matched = sum(1 for e in forward if e.matched is not None)
    lead = forecast.metrics.get("lead_s")
    issued = None if lead is None else forecast.epoch - float(lead)
    pair = {
        "schema": PAIR_SCHEMA, "version": VERSION, "forecast": name,
        "circuit": {"tx": truth.circuit.tx, "rx": truth.circuit.rx},
        "issued_at": _utc(issued), "valid_at": _utc(forecast.epoch),
        "lead_s": _finite(lead),
        "lead_actual_s": None if issued is None else float(truth.epoch - issued),
        "from": forecast.source, "to": truth.source,
        "muf_mhz": _bounded(forecast.metrics.get("muf_mhz"),
                            truth.metrics.get("muf_mhz"), forecast, truth, above=True),
        "mode_muf_mhz": _bounded(forecast.muf_mhz, truth.muf_mhz, forecast, truth,
                                 above=True),
        "lof_mhz": _bounded(_lof(forecast), _lof(truth), forecast, truth, above=False),
        "modes": {
            "forecast": len(forecast.modes), "truth": len(truth.modes),
            "matched": matched, "missed": len(forward) - matched,
            "false": sum(1 for e in backward if e.matched is None),
            "pairings_agree": ({(e.matched, e.name) for e in forward if e.matched}
                               == {(e.name, e.matched) for e in backward if e.matched}),
        },
        "excluded_prior_fields": sum(len(r["excluded"]) for r in rows),
        "carriers": {
            "n": len(at_carriers), "outside_band": len(carriers) - len(seen),
            "bandwidth_hz": float(bandwidth_hz),
            "snr_min_db": float(snr_min_db),
            "events": {event: contingency((r["events"][event]["forecast"],
                                           r["events"][event]["truth"])
                                          for r in at_carriers)
                       for event in EVENTS},
        },
    }
    return PairScore(pair, rows, at_carriers)


def sounded(state: ChannelState, copies: int, *, seed: int = 0,
            gate_km: tuple[float, float] | None = None,
            workdir: str | Path | None = None) -> list:
    """``state``'s synthetic soundings, seeds ``seed`` on, as ionograms.

    Each 80 MB ``.lfs`` is written, loaded at ``gate_km`` and deleted at
    once, so a run of them holds one on disk at a time.
    """
    own = workdir is None
    folder = Path(tempfile.mkdtemp(prefix="muf-forecast-")) if own else Path(workdir)
    out = []
    try:
        for k in range(copies):
            path = folder / sounding.default_name(state, tag=f"copy{k:02d}")
            sounding.write_lfs(state, path, seed=seed + k)
            try:
                out.append(observe.load(path, gate_km=gate_km))
            finally:
                path.unlink()
    finally:
        if own:
            shutil.rmtree(folder, ignore_errors=True)
    return out


def common_span(forecast: tuple[float, float],
                truth: tuple[float, float]) -> tuple[float, float] | None:
    """The span of range or frequency two ionograms share, or None under
    :data:`MIN_COMMON` of the truth's."""
    lo, hi = max(forecast[0], truth[0]), min(forecast[1], truth[1])
    if hi - lo < MIN_COMMON * (truth[1] - truth[0]):
        return None
    return lo, hi


def on_band(ion: spectro.Ionogram, band: tuple[float, float]) -> spectro.Ionogram:
    """``ion``'s rows within ``band``, MHz, half a bin either side."""
    half = 0.5 * ion.cal.freq_step_mhz
    rows = np.flatnonzero((ion.freq >= band[0] - half) & (ion.freq <= band[1] + half))
    cal = replace(ion.cal, freq=ion.freq[rows], freq_stop=float(ion.freq[rows[-1]]))
    return replace(ion, power=ion.power[rows], cal=cal, _db=None)


def _z(r: float) -> float:
    return float(np.arctanh(r))


def image(a, b, copies: list, modes: Sequence[Mode]) -> dict:
    """Group D on ionograms on one grid: the forecast's sounding ``a`` and its
    state's ``copies`` against the truth's sounding ``b``.

    Compared in dB within :data:`muf.sim.check.CORRELATION_HALF_BINS` of
    ``modes``' traces -- the forecast's and the truth's together, so that a
    mode that appeared is in the comparison -- less every cell the cleaning
    took out of any of them, and every cell a map has no value for, as where
    a recording cut short ends in a row of NaN; on each of
    :data:`muf.sim.check.CORRELATION_MAPS`:

    * ``ion``: ``a`` against ``b``, the persistence of the ionogram itself;
    * ``state``: each copy against ``b``, averaged in Fisher's z;
    * ``expected``: the copies' mean linear power against ``b``, the forecast
      with the fading averaged out;
    * ``ceiling``: the copies against each other, as like ``b`` as one state
      can make a sounding;
    * ``dz_ceiling`` = z(ceiling) - z(state), how far the state falls short
      of it, and ``dz_ion`` = z(state) - z(ion), at or over zero where the
      state five minutes old does as well as the ionogram five minutes old.

    From :data:`muf.sim.check.REALISM_SOUNDINGS` copies, check's level 2
    verdict on ``b`` as one more copy. And the level: the expected
    ionogram's dB less ``b``'s on the smoothed map.
    """
    if any(not check.same_grid(b, ion) for ion in (a, *copies)):
        raise ValueError("the ionograms compared are not on one grid")
    clean = [check.cleaned(ion) for ion in (a, b, *copies)]
    maps = [check.db_maps(c) for c, _ in clean]
    taken = np.logical_or.reduce([t for _, t in clean]
                                 + [~np.isfinite(m[key]) for m in maps
                                    for key in check.CORRELATION_MAPS])
    mask = check.trace_mask(modes, b) & ~taken
    out = {"copies": len(copies), "cells": int(np.count_nonzero(mask)),
           "half_bins": check.CORRELATION_HALF_BINS}
    if out["cells"] < 2:
        return {**out, "skipped": "no cells around the traces"}
    map_a, map_b, made = maps[0], maps[1], maps[2:]
    ion = check.correlate(map_a, map_b, mask)
    for key in check.CORRELATION_MAPS:
        out[key] = {"ion": ion[key]}
    if not copies:
        return out
    power = np.mean([np.asarray(c.power, dtype=np.float64) for c, _ in clean[2:]],
                    axis=0)
    expected = check.db_maps(SimpleNamespace(power=power, freq=np.asarray(b.freq)))
    against = [check.correlate(m, map_b, mask) for m in made]
    pairs = [check.correlate(made[j], made[k], mask)
             for j, k in itertools.combinations(range(len(made)), 2)]
    mean_expected = check.correlate(expected, map_b, mask)
    verdict = (check.compare_maps(map_b, made, mask)
               if len(copies) >= check.REALISM_SOUNDINGS else {})
    for key in check.CORRELATION_MAPS:
        z_state = float(np.mean([_z(r[key]) for r in against]))
        entry = out[key]
        entry.update(state=float(np.tanh(z_state)), expected=mean_expected[key],
                     dz_ion=z_state - _z(ion[key]))
        if pairs:
            z_ceiling = float(np.mean([_z(r[key]) for r in pairs]))
            entry.update(ceiling=float(np.tanh(z_ceiling)),
                         dz_ceiling=z_ceiling - z_state)
        if "real_inside" in verdict.get(key, {}):
            entry.update(z=verdict[key]["real_z"], inside=verdict[key]["real_inside"])
    d = expected["smoothed"][mask] - map_b["smoothed"][mask]
    out["level_db"] = {"bias": float(d.mean()), "rms": float(np.sqrt(np.mean(d ** 2)))}
    return out


def on_one_grid(origin: str | Path, truth_path: str | Path, *,
                window: int = spectro.DEFAULT_WINDOW,
                stations: Mapping[str, tuple[float, float]] | None = None,
                ) -> tuple | None:
    """The two soundings' ionograms on the range and band they share, bin for
    bin.

    None when they share under :data:`MIN_COMMON` of the truth's
    (:func:`common_span`). The forecast's copies are to be loaded on the
    truth's axis, ``check.gate_of(b)``, and cut to its band (:func:`on_band`).
    """
    a = observe.load(origin, window=window, stations=stations)
    b = observe.load(truth_path, window=window, stations=stations)
    gate = common_span(check.gate_of(a), check.gate_of(b))
    band = common_span((a.freq[0], a.freq[-1]), (b.freq[0], b.freq[-1]))
    if gate is None or band is None:
        return None
    if not (a.vrange.shape == b.vrange.shape and np.allclose(a.vrange, b.vrange)):
        a = observe.load(origin, window=window, stations=stations, gate_km=gate)
        b = observe.load(truth_path, window=window, stations=stations, gate_km=gate)
    return on_band(a, band), on_band(b, band)


def image_row(forecast: ChannelState, truth: ChannelState, grid: tuple | None,
              made: list) -> dict:
    """Group D for a pair: the row of ``ionogram.jsonl``.

    ``grid`` is :func:`on_one_grid`'s, ``made`` the forecast's copies on it,
    seeds 0 on as :func:`muf.sim.check.closed_loop` seeds them -- which is
    what lets the loop on the forecast's own sounding reuse them.
    """
    head = {"schema": IMAGE_SCHEMA, "version": VERSION, "from": forecast.source,
            "to": truth.source}
    if grid is None:
        return {**head, "copies": len(made),
                "skipped": f"the soundings share under {MIN_COMMON:.0%} of the "
                           f"truth's range or band"}
    a, b = grid
    made = [on_band(c, (b.freq[0], b.freq[-1])) for c in made]
    return {**head, **image(a, b, made, [*forecast.modes, *truth.modes])}


def evaluate(forecast: ChannelState, truth_path: str | Path, *,
             origin: str | Path | None = None, copies: int = 0, seed: int = 0,
             workdir: str | Path | None = None, method: str = "contour",
             window: int = spectro.DEFAULT_WINDOW,
             stations: Mapping[str, tuple[float, float]] | None = None,
             doppler: str | float = observe.DEFAULT_DOPPLER_PRIOR,
             knot_mhz: float = observe.KNOT_MHZ,
             name: str = "persistence", carriers: Sequence[float] | None = None,
             bandwidth_hz: float = link.BANDWIDTH_HZ,
             snr_min_db: float = link.SNR_MIN_DB) -> PairScore:
    """``forecast`` against the sounding at ``truth_path``.

    ``origin`` is the sounding the forecast was made from. The truth is
    observed as it was -- the caller passes the same extractor, window,
    knots and Doppler prior -- and loaded on its range axis
    (:func:`muf.sim.check.gate_of`), so that both are read on one grid.
    ``copies`` adds group D (:func:`image_row`), which needs ``origin``. The
    rest is :func:`score`'s.
    """
    if copies and origin is None:
        raise ValueError("comparing the ionograms needs the forecast's own sounding")
    start = time.perf_counter()
    gate = (None if origin is None
            else check.gate_of(observe.load(origin, window=window, stations=stations)))
    ion = observe.load(truth_path, window=window, gate_km=gate, stations=stations)
    truth = observe.from_ionogram(ion, method=method, source=Path(truth_path).name,
                                  doppler=doppler, knot_mhz=knot_mhz)
    out = score(forecast, truth, name=name, carriers=carriers,
                bandwidth_hz=bandwidth_hz, snr_min_db=snr_min_db)
    if copies:
        grid = on_one_grid(origin, truth_path, window=window, stations=stations)
        made = ([] if grid is None
                else sounded(forecast, copies, seed=seed,
                             gate_km=check.gate_of(grid[1]), workdir=workdir))
        out.ionogram = image_row(forecast, truth, grid, made)
    out.pair.update(method=method, seconds=round(time.perf_counter() - start, 2))
    return out


def _value(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _bounded_line(label: str, entry: dict) -> str:
    line = (f"  {label:<17} forecast {_value(entry['forecast'], '.3f')}  "
            f"truth {_value(entry['truth'], '.3f')}  "
            f"error {_value(entry['error'], '+.3f')} MHz")
    if entry["censored_truth"]:
        line += ("; truth censored, error at least "
                 + _value(entry["error_bound"], ".3f"))
    if entry["censored_forecast"]:
        line += "; forecast censored"
    return line


def _mode_line(row: dict) -> str:
    rms = row["delay_rms_us"]
    km = f" ({rms * check.GROUP_KM_PER_US:.1f} km)" if rms is not None else ""
    return (f"  {row['forecast_mode']:>10} -> {row['truth_mode']:<10} "
            f"{row['class']:<5}  "
            f"delay {_value(row['delay_bias_us'], '+.1f')} "
            f"rms {_value(row['delay_rms_us'], '.1f')} us{km}  "
            f"GDD {_value(row['gdd_bias_us_per_mhz'], '+.2f')} "
            f"rms {_value(row['gdd_rms_us_per_mhz'], '.2f')} us/MHz  "
            f"C/N0 {_value(row['cn0_bias_db'], '+.1f')} "
            f"rms {_value(row['cn0_rms_db'], '.1f')} dB  "
            f"f_lo {_value(row['f_lo_error_mhz'], '+.2f')} "
            f"f_hi {_value(row['f_hi_error_mhz'], '+.2f')} MHz")


def format_score(score: PairScore) -> str:
    """A few lines for a terminal; the whole score is :meth:`PairScore.to_dict`."""
    p = score.pair
    lead = (f"lead {p['lead_s']:g} s, {p['lead_actual_s']:.1f} s between the soundings"
            if p["lead_s"] is not None else "no lead recorded")
    lines = [f"{p['forecast']}: {p['from']} -> {p['to']} ({lead})",
             _bounded_line("MUF (extractor)", p["muf_mhz"]),
             _bounded_line("MUF (modes)", p["mode_muf_mhz"]),
             _bounded_line("LOF (modes)", p["lof_mhz"])]
    m = p["modes"]
    lines.append(f"  modes: forecast {m['forecast']}, truth {m['truth']}, "
                 f"matched {m['matched']}, missed {m['missed']}, false {m['false']}"
                 + ("" if m["pairings_agree"]
                    else "; the two directions paired them differently"))
    for row in score.modes:
        if row["status"] == "matched":
            lines.append(_mode_line(row))
        elif row["status"] == "missed":
            lines.append(f"  {'':>10} -> {row['truth_mode']:<10} missed: new in "
                         f"the truth")
        else:
            lines.append(f"  {row['forecast_mode']:>10} -> {'':<10} false: gone from "
                         f"the truth")
    lines.append(f"  {p['excluded_prior_fields']} mode fields not scored: assumed, "
                 f"not measured, in one state or both")
    return "\n".join(lines + _carrier_lines(score) + _image_lines(score.ionogram))


def _image_lines(row: dict | None) -> list[str]:
    if row is None:
        return []
    if "skipped" in row:
        return [f"  ionogram not compared: {row['skipped']}"]
    lines = [f"  ionogram, {row['copies']} copies, dB within +-{row['half_bins']} "
             f"range bins of both states' traces ({row['cells']} cells):"]
    labels = {"rows": "row by row", "smoothed": f"over +-{observe.SMOOTH_MHZ:g} MHz"}
    for key in check.CORRELATION_MAPS:
        e = row[key]
        line = f"    {labels[key]:<16} r sounding to sounding {e['ion']:.3f}"
        if "state" in e:
            line += (f", copies {e['state']:.3f}, expected {e['expected']:.3f}"
                     f"; dz over the sounding {e['dz_ion']:+.3f}")
        if "ceiling" in e:
            line += (f", ceiling {e['ceiling']:.3f}, dz under it "
                     f"{e['dz_ceiling']:+.3f}")
        if "inside" in e:
            line += f" ({e['z']:+.1f} sd) " + ("inside" if e["inside"] else "OUTSIDE")
        lines.append(line)
    if "level_db" in row:
        lines.append(f"    expected against the truth, smoothed: level "
                     f"{row['level_db']['bias']:+.2f} dB, rms "
                     f"{row['level_db']['rms']:.2f} dB")
    return lines


#: Carrier figures the terminal summary quotes, with their units and digits.
QUOTED = (("first_delay_us", "first delay", "us", ".1f"),
          ("delay_spread_us", "delay spread", "us", ".1f"),
          ("cn0_dbhz", "C/N0", "dB", ".1f"),
          ("coherence_bandwidth_khz", "coherence bandwidth", "kHz", ".2f"))


def _table_line(label: str, t: dict) -> str:
    return (f"    {label:<12} hits {t['hits']}, false alarms {t['false_alarms']}, "
            f"misses {t['misses']}, correct no {t['correct_negatives']}: "
            f"POD {_value(t['pod'], '.2f')}  FAR {_value(t['far'], '.2f')}  "
            f"CSI {_value(t['csi'], '.2f')}  "
            f"(in the truth {_value(t['base_rate'], '.2f')})")


def _carrier_lines(score: PairScore) -> list[str]:
    c = score.pair["carriers"]
    if not c["n"]:
        return []
    f0 = [r["f0_mhz"] for r in score.carriers]
    lines = [f"  {c['n']} carriers {min(f0):.2f}-{max(f0):.2f} MHz, usable from SNR "
             f"{c['snr_min_db']:g} dB in {c['bandwidth_hz']:g} Hz:",
             _table_line("a mode seen", c["events"]["mode"]),
             _table_line("usable", c["events"]["usable"])]
    parts = []
    for key, label, unit, digits in QUOTED:
        errors = [abs(r["figures"][key]["error"]) for r in score.carriers
                  if r["figures"][key]["error"] is not None]
        if errors:
            parts.append(f"{label} {format(float(np.median(errors)), digits)} {unit}")
    if parts:
        lines.append("    median |error| where both see a mode: " + ", ".join(parts))
    return lines
