"""One mode over one band: 1F over 1 MHz at 0.75 of its MUF, as a complex signal.

The full simulator models every mode over the whole sweep, so its errors mix
the modes' problems. A link needs only one part of the channel: 1 MHz of the
1F mode around 0.75 of its MUF (#92). This module reads that part alone,
precisely, and makes the channel state of it.

**The band.** 1F is the earliest-arriving low ray, and its MUF the top of
that ray (:func:`band_of`). The band is ``FRACTION`` of that MUF,
``WIDTH_MHZ`` wide. A censored MUF, or no low ray that covers the whole band,
gives no band.

**The signal.** Over the band the dechirped record is dechirped a second time
along 1F's trace and low-passed, as :mod:`muf.sim.fast` does: 1F's tone comes
to zero frequency and every other mode, farther than the stop band, is
discarded. What is left is 1F's complex signal a(t), the o and x waves
together. Its power gives the level; its phase, added to the trace it was
brought down along, gives the echo's beat along the sweep, and with it the
delay, the GDD and the TOD in the band, far finer than the ionogram's trace;
its fading gives the Doppler spread and the Rice factor, as the fast part
reads them.

**The band state.** 1F alone, its curve and level from the signal, its sweep
cut to the band: a :class:`~muf.sim.state.ChannelState` like any other, so a
sounding, a channel or a link is made from it unchanged.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import fast
from .state import MEASURED, ChannelState, Mode, Provenance

#: Where the band is centred, as a fraction of 1F's MUF, and how wide it is.
FRACTION = 0.75
WIDTH_MHZ = 1.0

#: How far beyond the band the band state's mode reaches, MHz: past the
#: synthetic sounding's edge taper and the filter's trimmed ends, so a copy
#: is whole across the band.
MARGIN_MHZ = 0.2

#: Spacing of the band state's knots, MHz.
KNOT_STEP_MHZ = 0.05

#: The phase is read over lags this long, s: long enough that the noise in
#: each step is small, short enough that the residual beat cannot turn it
#: by half a cycle.
PHASE_LAG_S = 0.1

#: What a reading holds, and what its copies are compared on.
FIELDS = ("cn0_dbhz", "delay_us", "gdd_us_per_mhz", "tod_us_per_mhz2", "spread_hz")


@dataclass(frozen=True)
class Band:
    """The part of the sweep a link uses: one mode, from ``lo_mhz`` to ``hi_mhz``."""

    mode: str
    muf_mhz: float
    lo_mhz: float
    hi_mhz: float

    @property
    def centre_mhz(self) -> float:
        return (self.lo_mhz + self.hi_mhz) / 2.0


@dataclass(frozen=True)
class Signal:
    """One mode's complex signal over one band, the mode alone."""

    samples: np.ndarray
    rate_hz: float
    freq_mhz: np.ndarray
    beat_hz: np.ndarray
    pass_hz: float
    noise_power: float
    noise_r: np.ndarray
    hz_per_us: float
    zero_hz: float


def band_of(state: ChannelState, *, fraction: float = FRACTION,
            width_mhz: float = WIDTH_MHZ) -> Band:
    """The band at ``fraction`` of 1F's MUF, or ValueError saying why there is none.

    1F is the earliest-arriving low ray that covers ``fraction`` of its own
    MUF, whatever its label. It arrives before every other F mode, and a
    receiver clock that is off moves every delay alike, so it cannot change
    which arrives first -- where it moves the hop count the geometry reads off
    the delays (yoshkar-ola's, from 18 October 2026). An earlier fragment too
    short to hold its band is passed over.
    """
    rays = sorted((m for m in state.modes if m.branch != "high"),
                  key=lambda m: float(np.median(m.delay_us)))
    if not rays:
        raise ValueError("the state has no low ray")
    for mode in rays:
        lo = fraction * mode.f_hi - width_mhz / 2.0
        hi = fraction * mode.f_hi + width_mhz / 2.0
        if mode.f_lo > lo or mode.f_hi < hi:
            continue
        edge = state.metrics.get("band_edge_mhz")
        if edge is not None and math.isfinite(edge) and mode.f_hi >= edge:
            raise ValueError(f"1F's MUF is censored: {mode.name} reaches "
                             f"{mode.f_hi:.2f} MHz, the sweep's edge {edge:.2f} MHz")
        return Band(mode.name, mode.f_hi, lo, hi)
    raise ValueError(f"no low ray covers {fraction:g} of its own MUF, "
                     f"{width_mhz:g} MHz wide")


def isolate(record: fast.Record, mode: Mode, band: Band, others=(), *,
            passes: tuple[float, ...] = fast.PASSES_HZ) -> Signal:
    """``mode``'s signal over ``band``, every other mode discarded, or ValueError."""
    t_lo, t_hi = record.time_at([band.lo_mhz, band.hi_mhz])
    i0 = math.ceil((t_lo - fast.TRIM_S - record.t_first_s) * record.sr_hz)
    i1 = math.floor((t_hi + fast.TRIM_S - record.t_first_s) * record.sr_hz)
    if i0 < 0 or i1 > record.samples.size:
        raise ValueError(f"the record does not cover {band.lo_mhz:.2f}-"
                         f"{band.hi_mhz:.2f} MHz")
    t = record.t_first_s + np.arange(i0, i1) / record.sr_hz
    coarse = t[::64]
    beat = np.interp(t, coarse, fast._beats(record, mode, coarse))
    widest = fast.STOP_RATIO * max(passes)
    edge = 0.45 * record.sr_hz - widest
    if np.max(np.abs(beat)) > edge:
        raise ValueError(f"{mode.name}'s beat is outside the record's band")
    f = record.freq_at(t)
    margin = widest + fast.ISOLATION_MARGIN_HZ
    offsets = [o for o in fast.NOISE_OFFSETS_HZ
               if np.max(np.abs(beat + o)) < edge
               and not any(np.any(m.covers(f) & (np.abs(fast._their_beat(record, m, f)
                                                         - beat - o) < margin))
                           for m in others)]
    if not offsets:
        raise ValueError("no offset clear of every other mode for the noise reference")
    z, rate = fast._down(record, i0, i1, beat, pass_hz=max(passes), stop_hz=widest)
    p_n, offset, _ = _noise(record, i0, i1, beat, offsets, max(passes), widest)
    p_s = float(np.mean(np.abs(z) ** 2)) - p_n
    if p_s <= 0:
        raise ValueError(f"no {mode.name} signal above the noise in the band")
    beat = beat + fast._recentre(z, rate, t[0] + fast.TRIM_S, t, p_n)
    separate = [o for o in fast._conflicts(record, mode, list(others), coarse,
                                           beat[::64], margin)
                if fast._separate(record, mode, o, i0, i1, beat, p_s, offset)]
    step = max(1, math.floor(record.sr_hz / fast.OUT_RATE_HZ))
    reason = ""
    for pass_hz in passes:
        stop = fast.STOP_RATIO * pass_hz
        near = [o.name for o in separate
                if np.any(o.covers(f) & (np.abs(fast._their_beat(record, o, f) - beat)
                                         < stop + fast.ISOLATION_MARGIN_HZ))]
        if near:
            reason = (f"a second echo ({', '.join(near)}) within "
                      f"{stop + fast.ISOLATION_MARGIN_HZ:g} Hz of {mode.name}")
            continue
        z, rate = fast._down(record, i0, i1, beat, pass_hz=pass_hz, stop_hz=stop)
        p_n, _, noise = _noise(record, i0, i1, beat, offsets, pass_hz, stop)
        p_s = float(np.mean(np.abs(z) ** 2)) - p_n
        snr = 10.0 * math.log10(p_s / p_n) if p_s > 0 else -math.inf
        if snr < fast.MIN_SNR_DB:
            reason = f"SNR {snr:.1f} dB in {2 * pass_hz:g} Hz"
            continue
        trim = round(fast.TRIM_S * rate)
        kept = slice(trim, trim + z.size)
        max_lag = min(z.size // 3, round(fast.MAX_LAG_S * rate))
        return Signal(samples=z, rate_hz=rate, freq_mhz=f[::step][kept],
                      beat_hz=beat[::step][kept], pass_hz=pass_hz, noise_power=p_n,
                      noise_r=np.abs(fast._autocorrelation(noise, max_lag)),
                      hz_per_us=record.km_per_us / record.km_per_hz,
                      zero_hz=-(record.range_zero_km / record.km_per_hz
                                + record.shift_hz))
    raise ValueError(reason)


def _noise(record: fast.Record, i0: int, i1: int, beat: np.ndarray, offsets,
           pass_hz: float, stop_hz: float) -> tuple[float, float, np.ndarray]:
    """The noise reference: (power, offset, samples) of the offset of median power.

    The offsets are clear of every other mode, so none holds an echo and
    the median of their powers is the noise's, a few percent finer than one
    offset's -- and the level is the signal's power over it.
    """
    rows = sorted(((float(np.mean(np.abs(n) ** 2)), o, n) for o, n in
                   ((o, fast._down(record, i0, i1, beat + o, pass_hz=pass_hz,
                                   stop_hz=stop_hz)[0]) for o in offsets)),
                  key=lambda row: row[0])
    return rows[len(rows) // 2]


def reading(signal: Signal, band: Band) -> dict:
    """What a band's signal gives: level, delay, GDD, TOD and fading at its centre.

    The echo's beat is the one the signal was brought down along plus the
    signal's own, read from its phase over :data:`PHASE_LAG_S` and weighted
    by its amplitude; a quadratic in frequency through it is the delay curve.
    The level is the signal's power over the noise reference's, times the
    filter's noise bandwidth.
    """
    z = signal.samples
    power = np.abs(z) ** 2
    p_s = float(np.mean(power)) - signal.noise_power
    noise_hz = fast._bandwidth(signal.pass_hz, fast.STOP_RATIO * signal.pass_hz)
    lag = max(1, round(PHASE_LAG_S * signal.rate_hz))
    step = z[lag:] * np.conj(z[:-lag])
    own = np.angle(step) * signal.rate_hz / (2.0 * np.pi * lag)
    beat = (signal.beat_hz[lag:] + signal.beat_hz[:-lag]) / 2.0 + own
    x = (signal.freq_mhz[lag:] + signal.freq_mhz[:-lag]) / 2.0 - band.centre_mhz
    c2, c1, c0 = np.polyfit(x, beat, 2, w=np.sqrt(np.abs(step)))
    est = fast.BandEstimate(band.mode, band.lo_mhz, band.hi_mhz, used=True,
                            seconds=z.size / signal.rate_hz, rate_hz=signal.rate_hz,
                            pass_hz=signal.pass_hz, power=power,
                            noise_r=signal.noise_r, noise_power=signal.noise_power)
    fit = fast._pooled_fit([est])
    fast._fill(est, fit)
    return {"cn0_dbhz": 10.0 * math.log10(p_s / signal.noise_power * noise_hz),
            "snr_db": 10.0 * math.log10(p_s / signal.noise_power),
            "delay_us": float((c0 - signal.zero_hz) / signal.hz_per_us),
            "gdd_us_per_mhz": float(c1 / signal.hz_per_us),
            "tod_us_per_mhz2": float(2.0 * c2 / signal.hz_per_us),
            "spread_hz": float(est.spread_hz), "shape": est.shape, "bound": est.bound,
            "rice_k_db": 10.0 * math.log10(fit["k"]) if fit["k"] > 0 else None,
            "pass_hz": signal.pass_hz, "seconds": est.seconds}


def narrow(state: ChannelState, band: Band, measured: dict) -> ChannelState:
    """The band state: ``band``'s mode alone, its curve, level and fading ``measured``.

    The delay and the GDD come from the band's phase, the TOD from the
    trace: one band, faded, gives the curvature only to tens of us/MHz^2.
    The level is flat at the band's. The fading is the band's unless the
    band only bounded it. The sweep is cut to the band and its margins.
    """
    out = ChannelState.from_dict(state.to_dict())
    mode = out.mode(band.mode)
    lo, hi = band.lo_mhz - MARGIN_MHZ, band.hi_mhz + MARGIN_MHZ
    knots = np.linspace(lo, hi, round((hi - lo) / KNOT_STEP_MHZ) + 1)
    x = knots - band.centre_mhz
    tod = float(mode.tod_at(band.centre_mhz))
    note = f"the band's phase, {band.lo_mhz:.2f}-{band.hi_mhz:.2f} MHz"
    provenance = dict(mode.provenance)
    provenance["delay"] = Provenance(MEASURED, f"{note}; TOD {tod:.2f} us/MHz^2 from "
                                               "the trace")
    provenance["cn0"] = Provenance(MEASURED, f"flat at the level of {note}")
    provenance["band"] = Provenance(
        MEASURED, f"{band.lo_mhz:.2f}-{band.hi_mhz:.2f} MHz, {FRACTION:g} of 1F's "
                  f"MUF {band.muf_mhz:.2f} MHz, +-{MARGIN_MHZ:g} MHz")
    fading = {}
    if not measured["bound"]:
        fading = {"doppler_spread_hz": measured["spread_hz"],
                  "doppler_shape": measured["shape"],
                  "rician_k_db": measured["rice_k_db"], "delay_spread_us": 0.0}
        provenance["doppler_spread"] = Provenance(MEASURED, f"{note}; "
                                                  f"{measured['shape']} by "
                                                  "autocorrelation fit")
        provenance["fading"] = Provenance(MEASURED, f"{note}; Rice K by moments")
        provenance["delay_spread"] = Provenance(
            MEASURED, f"folded into the fading of {note}: along an up-sweep a delay "
                      "spread fades as a Doppler spread does, and one sweep cannot "
                      "tell them apart")
    out.modes = [replace(mode, freq_mhz=knots,
                         delay_us=measured["delay_us"] + measured["gdd_us_per_mhz"] * x
                         + 0.5 * tod * x ** 2,
                         cn0_dbhz=np.full(knots.size, measured["cn0_dbhz"]),
                         provenance=provenance, **fading)]
    if out.sweep is not None:
        pad = MARGIN_MHZ + 2.0 * fast.TRIM_S * out.sweep.rate_hz_s / 1e6
        seconds = math.ceil((hi - lo + 2.0 * pad) * 1e6 / out.sweep.rate_hz_s)
        out.sweep = replace(out.sweep, start_mhz=band.lo_mhz - MARGIN_MHZ - pad,
                            duration_s=float(seconds))
    out.notes.append(f"band state: {band.mode} alone over "
                     f"{band.lo_mhz:.2f}-{band.hi_mhz:.2f} MHz")
    out.metrics["band_lo_mhz"], out.metrics["band_hi_mhz"] = band.lo_mhz, band.hi_mhz
    out.metrics["band_muf_mhz"] = band.muf_mhz
    return out


def copies(banded: ChannelState, band: Band, *, count: int,
           seed: int = 0) -> list[dict]:
    """Readings of ``count`` synthetic soundings of the band state, read as real."""
    from . import sounding

    folder = Path(tempfile.mkdtemp(prefix="muf-band-"))
    try:
        rows = []
        for k in range(count):
            path = sounding.write_lfs(banded, folder / f"copy_{k}.lfs", seed=seed + k)
            signal = isolate(fast.record_of(path), banded.modes[0], band)
            rows.append(reading(signal, band))
            path.unlink()
        return rows
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def check(path: str | Path, *, count: int | None = None, seed: int = 0,
          method: str = "contour", stations=None) -> dict:
    """A sounding's band: its reading, its band state, and the copies' verdicts."""
    from . import check as loop
    from . import observe

    state = observe.from_sounding(path, method=method, stations=stations)
    band = band_of(state)
    mode = state.mode(band.mode)
    real = reading(isolate(fast.record_of(path, stations), mode, band,
                           [m for m in state.modes if m is not mode]), band)
    banded = narrow(state, band, real)
    rows = copies(banded, band, count=count or loop.REALISM_SOUNDINGS, seed=seed)
    return {"source": Path(path).name, "band": asdict(band), "real": real,
            "copies": {k: loop._stats([r[k] for r in rows]) for k in FIELDS},
            "verdicts": {k: loop._verdict(real[k], [r[k] for r in rows])
                         for k in FIELDS},
            "state": banded.to_dict()}


def format_report(report: dict) -> str:
    """The report as lines: the band, then each field real against its copies."""
    b, real = report["band"], report["real"]
    lines = [f"{report['source']}: {b['mode']} over {b['lo_mhz']:.2f}-"
             f"{b['hi_mhz']:.2f} MHz, {FRACTION:g} of its MUF {b['muf_mhz']:.2f} MHz; "
             f"SNR {real['snr_db']:.1f} dB in {2 * real['pass_hz']:g} Hz"]
    for k in FIELDS:
        s, v = report["copies"][k], report["verdicts"][k]
        spread = (f"copies {s['mean']:.3f} +- {s['std']:.3f}" if s.get("n")
                  else "no copies")
        where = "inside" if v.get("real_inside") else "OUTSIDE"
        verdict = f"z {v['real_z']:+.1f}, {where}" if v else "no verdict"
        lines.append(f"  {k:>16}: real {real[k]:.3f}; {spread}; {verdict}")
    return "\n".join(lines)
