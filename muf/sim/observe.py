"""From a sounding to a channel state: the slow part, off the ionogram.

The trace is found the way ``muf run`` finds it -- an extractor, then
``muf.trace``: points per contiguous run, tracks followed across frequency,
hop labels from the geometry, low and high rays paired at the nose. Each
track becomes one mode:

* its delay curve is :func:`muf.trace.reconstruct`'s smoothing spline,
  sampled on knots and converted from range with the format's own axis
  (the ``.lfs`` axis is calibrated with c = 3e5 km/s, v2's with c exact);
* its C/N0 is the ionogram's power along the curve, summed over the five
  range bins around it so that where the echo falls between bins does not
  matter (a Hann window's tone energy spans 1.5 bins), less the noise those
  bins carry, smoothed over a few tenths of a MHz to average the fading.

The conversion from muf's scale is exact for ``.lfs``: muf divides each
row by 4 ln2 times its median, so a tone of carrier-to-noise density C/N0
peaks at ``C/N0 * T / (6 ln^2 2)`` for a window of T seconds, and a noise
bin averages ``1 / (4 ln^2 2)``. v2 products put their own spectrogram on
the same scale only approximately, and say so in the provenance.

What a power ionogram cannot give -- Doppler shift and spread, the fading
law, the o/x split -- is filled from a prior and marked as one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .. import calibrate, extractors, interference, loader, spectro, trace
from ..io_lfs import LfsHeader
from . import presets
from .state import MEASURED, PRIOR, ChannelState, Circuit, Mode, Provenance, Sweep

#: A Hann window's equivalent noise bandwidth, bins: the share of a tone's
#: energy that is not in its peak bin is what summing neighbours recovers.
HANN_ENBW_BINS = 1.5

#: Range bins summed either side of the curve.
SUM_HALF_BINS = 2

#: muf's mean noise power per bin: |FFT(hann x)|^2 averages 3W/8 per unit
#: variance, and the floor divided by is 4 ln2 * (its median, ln2 * mean).
NOISE_MEAN_POWER = 1.0 / (4.0 * math.log(2.0) ** 2)

#: Peak power of a tone, per unit C/N0 and second of window: W^2/4 over the
#: floor (3/2) ln^2(2) W sigma^2, with |a|^2 = (C/N0) sigma^2 / fs.
TONE_PER_CN0_SECOND = 1.0 / (6.0 * math.log(2.0) ** 2)

#: Default knot spacing, MHz.
KNOT_MHZ = 0.1

#: A gap without points wider than this is joined straight, not by the
#: spline, MHz; the delay's provenance names the widest. A smoothing spline's
#: knots sit on its points, so across a gap one knot interval holds no data:
#: only continuity holds the cubic there, and it can swing thousands of km off
#: the trace while passing close to every point.
FIT_GAP_MHZ = 0.5

#: Half-width over which the C/N0 along a mode is averaged, MHz. A
#: Rayleigh-fading echo's level is the mean of its power, not a sample of it.
SMOOTH_MHZ = 0.2

DEFAULT_DOPPLER_PRIOR = "mid-moderate"

#: Rows weaker than this are left out of the level statistics: near the
#: noise a row's level scatters by the noise's estimate, not the fading's.
LEVEL_ABOVE_DBHZ = 30.0

#: Frequency lags, MHz, of the level's structure function along a mode.
LEVEL_LAGS_MHZ = (0.05, 0.2, 0.5)


def level_statistics(f_rows: np.ndarray, cn0_rows: np.ndarray) -> dict | None:
    """How a mode's level scatters along its trace, row to row.

    The rows' C/N0 in dB less its smoothed curve: the standard deviation, and
    the structure function D(df) = <(L(f + df) - L(f))^2> at
    ``LEVEL_LAGS_MHZ``. Along a sweep both are set by the fast part of the
    channel -- the Doppler spread against the window, the Rice factor, o/x
    beating -- which a power ionogram cannot measure but does *show*. A
    Rayleigh-fading row's power is exponential, 5.6 dB r.m.s. in dB; a
    window long against the fading averages that down.
    """
    smooth = _smooth(f_rows, cn0_rows, SMOOTH_MHZ)
    level = 10.0 * np.log10(np.maximum(cn0_rows, 1e-6))
    trend = 10.0 * np.log10(np.maximum(smooth, 1e-6))
    strong = trend >= LEVEL_ABOVE_DBHZ
    if np.count_nonzero(strong) < 10:
        return None
    resid = (level - trend)[strong]
    step = float(np.median(np.diff(f_rows))) if f_rows.size > 1 else 0.0
    sf = {}
    for lag in LEVEL_LAGS_MHZ:
        k = round(lag / step) if step > 0 else 0
        if k < 1 or k >= level.size:
            continue
        both = strong[k:] & strong[:-k]
        if np.count_nonzero(both) >= 10:
            d = (level[k:] - level[:-k])[both]
            sf[f"{lag:g}"] = float(np.mean(d ** 2))
    return {"rows": int(np.count_nonzero(strong)), "std_db": float(np.std(resid)),
            "sf_db2": sf}


#: Range bins either side of the trace its width is read over.
SPREAD_HALF_BINS = 4

#: Rows the width is read on: strong enough that it is the echo's, not the
#: noise's -- the peak bin some 17 dB over the noise.
SPREAD_ABOVE_DBHZ = 28.0

#: Rows a reading needs, clear of other modes.
SPREAD_MIN_ROWS = 20

#: Row-to-row correlation of the trace's centres above which its width is not
#: taken for scatter. Paths fading independently leave the centres
#: independent: -0.4 to +0.3 on synthetic soundings of 10 to 80 us. A trace
#: whose centres move together is wide for another reason than scatter.
SCATTER_MAX_R1 = 0.4


def hann_response(x) -> np.ndarray:
    """Power a Hann-windowed FFT puts ``x`` bins from a tone, 1 at the tone.

    The periodic Hann window's transform is a sinc and half of each of its
    neighbours; muf's spectrogram uses that window with no zero padding.
    """
    x = np.asarray(x, dtype=np.float64)
    return (np.sinc(x) + 0.5 * np.sinc(x - 1.0) + 0.5 * np.sinc(x + 1.0)) ** 2


def cross_trace(ion, mode: Mode, others=(), *,
                half_bins: int = SPREAD_HALF_BINS) -> dict | None:
    """How wide the trace is in delay, against a point echo: its delay spread.

    Row by row, the power over ``half_bins`` either side of the curve, less
    the noise: its centre and its variance about that centre. A point echo
    at the same centre would have the Hann response's variance, and the
    excess, pooled over the rows by power, is the spread *within* a row.
    That is not all of it. One row is one 0.2 s snapshot of paths fading
    independently, narrower than their mean profile and off its centre by
    chance, so the centres scatter too; the two variances add to the
    profile's (``spread_us``, 2-7 % low from 10 to 60 us on synthetic
    soundings, 0-8 us at no spread). The centres' correlation from
    one row to the next tells the two causes apart: a spread's snapshots are
    independent rows apart, a trace wandering about the smooth curve is not.
    Rows within reach of another mode are left out. None under
    :data:`SPREAD_MIN_ROWS` rows.
    """
    freq, vrange = np.asarray(ion.freq), np.asarray(ion.vrange)
    scale = km_per_us(ion.header)
    bin_us = float(np.median(np.abs(np.diff(vrange)))) / scale
    reach = (half_bins + 3) * bin_us
    t_w = window_seconds(ion)
    power = np.asarray(ion.power, dtype=np.float64)
    excess, weight, centres = 0.0, 0.0, []
    for i in np.flatnonzero((freq >= mode.f_lo) & (freq <= mode.f_hi)):
        f = freq[i]
        tau = float(mode.delay_at(f))
        if any(o.covers(f) and abs(float(o.delay_at(f)) - tau) < reach for o in others):
            continue
        j = int(np.argmin(np.abs(vrange - tau * scale)))
        if j - half_bins < 0 or j + half_bins >= vrange.size:
            continue
        w = power[i, j - half_bins:j + half_bins + 1] - NOISE_MEAN_POWER
        a, b = j - SUM_HALF_BINS, j + SUM_HALF_BINS + 1
        tone = (power[i, a:b].sum() - (b - a) * NOISE_MEAN_POWER) / HANN_ENBW_BINS
        if tone <= 0 or 10 * math.log10(tone / (TONE_PER_CN0_SECOND * t_w)) \
                < SPREAD_ABOVE_DBHZ:
            continue
        d = vrange[j - half_bins:j + half_bins + 1] / scale - tau     # + later
        total = float(w.sum())
        centre = float(np.sum(d * w) / total)
        var = float(np.sum((d - centre) ** 2 * w) / total)
        k = hann_response((d - centre) / bin_us)
        point = float(np.sum((d - centre) ** 2 * k) / k.sum())
        excess += total * (var - point)
        weight += total
        centres.append((i, total, centre))
    if len(centres) < SPREAD_MIN_ROWS:
        return None
    within = excess / weight
    rows, w, c = (np.array(x) for x in zip(*centres))
    mean = float(np.sum(w * c) / w.sum())
    scatter = float(np.sum(w * (c - mean) ** 2) / w.sum())
    next_row = np.flatnonzero(np.diff(rows) == 1)
    r1 = (float(np.corrcoef(c[next_row], c[next_row + 1])[0, 1])
          if next_row.size >= 3 else float("nan"))
    return {"rows": int(rows.size),
            "spread_us": math.sqrt(max(within + scatter, 0.0)),
            "within_us": math.sqrt(max(within, 0.0)),
            "centre_rms_us": math.sqrt(scatter), "centre_r1": r1,
            "offset_us": mean, "bin_us": bin_us}


def _delay_spread(state: ChannelState, ion) -> None:
    """Each mode's delay spread from its trace's width, where it is scatter.

    The width is kept in ``metrics["cross_trace"]`` whatever it is; it becomes
    the mode's delay spread only when its centres are independent from row to
    row, as scattered paths leave them (:data:`SCATTER_MAX_R1`).
    """
    for m in state.modes:
        others = [o for o in state.modes if o is not m]
        width = cross_trace(ion, m, others)
        if width is None:
            m.provenance["delay_spread"] = Provenance(
                PRIOR, "one path, as in F.1487: too few strong rows clear of other "
                       "modes to read the trace's width")
            continue
        state.metrics.setdefault("cross_trace", {})[m.name] = width
        read = (f"the trace's width over {width['rows']} rows, +-{SPREAD_HALF_BINS} "
                f"bins, against a point echo, {width['spread_us']:.1f} us: "
                f"{width['within_us']:.1f} within a row, the centres "
                f"{width['centre_rms_us']:.1f} RMS, correlated "
                f"{width['centre_r1']:+.2f} row to row")
        if not width["centre_r1"] <= SCATTER_MAX_R1:
            m.provenance["delay_spread"] = Provenance(
                PRIOR, f"one path: {read} -- coherent structure, not scatter, which "
                       f"would leave them independent")
            continue
        m.delay_spread_us = width["spread_us"]
        m.provenance["delay_spread"] = Provenance(
            MEASURED, f"{read}, as independent paths leave them; effective -- the "
                      f"phase (--fast) says whether it fades as scatter would")


def km_per_us(header) -> float:
    """Range-axis km per microsecond of group delay, for this format."""
    if isinstance(header, LfsHeader):
        return calibrate.C_KM_S * 1e-6 * 2.0 / header.div_coef
    from .. import io_chirp

    # v2's axis is c/rate times the beat, with c exact and no path divisor.
    return io_chirp.C_M_S * 1e-9


def window_seconds(ion) -> float:
    header = ion.header
    return ion.window / (float(header.sample_rate) / float(header.dec))


def sweep_of(header) -> Sweep:
    start, _stop = calibrate.sweep_bounds(header)
    sr = float(header.sample_rate) / float(header.dec)
    return Sweep(start_mhz=float(start), rate_hz_s=float(header.rate),
                 duration_s=float(header.dur), sample_rate_hz=sr,
                 adc_rate_hz=float(header.sample_rate),
                 rmin_km=int(header.rmin), rmax_km=int(header.rmax),
                 repetition_s=int(getattr(header, "rep", 300) or 300),
                 chirpt=int(getattr(header, "chirpt", 10) or 10))


def epoch_of(header) -> float:
    if isinstance(header, LfsHeader):
        return float(header.start_epoch)
    return float(header.t0)


def _curve(seg) -> tuple[np.ndarray, np.ndarray, str]:
    """A segment's range curve over its own span, and how it was made."""
    rec = trace.reconstruct(seg)
    if rec.ok and rec.freq.size >= 2:
        how = (f"muf.trace.reconstruct, rms residual {rec.rms_residual_km:.1f} km"
               if np.isfinite(rec.rms_residual_km) else "muf.trace.reconstruct")
        return (*_bridged(seg, rec.freq, rec.vrange), how)
    # Too short to reconstruct: medians per 0.1 MHz, the data as it is.
    edges = np.arange(seg.freq.min(), seg.freq.max() + KNOT_MHZ, KNOT_MHZ)
    idx = np.clip(np.digitize(seg.freq, edges) - 1, 0, edges.size - 1)
    f, r = [], []
    for k in np.unique(idx):
        sel = idx == k
        f.append(float(np.median(seg.freq[sel])))
        r.append(float(np.median(seg.vrange[sel])))
    return np.asarray(f), np.asarray(r), f"bin medians ({rec.reason})"


def _bridged(seg, f: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A segment's spline with no samples in its gaps, so the mode joins them straight.

    The gaps are those without points wider than :data:`FIT_GAP_MHZ`. Between
    them the spline is kept as it is, held by points on either side; a
    stretch it has no sample within -- a lone point beyond a gap -- is its
    points' median, since the spline's samples beside it can be anywhere. The
    mode's knots inside a gap then lie on the line between its edges: a fade
    says nothing of the curve, and a line claims least.

    The two other ways out were worse. Fitting each stretch apart leaves short
    stretches to splines of their own, which overshoot a step in range the
    whole fit followed. Splitting the mode at the gap shortens the bands that
    scoring and the top's error read, where :mod:`muf.trace` follows a track
    across a fade as one on purpose.
    """
    cut = np.flatnonzero(np.diff(seg.freq) > FIT_GAP_MHZ) + 1
    if cut.size == 0:
        return f, r
    kept = []
    for freq, vrange in zip(np.split(seg.freq, cut), np.split(seg.vrange, cut)):
        inside = (f >= freq[0]) & (f <= freq[-1])
        kept.append((f[inside], r[inside]) if inside.any() else
                    (np.median(freq, keepdims=True), np.median(vrange, keepdims=True)))
    return tuple(np.concatenate(x) for x in zip(*kept))


def _cn0_along(ion, f_curve: np.ndarray, r_curve: np.ndarray,
               lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """(row frequency MHz, C/N0 linear) along a curve, from the ionogram."""
    rows = np.flatnonzero((ion.freq >= lo) & (ion.freq <= hi))
    if rows.size == 0:
        return np.empty(0), np.empty(0)
    t_w = window_seconds(ion)
    vrange = np.asarray(ion.vrange)
    n_col = vrange.size
    out = np.empty(rows.size)
    for k, i in enumerate(rows):
        r = float(np.interp(ion.freq[i], f_curve, r_curve))
        j = int(np.argmin(np.abs(vrange - r)))
        a, b = max(0, j - SUM_HALF_BINS), min(n_col, j + SUM_HALF_BINS + 1)
        band = np.asarray(ion.power[i, a:b], dtype=np.float64)
        tone = (band.sum() - band.size * NOISE_MEAN_POWER) / HANN_ENBW_BINS
        out[k] = max(tone, 1e-6) / (TONE_PER_CN0_SECOND * t_w)
    return np.asarray(ion.freq[rows], dtype=np.float64), out


def _smooth(freq: np.ndarray, value: np.ndarray, half: float) -> np.ndarray:
    out = np.empty_like(value)
    for k, f in enumerate(freq):
        out[k] = value[np.abs(freq - f) <= half].mean()
    return out


def _name(seg, taken: set[str]) -> str:
    base = f"{seg.hops}F" if seg.hops else "F"
    if seg.branch == "high":
        base += "-high"
    name, k = base, 2
    while name in taken:
        name, k = f"{base}#{k}", k + 1
    taken.add(name)
    return name


def from_ionogram(ion, *, method: str = "contour", source: str = "",
                  doppler: str | float = DEFAULT_DOPPLER_PRIOR,
                  knot_mhz: float = KNOT_MHZ) -> ChannelState:
    """A state from an ionogram already in hand. See :func:`from_sounding`."""
    header = ion.header
    if getattr(ion, "zero_periods", 0):
        raise ValueError("C/N0 is calibrated for an unpadded spectrogram; "
                         "load with zero_periods=0")
    approx = "" if isinstance(header, LfsHeader) else ", v2 scale approximate"

    if isinstance(doppler, str):
        spread, doppler_origin = presets.doppler_prior(doppler)
    else:
        spread = float(doppler)
        doppler_origin = Provenance(PRIOR, f"given: {doppler} Hz")

    from ..pipeline import Options, band_edge_mhz, circuit_ceiling, lof_edge_mhz
    ion, _rejected = interference.apply(ion, Options())
    result = extractors.run(ion, methods=(method,))[method]

    circuit = Circuit(header.tx_name, header.rx_name,
                      float(header.tx_latitude), float(header.tx_longitude),
                      float(header.rx_latitude), float(header.rx_longitude))
    corrected = getattr(header, "from_registry", ())
    registry = (f", {'+'.join(corrected)} from the station registry"
                if corrected else "")
    state = ChannelState(
        circuit=circuit, epoch=epoch_of(header), sweep=sweep_of(header), source=source,
        provenance={
            "epoch": Provenance(MEASURED, "sounding header"),
            "circuit": Provenance(MEASURED, "sounding header" + registry),
            "sweep": Provenance(MEASURED, "sounding header"),
        })
    state.metrics["window"] = float(ion.window)
    # Where a MUF or an LOF stops being a measurement, as muf run has it, and
    # whether the recording holds its whole sweep: a forecast scored against
    # this state reads its censoring here.
    state.metrics["band_edge_mhz"] = band_edge_mhz(ion.cal,
                                                   circuit_ceiling(header, Options()))
    state.metrics["lof_edge_mhz"] = lof_edge_mhz(ion.cal)
    state.metrics["sweep_complete"] = float(ion.cal.sweep_complete)
    if not result.ok:
        state.notes.append(f"{method} found no trace"
                           + (f": {result.error}" if result.error else ""))
        return state
    state.metrics["muf_mhz"] = float(result.pick.muf_mhz)
    state.metrics["pick_vrange_km"] = float(result.pick.vrange_km)

    freq, vrange, weight = trace.extract_points(ion, result)
    segments = trace.group_tracks(freq, vrange, weight)
    ground = calibrate.ground_range_km(header) if isinstance(header, LfsHeader) else (
        circuit.ground_km)
    if ground:
        segments = trace.identify_hops(segments, ground)
    segments = trace.merge_branches(segments)
    # The high ray is labelled through its low ray: they are one mode's two
    # branches, and the range that would place the high ray by itself
    # matches no hop count well.
    hops_of = {s.group: s.hops for s in segments if s.group is not None and s.hops}
    for s in segments:
        if s.hops is None and s.group in hops_of:
            s.hops = hops_of[s.group]

    scale = km_per_us(header)
    taken: set[str] = set()
    for seg in sorted(segments, key=lambda s: (s.median_range, s.freq_span[0])):
        lo, hi = seg.freq_span
        if hi - lo < knot_mhz:
            continue
        f_curve, r_curve, how = _curve(seg)
        knots = np.arange(lo, hi, knot_mhz)
        knots = np.append(knots[knots < hi - 1e-9], hi)
        r_k = np.interp(knots, f_curve, r_curve)
        f_rows, cn0_rows = _cn0_along(ion, f_curve, r_curve, lo, hi)
        if f_rows.size < 2:
            continue
        cn0_k = np.interp(knots, f_rows, _smooth(f_rows, cn0_rows, SMOOTH_MHZ))
        levels = level_statistics(f_rows, cn0_rows)
        gaps = np.diff(np.unique(seg.freq))
        widest = float(gaps.max()) if gaps.size else 0.0
        name = _name(seg, taken)
        if levels is not None:
            state.metrics.setdefault("levels", {})[name] = levels
        state.modes.append(Mode(
            name=name, freq_mhz=knots, delay_us=r_k / scale,
            cn0_dbhz=10.0 * np.log10(cn0_k), doppler_spread_hz=spread,
            hops=seg.hops, branch=seg.branch,
            provenance={
                "delay": Provenance(MEASURED, f"{method} trace, {how}"
                                    + (f"; widest gap without points {widest:.2f} MHz, "
                                       f"gaps over {FIT_GAP_MHZ:g} MHz joined straight"
                                       if widest > FIT_GAP_MHZ else "")),
                "band": Provenance(MEASURED, "extent of the trace segment"),
                "cn0": Provenance(MEASURED, f"ionogram power along the trace, "
                                            f"+-{SUM_HALF_BINS} bins, "
                                            f"window {ion.window}{approx}"),
                "doppler_spread": doppler_origin,
                "doppler_shift": Provenance(PRIOR, "none: a power ionogram "
                                                   "has no phase"),
                "fading": Provenance(PRIOR, "Rayleigh (ITU-R F.1487 Annex 1)"),
                "magnetoionic": Provenance(PRIOR, "not resolved on a power ionogram; "
                                                  "not split"),
            }))
    if not state.modes:
        state.notes.append(f"{method} found a trace but no segment long enough "
                           f"for a mode")
    _delay_spread(state, ion)
    return state


def load(path: str | Path, *, window: int = spectro.DEFAULT_WINDOW,
         gate_km: tuple[float, float] | None = None,
         stations: Mapping[str, tuple[float, float]] | None = None,
         cache_dir: str | Path | None = None):
    """The ionogram :func:`from_sounding` reads a state from.

    An ``.h5`` product carrying an echo window cut from an ``.lfs`` is read
    from the window: the capture's own row length and full-band noise, so the
    same ionogram the capture gives, over the window's range. The product's
    power beside it is ionozond's, formed at another row length (16384 at
    100 kHz/s) with its own noise, and the synthetic soundings the state is
    checked against are ``.lfs``.
    """
    path = Path(path)
    stations = loader.resolve_stations(stations)
    if path.suffix == ".h5":
        from .. import echo_window

        if (echo_window.has_window(path) and echo_window.read(
                path, samples=False).source == echo_window.SOURCE_LFS):
            return echo_window.reprocess_window(path, window, gate_km,
                                                stations=stations)
    return loader.load(path, window=window, gate_km=gate_km, cache_dir=cache_dir,
                       stations=stations)


def from_sounding(path: str | Path, *, method: str = "contour",
                  window: int = spectro.DEFAULT_WINDOW,
                  gate_km: tuple[float, float] | None = None,
                  stations: Mapping[str, tuple[float, float]] | None = None,
                  doppler: str | float = DEFAULT_DOPPLER_PRIOR,
                  knot_mhz: float = KNOT_MHZ,
                  cache_dir: str | Path | None = None) -> ChannelState:
    """Read a sounding of any format muf reads and describe its channel.

    ``doppler`` is the prior for what the ionogram cannot measure: an F.1487
    condition name, or a frequency spread in Hz.
    """
    ion = load(path, window=window, gate_km=gate_km, stations=stations,
               cache_dir=cache_dir)
    return from_ionogram(ion, method=method, source=Path(path).name, doppler=doppler,
                         knot_mhz=knot_mhz)
