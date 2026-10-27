"""Compare a channel simulator's soundings with real ones, as echo windows.

`docs/echo-window.md` keeps each sounding's echo band as complex voltage so
that an HF channel simulator can be built from it. This judges such a
simulator. Given real windows and simulated ones in the same format, it
measures both with the same code, and says metric by metric whether the
simulator is as close to the real channel as the real channel is to itself.

**Per sounding** -- what the ionosphere sets at that hour, compared pair by
pair. A simulated window is paired with the real sounding of its circuit that
starts within ``--pair-s`` of it.

* MUF and LOF, MHz, from the pipeline (``--method``, contour by default). A
  pair where either side is a bound -- the trace at the top or the bottom of
  the sweep -- is left out, as `muf/compare.py` leaves it out.
* Trace position, km: the RMS distance in group path between the two
  primary modes (`muf/trace.py`) over the frequencies both reach. It compares
  absolute delay, which a station's clock moves by milliseconds: trust it
  only where the station's `epoch_offset_s` health metric is ok.
* Modes: the difference in how many modes the trace splits into. A pair
  where neither side has a trace is left out: two empty soundings agree
  about nothing.

**Statistical** -- random by nature, compared as distributions. A simulator
draws its own realisation, so no one sounding should match.

* Doppler spread, Hz: the RMS width of the strongest mode's Doppler
  spectrum, by the estimator of `tools/echo_window_check.py`, along the
  stretch of sweep where its own trace holds that mode unbroken (or
  ``--band``).
* Delay spread, ms: the power-weighted RMS spread in delay of everything
  10 dB above the noise at each frequency the primary mode reaches, the
  median over those frequencies.
* Fading of the strongest mode, along the same stretch and within the
  estimator's +/-3 Hz of it: S4, the standard deviation of its intensity
  over the mean once the slow change with frequency is divided out; how many
  times a second it falls 10 dB below its median; and how long it stays
  there. A Rice K-factor follows from S4 and is in the JSON.

No Doppler *shift*. A chirp's beat is the group delay and the Doppler shift
added together, so within one sweep a shift of 1 Hz is the trace moved by
``|c / rate|`` -- 3 km at 100 kHz/s -- and the estimator's trajectory takes
it up: a 0.3 Hz shift reads as 0.0. A simulator's shift is judged inside the
trace position, which it moves by that much.

o/x splitting is not measured: these receivers have no polarimetry, and the
O/X hypothesis was rejected on these soundings (`muf/export/saoxml.py`).

**The tolerance is the real channel's own variation**, as large as it comes
in 95 % of draws at the comparison's own size. Not its typical value: a
simulator as good as the real channel is then a second draw from the same
distribution, and would fail half the time.

* Per sounding: each real sounding against the next real one of its circuit
  within ``--neighbour-s``. A metric passes when the median simulated-to-real
  difference is no further above the median real-to-real one than the
  median of as many real-to-real differences, resampled, comes above it.
* Statistical: a metric passes when the Kolmogorov-Smirnov distance between
  the simulated and the real values is no larger than the distance between
  the real soundings split at random in two reaches, scaled to the sizes
  compared.

A metric with too few values to judge says so rather than passing. The
resampling is seeded: one input gives one report.

    python tools/channel_compare.py --real real/*.echo.h5 --sim sim/*.echo.h5 \\
        --json report.json

The exit status is 1 when any metric fails.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muf import calibrate, echo_window, extractors, interference, loader, trace
from muf.pipeline import Options, process_file, read_header
from tools import echo_window_check as estimator

#: Whose MUF and LOF are compared: the method the circuit pages and the live
#: view show.
METHOD = "contour"

#: How far apart a simulated and a real sounding may start and still be a pair.
PAIR_S = 60.0

#: How far apart two real soundings may start and still be neighbours: a
#: 600 s schedule's next sounding, and some to spare.
NEIGHBOUR_S = 900.0

#: Fewer pairs, neighbours or values than this is not judged.
MIN_COUNT = 3

#: How much of the real-to-real variation a simulator may reach. Over 400
#: synthetic draws, a simulator drawn from the real channel itself failed a
#: distribution 4.5 % of the time and a per-sounding metric 7-8 %: a
#: resampled median moves in steps, and its margin comes out a little tight.
CONFIDENCE = 0.95
RESAMPLES = 2000
SEED = 0

#: Two traces overlapping by less than this in frequency are not compared.
MIN_OVERLAP_MHZ = 0.5

#: A jump in beat between traced segments worth this much group path is
#: another mode. `muf/trace.py` measured one trace scattering by 12-37 km and
#: modes 130-180 km apart.
MODE_JUMP_KM = 60.0

#: A gap between traced segments longer than this ends a stretch.
MAX_GAP_S = 1.0

#: The Doppler stretch: at least two of the estimator's Welch segments, and
#: at most the 40 s that `docs/echo-window.md` sec. 6.2 measured on.
MIN_BAND_S = 2 * estimator.WELCH_S
MAX_BAND_S = 40.0

#: The fading series' rate, Hz: the Doppler estimator's own 100 Hz.
SERIES_HZ = 100.0

#: The mode's intensity changes slowly as the sweep moves through frequency;
#: dividing by a polynomial of this degree, fitted to its log, leaves the
#: fading. A running mean short enough to follow that change also takes a
#: tenth off a Rayleigh S4; a quadratic over the stretch leaves it. In the
#: log because a fit to the intensity itself can cross zero where the mode
#: fades out, and the fading's constant bias in the log cancels in S4.
DETREND_DEGREE = 2

#: A fade is the intensity this far below its median.
FADE_DB = 10.0

#: The fading is read within this much of the mode, Hz: what the Doppler
#: estimator takes for the echo. The rest of the 100 Hz series is noise, and
#: on cyprus1 it made fades of one or two samples, 18 ms, out of a mode whose
#: 0.9 Hz spread fades over a tenth of a second.
FADING_HZ = estimator.CORE_HZ

#: The low-pass that keeps it, s: a Hamming-windowed sinc this long, half of
#: it dropped at either end rather than wrapped round. An FFT filter would
#: join the stretch's last sample to its first, and a mode is never as
#: strong at both ends.
FADING_TAPS_S = 2.0

PER_SOUNDING = {"muf_mhz": "MUF, MHz", "lof_mhz": "LOF, MHz",
                "trace_km": "Trace position, km", "modes": "Modes"}
STATISTICAL = {"doppler_rms_hz": "Doppler spread, Hz",
               "delay_spread_ms": "Delay spread, ms", "s4": "Fading S4",
               "fade_rate_hz": "Fades per second", "fade_s": "Fade length, s"}


@dataclass
class Sounding:
    """One echo window, measured."""

    path: Path
    tx: str
    rx: str
    t0: float
    values: dict = field(default_factory=dict)
    #: The primary mode as ``(MHz, km)``, or None where there is none.
    trace: tuple | None = None

    @property
    def circuit(self) -> tuple[str, str]:
        return self.tx.lower(), self.rx.lower()


# --- measuring ---------------------------------------------------------------

def measure(path: Path, method: str = METHOD, band=None) -> Sounding:
    """Every metric of one window. Raises ValueError for one the pipeline refuses."""
    options = Options(methods=(method,))
    row = process_file(path, options)
    if "error" in row:
        raise ValueError(row["error"])
    window = echo_window.read(path)
    found = Sounding(Path(path), str(row.get("tx") or ""), str(row.get("rx") or ""),
                     float(window.t0))
    modes = row.get(f"nseg_{method}")
    found.values.update(
        muf_mhz=None if row.get(f"limited_{method}") else _finite(
            row.get(f"muf_{method}")),
        lof_mhz=None if row.get(f"loflim_{method}") else _finite(
            row.get(f"lof_{method}")),
        modes=0 if modes is None and not row.get(f"err_{method}") else modes)
    ion, found.trace = primary(path, options, method, row.get("path_km"))
    found.values["delay_spread_ms"] = (
        None if found.trace is None
        else delay_spread_ms(ion, (found.trace[0].min(), found.trace[0].max())))
    found.values.update(channel(window, band))
    return found


def primary(path: Path, options: Options, method: str, path_km):
    """The window's ionogram as `process_file` forms it, and its primary mode."""
    header = read_header(path, format=options.format, stations=options.stations)
    ion = loader.load(path, window=options.window, zero_periods=options.zero_periods,
                      gate_km=options.gate_km, cache_dir=options.cache_dir,
                      format=options.format, header=header,
                      stations=options.stations)
    ion, _ = interference.apply(ion, options)
    result = extractors.run(ion, methods=(method,), **options.per_method())[method]
    if not result.ok or path_km is None:
        return ion, None
    freq, vrange, weight = trace.extract_points(ion, result)
    best = trace.primary_segment(trace.merge_branches(trace.identify_hops(
        trace.group_tracks(freq, vrange, weight), path_km)))
    if best is None:
        return ion, None
    return ion, (np.asarray(best.freq, dtype=np.float64),
                 np.asarray(best.vrange, dtype=np.float64))


def delay_spread_ms(ion, span_mhz) -> float | None:
    """The median over ``span_mhz`` of the power-weighted RMS delay spread.

    Only where the channel is open: below the LOF and above the MUF a row
    holds noise, whose odd cell above the threshold has no spread at all.
    """
    freq = np.asarray(ion.freq, dtype=np.float64)
    db = np.asarray(ion.db, dtype=np.float64)[(freq >= span_mhz[0])
                                              & (freq <= span_mhz[1])]
    km_per_ms = calibrate.C_KM_S / 1e3 * 2.0 / float(ion.header.div_coef)
    delay = np.asarray(ion.vrange, dtype=np.float64) / km_per_ms
    spreads = []
    for row in db:
        lit = row >= echo_window.NOISE_DB + echo_window.POWER_CHECK_ABOVE_NOISE_DB
        if not lit.any():
            continue
        power, at = 10.0 ** (row[lit] / 10.0), delay[lit]
        mean = np.sum(power * at) / np.sum(power)
        spreads.append(math.sqrt(np.sum(power * (at - mean) ** 2) / np.sum(power)))
    return float(np.median(spreads)) if spreads else None


def channel(window, band=None) -> dict:
    """Doppler spread and fading of the strongest mode, from the window's samples."""
    out = dict.fromkeys(("band_mhz", "doppler_rms_hz", "s4", "rice_k",
                         "fade_rate_hz", "fade_s"))
    fs, rate = window.source_sr_hz, window.rate_hz_s
    f0 = window.sweep_hz_at_t0 / 1e6
    half = window.plan.passband_hz
    beat = (window.shift_hz - half, window.shift_hz + half)
    x = echo_window.reconstruct(window)
    band = band or stretch(x, fs, rate, f0, beat, window.km_per_hz)
    if band is None:
        return out
    block = max(1, round(fs / SERIES_HZ))
    try:
        fit = estimator.trajectory(x, fs, rate, f0, band, noise_band=beat,
                                   beat_hz=beat)
        found = estimator.doppler(x, fs, rate, f0, band, fit=fit, block=block)
        series = estimator.along(x, fs, rate, f0, band, fit[0], block)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return dict(out, band_mhz=list(band))
    return dict(out, band_mhz=list(band), doppler_rms_hz=found["rms_hz"],
                **fading(series, fs / block))


def stretch(x, fs, rate, f0_mhz, beat_hz, km_per_hz) -> tuple | None:
    """The stretch of sweep, MHz, where one mode stands out unbroken.

    The estimator's own tracer, per 0.1 s: the longest run of segments at its
    SNR threshold with no gap over ``MAX_GAP_S`` and no jump in beat over
    ``MODE_JUMP_KM``, its middle ``MAX_BAND_S`` at most. None when no run
    lasts ``MIN_BAND_S``.
    """
    tseg, fb, snr = estimator._trace(x, fs, rate, beat_hz, beat_hz)
    good = np.flatnonzero(snr >= estimator.TRACE_SNR_DB)
    if good.size == 0:
        return None
    jump = MODE_JUMP_KM / abs(km_per_hz) if km_per_hz else math.inf
    runs, start = [], good[0]
    for before, now in zip(good[:-1], good[1:]):
        if tseg[now] - tseg[before] > MAX_GAP_S or abs(fb[now] - fb[before]) > jump:
            runs.append((start, before))
            start = now
    runs.append((start, good[-1]))
    lo, hi = max(((tseg[a], tseg[b]) for a, b in runs), key=lambda r: r[1] - r[0])
    if hi - lo < MIN_BAND_S:
        return None
    middle, half = (lo + hi) / 2, min(hi - lo, MAX_BAND_S) / 2
    return (f0_mhz + rate * (middle - half) / 1e6,
            f0_mhz + rate * (middle + half) / 1e6)


def fading(series: np.ndarray, fs: float) -> dict:
    """S4, Rice K, fade rate and length of one mode's complex amplitude."""
    out = dict.fromkeys(("s4", "rice_k", "fade_rate_hz", "fade_s"))
    series = np.asarray(series)
    if series.size < MIN_BAND_S * fs:
        return out
    k = np.arange(int(FADING_TAPS_S * fs) // 2 * 2 + 1)
    k -= k.size // 2
    taps = np.sinc(2.0 * FADING_HZ / fs * k) * np.hamming(k.size)
    intensity = np.abs(np.convolve(series, taps / taps.sum(), mode="valid")) ** 2
    if not np.all(intensity > 0):
        return out
    t = np.arange(intensity.size) / fs
    level = intensity / np.exp(np.polyval(np.polyfit(t, np.log(intensity),
                                                     DETREND_DEGREE), t))
    s4 = float(np.std(level) / np.mean(level))
    fades = _runs(level < np.median(level) * 10.0 ** (-FADE_DB / 10.0))
    return {"s4": s4, "rice_k": rice_k(s4),
            "fade_rate_hz": len(fades) * fs / level.size,
            "fade_s": float(np.mean([b - a for a, b in fades])) / fs if fades else None}


def rice_k(s4: float) -> float | None:
    """The Rice K-factor whose intensity has this S4: 0 for Rayleigh fading.

    From ``S4**2 = (1 + 2K) / (1 + K)**2``. None for an S4 of 0, a steady
    signal, whose K is infinite.
    """
    if s4 >= 1.0:
        return 0.0
    if s4 <= 0.0:
        return None
    q = 1.0 - s4 * s4
    return (q + math.sqrt(q)) / (s4 * s4)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# --- comparing ---------------------------------------------------------------

def paired(real: list, sim: list, within_s: float = PAIR_S) -> list:
    """Each simulated sounding with the real one of its circuit nearest in start.

    Within ``within_s``, and each real sounding taken once.
    """
    out, used = [], set()
    for s in sorted(sim, key=lambda x: x.t0):
        best = min((r for r in real if r.circuit == s.circuit and id(r) not in used
                    and abs(r.t0 - s.t0) <= within_s),
                   key=lambda r: abs(r.t0 - s.t0), default=None)
        if best is not None:
            used.add(id(best))
            out.append((best, s))
    return out


def neighbours(real: list, within_s: float = NEIGHBOUR_S) -> list:
    """Each real sounding with the next one of its circuit, within ``within_s``."""
    by = collections.defaultdict(list)
    for r in real:
        by[r.circuit].append(r)
    out = []
    for group in by.values():
        group.sort(key=lambda x: x.t0)
        out += [(a, b) for a, b in zip(group, group[1:]) if b.t0 - a.t0 <= within_s]
    return out


def difference(a: Sounding, b: Sounding, metric: str) -> float | None:
    """How far apart two soundings are in one per-sounding metric."""
    if metric == "trace_km":
        return trace_distance_km(a.trace, b.trace)
    x, y = a.values.get(metric), b.values.get(metric)
    if x is None or y is None or (metric == "modes" and not (x or y)):
        return None
    return abs(float(x) - float(y))


def trace_distance_km(one, other) -> float | None:
    """RMS group-path distance between two traces where both reach, or None."""
    if one is None or other is None:
        return None
    (fa, ra), (fb, rb) = one, other
    lo, hi = max(fa.min(), fb.min()), min(fa.max(), fb.max())
    if hi - lo < MIN_OVERLAP_MHZ:
        return None
    inside = (fa >= lo) & (fa <= hi)
    order = np.argsort(fb, kind="stable")
    there = np.interp(fa[inside], fb[order], rb[order])
    return float(np.sqrt(np.mean((ra[inside] - there) ** 2)))


def judge_pairs(pairs: list, near: list, metric: str, rng=None) -> dict:
    """One per-sounding metric: simulated against real, real against its next.

    The tolerance is the median real-to-real difference, plus as far as the
    median of as many of them as there are pairs comes above the median of
    them all in ``CONFIDENCE`` of resamplings. Both medians are draws, so
    each is resampled at its own size: a tolerance from the scatter of one
    alone fails a simulator drawn from the real channel one time in eight.
    """
    errors = [d for d in (difference(r, s, metric) for r, s in pairs) if d is not None]
    spread = [d for d in (difference(a, b, metric) for a, b in near) if d is not None]
    row = {"metric": metric, "label": PER_SOUNDING[metric], "pairs": len(errors),
           "neighbours": len(spread), "sim_vs_real": _median(errors),
           "real_vs_real": _median(spread), "tolerance": None,
           "verdict": "not judged"}
    if len(errors) >= MIN_COUNT and len(spread) >= MIN_COUNT:
        rng = rng or np.random.default_rng(SEED)
        gap = (np.median(rng.choice(spread, (RESAMPLES, len(errors))), axis=1)
               - np.median(rng.choice(spread, (RESAMPLES, len(spread))), axis=1))
        row["tolerance"] = row["real_vs_real"] + float(np.quantile(gap, CONFIDENCE))
        row["verdict"] = "pass" if row["sim_vs_real"] <= row["tolerance"] else "fail"
    return row


def judge_distributions(real: list, sim: list, metric: str, rng=None) -> dict:
    """One statistical metric: the simulated distribution against the real one.

    The tolerance is the KS distance between the real soundings split at
    random in two, as large as it comes in ``CONFIDENCE`` of splits, carried
    to the sizes compared by the ``sqrt(n m / (n + m))`` the distance scales
    with.
    """
    values = [v for v in (r.values.get(metric) for r in real) if v is not None]
    drawn = [v for v in (s.values.get(metric) for s in sim) if v is not None]
    row = {"metric": metric, "label": STATISTICAL[metric], "real": len(values),
           "sim": len(drawn), "real_median": _median(values),
           "real_iqr": _iqr(values), "sim_median": _median(drawn),
           "sim_iqr": _iqr(drawn), "ks": None, "tolerance": None,
           "verdict": "not judged"}
    if len(values) >= 2 * MIN_COUNT and len(drawn) >= MIN_COUNT:
        rng = rng or np.random.default_rng(SEED)
        values, half = np.asarray(values, float), len(values) // 2
        splits = [_scale(half, len(values) - half) * ks_distance(v[:half], v[half:])
                  for v in (rng.permutation(values) for _ in range(RESAMPLES))]
        row["ks"] = ks_distance(values, drawn)
        row["tolerance"] = float(np.quantile(splits, CONFIDENCE)
                                 / _scale(len(values), len(drawn)))
        row["verdict"] = "pass" if row["ks"] <= row["tolerance"] else "fail"
    return row


def ks_distance(a, b) -> float:
    """The two-sample Kolmogorov-Smirnov distance: the largest gap between
    the two empirical distribution functions."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    at = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, at, side="right") / a.size
                               - np.searchsorted(b, at, side="right") / b.size)))


def _scale(n: int, m: int) -> float:
    return math.sqrt(n * m / (n + m))


def _median(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def _iqr(values) -> list | None:
    return [float(v) for v in np.percentile(values, [25, 75])] if len(values) else None


def compare(real: list, sim: list, *, pair_s: float = PAIR_S,
            neighbour_s: float = NEIGHBOUR_S) -> dict:
    """The whole comparison, as the JSON report carries it."""
    pairs, near = paired(real, sim, pair_s), neighbours(real, neighbour_s)
    rng = np.random.default_rng(SEED)
    return {
        "pairs": [dict({m: difference(r, s, m) for m in PER_SOUNDING},
                       real=r.path.name, sim=s.path.name, t0=r.t0)
                  for r, s in pairs],
        "neighbours": len(near),
        "per_sounding": [judge_pairs(pairs, near, m, rng) for m in PER_SOUNDING],
        "statistical": [judge_distributions(real, sim, m, rng) for m in STATISTICAL],
    }


# --- reporting ---------------------------------------------------------------

def _num(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def describe(result: dict, real: int, sim: int) -> str:
    """The comparison as two plain-text tables."""
    lines = [f"Per sounding: {len(result['pairs'])} pairs; the tolerance from "
             f"{result['neighbours']} neighbouring real soundings",
             f"  {'metric':<20}{'pairs':>6}{'sim-real':>10}{'real-real':>11}"
             f"{'tolerance':>11}  verdict"]
    for row in result["per_sounding"]:
        lines.append(f"  {row['label']:<20}{row['pairs']:>6}"
                     f"{_num(row['sim_vs_real']):>10}{_num(row['real_vs_real']):>11}"
                     f"{_num(row['tolerance']):>11}  {row['verdict']}")
    lines += ["", f"Statistical: {real} real, {sim} simulated",
              f"  {'metric':<20}{'real median':>12}{'sim median':>12}"
              f"{'KS':>8}{'tolerance':>11}  verdict"]
    for row in result["statistical"]:
        lines.append(f"  {row['label']:<20}{_num(row['real_median']):>12}"
                     f"{_num(row['sim_median']):>12}{_num(row['ks']):>8}"
                     f"{_num(row['tolerance']):>11}  {row['verdict']}")
    return "\n".join(lines)


def _measured(paths: list, method: str, band, kind: str) -> list:
    out = []
    for path in paths:
        try:
            out.append(measure(path, method, band))
        except (OSError, ValueError, KeyError) as exc:
            print(f"skipped {kind} {path.name}: {exc}", file=sys.stderr, flush=True)
    return out


def _band(text: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not LO-HI in MHz") from None
    if not lo < hi:
        raise argparse.ArgumentTypeError(f"{text!r}: LO must be below HI")
    return lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", type=Path, nargs="+", required=True,
                    help="echo windows of real soundings")
    ap.add_argument("--sim", type=Path, nargs="+", required=True,
                    help="echo windows the simulator wrote, in the same format")
    ap.add_argument("--method", default=METHOD,
                    help="whose MUF and LOF to compare (default: %(default)s)")
    ap.add_argument("--pair-s", type=float, default=PAIR_S,
                    help="a simulated and a real sounding pair when they start "
                         "this close (default: %(default)s s)")
    ap.add_argument("--neighbour-s", type=float, default=NEIGHBOUR_S,
                    help="two real soundings are neighbours this close "
                         "(default: %(default)s s)")
    ap.add_argument("--band", type=_band, default=None, metavar="LO-HI",
                    help="the Doppler and fading stretch in MHz, for every window; "
                         "by default each window's own trace chooses it")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write every value and verdict here")
    args = ap.parse_args(argv)
    if args.method not in extractors.ALL_METHODS:
        ap.error(f"--method must be one of {', '.join(extractors.ALL_METHODS)}")

    real = _measured(args.real, args.method, args.band, "real")
    sim = _measured(args.sim, args.method, args.band, "simulated")
    result = compare(real, sim, pair_s=args.pair_s, neighbour_s=args.neighbour_s)
    print(describe(result, len(real), len(sim)))
    if args.json:
        result["soundings"] = {kind: [dict(s.values, file=s.path.name, tx=s.tx,
                                           rx=s.rx, t0=s.t0) for s in found]
                               for kind, found in (("real", real), ("sim", sim))}
        args.json.write_text(json.dumps(result, indent=1))
    verdicts = [r["verdict"] for r in result["per_sounding"] + result["statistical"]]
    return 1 if "fail" in verdicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
