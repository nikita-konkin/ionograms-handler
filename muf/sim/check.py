"""The closed loop: does the simulator reproduce what it was built from?

Three levels, as in the prototype this package grew out of:

0. **The generator does what it is told.** A mode put in at a delay comes
   out there; a spread put in is the spread measured. That is the unit tests
   (``tests/test_sim_*.py``).
1. **Recovery.** A state, sounded synthetically and read back through the
   same muf processing, gives the state again: delay curve, band, C/N0. The
   errors and their spread over realisations are what this module reports.
2. **Realism.** The real sounding the state was made from lies inside the
   spread of its synthetic soundings: its MUF among theirs, and how its level
   scatters along each mode's trace -- which the fast part of the channel
   sets and the state's slow curves do not, so a wrong Doppler prior shows
   here first. And the ionograms themselves: around the state's traces the
   real one's dB map should correlate with each synthetic one as well as two
   synthetic ones correlate with each other, which is as far as one state
   can reproduce a sounding. Every verdict is drawn one way: the real
   value's distance from the synthetic soundings' mean in their standard
   deviations, 3 at most, from 12 soundings at least.

With ``fast=True`` the fast part joins the loop: the real sounding's spread,
shape, Rice factor and o/x split are measured from its phase
(:mod:`muf.sim.fast`), sounded into every synthetic sounding, and measured
back from each the same way.

Two references put the errors in scale. Delays are also quoted as group
path, c times the delay in km -- the unit Lv et al. (Remote Sens. 14, 4133,
2022) validate their channel model in against oblique sounding. And the
**natural change** between two real soundings of the same circuit, one
repetition apart, is what the channel itself does in that time: a recovery
error well inside it is a simulator that resolves more than the channel
holds still for.

Synthetic soundings are 80 MB each and are deleted once read, unless kept.
"""

from __future__ import annotations

import itertools
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .. import interference
from . import observe, sounding
from .state import ChannelState, Mode

#: Overlap below which two modes are not the same mode, MHz.
MIN_OVERLAP_MHZ = 0.5

#: Knots weaker than this are left out of the C/N0 comparison: near the noise
#: the recovered level is the noise's, not the mode's.
CN0_COMPARE_ABOVE_DBHZ = 25.0

#: Group path per microsecond of group delay, km: c, exactly.
GROUP_KM_PER_US = 0.299792458

#: Range bins either side of a mode's trace that the ionogram correlation
#: reads: the +-2 its C/N0 is summed over and three more, so that a trace in
#: the wrong place or of the wrong width counts, not only its level.
CORRELATION_HALF_BINS = 5

#: Power is clipped here before the logarithm. The noise's mean is 0.52; a
#: cell a thousand times under it says nothing more, and would only bring the
#: logarithm's long tail into the correlation.
CORRELATION_FLOOR = 1e-3

#: The maps correlated: every row as read, fading and all, and each range
#: bin averaged over +-SMOOTH_MHZ of frequency, the resolution the state
#: keeps C/N0 at.
CORRELATION_MAPS = ("rows", "smoothed")

#: Synthetic soundings a level-2 verdict needs: their spread is its scale,
#: and the spread of three is chance. With fewer the numbers are reported
#: and no verdict is drawn.
REALISM_SOUNDINGS = 12

#: How far from its synthetic soundings, in standard deviations, a real
#: sounding may lie and still be taken for one more of them. Three, not two,
#: for the heavy tails; calibrated in docs/simulator.md, level 2.
REALISM_Z = 3.0

#: Synthetic soundings the fast part is read back from, at most: some 13 s
#: each on a 200 s sweep, against 0.65 s for all the rest of the loop.
FAST_REALISATIONS = 3


@dataclass
class ModeError:
    """One true mode against the recovered mode matched to it."""

    name: str
    matched: str | None
    overlap_mhz: float = 0.0
    delay_bias_us: float = float("nan")
    delay_rms_us: float = float("nan")
    gdd_bias_us_per_mhz: float = float("nan")
    gdd_rms_us_per_mhz: float = float("nan")
    cn0_bias_db: float = float("nan")
    cn0_rms_db: float = float("nan")
    f_lo_error_mhz: float = float("nan")
    f_hi_error_mhz: float = float("nan")


def _overlap(a: Mode, b: Mode) -> tuple[float, float]:
    return max(a.f_lo, b.f_lo), min(a.f_hi, b.f_hi)


def match(truth: ChannelState, got: ChannelState, *,
          both_strong: bool = False) -> list[ModeError]:
    """Pair each true mode with the recovered one nearest it in delay.

    Candidates must overlap it by ``MIN_OVERLAP_MHZ``; among them the one
    with the smallest median delay difference wins, and each recovered mode
    is used once.

    C/N0 is compared where the truth is over
    :data:`CN0_COMPARE_ABOVE_DBHZ` -- right for a recovery, whose truth is
    known exactly. When both are measurements, each with its own noise,
    ``both_strong`` asks both to be over it: choosing the points by one side
    alone keeps those where that side read high, and biases the difference
    its way.
    """
    out, used = [], set()
    for m in truth.modes:
        best, best_score = None, np.inf
        for g in got.modes:
            if g.name in used:
                continue
            lo, hi = _overlap(m, g)
            if hi - lo < MIN_OVERLAP_MHZ:
                continue
            f = np.linspace(lo, hi, 64)
            score = float(np.nanmedian(np.abs(g.delay_at(f) - m.delay_at(f))))
            if score < best_score:
                best, best_score = g, score
        if best is None:
            out.append(ModeError(m.name, None))
            continue
        used.add(best.name)
        lo, hi = _overlap(m, best)
        f = np.linspace(lo, hi, 128)
        d = best.delay_at(f) - m.delay_at(f)
        g = best.gdd_at(f) - m.gdd_at(f)
        strong = m.cn0_at(f) >= CN0_COMPARE_ABOVE_DBHZ
        if both_strong:
            strong &= best.cn0_at(f) >= CN0_COMPARE_ABOVE_DBHZ
        c = (best.cn0_at(f) - m.cn0_at(f))[strong]
        out.append(ModeError(
            name=m.name, matched=best.name, overlap_mhz=hi - lo,
            delay_bias_us=float(np.nanmean(d)),
            delay_rms_us=float(np.sqrt(np.nanmean(d ** 2))),
            gdd_bias_us_per_mhz=float(np.nanmean(g)),
            gdd_rms_us_per_mhz=float(np.sqrt(np.nanmean(g ** 2))),
            cn0_bias_db=float(np.mean(c)) if c.size else float("nan"),
            cn0_rms_db=float(np.sqrt(np.mean(c ** 2))) if c.size else float("nan"),
            f_lo_error_mhz=best.f_lo - m.f_lo, f_hi_error_mhz=best.f_hi - m.f_hi))
    return out


def _stats(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max())}


def _verdict(real: float, values, *, resolution: float = 0.0,
             below_only: bool = False) -> dict:
    """The real value against the synthetic ones': z, and inside or not.

    z is the real value's distance from the synthetic mean in the standard
    deviation one more synthetic value would have: the sample's, times
    sqrt(1 + 1/n) for the mean's own error. ``resolution`` is the grid the
    values are read on; rounding to it adds resolution^2/12 of variance,
    which a run of identical synthetic values would otherwise hide.
    ``below_only`` for a value that is only wrong one way. Empty under
    :data:`REALISM_SOUNDINGS` values.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size < REALISM_SOUNDINGS or not np.isfinite(real):
        return {}
    d = v - v[0]                # identical values give exactly no variance
    sd = math.sqrt((d.var(ddof=1) + resolution ** 2 / 12.0) * (1.0 + 1.0 / v.size))
    if sd == 0.0:
        return {}
    z = float((real - v[0] - d.mean()) / sd)
    return {"real_z": z,
            "real_inside": bool(z >= -REALISM_Z if below_only else abs(z) <= REALISM_Z)}


def _fast_reading(estimates, errors) -> dict:
    """The fast part each true mode's matched mode was measured with."""
    by_name = {e.mode: e for e in estimates}
    out = {}
    for err in errors:
        e = by_name.get(err.matched) if err.matched else None
        if e is None or not e.used:
            continue
        out[err.name] = {"spread_hz": float(e.spread_hz), "shape": e.shape,
                         "bound": e.bound, "rice_k_db": e.rice_k_db,
                         "split_us": float(e.split_us), "bands": e.used}
    return out


def _fast_summary(m: Mode, rows: list[dict]) -> dict:
    values = [r for r in rows if r]
    return {"truth": {"spread_hz": m.doppler_spread_hz, "shape": m.doppler_shape,
                      "rice_k_db": m.rician_k_db, "split_us": m.ox_split_us},
            "measured": len(values),
            "spread_hz": _stats(r["spread_hz"] for r in values if not r["bound"]),
            "same_shape": sum(1 for r in values if r["shape"] == m.doppler_shape),
            "split_us": _stats(r["split_us"] for r in values)}


def trace_mask(modes, ion, *, half_bins: int = CORRELATION_HALF_BINS) -> np.ndarray:
    """Cells within ``half_bins`` range bins of the modes' traces, over their bands."""
    freq, vrange = np.asarray(ion.freq), np.asarray(ion.vrange)
    scale = observe.km_per_us(ion.header)
    mask = np.zeros((freq.size, vrange.size), dtype=bool)
    for m in modes:
        rows = np.flatnonzero((freq >= m.f_lo) & (freq <= m.f_hi))
        if rows.size == 0:
            continue
        km = m.delay_at(freq[rows]) * scale
        centres = np.abs(vrange[None, :] - km[:, None]).argmin(axis=1)
        for i, j in zip(rows, centres):
            mask[i, max(0, j - half_bins):j + half_bins + 1] = True
    return mask


def _along_frequency(power: np.ndarray, freq: np.ndarray,
                     half_mhz: float) -> np.ndarray:
    """Each range bin's mean over +-``half_mhz``; fewer rows at the edges."""
    h = round(half_mhz / float(np.median(np.abs(np.diff(freq)))))
    total = np.cumsum(np.pad(power, ((h + 1, h), (0, 0))), axis=0)
    count = np.cumsum(np.pad(np.ones(power.shape[0]), (h + 1, h)))
    width = 2 * h + 1
    return (total[width:] - total[:-width]) / (count[width:] - count[:-width])[:, None]


def cleaned(ion):
    """The ionogram as the extractor reads it, and the cells taken out of it."""
    from ..pipeline import Options

    clean, _ = interference.apply(ion, Options())
    return clean, np.asarray(clean.power) != np.asarray(ion.power)


def same_grid(a, b) -> bool:
    return (np.shape(a.power) == np.shape(b.power)
            and np.allclose(a.freq, b.freq) and np.allclose(a.vrange, b.vrange))


def gate_of(ion) -> tuple[float, float]:
    """The range ``ion`` covers, a quarter bin wide each side, as a load gate.

    A synthetic ``.lfs`` loads at the capture's default gate. An echo window
    is narrower, so its synthetic copies are loaded at this gate to land on
    its axis bin for bin; for an ``.lfs`` it is the default gate again.
    """
    vrange = np.asarray(ion.vrange, dtype=np.float64)
    pad = 0.25 * float(np.median(np.abs(np.diff(vrange))))
    return float(vrange.min()) - pad, float(vrange.max()) + pad


def db_maps(ion) -> dict[str, np.ndarray]:
    """The whole ionogram in dB, as each of :data:`CORRELATION_MAPS`."""
    power = np.asarray(ion.power, dtype=np.float64)
    maps = {"rows": power,
            "smoothed": _along_frequency(power, np.asarray(ion.freq),
                                         observe.SMOOTH_MHZ)}
    return {key: 10.0 * np.log10(np.maximum(v, CORRELATION_FLOOR))
            for key, v in maps.items()}


def correlate(a: dict, b: dict, mask: np.ndarray) -> dict[str, float]:
    """Pearson's r of two ionograms' :func:`db_maps` inside ``mask``."""
    return {key: float(np.corrcoef(a[key][mask], b[key][mask])[0, 1])
            for key in CORRELATION_MAPS}


def _masks(state: ChannelState, ion, taken: np.ndarray) -> dict[str, np.ndarray]:
    """All the state's traces together (``""``), and each mode's by its name."""
    masks = {"": trace_mask(state.modes, ion) & ~taken}
    masks.update({m.name: trace_mask([m], ion) & ~taken for m in state.modes})
    return {name: mask for name, mask in masks.items()
            if np.count_nonzero(mask) >= 2}


def ionogram_correlation(state: ChannelState, ion_a, ion_b) -> dict | None:
    """r of two ionograms around ``state``'s traces; None if not comparable.

    Both are cleaned as the extractor cleans them, and cells either loses to
    the cleaning are left out. All the traces together at the top level, each
    mode's own under ``modes``.
    """
    if not same_grid(ion_a, ion_b):
        return None
    (clean_a, taken_a), (clean_b, taken_b) = cleaned(ion_a), cleaned(ion_b)
    masks = _masks(state, ion_a, taken_a | taken_b)
    if "" not in masks:
        return None
    a, b = db_maps(clean_a), db_maps(clean_b)
    out = correlate(a, b, masks.pop(""))
    out["modes"] = {name: correlate(a, b, mask) for name, mask in masks.items()}
    return out


def compare_maps(real, synthetic: list, mask: np.ndarray) -> dict:
    """Real against each synthetic map, and the synthetic pairs, in one mask.

    The verdict is leave-one-out. Each synthetic sounding's mean r with the
    others is what one more realisation gets; the real one's mean r with all
    of them is set against their spread (:func:`_verdict`). Its fading is
    common to all its pairs, so it is these means that are alike if the
    state is complete, not the real one's mean and single pairs. One-sided:
    a real sounding more like the copies than they are like each other is
    no defect.
    """
    n = len(synthetic)
    pairs = {(j, k): correlate(synthetic[j], synthetic[k], mask)
             for j, k in itertools.combinations(range(n), 2)}
    against = [correlate(real, s, mask) for s in synthetic] if real is not None else []
    out = {"cells": int(np.count_nonzero(mask))}
    for key in CORRELATION_MAPS:
        entry = {"synthetic": _stats(p[key] for p in pairs.values())}
        if against:
            entry["real"] = _stats(r[key] for r in against)
            # In Fisher's z = atanh r: r near 1 is squeezed against its bound,
            # and its scatter is skewed there.
            each = [np.mean([np.arctanh(pairs[min(j, k), max(j, k)][key])
                             for k in range(n) if k != j]) for j in range(n)]
            real_z = float(np.mean([np.arctanh(r[key]) for r in against]))
            entry.update(_verdict(real_z, each, below_only=True))
        out[key] = entry
    if against:
        out["per_run"] = against
    return out


def _correlation_summary(truth: ChannelState, real, synthetic: list) -> dict:
    """Level 2 on the ionograms: real against synthetic, and synthetic pairs.

    The pairs are the ceiling: two soundings of one state differ only by
    fading and noise, so a real sounding the state described fully would
    correlate with them as they do with each other. Cells the cleaning takes
    out of any of them are left out of all.
    """
    grid = real if real is not None else synthetic[0]
    if any(not same_grid(grid, ion) for ion in synthetic):
        return {"skipped": "the synthetic ionograms' grid is not the real one's"}
    clean = [cleaned(ion) for ion in synthetic]
    if real is not None:
        clean.append(cleaned(real))
    taken = np.logical_or.reduce([t for _, t in clean])
    maps = [db_maps(c) for c, _ in clean]
    real_maps = maps.pop() if real is not None else None
    masks = _masks(truth, grid, taken)
    out = {"soundings": len(synthetic), "half_bins": CORRELATION_HALF_BINS,
           "smooth_mhz": observe.SMOOTH_MHZ}
    if "" not in masks:
        return {"skipped": "no cells around the traces", **out}
    out.update(compare_maps(real_maps, maps, masks.pop("")))
    out["modes"] = {name: compare_maps(real_maps, maps, mask)
                    for name, mask in masks.items()}
    return out


def closed_loop(truth: ChannelState, *, realisations: int = REALISM_SOUNDINGS,
                seed: int = 0, method: str = "contour",
                workdir: str | Path | None = None, keep: bool = False,
                real_muf_mhz: float | None = None, fast: bool = False,
                fast_realisations: int = FAST_REALISATIONS, real=None,
                synthetic: Sequence = (),
                progress: Callable[[int, int], None] | None = None) -> dict:
    """Sound ``truth`` ``realisations`` times, read each back, and compare.

    ``real``, the ionogram ``truth`` was read from, is correlated with every
    synthetic ionogram, each loaded on its range axis (:func:`gate_of`);
    without it only the synthetic pairs are. Level 2's verdicts need
    :data:`REALISM_SOUNDINGS`. With ``fast``, the fast part is read back from
    the first ``fast_realisations``.

    ``synthetic`` are synthetic ionograms of ``truth`` already in hand -- the
    copies a forecast of the same curves was scored with -- seeds ``seed``
    on and loaded on ``real``'s axis; only the rest are sounded. Their files
    are gone, so the fast part cannot be read back from them.

    ``progress`` hears (realisations read back, realisations) before the
    first and after each.
    """
    if truth.sweep is None:
        raise ValueError("the state has no sweep to sound")
    if fast and synthetic:
        raise ValueError("the fast part is read from the synthetic files, and "
                         "synthetic ionograms in hand have none")
    own = workdir is None
    folder = Path(tempfile.mkdtemp(prefix="muf-sim-")) if own else Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    gate = gate_of(real) if real is not None else None
    runs, ionograms = [], []
    if progress is not None:
        progress(0, realisations)
    try:
        for k in range(realisations):
            path = folder / sounding.default_name(truth, tag=f"sim{k:02d}")
            if k < len(synthetic):
                ion = synthetic[k]
            else:
                sounding.write_lfs(truth, path, seed=seed + k)
                ion = observe.load(path, gate_km=gate)
            ionograms.append(ion)
            got = observe.from_ionogram(ion, method=method, source=path.name)
            errors = match(truth, got)
            levels = got.metrics.get("levels", {})
            widths = got.metrics.get("cross_trace", {})
            run = {"file": path.name, "seed": seed + k,
                   "muf_mhz": got.metrics.get("muf_mhz", float("nan")),
                   "modes": [asdict(e) for e in errors],
                   "levels": {e.name: levels.get(e.matched) for e in errors
                              if e.matched},
                   "widths": {e.name: widths.get(e.matched) for e in errors
                              if e.matched},
                   "recovered": [m.name for m in got.modes]}
            if fast and k < fast_realisations:
                from . import fast as fast_part

                run["fast"] = _fast_reading(
                    fast_part.estimate(fast_part.record_of(path), got), errors)
            runs.append(run)
            if not keep and path.exists():
                path.unlink()
            if progress is not None:
                progress(k + 1, realisations)
    finally:
        if own and not keep:
            shutil.rmtree(folder, ignore_errors=True)

    summary = {}
    real_levels = truth.metrics.get("levels", {})
    for m in truth.modes:
        rows = [next(e for e in r["modes"] if e["name"] == m.name) for r in runs]
        summary[m.name] = {
            "found": sum(1 for e in rows if e["matched"]),
            **{key: _stats(e[key] for e in rows)
               for key in ("delay_bias_us", "delay_rms_us", "cn0_bias_db",
                           "f_lo_error_mhz", "f_hi_error_mhz")},
        }
        synth = [(r["levels"].get(m.name) or {}).get("std_db", float("nan"))
                 for r in runs]
        level = {"synthetic_std_db": _stats(synth)}
        if m.name in real_levels:
            real_std = float(real_levels[m.name]["std_db"])
            level["real_std_db"] = real_std
            if level["synthetic_std_db"].get("n"):
                level["real_offset_db"] = real_std - level["synthetic_std_db"]["mean"]
                level.update(_verdict(real_std, synth))
        summary[m.name]["level"] = level
        read = [(r["widths"].get(m.name) or {}).get("spread_us", float("nan"))
                for r in runs]
        width = {"delay_spread_us": m.delay_spread_us, "synthetic_us": _stats(read)}
        real_width = truth.metrics.get("cross_trace", {}).get(m.name)
        if real_width is not None:
            width["real_us"] = float(real_width["spread_us"])
            width["real_centre_r1"] = float(real_width["centre_r1"])
            width.update(_verdict(width["real_us"], read))
        summary[m.name]["width"] = width
        if fast:
            summary[m.name]["fast"] = _fast_summary(
                m, [r["fast"].get(m.name) for r in runs if "fast" in r])
    mufs = _stats(r["muf_mhz"] for r in runs)
    report = {"source": truth.source, "method": method, "realisations": realisations,
              "seed": seed, "modes": summary, "muf_mhz": mufs, "runs": runs,
              "kept": str(folder) if keep else None}
    if truth.modes and ionograms:
        report["correlation"] = _correlation_summary(truth, real, ionograms)
    if realisations < REALISM_SOUNDINGS:
        report["note"] = (f"level 2 draws no verdict from fewer than "
                          f"{REALISM_SOUNDINGS} synthetic soundings")
    if real_muf_mhz is not None and np.isfinite(real_muf_mhz) and mufs.get("n"):
        report["real_muf_mhz"] = float(real_muf_mhz)
        report["real_muf_offset_mhz"] = float(real_muf_mhz - mufs["mean"])
        # The MUF is read on the ionogram's rows.
        row_mhz = float(np.median(np.abs(np.diff(ionograms[0].freq))))
        muf = _verdict(real_muf_mhz, [r["muf_mhz"] for r in runs], resolution=row_mhz)
        if muf:
            report["real_muf_z"] = muf["real_z"]
            report["real_muf_inside"] = muf["real_inside"]
    return report


def persistence(path_a: str | Path, path_b: str | Path, *,
                method: str = "contour") -> dict:
    """What the channel itself changes between two real soundings.

    ``path_b`` is matched against ``path_a`` as a recovery would be, so the
    numbers are directly comparable with :func:`closed_loop`'s: the natural
    change over one repetition. ``path_b`` is loaded on ``path_a``'s range
    axis, as the synthetic soundings are.
    """
    ion_a = observe.load(path_a)
    ion_b = observe.load(path_b, gate_km=gate_of(ion_a))
    a = observe.from_ionogram(ion_a, method=method, source=Path(path_a).name)
    b = observe.from_ionogram(ion_b, method=method, source=Path(path_b).name)
    errors = match(a, b)
    out = {"from": a.source, "to": b.source, "dt_s": float(b.epoch - a.epoch),
           "modes": {e.name: asdict(e) for e in errors}}
    r = ionogram_correlation(a, ion_a, ion_b) if a.modes else None
    if r is not None:
        out["correlation"] = r
    muf_a, muf_b = a.metrics.get("muf_mhz"), b.metrics.get("muf_mhz")
    if muf_a is not None and muf_b is not None:
        out["muf_change_mhz"] = float(muf_b - muf_a)
    return out


def check_sounding(path: str | Path, *, next_sounding: str | Path | None = None,
                   fast: bool = False, **options) -> tuple[ChannelState, dict]:
    """Level 1 and 2 for one real sounding: its state, and the loop's report.

    ``next_sounding``, the circuit's following sounding, adds the natural
    change between the two as the scale the recovery errors are read on.
    ``fast`` measures the state's fast part from the sounding's phase first,
    and reads it back from every synthetic sounding. A sounding the reader
    finds no trace in is refused before the loop, which would recover nothing.
    """
    method = options.get("method", "contour")
    real = observe.load(path)
    truth = observe.from_ionogram(real, method=method, source=Path(path).name)
    if not truth.modes:
        raise ValueError("the sounding has no trace to simulate")
    if fast:
        from . import fast as fast_part

        fast_part.measure(path, truth)
    report = closed_loop(truth, real_muf_mhz=truth.metrics.get("muf_mhz"), fast=fast,
                         real=real, **options)
    if next_sounding is not None:
        report["natural"] = persistence(path, next_sounding, method=method)
        if fast:
            from . import fast as fast_part

            later = observe.from_sounding(next_sounding, method=method)
            errors = match(truth, later)
            report["natural"]["fast"] = _fast_reading(
                fast_part.estimate(fast_part.record_of(next_sounding), later), errors)
    return truth, report


def _spread(stats: dict, unit: str, digits: int = 1) -> str:
    if not stats.get("n"):
        return "-"
    return f"{stats['mean']:+.{digits}f} +- {stats['std']:.{digits}f} {unit}"


def _km(stats: dict) -> str:
    if not stats.get("n"):
        return ""
    return f" ({stats['mean'] * GROUP_KM_PER_US:.1f} km group path)"


def _where(entry: dict, z: str = "real_z", inside: str = "real_inside") -> str:
    if inside not in entry:
        return ""
    return (f" ({entry[z]:+.1f} sd) "
            + ("inside" if entry[inside] else "OUTSIDE"))


def _r(stats: dict) -> str:
    if not stats.get("n"):
        return "-"
    return f"{stats['mean']:.3f} +- {stats['std']:.3f}"


def _correlation_labels(corr: dict) -> dict[str, str]:
    return {"rows": "row by row", "smoothed": f"over +-{corr['smooth_mhz']:g} MHz"}


def _natural_correlation(report: dict) -> dict:
    return (report.get("natural") or {}).get("correlation") or {}


def _correlation_lines(report: dict) -> list[str]:
    corr = report.get("correlation")
    if not corr:
        return []
    if "skipped" in corr:
        return [f"  ionogram correlation skipped: {corr['skipped']}"]
    natural = _natural_correlation(report)
    labels = _correlation_labels(corr)
    lines = [f"  ionogram correlation over {corr['soundings']} synthetic soundings, "
             f"dB within +-{corr['half_bins']} range bins of the traces "
             f"({corr['cells']} cells):"]
    for key in CORRELATION_MAPS:
        e = corr[key]
        line = f"    {labels[key]:<16} synthetic pairs {_r(e['synthetic'])}"
        if "real" in e:
            line += f", real {_r(e['real'])}{_where(e)}"
        if natural:
            line += f"; real to the next sounding {natural[key]:.3f}"
        lines.append(line)
    return lines


def _mode_correlation_lines(report: dict, name: str) -> list[str]:
    corr = report.get("correlation") or {}
    mode = corr.get("modes", {}).get(name)
    if not mode:
        return []
    natural = _natural_correlation(report).get("modes", {}).get(name)
    labels = _correlation_labels(corr)
    parts = []
    for key in CORRELATION_MAPS:
        e = mode[key]
        st = e["synthetic"]
        part = f"{labels[key]} " + (f"{st['mean']:.3f}" if st.get("n") else "-")
        if "real" in e:
            part += f" / {e['real']['mean']:.3f}"
            if "real_inside" in e:
                part += " inside" if e["real_inside"] else " OUTSIDE"
        if natural:
            part += f" / next {natural[key]:.3f}"
        parts.append(part)
    who = "synthetic pairs / real" + (" / next" if natural else "")
    return [f"  {'':<10} ionogram r ({mode['cells']} cells), {who}: "
            + ", ".join(parts)]


def format_report(report: dict) -> str:
    lines = [f"{report['source']}: {report['realisations']} synthetic soundings, "
             f"method {report['method']}"]
    for name, s in report["modes"].items():
        lines.append(f"  {name:<10} found {s['found']}/{report['realisations']}  "
                     f"delay bias {_spread(s['delay_bias_us'], 'us')}  "
                     f"rms {_spread(s['delay_rms_us'], 'us')}{_km(s['delay_rms_us'])}  "
                     f"C/N0 {_spread(s['cn0_bias_db'], 'dB')}  "
                     f"f_hi {_spread(s['f_hi_error_mhz'], 'MHz', 3)}")
        level = s.get("level", {})
        synth = level.get("synthetic_std_db", {})
        if synth.get("n") or "real_std_db" in level:
            line = f"  {'':<10} level scatter along the trace: synthetic "
            line += (f"{synth['mean']:.2f} +- {synth['std']:.2f} dB" if synth.get("n")
                     else "-")
            if "real_std_db" in level:
                line += f", real {level['real_std_db']:.2f} dB{_where(level)}"
            lines.append(line)
        wd = s.get("width", {})
        if wd.get("synthetic_us", {}).get("n") or "real_us" in wd:
            st = wd["synthetic_us"]
            line = (f"  {'':<10} trace width over a point echo: synthetic "
                    + (f"{st['mean']:.1f} +- {st['std']:.1f} us" if st.get("n")
                       else "-"))
            if "real_us" in wd:
                line += (f", real {wd['real_us']:.1f} us{_where(wd)}, centres "
                         f"correlated {wd['real_centre_r1']:+.2f} row to row")
            line += f"; delay spread in the state {wd['delay_spread_us']:.1f} us"
            lines.append(line)
        lines += _mode_correlation_lines(report, name)
        fp = s.get("fast")
        if fp:
            t = fp["truth"]
            sp = fp["spread_hz"]
            read = f"{sp['mean']:.3f} +- {sp['std']:.3f} Hz" if sp.get("n") else "-"
            line = (f"  {'':<10} fast part: spread {t['spread_hz']:.3f} Hz "
                    f"{t['shape']} read {read}, "
                    f"shape {fp['same_shape']}/{fp['measured']}")
            if t["split_us"] is not None:
                st = fp["split_us"]
                line += (f"; o/x split {t['split_us']:.1f} us read "
                         + (f"{st['mean']:.1f} +- {st['std']:.1f} us" if st.get("n")
                            else "-"))
            lines.append(line)
    mu = report["muf_mhz"]
    if mu.get("n"):
        line = (f"  MUF synthetic {mu['mean']:.3f} +- {mu['std']:.3f} MHz "
                f"({mu['min']:.3f}-{mu['max']:.3f})")
        if "real_muf_mhz" in report:
            line += (f"; real {report['real_muf_mhz']:.3f} MHz"
                     + _where(report, "real_muf_z", "real_muf_inside"))
        lines.append(line)
    lines += _correlation_lines(report)
    if "note" in report:
        lines.append(f"  {report['note']}")
    natural = report.get("natural")
    if natural:
        lines.append(f"  natural change, {natural['from']} -> {natural['to']} "
                     f"({natural['dt_s']:.0f} s):")
        for name, e in natural["modes"].items():
            if not e["matched"]:
                lines.append(f"    {name:<10} not found in the next sounding")
                continue
            ratio = ""
            rec = report["modes"].get(name, {}).get("delay_rms_us", {})
            if rec.get("n") and e["delay_rms_us"] > 0:
                ratio = f"; recovery rms is {rec['mean'] / e['delay_rms_us']:.2f} of it"
            lines.append(f"    {name:<10} delay rms {e['delay_rms_us']:.1f} us "
                         f"({e['delay_rms_us'] * GROUP_KM_PER_US:.1f} km), "
                         f"C/N0 {e['cn0_bias_db']:+.1f} dB{ratio}")
        if "muf_change_mhz" in natural:
            lines.append(f"    MUF {natural['muf_change_mhz']:+.3f} MHz")
        for name, r in natural.get("fast", {}).items():
            lines.append(f"    {name:<10} fast part next: spread "
                         f"{r['spread_hz']:.3f} Hz {r['shape']}"
                         + (f", o/x split {r['split_us']:.1f} us"
                            if np.isfinite(r["split_us"]) else ""))
    return "\n".join(lines)
