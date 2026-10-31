"""A frozen benchmark of the simulator against real soundings: ``muf sim bench``.

Three questions are kept apart, as ``docs/simulator-quality-plan.md`` has
them:

- **Reconstruction.** A real sounding's state, synthesised at its own time,
  against the sounding itself, cell for cell: the level at each mode's ridge
  and the energy around it (:func:`levels`), and which echoes either side
  has that the other lacks (:func:`coverage`).
- **Recovery.** The same synthetic soundings read back, against the state
  they were made from: level 1 of :func:`muf.sim.check.closed_loop`, with
  its level 2 correlation of the real sounding with them.
- **Forecast.** A sounding's persisted state a lead later, against the
  sounding then (:func:`muf.sim.series.run_series`), split by split.

A benchmark is a manifest frozen once (:func:`freeze`): every input by its
path under a root and its SHA-256, its circuit, time, format and the phase
it carries (:func:`entry`); the whole UTC days each split holds; the
diagnostic soundings; and every parameter, seeds included. A run
(:func:`run`) refuses an input whose bytes changed, leaves the final split
alone unless asked, and writes its rows with a digest over all of them but
their timings, so two runs of one manifest are one result.

The coverage is read off a threshold, :data:`ECHO_DB`, not off the traces
the extractor found, so it can see an echo the extractor missed. It is not
a reviewed echo mask: a person still has to say which cells are echoes.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .. import calibrate, echo_window, io_lfs, loader, spectro
from . import check, clock, live, observe, series, summary
from . import forecast as fc
from .state import ChannelState

MANIFEST_SCHEMA = "muf.sim.bench.manifest"
DIAGNOSIS_SCHEMA = "muf.sim.bench.diagnosis"
RUN_SCHEMA = "muf.sim.bench.run"
VERSION = 1

#: The splits a day can be in; ``final`` is scored only when asked.
SPLITS = ("tune", "validation", "final")

#: How a benchmark observes and sounds, unless its manifest says otherwise.
#: ``leads_s`` None is the circuit's next two soundings, as ``muf sim live``.
PARAMS = {**series.OBSERVE, "copies": check.REALISM_SOUNDINGS, "seed": series.SEED,
          "forecast_copies": 0, "leads_s": None, "tolerance_s": series.TOLERANCE_S}

#: Range bins either side of a mode's curve its ridge is the brightest of:
#: the strip the observer sums its C/N0 over.
PEAK_HALF_BINS = observe.SUM_HALF_BINS

#: Range bins either side of a mode's curve its energy is summed over, wide
#: enough to hold an echo the strip would clip.
ENERGY_HALF_BINS = 3 * observe.SUM_HALF_BINS

#: dB on the shared scale a cell must reach to count as an echo: the
#: detection threshold, 13 dB over the noise.
ECHO_DB = echo_window.DETECTION_DB

#: dB over the noise mean a smoothed map must reach for the picture to show
#: its difference.
QUIET_DB = 6.0

#: A coverage block: the observer's knot spacing by its five-bin strip.
COVER_MHZ = observe.KNOT_MHZ
COVER_BINS = 2 * observe.SUM_HALF_BINS + 1

#: The share of a sweep's samples a sounding must hold to be benchmarked.
COMPLETE = 0.99

CHUNK = 1 << 20


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _utc(epoch: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc)


def _plain(value):
    """``value`` as strict JSON holds it: non-finite numbers as None."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def phase_of(path: Path, header, fmt: str) -> str:
    """What of the sounding's voltage the file keeps: ``lfs`` all of it, ``z``
    v2's whole dechirped record, ``echo window`` the part around the echoes,
    ``power`` none."""
    if fmt == loader.LFS:
        return "lfs"
    if echo_window.has_window(path):
        return "echo window"
    if getattr(header, "has_raw_voltage", False):
        return "z"
    return "power"


def _window(path: Path) -> dict:
    found = echo_window.read(path, samples=False)
    return {"source": found.source, "decimation": found.decimation,
            "sr_hz": found.sr_hz, "source_sr_hz": found.source_sr_hz,
            "source_window": found.source_window,
            "range_km": [found.range_lo_km, found.range_hi_km],
            "selection": found.selection.get("method"),
            "truncated_suspect": found.truncated_suspect,
            "trace_at_edge": found.trace_at_edge, "writer": found.writer}


def _expected(header) -> float:
    return float(header.dur) * float(header.sample_rate) / float(header.dec)


def sweep_fraction(path: Path, fmt: str) -> float | None:
    """The share of its sweep's samples the sounding holds; None for a
    product that keeps no voltage to count."""
    if fmt == loader.LFS:
        return io_lfs.n_samples(path) / _expected(io_lfs.read_header(path))
    if echo_window.has_window(path):
        found = echo_window.read(path, samples=False)
        if found.lfs_header is not None:
            header = io_lfs.parse_header(found.lfs_header, path)
            return found.source_samples / _expected(header)
    return None


def _described(path: Path) -> dict:
    fmt = loader.format_of(path)
    fraction = sweep_fraction(path, fmt)
    if fraction is not None and fraction < COMPLETE:
        raise ValueError(f"it holds {fraction:.1%} of the sweep's samples")
    header = loader.read_header(path)
    epoch = observe.epoch_of(header)
    start, stop = calibrate.sweep_bounds(header)
    out = {"format": fmt, "tx": header.tx_name, "rx": header.rx_name,
           "epoch": epoch, "utc": _utc(epoch).strftime("%Y-%m-%d %H:%M:%S"),
           "day": _utc(epoch).strftime("%Y-%m-%d"),
           "phase": phase_of(path, header, fmt),
           "sweep": {"start_mhz": start, "stop_mhz": stop,
                     "rate_hz_s": float(header.rate),
                     "sample_rate_hz": float(header.sample_rate) / float(header.dec)},
           "range_km": [float(header.rmin), float(header.rmax)],
           "sweep_fraction": fraction}
    if out["phase"] == "echo window":
        out["echo_window"] = _window(path)
    if fmt == loader.CHIRP2:
        out.update(range_offset_applied=bool(header.range_offset_applied),
                   range_relative=header.range_relative_reason or None,
                   git_dirty=header.git_dirty)
    return out


def entry(path: str | Path, root: str | Path) -> dict:
    """One input as the manifest lists it; ``reason`` says why one cannot be
    benchmarked: it does not read, or its sweep is cut short."""
    path = Path(path)
    out = {"file": path.name, "path": _relative(path, Path(root)),
           "bytes": path.stat().st_size, "sha256": sha256(path)}
    try:
        out.update(_described(path))
    except Exception as exc:
        out["reason"] = _failure(exc)
    return _plain(out)


def _files(paths: Iterable) -> list[Path]:
    found = set()
    for item in paths:
        item = Path(item)
        found.update(p.resolve() for p in
                     (loader.find_soundings(item) if item.is_dir() else [item]))
    return sorted(found)


def inventory(paths: Iterable, root: str | Path) -> list[dict]:
    """Every sounding under ``paths``, in time order, those set aside last."""
    rows = [entry(p, root) for p in _files(paths)]
    return sorted(rows, key=lambda e: ("reason" in e, e.get("epoch", 0.0), e["file"]))


def _once(entries: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """One input for each sounding, and the rest set aside.

    A station may keep a sounding twice, as its capture and as the product
    cut from it; the product is kept, being what the live loop reads.
    """
    by_time: dict[int, list[dict]] = {}
    for e in entries:
        by_time.setdefault(round(e["epoch"]), []).append(e)
    kept, twins = [], []
    for group in by_time.values():
        first, *rest = sorted(group,
                              key=lambda e: (e["format"] == loader.LFS, e["path"]))
        kept.append(first)
        twins.extend({**e, "reason": f"the same sounding as {first['file']}"}
                     for e in rest)
    return sorted(kept, key=lambda e: (e["epoch"], e["file"])), twins


def splits_of(named: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """Each UTC day, ``YYYY-MM-DD``, by the split ``named`` puts it in."""
    unknown = sorted(set(named) - set(SPLITS))
    if unknown:
        raise ValueError(f"no split {', '.join(unknown)}; there are "
                         f"{', '.join(SPLITS)}")
    days: dict[str, str] = {}
    for split in SPLITS:
        for day in named.get(split, ()):
            if day in days:
                raise ValueError(f"{day} is in both {days[day]} and {split}")
            days[day] = split
    return days


def assign(entries: Sequence[dict], days: Mapping[str, str]) -> list[dict]:
    """The entries, each with the split of its day; None in no split."""
    return [{**e, "split": days.get(e.get("day"))} for e in entries]


def _circuit(entries: Sequence[dict], circuit: str | None) -> str:
    names = sorted({f"{e['tx']}:{e['rx']}" for e in entries})
    if circuit is not None:
        return circuit
    if len(names) != 1:
        raise ValueError(f"soundings of {len(names)} circuits "
                         f"({', '.join(names) or 'none'}): choose one as tx:rx")
    return names[0]


def _leads(entries: Sequence[dict]) -> list[float]:
    """The circuit's next two soundings, as ``muf sim live`` leads, at the
    median of all its gaps: a run of the last few may be the archive's
    scattered extras."""
    gaps = np.diff(np.sort([e["epoch"] for e in entries]))
    return [] if gaps.size == 0 else [k * float(np.median(gaps)) for k in live.STEPS]


def freeze(paths: Iterable, root: str | Path, out: str | Path, *,
           splits: Mapping[str, Iterable[str]], diagnostic: Sequence[str] = (),
           params: Mapping | None = None, circuit: str | None = None) -> dict:
    """The benchmark of the soundings under ``paths``, written to
    ``out/manifest.json``.

    Paths are kept against ``root``, so the archive can move. ``splits``
    names the UTC days of each of :data:`SPLITS`; ``diagnostic`` the files
    reconstructed one by one. Only ``circuit``'s soundings are kept, the
    only circuit's when None; a sounding kept twice is benchmarked once
    (:func:`_once`). What is not benchmarked is listed in ``set_aside`` with
    its reason.
    """
    rows = inventory(paths, root)
    readable = [e for e in rows if "reason" not in e]
    chosen = _circuit(readable, circuit)
    kept, twins = _once([e for e in readable if f"{e['tx']}:{e['rx']}" == chosen])
    names = {e["file"] for e in kept}
    missing = [n for n in diagnostic if n not in names]
    if missing:
        raise ValueError(f"{', '.join(missing)}: not in the manifest")
    days = splits_of(splits)
    merged = {**PARAMS, **(params or {})}
    if merged["leads_s"] is None:
        merged["leads_s"] = _leads(kept)
    manifest = _plain({
        "schema": MANIFEST_SCHEMA, "version": VERSION,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "muf": series.code_version(), "circuit": chosen, "params": merged,
        "splits": {s: sorted(d for d, k in days.items() if k == s) for s in SPLITS},
        "diagnostic": list(diagnostic), "entries": assign(kept, days),
        "set_aside": [*twins, *(e for e in rows if "reason" in e)]})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False)
                                       + "\n", encoding="utf-8", newline="\n")
    return manifest


def check_inputs(manifest: Mapping, root: str | Path) -> None:
    """Refuse a manifest whose inputs are gone or changed since it was frozen."""
    for e in manifest["entries"]:
        path = Path(root) / e["path"]
        if not path.is_file():
            raise ValueError(f"{e['path']}: not under {root}")
        if sha256(path) != e["sha256"]:
            raise ValueError(f"{e['path']}: changed since the manifest was frozen")


def _centres(mode, ion) -> tuple[np.ndarray, np.ndarray]:
    """The rows of ``mode``'s band, and the range bin its curve is in at each."""
    freq, vrange = np.asarray(ion.freq), np.asarray(ion.vrange)
    rows = np.flatnonzero((freq >= mode.f_lo) & (freq <= mode.f_hi))
    km = mode.delay_at(freq[rows]) * observe.km_per_us(ion.header)
    return rows, np.abs(vrange[None, :] - km[:, None]).argmin(axis=1)


def _cleaned(real, copies) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """The powers as the extractor reads them, and every cell any cleaning took."""
    found = [check.cleaned(ion) for ion in (real, *copies)]
    taken = np.logical_or.reduce([t for _, t in found])
    powers = [np.asarray(c.power, dtype=np.float64) for c, _ in found]
    return powers[0], powers[1:], taken


def _strips(centres: np.ndarray, others: list[np.ndarray], half: int, size: int,
            taken_rows: np.ndarray) -> np.ndarray:
    """Rows whose strip lies inside the axis, clear of taken cells and of
    every other mode's strip."""
    keep = (centres - half >= 0) & (centres + half < size)
    for other in others:
        keep &= np.abs(other - centres) > 2 * half
    for i in np.flatnonzero(keep):
        keep[i] = not taken_rows[i, centres[i] - half:centres[i] + half + 1].any()
    return keep


def _ratio_db(a: float, b: float) -> float | None:
    return 10.0 * math.log10(a / b) if a > 0 and b > 0 else None


def levels(state: ChannelState, real, copies: Sequence) -> dict:
    """Each mode's level in the synthetic soundings against the real one.

    ``peak_bias_db`` is the copies' mean ridge power, the brightest cell
    within :data:`PEAK_HALF_BINS` of the curve, over the real one's, both
    averaged in linear power over the rows first; ``peak_median_db`` the
    median of that ratio row by row. ``energy_bias_db`` is the same for the
    power over the noise summed :data:`ENERGY_HALF_BINS` either side, which
    a broad echo keeps when its ridge is low. Rows another mode's strip
    reaches, or the cleaning took a cell of, are left out.
    """
    power, synthetic, taken = _cleaned(real, copies)
    size = power.shape[1]
    centres = {m.name: _centres(m, real) for m in state.modes}
    out = {}
    for m in state.modes:
        rows, at = centres[m.name]
        others = []
        for o in state.modes:
            if o.name != m.name:
                orow, oat = centres[o.name]
                lookup = dict(zip(orow.tolist(), oat.tolist()))
                others.append(np.array([lookup.get(r, -10 * size) for r in rows]))
        found = {"rows": 0, "peak_bias_db": None, "peak_median_db": None,
                 "energy_rows": 0, "energy_bias_db": None}
        peak_ok = _strips(at, others, PEAK_HALF_BINS, size, taken[rows])
        if peak_ok.any():
            r, c = rows[peak_ok], at[peak_ok]
            cols = c[:, None] + np.arange(-PEAK_HALF_BINS, PEAK_HALF_BINS + 1)
            real_peak = power[r[:, None], cols].max(axis=1)
            synth_peak = np.mean([p[r[:, None], cols].max(axis=1) for p in synthetic],
                                 axis=0)
            found.update(rows=int(r.size),
                         peak_bias_db=_ratio_db(synth_peak.mean(), real_peak.mean()),
                         peak_median_db=float(np.median(
                             10.0 * np.log10(synth_peak / real_peak))))
        energy_ok = _strips(at, others, ENERGY_HALF_BINS, size, taken[rows])
        if energy_ok.any():
            r, c = rows[energy_ok], at[energy_ok]
            cols = c[:, None] + np.arange(-ENERGY_HALF_BINS, ENERGY_HALF_BINS + 1)
            noise = observe.NOISE_MEAN_POWER * cols.size
            real_energy = float((power[r[:, None], cols].sum(axis=1) - noise).sum())
            synth_energy = float(np.mean(
                [(p[r[:, None], cols].sum(axis=1) - noise).sum() for p in synthetic]))
            found.update(energy_rows=int(r.size),
                         energy_bias_db=_ratio_db(synth_energy, real_energy))
        out[m.name] = found
    return out


def together(state: ChannelState, real, copies: Sequence) -> dict:
    """:func:`levels` for all the modes as one, over the union of their strips.

    A trace the extractor broke into fragments, or two modes that cross,
    are counted once and not left out: each row's ridge is the brightest
    cell any mode's strip holds, and the energy is summed over every cell
    within :data:`ENERGY_HALF_BINS` of any mode. Cells the cleaning took are
    left out.
    """
    power, synthetic, taken = _cleaned(real, copies)
    strip = check.trace_mask(state.modes, real, half_bins=PEAK_HALF_BINS) & ~taken
    wide = check.trace_mask(state.modes, real, half_bins=ENERGY_HALF_BINS) & ~taken
    rows = np.flatnonzero(strip.any(axis=1))
    out = {"rows": int(rows.size), "peak_bias_db": None, "peak_median_db": None,
           "cells": int(np.count_nonzero(wide)), "energy_bias_db": None}
    if rows.size:
        real_peak = np.where(strip, power, 0.0).max(axis=1)[rows]
        synth_peak = np.mean([np.where(strip, p, 0.0).max(axis=1)[rows]
                              for p in synthetic], axis=0)
        out.update(peak_bias_db=_ratio_db(synth_peak.mean(), real_peak.mean()),
                   peak_median_db=float(np.median(10.0 * np.log10(synth_peak
                                                                 / real_peak))))
    if out["cells"]:
        out["energy_bias_db"] = _ratio_db(
            float(np.mean([(p[wide] - observe.NOISE_MEAN_POWER).sum()
                           for p in synthetic])),
            float((power[wide] - observe.NOISE_MEAN_POWER).sum()))
    return out


def _blocks(mask: np.ndarray, rows: int, bins: int) -> np.ndarray:
    """``mask`` pooled to blocks of ``rows`` by ``bins``: any cell set."""
    r = np.arange(0, mask.shape[0], rows)
    c = np.arange(0, mask.shape[1], bins)
    return np.logical_or.reduceat(np.logical_or.reduceat(mask, r, axis=0), c, axis=1)


def coverage(state: ChannelState, real, copies: Sequence) -> dict:
    """Which echoes the synthetic soundings and the real one share.

    An echo is a block of :data:`COVER_MHZ` by :data:`COVER_BINS` with a
    cell at :data:`ECHO_DB` or more, cleaned as the extractor cleans.
    ``missed`` is the share of the real echoes a copy lacks, ``false`` the
    share of a copy's echoes the real one lacks, each the mean over the
    copies; ``untraced`` the share of the real echoes off every trace of
    ``state``, which the extractor never found.
    """
    power, synthetic, taken = _cleaned(real, copies)
    step = float(np.median(np.abs(np.diff(np.asarray(real.freq)))))
    rows = max(1, round(COVER_MHZ / step))

    def echoes(p: np.ndarray) -> np.ndarray:
        return _blocks((spectro.to_db(p) >= ECHO_DB) & ~taken, rows, COVER_BINS)

    seen = echoes(power)
    copied = [echoes(p) for p in synthetic]
    traced = _blocks(check.trace_mask(state.modes, real), rows, COVER_BINS)
    n = int(np.count_nonzero(seen))
    missed = [np.count_nonzero(seen & ~c) / n for c in copied] if n else []
    false = [np.count_nonzero(c & ~seen) / np.count_nonzero(c)
             for c in copied if np.count_nonzero(c)]
    return {"blocks": n, "block_mhz": rows * step, "block_bins": COVER_BINS,
            "missed": float(np.mean(missed)) if missed else None,
            "false": float(np.mean(false)) if false else None,
            "untraced": np.count_nonzero(seen & ~traced) / n if n else None}


def _clock(state: ChannelState) -> dict:
    offset = clock.offset_us(state)
    ground = state.circuit.ground_km
    if offset is None or not ground:
        return {"offset_us": offset, "inside": None}
    early, late = clock.window_us(ground)
    return {"offset_us": offset, "inside": bool(early < offset < late)}


def diagnose(path: str | Path, params: Mapping, *, stations: Mapping | None = None,
             draw: str | Path | None = None) -> dict:
    """The reconstruction and recovery of the sounding at ``path``: one row.

    Its state is observed as ``params`` say (:data:`PARAMS`), sounded
    ``copies`` times from ``seed`` on its own range axis, and the copies are
    set against the real sounding (:func:`levels`, :func:`coverage`, the
    closed loop's correlation) and read back against the state (the closed
    loop's level 1). ``draw`` writes the picture of it there.
    """
    path = Path(path)
    begun = time.perf_counter()
    real = observe.load(path, window=params["window"], stations=stations)
    state = observe.from_ionogram(real, method=params["method"], source=path.name,
                                  doppler=params["doppler"],
                                  knot_mhz=params["knot_mhz"])
    if not state.modes:
        raise ValueError(f"{path.name}: no trace to reconstruct")
    copies = fc.sounded(state, int(params["copies"]), seed=int(params["seed"]),
                        gate_km=check.gate_of(real), window=int(params["window"]))
    report = check.closed_loop(state, realisations=len(copies),
                               seed=int(params["seed"]), method=params["method"],
                               real=real, synthetic=copies,
                               real_muf_mhz=state.metrics.get("muf_mhz"))
    if draw is not None:
        picture(state, real, copies, draw)
    return _plain({
        "schema": DIAGNOSIS_SCHEMA, "version": VERSION, "file": path.name,
        "epoch": state.epoch, "modes": [m.name for m in state.modes],
        "muf_mhz": state.metrics.get("muf_mhz"), "clock": _clock(state),
        "reconstruction": {"together": together(state, real, copies),
                           "levels": levels(state, real, copies),
                           "coverage": coverage(state, real, copies),
                           "correlation": report.get("correlation")},
        "recovery": {k: report.get(k) for k in
                     ("modes", "muf_mhz", "real_muf_offset_mhz", "real_muf_z",
                      "real_muf_inside")},
        "seconds": time.perf_counter() - begun})


def picture(state: ChannelState, real, copies: Sequence, path: str | Path) -> Path:
    """The sounding, a synthetic copy, the state's traces over the sounding,
    and the copies' mean less the sounding.

    The difference is taken on the closed loop's smoothed maps, each range
    bin averaged in linear power over +-0.2 MHz: a single sounding's fading
    then no longer reads 2.5 dB under a mean of copies, as it does cell by
    cell. Where neither map stands :data:`QUIET_DB` over the noise it is
    left grey, so the noise does not drown the echoes.
    """
    from matplotlib.figure import Figure

    freq, vrange = np.asarray(real.freq), np.asarray(real.vrange)
    real_db = np.asarray(real.db, dtype=np.float64)
    mean = dataclasses.replace(real, power=np.mean(
        [np.asarray(c.power, dtype=np.float64) for c in copies], axis=0), _db=None)
    smooth_real, smooth_mean = (check.db_maps(ion)["smoothed"] for ion in (real, mean))
    quiet = 10.0 * np.log10(observe.NOISE_MEAN_POWER) + QUIET_DB
    difference = np.where(np.maximum(smooth_real, smooth_mean) < quiet, np.nan,
                          smooth_mean - smooth_real)
    figure = Figure(figsize=(13, 8), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    shown = {"vmin": spectro.NOISE_FLOOR_DB, "vmax": spectro.NOISE_FLOOR_DB + 40,
             "cmap": "viridis", "shading": "auto"}
    for ax, values, title in ((axes[0, 0], real_db, "sounding"),
                              (axes[0, 1], np.asarray(copies[0].db),
                               "synthetic, first seed"),
                              (axes[1, 0], real_db, "sounding and the state's traces")):
        mesh = ax.pcolormesh(freq, vrange, values.T, **shown)
        ax.set_title(title)
    figure.colorbar(mesh, ax=axes[0, :], label="dB, noise at 30")
    scale = observe.km_per_us(real.header)
    for m in state.modes:
        f = np.linspace(m.f_lo, m.f_hi, 200)
        axes[1, 0].plot(f, m.delay_at(f) * scale, lw=0.8, color="w")
    axes[1, 1].set_facecolor("0.85")
    residual = axes[1, 1].pcolormesh(freq, vrange, difference.T, vmin=-10, vmax=10,
                                     cmap="RdBu_r", shading="auto")
    axes[1, 1].set_title(f"mean of {len(copies)} copies less the sounding, "
                         f"+-{observe.SMOOTH_MHZ:g} MHz")
    figure.colorbar(residual, ax=axes[1, 1], label="dB")
    for ax in axes[1]:
        ax.set_xlabel("frequency, MHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("range, km")
    figure.suptitle(f"{state.source}  {_utc(state.epoch):%Y-%m-%d %H:%M:%S} UTC")
    path = Path(path)
    figure.savefig(path, dpi=90)
    return path


def format_inventory(rows: Sequence[Mapping]) -> str:
    """Each day's soundings counted by the phase they carry, those set aside apart."""
    days: dict[str, dict[str, int]] = {}
    for e in rows:
        if "reason" not in e:
            phases = days.setdefault(e["day"], {})
            phases[e["phase"]] = phases.get(e["phase"], 0) + 1
    lines = [f"{day}  {sum(p.values())} soundings: "
             + ", ".join(f"{n} {phase}" for phase, n in sorted(p.items()))
             for day, p in sorted(days.items())]
    aside = [e for e in rows if "reason" in e]
    if aside:
        lines.append(f"{len(aside)} set aside, the first {aside[0]['file']}: "
                     f"{aside[0]['reason']}")
    return "\n".join(lines)


def _db(value) -> str:
    return "-" if value is None else f"{value:+.1f} dB"


def _share(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def format_diagnosis(row: Mapping) -> str:
    """One diagnosed sounding on one line: the ridge and energy bias of all
    its modes together, the echoes either side lacks, and its correlation
    with the copies beside theirs with one another."""
    found = row["reconstruction"]
    level = found["together"]
    cover = found["coverage"]
    rows = (found.get("correlation") or {}).get("rows", {})
    r, ceiling = (rows.get(k, {}).get("mean") for k in ("real", "synthetic"))
    return (f"{row['file']}  {_utc(row['epoch']):%Y-%m-%d %H:%M}  "
            f"{len(row['modes'])} modes: peak {_db(level['peak_bias_db'])}, energy "
            f"{_db(level['energy_bias_db'])}  |  {cover['blocks']} echoes: missed "
            f"{_share(cover['missed'])}, false {_share(cover['false'])}, untraced "
            f"{_share(cover['untraced'])}"
            + ("" if r is None or ceiling is None
               else f"  |  r {r:.2f} of {ceiling:.2f}"))


def _timeless(value):
    if isinstance(value, Mapping):
        return {k: _timeless(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


def digest(rows: Iterable[Mapping]) -> str:
    """One SHA-256 over ``rows`` in any order, their timings left out."""
    lines = sorted(json.dumps(_timeless(r), sort_keys=True) for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _folder_digest(folder: Path) -> str:
    rows = []
    for name in (*series.ROWS, "skipped"):
        rows.extend({"rows": name, **r}
                    for r in series.read_rows(folder / f"{name}.jsonl"))
    return digest(rows)


def _write_rows(path: Path, rows: Sequence[Mapping]) -> None:
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")


def _diagnoses(manifest: Mapping, root: Path, out: Path, stations) -> tuple[list, list]:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    by_name = {e["file"]: e for e in manifest["entries"]}
    rows, skipped = [], []
    for name in manifest["diagnostic"]:
        try:
            rows.append(diagnose(root / by_name[name]["path"], manifest["params"],
                                 stations=stations, draw=gallery / f"{name}.png"))
        except Exception as exc:
            skipped.append({"file": name, "stage": "diagnosis",
                            "reason": _failure(exc)})
    return rows, skipped


def _forecasts(manifest: Mapping, root: Path, out: Path, *, final: bool, jobs: int,
               stations) -> dict[str, str]:
    params, found = manifest["params"], {}
    for split in SPLITS:
        if split == "final" and not final:
            continue
        paths = [root / e["path"] for e in manifest["entries"] if e["split"] == split]
        if not paths:
            continue
        for lead in params["leads_s"]:
            folder = out / "forecast" / split / f"{lead:g}s"
            series.run_series(paths, folder, circuit=manifest["circuit"], lead_s=lead,
                              tolerance_s=params["tolerance_s"],
                              copies=int(params["forecast_copies"]), recovery_every=0,
                              method=params["method"], window=int(params["window"]),
                              knot_mhz=params["knot_mhz"], doppler=params["doppler"],
                              stations=stations, jobs=jobs)
            (folder / "summary.json").write_text(
                json.dumps(summary.summarise(folder), indent=1) + "\n",
                encoding="utf-8", newline="\n")
            found[f"{split}/{lead:g}s"] = _folder_digest(folder)
    return found


def run(manifest_path: str | Path, out: str | Path, *, root: str | Path,
        diagnostic: bool = True, forecast: bool = True, final: bool = False,
        jobs: int = 1, stations: Mapping | None = None) -> dict:
    """The benchmark frozen at ``manifest_path``, run into ``out``.

    ``root`` is where the manifest's paths are read from; an input that is
    gone or changed stops the run before it begins. The diagnostic
    soundings go to ``diagnosis.jsonl`` and their pictures to ``gallery/``;
    each split's forecast at each lead to ``forecast/<split>/<lead>s``, the
    final split only with ``final``, summed up there with block-bootstrap
    intervals (:func:`muf.sim.summary.summarise`); what could not be scored to
    ``skipped.jsonl``; and the record, with the digest of it all, to
    ``run.json``.
    """
    manifest_path, out, root = Path(manifest_path), Path(out), Path(root)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != VERSION:
        raise ValueError(f"{manifest_path}: not a version {VERSION} benchmark manifest")
    check_inputs(manifest, root)
    out.mkdir(parents=True, exist_ok=True)
    started, begun = _dt.datetime.now(_dt.timezone.utc), time.perf_counter()
    rows, skipped = (_diagnoses(manifest, root, out, stations) if diagnostic
                     else ([], []))
    _write_rows(out / "diagnosis.jsonl", rows)
    _write_rows(out / "skipped.jsonl", skipped)
    forecasts = (_forecasts(manifest, root, out, final=final, jobs=jobs,
                            stations=stations) if forecast else {})
    record = _plain({
        "schema": RUN_SCHEMA, "version": VERSION, "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "muf": series.code_version(), "params": manifest["params"],
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "diagnosed": len(rows), "skipped": len(skipped), "forecast": forecasts,
        "digest": digest([*({"rows": "diagnosis", **r} for r in rows),
                          *({"rows": "skipped", **r} for r in skipped),
                          {"rows": "forecast", **forecasts}]),
        "seconds": time.perf_counter() - begun})
    (out / "run.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n",
                                  encoding="utf-8", newline="\n")
    return record
