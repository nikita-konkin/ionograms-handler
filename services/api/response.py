"""A stored sounding's frequency response, subchannel by subchannel (#103).

The measurement is :mod:`muf.sim.response`; this is what the hub needs around
it. The modes' curves come from the sounding's state, observed with muf's
defaults -- every mode, spline curves, where the live loop keeps the 1F
family on a layer -- and cached beside the consoles by
:func:`muf.sim.series.state_of`. With them goes a small ionogram about the
modes, for the page to pick a stretch on.

Reading a recording is the slow part -- an ``.lfs`` is 80 MB and its
ionogram a spectrogram of all of it -- and a page asks again at every change
of a setting, so the last sounding's record, state and ionogram are kept in
memory. A sliding series at every sample of an ``.lfs`` is 400,000 samples a
megahertz; a page is a thousand pixels wide, so at most :data:`POINTS` of a
series are sent, every k-th, after its phase is unwrapped at full density.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import numpy as np

#: Most samples of one series sent to a page.
POINTS = 4000

#: The modes' curves are sent sampled every this many MHz.
CURVE_STEP_MHZ = 0.02

#: The ionogram sent: at most this many columns of frequency and rows of
#: delay, this much delay either side of the modes, dB above the noise.
MAP_COLS, MAP_ROWS = 480, 220
MAP_PAD_US = 400.0
MAP_DB = (0.0, 40.0)

#: Products whose answer to :func:`keeps_phase` is remembered, at most.
PHASE_MEMO = 20_000

_lock = threading.Lock()
_last: dict = {}
_phase: dict = {}


def forget() -> None:
    """Drop the sounding kept in memory."""
    with _lock:
        _last.clear()
    _phase.clear()


def keeps_phase(path: Path) -> bool:
    """Whether the file keeps the dechirped voltage: an .lfs, or a v2 product
    with ``z`` or an echo window.

    A product is opened once for as long as it stays the same size and age:
    the picker asks this of a circuit's three days of soundings at every
    visit, and an echo window can be added to a product after it is written.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".lfs":
        return path.is_file()
    if suffix != ".h5":
        return False
    import h5py

    from muf import echo_window, io_chirp

    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in _phase:
            with h5py.File(path, "r") as fh:
                found = io_chirp.RAW_VOLTAGE_KEY in fh or echo_window.GROUP in fh
            if len(_phase) >= PHASE_MEMO:
                _phase.clear()
            _phase[key] = found
        return _phase[key]
    except OSError:
        return False


def _inputs(path: Path, cache: Path):
    """The sounding's record, state and ionogram, from memory when it is the last."""
    from muf.sim import fast, observe, series

    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _lock:
        if _last.get("key") == key:
            return _last["record"], _last["state"], _last["ionogram"]
    record = fast.record_of(path)
    state = series.state_of(path, Path(cache) / "response")
    ionogram = observe.load(path, window=series.OBSERVE["window"])
    with _lock:
        _last.clear()
        _last.update(key=key, record=record, state=state, ionogram=ionogram)
    return record, state, ionogram


def _curve(mode) -> dict:
    freq = np.append(np.arange(mode.f_lo, mode.f_hi, CURVE_STEP_MHZ), mode.f_hi)
    return {"freq_mhz": np.round(freq, 4).tolist(),
            "delay_us": np.round(mode.delay_at(freq), 2).tolist()}


def _map(ion, km_per_us: float, modes) -> dict:
    """The ionogram's dB about the modes, pooled to at most MAP_COLS x MAP_ROWS, as
    bytes column by column, delay ascending within each."""
    freq = np.asarray(ion.freq, dtype=np.float64)
    delay = np.asarray(ion.vrange, dtype=np.float64) / km_per_us
    curves = np.concatenate([m.delay_us for m in modes])
    d0 = max(float(delay.min()), float(curves.min()) - MAP_PAD_US)
    d1 = min(float(delay.max()), float(curves.max()) + MAP_PAD_US)
    keep = np.nonzero((delay >= d0) & (delay <= d1))[0]
    keep = keep[np.argsort(delay[keep])]
    level = 10.0 * np.log10(np.maximum(np.asarray(ion.power)[:, keep], 1e-12))
    cols, rows = min(MAP_COLS, freq.size), min(MAP_ROWS, keep.size)
    f0, f1 = float(freq.min()), float(freq.max())
    col = np.minimum(((freq - f0) / max(f1 - f0, 1e-9) * cols).astype(int), cols - 1)
    row = np.minimum(((delay[keep] - d0) / max(d1 - d0, 1e-9) * rows).astype(int),
                     rows - 1)
    grid = np.full((cols, rows), MAP_DB[0])
    np.maximum.at(grid, (col[:, None], row[None, :]), level)
    scaled = np.clip((grid - MAP_DB[0]) / (MAP_DB[1] - MAP_DB[0]) * 255.0, 0, 255)
    return {"freq_mhz": [f0, f1], "delay_us": [d0, d1], "cols": cols, "rows": rows,
            "db": list(MAP_DB),
            "map": base64.b64encode(scaled.astype(np.uint8).tobytes()).decode("ascii")}


def _thinned(series: dict, points: int) -> dict:
    size = len(series["freq_mhz"])
    if size <= points:
        return series
    every = -(-size // points)
    return {k: v[::every] if isinstance(v, list) else v for k, v in series.items()}


def reading(path: Path, cache: Path, *, mode: str | None = None,
            lo: float | None = None, hi: float | None = None,
            subchannel_khz: float = 20.48, overlap: float = 0.5, held: bool = False,
            shift_us: float = 0.0, half_us: float | None = None) -> dict:
    """The page's data: the response, the modes' curves, the ionogram and MUF."""
    from muf.sim import observe
    from muf.sim import response as fr

    path = Path(path)
    if not keeps_phase(path):
        raise ValueError(f"{path.name} keeps no phase: an .lfs, or a v2 product "
                         f"with z or an echo window, is needed")
    record, state, ion = _inputs(path, cache)
    names = [m.name for m in state.modes]
    if not names:
        raise ValueError(f"{path.name}: no mode was observed")
    if mode is not None and mode not in names:
        raise ValueError(f"{path.name}: no mode {mode} among {', '.join(names)}")
    chosen = state.mode(mode or names[0])
    band = fr.default_band(chosen) if lo is None or hi is None else (lo, hi)
    got = fr.measure(record, chosen, *band, subchannel_hz=subchannel_khz * 1e3,
                     overlap=overlap,
                     gate=fr.Gate(held=held, shift_us=shift_us, half_us=half_us))
    out = got.as_dict()
    out["sliding"] = _thinned(out["sliding"], POINTS)
    out["stepped"] = _thinned(out["stepped"], POINTS)
    ends = record.freq_at(np.array([record.t_first_s,
                                    record.t_first_s + record.duration_s]))
    muf = state.metrics.get("muf_mhz")
    return {"source": path.name,
            "record": {"f_lo_mhz": float(ends.min()), "f_hi_mhz": float(ends.max())},
            "muf_mhz": float(muf) if muf is not None and np.isfinite(muf) else None,
            "modes": [{"name": m.name, "f_lo_mhz": m.f_lo, "f_hi_mhz": m.f_hi,
                       "curve": _curve(m)} for m in state.modes],
            "ionogram": _map(ion, observe.km_per_us(ion.header), state.modes),
            "response": out}
