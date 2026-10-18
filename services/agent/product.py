"""One archived ``lfm_ionogram-*.h5`` in the live view's own units and grid.

The live feed sends a sounding as one byte per frequency step and range row,
a quarter of a dB per count above the noise floor, the range folded to at
most :data:`services.agent.live.ROWS` rows. A product read here comes out the
same way, so the backfill summarises it as the server summarises a live
sounding (issue #12), and the live page draws an archived ionogram exactly as
it draws a live one.

**Gated by the path, as a live capture is.** A product stores the whole range
the sounder kept, 0 to 8000 km on the Cyprus path, and folding all of it into
the live rows made the history three times coarser than live. The path's
window comes from the product's own coordinates or, when it has none, from
the station registry by the transmitter and receiver names. A path neither
knows keeps the whole range.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from muf import calibrate, io_chirp, spectro
from muf.stations import default_registry

from . import summary
from .live import ROWS


def read(path: Path) -> tuple[dict, np.ndarray]:
    """The product's axes as a live sounding's head, and its cells by step and row."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ion = io_chirp.load(path)
    vrange = np.asarray(ion.vrange, dtype=np.float64)[::-1]
    freq = np.asarray(ion.freq, dtype=np.float64)
    db = ion.db[:, ::-1]
    window = gate(ion.header)
    if window is not None:
        kept = (vrange >= window[0]) & (vrange <= window[1])
        if np.count_nonzero(kept) >= 2:
            vrange, db = vrange[kept], db[:, kept]
    if freq.size < 2 or vrange.size < 2:
        raise ValueError("%s has too few frequencies or ranges to read"
                         % path.name)
    fold = -(-vrange.size // ROWS)
    cells = summary.counts(np.maximum.reduceat(
        db - spectro.NOISE_FLOOR_DB, np.arange(0, vrange.size, fold), axis=1))
    return {"tx": ion.header.tx_name, "t0": round(ion.header.t0),
            "f_lo_mhz": float(freq[0]), "f_step_mhz": float(np.median(np.diff(freq))),
            "columns": int(cells.shape[0]), "range_lo_km": float(vrange[0]),
            "range_step_km": float(np.median(np.diff(vrange))) * fold,
            "rows": int(cells.shape[1]),
            "km_per_ms": calibrate.C_KM_S / 1e3 * 2.0 / ion.header.div_coef}, cells


def gate(header) -> tuple[float, float] | None:
    """The range an echo on the product's path can occupy, or None if unknown."""
    found = calibrate.geometry_gate(header)
    if found is not None or not header.is_oblique:
        return found
    sites = default_registry()
    tx, rx = sites.station(header.tx_name), sites.station(header.rx_name)
    if tx is None or rx is None:
        return None
    return calibrate.geometry_gate(SimpleNamespace(
        is_oblique=True, tx_latitude=tx.latitude, tx_longitude=tx.longitude,
        rx_latitude=rx.latitude, rx_longitude=rx.longitude))
