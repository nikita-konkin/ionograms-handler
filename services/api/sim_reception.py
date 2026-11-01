"""A synthetic forecast matched to the sounding currently arriving live.

The left pane draws the synthetic sounding the live loop kept when it issued
the forecast, the one the forecast's panels draw and its scores were read
from. It is not synthesised again. The live header supplies display axes;
received powers never enter it.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from . import apart, db

_lock = threading.Lock()
_cache = collections.OrderedDict()
_queue = collections.deque()
_worker = None
LIMIT = 8


def select(ledger: dict, t0: float) -> dict | None:
    """The forecast waiting with the shortest issued lead within 30 s of this
    sweep, never one from a future source."""
    matches = []
    for pending in ledger.get("pending", []):
        state = pending["forecast"]
        lead = float(state.get("metrics", {}).get("lead_s", 0))
        if lead > 0 and state["epoch"] - lead < t0:
            delta = abs(state["epoch"] - t0)
            if delta <= 30:
                matches.append((delta, lead, pending))
    return min(matches, key=lambda p: p[:2])[2] if matches else None


def grid(head: dict) -> dict:
    """The page's raster of the live display: as many cells as the page folds
    the received sweep into, over the same extent."""
    out = {k: head[k] for k in ("t0", "columns", "rows", "f_lo_mhz",
                               "f_step_mhz", "range_lo_km", "range_step_km",
                               "km_per_ms")}
    for n, step, cap in (("columns", "f_step_mhz", 800),
                         ("rows", "range_step_km", 256)):
        size = min(cap, out[n])
        out[step] *= out[n] / size
        out[n] = size
    return out


def gate_of(head: dict) -> tuple[float, float]:
    """The range a sounding is read over to fill the live display's rows."""
    return (head["range_lo_km"] - head["range_step_km"],
            head["range_lo_km"] + head["rows"] * head["range_step_km"])


def cells(ion, head: dict) -> np.ndarray:
    """``ion`` as the page's quarter-dB cells, columns by rows.

    Each bin goes to the live sweep's column and folded row it falls in, and
    on to the page's raster as ``sim-reception.js`` folds the received
    sweep: cell ``floor(i * size / n)`` of ``n``, by the maximum.
    """
    from muf import spectro

    raster = grid(head)
    vrange = np.asarray(ion.vrange)
    column, row = np.meshgrid(
        np.floor((np.asarray(ion.freq) - head["f_lo_mhz"]) / head["f_step_mhz"]),
        np.floor((vrange - head["range_lo_km"] + np.median(np.abs(np.diff(vrange))) / 2)
                 / head["range_step_km"]),
        indexing="ij")
    inside = ((column >= 0) & (column < head["columns"])
              & (row >= 0) & (row < head["rows"]))
    out = np.zeros((raster["columns"], raster["rows"]))
    folded = (column[inside].astype(int) * raster["columns"] // head["columns"],
              row[inside].astype(int) * raster["rows"] // head["rows"])
    np.maximum.at(out, folded, np.asarray(ion.db)[inside] - spectro.NOISE_FLOOR_DB)
    return np.clip(np.rint(out * 4), 0, 255).astype(np.uint8)


def _product(pending: dict) -> Path | None:
    found = (pending.get("simulated") or {}).get("product")
    return Path(found) if found else None


def render(pending: dict, head: dict) -> dict:
    """The forecast's issued synthetic sounding on the live sweep's own bins.

    It is read back with the FFT window the sweep is received with, so a
    trace is as thin as the received one, and folded as the page folds the
    received sweep (:func:`cells`).
    """
    from muf.sim import observe
    from muf.sim.state import ChannelState

    sweep = ChannelState.from_dict(pending["forecast"]).sweep
    ion = observe.load(_product(pending),
                       window=round(head["f_step_mhz"] * 1e6 / sweep.rate_hz_s
                                    * sweep.sample_rate_hz),
                       gate_km=gate_of(head))
    return {"head": grid(head),
            "cells": base64.b64encode(cells(ion, head).tobytes()).decode("ascii")}


def image(pending: dict, head: dict) -> dict:
    """The pane's picture of ``pending`` on ``head``: ready, or being built, or
    ``gone`` when the forecast's synthetic sounding is no longer kept."""
    global _worker
    product = _product(pending)
    if product is None or not product.is_file():
        return {"status": "gone"}
    key = hashlib.sha256(json.dumps([pending, head], sort_keys=True).encode()
                         ).hexdigest()
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return dict(_cache[key])
        # Keep queued/running entries until they finish; bound work requested
        # by multiple browsers rather than queueing the same sweep repeatedly.
        while len(_cache) >= LIMIT:
            old = next((k for k, v in _cache.items()
                        if v["status"] in ("ready", "failed")), None)
            if old is None:
                return {"status": "busy"}
            del _cache[old]
        _cache[key] = {"status": "building"}
        _queue.append((key, pending, head))
        if _worker is None:
            _worker = threading.Thread(target=_work, daemon=True, name="sim-reception")
            _worker.start()
        return {"status": "building"}


def _work():
    global _worker
    while True:
        with _lock:
            if not _queue:
                _worker = None
                return
            key, pending, head = _queue.popleft()
        try:
            result = apart.run(render, pending, head, timeout_s=120,
                               what="live simulated ionogram")
            result["status"] = "ready"
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
            print(f"live simulated ionogram failed: {exc}", flush=True)
        with _lock:
            _cache[key] = result


def view(conn, hub, tx: str, rx: str, station: str | None = None) -> dict:
    stations = sorted(s for s, board in tuple(hub.boards.items())
                      if tx in board.current or tx in board.last or tx in board.status)
    matches = [s for s in stations if rx in db.receivers(conn, s)]
    chosen = (station if station in stations
              else (matches[0] if len(matches) == 1 else None))
    result = {"stations": stations, "station": chosen, "status": "choose_station"}
    if chosen is None:
        return result
    board = hub.boards[chosen]
    capture = board.current.get(tx) or board.last.get(tx)
    if capture is None:
        return dict(result, status="waiting")
    head = capture.head
    result.update(t0=head["t0"], status="no_forecast")
    held = db.one(conn, "SELECT ledger FROM live_circuit WHERE tx=? AND rx=?", (tx, rx))
    if held is None:
        return result
    pending = select(json.loads(held["ledger"]), head["t0"])
    if pending is None:
        return result
    state = pending["forecast"]
    result.update(image(pending, head))
    result.update(issued_at=state["epoch"] - state["metrics"]["lead_s"],
                  valid_at=state["epoch"], lead_s=state["metrics"]["lead_s"])
    return result
