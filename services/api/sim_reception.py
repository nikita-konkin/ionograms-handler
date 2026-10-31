"""A synthetic forecast matched to the sounding currently arriving live.

Only the issued state is used to synthesise the left pane. The live header
supplies display axes; received powers never enter the simulation.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import threading

import numpy as np

from . import apart, db

_lock = threading.Lock()
_cache = collections.OrderedDict()
_queue = collections.deque()
_worker = None
LIMIT = 8


def select(ledger: dict, t0: float) -> dict | None:
    """Shortest issued lead within 30 s of this sweep, never a future source."""
    matches = []
    for pending in ledger.get("pending", []):
        state = pending["forecast"]
        lead = float(state.get("metrics", {}).get("lead_s", 0))
        if lead > 0 and state["epoch"] - lead < t0:
            delta = abs(state["epoch"] - t0)
            if delta <= 30:
                matches.append((delta, lead, state))
    return min(matches, key=lambda p: p[:2])[2] if matches else None


def grid(head: dict) -> dict:
    """Bound the browser raster while preserving the live display's extent."""
    out = {k: head[k] for k in ("t0", "columns", "rows", "f_lo_mhz",
                               "f_step_mhz", "range_lo_km", "range_step_km",
                               "km_per_ms")}
    for n, step, cap in (("columns", "f_step_mhz", 800),
                         ("rows", "range_step_km", 256)):
        fold = max(1, int(np.ceil(out[n] / cap)))
        size = int(np.ceil(out[n] / fold))
        out[step] *= out[n] / size
        out[n] = size
    return out


def render(state: dict, head: dict) -> dict:
    """The forecast's synthetic sounding on the live sweep's own bins.

    It is read back with the FFT window the sweep is received with, so a
    trace is as thin as the received one, and folded into the page's raster
    by the maximum, as the station folds rows and the page folds columns.
    """
    from muf import spectro
    from muf.sim import forecast
    from muf.sim.state import ChannelState

    issued = ChannelState.from_dict(state)
    raster = grid(head)
    ion = forecast.sounded(
        issued, 1, seed=0,
        window=round(head["f_step_mhz"] * 1e6 / issued.sweep.rate_hz_s
                     * issued.sweep.sample_rate_hz),
        gate_km=(head["range_lo_km"] - head["range_step_km"],
                 head["range_lo_km"] + head["rows"] * head["range_step_km"]))[0]
    vrange = np.asarray(ion.vrange)
    column, row = np.meshgrid(
        np.floor((np.asarray(ion.freq) - head["f_lo_mhz"]) / raster["f_step_mhz"]),
        np.floor((vrange - head["range_lo_km"] + np.median(np.abs(np.diff(vrange))) / 2)
                 / raster["range_step_km"]),
        indexing="ij")
    inside = ((column >= 0) & (column < raster["columns"])
              & (row >= 0) & (row < raster["rows"]))
    cells = np.zeros((raster["columns"], raster["rows"]))
    np.maximum.at(cells, (column[inside].astype(int), row[inside].astype(int)),
                  np.asarray(ion.db)[inside] - spectro.NOISE_FLOOR_DB)
    return {"head": raster, "cells": base64.b64encode(
        np.clip(np.rint(cells * 4), 0, 255).astype(np.uint8).tobytes()).decode("ascii")}


def image(state: dict, head: dict) -> dict:
    global _worker
    key = hashlib.sha256(json.dumps([state, head], sort_keys=True).encode()).hexdigest()
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
        _queue.append((key, state, head))
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
            key, state, head = _queue.popleft()
        try:
            result = apart.run(render, state, head, timeout_s=120,
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
    state = select(json.loads(held["ledger"]), head["t0"])
    if state is None:
        return result
    result.update(image(state, head))
    result.update(issued_at=state["epoch"] - state["metrics"]["lead_s"],
                  valid_at=state["epoch"], lead_s=state["metrics"]["lead_s"])
    return result
