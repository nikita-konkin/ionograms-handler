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
    from scipy.interpolate import RegularGridInterpolator

    from muf import spectro
    from muf.sim import forecast
    from muf.sim.state import ChannelState

    ion = forecast.sounded(ChannelState.from_dict(state), 1, seed=0)[0]
    fs, rs = np.argsort(ion.freq), np.argsort(ion.vrange)
    interp = RegularGridInterpolator(
        (np.asarray(ion.freq)[fs], np.asarray(ion.vrange)[rs]),
        np.asarray(ion.db)[np.ix_(fs, rs)] - spectro.NOISE_FLOOR_DB,
        bounds_error=False, fill_value=0)
    f = head["f_lo_mhz"] + np.arange(head["columns"]) * head["f_step_mhz"]
    r = head["range_lo_km"] + np.arange(head["rows"]) * head["range_step_km"]
    ff, rr = np.meshgrid(f, r, indexing="ij")
    cells = np.clip(np.rint(interp(np.stack((ff, rr), axis=-1)) * 4), 0, 255)
    return {"head": head, "cells": base64.b64encode(cells.astype(np.uint8).tobytes())
            .decode("ascii")}


def image(state: dict, head: dict) -> dict:
    global _worker
    head = grid(head)
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
