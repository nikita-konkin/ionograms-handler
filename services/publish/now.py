"""The public page's live pass: the sounding in progress, every few seconds (#105).

The hub's live view (`services/api/live.py`) holds each station's sounding as
its columns arrive, in the api's memory. This pass reads it from the api
inside the hub's own network and writes two kinds of file into the public
folder beside the slow pass's::

    now.json                       circuits being sounded, and when it looked
    now/{slug}-{t0}-{n}.webp       the columns so far, as one small picture

The public host pulls only these every few seconds; nothing on its side can
reach the api. **Allow-listed, as the rest of the page**: only circuits named
in the publisher's list and not muted, and only the fields in :data:`FIELDS`
-- no station ids, addresses or paths. Each picture's name carries the
sounding and how many columns it holds, so a browser never shows a cached
one for a newer state, and the pictures no longer named are pruned.

A station whose snapshot cannot be read costs its own circuits; the pass and
the other stations go on.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from services.api import db

from .publisher import parse_circuits, slug, station_name, write_atomic

__all__ = ["FIELDS", "fetcher", "parse_circuits", "write_now"]

#: Everything a circuit in `now.json` may carry.
FIELDS = ("slug", "tx", "rx", "tx_name", "rx_name", "started", "f_lo_mhz",
          "f_hi_mhz", "f_now_mhz", "progress", "range_km", "img")

#: dB per count in a live column, as the agent encodes them, and the span the
#: preview's colours cover.
DB_STEP = 0.25
DB_SPAN = (0.0, 40.0)

#: Seconds one station's snapshot may take before it is given up on.
TIMEOUT_S = 5.0


def fetcher(api: str, token: str = ""):
    """Read a station's live snapshot from the api at ``api``."""
    def fetch(station: str) -> dict:
        request = urllib.request.Request(f"{api.rstrip('/')}/stations/{station}/live")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
            return json.load(reply)
    return fetch


def _stations(conn) -> list[str]:
    found = db.rows(conn, "SELECT station FROM health_report"
                          " UNION SELECT station FROM station_receiver")
    return sorted({str(r["station"]) for r in found})


def _cells(sounding: dict) -> np.ndarray:
    """The columns so far as dB, NaN where a column has not arrived."""
    cells = np.full((int(sounding["columns"]), int(sounding["rows"])), np.nan)
    for index, text in (sounding.get("data") or {}).items():
        try:
            column = np.frombuffer(base64.b64decode(text), np.uint8)
        except (binascii.Error, ValueError):
            continue
        i = int(index)
        if 0 <= i < cells.shape[0] and column.size == cells.shape[1]:
            cells[i] = column * DB_STEP
    return cells


def _preview(cells: np.ndarray) -> bytes:
    """Frequency across, range upwards, the columns still to come left dark."""
    from matplotlib import colormaps
    from PIL import Image

    lut = colormaps["viridis"](np.linspace(0.0, 1.0, 256))[:, :3]
    lut = (lut * 255).astype(np.uint8)
    scaled = np.clip((cells - DB_SPAN[0]) / (DB_SPAN[1] - DB_SPAN[0]), 0.0, 1.0)
    index = np.nan_to_num(scaled * 255.0).astype(np.uint8)
    rgb = lut[index]
    rgb[np.isnan(cells)] = (11, 13, 16)
    image = Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1]), "RGB")
    out = io.BytesIO()
    image.save(out, format="WEBP", quality=80, method=4)
    return out.getvalue()


def _entry(sounding: dict, tx: str, rx: str) -> dict | None:
    cells = _cells(sounding)
    arrived = np.nonzero(~np.isnan(cells[:, 0]))[0] if cells.size else []
    if not len(arrived):
        return None
    name = slug(tx, rx)
    t0 = float(sounding["t0"])
    lo, step = float(sounding["f_lo_mhz"]), float(sounding["f_step_mhz"])
    top = int(arrived.max())
    r_lo, r_step = float(sounding["range_lo_km"]), float(sounding["range_step_km"])
    return {
        "slug": name, "tx": tx, "rx": rx,
        "tx_name": station_name(tx), "rx_name": station_name(rx),
        "started": datetime.fromtimestamp(t0, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "f_lo_mhz": round(lo, 4),
        "f_hi_mhz": round(lo + (cells.shape[0] - 1) * step, 4),
        "f_now_mhz": round(lo + top * step, 4),
        "progress": round(len(arrived) / cells.shape[0], 4),
        "range_km": [round(r_lo, 1), round(r_lo + cells.shape[1] * r_step, 1)],
        "img": f"now/{name}-{int(t0)}-{len(arrived)}.webp",
        "_cells": cells,
    }


def write_now(conn, site_dir: Path, *, fetch, circuits: list[tuple[str, str]],
              now: datetime | None = None) -> dict:
    """One live pass into ``site_dir``. Returns counts for the log line."""
    now = now or datetime.now(timezone.utc)
    site_dir = Path(site_dir)
    allowed = set(circuits)
    muted = db.muted_matcher(conn)
    result = {"circuits": 0, "errors": []}
    listed: list[dict] = []
    for station in _stations(conn):
        try:
            snapshot = fetch(station)
        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            continue
        receivers = db.receivers(conn, station)
        for tx, sounding in sorted(((snapshot or {}).get("current") or {}).items()):
            for rx in receivers:
                if (tx.lower(), rx.lower()) not in allowed or muted(tx, rx):
                    continue
                try:
                    entry = _entry(sounding, tx, rx)
                except (KeyError, TypeError, ValueError) as exc:
                    result["errors"].append(
                        f"{slug(tx, rx)}: {type(exc).__name__}: {exc}")
                    continue
                if entry is not None:
                    listed.append(entry)
                break
    keep = set()
    for entry in listed:
        cells = entry.pop("_cells")
        target = site_dir / entry["img"]
        if not target.exists():
            write_atomic(target, _preview(cells))
        keep.add(entry["img"])
    write_atomic(site_dir / "now.json", json.dumps(
        {"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "circuits": [{k: e[k] for k in FIELDS} for e in listed]},
        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for stale in (site_dir / "now").glob("*.webp"):
        if stale.relative_to(site_dir).as_posix() not in keep:
            stale.unlink(missing_ok=True)
    result["circuits"] = len(listed)
    return result
