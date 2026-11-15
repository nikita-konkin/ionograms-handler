"""The public page's archive: past soundings as echo windows (#151).

The echo window (`docs/echo-window.md`) is the one product small enough to
publish and complete enough to draw again, about 1.6 MB a sounding. This pass
copies each one into the public folder beside the slow pass's files::

    archive/index.json                                  circuits and their days
    archive/{slug}/{YYYY}/{MM}/{DD}/index.json          one day's soundings
    archive/{slug}/{YYYY}/{MM}/{DD}/{stamp}.echo.h5     one window

**Byte for byte.** A window leaves the hub exactly as the archive holds it, so
a download can be checked against the hub by its digest. It therefore carries
the capture's file name, the station coordinates, the writer's commit and the
raw header: the one exception, decided in #151, to the page's rule that no
file names or internals leave the server. Nothing but a readable window is
ever copied, and a raw capture never is.

**Allow-listed on its own list.** Only circuits named in
``--archive-circuits`` and not muted appear, whatever the page's own list
says, so a circuit sounded years ago does not show on the page as a dead one.

**Never rewritten, and bounded.** A window is named for its sounding's time,
which makes the file on disk its own cache and lets the web server mark it
immutable. A pass copies at most `BATCH` new windows and holds the rest for
the next one, so a first copy of a whole archive cannot keep the live picture
waiting. A day's index lists every sounding of that day; one with no window
on the page yet has no file. Windows nothing lists any more are deleted only
after the indexes that stopped listing them are in place.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from services.api import db

from .publisher import (
    IMG_STAMP,
    TMP_PREFIX,
    _json_bytes,
    _parse_db_time,
    resolve,
    slug,
    station_name,
    write_atomic,
)

__all__ = ["BATCH", "FIELDS", "write_archive"]

#: Most windows one pass copies: about 160 MB, a minute from a network mount.
BATCH = 100

#: Everything a sounding in a day's index may carry.
FIELDS = ("t", "file", "bytes", "range_km", "trace_at_edge")


def _circuits(conn, circuits: list[tuple[str, str]]) -> dict[str, list]:
    """Each allowed, unmuted circuit's spellings by slug, newest first."""
    muted = db.muted_matcher(conn)
    held = db.rows(conn, "SELECT tx, rx, MAX(datetime) AS last FROM sounding"
                         " WHERE tx IS NOT NULL AND rx IS NOT NULL"
                         " GROUP BY tx, rx ORDER BY last DESC")
    found: dict[str, list] = {}
    for want in circuits:
        for row in held:
            if ((str(row["tx"]).lower(), str(row["rx"]).lower()) == want
                    and not muted(row["tx"], row["rx"])):
                found.setdefault(slug(row["tx"], row["rx"]), []).append(
                    (row["tx"], row["rx"]))
    return found


def _days(conn, spellings: list) -> dict[str, list]:
    """A circuit's soundings by day, oldest first, for days with a window."""
    from muf import loader

    rows = []
    for tx, rx in spellings:
        rows += db.rows(conn, "SELECT datetime, path, format FROM sounding"
                              " WHERE tx = ? AND rx = ?", (tx, rx))
    days: dict[str, list] = {}
    for row in sorted(rows, key=lambda r: str(r["datetime"]).replace("T", " ")):
        days.setdefault(str(row["datetime"])[:10], []).append(row)
    return {day: found for day, found in days.items()
            if any(r["format"] == loader.ECHO_WINDOW for r in found)}


def _described(path: Path) -> dict:
    """What a day's index says of a window, read from the window itself."""
    import h5py

    from muf import echo_window

    with h5py.File(path, "r") as handle:
        attrs = handle[echo_window.GROUP].attrs
        edge = attrs.get("trace_at_edge")
        return {
            "bytes": path.stat().st_size,
            "range_km": [round(float(attrs["range_lo_km"]), 2),
                         round(float(attrs["range_hi_km"]), 2)],
            "trace_at_edge": None if edge is None else bool(edge),
        }


def _copy(source: Path, target: Path) -> dict:
    """The window into place under a temporary name, described on the way."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{TMP_PREFIX}{target.name}.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        described = _described(temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return described


def _cached(index: Path) -> dict[str, dict]:
    """The entries a day's index already holds, by file name."""
    try:
        listed = json.loads(index.read_text(encoding="utf-8"))["soundings"]
        return {e["file"]: {k: e[k] for k in FIELDS} for e in listed if e["file"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _day(site_dir: Path, folder: str, rows: list, archive_root: Path,
         limit: int, result: dict) -> list[dict]:
    """One day's soundings, copying the windows the page does not hold yet."""
    from muf import loader

    cached = _cached(site_dir / folder / "index.json")
    entries = []
    for row in rows:
        when = _parse_db_time(row["datetime"])
        name = f"{when.strftime(IMG_STAMP)}{loader.ECHO_WINDOW_SUFFIX}"
        entry = dict.fromkeys(FIELDS)
        entry["t"] = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        if row["format"] == loader.ECHO_WINDOW:
            target = site_dir / folder / name
            try:
                if target.exists():
                    entry.update(cached.get(name) or _described(target), file=name)
                elif result["archived"] >= limit:
                    result["held"] += 1
                else:
                    entry.update(_copy(resolve(row["path"], archive_root), target),
                                 file=name)
                    result["archived"] += 1
            except Exception as exc:
                result["skipped"] += 1
                result["errors"].append(
                    f"{folder}/{name}: {type(exc).__name__}: {exc}")
        entries.append(entry)
    return entries


def _prune(site_dir: Path, keep: set[str]) -> int:
    """Delete what no index lists, and the folders that leaves empty."""
    root = site_dir / "archive"
    pruned = 0
    for stale in [*sorted(root.rglob("*"), reverse=True), root]:
        if not stale.exists():
            continue
        if stale.is_dir():
            try:
                stale.rmdir()
            except OSError:
                pass
        elif stale.relative_to(site_dir).as_posix() not in keep:
            stale.unlink(missing_ok=True)
            pruned += 1
    return pruned


def write_archive(conn, site_dir: Path, *, circuits: list[tuple[str, str]],
                  archive_root: Path, limit: int = BATCH) -> dict:
    """One archive pass into ``site_dir``. Returns counts for the log line."""
    site_dir = Path(site_dir)
    result = {"archived": 0, "held": 0, "skipped": 0, "pruned": 0, "errors": []}
    keep: set[str] = set()
    listed = []
    for name, spellings in _circuits(conn, circuits).items():
        days = []
        for day, rows in _days(conn, spellings).items():
            folder = f"archive/{name}/{day.replace('-', '/')}"
            entries = _day(site_dir, folder, rows, archive_root, limit, result)
            windows = [e for e in entries if e["file"]]
            if windows:
                write_atomic(site_dir / folder / "index.json", _json_bytes(
                    {"slug": name, "day": day, "soundings": entries}))
                keep.add(f"{folder}/index.json")
                keep.update(f"{folder}/{e['file']}" for e in windows)
                days.append({"day": day, "soundings": len(entries),
                             "windows": len(windows),
                             "bytes": sum(e["bytes"] for e in windows)})
        if days:
            tx, rx = spellings[0]
            listed.append({"slug": name, "tx": tx, "rx": rx,
                           "tx_name": station_name(tx),
                           "rx_name": station_name(rx), "days": days})
    if listed:
        write_atomic(site_dir / "archive" / "index.json",
                     _json_bytes({"circuits": listed}))
        keep.add("archive/index.json")
    result["pruned"] = _prune(site_dir, keep)
    return result
