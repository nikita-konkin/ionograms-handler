"""Indexing the folders someone registered, without reimplementing the indexer.

:mod:`services.api.watch` already does the hard part -- it enumerates a target,
subtracts what the database holds, and hands :mod:`services.api.ingest` only
the difference. It has also already learned the failures that matter: files
still being written, a NAS clock running five hours fast, a locked database, a
directory that vanishes mid-scan. None of that is repeated here. This module
is the registry's side of it: which folders, when, one at a time, and what to
tell the page afterwards.

**Why a lock and not a queue.** A scan runs the whole ``muf`` pipeline over
every new file, so it is CPU-bound, and it writes to SQLite, which locks the
file. Two at once would contend for both and finish later than one after the
other. A press while a scan is running is therefore refused with the name of
what is running, rather than queued -- a queue here would let an impatient
operator stack up hours of work behind a button that still says "scan now".

**Why its own connection.** ``watch.connect`` sets
``PRAGMA busy_timeout = 30000``; ``app.state.db`` does not, and the API reads
on every request. A scan sharing that connection turns an ordinary page load
into a ``database is locked`` error.

**What indexing produces.** Everything downstream of the pipeline: MUF, LOF,
group range, SNR and the fit columns land in ``extraction``, and the sounding
page and ``/soundings/{id}/sao.xml`` build the full scaling from them. There
is no separate "compute characteristics" step to run, on ``.lfs`` or anything
else.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import db, watch

#: How many soundings one pass may ingest. A first index of a large folder is
#: hours of pipeline; unbounded, it would hold one transaction open for all of
#: it and report nothing until the end. Bounded, each pass commits and the page
#: shows progress, and the periodic loop picks the rest up.
DEFAULT_BATCH = int(os.environ.get("ARCHIVE_SCAN_BATCH", "200"))

#: Seconds between automatic passes over the enabled archives. ``0`` disables
#: the loop entirely, which is what the tests and a CLI-only deployment want.
DEFAULT_INTERVAL_S = float(os.environ.get("ARCHIVE_SCAN_INTERVAL_S", "900"))

#: Worker processes for one pass, handed to the pipeline.
DEFAULT_JOBS = int(os.environ.get("ARCHIVE_SCAN_JOBS", "1"))

#: One scan at a time, process-wide. See the module docstring.
_LOCK = threading.Lock()

#: What the page polls. Guarded by `_STATUS_LOCK` rather than `_LOCK`, so
#: reading the status never waits behind a scan that holds the other one.
_STATUS_LOCK = threading.Lock()


@dataclass
class Status:
    """What a scan is doing, for a page that cannot see the thread."""

    archive_id: int | None = None
    name: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    result: str = ""
    ok: bool | None = None
    error: str = ""

    @property
    def running(self) -> bool:
        return self.started_at is not None and self.finished_at is None

    def as_dict(self) -> dict:
        out = {
            "running": self.running,
            "archive_id": self.archive_id,
            "name": self.name,
            "result": self.result,
            "ok": self.ok,
            "error": self.error,
        }
        if self.started_at is not None:
            out["elapsed_s"] = round(
                (self.finished_at or time.time()) - self.started_at, 1)
        return out


_STATUS = Status()


def status() -> dict:
    with _STATUS_LOCK:
        return _STATUS.as_dict()


def _set_status(**fields) -> None:
    with _STATUS_LOCK:
        for key, value in fields.items():
            setattr(_STATUS, key, value)


def is_scanning() -> bool:
    return status()["running"]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

class ArchiveError(ValueError):
    """A folder that cannot be registered, with the reason for the operator."""


def resolve(relpath: str, archive_root: str | os.PathLike) -> tuple[str, Path]:
    """``(stored relpath, absolute path)``, or refuse with why.

    The stored form is always relative to ``archive_root``. An absolute path
    is accepted *only* if it already sits inside that root, and is stored
    relative anyway -- so a database written on the host still resolves inside
    the container, where the same files live under ``/archive``.

    Refusing anything outside the root is not merely tidiness. Under Docker
    the API container mounts one host path read-only, so a folder outside it
    is not slow or awkward to read, it is **invisible** -- and a scan of an
    invisible folder succeeds while loading nothing, which is the shape of
    failure this whole page exists to end.
    """
    root = Path(archive_root).resolve()
    raw = (relpath or "").strip()
    if not raw:
        raise ArchiveError("a path is required")

    candidate = Path(raw)
    absolute = (candidate if candidate.is_absolute() else root / candidate)
    # `resolve` collapses `..` before the containment test, so traversal is
    # caught by the test rather than by pattern-matching the string.
    absolute = absolute.resolve()

    try:
        stored = absolute.relative_to(root)
    except ValueError:
        raise ArchiveError(
            f"{raw} is outside the archive root ({root}). This server can "
            f"only index what is under that root -- in the container it is "
            f"the one path mounted at /archive, so a folder elsewhere is not "
            f"just unreadable, it is invisible, and a scan of it would report "
            f"success having loaded nothing. Move the folder under the root, "
            f"or point ARCHIVE_HOST_PATH at a parent that contains it and "
            f"redeploy.") from None

    if stored == Path("."):
        raise ArchiveError(
            "that is the archive root itself. Register the folders inside it "
            "instead, so each can be scanned, disabled and reported on its "
            "own.")
    if not absolute.is_dir():
        raise ArchiveError(f"{absolute} is not a directory that exists here")
    return stored.as_posix(), absolute


def mount() -> dict:
    """What folder is actually mounted here, as far as this process can tell.

    Two paths, because under Docker they are different and only one of them is
    meaningful to the person editing `deploy/.env`:

    * ``root`` -- what this process sees. ``/archive`` in a container.
    * ``host`` -- the folder the operator mounted there, from
      ``ARCHIVE_HOST_PATH``. The container cannot discover this; compose has to
      pass it in, which is why it is now in the ``environment:`` block beside
      the volume that uses it. Absent, the page says so rather than inventing
      one.

    ``readable`` is checked rather than assumed. A bind mount whose source was
    renamed on the host still exists inside the container -- as an empty
    directory. Every scan then reports "0 on disk" truthfully and forever, and
    that is the state this whole page exists to make visible.
    """
    root = Path(os.environ.get("ARCHIVE_ROOT", "."))
    host = os.environ.get("ARCHIVE_HOST_PATH") or ""
    exists = root.is_dir()
    entries = None
    if exists:
        try:
            entries = sum(1 for _ in root.iterdir())
        except OSError:
            exists = False
    return {
        "root": str(root),
        "host": host,
        "in_container": Path("/.dockerenv").exists(),
        "exists": exists,
        "entries": entries,
        "empty": exists and entries == 0,
    }


def candidates(conn, archive_root, *, limit: int = 60) -> list[dict]:
    """Folders under the root that could be registered, and what is in them.

    So that adding one is picking from what is mounted rather than typing a
    path and hoping. A folder already registered is listed too, marked, so the
    answer to "is this one indexed?" is on the same screen as the folders.

    Counts come from `loader.find_soundings`, which walks the tree, so this is
    a directory scan per candidate. Bounded by `limit` because a root with
    hundreds of dated day-folders would otherwise turn a page load into a walk
    of the whole archive.
    """
    root = Path(archive_root)
    if not root.is_dir():
        return []
    registered = {row["relpath"]: row for row in db.archives(conn)}
    out = []
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    for child in children[:limit]:
        name = child.name
        try:
            found = survey(child)
        except ArchiveError:
            continue
        row = registered.get(name)
        out.append({
            "path": name,
            "soundings": found["soundings"],
            "by_format": found["by_format"],
            "registered": row is not None,
            "archive_id": row["id"] if row else None,
        })
    return out


#: Why a method cannot be used here, when it cannot. Keyed by method name.
#:
#: Checked rather than assumed because requesting an unusable method is not a
#: loud failure, it is a **silent loop**: `watch.already_done` counts a
#: sounding finished only when it holds a row for every requested method, so a
#: method that never produces one leaves every sounding in the archive
#: permanently unfinished and re-scans the whole folder on every pass, forever.
def method_availability() -> dict[str, dict]:
    from muf import extractors

    importable = set(extractors.available())
    out: dict[str, dict] = {}
    for name in extractors.ALL_METHODS:
        if name not in importable:
            out[name] = {"usable": False,
                         "why": "not installed on this server"}
            continue
        if name == "cnn":
            # Importable is not the same as usable: the CNN needs a model
            # trained on this geometry, and without one it raises per file.
            try:
                from muf.extractors import cnn as cnn_mod

                cnn_mod.find_model()
            except Exception as exc:                          # noqa: BLE001
                out[name] = {"usable": False,
                             "why": str(exc).split(".")[0] or type(exc).__name__}
                continue
        out[name] = {"usable": True, "why": ""}
    return out


def usable_methods() -> tuple[str, ...]:
    return tuple(n for n, v in method_availability().items() if v["usable"])


def survey(path: Path, format: str | None = None) -> dict:
    """What an indexer would find in a folder, before anything is committed.

    Registration reports this so a folder that will index nothing is caught
    while the operator is looking at it, rather than after a scan that
    truthfully says it loaded zero.
    """
    from muf import loader

    try:
        found = loader.find_soundings(path, format=format)
    except FileNotFoundError:
        return {"soundings": 0, "by_format": {}}
    except loader.FormatError as exc:
        raise ArchiveError(str(exc)) from None

    by_format: dict[str, int] = {}
    for fmt in (loader.LFS, loader.CHIRP2, loader.DIGISONDE):
        if format is not None and fmt != format:
            continue
        try:
            by_format[fmt] = len(loader.find_soundings(path, format=fmt))
        except FileNotFoundError:
            continue
    return {"soundings": len(found),
            "by_format": {k: v for k, v in by_format.items() if v}}


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def methods_of(row: dict) -> tuple[str, ...]:
    return tuple(m.strip() for m in (row["methods"] or "").split(",") if m.strip())


class ArchiveGone(ArchiveError):
    """A registered folder that is no longer there."""


def scan_once(row: dict, *, archive_root, db_path=None, batch=None,
              jobs=None, min_age_s=watch.DEFAULT_MIN_AGE_S) -> dict:
    """One pass over one archive. Blocking; the caller decides about threads.

    **The folder must still exist.** `watch.find_new` treats a target holding
    nothing as skippable, which is right for a target somebody just typed on
    the command line -- an archive holds detection trees and empty days beside
    the ionograms, and one of those must not stop a scan. It is wrong for a
    *stored* registration, which can go stale on its own: an unmounted share
    or a folder renamed underneath us would scan clean, report "0 on disk, 0
    new", and set `last_scan_ok` for as long as anyone left it running. That
    is a page full of green saying nothing is arriving, which is the failure
    this whole thing exists to stop being invisible.
    """
    target = Path(archive_root) / row["relpath"]
    if not target.is_dir():
        raise ArchiveGone(
            f"{row['relpath']} is registered but not on disk at {target}. "
            f"Nothing was scanned. A share that stopped being mounted looks "
            f"exactly like an empty folder to the indexer, so this is "
            f"reported rather than counted as a clean pass.")
    conn = watch.connect(db_path)
    try:
        result = watch.run_once(
            [target], conn,
            methods=methods_of(row), archive_root=Path(archive_root),
            jobs=DEFAULT_JOBS if jobs is None else jobs,
            batch=DEFAULT_BATCH if batch is None else batch,
            min_age_s=min_age_s, quiet=True)
        db.record_scan(conn, row["id"], result=watch.describe(result), ok=True)
        return result
    finally:
        conn.close()


def scan(row: dict, *, archive_root, db_path=None, **kw) -> bool:
    """Take the lock and scan, recording what happened. ``False`` if busy.

    Every outcome is written to the archive row, failures included: a scan
    that raised must not leave the previous pass's cheerful summary standing
    as though it were current.
    """
    if not _LOCK.acquire(blocking=False):
        return False
    try:
        _set_status(archive_id=row["id"], name=row["name"],
                    started_at=time.time(), finished_at=None,
                    result="", ok=None, error="")
        try:
            result = scan_once(row, archive_root=archive_root,
                               db_path=db_path, **kw)
            _set_status(finished_at=time.time(), ok=True,
                        result=watch.describe(result))
        except Exception as exc:                              # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            _set_status(finished_at=time.time(), ok=False, error=message,
                        result=f"failed -- {message}")
            conn = watch.connect(db_path)
            try:
                db.record_scan(conn, row["id"],
                               result=f"failed -- {message}", ok=False)
            finally:
                conn.close()
            return True
        return True
    finally:
        _LOCK.release()


def scan_in_background(row: dict, *, archive_root, db_path=None, **kw):
    """Start a scan on a daemon thread. Returns it, or ``None`` if busy.

    Daemon so a long pipeline cannot hold up a shutdown. Losing a pass costs
    nothing that is not recoverable: ingest is idempotent on ``(file, method)``
    and the next pass simply finds the same work.
    """
    if is_scanning():
        return None
    thread = threading.Thread(
        target=scan, args=(row,),
        kwargs={"archive_root": archive_root, "db_path": db_path, **kw},
        daemon=True, name=f"archive-scan-{row['id']}")
    thread.start()
    return thread


def scan_all(conn, *, archive_root, db_path=None, **kw) -> int:
    """One pass over every enabled archive, in order. Returns how many ran."""
    ran = 0
    for row in db.archives(conn, enabled_only=True):
        if scan(row, archive_root=archive_root, db_path=db_path, **kw):
            ran += 1
    return ran


def start_periodic(app, *, interval_s: float | None = None):
    """A daemon loop that keeps the enabled archives current.

    Off the request path for the same reason the census is, and on a loop
    rather than once because an archive, unlike a census of write-once files,
    gains files while the server runs.
    """
    interval = DEFAULT_INTERVAL_S if interval_s is None else interval_s
    if interval <= 0:
        return None

    def loop():
        while True:
            time.sleep(interval)
            try:
                conn = db.connect()
                try:
                    scan_all(conn, archive_root=app.state.archive_root)
                finally:
                    conn.close()
            except Exception as exc:                          # noqa: BLE001
                # A pass that raises must not kill the loop. The usual causes
                # -- a locked database, a share that went away -- all clear on
                # their own, and the next pass finds the same work waiting.
                print(f"{db.utcnow()}  archive scan failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    thread = threading.Thread(target=loop, daemon=True, name="archive-scan")
    thread.start()
    return thread
