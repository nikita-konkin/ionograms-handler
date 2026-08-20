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

**Why progress is chunked.** `watch.run_once` does enumeration and ingestion
in one call, which is right for a CLI with a tqdm bar and wrong for a page:
nothing can be reported until the whole thing returns, and on a large archive
that is many minutes of a spinner that looks exactly like a hung server. The
scan here takes `watch.find_new` and `ingest` -- the same two steps, neither
reimplemented -- and reports between chunks of them.

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

#: How many soundings one **automatic** pass may ingest. The background loop
#: runs unattended on a box that is also serving pages, so it takes a bite and
#: leaves; the next pass gets the rest.
#:
#: A pass asked for by hand is **not** capped -- see `scan_in_background`. The
#: cap once doubled as the progress mechanism, since a bounded pass at least
#: committed and reported something; now that a scan reports per chunk, that
#: job is done properly and the cap can go back to meaning what it says.
#: Applied to a manual press it was simply an obstacle: 5793 files at 200 a
#: press is 29 presses, or seven hours of waiting on the interval.
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


#: Files handed to the pipeline per chunk. The unit of progress: nothing can
#: be reported until a chunk returns, so this is the granularity of the bar
#: and also the most work that can be lost to a restart.
#:
#: Not 1. Each chunk is a fresh `ingest` call, and with `jobs > 1` that means
#: building a process pool -- per file, the pool would cost more than the work
#: in it.
DEFAULT_CHUNK = int(os.environ.get("ARCHIVE_SCAN_CHUNK", "20"))


@dataclass
class Status:
    """What a scan is doing, for a page that cannot see the thread.

    Elapsed seconds alone were the whole story here once, and they are
    indistinguishable from a hang: an operator watching "scanning, 240s
    elapsed" has no way to tell a large archive from a wedged server, and the
    reasonable guess is the wrong one. So this carries a phase and a count.

    **The phase matters as much as the count.** Before a single file can be
    reported, `watch.find_new` walks the whole tree and asks the database what
    it already holds -- minutes on a large archive, with nothing to show. Named,
    that silence is a step; unnamed, it is the part that looks broken.
    """

    archive_id: int | None = None
    name: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    result: str = ""
    ok: bool | None = None
    error: str = ""
    #: "" | "reading" (enumerating the folder) | "indexing" | "done"
    phase: str = ""
    done: int = 0
    total: int = 0
    loaded: int = 0
    skipped: int = 0

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
            "phase": self.phase,
            "done": self.done,
            "total": self.total,
            "loaded": self.loaded,
            "skipped": self.skipped,
        }
        elapsed = None
        if self.started_at is not None:
            elapsed = (self.finished_at or time.time()) - self.started_at
            out["elapsed_s"] = round(elapsed, 1)
        if self.total:
            out["percent"] = round(100.0 * self.done / self.total, 1)
            # Only once a chunk has actually returned. A rate extrapolated
            # from zero completed files is not an estimate, it is a number
            # shaped like one.
            if self.done and elapsed and self.running:
                rate = self.done / elapsed
                if rate > 0:
                    out["eta_s"] = int((self.total - self.done) / rate)
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

    Accepts a folder under any root in :func:`roots`, and stores it the way
    that root is stored: **relative** under the primary, so the row stays
    readable from the host and from inside the container, and **absolute**
    under any other, because there is nothing to be relative to that both
    would agree on. That is the same trade `ingest_row` already makes for a
    sounding outside the configured root.

    Refusing anything under none of them is not tidiness. A container's
    filesystem is fixed when it starts, so an unmounted folder is not slow or
    awkward to read, it is **invisible** -- and a scan of an invisible folder
    succeeds while loading nothing, which is the failure this page exists to
    end.
    """
    primary = Path(archive_root).resolve()
    every = [primary] + [p.resolve() for p in roots()[1:]]
    raw = (relpath or "").strip()
    if not raw:
        raise ArchiveError("a path is required")

    candidate = Path(raw)
    # A bare relative path belongs to the primary root; that is the common
    # case and the one the field is prefilled for.
    absolute = (candidate if candidate.is_absolute() else primary / candidate)
    # `resolve` collapses `..` before the containment test, so traversal is
    # caught by the test rather than by pattern-matching the string.
    absolute = absolute.resolve()

    owner = None
    for root in every:
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            continue
        owner, stored_rel = root, relative
        break

    if owner is None:
        listed = ", ".join(str(r) for r in every)
        raise ArchiveError(
            f"{raw} is under none of this server's archive roots ({listed}). "
            f"A container's filesystem is fixed when it starts, so a folder "
            f"outside them is not just unreadable, it is invisible, and a "
            f"scan of it would report success having loaded nothing. Add a "
            f"volume for it and list its container path in "
            f"{ROOTS_ENV}, then redeploy.")

    if stored_rel == Path("."):
        raise ArchiveError(
            f"that is the archive root {owner} itself. Register the folders "
            f"inside it instead, so each can be scanned, disabled and "
            f"reported on its own.")
    if not absolute.is_dir():
        raise ArchiveError(f"{absolute} is not a directory that exists here")

    # Relative under the primary root, so the row stays portable between host
    # and container. Absolute under any other, which is the same choice
    # `ingest_row` already makes for a sounding outside the configured root.
    stored = (stored_rel.as_posix() if owner == primary
              else absolute.as_posix())
    return stored, absolute


#: Extra places to look, beyond ``ARCHIVE_ROOT``. Colon-separated **container**
#: paths, like ``PATH``: ``/archive:/archive2:/archive3``.
#:
#: Colons are safe here and would not be in the host variables. Container paths
#: are always POSIX; ``ARCHIVE_HOST_PATH`` is routinely a Windows path with a
#: drive letter (``F:/MyData/ND/lfs``), which is why the host side is numbered
#: variables instead of a list.
#:
#: **This does not create mounts.** A container's filesystem is fixed when it
#: starts, so every root here still needs its own ``volumes:`` line and a
#: redeploy. What the list buys is that the api will *look* at more than one
#: place, which one variable in a volume spec could never express.
ROOTS_ENV = "ARCHIVE_ROOTS"


def roots() -> list[Path]:
    """Every root this server will index, primary first.

    The primary is ``ARCHIVE_ROOT`` and stays special: paths under it are
    stored relative, which is what keeps one database readable from the host
    and from inside the container. Paths under the others are stored absolute
    -- `ingest_row` already does this deliberately for anything outside the
    configured root, on the grounds that a half-ingested archive is worse than
    a non-portable row. The trade is real and belongs on the page, not hidden.
    """
    primary = Path(os.environ.get("ARCHIVE_ROOT", "."))
    out = [primary]
    for raw in (os.environ.get(ROOTS_ENV) or "").split(":"):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if path not in out:
            out.append(path)
    return out


def _root_host(index: int) -> str:
    """The host folder behind root ``index``, for display.

    Numbered rather than a list because a host path may contain a colon --
    ``F:/MyData/ND/lfs`` is in `.env.example` -- so the separator that works
    for container paths would split a Windows path in half.
    """
    name = "ARCHIVE_HOST_PATH" if index == 0 else f"ARCHIVE_HOST_PATH_{index + 1}"
    return os.environ.get(name) or ""


def _root_state(path: Path, index: int) -> dict:
    exists = path.is_dir()
    entries = None
    if exists:
        try:
            entries = sum(1 for _ in path.iterdir())
        except OSError:
            exists = False
    return {"root": str(path), "host": _root_host(index),
            "primary": index == 0, "exists": exists, "entries": entries,
            "empty": exists and entries == 0}


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
    every = [_root_state(path, i) for i, path in enumerate(roots())]
    primary = every[0]
    return {
        # The primary, flat, because most of the page and every caller that
        # predates multi-root asks about it directly.
        **primary,
        "in_container": Path("/.dockerenv").exists(),
        # ...and all of them, primary first.
        "roots": every,
        "extra": len(every) - 1,
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
    primary = Path(archive_root)
    every = [primary] + [p for p in roots()[1:]]
    registered = {row["relpath"]: row for row in db.archives(conn)}
    out = []
    for root in every:
        if not root.is_dir():
            continue
        try:
            children = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children[:limit]:
            try:
                found = survey(child)
            except ArchiveError:
                continue
            # The key a row is stored under: relative for the primary root,
            # absolute for the others. Same rule as `resolve`.
            key = (child.name if root == primary else str(child))
            row = registered.get(key)
            out.append({
                "path": key,
                "root": str(root),
                "primary": root == primary,
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
              jobs=None, chunk=None,
              min_age_s=watch.DEFAULT_MIN_AGE_S) -> dict:
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

    from muf import pipeline

    from . import ingest as ingest_mod

    methods = methods_of(row)
    batch = DEFAULT_BATCH if batch is None else batch
    jobs = DEFAULT_JOBS if jobs is None else jobs
    conn = watch.connect(db_path)
    try:
        # `watch.run_once` would do all of this in one call, and did until the
        # progress bar. The loop below is the same two steps it takes --
        # `find_new` then `ingest` -- split so that the second one reports
        # between chunks. Neither step is reimplemented; only the seam between
        # them is new.
        _set_status(phase="reading", done=0, total=0, loaded=0, skipped=0)
        new, found, fresh, skewed = watch.find_new(
            [target], conn, methods, min_age_s)

        held_back = 0
        if batch and len(new) > batch:
            held_back = len(new) - batch
            new = new[:batch]

        result = {"found": found, "new": len(new), "too_fresh": fresh,
                  "future_dated": skewed, "held_back": held_back,
                  "loaded": 0, "skipped": 0}
        if not new:
            _set_status(phase="done", total=0, done=0)
            db.record_scan(conn, row["id"], result=watch.describe(result),
                           ok=True)
            return result

        _set_status(phase="indexing", total=len(new), done=0)
        options = pipeline.Options(methods=methods)
        size = max(1, DEFAULT_CHUNK if chunk is None else chunk)
        for start in range(0, len(new), size):
            part = new[start:start + size]
            counts = ingest_mod.ingest(part, conn, options,
                                       archive_root=Path(archive_root),
                                       jobs=jobs, progress=False)
            result["loaded"] += counts["loaded"]
            result["skipped"] += counts["skipped"]
            _set_status(done=start + len(part), loaded=result["loaded"],
                        skipped=result["skipped"])

        _set_status(phase="done")
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

    **Uncapped by default**, unlike the background loop. Someone pressing scan
    has asked for this archive to be indexed, and stopping at
    `DEFAULT_BATCH` would answer that by doing an arbitrary fraction and
    saying "held for the next pass". They can watch the bar, and they can
    close the tab -- the scan is server-side and survives it.
    """
    if is_scanning():
        return None
    kw.setdefault("batch", 0)
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
