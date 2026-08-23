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

import errno
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from . import daydir, db, watch

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

#: Worker processes for one pass, handed to the pipeline. ``0`` means one per
#: core bar one.
#:
#: Still 1 by default: indexing shares this box with the pages it serves, and
#: taking every core for a background scan makes the console crawl exactly when
#: someone has opened it to watch the scan. Raising it is now safe -- the
#: pipeline no longer *forks* its workers
#: (`muf.pipeline.POOL_START_METHODS`), which is what made `jobs > 1` deadlock
#: a server that had already rendered an ionogram.
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
#:
#: 50 rather than 20 because the pool is rebuilt per chunk and a small chunk
#: never amortises it. Measured on 90 real soundings at `jobs=4`: 20 gave
#: 5.1 s, 45 gave 3.8 s, 90 gave 3.2 s, against 14.2 s at `jobs=1`. The pull
#: the other way is that a chunk is also the unit of progress and the most work
#: a restart can lose, so this stops well short of "the whole folder".
DEFAULT_CHUNK = int(os.environ.get("ARCHIVE_SCAN_CHUNK", "50"))


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

    # The root itself is allowed, and is the right answer whenever the day
    # directories sit directly under it. Refusing it used to force one archive
    # row per day -- fifteen of them on the station's own rig -- and made every
    # new day the receiver created into a manual registration. `find_soundings`
    # is recursive, so registering the folder that *contains* the days is what
    # makes new ones arrive on their own.
    #
    # It is stored as "." rather than "": that is what `Path.relative_to`
    # returns, `archive_root / "."` is `archive_root`, and a row with an empty
    # string in it reads like a bug.
    try:
        present = absolute.is_dir()
    except OSError as exc:
        # "not a directory that exists here" would be a lie about a folder that
        # does exist on a share that stopped answering, and would send the
        # operator looking for a path problem they do not have.
        raise ArchiveError(
            f"{absolute} could not be read -- {exc}. The folder may well be "
            f"there; the mount is not answering. Fix it on the host and "
            f"register again.") from None
    if not present:
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


#: Errnos meaning "something is mounted here and the storage behind it is not
#: answering" -- as opposed to "nothing is mounted here". The distinction is
#: the whole remedy: a missing bind mount is fixed in `deploy/.env` and a
#: redeploy, while a NAS that has gone away is fixed on the host and a
#: redeploy changes nothing. Sending an operator to the wrong one of those
#: costs an afternoon, and the work server produced both EIO and EHOSTDOWN in
#: the same minute on 2026-08-21.
UNREACHABLE_ERRNOS = frozenset(filter(None, (
    errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.ENOTCONN,
    errno.ECONNABORTED, errno.ECONNRESET,
    getattr(errno, "EHOSTDOWN", None), getattr(errno, "EHOSTUNREACH", None),
    getattr(errno, "EREMOTEIO", None), getattr(errno, "ENOLINK", None),
)))

#: Mounted and readable by somebody, just not by this process.
DENIED_ERRNOS = frozenset({errno.EACCES, errno.EPERM})


def mount_fault(exc: OSError) -> str:
    """Which kind of unreadable this is, and therefore which fix applies."""
    if exc.errno in UNREACHABLE_ERRNOS:
        return "unreachable"
    if exc.errno in DENIED_ERRNOS:
        return "denied"
    return "missing"


def listable(path: Path) -> bool:
    """``path.is_dir()``, with a dead mount answering False instead of raising.

    ``Path.is_dir()`` swallows ENOENT and ENOTDIR and nothing else, so every
    way a *mounted* share can fail underneath it -- EIO, ESTALE, EHOSTDOWN --
    comes back out as an OSError. On a page that walks several roots, one dead
    share would take the whole page down with it, which is how ``/ui/archives``
    came to answer 500 at exactly the moment it was the page you needed.

    Answering False rather than reporting the fault is right *here* only
    because :func:`_root_state` reports it, on the same page, in the panel that
    exists for it. A caller without that panel should use :func:`mount_fault`.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def _root_state(path: Path, index: int) -> dict:
    """What this root looks like, or why it cannot be looked at.

    **Never raises.** This function exists to report the mount's condition, so
    an unreachable mount is its subject, not an error case -- and it is the
    condition that most needs reporting, because nothing else on the server
    notices. `/healthz` answers "is the process up" and deliberately touches no
    storage, so a NAS that has gone away leaves a container marked healthy
    while this page 500s.

    ``Path.is_dir()`` is the trap and the reason for the outer guard: it
    swallows only ENOENT, ENOTDIR, EBADF and ELOOP. A network mount that is
    present but sick raises EIO, ESTALE, EACCES or ETIMEDOUT, and every one of
    those propagates. `iterdir` was already guarded; `is_dir` was not, so the
    page died on the one call that had no reason to be trusted.
    """
    populated = None
    error = fault = None
    try:
        exists = path.is_dir()
    except OSError as exc:
        # Present in the namespace, unreadable in practice: the mount is there
        # and the filesystem behind it is not answering.
        exists, error, fault = False, str(exc), mount_fault(exc)
    if exists:
        try:
            # One entry, not a count, and `os.scandir` rather than `iterdir`.
            # This runs on `GET /archives`, which the page polls once a second
            # while a scan is running. Counting was `sum(1 for _ in
            # path.iterdir())` -- a full enumeration of the root, and on
            # Python 3.12 `Path.iterdir` is `os.listdir` underneath, so it
            # reads the whole directory eagerly before yielding anything. On a
            # local disk with nineteen day-folders that is 0.1 ms. On an SMB
            # share it is 6.3 ms per entry (`sources.DEFAULT_MAX_AGE_S` records
            # 293.8 s for 46,436), once a second, competing with the indexer
            # for the very mount this panel exists to report on -- which is how
            # polling the page could take the share down.
            #
            # `os.scandir` is lazy, so the first entry costs one round trip and
            # the rest are never fetched. What the panel needs is whether the
            # mount answers and whether anything is in it; both are in that
            # first entry. The exact count belonged to the candidates panel,
            # which already has it and already caches it.
            with os.scandir(path) as entries_it:
                populated = next(entries_it, None) is not None
        except OSError as exc:
            exists, error, fault = False, str(exc), mount_fault(exc)
    elif error is None:
        fault = "missing"
    return {"root": str(path), "host": _root_host(index),
            "primary": index == 0, "exists": exists, "populated": populated,
            "empty": exists and populated is False, "error": error,
            "fault": fault}


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


#: Directories a candidate survey will not walk into. Filesystem and NAS
#: furniture, not anybody's data -- so skipping them cannot hide a folder an
#: operator meant to index, and walking them is pure cost. `#recycle` and
#: `@eaDir` are Synology's (deleted files, and thumbnail/index sidecars, the
#: latter scattered through every folder on the volume); `lost+found` is the
#: fsck bin. On a 16 TB general-purpose volume this is the difference between
#: a survey that finishes and one that does not.
#:
#: Deliberately NOT a general exclusion list. Folders with ordinary names
#: stay on offer even when they turn out to hold nothing -- the page reports
#: "nothing this server can read" beside them, which is an answer. Silently
#: omitting a folder someone is looking for is not.
UNWALKABLE = frozenset({"#recycle", "#snapshot", "@eaDir", "lost+found",
                        ".Trash", ".Trashes", "$RECYCLE.BIN",
                        "System Volume Information"})


def _worth_surveying(path: Path) -> bool:
    name = path.name
    return name not in UNWALKABLE and not name.startswith(".")


#: How far below a root to look for a dataset folder. Two is enough for both
#: layouts seen in service -- day directories at the root, and day directories
#: one folder down -- and the bound is what stops discovery walking an archive
#: of 46,000 files looking for a third.
DISCOVERY_DEPTH = 2

#: How many day folders to probe before concluding one holds no soundings.
#: A dataset is recognised by its *first* day with data, so this only bites on
#: a folder that is not a dataset, where the answer is "no" either way.
PROBE_DAYS = 8


def datasets(root: Path, *, max_depth: int = DISCOVERY_DEPTH) -> list[Path]:
    """Folders that hold daily sounding data, shallowest first.

    **What the archives page should be offering.** Listing a root's immediate
    subdirectories offers whatever happens to be one level down, which for the
    station's layout is 18 day folders -- so registering an archive means
    registering every day by hand, and every new day the receiver creates
    needs another one. It also offers folders with nothing in them, because
    nothing filtered on whether a candidate held soundings at all.

    A folder is a **dataset** when its day-named children hold soundings, or
    when it holds sounding files directly. That single rule covers both
    layouts in service:

    * ``/archive/2026-08-04/*.h5`` -- the root itself is the dataset. One
      registration, and a new day appears inside it with no action at all,
      because `find_soundings` is recursive.
    * ``/archive/ionozond_data2/2026-08-04/*.h5`` -- the root's children are
      not days, so it descends and offers ``ionozond_data2`` and its siblings
      separately, which is what keeps a `.lfs` folder and a `chirp2` folder on
      their own formats and their own estimators.

    A dataset is never descended into, so a folder and the day directories
    under it are never both offered. That is the property that makes the list
    a set of choices rather than an inventory.
    """
    from muf import loader

    found: list[Path] = []
    queue: list[tuple[Path, int]] = [(Path(root), 0)]

    while queue:
        path, depth = queue.pop(0)
        try:
            children = sorted(p for p in path.iterdir()
                              if p.is_dir() and _worth_surveying(p))
        except OSError:
            # One unreadable folder is one folder left off the list, never a
            # discovery pass that fails. See `listable`.
            continue

        # Newest first: a dataset is recognised by its first day with data,
        # and the newest day is the one most likely to have any. By the date
        # the name means, not by the name -- only `PROBE_DAYS` of these are
        # ever opened, so getting the order wrong spends the whole budget on
        # the oldest folders. See `daydir.newest`.
        days = daydir.newest(children)
        if (days and any(loader.has_soundings(day) for day in days[:PROBE_DAYS])
                or loader.has_soundings(path, recursive=False)):
            found.append(path)

        # Descend into the children that are *not* days, whether or not this
        # folder was itself a dataset. A dataset's day folders are covered by
        # it and must never be offered separately -- that was the original
        # bug -- but a folder sitting *beside* those days is a different
        # archive that happens to share a parent, and stopping here made it
        # invisible. That is the exact case of adding a folder of older `.lfs`
        # recordings next to a receiver writing `.h5` days into the same root.
        #
        # A folder offered alongside an ancestor that is also offered is not a
        # contradiction: they are two ways to register the same tree, one row
        # or several, and `overlapping` refuses taking both by accident.
        if depth < max_depth:
            queue.extend((child, depth + 1)
                         for child in children if not daydir.is_day(child))

    return found


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
        if not listable(root):
            continue
        discovered = datasets(root)
        for path in discovered[:limit]:
            # The key a row is stored under: relative for the primary root,
            # absolute for the others. Same rule as `resolve`.
            if root == primary:
                try:
                    key = path.relative_to(root).as_posix()
                except ValueError:
                    continue
            else:
                key = str(path)
            row = registered.get(key)
            if row is not None:
                # Already registered: the count is in the database, derived
                # from `sounding.path`, and walking the folder again to
                # recompute it would be the most expensive way to learn
                # something already known.
                found = {"soundings": row["soundings"], "by_format": None}
            else:
                try:
                    found = survey(path)
                except (ArchiveError, OSError):
                    # OSError as well as ArchiveError: `survey` walks the
                    # folder, and a share that dies mid-walk raises from the
                    # walk rather than from the guard above it. One candidate
                    # that cannot be counted is a candidate left off the list,
                    # not a page that fails to render.
                    continue
            out.append({
                "path": key,
                "root": str(root),
                "primary": root == primary,
                # Which other offered folder already contains this one, if
                # any. Both are legitimate choices -- one row or several --
                # and saying so beats letting the operator find out from a
                # 409 after filling in the form.
                "inside": next(
                    (str(other.relative_to(root).as_posix())
                     for other in discovered
                     if other != path and path.is_relative_to(other)), None),
                "days": _day_count(path),
                "soundings": found["soundings"],
                "by_format": found["by_format"],
                "registered": row is not None,
                "archive_id": row["id"] if row else None,
            })
    return out


def _day_count(path: Path) -> int:
    """How many day directories this dataset covers, for the page to say so.

    The number that makes a single row legible as "and every day inside it",
    which is the whole difference between this list and the one that offered
    the days themselves.
    """
    try:
        return sum(1 for child in path.iterdir()
                   if daydir.is_day(child) and child.is_dir())
    except OSError:
        return 0


#: Why a method cannot be used here, when it cannot. Keyed by method name.
#:
#: Checked rather than assumed because requesting an unusable method is not a
#: loud failure, it is a **silent loop**: `watch.already_done` counts a
#: sounding finished only when it holds a row for every requested method, so a
#: method that never produces one leaves every sounding in the archive
#: permanently unfinished and re-scans the whole folder on every pass, forever.
#: How long a candidate survey stays good for. It is a walk of the archive
#: tree, so it is priced like one: recomputed on a timer, never on demand.
#: An hour, not minutes: what folders exist on a mount changes when someone
#: puts one there, which is rare, while the walk to find out is expensive.
#: Registering or removing one calls `forget_candidates` and re-surveys at
#: once, so the slow timer never stands between an operator and a change they
#: just made -- it only governs folders that appeared behind this server's
#: back.
CANDIDATE_TTL_S = float(os.environ.get("ARCHIVE_CANDIDATE_TTL_S", "3600"))

_CAND_LOCK = threading.Lock()
_CAND: dict[str, tuple[float, list[dict]]] = {}
_CAND_REFRESHING = False

#: Bumped whenever the answer a survey would give has changed underneath one
#: that is already running -- a folder registered, a root reconfigured. A
#: refresh records this when it starts and discards its result if it moved,
#: because `forget_candidates` on its own only empties the cache: a walk that
#: began before the change lands after it and refills the cache with the old
#: answer, which then stands for the whole TTL.
_CAND_GEN = 0


def candidates_cached(conn, archive_root, *, limit: int = 60,
                      db_path=None) -> dict:
    """`candidates`, but never on the thread serving a request.

    `candidates` walks every unregistered folder under every root to count
    what is in it. That is a recursive stat of the whole archive -- fine as a
    one-off, ruinous as part of a page load, and worse than ruinous as part of
    a poll: this used to be reached from ``GET /archives``, which the page
    hits once a second while a scan runs, so a work server with three large
    .lfs folders spent every second of an index re-walking them on the request
    thread. The page did not render at all.

    So: served from a cache, refreshed in the background, and **never
    refreshed while a scan is running** -- the scan already has that disk, and
    a convenience list is not worth competing with it for one.

    Returns the list with `ready`/`age_s` beside it, so a caller that arrives
    before the first survey finishes can say "looking" instead of "none".
    """
    global _CAND_REFRESHING
    key = str(archive_root)
    now = time.time()
    with _CAND_LOCK:
        hit = _CAND.get(key)
        busy = _CAND_REFRESHING
    age = None if hit is None else now - hit[0]
    fresh = age is not None and age < CANDIDATE_TTL_S

    scanning = is_scanning()
    if not fresh and not busy and not scanning:
        with _CAND_LOCK:
            if not _CAND_REFRESHING:
                _CAND_REFRESHING = True
                busy = True
                _start_candidate_refresh(archive_root, limit,
                                         db_path, _CAND_GEN)

    return {
        "items": [] if hit is None else hit[1],
        "ready": hit is not None,
        "refreshing": busy,
        "age_s": None if age is None else round(age, 1),
        # Why an empty or old list is what it is, so the page can say so
        # rather than implying the folders are gone.
        "why": ("a scan is running -- the folder list is not refreshed while "
                "it has the disk" if scanning and not fresh else ""),
    }


def _start_candidate_refresh(archive_root, limit: int, db_path=None,
                             started_at_gen: int = 0) -> None:
    """Walk on a daemon thread, with its own connection.

    Its own because `app.state.db` is shared with every request handler, and
    a survey holds it for as long as the walk takes.

    `started_at_gen` is read by the caller, which is already inside
    `_CAND_LOCK` when it decides to refresh -- taking that lock again here
    would deadlock rather than wait.
    """
    def run() -> None:
        global _CAND_REFRESHING
        conn = None
        try:
            conn = watch.connect(db_path)
            found = candidates(conn, archive_root, limit=limit)
            with _CAND_LOCK:
                if _CAND_GEN == started_at_gen:
                    _CAND[str(archive_root)] = (time.time(), found)
        except Exception as exc:                              # noqa: BLE001
            warnings.warn(f"candidate survey failed: {exc!r}", stacklevel=2)
        finally:
            if conn is not None:
                conn.close()
            with _CAND_LOCK:
                _CAND_REFRESHING = False

    threading.Thread(target=run, daemon=True,
                     name="archive-candidates").start()


def forget_candidates() -> None:
    """Drop the cache, so the next look re-walks.

    Called when the set of registered archives changes: a folder that was
    just registered should stop being offered as a candidate immediately,
    not in five minutes.
    """
    global _CAND_GEN
    with _CAND_LOCK:
        _CAND.clear()
        _CAND_GEN += 1


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


def overlapping(conn, relpath: str, archive_root) -> list[dict]:
    """Registered archives that already cover this folder, or that it covers.

    Scanning is recursive, so a folder and its parent both being registered
    means every sounding under the child is walked, hashed and looked up twice
    on every pass. Nothing breaks -- dedup is by name, so no row is doubled --
    which is exactly why it needs saying out loud: the only symptom is a scan
    that takes twice as long for no reason anybody can see.

    This became reachable the moment the root stopped being refused. The
    station's own rig had fifteen day folders registered individually, and
    registering the root over the top of them is the obvious next move and the
    one that would silently double the work.
    """
    root = Path(archive_root)
    try:
        target = (root / relpath).resolve()
    except OSError:
        return []

    clashes = []
    for row in db.archives(conn):
        stored = Path(row["relpath"])
        other = stored if stored.is_absolute() else root / stored
        try:
            other = other.resolve()
        except OSError:
            continue
        if other == target:
            continue
        if other.is_relative_to(target):
            clashes.append(dict(row, relation="inside"))
        elif target.is_relative_to(other):
            clashes.append(dict(row, relation="contains"))
    return clashes


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
    except OSError as exc:
        # A walk of a share that stopped answering. Distinct from the
        # FileNotFoundError above, which means the folder is genuinely not
        # there: "0 soundings" would be a measurement, and this is not one.
        raise ArchiveError(
            f"{path} could not be walked -- {exc}. Nothing was counted, so "
            f"this says nothing about what the folder holds. Fix the mount on "
            f"the host and try again.") from None

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
        # The archive's own format, honoured here and not only when the
        # folder was registered. A folder narrowed to one format must stop
        # ingesting the others, or nothing removed from it stays removed.
        new, found, fresh, skewed = watch.find_new(
            [target], conn, methods, min_age_s, format=row.get("format"))

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
