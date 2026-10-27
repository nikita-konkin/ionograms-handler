"""The simulator console, built on demand: one sounding at a time, and kept.

Running the simulator on a sounding is :mod:`muf.sim.check`'s closed loop --
twelve synthetic soundings of 80 MB each, written, read and deleted -- about
half a minute of CPU for an `.lfs`. Too slow for a request and too costly to
repeat, so a request queues it, one build runs at a time in a process of its
own, and the result is kept beside the database. It is kept per image, the
image's build in the file name: a console built by another image is built
again rather than served, since the simulator it shows may have changed.

The process is for the server's sake, not the build's. The loop is heavy
numerics with long stretches of Python between them; on a thread it would
hold the interpreter the request path runs on, and every page would wait.
While it runs it says how far it has got in a file beside the console's
(:class:`services.api.apart.Progress`), which :func:`status` reads. The
file goes when the build ends, under the lock :func:`status` reads it in,
since Windows cannot remove a file being read; one another program holds
open stays behind, harmless, as it is read only while its build runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

from . import apart, build, db

#: What built a kept console. A different value builds it again.
VERSION = build.SHA or "source"

#: Soundings that may wait at once. A full queue refuses rather than grows.
QUEUE_LIMIT = 20

#: Longest one build may take before it is stopped.
TIMEOUT_S = 900.0

TEMPLATE = Path(__file__).with_name("sim_console.html")

_lock = threading.Lock()
_queue: deque[tuple[Path, Path]] = deque()
_failed: dict[Path, str] = {}
_running: Path | None = None
_worker: threading.Thread | None = None


def folder() -> Path:
    """Where built consoles are kept: ``SIM_CACHE``, or beside the database."""
    return Path(os.environ.get("SIM_CACHE") or Path(db.DEFAULT_DB).parent / "sim")


def _file(cache: Path, sounding_id: int) -> Path:
    return Path(cache) / f"{sounding_id}-{VERSION}.json"


def data(cache: Path, sounding_id: int) -> dict | None:
    """The console's data for a sounding, if this image built it."""
    try:
        return json.loads(_file(cache, sounding_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def built(cache: Path) -> set[int]:
    """The soundings whose console this image has built."""
    tail = f"-{VERSION}.json"
    return {int(f.name[:-len(tail)]) for f in Path(cache).glob("*" + tail)
            if f.name[:-len(tail)].isdigit()}


def status(cache: Path, sounding_id: int) -> dict:
    """Where a sounding's console stands: absent, queued, running, done or failed."""
    with _lock:
        return _standing(cache, sounding_id)


def _standing(cache: Path, sounding_id: int) -> dict:
    target = _file(cache, sounding_id)
    if _running == target:
        return {"state": "running", "progress": apart.progress(_told(target))}
    waiting = [t for t, _ in _queue]
    if target in waiting:
        return {"state": "queued", "ahead": waiting.index(target)}
    if data(cache, sounding_id) is not None:
        return {"state": "done"}
    error = _failed.get(target)
    return {"state": "failed", "error": error} if error else {"state": "absent"}


def run(cache: Path, sounding_id: int, path: Path) -> dict:
    """Queue a build of the sounding at ``path``, unless it is built or on its way."""
    global _worker
    with _lock:
        now = _standing(cache, sounding_id)
        if now["state"] in ("queued", "running", "done"):
            return now
        if len(_queue) >= QUEUE_LIMIT:
            raise OverflowError(f"{len(_queue)} soundings are waiting already, so try "
                                f"again when they are done")
        target = _file(cache, sounding_id)
        _failed.pop(target, None)
        _queue.append((target, Path(path)))
        if _worker is None:
            _worker = threading.Thread(target=_work, daemon=True, name="sim-console")
            _worker.start()
        return {"state": "queued", "ahead": len(_queue) - 1}


def _work() -> None:
    global _running, _worker
    while True:
        with _lock:
            if not _queue:
                _worker = None
                return
            target, path = _queue.popleft()
            _running = target
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _keep(target, _execute(path, apart.Progress(_told(target), time.time())))
            error = None
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        with _lock:
            _running = None
            if error is not None:
                _failed[target] = error
            with contextlib.suppress(OSError):
                _told(target).unlink(missing_ok=True)


def _told(target: Path) -> Path:
    """Where the build of ``target`` says how far it has got."""
    return target.with_suffix(".progress")


def _keep(target: Path, built: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    partial.write_text(json.dumps(built, ensure_ascii=False, allow_nan=False),
                       encoding="utf-8")
    os.replace(partial, target)


def _execute(path: Path, progress: apart.Progress) -> dict:
    """The console's data, built in a process of its own and stopped if it hangs,
    telling ``progress`` how far it has got."""
    return apart.run(_build, str(path), progress, timeout_s=TIMEOUT_S,
                     what=f"the simulator on {Path(path).name}")


def _build(path: str, progress: apart.Progress) -> dict:
    from muf.sim import console

    return console.build(path, progress=progress)


def page(built: dict) -> str:
    """The console page with one sounding's data in it."""
    blob = json.dumps([built], ensure_ascii=False, separators=(",", ":"))
    return TEMPLATE.read_text(encoding="utf-8").replace(
        "const DATA = /*DATA*/null", "const DATA = " + blob.replace("</", "<\\/"))
