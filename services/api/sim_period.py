"""A period of the archive scored on the hub: ``muf sim series``, then ``summary``.

The live loop scores forecasts only from the moment a circuit is followed.
This scores the persistence forecast over a stretch the archive already
holds: an operator picks a circuit, a first and a last day and a lead, and
the hub runs :func:`muf.sim.series.run_series` on the circuit's soundings of
those days, then :func:`muf.sim.summary.summarise`. Synthetic copies and the
closed loop are left out, so a day takes minutes rather than hours; pairs the
receiver's clock broke are set apart, as a summary always sets them.

One run at a time, in a process of its own (:mod:`services.api.apart`),
saying how far it has got as a console's build does: each sounding observed,
each pair scored, then the summary. Each run is kept in a folder of its own
under :func:`folder`: ``job.json`` what was asked, and ``run/`` the series'
rows with their ``summary.json``. The states the series observed are deleted
once the summary is written; they serve a resume, and the hub never resumes.

A run covers at most :data:`MAX_DAYS` days and :data:`MAX_SOUNDINGS`
soundings, so that one run cannot hold the hub for a day. The queue is in
memory: a run queued or running when the hub stopped reads as failed, to be
run again.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import shutil
import statistics
import threading
import time
from collections import deque
from pathlib import Path

from . import apart, db

#: Days one run may cover.
MAX_DAYS = 7

#: Soundings one run may read.
MAX_SOUNDINGS = 2000

#: Runs that may wait at once. A full queue refuses rather than grows.
QUEUE_LIMIT = 5

#: Longest one run may take before it is stopped.
TIMEOUT_S = 6 * 3600.0

#: A finished run's files that may be downloaded.
DOWNLOADS = ("summary.json", "pairs.jsonl", "modes.jsonl", "carriers.jsonl",
             "skipped.jsonl", "run.json")

_lock = threading.Lock()
_queue: deque[int] = deque()
_running: int | None = None
_worker: threading.Thread | None = None


def folder(cache: Path) -> Path:
    """Where the runs are kept, under the simulator's cache."""
    return Path(cache) / "periods"


def _job(cache: Path, run_id: int) -> Path:
    return folder(cache) / str(run_id) / "job.json"


def _out(cache: Path, run_id: int) -> Path:
    return folder(cache) / str(run_id) / "run"


def _told(cache: Path, run_id: int) -> Path:
    """Where run ``run_id`` says how far it has got."""
    return folder(cache) / str(run_id) / "progress.json"


def circuits(conn) -> list[dict]:
    """Every circuit the hub holds soundings of: its first and last day, how
    many soundings, and the interval between its latest ones."""
    found = db.rows(conn, "SELECT tx, rx, MIN(datetime) AS first, "
                          "MAX(datetime) AS last, COUNT(*) AS soundings FROM sounding "
                          "WHERE tx IS NOT NULL AND rx IS NOT NULL "
                          "AND datetime IS NOT NULL GROUP BY tx, rx ORDER BY tx, rx")
    for c in found:
        recent = db.rows(conn, "SELECT datetime FROM sounding WHERE tx = ? AND rx = ? "
                               "ORDER BY datetime DESC LIMIT 13", (c["tx"], c["rx"]))
        c.update(first=str(c["first"])[:10], last=str(c["last"])[:10],
                 interval_s=_interval([r["datetime"] for r in recent]))
    return found


def _epoch(stamp: str) -> float:
    return _dt.datetime.fromisoformat(str(stamp)).replace(
        tzinfo=_dt.timezone.utc).timestamp()


def _interval(stamps: list[str]) -> float | None:
    """The median gap between soundings, s; None for fewer than two."""
    times = sorted(_epoch(s) for s in stamps)
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return float(statistics.median(gaps)) if gaps else None


def submit(conn, cache: Path, root: Path, *, tx: str, rx: str, first: _dt.date,
           last: _dt.date, lead_s: float | None = None) -> dict:
    """Queue a run over the circuit ``tx`` -> ``rx``'s soundings from ``first``
    to ``last``, both days whole, UTC.

    ``root`` is where a sounding's relative path is read from. ``lead_s``
    is by default the soundings' interval. ValueError for a period the limits
    refuse, LookupError for one with none of the circuit's soundings, and
    OverflowError while :data:`QUEUE_LIMIT` runs wait.
    """
    global _worker
    if last < first:
        raise ValueError(f"the last day, {last}, is before the first, {first}")
    days = (last - first).days + 1
    if days > MAX_DAYS:
        raise ValueError(f"{days} days asked for, and a run covers at most {MAX_DAYS}")
    if lead_s is not None and not lead_s > 0:
        raise ValueError(f"a lead of {lead_s:g} s forecasts nothing")
    found = db.rows(conn, "SELECT path, datetime FROM sounding "
                          "WHERE tx = ? AND rx = ? AND datetime >= ? AND datetime < ? "
                          "ORDER BY datetime, id",
                    (tx, rx, first.isoformat(),
                     (last + _dt.timedelta(days=1)).isoformat()))
    if not found:
        raise LookupError(f"no soundings of {tx} → {rx} from {first} to {last}")
    if len(found) > MAX_SOUNDINGS:
        raise ValueError(f"{len(found)} soundings from {first} to {last}, and a run "
                         f"reads at most {MAX_SOUNDINGS}: choose fewer days")
    lead = lead_s or _interval([r["datetime"] for r in found])
    if lead is None:
        raise ValueError(f"one sounding of {tx} → {rx} from {first} to {last} makes "
                         f"no pair")
    paths = [str(p if p.is_absolute() else Path(root) / p)
             for p in (Path(r["path"]) for r in found)]
    with _lock:
        if len(_queue) >= QUEUE_LIMIT:
            raise OverflowError(f"{len(_queue)} runs are waiting already, so try again "
                                f"when they are done")
        held = [int(p.name) for p in folder(cache).glob("*") if p.name.isdigit()]
        run_id = max(held, default=0) + 1
        job = {"id": run_id, "tx": tx, "rx": rx, "first": first.isoformat(),
               "last": last.isoformat(), "lead_s": float(lead), "soundings": len(paths),
               "asked_at": db.utcnow(), "paths": paths}
        _job(cache, run_id).parent.mkdir(parents=True)
        _keep(_job(cache, run_id), job)
        _queue.append(run_id)
        if _worker is None:
            _worker = threading.Thread(target=_work, args=(Path(cache),), daemon=True,
                                       name="sim-period")
            _worker.start()
        return _standing(cache, job)


def _keep(path: Path, job: dict) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    partial.replace(path)


def _read(cache: Path, run_id: int) -> dict | None:
    try:
        return json.loads(_job(cache, run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(cache: Path, run_id: int) -> dict:
    """What run ``run_id`` was asked to do, and where it stands: queued,
    running, done or failed. LookupError for a run never asked for."""
    with _lock:
        job = _read(cache, run_id)
        if job is None:
            raise LookupError(f"no run {run_id}")
        return _standing(cache, job)


def runs(cache: Path) -> list[dict]:
    """Every run kept, newest first, each as :func:`status` gives it."""
    held = sorted((int(p.name) for p in folder(cache).glob("*") if p.name.isdigit()),
                  reverse=True)
    with _lock:
        jobs = [job for job in (_read(cache, k) for k in held) if job is not None]
        return [_standing(cache, job) for job in jobs]


def _standing(cache: Path, job: dict) -> dict:
    run_id = job["id"]
    shown = {k: v for k, v in job.items() if k != "paths"}
    if _running == run_id:
        return {**shown, "state": "running",
                "progress": apart.progress(_told(cache, run_id))}
    if run_id in _queue:
        return {**shown, "state": "queued", "ahead": list(_queue).index(run_id)}
    if (_out(cache, run_id) / "summary.json").is_file():
        return {**shown, "state": "done"}
    return {**shown, "state": "failed",
            "error": job.get("error") or "the hub stopped before the run finished, so "
                                         "score the period again"}


def summary(cache: Path, run_id: int) -> dict | None:
    """A finished run's summary; None before it has one."""
    try:
        return json.loads((_out(cache, run_id) / "summary.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def files(cache: Path, run_id: int) -> list[str]:
    """The :data:`DOWNLOADS` run ``run_id`` wrote."""
    return [name for name in DOWNLOADS if (_out(cache, run_id) / name).is_file()]


def download(cache: Path, run_id: int, name: str) -> Path:
    """One of a finished run's :data:`DOWNLOADS`; LookupError for any other
    file, or one the run did not write."""
    if name not in DOWNLOADS:
        raise LookupError(f"{name} is not one of a run's files: "
                          f"{', '.join(DOWNLOADS)}")
    path = _out(cache, run_id) / name
    if not path.is_file():
        raise LookupError(f"run {run_id} wrote no {name}")
    return path


def _work(cache: Path) -> None:
    global _running, _worker
    while True:
        with _lock:
            if not _queue:
                _worker = None
                return
            run_id = _queue.popleft()
            _running = run_id
        job = _read(cache, run_id) or {"id": run_id, "paths": [], "lead_s": 0.0}
        begun = time.monotonic()
        try:
            _execute(job["paths"], str(_out(cache, run_id)), job["lead_s"],
                     apart.Progress(_told(cache, run_id), time.time()))
        except Exception as exc:
            job["error"] = str(exc) or type(exc).__name__
        job.update(finished_at=db.utcnow(), seconds=round(time.monotonic() - begun, 1))
        with _lock:
            _running = None
            with contextlib.suppress(OSError):
                _keep(_job(cache, run_id), job)
            with contextlib.suppress(OSError):
                _told(cache, run_id).unlink(missing_ok=True)


def _execute(paths: list[str], out: str, lead_s: float,
             progress: apart.Progress) -> None:
    """:func:`_score`, in a process of its own and stopped if it runs too long."""
    apart.run(_score, paths, out, lead_s, progress, timeout_s=TIMEOUT_S,
              what=f"the period of {len(paths)} soundings")


def _score(paths: list[str], out: str, lead_s: float, progress) -> None:
    """The series over ``paths`` into ``out``, then its summary, then its
    states deleted. ``progress`` hears the series' steps and the summary's."""
    from muf.sim import series
    from muf.sim import summary as summing

    out = Path(out)
    series.run_series(paths, out, lead_s=lead_s, copies=0, recovery_every=0, jobs=1,
                      progress=lambda done, total: progress(done, total + 1))
    found = summing.summarise(out)
    (out / "summary.json").write_text(json.dumps(found, indent=1) + "\n",
                                      encoding="utf-8")
    shutil.rmtree(out / "states", ignore_errors=True)
