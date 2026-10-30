"""A period's forecasts summed up on the hub, as ``muf sim summary`` sums a run.

The live loop scores forecasts only from the moment a circuit is followed.
A period of the archive is scored afresh: an operator picks a circuit, a
first and a last day and a lead, and the hub runs
:func:`muf.sim.series.run_series` on the circuit's soundings of those days,
then :func:`muf.sim.summary.summarise`. Synthetic copies and the closed loop
are left out, so a day takes minutes rather than hours; pairs the receiver's
clock broke are set apart, as a summary always sets them.

A period of the live loop is not scored again: the forecasts the loop scored
then, kept in ``live_event``, are written out as a series writes its rows
(:func:`muf.sim.live.as_run`) and summed up the same way, so the operational
forecast is measured as the archive is, over the days it ran (:data:`SOURCES`).

One run at a time, in a process of its own (:mod:`services.api.apart`),
saying how far it has got as a console's build does: each sounding observed,
each pair scored, then the summary. Each run is kept in a folder of its own
under :func:`folder`: ``job.json`` what was asked, and ``run/`` the series'
rows with their ``summary.json``. The states the series observed are deleted
once the summary is written; they serve a resume, and the hub never resumes.

A run covers at most :data:`MAX_DAYS` days and, of the archive,
:data:`MAX_SOUNDINGS` soundings, so that one run cannot hold the hub for a
day. The queue is in
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

#: Where a period's forecasts come from: the archive, scored afresh, or the
#: live loop, as it scored them.
SOURCES = ("archive", "live")

#: The leads the live loop forecast are offered to this, s.
LEAD_STEP_S = 10.0

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


def live_circuits(conn) -> list[dict]:
    """Every circuit the live loop scored forecasts on: its first and last
    day, and the leads it forecast, s, each to :data:`LEAD_STEP_S`."""
    leads: dict[tuple[str, str], set[float]] = {}
    for r in db.rows(conn, "SELECT DISTINCT tx, rx, lead_s FROM live_event "
                           "WHERE kind = 'scored'"):
        leads.setdefault((r["tx"], r["rx"]), set()).add(_lead(r["lead_s"]))
    spans = [(tx, rx, db.one(conn, "SELECT MIN(epoch) AS first, MAX(epoch) AS last "
                                   "FROM live_event WHERE tx = ? AND rx = ?", (tx, rx)))
             for tx, rx in sorted(leads)]
    return [{"tx": tx, "rx": rx, "first": _day(span["first"]),
             "last": _day(span["last"]), "leads": sorted(leads[(tx, rx)])}
            for tx, rx, span in spans]


def _lead(lead_s: float) -> float:
    return round(float(lead_s) / LEAD_STEP_S) * LEAD_STEP_S


def _day(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).date().isoformat()


def _epoch(stamp: str) -> float:
    return _dt.datetime.fromisoformat(str(stamp)).replace(
        tzinfo=_dt.timezone.utc).timestamp()


def _interval(stamps: list[str]) -> float | None:
    """The median gap between soundings, s; None for fewer than two."""
    times = sorted(_epoch(s) for s in stamps)
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return float(statistics.median(gaps)) if gaps else None


def submit(conn, cache: Path, root: Path, *, tx: str, rx: str, first: _dt.date,
           last: _dt.date, lead_s: float | None = None,
           source: str = "archive") -> dict:
    """Queue a run over the circuit ``tx`` -> ``rx``'s forecasts from ``first``
    to ``last``, both days whole, UTC, taken from ``source``, one of
    :data:`SOURCES`.

    ``root`` is where a sounding's relative path is read from. ``lead_s`` is
    by default the soundings' interval, or the shortest lead the live loop
    forecast. ValueError for a period the limits refuse, LookupError for one
    with nothing of the circuit's to sum up, and OverflowError while
    :data:`QUEUE_LIMIT` runs wait.
    """
    global _worker
    if source not in SOURCES:
        raise ValueError(f"{source!r} is not where forecasts come from, which is "
                         f"one of {', '.join(SOURCES)}")
    if last < first:
        raise ValueError(f"the last day, {last}, is before the first, {first}")
    days = (last - first).days + 1
    if days > MAX_DAYS:
        raise ValueError(f"{days} days asked for, and a run covers at most {MAX_DAYS}")
    if lead_s is not None and not lead_s > 0:
        raise ValueError(f"a lead of {lead_s:g} s forecasts nothing")
    asked = (_live(conn, tx, rx, first, last, lead_s) if source == "live"
             else _archive(conn, Path(root), tx, rx, first, last, lead_s))
    with _lock:
        if len(_queue) >= QUEUE_LIMIT:
            raise OverflowError(f"{len(_queue)} runs are waiting already, so try again "
                                f"when they are done")
        held = [int(p.name) for p in folder(cache).glob("*") if p.name.isdigit()]
        run_id = max(held, default=0) + 1
        job = {"id": run_id, "source": source, "tx": tx, "rx": rx,
               "first": first.isoformat(), "last": last.isoformat(), **asked,
               "asked_at": db.utcnow()}
        _job(cache, run_id).parent.mkdir(parents=True)
        if source == "live":
            _copy(conn, tx, rx, first, last, job["lead_s"], _out(cache, run_id))
        _keep(_job(cache, run_id), job)
        _queue.append(run_id)
        if _worker is None:
            _worker = threading.Thread(target=_work, args=(Path(cache),), daemon=True,
                                       name="sim-period")
            _worker.start()
        return _standing(cache, job)


def _archive(conn, root: Path, tx: str, rx: str, first: _dt.date, last: _dt.date,
             lead_s: float | None) -> dict:
    """The lead, the soundings and their paths a run of the archive reads."""
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
    paths = [str(p if p.is_absolute() else root / p)
             for p in (Path(r["path"]) for r in found)]
    return {"lead_s": float(lead), "soundings": len(paths), "paths": paths}


def _span(tx: str, rx: str, first: _dt.date, last: _dt.date) -> tuple:
    """The live events of ``tx`` -> ``rx`` from ``first`` to ``last``, as the
    terms of a query on ``live_event``."""
    return (tx, rx, _epoch(first.isoformat()),
            _epoch((last + _dt.timedelta(days=1)).isoformat()))


def _live(conn, tx: str, rx: str, first: _dt.date, last: _dt.date,
          lead_s: float | None) -> dict:
    """The lead a run of the live loop sums up, and the soundings the loop
    turned on."""
    from muf.sim.live import TOLERANCE_S

    where = "tx = ? AND rx = ? AND epoch >= ? AND epoch < ?"
    span = _span(tx, rx, first, last)
    shortest = db.one(conn, f"SELECT MIN(lead_s) AS lead FROM live_event "
                            f"WHERE {where} AND kind = 'scored'", span)["lead"]
    if shortest is None:
        raise LookupError(f"no live forecasts of {tx} → {rx} scored from {first} "
                          f"to {last}")
    lead = lead_s or _lead(shortest)
    if not db.one(conn, f"SELECT COUNT(*) AS n FROM live_event WHERE {where} "
                        f"AND kind = 'scored' AND ABS(lead_s - ?) <= ?",
                  (*span, lead, TOLERANCE_S))["n"]:
        raise LookupError(f"no live forecasts of {tx} → {rx} {lead / 60:g} min ahead "
                          f"scored from {first} to {last}")
    return {"lead_s": float(lead),
            "soundings": db.one(conn, f"SELECT COUNT(DISTINCT sounding_id) AS n "
                                      f"FROM live_event WHERE {where}", span)["n"]}


def _copy(conn, tx: str, rx: str, first: _dt.date, last: _dt.date, lead_s: float,
          out: Path) -> None:
    """The live loop's forecasts of ``lead_s`` from ``first`` to ``last``
    written to ``out`` as a series writes its rows."""
    from muf.sim import live

    live.as_run((json.loads(r["body"]) for r in conn.execute(
        "SELECT body FROM live_event WHERE tx = ? AND rx = ? AND epoch >= ? "
        "AND epoch < ? AND kind IN ('scored', 'lapsed') ORDER BY epoch, id",
        _span(tx, rx, first, last))), out, lead_s=lead_s)


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
    shown = {"source": "archive", **{k: v for k, v in job.items() if k != "paths"}}
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
            if job.get("source") == "live":
                _sum(str(_out(cache, run_id)))
            else:
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


def _sum(out: str) -> None:
    """:func:`_summarise`, in a process of its own and stopped if it runs too
    long."""
    apart.run(_summarise, out, timeout_s=TIMEOUT_S, what=f"the summary of {out}")


def _score(paths: list[str], out: str, lead_s: float, progress) -> None:
    """The series over ``paths`` into ``out``, then its summary, then its
    states deleted. ``progress`` hears the series' steps and the summary's."""
    from muf.sim import series

    series.run_series(paths, Path(out), lead_s=lead_s, copies=0, recovery_every=0,
                      jobs=1, progress=lambda done, total: progress(done, total + 1))
    _summarise(out)
    shutil.rmtree(Path(out) / "states", ignore_errors=True)


def _summarise(out: str) -> None:
    """The summary of the rows in ``out``, written beside them."""
    from muf.sim import summary as summing

    (Path(out) / "summary.json").write_text(
        json.dumps(summing.summarise(Path(out)), indent=1) + "\n", encoding="utf-8")
