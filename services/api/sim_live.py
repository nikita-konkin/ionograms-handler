"""The simulator's forecasts, issued and scored as soundings reach the hub.

:mod:`muf.sim.live` is the loop: on each sounding of a circuit, the
forecasts issued for its time are scored, and new ones are issued for the
circuit's next two soundings. This module runs it on what the archive scans
store, as it comes, for the circuits an operator chose to follow on the page
(:func:`follow`, :func:`stop`); the others cost nothing. A thread looks every
``LIVE_INTERVAL_S`` seconds (60; 0 turns it off) for each followed circuit's
soundings newer than the last it took, takes them in time order, and keeps
the circuit's ledger and every event in the database (``live_circuit``,
``live_event``).

A circuit followed begins at its newest sounding, so the archive's past is
not replayed, and its first forecasts come from the next sounding; one
stopped and followed again begins afresh. A sounding stored after a later
one was taken came too late to be forecast live and is passed over. A muted
circuit is left alone.

Each turn runs in a process of its own (:mod:`services.api.apart`): a
sounding is read and its state fitted, seconds of numerics. The states are
kept :data:`KEEP_S` in :func:`folder`, under the image's build, since a
forecast is scored against a later sounding on its own range. The events
are kept for good: they are the live forecast's record, and a period of them
is summed up as a period of the archive is (:mod:`services.api.sim_period`).
``LIVE_KEEP_DAYS`` above 0 forgets those older than that many days.
:func:`events` reads them out, each with its forecast and truth side by
side, for a copy kept elsewhere (``tools/pull_live_events.py``).

The scans set the pace. At the default of a scan every 15 minutes, a
five-minute circuit's forecasts are issued and scored three soundings at a
time; a hub that shows them live scans every minute or two
(``ARCHIVE_SCAN_INTERVAL_S``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import apart, build, db

#: Seconds between looks for new soundings; 0 turns the loop off.
INTERVAL_S = 60.0

#: Longest one turn may take before it is stopped.
TIMEOUT_S = 600.0

#: Seconds a sounding's state is kept for the forecasts that need it.
KEEP_S = 6 * 3600.0

#: Days of events kept; 0 keeps them all.
KEEP_DAYS = 0.0

#: Seconds of pairs the page's short and long figures cover.
HOUR_S, DAY_S = 3600.0, 86400.0

#: Events the page lists.
RECENT = 40

#: Events one export answers with by default, and at most.
PAGE, PAGE_MAX = 1000, 10_000


def interval() -> float:
    """Seconds between the loop's looks, ``LIVE_INTERVAL_S``; 0 when it is off."""
    return float(os.environ.get("LIVE_INTERVAL_S", INTERVAL_S))


def folder() -> Path:
    """Where the states are kept: ``LIVE_CACHE``, or beside the database, under
    the image's build."""
    return (Path(os.environ.get("LIVE_CACHE") or Path(db.DEFAULT_DB).parent / "live")
            / (build.SHA or "source"))


def _epoch(stamp: str) -> float:
    return _dt.datetime.fromisoformat(str(stamp)).replace(
        tzinfo=_dt.timezone.utc).timestamp()


def _turn(ledger: dict, path: str, epoch: float, tx: str, rx: str,
          cache: str) -> tuple[dict, list[dict]]:
    """One turn of the loop on the sounding at ``path``: the ledger after it,
    and its events."""
    from muf.sim import live, series

    found, events = live.take(live.Ledger.from_dict(ledger),
                              series.Session(Path(path), epoch, tx, rx), cache)
    return found.to_dict(), events


def _apart(ledger: dict, path: str, epoch: float, tx: str, rx: str,
           cache: str) -> tuple[dict, list[dict]]:
    return apart.run(_turn, ledger, str(path), epoch, tx, rx, str(cache),
                     timeout_s=TIMEOUT_S,
                     what=f"the live forecast on {Path(path).name}")


def _hold(conn, tx: str, rx: str, taken_at: str, ledger: dict) -> None:
    conn.execute(
        "INSERT INTO live_circuit (tx, rx, taken_at, ledger, updated_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT (tx, rx) DO UPDATE SET "
        "taken_at = excluded.taken_at, ledger = excluded.ledger, "
        "updated_at = excluded.updated_at",
        (tx, rx, taken_at, json.dumps(ledger, allow_nan=False), db.utcnow()))


def _store(conn, tx: str, rx: str, sounding: dict, events: list[dict]) -> None:
    epoch = _epoch(sounding["datetime"])
    for e in events:
        source = db.one(conn, "SELECT id FROM sounding WHERE file = ?", (e["from"],))
        conn.execute(
            "INSERT INTO live_event (tx, rx, epoch, kind, lead_s, sounding_id, "
            "source_id, body) VALUES (?,?,?,?,?,?,?,?)",
            (tx, rx, epoch, e["event"], e.get("lead_s"), sounding["id"],
             None if source is None else source["id"],
             json.dumps(e, allow_nan=False)))


def events(conn, tx: str, rx: str, *, after: int = 0, limit: int = PAGE) -> list[dict]:
    """The circuit's stored events after event ``after``, oldest first, at most
    ``limit``: each as :mod:`muf.sim.live` wrote it, with the hub's ``id``,
    ``tx`` and ``rx`` added."""
    return [{**json.loads(r["body"]), "id": r["id"], "tx": tx, "rx": rx}
            for r in db.rows(conn, "SELECT id, body FROM live_event "
                                   "WHERE tx = ? AND rx = ? AND id > ? "
                                   "ORDER BY id LIMIT ?", (tx, rx, after, limit))]


def circuits(conn) -> list[dict]:
    """Every circuit the hub holds soundings of or follows, the followed first,
    each with whether it is."""
    followed = {(r["tx"], r["rx"])
                for r in db.rows(conn, "SELECT tx, rx FROM live_circuit")}
    found = db.rows(conn, "SELECT tx, rx FROM sounding "
                          "WHERE tx IS NOT NULL AND rx IS NOT NULL "
                          "UNION SELECT tx, rx FROM live_circuit")
    return sorted(({**c, "followed": (c["tx"], c["rx"]) in followed} for c in found),
                  key=lambda c: (not c["followed"], c["tx"], c["rx"]))


def follow(conn, tx: str, rx: str) -> dict:
    """Follow the circuit ``tx`` -> ``rx`` from its newest sounding on.

    A circuit already followed is left as it is. LookupError for one with no
    soundings, which has nowhere to begin.
    """
    from muf.sim import live

    held = db.one(conn, "SELECT taken_at FROM live_circuit WHERE tx = ? AND rx = ?",
                  (tx, rx))
    if held is not None:
        return {"tx": tx, "rx": rx, "followed": True, "taken_at": held["taken_at"]}
    recent = db.rows(conn, "SELECT datetime FROM sounding WHERE tx = ? AND rx = ? "
                           "ORDER BY datetime DESC LIMIT ?", (tx, rx, live.GAPS + 1))
    if not recent:
        raise LookupError(f"no soundings of {tx} -> {rx}")
    _hold(conn, tx, rx, recent[0]["datetime"],
          live.Ledger(times=tuple(sorted(_epoch(r["datetime"]) for r in recent)))
          .to_dict())
    conn.commit()
    return {"tx": tx, "rx": rx, "followed": True, "taken_at": recent[0]["datetime"]}


def stop(conn, tx: str, rx: str) -> dict:
    """Stop following the circuit ``tx`` -> ``rx``: its forecasts waiting are
    dropped, and its events kept."""
    conn.execute("DELETE FROM live_circuit WHERE tx = ? AND rx = ?", (tx, rx))
    conn.commit()
    return {"tx": tx, "rx": rx, "followed": False}


def advance(conn, root: Path, cache: Path, *,
            turn: Callable[..., tuple[dict, list[dict]]] = _apart,
            keep_days: float = KEEP_DAYS) -> int:
    """Take each followed circuit's soundings newer than the last it took, in
    time order.

    ``root`` is where a sounding's relative path is read from, ``cache``
    where the states are kept, and ``turn`` runs one turn
    (:func:`_turn`, in a process of its own). A turn that fails is an
    ``unobserved`` event and the loop moves past its sounding, so one bad file
    cannot hold a circuit up. Returns how many soundings were taken.
    """
    from muf.sim import live

    taken = 0
    for held in db.rows(conn, "SELECT tx, rx, taken_at, ledger FROM live_circuit"):
        tx, rx = held["tx"], held["rx"]
        if db.is_muted(conn, tx, rx):
            continue
        ledger = json.loads(held["ledger"])
        for s in db.rows(conn, "SELECT id, file, path, datetime FROM sounding "
                               "WHERE tx = ? AND rx = ? AND datetime > ? "
                               "ORDER BY datetime, id", (tx, rx, held["taken_at"])):
            epoch = _epoch(s["datetime"])
            try:
                ledger, events = turn(ledger, str(Path(root) / s["path"]), epoch,
                                      tx, rx, str(cache))
            except Exception as exc:
                events = [live.event("unobserved", epoch,
                                     **{"from": s["file"], "reason": str(exc)})]
            _store(conn, tx, rx, s, events)
            _hold(conn, tx, rx, s["datetime"], ledger)
            conn.commit()
            taken += 1
    newest = db.one(conn, "SELECT MAX(epoch) AS t FROM live_event")["t"]
    if newest is not None and keep_days > 0:
        conn.execute("DELETE FROM live_event WHERE epoch < ?",
                     (newest - keep_days * 86400.0,))
        conn.commit()
    return taken


def _clock(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).strftime("%H:%M:%S")


def _stamp(epoch: float) -> str:
    """The whole moment, for the page to show in UTC and Moscow alike."""
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S")


def view(conn, tx: str, rx: str) -> dict | None:
    """What the live page shows of a circuit; None for one the loop has not begun.

    The newest sounding taken and the circuit's interval; the forecasts
    waiting; each lead's latest score, with its two soundings' ids for their
    ionograms; the figures (:func:`muf.sim.live.figures`) of the last hour
    and day before the newest sounding, not before the clock, so a circuit
    whose files come late reads the same; and the latest events as lines,
    each marked when the receiver's clock broke its pair. The figures leave
    those pairs out; a latest score's own figures keep it.
    """
    from muf.sim import clock, live

    held = db.one(conn, "SELECT taken_at, ledger FROM live_circuit "
                        "WHERE tx = ? AND rx = ?", (tx, rx))
    if held is None:
        return None
    ledger, now = json.loads(held["ledger"]), _epoch(held["taken_at"])
    rows = db.rows(conn, "SELECT sounding_id, source_id, body FROM live_event "
                         "WHERE tx = ? AND rx = ? AND kind = 'scored' AND epoch >= ? "
                         "ORDER BY epoch DESC, id DESC", (tx, rx, now - DAY_S))
    bodies = [json.loads(r["body"]) for r in rows]
    latest: dict[float, dict] = {}
    for r, e in zip(rows, bodies):
        lead = float(e["lead_s"])
        latest.setdefault(lead, {"event": e, "sounding_id": r["sounding_id"],
                                 "source_id": r["source_id"],
                                 "figures": live.figures([e], suspect=True)[lead]})
    pending = []
    for p in ledger["pending"]:
        state = p["forecast"]
        lead = float(state["metrics"]["lead_s"])
        pending.append({"lead_s": lead, "source": Path(p["source"]).name,
                        "issued": _clock(state["epoch"] - lead),
                        "valid": _clock(state["epoch"]),
                        "issued_at": _stamp(state["epoch"] - lead),
                        "valid_at": _stamp(state["epoch"]),
                        "muf_mhz": state["metrics"].get("muf_mhz"),
                        "modes": len(state["modes"])})
    pending.sort(key=lambda p: p["lead_s"])
    recent = db.rows(conn, "SELECT body FROM live_event WHERE tx = ? AND rx = ? "
                           "ORDER BY epoch DESC, id DESC LIMIT ?", (tx, rx, RECENT))
    return {"tx": tx, "rx": rx, "taken_at": held["taken_at"],
            "interval_s": live.cadence(ledger["times"]), "pending": pending,
            "latest": latest,
            "leads": sorted({p["lead_s"] for p in pending} | set(latest)),
            "figures": {"hour": live.figures(bodies, since=now - HOUR_S),
                        "day": live.figures(bodies)},
            "recent": [{"line": live.format_event(e), "suspect": clock.suspect(e)}
                       for e in (json.loads(r["body"]) for r in recent)]}


def _prune(cache: Path) -> None:
    """The states older than :data:`KEEP_S` forgotten; they are observed again
    if a forecast still needs them."""
    if not cache.is_dir():
        return
    edge = time.time() - KEEP_S
    for f in cache.iterdir():
        if f.is_file() and f.stat().st_mtime < edge:
            f.unlink(missing_ok=True)


def start(app) -> Callable[[], None]:
    """:func:`advance` every :func:`interval` seconds on a daemon thread, until
    the function returned is called. A pass that fails is logged, and the
    next one finds the same soundings waiting."""
    stop = threading.Event()
    every = interval()
    keep_days = float(os.environ.get("LIVE_KEEP_DAYS", KEEP_DAYS))
    if every <= 0:
        return stop.set

    def loop() -> None:
        while not stop.wait(every):
            try:
                cache = folder()
                conn = db.connect()
                try:
                    taken = advance(conn, Path(app.state.archive_root), cache,
                                    keep_days=keep_days)
                finally:
                    conn.close()
                _prune(cache)
                if taken:
                    print(f"{db.utcnow()}  live forecast: took {taken} sounding(s)",
                          flush=True)
            except Exception as exc:
                print(f"{db.utcnow()}  live forecast pass failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    threading.Thread(target=loop, daemon=True, name="live-forecast").start()
    return stop.set
