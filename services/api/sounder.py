"""The ionozond schedule as the server sees it: reported, vetted, journalled.

Phase 3 of the live console (issue #4). An ionozond station's health report
carries ``sounder_config``: the console's editable document with its ``base``,
and what the sounder reads with its ``sounder_hash``. Both halves of the
checking live in ``services.agent.ionozond``, so the server vets a change with
the same code the station applies it with, against the station's last report,
while the operator is still looking at the screen.

**Only what the sounder reads opens an epoch.** A queued change opens one when
it is acknowledged, as every parameter change does. A change made on the
laptop, in the Qt console, is seen only as a new ``sounder_hash`` between two
consecutive reports, and :func:`observe` opens an epoch for it with
``changed_by`` "station". It opens none for a hash an acknowledgement already
recorded, and none while a delivered ``set_ionozond`` still awaits its
acknowledgement: the report can be collected between the write and the reply,
and that reply will open the epoch itself.
"""

from __future__ import annotations

import json
import sqlite3

from services.agent import control, ionozond, schedulecheck

from . import db

#: How many captures to show ahead of now.
AHEAD = 6


class Unvetted(Exception):
    """A change the server cannot judge, with the HTTP status that says why."""

    def __init__(self, status: int, text: str):
        super().__init__(text)
        self.status = status


def reported(conn: sqlite3.Connection, station: str) -> dict:
    """The station's last ``sounder_config``, or why there is none to vet against."""
    row = db.latest_health(conn, station)
    if row is None:
        raise Unvetted(409, (
            "%s has sent no health report, so there is no schedule to check this "
            "change against: start the health loop (chirp-agent.service) on the "
            "laptop" % station))
    document = json.loads(row["document"] or "{}")
    sounder = _config(document)
    if not sounder:
        names = {m.get("name") for m in document.get("metrics") or []}
        raise Unvetted(409, (
            "%s runs ionozond, but its agent is too old to report the schedule: "
            "update the agent on the laptop" % station
            if "newest_lfp_age_s" in names else
            "%s does not run ionozond, so it has no ionozond schedule" % station))
    if not sounder.get("base"):
        raise Unvetted(409, "%s could not read its schedule: %s" % (
            station, "; ".join(sounder.get("problems") or ["no reason given"])))
    return sounder


def checked(conn: sqlite3.Connection, station: str, params: dict) -> dict:
    """A ``set_ionozond`` judged against the station's last report.

    Raises :class:`Unvetted` when the change cannot be judged or is refused
    outright; a schedule the checker finds wrong comes back with ``ok`` false.
    """
    if not isinstance(params, dict):
        raise Unvetted(400, "set_ionozond takes an object of 'base', 'schedule' "
                            "and 'general'")
    sounder = reported(conn, station)
    try:
        wanted, found = ionozond.vetted(params, sounder)
    except control.ControlError as exc:
        raise Unvetted(400, str(exc)) from None
    return {"ok": all(i.level != schedulecheck.ERROR for i in found),
            "issues": [i.to_dict() for i in found],
            "edits": ionozond.edits(sounder["document"], wanted),
            "wanted": wanted}


def observe(conn: sqlite3.Connection, station: str, previous: dict | None,
            document: dict) -> int | None:
    """Open an epoch for a change the sounder now reads that no command made."""
    if previous is None:
        return None
    before = _config(json.loads(previous["document"] or "{}"))
    after = _config(document)
    old, new = before.get("sounder_hash"), after.get("sounder_hash")
    if not old or not new or old == new:
        return None
    epoch = db.open_epoch(conn, station)
    if epoch is not None and epoch.get("config_hash") == new:
        return None
    if db.one(conn, "SELECT id FROM command WHERE station = ? AND name = ?"
                    " AND acked_at IS NULL AND delivered_at >= ?",
              (station, ionozond.NAME, previous["received_at"])):
        return None
    return db.record_epoch(
        conn, station, ionozond.diff(_flat(before), _flat(after)),
        changed_by="station", config_hash=new,
        note="chirp_config.py changed on the station, not by a queued command")


def _config(document: dict) -> dict:
    found = document.get("sounder_config")
    return found if isinstance(found, dict) else {}


def _flat(config: dict) -> dict:
    found = config.get("sounder")
    return found if isinstance(found, dict) else {}
