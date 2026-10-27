"""The ionozond schedule as the server sees it: reported, vetted, journalled.

Phase 3 of the live console (issue #4). An ionozond station's health report
carries ``sounder_config``: the console's editable document with its ``base``,
and what the sounder reads with its ``sounder_hash``. Both halves of the
checking live in ``services.agent.ionozond``, so the server vets a change with
the same code the station applies it with, against the station's last report,
while the operator is still looking at the screen.

**Only what the sounder reads opens an epoch.** A queued change opens one when
it is acknowledged, as every parameter change does. A change made on the
laptop, by hand, is seen only as a report whose ``sounder_hash`` differs from
the open epoch's ``config_hash``, or from the previous report's where the
epoch has none, and :func:`observe` opens an epoch for it with ``changed_by``
"station". It opens none while a delivered ``set_ionozond`` or
``regenerate_ionozond`` still awaits its acknowledgement: the report can be
collected between the write and the reply, and that reply will open the epoch
itself. If the reply never comes, the next report opens it (phase 4, issue #8).

Every report's literals are filed under their hash (:func:`keep`), so an
epoch's changes are worked out against the configuration it replaces.

**A short report carries the summary alone** (``ionozond.SUMMARY``): the agent
sends the block whole only when the summary changes, and hourly otherwise. The
server keeps the last whole block, and :func:`latest` hands it out, marked
``stale`` when the latest summary no longer matches it.

**Seeding** (phase 4, decision 7) turns one verified slot of any station's
transmitter table into a group for the schedule, switched off, which then goes
through the ordinary check and Apply. Here rather than on the page, so the
conversion the page offers is the one the tests check.

**Planned against recorded** (:func:`tally`) holds the schedule's own replay,
the one that lists the next captures, to the soundings the live view has
summarised: what the checker predicts about lost sessions, checked against
what the sounder did.
"""

from __future__ import annotations

import calendar
import collections
import json
import math
import sqlite3
import statistics
import time

from muf.stations import default_registry
from services.agent import control, ionozond, schedulecheck, siteconfig

from . import db

#: How many captures to show ahead of now.
AHEAD = 6

#: The longest span the planned-and-recorded counts cover.
TALLY_S = 86400.0

#: How far a summarised sounding may start from a planned capture and still be
#: that capture. The sounder starts each one on its scheduled second -- a
#: product's start, modulo the period, is its offset, which
#: `tools/rename_transmitter.py` relies on -- so this only absorbs rounding.
#: Two captures of one transmitter are a whole period apart.
MATCH_S = 5.0

#: How long after a planned capture ends before a missing summary counts. The
#: live feed sends one with the last column; this is room for the products and
#: a post or two.
DUE_S = 120.0

#: Unseen captures listed by their start, the newest.
UNSEEN_LISTED = 10


class Unvetted(Exception):
    """A change the server cannot judge, with the HTTP status that says why."""

    def __init__(self, status: int, text: str):
        super().__init__(text)
        self.status = status


def latest(conn: sqlite3.Connection, station: str) -> tuple:
    """The station's last health report, and the whole ``sounder_config`` in force."""
    row = db.latest_health(conn, station)
    if row is None:
        return None, {}
    found = _config(json.loads(row["document"] or "{}"))
    if not found.get("short"):
        return row, found
    kept = db.block(conn, station) or {}
    return row, dict(kept or found, stale=ionozond.summary(kept) != found)


def reported(conn: sqlite3.Connection, station: str) -> dict:
    """The station's last ``sounder_config``, or why there is none to vet against."""
    row, sounder = latest(conn, station)
    if row is None:
        raise Unvetted(409, (
            "%s has sent no health report, so there is no schedule to check this "
            "change against: start the health loop (chirp-agent.service) on the "
            "laptop" % station))
    if sounder.get("stale"):
        raise Unvetted(409, (
            "%s's last whole schedule report is out of date: wait for the next "
            "report, which goes whole within a minute of any change" % station))
    if not sounder:
        names = {m.get("name") for m in json.loads(
            row["document"] or "{}").get("metrics") or []}
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


def vet_regenerate(conn: sqlite3.Connection, station: str, params: dict) -> None:
    """Refuse a ``regenerate_ionozond`` the station's last report says cannot go."""
    if not isinstance(params, dict):
        raise Unvetted(400, "regenerate_ionozond takes an object of 'base' and "
                            "'sounder_hash'")
    try:
        ionozond.regenerable(params, reported(conn, station))
    except control.ControlError as exc:
        raise Unvetted(400, str(exc)) from None


def vet_site(conn: sqlite3.Connection, station: str, params: dict) -> None:
    """Refuse a ``set_ionozond_site`` the station's last report says cannot go."""
    if not isinstance(params, dict):
        raise Unvetted(400, "set_ionozond_site takes an object of 'base' and 'site'")
    found = reported(conn, station).get("site")
    if not isinstance(found, dict):
        raise Unvetted(409, (
            "%s reports no site settings: update the agent on the laptop" % station))
    problems = siteconfig.vetted(params, found)
    if problems:
        raise Unvetted(400, "refused: " + "; ".join(problems))


def seeded(conn: sqlite3.Connection, station: str, params: dict) -> dict:
    """A slot verified at ``params["from"]`` as a group for this station's draft.

    Raises :class:`Unvetted` for a transmitter or slot nobody verified, a low
    edge that is not a frequency, or a station with no report to size it by.
    """
    if not isinstance(params, dict):
        raise Unvetted(400, "seeding takes an object of 'from', 'code', 'slot' and "
                            "'low_edge_khz'")
    found = db.transmitter(conn, str(params.get("from") or ""),
                           str(params.get("code") or ""))
    if found is None:
        raise Unvetted(404, "%s has verified no transmitter %r" % (
            params.get("from"), params.get("code")))
    index, timings = params.get("slot"), found["timings"]
    if not isinstance(index, int) or not 0 <= index < len(timings):
        raise Unvetted(404, "%s's %s has no slot %s: it was verified with %d" % (
            found["station"], found["code"], index, len(timings)))
    try:
        edge = float(params.get("low_edge_khz"))
        if not math.isfinite(edge) or edge < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise Unvetted(400, "the band's low edge must be a frequency of 0 kHz or "
                            "more, not %r" % params.get("low_edge_khz")) from None
    return seed(found["code"], timings[index], edge, reported(conn, station))


def seed(code: str, slot: dict, edge: float, config: dict) -> dict:
    """One verified ``{chirp-rate, rep, chirpt}`` slot as a group, switched off.

    The phase 4 design, section 6. A verified ``chirpt`` is the second the
    sweep is at 0 Hz, chirpsounder2's convention; ionozond's is the second it
    reaches the band's low edge, ``cf`` less half the sample rate, so the edge's
    sweep time is added within the cycle. Confirmed on 2026-10-17 against the
    hand-made Cyprus groups: NIC1, verified at 235 s, is chirpt 10 at a 7.5 MHz
    edge and 100 kHz/s. The sounding spans the whole receive band, shortened to
    whole seconds when it and its products would outlast the unit's
    ``TimeoutStopSec``.
    """
    rate, rep = float(slot["chirp-rate"]) / 1e3, float(slot["rep"])
    sample = float(config["settings"]["sample_rate_khz"])
    full = sample / rate
    dur = _fitted(full, config.get("timeout_stop_s"))
    site = default_registry().station(code)
    return {"name": code, "located": site is not None, "capped": dur < full,
            "full_dur": _text(full),
            "group": {"active": "false", "cf": _text(edge + sample / 2),
                      "rate": _text(rate), "rep": _text(rep),
                      "chirpt": _text(round(float(slot["chirpt"]) + edge / rate, 6)
                                      % rep),
                      "dur": _text(dur),
                      "lat": _text(site.latitude) if site else "",
                      "lon": _text(site.longitude) if site else ""}}


def keep(conn: sqlite3.Connection, station: str, document: dict) -> None:
    """Keep a whole block, and file what it says the sounder reads under its hash."""
    found = _config(document)
    if found and not found.get("short"):
        db.keep_block(conn, station, found)
    if found.get("sounder_hash") and isinstance(found.get("sounder"), dict):
        db.keep_literals(conn, station, found["sounder_hash"], found["sounder"])


def observe(conn: sqlite3.Connection, station: str, previous: dict | None,
            document: dict) -> int | None:
    """Open an epoch for a change the sounder now reads that no command made."""
    if previous is None:
        return None
    before = _config(json.loads(previous["document"] or "{}"))
    after = _config(document)
    old = ((db.open_epoch(conn, station) or {}).get("config_hash")
           or before.get("sounder_hash"))
    new = after.get("sounder_hash")
    if not old or not new or old == new:
        return None
    if db.one(conn, "SELECT id FROM command WHERE station = ? AND name IN (?, ?)"
                    " AND acked_at IS NULL AND delivered_at >= ?",
              (station, ionozond.NAME, ionozond.REGENERATE, previous["received_at"])):
        return None
    return db.record_epoch(
        conn, station, ionozond.diff(_read(conn, station, old, before),
                                     _read(conn, station, new, after)),
        changed_by="station", config_hash=new,
        note="chirp_config.py changed on the station, not by a queued command")


def tally(conn: sqlite3.Connection, station: str, config: dict,
          epoch: dict | None, now: float) -> dict | None:
    """The captures the schedule plans, beside the soundings summarised.

    Per transmitter. ``planned`` is :func:`ionozond.planned`, the schedule
    replayed as the sounder picks its captures, counting those that have
    ended. ``recorded`` is a planned capture with a summary, live or archived,
    within ``MATCH_S`` of its start; ``unseen`` one that ended ``DUE_S`` ago
    with none -- not recorded, or not yet backfilled from the station's
    archive. ``unplanned`` is a summary no planned capture accounts for, and
    ``offset_s`` the median of recorded less planned start.

    Over the last ``TALLY_S``, or since the open epoch began if that is later:
    the schedule in force did not make the soundings before it.
    ``comparable`` is false when the schedule reported is not the one the
    sounder runs -- drifted, written but not yet taken up, or a stale block.
    None when the report holds no schedule.
    """
    document = config.get("document")
    if not isinstance(document, dict) or not isinstance(document.get("schedule"), dict):
        return None
    since = max(now - TALLY_S, _seconds_of((epoch or {}).get("valid_from")) or 0.0)
    plan = collections.defaultdict(list)
    for capture in ionozond.planned(document, since, now):
        plan[capture["tx"]].append(capture)
    seen = db.summary_starts(conn, station, since - MATCH_S, now)
    return {"since": since, "until": now,
            "comparable": not any(config.get(k) for k in ("drift", "pending", "stale")),
            "rows": [_counted(tx, plan.get(tx, []), seen.get(tx, []), since, now)
                     for tx in sorted(set(plan) | set(seen))]}


def _counted(tx: str, planned: list, recorded: list, since: float,
             now: float) -> dict:
    """One transmitter's planned captures matched to its summaries, in order."""
    taken, offsets, unseen, i = set(), [], [], 0
    for capture in planned:
        start = capture["start"]
        while i < len(recorded) and recorded[i] < start - MATCH_S:
            i += 1
        if i < len(recorded) and recorded[i] <= start + MATCH_S:
            taken.add(i)
            offsets.append(recorded[i] - start)
            i += 1
        elif start + capture["dur"] + DUE_S <= now:
            unseen.append(start)
    return {"tx": tx, "planned": len(offsets) + len(unseen), "recorded": len(offsets),
            "unseen": len(unseen), "unseen_at": unseen[-UNSEEN_LISTED:],
            "unplanned": sum(1 for k, t0 in enumerate(recorded)
                             if k not in taken and t0 >= since),
            "offset_s": round(statistics.median(offsets), 1) if offsets else None}


def _seconds_of(text) -> float | None:
    """A database instant, ``db.TIME_FORMAT`` or with a space, as unix seconds."""
    try:
        return float(calendar.timegm(time.strptime(
            str(text).replace(" ", "T")[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None


def _fitted(full: float, limit: float | None) -> float:
    """The longest sounding up to ``full`` that, with its products, fits ``limit``.

    ``full`` when nothing whole fits: the checker then says why, once the group
    is switched on.
    """
    if limit is None or full + schedulecheck.processing_estimate_s(full) <= limit:
        return full
    return next((float(d) for d in range(math.floor(full), 0, -1)
                 if d + schedulecheck.processing_estimate_s(d) <= limit), full)


def _text(value: float) -> str:
    """A number as ``schedule.ini`` holds it, with no trailing zeros."""
    return ("%.6f" % value).rstrip("0").rstrip(".")


def _config(document: dict) -> dict:
    found = document.get("sounder_config")
    return found if isinstance(found, dict) else {}


def _read(conn: sqlite3.Connection, station: str, digest: str,
          config: dict) -> dict:
    kept = db.literals(conn, station, digest)
    if kept is not None:
        return kept
    return _flat(config) if config.get("sounder_hash") == digest else {}


def _flat(config: dict) -> dict:
    found = config.get("sounder")
    return found if isinstance(found, dict) else {}
