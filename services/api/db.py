"""SQLite access for the api service.

Stdlib ``sqlite3`` and hand-written SQL rather than an ORM. The schema is ten
tables of flat columns (``schema.sql``); an ORM would add a dependency, a
migration tool and a query language to learn, in exchange for nothing this
service needs.

**Every connection sets ``foreign_keys = ON``.** SQLite defaults it off, per
connection, silently -- so ``ON DELETE CASCADE`` on ``health_metric`` does
nothing unless each connection asks. A rig that accumulates orphaned metric
rows would look fine until someone counted them.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = Path(__file__).with_name("schema.sql")

#: Where the database lives. Overridden by ``API_DB`` so the container can put
#: it on a mounted volume without the code knowing.
DEFAULT_DB = Path(os.environ.get("API_DB", "data/ionograms.sqlite3"))

#: Root the ``sounding.path`` column is relative to. Stored relative because
#: the same database is read from the host and from inside a container, which
#: mount the archive at different paths.
ARCHIVE_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "."))


def time_bound(value: str | None, *, end: bool = False) -> str | None:
    """Normalize a ``from``/``to`` bound to the spelling ``datetime`` is stored in.

    ``sounding.datetime`` holds ``2026-08-09 00:00:00.009633`` -- a **space**
    separator -- and SQLite compares these as text, character by character.
    Two consequences, and both answer the wrong question in silence rather
    than raising:

    * **An ISO bound excludes the day it names.** Space is 0x20 and ``T`` is
      0x54, so ``'2026-08-09 00:00:00' >= '2026-08-09T00:00:00'`` is false.
      ``from=2026-08-09T00:00:00`` returned nothing at all.
    * **A bare date truncates as an upper bound.** ``'2026-08-09'`` is a prefix
      of every timestamp on that day, and a prefix sorts first, so
      ``to=2026-08-09`` dropped all of the 9th. As a *lower* bound the same
      property is what makes it work, which is why this went unnoticed.

    The same trap one level down: **a whole-second upper bound excludes its own
    second**, because ``'…23:59:59.999999'`` is longer than ``'…23:59:59'`` with
    the same prefix and so compares greater. Timestamps here carry
    microseconds, so ``to=2026-08-09T23:59:59`` dropped the last second of the
    day -- and every sounding in it.

    So ``T`` becomes a space; a date with no time becomes the first or last
    instant of that day; and a whole-second upper bound is extended to cover
    the microseconds inside it. A partial time (``2026-08-09 10``) is passed
    through as written -- padding it would mean guessing which field was
    meant, and the guess is wrong often enough to be worse than the literal
    reading.
    """
    if not value:
        return None
    text = value.strip().replace("T", " ")
    if len(text) == 10:                       # YYYY-MM-DD, no time
        # Microseconds are the finest thing stored, so this is the last
        # instant of the day that any row can carry.
        return f"{text} 23:59:59.999999" if end else f"{text} 00:00:00"
    if end and len(text) == 19 and "." not in text:
        return f"{text}.999999"               # YYYY-MM-DD HH:MM:SS
    return text


#: The spelling of ``received_at`` and friends. Named so that anything
#: comparing against those columns -- a retention cutoff, say -- formats its
#: own timestamp the identical way; these are compared as text.
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utcnow() -> str:
    """ISO-8601 UTC, second resolution, with the ``Z``.

    One spelling of "now" for the whole service. Mixing naive local timestamps
    into a table that also holds station-reported UTC is the kind of thing that
    reads correctly for months and then produces a negative age.
    """
    return datetime.now(timezone.utc).strftime(TIME_FORMAT)


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create anything missing. Safe to call on every start."""
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()
    return conn


@contextmanager
def session(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = init(connect(path))
    try:
        yield conn
    finally:
        conn.close()


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

def store_health(conn: sqlite3.Connection, station: str, document: dict) -> int:
    """Record one pushed health document and its metrics.

    The raw document is kept verbatim alongside the exploded metrics. The
    columns are for querying; the blob is so a document written by a newer
    agent than this server is never silently truncated to the fields this
    version happens to know about.
    """
    cur = conn.execute(
        "INSERT INTO health_report"
        " (station, received_at, reported_at, healthy, agent_version, document)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (station, utcnow(), document.get("timestamp"),
         _tri(document.get("healthy")), document.get("agent_version"),
         json.dumps(document, sort_keys=True)),
    )
    report_id = int(cur.lastrowid)

    for metric in document.get("metrics", []) or []:
        conn.execute(
            "INSERT OR REPLACE INTO health_metric"
            " (report_id, name, value, ok, detail) VALUES (?, ?, ?, ?, ?)",
            (report_id, str(metric.get("name")),
             None if metric.get("value") is None else str(metric.get("value")),
             _tri(metric.get("ok")), metric.get("detail") or ""),
        )
    conn.commit()
    return report_id


def _tri(value: Any) -> int | None:
    """Bool to integer, preserving None.

    ``int(None)`` raises and ``bool(None)`` is False; the second is the
    dangerous one, because it turns "could not measure" into "failing".
    """
    return None if value is None else int(bool(value))


def latest_health(conn: sqlite3.Connection, station: str) -> dict | None:
    return one(conn,
               "SELECT * FROM health_report WHERE station = ?"
               " ORDER BY received_at DESC, id DESC LIMIT 1", (station,))


def metrics_for(conn: sqlite3.Connection, report_id: int) -> list[dict]:
    return rows(conn, "SELECT name, value, ok, detail FROM health_metric"
                      " WHERE report_id = ? ORDER BY name", (report_id,))


def stations(conn: sqlite3.Connection) -> list[str]:
    """Every station that has ever reported, plus any that has a command."""
    seen = {r["station"] for r in rows(conn, "SELECT DISTINCT station FROM health_report")}
    seen |= {r["station"] for r in rows(conn, "SELECT DISTINCT station FROM command")}
    return sorted(seen)


def forget_station(conn: sqlite3.Connection, station: str) -> dict:
    """Drop a station's health reports, commands and previews. Returns what went.

    The two tables :func:`stations` reads, plus the pictures that hang off the
    panel they build, because the point is to
    remove the console panel of a receiver that has been renamed or retired --
    one that will never push again, and whose permanently STALE panel trains
    the reader to ignore a colour that means something on the others.

    **Not the identifications, not the configuration epochs.** A transmitter
    row is a human judgement with its evidence attached, and a config epoch is
    what ``sounding.config_epoch_id`` points at -- deleting either to tidy a
    panel would destroy the provenance of products already on disk. They
    survive under the old station name and can be moved to the new one; the
    counts are returned so a caller can say what stayed behind.
    """
    kept = {
        "transmitters": len(transmitters(conn, station)),
        "config_epochs": (one(conn, "SELECT COUNT(*) AS n FROM config_epoch"
                                    " WHERE station = ?", (station,))
                          or {"n": 0})["n"],
    }
    # `health_metric` cascades from `health_report`, but only while
    # `PRAGMA foreign_keys` is on -- it is per connection, and a maintenance
    # script that forgot it would leave the metrics orphaned and invisible.
    conn.execute("DELETE FROM health_metric WHERE report_id IN"
                 " (SELECT id FROM health_report WHERE station = ?)", (station,))
    reports = conn.execute("DELETE FROM health_report WHERE station = ?",
                           (station,)).rowcount
    commands = conn.execute("DELETE FROM command WHERE station = ?",
                            (station,)).rowcount
    previews = conn.execute("DELETE FROM station_preview WHERE station = ?",
                            (station,)).rowcount
    conn.commit()
    return {"health_reports": reports, "commands": commands,
            "previews": previews, "kept": kept}


# --------------------------------------------------------------------------
# Previews (services/agent/preview.py)
# --------------------------------------------------------------------------

#: How long a circuit's thumbnail outlives its last push. Long enough that a
#: transmitter sounded once a day is not swept between soundings; short enough
#: that a renamed or retired one does not leave a picture up forever, which is
#: the same failure `forget_station` exists to fix one level up.
PREVIEW_RETENTION_DAYS = 7


def save_preview(conn: sqlite3.Connection, station: str, tx: str,
                 image: bytes, **meta: Any) -> None:
    """Store, or replace, one station's newest picture of one transmitter.

    An upsert on ``(station, tx)``: this table is a live view and the newest
    picture is the only one worth keeping. The retention sweep runs here rather
    than on a timer because a push is the only moment this table is written at
    all -- a separate scheduler for a handful of rows would be more machinery
    than the problem.
    """
    conn.execute(
        "INSERT INTO station_preview"
        " (station, tx, t0, received_at, width, height, freq_lo_hz,"
        "  freq_hi_hz, range_lo_m, range_hi_m, cropped, image)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(station, tx) DO UPDATE SET"
        "   t0 = excluded.t0, received_at = excluded.received_at,"
        "   width = excluded.width, height = excluded.height,"
        "   freq_lo_hz = excluded.freq_lo_hz, freq_hi_hz = excluded.freq_hi_hz,"
        "   range_lo_m = excluded.range_lo_m, range_hi_m = excluded.range_hi_m,"
        "   cropped = excluded.cropped, image = excluded.image",
        (station, tx, meta.get("t0"), utcnow(),
         meta.get("width"), meta.get("height"),
         meta.get("freq_lo_hz"), meta.get("freq_hi_hz"),
         meta.get("range_lo_m"), meta.get("range_hi_m"),
         _tri(meta.get("cropped")), sqlite3.Binary(image)),
    )
    conn.execute(
        "DELETE FROM station_preview WHERE received_at < ?",
        ((datetime.now(timezone.utc)
          - timedelta(days=PREVIEW_RETENTION_DAYS)).strftime(TIME_FORMAT),))
    conn.commit()


def previews(conn: sqlite3.Connection, station: str) -> list[dict]:
    """Every circuit's preview metadata for one station -- **without the image**.

    The console renders one panel per station on every 15 s refresh; carrying
    the PNGs through that query would put a few hundred kilobytes of image into
    the HTML render path to decide where to put an ``<img>`` tag.
    """
    return rows(conn,
                "SELECT station, tx, t0, received_at, width, height,"
                " freq_lo_hz, freq_hi_hz, range_lo_m, range_hi_m, cropped"
                " FROM station_preview WHERE station = ?"
                " ORDER BY t0 DESC, tx", (station,))


def preview_image(conn: sqlite3.Connection, station: str,
                  tx: str) -> bytes | None:
    row = one(conn, "SELECT image FROM station_preview"
                    " WHERE station = ? AND tx = ?", (station, tx))
    return None if row is None else bytes(row["image"])


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def enqueue(conn: sqlite3.Connection, station: str, name: str,
            params: dict | None = None, issued_by: str = "web") -> str:
    import uuid

    command_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO command (id, station, name, params, issued_at, issued_by)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (command_id, station, name, json.dumps(params or {}), utcnow(), issued_by),
    )
    conn.commit()
    return command_id


def take_pending(conn: sqlite3.Connection, station: str,
                 limit: int = 10) -> list[dict]:
    """Undelivered commands for one station, marked delivered as they go out.

    Marking on delivery rather than on acknowledgement is what stops a command
    that fails from being handed out forever. The agent acknowledges failures
    explicitly, so a command that vanishes without an ack stays visible in the
    UI as delivered-but-unacked, which is the state a human should look at.
    """
    pending = rows(conn,
                   "SELECT * FROM command WHERE station = ? AND delivered_at IS NULL"
                   " ORDER BY issued_at LIMIT ?", (station, limit))
    if pending:
        conn.executemany("UPDATE command SET delivered_at = ? WHERE id = ?",
                         [(utcnow(), c["id"]) for c in pending])
        conn.commit()
    return pending


def acknowledge(conn: sqlite3.Connection, command_id: str, results: list) -> bool:
    ok = all(bool(r.get("ok")) for r in results) if results else False
    cur = conn.execute(
        "UPDATE command SET acked_at = ?, ok = ?, results = ? WHERE id = ?",
        (utcnow(), int(ok), json.dumps(results), command_id))
    conn.commit()
    known = cur.rowcount > 0
    if known:
        _journal_epoch(conn, command_id, results)
    return known


def _journal_epoch(conn: sqlite3.Connection, command_id: str,
                   results: list) -> int | None:
    """Open a `config_epoch` if this acknowledgement says the ini was rewritten.

    Architecture sec. 2.5: "Control endpoints touching acquisition parameters
    write a `config_epoch` row so every sounding can be attributed to the
    configuration that produced it." The agent hands back a ``journal`` on the
    one result that did the write, and that result is the trigger -- not the
    command's overall ``ok``.

    The distinction is load-bearing. ``apply_and_restart`` is stop, write,
    start; if the *start* fails the command is not ok, but the file on the
    station has already changed and anything it records from now on was
    recorded under the new configuration. Keying the epoch off the overall
    result would leave those soundings attributed to a configuration that no
    longer exists.
    """
    for result in results or []:
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        journal = result.get("journal")
        if not isinstance(journal, dict) or not journal.get("changes"):
            continue
        row = one(conn, "SELECT station, issued_by FROM command WHERE id = ?",
                  (command_id,))
        if row is None:                                       # pragma: no cover
            return None
        return record_epoch(
            conn, row["station"], journal["changes"],
            changed_by=row["issued_by"],
            note=f"command {command_id}: {result.get('detail', '')}".strip())
    return None


def recent_commands(conn: sqlite3.Connection, station: str,
                    limit: int = 20) -> list[dict]:
    return rows(conn, "SELECT * FROM command WHERE station = ?"
                      " ORDER BY issued_at DESC LIMIT ?", (station, limit))


# --------------------------------------------------------------------------
# Verified transmitters
# --------------------------------------------------------------------------

def next_sounder_id(conn: sqlite3.Connection, station: str) -> int:
    """The next free chirpsounder2 ``id`` for this receiver, counting from 1.

    Reuses nothing. An id freed by deleting a transmitter stays free, because
    products already on disk carry it in their file names and handing it to a
    different transmitter would make two sites share one number in one
    archive.
    """
    row = one(conn, "SELECT MAX(sounder_id) AS top FROM transmitter"
                    " WHERE station = ?", (station,))
    return int((row or {}).get("top") or 0) + 1


def save_transmitter(conn: sqlite3.Connection, station: str, code: str,
                     timings: list, *, name: str | None = None,
                     sounder_id: int | None = None,
                     evidence: Any = None, verified_by: str = "web",
                     note: str | None = None) -> dict:
    """Record, or re-record, one identified transmitter for one receiver.

    An upsert on ``(station, code)``. Re-identifying is the normal case, not an
    error: the operator watches a census for a few days and narrows the slots,
    and the second save is the better one. What is *not* overwritten silently
    is the provenance -- ``verified_at`` and ``verified_by`` move with the
    timings, so the record always says when the current numbers were chosen.

    The entries are normalised on the way in, so what is stored is exactly what
    the ini will receive: ``transmit_name`` is the code and ``id`` is this
    transmitter's ``sounder_id``, whatever the caller sent. Those two fields
    are read by ``calc_ionograms.py`` with a bare subscript, and letting a
    caller set them independently is how they come to disagree with the record
    they are supposed to name.
    """
    existing = transmitter(conn, station, code)
    if sounder_id is None:
        sounder_id = (existing["sounder_id"] if existing
                      else next_sounder_id(conn, station))
    entries = [dict(entry, id=int(sounder_id), transmit_name=code)
               for entry in timings]

    conn.execute(
        "INSERT INTO transmitter"
        " (station, code, name, sounder_id, timings, evidence, verified_at,"
        "  verified_by, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(station, code) DO UPDATE SET"
        "   name = excluded.name, timings = excluded.timings,"
        "   evidence = excluded.evidence, verified_at = excluded.verified_at,"
        "   verified_by = excluded.verified_by, note = excluded.note",
        (station, code, name, int(sounder_id), json.dumps(entries),
         None if evidence is None else json.dumps(evidence),
         utcnow(), verified_by, note),
    )
    conn.commit()
    found = transmitter(conn, station, code)
    if found is None:                                         # pragma: no cover
        raise RuntimeError(f"{station}/{code} vanished between write and read")
    return found


def _transmitter(row: dict) -> dict:
    row = dict(row)
    row["timings"] = json.loads(row["timings"] or "[]")
    row["evidence"] = json.loads(row["evidence"]) if row.get("evidence") else None
    return row


def transmitters(conn: sqlite3.Connection, station: str) -> list[dict]:
    return [_transmitter(r) for r in
            rows(conn, "SELECT * FROM transmitter WHERE station = ?"
                       " ORDER BY code", (station,))]


def transmitter(conn: sqlite3.Connection, station: str,
                code: str) -> dict | None:
    row = one(conn, "SELECT * FROM transmitter WHERE station = ? AND code = ?",
              (station, code))
    return _transmitter(row) if row else None


def delete_transmitter(conn: sqlite3.Connection, station: str,
                       code: str) -> bool:
    cur = conn.execute("DELETE FROM transmitter WHERE station = ? AND code = ?",
                       (station, code))
    conn.commit()
    return cur.rowcount > 0


# --------------------------------------------------------------------------
# Configuration epochs
# --------------------------------------------------------------------------

def open_epoch(conn: sqlite3.Connection, station: str) -> dict | None:
    """The configuration currently in force, or None if none was ever recorded.

    Open-ended by definition: an epoch ends when the next one starts, so
    ``valid_to IS NULL`` is "what the station is running now" rather than a
    row waiting to be completed.
    """
    row = one(conn, "SELECT * FROM config_epoch WHERE station = ?"
                    " AND valid_to IS NULL ORDER BY valid_from DESC, id DESC"
                    " LIMIT 1", (station,))
    if row is None:
        return None
    row = dict(row)
    row["changes"] = json.loads(row["changes"] or "{}")
    return row


def record_epoch(conn: sqlite3.Connection, station: str, changes: dict,
                 changed_by: str | None = None,
                 note: str | None = None) -> int:
    """Close the open epoch and start a new one.

    Called when a parameter change is *acknowledged*, not when it is queued.
    A command sitting in the queue has changed nothing, and an epoch opened at
    enqueue time would attribute every sounding recorded while the station was
    unreachable to a configuration it was not running.

    Closing first is what keeps the table readable as intervals. Two open
    epochs for one station is not a worse row, it is a station with two
    current configurations, and every attribution after it is a guess.
    """
    now = utcnow()
    conn.execute("UPDATE config_epoch SET valid_to = ?"
                 " WHERE station = ? AND valid_to IS NULL", (now, station))
    cur = conn.execute(
        "INSERT INTO config_epoch (station, valid_from, valid_to, changes,"
        " changed_by, note) VALUES (?, ?, NULL, ?, ?, ?)",
        (station, now, json.dumps(changes, sort_keys=True), changed_by, note))
    conn.commit()
    return int(cur.lastrowid)


def recent_epochs(conn: sqlite3.Connection, station: str,
                  limit: int = 10) -> list[dict]:
    out = []
    for row in rows(conn, "SELECT * FROM config_epoch WHERE station = ?"
                          " ORDER BY valid_from DESC, id DESC LIMIT ?",
                    (station, limit)):
        row["changes"] = json.loads(row["changes"] or "{}")
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Archives -- the folders that are meant to be indexed
# --------------------------------------------------------------------------

def archives(conn: sqlite3.Connection, *, enabled_only: bool = False) -> list[dict]:
    """Every registered archive, with how many soundings came from it.

    The count is derived from `sounding.path` rather than stored, because a
    stored count is a second source of truth that goes wrong quietly: ingest
    writes `sounding` and would have to remember to touch a counter here too,
    and the day it forgets, the page reports a number no query can reproduce.
    """
    where = " WHERE enabled = 1" if enabled_only else ""
    rows_ = rows(conn, f"SELECT * FROM archive{where} ORDER BY name")
    for row in rows_:
        # `path` is stored relative to ARCHIVE_ROOT, so an archive's soundings
        # are the ones whose path sits under its relpath. LIKE with a trailing
        # separator, not a bare prefix: `2026.02.04` must not also match
        # `2026.02.045`.
        prefix = row["relpath"].rstrip("/") + "/"
        row["soundings"] = (one(
            conn, "SELECT COUNT(*) AS n FROM sounding WHERE path LIKE ?",
            (prefix + "%",)) or {"n": 0})["n"]
    return rows_


def archive(conn: sqlite3.Connection, archive_id: int) -> dict | None:
    return one(conn, "SELECT * FROM archive WHERE id = ?", (archive_id,))


def add_archive(conn: sqlite3.Connection, *, name: str, relpath: str,
                methods: str, format: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO archive (name, relpath, format, methods, enabled,"
        " added_at) VALUES (?, ?, ?, ?, 1, ?)",
        (name, relpath, format, methods, utcnow()))
    conn.commit()
    return int(cur.lastrowid)


def set_archive_enabled(conn: sqlite3.Connection, archive_id: int,
                        enabled: bool) -> bool:
    cur = conn.execute("UPDATE archive SET enabled = ? WHERE id = ?",
                       (1 if enabled else 0, archive_id))
    conn.commit()
    return cur.rowcount > 0


def set_archive_methods(conn: sqlite3.Connection, archive_id: int,
                        methods: str) -> bool:
    cur = conn.execute("UPDATE archive SET methods = ? WHERE id = ?",
                       (methods, archive_id))
    conn.commit()
    return cur.rowcount > 0


def record_scan(conn: sqlite3.Connection, archive_id: int, *,
                result: str, ok: bool) -> None:
    conn.execute(
        "UPDATE archive SET last_scan_at = ?, last_scan_result = ?,"
        " last_scan_ok = ? WHERE id = ?",
        (utcnow(), result, 1 if ok else 0, archive_id))
    conn.commit()


def remove_archive(conn: sqlite3.Connection, archive_id: int) -> bool:
    """Unregister a folder. **Never touches its soundings.**

    Removing a row here says "stop indexing this", not "forget what was
    measured". The soundings and their extractions are the record; the archive
    row is only the instruction to keep looking. Re-registering the same path
    picks the existing soundings straight back up, because `watch.already_done`
    keys on the file name and finds them still there.
    """
    cur = conn.execute("DELETE FROM archive WHERE id = ?", (archive_id,))
    conn.commit()
    return cur.rowcount > 0
