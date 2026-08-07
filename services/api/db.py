"""SQLite access for the api service.

Stdlib ``sqlite3`` and hand-written SQL rather than an ORM. The schema is nine
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
from datetime import datetime, timezone
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


def utcnow() -> str:
    """ISO-8601 UTC, second resolution, with the ``Z``.

    One spelling of "now" for the whole service. Mixing naive local timestamps
    into a table that also holds station-reported UTC is the kind of thing that
    reads correctly for months and then produces a negative age.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return cur.rowcount > 0


def recent_commands(conn: sqlite3.Connection, station: str,
                    limit: int = 20) -> list[dict]:
    return rows(conn, "SELECT * FROM command WHERE station = ?"
                      " ORDER BY issued_at DESC LIMIT ?", (station, limit))
