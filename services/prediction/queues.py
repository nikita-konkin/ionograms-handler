"""The two work queues, and the only module that writes their tables.

``model_upload`` and ``train_job`` are each written from two sides -- the api
enqueues, a worker settles -- and the two sides live in different containers.
Putting the SQL in one place is what keeps the state machine honest: there is
exactly one function that can move an upload out of ``pending`` and exactly one
that can claim a job, so "how does a row reach this state" has one answer.

**Nothing here loads an artifact or fits anything.** This module is imported by
the api, which must be able to enqueue without ever gaining the ability to run
what it enqueued. It depends on ``services.api.db`` and the standard library,
and that is deliberate rather than incidental.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..api import db

#: What the operator may set on an upload. The rest of the row -- state,
#: detail, model_id, the timestamps -- belongs to the state machine.
UPLOAD_FIELDS = ("filename", "sha256", "bytes", "name", "param", "tx", "rx",
                 "origin", "target_src", "period", "note")

PENDING = "pending"
REGISTERED = "registered"
REFUSED = "refused"

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

#: Upload states from which nothing further happens on its own.
SETTLED_UPLOAD = (REGISTERED, REFUSED)


class QueueError(RuntimeError):
    """A queue operation was refused."""


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------

def add_upload(conn: sqlite3.Connection, *, by: str | None = None,
               **fields) -> dict:
    """Record an artifact that has been quarantined and awaits registration."""
    unknown = set(fields) - set(UPLOAD_FIELDS)
    if unknown:
        raise QueueError(f"not an upload field: {sorted(unknown)}")
    for required in ("filename", "sha256", "bytes", "param"):
        if fields.get(required) in (None, ""):
            raise QueueError(f"an upload needs {required}")

    columns = list(fields) + ["state", "uploaded_at", "uploaded_by"]
    values = list(fields.values()) + [PENDING, db.utcnow(), by]
    cursor = conn.execute(
        f"INSERT INTO model_upload ({','.join(columns)}) "
        f"VALUES ({','.join('?' * len(columns))})", tuple(values))
    conn.commit()
    return upload(conn, int(cursor.lastrowid))


def upload(conn: sqlite3.Connection, upload_id: int) -> dict | None:
    return db.one(conn, "SELECT * FROM model_upload WHERE id = ?", (upload_id,))


def uploads(conn: sqlite3.Connection, state: str | None = None,
            limit: int = 50) -> list[dict]:
    """Newest first, because the one just added is the one being watched."""
    sql = "SELECT * FROM model_upload"
    params: tuple = ()
    if state:
        sql += " WHERE state = ?"
        params = (state,)
    sql += " ORDER BY id DESC LIMIT ?"
    return db.rows(conn, sql, params + (int(limit),))


def pending_uploads(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Oldest first: a queue is drained in the order it was filled."""
    return db.rows(
        conn,
        "SELECT * FROM model_upload WHERE state = ? ORDER BY id LIMIT ?",
        (PENDING, int(limit)))


def settle_upload(conn: sqlite3.Connection, upload_id: int, state: str,
                  detail: str | None = None,
                  model_id: int | None = None) -> dict | None:
    if state not in SETTLED_UPLOAD:
        raise QueueError(f"not a settled state: {state!r}")
    with conn:
        conn.execute(
            "UPDATE model_upload SET state = ?, detail = ?, model_id = ?, "
            "settled_at = ? WHERE id = ?",
            (state, detail, model_id, db.utcnow(), upload_id))
    return upload(conn, upload_id)


def delete_upload(conn: sqlite3.Connection, upload_id: int) -> dict | None:
    """Forget an upload. Returns the row as it was, so its blob can be reaped."""
    row = upload(conn, upload_id)
    if row is None:
        return None
    with conn:
        conn.execute("DELETE FROM model_upload WHERE id = ?", (upload_id,))
    return row


def blob_is_referenced(conn: sqlite3.Connection, digest: str,
                       ignoring: int | None = None) -> bool:
    """Whether any unsettled row still needs the quarantined bytes.

    A refusal is usually fixed by re-registering the same file with an explicit
    feature list, so a refused row keeps its blob. Only when every row for a
    digest is registered or gone are the quarantined bytes reaped -- the object
    store holds them by then.
    """
    sql = ("SELECT COUNT(*) AS n FROM model_upload "
           "WHERE sha256 = ? AND state IN (?, ?)")
    params: tuple = (digest, PENDING, REFUSED)
    if ignoring is not None:
        sql += " AND id != ?"
        params += (ignoring,)
    return bool((db.one(conn, sql, params) or {}).get("n"))


# --------------------------------------------------------------------------
# Training jobs
# --------------------------------------------------------------------------

def add_job(conn: sqlite3.Connection, *, param: str, tx: str, rx: str,
            method: str = "contour", spec: dict[str, Any] | None = None,
            by: str | None = None) -> dict:
    cursor = conn.execute(
        "INSERT INTO train_job (param, tx, rx, method, spec, state, "
        "requested_at, requested_by) VALUES (?,?,?,?,?,?,?,?)",
        (param, tx, rx, method, json.dumps(spec or {}), QUEUED,
         db.utcnow(), by))
    conn.commit()
    return job(conn, int(cursor.lastrowid))


def job(conn: sqlite3.Connection, job_id: int) -> dict | None:
    row = db.one(conn, "SELECT * FROM train_job WHERE id = ?", (job_id,))
    return _decode_job(row) if row else None


def jobs(conn: sqlite3.Connection, state: str | None = None,
         limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM train_job"
    params: tuple = ()
    if state:
        sql += " WHERE state = ?"
        params = (state,)
    sql += " ORDER BY id DESC LIMIT ?"
    return [_decode_job(r) for r in db.rows(conn, sql, params + (int(limit),))]


#: Tables `_claim` may touch. The table name is interpolated into SQL -- it is
#: a module constant here and never a caller's string, and this list is what
#: keeps it that way.
CLAIMABLE = ("train_job", "infer_job")


def _claim(conn: sqlite3.Connection, table: str) -> dict | None:
    """Take the oldest queued row in ``table`` and mark it running.

    The claim is a conditional UPDATE rather than a SELECT followed by an
    UPDATE: two workers on one database would otherwise both read the same
    queued row and both act on it. There is one worker per queue today; the
    guarantee costs a line and removes the need to remember that.

    ``RETURNING`` rather than a read-back, and that is the second half of the
    same guarantee. Re-selecting "the most recently started running row" looks
    equivalent and is not: ``db.utcnow`` has second resolution, so a row left
    ``running`` by a worker that died -- or claimed by a second worker inside
    the same second -- can tie on ``started_at`` and win the ordering. Then the
    caller runs a job it did not claim. SQLite has had ``RETURNING`` since
    3.35; the inference image carries 3.46.
    """
    if table not in CLAIMABLE:
        raise QueueError(f"not a claimable queue: {table!r}")
    with conn:
        row = conn.execute(
            f"UPDATE {table} SET state = ?, started_at = ? "
            f"WHERE id = (SELECT id FROM {table} WHERE state = ? "
            f"            ORDER BY id LIMIT 1) RETURNING *",
            (RUNNING, db.utcnow(), QUEUED)).fetchone()
    return dict(row) if row else None


def claim_job(conn: sqlite3.Connection) -> dict | None:
    """Take the oldest queued training job. Returns it, or None."""
    row = _claim(conn, "train_job")
    return _decode_job(row) if row else None


def settle_job(conn: sqlite3.Connection, job_id: int, state: str,
               detail: str | None = None,
               model_id: int | None = None) -> dict | None:
    if state not in (DONE, FAILED, CANCELLED):
        raise QueueError(f"not a settled state: {state!r}")
    with conn:
        conn.execute(
            "UPDATE train_job SET state = ?, detail = ?, model_id = ?, "
            "settled_at = ? WHERE id = ?",
            (state, detail, model_id, db.utcnow(), job_id))
    return job(conn, job_id)


def cancel_job(conn: sqlite3.Connection, job_id: int) -> dict:
    """Cancel a job that has not started. A running fit is left alone.

    Killing a fit half way would leave the worker's temporary artifact and its
    claim behind with nothing to tidy them; a job that is already running is
    therefore refused here rather than raced.
    """
    row = job(conn, job_id)
    if row is None:
        raise QueueError(f"no training job {job_id}")
    if row["state"] != QUEUED:
        raise QueueError(
            f"job {job_id} is {row['state']}, not queued, so there is nothing "
            f"to cancel. A fit that has started runs to its end.")
    return settle_job(conn, job_id, CANCELLED, "cancelled before it started")


def _decode_job(row: dict) -> dict:
    out = dict(row)
    if isinstance(out.get("spec"), str) and out["spec"]:
        try:
            out["spec"] = json.loads(out["spec"])
        except json.JSONDecodeError:
            pass
    return out


# --------------------------------------------------------------------------
# Inference passes
#
# Drained by `infer` itself, between the slices its interval is now cut into.
# Nothing else in this module needs to know that; what matters here is that a
# requested pass is a row somebody can see the state of, the same as an upload
# or a fit.
# --------------------------------------------------------------------------

def add_run(conn: sqlite3.Connection, *, param: str, tx: str, rx: str,
            method: str = "contour", model_id: int | None = None,
            by: str | None = None) -> dict:
    cursor = conn.execute(
        "INSERT INTO infer_job (param, tx, rx, method, model_id, state, "
        "requested_at, requested_by) VALUES (?,?,?,?,?,?,?,?)",
        (param, tx, rx, method, model_id, QUEUED, db.utcnow(), by))
    conn.commit()
    return run(conn, int(cursor.lastrowid))


def run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    return db.one(conn, "SELECT * FROM infer_job WHERE id = ?", (run_id,))


def runs(conn: sqlite3.Connection, state: str | None = None,
         limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM infer_job"
    params: tuple = ()
    if state:
        sql += " WHERE state = ?"
        params = (state,)
    sql += " ORDER BY id DESC LIMIT ?"
    return db.rows(conn, sql, params + (int(limit),))


def pending_run(conn: sqlite3.Connection) -> dict | None:
    """Whether anything is waiting, without claiming it.

    `infer` checks this between sleep slices and only opens a write
    transaction when the answer is yes -- the api holds the write lock for
    minutes during an archive scan, and a claim attempted every ten seconds
    against that would be ten seconds of contention for nothing.
    """
    return db.one(conn, "SELECT * FROM infer_job WHERE state = ? "
                        "ORDER BY id LIMIT 1", (QUEUED,))


def claim_run(conn: sqlite3.Connection) -> dict | None:
    return _claim(conn, "infer_job")


def settle_run(conn: sqlite3.Connection, run_id: int, state: str,
               detail: str | None = None, written: int | None = None,
               backtest: bool | None = None) -> dict | None:
    if state not in (DONE, FAILED, CANCELLED):
        raise QueueError(f"not a settled state: {state!r}")
    with conn:
        conn.execute(
            "UPDATE infer_job SET state = ?, detail = ?, written = ?, "
            "backtest = ?, settled_at = ? WHERE id = ?",
            (state, detail, written,
             None if backtest is None else int(backtest), db.utcnow(), run_id))
    return run(conn, run_id)


def cancel_run(conn: sqlite3.Connection, run_id: int) -> dict:
    """Cancel a pass that has not started. A running one is left alone."""
    row = run(conn, run_id)
    if row is None:
        raise QueueError(f"no inference run {run_id}")
    if row["state"] != QUEUED:
        raise QueueError(
            f"run {run_id} is {row['state']}, not queued, so there is nothing "
            f"to cancel. A pass that has started runs to its end.")
    return settle_run(conn, run_id, CANCELLED, "cancelled before it started")
