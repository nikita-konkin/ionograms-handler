"""Persistent, verified .lfs replacement jobs. DSP runs only in the worker.

The manifest is reviewed before deletion is enabled. A verified entry is a
recovery journal: it commits both checksums before unlinking anything. A crash
between unlink and SQLite commit can therefore finish the path update without
guessing whether an arbitrary existing HDF5 file is a valid replacement.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from . import archives, db


class ConversionError(ValueError):
    pass


def get(conn, job_id):
    return db.one(conn, "SELECT * FROM archive_conversion WHERE id = ?", (job_id,))


def listing(conn):
    return db.rows(conn, "SELECT * FROM archive_conversion ORDER BY id DESC LIMIT 50")


def folder_for(conn, archive_id, subpath="."):
    archive = db.archive(conn, archive_id)
    if archive is None:
        raise ConversionError("archive not found")
    if archive["format"] not in (None, "lfs", "echo_window"):
        raise ConversionError("choose an archive that allows LFS or echo files")
    _, base = archives.resolve(archive["relpath"], archives.roots()[0])
    sub = Path(subpath or ".")
    if sub.is_absolute() or ".." in sub.parts:
        raise ConversionError("choose a folder inside the selected archive")
    folder = base / sub
    if folder.is_symlink() or not folder.resolve().is_relative_to(base.resolve()):
        raise ConversionError("linked folders outside the archive are not allowed")
    stored, folder = archives.resolve(str(folder), archives.roots()[0])
    return stored, folder


def enqueue(conn, archive_id, subpath, by):
    folder, _ = folder_for(conn, archive_id, subpath)
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO archive_conversion"
                " (archive_id, folder, requested_by, created_at)"
                " VALUES (?,?,?,?)",
                (archive_id, folder, by, db.utcnow()),
            )
    except sqlite3.IntegrityError as exc:
        raise ConversionError("another conversion job is queued or running") from exc
    return get(conn, cursor.lastrowid)


def start(conn, job_id, confirmed, by="local"):
    job = get(conn, job_id)
    if job is None:
        raise ConversionError("job not found")
    if confirmed is not True:
        raise ConversionError(
            "confirm replacement and deletion of the listed .lfs files"
        )
    if job["state"] not in ("ready", "failed", "cancelled"):
        raise ConversionError("wait for the folder inspection to finish")
    # A cancelled survey is incomplete and must be inspected again.
    mode = (
        "survey" if job["mode"] == "survey" and job["state"] != "ready" else "convert"
    )
    try:
        with conn:
            conn.execute(
                "UPDATE archive_conversion SET state = 'queued', mode = ?,"
                " cancel_requested = 0, finished_at = NULL, detail = '',"
                " approved_by = ?, approved_at = ? WHERE id = ?",
                (
                    mode,
                    by if mode == "convert" else None,
                    db.utcnow() if mode == "convert" else None,
                    job_id,
                ),
            )
            conn.execute(
                "UPDATE archive_conversion_file SET state = 'pending', detail = ''"
                " WHERE job_id = ? AND state = 'failed'",
                (job_id,),
            )
    except sqlite3.IntegrityError as exc:
        raise ConversionError("another conversion job is queued or running") from exc
    return get(conn, job_id)


def cancel(conn, job_id):
    with conn:
        conn.execute(
            "UPDATE archive_conversion SET cancel_requested = 1,"
            " state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END"
            " WHERE id = ? AND state IN ('queued', 'running')",
            (job_id,),
        )
    return get(conn, job_id)


def signature(path):
    stat = path.stat()
    return json.dumps([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns])


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _durable_target(path):
    # HDF5 close flushes its cache; fsync persists the data and the rename
    # before the source can be removed. Storage must honor these flushes.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _self_contained(path):
    """The replacement must not depend on data outside this HDF5 file."""
    import h5py

    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"echo_window"}:
            raise ConversionError("replacement must be a standalone echo product")

        def check(_name, obj):
            if isinstance(obj, h5py.Group):
                for name in obj:
                    if not isinstance(obj.get(name, getlink=True), h5py.HardLink):
                        raise ConversionError("replacement contains linked HDF5 data")
            elif isinstance(obj, h5py.Dataset) and (obj.is_virtual or obj.external):
                raise ConversionError("replacement depends on external HDF5 storage")

        check("/", handle)
        handle.visititems(check)


def _safe(path, folder):
    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
        raise ConversionError(f"{path.name}: linked files are not converted")
    for parent in path.parents:
        if parent == folder:
            break
        if parent.is_symlink():
            raise ConversionError("linked directories are not converted")


def _stored(path):
    try:
        return path.resolve().relative_to(archives.roots()[0].resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _relink(conn, source, target):
    old = db.one(conn, "SELECT id, path FROM sounding WHERE file = ?", (source.name,))
    new = db.one(conn, "SELECT id, path FROM sounding WHERE file = ?", (target.name,))
    if old:
        stored = Path(old["path"])
        actual = stored if stored.is_absolute() else archives.roots()[0] / stored
        if actual.resolve() != source.resolve():
            raise ConversionError(
                "an indexed capture with the same name belongs to another folder"
            )
        if new and new["id"] != old["id"]:
            raise ConversionError(
                "replacement is already indexed as a separate sounding"
            )
        conn.execute(
            "UPDATE sounding SET file = ?, path = ?, format = 'echo_window'"
            " WHERE id = ?",
            (target.name, _stored(target), old["id"]),
        )


def replace_one(conn, job, item, folder):
    """Replace one reviewed source; any pre-deletion error keeps its raw file."""
    from muf import echo_window, io_lfs, loader

    source = folder / item["path"]
    target = source.with_name(source.stem + ".echo.h5")
    _safe(source, folder)
    _safe(target, folder)
    if item["state"] == "verified":
        # Includes recovery after raw unlink but before DB commit.
        if not target.is_file() or digest(target) != item["target_sha256"]:
            raise ConversionError(
                "verified replacement has changed; automatic deletion refused"
            )
        if source.exists() and (
            signature(source) != item["signature"]
            or digest(source) != item["source_sha256"]
        ):
            raise ConversionError("source changed after verification")
    else:
        if not source.is_file() or signature(source) != item["signature"]:
            raise ConversionError(
                "source changed since folder inspection; inspect it again"
            )
        if time.time() - source.stat().st_mtime < 60:
            raise ConversionError("source was modified less than a minute ago")
        source_hash = digest(source)
        if target.exists():
            if not target.is_file():
                raise ConversionError("replacement path is not a regular file")
            # write() can append a group to an existing HDF5 product. That
            # behavior is useful for --into, but is never safe for replacement.
            _self_contained(target)
            existing = echo_window.read(target, samples=False)
            with source.open("rb") as handle:
                original_header = handle.read(io_lfs.HEADER_SIZE)
            if existing.source != "lfs" or existing.lfs_header != original_header:
                raise ConversionError("existing output belongs to another capture")
        report = echo_window.write(source, verify=True)
        if (
            report.target != target
            or report.verification is None
            or not report.verification.ok
        ):
            raise ConversionError("replacement did not pass source verification")
        _self_contained(target)
        window = echo_window.read(target, samples=False)
        with source.open("rb") as handle:
            header = handle.read(io_lfs.HEADER_SIZE)
        if window.source != "lfs" or window.lfs_header != header:
            raise ConversionError(
                "replacement does not carry this capture's original header"
            )
        if window.selection.get("rule", 1) < 2 or window.trace_at_edge is not False:
            raise ConversionError(
                "echo selection is old or may crop a trace; raw capture kept"
            )
        if (
            window.selection.get("manual_km") is not None
            or window.selection.get("max_hops", 0) < 3
            or window.selection.get("margin_km", 0) < 150
        ):
            raise ConversionError(
                "existing echo window uses a narrower selection policy"
            )
        # Make sure downstream readers can open it before removing the source.
        loader.load(target)
        _safe(source, folder)
        _safe(target, folder)
        if signature(source) != item["signature"] or digest(source) != source_hash:
            raise ConversionError("source changed during conversion")
        target_hash = digest(target)
        _durable_target(target)
        with conn:
            conn.execute(
                "UPDATE archive_conversion_file SET state = 'verified',"
                " source_sha256 = ?, target_sha256 = ?, output_bytes = ?"
                " WHERE job_id = ? AND path = ?",
                (
                    source_hash,
                    target_hash,
                    target.stat().st_size,
                    job["id"],
                    item["path"],
                ),
            )
        item = db.one(
            conn,
            "SELECT * FROM archive_conversion_file WHERE job_id = ? AND path = ?",
            (job["id"], item["path"]),
        )

    # The journal above is committed independently of this transaction. If
    # unlink succeeds and this transaction rolls back, a restart can relink.
    with conn:
        _safe(source, folder)
        _safe(target, folder)
        if digest(target) != item["target_sha256"]:
            raise ConversionError("replacement changed before deletion")
        if source.exists() and signature(source) != item["signature"]:
            raise ConversionError("source changed before deletion")
        _relink(conn, source, target)
        if source.exists():
            source.unlink()
            _sync_directory(source.parent)
        conn.execute(
            "UPDATE archive_conversion_file SET state = 'replaced', detail = ''"
            " WHERE job_id = ? AND path = ?",
            (job["id"], item["path"]),
        )


def _progress(conn, job_id, detail=""):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO archive_conversion_worker VALUES (1, ?)",
            (db.utcnow(),),
        )
        conn.execute(
            "UPDATE archive_conversion SET heartbeat = ?, detail = ?,"
            " done = (SELECT COUNT(*) FROM archive_conversion_file WHERE job_id = ?"
            " AND state = 'replaced'), failed = (SELECT COUNT(*)"
            " FROM archive_conversion_file"
            " WHERE job_id = ? AND state = 'failed'), output_bytes = COALESCE("
            " (SELECT SUM(output_bytes) FROM archive_conversion_file WHERE job_id = ?"
            " AND state = 'replaced'), 0) WHERE id = ?",
            (db.utcnow(), detail, job_id, job_id, job_id, job_id),
        )


def _survey(conn, job, folder):
    # No following directory symlinks; never flatten the original day tree.
    def unreadable(error):
        raise error

    for current, dirs, files in os.walk(folder, followlinks=False, onerror=unreadable):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        if get(conn, job["id"])["cancel_requested"]:
            return False
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO archive_conversion_worker VALUES (1, ?)",
                (db.utcnow(),),
            )
            for name in sorted(files):
                path = Path(current) / name
                if path.suffix.lower() != ".lfs" or path.is_symlink():
                    continue
                _safe(path, folder)
                if time.time() - path.stat().st_mtime < 60:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO archive_conversion_file"
                    " (job_id, path, signature, source_bytes) VALUES (?,?,?,?)",
                    (
                        job["id"],
                        path.relative_to(folder).as_posix(),
                        signature(path),
                        path.stat().st_size,
                    ),
                )
            conn.execute(
                "UPDATE archive_conversion SET heartbeat = ?,"
                " total = (SELECT COUNT(*) FROM archive_conversion_file"
                " WHERE job_id = ?), source_bytes = COALESCE("
                " (SELECT SUM(source_bytes) FROM archive_conversion_file"
                " WHERE job_id = ?), 0) WHERE id = ?",
                (db.utcnow(), job["id"], job["id"], job["id"]),
            )
    return True


def run_next(conn):
    """One serial job, called under the worker's process lock."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO archive_conversion_worker VALUES (1, ?)",
            (db.utcnow(),),
        )
    job = db.one(
        conn,
        "SELECT * FROM archive_conversion WHERE state IN ('queued','running')"
        " ORDER BY id LIMIT 1",
    )
    if job is None:
        return None
    with conn:
        conn.execute(
            "UPDATE archive_conversion SET state = 'running', started_at = COALESCE("
            " started_at, ?), heartbeat = ? WHERE id = ?",
            (db.utcnow(), db.utcnow(), job["id"]),
        )
    try:
        if job["archive_id"] is None:
            raise ConversionError("archive registration was removed")
        _, folder = archives.resolve(job["folder"], archives.roots()[0])
        _, registered = folder_for(conn, job["archive_id"])
        if not folder.is_relative_to(registered):
            raise ConversionError("archive registration changed")
        if job["mode"] == "survey":
            state = "ready" if _survey(conn, job, folder) else "cancelled"
        else:
            # Read the manifest incrementally; multi-month jobs need no list
            # of voltage arrays or thousands of open HDF5 handles.
            while True:
                if get(conn, job["id"])["cancel_requested"]:
                    break
                item = db.one(
                    conn,
                    "SELECT * FROM archive_conversion_file WHERE job_id = ?"
                    " AND state IN ('pending', 'verified') ORDER BY path LIMIT 1",
                    (job["id"],),
                )
                if item is None:
                    break
                try:
                    replace_one(conn, job, item, folder)
                except Exception as exc:
                    # A verified journal with a missing source must survive
                    # for recovery, rather than being retried as an unverified file.
                    journal = db.one(
                        conn,
                        "SELECT state FROM archive_conversion_file"
                        " WHERE job_id = ? AND path = ?",
                        (job["id"], item["path"]),
                    )
                    if (
                        journal["state"] == "verified"
                        and not (folder / item["path"]).exists()
                    ):
                        raise
                    with conn:
                        conn.execute(
                            "UPDATE archive_conversion_file"
                            " SET state = 'failed', detail = ?"
                            " WHERE job_id = ? AND path = ?",
                            (str(exc), job["id"], item["path"]),
                        )
                _progress(conn, job["id"], item["path"])
            _progress(conn, job["id"])
            latest = get(conn, job["id"])
            state = (
                "cancelled"
                if latest["cancel_requested"]
                else "failed"
                if latest["failed"]
                else "completed"
            )
        with conn:
            conn.execute(
                "UPDATE archive_conversion SET state = ?, finished_at = ? WHERE id = ?",
                (state, db.utcnow(), job["id"]),
            )
    except Exception as exc:
        with conn:
            conn.execute(
                "UPDATE archive_conversion SET state = 'failed', detail = ?,"
                " finished_at = ? WHERE id = ?",
                (str(exc), db.utcnow(), job["id"]),
            )
    return get(conn, job["id"])
