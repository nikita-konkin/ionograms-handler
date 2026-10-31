"""Durable station uploads, ingested without walking or waiting for the NAS.

Files and receipts live beside API_DB, on local persistent storage. A receipt
is acknowledged only after the bytes and the pending job are on disk. Failed
jobs stay pending across restarts; numerics run outside the API process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from . import apart, db, ingest
from .auth import Capability, require

router = APIRouter(tags=["station files"])
LOCK = threading.Lock()
MAX_BYTES = 256 * 1024 * 1024
SIGNATURE = b"\x89HDF\r\n\x1a\n"


def enabled() -> bool:
    return os.environ.get("DIRECT_INGEST", "0") == "1"


def root() -> Path:
    return ingest.direct_root()


def _paths(station: str, name: str) -> tuple[Path, Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", station):
        raise HTTPException(400, "invalid station name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,230}\.h5", name):
        raise HTTPException(400, "expected an HDF5 basename")
    folder = root() / station
    return folder / name, folder / (name + ".json"), folder / (name + ".pending")


def _save(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _sync_dir(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _publish(tmp: Path, station: str, name: str, digest: str, size: int) -> dict:
    path, receipt, pending = _paths(station, name)
    with LOCK:
        previous = receipt if receipt.exists() else pending
        if previous.exists():
            old = json.loads(previous.read_text(encoding="utf-8"))
            if old["sha256"] != digest:
                raise HTTPException(409, "a different file already has this name")
            # A retry after an interrupted publication repairs the queue.
            if old["status"] == "queued" and not pending.exists():
                _save(pending, old)
            if not receipt.exists():
                _save(receipt, old)
            return old
        os.replace(tmp, path)
        job = {"station": station, "file": name, "sha256": digest,
               "bytes": size, "received_at": db.utcnow(), "status": "queued"}
        _save(pending, job)
        _save(receipt, job)
        return job


@router.put("/stations/{station}/files/{name}", status_code=202,
            dependencies=[Depends(require(Capability.AGENT))])
async def upload(station: str, name: str, request: Request) -> dict:
    if not enabled():
        raise HTTPException(503, "direct ingestion is disabled (DIRECT_INGEST=1)")
    path, _, _ = _paths(station, name)
    digest = request.headers.get("x-content-sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HTTPException(400, "X-Content-SHA256 is required")
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError:
        raise HTTPException(400, "invalid Content-Length") from None
    if length < len(SIGNATURE) or length > MAX_BYTES:
        raise HTTPException(413, "file must be between 8 bytes and 256 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".upload")
    tmp = Path(temporary)
    sha, size, prefix = hashlib.sha256(), 0, b""
    try:
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > length:
                    raise HTTPException(413, "body exceeds declared length")
                prefix = (prefix + chunk)[:8] if len(prefix) < 8 else prefix
                sha.update(chunk)
                await run_in_threadpool(f.write, chunk)
            await run_in_threadpool(f.flush)
            await run_in_threadpool(os.fsync, f.fileno())
        if size != length or sha.hexdigest() != digest:
            raise HTTPException(400, "incomplete file or checksum mismatch")
        if prefix != SIGNATURE:
            raise HTTPException(400, "not an HDF5 file")
        return await run_in_threadpool(_publish, tmp, station, name, digest, size)
    finally:
        tmp.unlink(missing_ok=True)


@router.get("/stations/{station}/files/{name}",
            dependencies=[Depends(require(Capability.AGENT))])
def status(station: str, name: str) -> dict:
    _, receipt, _ = _paths(station, name)
    if not receipt.exists():
        raise HTTPException(404, "file not received")
    return json.loads(receipt.read_text(encoding="utf-8"))


def _extract(path: str) -> dict:
    from muf.pipeline import Options, process_file

    row = process_file(Path(path), Options(methods=("contour",)))
    if row.get("error") or row.get("datetime") is None:
        raise ValueError(row.get("error") or "sounding has no timestamp")
    return row


def extract(path: str) -> dict:
    return apart.run(_extract, path, timeout_s=120, what="direct sounding ingest")


def advance(conn, archive_root: Path, *, process=extract) -> int:
    """One bounded local queue pass. No NAS stat, enumeration, or scan lock."""
    count = 0
    for pending in sorted(root().glob("*/*.pending")):
        try:
            job = json.loads(pending.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"direct ingest receipt {pending.name}: {exc}", flush=True)
            continue
        if job.get("retry_at", 0) > time.time():
            continue
        path, receipt, _ = _paths(job["station"], job["file"])
        try:
            row = process(str(path))
            # The original name is the archive's de-duplication key.
            row["file"] = path.name
            with conn:
                sounding = ingest.ingest_row(conn, row, path, archive_root,
                                             ("contour",))
            job.update(status="ingested" if sounding else "muted",
                       sounding_id=sounding, ingested_at=db.utcnow())
            job.pop("error", None)
            with LOCK:
                _save(receipt, job)
                pending.unlink()
        except Exception as exc:
            conn.rollback()
            attempts = job.get("attempts", 0) + 1
            job.update(status="queued", error=str(exc), attempts=attempts,
                       retry_at=time.time() + min(3600, 30 * 2 ** min(attempts, 7)))
            with LOCK:
                _save(pending, job)
                _save(receipt, job)
            print(f"direct ingest {path.name}: {exc}", flush=True)
        count += 1
        if count >= 8:
            break
    return count


def start(app):
    stop = threading.Event()
    if not enabled():
        return stop.set

    def loop():
        while not stop.wait(5):
            try:
                conn = db.connect()
                try:
                    conn.execute("PRAGMA busy_timeout = 30000")
                    advance(conn, Path(app.state.archive_root))
                finally:
                    conn.close()
            except Exception as exc:
                print(f"direct ingest pass failed: {exc}", flush=True)

    threading.Thread(target=loop, daemon=True, name="direct-ingest").start()
    return stop.set
