"""Archive-authorized job controls; no conversion on HTTP request threads."""

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from . import conversion, db
from .auth import Capability, Principal, require

router = APIRouter(dependencies=[Depends(require(Capability.ARCHIVE))])


@router.get("/archive-conversions")
def list_jobs(request: Request):
    conn = request.app.state.db
    return {
        "jobs": conversion.listing(conn),
        "worker": db.one(
            conn, "SELECT heartbeat FROM archive_conversion_worker WHERE id = 1"
        ),
    }


@router.post("/archive-conversions", status_code=202)
def inspect_folder(
    request: Request,
    payload: dict = Body(...),
    who: Principal = Depends(require(Capability.ARCHIVE)),
):
    try:
        archive_id = int(payload.get("archive_id", 0))
        return conversion.enqueue(
            request.app.state.db,
            archive_id,
            str(payload.get("subpath") or "."),
            who.name,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/archive-conversions/{job_id}")
def job_details(job_id: int, request: Request, offset: int = 0):
    conn = request.app.state.db
    job = conversion.get(conn, job_id)
    if job is None:
        raise HTTPException(404, "conversion job not found")
    items = db.rows(
        conn,
        "SELECT path, state, source_bytes, output_bytes, detail"
        " FROM archive_conversion_file WHERE job_id = ?"
        " ORDER BY CASE WHEN state = 'failed' THEN 0 ELSE 1 END, path"
        " LIMIT 100 OFFSET ?",
        (job_id, max(0, offset)),
    )
    return {"job": job, "files": items}


@router.post("/archive-conversions/{job_id}/start", status_code=202)
def start_job(
    job_id: int,
    request: Request,
    payload: dict = Body(...),
    who: Principal = Depends(require(Capability.ARCHIVE)),
):
    try:
        return conversion.start(
            request.app.state.db, job_id, payload.get("confirm_delete"), who.name
        )
    except conversion.ConversionError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/archive-conversions/{job_id}/cancel")
def cancel_job(job_id: int, request: Request):
    job = conversion.cancel(request.app.state.db, job_id)
    if job is None:
        raise HTTPException(404, "conversion job not found")
    return job
