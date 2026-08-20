"""Registering folders to index, and asking for a pass over one.

Reads are open and writes need the control token, matching
:mod:`services.api.control_routes`. Registering a folder does not touch a
radio, but it does commit this server to hours of pipeline over data nobody
else chose, so it sits on the same side of the fence as the other writes.

The real work is in :mod:`services.api.archives`; this is the HTTP shape of
it. Validation happens **before** the row is written, and reports what an
indexer would find, so a folder that will load nothing is refused while the
operator is looking at it rather than after a scan that truthfully reports
zero.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from muf import loader
from muf.extractors import ALL_METHODS, DEFAULT_METHODS

from . import archives as archives_mod
from . import db
from .auth import require_control

router = APIRouter(tags=["archives"])


def _clean_methods(value) -> str:
    """Comma-separated method names, defaulted and checked against what exists.

    An unknown method is refused rather than dropped: `watch.already_done`
    counts a sounding finished only when it holds a row for every *requested*
    method, so a name nothing can produce would make every sounding in the
    archive look permanently unfinished and re-scan the whole folder on every
    pass, forever.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return ",".join(DEFAULT_METHODS)
    if isinstance(value, str):
        wanted = [m.strip() for m in value.split(",") if m.strip()]
    else:
        wanted = [str(m).strip() for m in value if str(m).strip()]
    unknown = [m for m in wanted if m not in ALL_METHODS]
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown extraction method(s): {', '.join(unknown)}; "
            f"available: {', '.join(ALL_METHODS)}")

    # Known but not usable here is the more dangerous case, and the one worth
    # the longer message: `cnn` imports on a machine with Keras yet still
    # needs a model trained on this geometry. Requested without one it raises
    # per file, which by itself would only be noisy -- but `already_done`
    # counts a sounding finished when it holds a row for *every* requested
    # method, so a method that can never produce one leaves the whole archive
    # permanently unfinished and re-scanned on every pass, forever.
    reasons = archives_mod.method_availability()
    unusable = [m for m in wanted if not reasons.get(m, {}).get("usable")]
    if unusable:
        detail = "; ".join(
            f"{m} ({reasons.get(m, {}).get('why') or 'unavailable'})"
            for m in unusable)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"not usable on this server: {detail}. Requesting it would not "
            f"merely fail -- every sounding in the archive would stay "
            f"unfinished and be re-scanned on every pass. Usable here: "
            f"{', '.join(archives_mod.usable_methods())}.")
    if not wanted:
        return ",".join(DEFAULT_METHODS)
    return ",".join(wanted)


def _row(request: Request, archive_id: int) -> dict:
    row = db.archive(request.app.state.db, archive_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such archive")
    return row


@router.get("/archives")
def list_archives(request: Request) -> dict:
    conn = request.app.state.db
    return {
        "archives": db.archives(conn),
        "archive_root": str(request.app.state.archive_root),
        # What is mounted here, and which host folder it came from. See
        # `archives.mount`: the second half only exists because compose passes
        # ARCHIVE_HOST_PATH in, and without it "/archive" alone cannot tell an
        # operator whether the .env they edited took effect.
        "mount": archives_mod.mount(),
        "candidates": archives_mod.candidates(
            conn, request.app.state.archive_root),
        "status": archives_mod.status(),
        "formats": list(loader.FORMATS),
        "methods": archives_mod.method_availability(),
        "methods_available": list(archives_mod.usable_methods()),
        "methods_default": list(DEFAULT_METHODS),
    }


@router.post("/archives")
def add_archive(request: Request, payload: dict = Body(...),
                _: str = Depends(require_control)) -> dict:
    root = request.app.state.archive_root
    fmt = (payload.get("format") or "").strip() or None
    if fmt is not None and fmt not in loader.FORMATS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown format {fmt!r}; choose from {', '.join(loader.FORMATS)}"
            f" or leave it empty for any")

    try:
        relpath, absolute = archives_mod.resolve(payload.get("path"), root)
        found = archives_mod.survey(absolute, fmt)
    except archives_mod.ArchiveError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    if not found["soundings"]:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{relpath} holds no soundings this server can read"
            + (f" in format {fmt}" if fmt else "")
            + ". Registering it would schedule a scan that loads nothing on "
              "every pass. Check the path, or drop the format filter if the "
              "files are of another kind.")

    name = (str(payload.get("name") or "").strip() or relpath)
    methods = _clean_methods(payload.get("methods"))

    conn = request.app.state.db
    if db.one(conn, "SELECT id FROM archive WHERE relpath = ?", (relpath,)):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"{relpath} is already registered")
    if db.one(conn, "SELECT id FROM archive WHERE name = ?", (name,)):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"an archive named {name!r} already exists")

    archive_id = db.add_archive(conn, name=name, relpath=relpath,
                                methods=methods, format=fmt)
    return {"ok": True, "id": archive_id, "name": name, "path": relpath,
            "methods": methods, "found": found,
            "note": "registered; press scan to index it"}


@router.post("/archives/{archive_id}/scan")
def scan_archive(archive_id: int, request: Request,
                 _: str = Depends(require_control)) -> dict:
    row = _row(request, archive_id)
    started = archives_mod.scan_in_background(
        row, archive_root=request.app.state.archive_root)
    if started is None:
        running = archives_mod.status()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"a scan of {running.get('name') or 'another archive'} is already "
            f"running. Scans are one at a time on purpose: they are CPU-bound "
            f"and they lock the database, so two would finish later than one "
            f"after the other.")
    return {"ok": True, "id": archive_id, "name": row["name"],
            "note": "scanning; the result appears on this page when it ends"}


@router.post("/archives/{archive_id}/enabled")
def set_enabled(archive_id: int, request: Request, payload: dict = Body(...),
                _: str = Depends(require_control)) -> dict:
    row = _row(request, archive_id)
    enabled = bool(payload.get("enabled"))
    db.set_archive_enabled(request.app.state.db, archive_id, enabled)
    return {"ok": True, "id": archive_id, "name": row["name"],
            "enabled": enabled}


@router.post("/archives/{archive_id}/methods")
def set_methods(archive_id: int, request: Request, payload: dict = Body(...),
                _: str = Depends(require_control)) -> dict:
    _row(request, archive_id)
    methods = _clean_methods(payload.get("methods"))
    db.set_archive_methods(request.app.state.db, archive_id, methods)
    return {"ok": True, "id": archive_id, "methods": methods,
            "note": "soundings missing any of these are revisited on the next "
                    "scan; nothing already computed is recomputed"}


@router.delete("/archives/{archive_id}")
def delete_archive(archive_id: int, request: Request,
                   _: str = Depends(require_control)) -> dict:
    row = _row(request, archive_id)
    conn = request.app.state.db
    kept = next((a["soundings"] for a in db.archives(conn)
                 if a["id"] == archive_id), 0)
    db.remove_archive(conn, archive_id)
    return {"ok": True, "id": archive_id, "name": row["name"],
            "soundings_kept": kept,
            "note": f"unregistered. {kept} sounding(s) and their extractions "
                    f"stay in the database -- this stops the indexing, it "
                    f"does not discard what was measured."}
