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
from .auth import Capability, Principal, require

router = APIRouter(tags=["archives"])


def _any(value: str | None) -> str | None:
    """``*`` and the empty string both mean "every receiver".

    A path segment cannot be empty, so the wildcard needs a spelling that
    survives a URL. `*` is the one every operator already reaches for.
    """
    if value is None:
        return None
    value = value.strip()
    return None if value in ("", "*") else value


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


@router.get("/archives/status")
def scan_status() -> dict:
    """Just the running scan, for the progress bar to poll.

    Its own endpoint because the bar asks once a second and `GET /archives`
    is not cheap: it counts soundings per archive and reports the candidate
    list. Polling that was what made the page unusable during an index --
    every tick re-walked the archive tree on the request thread.

    This one reads a module-level dataclass under a lock. No disk, no
    database, nothing that grows with the size of the archive.
    """
    return archives_mod.status()


@router.get("/archives/candidates")
def list_candidates(request: Request) -> dict:
    """Registerable folders, surveyed in the background.

    Returns immediately whatever the last survey found, with `ready` false
    until there has been one. The page asks again while it is false rather
    than showing "no folders", which would be a lie about a mounted disk.
    """
    return archives_mod.candidates_cached(
        request.app.state.db, request.app.state.archive_root)


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
        # From cache, never walked here -- see `archives.candidates_cached`.
        # This endpoint is polled once a second during a scan.
        "candidates": archives_mod.candidates_cached(
            conn, request.app.state.archive_root)["items"],
        "status": archives_mod.status(),
        "formats": list(loader.FORMATS),
        "methods": archives_mod.method_availability(),
        "methods_available": list(archives_mod.usable_methods()),
        "methods_default": list(DEFAULT_METHODS),
    }


@router.post("/archives")
def add_archive(request: Request, payload: dict = Body(...),
                _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
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

    # Refused rather than warned. A scan is recursive, so overlapping rows
    # walk the same files twice on every pass and the only symptom is that
    # indexing takes twice as long -- invisible unless you already suspect it.
    # Registering the root over folders already registered inside it is the
    # obvious move once the root is allowed, so it is the one to catch.
    clashes = archives_mod.overlapping(request.app.state.db, relpath, root)
    if clashes and not payload.get("replace"):
        inside = [c["relpath"] for c in clashes if c["relation"] == "inside"]
        outside = [c["relpath"] for c in clashes if c["relation"] == "contains"]
        if outside:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"{relpath} is already covered by {', '.join(outside)}, which "
                f"is registered and scanned recursively. Registering both "
                f"walks these files twice on every pass. Remove the other "
                f"registration first, or register something outside it.")
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{relpath} contains {len(inside)} folder(s) that are already "
            f"registered on their own: {', '.join(inside[:6])}"
            + (" and others" if len(inside) > 6 else "")
            + ". Scanning is recursive, so keeping both walks every file "
              "twice. Send replace=true to register this folder and drop "
              "those rows -- the soundings they already indexed stay in the "
              "database, because they are keyed by file rather than by "
              "archive.")

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

    # `replace` got this far only by being explicit, and the refusal above
    # said what it would do. The rows go; the soundings they indexed stay,
    # because `sounding` is keyed by file and has no archive_id -- which is
    # what makes consolidating fifteen day folders into one root a bookkeeping
    # change rather than a reindex.
    dropped = []
    if payload.get("replace"):
        for clash in archives_mod.overlapping(conn, relpath, root):
            if clash["relation"] == "inside":
                db.remove_archive(conn, clash["id"])
                dropped.append(clash["relpath"])

    archive_id = db.add_archive(conn, name=name, relpath=relpath,
                                methods=methods, format=fmt)
    archives_mod.forget_candidates()
    note = "registered; press scan to index it"
    if dropped:
        note = (f"registered, and dropped {len(dropped)} folder(s) now covered "
                f"by it; their soundings stay indexed")
    return {"ok": True, "id": archive_id, "name": name, "path": relpath,
            "methods": methods, "found": found, "replaced": dropped,
            "note": note}


@router.post("/archives/{archive_id}/scan")
def scan_archive(archive_id: int, request: Request,
                 _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
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
                _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    row = _row(request, archive_id)
    enabled = bool(payload.get("enabled"))
    db.set_archive_enabled(request.app.state.db, archive_id, enabled)
    return {"ok": True, "id": archive_id, "name": row["name"],
            "enabled": enabled}


@router.post("/archives/{archive_id}/methods")
def set_methods(archive_id: int, request: Request, payload: dict = Body(...),
                _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    _row(request, archive_id)
    methods = _clean_methods(payload.get("methods"))
    db.set_archive_methods(request.app.state.db, archive_id, methods)
    return {"ok": True, "id": archive_id, "methods": methods,
            "note": "soundings missing any of these are revisited on the next "
                    "scan; nothing already computed is recomputed"}


def _clean_format(value) -> str | None:
    """A format name, or ``None`` for "any". Mirrors `_clean_methods`."""
    fmt = (str(value).strip() if value is not None else "")
    if not fmt or fmt in ("any", "None"):
        return None
    if fmt not in loader.FORMATS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown format {fmt!r}; choose from {', '.join(loader.FORMATS)}"
            f" or leave it empty for any")
    return fmt


@router.post("/archives/{archive_id}/format")
def set_format(archive_id: int, request: Request, payload: dict = Body(...),
               _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Narrow or widen what this folder may contribute.

    Unlike the method list, this takes effect on the **next scan** as well as
    retrospectively: `scan_once` hands it to `watch.find_new`, so a narrowed
    archive stops ingesting the other formats rather than merely declaring a
    preference. That is deliberate, and it is what makes removal stick.

    Narrowing does not delete anything by itself. The response reports what is
    now out of scope so the page can offer to remove it as a separate,
    confirmed act -- changing a rule and destroying rows are different
    decisions and should not share one button.
    """
    _row(request, archive_id)
    fmt = _clean_format(payload.get("format"))
    conn = request.app.state.db
    db.set_archive_format(conn, archive_id, fmt)
    orphans = db.archive_orphans(conn, archive_id)
    return {"ok": True, "id": archive_id, "format": fmt,
            "orphans": orphans,
            "note": ("every format" if fmt is None else
                     f"only {fmt} from now on")
                    + (f"; {orphans['total']} already-indexed sounding(s) no "
                       f"longer match" if orphans["total"] else "")}


@router.get("/archives/{archive_id}/orphans")
def list_orphans(archive_id: int, request: Request) -> dict:
    """What this archive holds that its own format no longer allows.

    Open, like the other reads. It counts; it does not remove.
    """
    _row(request, archive_id)
    return {"id": archive_id,
            **db.archive_orphans(request.app.state.db, archive_id)}


@router.delete("/archives/{archive_id}/orphans")
def delete_orphans(archive_id: int, request: Request,
                   _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Remove exactly what the preview counted.

    The same query decides both, so what was shown is what goes. Their
    `extraction` and `reference` rows go too, by cascade.

    This is the one destructive act on this page, and it is only safe because
    the archive's format keeps them gone: `find_new` treats a file with no row
    as new forever, so without the rule these would be back on the next pass.
    Widening the format again brings them back on the next scan, which is the
    honest undo -- the files were never touched, the mount is read-only.
    """
    row = _row(request, archive_id)
    conn = request.app.state.db
    if not row["format"]:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{row['name']} admits every format, so nothing in it is out of "
            f"scope. Narrow the format first -- otherwise this would delete "
            f"soundings that the very next scan would ingest again.")
    before = db.archive_orphans(conn, archive_id)
    removed = db.delete_archive_orphans(conn, archive_id)
    return {"ok": True, "id": archive_id, "removed": removed,
            "was": before["by_format"],
            "note": f"{removed} sounding(s) removed, with their extractions "
                    f"and modelled values. The files are untouched -- widen "
                    f"the format and the next scan reads them again."}


@router.delete("/archives/{archive_id}")
def delete_archive(archive_id: int, request: Request,
                   _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    row = _row(request, archive_id)
    conn = request.app.state.db
    kept = next((a["soundings"] for a in db.archives(conn)
                 if a["id"] == archive_id), 0)
    db.remove_archive(conn, archive_id)
    archives_mod.forget_candidates()
    return {"ok": True, "id": archive_id, "name": row["name"],
            "soundings_kept": kept,
            "note": f"unregistered. {kept} sounding(s) and their extractions "
                    f"stay in the database -- this stops the indexing, it "
                    f"does not discard what was measured."}


# --------------------------------------------------------------------------
# Circuits, and ruling one out
# --------------------------------------------------------------------------
#
# A circuit here is a (tx, rx) pair the database actually holds, not one
# anybody configured. That is the point: `unkown -> DOB` was never configured
# by anyone -- chirp v2 writes `unkown` when it cannot identify the
# transmitter (`muf/stations.py:UNIDENTIFIED`, upstream's spelling), so every
# unidentified emitter in an archive arrives under one string. The result is
# not a circuit; it is a pile of different paths wearing one name, on a range
# axis with no absolute zero.
#
# **Deleting the rows is not enough on its own.** `find_new` treats a file
# with no row as new forever, so the next scan reads them straight back in.
# That is the same trap `delete_orphans` refuses to walk into, and a mute rule
# is the equivalent guard for a circuit: `ingest` declines to write a muted
# one, so the deletion stays done. Unmuting and re-scanning is the undo, and
# it works because nothing here touches a file -- the mount is read-only.

@router.get("/circuits")
def list_circuits(request: Request) -> dict:
    """Every circuit in the database, with what hangs off it.

    Open, like the other reads. It counts; it does not remove.
    """
    conn = request.app.state.db
    return {"circuits": db.circuits(conn), "muted": db.muted_circuits(conn)}


@router.get("/circuits/{tx}/{rx}")
def circuit_holdings(tx: str, rx: str, request: Request) -> dict:
    """What deleting this circuit would take with it, counted before anything.

    The same query the delete uses, so what was shown is what goes.
    """
    return db.circuit_holdings(request.app.state.db, tx, _any(rx))


@router.post("/circuits/mute")
def mute_circuit(request: Request, spec: dict = Body(...),
                 who: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Refuse this circuit at ingest from here on.

    `rx` omitted or `*` means every receiver, which is what a marker like
    `unkown` wants: it is not one emitter, so pinning the rule to one receiver
    would leave the same bad string arriving on the others.

    Muting alone removes nothing. It stops the bleeding; `DELETE
    /circuits/{tx}/{rx}` is what clears what is already there.
    """
    conn = request.app.state.db
    tx = (spec.get("tx") or "").strip()
    if not tx:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "a mute rule needs a transmitter")
    rx = _any(spec.get("rx"))
    rule = db.mute_circuit(conn, tx, rx, note=spec.get("note"), by=who.name)
    holds = db.circuit_holdings(conn, tx, rx)
    scope = f"{tx} -> {rx}" if rx else f"{tx} -> any receiver"
    return {"ok": True, "muted": rule, "holds": holds,
            "detail": f"{scope} will not be ingested again. "
                      + (f"{holds['soundings']} sounding(s) are already in the "
                         f"database -- muting does not remove them."
                         if holds["soundings"] else
                         "Nothing of it is in the database.")}


@router.delete("/circuits/mute/{rule_id}")
def unmute_circuit(rule_id: int, request: Request,
                   _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Drop a mute rule. The next scan reads those files again."""
    row = db.unmute_circuit(request.app.state.db, rule_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no mute rule {rule_id}")
    scope = f"{row['tx']} -> {row['rx'] or 'any receiver'}"
    return {"ok": True, "unmuted": row,
            "detail": f"{scope} is no longer refused. The files were never "
                      f"touched, so the next scan reads them again."}


@router.delete("/circuits/{tx}/{rx}")
def delete_circuit(tx: str, rx: str, request: Request,
                   _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Remove a circuit's soundings, with their extractions and references.

    **Refused unless the circuit is muted first**, for the reason
    `delete_orphans` refuses on an archive that admits every format: without a
    rule keeping them out, the next scan ingests every one of these files
    again and the delete was theatre. Mute, then delete, and it stays done.
    """
    conn = request.app.state.db
    rx = _any(rx)
    if not db.is_muted(conn, tx, rx):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{tx} -> {rx or 'any receiver'} is not muted. Mute it first -- "
            f"otherwise the next scan ingests every one of these files again "
            f"and nothing has been achieved.")
    before = db.circuit_holdings(conn, tx, rx)
    if not before["soundings"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"no soundings for {tx} -> {rx or 'any receiver'}")
    removed = db.delete_circuit(conn, tx, rx)
    return {"ok": True, "removed": removed, "was": before,
            "note": f"{removed} sounding(s) removed, with "
                    f"{before['extractions']} extraction(s) and "
                    f"{before['references']} modelled value(s). The files are "
                    f"untouched -- unmute and the next scan reads them again."}
