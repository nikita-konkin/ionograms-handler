"""Queueing commands for a station.

Control scope only. This is the endpoint that can stop a radio, so two things
are true of it that are not true of anything else in the service:

**The verb is allow-listed here as well as in the agent.** ``control.py``
already refuses anything outside ``ALLOWED_VERBS``, and duplicating that check
is deliberate: the agent's list is the last line, and a server that will queue
anything means an unreachable station accumulates commands nobody vetted. Two
checks, and the outer one is the one an operator can see.

**Process verbs, plus one parameter edit.** ``control.py`` implements
validated edits to mode, sounder_timings, output_dir, max_range_extent and
save_raw_voltage. Only ``mode`` and ``sounder_timings`` are routed here, and
only together, because those two are the sounding mode -- what the station
records at all -- and an operator who can see the emitter census on
``/ui/sources`` has no other way to act on it than editing the ini by hand on
the station.

**Naming a transmitter is here too, and it is not an aside.** A detection is
anonymous: the census can tell you a 100 kHz/s emitter arrives at second 235
of every 300, and nothing more. But ``calc_ionograms.py`` reads
``transmit_name`` off every schedule entry and writes it into the product's
file name and into ``ho["txname"]``, which this pipeline ingests as
``sounding.tx`` and resolves against ``muf/stations.py`` for the path geometry
and the band ceiling. So the identification an operator makes -- the same
judgement that resolved ``cyprus1`` to ``NIC`` -- is the hinge the whole
downstream chain turns on, and it is recorded, with its evidence, before it
can be scheduled.

The rest stay unrouted. ``output_dir`` decides where a week of data lands and
a typo is unrecoverable from here; the other two change how products are
formed and want a considered decision rather than a button.

This is a widening of what the web can do to a radio, so it is narrow on
purpose:

* the parameter allow-list is checked here as well as in the agent;
* ``mode`` is checked against ``control.MODES`` rather than passed through;
* leaving search mode requires ``sounder_timings`` in the *same* command,
  which is the agent's own rule (a scheduled station with no schedule records
  nothing while every process reports healthy);
* the change rewrites the ini and takes effect on the next restart, so it is
  two deliberate actions, not one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from fastapi import (APIRouter, Body, Depends, HTTPException, Query, Request,
                     status)

from muf.stations import default_registry

from . import acquisition, db
from .auth import require_control
from .read_routes import _age_seconds

router = APIRouter(tags=["control"])

#: What the web interface may queue. A strict subset of the agent's own
#: ``ALLOWED_VERBS`` plus ``set_config``; see the module docstring.
QUEUEABLE = ("start", "stop", "restart", "set_config")

#: Settings ``set_config`` may carry from the web. A strict subset of
#: ``control.EDITABLE`` plus ``control.COMPOSITE`` -- see the module docstring
#: for what is left out.
#:
#: ``set_band`` is a composite, not a key: it carries ``band_start_mhz`` and
#: optionally an analysis window, and the agent expands it into the six ini
#: entries of ``control.BAND_INI``. Those six are deliberately **not** here and
#: not in ``control.EDITABLE`` either -- the band is five values that must
#: agree, and offering them as five fields is the exact shape that blinded the
#: station twice on 2026-08-19.
#:
#: Safe to expose only because the agent refuses a band change outright when
#: the deployed recorder has no ``--center-freq`` (patch 0014). A station on an
#: older binary rejects the command rather than tuning nowhere, so this does
#: not have to wait on every station being rebuilt.
WEB_EDITABLE = ("mode", "sounder_timings", "set_band")


@router.post("/stations/{station}/commands")
def enqueue(station: str, request: Request,
            payload: dict = Body(...),
            _: str = Depends(require_control)) -> dict:
    name = str(payload.get("name", "")).strip().lower().replace("-", "_")
    if name not in QUEUEABLE:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{name!r} is not queueable from the web interface; "
            f"allowed: {', '.join(QUEUEABLE)}")

    params = payload.get("params") or {}
    if name == "set_config":
        params = _vet_config(params)

    command_id = db.enqueue(request.app.state.db, station, name,
                            params=params,
                            issued_by=str(payload.get("issued_by") or "web"))
    return {"ok": True, "id": command_id, "name": name, "station": station,
            "note": "queued; the station will collect it on its next pull"}


def _vet_config(params: dict) -> dict:
    """Check a ``set_config`` payload before it is queued.

    The agent checks all of this again -- that is the last line and it stays.
    This is the outer check, and it exists so a bad command is refused while an
    operator is looking at the screen rather than accepted, queued, and
    rejected minutes later on a station nobody is watching.
    """
    from services.agent import control

    changes = params.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "set_config needs a non-empty 'changes' object")

    unknown = sorted(set(changes) - set(WEB_EDITABLE))
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{', '.join(unknown)} cannot be set from the web interface; "
            f"allowed: {', '.join(WEB_EDITABLE)}")

    if "set_band" in changes:
        _vet_band(changes["set_band"])

    mode = str(changes.get("mode", "")).lower()
    if "mode" in changes and mode not in control.MODES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"mode {changes['mode']!r} unknown; "
            f"choose from {sorted(set(control.MODES))}")

    # Leaving search mode without a schedule is the failure the agent exists to
    # prevent: every process reports healthy and nothing is recorded. Checked
    # here too so the refusal reaches the operator immediately.
    if mode in ("scheduled", "schedule"):
        timings = changes.get("sounder_timings")
        entries = _parse_timings(timings)
        if not entries:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "scheduled mode needs sounder_timings in the same command, or "
                "the station would record nothing while reporting healthy")
        for entry in entries:
            missing = sorted(set(acquisition.REQUIRED_ENTRY_KEYS) - set(entry))
            if missing:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"sounder_timings entry {entry} is missing {missing}. "
                    f"calc_ionograms.py reads all five keys with a bare "
                    f"subscript, and `transmit_name` is the name the product "
                    f"file is called after -- which is why a schedule is "
                    f"composed from identified transmitters "
                    f"(POST /stations/{{id}}/schedule) rather than straight "
                    f"from the emitter census.")
    return {"changes": changes}


def _vet_band(request) -> None:
    """Shape-check a ``set_band`` payload while the operator is still looking.

    Only the shape and the arithmetic that needs nothing from the station. The
    checks that matter -- the window against the digitised band, the sweep
    against every transmitter's ``rep``, the block-grid alignment, and whether
    the deployed recorder even reads ``center_freq`` -- all need the station's
    own ini and its own binary, so they belong to the agent and stay there.
    This is the outer check, and it is the one that catches a typo before it
    costs a round trip.
    """
    from services.agent import control

    if not isinstance(request, dict):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"set_band takes an object with {list(control.BAND_ARGS)}")
    unknown = sorted(set(request) - set(control.BAND_ARGS))
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"set_band: unknown field(s) {unknown}; "
            f"allowed: {list(control.BAND_ARGS)}")
    if "band_start_mhz" not in request:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "set_band needs band_start_mhz")
    for name in control.BAND_ARGS:
        value = request.get(name)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"set_band: {name} = {request[name]!r} is not a number") from None
        if value < 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"set_band: {name} is below zero")
    lo, hi = request.get("analysis_min_mhz"), request.get("analysis_max_mhz")
    if lo is not None and hi is not None and float(hi) <= float(lo):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"set_band: analysis_max_mhz {hi} is not above "
            f"analysis_min_mhz {lo}")


def _parse_timings(value):
    """`sounder_timings` as a list of entries, however it arrived.

    The station's ini stores a nested list; a browser form sends a JSON string.
    Both are accepted, and anything else is not a schedule. One reader for both
    shapes, in :mod:`services.api.acquisition`, because the rank grouping is
    also what the console panel and the schedule composer read.
    """
    return acquisition.entries(value)


# --------------------------------------------------------------------------
# Verified transmitters, and the schedule built from them
# --------------------------------------------------------------------------

#: What a transmitter code may contain.
#:
#: **No dash.** The code is written into the schedule as `transmit_name`, and
#: `calc_ionograms.py:344` builds the product's file name as
#: `lfm_ionogram-{txname}-{station}-{ch}-{cid:03d}-{t0:.2f}.h5`, which
#: `muf/io_chirp.py:188` reads back with a dash-delimited regex. A dash inside
#: the name does not fail to parse -- it parses into the *next* field, so the
#: transmitter's tail becomes the receiver and every field after it shifts by
#: one. A week of products would carry a plausible wrong receiver.
#:
#: No whitespace or separators either, for the ordinary reason: this string
#: becomes a path component on the station.
CODE_RE = re.compile(r"^[A-Za-z0-9_.]{1,24}$")


@router.post("/stations/{station}/transmitters")
def save_transmitter(station: str, request: Request,
                     payload: dict = Body(...),
                     _: str = Depends(require_control)) -> dict:
    """Identify one emitter from the census as a named transmitter.

    Control scope, though it stops no radio. What it writes is what a later
    schedule is composed from, and the name it fixes ends up in the file name
    of every product the station records from that transmitter -- so it is a
    change to the station's output, made one step earlier.
    """
    code = str(payload.get("code") or "").strip()
    if not CODE_RE.match(code):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{code!r} is not a usable transmitter code. Letters, digits, "
            f"underscore and dot, up to 24 characters -- no dash: the code "
            f"becomes `transmit_name`, which is a dash-delimited field of the "
            f"product's file name, and a dash inside it silently shifts every "
            f"field after it.")

    timings = payload.get("timings")
    if not isinstance(timings, list) or not timings:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "timings must be a non-empty list of {chirp-rate, rep, chirpt} "
            "entries -- one per slot of this transmitter you want sounded")
    for entry in timings:
        if not isinstance(entry, dict):
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"timings entry {entry!r} is not an object")
        missing = sorted({"chirp-rate", "rep", "chirpt"} - set(entry))
        if missing:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"timings entry {entry} is missing {missing}")
        try:
            if float(entry["rep"]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"timings entry {entry} has no usable `rep`; the cycle is what "
                f"places the slot on a clock") from None

    record = db.save_transmitter(
        request.app.state.db, station, code, timings,
        name=(str(payload.get("name")).strip() or None
              if payload.get("name") else None),
        evidence=payload.get("evidence"),
        verified_by=str(payload.get("verified_by") or "web"),
        note=payload.get("note"))
    response = {"ok": True, "station": station, "transmitter": record}

    # A code `muf/stations.py` cannot resolve is saved, not refused: a newly
    # heard emitter has to be nameable before anyone knows where it is, and
    # refusing here would make identifying one impossible. But it is said out
    # loud, because nothing downstream will.
    #
    # `io_chirp._coords_for` returns NaN for an unknown name by design, so the
    # consequence is silent and hours away: the range gate falls back to the
    # full span, `path_km` is NULL, `sounded_ceiling` loses its measured limit,
    # and IRI is finally asked for a foF2 at `nanS nanW`. That is the first
    # sentence the operator sees, on a sounding page, with nothing connecting
    # it to the name they typed. It happened on 2026-08-16 with NIC1 and NIC3.
    if default_registry().station(code) is None:
        response["warning"] = (
            f"Saved, but {code!r} is not in the station registry, so products "
            f"from it will have no transmitter coordinates: no path length, no "
            f"M-factor, a full-span range gate, and IRI will report a foF2 at "
            f"nanS nanW. Either reuse the code of a site already known, or add "
            f"{code!r} to muf/stations.py -- as an alias if this is another "
            f"slot of an emitter already there.")
    return response


@router.delete("/stations/{station}/transmitters/{code}")
def forget_transmitter(station: str, code: str, request: Request,
                       _: str = Depends(require_control)) -> dict:
    """Forget an identification. Products already recorded keep the name."""
    removed = db.delete_transmitter(request.app.state.db, station, code)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"{station} has no transmitter {code!r}")
    return {"ok": True, "station": station, "code": code}


@router.delete("/stations/{station}")
def forget_station(station: str, request: Request, force: bool = False,
                   _: str = Depends(require_control)) -> dict:
    """Remove a renamed or retired receiver from the console.

    The console lists every station that has ever pushed, so a receiver that
    changed its name leaves a panel behind that is STALE for good. That is the
    one thing a health console cannot afford: a red that is always red is a
    red nobody reads, and the next one that means something arrives beside it.

    **A station still reporting is refused.** Deleting its history would clear
    the panel until the next push and lose the record of when reports stopped
    -- the question sec. 5.4 keeps every report to answer. The name lives in
    the station's own ``~/agent.json``, so the fix for a live station is
    there, not here. ``?force=true`` overrides for the case the operator
    means it.

    Identifications and configuration epochs stay, under the old name. See
    :func:`services.api.db.forget_station`.
    """
    conn = request.app.state.db
    if station not in db.stations(conn):
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"no health reports or commands for {station!r}")

    latest = db.latest_health(conn, station)
    age_s = _age_seconds(latest["received_at"]) if latest else None
    if not force and age_s is not None and age_s <= acquisition.STALE_AFTER_S:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{station} reported {age_s:.0f}s ago, so an agent is still "
            f"pushing under that name and the panel would return on its next "
            f"push. Change `station` in that agent's config first. Pass "
            f"force=true to delete the history anyway.")

    removed = db.forget_station(conn, station)
    kept = removed["kept"]
    return {
        "ok": True, "station": station, **removed,
        "note": (f"{removed['health_reports']} report(s) and "
                 f"{removed['commands']} command(s) removed. "
                 f"{kept['transmitters']} identification(s) and "
                 f"{kept['config_epochs']} config epoch(s) were kept under "
                 f"this name -- they are provenance, not console furniture."),
    }


@router.post("/stations/{station}/schedule")
def set_schedule(station: str, request: Request,
                 payload: dict = Body(...),
                 _: str = Depends(require_control)) -> dict:
    """Sound these verified transmitters, and nothing else.

    ``architecture.md`` sec. 4.3. The body names transmitters, not numbers:
    the numbers are the ones that were verified, and re-typing them at the
    moment of scheduling is how a schedule comes to disagree with the record
    it was supposed to come from.

    One MPI rank group per transmitter, matching upstream's own arrangement.
    The station's launcher must start ``calc_ionograms.py`` with that many
    ranks -- the agent checks it against the launcher and refuses a mismatch,
    because too few ranks means the transmitters past the cut are never
    sounded and nothing says so.
    """
    codes = payload.get("codes")
    if not isinstance(codes, list) or not codes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "`codes` must name at least one verified "
                            "transmitter; see POST /stations/{id}/transmitters")

    conn = request.app.state.db
    records, unknown = [], []
    for code in codes:
        found = db.transmitter(conn, station, str(code).strip())
        (records.append(found) if found else unknown.append(str(code)))
    if unknown:
        known = [t["code"] for t in db.transmitters(conn, station)]
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{station} has no verified transmitter named "
            f"{', '.join(unknown)}. Identified so far: "
            f"{', '.join(known) if known else 'none'}.")

    groups = acquisition.compose(records)
    faults = acquisition.problems(groups)
    if faults:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "; ".join(faults))

    mode = str(payload.get("mode") or "scheduled").lower()
    changes = {"mode": mode, "sounder_timings": json.dumps(groups)}
    command_id = db.enqueue(conn, station, "set_config",
                            params=_vet_config({"changes": changes}),
                            issued_by=str(payload.get("issued_by") or "web"))
    return {
        "ok": True, "id": command_id, "station": station, "mode": mode,
        "ranks": len(groups),
        "entries": sum(len(g) for g in groups),
        "transmitters": [r["code"] for r in records],
        "sounder_timings": groups,
        "note": (f"queued; the station applies it on its next pull and it "
                 f"takes effect on restart. calc_ionograms.py must be started "
                 f"with -np {len(groups)}."),
    }


# --------------------------------------------------------------------------
# Forecasting models
# --------------------------------------------------------------------------

@router.post("/models/{model_id}/activate")
def activate_model(model_id: int, request: Request,
                   who: str = Depends(require_control)) -> dict:
    """Make a model the live forecast for its circuit.

    **Control scope, not read scope.** Nothing here touches a radio, so the
    obvious argument is that read scope would do -- and it would not. This
    changes what every consumer of ``GET /forecast`` receives from the next
    request onwards, silently and with no other signal that it happened. That
    is an operational change to a published product, and it belongs behind the
    same token as the other operational changes.

    The rules it can fail on live in the schema, not here: a model fitted
    against a modelled target, or bound to no circuit, is refused by a CHECK
    constraint that a direct SQL session could not talk its way past either.
    This route's job is to turn that refusal into a sentence.
    """
    from ..prediction import registry

    try:
        result = registry.activate(request.app.state.db, model_id, by=who)
    except registry.RegistryError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    activated = result["activated"]
    deactivated = result["deactivated"]
    return {
        "ok": True,
        "activated": {"id": activated["id"], "name": activated["name"],
                      "param": activated["param"],
                      "tx": activated["tx"], "rx": activated["rx"]},
        "deactivated": ({"id": deactivated["id"], "name": deactivated["name"]}
                        if deactivated else None),
        "detail": (f"{activated['name']} is now the {activated['param']} "
                   f"forecast for {activated['tx']} -> {activated['rx']}"
                   + (f", replacing {deactivated['name']}" if deactivated else "")),
    }


@router.post("/models/{model_id}/retire")
def retire_model(model_id: int, request: Request,
                 who: str = Depends(require_control)) -> dict:
    """Take a model out of service, keeping its row and its forecasts.

    Retiring is not deleting. The forecasts it issued are what its scores were
    computed from, and re-activating it is how a promotion is rolled back --
    both need the history to still be there.
    """
    from ..prediction import registry

    row = registry.retire(request.app.state.db, model_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no model {model_id}")
    return {"ok": True, "retired": {"id": row["id"], "name": row["name"]},
            "detail": f"{row['name']} is no longer the live forecast; its "
                      f"rows and its forecasts are kept."}


# --------------------------------------------------------------------------
# Adding a model from the console
# --------------------------------------------------------------------------
#
# **This process takes the bytes and never opens them.** A `.sav` is a joblib
# pickle, and loading one runs code out of the file;
# `services/prediction/importer.py` refused an inbound route for exactly that
# reason and the reason still holds. What changed is that the surface is split
# rather than opened: the api hashes an upload, checks four magic bytes, writes
# it to a quarantine volume and records a row. A worker with no listening
# socket -- `services/prediction/registrar.py` -- is what loads it.
#
# The same split covers training. `POST /models/train` writes a row; the fit
# happens in `services/prediction/trainer.py`, in a container that answers no
# requests. Nothing under `services/api/` unpickles or fits, and
# `tests/test_prediction_upload.py` greps this tree to keep it that way.

#: Where quarantined uploads land. Its own volume, not the models one: the api
#: has to be able to write here, and it must never be able to write into the
#: directory the inference service runs code out of.
UPLOAD_DIR = Path(os.environ.get("MODEL_UPLOADS", "/uploads"))

#: A ceiling, because the body is streamed and an unbounded stream fills a
#: volume. 64 MiB clears every artifact in the archive by a wide margin -- the
#: legacy `.sav` files are 2-7 KB and the one Keras model is 11.9 MB.
MAX_UPLOAD_BYTES = int(os.environ.get("MODEL_UPLOAD_MAX_BYTES", 64 * 1024 * 1024))

#: How much of the stream is read at a time.
CHUNK = 1 << 20

#: Starlette renamed this constant mid-series -- `HTTP_413_REQUEST_ENTITY_TOO_LARGE`
#: became `HTTP_413_CONTENT_TOO_LARGE` and the old spelling now warns on every
#: use. The number is the part that has not changed in twenty-five years.
TOO_LARGE = 413


@router.post("/models/upload")
async def upload_model(request: Request,
                       filename: str = Query(..., min_length=1),
                       param: str = Query(..., pattern="^(muf|lof)$"),
                       tx: str | None = None, rx: str | None = None,
                       name: str | None = None,
                       origin: str = Query("imported",
                                           pattern="^(legacy|trained|imported)$"),
                       target_src: str | None = Query(
                           None, pattern="^(measured|modelled)$"),
                       period: int | None = None,
                       note: str | None = None,
                       who: str = Depends(require_control)) -> dict:
    """Quarantine an uploaded artifact and queue it for registration.

    The file arrives as the **raw request body**, not as multipart: a browser
    `fetch(url, {body: fileObject})` sends it that way, and it keeps
    `python-multipart` out of an image whose dependency list is argued for line
    by line. The metadata rides in the query string, where a circuit code and a
    filename are unremarkable.

    Refusals happen before anything is stored, and both are cheap:

    * a body that does not begin with a pickle or zip magic is **415**. This is
      a format check, not a safety check -- nothing here would make an
      arbitrary pickle safe -- but it turns "an operator picked the wrong file"
      into a sentence instead of a worker-log traceback ten seconds later.
    * a body over `MAX_UPLOAD_BYTES` is **413**, with the partial file removed.

    What lands is `<quarantine>/<sha256>`, so two uploads of the same bytes
    converge on one blob rather than racing.
    """
    from ..prediction import artifacts, queues

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"the upload quarantine {UPLOAD_DIR} is not writable ({exc}). "
            f"Mount a volume there for the api, as deploy/docker-compose.yml "
            f"does.") from exc

    digest = hashlib.sha256()
    total = 0
    head = b""
    handle, temporary = tempfile.mkstemp(dir=UPLOAD_DIR, prefix=".incoming-")
    temporary_path = Path(temporary)

    try:
        with os.fdopen(handle, "wb") as writer:
            async for chunk in request.stream():
                if not chunk:
                    continue
                if not head:
                    head = chunk[:4]
                    if not (head.startswith(artifacts.PICKLE_MAGIC)
                            or head.startswith(artifacts.ZIP_MAGIC)):
                        raise HTTPException(
                            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"{filename} does not begin like a model artifact "
                            f"(first bytes: {head!r}). Expected a joblib "
                            f"pickle (.sav, .joblib) or a Keras .keras file.")
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        TOO_LARGE,
                        f"{filename} is larger than the {MAX_UPLOAD_BYTES} "
                        f"byte limit. Raise MODEL_UPLOAD_MAX_BYTES if this is "
                        f"a model rather than a mistake.")
                digest.update(chunk)
                writer.write(chunk)
    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"could not write the upload: {exc}") from exc

    if not total:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "the request body was empty; there is no file in it.")

    sha = digest.hexdigest()
    os.replace(temporary_path, UPLOAD_DIR / sha)

    row = queues.add_upload(
        request.app.state.db, by=who, filename=filename, sha256=sha,
        bytes=total, name=name, param=param, tx=tx, rx=rx, origin=origin,
        target_src=target_src, period=period, note=note)

    return {
        "ok": True, "upload": row,
        "detail": (f"{filename} accepted ({total} bytes, sha256 {sha[:12]}). "
                   f"It is queued for registration; nothing has loaded it yet."),
    }


@router.delete("/models/uploads/{upload_id}")
def forget_upload(upload_id: int, request: Request,
                  _: str = Depends(require_control)) -> dict:
    """Drop a queued or refused upload, and its quarantined bytes with it."""
    from ..prediction import queues

    conn = request.app.state.db
    row = queues.delete_upload(conn, upload_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no upload {upload_id}")

    reaped = False
    if not queues.blob_is_referenced(conn, row["sha256"]):
        blob = UPLOAD_DIR / row["sha256"]
        if blob.exists():
            blob.unlink()
            reaped = True

    return {"ok": True, "forgotten": {"id": upload_id,
                                      "filename": row["filename"]},
            "reaped": reaped,
            "detail": f"{row['filename']} is no longer queued."
                      + ("" if reaped else " Its bytes are still referenced by "
                                           "another upload row.")}


@router.post("/models/train")
def queue_training(request: Request, spec: dict = Body(...),
                   who: str = Depends(require_control)) -> dict:
    """Queue a training run. The fit happens in the trainer, not here.

    Validated at the door rather than in the worker, because a job that is
    going to be refused should be refused while the operator is still looking
    at the form. What cannot be checked here -- whether the circuit has enough
    history for the requested lead -- is arithmetic the trainer does against
    the tracked series, and it reports it as a sentence.
    """
    from ..prediction import queues, train

    conn = request.app.state.db
    try:
        plan = train.vet(spec)
    except train.TrainError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    row = queues.add_job(conn, param=plan["param"], tx=plan["tx"],
                         rx=plan["rx"], method=plan["method"],
                         spec=plan["spec"], by=who)
    return {"ok": True, "job": row,
            "detail": (f"queued: {plan['estimator']} for {plan['param']} on "
                       f"{plan['tx']} -> {plan['rx']} at "
                       f"{plan['lead_h']:g} h lead. The trainer picks it up "
                       f"within a minute.")}


@router.delete("/models/jobs/{job_id}")
def cancel_training(job_id: int, request: Request,
                    _: str = Depends(require_control)) -> dict:
    from ..prediction import queues

    try:
        row = queues.cancel_job(request.app.state.db, job_id)
    except queues.QueueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"ok": True, "cancelled": {"id": row["id"]},
            "detail": f"training job {job_id} cancelled."}
