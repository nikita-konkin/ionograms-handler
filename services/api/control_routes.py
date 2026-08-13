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

import json
import re

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from . import acquisition, db
from .auth import require_control

router = APIRouter(tags=["control"])

#: What the web interface may queue. A strict subset of the agent's own
#: ``ALLOWED_VERBS`` plus ``set_config``; see the module docstring.
QUEUEABLE = ("start", "stop", "restart", "set_config")

#: Settings ``set_config`` may carry from the web. A strict subset of
#: ``control.EDITABLE`` -- see the module docstring for what is left out.
WEB_EDITABLE = ("mode", "sounder_timings")


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
    return {"ok": True, "station": station, "transmitter": record}


@router.delete("/stations/{station}/transmitters/{code}")
def forget_transmitter(station: str, code: str, request: Request,
                       _: str = Depends(require_control)) -> dict:
    """Forget an identification. Products already recorded keep the name."""
    removed = db.delete_transmitter(request.app.state.db, station, code)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"{station} has no transmitter {code!r}")
    return {"ok": True, "station": station, "code": code}


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
