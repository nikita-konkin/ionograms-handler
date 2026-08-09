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

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from . import db
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
            missing = sorted({"chirp-rate", "rep", "chirpt"} - set(entry))
            if missing:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"sounder_timings entry {entry} is missing {missing}")
    return {"changes": changes}


def _parse_timings(value):
    """`sounder_timings` as a list of entries, however it arrived.

    The station's ini stores a nested list; a browser form sends a JSON string.
    Both are accepted, and anything else is not a schedule.
    """
    import json

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    # `[[{...}]]` in the ini is per-MPI-rank; flatten one level if present.
    flat = []
    for item in value:
        flat.extend(item if isinstance(item, list) else [item])
    return [e for e in flat if isinstance(e, dict)]
