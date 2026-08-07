"""Queueing commands for a station.

Control scope only. This is the endpoint that can stop a radio, so two things
are true of it that are not true of anything else in the service:

**The verb is allow-listed here as well as in the agent.** ``control.py``
already refuses anything outside ``ALLOWED_VERBS``, and duplicating that check
is deliberate: the agent's list is the last line, and a server that will queue
anything means an unreachable station accumulates commands nobody vetted. Two
checks, and the outer one is the one an operator can see.

**Only process verbs are exposed for this deployment.** ``control.py`` also
implements validated parameter edits -- mode, sounder_timings, output_dir --
and they are deliberately not routed. A parameter change rewrites the
station's ``.ini`` and needs a restart to take effect; proving start/stop
first is worth more than the extra surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from . import db
from .auth import require_control

router = APIRouter(tags=["control"])

#: What the web interface may queue. A strict subset of the agent's own
#: ``ALLOWED_VERBS``; see the module docstring for why parameter edits are out.
QUEUEABLE = ("start", "stop", "restart")


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

    command_id = db.enqueue(request.app.state.db, station, name,
                            params=payload.get("params") or {},
                            issued_by=str(payload.get("issued_by") or "web"))
    return {"ok": True, "id": command_id, "name": name, "station": station,
            "note": "queued; the station will collect it on its next pull"}
