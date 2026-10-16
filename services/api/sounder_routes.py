"""The ionozond schedule on the live console: read it, and check an edit.

Queueing the edit is ``POST /stations/{id}/commands`` with ``set_ionozond``,
like every other command; see ``control_routes``. These two routes are what
the schedule panel on ``/ui/live`` needs around that: the schedule in force
with its next captures, and a check of an edit that queues nothing, so the
operator sees collisions while still typing.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from services.agent import ionozond

from . import db, sounder
from .auth import Capability, Principal, require, require_read

router = APIRouter(tags=["ionozond"])


@router.get("/stations/{station}/ionozond", dependencies=[Depends(require_read)])
def schedule(station: str, request: Request) -> dict:
    conn = request.app.state.db
    row = db.latest_health(conn, station)
    config = json.loads(row["document"] or "{}").get("sounder_config") if row else None
    if not isinstance(config, dict):
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"{station} reports no ionozond schedule")
    document = config.get("document")
    return {"station": station, "reported_at": row["received_at"],
            "sounder_config": config,
            "next": (ionozond.captures(document, time.time(), sounder.AHEAD)
                     if isinstance(document, dict) else []),
            "epoch": db.open_epoch(conn, station)}


@router.post("/stations/{station}/ionozond/check")
def check(station: str, request: Request, params: dict = Body(...),
          _: Principal = Depends(require(Capability.STATION))) -> dict:
    try:
        result = sounder.checked(request.app.state.db, station, params)
    except sounder.Unvetted as exc:
        raise HTTPException(exc.status, str(exc)) from None
    wanted = result.pop("wanted")
    return dict(result, next=ionozond.captures(wanted, time.time(), sounder.AHEAD))
