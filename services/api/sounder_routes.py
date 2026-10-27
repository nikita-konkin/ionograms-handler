"""The ionozond schedule on the live console: read it, check an edit, seed a group.

Queueing the edit is ``POST /stations/{id}/commands`` with ``set_ionozond``,
like every other command; see ``control_routes``. These three routes are what
the schedule panel on ``/ui/live`` needs around that: the schedule in force
with its next captures, a check of an edit that queues nothing, so the
operator sees collisions while still typing, and a verified slot converted
into a group for the edit (phase 4, issue #8).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from services.agent import ionozond

from . import db, sounder
from .auth import Capability, Principal, require, require_read

router = APIRouter(tags=["ionozond"])


@router.get("/stations/{station}/ionozond", dependencies=[Depends(require_read)])
def schedule(station: str, request: Request) -> dict:
    conn = request.app.state.db
    row, config = sounder.latest(conn, station)
    if not config:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"{station} reports no ionozond schedule")
    document, epoch, now = (config.get("document"), db.open_epoch(conn, station),
                            time.time())
    return {"station": station, "reported_at": row["received_at"],
            "sounder_config": config,
            "next": (ionozond.captures(document, now, sounder.AHEAD)
                     if isinstance(document, dict) else []),
            "epoch": epoch,
            "tally": sounder.tally(conn, station, config, epoch, now)}


@router.post("/stations/{station}/ionozond/check")
def check(station: str, request: Request, params: dict = Body(...),
          _: Principal = Depends(require(Capability.STATION))) -> dict:
    try:
        result = sounder.checked(request.app.state.db, station, params)
    except sounder.Unvetted as exc:
        raise HTTPException(exc.status, str(exc)) from None
    wanted = result.pop("wanted")
    return dict(result, next=ionozond.captures(wanted, time.time(), sounder.AHEAD))


@router.post("/stations/{station}/ionozond/seed")
def seed(station: str, request: Request, params: dict = Body(...),
         _: Principal = Depends(require(Capability.STATION))) -> dict:
    try:
        return sounder.seeded(request.app.state.db, station, params)
    except sounder.Unvetted as exc:
        raise HTTPException(exc.status, str(exc)) from None
