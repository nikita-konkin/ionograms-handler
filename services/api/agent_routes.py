"""The three endpoints the station agent already speaks.

**The paths here match ``services/agent/client.py``, not
``architecture.md`` sec. 4.3.** The doc proposed ``POST /health/report``; the
agent that exists and is tested posts to ``/stations/health`` and pulls from
``/stations/{id}/commands``. The agent is the deployed half and the awkward one
to change -- it lives on an acquisition laptop reached over AnyDesk -- so the
server serves what the client speaks and sec. 4.3 has been corrected to match.

All three require the control scope. A health push is not a read: it is a
station identifying itself, and accepting anonymous pushes would let anyone
fabricate a healthy report for a station that is down -- which under sec. 5.4,
where silence is the alert, is precisely how you would hide an outage.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from . import db
from .auth import require_control

router = APIRouter(tags=["agent"])


@router.post("/stations/health")
async def push_health(request: Request,
                      _: str = Depends(require_control)) -> dict:
    """Receive one health document.

    The station name comes from the document, not from the URL, because that
    is what the agent sends. Rejecting a nameless document rather than filing
    it under "" keeps an unattributable report out of the station list.
    """
    document = await request.json()
    if not isinstance(document, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "expected a JSON object")
    station = str(document.get("station") or "").strip()
    if not station:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "document has no `station`")

    report_id = db.store_health(request.app.state.db, station, document)
    return {"ok": True, "report_id": report_id}


@router.get("/stations/{station}/commands")
def pull_commands(station: str, request: Request,
                  _: str = Depends(require_control)) -> dict:
    """Hand out pending work. An empty list is the normal answer."""
    pending = db.take_pending(request.app.state.db, station)
    return {"commands": [
        {"id": c["id"], "name": c["name"],
         "params": _json(c["params"])} for c in pending]}


@router.post("/stations/{station}/commands/{command_id}/ack")
async def acknowledge(station: str, command_id: str, request: Request,
                      _: str = Depends(require_control)) -> dict:
    """Record what a command did.

    Accepted even for an unknown id. The agent has already acted by the time it
    acknowledges, so refusing the ack would make it retry an action that
    already happened -- and a retried "restart acquisition" is worse than the
    bookkeeping gap it would fix.
    """
    payload = await request.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    known = db.acknowledge(request.app.state.db, command_id,
                           results if isinstance(results, list) else [])
    return {"ok": True, "known": known}


def _json(text: str | None) -> Any:
    import json

    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
