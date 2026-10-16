"""Routes of the live view: the agent posts, browsers watch.

The post takes the agent capability, like every other path a station speaks:
the live view is what an operator trusts to say the receiver is working, so a
forged column is as bad as a forged health report. Watching is a read.

The stream is Server-Sent Events rather than a WebSocket. It only ever flows
one way, it survives the proxies and the SSH tunnel this console is reached
through without configuration, and the browser half is forty lines with no
library. See ``services/api/live.py`` for what flows.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from . import db, live
from .auth import Capability, Principal, require, require_read
from .web_routes import templates

router = APIRouter(tags=["live"])

#: Events one batch may carry. The agent sends a handful every two seconds and
#: holds at most a few thousand through an outage; anything past this is not
#: an agent catching up.
MAX_EVENTS = 5000


@router.post("/stations/{station}/live")
async def push_live(station: str, request: Request,
                    _: Principal = Depends(require(Capability.AGENT))) -> dict:
    payload = await request.json()
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "expected {\"events\": [...]}")
    if len(events) > MAX_EVENTS:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"{len(events)} events in one batch, more than {MAX_EVENTS}")
    try:
        accepted = request.app.state.live.publish(station, events)
    except live.Refused as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return {"ok": True, "accepted": accepted,
            "queued": db.count_pending(request.app.state.db, station)}


@router.get("/stations/{station}/live", dependencies=[Depends(require_read)])
def live_snapshot(station: str, request: Request) -> dict:
    return request.app.state.live.snapshot(station)


@router.get("/stations/{station}/live/events", dependencies=[Depends(require_read)])
def live_events(station: str, request: Request) -> StreamingResponse:
    return StreamingResponse(
        live.stream(request.app.state.live, station, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/ui/live", include_in_schema=False, dependencies=[Depends(require_read)])
def live_page(request: Request, station: str | None = None):
    hub = request.app.state.live
    known = sorted(set(hub.boards) | set(db.stations(request.app.state.db)))
    chosen = station or next(iter(sorted(hub.boards)), None) or next(iter(known), "")
    return templates.TemplateResponse(request, "live.html", {
        "station": chosen, "stations": known})
