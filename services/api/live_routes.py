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

import base64
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from muf import calibrate
from muf.extractors import ALL_METHODS
from muf.stations import default_registry
from services.agent import product, schedulecheck

from . import db, live, sounder
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


#: Summaries one read may return: three days of a sounding every 300 s, and
#: some to spare for a schedule that sounds more often.
MAX_SUMMARIES = 2000

#: Whose pick the live view shows: the method forecasts are trained on, and
#: the public page's default (`PUBLISH_METHOD`).
PICK_METHOD = "contour"


@router.get("/stations/{station}/live/summaries", dependencies=[Depends(require_read)])
def live_summaries(station: str, tx: str, since: float, until: float,
                   request: Request) -> dict:
    conn = request.app.state.db
    return {"summaries": db.summaries(conn, station, tx, since, until, MAX_SUMMARIES),
            "earliest": db.earliest_summary(conn, station, tx)}


@router.get("/stations/{station}/live/days", dependencies=[Depends(require_read)])
def live_days(station: str, request: Request) -> dict:
    return {"days": db.summary_days(request.app.state.db, station)}


@router.get("/stations/{station}/live/archive", dependencies=[Depends(require_read)])
def live_archive(station: str, tx: str, request: Request, before: float | None = None,
                 after: float | None = None) -> dict:
    """The archived sounding nearest a time, drawn as the live view draws its own.

    One side of the time or the other, so the page can step back through the
    archive and forward again.
    """
    if (before is None) == (after is None):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "give exactly one of before and after")
    row = db.archived(request.app.state.db, station, tx, before=before, after=after)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "the archive holds no %s sounding %s that time" % (
                                tx, "before" if before is not None else "after"))
    path = Path(row["path"])
    if not path.is_absolute():
        path = Path(request.app.state.archive_root) / path
    try:
        head, cells = product.read(path)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "cannot read %s: %s" % (row["file"], exc)) from None
    return dict(head, tx=tx, data={str(i): base64.b64encode(column.tobytes()).decode()
                                   for i, column in enumerate(cells)})


@router.get("/stations/{station}/live/muf", dependencies=[Depends(require_read)])
def live_muf(station: str, tx: str, request: Request,
             method: str = PICK_METHOD) -> dict:
    """The pipeline's MUF and LOF for the newest archived sounding of ``tx``.

    From the archive, not the live columns: those are folded and rounded for
    the screen, and a MUF read off them would disagree with the one every
    other page shows. So it arrives once the hub has ingested the product, up
    to an ingest interval after the sounding ends.
    """
    if method not in ALL_METHODS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "method must be one of %s" % ", ".join(ALL_METHODS))
    row = db.latest_pick(request.app.state.db, station, tx, method)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "%s has not run on any %s sounding at %s"
                            % (method, tx, station))
    return {"tx": tx, "method": method, "t0": sounder._seconds_of(row["datetime"]),
            "muf": row["muf"], "lof": row["lof"],
            "limited": bool(row["limited"]), "loflim": bool(row["loflim"])}


@router.get("/stations/{station}/live/path", dependencies=[Depends(require_read)])
def live_path(station: str, tx: str, request: Request) -> dict:
    """How far the transmitter is from this station, for the ionograms' height axis.

    From the coordinates the station's schedule reports, as the sounder itself
    gates by, and from the station registry by name where the report has none.
    """
    conn = request.app.state.db
    try:
        found = sounder.reported(conn, station)
    except sounder.Unvetted:
        found = {}
    group = ((found.get("document") or {}).get("schedule") or {}).get(tx) or {}
    receivers = found.get("receivers") or [{}]
    sites = default_registry()
    known = (sites.station(name) for name in (station, *db.receivers(conn, station)))
    ends = [_place(group, sites.station(tx)),
            _place(receivers[0], next((site for site in known if site), None))]
    km = None if None in ends else calibrate.ground_range_km(SimpleNamespace(
        is_oblique=True, tx_latitude=ends[0][0], tx_longitude=ends[0][1],
        rx_latitude=ends[1][0], rx_longitude=ends[1][1]))
    if km is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "cannot place both ends of %s to %s" % (tx, station))
    return {"ground_km": km}


def _place(group: dict, site) -> tuple[float, float] | None:
    try:
        return float(group["lat"]), float(group["lon"])
    except (KeyError, TypeError, ValueError):
        return None if site is None else (site.latitude, site.longitude)


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
        "station": chosen, "stations": known,
        "fft_sizes": schedulecheck.FFT_COUNT_LIST})
