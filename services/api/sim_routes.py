"""The simulator's routes: a sounding's console, and the circuits followed live.

Building a console runs the simulator for about half a minute of CPU, and
following a circuit runs it on every sounding the circuit sends, so both
take the archive capability, as a rescan does; reading what is built, or a
circuit's live events, is a read like any other. See :mod:`services.api.sim`
for the queue and cache, and :mod:`services.api.sim_live` for the loop.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, Response

from . import sim, sim_live
from .auth import Capability, Principal, require, require_read
from .read_routes import _sounding_path

router = APIRouter(tags=["sim"])


@router.get("/sim/{sounding_id}", dependencies=[Depends(require_read)])
def sim_status(sounding_id: int, request: Request) -> dict:
    """Where the sounding's console stands."""
    return sim.status(request.app.state.sim_cache, sounding_id)


@router.post("/sim/{sounding_id}")
def sim_run(sounding_id: int, request: Request,
            _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Build the sounding's console, unless it is built or on its way."""
    path = _sounding_path(request, sounding_id)
    try:
        return sim.run(request.app.state.sim_cache, sounding_id, path)
    except OverflowError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from None


@router.post("/sim/live/{tx}/{rx}")
def live_follow(tx: str, rx: str, request: Request,
                _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Follow a circuit live, from its newest sounding on."""
    try:
        return sim_live.follow(request.app.state.db, tx, rx)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None


@router.delete("/sim/live/{tx}/{rx}")
def live_stop(tx: str, rx: str, request: Request,
              _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Stop following a circuit live; its events are kept."""
    return sim_live.stop(request.app.state.db, tx, rx)


@router.get("/sim/live/{tx}/{rx}/events", dependencies=[Depends(require_read)])
def live_events(tx: str, rx: str, request: Request, after: int = 0,
                limit: int = Query(sim_live.PAGE, ge=1, le=sim_live.PAGE_MAX),
                ) -> Response:
    """The circuit's live events after event ``after``, oldest first, as JSON Lines."""
    return Response("".join(json.dumps(e) + "\n"
                            for e in sim_live.events(request.app.state.db, tx, rx,
                                                     after=after, limit=limit)),
                    media_type="application/x-ndjson")


@router.get("/sim/{sounding_id}/console", response_class=HTMLResponse,
            include_in_schema=False, dependencies=[Depends(require_read)])
def sim_console(sounding_id: int, request: Request) -> HTMLResponse:
    """The console page for a sounding already built."""
    built = sim.data(request.app.state.sim_cache, sounding_id)
    if built is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"the console of sounding {sounding_id} is not built")
    return HTMLResponse(sim.page(built))
