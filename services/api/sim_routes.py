"""The simulator's routes: a sounding's console, and the circuits followed live.

Building a console runs the simulator for about half a minute of CPU, and
following a circuit runs it on every sounding the circuit sends, so both
take the archive capability, as a rescan does; reading what is built, or a
circuit's live events, is a read like any other. See :mod:`services.api.sim`
for the queue and cache, and :mod:`services.api.sim_live` for the loop.

Scoring a period of the archive (:mod:`services.api.sim_period`) runs for
minutes to hours, and takes the archive capability too. Its routes come
first: declared after ``/sim/{sounding_id}``, ``/sim/periods`` would be read
as a sounding's id.
"""

from __future__ import annotations

import datetime as _dt
import json

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, Response

from . import sim, sim_live, sim_period
from .auth import Capability, Principal, require, require_read
from .read_routes import _sounding_path

router = APIRouter(tags=["sim"])


def _day(payload: dict, key: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(str(payload.get(key)))
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"the {key} day is {payload.get(key)!r}, not a "
                            f"YYYY-MM-DD") from None


@router.get("/sim/periods", dependencies=[Depends(require_read)])
def period_runs(request: Request) -> list[dict]:
    """Every period scored or being scored, newest first."""
    return sim_period.runs(request.app.state.sim_cache)


@router.post("/sim/periods")
def period_run(request: Request, payload: dict = Body(...),
               _: Principal = Depends(require(Capability.ARCHIVE))) -> dict:
    """Score the persistence forecast over a circuit's soundings from ``first``
    to ``last``, at ``lead_s`` or the soundings' interval."""
    try:
        lead = payload.get("lead_s")
        lead = None if lead in (None, "") else float(lead)
    except (TypeError, ValueError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"the lead is {payload.get('lead_s')!r}, not a number "
                            f"of seconds") from None
    try:
        return sim_period.submit(request.app.state.db, request.app.state.sim_cache,
                                 request.app.state.archive_root,
                                 tx=str(payload.get("tx")), rx=str(payload.get("rx")),
                                 first=_day(payload, "first"),
                                 last=_day(payload, "last"), lead_s=lead)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    except OverflowError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from None


@router.get("/sim/periods/{run_id}", dependencies=[Depends(require_read)])
def period_status(run_id: int, request: Request) -> dict:
    """What a period run was asked to do, and where it stands."""
    try:
        return sim_period.status(request.app.state.sim_cache, run_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None


@router.get("/sim/periods/{run_id}/{name}", dependencies=[Depends(require_read)])
def period_file(run_id: int, name: str, request: Request) -> FileResponse:
    """One of a finished period run's files: its summary, or its rows."""
    try:
        path = sim_period.download(request.app.state.sim_cache, run_id, name)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    return FileResponse(path, filename=f"period-{run_id}-{name}",
                        media_type="application/json" if name.endswith(".json")
                        else "application/x-ndjson")


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
