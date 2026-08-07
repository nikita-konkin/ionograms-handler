"""Server-rendered pages.

Jinja templates and a little inline fetch, no build step. For a deployment
that is meant to be temporary, a JavaScript toolchain is a liability: it adds
an install, a lockfile and a second thing that can fail to start, in exchange
for interactions this interface does not have.

The pages are read-scope. The two control buttons post to
``/stations/{id}/commands``, which requires the control token -- the browser
supplies it from a field the operator pastes into, so the token is never baked
into a page that might be left open.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from . import db
from .auth import require_read
from .read_routes import _age_seconds, _command, _tri

router = APIRouter(include_in_schema=False, dependencies=[Depends(require_read)])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

#: Age past which a station is shown as stale rather than as whatever it last
#: said. Three push intervals at the agent's 60 s default: one missed push is
#: a hiccup, three is a story.
STALE_AFTER_S = 180.0


@router.get("/ui")
def console(request: Request):
    conn = request.app.state.db
    stations = []
    for station in db.stations(conn):
        latest = db.latest_health(conn, station)
        metrics = db.metrics_for(conn, int(latest["id"])) if latest else []
        age = _age_seconds(latest["received_at"]) if latest else None
        stations.append({
            "name": station,
            "age_s": age,
            "stale": age is None or age > STALE_AFTER_S,
            "healthy": _tri(latest["healthy"]) if latest else None,
            "agent_version": latest["agent_version"] if latest else None,
            "metrics": [{**m, "ok": _tri(m["ok"])} for m in metrics],
            "commands": [_command(c) for c in db.recent_commands(conn, station, 8)],
        })

    counts = db.one(conn, "SELECT COUNT(*) AS n FROM sounding") or {"n": 0}
    methods = db.rows(conn, "SELECT method, COUNT(*) AS n,"
                            " SUM(CASE WHEN muf IS NOT NULL THEN 1 ELSE 0 END)"
                            " AS picks FROM extraction GROUP BY method"
                            " ORDER BY method")
    return templates.TemplateResponse(request, "console.html", {
        "stations": stations, "n_soundings": counts["n"], "methods": methods,
        "stale_after": STALE_AFTER_S,
    })


@router.get("/ui/series")
def series(request: Request, method: str = "algo"):
    conn = request.app.state.db
    methods = [r["method"] for r in
               db.rows(conn, "SELECT DISTINCT method FROM extraction ORDER BY method")]
    points = db.rows(
        conn,
        "SELECT s.id, s.datetime, s.tx, s.rx, e.muf, e.lof, e.limited, e.loflim"
        " FROM extraction e JOIN sounding s ON s.id = e.sounding_id"
        " WHERE e.method = ? AND e.muf IS NOT NULL"
        " ORDER BY s.datetime LIMIT 5000", (method,))
    return templates.TemplateResponse(request, "series.html", {
        "method": method, "methods": methods, "points": points,
    })


@router.get("/ui/soundings")
def soundings(request: Request, limit: int = 200, offset: int = 0):
    conn = request.app.state.db
    rows = db.rows(conn,
                   "SELECT s.*, COUNT(e.method) AS n_methods,"
                   " SUM(CASE WHEN e.muf IS NOT NULL THEN 1 ELSE 0 END) AS n_picks"
                   " FROM sounding s LEFT JOIN extraction e ON e.sounding_id = s.id"
                   " GROUP BY s.id ORDER BY s.datetime LIMIT ? OFFSET ?",
                   (limit, offset))
    total = (db.one(conn, "SELECT COUNT(*) AS n FROM sounding") or {"n": 0})["n"]
    return templates.TemplateResponse(request, "soundings.html", {
        "soundings": rows, "limit": limit, "offset": offset, "total": total,
    })


@router.get("/ui/sounding/{sounding_id}")
def sounding(request: Request, sounding_id: int, gate: str = "auto"):
    conn = request.app.state.db
    row = db.one(conn, "SELECT * FROM sounding WHERE id = ?", (sounding_id,))
    extractions = db.rows(conn, "SELECT * FROM extraction WHERE sounding_id = ?"
                                " ORDER BY method", (sounding_id,))
    return templates.TemplateResponse(request, "sounding.html", {
        "sounding": row, "extractions": extractions, "gate": gate,
    })
