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

from fastapi import APIRouter, Depends, Query, Request
from fastapi.templating import Jinja2Templates

from muf.reference import indices

from . import acquisition, db
from . import net as net_mod
from . import sources as sources_mod
from .auth import require_read
from .read_routes import _age_seconds, _command, _tri

router = APIRouter(include_in_schema=False, dependencies=[Depends(require_read)])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

#: Age past which a station is shown as stale rather than as whatever it last
#: said. Defined in `acquisition` because the acquiring/stopped indicator
#: turns on the same threshold: a report too old to show is too old to
#: conclude anything from.
STALE_AFTER_S = acquisition.STALE_AFTER_S


def _duration(seconds) -> str:
    """Seconds as something readable at a glance, with a sign for the past.

    An operator reading "is it working this minute" should not have to divide
    by sixty. ``None`` prints as an em dash rather than as zero, because "not
    known" and "now" are the two answers this must never confuse.
    """
    if seconds is None:
        return "—"
    sign, value = ("-", -seconds) if seconds < 0 else ("", seconds)
    if value < 90:
        return f"{sign}{value:.0f}s"
    if value < 5400:
        return f"{sign}{int(value) // 60}m{int(value) % 60:02d}s"
    if value < 172800:
        return f"{sign}{int(value) // 3600}h{(int(value) % 3600) // 60:02d}m"
    return f"{sign}{value / 86400:.1f}d"


templates.env.filters["duration"] = _duration


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
            # What it is sounding this minute, which is the question the unit
            # states cannot answer: every process can be active while the
            # schedule points at a transmitter that stopped months ago.
            "acquisition": acquisition.current(conn, station),
        })

    counts = db.one(conn, "SELECT COUNT(*) AS n FROM sounding") or {"n": 0}
    methods = db.rows(conn, "SELECT method, COUNT(*) AS n,"
                            " SUM(CASE WHEN muf IS NOT NULL THEN 1 ELSE 0 END)"
                            " AS picks FROM extraction GROUP BY method"
                            " ORDER BY method")
    return templates.TemplateResponse(request, "console.html", {
        "stations": stations, "n_soundings": counts["n"], "methods": methods,
        "stale_after": STALE_AFTER_S,
        # Read, never probed: `current` returns the last background reading
        # without blocking, so a page load costs nothing even with every index
        # host unreachable.
        "net": net_mod.current(),
        "sources": {s.key: s for s in indices.SOURCES},
    })


@router.get("/ui/series")
def series(request: Request, method: str = "algo",
           circuit: str | None = None,
           start: str | None = Query(None, alias="from"),
           end: str | None = Query(None, alias="to")):
    """MUF against time, for one circuit unless told otherwise.

    ``from``/``to`` are spelled and bounded exactly as ``/series/muf`` spells
    them -- inclusive both ends, aliased off ``start``/``end`` because ``from``
    is a Python keyword. Two routes over the same table answering differently
    to the same query string would be worse than having no filter at all.
    Both go through :func:`db.time_bound`, which is what makes a bare date and
    an ISO timestamp mean what they look like.

    **A circuit is the unit, not the receiver.** MUF is a property of a path:
    its length sets the obliquity, its midpoint sets the local time at the
    reflection, and the recorder sets the band ceiling. Two circuits drawn on
    one axis produce a curve that measures neither, so ``circuit`` defaults to
    whichever has the most picks rather than to all of them. ``circuit=all``
    overlays them, coloured and named, for when comparing is the point.

    Without any window the axis spans everything ingested, and an archive of a
    few days months apart draws them as vertical stripes with nothing legible
    between; ``days`` gives the template one link per day that has picks.
    """
    conn = request.app.state.db
    methods = [r["method"] for r in
               db.rows(conn, "SELECT DISTINCT method FROM extraction ORDER BY method")]

    # Circuits carrying picks *for this method*, commonest first. A circuit
    # every estimator missed -- SGO -> DOB is 381 soundings and no picks at
    # all -- would otherwise be offered as a choice that silently draws
    # nothing.
    circuits = [f"{r['tx']} -> {r['rx']}" for r in db.rows(
        conn, "SELECT s.tx AS tx, s.rx AS rx, COUNT(*) AS n"
              " FROM extraction e JOIN sounding s ON s.id = e.sounding_id"
              " WHERE e.method = ? AND e.muf IS NOT NULL"
              " GROUP BY s.tx, s.rx ORDER BY n DESC", (method,))]
    if circuit not in ("all", *circuits):
        circuit = circuits[0] if circuits else "all"

    days = [r["day"] for r in db.rows(
        conn, "SELECT DISTINCT substr(s.datetime, 1, 10) AS day"
              " FROM extraction e JOIN sounding s ON s.id = e.sounding_id"
              " WHERE e.muf IS NOT NULL ORDER BY day")]

    sql = ["SELECT s.id, s.datetime, s.tx, s.rx, s.path_km,"
           " e.muf, e.lof, e.limited, e.loflim",
           "FROM extraction e JOIN sounding s ON s.id = e.sounding_id",
           "WHERE e.method = ? AND e.muf IS NOT NULL"]
    params: list = [method]
    if circuit != "all":
        tx, _, rx = circuit.partition(" -> ")
        sql.append("AND s.tx = ? AND s.rx = ?")
        params += [tx, rx]
    if start:
        sql.append("AND s.datetime >= ?")
        params.append(db.time_bound(start))
    if end:
        sql.append("AND s.datetime <= ?")
        params.append(db.time_bound(end, end=True))
    sql.append("ORDER BY s.datetime LIMIT 5000")

    points = db.rows(conn, " ".join(sql), tuple(params))
    return templates.TemplateResponse(request, "series.html", {
        "method": method, "methods": methods, "points": points,
        "days": days, "start": start or "", "end": end or "",
        "circuit": circuit, "circuits": circuits,
    })


@router.get("/ui/sources")
def sources_page(request: Request,
                 max_days: int = sources_mod.DEFAULT_MAX_DAYS,
                 min_count: int = sources_mod.DEFAULT_MIN_COUNT):
    """Transmitters heard, and the schedule they would become.

    The page exists because `control.py` refuses to leave search mode without
    a `sounder_timings` list, and until now the only way to get one was to run
    `muf detect` on the station and transcribe the numbers.
    """
    conn = request.app.state.db
    census = sources_mod.census(request.app.state.archive_root,
                                max_days=max_days, min_count=min_count)
    known = db.stations(conn)
    return templates.TemplateResponse(request, "sources.html", {
        "census": census, "max_days": max_days, "min_count": min_count,
        # Rendered into the page rather than fetched, so the verified list
        # survives a READ_TOKEN being set: the page is already authorised, and
        # a second fetch would need a token this page has no field for.
        "verified": {s: db.transmitters(conn, s) for s in known},
        # `control.MODES` holds four keys for two modes -- "serendipitous" and
        # "schedule" are the ini's own vocabulary, kept so a command written
        # by hand in either dialect works. Offering all four as choices would
        # suggest four modes, so the page shows the canonical pair.
        "modes": ("search", "scheduled"),
        "stations": known,
        # The page decides which census row is already someone, and the server
        # decides which verified entry a slot belongs to. Both are the same
        # judgement, so both use the same tolerances -- sent rather than
        # duplicated, because a page that drifts from the server would mark a
        # row identified that the schedule does not.
        "match": {"rate_hz": acquisition.MATCH_RATE_HZ,
                  "slot_s": acquisition.MATCH_SLOT_S},
    })


#: Sortable columns, by the name that appears in the query string.
#:
#: A whitelist rather than interpolation: `sort` reaches SQL as an identifier
#: and cannot be a bound parameter, so the only safe version of this is a
#: fixed map from a name the caller may send to an expression we wrote.
SOUNDING_SORTS = {
    "time": "s.datetime",
    "tx": "s.tx",
    "rx": "s.rx",
    "format": "s.format",
    "sweep": "s.freq_stop",
    "complete": "s.sweep_fraction",
    "picks": "n_picks",
}


@router.get("/ui/soundings")
def soundings(request: Request, limit: int = 200, offset: int = 0,
              sort: str = "time", dir: str = "asc",
              tx: str | None = None, fmt: str | None = None,
              picks: str | None = None,
              start: str | None = Query(None, alias="from"),
              end: str | None = Query(None, alias="to")):
    """The sounding table, sorted and filtered.

    ``picks`` is the one filter worth having that is not a column: ``some``
    and ``none`` split the table on whether any estimator found anything,
    which is how you find the soundings worth looking at in an archive where
    most of them are noise.

    Filters apply before the limit, so the pager walks the filtered set rather
    than paging through the whole table and hiding rows -- otherwise a page
    could legitimately come back empty with more matches further on.
    """
    conn = request.app.state.db

    order = SOUNDING_SORTS.get(sort, SOUNDING_SORTS["time"])
    descending = dir == "desc"
    where, params = ["1 = 1"], []
    for column, value in (("s.tx", tx), ("s.format", fmt)):
        if value:
            where.append(f"AND {column} = ?")
            params.append(value)
    if start:
        where.append("AND s.datetime >= ?")
        params.append(db.time_bound(start))
    if end:
        where.append("AND s.datetime <= ?")
        params.append(db.time_bound(end, end=True))
    # HAVING, not WHERE: n_picks is an aggregate over the joined extractions.
    having = ""
    if picks == "some":
        having = " HAVING n_picks > 0"
    elif picks == "none":
        having = " HAVING n_picks = 0"

    base = ("SELECT s.*, COUNT(e.method) AS n_methods,"
            " SUM(CASE WHEN e.muf IS NOT NULL THEN 1 ELSE 0 END) AS n_picks"
            " FROM sounding s LEFT JOIN extraction e ON e.sounding_id = s.id"
            " WHERE " + " ".join(where) + " GROUP BY s.id" + having)

    rows = db.rows(conn, f"{base} ORDER BY {order} {'DESC' if descending else 'ASC'},"
                         " s.id LIMIT ? OFFSET ?", (*params, limit, offset))
    total = (db.one(conn, f"SELECT COUNT(*) AS n FROM ({base})", tuple(params))
             or {"n": 0})["n"]

    facets = {
        "tx": [r["v"] for r in db.rows(
            conn, "SELECT DISTINCT tx AS v FROM sounding"
                  " WHERE tx IS NOT NULL ORDER BY v")],
        "format": [r["v"] for r in db.rows(
            conn, "SELECT DISTINCT format AS v FROM sounding"
                  " WHERE format IS NOT NULL ORDER BY v")],
    }
    return templates.TemplateResponse(request, "soundings.html", {
        "soundings": rows, "limit": limit, "offset": offset, "total": total,
        "sort": sort, "dir": dir, "facets": facets,
        "tx": tx or "", "fmt": fmt or "", "picks": picks or "",
        "start": start or "", "end": end or "",
    })


@router.get("/ui/sounding/{sounding_id}")
def sounding(request: Request, sounding_id: int, gate: str = "auto"):
    conn = request.app.state.db
    row = db.one(conn, "SELECT * FROM sounding WHERE id = ?", (sounding_id,))
    extractions = db.rows(conn, "SELECT * FROM extraction WHERE sounding_id = ?"
                                " ORDER BY method", (sounding_id,))

    # Neighbours in time, not in id. Ids are ingest order, which matches time
    # only until a day is back-filled or re-ingested -- and stepping through a
    # day is the whole point of having these. `id` breaks the tie so two
    # soundings sharing a timestamp still order deterministically, and so
    # next(prev(x)) is x rather than a cycle between them.
    neighbours = {"prev": None, "next": None}
    if row is not None:
        for name, cmp, order in (("prev", "<", "DESC"), ("next", ">", "ASC")):
            found = db.one(
                conn,
                "SELECT id, datetime FROM sounding"
                f" WHERE datetime {cmp} ? OR (datetime = ? AND id {cmp} ?)"
                f" ORDER BY datetime {order}, id {order} LIMIT 1",
                (row["datetime"], row["datetime"], sounding_id))
            neighbours[name] = found

    return templates.TemplateResponse(request, "sounding.html", {
        "sounding": row, "extractions": extractions, "gate": gate,
        "prev": neighbours["prev"], "next": neighbours["next"],
    })
