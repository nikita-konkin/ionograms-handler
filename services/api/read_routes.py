"""Read endpoints: soundings, series, health views, ionograms.

``architecture.md`` sec. 4.3, read scope only. Nothing here can change the
station.

Ionograms are rendered **on request** (sec. 4.2). 288 images a day
precomputed is 105k files a year that mostly nobody opens; rendering from the
stored product is about 0.2 s, and the file is on disk anyway.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from . import db
from . import net as net_mod
from . import sources as sources_mod
from .auth import require_read

router = APIRouter(tags=["read"], dependencies=[Depends(require_read)])

#: Soundings returned when the caller does not say. High enough to draw a day,
#: low enough that a bare `/soundings` cannot pull the whole archive into one
#: response.
DEFAULT_LIMIT = 500
MAX_LIMIT = 5000


@router.get("/stations")
def list_stations(request: Request) -> dict:
    """Every station, its latest report, and how long ago it arrived.

    ``age_s`` is the number the operator actually reads. Under sec. 5.4
    silence is the alert, so a station that stopped reporting must be visible
    as *stale*, not merely as whatever it last claimed.
    """
    conn = request.app.state.db
    out = []
    for station in db.stations(conn):
        latest = db.latest_health(conn, station)
        entry: dict = {"station": station, "last_report": None, "age_s": None,
                       "healthy": None, "failing": 0, "unknown": 0, "ok": 0}
        if latest:
            metrics = db.metrics_for(conn, int(latest["id"]))
            entry.update(
                last_report=latest["received_at"],
                age_s=_age_seconds(latest["received_at"]),
                healthy=_tri(latest["healthy"]),
                agent_version=latest["agent_version"],
                failing=sum(1 for m in metrics if m["ok"] == 0),
                unknown=sum(1 for m in metrics if m["ok"] is None),
                ok=sum(1 for m in metrics if m["ok"] == 1),
            )
        out.append(entry)
    return {"stations": out}


@router.get("/stations/{station}/health")
def station_health(station: str, request: Request) -> dict:
    """Latest report and recent commands for one station.

    A station with queued commands but no report yet is **not** a 404. It
    appears in ``/stations`` the moment a command is queued for it, so 404ing
    the detail view would make the list link to a page that denies the station
    exists -- and "commanded but never reported" is a state worth looking at,
    not one to hide. Only a name with neither reports nor commands is unknown.
    """
    conn = request.app.state.db
    latest = db.latest_health(conn, station)
    commands = [_command(c) for c in db.recent_commands(conn, station)]
    if latest is None and not commands:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"no reports and no commands for {station!r}")

    if latest is None:
        return {"station": station, "received_at": None, "age_s": None,
                "healthy": None, "agent_version": None, "metrics": [],
                "commands": commands,
                "note": "commands queued, but this station has never reported"}

    return {
        "station": station,
        "received_at": latest["received_at"],
        "age_s": _age_seconds(latest["received_at"]),
        "healthy": _tri(latest["healthy"]),
        "agent_version": latest["agent_version"],
        "metrics": [
            {**m, "ok": _tri(m["ok"])} for m in db.metrics_for(conn, int(latest["id"]))
        ],
        "commands": commands,
    }


@router.get("/stations/{station}/health/history")
def station_history(station: str, request: Request,
                    limit: int = Query(200, ge=1, le=5000)) -> dict:
    """Report-level history, for trends and for spotting gaps."""
    rows = db.rows(request.app.state.db,
                   "SELECT id, received_at, healthy FROM health_report"
                   " WHERE station = ? ORDER BY received_at DESC LIMIT ?",
                   (station, limit))
    return {"station": station,
            "history": [{"received_at": r["received_at"],
                         "healthy": _tri(r["healthy"])} for r in rows]}


@router.get("/stations/{station}/transmitters")
def station_transmitters(station: str, request: Request) -> dict:
    """Transmitters an operator has identified at this receiver.

    Read scope: knowing who is on air is not a control action, and this is the
    list a second operator needs before they can be told what the schedule
    means.
    """
    found = db.transmitters(request.app.state.db, station)
    return {"station": station, "count": len(found), "transmitters": found}


@router.get("/stations/{station}/schedule")
def station_schedule(station: str, request: Request,
                     limit: int = Query(6, ge=1, le=50)) -> dict:
    """What this station is sounding right now, and what is due next.

    ``architecture.md`` sec. 4.3. The schedule comes from the open
    ``config_epoch`` -- what the station was last *seen* configured with,
    written when it acknowledged the change -- so a station configured by hand
    reports no schedule rather than a stale one.
    """
    from . import acquisition

    return acquisition.current(request.app.state.db, station,
                               limit=limit).as_dict()


@router.get("/soundings")
def soundings(request: Request,
              tx: str | None = None, rx: str | None = None,
              start: str | None = Query(None, alias="from"),
              end: str | None = Query(None, alias="to"),
              limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)) -> dict:
    sql = ["SELECT * FROM sounding WHERE 1 = 1"]
    params: list = []
    for column, value in (("tx", tx), ("rx", rx)):
        if value:
            sql.append(f"AND {column} = ?")
            params.append(value)
    if start:
        sql.append("AND datetime >= ?")
        params.append(db.time_bound(start))
    if end:
        sql.append("AND datetime <= ?")
        params.append(db.time_bound(end, end=True))
    sql.append("ORDER BY datetime LIMIT ?")
    params.append(limit)

    found = db.rows(request.app.state.db, " ".join(sql), tuple(params))
    return {"count": len(found), "soundings": found}


@router.get("/series/muf")
def series_muf(request: Request,
               method: str = "algo",
               tx: str | None = None, rx: str | None = None,
               start: str | None = Query(None, alias="from"),
               end: str | None = Query(None, alias="to"),
               smoothed: bool = False,
               limit: int = Query(MAX_LIMIT, ge=1, le=MAX_LIMIT)) -> dict:
    """MUF against time for one estimator.

    ``limited`` travels with every point rather than being filtered out here.
    A midday lower bound is a real observation and dropping it silently would
    bias the series high; the caller decides what to do with it, having been
    told. See ``BACKLOG.md`` sec. 3.
    """
    column = "e.muf_smooth" if smoothed else "e.muf"
    sql = ["SELECT s.id, s.datetime, s.tx, s.rx,", column, "AS muf,",
           "e.lof, e.snr, e.run, e.limited, e.loflim",
           "FROM extraction e JOIN sounding s ON s.id = e.sounding_id",
           "WHERE e.method = ? AND", column, "IS NOT NULL"]
    params: list = [method]
    for name, value in (("s.tx", tx), ("s.rx", rx)):
        if value:
            sql.append(f"AND {name} = ?")
            params.append(value)
    if start:
        sql.append("AND s.datetime >= ?")
        params.append(db.time_bound(start))
    if end:
        sql.append("AND s.datetime <= ?")
        params.append(db.time_bound(end, end=True))
    sql.append("ORDER BY s.datetime LIMIT ?")
    params.append(limit)

    points = db.rows(request.app.state.db, " ".join(sql), tuple(params))
    return {"method": method, "smoothed": smoothed,
            "count": len(points), "points": points}


@router.get("/models")
def models(request: Request,
           param: str | None = None,
           tx: str | None = None, rx: str | None = None) -> dict:
    """Registered forecasting models: what exists, what is live, how it scored.

    Read scope, like every other page here. Promotion is a control-scope POST
    in ``control_routes`` -- seeing which model is live is public, changing it
    is not.
    """
    from ..prediction import registry

    rows = registry.models(request.app.state.db, param=param, tx=tx, rx=rx)
    for row in rows:
        # The stored input row is an implementation detail of the golden check
        # and is meaningless on a page; the *result* of that check is not.
        row.pop("golden_input", None)
        row["golden"] = "recorded" if row.pop("golden_output", None) is not None \
            else "absent"
    return {"count": len(rows), "models": rows}


@router.get("/forecast")
def forecast(request: Request,
             param: str = "muf",
             tx: str | None = None, rx: str | None = None,
             model: int | None = None,
             issued: str | None = None,
             start: str | None = Query(None, alias="from"),
             end: str | None = Query(None, alias="to"),
             limit: int = Query(MAX_LIMIT, ge=1, le=MAX_LIMIT)) -> dict:
    """Forecast values against time.

    **Active models only, unless one is named.** Every registered model may
    write forecasts -- that is how a comparison is run -- so an unfiltered
    query would interleave the live curve with every candidate's and with every
    legacy import's. ``?model=`` fetches one deliberately.

    Defaults to the **latest issue**: a forecast table holds every issue ever
    made, because that is what horizon scoring is computed from, and drawing
    all of them at once is never what a caller means.
    """
    sql = ["SELECT f.*, m.name AS model_name, m.origin, m.active",
           "FROM forecast f JOIN model_registry m ON m.id = f.model_id",
           "WHERE f.param = ?"]
    params: list = [param]

    if model is not None:
        sql.append("AND f.model_id = ?")
        params.append(model)
    else:
        sql.append("AND m.active = 1")

    for column, value in (("f.tx", tx), ("f.rx", rx)):
        if value:
            sql.append(f"AND {column} = ?")
            params.append(value)

    if issued:
        sql.append("AND f.issued_at = ?")
        params.append(issued)
    else:
        sql.append("AND f.issued_at = (SELECT MAX(issued_at) FROM forecast x "
                   "WHERE x.model_id = f.model_id AND x.param = f.param "
                   "AND x.tx = f.tx AND x.rx = f.rx)")

    if start:
        sql.append("AND f.valid_at >= ?")
        params.append(db.time_bound(start))
    if end:
        sql.append("AND f.valid_at <= ?")
        params.append(db.time_bound(end, end=True))

    sql.append("ORDER BY f.valid_at LIMIT ?")
    params.append(limit)

    points = db.rows(request.app.state.db, " ".join(sql), tuple(params))
    for point in points:
        if point.get("quality"):
            try:
                point["quality"] = json.loads(point["quality"])
            except json.JSONDecodeError:
                pass
    return {"param": param, "model": model, "count": len(points),
            "points": points}


@router.get("/scores")
def scores(request: Request,
           param: str = "muf",
           tx: str | None = None, rx: str | None = None,
           flat: bool = False) -> dict:
    """How every model and every baseline did, by horizon.

    **Baselines come back in the same list as the models**, marked by `kind`.
    They are what makes an MAE mean anything -- 0.94 MHz is good or useless
    depending on what yesterday's value would have scored -- so a caller cannot
    fetch the models' numbers without also being handed what they are judged
    against.

    Folded one row per subject by default, with a column per horizon; `flat=1`
    returns the stored rows instead, which is what a rescoring diff wants.
    """
    from ..prediction import scoring

    conn = request.app.state.db
    if flat:
        rows = scoring.scores(conn, param, tx, rx)
        return {"param": param, "tx": tx, "rx": rx,
                "horizons": list(scoring.HORIZONS), "count": len(rows),
                "scores": rows}

    if not (tx and rx):
        raise HTTPException(
            status_code=400,
            detail="a leaderboard is per circuit: pass tx and rx, or flat=1 "
                   "for the stored rows across all of them.")
    board = scoring.leaderboard(conn, param, tx, rx)
    return {"param": param, "tx": tx, "rx": rx,
            "horizons": list(scoring.HORIZONS), "count": len(board),
            "leaderboard": board,
            # Reported, never enforced: a service that demoted a model on its
            # own would change a published product with nothing in the logs
            # having asked for it.
            "drift": scoring.drift(board)}


@router.get("/sources")
def sources(request: Request,
            max_days: int = Query(sources_mod.DEFAULT_MAX_DAYS, ge=1, le=14),
            cycle_s: float | None = None,
            min_count: int = Query(3, ge=1, le=100)) -> dict:
    """Repeating transmitters the station has heard, newest days first.

    Read scope, because it is a census of what is on air -- but it is the list
    a schedule is built from, so each row carries the `sounder_timings` entry
    it would become. See `services.api.sources` for why the seconds are as
    received rather than as transmitted.
    """
    # Never on the request path: the scan alone costs minutes on the archive
    # this serves. Answers from the last completed census and refreshes in the
    # background -- `age_s` says how old it is, `building` that there is
    # nothing yet. See `sources.DEFAULT_MAX_AGE_S`.
    return sources_mod.census(request.app.state.archive_root,
                              max_days=max_days, cycle_s=cycle_s,
                              min_count=min_count, block=False,
                              max_age_s=sources_mod.DEFAULT_MAX_AGE_S)


@router.get("/net")
def net_state() -> dict:
    """Whether this host can still reach the solar index servers.

    Read scope. It reports three public hostnames, a round-trip time and the
    age of each cached file -- nothing about the station, and nothing a caller
    could not learn by trying the same HEADs itself.

    Always the *last* reading, never a fresh probe: a route that probed on
    demand would be a way for anyone who can reach this port to make the
    server issue outbound requests, and would put a four-second timeout on the
    request path. ``state`` is ``unknown`` until the first background pass
    finishes, a second or so after startup.
    """
    return net_mod.current().as_dict()


@router.get("/methods")
def methods(request: Request) -> dict:
    rows = db.rows(request.app.state.db,
                   "SELECT method, COUNT(*) AS n,"
                   " SUM(CASE WHEN muf IS NOT NULL THEN 1 ELSE 0 END) AS picks"
                   " FROM extraction GROUP BY method ORDER BY method")
    return {"methods": rows}


@router.get("/ionogram/{sounding_id}.png")
def ionogram(sounding_id: int, request: Request,
             gate: str | None = None,
             trace: bool = False,
             muf: bool = True,
             bare: bool = False,
             dpi: int = Query(110, ge=40, le=300)) -> Response:
    """Render one sounding on demand.

    ``gate=auto`` fits the range window to where the echo is, which is what
    makes a search-mode v2 product legible at all -- its stored axis is
    +/-3998 km and the trace occupies a few hundred.

    ``bare=true`` drops the axes, title, colourbar and every overlay, leaving
    the raster alone. That is not a styling option: it is the backing image
    for the interactive plot, which supplies its own axes and draws the traces
    itself, and it only lines up because the pixels stop exactly where the
    data does.
    """
    path = _sounding_path(request, sounding_id)
    png = _render(path, gate=gate, trace=trace, muf=muf, dpi=dpi, bare=bare)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/preview/{station}/{tx}.png")
def station_preview(station: str, tx: str, request: Request,
                    t: str | None = None) -> Response:
    """The newest thumbnail a station pushed for one transmitter.

    Encoded on the station itself; see ``services/agent/preview.py`` for why
    that is where it belongs and what it costs there.

    ``t`` is ignored and exists to be in the URL. The console reloads itself
    entirely every 15 s, so every ``<img>`` is requested again; with a plain
    URL and this ``max-age`` the picture would freeze for an hour, and with
    ``no-cache`` every open console would re-transfer every thumbnail four
    times a minute. Keying the URL on the sounding's own ``t0`` makes an
    unchanged product a cache hit and a new one a miss, which is exactly the
    question being asked.
    """
    image = db.preview_image(request.app.state.db, station, tx)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"no preview for {tx} at {station}")
    return Response(image, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/soundings/{sounding_id}/sao.xml")
def sounding_sao(sounding_id: int, request: Request,
                 gate: str = "auto") -> Response:
    """This sounding as SAO.XML 5.0, one ``<SAORecord>`` per estimator.

    The same document ``muf export`` writes, from the same code -- so an
    archive scaled on the station and one pulled from here are the same file,
    which is the only reason it is worth serving at all.

    Separate records rather than one merged scaling: the spec keeps
    independent scalings apart (sec. 1.3.4), and the disagreement between
    three estimators on one trace is the closest thing this pipeline has to an
    error bar. Merging them would throw it away.
    """
    from muf.export import saoxml

    from . import sao as sao_mod

    path = _sounding_path(request, sounding_id)
    scaling = sao_mod.build(path, gate=gate)
    body = saoxml.to_string(scaling.root)
    name = Path(path).stem
    return Response(
        body, media_type="application/xml",
        headers={"Cache-Control": "public, max-age=3600",
                 "Content-Disposition":
                     f'inline; filename="{name}.SAO.XML"'})


def _sounding_path(request: Request, sounding_id: int) -> Path:
    """The product behind one sounding row, or the reason it cannot be served."""
    row = db.one(request.app.state.db,
                 "SELECT * FROM sounding WHERE id = ?", (sounding_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such sounding")

    path = Path(row["path"])
    if not path.is_absolute():
        path = Path(request.app.state.archive_root) / path
    # Guarded, and the two outcomes are genuinely different. `Path.is_file`
    # swallows only ENOENT/ENOTDIR/EBADF/ELOOP, so a mount that is present but
    # not answering -- EIO, EHOSTDOWN, ESTALE -- raises straight out of the
    # request. Reporting that as 410 Gone would be two lies at once: the file
    # is probably fine, and the suggested fix (check ARCHIVE_ROOT) is for a
    # different fault entirely. 503 is what "the storage is down, try later"
    # means, and it keeps a caching client from treating it as permanent.
    try:
        present = path.is_file()
    except OSError as exc:
        from . import archives as archives_mod
        detail = (f"the archive is mounted but not answering -- {exc}. "
                  f"The database row is intact; this is storage, not data. "
                  f"Fix the mount on the host and retry.")
        if archives_mod.mount_fault(exc) == "denied":
            detail = (f"the archive cannot be read by this process -- {exc}. "
                      f"The api runs as uid 10001, which needs read access to "
                      f"the mounted folder.")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail) from None

    if not present:
        raise HTTPException(
            status.HTTP_410_GONE,
            f"{row['file']} is in the database but not on disk at {path}. "
            f"Check ARCHIVE_ROOT matches the root it was ingested with.")
    return path


def _render(path: Path, *, gate, trace, muf, dpi, bare: bool = False) -> bytes:
    import matplotlib
    matplotlib.use("Agg")                      # no display in a container

    from muf import extractors, render

    from . import sao as sao_mod

    # Gated by the same function the plot frame uses. If these two ever gate
    # differently the raster and the axes drawn over it describe different
    # range windows, and the picture is quietly wrong rather than broken.
    ion = sao_mod.load_ion(path, gate=gate)

    if bare:
        # No extractors, no trace fit: nothing is drawn on this image, and
        # running the estimators to throw the answer away would triple what
        # the page waits for.
        buffer = io.BytesIO()
        render.plot(ion, buffer, None, axes=False, dpi=dpi)
        return buffer.getvalue()

    results = extractors.run(ion) if muf else None
    segments = reconstruction = None
    if trace and results:
        from muf import fit as fit_module, geometry, trace as trace_module

        best = next((r for r in results.values() if r.ok), None)
        if best is not None:
            _, _, ground = geometry.path_of(ion.header)
            freq, vrange, weight = trace_module.extract_points(ion, best)
            segments, reconstruction = trace_module.analyse(
                freq, vrange, ground, weight=weight)

    buffer = io.BytesIO()
    render.plot(ion, buffer, results, axes=True, dpi=dpi,
                segments=segments, reconstruction=reconstruction)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _tri(value):
    """SQLite integer back to a tri-state. ``None`` stays ``None``."""
    return None if value is None else bool(value)


def _age_seconds(received_at: str | None) -> float | None:
    from datetime import datetime, timezone

    if not received_at:
        return None
    try:
        when = datetime.strptime(received_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None
    return round((datetime.now(timezone.utc) - when).total_seconds(), 1)


def _preview(row: dict) -> dict:
    """One thumbnail's metadata, ready for a template. No image bytes.

    ``age_s`` is measured from ``t0`` -- the sounding's own start time -- not
    from ``received_at``. "How old is this picture of the ionosphere" is the
    question being asked, and a push that was delayed does not make the
    sounding in it any newer. It is the same choice ``newest_product_age``
    makes on the station, and it can come out negative for the same reason:
    the recorder's clock disagreeing with this one, which the health metrics
    are where to go and look at.
    """
    t0 = row["t0"]
    url = f"/preview/{quote(row['station'])}/{quote(str(row['tx']))}.png"
    return {
        "tx": row["tx"],
        "t0": t0,
        # `t0` in the query string is the cache key and nothing else; see the
        # `station_preview` endpoint for why the URL needs one at all.
        "url": url if t0 is None else f"{url}?t={t0:.0f}",
        "age_s": None if t0 is None else time.time() - float(t0),
        "received_at": row["received_at"],
        "width": row["width"], "height": row["height"],
        "freq_mhz": _span(row["freq_lo_hz"], row["freq_hi_hz"], 1e6),
        "range_km": _span(row["range_lo_m"], row["range_hi_m"], 1e3),
        "cropped": bool(row["cropped"]),
    }


def _span(lo, hi, scale: float) -> str:
    """``"0.6-16.0"``, or empty when the agent did not say."""
    if lo is None or hi is None:
        return ""
    return f"{float(lo) / scale:.1f}–{float(hi) / scale:.1f}"


def _command(row: dict) -> dict:
    # `results` is what the agent said, and until now it stayed in the
    # database: the console printed the command's *parameters* in a column
    # headed "result", so every refusal rendered as a bare red `failed` with
    # the request echoed back. The agent always sends a reason -- "no systemd
    # target configured for this station", "the schedule has 2 rank group(s)
    # but the launcher starts calc_ionograms.py with -np 1" -- and that reason
    # is the only thing that tells an operator what to change.
    return {
        "id": row["id"], "name": row["name"],
        "params": json.loads(row["params"] or "{}"),
        "results": json.loads(row["results"] or "[]"),
        "detail": _detail(row["results"]),
        "issued_at": row["issued_at"], "issued_by": row["issued_by"],
        "delivered_at": row["delivered_at"], "acked_at": row["acked_at"],
        "ok": _tri(row["ok"]),
        "state": ("acked" if row["acked_at"] else
                  "delivered" if row["delivered_at"] else "pending"),
    }


def _detail(results: str | None) -> str:
    """The agent's reasons, joined. Empty when it did not give one."""
    try:
        parsed = json.loads(results or "[]")
    except (TypeError, ValueError):
        return ""
    return "; ".join(
        str(r.get("detail", "")).strip() for r in parsed
        if isinstance(r, dict) and str(r.get("detail", "")).strip())
