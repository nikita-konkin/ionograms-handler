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
from jinja2 import pass_context

from muf.reference import indices

from . import acquisition, db, i18n
from . import net as net_mod
from . import sao as sao_mod
from . import series as series_mod
from . import sources as sources_mod
from .auth import require_read
from .read_routes import _age_seconds, _command, _preview, _tri

router = APIRouter(include_in_schema=False, dependencies=[Depends(require_read)])
# `context_processors` puts `t`, `plural` and `lang` into every render
# without any route function or context dict having to carry a language
# around -- see `i18n.context`. The alternative, threading a language
# through eight route signatures, is eight chances to forget one.
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates"),
    context_processors=[i18n.context])

#: Age past which a station is shown as stale rather than as whatever it last
#: said. Defined in `acquisition` because the acquiring/stopped indicator
#: turns on the same threshold: a report too old to show is too old to
#: conclude anything from.
STALE_AFTER_S = acquisition.STALE_AFTER_S

#: Lead times the training form offers, in hours. A select rather than a box
#: for the lag in samples: the operator is choosing how far ahead to predict,
#: and `288` is an implementation detail of a five-minute grid. These four are
#: the horizons the archive's own models were fitted at, so a trained model
#: lands in the same scoring bucket as the import it is meant to replace.
LEADS = (24, 48, 72, 120)


def _duration(seconds, lang: str = i18n.DEFAULT) -> str:
    """Seconds as something readable at a glance, with a sign for the past.

    An operator reading "is it working this minute" should not have to divide
    by sixty. ``None`` prints as an em dash rather than as zero, because "not
    known" and "now" are the two answers this must never confuse.

    The unit letters come from the catalog: ``9h04m`` is ``9ч04м`` in Russian,
    and a console that prints one of them beside a Russian sentence is a
    console somebody has to read in two languages at once.
    """
    if seconds is None:
        return "—"
    sec, minute, hour, day = (str(i18n.t(f"unit.{u}", lang))
                              for u in ("s", "m", "h", "d"))
    sign, value = ("-", -seconds) if seconds < 0 else ("", seconds)
    if value < 90:
        return f"{sign}{value:.0f}{sec}"
    if value < 5400:
        return f"{sign}{int(value) // 60}{minute}{int(value) % 60:02d}{sec}"
    if value < 172800:
        return f"{sign}{int(value) // 3600}{hour}{(int(value) % 3600) // 60:02d}{minute}"
    return f"{sign}{value / 86400:.1f}{day}"


def _lead(seconds, lang: str = i18n.DEFAULT) -> str:
    """A horizon as a column heading: ``1 h``, ``24 h``, ``7 d``.

    Separate from `_duration`, which answers "how long ago" and prints
    ``1h00m``. A leaderboard column is a label, not a measurement, and the
    minutes in it are noise.
    """
    if seconds is None:
        return "—"
    seconds = int(seconds)
    unit = "h" if seconds < 172800 else "d"
    count = seconds // 3600 if seconds < 172800 else seconds // 86400
    return f"{count} {i18n.t('unit.' + unit, lang)}"


# Wrapped rather than registered directly: a Jinja filter has no request, and
# `pass_context` is how it reaches the `lang` the context processor put there.
# Keeping the plain functions callable means the units stay testable without a
# render.
templates.env.filters["duration"] = pass_context(
    lambda ctx, seconds: _duration(seconds, ctx.get("lang", i18n.DEFAULT)))
templates.env.filters["lead"] = pass_context(
    lambda ctx, seconds: _lead(seconds, ctx.get("lang", i18n.DEFAULT)))


#: How far the observed band may sit inside the configured one before the
#: console calls it a disagreement, in MHz.
#:
#: The legitimate gap is small and known: `calc_ionograms.py:327` selects with
#: strict inequalities (`freqs > min_freq`), which drops the edge bins, and the
#: top of the sweep loses a few more -- 7.5-32.5 configured is stored as
#: 7.55-32.30, a 0.2 MHz shortfall. The failure this is looking for is nothing
#: like that size: an LO left at 12.5 MHz while the ini said 20 MHz puts the
#: observed band 7.5 MHz away. One MHz sits clear of the first and nowhere near
#: the second.
BAND_TOLERANCE_MHZ = 1.0


def _band(metrics: list[dict], acq) -> dict:
    """The configured band, the observed one, and whether they agree.

    Two independent sources on purpose. The configured half comes from the
    station's health report, which reads its ini; the observed half comes from
    the products in this database. Nothing on the station can make them agree
    by accident, which is exactly what makes the comparison worth printing:
    on 2026-08-19 the ini said one band and the recorder used another, and
    every unit stayed green for hours.
    """
    values = {m["name"]: m["value"] for m in metrics if m.get("value") is not None}

    def mhz(name):
        raw = values.get(name)
        return None if raw is None else float(raw) / 1e6

    configured = {
        "center_mhz": mhz("center_freq_hz"),
        "start_mhz": mhz("band_start_hz"),
        "stop_mhz": mhz("band_stop_hz"),
        "analysis_min_mhz": mhz("analysis_min_hz"),
        "analysis_max_mhz": mhz("analysis_max_hz"),
    }
    if configured["center_mhz"] is None:
        configured = None

    recorder = next((m for m in metrics
                     if m["name"] == "recorder_reads_ini"), None)
    can_change = bool(recorder and recorder.get("value"))
    why_not = "" if can_change else (
        (recorder or {}).get("detail")
        or "this station has not reported whether its recorder reads the ini")

    observed = getattr(acq, "observed_band", None)
    disagreement = None
    if configured and observed:
        gaps = []
        if configured["analysis_min_mhz"] is not None:
            gaps.append(abs(observed["start_mhz"] - configured["analysis_min_mhz"]))
        if configured["analysis_max_mhz"] is not None:
            gaps.append(abs(observed["stop_mhz"] - configured["analysis_max_mhz"]))
        if gaps and max(gaps) > BAND_TOLERANCE_MHZ:
            disagreement = (
                f"the products are {max(gaps):.2f} MHz from the configured "
                f"window. A recorder tuned somewhere other than center_freq "
                f"produces exactly this: healthy units, fresh products, wrong "
                f"spectrum.")

    return {"configured": configured, "observed": observed,
            "can_change": can_change, "why_not": why_not,
            "disagreement": disagreement,
            "tolerance_mhz": BAND_TOLERANCE_MHZ}


@router.get("/ui")
def console(request: Request):
    conn = request.app.state.db
    stations = []
    for station in db.stations(conn):
        latest = db.latest_health(conn, station)
        metrics = db.metrics_for(conn, int(latest["id"])) if latest else []
        age = _age_seconds(latest["received_at"]) if latest else None
        shown_metrics = [{**m, "ok": _tri(m["ok"])} for m in metrics]
        acq = acquisition.current(conn, station)
        stations.append({
            "name": station,
            "age_s": age,
            "stale": age is None or age > STALE_AFTER_S,
            "healthy": _tri(latest["healthy"]) if latest else None,
            "agent_version": latest["agent_version"] if latest else None,
            "metrics": shown_metrics,
            # Configured band beside observed band. See `_band`: two sources
            # that cannot agree by accident, which is the point.
            "band": _band(shown_metrics, acq),
            "commands": [_command(c) for c in db.recent_commands(conn, station, 8)],
            # What it is sounding this minute, which is the question the unit
            # states cannot answer: every process can be active while the
            # schedule points at a transmitter that stopped months ago.
            "acquisition": acq,
            # What it *could* sound. Read from the database, not the archive:
            # this is the identify workflow's output, so the console carries
            # the chooser without inheriting the census that made it -- see
            # `sources_page`, where the same list used to live behind a scan
            # of every detection file under the archive root.
            "verified": db.transmitters(conn, station),
            # The newest picture from each transmitter, straight off the
            # station's own disk. Metadata only -- the PNGs are fetched by the
            # browser from `/preview/...`, so no image travels inside the HTML.
            "previews": [_preview(p) for p in db.previews(conn, station)],
        })

    counts = db.one(conn, "SELECT COUNT(*) AS n FROM sounding") or {"n": 0}
    methods = db.rows(conn, "SELECT method, COUNT(*) AS n,"
                            " SUM(CASE WHEN muf IS NOT NULL THEN 1 ELSE 0 END)"
                            " AS picks FROM extraction GROUP BY method"
                            " ORDER BY method")
    return templates.TemplateResponse(request, "console.html", {
        "stations": stations, "n_soundings": counts["n"], "methods": methods,
        "stale_after": STALE_AFTER_S,
        # `control.MODES` holds four keys for two modes; the pair below is the
        # canonical spelling, and offering all four would suggest four modes.
        "modes": ("search", "scheduled"),
        # Read, never probed: `current` returns the last background reading
        # without blocking, so a page load costs nothing even with every index
        # host unreachable.
        "net": net_mod.current(),
        "sources": {s.key: s for s in indices.SOURCES},
    })


#: What counts as a point on the series page.
#:
#: MUF *or* LOF, not MUF alone. The page draws both ends of the band now, and
#: a sounding whose trace faded out before it reached the ceiling has a real
#: LOF and no MUF -- under the old condition it was not on the chart at all,
#: which reads as "nothing was recorded" rather than "the top was never seen".
#: Used for the chooser and the day list too, so a circuit or a day is offered
#: exactly when selecting it draws something.
HAS_A_PICK = "(e.muf IS NOT NULL OR e.lof IS NOT NULL)"


@router.get("/ui/series")
def series(request: Request, method: str = "algo",
           circuit: str | None = None,
           model: str = "iri",
           forecast: int | None = None,
           start: str | None = Query(None, alias="from"),
           end: str | None = Query(None, alias="to")):
    """MUF, LOF, equivalent foF2 and a model, against time, for one circuit.

    ``from``/``to`` are spelled and bounded exactly as ``/series/muf`` spells
    them -- inclusive both ends, aliased off ``start``/``end`` because ``from``
    is a Python keyword. Two routes over the same table answering differently
    to the same query string would be worse than having no filter at all.
    Both go through :func:`db.time_bound`, which is what makes a bare date and
    an ISO timestamp mean what they look like.

    **A circuit is the unit, not the receiver.** Every parameter here belongs
    to a path: its length sets the obliquity that relates foF2 to MUF, its
    control point is where the model is evaluated, and the recorder sets the
    band ceiling that censors a pick. Two circuits drawn on one axis produce
    curves that measure neither, so ``circuit`` defaults to whichever has the
    most picks rather than to all of them. ``circuit=all`` overlays them,
    coloured and named -- each with its own model, because they do not share a
    control point.

    Without any window the axis spans everything ingested, and an archive of a
    few days months apart draws them as vertical stripes with nothing legible
    between; ``days`` gives the template one link per day that has picks.

    ``model=off`` drops the reference. It is the only part of this page that
    can reach the network -- IRI needs a solar driver -- so it has to be
    refusable from the query string and not only from the environment.

    ``forecast=<model_id>`` overlays one registered model's latest issue on the
    measured curve, which is the comparison this page exists to make possible.
    Without it only an *active* model is drawn -- see `_live_forecasts`. The
    two are not the same trace and are not drawn the same: an operational
    forecast is the answer, a named one is a candidate being judged against the
    measurement beside it.
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
              f" WHERE e.method = ? AND {HAS_A_PICK}"
              " GROUP BY s.tx, s.rx ORDER BY n DESC", (method,))]
    if circuit not in ("all", *circuits):
        circuit = circuits[0] if circuits else "all"

    days = [r["day"] for r in db.rows(
        conn, "SELECT DISTINCT substr(s.datetime, 1, 10) AS day"
              " FROM extraction e JOIN sounding s ON s.id = e.sounding_id"
              f" WHERE {HAS_A_PICK} ORDER BY day")]

    sql = ["SELECT s.id, s.datetime, s.tx, s.rx, s.path_km,"
           " s.tx_lat, s.tx_lon, s.rx_lat, s.rx_lon, s.freq_stop,"
           " e.muf, e.lof, e.limited, e.loflim, e.muf_smooth, e.snr",
           "FROM extraction e JOIN sounding s ON s.id = e.sounding_id",
           f"WHERE e.method = ? AND {HAS_A_PICK}"]
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

    # Every model that has actually written a forecast for a circuit on this
    # page. Registered-but-never-run is deliberately not offered: it would be
    # a chooser entry that draws nothing, and "I picked it and no line
    # appeared" is indistinguishable from a broken page.
    candidates = db.rows(
        conn,
        "SELECT DISTINCT m.id, m.name, m.param, m.active, f.tx, f.rx "
        "FROM forecast f JOIN model_registry m ON m.id = f.model_id "
        + ("WHERE f.tx = ? AND f.rx = ? " if circuit != "all" else "")
        + "ORDER BY m.active DESC, m.name",
        (circuit.partition(" -> ")[0], circuit.partition(" -> ")[2])
        if circuit != "all" else ())

    if forecast is not None and not any(c["id"] == forecast for c in candidates):
        forecast = None                    # a stale link, not a reason to 404

    return templates.TemplateResponse(request, "series.html", {
        "method": method, "methods": methods, "points": points,
        "days": days, "start": start or "", "end": end or "",
        "circuit": circuit, "circuits": circuits,
        "model": model,
        "forecast": forecast, "candidates": candidates,
        "frame": series_mod.frame(
            points, model=model,
            forecasts=_live_forecasts(conn, points, forecast)),
    })


def _live_forecasts(conn, points, model_id: int | None = None) -> dict:
    """The latest issue for the circuits on the page.

    **Active models only, unless one is named** -- the rule `/forecast`
    already states and for the same reason. Every registered model may write
    forecasts, that being how a comparison is run, so drawing them all would
    put a legacy import on the same axis as the measurement in the same style
    as the operational curve. Naming one is a deliberate act and is marked as
    such: `comparison` rides along on the entry so the plot can draw it as a
    candidate rather than as the forecast.

    A named model is fetched **whatever its state**, including one bound to
    another circuit. That is not an oversight: running somebody else's model
    against this path is exactly what a comparison is, `infer --model` already
    allows it, and the alias in the hover says whose data it learned.

    Failures are swallowed to an empty dict: the series page is the page an
    operator opens to look at data, and it existed and worked before there was
    a prediction service at all. A forecast that cannot be read is a missing
    trace, not a 500.
    """
    circuits = {(row["tx"], row["rx"]) for row in points
                if row.get("tx") and row.get("rx")}
    if not circuits:
        return {}

    # The only difference between the two modes, kept to one line so the
    # issued_at correlation below cannot drift between them.
    where, key = (("m.id = ?", model_id) if model_id is not None
                  else ("m.active = 1", None))

    found: dict = {}
    for tx, rx in sorted(circuits):
        for param in ("muf", "lof"):
            try:
                rows = db.rows(
                    conn,
                    "SELECT f.valid_at, f.value, f.sigma, m.name, m.id, "
                    "       m.active, m.target_alias "
                    "FROM forecast f JOIN model_registry m ON m.id = f.model_id "
                    f"WHERE {where} AND f.param = ? AND f.tx = ? AND f.rx = ? "
                    "AND f.issued_at = (SELECT MAX(issued_at) FROM forecast x "
                    "  WHERE x.model_id = f.model_id AND x.param = f.param "
                    "  AND x.tx = f.tx AND x.rx = f.rx) "
                    "ORDER BY f.valid_at",
                    ((key, param, tx, rx) if key is not None
                     else (param, tx, rx)),
                )
            except Exception:                     # pragma: no cover - defensive
                continue
            if not rows:
                continue
            found[(tx, rx, param)] = {
                "tx": tx, "rx": rx, "param": param,
                "model": rows[0]["name"],
                "model_id": rows[0]["id"],
                "alias": rows[0]["target_alias"],
                "comparison": not rows[0]["active"],
                "t": [_iso_stamp(row["valid_at"]) for row in rows],
                "value": [row["value"] for row in rows],
                "sigma": [row["sigma"] for row in rows],
            }
    return found


def _iso_stamp(text: str) -> str:
    """`2026-08-13T04:00:00Z` as the page's own spelling of an instant.

    The forecast table stamps with the trailing ``Z`` that `db.utcnow` puts on
    everything; the soundings are naive UTC. Plotly would place two such axes
    an offset apart, which draws a forecast that lags its own measurement by
    the local timezone.
    """
    return str(text).rstrip("Z")


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
                                max_days=max_days, min_count=min_count,
                                block=False,
                                max_age_s=sources_mod.DEFAULT_MAX_AGE_S)
    known = db.stations(conn)
    return templates.TemplateResponse(request, "sources.html", {
        "census": census, "max_days": max_days, "min_count": min_count,
        "max_age_s": sources_mod.DEFAULT_MAX_AGE_S,
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


@router.get("/ui/archives")
def archives_page(request: Request):
    """The folders this server indexes, and what each pass did.

    Indexing is also where every characteristic comes from -- the pipeline
    runs per file, so MUF, LOF and the rest land in `extraction` and the
    sounding page builds the SAO record from them. There is no separate
    "compute characteristics" step for this page to offer.
    """
    from muf import loader
    from muf.extractors import DEFAULT_METHODS

    from . import archives as archives_mod

    conn = request.app.state.db
    return templates.TemplateResponse(request, "archives.html", {
        "archives": db.archives(conn),
        "archive_root": str(request.app.state.archive_root),
        "mount": archives_mod.mount(),
        # Cached, and refreshed on a background thread -- surveying the
        # folders here meant the page could not render until a recursive
        # walk of the whole archive finished, which during an index is
        # exactly when someone opens it. The list arrives by fetch instead.
        "candidates": archives_mod.candidates_cached(
            conn, request.app.state.archive_root)["items"],
        "status": archives_mod.status(),
        "formats": loader.FORMATS,
        "methods": archives_mod.method_availability(),
        "methods_default": DEFAULT_METHODS,
        "interval_s": archives_mod.DEFAULT_INTERVAL_S,
    })


@router.get("/ui/forecast")
def forecast_page(request: Request, param: str | None = None,
                  tx: str | None = None, rx: str | None = None):
    """What is live, what is registered, how it scored, and what may be promoted.

    Read scope, like every page here. The buttons post to the control routes
    with a token the operator pasted on the console page, which is the same
    contract the start/stop buttons and the archives page already use.

    ``param``/``tx``/``rx`` pick the circuit the leaderboard is drawn for. A
    leaderboard is only meaningful per circuit -- MAE in MHz on a 2400 km path
    is not comparable with MAE on a 700 km one -- so there is no "all circuits"
    view to mistake for one.
    """
    from ..prediction import dataset, queues, registry, scoring, train

    conn = request.app.state.db
    models = registry.models(conn)
    for model in models:
        model.pop("golden_input", None)
        model["golden"] = "recorded" if model.pop("golden_output", None) is not None \
            else "absent"

    # One row per circuit and parameter that has data, whether or not anything
    # is live for it -- "no model" is the answer an operator most needs on a
    # fresh deployment, and a table built only from the models would omit it.
    live = []
    for param in ("muf", "lof"):
        for circuit in dataset.circuits(conn, param, "contour"):
            model = registry.active(conn, param, circuit["tx"], circuit["rx"])
            issued = db.one(
                conn,
                "SELECT MAX(issued_at) AS issued_at FROM forecast "
                "WHERE param = ? AND tx = ? AND rx = ?",
                (param, circuit["tx"], circuit["rx"]),
            )
            issued_at = (issued or {}).get("issued_at")
            age = _age_seconds(issued_at) if issued_at else None
            live.append({
                "tx": circuit["tx"], "rx": circuit["rx"], "param": param,
                "model": model, "issued_at": issued_at, "age_s": age,
                # Stale against the default cadence, with slack: a run that is
                # merely late is not a fault, one that is a whole cycle late is.
                "stale": bool(age is not None and age > 2 * 21600),
            })

    # The circuit the leaderboard is drawn for: what was asked for, else the
    # first one that has actually been scored, else the first with data. A page
    # that defaulted to circuit one would show an empty board on a deployment
    # where only circuit four has ever run.
    scored = {(row["param"], row["tx"], row["rx"])
              for row in scoring.scores(conn) if row["n"]}
    chosen = None
    if param and tx and rx:
        chosen = (param, tx, rx)
    else:
        for row in live:
            key = (row["param"], row["tx"], row["rx"])
            if key in scored:
                chosen = key
                break
        else:
            chosen = ((live[0]["param"], live[0]["tx"], live[0]["rx"])
                      if live else None)

    board = scoring.leaderboard(conn, *chosen) if chosen else []

    # Every circuit that has data, for the upload and training forms. Built
    # from `live` rather than from the models, so a fresh deployment with
    # nothing registered still offers the circuits it is actually recording.
    circuits = sorted({(row["tx"], row["rx"]) for row in live})

    # Lead time in seconds, so the table can use the same `lead` filter the
    # leaderboard's horizon columns do. Computed here because the conversion
    # needs the grid step, and a template that hard-codes 300 is a template
    # that lies the day the step changes.
    jobs = queues.jobs(conn, limit=25)
    for entry in jobs:
        entry["lead_s"] = (entry.get("spec") or {}).get("lag", 0) \
            * dataset.DEFAULT_STEP_S

    return templates.TemplateResponse(request, "forecast.html", {
        "models": models,
        "live": live,
        "chosen": chosen,
        "board": board,
        # Surfaced, not acted on: see `scoring.drift`. The console says the
        # live model has been overtaken; demoting it stays a human decision.
        "drift": scoring.drift(board),
        "horizons": scoring.HORIZONS,
        "baselines": scoring.BASELINES,
        "circuits": circuits,
        # Only what is still moving or still needs reading. A registered
        # upload is already in the models table above, and listing it twice
        # would make the queue look permanently full.
        "uploads": [row for row in queues.uploads(conn, limit=25)
                    if row["state"] != queues.REGISTERED],
        "jobs": jobs,
        # Only what is still moving or still worth reading. A pass that ran
        # hours ago has already told its story in the Last issue column above.
        "runs": queues.runs(conn, limit=10),
        "leads": LEADS,
        "estimators": train.ESTIMATORS,
    })


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
def sounding(request: Request, sounding_id: int, gate: str = "auto",
             method: str = "algo", plot: str = "interactive"):
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

    # Scaled here rather than fetched by the page. It costs about a second
    # cold and nothing warm, and the alternative -- render the frame, then
    # fill it in -- means an operator stepping through a day sees a page that
    # is briefly wrong rather than briefly empty.
    scaling = frame = None
    sao_error = ""
    if row is not None and plot == "interactive":
        try:
            scaling = sao_mod.build(_product_path(request, row), gate=gate)
            frame = sao_mod.plot_data(scaling, method)
        except Exception as exc:                                  # noqa: BLE001
            # A missing file, an unreadable product, a detector that threw.
            # Named on the page: the rest of it -- the row, the neighbours,
            # the stored extractions -- is still worth reading, and a page
            # that 500s because one panel could not be drawn hides all of it.
            sao_error = f"{type(exc).__name__}: {exc}"

    return templates.TemplateResponse(request, "sounding.html", {
        "sounding": row, "extractions": extractions, "gate": gate,
        "prev": neighbours["prev"], "next": neighbours["next"],
        "scaling": scaling, "frame": frame, "sao_error": sao_error,
        "method": method, "methods": sao_mod.METHODS, "plot": plot,
    })


def _product_path(request: Request, row) -> Path:
    """Where a sounding row's file actually is. Relative paths are archive-root."""
    path = Path(row["path"])
    if not path.is_absolute():
        path = Path(request.app.state.archive_root) / path
    return path
