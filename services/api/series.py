"""What the series page draws: four parameters over time, and a model beside them.

The page used to draw one number. MUF is the number this pipeline exists to
produce, but on its own it cannot be *judged*: a curve that looks plausible is
indistinguishable from one that is wrong by a constant, and a day of picks
pinned at the band ceiling looks like a good day at the top of the band. Read
together, four series settle most of that between them:

``MUF``
    The measured operational MUF, carrying ``limited`` -- the pick sat at the
    top of the sweep, so the value is a lower bound. Bounds are drawn, never
    dropped: filtering them out biases the curve high exactly when the
    ionosphere is best (``BACKLOG.md`` sec. 3).

``LOF``
    The lowest observed frequency, carrying ``loflim`` for the same censoring
    at the other end of the band. **LOF, not LUF** -- P.533-13 sec. 9 defines
    the lowest *usable* frequency with a required signal-to-noise ratio and a
    monthly median, and one sounding has neither. :mod:`muf.lof` sets out the
    argument. What the low end buys here is independent of the MUF: it tracks
    D-region absorption, so it follows solar illumination directly, and a MUF
    that moves with a LOF that does not is a MUF worth doubting.

``foF2``
    The vertical critical frequency the measured MUF implies, through the
    secant law at a stated height. Not a measurement: an oblique sounder never
    sees vertical incidence (``saoxml`` writes it as ``<Modeled>`` for the same
    reason). It earns its place because it is the only form in which this
    circuit's measurement can be set against a vertical-incidence model, or
    against a nearby ionosonde, without the obliquity of *this* path in the way.

``IRI``
    The International Reference Ionosphere at the path's control point, and the
    residual against it. This is the one series here that did not come out of
    the recording, and it is the only one that can catch a bias shared by all
    three estimators -- they share a spectrogram, a gate, a threshold and a
    picker, so their agreement shows consistency and not accuracy.

**The model is evaluated at the sounding instants, one call per day.** At the
instants because that is what makes a residual subtractable; per day because
:func:`muf.reference.iri.predict` reads its solar driver from the *first*
timestamp it is given, and one F10.7 stretched across a window holding February
and August would be wrong in a way nothing on the page would show. PyIRI costs
one evaluation per day per control point either way, so this is close to free.

**Nothing here is NaN.** Every array crosses into the template through
``|tojson``, and Python's ``json`` writes a bare ``NaN`` that ``JSON.parse``
refuses -- one absent pick would blank the whole plot with an error visible
only in the browser console. Missing values are ``None``, which is ``null``,
which plotly draws as a gap.
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import threading
from collections import OrderedDict

from muf.geometry import (DEFAULT_HMF2_KM, Point, describe_path, hop_count,
                          muf_to_fof2)

#: Whether the page draws a model beside the measurements. ``SERIES_MODEL=0``
#: turns it off.
#:
#: Same knob shape as :data:`services.api.sao.MODEL` and for the same reason:
#: the model is the one part of this page that can reach the network, because
#: PyIRI needs a solar driver and an unwarmed index cache means a fetch. A
#: module-level name rather than a default argument so a test can monkeypatch
#: it.
MODEL = os.environ.get("SERIES_MODEL", "1") not in ("0", "", "false")

#: Height assumed when inverting a measured MUF back to a vertical foF2.
#:
#: The same constant ``muf.export.saoxml`` writes into its ``<Modeled>`` foF2,
#: shared so the page and the SAO download cannot disagree about what the
#: measurement implies. It is a stated assumption, not a measurement: real hmF2
#: runs 250-400 km, and the page says so.
EQUIVALENT_HMF2_KM = DEFAULT_HMF2_KM

#: Days the model will cover in one page load.
#:
#: The cost is one IRI evaluation per day per control point, paid while the
#: request is open. A window of a few days is the case this page is for -- the
#: day pills exist because the archive is days months apart -- and a request
#: that quietly takes a minute is worse than one that says why it declined.
MAX_MODEL_DAYS = 31

#: Modelled series kept in memory, keyed by circuit and by the instants asked
#: for. Small -- a few thousand floats each -- and the access pattern is a
#: reader stepping between days and methods over the same circuit.
CACHE_SIZE = 8

_LOCK = threading.Lock()
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()


def circuit_name(row) -> str:
    """The circuit a row belongs to, spelled as the route's chooser spells it."""
    return f"{row['tx']} -> {row['rx']}"


def _finite(value):
    """A float, or ``None`` for anything that is not one -- NaN included."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _iso(text) -> str:
    """A stored timestamp as ``YYYY-MM-DDTHH:MM:SS.mmm``.

    The database keeps ``2026-02-04 00:00:00.009633``: a space separator, and
    microseconds. Plotly's date axis is documented to milliseconds, and this
    page's axis spans hours -- the trailing three digits cannot be resolved on
    it and are not worth the risk of a parser that stops at a date it cannot
    read.
    """
    stamp = str(text).strip().replace(" ", "T")
    head, dot, frac = stamp.partition(".")
    return f"{head}.{frac[:3]:0<3}" if dot else head


def _point(lat, lon) -> Point | None:
    lat, lon = _finite(lat), _finite(lon)
    return None if lat is None or lon is None else Point(lat, lon)


def endpoints(rows) -> tuple[Point | None, Point | None, float | None]:
    """The circuit's geometry, off the first row that carries it.

    Read from the stored sounding rather than looked up in :mod:`muf.stations`:
    the coordinates that produced this MUF are the ones that must produce the
    model beside it. A station table corrected after ingest would otherwise
    move the control point without moving anything it is compared against, and
    the residual would change with no sounding having changed.
    """
    for row in rows:
        tx = _point(row.get("tx_lat"), row.get("tx_lon"))
        rx = _point(row.get("rx_lat"), row.get("rx_lon"))
        if tx is not None and rx is not None:
            return tx, rx, _finite(row.get("path_km"))
    return None, None, None


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def model_for(tx: Point, rx: Point, stamps: list[str]) -> dict:
    """IRI at these instants for this circuit. Memoised on both.

    Returns ``{"muf": [...], "fof2": [...], "hmf2": [...], "source": str}``, or
    a dict whose only key is ``error``. Never raises: a model that is not
    installed, or a host that cannot reach its solar driver, is a normal
    condition here and has to degrade to a stated absence rather than to a
    blank panel that reads like a measurement of nothing.
    """
    digest = hashlib.sha1("\n".join(stamps).encode()).hexdigest()
    key = (tx.lat, tx.lon, rx.lat, rx.lon, len(stamps), digest)

    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit

    got = _model(tx, rx, stamps)

    with _LOCK:
        _CACHE[key] = got
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return got


def _model(tx: Point, rx: Point, stamps: list[str]) -> dict:
    from muf.reference import iri

    if not iri.available():
        return {"error": f"not installed. {iri.INSTALL_HINT}"}

    import pandas as pd

    index = pd.DatetimeIndex(pd.to_datetime(stamps))
    days = index.normalize().unique()
    if len(days) > MAX_MODEL_DAYS:
        return {"error": f"this window spans {len(days)} days and the model is "
                         f"drawn for up to {MAX_MODEL_DAYS}; pick a day, or a "
                         f"narrower range"}

    muf: list = [None] * len(index)
    fof2: list = [None] * len(index)
    hmf2: list = [None] * len(index)
    sources: list[str] = []
    failures: list[str] = []

    # One call per day, so each day is driven by its own F10.7. `predict` reads
    # the driver off `index[0]` alone, which over a window holding two seasons
    # would model August with February's sun and say nothing about it.
    for day in days:
        rows = [i for i, when in enumerate(index) if when.normalize() == day]
        try:
            got = iri.predict(tx, rx, index[rows])
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"{day.date()}: {type(exc).__name__}: {exc}")
            continue
        if got.error:
            failures.append(f"{day.date()}: {got.error}")
            continue

        sources.append(got.source)
        values = list(got.muf)
        detail = got.detail
        criticals = list(detail["fof2"]) if detail is not None \
            and "fof2" in detail else [None] * len(rows)
        heights = list(detail["hmf2"]) if detail is not None \
            and "hmf2" in detail else [None] * len(rows)
        for position, row in enumerate(rows):
            muf[row] = _finite(values[position])
            fof2[row] = _finite(criticals[position])
            hmf2[row] = _finite(heights[position])

    if not any(value is not None for value in muf):
        return {"error": failures[0] if failures else
                "the model returned nothing usable for this circuit"}

    # The driver is part of what the numbers mean, so it is reported rather
    # than assumed. Distinct sources are counted rather than listed: over a
    # month that is thirty near-identical sentences differing in one figure.
    unique = list(dict.fromkeys(sources))
    source = unique[0]
    if len(unique) > 1:
        source += f" (+{len(unique) - 1} more driver(s) across the window)"

    out = {"muf": muf, "fof2": fof2, "hmf2": hmf2, "source": source}
    if failures:
        out["note"] = f"{len(failures)} day(s) unmodelled: {failures[0]}"
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _pearson(a: list[float], b: list[float]):
    """Correlation, or ``None`` when there is not enough of it to have one."""
    if len(a) < 3:
        return None
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    da = [x - mean_a for x in a]
    db = [x - mean_b for x in b]
    spread = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    return sum(x * y for x, y in zip(da, db)) / spread if spread else None


def compare(measured: list, modelled: list, limited: list) -> dict:
    """Measured against modelled, with the lower bounds left out of the summary.

    A ``limited`` MUF is a lower bound, not a measurement: the pick sat at the
    top of the sweep and the ionosphere was supporting at least that. Counting
    it as a residual would report the *recorder's* band ceiling as a modelling
    error, and on a ceiling-limited circuit that is most of the daytime. The
    excluded ones are counted and reported rather than silently dropped -- a
    bias computed from four of forty points is a different claim from one
    computed from forty.
    """
    kept, held = [], 0
    for value, model, bound in zip(measured, modelled, limited):
        if value is None or model is None:
            continue
        if bound:
            held += 1
        else:
            kept.append((value, model))

    out = {"n": len(kept), "excluded": held}
    if not kept:
        return out

    diffs = [value - model for value, model in kept]
    out["bias"] = statistics.median(diffs)
    out["rms"] = math.sqrt(statistics.fmean(d * d for d in diffs))
    out["r"] = _pearson([v for v, _ in kept], [m for _, m in kept])
    return out


def _summary(values: list) -> dict:
    """Count and median of whatever is actually there."""
    present = [v for v in values if v is not None]
    return {"n": len(present),
            "median": statistics.median(present) if present else None}


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------

def frame(rows, *, model: str = "iri", forecasts: dict | None = None) -> dict:
    """Everything the page draws, grouped by circuit.

    Grouped because a circuit is the unit of every parameter here, not just of
    the MUF: the obliquity that turns foF2 into a MUF is the path's, the
    control point the model is evaluated at is the path's, and the band ceiling
    that censors a pick is the recorder's. One array spanning two circuits
    would describe neither.

    ``forecasts`` is keyed by ``(tx, rx, param)`` and carries its **own** time
    axis rather than being resampled onto the soundings'. A forecast that
    reaches past the last measurement is the only interesting part of one, and
    aligning it to the picks would crop off exactly that.
    """
    wanted = model == "iri" and MODEL

    by_circuit: "OrderedDict[str, list]" = OrderedDict()
    for row in rows:
        by_circuit.setdefault(circuit_name(row), []).append(row)

    circuits = []
    for name, group in by_circuit.items():
        tx, rx, path_km = endpoints(group)
        hops = hop_count(path_km) if path_km else 1
        # The obliquity is set by one hop's ground distance, not by the whole
        # path -- and `iri.predict` converts its foF2 the same way. Inverting
        # the measurement over the full distance on a two-hop path would put
        # the two foF2 curves on different geometries and make the comparison
        # the page exists for meaningless.
        hop_km = path_km / hops if path_km else None

        stamps = [_iso(row["datetime"]) for row in group]
        muf = [_finite(row.get("muf")) for row in group]
        lof = [_finite(row.get("lof")) for row in group]
        smooth = [_finite(row.get("muf_smooth")) for row in group]
        fof2 = [muf_to_fof2(value, hop_km, EQUIVALENT_HMF2_KM)
                if value is not None and hop_km else None for value in muf]

        got: dict = {}
        if wanted:
            if tx is None or rx is None:
                got = {"error": "no coordinates stored for this circuit; the "
                                "model needs both ends to find a control point"}
            else:
                got = model_for(tx, rx, stamps)

        modelled = got.get("muf") or [None] * len(group)
        limited = [1 if row.get("limited") else 0 for row in group]
        residual = [value - point if value is not None and point is not None
                    else None for value, point in zip(muf, modelled)]

        circuits.append({
            "name": name,
            "tx": group[0]["tx"], "rx": group[0]["rx"],
            "path_km": path_km, "hops": hops,
            "geometry": describe_path(tx, rx) if tx and rx else "",
            "n": len(group),
            "id": [row["id"] for row in group],
            "t": stamps,
            "muf": muf, "lof": lof, "fof2": fof2, "muf_smooth": smooth,
            "limited": limited,
            "loflim": [1 if row.get("loflim") else 0 for row in group],
            "sweep_top": [_finite(row.get("freq_stop")) for row in group],
            "model": got,
            "forecast": [entry for entry in (forecasts or {}).values()
                         if entry["tx"] == group[0]["tx"]
                         and entry["rx"] == group[0]["rx"]],
            "residual": residual,
            "stats": {
                "muf": _summary(muf), "lof": _summary(lof),
                "fof2": _summary(fof2),
                "limited": sum(limited),
                "loflim": sum(1 for row in group if row.get("loflim")),
                "model": compare(muf, modelled, limited),
            },
        })

    return {
        "circuits": circuits,
        "model": model if wanted else "off",
        "hmf2_km": EQUIVALENT_HMF2_KM,
        "any_model": any(c["model"].get("muf") for c in circuits),
        "any_forecast": any(c["forecast"] for c in circuits),
    }


def clear() -> None:
    """Drop the memo. For tests, which must not inherit each other's."""
    with _LOCK:
        _CACHE.clear()
