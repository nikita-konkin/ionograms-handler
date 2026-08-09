"""Real ionosonde measurements from GIRO / DIDBase.

The strongest reference available: an independent instrument measuring the same
ionosphere the oblique path reflects off. For Cyprus -> Yoshkar-Ola the control
point is 45.88N 39.45E and station **RV149 ROSTOV** sits 148 km away, comfortably
inside the scale over which the F2 layer stays correlated.

Two ways to get an oblique MUF out of it:

* ask DIDBase for ``MUFD`` with ``DMUF=<path km>``, letting the server apply its
  own obliquity factor; or
* take ``foF2`` and ``hmF2`` and convert here with :func:`muf.geometry.fof2_to_muf`.

Both are fetched. The second is preferred when ``hmF2`` is present, because it
uses the measured reflection height rather than a nominal one, and because the
conversion is then visible rather than happening server-side.

Data are subject to the LGDC Rules of the Road:
https://ulcar.uml.edu/DIDB/RulesOfTheRoadForDIDBase.htm
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry import (DEFAULT_HMF2_KM, Point, great_circle_km, fof2_to_muf,
                        midpoint)
from . import ReferenceSeries, as_index

DIDB_URL = "https://lgdc.uml.edu/common/DIDBGetValues"
TIMEOUT_S = 90
DEFAULT_CACHE = Path.home() / ".cache" / "muf" / "giro"

#: Stations near the paths this instrument sounds. URSI code -> (name, lat, lon).
#: The full list lives at https://lgdc.uml.edu/common/DIDBStationList
STATIONS: dict[str, tuple[str, float, float]] = {
    "RV149": ("Rostov-on-Don", 47.20, 39.70),
    "MO155": ("Moscow", 55.47, 37.30),
    "KL154": ("Kaliningrad", 54.70, 20.60),
    "SQ143": ("Sofia", 42.68, 23.35),
    "AT138": ("Athens", 38.00, 23.50),
    "NI135": ("Nicosia", 35.03, 33.16),
    "PQ052": ("Pruhonice", 50.00, 14.60),
    "JR055": ("Juliusruh", 54.60, 13.40),
    # The digisondes DOB receives obliquely (`muf.io_digisonde`). Coordinates
    # from https://lgdc.uml.edu/common/DIDBFastStationList, converted to
    # -180..180. Chilton publishes no GIRO feed -- DIDBase carries it, the
    # real-time mirror does not -- so `history` returns nothing for RL052.
    "RL052": ("Chilton", 51.50, -0.60),
    "DB049": ("Dourbes", 50.10, 4.60),
    "TR169": ("Tromso", 69.60, 19.20),
    "EA036": ("El Arenosillo", 37.10, -6.70),
    "IR352": ("Irkutsk", 52.30, 104.30),
    "NV355": ("Novosibirsk", 54.60, 83.20),
}

#: Beyond this the station's ionosphere is no longer a fair stand-in for the
#: control point's. The F2 layer decorrelates over a few hundred km.
MAX_STATION_DISTANCE_KM = 500.0


def nearest_station(target: Point,
                    max_km: float = MAX_STATION_DISTANCE_KM
                    ) -> tuple[str, str, float] | None:
    """Closest known station to ``target``: ``(ursi, name, km)`` or None."""
    best = None
    for ursi, (name, lat, lon) in STATIONS.items():
        km = great_circle_km(target, Point(lat, lon))
        if best is None or km < best[2]:
            best = (ursi, name, km)
    if best is None or best[2] > max_km:
        return None
    return best


def build_url(ursi: str, start: dt.datetime, stop: dt.datetime,
              path_km: float | None = None) -> str:
    """DIDBase query URL. Dates are ``YYYY/MM/DD HH:MM:SS``."""
    chars = "foF2,hmF2,MUFD"
    params = {
        "ursiCode": ursi,
        "charName": chars,
        "fromDate": start.strftime("%Y/%m/%d %H:%M:%S"),
        "toDate": stop.strftime("%Y/%m/%d %H:%M:%S"),
    }
    if path_km:
        # Makes the server return MUFD scaled to this path length.
        params["DMUF"] = str(int(round(path_km)))
    return DIDB_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def fetch(url: str, cache_dir: Path | None = None, offline: bool = False) -> str:
    """Fetch a DIDBase response, caching by URL."""
    cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = str(abs(hash(url)))
        cache = cache_dir / f"didb_{key}.txt"
        if cache.exists():
            return cache.read_text(encoding="utf-8", errors="replace")

    if offline:
        raise RuntimeError("offline and this query is not cached")

    request = urllib.request.Request(url, headers={"User-Agent": "muf-pipeline/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"DIDBase returned HTTP {exc.code} for {url}. The service moves "
            f"occasionally -- check https://giro.uml.edu/didbase/scaled.php for "
            f"the current endpoint."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"could not reach DIDBase: {exc}") from exc

    if cache:
        cache.write_text(text, encoding="utf-8")
    return text


def parse(text: str) -> pd.DataFrame:
    """Parse a DIDBase text table.

    Comment lines start with ``#``; the last of them is the column header. Values
    may carry a one-character qualifier or descriptive letter, and missing
    entries appear as ``---``.
    """
    header: list[str] = []
    rows: list[list[str]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").split()
            if candidate and any(c.lower().startswith("time") for c in candidate):
                header = candidate
            continue
        rows.append(stripped.split())

    if not rows:
        return pd.DataFrame()

    width = max(len(r) for r in rows)
    if len(header) != width:
        header = ["time"] + [f"c{i}" for i in range(1, width)]

    frame = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], columns=header)

    time_col = next((c for c in frame.columns if c.lower().startswith("time")),
                    frame.columns[0])
    index = pd.to_datetime(frame[time_col], errors="coerce", utc=True, format="mixed")
    frame = frame[index.notna()]
    frame.index = pd.DatetimeIndex(index[index.notna()]).tz_localize(None)
    frame = frame.drop(columns=[time_col])

    def numeric(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(r"[^\d.\-+eE]", "", regex=True)
        return pd.to_numeric(cleaned.replace({"": None, "-": None}), errors="coerce")

    out = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        lowered = column.lower()
        if "fof2" in lowered:
            out["fof2"] = numeric(frame[column])
        elif "hmf2" in lowered:
            out["hmf2"] = numeric(frame[column])
        elif "muf" in lowered:
            out["mufd"] = numeric(frame[column])
    return out.dropna(how="all")


def predict(
    tx: Point,
    rx: Point,
    times,
    ursi: str | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    use_measured_height: bool = True,
    max_km: float = MAX_STATION_DISTANCE_KM,
    **_,
) -> ReferenceSeries:
    """MUF for the path, from the nearest ionosonde's measurements."""
    index = as_index(times)
    if not len(index):
        return ReferenceSeries("giro", error="no timestamps given")

    path_km = great_circle_km(tx, rx)
    control = midpoint(tx, rx)

    if ursi is None:
        found = nearest_station(control, max_km)
        if found is None:
            return ReferenceSeries(
                "giro",
                error=(f"no known station within {max_km:.0f} km of the control "
                       f"point {control}. Add one from "
                       f"https://lgdc.uml.edu/common/DIDBStationList to "
                       f"giro.STATIONS, or pass ursi=."),
            )
        ursi, station_name, station_km = found
    else:
        entry = STATIONS.get(ursi)
        station_name = entry[0] if entry else ursi
        station_km = great_circle_km(control, Point(entry[1], entry[2])) if entry else float("nan")

    start = index.min().to_pydatetime() - dt.timedelta(minutes=30)
    stop = index.max().to_pydatetime() + dt.timedelta(minutes=30)
    url = build_url(ursi, start, stop, path_km)

    try:
        measured = parse(fetch(url, cache_dir, offline))
    except RuntimeError as exc:
        return ReferenceSeries("giro", error=str(exc))

    if measured.empty:
        return ReferenceSeries(
            "giro",
            error=(f"{ursi} ({station_name}) returned no scaled data for "
                   f"{start:%Y-%m-%d %H:%M} to {stop:%H:%M} UTC"),
        )

    # Prefer converting foF2 ourselves: it uses the measured reflection height
    # and keeps the conversion visible.
    if "fof2" in measured and measured["fof2"].notna().any():
        heights = (measured["hmf2"] if use_measured_height and "hmf2" in measured
                   else pd.Series(DEFAULT_HMF2_KM, index=measured.index))
        heights = heights.fillna(DEFAULT_HMF2_KM)
        muf = pd.Series(
            [fof2_to_muf(f, path_km, h) if np.isfinite(f) else np.nan
             for f, h in zip(measured["fof2"], heights)],
            index=measured.index,
        )
        how = "foF2 x secant law"
    elif "mufd" in measured:
        muf = measured["mufd"]
        how = f"server MUFD at DMUF={path_km:.0f}"
    else:
        return ReferenceSeries(
            "giro", error=f"{ursi} returned no foF2 or MUFD column"
        )

    muf = muf.dropna().sort_index()
    resampled = _to_times(muf, index)

    return ReferenceSeries(
        name="giro",
        muf=resampled,
        detail=measured,
        source=(f"{ursi} {station_name}, {station_km:.0f} km from control point "
                f"{control}; {how}"),
    )


def _to_times(series: pd.Series, index: pd.DatetimeIndex,
              tolerance_min: int = 20) -> pd.Series:
    """Align measurements onto our sounding times by nearest neighbour.

    Ionosondes typically run every 15 minutes against our 5, so interpolating
    would invent structure; nearest-within-tolerance keeps each value a
    measurement.
    """
    if series.empty:
        return pd.Series(index=index, dtype=float)
    aligned = series.reindex(
        index, method="nearest", tolerance=pd.Timedelta(minutes=tolerance_min)
    )
    return aligned.astype(float)


#: Real-time mirror of GIRO scaled characteristics, used when DIDBase itself is
#: unreachable. It is not a substitute: it carries the last ~7 days for the
#: subset of stations feeding the propagation maps, where DIDBase holds the
#: full archive for every station. But when `DIDBGetValues` is down -- which it
#: was on 2026-08-09, GIRO's own documented example URLs returning 404 -- this
#: is the difference between a comparison and none.
HISTORY_URL = "https://prop.kc2g.com/api/history.json"

#: Field order of one `history` entry. Verified against the same station's
#: live record rather than assumed: the feed documents neither.
HISTORY_FIELDS = ("time", "confidence", "fof2", "mufd", "hmf2")


def history(ursi: str, cache_dir: Path | None = None,
            offline: bool = False) -> pd.DataFrame:
    """Recent scaled characteristics for one station, from the GIRO mirror.

    Returns a frame indexed by UTC time with ``fof2``, ``mufd``, ``hmf2`` and
    ``confidence``; empty when the station has no feed. Roughly a week of
    history at the station's own cadence.

    Use it for the check nothing else in this pipeline can make. Every
    estimator here shares a spectrogram, a gate, a threshold and a picker, so a
    common bias cannot show up in their agreement -- and IRI is a model. A
    digisonde's *own vertical* ``foF2``, measured by a different instrument at
    the same instant, converts through :func:`muf.geometry.fof2_to_muf` into
    the oblique MUF this receiver should be seeing on that path. On
    2026-08-09 that comparison is what showed the DOB picks to be interference:
    Juliusruh's own foF2 rose 2.59 to 5.28 MHz through the morning, implying an
    oblique MUF from 3.8 to 11.3 MHz, while every pick sat flat at 3.05.
    """
    text = fetch(HISTORY_URL, cache_dir=cache_dir, offline=offline)
    stations = json.loads(text)
    for station in stations:
        if str(station.get("code", "")).upper() != ursi.upper():
            continue
        rows = station.get("history") or []
        if not rows:
            break
        frame = pd.DataFrame(rows, columns=list(HISTORY_FIELDS))
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        for column in ("fof2", "mufd", "hmf2", "confidence"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["time"]).set_index("time").sort_index()
    return pd.DataFrame(
        columns=[c for c in HISTORY_FIELDS if c != "time"],
        index=pd.DatetimeIndex([], tz="UTC", name="time"))
