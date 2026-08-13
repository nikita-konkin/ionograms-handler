"""The International Reference Ionosphere.

IRI is the standard empirical model of the ionosphere (COSPAR/URSI). It gives
``foF2`` and ``hmF2`` at a point and time, which :mod:`muf.geometry` turns into
an oblique MUF for the path.

IRI itself is Fortran. Two Python routes are supported, tried in order:

``PyIRI``
    Pure Python, no compiler needed. ``pip install PyIRI``
``iri2016``
    Wraps the official Fortran. More faithful, needs a Fortran toolchain.
    ``pip install iri2016``

Neither is a hard dependency; without one this reference reports why it is
unavailable and the run continues.

A caveat that matters for recent dates: IRI is normally driven by a *smoothed*
solar index, and R12 does not exist until six months after the fact -- see
:mod:`muf.reference.indices`. For a February 2026 sounding the driver is
necessarily an approximation, so treat IRI here as indicative rather than
authoritative.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry import Point, great_circle_km, fof2_to_muf, midpoint
from . import ReferenceSeries, as_index
from .indices import IndexUnavailable, solar_indices

INSTALL_HINT = (
    "IRI needs one of: `pip install PyIRI` (pure Python) or "
    "`pip install iri2016` (wraps the Fortran, needs a Fortran compiler)."
)


def _backend():
    """Return ``(name, module)`` for whichever IRI binding is installed."""
    try:
        import PyIRI  # type: ignore
        import PyIRI.main_library  # type: ignore
        return "PyIRI", PyIRI
    except ImportError:
        pass
    try:
        import iri2016  # type: ignore
        return "iri2016", iri2016
    except ImportError:
        pass
    return None, None


def available() -> bool:
    return _backend()[0] is not None


def _coeff_dir(module) -> str:
    """Where PyIRI keeps its CCIR/URSI coefficient files."""
    import os

    explicit = getattr(module, "coeff_dir", None)
    if explicit and os.path.isdir(explicit):
        return explicit
    return os.path.join(os.path.dirname(module.__file__), "coefficients")


def _pyiri_day(module, day: dt.date, hours: np.ndarray, point: Point,
               f107: float) -> tuple[np.ndarray, np.ndarray]:
    """foF2 and hmF2 for one date at many UT hours.

    One call covers the whole day: ``IRI_density_1day`` takes an array of hours,
    so a 288-sounding day costs one evaluation rather than 288.
    """
    from PyIRI import main_library as lib

    f2 = lib.IRI_density_1day(
        day.year, day.month, day.day,
        np.asarray(hours, dtype=float),
        np.array([point.lon], dtype=float),
        np.array([point.lat], dtype=float),
        np.array([250.0, 300.0, 350.0]),      # F2 params do not depend on this
        float(f107),
        _coeff_dir(module),
        0,                                     # 0 = CCIR coefficients
    )[0]

    fof2 = np.asarray(f2["fo"], dtype=float).reshape(len(hours), -1)[:, 0]
    hmf2 = np.asarray(f2["hm"], dtype=float).reshape(len(hours), -1)[:, 0]
    return fof2, hmf2


def _fof2_iri2016(module, when: dt.datetime, point: Point) -> float:
    """foF2 from the Fortran IRI at one time and place."""
    profile = module.IRI(when, (100, 1000, 50), point.lat, point.lon)
    value = profile["NmF2"] if "NmF2" in profile else None
    if value is None:
        return float("nan")
    nmf2 = float(np.asarray(value).ravel()[0])       # electrons per m^3
    if not np.isfinite(nmf2) or nmf2 <= 0:
        return float("nan")
    # Plasma frequency: f[Hz] = 8.98 * sqrt(Ne[m^-3])
    return 8.98e-6 * np.sqrt(nmf2)


def _hmf2_iri2016(module, when: dt.datetime, point: Point) -> float:
    try:
        profile = module.IRI(when, (100, 1000, 50), point.lat, point.lon)
        return float(np.asarray(profile["hmF2"]).ravel()[0])
    except Exception:
        return float("nan")


def predict(
    tx: Point,
    rx: Point,
    times,
    cache_dir: Path | None = None,
    offline: bool = False,
    f107: float | None = None,
    **_,
) -> ReferenceSeries:
    """MUF for the path, from IRI's foF2 at the control point."""
    index = as_index(times)
    if not len(index):
        return ReferenceSeries("iri", error="no timestamps given")

    name, module = _backend()
    if name is None:
        return ReferenceSeries("iri", error=f"not installed. {INSTALL_HINT}")

    path_km = great_circle_km(tx, rx)
    control = midpoint(tx, rx)

    driver_note = ""
    if f107 is None:
        try:
            si = solar_indices(index[0].date(), cache_dir=cache_dir, offline=offline)
            # `f107_driver`, not `f107`: the CCIR and URSI maps PyIRI
            # interpolates were fitted against a smoothed index, so the 81-day
            # mean is the number they were built for. The daily flux is a
            # better description of *today* and a worse input to *these maps* --
            # it would swing foF2 across a solar rotation in a way the
            # climatology never claimed to predict.
            f107 = si.f107_driver
            driver_note = f"; driver {si.source}"
            if f107 is None and si.ssn_daily is not None:
                # Rough SSN -> F10.7 (Holland & Vaughn); only a fallback.
                f107 = 63.7 + 0.728 * si.ssn_daily + 0.00089 * si.ssn_daily ** 2
                driver_note += " (F10.7 estimated from SSN)"
        except IndexUnavailable as exc:
            return ReferenceSeries("iri", error=f"no solar driver: {exc}")
    if f107 is None:
        return ReferenceSeries("iri", error="no solar driver available")

    values = np.full(len(index), np.nan)
    heights = np.full(len(index), np.nan)

    if name == "PyIRI":
        # Grouped by date so each day costs one model evaluation.
        dates = pd.Series(index.date, index=range(len(index)))
        for day, positions in dates.groupby(dates).groups.items():
            rows = np.asarray(positions, dtype=int)
            hours = np.array([
                index[i].hour + index[i].minute / 60 + index[i].second / 3600
                for i in rows
            ])
            try:
                fof2, hmf2 = _pyiri_day(module, day, hours, control, float(f107))
                values[rows], heights[rows] = fof2, hmf2
            except Exception as exc:
                driver_note = driver_note or f"; {type(exc).__name__} on {day}"
    else:
        for position, when in enumerate(index):
            stamp = when.to_pydatetime()
            try:
                values[position] = _fof2_iri2016(module, stamp, control)
                heights[position] = _hmf2_iri2016(module, stamp, control)
            except Exception as exc:        # one bad sample must not kill the run
                driver_note = driver_note or f"; {type(exc).__name__} on some samples"

    fof2_series = pd.Series(values, index=index, dtype=float)
    if not fof2_series.notna().any():
        return ReferenceSeries(
            "iri", error=f"{name} returned no usable foF2 for {control}"
        )

    from ..geometry import DEFAULT_HMF2_KM
    hmf2_series = pd.Series(heights, index=index, dtype=float).fillna(DEFAULT_HMF2_KM)

    muf = pd.Series(
        [fof2_to_muf(f, path_km, h) if np.isfinite(f) else np.nan
         for f, h in zip(fof2_series, hmf2_series)],
        index=index,
    )

    return ReferenceSeries(
        name="iri",
        muf=muf,
        detail=pd.DataFrame({"fof2": fof2_series, "hmf2": hmf2_series}),
        source=f"{name} at control point {control}, F10.7={f107:.0f}{driver_note}",
    )
