"""A transparent solar-zenith model of the diurnal MUF variation.

Every constant here is either derived or stated, and the model is deliberately
simple. It exists to answer one question: **does the extracted MUF vary through
the day the way photochemistry says it should?**

An alpha-Chapman layer in photochemical equilibrium has peak density
``NmF2 ~ sqrt(cos X)`` for solar zenith angle ``X``, and plasma frequency goes
as the square root of density, so

    foF2 ~ (cos X)^0.25

That quarter-power law is the whole content of the model. Night is handled with
a floor, because the F2 layer is maintained after dark by downward diffusion
rather than by photoionisation.

**What this can and cannot do.** Its amplitude is *fitted to the data it is
being compared against*, so it cannot corroborate a scale error -- if the
pipeline read every MUF 20% high, this would fit 20% high too and report perfect
agreement. It tests shape only. For an absolute reference use
:mod:`muf.reference.giro` (measurements) or :mod:`muf.reference.iri` (a real
model).

Reported as ``r2_shape`` rather than as a MUF prediction, to keep that
distinction visible.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry import Point, great_circle_km, midpoint
from . import ReferenceSeries, as_index

#: alpha-Chapman: Nm ~ sqrt(cos X), and f ~ sqrt(Nm), giving foF2 ~ cos(X)^0.25.
CHAPMAN_EXPONENT = 0.25

#: Night floor as a fraction of the noon value. The F2 layer persists after
#: dark; mid-latitude night foF2 typically runs 30-45% of the noon figure.
DEFAULT_NIGHT_FRACTION = 0.35


def solar_declination(when: dt.datetime) -> float:
    """Solar declination in radians (Cooper's approximation, ~0.5 deg)."""
    day = when.timetuple().tm_yday
    return math.radians(23.45) * math.sin(2 * math.pi * (284 + day) / 365.0)


def solar_zenith_cos(when: dt.datetime, point: Point) -> float:
    """``cos`` of the solar zenith angle. Negative when the sun is down."""
    declination = solar_declination(when)
    ut_hours = when.hour + when.minute / 60.0 + when.second / 3600.0

    # Local solar time from UT and longitude; 15 degrees per hour.
    hour_angle = math.radians(15.0 * (ut_hours - 12.0) + point.lon)
    latitude = math.radians(point.lat)

    return (math.sin(latitude) * math.sin(declination)
            + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle))


def shape(times, point: Point,
          night_fraction: float = DEFAULT_NIGHT_FRACTION) -> pd.Series:
    """Diurnal shape over ``times``: 1.0 at the daily peak, ``night_fraction`` at night.

    The daylight term is normalised by its own maximum over the window, so the
    curve reaches 1.0 at local noon whatever the season and latitude. Without
    that, the peak would only reach 1.0 with the sun directly overhead --
    at 46N in February it tops out near 0.87 -- and both parameters would mean
    something other than they say.
    """
    index = as_index(times)
    daylight = np.array([
        max(0.0, solar_zenith_cos(when.to_pydatetime(), point)) ** CHAPMAN_EXPONENT
        for when in index
    ])

    peak = daylight.max()
    if peak > 0:
        daylight = daylight / peak

    return pd.Series(night_fraction + (1.0 - night_fraction) * daylight,
                     index=index, dtype=float)


def predict(
    tx: Point,
    rx: Point,
    times,
    observed: pd.Series | None = None,
    night_fraction: float = DEFAULT_NIGHT_FRACTION,
    scale_mhz: float | None = None,
    **_,
) -> ReferenceSeries:
    """The diurnal shape, scaled to the observations it will be compared against.

    Args:
        observed: the measured MUF series. Its 95th percentile sets the scale.
            Without it, ``scale_mhz`` must be given.
        night_fraction: night MUF as a fraction of the noon value.
        scale_mhz: fix the noon value explicitly instead of fitting it.
    """
    index = as_index(times)
    if not len(index):
        return ReferenceSeries("chapman", error="no timestamps given")

    control = midpoint(tx, rx)
    curve = shape(index, control, night_fraction)

    fitted = ""
    if scale_mhz is None:
        if observed is None or not len(pd.Series(observed).dropna()):
            return ReferenceSeries(
                "chapman",
                error="needs `observed` to fit its amplitude, or an explicit "
                      "`scale_mhz` -- this model has no absolute scale of its own",
            )
        values = pd.Series(observed).dropna().astype(float)
        # 95th percentile rather than the max: robust to a stray high pick.
        scale_mhz = float(np.percentile(values, 95)) / float(curve.max())
        fitted = " (amplitude fitted to the data -- shape check only)"

    return ReferenceSeries(
        name="chapman",
        muf=curve * scale_mhz,
        detail=pd.DataFrame({"shape": curve}),
        source=(f"alpha-Chapman cos(X)^{CHAPMAN_EXPONENT} at {control}, "
                f"night fraction {night_fraction:.2f}{fitted}"),
    )
