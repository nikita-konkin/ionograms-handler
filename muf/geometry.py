"""Path geometry and the vertical-to-oblique MUF conversion.

Reference models predict or measure the ionosphere *above a point* -- usually as
``foF2``, the F2-layer critical frequency, which is the highest frequency
returned at vertical incidence. Comparing that against an oblique sounding needs
two things: the point over which the path reflects, and the factor relating
vertical to oblique.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0

#: Height of the F2 reflection point when nothing better is known. Real hmF2
#: varies roughly 250-400 km; DIDBase reports it per sounding, and IRI models
#: it, so this is only the fallback.
DEFAULT_HMF2_KM = 300.0

#: Beyond this the path needs more than one hop, and a single midpoint control
#: point stops being appropriate.
MAX_SINGLE_HOP_KM = 4000.0


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float

    def __str__(self) -> str:
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"{abs(self.lat):.2f}{ns} {abs(self.lon):.2f}{ew}"


def great_circle_km(a: Point, b: Point) -> float:
    """Distance along the surface, in km."""
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = p2 - p1
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def midpoint(a: Point, b: Point) -> Point:
    """Great-circle midpoint -- the control point for a single-hop path."""
    p1, l1 = math.radians(a.lat), math.radians(a.lon)
    p2, l2 = math.radians(b.lat), math.radians(b.lon)

    bx = math.cos(p2) * math.cos(l2 - l1)
    by = math.cos(p2) * math.sin(l2 - l1)
    lat = math.atan2(math.sin(p1) + math.sin(p2),
                     math.sqrt((math.cos(p1) + bx) ** 2 + by ** 2))
    lon = l1 + math.atan2(by, math.cos(p1) + bx)
    return Point(math.degrees(lat), math.degrees(math.atan2(math.sin(lon),
                                                            math.cos(lon))))


def control_points(a: Point, b: Point) -> list[Point]:
    """Points whose ionosphere governs the path.

    One midpoint for a single hop. Longer paths reflect more than once and are
    limited by the worst control point, conventionally taken 2000 km in from
    each end -- the convention MINIMUF and the CCIR methods use.
    """
    if great_circle_km(a, b) <= MAX_SINGLE_HOP_KM:
        return [midpoint(a, b)]
    return [intermediate(a, b, 2000.0), intermediate(b, a, 2000.0)]


def hop_count(path_km: float) -> int:
    """How many F2 reflections a path of this length takes.

    The companion to :func:`control_points`: that one says *where* to sample
    the ionosphere, this one says what distance the sample then applies to.
    :func:`m_factor` converts foF2 to an oblique MUF for **one hop**, so a
    multi-hop path is converted at ``D/n``, not at ``D``.

    Passing the whole distance does not merely lose accuracy, it describes a
    ray that does not exist: the geometric factor peaks at 3840 km (M = 3.373
    for a 300 km layer) and falls away after, because past that the reflection
    would have to happen below the horizon. An 8000 km path evaluated whole
    gives 2.665 where its two real hops give 3.370 -- a MUF understated by a
    fifth, which reads as the instrument over-picking rather than the model
    being asked the wrong question.
    """
    if path_km <= MAX_SINGLE_HOP_KM:
        return 1
    return math.ceil(path_km / MAX_SINGLE_HOP_KM)


def describe_path(a: Point, b: Point) -> str:
    """Where a reference model was evaluated, for SAO ``ModelOptions``.

    One phrasing, so the CLI's export and the server's cannot describe the same
    circuit differently -- both used to format a bare midpoint themselves, and
    a midpoint is the wrong answer for a path that hops twice.
    """
    path_km = great_circle_km(a, b)
    points = control_points(a, b)
    hops = hop_count(path_km)

    label = "control point" if len(points) == 1 else "control points"
    where = " / ".join(str(point) for point in points)
    return (f"{label} {where}, D={path_km:.0f}km"
            + ("" if hops == 1 else f", {hops} hops"))


def intermediate(a: Point, b: Point, distance_km: float) -> Point:
    """The point ``distance_km`` from ``a`` along the great circle toward ``b``."""
    total = great_circle_km(a, b)
    if total == 0:
        return a
    f = max(0.0, min(1.0, distance_km / total))

    d = total / EARTH_RADIUS_KM
    p1, l1 = math.radians(a.lat), math.radians(a.lon)
    p2, l2 = math.radians(b.lat), math.radians(b.lon)

    sd = math.sin(d)
    if sd == 0:
        return a
    x = math.sin((1 - f) * d) / sd
    y = math.sin(f * d) / sd

    xc = x * math.cos(p1) * math.cos(l1) + y * math.cos(p2) * math.cos(l2)
    yc = x * math.cos(p1) * math.sin(l1) + y * math.cos(p2) * math.sin(l2)
    zc = x * math.sin(p1) + y * math.sin(p2)

    return Point(math.degrees(math.atan2(zc, math.hypot(xc, yc))),
                 math.degrees(math.atan2(yc, xc)))


def m_factor(path_km: float, hmf2_km: float = DEFAULT_HMF2_KM) -> float:
    """Vertical-to-oblique conversion factor, ``MUF = foF2 * M``.

    The secant law: a ray reaching the layer at incidence angle ``phi`` is
    returned up to ``foF2 / cos(phi)``. For a curved Earth and a reflection
    height ``hmF2``, the half-path subtends ``d/2`` at the centre and

        tan(phi) = R sin(d/2R) / (R + h - R cos(d/2R))

    which reduces to the flat-Earth ``sqrt(1 + (D/2h)^2)`` for short paths.

    This is the geometric factor only. Real M(3000)F2 is a little lower because
    the ray refracts through a thick layer rather than reflecting off a mirror
    at one height; the deviation is a few percent for oblique paths.
    """
    if path_km <= 0:
        return 1.0

    half = path_km / 2.0
    theta = half / EARTH_RADIUS_KM        # half-path central angle, radians
    numerator = EARTH_RADIUS_KM * math.sin(theta)
    denominator = (EARTH_RADIUS_KM + hmf2_km) - EARTH_RADIUS_KM * math.cos(theta)
    if denominator <= 0:
        return 1.0

    phi = math.atan2(numerator, denominator)
    cos_phi = math.cos(phi)
    if cos_phi <= 1e-6:
        return 1.0
    return 1.0 / cos_phi


def fof2_to_muf(fof2_mhz: float, path_km: float,
                hmf2_km: float = DEFAULT_HMF2_KM) -> float:
    """Oblique MUF implied by a vertical-incidence critical frequency."""
    return fof2_mhz * m_factor(path_km, hmf2_km)


def muf_to_fof2(muf_mhz: float, path_km: float,
                hmf2_km: float = DEFAULT_HMF2_KM) -> float:
    """The inverse: what vertical foF2 a measured oblique MUF implies."""
    factor = m_factor(path_km, hmf2_km)
    return muf_mhz / factor if factor else muf_mhz


def path_of(header) -> tuple[Point, Point, float]:
    """Transmitter, receiver and path length from an :class:`LfsHeader`."""
    tx = Point(header.tx_latitude, header.tx_longitude)
    rx = Point(header.rx_latitude, header.rx_longitude)
    return tx, rx, great_circle_km(tx, rx)
