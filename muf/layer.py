"""A quasi-parabolic layer and the oblique trace it returns.

A trace tracked point by point can step from one echo onto the next: on a
long path the echoes between 1F and 3F are often stacked tens of km apart,
and nothing in a smoothing spline knows which shape a trace can take. A layer
does. Its rays at the circuit's ground range give a low ray rising to the
nose at the MUF and a high ray falling away from it, and neither can step
between echoes, so a layer laid on the sounding's echoes draws a 1F trace
that stays on one.

The layer is quasi-parabolic over a spherical Earth (Croft and Hoogasian,
*Radio Science* 3, 69-74, 1968):

    fN^2(r) = fc^2 [1 - ((r - rm) / ym)^2 (rb / r)^2],   rb = rm - ym

with no magnetic field. Its ground range and group path have a closed form:
with ``a = R cos(elevation)``, ``r^2 mu^2 - a^2 = A r^2 + B r + C``, and the
integrals of ``a / (r sqrt(.))`` and ``r / sqrt(.)`` from the layer's base to
the ray's apex are logarithms. The free-space legs below the base add
``2 R (gamma - elevation)`` and ``2 (rb sin gamma - R sin elevation)``, with
``cos gamma = a / rb``.

Both depend on frequency only through ``f / fc``, so the MUF scales with
``fc`` and a layer's trace at one critical frequency is every layer's of the
same height and thickness, rescaled in frequency. The fit uses that.

**The fit.** A sounding's echo points are scored against a layer by the echo
weight lying within :data:`REACH_KM` of either ray, at the range offset that
scores best: the station's clock moves every echo together, so the offset is
free. The MUF is searched within :data:`MUF_SPAN` of the extractor's pick,
the peak height and the thickness on a grid, then on a finer one about the
best. The offset trades against the peak height, which moves the whole trace
too, so the fitted layer is a curve that fits, not a measured profile, until
the station's timing holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import EARTH_RADIUS_KM

#: Launch elevations a trace is solved at, radians: a long path's low ray
#: leaves a few degrees up, its high ray some tens.
ELEVATIONS = np.radians(np.arange(0.2, 45.0, 0.05))

#: Bisection steps for the frequency that lands a ray at the ground range.
STEPS = 50

#: How near the ground range a ray has to land to count, km. A steep ray
#: reaches far only within a sliver of frequency below the one that goes
#: through the layer, where the range grows as a logarithm; past what the
#: bisection can resolve it lands short, and it is left out.
LANDED_KM = 0.05

#: Peak heights and semi-thicknesses the fit starts from, km.
HEIGHTS_KM = np.arange(150.0, 501.0, 10.0)
THICKNESSES_KM = np.arange(30.0, 201.0, 10.0)

#: The lowest base the fit allows, km: below it is the E region.
BASE_KM = 80.0

#: MUFs tried about the extractor's pick, as a share of it.
MUF_SPAN = 0.03
MUF_STEP = 0.005

#: Half-width of the window a point counts within, km: twice the scatter
#: of echo points about their track.
REACH_KM = 15.0

#: Range offsets allowed, km either way, and the bins they are found on.
OFFSET_KM = 500.0
BIN_KM = 1.0


@dataclass(frozen=True)
class Layer:
    """One quasi-parabolic layer: critical frequency, peak height, semi-thickness."""

    fc_mhz: float
    hm_km: float
    ym_km: float

    def ray(self, f_mhz, elevation) -> tuple[np.ndarray, np.ndarray]:
        """Ground range and group path, km, of rays launched at ``elevation`` radians.

        NaN for a ray that goes through the layer.
        """
        f = np.asarray(f_mhz, dtype=np.float64)
        rise = np.asarray(elevation, dtype=np.float64)
        rm = EARTH_RADIUS_KM + self.hm_km
        rb = rm - self.ym_km
        a = EARTH_RADIUS_KM * np.cos(rise)
        k = (self.fc_mhz / f) ** 2 * (rb / self.ym_km) ** 2
        quad = 1.0 - (self.fc_mhz / f) ** 2 + k
        lin = -2.0 * rm * k
        const = k * rm ** 2 - a ** 2
        disc = lin ** 2 - 4.0 * quad * const
        apex = (-lin - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * quad)
        back = (disc > 0) & (apex > rb) & (apex <= rm)
        sin_g = np.sqrt(np.maximum(1.0 - (a / rb) ** 2, 0.0))
        gamma = np.arccos(np.clip(a / rb, -1.0, 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            root_c, root_q = np.sqrt(const), np.sqrt(quad)
            entry = sin_g + root_c / rb + lin / (2.0 * root_c)
            ground = 2.0 * EARTH_RADIUS_KM * (
                gamma - rise
                - a / (2.0 * root_c) * np.log(disc / (4.0 * const * entry ** 2)))
            group = 2.0 * (rb * sin_g - EARTH_RADIUS_KM * np.sin(rise)) + 2.0 / quad * (
                -rb * sin_g - lin / (4.0 * root_q) * np.log(
                    disc / (2.0 * quad * rb + lin + 2.0 * rb * root_q * sin_g) ** 2))
        return np.where(back, ground, np.nan), np.where(back, group, np.nan)

    def trace(self, ground_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The rays landing at ``ground_km``: frequency MHz, group path km, elevation.

        In order of elevation, so the low ray up to the nose comes first and
        the high ray after it. At each elevation the frequency is bisected:
        a higher one reflects higher and lands further, and one through the
        layer lands nowhere.
        """
        lo = np.full(ELEVATIONS.shape, 0.05 * self.fc_mhz)
        hi = np.full(ELEVATIONS.shape, 30.0 * self.fc_mhz)
        short = self.ray(lo, ELEVATIONS)[0] < ground_km
        for _ in range(STEPS):
            mid = 0.5 * (lo + hi)
            far = ~(self.ray(mid, ELEVATIONS)[0] < ground_km)
            hi = np.where(far, mid, hi)
            lo = np.where(far, lo, mid)
        f = 0.5 * (lo + hi)
        ground, group = self.ray(f, ELEVATIONS)
        keep = short & (np.abs(ground - ground_km) < LANDED_KM)
        return f[keep], group[keep], ELEVATIONS[keep]


@dataclass(frozen=True)
class Fit:
    """A layer laid on a sounding's echoes, and the range offset that lays it there."""

    layer: Layer
    ground_km: float
    offset_km: float
    score: float

    def low(self) -> tuple[np.ndarray, np.ndarray]:
        """The low ray up to the nose: frequency MHz ascending, range km."""
        f, group, _ = self.layer.trace(self.ground_km)
        nose = int(np.argmax(f))
        return f[:nose + 1], group[:nose + 1] + self.offset_km

    def high(self) -> tuple[np.ndarray, np.ndarray]:
        """The high ray up to the nose: frequency MHz ascending, range km."""
        f, group, _ = self.layer.trace(self.ground_km)
        nose = int(np.argmax(f))
        return f[nose:][::-1], group[nose:][::-1] + self.offset_km


def fit(freq: np.ndarray, vrange: np.ndarray, weight: np.ndarray,
        ground_km: float, muf_mhz: float) -> Fit:
    """The layer whose rays at ``ground_km`` lie on the most echo weight.

    ``freq``, ``vrange`` and ``weight`` are echo points, one per echo per
    row, as :func:`muf.trace.extract_points` gives them; ``muf_mhz`` is the
    extractor's pick.
    """
    if np.size(freq) == 0:
        raise ValueError("cannot fit a layer to a sounding without echo points")
    points = (np.asarray(freq, dtype=np.float64), np.asarray(vrange, dtype=np.float64),
              np.asarray(weight, dtype=np.float64))
    shares = np.arange(-MUF_SPAN, MUF_SPAN + 1e-9, MUF_STEP)
    coarse = _best(points, ground_km,
                   [(h, y) for h in HEIGHTS_KM for y in THICKNESSES_KM],
                   muf_mhz * (1.0 + shares))
    fine = [(coarse.layer.hm_km + dh, coarse.layer.ym_km + dy)
            for dh in np.arange(-8.0, 9.0, 2.0) for dy in np.arange(-8.0, 9.0, 2.0)]
    muf = float(np.max(coarse.layer.trace(ground_km)[0]))
    return _best(points, ground_km, fine, muf * (1.0 + shares / 5.0))


def _best(points, ground_km: float, shapes, mufs) -> Fit:
    best = None
    for hm, ym in shapes:
        if hm - ym < BASE_KM or ym <= 0:
            continue
        f, group, _ = Layer(1.0, hm, ym).trace(ground_km)
        if f.size < 3:
            continue
        nose = int(np.argmax(f))
        for muf in mufs:
            scale = muf / f[nose]
            score, offset = _scored(points, f * scale, group, nose)
            if best is None or score > best.score:
                best = Fit(Layer(scale, hm, ym), ground_km, offset, score)
    if best is None:
        raise ValueError(f"no layer returns a ray over {ground_km:.0f} km")
    return best


def _scored(points, f: np.ndarray, group: np.ndarray, nose: int) -> tuple[float, float]:
    """The echo weight within reach of either ray at the best offset, and the offset."""
    freq, vrange, weight = points
    residuals, weights = [], []
    for ray_f, ray_r in ((f[:nose + 1], group[:nose + 1]),
                         (f[nose:][::-1], group[nose:][::-1])):
        inside = (freq >= ray_f[0]) & (freq <= ray_f[-1])
        residuals.append(vrange[inside] - np.interp(freq[inside], ray_f, ray_r))
        weights.append(weight[inside])
    residual, mass = np.concatenate(residuals), np.concatenate(weights)
    edges = np.arange(-OFFSET_KM - REACH_KM, OFFSET_KM + REACH_KM + BIN_KM, BIN_KM)
    hist, _ = np.histogram(residual, bins=edges, weights=mass)
    lag = np.arange(-REACH_KM, REACH_KM + BIN_KM, BIN_KM)
    kernel = np.maximum(1.0 - (lag / REACH_KM) ** 2, 0.0)
    swept = np.convolve(hist, kernel, mode="same")
    centres = 0.5 * (edges[1:] + edges[:-1])
    allowed = np.abs(centres) <= OFFSET_KM
    k = int(np.argmax(np.where(allowed, swept, -np.inf)))
    return float(swept[k]), float(centres[k])
