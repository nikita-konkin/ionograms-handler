"""Fitting a curve to the ionogram trace.

Approaching the MUF, the low-ray and high-ray branches of an oblique trace
converge and ``dh/df`` diverges -- the "nose". Near it the trace is locally

    f = f_MUF - A (h - h_v)^2

so a parabola fitted in the ``(virtual range, frequency)`` plane has its vertex
at the MUF. This is the idea behind OIASA (Ippolito et al., *J. Space Weather
Space Clim.* 8, A10, 2018), which slides parabola pairs across the ionogram and
reads MUF from the vertex.

**Feed it both branches.** Below the MUF two rays arrive at every frequency and
converge at the nose. Pass :func:`muf.trace.nose_points`, which supplies both,
and the vertex is bracketed by data instead of extrapolated off one side. That
roughly halves the error:

======================================  ===============  ===============
against the extractors' pick            one branch       both branches
======================================  ===============  ===============
bias                                    -0.12 MHz        **-0.05 MHz**
mean absolute error                     0.74 MHz         **0.37 MHz**
median residual                         0.31 MHz         **0.18 MHz**
======================================  ===============  ===============

An earlier version of this note claimed only the low-ray branch was visible in
these ionograms. That was wrong: the high ray is plainly there, and the trace
module was separating it into its own track and discarding it.

*Also useful for outlier detection.* Where the vertex disagrees with a pick by
more than 3 MHz, the pick is wrong -- every such flag on 2026-02-04 was
independently rejected by :mod:`muf.track` on temporal grounds. It now flags
less often than it used to, because it declines on marginal soundings rather
than guessing; the tracker is the primary outlier mechanism.

*Not a reliability filter.* ``rms_residual_mhz`` separates smooth traces from
ragged ones, but does not predict whether the pick is right -- filtering on it
made agreement between extractors slightly worse. A wrong pick usually has an
excellent residual: the trace fits a clean parabola, the pick just landed on the
wrong part of it.

*Not a way past the band edge.* Of soundings whose pick reached the top of the
sweep, none yield a usable extrapolation -- when the trace runs into the band
limit the nose was never reached, so the fit declines, correctly.

Fitting the whole trace rather than the nose is far worse (residuals to 5 MHz),
because most of it is nearly flat in range and swamps the nose.
:func:`fit_nose` declines rather than returning a number it cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extractors import MufResult
from .spectro import Ionogram

#: Frequency span below the trace's top used for the fit.
#:
#: 1 MHz once :func:`muf.trace.nose_points` supplies *both* branches: the vertex
#: is then bracketed by data and a tight window sits on the cusp itself. A wider
#: window pulls in the flat body of the trace and pulls the vertex low --
#: measured residuals roughly double between 1.0 and 3.0 MHz. Before branch
#: merging, only one side was available and 3.0 MHz was needed for stability.
DEFAULT_WINDOW_MHZ = 1.0

DEFAULT_MIN_POINTS = 6

#: Above this RMS residual the trace is not describing a smooth nose -- usually
#: interference or two merged propagation modes.
DEFAULT_MAX_RESIDUAL_MHZ = 1.0

#: Fraction of the worst-fitting points dropped and the fit repeated, once.
TRIM_FRACTION = 0.15

#: A vertex further than this beyond the observed data is extrapolating too far
#: to be believed.
DEFAULT_MAX_EXTRAPOLATION_MHZ = 3.0


@dataclass(frozen=True)
class TraceFit:
    """Outcome of fitting the nose of a trace."""

    muf_mhz: float                 # vertex frequency; NaN when the fit declined
    vrange_km: float               # virtual range at the vertex
    curvature: float               # quadratic coefficient; negative for a nose
    rms_residual_mhz: float
    n_points: int
    observed_max_mhz: float        # highest frequency actually detected
    extrapolation_mhz: float       # vertex minus observed max
    ok: bool
    reason: str = ""

    @property
    def is_extrapolated(self) -> bool:
        return self.ok and self.extrapolation_mhz > 0.0

    def __str__(self) -> str:
        if not self.ok:
            return f"TraceFit(declined: {self.reason})"
        return (f"TraceFit(MUF {self.muf_mhz:.2f} MHz @ {self.vrange_km:.0f} km, "
                f"residual {self.rms_residual_mhz:.2f}, "
                f"extrapolated {self.extrapolation_mhz:+.2f}, n={self.n_points})")


NO_FIT = TraceFit(np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, False,
                  "not attempted")


def trace_points(ion: Ionogram, result: MufResult) -> tuple[np.ndarray, np.ndarray]:
    """``(frequency, virtual range)`` of the trace, one point per detected bin.

    Where the estimator produced a mask the range is the power-weighted centroid
    of the masked cells, which is steadier than a bare peak; otherwise it is the
    strongest cell in that frequency bin.
    """
    indices = np.flatnonzero(result.presence)
    if indices.size == 0:
        return np.empty(0), np.empty(0)

    ranges = np.empty(indices.size)
    for position, i in enumerate(indices):
        cells = np.flatnonzero(result.mask[i]) if result.mask is not None \
            else np.empty(0, dtype=int)
        if cells.size:
            weights = np.maximum(ion.db[i, cells], 0.0)
            total = weights.sum()
            ranges[position] = (np.average(ion.vrange[cells], weights=weights)
                                if total > 0 else ion.vrange[cells].mean())
        else:
            ranges[position] = ion.vrange[int(ion.db[i].argmax())]

    return ion.freq[indices], ranges


def _parabola(vrange: np.ndarray, freq: np.ndarray):
    """Least-squares ``f = a h^2 + b h + c``; returns coefficients and vertex."""
    a, b, c = np.polyfit(vrange, freq, 2)
    if a >= 0 or not np.isfinite(a):
        return None
    vertex_range = -b / (2 * a)
    vertex_freq = c - b * b / (4 * a)
    residual = float(np.sqrt(np.mean((np.polyval([a, b, c], vrange) - freq) ** 2)))
    return a, vertex_freq, vertex_range, residual


def fit_nose(
    freq: np.ndarray,
    vrange: np.ndarray,
    window_mhz: float = DEFAULT_WINDOW_MHZ,
    min_points: int = DEFAULT_MIN_POINTS,
    max_residual_mhz: float = DEFAULT_MAX_RESIDUAL_MHZ,
    max_extrapolation_mhz: float = DEFAULT_MAX_EXTRAPOLATION_MHZ,
) -> TraceFit:
    """Fit the top of a trace and read the MUF off the vertex.

    Declines -- ``ok=False`` with a reason -- rather than returning a number it
    cannot support. Every gate below corresponds to a way the fit is known to
    fail on real soundings.
    """
    freq = np.asarray(freq, dtype=float)
    vrange = np.asarray(vrange, dtype=float)
    if freq.size < min_points:
        return TraceFit(np.nan, np.nan, np.nan, np.nan, int(freq.size), np.nan,
                        np.nan, False, f"only {freq.size} trace points")

    observed_max = float(freq.max())
    near_nose = freq >= observed_max - window_mhz
    if near_nose.sum() < min_points:
        return TraceFit(np.nan, np.nan, np.nan, np.nan, int(near_nose.sum()),
                        observed_max, np.nan, False,
                        f"only {int(near_nose.sum())} points within "
                        f"{window_mhz} MHz of the top")

    f_nose, h_nose = freq[near_nose], vrange[near_nose]
    if np.ptp(h_nose) < 1e-6:
        return TraceFit(np.nan, np.nan, np.nan, np.nan, int(f_nose.size),
                        observed_max, np.nan, False,
                        "trace is flat in range; no nose to fit")

    fitted = _parabola(h_nose, f_nose)
    if fitted is None:
        return TraceFit(np.nan, np.nan, np.nan, np.nan, int(f_nose.size),
                        observed_max, np.nan, False,
                        "parabola opens upward; no maximum")

    # One robust pass: drop the worst-fitting points and refit.
    a, vertex_freq, vertex_range, residual = fitted
    if f_nose.size >= min_points * 2:
        errors = np.abs(np.polyval(np.polyfit(h_nose, f_nose, 2), h_nose) - f_nose)
        keep = errors <= np.quantile(errors, 1.0 - TRIM_FRACTION)
        if keep.sum() >= min_points and np.ptp(h_nose[keep]) > 1e-6:
            refitted = _parabola(h_nose[keep], f_nose[keep])
            if refitted is not None:
                a, vertex_freq, vertex_range, residual = refitted
                f_nose, h_nose = f_nose[keep], h_nose[keep]

    extrapolation = float(vertex_freq - observed_max)
    common = dict(
        muf_mhz=float(vertex_freq), vrange_km=float(vertex_range),
        curvature=float(a), rms_residual_mhz=residual,
        n_points=int(f_nose.size), observed_max_mhz=observed_max,
        extrapolation_mhz=extrapolation,
    )

    if residual > max_residual_mhz:
        return TraceFit(**common, ok=False,
                        reason=f"residual {residual:.2f} MHz exceeds "
                               f"{max_residual_mhz} -- trace is not a clean nose")
    if extrapolation > max_extrapolation_mhz:
        return TraceFit(**common, ok=False,
                        reason=f"vertex is {extrapolation:.2f} MHz beyond the "
                               f"data; too far to extrapolate")
    if extrapolation < -window_mhz:
        return TraceFit(**common, ok=False,
                        reason=f"vertex {abs(extrapolation):.2f} MHz below the "
                               f"observed top; the nose was not reached")

    return TraceFit(**common, ok=True)


def fit_result(
    ion: Ionogram,
    result: MufResult,
    **options,
) -> TraceFit:
    """Fit the nose of an estimator's trace. Convenience over the two steps."""
    if not result.ok:
        return TraceFit(np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan,
                        False, "estimator produced no pick")
    freq, vrange = trace_points(ion, result)
    return fit_nose(freq, vrange, **options)
