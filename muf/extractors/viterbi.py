"""One continuous trace, found by dynamic programming.

The other three estimators decide **cell by cell** whether something is trace,
then hope a run emerges from the decisions. `algo` thresholds and checks
neighbours; `contour` thresholds and morphologically closes; `kmeans` clusters
the same cells. All three then hand a per-frequency presence array to
:func:`muf.pick.pick_muf`, which is where continuity is finally required -- and
by then the evidence that would have justified a gap is gone.

That ordering is what
``docs/2026-08-30-segmentation-quality.md`` measured going wrong. Detection
reached 28.45 MHz on the worst sounding and the picker returned 18.75, because
the high-frequency detections were real but too fragmented to assemble a run.
Bridging (:data:`muf.pick.DEFAULT_BRIDGE`) patched that with a fixed allowance
of two bins, which helps and is still arbitrary: two bins is right for the fades
that were measured and wrong for the next one.

**This module inverts the order.** The physics says the trace is a single
connected curve whose virtual range varies smoothly with frequency. That is not
a property to check afterwards; it is the thing being looked for. Stated as a
search -- pick one range bin per frequency, maximise total power, pay for
discontinuity -- it is a shortest-path problem, and the optimum is exact and
cheap by dynamic programming. The result **cannot fragment by construction**,
so there is nothing for a bridge parameter to repair.

This is what OIASA's maximum-contrast method does in spirit, and it is the
oblique/chirp-sounder state of the art. Like OIASA it needs no training data,
which matters here: there is not one labelled ionogram in the archive.

**The trace is one span, and that is load-bearing rather than a
simplification.** The first version of this module let the path leave the trace
and rejoin it later, which reads as the more general model. It is worse, and
measurably: rejoining is not constrained by the slope limit -- the gap resets
the range -- so a bright interferer anywhere in the array could be joined for
the price of leaving and returning. On a synthetic ionogram, six bins of
interference 500 km off the trace and 200 bins above its end were picked up as
the MUF. Generality bought a way to smuggle in exactly the discontinuity the
module exists to forbid.

So the trace starts once and ends once. Each cell scores what it is worth
against :data:`DEFAULT_THRESHOLD_DB` -- earning above it, costing below -- and
the best scoring span wins. This is Kadane's rule carried along a path: a
predecessor worth less than nothing is dropped rather than inherited, which is
the entire start rule, and the end is wherever the running total peaked.

Bridging then stops being a parameter and becomes an inference, with no
constant to pick. A fade is crossed when the trace beyond it is worth more than
the fade costs, and a trace ends when it is not. That is the argument for this
module over :data:`muf.pick.DEFAULT_BRIDGE` -- not that two is the wrong
number, but that no fixed number can be the right one.

**It follows that there is no upper bound on the gap this will cross**, and
that is a real difference in behaviour rather than a nicety. Measured on a
synthetic sounding, forty dead bins are crossed when six hundred bins of strong
trace lie beyond them; a fixed ``bridge=2`` would have stopped dead. What holds
it honest is that the range limit still applies *through* the gap -- the path
moves at most ``width`` bins per frequency whether or not anything is lit there
-- so the far side has to sit at a range the trace could plausibly have reached.
A gap is crossed on the strength and the position of what is past it, never on
its length alone. Whether that is right on real data is the thing to measure,
and it is why this is not a default.

Not enabled by default. `DEFAULT_METHODS` is unchanged, and every stored result
was produced without this; it joins the registry as ``dp`` so it can be run
alongside the others and compared. See
``docs/2026-08-30-segmentation-quality.md`` sec. 6a.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..pick import DEFAULT_MAX_RANGE_SLOPE, DEFAULT_MIN_RUN, pick_muf
from ..spectro import Ionogram
from . import MufResult

#: The level a cell must beat to be worth being on the trace for, in dB on
#: ``Ionogram.db``. Deliberately the same physical level as the other three
#: estimators (``contour.DEFAULT_THRESHOLD_DB``, ``algo``'s linear 20), so a
#: disagreement between them is about *method* and never about where the bar
#: was set. With ``spectro.NOISE_FLOOR_DB`` at 30, this is a 13 dB SNR.
#:
#: Here it is not a detection threshold but the zero of the scoring: a cell
#: above it earns its place on the trace, a cell below it costs. Nothing is
#: excluded by it, so a dim cell inside a fade can still be crossed when what
#: lies beyond pays for it. Thresholding is a consequence of the optimisation
#: rather than a step before it.
DEFAULT_THRESHOLD_DB = 43.0

#: Registry name.
NAME = "dp"


def trace(
    db: np.ndarray,
    freq: np.ndarray,
    vrange: np.ndarray,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    max_range_slope: float = DEFAULT_MAX_RANGE_SLOPE,
) -> tuple[np.ndarray, np.ndarray]:
    """Best single trace through ``db``.

    Args:
        db: ``[n_freq, n_range]`` power in dB, as ``Ionogram.db`` produces.
        freq: frequency axis in MHz, ascending.
        vrange: virtual-range axis in km.
        threshold_db: the level a cell must beat to be worth occupying; see
            :data:`DEFAULT_THRESHOLD_DB`.
        max_range_slope: km/MHz the trace may move between adjacent frequency
            bins. The same limit ``pick_muf`` applies, from the same constant,
            so the two modules cannot drift apart about what a real trace does.

    Returns:
        ``(present, range_index)``. ``present`` is bool per frequency bin;
        ``range_index`` is the chosen range bin there, and ``-1`` where absent.
    """
    n_freq, n_range = db.shape
    if n_freq == 0 or n_range == 0:
        return np.zeros(n_freq, bool), np.full(n_freq, -1, int)

    # How many range bins the trace may move per frequency step, straight from
    # the slope limit.
    #
    # Deliberately *not* `pick._range_tolerance`, which floors the allowance at
    # `RANGE_SLOPE_FLOOR_BINS` range bins. That floor is right where it lives:
    # it absorbs the jitter of peak-finding between two independent detections.
    # Here the same number would be a per-step *drift budget* on a path that
    # takes hundreds of steps, and it compounds. On a 20.5 kHz axis with 5 km
    # range bins it allowed 10 km per step -- 488 km/MHz against a stated limit
    # of 150, enough for the path to walk 650 km through noise and rejoin a
    # different feature, which is the exact discontinuity this module forbids.
    #
    # One bin is the floor instead, because a path that may not move at all is
    # a horizontal line rather than a trace. Where the range axis is coarser
    # than the allowed movement that floor still exceeds the limit, and that is
    # a property of the axis, not a choice made here.
    range_step = (float(np.median(np.abs(np.diff(vrange))))
                  if np.size(vrange) > 1 else 0.0)
    freq_step = (float(np.median(np.abs(np.diff(freq))))
                 if np.size(freq) > 1 else 0.0)
    width = (max(1, int(round(max_range_slope * freq_step / range_step)))
             if range_step else 1)

    # Every cell scores what it is worth *relative to not being on the trace*.
    # Above threshold earns, below threshold costs, and the zero is the same
    # physical level the other three estimators threshold at.
    gain = db.astype(np.float64) - threshold_db

    # Forward pass. `best[f, r]` is the score of the best trace that ends at
    # frequency f on range bin r. Carrying `max(carried, 0)` is what lets a
    # trace *start* at f: a predecessor worth less than nothing is dropped
    # rather than inherited. That single clamp is the whole of the start rule,
    # and the end rule is just the global argmax below.
    #
    # float64 because these accumulate over hundreds of frequency bins, and
    # float32 loses the ordering between near-equal paths.
    best = np.empty((n_freq, n_range), dtype=np.float64)
    best[0] = gain[0]
    for f in range(1, n_freq):
        # Best predecessor within `width` range bins: a sliding-window maximum,
        # so this is O(n_range) per frequency rather than the O(n_range^2) the
        # transition matrix implies.
        reachable = maximum_filter1d(best[f - 1], size=2 * width + 1,
                                     mode="constant", cval=-np.inf)
        best[f] = gain[f] + np.maximum(reachable, 0.0)

    present = np.zeros(n_freq, dtype=bool)
    chosen = np.full(n_freq, -1, dtype=int)
    if not np.isfinite(best).any() or best.max() <= 0:
        return present, chosen                  # nothing beat the noise

    # Backtrack from the best endpoint. The predecessor is recovered by
    # searching the window rather than stored: the window is a few bins wide
    # and this runs once per frequency, which is far cheaper than carrying an
    # [n_freq, n_range] index array alongside the scores.
    end = int(np.argmax(best))
    f, here = divmod(end, n_range)
    while f >= 0:
        present[f], chosen[f] = True, here
        if f == 0:
            break
        low = max(0, here - width)
        high = min(n_range, here + width + 1)
        window = best[f - 1][low:high]
        if window.max() <= 0:
            break                               # the trace started at f
        here = low + int(np.argmax(window))
        f -= 1

    return present, chosen


def extract(
    ion: Ionogram,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    max_range_slope: float = DEFAULT_MAX_RANGE_SLOPE,
    min_run: int = DEFAULT_MIN_RUN,
    percentile: float = 100.0,
) -> MufResult:
    """Estimate MUF from the best continuous trace.

    The picker still makes the final call, for the same reason the other three
    estimators defer to it: it is where sub-bin range interpolation and the
    reported SNR live, and a result that skipped it would not be comparable.

    But it is asked for none of its continuity work here -- no bridging, no
    slope test. Both are already satisfied exactly rather than approximately,
    and applying them twice would discard traces this module accepted on
    stronger evidence than the picker has access to. ``min_run`` is kept: the
    span this module returns is contiguous but may still be a single bright
    cell, and that is not a trace.
    """
    present, chosen = trace(ion.db, ion.freq, ion.vrange, threshold_db,
                            max_range_slope)

    mask = np.zeros(ion.db.shape, dtype=bool)
    lit = np.flatnonzero(present)
    mask[lit, chosen[lit]] = True

    pick = pick_muf(
        present, ion.freq,
        power_db=ion.db, vrange=ion.vrange,
        min_run=min_run, percentile=percentile,
        max_range_slope=None, bridge=0,
    )
    return MufResult(
        method=NAME, pick=pick, presence=present, mask=mask,
    )
