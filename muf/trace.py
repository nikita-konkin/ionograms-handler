"""Segmenting and reconstructing the ionogram trace.

An extractor returns the frequency bins where it found signal. That set is not a
curve: on this instrument it covers only 18-48% of the frequency span it reaches
across, carries 12-37 km of scatter, and -- the part that matters -- it is
usually **more than one propagation mode stitched together**.

Measured on 2026-02-04, the gaps in a trace fall into two clear classes:

===================  ==========================================================
change in range      what it is
===================  ==========================================================
+15 to +37 km        a fade inside one trace; safe to bridge
-130 to -180 km      a mode boundary; bridging it would be nonsense
===================  ==========================================================

Virtual range *falling* as frequency rises is backwards for a single trace --
range must rise toward the MUF -- so a large drop marks the point where a
different mode takes over. A 06:00 sounding has a 5.5 MHz gap with the range
dropping 176 km across it; a 12:00 sounding, 4.4 MHz and 132 km. The extractors'
continuity rule is satisfied straight across those, because it only looks at
frequency.

So this module segments first and reconstructs second:

1. :func:`segment` splits at range discontinuities.
2. :func:`identify_hops` labels each segment by how many hops its range implies.
3. :func:`reconstruct` fits a weighted smoothing spline to one segment, filling
   the small fades and suppressing scatter, without ever crossing a boundary.

The result is a continuous single-mode ``h(f)``, which is what an
electron-density inversion needs and what a sparse point set cannot provide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: A range change larger than this between neighbouring trace points marks a
#: mode boundary rather than a fade. Well above the 12-37 km scatter seen here,
#: well below the 130-180 km jumps that separate modes.
DEFAULT_MAX_JUMP_KM = 80.0

#: Points either side of a step used to judge it, so one noisy point cannot
#: manufacture a boundary.
DEFAULT_JUDGE_WINDOW = 5

#: The jump threshold is also required to be this many times the trace's own
#: scatter. Scatter varies three-fold between soundings here (12-37 km), and a
#: fixed threshold splits the noisiest traces on noise alone.
DEFAULT_JUMP_SIGMA = 4.0

#: Segments shorter than this are dropped: too little to reconstruct or trust.
DEFAULT_MIN_SEGMENT_POINTS = 8

#: How long a track may go unextended before it is closed. Generous, because
#: real traces fade for several MHz at a time -- a 06:00 sounding here has a
#: 7 MHz fade in the middle of one continuous echo, and closing the track there
#: chops it into fragments. What actually prevents wrong joins is the range
#: window (:data:`DEFAULT_MAX_JUMP_KM`): modes are separated by 130-180 km,
#: comfortably more than the 80 km a track may step.
DEFAULT_MAX_GAP_MHZ = 8.0

#: How the curve is fitted. ``spline`` is a weighted smoothing spline;
#: ``pchip`` and ``makima`` bin the points to medians first and interpolate
#: through those, which cannot overshoot. The literature on vertical ionogram
#: reconstruction (Adv. Space Res., 2025) compares exactly these.
DEFAULT_METHOD = "spline"

#: No output is produced further than this from a measured point. Without it a
#: smoothing spline will happily arc hundreds of km above the data across a
#: gap, which it did on a real 22:00 sounding with a 2.1 MHz hole.
DEFAULT_MAX_SUPPORT_GAP_MHZ = 1.0

#: Effective mirror heights scanned when matching a segment's range to a hop
#: count. Wider than the real hmF2 band (250-400 km) because virtual range
#: exceeds the true path -- the wave slows near reflection, so the equivalent
#: mirror sits above the density peak.
HOP_HEIGHTS_KM = (250.0, 300.0, 350.0, 400.0, 450.0, 500.0)

#: A segment further than this from every predicted hop range is left unlabelled
#: rather than forced onto the nearest one.
HOP_TOLERANCE_KM = 250.0

#: The best hop candidate must beat the runner-up by this much to be accepted.
#:
#: Hop identification is inherently weak on a short path. Over 2588 km a 1-hop
#: echo spans 2636-2774 km across plausible heights and a 2-hop echo 2774-3182:
#: the ranges touch, so many segments are genuinely ambiguous and are left
#: unlabelled. The discrimination sharpens considerably on longer paths, where
#: the hop families separate. Segmentation does not depend on this and remains
#: useful either way.
HOP_MARGIN_KM = 40.0


#: Two tracks are branches of one mode when their high-frequency ends meet:
#: within this much in frequency, and in range, of each other. Measured on real
#: noses the ends agree to 8-38 km, while genuinely different modes differ by
#: 160-225 km, so the criterion separates them comfortably.
BRANCH_TOP_GAP_MHZ = 1.0
BRANCH_RANGE_GAP_KM = 120.0

#: Range change per MHz below which a track is called flat rather than a
#: rising or falling branch.
BRANCH_SLOPE_FLOOR = 1.0


@dataclass
class Segment:
    """One contiguous propagation mode within a trace."""

    freq: np.ndarray                  # MHz, ascending
    vrange: np.ndarray                # km
    weight: np.ndarray | None = None  # relative confidence per point
    hops: int | None = None           # 1 for one-hop, 2 for two-hop, ...
    height_km: float | None = None    # reflection height implied by the match
    branch: str | None = None         # "low", "high" or None -- see merge_branches
    group: int | None = None          # nose group: branches of one mode share it

    @property
    def n_points(self) -> int:
        return len(self.freq)

    @property
    def freq_span(self) -> tuple[float, float]:
        return float(self.freq.min()), float(self.freq.max())

    @property
    def median_range(self) -> float:
        return float(np.median(self.vrange))

    def __str__(self) -> str:
        low, high = self.freq_span
        mode = f"{self.hops}-hop" if self.hops else "unlabelled"
        return (f"Segment({low:.2f}-{high:.2f} MHz, {self.n_points} points, "
                f"{self.median_range:.0f} km, {mode})")


@dataclass
class Reconstruction:
    """A segment resampled onto a regular frequency grid."""

    freq: np.ndarray                  # regular grid, MHz
    vrange: np.ndarray                # smoothed and gap-filled, km
    segment: Segment = field(repr=False)
    rms_residual_km: float = np.nan
    knots: np.ndarray | None = field(default=None, repr=False)
    ok: bool = True
    reason: str = ""

    @property
    def muf_mhz(self) -> float:
        """Top of the reconstructed segment."""
        return float(self.freq.max()) if self.ok and len(self.freq) else np.nan

    def __str__(self) -> str:
        if not self.ok:
            return f"Reconstruction(declined: {self.reason})"
        return (f"Reconstruction({self.freq.min():.2f}-{self.freq.max():.2f} MHz, "
                f"{len(self.freq)} points, residual {self.rms_residual_km:.1f} km)")


def trace_scatter_km(vrange: np.ndarray, window: int = 9) -> float:
    """Point-to-point scatter about the local trend, in km.

    Deviation from a short running median: follows the curve without following
    the noise, so it measures scatter rather than the trace's real slope.
    """
    from scipy.ndimage import median_filter

    vrange = np.asarray(vrange, dtype=float)
    if vrange.size < 3:
        return 0.0
    size = min(window, vrange.size | 1)
    return float(np.std(vrange - median_filter(vrange, size=size)))


def _local_median(values: np.ndarray, index: int, window: int,
                  forward: bool) -> float:
    """Median of up to ``window`` points from ``index``, one way or the other."""
    if forward:
        chunk = values[index:index + window]
    else:
        chunk = values[max(0, index - window + 1):index + 1]
    return float(np.median(chunk)) if len(chunk) else float(values[index])


def segment(
    freq: np.ndarray,
    vrange: np.ndarray,
    weight: np.ndarray | None = None,
    max_jump_km: float = DEFAULT_MAX_JUMP_KM,
    judge_window: int = DEFAULT_JUDGE_WINDOW,
    min_points: int = DEFAULT_MIN_SEGMENT_POINTS,
    jump_sigma: float = DEFAULT_JUMP_SIGMA,
) -> list[Segment]:
    """Split a trace into contiguous propagation modes.

    A boundary is declared where the local median range shifts by more than the
    threshold between neighbouring points. Local medians rather than raw values,
    so a single noisy point cannot split a good trace, and the threshold rises
    with the trace's own scatter, so a noisy sounding is not shredded by it.
    """
    freq = np.asarray(freq, dtype=float)
    vrange = np.asarray(vrange, dtype=float)
    if freq.size == 0:
        return []
    order = np.argsort(freq)
    freq, vrange = freq[order], vrange[order]
    if weight is not None:
        weight = np.asarray(weight, dtype=float)[order]

    threshold = max(max_jump_km, jump_sigma * trace_scatter_km(vrange))

    # Detect with local medians -- robust, but they straddle the boundary, so a
    # single discontinuity lights up several neighbouring indices. Each run of
    # them is one boundary, placed where the raw step is actually largest.
    strength = np.zeros(freq.size - 1)
    for i in range(freq.size - 1):
        before = _local_median(vrange, i, judge_window, forward=False)
        after = _local_median(vrange, i + 1, judge_window, forward=True)
        strength[i] = abs(after - before)

    firing = strength > threshold
    raw_step = np.abs(np.diff(vrange))

    boundaries = [0]
    start = None
    for i in range(freq.size):
        lit = bool(firing[i]) if i < firing.size else False
        if lit and start is None:
            start = i
        elif not lit and start is not None:
            run = np.arange(start, i)
            boundaries.append(int(run[np.argmax(raw_step[run])]) + 1)
            start = None
    if start is not None:
        run = np.arange(start, firing.size)
        boundaries.append(int(run[np.argmax(raw_step[run])]) + 1)
    boundaries.append(freq.size)

    segments = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start < min_points:
            continue
        segments.append(Segment(
            freq=freq[start:stop],
            vrange=vrange[start:stop],
            weight=None if weight is None else weight[start:stop],
        ))
    return segments


def hop_range_km(ground_km: float, hops: int, height_km: float) -> float:
    """Virtual range of an ``hops``-hop path over ``ground_km``.

    Each hop covers ``ground/hops`` and reflects at ``height``, so its path is
    ``sqrt((ground/hops)^2 + 4 height^2)``. Flat-Earth, which is ample for
    telling one hop from two.
    """
    if hops < 1:
        raise ValueError("hops must be at least 1")
    leg = ground_km / hops
    return hops * math.hypot(leg, 2.0 * height_km)


def identify_hops(
    segments: list[Segment],
    ground_km: float,
    max_hops: int = 4,
    tolerance_km: float = HOP_TOLERANCE_KM,
    margin_km: float = HOP_MARGIN_KM,
) -> list[Segment]:
    """Label each segment with the hop count its virtual range implies.

    A segment is labelled only when one hop count fits clearly better than the
    rest. Segments matching nothing within ``tolerance_km``, or sitting between
    two candidates without favouring either by ``margin_km``, are left
    unlabelled -- an honest "don't know" beats a coin-flip label that later
    analysis would take at face value.
    """
    for item in segments:
        observed = item.median_range

        # Best error for each hop count, minimised over plausible heights.
        by_hops = {
            hops: min(abs(observed - hop_range_km(ground_km, hops, height))
                      for height in HOP_HEIGHTS_KM)
            for hops in range(1, max_hops + 1)
        }
        ranked = sorted(by_hops.items(), key=lambda kv: kv[1])
        (best_hops, best_error) = ranked[0]
        runner_up_error = ranked[1][1] if len(ranked) > 1 else float("inf")

        if best_error > tolerance_km:
            continue
        if runner_up_error - best_error < margin_km:
            continue           # genuinely ambiguous for this geometry

        item.hops = best_hops
        item.height_km = min(
            HOP_HEIGHTS_KM,
            key=lambda h: abs(observed - hop_range_km(ground_km, best_hops, h)),
        )
    return segments


def slope_km_per_mhz(item: Segment) -> float:
    """How fast a track's range changes with frequency."""
    if item.n_points < 3 or np.ptp(item.freq) <= 0:
        return 0.0
    return float(np.polyfit(item.freq, item.vrange, 1)[0])


def merge_branches(
    segments: list[Segment],
    top_gap_mhz: float = BRANCH_TOP_GAP_MHZ,
    range_gap_km: float = BRANCH_RANGE_GAP_KM,
) -> list[Segment]:
    """Recognise the low-ray and high-ray branches of a single mode.

    Below the MUF two rays reach the receiver at every frequency: the low ray,
    whose virtual range rises gently with frequency, and the high ray, whose
    range falls steeply. They converge at the nose, and the frequency where they
    meet *is* the MUF.

    Track-following treats them as two modes, because they are separated in
    range. That has two costs: the "primary" track comes out as the high ray --
    a short steep fragment near the nose rather than the main trace -- and the
    nose fit in :mod:`muf.fit` sees only one side of the vertex, which is why it
    performed poorly as an estimator.

    Branches are tagged ``low``/``high`` and given a shared ``group``. Tracks
    that pair with nothing keep ``branch=None`` and get their own group.
    """
    rising = [s for s in segments if slope_km_per_mhz(s) > BRANCH_SLOPE_FLOOR]
    falling = [s for s in segments if slope_km_per_mhz(s) < -BRANCH_SLOPE_FLOOR]

    def range_at_top(item: Segment, top: float) -> float:
        near = item.freq >= top - 0.4
        return float(item.vrange[near].mean()) if near.any() else float(item.vrange[-1])

    pairs = []
    for low in rising:
        for high in falling:
            top_gap = abs(low.freq.max() - high.freq.max())
            if top_gap > top_gap_mhz:
                continue
            top = min(low.freq.max(), high.freq.max())
            separation = abs(range_at_top(low, top) - range_at_top(high, top))
            if separation <= range_gap_km:
                # Each term normalised by its own tolerance, so neither
                # dominates: raw MHz against raw km would let a 0.16 MHz
                # difference outweigh 56 km of range separation.
                score = top_gap / top_gap_mhz + separation / range_gap_km
                pairs.append((score, low, high))

    # Closest pairs first, and each track joins at most one nose.
    group = 0
    for _, low, high in sorted(pairs, key=lambda p: p[0]):
        if low.group is not None or high.group is not None:
            continue
        low.branch, high.branch = "low", "high"
        low.group = high.group = group
        group += 1

    for item in segments:
        if item.group is None:
            item.group = group
            group += 1
    return segments


def primary_segment(segments: list[Segment]) -> Segment | None:
    """The track carrying the MUF.

    The nose group reaching the highest frequency, and within it the low ray --
    the long main trace rather than the short steep high-ray fragment beside it.
    """
    if not segments:
        return None

    best_group = max(segments, key=lambda s: s.freq.max()).group
    if best_group is None:
        return max(segments, key=lambda s: s.freq.max())

    members = [s for s in segments if s.group == best_group]
    low = [s for s in members if s.branch == "low"]
    if low:
        return max(low, key=lambda s: s.n_points)
    return max(members, key=lambda s: s.freq.max())


def nose_points(segments: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    """Points from both branches of the primary nose, for fitting its vertex.

    With both sides present the vertex is bracketed by data rather than
    extrapolated from the curvature of one branch.
    """
    primary = primary_segment(segments)
    if primary is None:
        return np.empty(0), np.empty(0)

    members = [s for s in segments if s.group == primary.group] or [primary]
    freq = np.concatenate([s.freq for s in members])
    vrange = np.concatenate([s.vrange for s in members])
    order = np.argsort(freq)
    return freq[order], vrange[order]


def _smoothed_knots(freq, vrange, weight, bin_mhz):
    """Robust summary points: the median range within each frequency bin.

    Shape-preserving interpolators pass through every point, so feeding them raw
    scattered data reproduces the noise. Binning first gives them something
    already smooth to follow.
    """
    edges = np.arange(freq.min(), freq.max() + bin_mhz, bin_mhz)
    index = np.clip(np.digitize(freq, edges) - 1, 0, len(edges) - 1)

    centres, values = [], []
    for b in np.unique(index):
        inside = index == b
        centres.append(float(freq[inside].mean()))
        values.append(float(np.median(vrange[inside])))
    return np.asarray(centres), np.asarray(values)


def reconstruct(
    item: Segment,
    step_mhz: float | None = None,
    smoothing_km: float | None = None,
    min_points: int = DEFAULT_MIN_SEGMENT_POINTS,
    method: str = DEFAULT_METHOD,
    max_support_gap_mhz: float = DEFAULT_MAX_SUPPORT_GAP_MHZ,
) -> Reconstruction:
    """Reconstruct one segment as a continuous curve.

    Args:
        item: a single mode, from :func:`group_tracks` or :func:`segment`.
        step_mhz: output grid spacing. Defaults to the smallest spacing present,
            so gaps are filled at the instrument's own resolution.
        smoothing_km: expected scatter about the true curve. Estimated from the
            data when omitted.
        method: ``spline`` fits a weighted smoothing spline; ``pchip`` and
            ``makima`` bin the points to medians and interpolate through those,
            and cannot overshoot.
        max_support_gap_mhz: output is not produced further than this from a
            measured point, so a wide gap is left empty rather than bridged by
            whatever the fit does in between.
    """
    from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, \
        UnivariateSpline

    if item.n_points < min_points:
        return Reconstruction(np.empty(0), np.empty(0), item, ok=False,
                              reason=f"only {item.n_points} points")

    freq, vrange = item.freq, item.vrange
    if np.ptp(freq) <= 0:
        return Reconstruction(np.empty(0), np.empty(0), item, ok=False,
                              reason="zero frequency span")

    if smoothing_km is None:
        # Scatter about a short running median, which follows the curve without
        # following the noise.
        from scipy.ndimage import median_filter
        window = min(9, item.n_points | 1)
        smoothing_km = float(np.std(vrange - median_filter(vrange, size=window)))
        smoothing_km = max(smoothing_km, 1.0)

    weights = None
    if item.weight is not None and np.all(item.weight > 0):
        weights = np.sqrt(item.weight / item.weight.max()) / smoothing_km
    else:
        weights = np.full(item.n_points, 1.0 / smoothing_km)

    knots = None
    try:
        if method == "spline":
            # s = n is the standard target when weights are 1/sigma: fit to
            # within one standard deviation per point, no closer.
            curve = UnivariateSpline(freq, vrange, w=weights, k=3,
                                     s=float(item.n_points))
            knots = np.asarray(curve.get_knots())
        elif method in ("pchip", "makima"):
            bin_mhz = max(np.ptp(freq) / 12.0, float(np.min(np.diff(freq)[np.diff(freq) > 0]))
                          if np.any(np.diff(freq) > 0) else 0.05)
            centres, medians = _smoothed_knots(freq, vrange, item.weight, bin_mhz)
            if len(centres) < 3:
                return Reconstruction(np.empty(0), np.empty(0), item, ok=False,
                                      reason=f"only {len(centres)} bins for {method}")
            curve = (PchipInterpolator(centres, medians, extrapolate=False)
                     if method == "pchip"
                     else Akima1DInterpolator(centres, medians))
            knots = centres
        else:
            raise ValueError(f"unknown reconstruction method {method!r}")
    except ValueError:
        raise
    except Exception as exc:
        return Reconstruction(np.empty(0), np.empty(0), item, ok=False,
                              reason=f"{method} failed: {type(exc).__name__}: {exc}")

    if step_mhz is None:
        # The *smallest* spacing present, which is the instrument's own bin
        # width -- so gaps are filled at the native resolution. Using the median
        # instead would reproduce the input's own sparseness and fill nothing.
        gaps = np.diff(freq)
        gaps = gaps[gaps > 0]
        step_mhz = float(gaps.min()) if gaps.size else 0.02
        step_mhz = max(step_mhz, 1e-4)

    grid = np.arange(freq.min(), freq.max() + step_mhz * 0.5, step_mhz)
    if grid.size < 2:
        grid = np.array([freq.min(), freq.max()])

    # Emit only where a measurement is nearby. Across a wide gap the curve is
    # whatever the fit happens to do, which on a real sounding here meant a
    # smoothing spline arcing 200 km above the data over a 2 MHz hole.
    nearest = np.min(np.abs(grid[:, None] - freq[None, :]), axis=1)
    supported = nearest <= max_support_gap_mhz
    if supported.sum() < 2:
        return Reconstruction(np.empty(0), np.empty(0), item, ok=False,
                              reason="no region is well enough supported")
    grid = grid[supported]

    values = np.asarray(curve(grid), dtype=float)
    fitted = np.asarray(curve(freq), dtype=float)
    good = np.isfinite(fitted)
    residual = (float(np.sqrt(np.mean((fitted[good] - vrange[good]) ** 2)))
                if good.any() else float("nan"))

    finite = np.isfinite(values)
    return Reconstruction(
        freq=grid[finite], vrange=values[finite], segment=item,
        rms_residual_km=residual, knots=knots,
    )


def analyse(
    freq: np.ndarray,
    vrange: np.ndarray,
    ground_km: float,
    weight: np.ndarray | None = None,
    group: bool = True,
    **options,
) -> tuple[list[Segment], Reconstruction | None]:
    """Separate the modes in a trace, label them, and reconstruct the primary.

    Returns ``(segments, reconstruction_of_the_primary_segment)``.

    ``group=True`` uses :func:`group_tracks`, which follows each mode across
    frequency and handles modes that overlap in frequency. ``group=False`` uses
    :func:`segment`, which splits a frequency-ordered sequence at range jumps --
    correct only when at most one mode is present at a time, and appropriate
    when the input has a single range per frequency bin.
    """
    if group:
        segments = group_tracks(
            freq, vrange, weight=weight,
            max_step_km=options.pop("max_step_km", DEFAULT_MAX_JUMP_KM),
            max_gap_mhz=options.pop("max_gap_mhz", DEFAULT_MAX_GAP_MHZ),
            min_points=options.pop("min_points", DEFAULT_MIN_SEGMENT_POINTS),
        )
        options.pop("max_jump_km", None)
        options.pop("judge_window", None)
    else:
        segments = segment(
            freq, vrange, weight=weight,
            max_jump_km=options.pop("max_jump_km", DEFAULT_MAX_JUMP_KM),
            judge_window=options.pop("judge_window", DEFAULT_JUDGE_WINDOW),
            min_points=options.pop("min_points", DEFAULT_MIN_SEGMENT_POINTS),
        )
    segments = identify_hops(segments, ground_km,
                             tolerance_km=options.pop("tolerance_km",
                                                      HOP_TOLERANCE_KM))
    segments = merge_branches(
        segments,
        top_gap_mhz=options.pop("top_gap_mhz", BRANCH_TOP_GAP_MHZ),
        range_gap_km=options.pop("range_gap_km", BRANCH_RANGE_GAP_KM),
    )
    primary = primary_segment(segments)
    if primary is None:
        return segments, None
    return segments, reconstruct(primary, **options)


def trace_weights(ion, result) -> np.ndarray:
    """Per-point confidence from the signal level at each detected bin."""
    indices = np.flatnonzero(result.presence)
    if indices.size == 0:
        return np.empty(0)
    return np.array([float(ion.db[i].max()) for i in indices])


def extract_points(ion, result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every trace point in a sounding: ``(freq, vrange, weight)``.

    Unlike :func:`muf.fit.trace_points`, which returns one range per frequency
    bin, this emits one point per *contiguous run* in the detection mask. Two
    propagation modes are routinely present at the same frequency -- a 06:00
    sounding here carries traces near 2680, 2800 and 2950 km simultaneously --
    and collapsing them to a single range per bin makes the extractor alternate
    between them, which then looks like a trace jumping about.

    Frequencies repeat in the output, once per mode present.
    """
    if result.mask is None:
        freq, vrange = _single_points(ion, result)
        return freq, vrange, trace_weights(ion, result)

    freqs, ranges, weights = [], [], []
    for i in np.flatnonzero(result.presence):
        row = result.mask[i]
        cells = np.flatnonzero(row)
        if cells.size == 0:
            continue
        # Split the lit cells into contiguous runs; each is one mode's echo.
        breaks = np.flatnonzero(np.diff(cells) > 1) + 1
        for run in np.split(cells, breaks):
            if run.size == 0:
                continue
            power = np.maximum(ion.db[i, run], 0.0)
            total = power.sum()
            centre = (np.average(ion.vrange[run], weights=power) if total > 0
                      else float(ion.vrange[run].mean()))
            freqs.append(float(ion.freq[i]))
            ranges.append(float(centre))
            weights.append(float(ion.db[i, run].max()))

    return (np.asarray(freqs), np.asarray(ranges), np.asarray(weights))


def _single_points(ion, result) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(result.presence)
    ranges = np.array([ion.vrange[int(ion.db[i].argmax())] for i in indices])
    return ion.freq[indices], ranges


def group_tracks(
    freq: np.ndarray,
    vrange: np.ndarray,
    weight: np.ndarray | None = None,
    max_step_km: float = DEFAULT_MAX_JUMP_KM,
    max_gap_mhz: float = DEFAULT_MAX_GAP_MHZ,
    min_points: int = DEFAULT_MIN_SEGMENT_POINTS,
) -> list[Segment]:
    """Group points into tracks by following each mode across frequency.

    Walks up the frequency axis holding a set of open tracks and extending each
    with the nearest point in range. This is the right operation when modes
    overlap in frequency -- which they usually do -- where splitting a
    frequency-ordered sequence at range jumps is not.

    Args:
        max_step_km: how far a track's range may move between frequency bins.
        max_gap_mhz: a track unextended for longer than this is closed, so two
            unrelated echoes at the same range are not joined across the band.
    """
    freq = np.asarray(freq, dtype=float)
    vrange = np.asarray(vrange, dtype=float)
    if freq.size == 0:
        return []
    if weight is None:
        weight = np.ones_like(freq)
    weight = np.asarray(weight, dtype=float)

    order = np.argsort(freq, kind="stable")
    freq, vrange, weight = freq[order], vrange[order], weight[order]

    open_tracks: list[dict] = []
    closed: list[dict] = []

    for value in np.unique(freq):
        at = np.flatnonzero(freq == value)

        for track in open_tracks:
            if value - track["last_freq"] > max_gap_mhz:
                track["stale"] = True
        closed.extend(t for t in open_tracks if t.get("stale"))
        open_tracks = [t for t in open_tracks if not t.get("stale")]

        # Greedy nearest-in-range assignment: closest pairs claimed first.
        pairs = sorted(
            ((abs(vrange[p] - t["last_range"]), p, ti)
             for p in at for ti, t in enumerate(open_tracks)),
            key=lambda item: item[0],
        )
        taken_points: set[int] = set()
        taken_tracks: set[int] = set()
        for distance, p, ti in pairs:
            if distance > max_step_km or p in taken_points or ti in taken_tracks:
                continue
            track = open_tracks[ti]
            track["freq"].append(freq[p])
            track["vrange"].append(vrange[p])
            track["weight"].append(weight[p])
            track["last_freq"], track["last_range"] = freq[p], vrange[p]
            taken_points.add(p)
            taken_tracks.add(ti)

        for p in at:
            if p in taken_points:
                continue
            open_tracks.append({
                "freq": [freq[p]], "vrange": [vrange[p]], "weight": [weight[p]],
                "last_freq": freq[p], "last_range": vrange[p],
            })

    closed.extend(open_tracks)

    segments = [
        Segment(freq=np.asarray(t["freq"]), vrange=np.asarray(t["vrange"]),
                weight=np.asarray(t["weight"]))
        for t in closed if len(t["freq"]) >= min_points
    ]
    return sorted(segments, key=lambda s: s.freq.min())
