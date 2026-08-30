"""The shared rule that turns a detected trace into a MUF value.

Every method inherited from the two source projects ends the same way: take the
right-most bright thing. That is one noise spike away from being wrong, and each
script grew its own partial defence -- ``ionogr_clustering_0.03.py`` required a
run of 2 consecutive columns, ``ionogr_clustering_and_MUF_0.01.py`` walked
neighbours with a 0.003 MHz step, ``kmeans_clustering.py`` had none at all.

This module generalises those into one function that every extractor calls, so
a change to the decision rule applies uniformly and the methods stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 5 bins x 20.5 kHz ~ 100 kHz of continuous trace. Long enough to reject
# isolated interference, short enough not to clip a genuine MUF edge, where the
# trace fades over a few bins.
DEFAULT_MIN_RUN = 5

#: Steepest range-against-frequency slope a real trace is allowed, km/MHz.
#:
#: An ionospheric echo's virtual range is a smooth function of frequency:
#: measured on this instrument the low ray runs +2 to +17 km/MHz and the high
#: ray -24 to -76, steepening only at the nose. Interference has no such
#: constraint -- consecutive frequency bins light up at unrelated ranges,
#: because there is no propagation path tying them together.
#:
#: This is not a novel test. It is the "good continuation" half of the pair of
#: Gestalt grouping principles ARTIST 5 rests its echo grouping on -- Galkin
#: and Reinisch, *The New ARTIST 5 for all Digisondes*, INAG, 2008, the same
#: authors as the SAO.XML spec `muf.export.saoxml` writes -- and the same
#: criterion Ding et al. state as "the continuity of the slope of the single
#: layer trace and rejection of impractical changes in slope when the ionogram
#: is traversed in the frequency axis". Both arrived at it because thresholding
#: alone does not separate an echo from a crowded band.
#:
#: 150 is roughly twice the steepest real leg measured here, so it admits the
#: nose while rejecting a range that jumps. It is a *rate*, applied per
#: frequency step, so it means the same thing on a 25 kHz digisonde axis and a
#: 20.5 kHz .lfs one.
DEFAULT_MAX_RANGE_SLOPE = 150.0

#: Frequency bins a run may skip over and still count as one run.
#:
#: A real trace fades. Requiring `min_run` bins with no gap at all treats a
#: two-bin dropout as the end of the trace, and on this instrument that is the
#: single largest source of disagreement between estimators: measured
#: 2026-08-30 over 3421 soundings, `algo` and `contour` differ by more than
#: 1 MHz on 11.1% of them, **every one** of those with `algo` reading lower, and
#: the rate is 31.7% where the shorter run is <= 8 bins against 3.0% where it is
#: longer. On the worst sounding `algo` detected trace out to 28.45 MHz and the
#: picker returned 18.75, because nothing past that point could assemble five
#: unbroken bins.
#:
#: Two, not more: three consecutive dead bins at 20.5 kHz is 60 kHz of silence,
#: which is longer than any fade measured here and starts to admit two
#: unrelated features as one run. The range-agreement test still applies across
#: a bridge -- `split_on_range_jumps` scales its tolerance by the gap -- so
#: bridging decides only whether bins may be *considered* together, never
#: whether they agree.
DEFAULT_BRIDGE = 2

#: Floor on the per-step range tolerance, in range bins. Two bins of jitter is
#: the peak-finding, not the ionosphere: without this the rule would reject a
#: real trace on a fine frequency axis, where the allowed slope works out
#: smaller than the range resolution.
RANGE_SLOPE_FLOOR_BINS = 2.0


@dataclass(frozen=True)
class MufPick:
    """Outcome of one MUF decision."""

    muf_mhz: float          # NaN when nothing qualified
    vrange_km: float        # NaN when nothing qualified
    n_detections: int       # frequency bins with trace presence
    run_len: int            # length of the run the pick sits in
    snr_db: float           # peak power at the picked bin, dB over the noise floor
    freq_index: int         # -1 when nothing qualified

    @property
    def ok(self) -> bool:
        return self.freq_index >= 0


NO_PICK = MufPick(np.nan, np.nan, 0, 0, np.nan, -1)


def find_runs(presence: np.ndarray, min_run: int,
              bridge: int = 0) -> list[tuple[int, int]]:
    """Inclusive ``(start, stop)`` spans of at least ``min_run`` detected bins.

    ``bridge`` gaps of up to that many undetected bins are carried through as
    part of the run, because a real trace fades and an estimator that treats a
    two-bin dropout as the end of the trace stops early -- see
    :data:`DEFAULT_BRIDGE` for what that costs in practice.

    A bridged run still has to satisfy ``min_run`` on its *detected* bins, not
    on its total width. Counting the gap would let ``min_run=5`` be met by three
    detections and two holes, which is the interference this rule exists to
    reject rather than the fade it is being relaxed for.
    """
    if presence.size == 0:
        return []
    presence = np.asarray(presence, dtype=bool)
    padded = np.concatenate(([False], presence, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]        # stops are exclusive here
    spans = [(int(a), int(b - 1)) for a, b in zip(starts, stops)]

    if bridge > 0 and len(spans) > 1:
        merged = [spans[0]]
        for a, b in spans[1:]:
            previous_a, previous_b = merged[-1]
            if a - previous_b - 1 <= bridge:
                merged[-1] = (previous_a, b)
            else:
                merged.append((a, b))
        spans = merged

    return [(a, b) for a, b in spans
            if int(presence[a:b + 1].sum()) >= min_run]


def echo_ranges(presence: np.ndarray, power_db: np.ndarray,
                vrange: np.ndarray) -> np.ndarray:
    """Brightest range at each detected frequency; NaN where nothing was found.

    The brightest cell rather than a centroid: a centroid of a row holding both
    a trace and an interferer sits between them, at a range neither occupies,
    which is precisely the reading this rule exists to catch.
    """
    out = np.full(presence.shape, np.nan, dtype=float)
    idx = np.flatnonzero(presence)
    if idx.size:
        out[idx] = np.asarray(vrange)[np.argmax(power_db[idx], axis=1)]
    return out


def split_on_range_jumps(runs, ranges: np.ndarray,
                         tolerance_km: float) -> list[tuple[int, int]]:
    """Break runs wherever the echo range jumps between adjacent bins.

    A run of consecutive lit frequency bins is only evidence of a trace if
    those bins agree about *where* the echo is. Interference satisfies the
    consecutive-bins test easily -- a crowded band lights up many neighbouring
    frequencies -- and fails this one, because nothing ties its ranges
    together.
    """
    out: list[tuple[int, int]] = []
    for a, b in runs:
        start = a
        # The last bin that actually had an echo, and how far back it is. A run
        # may now contain bridged bins with no range at all (`find_runs`), and
        # treating those as a break would undo the bridge the moment the slope
        # test is switched on -- the two rules would then disagree about what a
        # run is. Instead the comparison reaches across the gap and the
        # tolerance grows with it, which is what the slope limit means: km per
        # MHz, so two bins of separation allow twice the movement.
        last, since = ranges[a], 1
        for i in range(a + 1, b + 1):
            current = ranges[i]
            if not np.isfinite(current):
                since += 1
                continue
            broken = (not np.isfinite(last)
                      or abs(current - last) > tolerance_km * since)
            if broken:
                out.append((start, i - 1))
                start = i
            last, since = current, 1
        out.append((start, b))
    return out


def _range_tolerance(freq: np.ndarray, vrange: np.ndarray,
                     slope_km_per_mhz: float) -> float:
    """Per-step range tolerance in km, from the slope limit and the axes."""
    freq_step = float(np.median(np.abs(np.diff(freq)))) if freq.size > 1 else 0.0
    range_step = (float(np.median(np.abs(np.diff(vrange))))
                  if np.size(vrange) > 1 else 0.0)
    return max(slope_km_per_mhz * freq_step,
               RANGE_SLOPE_FLOOR_BINS * range_step)


def _parabolic_offset(y_prev: float, y_peak: float, y_next: float) -> float:
    """Sub-sample peak offset in bins, from three log-domain samples.

    Standard three-point parabolic interpolation. Returns 0.0 when the samples
    do not describe a peak.
    """
    denom = y_prev - 2.0 * y_peak + y_next
    if denom >= 0 or not np.isfinite(denom):
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -0.5, 0.5))


def _echo_range(
    row_db: np.ndarray, vrange: np.ndarray, parabolic: bool
) -> tuple[float, float]:
    """Virtual range and peak level of the strongest echo in one frequency bin."""
    j = int(np.argmax(row_db))
    peak = float(row_db[j])
    if not parabolic or j == 0 or j == len(row_db) - 1 or len(vrange) < 2:
        return float(vrange[j]), peak

    offset = _parabolic_offset(float(row_db[j - 1]), peak, float(row_db[j + 1]))
    step = float(vrange[1] - vrange[0])   # negative: the axis descends
    return float(vrange[j]) + offset * step, peak


def pick_muf(
    presence: np.ndarray,
    freq: np.ndarray,
    power_db: np.ndarray | None = None,
    vrange: np.ndarray | None = None,
    min_run: int = DEFAULT_MIN_RUN,
    percentile: float = 100.0,
    parabolic: bool = True,
    max_range_slope: float | None = None,
    bridge: int = DEFAULT_BRIDGE,
) -> MufPick:
    """Pick the MUF from per-frequency trace presence.

    Args:
        presence: bool, one entry per frequency bin -- trace detected there.
        freq: frequency axis in MHz, ascending, same length as ``presence``.
        power_db: optional ``[n_freq, n_range]`` array used to locate the echo's
            virtual range and report its level.
        vrange: virtual-range axis in km, matching ``power_db``'s columns.
        min_run: consecutive detected bins required to accept a run. Set to 1 to
            reproduce the historical "right-most bright thing" behaviour.
        percentile: 100 takes the highest qualifying bin; lower values trim the
            top of the distribution, which helps when a few bins of interference
            survive the run test.
        parabolic: interpolate the echo's range between bins. This recovers the
            sub-bin precision that zero-padding was previously used for, at no
            cost -- see ``calibrate.range_resolution_km``.
        bridge: undetected frequency bins a run may skip over. 0 requires an
            unbroken run, which is what every result before 2026-08-30 was
            produced with. See :data:`DEFAULT_BRIDGE`.
        max_range_slope: steepest range-against-frequency slope a run may have,
            in km/MHz. ``None`` disables the test, which is the default and
            what every ``.lfs`` result to date was produced with.

            The consecutive-bins rule asks whether neighbouring frequencies are
            lit; this asks whether they agree about *where*. On a crowded band
            the first is easy to satisfy by accident -- received obliquely at
            DOB, four different digisonde circuits all reported a MUF of
            3.05 MHz while their pick ranges wandered over 1700 km, which is
            not four ionospheres agreeing but one interferer being found four
            times.

    Returns:
        A ``MufPick``; ``NO_PICK`` when no run qualifies.
    """
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != freq.shape:
        raise ValueError(f"presence {presence.shape} does not match freq {freq.shape}")

    n_detections = int(presence.sum())
    runs = find_runs(presence, min_run, bridge)

    if (max_range_slope is not None and runs
            and power_db is not None and vrange is not None and len(vrange)):
        # Split first, then re-apply min_run: a run broken in half by a range
        # jump has to earn its length again, or a long stretch of interference
        # would survive as several short ones.
        ranges = echo_ranges(presence, power_db, vrange)
        tolerance = _range_tolerance(freq, vrange, max_range_slope)
        runs = [(a, b) for a, b in split_on_range_jumps(runs, ranges, tolerance)
                if int(presence[a:b + 1].sum()) >= min_run]

    if not runs:
        return MufPick(np.nan, np.nan, n_detections, 0, np.nan, -1)

    # Detected bins only, not whole spans. A bridged run spans bins where
    # nothing was found, and those must never *be* the answer: a split landing
    # inside a bridge leaves a span ending on a gap bin, and `percentile=100`
    # would then report a MUF at a frequency that showed no trace at all.
    # Every surviving run holds at least `min_run` detections, so this cannot
    # empty out.
    qualifying = np.concatenate([np.arange(a, b + 1) for a, b in runs])
    qualifying = qualifying[presence[qualifying]]

    if percentile >= 100.0:
        index = int(qualifying.max())
    else:
        index = int(round(float(np.percentile(qualifying, percentile))))
        index = int(qualifying[np.abs(qualifying - index).argmin()])

    # Detections, not width: a bridged run that reported its span would claim
    # evidence for the bins it skipped, and `run_len` is read as a confidence.
    run_len = next((int(presence[a:b + 1].sum())
                    for a, b in runs if a <= index <= b), 0)

    vrange_km, snr_db = np.nan, np.nan
    if power_db is not None and vrange is not None and len(vrange):
        vrange_km, snr_db = _echo_range(power_db[index], vrange, parabolic)

    return MufPick(
        muf_mhz=float(freq[index]),
        vrange_km=vrange_km,
        n_detections=n_detections,
        run_len=int(run_len),
        snr_db=snr_db,
        freq_index=index,
    )
