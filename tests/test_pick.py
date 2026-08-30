"""The shared decision rule, and the range-consistency test layered on it.

`min_run` asks whether neighbouring frequencies are lit. The range test asks
whether they agree about *where* the echo is, which is the part interference
cannot fake: a crowded band lights up many adjacent frequencies, but nothing
ties their ranges together.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import pick

FREQ = np.arange(40) * 0.025 + 1.0          # 25 kHz steps, as a digisonde
VRANGE = np.arange(200, 0, -3.0)            # 3 km bins, descending as the pipeline


def _power(ranges_km, *, lit, noise_db=25.0, echo_db=60.0):
    """Build a [n_freq, n_range] dB array with one bright cell per lit bin."""
    db = np.full((FREQ.size, VRANGE.size), noise_db)
    for i, r in enumerate(ranges_km):
        if lit[i] and np.isfinite(r):
            db[i, int(np.argmin(np.abs(VRANGE - r)))] = echo_db
    return db


def test_the_rule_is_off_unless_asked_for():
    """Every .lfs result to date was produced without it, so the default has to
    stay exactly what it was."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:20] = True
    ranges = np.where(lit, np.random.RandomState(0).uniform(30, 190, FREQ.size), np.nan)
    db = _power(ranges, lit=lit)

    loose = pick.pick_muf(lit, FREQ, db, VRANGE, min_run=5)
    assert loose.ok, "unchanged behaviour without the option"


def test_a_smooth_trace_survives():
    """A real echo's range is a smooth function of frequency -- +2 to +17 km/MHz
    on the low ray here, which at 25 kHz steps is well inside tolerance."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:25] = True
    ranges = np.where(lit, 120.0 + (np.arange(FREQ.size) - 10) * 0.4, np.nan)

    got = pick.pick_muf(lit, FREQ, _power(ranges, lit=lit), VRANGE,
                        min_run=5, max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert got.ok
    assert got.freq_index == 24, "the top of the run is still the pick"
    assert got.run_len == 15


def test_a_run_whose_range_jumps_is_rejected():
    """The DOB case: consecutive lit bins whose ranges are unrelated. This
    passes min_run easily and is not a trace."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:25] = True
    scatter = np.random.RandomState(1).uniform(30, 190, FREQ.size)
    ranges = np.where(lit, scatter, np.nan)

    got = pick.pick_muf(lit, FREQ, _power(ranges, lit=lit), VRANGE,
                        min_run=5, max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert not got.ok
    assert got.n_detections == 15, "the detections are still counted and reported"


def test_a_broken_run_must_earn_its_length_again():
    """Otherwise a long stretch of interference survives as several short ones."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:24] = True
    ranges = np.full(FREQ.size, np.nan)
    ranges[10:17] = 120.0                       # seven consistent
    ranges[17:24] = 40.0                        # then a jump, seven more

    eight = pick.pick_muf(lit, FREQ, _power(ranges, lit=lit), VRANGE,
                          min_run=8, max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert not eight.ok, "neither half is 8 long"

    five = pick.pick_muf(lit, FREQ, _power(ranges, lit=lit), VRANGE,
                         min_run=5, max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert five.ok and five.run_len == 7, "each half stands on its own"


def test_the_pick_comes_from_the_surviving_run():
    """A trace followed by interference must not have the interference's top
    reported as its MUF."""
    lit = np.zeros(FREQ.size, bool)
    lit[5:15] = True                            # smooth
    lit[20:30] = True                           # scattered
    ranges = np.full(FREQ.size, np.nan)
    ranges[5:15] = 120.0 + np.arange(10) * 0.3
    ranges[20:30] = np.random.RandomState(2).uniform(30, 190, 10)

    got = pick.pick_muf(lit, FREQ, _power(ranges, lit=lit), VRANGE,
                        min_run=5, max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert got.ok and got.freq_index == 14, "the smooth run, not the scattered one"


def test_the_tolerance_has_a_floor_in_range_bins():
    """On a fine frequency axis the allowed slope works out smaller than the
    range resolution, and the rule would reject a real trace for jitter of one
    bin. The floor is what stops that."""
    fine = np.arange(40) * 0.001 + 1.0          # 1 kHz steps
    slope_only = pick.DEFAULT_MAX_RANGE_SLOPE * 0.001      # 0.15 km
    tolerance = pick._range_tolerance(fine, VRANGE, pick.DEFAULT_MAX_RANGE_SLOPE)

    assert tolerance > slope_only
    assert tolerance == pytest.approx(pick.RANGE_SLOPE_FLOOR_BINS * 3.0)


def test_the_test_needs_the_array_and_declines_without_it():
    """`presence` alone cannot say where an echo was, so with no power array the
    rule has nothing to test and must not silently reject everything."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:20] = True

    got = pick.pick_muf(lit, FREQ, None, None, min_run=5,
                        max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert got.ok, "no array, no test"


def test_echo_ranges_takes_the_brightest_not_the_centroid():
    """A centroid of a row holding both a trace and an interferer sits between
    them, at a range neither occupies."""
    lit = np.zeros(FREQ.size, bool)
    lit[3] = True
    db = np.full((FREQ.size, VRANGE.size), 25.0)
    db[3, 5] = 70.0                                    # the trace
    db[3, 60] = 50.0                                   # something weaker

    ranges = pick.echo_ranges(lit, db, VRANGE)
    assert ranges[3] == pytest.approx(VRANGE[5])
    assert np.isnan(ranges[0])


# --------------------------------------------------------------------------
# Bridging a fade
#
# 2026-08-30. Requiring `min_run` bins with no gap at all treats a two-bin
# dropout as the end of the trace, and that was the largest single source of
# disagreement between estimators here: over 3421 soundings `algo` and
# `contour` differed by more than 1 MHz on 11.1%, **every one** of those with
# `algo` reading lower, and the rate was 31.7% where the shorter run was <= 8
# bins against 3.0% where it was longer. On the worst sounding `algo` detected
# trace out to 28.45 MHz while the picker returned 18.75, because nothing past
# that point could assemble five unbroken bins.
# --------------------------------------------------------------------------

def test_a_short_fade_no_longer_ends_the_trace():
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[17:23] = True          # one dead bin at 16

    assert pick.find_runs(lit, min_run=5, bridge=0) == [(10, 15), (17, 22)]
    assert pick.find_runs(lit, min_run=5, bridge=2) == [(10, 22)]


def test_a_long_gap_is_still_two_traces():
    """Three dead bins is 60 kHz of silence -- longer than any fade measured
    here, and wide enough to join two unrelated features."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[19:25] = True          # three dead bins at 16, 17, 18

    assert pick.find_runs(lit, min_run=5, bridge=2) == [(10, 15), (19, 24)]


def test_a_bridged_run_earns_min_run_on_detections_not_on_width():
    """Otherwise `min_run=5` is met by three detections and two holes, which is
    the interference the rule exists to reject rather than the fade it is being
    relaxed for."""
    lit = np.zeros(FREQ.size, bool)
    lit[10] = lit[12] = lit[14] = True      # 3 detections spanning 5 bins

    assert pick.find_runs(lit, min_run=5, bridge=2) == []
    assert pick.find_runs(lit, min_run=3, bridge=2) == [(10, 14)]


def test_run_len_reports_detections_not_the_span_it_bridged():
    """`run_len` is read as a confidence, so a bridged run must not claim
    evidence for the bins it skipped."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[17:23] = True

    got = pick.pick_muf(lit, FREQ, min_run=5, bridge=2)
    assert got.freq_index == 22, "the pick reaches past the fade"
    assert got.run_len == 12, "12 detections across a 13-bin span"


def test_bridging_off_reproduces_the_historical_rule():
    """Every result before 2026-08-30 was produced with an unbroken run."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[17:23] = True

    assert pick.pick_muf(lit, FREQ, min_run=5, bridge=0).run_len == 6
    assert pick.pick_muf(lit, FREQ, min_run=5, bridge=2).run_len == 12


def test_the_range_test_reaches_across_a_bridge_rather_than_breaking_on_it():
    """A bridged bin has no echo and so no range.

    Treating that as a break would undo the bridge the moment the slope test is
    switched on, and the two rules would then disagree about what a run is.
    The comparison reaches across the gap instead, with the tolerance scaled by
    its width -- which is what a slope limit means: km per MHz.
    """
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[17:23] = True
    ranges = np.where(lit, np.linspace(150, 130, FREQ.size), np.nan)
    db = _power(ranges, lit=lit)

    got = pick.pick_muf(lit, FREQ, db, VRANGE, min_run=5, bridge=2,
                        max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert got.ok and got.freq_index == 22, "a smooth trace survives its fade"


def test_a_bridge_does_not_smuggle_a_range_jump_past_the_slope_test():
    """Bridging decides only whether bins may be *considered* together.

    The two halves here each earn `min_run` on their own, so both survive and
    the picker takes the higher frequency -- which is correct. What must not
    happen is the two being treated as *one* run: that would report twelve
    bins of mutually-agreeing evidence for a trace that moves 120 km across
    the fade.
    """
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[17:23] = True
    ranges = np.where(lit, 150.0, np.nan)
    ranges[17:23] = 30.0                    # the far side sits 120 km away
    db = _power(ranges, lit=lit)

    got = pick.pick_muf(lit, FREQ, db, VRANGE, min_run=5, bridge=2,
                        max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert got.run_len == 6, "the jump split the run rather than being bridged"

    # And when only the far half is too short to stand alone, it goes entirely.
    lit[21:23] = False
    ranges = np.where(lit, 150.0, np.nan)
    ranges[17:21] = 30.0
    got = pick.pick_muf(lit, FREQ, _power(ranges, lit=lit), VRANGE,
                        min_run=5, bridge=2,
                        max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert got.freq_index == 15, "a short jumped-to fragment cannot win"


def test_a_bridged_run_never_reports_a_muf_at_an_unlit_bin():
    """The pick must land on a bin that actually showed something.

    A run is a *span*, and a bridged span covers bins where nothing was found.
    When a range jump splits such a span the cut can fall inside the bridge,
    leaving a span whose last bin is a gap -- and `percentile=100` takes the
    highest bin in the span. Reported naively that is a MUF at a frequency the
    sounder saw nothing at, biased upward by as much as `bridge` bins.
    """
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True                       # six bins at 150 km
    lit[17:21] = True                       # four bins at 30 km, too few alone
    ranges = np.where(lit, 150.0, np.nan)
    ranges[17:21] = 30.0
    db = _power(ranges, lit=lit)

    got = pick.pick_muf(lit, FREQ, db, VRANGE, min_run=5, bridge=2,
                        max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert lit[got.freq_index], "picked a frequency with no detection"
    assert got.freq_index == 15


def test_a_bridge_reaches_the_far_side_when_the_range_agrees():
    """The point of the whole exercise: a fade must not end the trace."""
    lit = np.zeros(FREQ.size, bool)
    lit[10:16] = True
    lit[18:24] = True                       # two dead bins between the halves
    ranges = np.where(lit, 150.0, np.nan)   # ...but the same range throughout
    db = _power(ranges, lit=lit)

    unbridged = pick.pick_muf(lit, FREQ, db, VRANGE, min_run=5, bridge=0,
                              max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    bridged = pick.pick_muf(lit, FREQ, db, VRANGE, min_run=5, bridge=2,
                            max_range_slope=pick.DEFAULT_MAX_RANGE_SLOPE)
    assert unbridged.freq_index == 23, "each half already qualifies on its own"
    assert bridged.freq_index == 23
    assert bridged.run_len == 12, "bridging joins the evidence"
    assert unbridged.run_len == 6
