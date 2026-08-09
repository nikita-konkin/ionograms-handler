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
