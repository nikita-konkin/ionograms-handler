"""The dynamic-programming trace extractor.

These are written against *behaviour the physics requires*, not against the
numbers the implementation happens to produce, because the whole claim of this
module is that continuity is a constraint rather than a post-hoc filter. If a
future rewrite still satisfies all of these, it is still the same estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import spectro
from muf.extractors import viterbi
from muf.spectro import NOISE_FLOOR_DB

from conftest import snapped_range, synth_iq

FREQ = np.arange(2.0, 30.0, 0.0205)         # a .lfs-like frequency axis
VRANGE = np.arange(0.0, 1000.0, 5.0)
ECHO_DB = 60.0


def _array(ranges_km, lit, echo_db=ECHO_DB, seed=0):
    """An ionogram-shaped dB array with one bright cell per lit frequency."""
    rng = np.random.default_rng(seed)
    db = (np.full((FREQ.size, VRANGE.size), NOISE_FLOOR_DB)
          + rng.normal(0.0, 1.5, (FREQ.size, VRANGE.size)))
    for i, (r, on) in enumerate(zip(ranges_km, lit)):
        if on and np.isfinite(r):
            db[i, int(np.argmin(np.abs(VRANGE - r)))] = echo_db
    return db


def _flat(start, stop, range_km=250.0, holes=()):
    lit = np.zeros(FREQ.size, bool)
    lit[start:stop] = True
    for a, b in holes:
        lit[a:b] = False
    return np.full(FREQ.size, range_km), lit


def test_it_finds_the_extent_of_a_clean_trace():
    ranges, lit = _flat(100, 700)
    present, chosen = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)

    found = np.flatnonzero(present)
    assert (found.min(), found.max()) == (100, 699)
    assert VRANGE[chosen[found]].std() == pytest.approx(0.0, abs=5.0)


def test_pure_noise_yields_no_trace_at_all():
    """The estimator has to be able to say "nothing here".

    Nothing beats `DEFAULT_THRESHOLD_DB`, so every candidate span scores
    negative and the answer is the empty one. An extractor that always returns
    its best guess would report a MUF for a dead band.
    """
    rng = np.random.default_rng(1)
    db = (np.full((FREQ.size, VRANGE.size), NOISE_FLOOR_DB)
          + rng.normal(0.0, 1.5, (FREQ.size, VRANGE.size)))

    present, chosen = viterbi.trace(db, FREQ, VRANGE)
    assert not present.any()
    assert (chosen == -1).all()


def test_a_short_fade_is_crossed_without_a_bridge_parameter():
    """The reason this module exists. `algo` needed `bridge=2` for this."""
    ranges, lit = _flat(100, 700, holes=[(400, 402)])
    present, _ = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)

    assert present[400] and present[401], "the fade ended the trace"
    assert np.flatnonzero(present).max() == 699


def test_a_gap_is_crossed_on_what_lies_beyond_it_not_on_its_length():
    """No fixed bridge count, in either direction.

    Forty dead bins are crossed when six hundred bins of strong trace lie
    beyond them, and the *same* forty are not crossed when nothing does. The
    length is identical in both cases, so length cannot be what decides.
    """
    ranges, lit = _flat(100, 700, holes=[(400, 440)])
    crossed, _ = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)
    assert crossed[400:440].any(), "a strong continuation did not pay for it"
    assert np.flatnonzero(crossed).max() == 699

    ranges, lit = _flat(100, 440)               # nothing past the gap at all
    stopped, _ = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)
    assert not stopped[440:].any(), "crossed a gap with nothing behind it"


def test_an_interferer_off_the_trace_cannot_become_the_muf():
    """The bug that killed the first design, kept as a test.

    An earlier version modelled "off the trace" as a state the path could
    enter and leave. Leaving reset the range, so the slope limit did not apply
    across the gap and a bright blob anywhere in the array could be joined for
    a fixed price -- here six bins, 500 km off the trace and 200 bins above its
    end, were returned as the MUF. One span, scored end to end, cannot do that.
    """
    ranges, lit = _flat(100, 700, range_km=250.0)
    db = _array(ranges, lit)
    db[900:906, 150] = ECHO_DB                  # 750 km, far above the trace

    present, _ = viterbi.trace(db, FREQ, VRANGE)
    assert np.flatnonzero(present).max() == 699
    assert not present[900:906].any()


def test_the_trace_is_followed_up_a_slope():
    """A MUF nose is a real trace, and must not be cut off as a range jump."""
    ranges = np.linspace(250.0, 400.0, FREQ.size)
    lit = np.zeros(FREQ.size, bool)
    lit[100:700] = True

    present, chosen = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)
    found = np.flatnonzero(present)

    assert (found.min(), found.max()) == (100, 699)
    climbed = VRANGE[chosen[found.max()]] - VRANGE[chosen[found.min()]]
    assert climbed == pytest.approx(ranges[699] - ranges[100], abs=15.0)


def test_the_slope_limit_holds_at_every_step_of_the_path():
    """The constraint that makes this an estimator and not a brightest-cell
    search.

    Stated as the invariant rather than as a consequence. An earlier version
    asserted that a path could not contain two features 650 km apart, which is
    not something a per-step limit can promise and not something it should: a
    real MUF nose climbs, and given enough frequency bins a legal path reaches
    a long way. What the limit actually guarantees is that it never gets there
    in one step, and that is what is checked.
    """
    ranges = np.full(FREQ.size, 250.0)
    ranges[400:] = 900.0                        # 650 km in one frequency step
    lit = np.zeros(FREQ.size, bool)
    lit[100:700] = True

    present, chosen = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)
    found = np.flatnonzero(present)
    steps = np.abs(np.diff(VRANGE[chosen[found]]))

    range_step = float(np.median(np.diff(VRANGE)))
    freq_step = float(np.median(np.diff(FREQ)))
    allowed = max(range_step, round(150.0 * freq_step / range_step) * range_step)
    assert steps.max() <= allowed + 1e-9, \
        f"moved {steps.max():.0f} km in one step, limit {allowed:.0f} km"
    assert found.size > 1


def test_the_drift_budget_comes_from_the_slope_not_from_the_jitter_floor():
    """Regression on a limit that was 3x too loose.

    `pick._range_tolerance` floors its allowance at `RANGE_SLOPE_FLOOR_BINS`
    range bins, which is right for jitter between two independent detections
    and wrong as a per-step budget for a path of hundreds of steps. Using it
    here allowed 10 km per step on this axis -- 488 km/MHz against a stated
    150 -- and the path could walk across a 650 km gap through pure noise.
    """
    from muf import pick

    range_step = float(np.median(np.diff(VRANGE)))
    freq_step = float(np.median(np.diff(FREQ)))

    floored = pick._range_tolerance(FREQ, VRANGE, 150.0)
    assert floored / freq_step > 3 * 150.0, "the floor is what made this loose"

    ranges = np.full(FREQ.size, 250.0)
    lit = np.zeros(FREQ.size, bool)
    lit[100:700] = True
    _, chosen = viterbi.trace(_array(ranges, lit), FREQ, VRANGE)
    present = chosen >= 0
    steps = np.abs(np.diff(VRANGE[chosen[present]]))
    assert steps.max() <= max(range_step, 150.0 * freq_step) + 1e-9


def test_extract_satisfies_the_mufresult_contract(synthetic_dp):
    """The same contract the other three estimators satisfy.

    Driven off the shared synthetic sounding in `test_extractors.py` rather
    than a hand-built array, so "dp finds the echo" is asserted against the
    same known answer the other estimators are held to.
    """
    result, expected_range, range_tolerance = synthetic_dp

    assert result.method == "dp"
    assert result.ok
    assert result.vrange_km == pytest.approx(expected_range, abs=range_tolerance)
    assert result.presence.shape[0] == result.mask.shape[0]
    # One cell per present frequency: the path is a function of frequency.
    assert result.mask.sum() == result.presence.sum()


def test_the_trace_can_never_end_on_a_bin_that_was_not_lit():
    """Why `extract` may hand the picker its raw path as a presence array.

    `pick_muf` reports the MUF at the highest qualifying frequency, and
    `pick.qualifying` keeps that off an unlit bin by filtering on `presence`.
    This module's path is present on the bins it crosses through a fade, so
    that guard is inert here -- a `dp` MUF is protected instead by the shape of
    the score: `best = gain + max(reachable, 0)`, and a below-threshold cell has
    negative `gain`, so it always scores *below* its own best predecessor. The
    global argmax cannot land on one.

    That is a property of the scoring, not of the data, and a future change to
    it -- clamping `gain` at zero, adding a continuity bonus -- would break the
    invariant silently and let a crossed fade become the answer. Hence a test
    rather than a comment. Checked here on a trace whose top end is a long fade,
    which is the case that would expose it.
    """
    ranges, lit = _flat(100, 700, holes=[(660, 700)])   # dies into a fade
    db = _array(ranges, lit)
    present, chosen = viterbi.trace(db, FREQ, VRANGE)
    found = np.flatnonzero(present)

    assert found.size
    top = found.max()
    assert db[top, chosen[top]] >= viterbi.DEFAULT_THRESHOLD_DB, \
        "the trace ended on a bin where nothing was detected"
    assert top == 659, "the trace ran past its last detection"


def test_it_is_in_the_registry_but_not_in_the_defaults():
    """Stored results were all produced without it; a default would change
    what re-running an old archive means."""
    from muf import extractors

    assert "dp" in extractors.available()
    assert "dp" in extractors.ALL_METHODS
    assert "dp" not in extractors.DEFAULT_METHODS
    assert extractors.get("dp") is viterbi.extract


def test_degenerate_arrays_do_not_raise():
    empty = np.zeros((0, 0))
    present, chosen = viterbi.trace(empty, np.zeros(0), np.zeros(0))
    assert present.size == 0 and chosen.size == 0

    one = np.full((1, 4), NOISE_FLOOR_DB)
    present, chosen = viterbi.trace(one, np.array([5.0]), VRANGE[:4])
    assert not present.any()


# The shared synthetic sounding from `test_extractors.py`, so `dp` is held to
# the same known answer as `algo`, `contour` and `kmeans`.
WINDOW = 512
N_FREQ = 200
HALF_SPAN = 60_000.0
ECHO_RANGE = 2700.0
ECHO_LAST_BIN = 120


@pytest.fixture
def synthetic_dp(make_lfs):
    iq = synth_iq(
        n_freq=N_FREQ, window=WINDOW, echo_range_km=ECHO_RANGE,
        half_span_km=HALF_SPAN, echo_last_bin=ECHO_LAST_BIN,
    )
    path = make_lfs(iq, name="synthetic_dp.lfs")
    ion = spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))
    return (viterbi.extract(ion),
            snapped_range(ECHO_RANGE, HALF_SPAN, WINDOW),
            2 * HALF_SPAN / WINDOW)
