"""Rejecting interference stripes.

The rule is one asymmetry: a burst has no delay and smears across every range
bin, an echo is narrow in range and continuous in frequency. These tests
police the boundary -- above all that a real echo, however strong or however
spread by multi-hop, is never mistaken for a burst.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import interference, spectro
from muf.pipeline import Options

# A range axis fine enough for a km-based rule to mean something: 2*60000/4096
# is 29.3 km per bin. Real products are finer still (v2 is 2 km), but at the
# 234 km of `window=512` a four-bin echo would exceed MAX_ECHO_RANGE_KM -- see
# MIN_BURST_BINS, which is the guard for exactly that.
WINDOW = 4096
N_FREQ = 40
HALF_SPAN = 60_000.0
BURST_ROW = 20


def _sounding(make_lfs, **header):
    from conftest import synth_iq

    iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
                  half_span_km=HALF_SPAN, echo_last_bin=N_FREQ - 1)
    return spectro.compute(make_lfs(iq, name="i.lfs", **header),
                           window=WINDOW, gate_km=(2000.0, 5000.0))


def _burst(ion, row: int):
    """Light up one frequency row across the whole range axis."""
    power = np.array(ion.power, copy=True)
    power[row, :] = 10 ** 6
    import dataclasses
    return dataclasses.replace(ion, power=power, _db=None)


# --------------------------------------------------------------------------
# Finding
# --------------------------------------------------------------------------

def test_a_burst_row_is_found(make_lfs):
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    found = interference.find(ion)

    assert found.any
    assert found.n_rows == 1
    assert found.rows[BURST_ROW]
    assert found.occupied_km[BURST_ROW] > interference.MAX_ECHO_RANGE_KM


def test_a_clean_sounding_has_no_rows(make_lfs):
    """The synthetic echo is strong and must survive untouched."""
    found = interference.find(_sounding(make_lfs))

    assert not found.any
    assert found.describe() == "no interference rows"


def test_a_strong_narrow_echo_is_never_a_burst(make_lfs):
    """The failure that would matter: rejecting the measurement itself."""
    ion = _sounding(make_lfs)
    power = np.array(ion.power, copy=True)
    centre = ion.power.shape[1] // 2
    power[:, centre - 2:centre + 3] = 10 ** 6      # a very bright, narrow trace
    import dataclasses
    ion = dataclasses.replace(ion, power=power, _db=None)

    assert not interference.find(ion).any


def test_multi_hop_spans_far_but_occupies_little(make_lfs):
    """Occupied range, not first-to-last span.

    Three hops 300 km apart span 600 km and occupy about 120 km. Measuring the
    span would reject a legitimate multi-hop family.
    """
    ion = _sounding(make_lfs)
    power = np.array(ion.power, copy=True)
    step = abs(float(ion.cal.vrange[1] - ion.cal.vrange[0]))
    apart = max(1, int(300.0 / step))
    centre = ion.power.shape[1] // 2
    for hop in (-apart, 0, apart):
        lo = centre + hop - 1
        power[:, lo:lo + 3] = 10 ** 6
    import dataclasses
    ion = dataclasses.replace(ion, power=power, _db=None)

    assert not interference.find(ion).any


def test_the_threshold_is_the_estimators_own(make_lfs):
    """Rejecting a row on evidence the estimators would ignore would be a
    different rule than the one documented."""
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)

    assert interference.find(ion, threshold_db=1e6).n_rows == 0
    assert interference.find(ion, threshold_db=43.0).n_rows == 1


def test_a_degenerate_axis_rejects_nothing(make_lfs):
    """A rule that cannot be evaluated must not fire."""
    ion = _sounding(make_lfs)
    one = ion.regated(sum(ion.cal.gate_km) / 2 - 0.01,
                      sum(ion.cal.gate_km) / 2 + 0.01)

    assert not interference.find(one).any


def test_too_few_bins_is_never_a_burst_however_coarse_the_axis(make_lfs):
    """MAX_ECHO_RANGE_KM is in km, so on a coarse axis a handful of bins can
    exceed it. At 234 km per bin a four-bin echo is 936 km, and without this
    guard the rule would reject the measurement it exists to protect."""
    from conftest import synth_iq

    iq = synth_iq(n_freq=200, window=512, echo_range_km=2700.0,
                  half_span_km=HALF_SPAN, echo_last_bin=199)
    coarse = spectro.compute(make_lfs(iq, name="coarse.lfs"),
                             window=512, gate_km=(2000.0, 5000.0))
    step = abs(float(coarse.cal.vrange[1] - coarse.cal.vrange[0]))
    assert step > interference.MAX_ECHO_RANGE_KM / interference.MIN_BURST_BINS

    import dataclasses
    power = np.array(coarse.power, copy=True)
    centre = power.shape[1] // 2
    power[:, centre - 2:centre + 2] = 10 ** 6      # four bins = 937 km here
    coarse = dataclasses.replace(coarse, power=power, _db=None)

    found = interference.find(coarse)
    assert found.occupied_km.max() > interference.MAX_ECHO_RANGE_KM
    assert not found.any, "km alone would have rejected an ordinary echo"


# --------------------------------------------------------------------------
# Suppressing
# --------------------------------------------------------------------------

def test_suppression_flattens_to_the_equalized_noise_floor(make_lfs):
    """Not zero, and not NaN: the value a cell that was never detected already
    reads, so a suppressed row is indistinguishable from an empty one."""
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    clean, found = interference.suppress(ion)

    assert np.allclose(clean.power[BURST_ROW], interference.EQUALIZED_NOISE_POWER)
    assert clean.db[BURST_ROW].max() == pytest.approx(spectro.NOISE_FLOOR_DB - 4.429,
                                               abs=0.01)
    assert found.n_rows == 1


def test_suppression_touches_no_other_row(make_lfs):
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    clean, _ = interference.suppress(ion)

    others = [i for i in range(N_FREQ) if i != BURST_ROW]
    assert np.array_equal(clean.power[others], ion.power[others])


def test_a_clean_sounding_is_returned_unchanged(make_lfs):
    """Identity, not a defensive copy: the common case must cost nothing."""
    ion = _sounding(make_lfs)
    same, found = interference.suppress(ion)

    assert same is ion
    assert not found.any


def test_suppression_does_not_mutate_its_input(make_lfs):
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    before = np.array(ion.power, copy=True)
    interference.suppress(ion)

    assert np.array_equal(ion.power, before)


# --------------------------------------------------------------------------
# The flag
# --------------------------------------------------------------------------

def test_apply_is_off_by_default(make_lfs):
    """It changes a MUF, so it must be asked for."""
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    same, found = interference.apply(ion, Options())

    assert same is ion and found is None


def test_apply_honours_the_flag(make_lfs):
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    clean, found = interference.apply(ion, Options(reject_interference=True))

    assert found.any
    assert clean is not ion


def test_describe_names_the_frequencies(make_lfs):
    """What the operator needs first: which part of the band is contaminated."""
    ion = _burst(_sounding(make_lfs), row=BURST_ROW)
    found = interference.find(ion)
    text = found.describe(ion.freq)

    assert f"{ion.freq[BURST_ROW]:.2f}" in text
    assert "1 row(s)" in text
    assert "km of range" in text
