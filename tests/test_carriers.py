"""Rejecting carriers: narrow in frequency, alone at their range.

The case that made the rule: Rostov -> Yoshkar-Ola on 2026-09-28, 16:11-16:21,
a bar 16.85-17.05 MHz by 1500-2000 km with nothing else in the ionogram, read
as MUF 17.05 by kmeans and contour. These pin that it goes, and above all the
three things measured on the archive that must *not* go: a trace's spread
nose, a trace the transmitter chops into columns, and a feature whose
surroundings were never recorded.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from muf import carriers, interference
from muf.pipeline import Options
from muf.spectro import Ionogram

STEP_MHZ = 0.05
FREQ = np.round(np.arange(8.0, 20.0 + STEP_MHZ / 2, STEP_MHZ), 4)
VRANGE = np.arange(3000.0, 1000.0, -2.0)        # km, descending like the real axis
HOT = 1e3                                       # 60 dB -- well over the 43 dB threshold


def _row(mhz: float) -> int:
    return int(np.argmin(np.abs(FREQ - mhz)))


def _col(km: float) -> int:
    return int(np.argmin(np.abs(VRANGE - km)))


def _ion(power: np.ndarray) -> Ionogram:
    cal = SimpleNamespace(freq=FREQ, vrange=VRANGE)
    return Ionogram(power=power.astype(np.float32), cal=cal, header=None,
                    window=0, zero_periods=0)


def _noise(seed: int = 0) -> np.ndarray:
    """Equalized noise: median ~0.36, nowhere near the threshold."""
    rng = np.random.default_rng(seed)
    return rng.exponential(0.52, size=(FREQ.size, VRANGE.size))


def _box(power, f_lo, f_hi, r_lo, r_hi, value=HOT):
    power[_row(f_lo):_row(f_hi) + 1, _col(r_hi):_col(r_lo) + 1] = value
    return power


def _carrier(power=None):
    """The Rostov shape: 0.25 MHz wide, 300 km tall, filled."""
    power = _noise() if power is None else power
    return _box(power, 16.85, 17.05, 1600.0, 1900.0)


def _trace(power, f_lo, f_hi, km_at_lo, km_per_mhz, width_km=10.0):
    for i in range(_row(f_lo), _row(f_hi) + 1):
        km = km_at_lo + km_per_mhz * (FREQ[i] - f_lo)
        _box(power, FREQ[i], FREQ[i], km - width_km / 2, km + width_km / 2)
    return power


# --------------------------------------------------------------------------
# What goes
# --------------------------------------------------------------------------

def test_the_rostov_carrier_is_found():
    found = carriers.find(_ion(_carrier()))

    assert found.n == 1
    (c,) = found.found
    assert c.freq_mhz == pytest.approx((16.85, 17.05))
    assert min(c.range_km) == pytest.approx(1600.0, abs=60)
    assert max(c.range_km) == pytest.approx(1900.0, abs=60)
    assert "16.85-17.05 MHz" in found.describe()


def test_suppressing_it_leaves_nothing_over_the_threshold():
    clean, found = carriers.suppress(_ion(_carrier()))

    assert found.any
    assert not (np.asarray(clean.db) > 43.0).any()


def test_a_stray_noise_row_does_not_save_it():
    """One row of noise at its range is a coincidence, not a trace."""
    power = _carrier()
    _box(power, 16.3, 16.3, 1700.0, 1710.0)
    assert carriers.find(_ion(power)).n == 1


def test_a_stripe_at_the_top_of_the_sweep_is_found():
    """The 30.2 MHz sweep-end transient: nothing above, recorded noise below."""
    power = _box(_noise(), 19.95, 20.0, 1100.0, 2900.0)
    found = carriers.find(_ion(power))
    assert found.n == 1
    assert found.found[0].freq_mhz[1] == pytest.approx(20.0)


def test_only_the_carriers_own_cells_are_flattened():
    """An echo at another range in the same rows is evidence; keep it."""
    power = _carrier()
    _box(power, 16.9, 17.0, 2600.0, 2620.0)         # not in the carrier's span
    _trace(power, 9.0, 17.0, 2600.0, 0.0)           # ...and part of a trace
    clean, found = carriers.suppress(_ion(power))

    assert found.n == 1
    hot = np.asarray(clean.db) > 43.0
    assert hot[_row(16.95), _col(2610.0)]
    assert not hot[_row(16.95), _col(1750.0)]


# --------------------------------------------------------------------------
# What stays
# --------------------------------------------------------------------------

def test_a_clean_sounding_has_none():
    found = carriers.find(_ion(_noise()))
    assert not found.any and found.describe() == "no carriers"


def test_a_trace_with_a_spread_nose_is_kept():
    """Near the MUF the rays converge and the echo spreads in range. Those top
    rows are wide and narrow in frequency -- but the trace is right below."""
    power = _trace(_noise(), 9.0, 16.8, 2700.0, 20.0)
    _box(power, 16.85, 17.0, 2750.0, 2950.0)
    assert not carriers.find(_ion(power)).any


def test_a_trace_chopped_into_columns_is_kept():
    """v2 transmitters skip bands: a trace becomes 0.15 MHz columns with empty
    rows between. Each column alone is as narrow as a carrier; 1,900 archived
    soundings had one. Its neighbours at the same range are what keep it."""
    power = _noise()
    for start in np.arange(8.0, 12.0, 0.4):
        _box(power, start, start + 0.1, 2700.0, 2800.0)
    assert not carriers.find(_ion(power)).any


def test_a_feature_where_recording_began_is_kept():
    """A late-analysed product has no rows below its first: one constant value
    each. No trace below is then not evidence of no trace below."""
    power = _carrier()
    power[:_row(16.85)] = 1.0 / interference.NOISE_COEF
    assert not carriers.find(_ion(power)).any


def test_a_feature_at_the_bottom_of_the_sweep_is_kept():
    power = _box(_noise(), 8.0, 8.15, 2600.0, 2900.0)
    assert not carriers.find(_ion(power)).any


def test_a_feature_wider_than_a_carrier_is_kept():
    power = _box(_noise(), 16.0, 16.6, 1600.0, 1900.0)
    assert not carriers.find(_ion(power)).any


def test_suppress_returns_the_same_object_when_nothing_is_found():
    ion = _ion(_noise())
    same, found = carriers.suppress(ion)
    assert same is ion and not found.any


# --------------------------------------------------------------------------
# The flag
# --------------------------------------------------------------------------

def test_apply_rejects_carriers_by_default():
    ion = _ion(_carrier())
    clean, rejected = interference.apply(ion, Options())

    assert clean is not ion
    assert rejected.any and rejected.carriers.n == 1
    assert rejected.bursts is None
    assert "16.85-17.05 MHz" in rejected.describe()


def test_apply_can_keep_them():
    ion = _ion(_carrier())
    same, rejected = interference.apply(ion, Options(reject_carriers=False))

    assert same is ion
    assert rejected.carriers is None and not rejected.any
    assert rejected.describe() == "nothing rejected"


def test_every_blob_in_the_same_rows_is_tested():
    """A transmitter stopping at 20 MHz can leave two or three blobs at
    different ranges in one search product. Taking only the heaviest left the
    others, and contour still read 19.997."""
    power = _box(_noise(), 19.85, 19.95, 1200.0, 1400.0)
    _box(power, 19.85, 19.95, 2400.0, 2550.0)
    found = carriers.find(_ion(power))
    assert found.n == 2
    clean, _ = carriers.suppress(_ion(power), found)
    assert not (np.asarray(clean.db) > 43.0).any()
