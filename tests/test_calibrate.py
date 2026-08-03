"""Axes and range gating.

The orientation test is the important one: it is the defect these modules were
written to fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import calibrate
from muf.io_lfs import read_header


@pytest.fixture
def header(make_lfs):
    return read_header(make_lfs(np.zeros(16, dtype=np.complex64)))


def test_sweep_bounds(header):
    start, stop = calibrate.sweep_bounds(header)
    assert start == pytest.approx(7.5)
    assert stop == pytest.approx(32.5)


def test_range_half_span(header):
    # 3e5 km/s * (40000 Hz / 2) / 100000 Hz/s
    assert calibrate.range_half_span(header) == pytest.approx(60_000.0)


def test_resolution_equals_unpadded_bin_spacing(header):
    """Zero-padding subdivides bins; it does not resolve more.

    The whole justification for defaulting zero_periods to 0.
    """
    window = 8192
    resolution = calibrate.range_resolution_km(header, window)
    assert resolution == pytest.approx(14.648, abs=0.01)

    unpadded = calibrate.build(header, n_freq=10, window=window, zero_periods=0)
    padded = calibrate.build(header, n_freq=10, window=window, zero_periods=10)

    assert unpadded.range_step == pytest.approx(resolution)
    assert padded.range_step == pytest.approx(resolution / 11, rel=1e-6)
    assert padded.resolution_km == pytest.approx(unpadded.resolution_km)


def test_range_axis_descends(header):
    """MUF.py's plot axis ascends from -R; the extractor needs the reverse."""
    cal = calibrate.build(header, n_freq=10, window=8192,
                          gate_km=(2000.0, 5000.0))
    assert cal.vrange[0] > cal.vrange[-1]
    assert np.all(np.diff(cal.vrange) < 0)


def test_echo_bin_maps_to_positive_range(header):
    """The anchor: bin 3909 is where the real 2026-02-04 echo sits.

    Under linspace(-R, +R) it maps to -2732 km, which no echo can occupy.
    """
    cal = calibrate.build(header, n_freq=10, window=8192, gate_km=(2000.0, 5000.0))
    mapped = cal.half_span - 3909 * cal.range_step

    assert mapped == pytest.approx(2739.3, abs=0.1)
    assert mapped > 0


def test_gate_indices_match_measurement(header):
    cal = calibrate.build(header, n_freq=10, window=8192, gate_km=(2000.0, 5000.0))
    assert cal.gate_idx == (3755, 3959)
    assert cal.n_range == 205
    # 205 of 8192 bins is the reduction that makes the pipeline cheap.
    assert cal.n_range_full / cal.n_range == pytest.approx(40, abs=1)


def test_gate_covers_requested_span(header):
    cal = calibrate.build(header, n_freq=10, window=8192, gate_km=(2500.0, 4000.0))
    assert cal.vrange.max() >= 4000.0 - cal.range_step
    assert cal.vrange.min() <= 2500.0 + cal.range_step


def test_empty_gate_rejected(header):
    with pytest.raises(ValueError, match="empty range gate"):
        calibrate.gate_indices(60_000.0, 14.648, 70_000.0, 80_000.0, 8192)


def test_gate_clamps_to_axis(header):
    """A gate wider than the axis clamps to it rather than overrunning."""
    lo, hi = calibrate.gate_indices(60_000.0, 14.648, -1e9, 1e9, 8192)
    assert (lo, hi) == (0, 8191)


def test_gate_maps_negative_ranges(header):
    """The axis runs to -60,000 km; -10,000 km is a real index, not a clamp."""
    _, hi = calibrate.gate_indices(60_000.0, 14.648, -10_000.0, 60_000.0, 8192)
    assert hi == 4778


def test_ground_range(header):
    """Cyprus to Yoshkar-Ola, from the coordinates in the header."""
    assert calibrate.ground_range_km(header) == pytest.approx(2588, abs=5)


def test_ground_range_none_for_vertical(make_lfs):
    header = read_header(
        make_lfs(np.zeros(16, dtype=np.complex64), tx_name="s", rx_name="s")
    )
    assert calibrate.ground_range_km(header) is None


def test_default_gate_excludes_below_ground_path(header):
    """No echo can arrive sooner than the ground path, whatever rmin says."""
    lo, hi = calibrate.default_gate(header)
    assert header.rmin == 0
    assert lo > 2000
    assert hi == 5000.0


def test_frequency_axis(header):
    """Bins are labelled at their centre, half a step above the window start."""
    cal = calibrate.build(header, n_freq=1220, window=8192)
    step = cal.freq_step_mhz

    assert cal.n_freq == 1220
    assert step * 1e3 == pytest.approx(20.48, abs=0.01)
    assert cal.freq[0] == pytest.approx(7.5 + step / 2)
    assert cal.freq[-1] == pytest.approx(32.5 - step / 2, abs=0.05)


def test_frequency_axis_comes_from_the_chirp_rate(header):
    """A complete sweep must land on the nominal endpoint either way.

    The rate-derived axis and the old linspace agree here; they diverge only
    when a recording is short, which is the case that matters.
    """
    cal = calibrate.build(header, n_freq=1220, window=8192)
    assert cal.freq_stop == pytest.approx(32.5, abs=0.05)
    assert cal.sweep_complete
    assert cal.sweep_fraction == pytest.approx(1.0, abs=0.01)


def test_truncated_recording_is_not_stretched(header):
    """A recording cut short must not have its axis stretched to the nominal top.

    Real case: 10 files in data 2026.02.05 hold 347 windows instead of 1220
    while still declaring dur=250. Stretching puts their last bin at 32.5 MHz
    when the transmitter had only reached ~14.6, inflating MUF by up to 2.2x.
    """
    cal = calibrate.build(header, n_freq=347, window=8192)

    assert cal.freq_stop == pytest.approx(14.6, abs=0.1)
    assert cal.freq[-1] < 15.0
    assert not cal.sweep_complete
    assert cal.sweep_fraction == pytest.approx(0.284, abs=0.01)
    # The nominal value stays available for reference.
    assert cal.freq_stop_nominal == pytest.approx(32.5)


def test_step_is_independent_of_recording_length(header):
    """Truncation changes how far the sweep got, never the bin spacing."""
    full = calibrate.build(header, n_freq=1220, window=8192)
    short = calibrate.build(header, n_freq=347, window=8192)

    assert short.freq_step_mhz == pytest.approx(full.freq_step_mhz)
    np.testing.assert_allclose(short.freq, full.freq[:347])
