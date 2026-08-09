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


# --------------------------------------------------------------------------
# auto_gate -- fitting the range window to the echo
# --------------------------------------------------------------------------

def _synthetic(n_freq=300, n_range=2000, trace_at=-40.0, half_span=2000.0,
               seed=0, trace_from=0.5, thickness=8):
    """A noise field with one horizontal trace, on a descending axis."""
    rng = np.random.default_rng(seed)
    vrange = np.linspace(half_span, -half_span, n_range)
    power = rng.gamma(1.0, 1.0, size=(n_freq, n_range)).astype(np.float32)
    centre = int(np.argmin(np.abs(vrange - trace_at)))
    power[int(n_freq * trace_from):, centre - thickness:centre + thickness] += 30.0
    return power, vrange


def test_auto_gate_finds_the_trace_and_ignores_the_empty_axis():
    """The v2 search-mode case: +/-3998 km stored, a few hundred occupied."""
    power, vrange = _synthetic(trace_at=-40.0)
    lo, hi = calibrate.auto_gate(power, vrange)

    assert lo < -40.0 < hi
    assert (hi - lo) < 600.0, "gate should be a window, not most of the axis"
    full = vrange.max() - vrange.min()
    assert full / (hi - lo) > 5.0


def test_auto_gate_declines_on_pure_noise():
    """Inventing a window would crop away the evidence there is nothing here."""
    rng = np.random.default_rng(1)
    power = rng.gamma(1.0, 1.0, size=(300, 2000)).astype(np.float32)
    vrange = np.linspace(2000.0, -2000.0, 2000)

    assert calibrate.auto_gate(power, vrange) is None


def test_auto_gate_survives_a_few_very_bright_noise_cells():
    """Counting bright cells per range beats summing their power.

    `storage_snr_threshold = 2` keeps a lot of noise and float16 saturates at
    65504, so a handful of isolated cells can carry more total power than the
    whole trace. A sum-based window follows them; a count-based one does not,
    because a trace is contiguous in frequency and they are not. Measured on
    the 2026-08-05 archive: SNR-weighted returned 88 % of the axis, this
    returns 6 %.
    """
    power, vrange = _synthetic(trace_at=0.0, seed=3)
    rng = np.random.default_rng(11)
    rows = rng.integers(0, power.shape[0], 40)
    cols = rng.integers(0, power.shape[1], 40)
    power[rows, cols] = 65504.0

    lo, hi = calibrate.auto_gate(power, vrange)
    assert (hi - lo) < 600.0
    assert lo < 0.0 < hi


def test_auto_gate_is_indifferent_to_axis_direction():
    """`.lfs` and v2 disagree about which way range runs; the gate must not."""
    power, vrange = _synthetic(trace_at=-40.0)
    down = calibrate.auto_gate(power, vrange)
    up = calibrate.auto_gate(power[:, ::-1], vrange[::-1])

    assert down == pytest.approx(up, abs=5.0)


def test_auto_gate_returns_none_rather_than_raising_on_junk():
    assert calibrate.auto_gate(np.zeros((0, 0)), np.array([])) is None
    assert calibrate.auto_gate(np.zeros((1, 5)), np.arange(5.0)) is None


def test_geometry_gate_brackets_the_hop_families():
    """Both edges come from `trace.hop_range_km`, the same formula
    `identify_hops` labels segments with, so a gate and a hop label cannot
    disagree about what a range means."""
    from muf import calibrate, trace

    class _H:
        is_oblique = True
        tx_latitude, tx_longitude = 54.63, 13.37      # Juliusruh
        rx_latitude, rx_longitude = 62.073, 9.111     # DOB

    lo, hi = calibrate.geometry_gate(_H(), margin_km=0.0)
    ground = calibrate.ground_range_km(_H())

    assert lo == pytest.approx(trace.hop_range_km(ground, 1,
                                                  min(trace.HOP_HEIGHTS_KM)))
    assert hi == pytest.approx(trace.hop_range_km(ground,
                                                  calibrate.DEFAULT_MAX_HOPS,
                                                  max(trace.HOP_HEIGHTS_KM)))
    assert 0 < lo < hi


def test_geometry_gate_is_none_without_a_usable_path():
    """Vertical sounding and missing coordinates both mean "no path", which is
    the caller's cue to fall back rather than to crop on a guess."""
    from muf import calibrate

    class _Vertical:
        is_oblique = False
        tx_latitude = tx_longitude = rx_latitude = rx_longitude = 0.0

    class _NoCoords:
        is_oblique = True
        tx_latitude = tx_longitude = float("nan")
        rx_latitude, rx_longitude = 62.0, 9.0

    assert calibrate.geometry_gate(_Vertical()) is None
    assert calibrate.geometry_gate(_NoCoords()) is None
