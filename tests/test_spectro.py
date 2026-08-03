"""Spectrogram formation, gating and caching."""

from __future__ import annotations

import numpy as np
import pytest

from muf import spectro
from muf.io_lfs import read_header

from conftest import snapped_range, synth_iq

WINDOW = 512
N_FREQ = 40
HALF_SPAN = 60_000.0


@pytest.fixture
def path(make_lfs):
    iq = synth_iq(
        n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
        half_span_km=HALF_SPAN, echo_last_bin=N_FREQ - 1,
    )
    return make_lfs(iq)


def test_shape_follows_gate(path):
    ion = spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))
    assert ion.shape[0] == N_FREQ
    assert ion.shape[1] == ion.cal.n_range
    assert ion.shape[1] < WINDOW            # the point of gating


def test_gating_does_not_move_the_echo(path):
    """A wider gate must not shift where the echo is reported."""
    narrow = spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))
    wide = spectro.compute(path, window=WINDOW, gate_km=(500.0, 20000.0))

    expected = snapped_range(2700.0, HALF_SPAN, WINDOW)
    for ion in (narrow, wide):
        peak = ion.vrange[int(ion.power[N_FREQ // 2].argmax())]
        assert peak == pytest.approx(expected, abs=2 * HALF_SPAN / WINDOW)


def test_noise_floor_is_equalised(path):
    """Median equalization pins the noise floor, so thresholds are portable.

    stuffr's NOISE_COEF scales the median to the mean of an exponential, which
    leaves the median of the equalized array at 1/(4 ln 2).
    """
    # Measured over a wide gate: the divisor is the median of the *full*
    # spectrum, and a gate tight around the echo is biased upward by it.
    ion = spectro.compute(path, window=WINDOW, gate_km=(-50_000.0, 50_000.0))
    assert float(np.median(ion.power)) == pytest.approx(1 / spectro.NOISE_COEF, rel=0.3)


def test_db_conversion_reference():
    """0 dB is 1e-3 in equalized power, so linear 20 is 43 dB.

    That correspondence is what lets the algorithmic and threshold estimators
    use the same physical level.
    """
    assert spectro.to_db(np.array([1e-3]))[0] == pytest.approx(0.0)
    assert spectro.to_db(np.array([20.0]))[0] == pytest.approx(43.0, abs=0.1)


def test_db_has_no_negative_infinity(path):
    """comprz_dB's job: keep zeros from becoming -inf."""
    ion = spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))
    ion.power[0, 0] = 0.0
    assert np.all(np.isfinite(spectro.to_db(ion.power)))


def test_zero_padding_subdivides_bins(path):
    """Padding gives more bins over the same span, at the same resolution."""
    plain = spectro.compute(path, window=WINDOW, zero_periods=0,
                            gate_km=(2000.0, 5000.0))
    padded = spectro.compute(path, window=WINDOW, zero_periods=3,
                             gate_km=(2000.0, 5000.0))

    assert padded.shape[1] > plain.shape[1]
    assert padded.cal.range_step == pytest.approx(plain.cal.range_step / 4, rel=1e-6)
    assert padded.cal.resolution_km == pytest.approx(plain.cal.resolution_km)


def test_cache_round_trip(path, tmp_path):
    cache = tmp_path / "cache"
    first = spectro.compute_cached(path, window=WINDOW, gate_km=(2000.0, 5000.0),
                                   cache_dir=cache)
    assert list(cache.glob("*.npz"))

    second = spectro.compute_cached(path, window=WINDOW, gate_km=(2000.0, 5000.0),
                                    cache_dir=cache)
    np.testing.assert_allclose(first.power, second.power)
    assert second.cal.gate_idx == first.cal.gate_idx
    assert second.header.datetime == first.header.datetime


def test_corrupt_cache_falls_back_to_recompute(path, tmp_path):
    cache = tmp_path / "cache"
    spectro.compute_cached(path, window=WINDOW, gate_km=(2000.0, 5000.0),
                           cache_dir=cache)
    for entry in cache.glob("*.npz"):
        entry.write_bytes(b"not an npz")

    ion = spectro.compute_cached(path, window=WINDOW, gate_km=(2000.0, 5000.0),
                                 cache_dir=cache)
    assert ion.shape[0] == N_FREQ


def test_cache_key_distinguishes_settings(path):
    keys = {
        spectro.cache_key(path, w, z, g)
        for w, z, g in [(512, 0, None), (1024, 0, None),
                        (512, 3, None), (512, 0, (2000.0, 5000.0))]
    }
    assert len(keys) == 4


def test_rejects_file_shorter_than_one_window(make_lfs):
    path = make_lfs(np.zeros(10, dtype=np.complex64))
    with pytest.raises(ValueError, match="shorter than one"):
        spectro.compute(path, window=WINDOW)


def test_real_sounding_geometry(real_file):
    ion = spectro.compute(real_file)
    header = read_header(real_file)

    assert ion.shape[0] == 1220
    assert ion.cal.freq_start == pytest.approx(7.5)
    # 1220 whole windows reach 32.4856 MHz; the leftover samples do not fill a
    # 1221st, so the actual top sits just under the nominal 32.5.
    assert ion.cal.freq_stop == pytest.approx(32.5, abs=ion.cal.freq_step_mhz)
    assert ion.cal.freq_stop_nominal == pytest.approx(32.5)
    assert ion.cal.sweep_complete
    assert ion.header.tx_name == header.tx_name
    # Gating must leave a small fraction of the 8192-bin axis.
    assert ion.shape[1] < 400
