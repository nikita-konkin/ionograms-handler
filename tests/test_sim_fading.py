"""muf.sim.fading: Watterson tap gains with F.1487's spread convention."""

from __future__ import annotations

import numpy as np
import pytest

from muf.sim import fading

RATE = 64.0


def _sampled(spread, seconds=4000.0, seed=3, **kw):
    gain = fading.draw(seconds, spread, rng=np.random.default_rng(seed), **kw)
    t = np.arange(0.0, seconds, 1.0 / RATE)
    return gain, gain(t)


def test_mean_power_is_one():
    _, g = _sampled(2.0)
    assert np.mean(np.abs(g) ** 2) == pytest.approx(1.0, rel=0.05)


def test_the_spread_is_two_sigma():
    """F.1487 quotes 2 sigma: a 2 Hz spread is a spectrum 1 Hz wide (RMS)."""
    _, g = _sampled(2.0)
    _, _, width = fading.doppler_spectrum(g, RATE, segment_s=20.0)
    assert width == pytest.approx(1.0, rel=0.05)


def test_the_envelope_is_rayleigh():
    _, g = _sampled(2.0)
    power = np.abs(g) ** 2 / np.mean(np.abs(g) ** 2)
    # Exponential power: median ln 2, and 1 - 1/e of it below the mean.
    assert np.median(power) == pytest.approx(np.log(2.0), rel=0.05)
    assert np.mean(power < 1.0) == pytest.approx(1.0 - np.exp(-1.0), abs=0.02)


def test_a_shift_moves_the_spectrum():
    _, g = _sampled(1.0, shift_hz=3.0)
    freq, power, _ = fading.doppler_spectrum(g, RATE, segment_s=20.0)
    assert np.sum(freq * power) / np.sum(power) == pytest.approx(3.0, abs=0.05)


def test_no_spread_is_no_fading():
    _, g = _sampled(0.0, seconds=10.0)
    np.testing.assert_allclose(np.abs(g), 1.0)


def test_a_specular_part_carries_its_share():
    """K = 10 dB: the mean of the gain is the specular part, 10/11 of the power."""
    _, g = _sampled(1.0, k_db=10.0)
    assert abs(np.mean(g)) ** 2 == pytest.approx(10.0 / 11.0, abs=0.03)
    assert np.mean(np.abs(g) ** 2) == pytest.approx(1.0, rel=0.05)


def test_the_same_seed_draws_the_same_gain():
    a, _ = _sampled(1.0, seconds=30.0, seed=7)
    b, _ = _sampled(1.0, seconds=30.0, seed=7)
    t = np.linspace(0.0, 30.0, 999)
    np.testing.assert_array_equal(a(t), b(t))


def test_a_negative_spread_is_refused():
    with pytest.raises(ValueError):
        fading.draw(1.0, -0.1)
