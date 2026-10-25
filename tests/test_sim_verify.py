"""muf.sim.verify: the generator against closed-form theory."""

from __future__ import annotations

import math

import numpy as np
import pytest

from muf import cli
from muf.sim import fading, presets, verify


def test_f1487_test_length():
    # Annex 3 sec. 6: the longer of 3000/spread and 100/(BER * rate).
    assert verify.test_length_s(0.5) == 6000.0
    assert verify.test_length_s(10.0, ber=1e-4, bit_rate=300.0) == pytest.approx(3333.3,
                                                                                rel=1e-4)


def test_rayleigh_crossing_rate_is_jakes_in_rms_terms():
    # Jakes: sqrt(2 pi) f_D rho e^-rho^2, with the RMS width f_D / sqrt 2.
    f_d, rho = 3.0, 0.7
    jakes = math.sqrt(2 * math.pi) * f_d * rho * math.exp(-rho ** 2)
    assert verify.crossing_rate(rho, f_d / math.sqrt(2)) == pytest.approx(jakes)


def test_rice_reduces_to_rayleigh():
    r = np.linspace(0.0, 2.5, 11)
    np.testing.assert_allclose(verify.rice_cdf(r, 1e-9), verify.rice_cdf(r, 0.0),
                               atol=1e-6)
    assert verify.crossing_rate(1.0, 0.5, 1e-9) == pytest.approx(
        float(verify.crossing_rate(1.0, 0.5, 0.0)), rel=1e-6)


def test_k_by_moments_takes_the_noise_out():
    rng = np.random.default_rng(0)
    k = 4.0
    n = 400000
    s = (math.sqrt(k / (k + 1))
         + math.sqrt(1 / (k + 1)) * (rng.standard_normal(n)
                                     + 1j * rng.standard_normal(n)) / math.sqrt(2))
    noise = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    p = np.abs(s + noise) ** 2
    assert verify.rice_k_moments(p, noise_power=0.01) == pytest.approx(k, rel=0.05)


def test_half_power_width_of_known_shapes():
    f = np.linspace(-5, 5, 2001)
    exact = 10 ** 6                     # noise-free: as if averaged forever
    sigma = 0.4
    gauss = np.exp(-f ** 2 / (2 * sigma ** 2))
    assert verify.half_power_width(f, gauss, exact) == pytest.approx(
        2 * sigma * math.sqrt(2 * math.log(2)), rel=0.01)
    gamma = 0.5
    width = verify.half_power_width(f, 1 / (1 + (f / gamma) ** 2), exact)
    assert width == pytest.approx(2 * gamma, rel=0.01)


def test_dpsk_theory_limits():
    # Slow fading: 1/(2(1+g)); a decorrelated gain: 1/2 whatever the SNR.
    assert verify.dpsk_ber_theory(10.0, 1.0) == pytest.approx(1 / 22)
    assert verify.dpsk_ber_theory(40.0, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("shape", [fading.GAUSSIAN, fading.LORENTZIAN])
def test_fading_statistics_pass(shape):
    checks = verify.fading_statistics(1.0, shape=shape, duration_s=1500.0, seed=3)
    assert all(c.passed for c in checks), verify.format_checks(checks)


def test_rician_fading_statistics_pass():
    checks = verify.fading_statistics(1.0, k_db=6.0, duration_s=1500.0, seed=4)
    assert all(c.passed for c in checks), verify.format_checks(checks)


def test_f1487_scattering_function_reads_the_table():
    checks = verify.scattering_checks(presets.f1487("mid-disturbed"), 10.0,
                                      duration_s=1500.0, seed=5)
    assert all(c.passed for c in checks), verify.format_checks(checks)
    names = {c.name for c in checks}
    assert {"path2 delay", "path2 spread (2 sigma)"} <= names


def test_a_wrong_generator_fails_its_check(monkeypatch):
    """The battery has teeth: a spread drawn 30 % wide is caught."""
    real_draw = fading.draw

    def wide(duration_s, spread_hz, **kw):
        return real_draw(duration_s, 1.3 * spread_hz, **kw)

    monkeypatch.setattr(fading, "draw", wide)
    checks = verify.scattering_checks(presets.f1487("mid-disturbed"), 10.0,
                                      duration_s=1500.0, seed=5)
    failed = {c.name for c in checks if not c.passed}
    assert {"path1 spread (2 sigma)", "path2 spread (2 sigma)"} <= failed


@pytest.mark.parametrize("fc", [21.0, 24.0])
def test_dispersion_smears_over_gdd_times_band(fc):
    c = verify.dispersion_extent(presets.law_mode(muf_mhz=25.0), fc)
    assert c.passed, (c.measured, c.expected)


def test_dpsk_ber_matches_theory():
    checks = verify.dpsk_checks(2.0, (0.0, 10.0), duration_s=400.0, seed=6)
    assert all(c.passed for c in checks), verify.format_checks(checks)


def test_cli_verify_quick(tmp_path, capsys):
    out = tmp_path / "v.json"
    assert cli.main(["sim", "verify", "--quick", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "passed" in text and "DBPSK" in text
    assert out.exists()


@pytest.mark.parametrize("sigma", [20.0, 50.0])
def test_delay_spread_is_its_frequency_correlation(sigma):
    checks = verify.delay_spread_checks(sigma, duration_s=1500.0, seed=3)
    assert all(c.passed for c in checks), verify.format_checks(checks)
