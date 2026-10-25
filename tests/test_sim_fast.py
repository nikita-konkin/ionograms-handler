"""muf.sim.fast: the fast part read off a sounding's phase.

Two kinds of test. On gains drawn by :mod:`muf.sim.fading` and filtered as a
band's record is, the estimator's statistics -- bias, scatter, Rice and o/x
detection -- over many realisations, cheaply. On synthetic soundings, once
each, the whole chain: the tracker's traces, the second dechirp, the bands.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from muf import cli
from muf.sim import fading, fast, observe, presets, sounding
from muf.sim.state import MEASURED, ChannelState

SR = 1000.0          # the gain's rate before the band's filter, Hz
BAND_S = 30.0


def _bands(gain, *, n_bands: int = 3, snr_db: float = 30.0, pass_hz: float = 40.0,
           level=None, seed: int = 0) -> list[fast.BandEstimate]:
    """Bands as :func:`fast.band_estimate` makes them, of ``gain(t)`` plus noise."""
    rng = np.random.default_rng(10_000 + seed)
    stop = fast.STOP_RATIO * pass_hz
    step = math.floor(SR / fast.OUT_RATE_HZ)
    rate = SR / step
    trim = round(fast.TRIM_S * rate)
    # Noise of the density that gives snr_db in the pass band, at unit signal.
    n0 = 10.0 ** (-snr_db / 10.0) / fast._bandwidth(pass_hz, stop)
    out = []
    for i in range(n_bands):
        t = i * BAND_S + np.arange(0.0, BAND_S, 1.0 / SR)
        x = gain(t) * (1.0 if level is None else np.sqrt(level(t)))

        def noise(size=t.size):
            w = rng.standard_normal(size) + 1j * rng.standard_normal(size)
            return w * math.sqrt(n0 * SR / 2.0)

        z = fast._lowpass(x + noise(), SR, pass_hz, stop)[::step][trim:-trim]
        n = fast._lowpass(noise(), SR, pass_hz, stop)[::step][trim:-trim]
        b = fast.BandEstimate("m", 0.0, 1.0, used=True, rate_hz=rate, pass_hz=pass_hz,
                              seconds=z.size / rate)
        b.power = np.abs(z) ** 2
        b.noise_power = float(np.mean(np.abs(n) ** 2))
        b.noise_r = np.abs(fast._autocorrelation(n, min(z.size // 3,
                                                        round(fast.MAX_LAG_S * rate))))
        out.append(b)
    return out


def _gain(spread, shape="gaussian", k_db=None, seed=0):
    return fading.draw(3 * BAND_S, spread, shape=shape, k_db=k_db,
                       rng=np.random.default_rng(seed))


def test_lorentzian_r_is_the_filtered_spectrum():
    lags = np.arange(40) / fast.OUT_RATE_HZ
    for gamma, pass_hz in [(0.25, 10.0), (1.0, 10.0), (2.0, 40.0)]:
        stop = fast.STOP_RATIO * pass_hz
        nu = np.arange(0.0, stop, 0.0005) + 0.00025
        h = 0.5 - 0.5 * np.cos(np.pi * np.clip((stop - nu) / (stop - pass_hz), 0, 1))
        w = h ** 2 / (gamma ** 2 + nu ** 2)
        brute = (np.cos(2 * np.pi * np.outer(lags, nu)) @ w) / w.sum()
        np.testing.assert_allclose(fast.lorentzian_r(lags, gamma, pass_hz, stop), brute,
                                   atol=2e-3)
    # Unfiltered, the exponential.
    np.testing.assert_allclose(fast.lorentzian_r(lags, 0.5),
                               np.exp(-np.pi * lags), rtol=1e-12)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("pass_hz", [40.0, 10.0])
def test_spread_unbiased_on_gains(shape, pass_hz):
    # 90 s of a 0.5 Hz gain: some 45 fades. Measured over 24 realisations the
    # median reads 1.00-1.01 of the truth and the 68 % interval is +-8 %.
    ratios, shapes = [], []
    for seed in range(24):
        e = fast._aggregate("m", _bands(_gain(0.5, shape, seed=seed), pass_hz=pass_hz,
                                        seed=seed))
        ratios.append(e.spread_hz / 0.5)
        shapes.append(e.shape)
    assert np.median(ratios) == pytest.approx(1.0, abs=0.08)
    assert np.percentile(ratios, 84) - np.percentile(ratios, 16) < 0.35
    assert shapes.count(shape) >= 22


def test_fast_lorentzian_through_a_narrow_pass():
    # A 4 Hz Lorentzian's cusp is rounded by a 10 Hz pass; modelled, not a bias.
    ratios = [fast._aggregate("m", _bands(_gain(4.0, "lorentzian", seed=s),
                                          pass_hz=10.0, seed=s)).spread_hz / 4.0
              for s in range(4)]
    assert np.median(ratios) == pytest.approx(1.0, abs=0.06)


def test_rayleigh_gamma_within_its_error():
    # GAMMA_SPREAD bounds the measured scatter of gamma on Rayleigh gains.
    for spread in (0.25, 1.0):
        gammas = [fast._pooled_fit(_bands(_gain(spread, seed=s), seed=s),
                                   rice=False)["gamma"] for s in range(30)]
        seconds = 3 * (BAND_S - 2 * fast.TRIM_S)
        assert np.std(gammas) < fast.gamma_error(seconds, spread)
        assert np.mean(gammas) == pytest.approx(1.0, abs=0.1)


def test_rice_found_and_not_invented():
    found = [fast._aggregate("m", _bands(_gain(1.0, k_db=10.0, seed=s), seed=s))
             for s in range(8)]
    assert all(e.rice_k_db is not None for e in found)
    assert np.median([e.rice_k_db for e in found]) == pytest.approx(10.0, abs=1.5)
    assert np.median([e.spread_hz for e in found]) == pytest.approx(1.0, rel=0.1)
    rayleigh = [fast._aggregate("m", _bands(_gain(1.0, seed=s), seed=s))
                for s in range(8)]
    assert sum(e.rice_k_db is not None for e in rayleigh) <= 1


def test_two_waves_beating():
    # o and x 30 us apart beat at 3 Hz along a 100 kHz/s sweep; read as one
    # wave that is a 3 Hz spread, as two, the 0.5 Hz each has.
    beat, ratio = 3.0, 10.0 ** (-3.0 / 10.0)
    for seed in range(3):
        g_o, g_x = _gain(0.5, seed=seed), _gain(0.5, seed=100 + seed)

        def gain(t, g_o=g_o, g_x=g_x):
            x = g_x(t) * np.exp(2j * np.pi * beat * t)
            return (g_o(t) + math.sqrt(ratio) * x) / math.sqrt(1.0 + ratio)

        e = fast._aggregate("m", _bands(gain, seed=seed))
        assert e.beat_hz == pytest.approx(beat, rel=0.05)
        assert e.weaker_db == pytest.approx(-3.0, abs=2.0)
        assert e.spread_hz == pytest.approx(0.5, rel=0.3)
        assert e.spread_one_wave_hz > 2.0


def test_no_pair_in_one_wave():
    for shape in ("gaussian", "lorentzian"):
        for seed in range(6):
            e = fast._aggregate("m", _bands(_gain(0.5, shape, seed=seed), seed=seed))
            assert not np.isfinite(e.beat_hz)


def test_frozen_channel_is_a_bound():
    e = fast._aggregate("m", _bands(_gain(0.005, seed=1), seed=1))
    assert e.bound == "upper"
    assert e.spread_hz < 0.05


def test_level_swings_taken_off():
    # 8 dB of level over 8 s on a 2 Hz gain: gamma far above 1, the level
    # comes off by a running mean, and the spread stands.
    def level(t):
        return 10.0 ** (0.4 * np.sin(2 * np.pi * t / 8.0))

    ratios = []
    for seed in range(4):
        e = fast._aggregate("m", _bands(_gain(2.0, seed=seed), level=level, seed=seed))
        assert np.isfinite(e.detrended_s)
        ratios.append(e.spread_hz / 2.0)
    assert np.median(ratios) == pytest.approx(1.0, abs=0.12)


# --- the whole chain, on synthetic soundings ---------------------------------------

@pytest.fixture(scope="module")
def ox_sounding(tmp_path_factory):
    truth = presets.law_state()
    for m in truth.modes:
        m.doppler_spread_hz, m.ox_split_us, m.x_to_o_db = 0.5, 30.0, -3.0
    path = tmp_path_factory.mktemp("fast") / "ox.lfs"
    sounding.write_lfs(truth, path, seed=0)
    return path


def test_sounding_spread_and_split(ox_sounding):
    state, estimates = fast.measure(ox_sounding)
    e = max(estimates, key=lambda e: e.used)
    assert e.used >= 2
    assert e.spread_hz == pytest.approx(0.5, rel=0.3)
    assert e.split_us == pytest.approx(30.0, abs=2.0)
    m = next(m for m in state.modes if m.name == e.mode)
    assert m.ox_split_us == pytest.approx(e.split_us)
    assert m.provenance["magnetoionic"].origin == MEASURED
    assert m.provenance["doppler_spread"].origin == MEASURED


def test_sounding_one_wave(tmp_path):
    truth = presets.law_state()
    for m in truth.modes:
        m.doppler_spread_hz, m.doppler_shape = 1.0, "lorentzian"
    path = tmp_path / "one.lfs"
    sounding.write_lfs(truth, path, seed=1)
    got = observe.from_sounding(path)
    e = max(fast.estimate(fast.record_of(path), got), key=lambda e: e.used)
    assert e.spread_hz == pytest.approx(1.0, rel=0.2)
    assert e.shape == "lorentzian"
    assert not np.isfinite(e.split_us)


def test_cli_fast(ox_sounding, tmp_path, capsys):
    out = tmp_path / "state.json"
    rc = cli.main(["sim", "fast", str(ox_sounding), "--out", str(out),
                   "--json", str(tmp_path / "fast.json")])
    assert rc == 0
    assert "two waves beating" in capsys.readouterr().out
    state = ChannelState.load(out)
    assert any(m.ox_split_us is not None for m in state.modes)
