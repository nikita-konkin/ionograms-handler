"""muf.sim.channel: a signal through the state at one carrier."""

from __future__ import annotations

import numpy as np
import pytest

from muf.sim import channel, presets
from muf.sim.state import ChannelState, Circuit, Mode

STILL = channel.Effects(fading=False)


def _state(*modes) -> ChannelState:
    return ChannelState(Circuit("a", "b"), 0.0, modes=list(modes), source="test")


def _flat(name="m", delay=1000.0, cn0=50.0, band=(8.0, 14.0), **kw) -> Mode:
    return Mode(name, list(band), [delay, delay], cn0, **kw)


def _linear(name="d", gdd=30.0, centre=11.0, delay=9000.0, band=(8.0, 14.0)) -> Mode:
    f = np.arange(band[0], band[1] + 1e-9, 0.05)
    return Mode(name, f, delay + gdd * (f - centre), 50.0)


def _impulse(n=4096):
    x = np.zeros(n, dtype=np.complex128)
    x[0] = 1.0
    return x


def test_two_paths_arrive_at_their_delays_with_equal_power():
    out = channel.simulate(presets.f1487("mid-moderate"), _impulse(), 48e3, 10.0,
                           effects=STILL, seed=1)
    y = out.samples
    peaks = sorted(np.argsort(-np.abs(y))[:2] - out.lead_samples)
    assert peaks == [0, 48]                        # 1 ms at 48 kHz
    p = np.abs(y[out.lead_samples + np.array(peaks)]) ** 2
    assert p[0] == pytest.approx(p[1], rel=1e-6)
    assert np.sum(np.abs(y) ** 2) == pytest.approx(1.0, rel=1e-3)


def test_a_doppler_shift_moves_a_tone():
    shift = 4.7
    state = _state(_flat(doppler_shift_hz=shift))
    fs, n = 2000.0, 20000
    out = channel.simulate(state, np.ones(n), fs, 10.0, effects=STILL, seed=2)
    y = out.samples[out.lead_samples + 100: out.lead_samples + n - 100]
    measured = np.angle(np.sum(y[1:] * np.conj(y[:-1]))) * fs / (2 * np.pi)
    assert measured == pytest.approx(shift, abs=1e-3)


def test_the_group_delay_is_the_curve():
    """-(1/2 pi) dphi/df of the full filter equals tau(fc + df) - tau_ref."""
    m = presets.law_mode(muf_mhz=25.0)
    state = _state(m)
    real = channel.Realisation(state, 22.0, 400e3, 0.1, effects=STILL, seed=0)
    path = real.paths[0]
    df = np.linspace(-150e3, 150e3, 3001)
    phase = np.unwrap(np.angle(real.response(path, df)))
    group_us = -np.gradient(phase, df) / (2 * np.pi) * 1e6
    want = m.delay_at(22.0 + df / 1e6) - real.reference_delay_us
    np.testing.assert_allclose(group_us[5:-5], want[5:-5], atol=0.01)


def test_gdd_truncation_is_linear_in_delay():
    m = presets.law_mode(muf_mhz=25.0)
    state = _state(m)
    real = channel.Realisation(state, 22.0, 400e3, 0.1, seed=0,
                               effects=channel.Effects(fading=False, dispersion="gdd"))
    path = real.paths[0]
    df = np.linspace(-150e3, 150e3, 3001)
    phase = np.unwrap(np.angle(real.response(path, df)))
    group_us = -np.gradient(phase, df) / (2 * np.pi) * 1e6
    want = path.gdd_us_per_mhz * df / 1e6
    np.testing.assert_allclose(group_us[5:-5], want[5:-5], atol=0.01)


def _burst(fs, n, offset_hz, centre, width):
    t = np.arange(n)
    envelope = np.exp(-0.5 * ((t - centre) / width) ** 2)
    return envelope * np.exp(2j * np.pi * offset_hz * t / fs)


def _arrival(y, fs, offset_hz, lead):
    """Envelope peak of the part of ``y`` near ``offset_hz``, in samples."""
    spectrum = np.fft.fft(y)
    df = np.fft.fftfreq(y.size, 1 / fs)
    sub = np.fft.ifft(spectrum * (np.abs(df - offset_hz) < 60e3))
    env = np.abs(sub)
    k = int(np.argmax(env))
    a, b, c = env[k - 1:k + 2]
    return k + 0.5 * (a - c) / (a - 2 * b + c) - lead


def test_gdd_separates_subcarriers_by_gdd_times_their_spacing():
    """30 us/MHz across 400 kHz: the upper burst arrives 12 us after the lower."""
    fs, n, gdd = 1e6, 1 << 14, 30.0
    state = _state(_linear(gdd=gdd))
    x = _burst(fs, n, -200e3, 3000, 60) + _burst(fs, n, 200e3, 3000, 60)
    out = channel.simulate(state, x, fs, 11.0, effects=STILL, seed=3)
    early = _arrival(out.samples, fs, -200e3, out.lead_samples)
    late = _arrival(out.samples, fs, 200e3, out.lead_samples)
    assert (late - early) / fs * 1e6 == pytest.approx(gdd * 0.4, abs=0.2)


def test_dispersion_none_keeps_the_subcarriers_together():
    fs, n = 1e6, 1 << 14
    state = _state(_linear(gdd=30.0))
    x = _burst(fs, n, -200e3, 3000, 60) + _burst(fs, n, 200e3, 3000, 60)
    out = channel.simulate(state, x, fs, 11.0, seed=3,
                           effects=channel.Effects(fading=False, dispersion="none"))
    early = _arrival(out.samples, fs, -200e3, out.lead_samples)
    late = _arrival(out.samples, fs, 200e3, out.lead_samples)
    assert abs(late - early) < 0.05


def test_nothing_passes_above_the_muf():
    m = _linear(band=(8.0, 11.05))
    real = channel.Realisation(_state(m), 11.0, 400e3, 0.1, effects=STILL, seed=0)
    path = real.paths[0]
    taper = channel.EDGE_TAPER_FRACTION * 400e3
    above = np.linspace(50e3 + taper / 2 + 1, 190e3, 50)
    below = np.linspace(-190e3, 50e3 - taper / 2 - 1, 50)
    assert np.all(np.abs(real.response(path, above)) == 0)
    assert np.all(np.abs(real.response(path, below)) > 0)


def test_the_ox_split_makes_two_arrivals():
    m = _flat(ox_split_us=50.0, x_to_o_db=-3.0)
    out = channel.simulate(_state(m), _impulse(), 1e6, 10.0, effects=STILL, seed=4)
    y = out.samples
    o, x = y[out.lead_samples], y[out.lead_samples + 50]
    assert 10 * np.log10(abs(x) ** 2 / abs(o) ** 2) == pytest.approx(-3.0, abs=0.01)
    assert abs(o) ** 2 + abs(x) ** 2 == pytest.approx(1.0, rel=1e-3)


def test_magnetoionic_off_leaves_one_arrival():
    m = _flat(ox_split_us=50.0, x_to_o_db=-3.0)
    out = channel.simulate(_state(m), _impulse(), 1e6, 10.0, seed=4,
                           effects=channel.Effects(fading=False, magnetoionic=False))
    assert abs(out.samples[out.lead_samples + 50]) < 1e-6


def test_the_noise_is_the_snr_asked_for():
    state = _state(_flat())
    x = np.exp(2j * np.pi * 0.05 * np.arange(200000))
    clean = channel.simulate(state, x, 48e3, 10.0, effects=STILL, seed=5)
    noisy = channel.simulate(state, x, 48e3, 10.0, effects=STILL, seed=5, snr_db=10.0)
    noise = noisy.samples - clean.samples
    assert np.mean(np.abs(noise) ** 2) == pytest.approx(0.1, rel=0.02)
    assert noisy.noise_variance == pytest.approx(0.1)


def test_snr_in_a_narrower_band_raises_the_density():
    state = _state(_flat())
    out = channel.simulate(state, np.ones(1000), 48e3, 10.0, effects=STILL, seed=5,
                           snr_db=10.0, snr_bandwidth_hz=3e3)
    assert out.noise_variance == pytest.approx(0.1 * 16.0)


def test_absolute_levels_are_the_states():
    """Against noise of unit density a path's power is its C/N0."""
    state = _state(_flat(cn0=40.0))
    real = channel.Realisation(state, 10.0, 1e6, 0.1, effects=STILL, seed=6)
    assert abs(real.transfer([0.0], [0.0])[0, 0]) ** 2 == pytest.approx(1e4)
    out = real.apply(_impulse(), absolute=True)
    assert out.signal_scale == 1.0
    assert out.noise_variance == pytest.approx(1e6)


def test_multipath_off_keeps_the_strongest_mode():
    state = _state(_flat("weak", delay=1000.0, cn0=40.0),
                   _flat("strong", delay=2000.0, cn0=50.0))
    out = channel.simulate(state, _impulse(), 48e3, 10.0, seed=7,
                           effects=channel.Effects(fading=False, multipath=False))
    assert [p["path"] for p in out.paths] == ["strong"]


def test_transfer_is_the_draw_apply_uses():
    state = _state(_flat(doppler_spread_hz=1.0))
    fs, n = 8e3, 8000
    real = channel.Realisation(state, 10.0, fs, 2.0, seed=8)
    out = real.apply(_impulse(n))
    t = np.arange(out.samples.size) / fs
    k = out.lead_samples
    h = real.transfer([0.0], t[k:k + 1])[0, 0] * out.signal_scale
    assert out.samples[k] == pytest.approx(h, rel=1e-6)


def test_a_carrier_no_mode_reaches_is_refused():
    with pytest.raises(ValueError, match="no mode"):
        channel.simulate(_state(_flat()), _impulse(), 48e3, 20.0)


def test_an_unknown_dispersion_is_refused():
    with pytest.raises(ValueError):
        channel.Effects(dispersion="all")


def test_a_delay_spread_is_independent_paths_about_the_delay():
    m = _flat(delay_spread_us=20.0, doppler_spread_hz=1.0)
    paths = channel.components(_state(m), 10.0, tap_step_us=5.0)
    shares = np.array([10 ** (p.level_db / 10) for p in paths])
    offsets = np.array([p.tap_us for p in paths])
    assert shares.sum() == pytest.approx(1.0)
    assert np.sqrt(np.sum(shares * offsets ** 2)) == pytest.approx(20.0, rel=0.005)
    assert [p.delay_us - 1000.0 for p in paths] == pytest.approx(offsets.tolist())
    real = channel.Realisation(_state(m), 10.0, 48e3, 1.0, seed=1)
    assert len({id(g) for g in real.gains}) == len(real.paths)
    off = channel.components(_state(m), 10.0,
                             channel.Effects(delay_spread=False))
    assert len(off) == 1 and off[0].tap_us == 0.0
