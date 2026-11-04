"""muf.sim.observe: from a sounding to a channel state."""

from __future__ import annotations

import random

import numpy as np

from muf.sim import observe

#: NIC to Yoshkar-Ola along the ground, by the station registry.
GROUND_KM = 2587.8

C_KM_S = 299792.458


def test_a_state_records_where_its_echo_begins(make_chirp_h5):
    from conftest import chirp_range_gates_m

    start = GROUND_KM + random.uniform(-900.0, 600.0)
    rows = random.randint(16, 24)
    power = np.random.exponential(1.0, (rows, 8192))
    km = chirp_range_gates_m(8192, 40_000.0, 100_000.0) / 1e3
    power[rows // 4:, (km >= start) & (km < start + 60.0)] *= 1e3
    path = make_chirp_h5(power, t0=1792843215.0, txname="NIC1",
                         station_name="yoshkar-ola", max_range_extent_km=8000.0,
                         name="lfm_ionogram-NIC1-yoshkar-ola-ch0-000-1792843215.00.h5")
    state = observe.from_ionogram(observe.load(path))
    assert abs(state.metrics["echo_offset_us"] - (start - GROUND_KM) / C_KM_S * 1e6) \
        < 200.0, "the state did not record where its echo begins"


def test_a_state_observed_for_one_hop_count_holds_no_other(make_chirp_h5):
    from conftest import chirp_range_gates_m

    rows = random.randint(40, 70)
    power = np.random.exponential(1.0, (rows, 8192))
    km = chirp_range_gates_m(8192, 40_000.0, 100_000.0) / 1e3
    low = random.uniform(2760.0, 2800.0)
    for start in (low, low + random.uniform(260.0, 300.0)):
        power[:, (km >= start) & (km < start + 15.0)] *= 1e3
    path = make_chirp_h5(power, t0=1792843215.0, txname="NIC1",
                         station_name="yoshkar-ola", max_range_extent_km=8000.0,
                         freqs_hz=np.linspace(10e6, random.uniform(18e6, 24e6), rows),
                         name="lfm_ionogram-NIC1-yoshkar-ola-ch0-000-1792843215.00.h5")
    state = observe.from_ionogram(observe.load(path), hops=1)
    assert {m.hops for m in state.modes} == {1}, \
        "a state observed for one hop count held modes of another"


def _layered(make_chirp_h5):
    """A sounding of a layer's 1F rays, a flat band leading up to its low ray."""
    from conftest import chirp_range_gates_m

    from muf.layer import Layer

    layer = Layer(random.uniform(8.0, 9.5), random.uniform(260.0, 360.0),
                  random.uniform(70.0, 130.0))
    f, group, _ = layer.trace(GROUND_KM)
    nose = int(np.argmax(f))
    muf = float(f[nose])
    offset = random.uniform(50.0, 120.0)
    freq = np.arange(10.0, muf + 3.0, 0.1)
    power = np.random.exponential(1.0, (freq.size, 16384))
    km = chirp_range_gates_m(16384, 40_000.0, 100_000.0) / 1e3
    start = random.uniform(13.0, 15.0)
    low = np.interp(freq, f[:nose + 1], group[:nose + 1]) + offset
    high = np.interp(freq, f[nose:][::-1], group[nose:][::-1]) + offset
    band = np.interp(start, freq, low) - random.uniform(30.0, 40.0)
    for i, mhz in enumerate(freq):
        for lit, at in ((start <= mhz <= muf, low[i]),
                        (0.75 * muf <= mhz <= muf, high[i]), (mhz < start, band)):
            if lit:
                power[i, np.abs(km - at) < 7.5] *= 1e3
    path = make_chirp_h5(power, t0=1792843215.0, txname="NIC1",
                         station_name="yoshkar-ola", max_range_extent_km=8000.0,
                         freqs_hz=freq * 1e6,
                         name="lfm_ionogram-NIC1-yoshkar-ola-ch0-000-1792843215.00.h5")
    return path, (freq, low)


def test_a_1f_mode_drawn_from_a_layer_keeps_off_the_band_below_it(make_chirp_h5):
    path, (freq, low) = _layered(make_chirp_h5)
    ion = observe.load(path)
    state = observe.from_ionogram(ion, curve="layer")
    mode = next(m for m in state.modes if m.name == "1F")
    assert np.max(np.abs(mode.delay_us * observe.km_per_us(ion.header)
                         - np.interp(mode.freq_mhz, freq, low))) < 15.0, \
        "a 1F mode observed through a layer strayed off its low ray"


def test_a_state_observed_through_a_layer_holds_its_two_1f_rays_alone(make_chirp_h5):
    path, _ = _layered(make_chirp_h5)
    state = observe.from_ionogram(observe.load(path), curve="layer")
    assert sorted(m.name for m in state.modes) == ["1F", "1F-high"], \
        "a state observed through a layer held other modes than its two 1F rays"
