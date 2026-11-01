"""muf.sim.observe: from a sounding to a channel state."""

from __future__ import annotations

import random

import numpy as np

from muf import trace
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


def hump(values):
    """How far a sequence rises and then falls again, the lesser of the two."""
    values = values[np.isfinite(values)]
    return float(np.max(np.minimum(values - np.minimum.accumulate(values),
                                   values - np.minimum.accumulate(values[::-1])[::-1])))


def test_no_mode_of_an_echo_that_humps_humps(make_chirp_h5):
    from conftest import chirp_range_gates_m

    rows = 200
    power = np.random.exponential(1.0, (rows, 8192))
    km = chirp_range_gates_m(8192, 40_000.0, 100_000.0) / 1e3
    top = GROUND_KM + random.uniform(300.0, 500.0)
    apex = random.uniform(0.4, 0.6)
    for row in range(rows):
        centre = top - random.uniform(150.0, 160.0) * ((row / rows - apex) / 0.5) ** 2
        power[row, (km >= centre - 15.0) & (km < centre + 15.0)] *= 1e3
    path = make_chirp_h5(power, t0=1792843215.0, txname="NIC1", fft_step=8000,
                         station_name="yoshkar-ola", max_range_extent_km=8000.0,
                         name="lfm_ionogram-NIC1-yoshkar-ola-ch0-000-1792843215.00.h5")
    assert max(hump(mode.delay_at(np.arange(mode.f_lo, mode.f_hi, 0.02)) * C_KM_S / 1e6)
               for mode in observe.from_ionogram(observe.load(path)).modes) \
        < trace.TURN_KM, "a mode of an echo that humps still humps"
