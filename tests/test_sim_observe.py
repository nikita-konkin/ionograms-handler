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
