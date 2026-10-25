"""muf.sim.link: what a link at one carrier sees of a state."""

from __future__ import annotations

import math

import numpy as np
import pytest

from muf.sim import fading, link
from muf.sim.state import ChannelState, Circuit, Mode, Sweep


def _flat(name, delay, cn0, band=(13.3, 21.7), **kw) -> Mode:
    return Mode(name, list(band), [delay, delay], cn0, **kw)


def _state(*modes) -> ChannelState:
    return ChannelState(Circuit("cyprus1", "yoshkar-ola"), 1770206410.0, list(modes))


def _two(**kw) -> ChannelState:
    return _state(_flat("1F", 9137.0, 41.3, **kw), _flat("2F", 9871.0, 33.7))


def _spread(delays, cn0s, sigmas) -> float:
    power = [10.0 ** (c / 10.0) for c in cn0s]
    mean = sum(p * t for p, t in zip(power, delays)) / sum(power)
    return math.sqrt(sum(p * ((t - mean) ** 2 + s ** 2)
                         for p, t, s in zip(power, delays, sigmas)) / sum(power))


# --- two modes at a carrier (T-8) --------------------------------------------

def test_the_multipath_is_the_spread_of_the_delays():
    assert link.at(_two(), 17.25)["multipath_us"] == pytest.approx(734.0)


def test_the_delay_spread_weighs_each_mode_by_its_power():
    figures = link.at(_two(delay_spread_us=12.0), 17.25)
    assert figures["delay_spread_us"] == pytest.approx(
        _spread([9137.0, 9871.0], [41.3, 33.7], [12.0, 0.0]))


def test_the_coherence_bandwidth_is_a_fifth_of_the_inverse_spread():
    figures = link.at(_two(), 17.25)
    assert figures["coherence_bandwidth_khz"] == pytest.approx(
        1e3 / (5.0 * _spread([9137.0, 9871.0], [41.3, 33.7], [0.0, 0.0])))


def test_the_total_cn0_adds_the_modes_power():
    assert link.at(_two(), 17.25)["cn0_dbhz"] == pytest.approx(
        10.0 * math.log10(10.0 ** 4.13 + 10.0 ** 3.37))


def test_the_snr_is_the_cn0_in_the_links_band():
    assert link.at(_two(), 17.25, bandwidth_hz=2400.0)["snr_db"] == pytest.approx(
        10.0 * math.log10(10.0 ** 4.13 + 10.0 ** 3.37) - 10.0 * math.log10(2400.0))


def test_the_first_delay_is_the_earliest_mode():
    assert link.at(_two(), 17.25)["first_delay_us"] == pytest.approx(9137.0)


def test_the_gdd_is_the_strongest_modes():
    state = _state(Mode("1F", [13.3, 21.7], [9100.0, 9300.0], 41.3),
                   _flat("2F", 9871.0, 33.7))
    assert link.at(state, 17.25)["gdd_us_per_mhz"] == pytest.approx(200.0 / 8.4)


def test_one_mode_has_no_multipath():
    assert link.at(_state(_flat("1F", 9137.0, 41.3)), 17.25)["multipath_us"] is None


def test_one_path_bounds_no_coherence_bandwidth():
    figures = link.at(_state(_flat("1F", 9137.0, 41.3)), 17.25)
    assert figures["coherence_bandwidth_khz"] is None


def test_a_mode_under_the_floor_is_not_seen():
    assert link.at(_state(_flat("1F", 9137.0, 24.6)), 17.25)["modes"] == 0


def test_a_carrier_outside_every_band_sees_no_mode():
    assert not link.at(_two(), 25.0)["has_mode"]


def test_a_link_over_the_snr_floor_is_usable():
    assert link.at(_state(_flat("1F", 9137.0, 44.9)), 17.25)["usable"]


def test_a_link_under_the_snr_floor_is_not_usable():
    assert not link.at(_state(_flat("1F", 9137.0, 44.7)), 17.25)["usable"]


# --- the coherence time (T-9) ------------------------------------------------

@pytest.mark.parametrize("shape", fading.SHAPES)
def test_the_coherence_time_is_where_the_correlation_halves(shape):
    assert fading.autocorrelation(link.coherence_time_s(0.73, shape), 0.73, shape) \
        == pytest.approx(0.5)


def test_the_gaussian_coherence_time_is_the_specs():
    assert link.coherence_time_s(0.73, "gaussian") == pytest.approx(0.375 / 0.73,
                                                                    rel=1e-3)


def test_the_lorentzian_coherence_time_is_the_specs():
    assert link.coherence_time_s(0.73, "lorentzian") == pytest.approx(
        math.log(2.0) / (math.pi * 0.73))


@pytest.mark.parametrize("shape", fading.SHAPES)
def test_a_drawn_gain_decorrelates_to_half_at_the_coherence_time(shape):
    gain = fading.draw(3000.0, 0.73, rng=np.random.default_rng(7), shape=shape)
    lag = round(link.coherence_time_s(0.73, shape) * gain.rate_hz)
    g = gain.samples
    r = abs(np.vdot(g[:-lag], g[lag:])) / np.vdot(g, g).real * g.size / (g.size - lag)
    assert r == pytest.approx(0.5, abs=0.05)


def test_no_spread_bounds_no_coherence_time():
    assert link.coherence_time_s(0.0, "gaussian") is None


def test_the_coherence_time_is_the_strongest_modes():
    state = _state(_flat("1F", 9137.0, 41.3, doppler_spread_hz=0.73),
                   _flat("2F", 9871.0, 33.7, doppler_spread_hz=2.1))
    assert link.at(state, 17.25)["coherence_time_s"] == pytest.approx(
        link.coherence_time_s(0.73, "gaussian"))


# --- the carriers -------------------------------------------------------------

def test_the_february_sweep_has_101_carriers():
    sweep = Sweep(start_mhz=7.5, rate_hz_s=125e3, duration_s=200.0,
                  sample_rate_hz=40e3)
    assert link.carrier_grid(sweep.start_mhz, sweep.stop_mhz).size == 101


def test_the_carriers_step_from_the_start():
    assert link.carrier_grid(7.5, 8.4, 0.25).tolist() == [7.5, 7.75, 8.0, 8.25]
