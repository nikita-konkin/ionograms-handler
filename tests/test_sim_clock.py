"""muf.sim.clock: the pairs the receiver's clock broke."""

from __future__ import annotations

import random

import pytest

from muf.sim import clock, presets
from muf.sim.state import ChannelState
from services.agent import health


def _state(offset_us: float | None) -> ChannelState:
    return ChannelState(presets.cyprus1_circuit(), 1792843215.0,
                        metrics={"echo_offset_us": offset_us})


def _late() -> float:
    return clock.window_us(presets.cyprus1_circuit().ground_km)[1]


def test_an_echo_that_moved_past_the_limit_makes_the_pair_suspect():
    a = random.uniform(100.0, 700.0)
    b = a + random.uniform(clock.JUMP_US + 5.0, 1200.0)
    assert clock.verdict(_state(a), _state(b))["suspect"] is True, \
        "an echo that moved past the limit did not make the pair suspect"


def test_an_echo_that_moved_within_the_limit_leaves_the_pair_alone():
    a = random.uniform(100.0, 700.0)
    b = a + random.uniform(-clock.JUMP_US + 5.0, clock.JUMP_US - 5.0)
    assert clock.verdict(_state(a), _state(b))["suspect"] is False, \
        "an echo that moved within the limit made the pair suspect"


def test_an_echo_before_the_ground_line_makes_the_pair_suspect():
    a = random.uniform(-4000.0, -600.0)
    found = clock.verdict(_state(a), _state(a + random.uniform(-90.0, 90.0)))
    assert found["suspect"] is True, \
        "an echo before the ground line did not make the pair suspect"


def test_an_echo_later_than_two_hops_makes_the_pair_suspect():
    b = _late() + random.uniform(100.0, 3000.0)
    found = clock.verdict(_state(b + random.uniform(-90.0, 90.0)), _state(b))
    assert found["suspect"] is True, \
        "an echo later than two hops did not make the pair suspect"


def test_an_echo_not_read_leaves_the_pair_alone():
    found = clock.verdict(_state(None), _state(random.uniform(100.0, 700.0)))
    assert found["suspect"] is False, "an echo not read made the pair suspect"


def test_the_jump_is_how_far_the_echo_moved():
    a, b = random.uniform(-300.0, 900.0), random.uniform(-300.0, 900.0)
    found = clock.verdict(_state(a), _state(b))
    assert found["jump_us"] == pytest.approx(abs(b - a)), \
        "the jump is not how far the echo moved"


def test_a_suspect_pair_says_the_echo_moved():
    a = random.uniform(100.0, 700.0)
    why = clock.verdict(_state(a), _state(a + random.uniform(400.0, 1200.0)))["why"]
    assert "moved" in why, "a suspect pair did not say its echo moved"


def test_the_window_is_the_one_the_station_judges_its_clock_by():
    ground = random.uniform(500.0, 4000.0)
    assert clock.window_us(ground) == pytest.approx(
        (-health.EPOCH_EARLY_LIMIT_S * 1e6,
         health.hop_excess_s(ground, health.ECHO_MAX_HOPS,
                             health.ECHO_MAX_HEIGHT_KM) * 1e6)), \
        "the simulator's clock window is not the station's"


def test_a_row_the_clock_broke_is_suspect():
    assert clock.suspect({"clock": {"suspect": True}}) is True, \
        "a row the clock broke was not suspect"


def test_a_row_from_before_the_clock_check_is_not_suspect():
    assert clock.suspect({"pair": {"muf_mhz": {}}}) is False, \
        "a row from before the clock check was suspect"
