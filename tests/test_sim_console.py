"""muf.sim.console: what the simulator page shows of one sounding."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from muf.sim import console, presets, sounding
from muf.sim.state import ChannelState, Sweep

SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)
T0 = 1770206410.0


def _soundings(folder: Path, count: int = 2) -> list[Path]:
    law = presets.law_mode(muf_mhz=13.6, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.5)
    out = []
    for k in range(count):
        state = ChannelState(presets.cyprus1_circuit(), T0 + 300.0 * k, [law],
                             sweep=SHORT, source="test")
        out.append(sounding.write_lfs(state, folder / sounding.default_name(state),
                                      seed=k))
    return out


def test_the_next_sounding_is_the_circuits_following_one(tmp_path):
    a, b, _ = _soundings(tmp_path, 3)
    assert console.next_of(a) == b


def test_the_last_sounding_has_none_following(tmp_path):
    *_, last = _soundings(tmp_path, 2)
    assert console.next_of(last) is None


def test_the_data_is_json_without_nan(tmp_path):
    a, _ = _soundings(tmp_path)
    json.dumps(console.build(a, realisations=2), allow_nan=False)


def test_the_page_gets_the_real_ionogram_its_copies_and_the_next(tmp_path):
    a, _ = _soundings(tmp_path)
    maps = console.build(a, realisations=2)["maps"]
    assert set(maps) == {"real", "synthetic1", "synthetic2", "synthetic3", "next"}


def test_every_mode_of_the_state_is_drawn(tmp_path):
    a, _ = _soundings(tmp_path)
    assert [m["name"] for m in console.build(a, realisations=2)["modes"]] == ["1F"]


def test_a_sounding_without_a_trace_fails_saying_so(tmp_path):
    state = ChannelState(presets.cyprus1_circuit(), T0, [], sweep=SHORT, source="test")
    path = sounding.write_lfs(state, tmp_path, seed=random.randrange(2**31))
    with pytest.raises(ValueError, match="no trace to simulate"):
        console.build(path, realisations=2)


def test_the_copies_shown_leave_no_file_behind(tmp_path):
    a, b = _soundings(tmp_path)
    console.build(a, realisations=2)
    assert sorted(tmp_path.iterdir()) == sorted([a, b]), "a copy shown was left on disk"


def test_a_build_reports_each_step_it_takes(tmp_path):
    a, _ = _soundings(tmp_path)
    calls = []
    console.build(a, realisations=2,
                  progress=lambda done, total: calls.append((done, total)))
    total = 2 + console.SHOWN + 1
    assert calls == [(k, total) for k in range(total + 1)], \
        f"the build did not report each step: {calls}"
