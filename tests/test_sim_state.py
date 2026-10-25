"""muf.sim.state: the channel state, its curves and its file."""

from __future__ import annotations

import numpy as np
import pytest

from muf import io_lfs
from muf.sim import presets, sounding
from muf.sim.state import (
    MEASURED,
    PRIOR,
    ChannelState,
    Circuit,
    Mode,
    Provenance,
    Sweep,
)


def _quadratic_mode() -> Mode:
    f = np.arange(10.0, 14.0001, 0.05)
    return Mode("1F", f, 9000.0 + 20.0 * (f - 12.0) + 4.0 * (f - 12.0) ** 2, 40.0,
                doppler_spread_hz=0.5,
                provenance={"delay": Provenance(MEASURED, "a test"),
                            "doppler_spread": "prior: ITU-R F.1487"})


def test_taylor_reads_a_known_curve():
    m = _quadratic_mode()
    tau, gdd, tod = m.taylor(12.53)
    assert tau == pytest.approx(9000.0 + 20.0 * 0.53 + 4.0 * 0.53 ** 2, abs=0.01)
    assert gdd == pytest.approx(20.0 + 8.0 * 0.53, rel=1e-3)
    assert tod == pytest.approx(8.0, rel=1e-3)


def test_tod_holds_at_the_band_edge():
    m = _quadratic_mode()
    assert float(m.tod_at(m.f_hi)) == pytest.approx(8.0, rel=0.02)


def test_a_mode_does_not_exist_outside_its_band():
    m = _quadratic_mode()
    assert np.isnan(m.delay_at(9.9)) and np.isnan(m.cn0_at(14.1))
    assert m.covers(10.0) and m.covers(14.0) and not m.covers(14.01)


def test_the_law_mode_follows_the_law():
    m = presets.law_mode(muf_mhz=25.0)
    f = np.array([16.0, 20.0, 24.0])
    np.testing.assert_allclose(m.gdd_at(f), presets.law_gdd(f, 25.0), rtol=2e-3)


def test_a_state_survives_its_file(tmp_path):
    state = ChannelState(
        circuit=Circuit("cyprus1", "yoshkar-ola", 35.18557, 33.38228, 56.38, 47.53),
        epoch=1770206410.0, modes=[_quadratic_mode()], sweep=presets.CYPRUS1_SWEEP,
        source="a test", notes=["kept"], metrics={"muf_mhz": 14.0},
        provenance={"epoch": Provenance(MEASURED, "header")})
    back = ChannelState.load(state.save(tmp_path / "state.json"))
    m, b = state.modes[0], back.modes[0]
    np.testing.assert_array_equal(b.freq_mhz, m.freq_mhz)
    np.testing.assert_array_equal(b.delay_us, m.delay_us)
    np.testing.assert_array_equal(b.cn0_dbhz, m.cn0_dbhz)
    assert b.provenance == m.provenance
    assert b.origin("doppler_spread") == PRIOR
    assert back.sweep == state.sweep and back.circuit == state.circuit
    assert back.metrics == state.metrics and back.notes == state.notes


def test_a_file_of_another_schema_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"schema": "something else", "version": 1}')
    with pytest.raises(ValueError, match="not a channel state"):
        ChannelState.load(path)


@pytest.mark.parametrize("bad", [
    {"freq_mhz": [10.0, 10.0, 11.0], "delay_us": [1.0, 2.0, 3.0]},
    {"freq_mhz": [10.0], "delay_us": [1.0]},
    {"freq_mhz": [10.0, 11.0], "delay_us": [1.0, np.nan]},
    {"freq_mhz": [10.0, 11.0], "delay_us": [1.0, 2.0], "doppler_spread_hz": -1.0},
])
def test_a_malformed_mode_is_refused(bad):
    with pytest.raises(ValueError):
        Mode("x", cn0_dbhz=40.0, **bad)


def test_an_unknown_origin_is_refused():
    with pytest.raises(ValueError):
        Provenance("guessed")


def test_mode_names_are_unique():
    m = _quadratic_mode()
    with pytest.raises(ValueError, match="repeat"):
        ChannelState(Circuit("a", "b"), 0.0, modes=[m, m])


def test_at_lists_the_modes_present_earliest_first():
    state = presets.f1487("mid-disturbed")
    names = [item.mode.name for item in state.at(10.0)]
    assert names == ["path1", "path2"]
    first, second = state.at(10.0)
    assert second.delay_us - first.delay_us == pytest.approx(2000.0)
    assert state.at(40.0) == []


def test_f1487_table_is_the_recommendations():
    # Annex 3, sections 2-4, as printed.
    assert presets.F1487["mid-quiet"] == (0.5, 0.1)
    assert presets.F1487["mid-moderate"] == (1.0, 0.5)
    assert presets.F1487["mid-disturbed"] == (2.0, 1.0)
    assert presets.F1487["high-disturbed"] == (7.0, 30.0)
    assert len(presets.F1487) == 10


def test_the_lfs_header_reads_back():
    state = presets.law_state()
    fields = sounding.header_fields(state)
    header = io_lfs.parse_header(io_lfs.pack_header(fields), "synthetic.lfs")
    for name, value in fields.items():
        got = getattr(header, name)
        if isinstance(value, float):
            assert got == pytest.approx(value, abs=1e-5), name
        else:
            assert got == (value.strip() if isinstance(value, str) else value), name


def test_a_header_needs_every_field():
    fields = sounding.header_fields(presets.law_state())
    del fields["rate"]
    with pytest.raises(ValueError, match="rate"):
        io_lfs.pack_header(fields)


def test_a_sweep_must_decimate_by_an_integer():
    state = presets.law_state(sweep=Sweep(7.5, 1e5, 10.0, 41e3))
    with pytest.raises(ValueError, match="integer"):
        sounding.header_fields(state)


@pytest.mark.parametrize("sigma", [2.0, 20.0, 50.0])
def test_delay_taps_are_the_gaussian(sigma):
    m = Mode("s", [10.0, 12.0], [9000.0, 9000.0], 40.0, delay_spread_us=sigma)
    taps = m.delay_taps(10.0)
    offsets = np.array([d for d, _w, _k in taps])
    shares = np.array([w for _d, w, _k in taps])
    assert shares.sum() == pytest.approx(1.0)
    assert np.sum(shares * offsets) == pytest.approx(0.0, abs=1e-9)
    assert np.sqrt(np.sum(shares * offsets ** 2)) == pytest.approx(sigma, rel=0.005)
    assert np.max(np.diff(offsets)) <= min(10.0, sigma / 2) + 1e-9
    assert all(k is None for _d, _w, k in taps)


def test_a_specular_part_is_not_spread():
    m = Mode("s", [10.0, 12.0], [9000.0, 9000.0], 40.0, delay_spread_us=20.0,
             rician_k_db=6.0)
    taps = m.delay_taps(10.0)
    assert sum(w for _d, w, _k in taps) == pytest.approx(1.0)
    k = 10 ** 0.6
    (_d, w0, k0), = [t for t in taps if t[0] == 0.0]
    scattered0 = (1 - k / (k + 1)) * (w0 * (k + 1) - k)    # its scattered share
    assert 10 ** (k0 / 10) * scattered0 == pytest.approx(k / (k + 1), rel=1e-9)
    assert all(t[2] is None for t in taps if t[0] != 0.0)
    assert Mode("p", [10.0, 12.0], [9000.0, 9000.0], 40.0).delay_taps(10.0) == [
        (0.0, 1.0, None)]


def test_delay_spread_survives_the_file_and_defaults_to_none(tmp_path):
    m = _quadratic_mode()
    m.delay_spread_us = 14.5
    back = Mode.from_dict(m.to_dict())
    assert back.delay_spread_us == 14.5
    old = m.to_dict()
    del old["delay_spread_us"]                  # a state written before it existed
    assert Mode.from_dict(old).delay_spread_us == 0.0
    with pytest.raises(ValueError, match="delay spread"):
        Mode("x", [10.0, 12.0], [9000.0, 9000.0], 40.0, delay_spread_us=-1.0)
