"""1F alone over 1 MHz at 0.75 of its MUF, read from the complex signal (#92)."""

from __future__ import annotations

import pytest

from muf import cli
from muf.sim import band, fast, presets, sounding
from muf.sim.state import ChannelState, Mode, Sweep


def sweep(start_mhz: float = 18.0, seconds: float = 15.0) -> Sweep:
    return Sweep(start_mhz=start_mhz, rate_hz_s=100e3, duration_s=seconds,
                 sample_rate_hz=40e3)


def state_of(*modes: Mode, metrics: dict | None = None,
             swept: Sweep | None = None) -> ChannelState:
    return ChannelState(circuit=presets.cyprus1_circuit(), epoch=1770206410.0,
                        modes=list(modes), sweep=swept or sweep(),
                        metrics=dict(metrics or {}))


def first_hop(**kw) -> Mode:
    return presets.law_mode("1F", **{"muf_mhz": 25.0, "spread_hz": 0.0, **kw})


def reading_of(state: ChannelState, folder, *, offset_us: float = 13.0,
               seed: int = 7) -> dict:
    path = sounding.write_lfs(state, folder / "sounding.lfs", seed=seed)
    m = state.modes[0]
    trace = Mode(name=m.name, freq_mhz=m.freq_mhz, delay_us=m.delay_us + offset_us,
                 cn0_dbhz=m.cn0_dbhz, hops=1)
    found = band.band_of(state)
    return band.reading(band.isolate(fast.record_of(path), trace, found,
                                     state.modes[1:]), found)


def test_the_band_is_centred_at_three_quarters_of_the_muf_of_1f():
    mode = first_hop(muf_mhz=23.7)
    assert band.band_of(state_of(mode)).centre_mhz == pytest.approx(0.75 * mode.f_hi), \
        "the band is not centred at 0.75 of 1F's MUF"


def test_the_band_is_one_megahertz_wide():
    found = band.band_of(state_of(first_hop(muf_mhz=27.3)))
    assert found.hi_mhz - found.lo_mhz == pytest.approx(1.0), \
        "the band is not 1 MHz wide"


def test_a_state_without_a_low_ray_has_no_band():
    high = presets.law_mode("1F-high", muf_mhz=26.1, f_lo_mhz=25.2, hops=1)
    high.branch = "high"
    with pytest.raises(ValueError, match="no low ray"):
        band.band_of(state_of(high))


def test_a_censored_muf_has_no_band():
    with pytest.raises(ValueError, match="censored"):
        band.band_of(state_of(first_hop(muf_mhz=26.2),
                              metrics={"band_edge_mhz": 25.9}))


def test_a_1f_that_starts_inside_the_band_has_no_band():
    with pytest.raises(ValueError, match="covers"):
        band.band_of(state_of(first_hop(muf_mhz=24.4, f_lo_mhz=18.5)))


def test_the_high_ray_does_not_set_the_muf():
    low = first_hop(muf_mhz=24.6)
    high = presets.law_mode("1F-high", muf_mhz=26.1, f_lo_mhz=25.2, hops=1)
    high.branch = "high"
    assert band.band_of(state_of(low, high)).muf_mhz == pytest.approx(low.f_hi), \
        "the high ray set 1F's MUF"


def test_the_band_belongs_to_the_earliest_arrival_whatever_its_label():
    late = presets.law_mode("1F", muf_mhz=24.3, tau0_us=9420.0)
    early = presets.law_mode("3F", muf_mhz=25.8, tau0_us=8870.0, hops=3)
    assert band.band_of(state_of(late, early)).mode == "3F", \
        "the band did not go to the earliest arrival"


def test_an_early_fragment_that_cannot_hold_its_band_is_passed_over():
    whole = presets.law_mode("F", muf_mhz=29.4, tau0_us=9440.0, hops=None)
    piece = presets.law_mode("F#2", muf_mhz=29.5, tau0_us=9360.0, f_lo_mhz=26.8,
                             hops=None)
    assert band.band_of(state_of(whole, piece)).mode == "F", \
        "the band went to an early fragment that does not cover it"


def test_a_steady_1f_reads_back_its_level(tmp_path):
    assert reading_of(state_of(first_hop(cn0_dbhz=43.0)), tmp_path)["cn0_dbhz"] \
        == pytest.approx(43.0, abs=0.3), "a steady 1F did not read back its level"


def test_a_steady_1f_reads_back_its_delay(tmp_path):
    mode = first_hop()
    found = band.band_of(state_of(mode))
    assert reading_of(state_of(mode), tmp_path)["delay_us"] \
        == pytest.approx(float(mode.delay_at(found.centre_mhz)), abs=0.05), \
        "a steady 1F did not read back its delay through a trace 13 us off"


def test_a_steady_1f_reads_back_its_gdd(tmp_path):
    mode = first_hop()
    found = band.band_of(state_of(mode))
    assert reading_of(state_of(mode), tmp_path)["gdd_us_per_mhz"] \
        == pytest.approx(float(mode.gdd_at(found.centre_mhz)), abs=0.1), \
        "a steady 1F did not read back its GDD"


def test_a_steady_1f_reads_back_its_tod(tmp_path):
    mode = first_hop()
    found = band.band_of(state_of(mode))
    assert reading_of(state_of(mode), tmp_path)["tod_us_per_mhz2"] \
        == pytest.approx(float(mode.tod_at(found.centre_mhz)), abs=1.0), \
        "a steady 1F did not read back its TOD"


def test_another_mode_leaves_the_reading_of_1f_alone(tmp_path):
    alone = reading_of(state_of(first_hop(spread_hz=0.3)), tmp_path)["cn0_dbhz"]
    other = presets.law_mode("2F", muf_mhz=21.0, tau0_us=9900.0, cn0_dbhz=51.0,
                             spread_hz=0.3, hops=2)
    assert reading_of(state_of(first_hop(spread_hz=0.3), other), tmp_path)["cn0_dbhz"] \
        == pytest.approx(alone, abs=0.1), "another mode leaked into 1F's signal"


def test_a_second_echo_too_close_to_1f_gives_no_signal(tmp_path):
    mode = first_hop()
    ghost = Mode(name="2F", freq_mhz=mode.freq_mhz, delay_us=mode.delay_us + 200.0,
                 cn0_dbhz=mode.cn0_dbhz, hops=2)
    with pytest.raises(ValueError, match="second echo"):
        reading_of(state_of(mode, ghost), tmp_path, offset_us=0.0)


def test_a_record_short_of_the_band_gives_no_signal(tmp_path):
    with pytest.raises(ValueError, match="does not cover"):
        reading_of(state_of(first_hop(), swept=sweep(18.4, 9.0)), tmp_path)


def measured(**kw) -> dict:
    return {"cn0_dbhz": 41.7, "delay_us": 8861.3, "gdd_us_per_mhz": 8.3,
            "tod_us_per_mhz2": -3.1, "spread_hz": 0.37, "shape": "lorentzian",
            "bound": "", "rice_k_db": None, **kw}


def test_the_band_state_holds_1f_alone():
    state = state_of(first_hop(), presets.law_mode("2F", muf_mhz=19.6, tau0_us=9870.0,
                                                   hops=2))
    assert [m.name for m in band.narrow(state, band.band_of(state), measured()).modes] \
        == ["1F"], "the band state holds more than 1F"


def test_the_band_state_takes_its_delay_from_the_reading():
    state = state_of(first_hop())
    found = band.band_of(state)
    assert float(band.narrow(state, found, measured()).modes[0]
                 .delay_at(found.centre_mhz)) == pytest.approx(8861.3), \
        "the band state did not take its delay from the reading"


def test_the_band_state_takes_its_gdd_from_the_reading():
    state = state_of(first_hop())
    found = band.band_of(state)
    assert float(band.narrow(state, found, measured()).modes[0]
                 .gdd_at(found.centre_mhz)) == pytest.approx(8.3, abs=0.01), \
        "the band state did not take its GDD from the reading"


def test_the_band_state_takes_its_level_from_the_reading():
    state = state_of(first_hop())
    found = band.band_of(state)
    assert float(band.narrow(state, found, measured()).modes[0]
                 .cn0_at(found.hi_mhz)) == pytest.approx(41.7), \
        "the band state did not take its level from the reading"


def test_the_band_state_takes_its_spread_from_the_reading():
    state = state_of(first_hop(spread_hz=0.9))
    assert band.narrow(state, band.band_of(state), measured()).modes[0] \
        .doppler_spread_hz == pytest.approx(0.37), \
        "the band state did not take its Doppler spread from the reading"


def test_the_band_state_folds_its_delay_spread_into_the_fading_it_read():
    state = state_of(first_hop(spread_hz=0.6))
    state.modes[0].delay_spread_us = 4.3
    assert band.narrow(state, band.band_of(state), measured()).modes[0] \
        .delay_spread_us == 0.0, "the band state counts its delay spread twice"


def test_a_spread_the_band_only_bounds_leaves_the_prior():
    state = state_of(first_hop(spread_hz=0.9))
    assert band.narrow(state, band.band_of(state), measured(bound="upper")).modes[0] \
        .doppler_spread_hz == pytest.approx(0.9), \
        "a bounded spread replaced the prior"


def test_the_band_state_reaches_past_the_band_by_its_margin():
    state = state_of(first_hop())
    found = band.band_of(state)
    assert band.narrow(state, found, measured()).modes[0].f_lo \
        == pytest.approx(found.lo_mhz - band.MARGIN_MHZ), \
        "the band state does not reach past the band by its margin"


def test_the_band_state_sweeps_the_band_and_its_margins():
    state = state_of(first_hop(), swept=presets.CYPRUS1_SWEEP)
    found = band.band_of(state)
    cut = band.narrow(state, found, measured()).sweep
    assert (cut.start_mhz < found.lo_mhz - band.MARGIN_MHZ
            and cut.stop_mhz > found.hi_mhz + band.MARGIN_MHZ
            and cut.duration_s < 30.0), "the band state's sweep is not cut to the band"


def test_cli_band_reports_the_band_of_a_synthetic_sounding(tmp_path, capsys):
    path = sounding.write_lfs(presets.law_state(spread_hz=0.2), tmp_path / "law.lfs",
                              seed=3)
    cli.main(["sim", "band", str(path), "--copies", "2"])
    assert "1F over" in capsys.readouterr().out, "muf sim band did not report the band"
