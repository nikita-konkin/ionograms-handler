"""muf.sim.response: a mode's frequency response read subchannel by subchannel.

Every test sounds a short synthetic record, 10-11 MHz at 100 kHz/s and 2 kHz,
and reads it back, so the truth -- the curve, the o/x split, the level -- is
known exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from muf import cli
from muf.sim import presets, response, sounding
from muf.sim.fast import Record
from muf.sim.state import ChannelState, Mode, Sweep


def _record(*modes, seed=1, **synth) -> Record:
    """A synthetic record of the modes on the .lfs axis: beat -rate * delay."""
    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=10.0, sample_rate_hz=2e3)
    state = ChannelState(presets.cyprus1_circuit(), 1770206410.0, modes=list(modes),
                         sweep=sweep, source="test")
    return Record(samples=sounding.synthesize(state, seed=seed, **synth), sr_hz=2e3,
                  t_first_s=0.0, sweep_hz_at_0=10e6, rate_hz_s=100e3, km_per_hz=-3.0,
                  range_zero_km=0.0, shift_hz=0.0, km_per_us=0.3, source="test")


def _ripple_us(series) -> float:
    """The delay, us, at which the amplitude ripples the most along frequency."""
    amp = 10.0 ** (series.level_db[series.valid] / 20.0)
    step = float(np.median(np.diff(series.freq_mhz[series.valid])))
    spectrum = np.abs(np.fft.rfft(amp - amp.mean()))
    delay = np.fft.rfftfreq(amp.size, step)
    keep = delay > 5.0
    return float(delay[keep][np.argmax(spectrum[keep])])


def test_a_single_path_comes_back_on_its_curve():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8).sliding
    miss = np.abs(got.delay_us[got.valid] - mode.delay_at(got.freq_mhz[got.valid]))
    assert miss.max() < 5.0, "the peaks did not follow the curve"


def test_the_phase_gives_the_curve_dispersion():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8)
    assert got.gdd_phase_us_per_mhz == pytest.approx(51.0, rel=0.05), \
        "the phase's curvature did not give the curve's GDD"


def test_the_peaks_give_the_curve_dispersion_on_a_wide_subchannel():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           subchannel_hz=102.4e3, overlap=1.0)
    assert got.gdd_trace_us_per_mhz == pytest.approx(51.0, rel=0.05), \
        "the peaks' slope did not give the curve's GDD"


def test_sparse_windows_on_a_wide_subchannel_report_phase_jumps():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           subchannel_hz=102.4e3, overlap=0.0)
    assert got.stepped.jumps > 0, "windows 100 kHz apart unwrapped as if unambiguous"


def test_the_level_is_the_echo_against_the_noise():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2037.4], 40.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           subchannel_hz=20.48e3)
    expected = 10.0 * np.log10(1e4 * got.window / 2e3 * 2.0 / 3.0)
    assert got.snr_db == pytest.approx(expected, abs=1.0), \
        "the level did not read as C/N0 over the window against the noise"


def test_two_waves_in_one_bin_ripple_at_their_split():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2037.4], 60.0, ox_split_us=27.3,
                x_to_o_db=-3.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.1, 10.9,
                           subchannel_hz=10.24e3, overlap=1.0)
    assert _ripple_us(got.sliding) == pytest.approx(27.3, abs=2.0), \
        "o and x sharing a bin did not beat at their split"


def test_a_wide_subchannel_resolves_the_two_waves():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2037.4], 60.0, ox_split_us=27.3,
                x_to_o_db=-3.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.1, 10.9,
                           subchannel_hz=102.4e3, overlap=1.0).sliding
    assert np.std(got.level_db[got.valid]) < 0.5, "o and x did not resolve in 102.4 kHz"


def test_without_the_split_nothing_ripples():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2037.4], 60.0, ox_split_us=27.3,
                x_to_o_db=-3.0)
    got = response.measure(_record(mode, fading_on=False, magnetoionic=False), mode,
                           10.1, 10.9, subchannel_hz=10.24e3, overlap=1.0).sliding
    assert np.std(got.level_db[got.valid]) < 0.5, "a single wave rippled"


def test_a_clean_path_unwraps_without_a_jump():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           overlap=0.9)
    assert got.sliding.jumps == 0, "the sliding phase jumped on a clean path"


def test_the_stepped_phase_keeps_its_turns_on_a_clean_path():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           overlap=0.9)
    assert not got.stepped.lost.any(), "the stepped phase lost a turn on a clean path"


def test_the_automatic_gate_is_sixty_microseconds_at_least():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           subchannel_hz=102.4e3)
    assert got.gate_half_us == 60.0, "the automatic gate was not 60 us at 102.4 kHz"


def test_the_automatic_gate_spans_one_and_a_half_bins_on_a_narrow_subchannel():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           subchannel_hz=5.12e3)
    assert got.gate_half_us == pytest.approx(1.5 * got.bin_us), \
        "the automatic gate did not span 1.5 bins on a narrow subchannel"


def test_a_gate_set_by_hand_is_used():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           gate=response.Gate(half_us=23.7))
    assert got.gate_half_us == 23.7, "the gate's half-width set by hand was ignored"


def test_a_shifted_gate_finds_the_other_echo():
    one = Mode("1F", [9.9, 11.1], [2037.4, 2037.4], 60.0)
    two = Mode("2F", [9.9, 11.1], [2391.8, 2391.8], 60.0)
    got = response.measure(_record(one, two, fading_on=False), one, 10.2, 10.8,
                           gate=response.Gate(shift_us=354.4)).sliding
    assert np.median(got.delay_us[got.valid]) == pytest.approx(2391.8, abs=5.0), \
        "the shifted gate did not find the second echo"


def test_a_held_gate_keeps_every_peak_near_its_one_delay():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2757.4], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8,
                           gate=response.Gate(held=True, half_us=60.0))
    held = float(mode.delay_at(10.5))
    away = np.abs(got.sliding.delay_us[got.sliding.valid] - held)
    assert away.max() <= 60.0 + got.bin_us / 2, "a peak left the held gate"


def test_windows_outside_the_mode_band_give_no_sample():
    mode = Mode("1F", [10.4, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8).sliding
    assert not got.valid[got.freq_mhz < 10.39].any(), \
        "a window below the mode's band gave a sample"


def test_every_sample_starts_a_window_at_full_overlap():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.4, 10.6,
                           overlap=1.0)
    assert got.hop == 1, "full overlap did not start a window at every sample"


def test_half_overlap_steps_by_half_a_window():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    got = response.measure(_record(mode, fading_on=False), mode, 10.4, 10.6,
                           overlap=0.5)
    assert got.hop == got.window // 2, "half overlap did not step by half a window"


def test_a_stretch_outside_the_record_is_refused():
    mode = Mode("1F", [9.9, 13.1], [2037.4, 2098.6], 60.0)
    with pytest.raises(ValueError, match=r"no .* window"):
        response.measure(_record(mode, fading_on=False), mode, 12.0, 13.0)


def test_a_stretch_that_ends_before_it_starts_is_refused():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    with pytest.raises(ValueError, match="ends before"):
        response.measure(_record(mode, fading_on=False), mode, 10.8, 10.2)


def test_an_overlap_above_one_is_refused():
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0)
    with pytest.raises(ValueError, match="overlap"):
        response.measure(_record(mode, fading_on=False), mode, 10.2, 10.8, overlap=1.3)


def _sounding(folder) -> tuple:
    """A synthetic .lfs of one split, dispersive mode, and its state's file."""
    mode = Mode("1F", [9.9, 11.1], [2037.4, 2098.6], 60.0, ox_split_us=27.3,
                x_to_o_db=-3.0)
    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=10.0, sample_rate_hz=2e3)
    state = ChannelState(presets.cyprus1_circuit(), 1770206410.0, modes=[mode],
                         sweep=sweep, source="test")
    return (sounding.write_lfs(state, folder / "sim.lfs", seed=3, fading_on=False),
            state.save(folder / "state.json"))


def test_the_command_prints_the_dispersion(tmp_path, capsys):
    lfs, state = _sounding(tmp_path)
    cli.main(["sim", "response", str(lfs), "--state", str(state),
              "--band", "10.2,10.8"])
    assert "us/MHz from the phase" in capsys.readouterr().out, \
        "the command did not print the GDD"


def test_the_command_writes_both_series_as_csv(tmp_path):
    lfs, state = _sounding(tmp_path)
    out = tmp_path / "fr.csv"
    cli.main(["sim", "response", str(lfs), "--state", str(state), "--band", "10.2,10.8",
              "--csv", str(out)])
    kinds = {line.split(",")[0] for line in out.read_text().splitlines()[1:]}
    assert kinds == {"stepped", "sliding"}, "the CSV did not hold both series"


def test_the_command_writes_the_response_as_json(tmp_path):
    lfs, state = _sounding(tmp_path)
    out = tmp_path / "fr.json"
    cli.main(["sim", "response", str(lfs), "--state", str(state), "--band", "10.2,10.8",
              "--subchannel", "40.96", "--json", str(out)])
    assert json.loads(out.read_text())["window"] == 819, \
        "the JSON did not carry a 40.96 kHz window"


def test_the_command_takes_a_megahertz_about_the_mode_by_default(tmp_path):
    lfs, state = _sounding(tmp_path)
    out = tmp_path / "fr.json"
    cli.main(["sim", "response", str(lfs), "--state", str(state), "--json", str(out)])
    got = json.loads(out.read_text())
    assert (got["f_lo_mhz"], got["f_hi_mhz"]) == (10.0, 11.0), \
        "the default stretch was not the megahertz about the mode's middle"


def test_the_command_passes_the_gate_on(tmp_path):
    lfs, state = _sounding(tmp_path)
    out = tmp_path / "fr.json"
    cli.main(["sim", "response", str(lfs), "--state", str(state), "--band", "10.2,10.8",
              "--held", "--shift", "12.5", "--half", "33", "--json", str(out)])
    got = json.loads(out.read_text())["gate"]
    assert (got["held"], got["shift_us"], got["half_us"]) == (True, 12.5, 33.0), \
        "the command did not pass the gate on"


def test_the_command_refuses_an_unknown_mode(tmp_path):
    lfs, state = _sounding(tmp_path)
    assert cli.main(["sim", "response", str(lfs), "--state", str(state),
                     "--mode", "4F"]) == 2, "an unknown mode was not refused"
