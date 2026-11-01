"""muf.sim.sounding and muf.sim.observe: a state sounded, and read back by muf."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import random
import warnings

import numpy as np
import pytest

from muf import cli, io_lfs, loader, spectro, trace
from muf.sim import check, observe, presets, sounding
from muf.sim.state import ChannelState, Mode, Sweep

#: A short sweep, 10-14 MHz in 40 s, so a sounding is 1.6 M samples.
SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)
KM_PER_US = 0.3          # the .lfs axis: c = 3e5 km/s, oblique


def _state(*modes, sweep=SHORT) -> ChannelState:
    return ChannelState(presets.cyprus1_circuit(), 1770206410.0, modes=list(modes),
                        sweep=sweep, source="test")


def _law(**kw) -> Mode:
    opts = {"muf_mhz": 14.0, "f_lo_mhz": 10.5, "cn0_dbhz": 45.0, "spread_hz": 0.0}
    opts.update(kw)
    return presets.law_mode(**opts)


def _flat(delay=9000.0, cn0=45.0, band=(10.5, 13.5), **kw) -> Mode:
    return Mode("flat", list(band), [delay, delay], cn0, **kw)


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return loader.load(path)


def test_the_file_is_an_lfs_of_the_sweep(tmp_path):
    state = _state(_flat())
    path = sounding.write_lfs(state, tmp_path, seed=1)
    header = io_lfs.read_header(path)
    assert header.tx_name == "cyprus1" and header.rx_name == "yoshkar-ola"
    assert header.start_epoch == 1770206410
    assert header.dur == 40 and header.rate == 100000
    assert header.sample_rate / header.dec == 40e3
    assert io_lfs.n_samples(path) == 40 * 40000
    assert path.name == "sim_cyprus1_20260204_120010.lfs"


def test_an_echo_lands_at_its_range(tmp_path):
    delay = 8950.0
    path = sounding.write_lfs(_state(_flat(delay=delay)), tmp_path, seed=2)
    ion = _load(path)
    rows = (ion.freq > 10.7) & (ion.freq < 13.3)
    peaks = ion.vrange[np.argmax(ion.power[rows], axis=1)]
    bin_km = abs(float(np.median(np.diff(ion.vrange))))
    assert np.all(np.abs(peaks - delay * KM_PER_US) <= bin_km / 2 + 1e-6)


def test_the_level_reads_as_its_cn0(tmp_path):
    """muf's scale and the state's C/N0 are one: the bridge's constant."""
    cn0 = 42.0
    path = sounding.write_lfs(_state(_flat(cn0=cn0)), tmp_path, seed=3)
    ion = _load(path)
    _, lin = observe._cn0_along(ion, np.array([10.0, 14.0]), np.array([2700.0, 2700.0]),
                                10.7, 13.3)
    assert 10 * np.log10(np.mean(lin)) == pytest.approx(cn0, abs=0.2)


def test_the_extractors_find_the_states_muf(tmp_path):
    state = _state(_law())
    ion = _load(sounding.write_lfs(state, tmp_path, seed=4))
    from muf import extractors

    for name, result in extractors.run(ion, methods=("contour", "algo", "dp")).items():
        assert result.muf_mhz == pytest.approx(state.muf_mhz, abs=0.05), name


def test_a_state_is_recovered_from_its_sounding(tmp_path):
    truth = _state(_law())
    got = observe.from_sounding(sounding.write_lfs(truth, tmp_path, seed=5))
    (err,) = check.match(truth, got)
    assert err.matched is not None
    assert err.delay_rms_us < 10.0                 # a range bin is 49 us
    assert abs(err.f_hi_error_mhz) < 0.06 and abs(err.f_lo_error_mhz) < 0.06
    assert abs(err.cn0_bias_db) < 1.0
    m = got.mode(err.matched)
    assert m.origin("delay") == "measured" and m.origin("doppler_spread") == "prior"


def test_two_modes_are_told_apart(tmp_path):
    truth = _state(_law(name="1F", tau0_us=8900.0),
                   _law(name="2F", tau0_us=9500.0, muf_mhz=12.5, cn0_dbhz=40.0, hops=2))
    got = observe.from_sounding(sounding.write_lfs(truth, tmp_path, seed=6))
    errors = check.match(truth, got)
    assert all(e.matched for e in errors)
    assert all(e.delay_rms_us < 15.0 for e in errors)


def test_fading_varies_the_level_along_the_trace(tmp_path):
    still = _load(sounding.write_lfs(_state(_flat()), tmp_path / "a.lfs", seed=7))
    fades = _load(sounding.write_lfs(_state(_flat(doppler_spread_hz=1.0)),
                                     tmp_path / "b.lfs", seed=7))

    def spread(ion):
        _, lin = observe._cn0_along(ion, np.array([10.0, 14.0]),
                                    np.array([2700.0, 2700.0]), 10.7, 13.3)
        return np.std(10 * np.log10(lin))
    assert spread(fades) > 3.0 > 1.0 > spread(still)


def test_a_mode_sharing_its_echo_with_another_takes_half_its_cn0(tmp_path):
    cn0 = random.uniform(38.0, 46.0)
    ion = _load(sounding.write_lfs(_state(_flat(cn0=cn0)), tmp_path,
                                   seed=random.randrange(1000)))
    curve = (np.array([10.0, 14.0]), np.full(2, 9000.0 * KM_PER_US))
    _, lin = observe._cn0_along(ion, *curve, 10.7, 13.3, others=[(*curve, 10.7, 13.3)])
    assert 10 * np.log10(np.mean(lin)) == pytest.approx(cn0 - 10 * np.log10(2),
                                                        abs=0.2), \
        "a mode sharing its echo with another did not take half its C/N0"


def test_a_spread_echo_carved_into_modes_reads_its_cn0_once(tmp_path):
    cn0 = random.uniform(40.0, 46.0)
    truth = _state(_flat(cn0=cn0, doppler_spread_hz=0.5,
                         delay_spread_us=random.uniform(30.0, 80.0)))
    got = observe.from_sounding(sounding.write_lfs(truth, tmp_path,
                                                   seed=random.randrange(1000)))
    f = np.linspace(11.0, 13.0, 21)
    total = sum(10 ** (np.where((f >= m.f_lo) & (f <= m.f_hi), m.cn0_at(f), -99.0) / 10)
                for m in got.modes)
    assert 10 * np.log10(np.mean(total)) == pytest.approx(cn0, abs=1.5), \
        "the modes of one spread echo did not read its C/N0 once between them"


def test_a_sounding_needs_a_sweep(tmp_path):
    state = _state(_flat(), sweep=None)
    with pytest.raises(ValueError, match="sweep"):
        sounding.synthesize(state)


def test_the_same_seed_sounds_the_same(tmp_path):
    state = _state(_flat(doppler_spread_hz=0.5))
    a = sounding.synthesize(state, seed=9, duration_s=2.0)
    b = sounding.synthesize(state, seed=9, duration_s=2.0)
    np.testing.assert_array_equal(a, b)


def test_closed_loop_reports_every_mode(tmp_path):
    truth = _state(_law(spread_hz=0.5))
    report = check.closed_loop(truth, realisations=2, seed=10, workdir=tmp_path)
    assert report["modes"]["1F"]["found"] == 2
    assert report["muf_mhz"]["n"] == 2
    corr = report["correlation"]
    assert corr["soundings"] == 2
    assert corr["rows"]["synthetic"]["n"] == 1      # one pair, no real
    assert "real" not in corr["rows"]
    assert "fewer than 12" in report["note"]        # two draw no verdict
    assert set(corr["modes"]) == {"1F"}
    # Averaging along frequency takes the fading off, and leaves the state.
    assert corr["smoothed"]["synthetic"]["mean"] > corr["rows"]["synthetic"]["mean"]
    assert not list(tmp_path.glob("*.lfs"))         # not kept unless asked
    json.dumps(report)                              # the CLI writes it as is


def test_closed_loop_reads_shared_copies_as_its_own(tmp_path):
    truth = _state(_law(spread_hz=0.5))
    shared = [_load(sounding.write_lfs(truth, tmp_path / f"c{k}.lfs", seed=k))
              for k in range(2)]
    own = check.closed_loop(truth, realisations=2, workdir=tmp_path / "own")
    reused = check.closed_loop(truth, realisations=2, workdir=tmp_path / "reused",
                               synthetic=shared)
    assert reused["modes"] == own["modes"]


def test_closed_loop_sounds_the_copies_it_was_not_given(tmp_path):
    truth = _state(_law(spread_hz=0.5))
    shared = [_load(sounding.write_lfs(truth, tmp_path / "c0.lfs", seed=0))]
    own = check.closed_loop(truth, realisations=2, workdir=tmp_path / "own")
    reused = check.closed_loop(truth, realisations=2, workdir=tmp_path / "reused",
                               synthetic=shared)
    assert reused["modes"] == own["modes"]


def test_closed_loop_refuses_the_fast_part_from_shared_copies(tmp_path):
    truth = _state(_law(spread_hz=0.5))
    shared = [_load(sounding.write_lfs(truth, tmp_path / "c0.lfs", seed=0))]
    with pytest.raises(ValueError, match="fast part"):
        check.closed_loop(truth, realisations=1, fast=True, synthetic=shared)


def test_check_sounding_says_a_sounding_without_a_trace_has_none_to_simulate(tmp_path):
    path = sounding.write_lfs(_state(), tmp_path, seed=random.randrange(2**31))
    with pytest.raises(ValueError, match="no trace to simulate"):
        check.check_sounding(path, realisations=2, workdir=tmp_path / "loop")


def test_check_sounding_runs_no_loop_on_a_sounding_without_a_trace(tmp_path):
    path = sounding.write_lfs(_state(), tmp_path, seed=random.randrange(2**31))
    with contextlib.suppress(ValueError):
        check.check_sounding(path, realisations=2, workdir=tmp_path / "loop")
    assert not (tmp_path / "loop").exists(), "the loop ran on a traceless sounding"


def test_trace_mask_is_around_the_trace(tmp_path):
    state = _state(_flat(delay=9000.0, band=(11.0, 13.0)))
    ion = _load(sounding.write_lfs(state, tmp_path, seed=1))
    mask = check.trace_mask(state.modes, ion, half_bins=3)
    rows = np.flatnonzero(mask.any(axis=1))
    assert ion.freq[rows[0]] >= 11.0 and ion.freq[rows[-1]] <= 13.0
    assert np.all(mask[rows].sum(axis=1) == 7)
    km = ion.vrange[mask[rows[0]]]
    assert km.min() < 9000.0 * KM_PER_US < km.max()


def test_an_echo_window_is_read_on_its_own_axis(tmp_path):
    from muf import echo_window

    lfs = sounding.write_lfs(_state(_flat(delay=9000.0)), tmp_path / "real.lfs",
                             seed=2)
    km = 9000.0 * KM_PER_US
    target = echo_window.write(lfs, out_dir=tmp_path,
                               manual_km=(km - 300.0, km + 300.0)).target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = loader.load(lfs)
        win = observe.load(target)
        again = observe.load(lfs, gate_km=check.gate_of(win))
        same = observe.load(lfs, gate_km=check.gate_of(whole))
    # The window is read from its samples, not from the product's own power,
    # and a capture loaded at its gate lands on its axis bin for bin.
    assert win.vrange.size < whole.vrange.size
    assert check.same_grid(win, again)
    rows = (win.freq > 10.7) & (win.freq < 13.3)

    def peak_km(ion):
        return ion.vrange[np.argmax(np.asarray(ion.power)[rows].mean(axis=0))]

    assert peak_km(win) == peak_km(again)
    # At the capture's own range the gate is the default one.
    assert check.same_grid(whole, same)
    assert np.array_equal(whole.power, same.power)


def _sounded(state, tmp_path, seed):
    path = sounding.write_lfs(state, tmp_path / f"real{seed}.lfs", seed=seed)
    ion = _load(path)
    path.unlink()
    return ion


def test_ionogram_correlation_places_the_real_sounding(tmp_path):
    # A "real" sounding that is another realisation of the state is one of
    # the synthetic ones; one whose trace is three range bins away falls far
    # under them, row by row and smoothed alike.
    truth = _state(_law(spread_hz=0.5))
    same = check.closed_loop(truth, seed=20, workdir=tmp_path,
                             real=_sounded(truth, tmp_path, seed=99))
    moved = copy.deepcopy(truth)
    m = moved.modes[0]
    moved.modes = [dataclasses.replace(m, delay_us=np.asarray(m.delay_us) + 150.0)]
    off = check.closed_loop(truth, seed=20, workdir=tmp_path,
                            real=_sounded(moved, tmp_path, seed=99))
    for key in check.CORRELATION_MAPS:
        st = same["correlation"][key]["synthetic"]
        assert same["correlation"][key]["real_inside"]
        assert off["correlation"][key]["real"]["mean"] < st["min"] - 0.1
        assert not off["correlation"][key]["real_inside"]
    assert len(same["correlation"]["per_run"]) == check.REALISM_SOUNDINGS
    json.dumps(same)


def test_verdict_needs_its_soundings_and_reads_the_grid():
    row = 0.0205
    same = [12.221] * check.REALISM_SOUNDINGS
    assert check._verdict(12.221, same[:-1], resolution=row) == {}   # too few
    assert check._verdict(12.221, same) == {}                        # no scale
    # Identical synthetic values still have the grid's rounding as spread.
    v = check._verdict(12.221, same, resolution=row)
    assert v["real_inside"] and v["real_z"] == 0.0
    assert not check._verdict(12.221 + 2 * row, same, resolution=row)["real_inside"]
    # One-sided: above the copies is no defect for a correlation.
    r = list(0.9 + 0.01 * np.random.default_rng(0).standard_normal(12))
    assert check._verdict(0.99, r, below_only=True)["real_inside"]
    assert not check._verdict(0.99, r)["real_inside"]
    assert not check._verdict(0.80, r, below_only=True)["real_inside"]


def test_muf_verdict_on_a_copy_and_off_it(tmp_path):
    truth = _state(_law(spread_hz=0.5))
    ion = _sounded(truth, tmp_path, seed=77)
    muf = observe.from_ionogram(ion).metrics["muf_mhz"]
    report = check.closed_loop(truth, seed=30, workdir=tmp_path, real_muf_mhz=muf)
    assert report["real_muf_inside"]
    row = float(np.median(np.diff(ion.freq)))
    mufs = [r["muf_mhz"] for r in report["runs"]]
    assert not check._verdict(muf - 5 * row, mufs, resolution=row)["real_inside"]


def test_cli_preset_sounding_state_and_channel(tmp_path, capsys):
    law = tmp_path / "law.json"
    assert cli.main(["sim", "preset", "law", "--muf", "25", "--out", str(law)]) == 0
    state = ChannelState.load(law)
    state.sweep = SHORT
    state.modes = [_law()]
    state.save(law)

    lfs = tmp_path / "x.lfs"
    assert cli.main(["sim", "sounding", str(law), "--out", str(lfs),
                     "--seed", "3"]) == 0
    assert io_lfs.n_samples(lfs) == 40 * 40000

    back = tmp_path / "back.json"
    assert cli.main(["sim", "state", str(lfs), "--out", str(back), "--fc", "13"]) == 0
    assert ChannelState.load(back).modes

    out = tmp_path / "y.npy"
    assert cli.main(["sim", "channel", str(back), "--fc", "13", "--fs", "48000",
                     "--test", "chirp", "--duration", "0.1", "--snr-db", "20",
                     "--out", str(out)]) == 0
    y = np.load(out)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert y.dtype == np.complex64 and y.size > 4800 and meta["paths"]
    assert cli.main(["sim", "show", str(back), "--provenance"]) == 0
    assert "measured" in capsys.readouterr().out


def test_cli_preset_list(capsys):
    assert cli.main(["sim", "preset", "--list"]) == 0
    assert "mid-moderate" in capsys.readouterr().out


def test_tone_constant_matches_the_spectrogram():
    """TONE_PER_CN0_SECOND straight from spectro's own normalisation."""
    fs, w = 40e3, spectro.DEFAULT_WINDOW
    cn0 = 1e4
    a = math.sqrt(cn0 / fs)
    t = np.arange(w * 64) / fs
    rng = np.random.default_rng(0)
    beat = 25 * fs / w                                   # on a bin
    noise = (rng.standard_normal(t.size)
             + 1j * rng.standard_normal(t.size)) / math.sqrt(2)
    x = a * np.exp(2j * np.pi * beat * t) + noise
    win = spectro.hanning(w)
    peaks = []
    for i in range(64):
        s = np.abs(np.fft.fft(win * x[i * w:(i + 1) * w])) ** 2
        peaks.append(s[25] / (spectro.NOISE_COEF * np.median(s)))
    expect = cn0 * (w / fs) * observe.TONE_PER_CN0_SECOND
    assert np.mean(peaks) == pytest.approx(expect, rel=0.02)


@pytest.mark.parametrize("sigma", [0.0, 30.0])
def test_a_delay_spread_widens_the_trace_by_itself(tmp_path, sigma):
    # Read back as the trace's width against a point echo; independent paths
    # leave the centres uncorrelated from one row to the next.
    m = _flat(doppler_spread_hz=0.5, delay_spread_us=sigma)
    ion = _load(sounding.write_lfs(_state(m), tmp_path, seed=5))
    width = observe.cross_trace(ion, m)
    assert width["rows"] >= 100
    if sigma:
        assert width["spread_us"] == pytest.approx(sigma, rel=0.15)
        assert abs(width["centre_r1"]) < 0.3
    else:
        assert width["spread_us"] < 8.0
    back = observe.from_ionogram(ion)
    if sigma:
        assert back.modes[0].origin("delay_spread") == "measured"
        # The extracted curve, not the true one: a few us apart.
        assert back.modes[0].delay_spread_us == pytest.approx(width["spread_us"],
                                                              abs=3.0)
    else:
        assert back.modes[0].delay_spread_us < 8.0


def test_a_coherent_width_is_not_taken_for_scatter(tmp_path, monkeypatch):
    # Centres correlated row to row: the width is kept, the spread is not.
    m = _flat(doppler_spread_hz=0.5, delay_spread_us=30.0)
    ion = _load(sounding.write_lfs(_state(m), tmp_path, seed=5))
    real = observe.cross_trace

    def correlated(*args, **kw):
        return {**real(*args, **kw), "centre_r1": 0.6}

    monkeypatch.setattr(observe, "cross_trace", correlated)
    back = observe.from_ionogram(ion)
    assert back.modes[0].delay_spread_us == 0.0
    assert "coherent" in back.modes[0].provenance["delay_spread"].note
    assert back.metrics["cross_trace"][back.modes[0].name]["spread_us"] > 20.0


def test_closed_loop_reports_each_realisation_as_it_is_read_back(tmp_path):
    calls = []
    check.closed_loop(_state(_law(spread_hz=0.5)), realisations=2, workdir=tmp_path,
                      progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 2), (1, 2), (2, 2)], \
        f"the closed loop did not report each realisation: {calls}"


def _gapped(rng) -> trace.Segment:
    """Rows of a trace, the last a step lower, and one point at that range 3 MHz on.

    The smoothing spline's knot interval across the gap holds no data, so it
    passes close to every point and swings far off the trace between them.
    """
    n = int(rng.integers(10, 13))
    near = (rng.uniform(10.2, 10.4) + 0.0205 * np.arange(n)
            + rng.uniform(-0.004, 0.004, n))
    km = rng.uniform(2650.0, 2750.0) + rng.normal(0.0, 4.0, n)
    km[-1] -= rng.uniform(60.0, 80.0)
    return trace.Segment(np.append(near, near[-1] + rng.uniform(3.0, 3.3)),
                         np.append(km, km[-1]))


def _across(seg, rng, tmp_path, monkeypatch) -> Mode:
    """The one mode from_ionogram makes of ``seg``, on a sounding of its band."""
    ion = _load(sounding.write_lfs(_state(_flat()), tmp_path,
                                   seed=int(rng.integers(1 << 16))))
    monkeypatch.setattr(observe.trace, "group_tracks", lambda *_: [seg])
    (mode,) = observe.from_ionogram(ion).modes
    return mode


def test_a_gap_in_a_trace_keeps_its_mode_on_its_points(tmp_path, monkeypatch):
    rng = np.random.default_rng()
    seg = _gapped(rng)
    km = _across(seg, rng, tmp_path, monkeypatch).delay_us * KM_PER_US
    assert max(km.max() - seg.vrange.max(), seg.vrange.min() - km.min()) < 30.0, \
        "the mode's delay does not stay within 30 km of its points across the gap"


def test_a_gap_in_a_trace_leaves_its_mode_the_whole_band(tmp_path, monkeypatch):
    rng = np.random.default_rng()
    seg = _gapped(rng)
    mode = _across(seg, rng, tmp_path, monkeypatch)
    assert (mode.f_lo, mode.f_hi) == pytest.approx(seg.freq_span), \
        "the mode does not span its whole trace across the gap"


def test_a_trace_without_a_gap_keeps_its_spline():
    rng = np.random.default_rng()
    freq = np.linspace(10.0, 14.0, 90) + rng.uniform(-0.01, 0.01, 90)
    seg = trace.Segment(freq, rng.uniform(2650.0, 2750.0) + 30.0 * (freq - 10.0) ** 2
                        + rng.normal(0.0, 5.0, freq.size))
    assert np.array_equal(observe._curve(seg)[1], trace.reconstruct(seg).vrange), \
        "a trace with no gap does not keep muf.trace.reconstruct's curve"
