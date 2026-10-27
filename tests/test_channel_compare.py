"""Judging a channel simulator against real echo windows (`tools/channel_compare.py`).

The estimators against signals whose answer is known, the verdicts against
values whose spread is known, and one window measured end to end.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")

from conftest import make_header_bytes

from muf import echo_window

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "channel_compare", ROOT / "tools" / "channel_compare.py")
cc = importlib.util.module_from_spec(spec)
sys.modules["channel_compare"] = cc
spec.loader.exec_module(cc)

SR = 40_000.0               # Hz, the synthetic capture's decimated rate
RATE = 100e3                # Hz/s
BEAT_HZ = 2900.0 / (-3e5 / RATE)    # an echo at 2900 km, oblique .lfs
FS = 100.0                  # Hz, the fading series' rate


def _gaussian(n: int, fs: float, sigma_hz: float, centre_hz: float = 0.0,
              seed: int = 0) -> np.ndarray:
    """A complex process with a Gaussian Doppler spectrum: Rayleigh fading."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, d=1.0 / fs)
    mag = np.sqrt(np.exp(-0.5 * ((f - centre_hz) / sigma_hz) ** 2))
    x = np.fft.ifft(mag * np.exp(2j * np.pi * rng.random(n)))
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _sounding(name="a", t0=0.0, tx="synthtx", rx="synthrx", trace=None, **values):
    return cc.Sounding(Path(name), tx, rx, t0, values, trace)


# --------------------------------------------------------------------------
# Fading
# --------------------------------------------------------------------------

def test_rayleigh_fading_reads_an_s4_of_about_one():
    found = [cc.fading(_gaussian(int(40 * FS), FS, 0.5, seed=s), FS)
             for s in range(10)]
    s4 = np.mean([f["s4"] for f in found])
    # A 40 s realisation holds some 130 independent fades, so one reading
    # scatters by a tenth; ten average to within a few per cent of 1.
    assert 0.85 < s4 < 1.05
    assert np.median([f["rice_k"] for f in found]) < 1.5
    assert all(0.05 < f["fade_s"] < 0.4 for f in found)


def test_a_steady_mode_does_not_fade_however_its_strength_drifts():
    t = np.arange(int(40 * FS)) / FS
    # 10 dB stronger by the end of the stretch, as a mode is nearer its MUF.
    found = cc.fading(10.0 ** (t / 40 / 2) * np.exp(2j * np.pi * 0.2 * t), FS)
    assert found["s4"] < 0.01
    assert found["rice_k"] > 1e3
    assert found["fade_rate_hz"] == 0 and found["fade_s"] is None


def test_noise_beyond_the_mode_does_not_make_fades():
    """Only 3 dB above the noise at 100 Hz, but 15 dB within the 3 Hz kept."""
    n = int(40 * FS)
    rng = np.random.default_rng(1)
    noise = 0.5 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    raw = np.abs(1.0 + noise) ** 2
    assert np.mean(raw < 0.1 * np.median(raw)) > 0.01      # unfiltered, it fades
    found = cc.fading(np.ones(n) + noise, FS)
    assert found["s4"] < 0.4
    assert found["fade_rate_hz"] == 0


def test_a_stretch_too_short_has_no_fading():
    found = cc.fading(_gaussian(int(cc.MIN_BAND_S * FS) - 1, FS, 0.5), FS)
    assert set(found.values()) == {None}


@pytest.mark.parametrize("k", [0.5, 2.0, 10.0])
def test_rice_k_inverts_its_s4(k):
    assert cc.rice_k(math.sqrt(1 + 2 * k) / (1 + k)) == pytest.approx(k)


def test_rice_k_at_its_ends():
    assert cc.rice_k(1.0) == 0.0 and cc.rice_k(1.3) == 0.0
    assert cc.rice_k(0.0) is None


# --------------------------------------------------------------------------
# Delay spread
# --------------------------------------------------------------------------

def _ionogram(rows: dict, freq=(10.0, 11.0, 12.0, 13.0)) -> SimpleNamespace:
    """Rows of lit cells, ``{row: {km: dB above noise}}``; the rest unlit."""
    vrange = np.arange(2500.0, 3500.0, 50.0)
    db = np.full((len(freq), vrange.size), echo_window.NOISE_DB)
    for i, cells in rows.items():
        for km, above in cells.items():
            db[i, int(np.searchsorted(vrange, km))] = echo_window.NOISE_DB + above
    return SimpleNamespace(freq=np.asarray(freq), vrange=vrange, db=db,
                           header=SimpleNamespace(div_coef=2))


def test_two_equal_modes_a_millisecond_apart_spread_half_a_millisecond():
    # Oblique: 300 km of group path to the millisecond.
    ion = _ionogram({0: {2700: 30, 3000: 30}, 1: {2700: 30, 3000: 30},
                     2: {2700: 30, 3000: 30}})
    assert cc.delay_spread_ms(ion, (10.0, 12.0)) == pytest.approx(0.5)


def test_one_mode_has_no_spread_and_the_weaker_one_counts_less():
    assert cc.delay_spread_ms(_ionogram({0: {2900: 30}}), (10.0, 10.0)) == 0.0
    weaker = cc.delay_spread_ms(_ionogram({0: {2700: 30, 3000: 20}}), (10.0, 10.0))
    assert 0.0 < weaker < 0.5


def test_delay_spread_reads_only_where_the_mode_reaches():
    # Above the MUF a row holds one noise cell, and would read no spread.
    ion = _ionogram({0: {2700: 30, 3000: 30}, 1: {2700: 30, 3000: 30},
                     2: {3300: 15}, 3: {2600: 15}})
    assert cc.delay_spread_ms(ion, (10.0, 11.0)) == pytest.approx(0.5)
    assert cc.delay_spread_ms(_ionogram({}), (10.0, 13.0)) is None


# --------------------------------------------------------------------------
# The Doppler stretch
# --------------------------------------------------------------------------

def _traced(monkeypatch, beat, snr):
    tseg = (np.arange(len(beat)) + 0.5) * 0.1
    monkeypatch.setattr(cc.estimator, "_trace",
                        lambda *a, **k: (tseg, np.asarray(beat, float),
                                         np.asarray(snr, float)))


def _stretch():
    return cc.stretch(None, SR, RATE, 10.0, (-1600, -400), -3.0)


def test_the_stretch_is_the_middle_of_the_longest_unbroken_run(monkeypatch):
    snr = np.full(600, 30.0)
    snr[:100] = 0.0                                     # 0-10 s: nothing
    _traced(monkeypatch, np.full(600, -966.0), snr)
    lo, hi = _stretch()
    # 10-60 s is traced; its middle 40 s, 15-55 s, at 100 kHz/s from 10 MHz.
    assert lo == pytest.approx(11.5, abs=0.02) and hi == pytest.approx(15.5, abs=0.02)


def test_a_jump_to_another_mode_ends_the_stretch(monkeypatch):
    beat = np.full(600, -966.0)
    beat[350:] -= cc.MODE_JUMP_KM / 3.0 + 5             # 35 s: another mode
    snr = np.full(600, 30.0)
    _traced(monkeypatch, beat, snr)
    lo, hi = _stretch()
    assert (hi - lo) / (RATE / 1e6) == pytest.approx(35.0, abs=0.2)
    assert lo == pytest.approx(10.0, abs=0.02)

    # A fade inside one mode wanders the beat by less, and keeps the run.
    beat[350:] = -966.0 - 20 / 3.0
    _traced(monkeypatch, beat, snr)
    assert _stretch()[1] - _stretch()[0] == pytest.approx(4.0, abs=0.02)


def test_a_gap_ends_the_stretch_and_a_short_one_is_none(monkeypatch):
    snr = np.full(400, 30.0)
    snr[150:170] = 0.0                                  # 2 s untraced at 15 s
    _traced(monkeypatch, np.full(400, -966.0), snr)
    assert _stretch() is not None
    lo, hi = _stretch()
    assert (hi - lo) / (RATE / 1e6) == pytest.approx(23.0, abs=0.2)

    snr[100:120] = 0.0                                  # and at 10 s
    snr[300:320] = 0.0                                  # and at 30 s
    _traced(monkeypatch, np.full(400, -966.0), snr)
    assert _stretch() is None


# --------------------------------------------------------------------------
# Pairs and neighbours
# --------------------------------------------------------------------------

def test_a_simulated_sounding_pairs_with_the_nearest_real_one_of_its_circuit():
    real = [_sounding("r0", 0), _sounding("r1", 300), _sounding("r2", 40, tx="other")]
    sim = [_sounding("s0", 30), _sounding("s1", 20), _sounding("s2", 290),
           _sounding("s3", 1000)]
    pairs = {s.path.name: r.path.name for r, s in cc.paired(real, sim)}
    # s1 is nearer r0 than s0 is, and takes it; s0 finds nothing else in 60 s.
    assert pairs == {"s1": "r0", "s2": "r1"}


def test_neighbours_are_consecutive_soundings_of_one_circuit():
    real = [_sounding("a", 600), _sounding("b", 0), _sounding("c", 1200),
            _sounding("d", 3000), _sounding("e", 300, tx="other")]
    near = [(a.path.name, b.path.name) for a, b in cc.neighbours(real)]
    assert near == [("b", "a"), ("a", "c")]


def test_traces_compare_where_both_reach():
    f = np.linspace(10.0, 20.0, 101)
    one, other = (f, 2900.0 + 0 * f), (f[50:] + 0.0, 2910.0 + 0 * f[50:])
    assert cc.trace_distance_km(one, other) == pytest.approx(10.0)
    assert cc.trace_distance_km(one, (f[:3], 2900 + 0 * f[:3])) is None
    assert cc.trace_distance_km(one, None) is None


def test_two_empty_soundings_agree_about_nothing():
    assert cc.difference(_sounding(modes=0), _sounding(modes=0), "modes") is None
    assert cc.difference(_sounding(modes=0), _sounding(modes=2), "modes") == 2
    assert cc.difference(_sounding(muf_mhz=None), _sounding(muf_mhz=20.0),
                         "muf_mhz") is None


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

def _day(muf, step_s=300.0, **kw):
    return [_sounding(f"r{i}", i * step_s, muf_mhz=m, **kw) for i, m in enumerate(muf)]


def test_a_simulator_drawn_from_the_real_channel_passes_in_nineteen_of_twenty():
    """The tolerance is real-to-real as large as it comes, not as it usually is.

    Were it the median, a simulator whose errors are a second draw from the
    real-to-real distribution would fail half the time.
    """
    passed = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        # The MUF wanders by 0.3 MHz a sounding, and the simulator misses it
        # by as much as one sounding misses the next.
        real = _day(20.0 + np.cumsum(rng.normal(0, 0.3, 40)))
        sim = [_sounding(f"s{i}", r.t0 + 5,
                         muf_mhz=r.values["muf_mhz"] + rng.normal(0, 0.3))
               for i, r in enumerate(real)]
        row = cc.judge_pairs(cc.paired(real, sim), cc.neighbours(real), "muf_mhz")
        passed += row["verdict"] == "pass"
    assert passed >= 88


def test_a_simulator_further_than_the_real_channel_moves_fails():
    rng = np.random.default_rng(0)
    real = _day(20.0 + np.cumsum(rng.normal(0, 0.3, 40)))
    sim = [_sounding(f"s{i}", r.t0 + 5, muf_mhz=r.values["muf_mhz"] + 1.0)
           for i, r in enumerate(real)]
    row = cc.judge_pairs(cc.paired(real, sim), cc.neighbours(real), "muf_mhz")
    assert row["pairs"] == 40 and row["neighbours"] == 39
    assert row["sim_vs_real"] == pytest.approx(1.0)
    assert row["real_vs_real"] < row["tolerance"] < 1.0
    assert row["verdict"] == "fail"


def test_too_few_pairs_are_not_judged():
    real = _day([20.0, 20.5, 21.0, 20.0])
    sim = [_sounding("s", 5, muf_mhz=25.0)]
    row = cc.judge_pairs(cc.paired(real, sim), cc.neighbours(real), "muf_mhz")
    assert row["verdict"] == "not judged" and row["tolerance"] is None
    assert row["sim_vs_real"] == pytest.approx(5.0)


def _spread(values, prefix="r"):
    return [_sounding(f"{prefix}{i}", i * 300.0, doppler_rms_hz=v)
            for i, v in enumerate(values)]


def test_the_same_distribution_passes_and_another_fails():
    rng = np.random.default_rng(3)
    real = _spread(rng.lognormal(0.0, 0.3, 60))
    same = cc.judge_distributions(real, _spread(rng.lognormal(0.0, 0.3, 30), "s"),
                                  "doppler_rms_hz")
    wider = cc.judge_distributions(real, _spread(rng.lognormal(0.5, 0.3, 30), "s"),
                                   "doppler_rms_hz")
    assert same["verdict"] == "pass" and same["ks"] <= same["tolerance"]
    assert wider["verdict"] == "fail" and wider["ks"] > wider["tolerance"]
    assert same["tolerance"] == wider["tolerance"]
    assert wider["sim_median"] == pytest.approx(math.e ** 0.5, rel=0.2)


def test_the_distribution_tolerance_scales_with_the_sizes_compared():
    rng = np.random.default_rng(4)
    real = _spread(rng.normal(1.0, 0.1, 80))
    few = cc.judge_distributions(real, _spread(rng.normal(1, 0.1, 5), "s"),
                                 "doppler_rms_hz")
    many = cc.judge_distributions(real, _spread(rng.normal(1, 0.1, 80), "s"),
                                  "doppler_rms_hz")
    # Five simulated values lie further from eighty real ones by chance alone.
    assert few["tolerance"] > 2 * many["tolerance"]


def test_too_few_values_are_not_judged():
    row = cc.judge_distributions(_spread([1.0] * 5), _spread([2.0] * 10, "s"),
                                 "doppler_rms_hz")
    assert row["verdict"] == "not judged" and row["ks"] is None


def test_ks_distance():
    assert cc.ks_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert cc.ks_distance([1, 2, 3], [4, 5]) == 1.0
    assert cc.ks_distance([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# One window, end to end
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    """Two synthetic echo windows ten minutes apart: one Rayleigh-fading mode
    at 2900 km with a 0.5 Hz Doppler spread, over 30 s of sweep."""
    out = tmp_path_factory.mktemp("channel_compare")
    n = int(30 * SR)
    found = []
    for i, minute in enumerate((0, 10)):
        rng = np.random.default_rng(i)
        iq = (rng.normal(size=n) + 1j * rng.normal(size=n)
              + 100.0 * _gaussian(n, SR, 0.5, BEAT_HZ, seed=10 + i))
        lfs = out / f"synth_20260204_00{minute:02d}10.lfs"
        lfs.write_bytes(make_header_bytes(start_minute=minute,
                                          start_epoch=1770163210 + 60 * minute)
                        + iq.astype(np.complex64).tobytes())
        found.append(echo_window.write(lfs, out_dir=out / "windows").target)
    return found


@pytest.mark.filterwarnings("ignore:.*re-derived from its echo window")
def test_one_window_measured(windows):
    found = cc.measure(windows[0])
    assert (found.tx, found.rx, found.t0) == ("synthtx", "synthrx", 1770163210.0)
    v = found.values
    assert v["modes"] == 1
    assert np.median(found.trace[1]) == pytest.approx(2900.0, abs=15)
    # The trace spans the whole sweep, so both ends are bounds and left out.
    assert v["muf_mhz"] is None and v["lof_mhz"] is None
    assert v["doppler_rms_hz"] == pytest.approx(0.5, rel=0.2)
    assert (v["band_mhz"][1] - v["band_mhz"][0]) / (RATE / 1e6) > cc.MIN_BAND_S
    assert 0.5 < v["s4"] < 1.4
    # One mode: the spread of its own cells about it, under a 14.65 km bin.
    assert 0.0 < v["delay_spread_ms"] < 14.65 / 300


@pytest.mark.filterwarnings("ignore:.*re-derived from its echo window")
def test_the_command_reports_and_writes_json(windows, tmp_path, capsys):
    report = tmp_path / "report.json"
    bogus = tmp_path / "not_a_window.echo.h5"
    bogus.write_bytes(b"nothing")
    status = cc.main(["--real", *map(str, windows), "--sim", str(windows[0]),
                      str(bogus), "--json", str(report)])
    out, err = capsys.readouterr()
    assert status == 0
    assert "Per sounding: 1 pairs" in out and "not judged" in out
    assert "skipped simulated not_a_window.echo.h5" in err
    result = json.loads(report.read_text())
    assert [p["real"] for p in result["pairs"]] == [windows[0].name]
    assert result["pairs"][0]["trace_km"] == 0.0
    assert result["neighbours"] == 1
    assert [len(result["soundings"][k]) for k in ("real", "sim")] == [2, 1]


def test_the_command_fails_when_a_metric_does(monkeypatch, capsys):
    rng = np.random.default_rng(0)
    real = {f"r{i}": s for i, s in enumerate(_day(20.0 + np.cumsum(
        rng.normal(0, 0.3, 10))))}
    sim = {f"s{i}": _sounding(f"s{i}", r.t0, muf_mhz=r.values["muf_mhz"] + 2.0)
           for i, r in enumerate(real.values())}
    monkeypatch.setattr(cc, "measure", lambda path, method, band: {**real, **sim}[
        path.name])
    assert cc.main(["--real", *real, "--sim", *sim]) == 1
    assert "MUF, MHz" in capsys.readouterr().out


def test_the_command_refuses_a_band_backwards():
    with pytest.raises(SystemExit):
        cc.main(["--real", "a", "--sim", "b", "--band", "26-22"])
