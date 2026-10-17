"""Keeping the echo window of the dechirped voltage (``docs/echo-window.md``).

The six criteria the task set, in its order, against both sources: a v2
product built by ``make_chirp_z_h5`` and a synthetic ``.lfs``.

Position comes first because that is where the sign trap is. v2 forms its
spectrogram on ``conj(z)`` and ``.lfs`` on the samples as they are, so one
echo sits at opposite beat frequencies on disk in the two formats -- and a
window cut on the wrong side keeps an empty band, decimates it beautifully,
and passes every test that does not ask *where the echo went*.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
signal = pytest.importorskip("scipy.signal")

from conftest import ECHO_RANGE_KM, make_header_bytes  # noqa: E402

from muf import cli, echo_window, io_chirp, loader, spectro  # noqa: E402

SR = 40_000.0               # Hz, the cyprus1 decimated rate
RATE = 100e3                # Hz/s
LFS_KM_PER_HZ = -3e5 / RATE                  # oblique .lfs: -c/rate
V2_KM_PER_HZ = -io_chirp.C_M_S / RATE / 1e3  # v2: -c/rate, v2's exact c

# A v2 product shaped like a real one, small enough to build per test.
V2_WINDOW = 2048
V2_STEP = 4096
V2_BIN_KM = io_chirp.C_M_S / RATE * SR / V2_WINDOW / 1e3     # 58.55 km

# .lfs at the pipeline's own window.
LFS_BIN_KM = 2 * (3e5 * SR / 2 / RATE) / spectro.DEFAULT_WINDOW   # 14.65 km

STATIONS = {"synthtx": (35.0, 34.0), "synthrx": (56.38, 47.53)}


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def _noise(n: int, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))


def _tone(n: int, beat_hz: float, amplitude: float = 30.0) -> np.ndarray:
    return amplitude * np.exp(2j * np.pi * beat_hz * np.arange(n) / SR)


def _lfs_beat(range_km: float) -> float:
    return range_km / LFS_KM_PER_HZ


#: v2 stores SNR as float16, which saturates at 65504 -- 33 dB at 2048 bins
#: for a unit tone over unit noise, and an amplitude of 30 would clip three
#: bins flat and leave the peak to argmax's tie-break.
V2_AMPLITUDE = 1.0


def _v2_beat(range_km: float) -> float:
    """On disk. The product's spectrogram is of the conjugate, so the tone
    that *reads* at ``range_km`` is stored at minus its beat."""
    return (range_km - ECHO_RANGE_KM) / V2_KM_PER_HZ


def _gaussian_echo(n: int, centre_hz: float, sigma_hz: float,
                   power: float = 1e4, seed: int = 5) -> np.ndarray:
    """A random process with an *exactly* Gaussian spectrum.

    Magnitudes from the Gaussian on the FFT grid, phases random: its
    periodogram is the Gaussian sample for sample, so the injected centre and
    width are the true ones and not a draw from them.
    """
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, d=1.0 / SR)
    mag = np.sqrt(np.exp(-0.5 * ((f - centre_hz) / sigma_hz) ** 2))
    spectrum = mag * np.exp(2j * np.pi * rng.random(n))
    x = np.fft.ifft(spectrum)
    return x * np.sqrt(power / np.mean(np.abs(x) ** 2))


@pytest.fixture
def make_lfs_iq(tmp_path):
    def _make(iq: np.ndarray, name: str = "synth_20260204_000010.lfs") -> object:
        path = tmp_path / name
        with open(path, "wb") as fh:
            fh.write(make_header_bytes())
            fh.write(np.asarray(iq, dtype=np.complex64).tobytes())
        return path

    return _make


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_km(ion) -> float:
    return float(ion.vrange[int(np.argmax(np.asarray(ion.power).mean(axis=0)))])


def _filtered(samples: np.ndarray, plan, times: int = 1) -> np.ndarray:
    """The source shifted and filtered ``times`` times at the source rate.

    Full convolution, so the filter's response either side of the record is
    kept: element ``m + times * (numtaps - 1) // 2`` is source sample ``m``.
    """
    out = echo_window._mix(samples, -plan.shift_hz, plan.source_sr_hz)
    for _ in range(times):
        out = signal.fftconvolve(out, plan.taps(), mode="full")
    return out


# --------------------------------------------------------------------------
# 1. Position -- the sign trap, first
# --------------------------------------------------------------------------

def test_an_lfs_echo_comes_back_at_its_range(make_lfs_iq, tmp_path):
    r_true = 198 * LFS_BIN_KM                         # 2900.4 km, on a bin
    n = int(20 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(r_true)))

    report = echo_window.write(lfs, out_dir=tmp_path)
    window = echo_window.read(report.target)
    assert window.range_lo_km < r_true < window.range_hi_km

    full = spectro.compute(lfs)
    derived = echo_window.reprocess_window(report.target)
    assert _peak_km(full) == pytest.approx(r_true, abs=LFS_BIN_KM / 2)
    assert _peak_km(derived) == pytest.approx(r_true, abs=LFS_BIN_KM / 2)


def test_a_full_band_window_loads_at_the_lfs_gate(make_lfs_iq, tmp_path):
    """The whole band, undecimated -- ionozond's ``--full`` -- re-derived with
    no gate lands on the axis the capture itself loads at, not +-60000 km."""
    r_true = 198 * LFS_BIN_KM
    n = int(20 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(r_true)))
    half = abs(LFS_KM_PER_HZ) * SR / 2

    report = echo_window.write(lfs, out_dir=tmp_path, manual_km=(-half, half))
    assert echo_window.read(report.target).plan.numtaps == 1

    full = spectro.compute(lfs)
    derived = echo_window.reprocess_window(report.target)
    np.testing.assert_allclose(derived.vrange, full.vrange)
    np.testing.assert_allclose(derived.power, full.power, rtol=1e-4)


def test_a_window_outside_the_lfs_gate_is_kept_whole(make_lfs_iq, tmp_path):
    n = int(20 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(1200.0)))
    report = echo_window.write(lfs, out_dir=tmp_path, manual_km=(800.0, 1600.0))

    derived = echo_window.reprocess_window(report.target)
    assert 800.0 <= derived.vrange.min() and derived.vrange.max() <= 1600.0
    assert _peak_km(derived) == pytest.approx(1200.0, abs=LFS_BIN_KM)


def test_a_v2_echo_comes_back_at_its_range(make_chirp_z_h5):
    r_true = ECHO_RANGE_KM + 7 * V2_BIN_KM          # 3119.9 km, on a bin
    n = int(10 * SR)
    path = make_chirp_z_h5(z=_noise(n) + _tone(n, _v2_beat(r_true), V2_AMPLITUDE),
                           window=V2_WINDOW, step=V2_STEP)
    stored = io_chirp.load(path)
    assert _peak_km(stored) == pytest.approx(r_true, abs=V2_BIN_KM / 2)

    echo_window.write(path)
    window = echo_window.read(path)
    assert window.range_lo_km < r_true < window.range_hi_km
    # The mirror image a missing conjugation would read the echo at.
    mirror = 2 * ECHO_RANGE_KM - r_true
    assert not window.range_lo_km < mirror < window.range_hi_km

    derived = echo_window.reprocess_window(path)
    assert _peak_km(derived) == pytest.approx(r_true, abs=V2_BIN_KM / 2)


def test_the_shift_has_the_formats_own_sign(make_lfs_iq, make_chirp_z_h5, tmp_path):
    """Both echoes are beyond the band's zero, so both shifts are negative --
    but v2's because of the conjugate and ``.lfs``'s because of the axis."""
    n = int(10 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    v2 = make_chirp_z_h5(z=_noise(n) + _tone(n, _v2_beat(3100.0), V2_AMPLITUDE),
                         window=V2_WINDOW, step=V2_STEP)
    lfs_window = echo_window.read(echo_window.write(lfs, out_dir=tmp_path).target)
    echo_window.write(v2)
    v2_window = echo_window.read(v2)

    for w, r_km in ((lfs_window, 2900.0), (v2_window, 3100.0)):
        assert w.range_lo_km < r_km < w.range_hi_km
        centre = (w.range_lo_km + w.range_hi_km) / 2
        assert w.shift_hz == pytest.approx(
            (centre - w.range_zero_km) / w.km_per_hz, rel=1e-9)
        assert w.range_zero_km + w.km_per_hz * w.shift_hz == pytest.approx(centre)
    assert lfs_window.km_per_hz < 0 and v2_window.km_per_hz < 0
    # float64 holds a 1.77e9 s epoch to 2.4e-7 s: 0.07 km of range.
    assert v2_window.range_zero_km == pytest.approx(ECHO_RANGE_KM, abs=0.1)


# --------------------------------------------------------------------------
# 2. Suppression
# --------------------------------------------------------------------------

def test_echoes_outside_the_window_are_attenuated_by_the_stopband(make_lfs_iq,
                                                                  tmp_path):
    n = int(10 * SR)
    lfs = make_lfs_iq(np.zeros(n))
    source = echo_window.source_of(lfs, STATIONS)
    plan = echo_window.plan_for(source, 2600.0, 3200.0)
    assert plan.attenuation_db >= echo_window.STOPBAND_DB
    assert plan.ripple_db <= echo_window.MAX_RIPPLE_DB

    # Just past the stopband edge on each side, and far away.
    edge_km = abs(plan.stopband_hz * LFS_KM_PER_HZ)
    centre = source.range_at(plan.shift_hz)
    trim = plan.numtaps // plan.decimation + 1
    for r_km in (centre + edge_km + 5.0, centre - edge_km - 5.0, 9000.0, -4000.0):
        amplitude = 1e3
        z_win = echo_window.cut(_tone(n, _lfs_beat(r_km), amplitude), plan)
        leaked = np.mean(np.abs(z_win[trim:-trim]) ** 2) / amplitude ** 2
        assert 10 * np.log10(leaked) <= -echo_window.STOPBAND_DB, r_km

    # ...and one inside comes through at unit gain.
    z_win = echo_window.cut(_tone(n, _lfs_beat(2900.0), 1e3), plan)
    kept = np.mean(np.abs(z_win[trim:-trim]) ** 2) / 1e6
    assert 10 * np.log10(kept) == pytest.approx(0.0, abs=echo_window.MAX_RIPPLE_DB)


# --------------------------------------------------------------------------
# 3. Doppler shift and spread
# --------------------------------------------------------------------------

def _doppler(samples, sr_hz, shift_hz, beat_hz, sigma_hint):
    """Centroid and RMS width of the echo, about its range's beat, in Hz.

    The periodogram over the whole record, noise floor from well outside the
    echo removed -- the estimator the real-data check uses, minus the
    second dechirp a moving trace would need. Centred on the peak, so a
    shift of several widths is not truncated by the integration limits.
    """
    spec = np.abs(np.fft.fft(samples)) ** 2
    f = np.fft.fftfreq(len(samples), d=1.0 / sr_hz) + shift_hz - beat_hz
    peak = f[int(np.argmax(spec))]
    core = np.abs(f - peak) <= 8 * sigma_hint
    away = (np.abs(f - peak) > 15 * sigma_hint) & (np.abs(f - peak) < 40 * sigma_hint)
    s = np.clip(spec[core] - np.median(spec[away]), 0.0, None)
    nu = float(np.sum(f[core] * s) / np.sum(s))
    sigma = float(np.sqrt(np.sum((f[core] - nu) ** 2 * s) / np.sum(s)))
    return nu, sigma


@pytest.mark.parametrize("nu_hz, sigma_hz", [(0.3, 0.5), (-1.2, 0.2)])
def test_doppler_shift_and_spread_survive_the_window(make_lfs_iq, tmp_path,
                                                     nu_hz, sigma_hz):
    """Estimated from the stored samples themselves, over exactly the record.

    A random-phase line spectrum is exactly Gaussian only on the grid it was
    built on. So the record is a whole number of window samples, the estimate
    spans the same 40 s, and the shift -- not a multiple of 1/40 Hz -- is
    mixed back out first, which is lossless. On any other grid the lines leak
    into one another and a 0.2 Hz width reads 10-15 % wide, from the
    reconstructed source just as from the window: the estimator's variance on
    one realisation, not anything the window lost.
    """
    window_km = (2600.0, 3200.0)
    probe = make_lfs_iq(np.zeros(16), name="probe.lfs")
    d = echo_window.plan_for(echo_window.source_of(probe), *window_km).decimation
    n = int(40 * SR) // d * d
    beat = _lfs_beat(198 * LFS_BIN_KM)
    iq = _noise(n) + _gaussian_echo(n, beat + nu_hz, sigma_hz)
    lfs = make_lfs_iq(iq)

    target = echo_window.write(lfs, out_dir=tmp_path, manual_km=window_km).target
    window = echo_window.read(target)
    assert window.decimation == d
    record = window.samples[window.plan.lead:window.plan.lead + n // d]
    record = echo_window._mix(record, window.shift_hz, window.sr_hz)
    aliased = (beat + window.sr_hz / 2) % window.sr_hz - window.sr_hz / 2
    nu_w, sigma_w = _doppler(record, window.sr_hz, 0.0, aliased, sigma_hz)
    nu_s, sigma_s = _doppler(iq, SR, 0.0, beat, sigma_hz)

    assert abs(nu_s - nu_hz) < 0.02 and sigma_s == pytest.approx(sigma_hz, rel=0.05)
    assert abs(nu_w - nu_hz) < 0.02
    assert sigma_w == pytest.approx(sigma_hz, rel=0.05)


# --------------------------------------------------------------------------
# 4. Group-delay bookkeeping
# --------------------------------------------------------------------------

def _dispersed(n: int, r0_km: float, km_per_mhz: float, amplitude=30.0):
    """An echo whose delay grows linearly with sweep frequency."""
    t = np.arange(n) / SR
    r = r0_km + km_per_mhz * RATE * t / 1e6
    beat = r / LFS_KM_PER_HZ
    phase = 2 * np.pi * np.cumsum(beat) / SR
    return amplitude * np.exp(1j * phase)


def _slope(ion) -> float:
    """km per MHz of the brightest range per row, sub-bin by parabola."""
    p = np.asarray(ion.power, dtype=np.float64)
    vr = np.asarray(ion.vrange)
    step = vr[0] - vr[1]
    peaks = []
    for row in p:
        i = int(np.clip(np.argmax(row), 1, len(row) - 2))
        a, b, c = np.log(row[i - 1:i + 2])
        peaks.append(vr[i] + step * 0.5 * (a - c) / (a - 2 * b + c))
    return float(np.polyfit(np.asarray(ion.freq), peaks, 1)[0])


def test_a_dispersed_echo_keeps_its_slope(make_lfs_iq, tmp_path):
    n = int(40 * SR)
    lfs = make_lfs_iq(_noise(n) + _dispersed(n, 2800.0, 60.0))
    target = echo_window.write(lfs, out_dir=tmp_path).target

    full = _slope(spectro.compute(lfs))
    derived = _slope(echo_window.reprocess_window(target))
    assert full == pytest.approx(60.0, rel=0.02)
    assert derived == pytest.approx(full, rel=1e-3)


def test_every_stored_sample_is_its_source_sample(make_lfs_iq, tmp_path):
    """Time, sweep frequency and content: ``z_win[k]`` *is* source sample
    ``k * D`` filtered. A one-sample slip would be a 0.05 rad error here."""
    n = int(10 * SR)
    iq = _noise(n) + _dispersed(n, 2800.0, 60.0)
    lfs = make_lfs_iq(iq)
    target = echo_window.write(lfs, out_dir=tmp_path).target
    window = echo_window.read(target)
    header = loader.read_header(lfs)
    d = window.decimation

    assert window.t0 == pytest.approx(header.start_epoch)
    source_index = np.arange(window.n) - window.plan.lead
    np.testing.assert_allclose(window.times_s(), source_index * d / SR)
    np.testing.assert_allclose(
        window.sweep_hz(),
        header.cf - header.sample_rate / 2 + RATE * source_index * d / SR)

    assert window.t_first_s == pytest.approx(-window.plan.lead * d / SR)

    # Stored sample k is source sample (k - lead) * D, filtered.
    half = (window.plan.numtaps - 1) // 2
    once = np.concatenate([np.zeros(d, complex),
                           _filtered(iq.astype(np.complex64), window.plan),
                           np.zeros(window.n * d, complex)])
    at = (np.arange(window.n) - window.plan.lead) * d + half + d
    error = np.abs(window.samples - once[at])
    assert error.max() < 1e-4 * 30.0
    # The same filter one sample late is nowhere near.
    assert np.abs(window.samples - once[at + 1]).max() > 1e-2


# --------------------------------------------------------------------------
# 5. Round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["lfs", "v2"])
def test_reconstruct_is_the_source_filtered_twice(make_lfs_iq, make_chirp_z_h5,
                                                  tmp_path, source):
    n = int(10 * SR)
    if source == "lfs":
        iq = (_noise(n) + _tone(n, _lfs_beat(2900.0))
              + _dispersed(n, 3000.0, 40.0, amplitude=10.0)).astype(np.complex64)
        target = echo_window.write(make_lfs_iq(iq), out_dir=tmp_path).target
    else:
        iq = (_noise(n) + _tone(n, _v2_beat(3100.0), V2_AMPLITUDE)).astype(np.complex64)
        target = make_chirp_z_h5(z=iq, window=V2_WINDOW, step=V2_STEP)
        echo_window.write(target)

    window = echo_window.read(target)
    plan = window.plan
    rebuilt = echo_window.reconstruct(window)
    assert len(rebuilt) == n

    half = (plan.numtaps - 1) // 2
    twice = _filtered(iq, plan, times=2)[2 * half:2 * half + n]
    reference = echo_window._mix(twice, plan.shift_hz, SR)
    rms = np.sqrt(np.mean(np.abs(rebuilt - reference) ** 2)
                  / np.mean(np.abs(reference) ** 2))
    assert rms < 1e-3


# --------------------------------------------------------------------------
# 6. Compatibility
# --------------------------------------------------------------------------

def _v2_product(make_chirp_z_h5, seconds: float = 10.0, **kw):
    n = int(seconds * SR)
    z = _noise(n) + _tone(n, _v2_beat(3100.0), V2_AMPLITUDE)
    return make_chirp_z_h5(z=z, window=V2_WINDOW, step=V2_STEP, **kw)


def test_required_datasets_are_unchanged():
    assert io_chirp.REQUIRED == ("SNR", "freqs", "ranges", "rate", "t0", "sr")


def test_a_product_with_the_group_loads_identically(make_chirp_z_h5):
    path = _v2_product(make_chirp_z_h5)
    before = loader.load(path)

    report = echo_window.write(path, verify=True)
    assert report.status == "written" and report.verification.ok

    header = loader.read_header(path)
    assert header.has_raw_voltage and header.has_echo_window
    after = loader.load(path)
    np.testing.assert_array_equal(after.power, before.power)
    np.testing.assert_array_equal(after.vrange, before.vrange)
    np.testing.assert_array_equal(after.freq, before.freq)


def test_writing_twice_is_a_no_op(make_chirp_z_h5):
    path = _v2_product(make_chirp_z_h5)
    echo_window.write(path)
    digest = _sha(path)

    again = echo_window.write(path)
    assert again.status == "present"
    assert _sha(path) == digest


def test_drop_raw_reclaims_the_space_and_keeps_the_product(make_chirp_z_h5):
    path = _v2_product(make_chirp_z_h5)
    before = loader.load(path)
    z_bytes = io_chirp.read_raw_voltage(path).nbytes
    size = path.stat().st_size

    report = echo_window.write(path, drop_raw=True)
    assert report.status == "raw-dropped" and report.verification.ok

    header = loader.read_header(path)
    assert not header.has_raw_voltage and header.has_echo_window
    assert path.stat().st_size < size - 0.9 * z_bytes
    after = loader.load(path)
    np.testing.assert_array_equal(after.power, before.power)
    np.testing.assert_array_equal(after.vrange, before.vrange)


def test_a_dropped_product_re_derives_from_its_window(make_chirp_z_h5):
    path = _v2_product(make_chirp_z_h5)
    echo_window.write(path, drop_raw=True)
    window = echo_window.read(path, samples=False)

    with pytest.warns(UserWarning, match="re-derived from its echo window"):
        ion = loader.load(path, window=V2_WINDOW // 2)
    assert ion.vrange.min() >= window.range_lo_km
    assert ion.vrange.max() <= window.range_hi_km
    assert _peak_km(ion) == pytest.approx(3100.0, abs=V2_BIN_KM)


def test_an_existing_window_can_still_drop_raw(make_chirp_z_h5):
    path = _v2_product(make_chirp_z_h5)
    echo_window.write(path)
    samples = echo_window.read(path).samples

    report = echo_window.write(path, drop_raw=True)
    assert report.status == "raw-dropped"
    assert not loader.read_header(path).has_raw_voltage
    np.testing.assert_array_equal(echo_window.read(path).samples, samples)


def test_an_interrupted_rewrite_leaves_the_original(make_chirp_z_h5, monkeypatch):
    path = _v2_product(make_chirp_z_h5)
    digest = _sha(path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(echo_window, "_same_except_z", interrupted)
    with pytest.raises(KeyboardInterrupt):
        echo_window.write(path, drop_raw=True)

    assert _sha(path) == digest
    assert loader.read_header(path).has_raw_voltage
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_a_failed_verification_keeps_z(make_chirp_z_h5, monkeypatch):
    path = _v2_product(make_chirp_z_h5)
    digest = _sha(path)
    monkeypatch.setattr(
        echo_window, "verify_window",
        lambda *a, **k: echo_window.Verification(False, ["forced"], {}))

    with pytest.raises(echo_window.VerificationError, match="raw `z` kept"):
        echo_window.write(path, drop_raw=True)
    assert _sha(path) == digest


def test_an_lfs_is_never_modified(make_lfs_iq, tmp_path):
    n = int(10 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    digest = _sha(lfs)

    with pytest.warns(UserWarning, match="never modified"):
        report = echo_window.write(lfs, out_dir=tmp_path / "out", drop_raw=True,
                                   verify=True)
    assert report.verification.ok
    assert _sha(lfs) == digest
    assert report.target.name == lfs.stem + ".echo.h5"


def test_a_standalone_window_carries_the_lfs_header(make_lfs_iq, tmp_path):
    n = int(10 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    target = echo_window.write(lfs, out_dir=tmp_path).target

    original = loader.read_header(lfs)
    carried = loader.read_header(target)
    for field in ("rate", "sample_rate", "dec", "cf", "start_epoch", "tx_name",
                  "rx_name", "tx_latitude", "rx_longitude"):
        assert getattr(carried, field) == getattr(original, field), field
    with h5py.File(target, "r") as fh:
        assert fh.attrs["rate_hz_s"] == RATE
        assert fh.attrs["t0_unix_s"] == original.start_epoch
        assert fh.attrs["sweep_start_hz"] == original.cf - original.sample_rate / 2
    assert loader.ECHO_WINDOW not in loader.FORMATS


def test_into_appends_to_a_product_of_the_same_capture(make_lfs_iq, make_chirp_h5,
                                                       tmp_path):
    n = int(10 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    product = make_chirp_h5(np.full((8, 64), 100.0))
    before = loader.load(product)

    report = echo_window.write(lfs, into=product)
    assert report.target == product
    assert loader.read_header(product).has_echo_window
    np.testing.assert_array_equal(loader.load(product).power, before.power)
    assert echo_window.read(product).source == echo_window.SOURCE_LFS


def test_no_candidate_on_a_relative_axis_has_nothing_to_keep(make_chirp_z_h5,
                                                             monkeypatch):
    """Nothing stands out and no geometry can place it: no window, not a guess.

    ``auto_gate`` is made to find nothing because whether it does on a short
    synthetic noise record is a property of ``auto_gate``, tested elsewhere.
    """
    path = make_chirp_z_h5(z=_noise(int(5 * SR)), window=V2_WINDOW, step=V2_STEP,
                           txname=io_chirp.UNIDENTIFIED_TX)
    digest = _sha(path)
    monkeypatch.setattr(echo_window.calibrate, "auto_gate", lambda *a, **k: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(echo_window.NoEchoError, match="nothing to keep"):
            echo_window.write(path)
    assert _sha(path) == digest


def test_an_echo_on_the_edge_is_flagged(make_lfs_iq, tmp_path):
    n = int(10 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    report = echo_window.write(lfs, out_dir=tmp_path, manual_km=(2860.0, 3400.0))
    window = echo_window.read(report.target)
    assert (window.range_lo_km, window.range_hi_km) == (2860.0, 3400.0)
    assert window.truncated_suspect
    assert "TRUNCATION SUSPECTED" in window.describe()
    assert window.selection["method"] == "manual"


def test_the_window_follows_the_energy_not_the_geometry(make_lfs_iq, tmp_path):
    """The 45 ms epoch error displaced an echo by 13,500 km. Geometry alone
    would keep an empty band; the union keeps the echo."""
    n = int(10 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(16400.0)))
    window = echo_window.read(echo_window.write(lfs, out_dir=tmp_path).target)

    selection = window.selection
    assert selection["geometry_km"] is not None
    assert selection["geometry_km"][1] < 16400.0
    assert window.range_lo_km < 16400.0 < window.range_hi_km


# --------------------------------------------------------------------------
# The filter travels with the samples
# --------------------------------------------------------------------------

def _lfs_window(make_lfs_iq, tmp_path):
    n = int(5 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    return lfs, echo_window.write(lfs, out_dir=tmp_path).target


def test_the_taps_are_stored_as_cut(make_lfs_iq, tmp_path):
    lfs, target = _lfs_window(make_lfs_iq, tmp_path)
    window = echo_window.read(target)

    with h5py.File(target, "r") as fh:
        stored = fh["echo_window/taps"]
        assert stored.dtype == np.float64
        assert stored.shape == (window.filter["numtaps"],)
    taps = window.taps
    assert taps.sum() == pytest.approx(1.0, abs=1e-12)        # unit DC gain
    np.testing.assert_array_equal(taps, taps[::-1])          # linear phase
    designed = echo_window.plan_for(echo_window.source_of(lfs),
                                    window.range_lo_km, window.range_hi_km)
    np.testing.assert_array_equal(taps, designed.taps())


def test_reading_needs_no_filter_designer(make_lfs_iq, tmp_path, monkeypatch):
    """What ionozond or any other reader relies on: the file is enough."""
    lfs, target = _lfs_window(make_lfs_iq, tmp_path)

    def no_designer(*args, **kwargs):
        raise AssertionError("a reader regenerated the taps")

    monkeypatch.setattr(echo_window, "_design", no_designer)
    window = echo_window.read(target)
    assert len(echo_window.reconstruct(window)) == window.source_samples
    echo_window.reprocess_window(target)
    check = echo_window.verify_window(target,
                                      source=echo_window.source_of(lfs))
    assert check.ok, check.describe()
    assert "filter" in check.metrics


def test_tampered_taps_fail_verification(make_lfs_iq, tmp_path):
    lfs, target = _lfs_window(make_lfs_iq, tmp_path)
    with h5py.File(target, "r+") as fh:
        taps = fh["echo_window/taps"]
        taps[taps.shape[0] // 2] *= 1.01

    check = echo_window.verify_window(target, source=echo_window.source_of(lfs))
    assert not check.ok
    assert any("re-cut differs" in p for p in check.problems)


def test_a_window_without_taps_is_refused(make_lfs_iq, tmp_path):
    _, target = _lfs_window(make_lfs_iq, tmp_path)
    with h5py.File(target, "r+") as fh:
        del fh["echo_window/taps"]

    with pytest.raises(ValueError, match="no 'taps'"):
        echo_window.read(target)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_dry_run_writes_nothing(make_lfs_iq, tmp_path, capsys):
    n = int(5 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    out = tmp_path / "out"

    assert cli.main(["echo-window", str(lfs), "--out", str(out), "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not out.exists()


def test_cli_writes_and_info_reports_it(make_lfs_iq, tmp_path, capsys):
    n = int(5 * SR)
    lfs = make_lfs_iq(_noise(n) + _tone(n, _lfs_beat(2900.0)))
    out = tmp_path / "out"
    out.mkdir()

    assert cli.main(["echo-window", str(lfs), "--out", str(out), "--verify"]) == 0
    assert "verified" in capsys.readouterr().out
    target = out / (lfs.stem + ".echo.h5")

    assert cli.main(["info", str(target)]) == 0
    text = capsys.readouterr().out
    assert "echo window" in text and "Hz (/" in text
