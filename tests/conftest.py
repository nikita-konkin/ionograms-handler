"""Fixtures: synthetic soundings with known content, and the real data if present.

The synthetic sounding is the backbone of the test suite. Real recordings have
no ground truth -- nobody knows the true MUF of a 2026-02-04 sounding -- so
correctness is established by building IQ whose echo sits at a chosen virtual
range and stops at a chosen frequency, then checking each estimator recovers
them.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from muf.io_lfs import HEADER_SIZE, _LAYOUT

#: Where the operational recordings live, when they are on this machine.
#: Recordings are large and get moved between drives, so the location is
#: overridable; tests that need them skip when it does not resolve.
#:
#:     set MUF_TEST_DATA=F:\MyData\ND\lfs\2026.02.04
REAL_DATA = Path(
    os.environ.get(
        "MUF_TEST_DATA",
        Path(__file__).resolve().parent.parent / "data" / "2026.02.04",
    )
)

#: Where the chirpsounder2 products live. Same arrangement as ``REAL_DATA``, but
#: a separate variable because the two formats are recorded by different
#: instruments and will not generally sit in the same tree.
#:
#:     set MUF_TEST_CHIRP_DATA=F:\MyData\ND\lfs\2026-08-04
REAL_CHIRP_DATA = Path(
    os.environ.get(
        "MUF_TEST_CHIRP_DATA",
        Path(__file__).resolve().parent.parent / "data" / "2026-08-04",
    )
)

#: Parameters of the real instrument, so synthetic files behave like it. Taken
#: from cyprus1_20260204_000010.lfs.
INSTRUMENT = dict(
    cf=20_000_000,
    sample_rate=25_000_000,
    dec=625,
    dur=250,
    rate=100_000,
    rmin=0,
    rmax=5000,
)


def make_header_bytes(**overrides) -> bytes:
    """Build a 512-byte LFS header.

    Field offsets come from ``io_lfs._LAYOUT``, so this exercises the parser's
    logic but cannot catch a wrong offset in the layout itself. The offsets are
    pinned separately, against real data, in ``test_io_lfs.py``.
    """
    fields = dict(
        format="LFSG", format_ver=1.0, header_id="fmt ", header_size=498,
        tx_name="synthtx", tx_latitude=35.0, tx_longitude=34.0,
        rx_name="synthrx", rx_latitude=56.38, rx_longitude=47.53,
        start_year=2026, start_daynumber=35, start_month=2, start_day=4,
        start_hour=0, start_minute=0, start_second=10,
        start_epoch=1770163210, chirpt=10, rep=300,
        whiten=1, whiten_len=8192, whiten_n=30000,
        **INSTRUMENT,
    )
    fields.update(overrides)

    buffer = bytearray(HEADER_SIZE)
    for name, offset, fmt in _LAYOUT:
        value = fields[name]
        if fmt.endswith("s"):
            value = str(value).encode("ascii")
        struct.pack_into("<" + fmt, buffer, offset, value)
    return bytes(buffer)


def snapped_range(echo_range_km: float, half_span_km: float, window: int) -> float:
    """The range a synthetic echo actually lands on.

    A tone only sits exactly on a bin at integer offsets, so the injected range
    is snapped to the nearest bin and tests compare against that rather than
    against the requested value.
    """
    step = 2 * half_span_km / window
    return round(echo_range_km / step) * step


def synth_iq(
    n_freq: int,
    window: int,
    echo_range_km: float,
    half_span_km: float,
    echo_last_bin: int,
    echo_first_bin: int = 0,
    amplitude: float = 60.0,
    noise: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """IQ whose spectrogram holds one echo at a known range and cutoff.

    A delay shows up as a beat tone. On the fftshifted axis the bin for virtual
    range ``r`` is ``window/2 - r/step``, so a tone at ``exp(-2*pi*i*m*t/window)``
    with ``m = r/step`` lands exactly there -- the same arithmetic
    ``calibrate.build`` inverts.
    """
    rng = np.random.default_rng(seed)
    step = 2 * half_span_km / window
    m = round(echo_range_km / step)     # integer: land the tone on a bin centre

    total = n_freq * window
    iq = (rng.normal(0, noise, total) + 1j * rng.normal(0, noise, total))
    iq = iq.astype(np.complex64)

    t = np.arange(window)
    tone = np.exp(-2j * np.pi * m * t / window).astype(np.complex64)
    for i in range(echo_first_bin, min(echo_last_bin + 1, n_freq)):
        iq[i * window:(i + 1) * window] += (amplitude * tone).astype(np.complex64)

    return iq


#: Speed of light in m/s as chirpsounder2 uses it (`scipy.constants.c`), for
#: building the range axis the way `calc_ionograms.py` does.
C_M_S = 299_792_458.0

#: Virtual range of the cyprus1 echo, `signal-chain.md` sec. 7.3.
ECHO_RANGE_KM = 2710.0

#: A `t0` whose fractional part encodes that range as propagation time, which
#: is what v2's detector produces: transmitters start on an integer second, so
#: everything past the second is one-way travel time. 2710 km is 9.04 ms.
ECHO_T0 = 1770163210.0 + ECHO_RANGE_KM * 1e3 / C_M_S


def chirp_range_offset_km(t0: float) -> float:
    """The term `plot_ionograms.py` adds when `range_offset_applied` is False."""
    return (t0 - np.floor(t0)) * C_M_S / 1e3


def chirp_stored_mask(fftlen: int, sr: float, rate: float,
                      max_range_extent_km: float = 2000.0) -> np.ndarray:
    """The `ridx` of `calc_ionograms.py`'s default branch.

    ``ridx = n.where(n.abs(range_gates) < conf.max_range_extent)`` with
    ``max_range_extent = 2000e3``. Note it is applied to the *raw* gates, so the
    window is +/-2000 km around the direct-path delay, not around 0 km absolute
    -- which is why it comfortably contains a 2710 km echo.
    """
    return np.abs(chirp_range_gates_m(fftlen, sr, rate)) < max_range_extent_km * 1e3


def chirp_range_gates_m(fftlen: int, sr: float, rate: float) -> np.ndarray:
    """The range axis ``calc_ionograms.py`` builds, in metres and *ascending*.

    ``range_gates = ds * fftshift(fftfreq(fftlen, d=1/sr_dec))`` with ``ds``
    converting beat frequency to group path, ``c/rate``. Ascending and signed --
    the opposite of this pipeline's descending axis, which is the flip
    ``io_chirp.load`` has to get right.
    """
    return (C_M_S / rate) * np.fft.fftshift(np.fft.fftfreq(fftlen, d=1.0 / sr))


def v2_snr(power: np.ndarray, storage_threshold: float = 2.0) -> np.ndarray:
    """Put a power spectrogram through ``calc_ionograms.py``'s exact transform.

    Reproduced line for line, because the whole point of the ``.h5`` reader is
    that it inverts *this* and not something like it::

        noise_floor[i] = n.median(SNR[i,:])
        SNR[i,:] = (SNR[i,:] - noise_floor[i]) / noise_floor[i]
        SNR[SNR < conf.storage_snr_threshold] = n.nan
        SNR = SNR.astype(n.float16)

    The median is over the *full* row, before range gating -- the ``S0`` line
    that would have gated first is commented out upstream. That is what makes
    the conversion exact, so a fixture that gated first would hide a real bug.
    """
    snr = np.array(power, dtype=np.float32)
    noise_floor = np.zeros(snr.shape[0])
    for i in range(snr.shape[0]):
        noise_floor[i] = np.median(snr[i, :])
        snr[i, :] = (snr[i, :] - noise_floor[i]) / noise_floor[i]
    snr[snr < storage_threshold] = np.nan
    return snr.astype(np.float16), noise_floor


@pytest.fixture
def make_chirp_z_h5(tmp_path):
    """Factory writing a v2 product that carries its dechirped voltage.

    Unlike ``make_chirp_h5``, which starts from a chosen power array, this one
    starts from a *waveform* and derives the stored ``SNR`` from it using v2's
    own spectrogram. That ordering is the point: it makes
    ``io_chirp.reprocess`` at the original window a round-trip that must
    return the stored array, which is the only way to check the port of
    ``calc_ionograms.spectrogram`` without a real ``save_raw_voltage``
    product to compare against.
    """
    h5py = pytest.importorskip("h5py")
    from muf import io_chirp

    def _make(*, n_freq: int = 24, window: int = 256, step: int = 64,
              rate: float = 100e3, sr: float = 40_000.0,
              echo_bin: int | None = None, echo_amplitude: float = 30.0,
              t0: float = ECHO_T0, txname: str = "synthtx",
              station_name: str = "synthrx", channel: str = "ch0",
              chirp_id: int = 3, seed: int = 11,
              storage_threshold: float = 2.0,
              name: str | None = None) -> Path:
        rng = np.random.default_rng(seed)
        n = n_freq * step + window + step
        z = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        if echo_bin is not None:
            # A constant beat frequency is a constant range: the simplest
            # thing that lands in one range bin at every frequency.
            beat = (echo_bin - window // 2) / window
            z += (echo_amplitude
                  * np.exp(2j * np.pi * beat * np.arange(n))).astype(np.complex64)

        spec = io_chirp.v2_spectrogram(np.conj(z), window=window, step=step)
        noise_floor = np.median(spec, axis=1)
        snr = (spec - noise_floor[:, None]) / noise_floor[:, None]
        # v2 sparsifies before storing, and that is not reversible: every cell
        # below the threshold becomes NaN and reads back as the row median.
        # A reprocess from `z` recovers them, so the two agree only above it.
        snr = np.where(snr < storage_threshold, np.nan, snr)
        freqs = rate * np.arange(spec.shape[0]) * step / sr
        ranges = (io_chirp.C_M_S / rate) * np.fft.fftshift(
            np.fft.fftfreq(window, d=1.0 / sr))

        path = tmp_path / (name or
                           f"lfm_ionogram-{txname}-{station_name}-{channel}-"
                           f"{chirp_id:03d}-{t0:.2f}.h5")
        with h5py.File(path, "w") as fh:
            fh.attrs["chirpsounder2_version"] = "0.2.0"
            fh["SNR"] = snr.astype(np.float16)
            fh["noise_floor"] = noise_floor
            fh["freqs"] = freqs
            fh["ranges"] = ranges
            fh["rate"] = float(rate)
            fh["sr"] = float(sr)
            fh["t0"] = float(t0)
            fh["id"] = chirp_id
            fh["txname"] = txname
            fh["station_name"] = station_name
            fh["ch"] = channel
            fh["range_offset_applied"] = False
            fh["range_gate_start_m"] = np.nan
            fh["range_gate_stop_m"] = np.nan
            fh[io_chirp.RAW_VOLTAGE_KEY] = z      # save_raw_voltage = true
        return path

    return _make


@pytest.fixture
def make_chirp_h5(tmp_path):
    """Factory writing a synthetic ``lfm_ionogram-*.h5``. Returns the path.

    Takes a raw power spectrogram and applies v2's own normalization, so a test
    can hand the *same* array to this and to ``spectro``'s normalization and
    compare the two pipelines' dB.
    """
    h5py = pytest.importorskip("h5py")

    def _make(power: np.ndarray, *, sr: float = 40_000.0, rate: float = 100_000.0,
              keep: slice | np.ndarray | None = None, fft_step: int | None = None,
              t0: float = ECHO_T0, txname: str = "synthtx",
              station_name: str = "synthrx", channel: str = "ch000",
              chirp_id: int = 7, range_offset_applied: bool = False,
              storage_threshold: float = 2.0, freqs_hz: np.ndarray | None = None,
              max_range_extent_km: float = 2000.0,
              name: str | None = None) -> Path:
        power = np.asarray(power, dtype=np.float64)
        n_freq, fftlen = power.shape

        snr, noise_floor = v2_snr(power, storage_threshold)
        ranges = chirp_range_gates_m(fftlen, sr, rate)
        if keep is None:
            # v2 never stores the whole axis; the default branch keeps the
            # max_range_extent window. A fixture storing all of it would let a
            # reader pass that mishandles the relative-to-absolute conversion.
            keep = chirp_stored_mask(fftlen, sr, rate, max_range_extent_km)
        if freqs_hz is None:
            # `freqs = rate * arange(n) * fft_step / sr_dec`. It starts at 0,
            # not at a band edge: v2's `t0` is the instant the sweep crosses
            # 0 Hz, so elapsed time times rate is absolute RF frequency. The
            # default step gives the 3:1 window overlap v2's own defaults
            # produce (fftlen 60000, fft_step 20000 for cyprus1).
            step = fftlen // 3 if fft_step is None else fft_step
            freqs_hz = rate * np.arange(n_freq) * step / sr

        if keep is not None:
            snr = snr[:, keep]
            ranges = ranges[keep]

        path = tmp_path / (name or
                           f"lfm_ionogram-{txname}-{station_name}-{channel}-"
                           f"{chirp_id:03d}-{t0:.2f}.h5")
        with h5py.File(path, "w") as fh:
            # `chirpsounder_version.tag_hdf5` stamps these on every product.
            fh.attrs["chirpsounder2_version"] = "0.2.0"
            fh.attrs["git_commit"] = "0d2712553063"
            fh.attrs["git_dirty"] = False
            fh.create_dataset("SNR", data=snr, compression="gzip",
                              compression_opts=9, shuffle=True)
            fh["noise_floor"] = noise_floor
            fh["freqs"] = np.asarray(freqs_hz, dtype=np.float64)
            fh["rate"] = rate
            fh["ranges"] = ranges
            fh["range_offset_applied"] = range_offset_applied
            # v2 writes NaN for both unless it gated in serendipitous mode.
            fh["range_gate_start_m"] = (float(ranges.min()) if range_offset_applied
                                        else np.nan)
            fh["range_gate_stop_m"] = (float(ranges.max()) if range_offset_applied
                                       else np.nan)
            fh["t0"] = t0
            fh["id"] = chirp_id
            fh["txname"] = txname
            fh["station_name"] = station_name
            fh["sr"] = float(sr)
            fh["ch"] = channel
        return path

    return _make


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: needs a live internet connection to a third party"
    )


@pytest.fixture
def make_lfs(tmp_path):
    """Factory writing a synthetic ``.lfs`` file. Returns the path."""

    def _make(iq: np.ndarray, name: str = "synth.lfs", **header_overrides) -> Path:
        path = tmp_path / name
        with open(path, "wb") as fh:
            fh.write(make_header_bytes(**header_overrides))
            fh.write(np.asarray(iq, dtype=np.complex64).tobytes())
        return path

    return _make


@pytest.fixture
def real_file() -> Path:
    """One real sounding known to contain a clear echo, or skip."""
    path = REAL_DATA / "cyprus1_20260204_030010.lfs"
    if not path.exists():
        pytest.skip(f"real recording not present: {path}")
    return path


@pytest.fixture
def real_dir() -> Path:
    if not REAL_DATA.is_dir() or not any(REAL_DATA.glob("*.lfs")):
        pytest.skip(f"real recordings not present: {REAL_DATA}")
    return REAL_DATA


@pytest.fixture
def real_chirp_dir() -> Path:
    """A directory of real v2 products, or skip.

    Deliberately the *directory* rather than one file: the v2 tree mixes
    ionogram products with detection files and, at DOB, with digisonde
    products, and keeping them together is what makes the discrimination tests
    meaningful.
    """
    pytest.importorskip("h5py")
    if not REAL_CHIRP_DATA.is_dir() or not any(
            REAL_CHIRP_DATA.glob("lfm_ionogram-*.h5")):
        pytest.skip(f"real v2 products not present: {REAL_CHIRP_DATA}")
    return REAL_CHIRP_DATA


@pytest.fixture
def real_chirp_file(real_chirp_dir) -> Path:
    """One real v2 product. The first by name, so it is stable across runs."""
    return sorted(real_chirp_dir.glob("lfm_ionogram-*.h5"))[0]


@pytest.fixture
def make_detection_h5(tmp_path):
    """Factory writing synthetic ``chirp-*.h5`` / ``par-*.h5`` / ``cdetections-*.h5``.

    A schedule is described the way the Twente chirp list describes one -- a
    cycle, the whole seconds a transmitter starts on, and how far away it is --
    plus a receiver epoch offset. That last argument is the point: the DOB
    fault was a receiver whose clock was out by nearly a second, and files
    written by such a receiver are internally perfect. A fixture that cannot
    produce them cannot test the code that has to survive them.
    """
    h5py = pytest.importorskip("h5py")
    C_M_S = 299_792_458.0
    # Each call gets its own directory. Two transmitters written into one tree
    # is a thing tests want to build deliberately, by passing the same `into`,
    # not something they should get by accident from calling the factory twice.
    calls = [0]

    def _times(cycle_s, transmit_seconds, distance_km, epoch_offset_s,
               cycles, base_epoch):
        tau = distance_km * 1e3 / C_M_S
        out = []
        for c in range(cycles):
            for second in transmit_seconds:
                out.append(base_epoch + c * cycle_s + second + tau + epoch_offset_s)
        return out

    def _make(kind: str = "par", *, rate: float = 100e3,
              cycle_s: float = 300.0, transmit_seconds=(235, 240, 245),
              distance_km: float = 3436.0, epoch_offset_s: float = 0.0,
              cycles: int = 4, base_epoch: float = 1785888000.0,
              snr: float = 60.0, channel: str = "ch0",
              station: str = "TST", jitter_s: float = 0.0,
              seed: int = 0, into: Path | None = None) -> Path:
        rng = np.random.default_rng(seed)
        times = _times(cycle_s, transmit_seconds, distance_km, epoch_offset_s,
                       cycles, base_epoch)
        if jitter_s:
            times = [t + float(rng.normal(0.0, jitter_s)) for t in times]

        if into is None:
            calls[0] += 1
            out = tmp_path / f"{kind}{calls[0]}"
        else:
            out = Path(into)
        out.mkdir(parents=True, exist_ok=True)

        if kind == "cdetections":
            data = np.array([[t, 0.0, 1.5e7, rate, snr] for t in times],
                            dtype=np.float64)
            path = out / f"cdetections-{station}-{int(base_epoch)}.h5"
            with h5py.File(path, "w") as fh:
                fh["data"] = data
            return out

        for t in times:
            if kind == "chirp":
                name = f"chirp-{channel}-{rate / 1e3:.0f}-0-{t:.0f}.h5"
                with h5py.File(out / name, "w") as fh:
                    fh["channel"] = channel
                    fh["chirp_rate"] = float(rate)
                    fh["chirp_time"] = float(t)
                    fh["f0"] = 1.5e7
                    fh["i0"] = 0
                    fh["n_samples"] = 5_000_000
                    fh["sample_rate"] = 25_000_000
                    fh["snr"] = np.float32(snr)
            elif kind == "par":
                name = f"par-{channel}-{t:.4f}.h5"
                with h5py.File(out / name, "w") as fh:
                    fh.attrs["chirpsounder2_version"] = "0.2.0"
                    fh.attrs["git_commit"] = "0d2712553063"
                    fh.attrs["git_dirty"] = True
                    fh["channel"] = channel
                    fh["chirp_rate"] = float(rate)
                    fh["t0"] = float(t)
                    fh["t0s"] = np.array([t, t, t], dtype=np.float64)
                    fh["f0"] = np.array([1.4e7, 1.5e7, 1.6e7], dtype=np.float64)
                    fh["snrs"] = np.array([snr, snr * 0.8, snr * 0.6],
                                          dtype=np.float32)
                    fh["num_detections"] = 3
            else:
                raise ValueError(f"unknown kind {kind!r}")
        return out

    return _make


@pytest.fixture
def make_digisonde_h5(tmp_path):
    """Factory writing a synthetic ``digisonde_ionogram-*.h5``. Returns the path.

    Mirrors ``receive_digisonde.py``'s write path: SNR is ``(P - median)/median``
    per (polarization, frequency) over range, and everything below
    ``snr_threshold`` is stored as NaN rather than as a number.
    """
    h5py = pytest.importorskip("h5py")

    def _make(power=None, *, n_pol=2, n_freq=32, n_range=64,
              freq0=1e6, dfreq=50e3, range_step_m=3e3, offset_us=2000.0,
              t0=1786245496.0, transmitter="Juliusruh", receiver="DOB",
              snr_threshold=2.0, kind="digisonde", drop=(),
              name=None) -> Path:
        if power is None:
            power = np.ones((n_pol, n_freq, n_range))
        power = np.asarray(power, dtype=np.float64)
        n_pol, n_freq, n_range = power.shape

        snr = np.zeros_like(power)
        for j in range(n_pol):
            for i in range(n_freq):
                nf = np.median(power[j, i, :])
                snr[j, i, :] = (power[j, i, :] - nf) / nf
        snr[snr < snr_threshold] = np.nan       # exactly what upstream stores

        path = tmp_path / (name or
                           f"digisonde_ionogram-{transmitter}-{receiver}-{t0:.2f}.h5")
        with h5py.File(path, "w") as fh:
            data = {
                "type": kind,
                "SNR": snr.astype(np.float32),
                "freqs": (freq0 + np.arange(n_freq) * dfreq).astype(np.float32),
                "ranges": (np.arange(n_range) * range_step_m).astype(np.float32),
                "noise_floor": np.ones((n_pol, n_freq), dtype=np.float32),
                "transmitter": transmitter,
                "receiver": receiver,
                "offset_us": offset_us,
                "t0": t0,
            }
            for key, value in data.items():
                if key not in drop:
                    fh[key] = value
        return path

    return _make


# --------------------------------------------------------------------------
# The api under test
#
# One copy. This fixture existed four times over -- test_api, test_archives'
# mount tests, test_prediction_api and test_web_handlers -- byte for byte
# identical, and only the first copy carried the comments saying why each
# background reader has to be off.
# --------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """A `TestClient` over the api, with nothing reading the world behind it.

    **The imports are inside the body deliberately.** This conftest is loaded
    for the whole suite, the pure-`muf` tests included, and fastapi is not a
    dependency of this package -- importing it at module scope would turn a
    pipeline-only install from "the api tests skip themselves" into "nothing
    collects at all". A fixture body runs only when a test asks for it.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from services.api import auth, db, main, net
    from services.api import series as series_mod

    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")

    # The startup warm-up reads an archive in a background thread and writes
    # to the census cache, which is module state every test shares -- left on,
    # its result lands in whichever test happens to be running when it
    # finishes. The test that owns it turns it back on deliberately.
    monkeypatch.setattr(main, "WARM_CENSUS", False)

    # Same hazard, worse: the reachability checker makes real HEAD requests to
    # three third-party hosts. A unit suite that reaches the internet is slow,
    # fails on a train, and quietly tests somebody else's uptime.
    monkeypatch.setattr(net, "ENABLED", False)
    net.reset()

    # Third of the same: the series page runs IRI, and IRI wants a solar
    # driver it may have to fetch. It is off by default here so that seeding a
    # sounding with real coordinates -- which is otherwise the most natural
    # thing to do -- cannot silently put a download in the middle of a test.
    # The tests that own the model turn it back on deliberately.
    monkeypatch.setattr(series_mod, "MODEL", False)
    series_mod.clear()

    with TestClient(main.app) as c:
        yield c


# --------------------------------------------------------------------------
# Forecasting artifacts
#
# `ALIAS` and `LAG` are module-level rather than fixtures because both
# prediction test modules assert against them directly. `from conftest import
# ALIAS, LAG` resolves because pytest's default `prepend` import mode puts
# `tests/` on `sys.path`.
#
# The `artifact` fixtures themselves stay where they are: they differ in the
# filename they write, and test_prediction_infer reads its metrics back out of
# that name.
# --------------------------------------------------------------------------

ALIAS = "MUF(3000)F2"
LAG = 288


def feature_names() -> list[str]:
    """The column names a legacy forecasting artifact was fitted on."""
    names = [f"{ALIAS}_lag_{LAG}"]
    names += [f"{ALIAS}_{c}_lag_{LAG}" for c in ("trend", "seasonal", "residual")]
    names += [f"{ALIAS}_rolling_{w}_{s}_lag_{LAG}"
              for w in (12, 48) for s in ("mean", "std")]
    names += ["hour", "minute"]
    return names
