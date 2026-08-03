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
