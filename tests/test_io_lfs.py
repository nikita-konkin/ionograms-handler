"""Header parsing, pinned against a real recording."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from muf.io_lfs import find_lfs, n_samples, read_header, read_iq


def test_reads_synthetic_header(make_lfs):
    path = make_lfs(np.zeros(16, dtype=np.complex64))
    header = read_header(path)

    assert header.format == "LFSG"
    assert header.tx_name == "synthtx"
    assert header.rx_name == "synthrx"
    assert header.cf == 20_000_000
    assert header.sample_rate == 25_000_000
    assert header.dec == 625


def test_strings_are_not_nul_padded(make_lfs):
    """Names are fixed-width fields; the padding must not leak into the value."""
    path = make_lfs(np.zeros(16, dtype=np.complex64), tx_name="cyprus1")
    assert read_header(path).tx_name == "cyprus1"


def test_datetime_and_path_type(make_lfs):
    path = make_lfs(np.zeros(16, dtype=np.complex64))
    header = read_header(path)

    assert header.datetime == dt.datetime(2026, 2, 4, 0, 0, 10, tzinfo=dt.timezone.utc)
    assert header.is_oblique
    assert header.path_type == "oblique"
    assert header.div_coef == 2.0


def test_same_tx_and_rx_is_vertical(make_lfs):
    path = make_lfs(np.zeros(16, dtype=np.complex64), tx_name="same", rx_name="same")
    header = read_header(path)

    assert not header.is_oblique
    assert header.path_type == "vertical"
    assert header.div_coef == 4.0


def test_iq_round_trip(make_lfs):
    iq = (np.arange(64) + 1j * np.arange(64)).astype(np.complex64)
    path = make_lfs(iq)

    np.testing.assert_allclose(read_iq(path), iq)
    assert n_samples(path) == 64


def test_rejects_non_lfs(tmp_path):
    path = tmp_path / "bogus.lfs"
    path.write_bytes(b"\x00" * 512)
    with pytest.raises(ValueError, match="not an LFS file"):
        read_header(path)


def test_rejects_truncated(tmp_path):
    path = tmp_path / "short.lfs"
    path.write_bytes(b"LFSG" + b"\x00" * 100)
    with pytest.raises(ValueError, match="truncated"):
        read_header(path)


def test_find_lfs_sorts(tmp_path):
    for name in ("c.lfs", "a.lfs", "b.lfs"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_lfs(tmp_path)] == ["a.lfs", "b.lfs", "c.lfs"]


# --- pinned against the real instrument --------------------------------------
#
# These assert the byte offsets themselves, which the synthetic fixture cannot:
# it writes through the same layout it reads back.

def test_real_header_matches_instrument(real_file):
    header = read_header(real_file)

    assert header.format == "LFSG"
    assert header.tx_name == "cyprus1"
    assert header.rx_name == "yoshkar-ola"
    assert header.cf == 20_000_000
    assert header.sample_rate == 25_000_000
    assert header.dec == 625
    assert header.dur == 250
    assert header.rate == 100_000
    assert (header.rmin, header.rmax) == (0, 5000)
    assert header.datetime == dt.datetime(2026, 2, 4, 3, 0, 10, tzinfo=dt.timezone.utc)


def test_real_receiver_longitude_is_not_the_latitude(real_file):
    """lfs_header.py:108 read rx_longitude at offset 150 -- rx_latitude's offset.

    Yoshkar-Ola is near 56.4N, 47.5E, so the two differ by ~9 degrees; reading
    the wrong field returns the latitude and this fails.
    """
    header = read_header(real_file)

    assert header.rx_latitude == pytest.approx(56.38, abs=0.01)
    assert header.rx_longitude == pytest.approx(47.53, abs=0.01)
    assert header.rx_longitude != pytest.approx(header.rx_latitude, abs=0.5)


def test_real_sample_count(real_file):
    """~10M samples, giving 1220 windows of 8192.

    The exact count varies by a few samples between recordings, so the
    invariant that matters is the window count -- which is the frequency-bin
    count, and the ``ion_col_num = 1220`` hardcoded throughout MUF_clustering.
    """
    count = n_samples(real_file)
    assert count == pytest.approx(10_000_000, abs=1000)
    assert count // 8192 == 1220
