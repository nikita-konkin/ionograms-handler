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


# --- the registry correction --------------------------------------------------
#
# An .lfs header always carries lat/lon, so unlike the v2 path this is a
# correction rather than a lookup: what has to be provable is which of the two
# sources a given number came from, since a wrong coordinate does not raise --
# it yields a plausible path length and a wrong virtual height.

NICOSIA = (35.18557, 33.38228)      # `NIC` in server.ini
YOSHKAR_OLA = (56.38, 47.53)


def test_the_registry_supersedes_the_header(make_lfs):
    path = make_lfs(np.zeros(16, dtype=np.complex64), tx_name="cyprus1")
    header = read_header(path, {"cyprus1": NICOSIA})

    assert header.tx_latitude == pytest.approx(35.18557, abs=1e-5)
    assert header.tx_longitude == pytest.approx(33.38228, abs=1e-5)
    assert header.from_registry == ("tx",)
    # The receiver is not in this table, so it keeps the file's own numbers.
    assert header.rx_latitude == pytest.approx(56.38, abs=1e-4)


def test_without_a_registry_the_file_is_read_verbatim(make_lfs):
    """`None` is no registry here, as in every reader in the package."""
    path = make_lfs(np.zeros(16, dtype=np.complex64), tx_name="cyprus1")
    header = read_header(path)

    assert (header.tx_latitude, header.tx_longitude) == (35.0, 34.0)
    assert header.from_registry == ()


def test_an_empty_registry_is_not_the_default_one(make_lfs):
    path = make_lfs(np.zeros(16, dtype=np.complex64), tx_name="cyprus1")
    header = read_header(path, {})

    assert (header.tx_latitude, header.tx_longitude) == (35.0, 34.0)
    assert header.from_registry == ()


def test_provenance_names_both_ends_when_both_are_known(make_lfs):
    path = make_lfs(np.zeros(16, dtype=np.complex64),
                    tx_name="cyprus1", rx_name="yoshkar-ola")
    header = read_header(path, {"cyprus1": NICOSIA, "yoshkar-ola": YOSHKAR_OLA})

    assert header.from_registry == ("tx", "rx")


def test_provenance_is_recorded_even_when_the_numbers_agree(make_lfs):
    """"The table said so" stays true on the day the table is corrected."""
    path = make_lfs(np.zeros(16, dtype=np.complex64), tx_name="cyprus1")
    header = read_header(path, {"cyprus1": (35.0, 34.0)})

    assert (header.tx_latitude, header.tx_longitude) == (35.0, 34.0)
    assert header.from_registry == ("tx",)


def test_the_default_table_moves_cyprus1_to_nicosia(real_file):
    """One site cannot have two positions depending on who logged it.

    The archive's ``cyprus1`` and v2's ``NIC`` are the same Nicosia
    transmitter, recorded 59.9 km apart. `loader.read_header` resolves the
    default table, so the path length changes from the 2588.4 km the header
    implies to the 2587.8 km the registry does.
    """
    from muf import loader
    from muf.geometry import path_of

    header = loader.read_header(real_file)

    assert header.tx_name == "cyprus1"
    assert header.tx_latitude == pytest.approx(35.18557, abs=1e-5)
    assert header.tx_longitude == pytest.approx(33.38228, abs=1e-5)
    assert path_of(header)[2] == pytest.approx(2587.8, abs=0.5)

    # Both ends, and the receiver's numbers do not move: `yoshkar-ola` was
    # transcribed into the table *from* an .lfs header. Provenance still says
    # the table, because that is where the value in play came from.
    assert header.from_registry == ("tx", "rx")
    assert header.rx_latitude == pytest.approx(56.38, abs=0.01)


def test_the_gated_tile_carries_the_corrected_geometry(make_lfs):
    """`spectro` threads the registry through to the header it hands back."""
    from muf import spectro

    path = make_lfs(np.zeros(8192 * 2, dtype=np.complex64), tx_name="cyprus1")
    ion = spectro.compute(path, window=8192, stations={"cyprus1": NICOSIA})

    assert ion.header.from_registry == ("tx",)
    assert ion.header.tx_latitude == pytest.approx(35.18557, abs=1e-5)


def test_a_cached_tile_picks_up_a_later_correction(make_lfs, tmp_path):
    """The cache holds the tile, not the geometry.

    The header is re-read from the source on every load, so correcting the
    table reaches soundings already on disk without invalidating them.
    """
    from muf import spectro

    path = make_lfs(np.zeros(8192 * 2, dtype=np.complex64), tx_name="cyprus1")
    cache = tmp_path / "cache"

    first = spectro.compute_cached(path, window=8192, cache_dir=cache)
    assert first.header.from_registry == ()
    assert first.header.tx_latitude == pytest.approx(35.0)

    second = spectro.compute_cached(path, window=8192, cache_dir=cache,
                                    stations={"cyprus1": NICOSIA})
    assert second.header.from_registry == ("tx",)
    assert second.header.tx_latitude == pytest.approx(35.18557, abs=1e-5)
    # Same tile, not a recompute: the correction did not cost the FFTs.
    np.testing.assert_array_equal(second.power, first.power)


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
