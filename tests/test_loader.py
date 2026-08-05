"""Format dispatch (``architecture.md`` sec. 3.2).

The point of this module is that ``run``, ``plot``, ``export`` and ``lof`` stop
caring which instrument produced a sounding. So the tests that matter are the
ones about the seams: which reader gets picked, what happens to flags that only
one format honours, and whether the two can share a cache directory without
one silently answering for the other.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from muf import loader, spectro
from muf.loader import CHIRP2, LFS, FormatError


def synth_iq(n: int = 8192 * 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_format_follows_the_extension(tmp_path):
    assert loader.format_of(tmp_path / "a.lfs") == LFS
    assert loader.format_of(tmp_path / "lfm_ionogram-x-y-ch0-000-1.00.h5") == CHIRP2
    assert loader.format_of(tmp_path / "A.LFS") == LFS      # case-insensitive


def test_override_beats_the_extension(tmp_path):
    """For a recording that was renamed, which is the only reason to have this."""
    assert loader.format_of(tmp_path / "a.lfs", CHIRP2) == CHIRP2
    assert loader.format_of(tmp_path / "a.h5", LFS) == LFS


def test_an_unknown_extension_names_what_it_expected(tmp_path):
    with pytest.raises(FormatError, match="no reader for .dat"):
        loader.format_of(tmp_path / "a.dat")
    with pytest.raises(FormatError, match="no extension"):
        loader.format_of(tmp_path / "recording")


def test_a_misspelled_override_fails_here_not_in_a_reader(tmp_path):
    with pytest.raises(FormatError, match="unknown format 'chirp'"):
        loader.format_of(tmp_path / "a.lfs", "chirp")
    with pytest.raises(FormatError, match="unknown format"):
        loader.find_soundings(tmp_path, format="chirp")


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def test_finds_both_formats_in_one_tree(make_lfs, make_chirp_h5, tmp_path):
    """A parallel run holds both by design (architecture.md sec. 2.4)."""
    make_lfs(synth_iq(), name="a.lfs")
    make_chirp_h5(np.full((4, 64), 100.0))

    found = loader.find_soundings(tmp_path)
    assert loader.describe_formats(found) == {LFS: 1, CHIRP2: 1}


def test_format_narrows_the_search(make_lfs, make_chirp_h5, tmp_path):
    make_lfs(synth_iq(), name="a.lfs")
    make_chirp_h5(np.full((4, 64), 100.0))

    assert loader.describe_formats(
        loader.find_soundings(tmp_path, format=LFS)) == {LFS: 1}
    assert loader.describe_formats(
        loader.find_soundings(tmp_path, format=CHIRP2)) == {CHIRP2: 1}


def test_detection_files_are_not_soundings(make_chirp_h5, make_detection_h5,
                                           tmp_path):
    """The v2 tree mixes ionograms with par/chirp/cdetections files.

    `io_detect` reads those; `find_soundings` must not hand one to `io_chirp`,
    which would fail on a schema that has no SNR array at all.
    """
    make_chirp_h5(np.full((4, 64), 100.0))
    make_detection_h5("par", cycles=3, into=tmp_path)
    make_detection_h5("chirp", cycles=3, into=tmp_path)

    found = loader.find_soundings(tmp_path)
    assert [p.name.split("-")[0] for p in found] == ["lfm_ionogram"]


def test_an_empty_tree_says_what_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="lfs or chirp2"):
        loader.find_soundings(tmp_path)
    with pytest.raises(FileNotFoundError, match="no chirp2 soundings"):
        loader.find_soundings(tmp_path, format=CHIRP2)


# --------------------------------------------------------------------------
# Headers
# --------------------------------------------------------------------------

def test_read_header_dispatches(make_lfs, make_chirp_h5, tmp_path):
    """Two header types, deliberately not a common base class.

    Note ``.format`` does not distinguish them the way it looks like it should:
    ``LfsHeader.format`` carries the file magic (``LFSG``) while
    ``ChirpHeader.format`` carries a format name (``chirp2``). Dispatch is by
    type, never by that field.
    """
    from muf.io_chirp import ChirpHeader
    from muf.io_lfs import LfsHeader

    lfs = make_lfs(synth_iq(), name="a.lfs")
    h5 = make_chirp_h5(np.full((4, 64), 100.0))

    assert isinstance(loader.read_header(lfs), LfsHeader)
    assert isinstance(loader.read_header(h5), ChirpHeader)
    # both answer what every consumer downstream actually asks
    for header in (loader.read_header(lfs), loader.read_header(h5)):
        assert header.tx_name and header.rx_name
        assert header.path_type in ("oblique", "vertical")
        assert header.datetime.tzinfo is not None


def test_a_registry_reaches_the_chirp_header(make_chirp_h5):
    """The keyword has to survive dispatch, or v2 geometry is never available."""
    path = make_chirp_h5(np.full((4, 64), 100.0), txname="AAA",
                         station_name="BBB")
    bare = loader.read_header(path)
    with_registry = loader.read_header(
        path, stations={"AAA": (10.0, 20.0), "BBB": (30.0, 40.0)})

    assert not bare.has_coordinates
    assert with_registry.has_coordinates
    assert with_registry.tx_latitude == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_loads_both_formats_into_the_same_type(make_lfs, make_chirp_h5):
    lfs = loader.load(make_lfs(synth_iq(), name="a.lfs"))
    h5 = loader.load(make_chirp_h5(np.full((4, 64), 100.0)))

    assert isinstance(lfs, spectro.Ionogram) and isinstance(h5, spectro.Ionogram)
    for ion in (lfs, h5):
        assert ion.power.shape == (ion.freq.size, ion.vrange.size)
        assert np.all(np.diff(ion.vrange) < 0)     # both descend


def test_window_flags_warn_for_chirp2_and_are_ignored(make_chirp_h5):
    """The silent no-op is what this exists to prevent.

    v2 fixed the window when it wrote the product and the raw IQ is gone, so
    `--window 4096` cannot be honoured. A run that quietly ignores it produces
    a table indistinguishable from one where it was applied.
    """
    path = make_chirp_h5(np.full((4, 64), 100.0))
    with pytest.warns(UserWarning, match="window=4096.*ignored|ignored.*window"):
        windowed = loader.load(path, window=4096)
    with pytest.warns(UserWarning, match="zero_periods"):
        loader.load(path, zero_periods=7)

    plain = loader.load(path)
    np.testing.assert_array_equal(windowed.power, plain.power)


def test_default_window_flags_do_not_warn(make_chirp_h5):
    """Only a value the caller actually chose is worth a warning."""
    path = make_chirp_h5(np.full((4, 64), 100.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loader.load(path, spectro.DEFAULT_WINDOW, spectro.DEFAULT_ZERO_PERIODS)


def test_window_flags_are_honoured_for_lfs(make_lfs):
    a = loader.load(make_lfs(synth_iq(), name="a.lfs"), window=2048)
    b = loader.load(make_lfs(synth_iq(), name="b.lfs"), window=4096)
    assert a.window == 2048 and b.window == 4096
    assert a.power.shape != b.power.shape


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_lfs_cache_keys_keep_their_historical_shape(tmp_path):
    """Existing caches must stay valid; only the .h5 key is new."""
    path = tmp_path / "a.lfs"
    assert (loader.cache_key(path, 8192, 4, None)
            == spectro.cache_key(path, 8192, 4, None))


def test_chirp2_cache_key_omits_the_meaningless_flags(tmp_path):
    path = tmp_path / "lfm_ionogram-x-y-ch0-000-1.00.h5"
    key = loader.cache_key(path, 8192, 4, None)
    assert CHIRP2 in key
    assert "_w8192" not in key and "_z4" not in key


def test_the_two_formats_cannot_collide_in_one_cache_dir(tmp_path):
    """Same stem, different instrument -- the tag is what separates them."""
    lfs = tmp_path / "sounding.lfs"
    h5 = tmp_path / "sounding.h5"
    assert loader.cache_key(lfs, 8192, 4, None) != loader.cache_key(h5, 8192, 4, None)


def test_chirp2_is_not_cached(make_chirp_h5, tmp_path):
    """No FFT to skip, and `spectro.load_cached` would rebuild the wrong axes.

    It restores a `Calibration` via `calibrate.build`, from header arithmetic;
    `io_chirp` builds one from the file's own `freqs` and `ranges` datasets.
    Round-tripping a v2 product through the cache would return a different
    axis and give no sign of it.
    """
    cache = tmp_path / "cache"
    path = make_chirp_h5(np.full((4, 64), 100.0))
    first = loader.load(path, cache_dir=cache)
    second = loader.load(path, cache_dir=cache)

    assert not cache.exists() or not list(cache.glob("*.npz"))
    np.testing.assert_array_equal(first.power, second.power)
    np.testing.assert_array_equal(first.vrange, second.vrange)


def test_lfs_still_caches(make_lfs, tmp_path):
    cache = tmp_path / "cache"
    path = make_lfs(synth_iq(), name="a.lfs")
    loader.load(path, cache_dir=cache)
    assert list(cache.glob("*.npz")), "the .lfs cache must be untouched by dispatch"


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

def test_real_v2_tree_is_discovered_and_loadable(real_chirp_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = loader.find_soundings(real_chirp_dir)
        ion = loader.load(found[0])

    assert loader.describe_formats(found) == {CHIRP2: len(found)}
    assert all(p.name.startswith("lfm_ionogram-") for p in found)
    assert ion.power.shape == (ion.freq.size, ion.vrange.size)
    assert np.isfinite(ion.db).all()
