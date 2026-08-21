"""How the ``.h5`` readers touch the filesystem, which on this station is a network.

The archive lives on an SMB mount. HDF5's own I/O is seek-heavy -- it walks the
b-tree and fetches each chunk where it lies -- so reading one product costs
~184 read syscalls. Locally that is free; over SMB every one is a round trip,
and a few thousand soundings became an hour of latency and then a dropped
mount. :func:`muf.io_chirp.open_h5` buffers the file instead.

These tests pin the property that fix depends on -- **h5py is handed bytes, not
a path** -- rather than the syscall count, which is real but not portable to a
CI runner. The size cap and the growing-file case are pinned too, because both
are the kind of thing a later reader would simplify away.
"""

from __future__ import annotations

import errno
import io
from pathlib import Path

import numpy as np
import pytest

from muf import io_chirp

h5py = pytest.importorskip("h5py")


@pytest.fixture
def product(make_chirp_h5) -> Path:
    rng = np.random.default_rng(0)
    return make_chirp_h5(rng.gamma(2.0, 1.0, size=(48, 512)))


def opened_with(monkeypatch) -> list:
    """Record what each ``h5py.File`` call was handed."""
    seen: list = []
    real = h5py.File

    def spy(name, *args, **kwargs):
        seen.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", spy)
    return seen


def test_h5py_is_handed_bytes_not_a_path(product, monkeypatch):
    """The whole point: HDF5 must not do its own seeking over the network."""
    seen = opened_with(monkeypatch)
    with io_chirp.open_h5(product) as fh:
        assert set(io_chirp.REQUIRED) <= set(fh.keys())

    assert len(seen) == 1
    assert isinstance(seen[0], io.BytesIO), (
        f"h5py was handed {type(seen[0]).__name__}; a path means HDF5 reads the "
        f"file in ~184 pieces, which is the SMB stall this exists to prevent"
    )


def test_the_buffered_read_returns_the_same_arrays(product, monkeypatch):
    with io_chirp.open_h5(product) as fh:
        buffered = {k: np.asarray(fh[k][()]) for k in fh.keys()}
    with h5py.File(product, "r") as fh:                   # HDF5's own path I/O
        direct = {k: np.asarray(fh[k][()]) for k in fh.keys()}

    assert buffered.keys() == direct.keys()
    for key in direct:
        # `equal_nan` because v2 stores NaN for every cell it dropped below the
        # storage threshold, and those must compare equal to count as identical.
        kinds = {buffered[key].dtype.kind, direct[key].dtype.kind}
        same = np.array_equal(buffered[key], direct[key],
                              equal_nan=kinds <= {"f", "c"})
        assert same, key


def test_an_oversized_file_is_streamed_instead_of_buffered(product, monkeypatch):
    """A file that is not a product must not be pulled into memory whole.

    With ``ARCHIVE_SCAN_JOBS`` workers each holding one, an unexpected raw
    capture would otherwise be multiplied by the worker count.
    """
    monkeypatch.setattr(io_chirp, "BUFFER_LIMIT_BYTES", 0)
    seen = opened_with(monkeypatch)
    with io_chirp.open_h5(product) as fh:
        assert "SNR" in fh

    assert len(seen) == 1
    assert not isinstance(seen[0], io.BytesIO)
    assert Path(seen[0]) == product


def test_a_file_that_grew_since_the_stat_is_not_truncated(product, monkeypatch):
    """The size comes from a stat taken before the read, so the read has to
    cope with the file being longer by the time it happens -- a product still
    being written. Truncating it would hand h5py a corrupt image."""
    full = product.read_bytes()
    real_stat = Path.stat

    def short_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self == product:
            class Shrunk:
                st_size = len(full) - 4096
            return Shrunk()
        return result

    monkeypatch.setattr(Path, "stat", short_stat)
    with io_chirp.open_h5(product) as fh:
        assert np.asarray(fh["SNR"][()]).size > 0


def test_a_dead_mount_surfaces_as_the_real_oserror(product, monkeypatch):
    """Not swallowed and not retried in pieces. A file that cannot be read
    whole cannot be read in pieces either, and the errno is what tells the
    archive page to say 'not readable' rather than 500."""
    def refuse(*args, **kwargs):
        raise OSError(errno.EIO, "Input/output error", str(product))

    monkeypatch.setattr(io_chirp, "open", refuse, raising=False)
    with pytest.raises(OSError) as caught:
        with io_chirp.open_h5(product):
            pass
    assert caught.value.errno == errno.EIO


def test_every_reader_goes_through_the_buffered_open():
    """A new ``h5py.File(path)`` added later would silently reintroduce the
    stall without failing anything else, so the readers are linted for one.

    The two calls inside :func:`muf.io_chirp.open_h5` are the implementation
    and are exempt by their indentation -- they sit inside the function, and
    every real call site sits at a function's top level.
    """
    root = Path(__file__).parent.parent / "muf"
    offenders = []
    for name in ("io_chirp", "io_digisonde", "io_detect"):
        for lineno, line in enumerate(
                (root / f"{name}.py").read_text().splitlines(), 1):
            if not line.strip().startswith("with h5py.File("):
                continue
            inside_open_h5 = name == "io_chirp" and (
                line.startswith(" " * 8) or "buffer" in line)
            if not inside_open_h5:
                offenders.append(f"muf/{name}.py:{lineno}: {line.strip()}")

    assert not offenders, (
        "these read an .h5 by path, so HDF5 fetches it in ~184 pieces -- over "
        "the station's SMB mount that is ~184 round trips a file. Use "
        "`open_h5(path)`:\n  " + "\n  ".join(offenders))
