"""Reading ``.lfs`` chirp-ionosonde files.

An ``.lfs`` file is a 512-byte header followed by interleaved complex64 IQ
samples. This module replaces the 30 sequential ``seek``/``unpack`` calls in
``lfs_header.py`` with a single fixed-layout unpack, and returns scalars rather
than the 1-tuples that forced ``header['cf'][0]`` at every call site.
"""

from __future__ import annotations

import datetime as _dt
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_SIZE = 512

# Field layout, little-endian, offsets in bytes from the start of the file.
# `s` fields are fixed-width NUL-padded ASCII.
#
# The one deviation from lfs_header.py is rx_longitude: that file reads it at
# offset 150, which is rx_latitude's offset, so header['rx_longitude'] has
# always held the latitude. rx_name occupies 86..149 and rx_latitude 150..153,
# which puts the longitude at 154.
_LAYOUT = (
    ("format", 0, "4s"),
    ("format_ver", 4, "f"),
    ("header_id", 8, "4s"),
    ("header_size", 12, "H"),
    ("tx_name", 14, "64s"),
    ("tx_latitude", 78, "f"),
    ("tx_longitude", 82, "f"),
    ("rx_name", 86, "64s"),
    ("rx_latitude", 150, "f"),
    ("rx_longitude", 154, "f"),
    ("start_year", 158, "H"),
    ("start_daynumber", 160, "H"),
    ("start_month", 162, "H"),
    ("start_day", 164, "H"),
    ("start_hour", 166, "H"),
    ("start_minute", 168, "H"),
    ("start_second", 170, "H"),
    ("start_epoch", 172, "I"),
    ("chirpt", 176, "I"),
    ("cf", 180, "I"),
    ("dur", 184, "H"),
    ("rate", 186, "I"),
    ("rep", 190, "I"),
    ("rmin", 194, "i"),
    ("rmax", 198, "i"),
    ("dec", 202, "I"),
    ("sample_rate", 206, "I"),
    ("whiten", 210, "H"),
    ("whiten_len", 212, "I"),
    ("whiten_n", 216, "I"),
)


def _clean(raw: bytes) -> str:
    """Decode a fixed-width NUL-padded ASCII field."""
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


@dataclass(frozen=True)
class LfsHeader:
    """Acquisition parameters for one sounding."""

    path: Path

    format: str
    format_ver: float
    header_id: str
    header_size: int

    tx_name: str
    tx_latitude: float
    tx_longitude: float
    rx_name: str
    rx_latitude: float
    rx_longitude: float

    start_year: int
    start_daynumber: int
    start_month: int
    start_day: int
    start_hour: int
    start_minute: int
    start_second: int
    start_epoch: int

    chirpt: int
    cf: int          # centre frequency, Hz
    dur: int         # sweep duration, s
    rate: int        # chirp rate, Hz/s
    rep: int
    rmin: int        # intended range gate, km
    rmax: int        # intended range gate, km
    dec: int         # decimation
    sample_rate: int  # Hz, before decimation
    whiten: int
    whiten_len: int
    whiten_n: int

    @property
    def datetime(self) -> _dt.datetime:
        return _dt.datetime(
            self.start_year, self.start_month, self.start_day,
            self.start_hour, self.start_minute, self.start_second,
            tzinfo=_dt.timezone.utc,
        )

    @property
    def is_oblique(self) -> bool:
        """True when transmitter and receiver are different sites."""
        return self.tx_name != self.rx_name

    @property
    def path_type(self) -> str:
        return "oblique" if self.is_oblique else "vertical"

    @property
    def div_coef(self) -> float:
        """Range-scaling divisor: 2 for oblique paths, 4 for vertical sounding.

        A vertical sounding's echo travels up and back over the same path, so
        the round-trip factor differs from the oblique case.
        """
        return 2.0 if self.is_oblique else 4.0

    def __str__(self) -> str:
        return (
            f"{self.tx_name}->{self.rx_name} ({self.path_type}) "
            f"{self.datetime:%Y-%m-%d %H:%M:%S}Z"
        )


def read_header(path: str | Path) -> LfsHeader:
    """Parse the 512-byte header of an ``.lfs`` file."""
    path = Path(path)
    with open(path, "rb") as fh:
        blob = fh.read(HEADER_SIZE)
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"{path}: truncated, {len(blob)} bytes < {HEADER_SIZE}")

    values: dict[str, object] = {"path": path}
    for name, offset, fmt in _LAYOUT:
        (raw,) = struct.unpack_from("<" + fmt, blob, offset)
        values[name] = _clean(raw) if fmt.endswith("s") else raw

    header = LfsHeader(**values)  # type: ignore[arg-type]
    if header.format != "LFSG":
        raise ValueError(f"{path}: not an LFS file (magic {header.format!r})")
    return header


def read_iq(path: str | Path, max_samples: int | None = None) -> np.ndarray:
    """Read the complex64 IQ payload that follows the header."""
    count = -1 if max_samples is None else int(max_samples)
    return np.fromfile(path, dtype=np.complex64, offset=HEADER_SIZE, count=count)


def n_samples(path: str | Path) -> int:
    """Number of complex64 samples in the payload, without reading it."""
    return (Path(path).stat().st_size - HEADER_SIZE) // 8


def find_lfs(target) -> list[Path]:
    """Return the ``.lfs`` files under ``target``, sorted by name.

    ``target`` may be a file, a directory (searched recursively), or any
    iterable of those -- so several days' folders can be processed in one go.
    Sorting by name sorts by time, since filenames embed the timestamp.
    Duplicates are removed, which matters when a directory and a file inside it
    are both named.
    """
    if isinstance(target, (str, Path)):
        targets = [Path(target)]
    else:
        targets = [Path(t) for t in target]
    if not targets:
        raise FileNotFoundError("no target given")

    found: list[Path] = []
    missing: list[Path] = []
    for item in targets:
        if item.is_file():
            found.append(item)
        elif item.is_dir():
            found.extend(item.rglob("*.lfs"))
        else:
            missing.append(item)

    if missing and not found:
        raise FileNotFoundError(", ".join(str(m) for m in missing))

    unique = {p.resolve(): p for p in found}
    return sorted(unique.values(), key=lambda p: (p.name, str(p)))
