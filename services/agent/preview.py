"""A thumbnail of the newest product, built on the station itself.

The console can call a station HEALTHY without anyone being able to see whether
it is making *usable* pictures. ``newest_product_age_s`` says a product exists;
nothing says what is in it. The archive reaches the server only on a timer, so
the newest sounding on the acquisition laptop's SSD is not on the server yet --
`acquisition.arrivals` deliberately measures the *archive*, and its own note in
the console says so. This module closes that gap by sending a picture small
enough that sending it costs nothing worth measuring.

**Why decoding here is cheap.** A chirpsounder2 v2 ``.h5`` holds the ionogram
already computed -- no FFT, no re-derivation. A normal DOB product is 74 KB
holding a ``(310, 450)`` float16 ``SNR``; decode, decimate and encode measured
**5 ms**. The worst case in the archive is a search-mode product,
``(486, 3999)`` and 2.5 MB on disk, which measured **53 ms**. Once per 60 s push
that is 0.09 % of one core, on a machine whose recorder is the thing that must
not be disturbed.

**Why h5py is imported inside the function.** This package is stdlib-only by
standing invariant (``systemd/chirp-agent.service``) so that a broken analysis
environment can never take the health reporting down with it. h5py and numpy are
chirpsounder2's own dependencies and are present in its ``.venv38`` in practice,
but their absence has to degrade to *no preview*, never to a failed pass. Same
pattern as :func:`health.epoch_offset` and its guarded ``muf`` import.

The arithmetic that turns v2's ``SNR`` into decibels is duplicated from
``muf.io_chirp``/``muf.spectro`` rather than imported, for the same reason the
filename regex in :mod:`health` is: ``muf`` is a server-side package and may not
be installed here at all. Every duplicated constant names its source, and
:mod:`tests.test_agent` pins them against the originals so the two cannot drift.
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

#: ``4 * ln(2)``. ``muf.spectro.NOISE_COEF`` -- the noise normalisation that
#: puts v2's ``SNR`` on the same scale ``spectro.compute`` produces.
NOISE_COEF = 4.0 * math.log(2.0)

#: ``muf.spectro.to_db``'s power floor.
DB_FLOOR = 1e-3

#: ``muf.render.DEFAULT_VMIN_DB`` / ``DEFAULT_VMAX_DB``. Sharing the scale is
#: the point: a bright cell in the thumbnail has to mean what a bright cell
#: means in the full render, or the thumbnail teaches the operator a second,
#: wrong reading of the same data.
VMIN_DB = 20.0
VMAX_DB = 75.0

#: ``muf.io_chirp.SNR_OFFSET_DB``. Zero, and the decode below omits the
#: multiply it would imply -- named here so that a day when it stops being zero
#: fails a test instead of quietly putting the thumbnail on a different scale
#: from the full render.
SNR_OFFSET_DB = 0.0

#: ``muf.io_chirp.C_M_S``.
C_M_S = 299_792_458.0

#: ``muf.io_chirp.UNIDENTIFIED_TX`` -- v2's own spelling of its marker for a
#: sounding whose transmitter nothing identified (``calc_ionograms.py``).
UNIDENTIFIED_TX = "unkown"

#: ``muf.io_chirp.MAX_VIRTUAL_RANGE_KM``. Past this the range offset implied by
#: ``t0`` is not a path length, so the axis is left relative.
MAX_VIRTUAL_RANGE_KM = 22_000.0

#: Widest range span a thumbnail will show before it crops around the echoes.
#: A search-mode product spans nearly 8,000 km while the trace occupies a few
#: hundred, so the uncropped thumbnail is a hairline in a field of noise. DOB's
#: ordinary 2,300-5,000 km gate is 2,700 km and passes through untouched, which
#: matters: for the normal case the thumbnail then frames exactly what
#: ``render.ionogram`` frames. The chosen span is reported with the image, so
#: the crop is never silent.
MAX_SPAN_KM = 3000.0

#: 16 samples of matplotlib's ``jet``, the colormap ``render`` uses. A palette
#: PNG at 4 bits per pixel is the same size as 4-bit greyscale -- 48 bytes of
#: PLTE -- so matching the full render's colours is free.
JET_16 = (
    (  0,   0, 128), (  0,   0, 205), (  0,   8, 255), (  0,  76, 255),
    (  0, 144, 255), (  0, 212, 255), ( 41, 255, 206), ( 96, 255, 151),
    (151, 255,  96), (206, 255,  41), (255, 230,   0), (255, 167,   0),
    (255, 104,   0), (255,  41,   0), (205,   0,   0), (128,   0,   0),
)


#: Most products decoded in one pass. A station handed eleven new circuits at
#: once must not turn a health pass into a second of CPU on the machine running
#: the recorder; the rest wait for the next pass 60 s later, which for a picture
#: of the ionosphere is no wait at all.
MAX_PER_PASS = 4


class PreviewUnavailable(Exception):
    """No preview can be built, and nothing is wrong with the station.

    Distinct from a decode failure: this is "the tools are not here", which is
    a `Metric.unknown`, not a red.
    """


@dataclass
class Preview:
    """One encoded thumbnail and what it is a picture of."""

    tx: str
    t0: float
    png: bytes
    width: int
    height: int
    freq_lo_hz: float
    freq_hi_hz: float
    range_lo_m: float
    range_hi_m: float
    cropped: bool = False


def due(newest: dict, sent: dict, budget: int = MAX_PER_PASS, cursor: int = 0
        ) -> tuple[list, int]:
    """Which transmitters to encode this pass, and where to resume next.

    ``newest`` is :func:`health.scan_products`' ``{tx: (path, t0)}``; ``sent``
    is ``{tx: t0}`` for what the server already has. A transmitter whose newest
    product is one the server has seen is skipped -- an idle circuit costs
    nothing at all, which is what makes running this every 60 s reasonable.

    The walk starts at ``cursor`` and the returned one resumes past everything
    examined, so a station with more new products than ``budget`` works through
    them in order rather than re-picking the same few. Sorting by ``t0``
    instead would let a busy circuit starve a quiet one forever, and the quiet
    one is usually the one worth looking at.
    """
    names = sorted(newest)
    if not names or budget < 1:
        return [], 0
    n = len(names)
    start = cursor % n
    picked, seen = [], 0
    while seen < n and len(picked) < budget:
        tx = names[(start + seen) % n]
        seen += 1
        path, t0 = newest[tx]
        if sent.get(tx) != t0:
            picked.append((tx, path, t0))
    return picked, (start + seen) % n


# --------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------

def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def encode_png(rows: list[bytes], width: int, palette=JET_16) -> bytes:
    """A 4-bit palette PNG, written with nothing but ``zlib`` and ``struct``.

    Twenty-odd lines against a dependency the station does not have: Pillow is
    not in chirpsounder2's environment, and adding one to the acquisition
    machine to make a 4 KB picture would be a poor trade. Each row of ``rows``
    is one palette index per pixel; two are packed per byte, high nibble first,
    and an odd width pads with zero as the format requires.

    Filter type 0 on every row. PNG's filters operate on *bytes*, and a byte
    here holds two unrelated pixels, so the predictors have nothing to predict;
    measured, they cost time and gain under a percent.
    """
    if len(palette) > 16:
        raise ValueError(f"palette has {len(palette)} entries, max 16 at 4 bits")
    raw = bytearray()
    for row in rows:
        if len(row) != width:
            raise ValueError(f"row is {len(row)} px, expected {width}")
        raw.append(0)                                    # filter: None
        for i in range(0, width - 1, 2):
            raw.append(((row[i] & 0xF) << 4) | (row[i + 1] & 0xF))
        if width % 2:
            raw.append((row[-1] & 0xF) << 4)

    plte = bytearray()
    for r, g, b in palette:
        plte += bytes((r, g, b))

    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, len(rows),
                                          4, 3, 0, 0, 0))
            + _chunk(b"PLTE", bytes(plte))
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + _chunk(b"IEND", b""))


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

def _block_max(a, n: int, axis: int):
    """Reduce ``axis`` to ``n`` samples, keeping the strongest cell of each block.

    **Not mean, and not striding.** A trace is one bright cell in a
    neighbourhood of noise; at the ~30x reduction a search-mode product needs,
    averaging dilutes it below the noise and point-sampling misses it outright.
    Keeping the maximum is what makes a 128-px-wide picture still show the
    trace, and it is the whole reason the full read is worth its 53 ms over a
    19 ms strided one.

    ``reduceat`` rather than a reshape so the block size need not divide the
    axis; where ``n`` exceeds the axis length the indices repeat and the axis is
    replicated, which is the sane answer for a product smaller than the
    thumbnail.
    """
    import numpy as np

    idx = (np.arange(n) * a.shape[axis]) // n
    return np.maximum.reduceat(a, idx, axis=axis)


def _absolute_ranges_km(ranges_m, t0: float, offset_applied: bool,
                        range_start_m: float, tx_name: str):
    """v2's stored range axis as virtual range in km. Mirrors
    ``muf.io_chirp._absolute_ranges``, including its arbitrary 0.75.

    A relative axis keeps every range *difference* correct and only loses the
    zero, so the two refusal cases below still make a perfectly readable
    picture -- the labels under it are what become approximate.
    """
    import numpy as np

    ranges_km = np.asarray(ranges_m, dtype=np.float64) / 1e3
    if offset_applied:
        if (math.isfinite(range_start_m) and ranges_km.size
                and float(np.nanmedian(ranges_km)) < 0.75 * range_start_m / 1e3):
            return ranges_km + range_start_m / 1e3
        return ranges_km

    offset_km = (t0 - math.floor(t0)) * C_M_S / 1e3
    if tx_name == UNIDENTIFIED_TX or offset_km > MAX_VIRTUAL_RANGE_KM:
        return ranges_km
    return ranges_km + offset_km


def _scalar(value):
    """h5py hands back 0-d arrays, 1-element arrays and bytes interchangeably."""
    try:
        value = value[()]
    except (TypeError, IndexError):
        pass
    try:
        if getattr(value, "shape", None) == (1,):
            value = value[0]
    except (TypeError, IndexError):                       # pragma: no cover
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def build(path: Path, size: tuple[int, int] = (128, 96)) -> Preview:
    """Decode one ``lfm_ionogram-*.h5`` into a thumbnail.

    ``size`` is ``(width, height)`` = ``(frequencies, ranges)``.

    Raises :class:`PreviewUnavailable` when numpy or h5py is missing, and the
    usual ``OSError``/``ValueError`` when the product itself is unreadable --
    which it routinely is, since the recorder writes products this may open
    mid-write. Callers treat both as "no picture this pass".
    """
    try:
        import h5py
        import numpy as np
    except Exception as exc:                # ImportError, but also a broken .so
        raise PreviewUnavailable(f"{type(exc).__name__}: {exc}") from exc

    width, height = int(size[0]), int(size[1])
    if width < 2 or height < 2:
        raise ValueError(f"preview size {size} is too small to be a picture")

    with h5py.File(path, "r") as fh:
        for key in ("SNR", "freqs", "ranges", "t0"):
            if key not in fh:
                raise ValueError(f"{path.name}: not an ionogram product, no {key!r}")
        snr = np.asarray(fh["SNR"][()])
        freqs_hz = np.asarray(fh["freqs"][()], dtype=np.float64)
        ranges_m = np.asarray(fh["ranges"][()], dtype=np.float64)
        t0 = float(_scalar(fh["t0"]))
        tx = str(_scalar(fh["txname"])) if "txname" in fh else ""
        offset_applied = (bool(_scalar(fh["range_offset_applied"]))
                          if "range_offset_applied" in fh else False)
        range_start_m = (float(_scalar(fh["range_gate_start_m"]))
                         if "range_gate_start_m" in fh else float("nan"))

    if snr.ndim != 2 or snr.shape != (freqs_hz.size, ranges_m.size):
        raise ValueError(f"{path.name}: SNR is {snr.shape} but the axes are "
                         f"{freqs_hz.size} x {ranges_m.size}")
    if snr.size == 0:
        raise ValueError(f"{path.name}: empty ionogram")

    vrange_km = _absolute_ranges_km(ranges_m, t0, offset_applied,
                                    range_start_m, tx)

    # v2's range axis ascends (fftshift of fftfreq); this pipeline's descends,
    # bin 0 holding the largest virtual range, and `render` puts that at the top
    # of the picture. Getting this wrong mirrors the ionogram top-to-bottom and
    # still looks entirely plausible -- which is why it is done here in the same
    # breath as the read, exactly as `io_chirp.load` does it.
    vrange_km = vrange_km[::-1]
    snr = snr[:, ::-1]

    # `snr_to_power` then `to_db`, in one expression because the intermediate
    # would be a second full-size float array. NaN is v2's "below the storage
    # threshold" and becomes SNR 0 -- the row median -- landing at 25.6 dB, well
    # under the 43 dB a detection needs, so sparsification cannot invent a trace.
    power = np.maximum(np.nan_to_num(snr.astype(np.float32), nan=0.0,
                                     posinf=np.float32(65504.0), neginf=-1.0)
                       + 1.0, 0.0) / NOISE_COEF
    db = 10.0 * np.log10(np.maximum(power, DB_FLOOR) / DB_FLOOR)

    lo_i, hi_i, cropped = _crop(db, vrange_km)
    db = db[:, lo_i:hi_i]
    vrange_km = vrange_km[lo_i:hi_i]

    grid = _block_max(_block_max(db, width, axis=0), height, axis=1)

    # Quantise after decimation, so a level boundary can never cost a peak that
    # the maximum was taken specifically to keep.
    level = (grid - VMIN_DB) * (len(JET_16) - 1) / (VMAX_DB - VMIN_DB)
    level = np.clip(np.rint(level), 0, len(JET_16) - 1).astype(np.uint8)

    # [freq, range] -> [row, col] with row 0 the largest virtual range, matching
    # the y axis `render.ionogram` draws.
    rows = [bytes(row) for row in level.T]
    png = encode_png(rows, width)

    return Preview(
        tx=tx, t0=t0, png=png, width=width, height=height,
        freq_lo_hz=float(freqs_hz.min()), freq_hi_hz=float(freqs_hz.max()),
        range_lo_m=float(vrange_km.min()) * 1e3,
        range_hi_m=float(vrange_km.max()) * 1e3,
        cropped=cropped,
    )


def _crop(db, vrange_km) -> tuple[int, int, bool]:
    """Range indices to keep. Everything, unless the extent is absurd.

    When it is, the window is centred on the strongest row rather than on the
    axis: the echo is what the picture is for, and on a search-mode product the
    stored extent is centred on a delay nobody has cross-checked.
    """
    import numpy as np

    n = vrange_km.size
    span = abs(float(vrange_km[0]) - float(vrange_km[-1]))
    if n < 4 or span <= MAX_SPAN_KM:
        return 0, n, False

    keep = max(int(round(n * MAX_SPAN_KM / span)), 2)
    centre = int(np.argmax(db.max(axis=0)))
    lo = min(max(centre - keep // 2, 0), n - keep)
    return lo, lo + keep, True
