"""Forming the range-gated ionogram from IQ samples.

Ports ``stuffr.py``'s ``hanning`` / ``spectrogram`` / ``medians`` /
``comprz_dB`` with two changes:

* **float32** rather than float64.
* **the range gate is applied inside the FFT loop**, so the full
  ``[n_freq, window*(1+zero_periods)]`` array is never held in memory. For the
  2026-02-04 data that is 1.0 MB instead of 40 MB (or 880 MB at ``-z 10``), and
  about 0.2 s per sounding.

The row median used for noise-floor equalization is still taken over the *full*
spectrum before gating, so gating does not bias it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import calibrate
from .calibrate import Calibration
from .io_lfs import LfsHeader, read_header, read_iq

# stuffr.medians() scales the median by 4*ln(2) to convert the median of an
# exponentially-distributed power spectrum into its mean.
NOISE_COEF = 4 * math.log(2)

DEFAULT_WINDOW = 8192
DEFAULT_ZERO_PERIODS = 0


def hanning(length: int) -> np.ndarray:
    """Hann window, matching ``stuffr.hanning``."""
    n = np.arange(length, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / length))


#: Where the equalized noise floor lands on the dB scale ``to_db`` produces.
#:
#: ``to_db`` divides by 1e-3 before taking the log, and median equalization puts
#: the noise at ~1.0 in linear power, so noise reads **30 dB, not 0**. Every dB
#: value in an ``Ionogram`` therefore sits 30 dB above the true power ratio: the
#: 43 dB detection threshold the estimators share is a 13 dB signal-to-noise,
#: which is what the historical linear threshold of 20 always meant.
#:
#: Anything that takes a *true* signal-to-noise from a caller has to add this
#: before comparing against ``Ionogram.db``. See ``muf.lof.luf_at_snr``.
NOISE_FLOOR_DB = 30.0


def to_db(power: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Log-compress median-equalized power.

    ``stuffr.comprz_dB`` clamps at the 5th percentile of non-zero values and
    divides by 1e-3 before taking the log. The percentile clamp is order
    statistics over the whole array and is only cosmetic once the noise floor
    is already equalized to ~1, so this keeps the ``/1e-3`` scaling (which sets
    where 0 dB falls, and therefore what the historical thresholds of 20 and 45
    dB mean) and clamps at a fixed floor instead.
    """
    return 10.0 * np.log10(np.maximum(power, floor) / floor)


@dataclass
class Ionogram:
    """One sounding: a gated power array plus the axes that describe it.

    ``power`` is median-equalized linear power -- the units the historical
    algorithmic threshold of 20 is expressed in. ``db`` is the log-compressed
    view used by the image-derived extractors. Rows are frequency, columns are
    virtual range (descending).
    """

    power: np.ndarray          # float32 [n_freq, n_range], noise floor ~1.0
    cal: Calibration
    header: LfsHeader
    window: int
    zero_periods: int

    _db: np.ndarray | None = None

    @property
    def db(self) -> np.ndarray:
        if self._db is None:
            self._db = to_db(self.power).astype(np.float32)
        return self._db

    @property
    def freq(self) -> np.ndarray:
        return self.cal.freq

    @property
    def vrange(self) -> np.ndarray:
        return self.cal.vrange

    @property
    def shape(self) -> tuple[int, int]:
        return self.power.shape

    def __str__(self) -> str:
        return (
            f"Ionogram({self.header}, {self.shape[0]}x{self.shape[1]}, "
            f"{self.cal.gate_km[0]:.0f}-{self.cal.gate_km[1]:.0f} km)"
        )


def compute(
    path: str | Path,
    window: int = DEFAULT_WINDOW,
    zero_periods: int = DEFAULT_ZERO_PERIODS,
    gate_km: tuple[float, float] | None = None,
    header: LfsHeader | None = None,
) -> Ionogram:
    """Read an ``.lfs`` file and form its range-gated ionogram."""
    path = Path(path)
    header = header or read_header(path)
    iq = read_iq(path)

    n_freq = len(iq) // window
    if n_freq == 0:
        raise ValueError(f"{path}: {len(iq)} samples is shorter than one {window}-sample window")

    cal = calibrate.build(
        header, n_freq=n_freq, window=window,
        zero_periods=zero_periods, gate_km=gate_km,
    )
    i_lo, i_hi = cal.gate_idx
    n_range_full = cal.n_range_full

    win = hanning(window)
    padded = np.zeros(n_range_full, dtype=np.complex128)
    out = np.empty((n_freq, i_hi - i_lo + 1), dtype=np.float32)

    for i in range(n_freq):
        padded[:window] = win * iq[i * window:(i + 1) * window]
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(padded))) ** 2
        # Noise floor from the full spectrum, before gating, so that restricting
        # the range window does not bias the estimate.
        floor = NOISE_COEF * np.median(spectrum)
        row = spectrum[i_lo:i_hi + 1]
        out[i] = row / floor if floor > 0 else row

    return Ionogram(
        power=out, cal=cal, header=header,
        window=window, zero_periods=zero_periods,
    )


# --- caching -----------------------------------------------------------------
#
# The FFTs dominate the cost and every extractor consumes the same array, so a
# cached tile makes extractor tuning essentially free to re-run.

def cache_key(path: Path, window: int, zero_periods: int,
              gate_km: tuple[float, float] | None) -> str:
    gate = "auto" if gate_km is None else f"{gate_km[0]:.0f}-{gate_km[1]:.0f}"
    return f"{path.stem}_w{window}_z{zero_periods}_g{gate}"


def load_cached(cache_dir: Path, key: str) -> Ionogram | None:
    """Return a cached ionogram, or None if absent or unreadable."""
    npz_path = cache_dir / f"{key}.npz"
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as blob:
            source = Path(str(blob["source"]))
            header = read_header(source)
            cal = calibrate.build(
                header,
                n_freq=int(blob["n_freq"]),
                window=int(blob["window"]),
                zero_periods=int(blob["zero_periods"]),
                gate_km=(float(blob["gate_lo"]), float(blob["gate_hi"])),
            )
            return Ionogram(
                power=blob["power"], cal=cal, header=header,
                window=int(blob["window"]), zero_periods=int(blob["zero_periods"]),
            )
    except Exception:
        # A corrupt or stale cache entry should cost a recompute, not a crash.
        return None


def save_cached(cache_dir: Path, key: str, ion: Ionogram) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{key}.npz",
        power=ion.power,
        source=str(ion.header.path),
        n_freq=ion.shape[0],
        window=ion.window,
        zero_periods=ion.zero_periods,
        gate_lo=ion.cal.gate_km[0],
        gate_hi=ion.cal.gate_km[1],
    )


def compute_cached(
    path: str | Path,
    window: int = DEFAULT_WINDOW,
    zero_periods: int = DEFAULT_ZERO_PERIODS,
    gate_km: tuple[float, float] | None = None,
    cache_dir: str | Path | None = None,
) -> Ionogram:
    """``compute`` with an optional on-disk cache of the gated tile."""
    path = Path(path)
    if cache_dir is None:
        return compute(path, window, zero_periods, gate_km)

    cache_dir = Path(cache_dir)
    key = cache_key(path, window, zero_periods, gate_km)
    cached = load_cached(cache_dir, key)
    if cached is not None:
        return cached

    ion = compute(path, window, zero_periods, gate_km)
    save_cached(cache_dir, key, ion)
    return ion
