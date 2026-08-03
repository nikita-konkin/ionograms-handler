"""Mapping spectrogram bin indices to physical frequency and virtual range.

This module is the single place the ionogram axes are defined.

The range axis is **descending**: bin 0 is the largest virtual range. This was
established empirically rather than assumed. For
``data/2026.02.04/cyprus1_20260204_000010.lfs`` the echo sits at raw fftshifted
bin 3909 of 8192; under the ascending convention ``linspace(-R, +R)`` used for
the y-axis in ``MUF.py:116`` that maps to -2732 km, which is not a physically
possible virtual range, while ``R - idx*step`` maps it to +2739 km -- correct
for a transmitter-receiver great-circle distance of ~2600 km.

``MUF.py`` gets a correct-looking *picture* only because it reverses the array
with ``[::-1]`` when plotting; its *extractor* path carries the sign error, and
``MUF.py:297``'s ``if vrng < 0: vrng = R + vrng`` is a patch over it. Defining
the axis once, correctly, removes the need for that patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .io_lfs import LfsHeader

C_KM_S = 3e8 / 1e3          # speed of light in km/s
EARTH_RADIUS_KM = 6371.0

# An oblique echo cannot arrive earlier than the ground path, but the header's
# own rmin is often 0. When the great-circle distance is unavailable we fall
# back to this floor rather than searching ranges no echo can occupy.
FALLBACK_MIN_RANGE_KM = 1500.0

# Fraction of the great-circle distance used as the low edge of the default
# gate. Slightly below 1.0 to leave room for calibration error in the geometry.
GROUND_RANGE_MARGIN = 0.90


def ground_range_km(header: LfsHeader) -> float | None:
    """Great-circle transmitter-receiver distance, or None if unusable.

    Returns None for vertical sounding and for headers whose coordinates are
    absent or implausible.
    """
    if not header.is_oblique:
        return None
    lat1, lon1 = header.tx_latitude, header.tx_longitude
    lat2, lon2 = header.rx_latitude, header.rx_longitude
    coords = (lat1, lon1, lat2, lon2)
    if any(c is None or not math.isfinite(c) for c in coords):
        return None
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return None
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        return None
    if lat1 == lat2 and lon1 == lon2:
        return None

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def sweep_bounds(header: LfsHeader) -> tuple[float, float]:
    """Start and stop frequency of the chirp sweep, in MHz."""
    freq_start = (header.cf - header.sample_rate / 2) / 1e6
    freq_stop = freq_start + (header.dur * header.rate) / 1e6
    return freq_start, freq_stop


def range_half_span(header: LfsHeader) -> float:
    """Half-width of the virtual-range axis, in km.

    The de-chirped beat frequency maps to delay through the chirp rate, so the
    post-decimation bandwidth ``sample_rate/dec`` spans +/- this many km.
    """
    sr = header.sample_rate / header.dec
    return C_KM_S * (sr / header.div_coef) / header.rate


def range_resolution_km(header: LfsHeader, window: int) -> float:
    """True delay resolution, in km, for a given FFT window length.

    A window of ``window`` samples at ``sr`` Hz lasts ``window/sr`` seconds and
    so resolves beat frequency to ``sr/window`` Hz; through the chirp rate that
    is ``C_KM_S * (sr/window) / rate`` km. That equals the *unpadded* bin
    spacing, which is the point: zero-padding subdivides bins without resolving
    anything further, so the true resolution is always
    ``range_step * (1 + zero_periods)``.

    For window=8192 at sr=40 kHz this is 14.65 km.
    """
    return 2 * range_half_span(header) / window


def default_gate(header: LfsHeader) -> tuple[float, float]:
    """Physically sensible virtual-range gate, in km.

    Upper edge from the header's ``rmax``; lower edge from the ground path when
    the geometry allows, otherwise from ``rmin`` with a floor.
    """
    hi = float(header.rmax) if header.rmax > 0 else 5000.0

    ground = ground_range_km(header)
    if ground is not None:
        lo = ground * GROUND_RANGE_MARGIN
    else:
        lo = max(float(header.rmin), FALLBACK_MIN_RANGE_KM)

    lo = max(lo, float(header.rmin))
    if lo >= hi:  # nonsense header; fall back to a wide but finite window
        lo, hi = FALLBACK_MIN_RANGE_KM, max(hi, 5000.0)
    return lo, hi


def gate_indices(
    half_span: float, step: float, lo_km: float, hi_km: float, n_range: int
) -> tuple[int, int]:
    """Inclusive index range on the descending axis covering ``lo_km..hi_km``.

    Because the axis descends, the *higher* range maps to the *lower* index.
    """
    axis_max = half_span
    axis_min = half_span - (n_range - 1) * step
    # Test for intersection before clamping: clamping first would silently turn
    # a gate that misses the axis entirely into a single valid bin.
    if hi_km < axis_min or lo_km > axis_max:
        raise ValueError(
            f"empty range gate {lo_km:.0f}..{hi_km:.0f} km "
            f"(axis spans {axis_min:.0f}..{axis_max:.0f} km)"
        )

    i_lo = max(0, min(int(math.ceil((half_span - hi_km) / step)), n_range - 1))
    i_hi = max(0, min(int(math.floor((half_span - lo_km) / step)), n_range - 1))
    if i_hi < i_lo:
        raise ValueError(
            f"empty range gate {lo_km:.0f}..{hi_km:.0f} km "
            f"(narrower than one {step:.1f} km bin)"
        )
    return i_lo, i_hi


def sweep_step_mhz(header: LfsHeader, window: int) -> float:
    """Frequency advance per FFT window, in MHz.

    One window spans ``window/sr`` seconds, during which the transmitter sweeps
    at ``rate`` Hz/s. This -- not the nominal endpoint -- is what fixes the
    frequency axis, and it stays correct when a recording is cut short.
    """
    sr = header.sample_rate / header.dec
    return (window / sr) * header.rate / 1e6


@dataclass(frozen=True)
class Calibration:
    """Physical axes for one ionogram."""

    freq: np.ndarray          # MHz, ascending, one per FFT window
    vrange: np.ndarray        # km, DESCENDING, gated slice
    freq_start: float
    freq_stop: float          # highest frequency actually swept
    freq_stop_nominal: float  # what the header's `dur` implies
    half_span: float          # km, half-width of the ungated range axis
    range_step: float         # km per range bin
    resolution_km: float      # true delay resolution (>= range_step)
    gate_km: tuple[float, float]
    gate_idx: tuple[int, int]  # inclusive, into the ungated axis
    n_range_full: int

    @property
    def freq_step_mhz(self) -> float:
        if len(self.freq) < 2:
            return 0.0
        return float(self.freq[1] - self.freq[0])

    @property
    def sweep_complete(self) -> bool:
        """False when the recording stopped before the sweep finished.

        Truncated files still carry a header claiming the full sweep, so this
        is the only way to notice. A MUF from an incomplete sweep is capped by
        where the recording stopped, not by the ionosphere.
        """
        return self.freq_stop >= self.freq_stop_nominal - self.freq_step_mhz

    @property
    def sweep_fraction(self) -> float:
        """How much of the intended sweep the recording actually contains."""
        span = self.freq_stop_nominal - self.freq_start
        return 1.0 if span <= 0 else (self.freq_stop - self.freq_start) / span

    @property
    def n_freq(self) -> int:
        return len(self.freq)

    @property
    def n_range(self) -> int:
        return len(self.vrange)


def build(
    header: LfsHeader,
    n_freq: int,
    window: int,
    zero_periods: int = 0,
    gate_km: tuple[float, float] | None = None,
) -> Calibration:
    """Construct the axes for a spectrogram of ``n_freq`` windows.

    ``window`` and ``zero_periods`` fix the ungated range-bin count; ``gate_km``
    overrides the header-derived default gate.
    """
    n_range_full = window * (1 + zero_periods)
    freq_start, freq_stop_nominal = sweep_bounds(header)
    half_span = range_half_span(header)
    step = 2 * half_span / n_range_full

    lo_km, hi_km = gate_km if gate_km is not None else default_gate(header)
    i_lo, i_hi = gate_indices(half_span, step, lo_km, hi_km, n_range_full)

    # From the chirp rate rather than by stretching to the nominal endpoint: a
    # recording cut short still declares the full sweep in its header, and
    # stretching would place its last window at 32.5 MHz when the transmitter
    # had only reached 14.6. Windows are labelled at their centre, since the
    # spectrum is an average over the window's duration.
    freq_step = sweep_step_mhz(header, window)
    freq = freq_start + (np.arange(n_freq) + 0.5) * freq_step

    return Calibration(
        freq=freq,
        vrange=half_span - np.arange(i_lo, i_hi + 1) * step,
        freq_start=freq_start,
        freq_stop=float(freq_start + n_freq * freq_step),
        freq_stop_nominal=freq_stop_nominal,
        half_span=half_span,
        range_step=step,
        resolution_km=range_resolution_km(header, window),
        gate_km=(lo_km, hi_km),
        gate_idx=(i_lo, i_hi),
        n_range_full=n_range_full,
    )
