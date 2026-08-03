"""The shared rule that turns a detected trace into a MUF value.

Every method inherited from the two source projects ends the same way: take the
right-most bright thing. That is one noise spike away from being wrong, and each
script grew its own partial defence -- ``ionogr_clustering_0.03.py`` required a
run of 2 consecutive columns, ``ionogr_clustering_and_MUF_0.01.py`` walked
neighbours with a 0.003 MHz step, ``kmeans_clustering.py`` had none at all.

This module generalises those into one function that every extractor calls, so
a change to the decision rule applies uniformly and the methods stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 5 bins x 20.5 kHz ~ 100 kHz of continuous trace. Long enough to reject
# isolated interference, short enough not to clip a genuine MUF edge, where the
# trace fades over a few bins.
DEFAULT_MIN_RUN = 5


@dataclass(frozen=True)
class MufPick:
    """Outcome of one MUF decision."""

    muf_mhz: float          # NaN when nothing qualified
    vrange_km: float        # NaN when nothing qualified
    n_detections: int       # frequency bins with trace presence
    run_len: int            # length of the run the pick sits in
    snr_db: float           # peak power at the picked bin, dB over the noise floor
    freq_index: int         # -1 when nothing qualified

    @property
    def ok(self) -> bool:
        return self.freq_index >= 0


NO_PICK = MufPick(np.nan, np.nan, 0, 0, np.nan, -1)


def find_runs(presence: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Inclusive ``(start, stop)`` spans of at least ``min_run`` consecutive True."""
    if presence.size == 0:
        return []
    padded = np.concatenate(([False], presence.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]        # stops are exclusive here
    return [
        (int(a), int(b - 1))
        for a, b in zip(starts, stops)
        if b - a >= min_run
    ]


def _parabolic_offset(y_prev: float, y_peak: float, y_next: float) -> float:
    """Sub-sample peak offset in bins, from three log-domain samples.

    Standard three-point parabolic interpolation. Returns 0.0 when the samples
    do not describe a peak.
    """
    denom = y_prev - 2.0 * y_peak + y_next
    if denom >= 0 or not np.isfinite(denom):
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -0.5, 0.5))


def _echo_range(
    row_db: np.ndarray, vrange: np.ndarray, parabolic: bool
) -> tuple[float, float]:
    """Virtual range and peak level of the strongest echo in one frequency bin."""
    j = int(np.argmax(row_db))
    peak = float(row_db[j])
    if not parabolic or j == 0 or j == len(row_db) - 1 or len(vrange) < 2:
        return float(vrange[j]), peak

    offset = _parabolic_offset(float(row_db[j - 1]), peak, float(row_db[j + 1]))
    step = float(vrange[1] - vrange[0])   # negative: the axis descends
    return float(vrange[j]) + offset * step, peak


def pick_muf(
    presence: np.ndarray,
    freq: np.ndarray,
    power_db: np.ndarray | None = None,
    vrange: np.ndarray | None = None,
    min_run: int = DEFAULT_MIN_RUN,
    percentile: float = 100.0,
    parabolic: bool = True,
) -> MufPick:
    """Pick the MUF from per-frequency trace presence.

    Args:
        presence: bool, one entry per frequency bin -- trace detected there.
        freq: frequency axis in MHz, ascending, same length as ``presence``.
        power_db: optional ``[n_freq, n_range]`` array used to locate the echo's
            virtual range and report its level.
        vrange: virtual-range axis in km, matching ``power_db``'s columns.
        min_run: consecutive detected bins required to accept a run. Set to 1 to
            reproduce the historical "right-most bright thing" behaviour.
        percentile: 100 takes the highest qualifying bin; lower values trim the
            top of the distribution, which helps when a few bins of interference
            survive the run test.
        parabolic: interpolate the echo's range between bins. This recovers the
            sub-bin precision that zero-padding was previously used for, at no
            cost -- see ``calibrate.range_resolution_km``.

    Returns:
        A ``MufPick``; ``NO_PICK`` when no run qualifies.
    """
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != freq.shape:
        raise ValueError(f"presence {presence.shape} does not match freq {freq.shape}")

    n_detections = int(presence.sum())
    runs = find_runs(presence, min_run)
    if not runs:
        return MufPick(np.nan, np.nan, n_detections, 0, np.nan, -1)

    qualifying = np.concatenate([np.arange(a, b + 1) for a, b in runs])

    if percentile >= 100.0:
        index = int(qualifying.max())
    else:
        index = int(round(float(np.percentile(qualifying, percentile))))
        index = int(qualifying[np.abs(qualifying - index).argmin()])

    run_len = next((b - a + 1 for a, b in runs if a <= index <= b), 0)

    vrange_km, snr_db = np.nan, np.nan
    if power_db is not None and vrange is not None and len(vrange):
        vrange_km, snr_db = _echo_range(power_db[index], vrange, parabolic)

    return MufPick(
        muf_mhz=float(freq[index]),
        vrange_km=vrange_km,
        n_detections=n_detections,
        run_len=int(run_len),
        snr_db=snr_db,
        freq_index=index,
    )
