"""Tracking MUF through the day as a continuous quantity.

Each sounding is currently scaled in isolation, which throws away the strongest
constraint available: the ionosphere does not change much in five minutes. A
MUF that jumps 10 MHz between consecutive soundings and back again is an
extraction failure, not physics -- but a per-sounding estimator has no way to
know that.

This module runs a constant-velocity Kalman filter over the day, with a
Rauch-Tung-Striebel backward pass so every estimate uses the whole day rather
than only the past. It buys three things a per-sounding estimator cannot:

* **gap filling** -- soundings where every estimator declined get an estimate
  from their neighbours, with honest uncertainty attached;
* **outlier rejection** -- a measurement too far from the prediction, measured
  in its own standard deviations, is not allowed to drag the state;
* **uncertainty** -- every point comes with a standard deviation, so downstream
  work can weight or threshold on it.

The state is ``[MUF, dMUF/dt]``. Process noise is set by how fast the MUF can
physically change; measurement noise comes from each pick's own quality, so a
long well-resolved trace counts for more than a marginal one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: How fast the MUF can change, in MHz per hour of random walk in the rate.
#: Sunrise and sunset are the fastest real transitions -- this instrument sees
#: about 8 MHz/hour there -- so the process noise must admit at least that
#: without letting noise through.
DEFAULT_PROCESS_NOISE_MHZ_PER_HOUR = 4.0

#: Measurement standard deviation for a good pick, in MHz. Roughly the spread
#: between independent estimators on clean soundings.
DEFAULT_BASE_SIGMA_MHZ = 0.35

#: Reject a measurement further than this many predicted standard deviations
#: from the prediction. 4 is loose enough to follow real sunrise transitions.
DEFAULT_GATE_SIGMA = 4.0

#: Below this many consecutive frequency bins a pick is treated as marginal and
#: its uncertainty inflated rather than being trusted at face value.
MARGINAL_RUN = 8


@dataclass
class Track:
    """A tracked MUF series."""

    frame: pd.DataFrame           # datetime, muf, sigma, measured, rejected
    n_measured: int
    n_filled: int
    n_rejected: int

    def __str__(self) -> str:
        return (f"Track({len(self.frame)} points: {self.n_measured} measured, "
                f"{self.n_filled} filled, {self.n_rejected} rejected)")


def measurement_sigma(
    frame: pd.DataFrame,
    method: str,
    base_sigma: float = DEFAULT_BASE_SIGMA_MHZ,
) -> np.ndarray:
    """Per-sounding measurement standard deviation, in MHz.

    A pick backed by a long continuous trace at good signal-to-noise deserves
    more weight than a marginal one. Both signals are already recorded per
    sounding, so the weighting costs nothing to compute.
    """
    n = len(frame)
    sigma = np.full(n, base_sigma)

    run = pd.to_numeric(frame.get(f"run_{method}"), errors="coerce") \
        if f"run_{method}" in frame else pd.Series(np.nan, index=frame.index)
    marginal = run.to_numpy() < MARGINAL_RUN
    sigma[np.nan_to_num(marginal, nan=False).astype(bool)] *= 3.0

    snr = pd.to_numeric(frame.get(f"snr_{method}"), errors="coerce") \
        if f"snr_{method}" in frame else pd.Series(np.nan, index=frame.index)
    if snr.notna().any():
        # Scale gently with signal level: a 40 dB weaker echo is ~2x less certain.
        reference = float(snr.median())          # skips NaN without warning
        if np.isfinite(reference):
            deficit = np.clip(reference - snr.to_numpy(), 0.0, 40.0)
            sigma *= 1.0 + np.nan_to_num(deficit, nan=0.0) / 40.0

    return sigma


def track(
    times,
    values,
    sigma,
    process_noise: float = DEFAULT_PROCESS_NOISE_MHZ_PER_HOUR,
    gate_sigma: float = DEFAULT_GATE_SIGMA,
) -> Track:
    """Kalman filter plus RTS smoother over an irregularly sampled series.

    Args:
        times: timestamps, ascending.
        values: measured MUF; NaN where no estimator produced a pick.
        sigma: per-measurement standard deviation, MHz.
        process_noise: random-walk rate on dMUF/dt, MHz per hour.
        gate_sigma: reject measurements beyond this many predicted sigma.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(times)))
    order = np.argsort(index.to_numpy())
    index = index[order]
    values = np.asarray(values, dtype=float)[order]
    sigma = np.asarray(sigma, dtype=float)[order]

    n = len(index)
    if n == 0:
        raise ValueError("no timestamps to track")
    if not np.isfinite(values).any():
        raise ValueError("no measurements to track")

    hours = np.diff(index.to_numpy().astype("datetime64[s]").astype(float)) / 3600.0
    hours = np.concatenate([[0.0], np.maximum(hours, 1e-6)])

    # --- forward pass ---
    state = np.array([np.nanmedian(values), 0.0])
    covariance = np.diag([25.0, 25.0])          # deliberately vague to start

    predicted_states = np.zeros((n, 2))
    predicted_covs = np.zeros((n, 2, 2))
    filtered_states = np.zeros((n, 2))
    filtered_covs = np.zeros((n, 2, 2))
    rejected = np.zeros(n, dtype=bool)

    observation = np.array([[1.0, 0.0]])

    for i in range(n):
        dt = hours[i]
        transition = np.array([[1.0, dt], [0.0, 1.0]])

        # Continuous white-noise acceleration.
        q = process_noise ** 2
        noise = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                              [dt ** 2 / 2.0, dt]])

        state = transition @ state
        covariance = transition @ covariance @ transition.T + noise
        predicted_states[i], predicted_covs[i] = state, covariance

        measurement = values[i]
        if np.isfinite(measurement):
            r = max(float(sigma[i]), 1e-3) ** 2
            innovation = measurement - (observation @ state).item()
            innovation_var = (observation @ covariance @ observation.T).item() + r

            if abs(innovation) > gate_sigma * np.sqrt(innovation_var):
                rejected[i] = True          # keep the prediction, ignore the point
            else:
                gain = covariance @ observation.T / innovation_var
                state = state + (gain * innovation).ravel()
                covariance = (np.eye(2) - gain @ observation) @ covariance

        filtered_states[i], filtered_covs[i] = state, covariance

    # --- RTS backward pass ---
    smoothed_states = filtered_states.copy()
    smoothed_covs = filtered_covs.copy()
    for i in range(n - 2, -1, -1):
        dt = hours[i + 1]
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        try:
            gain = filtered_covs[i] @ transition.T @ np.linalg.inv(predicted_covs[i + 1])
        except np.linalg.LinAlgError:        # singular: leave the filtered value
            continue
        smoothed_states[i] += gain @ (smoothed_states[i + 1] - predicted_states[i + 1])
        smoothed_covs[i] += gain @ (smoothed_covs[i + 1] - predicted_covs[i + 1]) @ gain.T

    measured = np.isfinite(values) & ~rejected
    frame = pd.DataFrame({
        "datetime": index,
        "muf": np.round(smoothed_states[:, 0], 3),
        "rate_mhz_per_hour": np.round(smoothed_states[:, 1], 3),
        "sigma": np.round(np.sqrt(np.maximum(smoothed_covs[:, 0, 0], 0.0)), 3),
        "measured": measured,
        "rejected": rejected,
    })

    return Track(
        frame=frame,
        n_measured=int(measured.sum()),
        n_filled=int((~np.isfinite(values)).sum()),
        n_rejected=int(rejected.sum()),
    )


def track_results(
    frame: pd.DataFrame,
    method: str | None = None,
    drop_limited: bool = True,
    base_sigma: float = DEFAULT_BASE_SIGMA_MHZ,
    process_noise: float = DEFAULT_PROCESS_NOISE_MHZ_PER_HOUR,
    gate_sigma: float = DEFAULT_GATE_SIGMA,
) -> Track:
    """Track the MUF in a results table from :mod:`muf.pipeline`.

    Band-limited picks are excluded by default: they are lower bounds, and
    letting them anchor the state would pull the midday peak down.
    """
    from .compare import flag, usable
    from .pipeline import _first_method

    method = method or _first_method(frame)
    frame = frame.sort_values("datetime").reset_index(drop=True)

    values = usable(frame, method, drop_limited=drop_limited).to_numpy(dtype=float)
    sigma = measurement_sigma(frame, method, base_sigma)

    if drop_limited:
        # Not measurements, so not tracked -- but their absence is a gap the
        # filter should fill rather than a hole in the output.
        values = np.where(flag(frame, f"limited_{method}").to_numpy(), np.nan, values)

    result = track(frame["datetime"], values, sigma,
                   process_noise=process_noise, gate_sigma=gate_sigma)
    result.frame.insert(1, "method", method)
    return result
