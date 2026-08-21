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


#: Per-parameter column names in a `muf.pipeline` results table. The censoring
#: flag differs at each end of the band -- `limited` is a pick at the top of
#: the sweep and a lower bound on MUF, `loflim` one at the floor and an upper
#: bound on LOF -- and so does the signal level each is measured at.
PARAMS = {
    "muf": {"censor": "limited", "snr": "snr"},
    "lof": {"censor": "loflim", "snr": "lofsnr"},
}


@dataclass
class Track:
    """A tracked series. The value column is named after the parameter."""

    frame: pd.DataFrame           # datetime, <param>, sigma, measured, rejected
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
    param: str = "muf",
) -> np.ndarray:
    """Per-sounding measurement standard deviation, in MHz.

    A pick backed by a long continuous trace at good signal-to-noise deserves
    more weight than a marginal one. Both signals are already recorded per
    sounding, so the weighting costs nothing to compute.

    LOF is weighted by ``lofsnr``, which the pipeline measures at the *bottom*
    of the trace. Reusing the MUF's ``snr`` would scale a LOF's uncertainty by
    the signal level at the other end of the band, and the two ends routinely
    differ.
    """
    snr_column = "lofsnr" if param == "lof" else "snr"
    n = len(frame)
    sigma = np.full(n, base_sigma)

    run = pd.to_numeric(frame.get(f"run_{method}"), errors="coerce") \
        if f"run_{method}" in frame else pd.Series(np.nan, index=frame.index)
    marginal = run.to_numpy() < MARGINAL_RUN
    sigma[np.nan_to_num(marginal, nan=False).astype(bool)] *= 3.0

    snr = pd.to_numeric(frame.get(f"{snr_column}_{method}"), errors="coerce") \
        if f"{snr_column}_{method}" in frame else pd.Series(np.nan, index=frame.index)
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
    param: str = "muf",
) -> Track:
    """Kalman filter plus RTS smoother over an irregularly sampled series.

    The filter itself knows nothing about which frequency it is tracking -- a
    random walk in rate is as good a description of the LOF as of the MUF.
    ``param`` only names the output column, so a tracked LOF is never handed
    on in a frame whose column says ``muf``.

    Args:
        times: timestamps, ascending.
        values: measurements; NaN where no estimator produced a pick.
        sigma: per-measurement standard deviation, MHz.
        process_noise: random-walk rate on d(value)/dt, MHz per hour.
        gate_sigma: reject measurements beyond this many predicted sigma.
        param: names the value column of the returned frame.
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
        param: np.round(smoothed_states[:, 0], 3),
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
    param: str = "muf",
) -> Track:
    """Track one parameter in a results table from :mod:`muf.pipeline`.

    Band-edge picks are excluded by default: they are bounds, and letting one
    anchor the state pulls the curve towards the edge it ran into -- the
    midday MUF down towards the top of the sweep, the dawn LOF up towards the
    band floor.
    """
    from .compare import flag, usable
    from .pipeline import _first_method

    if param not in PARAMS:
        raise ValueError(f"unknown parameter {param!r}; expected one of {sorted(PARAMS)}")
    censor = PARAMS[param]["censor"]

    method = method or _first_method(frame)
    frame = frame.sort_values("datetime").reset_index(drop=True)

    values = usable(frame, method, drop_limited=drop_limited,
                    param=param).to_numpy(dtype=float)
    sigma = measurement_sigma(frame, method, base_sigma, param=param)

    if drop_limited:
        # Not measurements, so not tracked -- but their absence is a gap the
        # filter should fill rather than a hole in the output.
        values = np.where(flag(frame, f"{censor}_{method}").to_numpy(), np.nan, values)

    result = track(frame["datetime"], values, sigma,
                   process_noise=process_noise, gate_sigma=gate_sigma,
                   param=param)
    result.frame.insert(1, "method", method)
    return result
