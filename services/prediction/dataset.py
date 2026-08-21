"""Database rows to a regular series a model can be fed.

The single place that knows both the schema and what a model expects, which
makes it the single place a leak could be introduced and the single place to
stop one. Everything it returns carries provenance: which points were measured
and which the tracker filled, how uncertain each is, and which were censored at
a band edge.

**The tracker is the resampler.** Extraction produces values at sounding
instants, irregularly, with gaps; a lagged-feature model needs a regular grid.
The obvious bridge is interpolation, and it is the wrong one -- it invents
points with no error bar and no notion of how fast the ionosphere may actually
move. ``muf.track`` already runs a constant-velocity Kalman filter with an RTS
smoother over exactly this series: it fills gaps with an estimate that carries
its own standard deviation and rejects outliers against a physical rate limit
rather than a distributional one. Feeding it the union of the sounding instants
and the target grid makes resampling and gap-filling the same operation.

**Censored picks are excluded from the fit and kept in the output.** A pick at
the top of the sweep is a lower bound, not a measurement, and letting it anchor
the state pulls the midday peak down -- so it is passed to the tracker as a gap.
It still appears in the returned frame, flagged, because a model with a censored
objective wants it and because a caller that silently dropped it would bias its
own scoring.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from muf import track as track_mod
from ..api import db

#: Grid the forecasting models were built around: five minutes, 288 a day.
DEFAULT_STEP_S = 300

#: Per-parameter column mapping. The censoring flag differs at each end of the
#: band -- `limited` is a pick at the top of the sweep (a lower bound on MUF),
#: `loflim` one at the bottom (an upper bound on LOF).
PARAMS = {
    "muf": {"value": "muf", "censor": "limited"},
    "lof": {"value": "lof", "censor": "loflim"},
}

#: Rows this many samples short of a full decomposition period cannot produce
#: features, so a window shorter than this is refused rather than returned
#: empty.
MIN_SAMPLES = 2 * 288


@dataclass
class Series:
    """A tracked series on a regular grid, with everything a caller may need."""

    frame: pd.DataFrame          # index: datetime; value, sigma, measured, censored
    param: str
    tx: str
    rx: str
    method: str
    n_measured: int
    n_filled: int
    n_rejected: int
    n_censored: int

    @property
    def values(self) -> pd.Series:
        return self.frame["value"]

    def __str__(self) -> str:
        return (f"{self.param} {self.tx}->{self.rx} [{self.method}]: "
                f"{len(self.frame)} points, {self.n_measured} measured, "
                f"{self.n_filled} filled, {self.n_rejected} rejected, "
                f"{self.n_censored} censored")


def observations(conn: sqlite3.Connection, param: str, tx: str, rx: str,
                 method: str, start: str | None = None,
                 end: str | None = None) -> pd.DataFrame:
    """Raw picks for one circuit, parameter and estimator, ascending."""
    if param not in PARAMS:
        raise ValueError(f"unknown parameter {param!r}; expected one of {sorted(PARAMS)}")
    columns = PARAMS[param]

    sql = [
        f"SELECT s.datetime, e.{columns['value']} AS value,",
        f"e.{columns['censor']} AS censored, e.run, e.snr",
        "FROM extraction e JOIN sounding s ON s.id = e.sounding_id",
        "WHERE e.method = ? AND s.tx = ? AND s.rx = ?",
    ]
    params: list = [method, tx, rx]
    if start:
        sql.append("AND s.datetime >= ?")
        params.append(db.time_bound(start))
    if end:
        sql.append("AND s.datetime <= ?")
        params.append(db.time_bound(end, end=True))
    sql.append("ORDER BY s.datetime")

    frame = pd.DataFrame(db.rows(conn, " ".join(sql), tuple(params)))
    if frame.empty:
        return pd.DataFrame(columns=["datetime", "value", "censored", "run", "snr"])

    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in ("value", "run", "snr"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["censored"] = frame["censored"].fillna(0).astype(bool)
    return frame


def measurement_sigma(frame: pd.DataFrame, param: str,
                      base: float = track_mod.DEFAULT_BASE_SIGMA_MHZ) -> np.ndarray:
    """Per-pick standard deviation, from the quality columns stored beside it.

    The same weighting ``muf.track.measurement_sigma`` applies to pipeline
    output, reading the database's column names instead of the CSV's
    per-method suffixes.

    **LOF is weighted by the same ``snr`` as MUF**, which is not quite right:
    the pipeline computes a separate ``lofsnr`` per pick and the ingest path
    does not store it. Until it does, a LOF's uncertainty is scaled by the
    signal level at the *MUF* end of the trace. Recorded here rather than
    hidden because it makes LOF sigmas optimistic for soundings whose two ends
    differ.
    """
    n = len(frame)
    sigma = np.full(n, float(base))

    run = pd.to_numeric(frame.get("run"), errors="coerce").to_numpy(dtype=float)
    marginal = np.nan_to_num(run < track_mod.MARGINAL_RUN, nan=False).astype(bool)
    sigma[marginal] *= 3.0

    snr = pd.to_numeric(frame.get("snr"), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(snr).any():
        reference = float(np.nanmedian(snr))
        if np.isfinite(reference):
            deficit = np.clip(reference - snr, 0.0, 40.0)
            sigma *= 1.0 + np.nan_to_num(deficit, nan=0.0) / 40.0

    return sigma


def tracked(conn: sqlite3.Connection, param: str, tx: str, rx: str,
            method: str = "algo", step_s: int = DEFAULT_STEP_S,
            start: str | None = None, end: str | None = None,
            drop_censored: bool = True,
            process_noise: float = track_mod.DEFAULT_PROCESS_NOISE_MHZ_PER_HOUR,
            ) -> Series:
    """Track one parameter and return it on a regular grid.

    The grid instants are handed to the filter alongside the sounding instants,
    with no measurement attached, so a grid point between two soundings is
    filled by the same mechanism that fills a real gap and carries the same
    honest sigma -- rather than being interpolated afterwards by something that
    does not know how fast a MUF can move.
    """
    picks = observations(conn, param, tx, rx, method, start=start, end=end)
    if picks.empty:
        raise ValueError(f"no {param} rows for {tx}->{rx} [{method}]")

    usable = picks["value"].to_numpy(dtype=float).copy()
    censored = picks["censored"].to_numpy(dtype=bool)
    if drop_censored:
        usable[censored] = np.nan

    if not np.isfinite(usable).any():
        raise ValueError(
            f"no usable {param} picks for {tx}->{rx} [{method}]: "
            f"{int(censored.sum())} of {len(picks)} were at a band edge and the "
            f"rest had no pick."
        )

    grid = pd.date_range(picks["datetime"].iloc[0].ceil(f"{step_s}s"),
                         picks["datetime"].iloc[-1], freq=f"{step_s}s")

    times = pd.DatetimeIndex(picks["datetime"]).append(grid)
    values = np.concatenate([usable, np.full(len(grid), np.nan)])
    sigma = np.concatenate([measurement_sigma(picks, param),
                            np.full(len(grid), np.nan)])
    order = np.argsort(times.to_numpy(), kind="stable")

    result = track_mod.track(times[order], values[order], sigma[order],
                             process_noise=process_noise, param=param)

    smoothed = result.frame.set_index("datetime")
    # Collapse duplicate instants (a sounding landing exactly on the grid)
    # before reindexing, or the reindex raises on a non-unique index.
    smoothed = smoothed[~smoothed.index.duplicated(keep="first")]
    on_grid = smoothed.reindex(grid)

    # `measured` cannot be read off the grid rows: the grid instants were added
    # to the filter *without* measurements, and a sounding stamped
    # 00:00:00.009633 never lands exactly on a 00:00:00 grid point, so the flag
    # would be False everywhere and say nothing. What a caller actually needs to
    # know is whether a real pick backs this grid point, so it is computed by
    # proximity: a measurement within half a step.
    tolerance = pd.Timedelta(seconds=step_s / 2)
    pick_times = pd.DatetimeIndex(picks["datetime"][np.isfinite(usable)])
    backed = np.zeros(len(grid), dtype=bool)
    censored_near = np.zeros(len(grid), dtype=bool)
    if len(pick_times):
        nearest = pick_times.get_indexer(grid, method="nearest")
        distance = np.abs(grid - pick_times[nearest])
        backed = distance <= tolerance

    all_times = pd.DatetimeIndex(picks["datetime"])
    if censored.any():
        censored_times = all_times[censored]
        nearest = censored_times.get_indexer(grid, method="nearest")
        censored_near = np.abs(grid - censored_times[nearest]) <= tolerance

    frame = pd.DataFrame({
        "value": on_grid[param].astype(float),
        "sigma": on_grid["sigma"].astype(float),
        "measured": backed,
        "censored": censored_near,
    }, index=grid)
    frame.index.name = "datetime"

    return Series(
        frame=frame, param=param, tx=tx, rx=rx, method=method,
        n_measured=int(frame["measured"].sum()),
        n_filled=int((~frame["measured"]).sum()),
        n_rejected=result.n_rejected,
        n_censored=int(censored.sum()),
    )


def circuits(conn: sqlite3.Connection, param: str = "muf",
             method: str = "algo") -> list[dict]:
    """Circuits with enough of this parameter to be worth running."""
    column = PARAMS[param]["value"]
    return db.rows(
        conn,
        f"SELECT s.tx, s.rx, COUNT(*) AS n, MIN(s.datetime) AS first, "
        f"MAX(s.datetime) AS last "
        f"FROM extraction e JOIN sounding s ON s.id = e.sounding_id "
        f"WHERE e.method = ? AND e.{column} IS NOT NULL "
        f"AND s.tx IS NOT NULL AND s.rx IS NOT NULL "
        f"GROUP BY s.tx, s.rx ORDER BY n DESC",
        (method,),
    )
