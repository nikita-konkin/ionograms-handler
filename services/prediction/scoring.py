"""Scoring: did the forecast beat doing nothing clever?

    python -m services.prediction.scoring --once --param muf,lof

A model that has been loaded, validated and run is still only a model that
produces numbers. This module is where it acquires a claim, and the claim is
comparative: not "the MAE is 0.94 MHz" but "the MAE is 0.94 MHz where
yesterday's value gives 1.02 and one solar rotation ago gives 1.19". Absolute
error on its own cannot answer the only question promotion turns on, which is
whether running this thing is better than not running it.

So **baselines are scored by the same code, over the same pairs, into the same
table**. They are not a footnote computed elsewhere and pasted in; if the
pairing rule or the censoring rule changes, it changes for the model and its
competitors together, and the comparison stays honest by construction.

Three rules the numbers depend on, each of which is a way to be quietly wrong:

**Truth is measured, never tracked.** :mod:`~services.prediction.dataset`
returns a Kalman-smoothed series on a regular grid, and it is the right input
to a model and the wrong yardstick for one. Half those points were filled by
the tracker; scoring against them would partly be scoring the model against
another model's smoothing, and a forecast that agreed with the filter through a
long gap would earn credit for a stretch where nothing was observed at all.
Here truth comes from :func:`dataset.observations` -- picks, at sounding
instants, or nothing.

**Censored picks are scored one-sidedly and counted apart.** A MUF pick at the
top of the sweep says the real MUF was *at least* that; a forecast above it is
not wrong, and charging it ``|predicted - observed|`` would penalise exactly
the midday hours a good model gets right. The error there is
``max(0, observed - predicted)``, mirrored for a LOF pick at the band floor,
and those pairs are reported in their own columns so a headline MAE is never
diluted by a bound.

**Persistence is offset by the lead, not by a fixed day.** At a 24 h lead that
is yesterday's value at the same UTC minute, which is the standard HF baseline
and the one the console names. At shorter leads it is what was actually known
when a forecast of that lead would have been issued -- always the stronger
comparison, and the one a model has to beat to be worth its container.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from muf.geometry import Point, control_points
from muf.reference import chapman
from ..api import db
from . import dataset, registry

#: Horizon buckets, in seconds: 1 h, 6 h, 24 h, 7 d. A forecast is filed under
#: the nearest one, so a model whose lead is 288 five-minute steps (86400 s)
#: lands in the 24 h column exactly and one at 82800 s lands there too.
HORIZONS = (3600, 21600, 86400, 604800)

#: How much recent history a scoring run covers by default.
DEFAULT_WINDOW_DAYS = 30

#: One solar rotation as seen from Earth. The classical HF recurrence baseline:
#: the same active regions face us again, so the ionosphere often repeats.
RECURRENCE = pd.Timedelta(days=27)

#: Diurnal harmonics fitted by the `harmonic` baseline, plus the zenith term.
HARMONICS = 2

#: A forecast and a pick are the same instant if they are within this of each
#: other. Half the model grid: any wider and two grid points compete for one
#: pick, any narrower and the sub-second jitter on a sounding instant
#: (`00:00:00.009633`) would throw away most of the pairs.
MATCH_TOLERANCE = pd.Timedelta(seconds=dataset.DEFAULT_STEP_S / 2)

#: The baselines every model is judged against, in the order the console shows
#: them.
BASELINES = ("persistence", "recurrence-27d", "iri", "harmonic")


class ScoringError(RuntimeError):
    """Something is wrong with the request, not with the forecast."""


@dataclass
class Pairs:
    """Predictions matched to the picks that judge them."""

    valid_at: pd.DatetimeIndex
    predicted: np.ndarray
    observed: np.ndarray
    censored: np.ndarray

    def __len__(self) -> int:
        return len(self.predicted)


# --------------------------------------------------------------------------
# Truth, and matching against it
# --------------------------------------------------------------------------

def _naive(stamp) -> pd.Timestamp:
    """A timestamp comparable with the rest of the database.

    `forecast.valid_at` is written with the trailing ``Z`` that `db.utcnow`
    puts on everything it stamps, while `sounding.datetime` is naive UTC.
    Parsed as they stand, the two cannot be compared at all -- pandas raises
    rather than guessing -- so every instant is flattened to naive UTC here,
    once, at the edge.
    """
    value = pd.Timestamp(stamp)
    return value.tz_convert(None) if value.tzinfo is not None else value


def truth(conn: sqlite3.Connection, param: str, tx: str, rx: str,
          method: str, start: str | None = None,
          end: str | None = None) -> pd.DataFrame:
    """Measured picks for one circuit: `value`, `censored`, indexed by instant.

    Deliberately *not* :func:`dataset.tracked`. See the module docstring: the
    tracker's filled points are an input to a model, never a yardstick for one.
    """
    picks = dataset.observations(conn, param, tx, rx, method, start=start, end=end)
    if picks.empty:
        return pd.DataFrame(columns=["value", "censored"],
                            index=pd.DatetimeIndex([], name="datetime"))
    picks = picks[np.isfinite(picks["value"])]
    frame = pd.DataFrame({"value": picks["value"].to_numpy(dtype=float),
                          "censored": picks["censored"].to_numpy(dtype=bool)},
                         index=pd.DatetimeIndex(picks["datetime"], name="datetime"))
    return frame[~frame.index.duplicated(keep="first")].sort_index()


def _nearest(index: pd.DatetimeIndex, wanted: pd.DatetimeIndex,
             tolerance: pd.Timedelta = MATCH_TOLERANCE) -> np.ndarray:
    """Position in ``index`` nearest each of ``wanted``, or -1 if too far."""
    if not len(index) or not len(wanted):
        return np.full(len(wanted), -1)
    position = index.get_indexer(wanted, method="nearest")
    distance = np.abs(wanted - index[position])
    return np.where(distance <= tolerance, position, -1)


def pair(predicted: pd.Series, observed: pd.DataFrame,
         tolerance: pd.Timedelta = MATCH_TOLERANCE) -> Pairs:
    """Match a predicted series to the picks that judge it."""
    predicted = predicted[np.isfinite(predicted.to_numpy(dtype=float))]
    position = _nearest(observed.index, pd.DatetimeIndex(predicted.index), tolerance)
    keep = position >= 0
    position = position[keep]
    return Pairs(
        valid_at=pd.DatetimeIndex(predicted.index[keep]),
        predicted=predicted.to_numpy(dtype=float)[keep],
        observed=observed["value"].to_numpy(dtype=float)[position],
        censored=observed["censored"].to_numpy(dtype=bool)[position],
    )


def absolute_error(pairs: Pairs, param: str) -> np.ndarray:
    """Absolute error, one-sided where the pick was a bound.

    For MUF a `limited` pick is a *lower* bound -- the sweep ran out before the
    trace did -- so a prediction above it is consistent with the observation
    and costs nothing. For LOF a `loflim` pick is an upper bound and the sign
    flips. Scoring a bound as though it were a measurement penalises the midday
    hours hardest, which is precisely where an operator needs the number.
    """
    error = np.abs(pairs.predicted - pairs.observed)
    if not pairs.censored.any():
        return error
    if param == "muf":
        one_sided = np.maximum(0.0, pairs.observed - pairs.predicted)
    else:
        one_sided = np.maximum(0.0, pairs.predicted - pairs.observed)
    return np.where(pairs.censored, one_sided, error)


def summarise(pairs: Pairs, param: str) -> dict:
    """MAE, RMSE and bias over the measured pairs; bounds counted apart."""
    error = absolute_error(pairs, param)
    free = ~pairs.censored
    residual = pairs.predicted - pairs.observed

    def _finite(value) -> float | None:
        value = float(value)
        return None if math.isnan(value) else round(value, 4)

    return {
        "n": int(free.sum()),
        "mae": _finite(error[free].mean()) if free.any() else None,
        "rmse": _finite(np.sqrt((residual[free] ** 2).mean())) if free.any() else None,
        "bias": _finite(residual[free].mean()) if free.any() else None,
        "n_censored": int(pairs.censored.sum()),
        "mae_censored": (_finite(error[pairs.censored].mean())
                         if pairs.censored.any() else None),
    }


def bucket(horizon_s: float) -> int:
    """The horizon column a lead time is filed under."""
    return min(HORIZONS, key=lambda edge: abs(math.log(max(horizon_s, 1) / edge)))


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def _shifted(observed: pd.DataFrame, offset: pd.Timedelta,
             at: pd.DatetimeIndex) -> pd.Series:
    """The measured value ``offset`` before each instant in ``at``.

    The engine behind persistence and 27-day recurrence. Instants with nothing
    within the match tolerance that far back are dropped rather than filled:
    a baseline that quietly interpolates is not the "do nothing" comparison it
    claims to be.
    """
    position = _nearest(observed.index, at - offset)
    keep = position >= 0
    return pd.Series(observed["value"].to_numpy(dtype=float)[position[keep]],
                     index=at[keep])


def circuit_point(conn: sqlite3.Connection, tx: str, rx: str) -> Point | None:
    """The path's control point, from the coordinates stored on the soundings.

    Read from `sounding` rather than looked up in :mod:`muf.stations`, for the
    reason `services.api.series.endpoints` gives: the coordinates that produced
    these picks are the ones that must produce anything compared against them.
    A station table corrected after ingest would otherwise move the control
    point without moving a single measurement.
    """
    row = db.one(
        conn,
        "SELECT tx_lat, tx_lon, rx_lat, rx_lon FROM sounding "
        "WHERE tx = ? AND rx = ? AND tx_lat IS NOT NULL AND rx_lat IS NOT NULL "
        "LIMIT 1",
        (tx, rx),
    )
    if not row:
        return None
    points = control_points(Point(row["tx_lat"], row["tx_lon"]),
                            Point(row["rx_lat"], row["rx_lon"]))
    return points[0]


def harmonic_design(index: pd.DatetimeIndex, point: Point) -> np.ndarray:
    """Columns of the harmonic baseline: constant, diurnal terms, ``cos chi``.

    The zenith term enters as ``max(0, cos chi) ** 0.25``, the same Chapman
    exponent :mod:`muf.reference.chapman` uses, so the two agree about what
    "the sun is up" means rather than each having its own idea.
    """
    fraction = ((index.hour * 3600 + index.minute * 60 + index.second)
                / 86400.0).to_numpy(dtype=float)
    columns = [np.ones(len(index))]
    for k in range(1, HARMONICS + 1):
        columns.append(np.cos(2 * np.pi * k * fraction))
        columns.append(np.sin(2 * np.pi * k * fraction))
    columns.append(np.array([
        max(0.0, chapman.solar_zenith_cos(when.to_pydatetime(), point))
        ** chapman.CHAPMAN_EXPONENT for when in index]))
    return np.column_stack(columns)


def harmonic(observed: pd.DataFrame, at: pd.DatetimeIndex, point: Point,
             train_before: pd.Timestamp) -> pd.Series:
    """Diurnal harmonics plus a zenith term, fitted **before** the scored window.

    Fitting on the same points it is then scored against would make this the
    strongest entry on the leaderboard and the most useless -- it would be
    measuring how well a smooth curve describes data it has already seen, and
    every real model would look bad beside it for the wrong reason.
    """
    train = observed[(observed.index < train_before) & (~observed["censored"])]
    if len(train) < 4 * (2 * HARMONICS + 2):
        raise ScoringError(
            f"only {len(train)} uncensored picks before {train_before:%Y-%m-%d}; "
            f"the harmonic baseline needs a training window earlier than the "
            f"one it is scored on, and there is not one here.")
    coefficients, *_ = np.linalg.lstsq(
        harmonic_design(pd.DatetimeIndex(train.index), point),
        train["value"].to_numpy(dtype=float), rcond=None)
    return pd.Series(harmonic_design(at, point) @ coefficients, index=at)


def iri(conn: sqlite3.Connection, param: str, tx: str, rx: str) -> pd.Series:
    """The stored IRI reference, as a series.

    Read from `reference`, never recomputed here: those rows were written
    against the sounding they belong to, with the solar driver of the day, and
    recomputing them now with today's indices would compare the model against a
    reference that has since moved.
    """
    if param != "muf":
        raise ScoringError(
            "IRI predicts the F2 peak and hence the MUF; it says nothing about "
            "the absorption floor that sets LOF, so there is no IRI baseline "
            "for this parameter.")
    rows = db.rows(
        conn,
        "SELECT s.datetime, r.value FROM reference r "
        "JOIN sounding s ON s.id = r.sounding_id "
        "WHERE r.source = 'iri' AND r.param = 'muf' AND s.tx = ? AND s.rx = ? "
        "AND r.value IS NOT NULL ORDER BY s.datetime",
        (tx, rx),
    )
    if not rows:
        raise ScoringError(
            "no IRI rows stored for this circuit; run the pipeline with "
            "`--ref-model iri` to populate them.")
    frame = pd.DataFrame(rows)
    return pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
                     index=pd.DatetimeIndex(pd.to_datetime(frame["datetime"])))


def baseline_series(conn: sqlite3.Connection, name: str, param: str,
                    tx: str, rx: str, observed: pd.DataFrame,
                    at: pd.DatetimeIndex, horizon_s: int,
                    train_before: pd.Timestamp) -> pd.Series:
    """One baseline's prediction at ``at``, or :class:`ScoringError` saying why not."""
    if name == "persistence":
        return _shifted(observed, pd.Timedelta(seconds=horizon_s), at)
    if name == "recurrence-27d":
        return _shifted(observed, RECURRENCE, at)
    if name == "iri":
        stored = iri(conn, param, tx, rx)
        position = _nearest(pd.DatetimeIndex(stored.index), at)
        keep = position >= 0
        return pd.Series(stored.to_numpy(dtype=float)[position[keep]], index=at[keep])
    if name == "harmonic":
        point = circuit_point(conn, tx, rx)
        if point is None:
            raise ScoringError(
                f"no coordinates stored for {tx}->{rx}, so the solar zenith "
                f"term cannot be computed.")
        return harmonic(observed, at, point, train_before)
    raise ScoringError(f"unknown baseline {name!r}; expected one of {BASELINES}")


# --------------------------------------------------------------------------
# Writing scores
# --------------------------------------------------------------------------

def store(conn: sqlite3.Connection, subject: str, param: str, tx: str, rx: str,
          horizon_s: int, result: dict, window: tuple[str, str] | None = None,
          detail: dict | None = None) -> None:
    """One score row, replacing whatever the last run put there."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO score (subject, param, tx, rx, horizon_s, "
            "scored_at, window_from, window_to, n, mae, rmse, bias, "
            "n_censored, mae_censored, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (subject, param, tx, rx, horizon_s, db.utcnow(),
             window[0] if window else None, window[1] if window else None,
             result.get("n", 0), result.get("mae"), result.get("rmse"),
             result.get("bias"), result.get("n_censored", 0),
             result.get("mae_censored"),
             json.dumps(detail) if detail else None),
        )


def scores(conn: sqlite3.Connection, param: str | None = None,
           tx: str | None = None, rx: str | None = None) -> list[dict]:
    """Every score row for a circuit, models and baselines alike."""
    sql = ["SELECT * FROM score WHERE 1 = 1"]
    params: list = []
    for column, value in (("param", param), ("tx", tx), ("rx", rx)):
        if value:
            sql.append(f"AND {column} = ?")
            params.append(value)
    sql.append("ORDER BY subject, horizon_s")
    rows = db.rows(conn, " ".join(sql), tuple(params))
    for row in rows:
        row["detail"] = json.loads(row["detail"]) if row.get("detail") else None
    return rows


def leaderboard(conn: sqlite3.Connection, param: str, tx: str, rx: str) -> list[dict]:
    """Score rows folded into one row per subject, keyed by horizon.

    The shape the console table wants: a name, where it came from, and a
    column per horizon. Model names are resolved here so the template does not
    have to know that `model:7` means anything.
    """
    known = {f"model:{row['id']}": row for row in registry.models(conn, param=param)}
    folded: dict[str, dict] = {}
    for row in scores(conn, param, tx, rx):
        entry = folded.setdefault(row["subject"], {
            "subject": row["subject"],
            "kind": "baseline" if row["subject"].startswith("baseline:") else "model",
            "name": row["subject"].split(":", 1)[1],
            "origin": "", "state": "", "mae": {}, "n": {}, "detail": {},
        })
        model = known.get(row["subject"])
        if model is not None:
            entry["name"] = model["name"]
            entry["origin"] = model["origin"]
            entry["state"] = registry.state_of(model)
            entry["model_id"] = model["id"]
        entry["mae"][str(row["horizon_s"])] = row["mae"]
        entry["n"][str(row["horizon_s"])] = row["n"]
        if row["detail"]:
            entry["detail"][str(row["horizon_s"])] = row["detail"]
    order = {"model": 0, "baseline": 1}
    return sorted(folded.values(),
                  key=lambda e: (order[e["kind"]],
                                 min([v for v in e["mae"].values() if v is not None],
                                     default=float("inf"))))


def drift(board: list[dict]) -> list[dict]:
    """Horizons where the live model is beaten by a baseline.

    **Surfaced, never acted on.** A service that silently swapped the model
    behind a published forecast the moment a rolling number crossed would be
    far harder to debug than one that says so and waits: the operator would be
    left explaining a change of product that nothing in the logs asked for.
    So this returns something for the console to draw, and promotion stays a
    human decision under the control scope.
    """
    active = next((e for e in board
                   if e["kind"] == "model" and e["state"] == "active"), None)
    if active is None:
        return []

    crossed = []
    for horizon, mae in active["mae"].items():
        if mae is None:
            continue
        beaten = [(e["name"], e["mae"][horizon]) for e in board
                  if e["kind"] == "baseline" and e["mae"].get(horizon) is not None
                  and e["mae"][horizon] < mae]
        if not beaten:
            continue
        name, best = min(beaten, key=lambda pair: pair[1])
        crossed.append({"horizon_s": int(horizon), "model": active["name"],
                        "model_mae": mae, "baseline": name,
                        "baseline_mae": best,
                        "n": active["n"].get(horizon, 0)})
    return sorted(crossed, key=lambda entry: entry["horizon_s"])


# --------------------------------------------------------------------------
# The passes
# --------------------------------------------------------------------------

def score_model(conn: sqlite3.Connection, model: dict, tx: str, rx: str,
                method: str = "contour",
                window_days: int = DEFAULT_WINDOW_DAYS,
                now: str | None = None) -> dict:
    """Score every issue of one model whose `valid_at` has already passed."""
    cutoff = _naive(now or db.utcnow())
    since = (cutoff - pd.Timedelta(days=window_days)).strftime(db.TIME_FORMAT)

    rows = db.rows(
        conn,
        "SELECT valid_at, horizon_s, value FROM forecast "
        "WHERE model_id = ? AND param = ? AND tx = ? AND rx = ? "
        "AND value IS NOT NULL AND valid_at <= ? AND valid_at >= ? "
        "ORDER BY issued_at, valid_at",
        (model["id"], model["param"], tx, rx,
         cutoff.strftime(db.TIME_FORMAT), since),
    )
    if not rows:
        return {"subject": f"model:{model['id']}", "name": model["name"],
                "tx": tx, "rx": rx, "param": model["param"], "scored": 0,
                "detail": "no forecast rows in the window whose valid_at has passed"}

    frame = pd.DataFrame(rows)
    frame["valid_at"] = [_naive(value) for value in frame["valid_at"]]
    # The latest issue wins where two forecasts cover the same instant at the
    # same lead: rows arrive in issue order, so keeping the last is keeping the
    # one made with the most information.
    frame["horizon"] = [bucket(value) for value in frame["horizon_s"]]

    observed = truth(conn, model["param"], tx, rx, method,
                     start=since, end=cutoff.strftime(db.TIME_FORMAT))
    if observed.empty:
        return {"subject": f"model:{model['id']}", "name": model["name"],
                "tx": tx, "rx": rx, "param": model["param"], "scored": 0,
                "detail": f"no measured {model['param']} picks in the window"}

    window = (since, cutoff.strftime(db.TIME_FORMAT))
    by_horizon: dict[str, dict] = {}
    for horizon, group in frame.groupby("horizon"):
        group = group.drop_duplicates(subset="valid_at", keep="last")
        series = pd.Series(group["value"].to_numpy(dtype=float),
                           index=pd.DatetimeIndex(group["valid_at"])).sort_index()
        result = summarise(pair(series, observed), model["param"])
        store(conn, f"model:{model['id']}", model["param"], tx, rx,
              int(horizon), result, window)
        by_horizon[str(int(horizon))] = result

    registry.set_metrics(conn, model["id"], {
        "scored_at": db.utcnow(), "method": method,
        "window_days": window_days, "by_horizon": by_horizon,
    })

    return {"subject": f"model:{model['id']}", "name": model["name"],
            "tx": tx, "rx": rx, "param": model["param"],
            "scored": sum(r["n"] for r in by_horizon.values()),
            "horizons": sorted(int(h) for h in by_horizon),
            "mae": {h: r["mae"] for h, r in by_horizon.items()}}


def score_baselines(conn: sqlite3.Connection, param: str, tx: str, rx: str,
                    method: str = "contour",
                    horizons: tuple[int, ...] = HORIZONS,
                    window_days: int = DEFAULT_WINDOW_DAYS,
                    now: str | None = None) -> list[dict]:
    """Score the four baselines over the same window and the same pairs.

    A baseline that cannot be computed is stored with `n = 0` and the reason in
    `detail`, not omitted. A missing row on the console reads as "nobody has
    got round to it"; a row saying *why* IRI is absent for LOF is an answer.
    """
    cutoff = _naive(now or db.utcnow())
    since = cutoff - pd.Timedelta(days=window_days)
    window = (since.strftime(db.TIME_FORMAT), cutoff.strftime(db.TIME_FORMAT))

    # The whole history, not just the window: persistence at a 24 h lead and
    # recurrence at 27 days both reach back before the scored window starts,
    # and a baseline cut off at the window edge would silently lose its first
    # day -- or its first month.
    observed = truth(conn, param, tx, rx, method)
    scored = observed[(observed.index >= since) & (observed.index <= cutoff)]
    if scored.empty:
        return [{"subject": f"baseline:{name}", "name": name, "tx": tx,
                 "rx": rx, "param": param, "scored": 0,
                 "detail": f"no measured {param} picks in the window"}
                for name in BASELINES]

    at = pd.DatetimeIndex(scored.index)
    results = []
    for name in BASELINES:
        by_horizon: dict[str, float | None] = {}
        pairs_scored = 0
        reason: str | None = None
        for horizon in horizons:
            try:
                predicted = baseline_series(conn, name, param, tx, rx, observed,
                                            at, int(horizon), since)
            except ScoringError as exc:
                reason = reason or str(exc)
                by_horizon[str(int(horizon))] = None
                store(conn, f"baseline:{name}", param, tx, rx, int(horizon),
                      {"n": 0}, window, {"unavailable": str(exc)})
                continue

            result = summarise(pair(predicted, observed), param)
            detail = None
            if not result["n"] and not result["n_censored"]:
                # A stored zero with no reason beside it reads as neglect. The
                # usual cause is simply that the archive does not reach back as
                # far as this baseline looks -- 27 days, for recurrence -- and
                # saying so is the difference between a gap and a mystery.
                detail = {"unavailable": (
                    f"{name} found no measurement at the offset it reaches "
                    f"back to; the history here does not go far enough.")}
                reason = reason or detail["unavailable"]
            store(conn, f"baseline:{name}", param, tx, rx, int(horizon),
                  result, window, detail)
            by_horizon[str(int(horizon))] = result["mae"]
            pairs_scored += result["n"]

        results.append({
            "subject": f"baseline:{name}", "name": name, "tx": tx, "rx": rx,
            "param": param, "scored": pairs_scored, "mae": by_horizon,
            "detail": reason,
        })
    return results


def run_once(conn: sqlite3.Connection, params: tuple[str, ...] = ("muf",),
             method: str = "contour", window_days: int = DEFAULT_WINDOW_DAYS,
             tx: str | None = None, rx: str | None = None,
             now: str | None = None) -> list[dict]:
    """Score every model that has produced forecasts, and the baselines.

    ``now`` moves the end of the scored window, which is what rescoring an
    archive needs: without it a run over 2023 data would compare a window
    ending today against forecasts that stopped two years ago and find nothing.
    """
    results: list[dict] = []
    now = now or db.utcnow()

    for param in params:
        for circuit in dataset.circuits(conn, param, method):
            if (tx and circuit["tx"] != tx) or (rx and circuit["rx"] != rx):
                continue
            issued = db.rows(
                conn,
                "SELECT DISTINCT model_id, horizon_s FROM forecast "
                "WHERE param = ? AND tx = ? AND rx = ?",
                (param, circuit["tx"], circuit["rx"]),
            )
            for model_id in sorted({row["model_id"] for row in issued}):
                model = registry.get(conn, model_id)
                if model is None:            # cascade should prevent this
                    continue
                results.append(score_model(conn, model, circuit["tx"],
                                           circuit["rx"], method, window_days, now))

            # The horizons the models actually produced, so the baseline rows
            # line up with them column for column. With no models yet, all four.
            horizons = tuple(sorted({bucket(row["horizon_s"]) for row in issued})) \
                or HORIZONS
            results.extend(score_baselines(conn, param, circuit["tx"],
                                           circuit["rx"], method, horizons,
                                           window_days, now))
    return results


def describe(result: dict) -> str:
    """One line per subject, in the style `infer.describe` set."""
    where = f"{result.get('tx')}->{result.get('rx')}"
    name = result.get("name") or result["subject"]
    if not result.get("scored"):
        return f"  {where} [{result.get('param','?')}] {name}: {result.get('detail','nothing')}"
    if "mae" in result and isinstance(result["mae"], dict):
        by = ", ".join(f"{int(h)//3600}h {v}" for h, v in sorted(
            result["mae"].items(), key=lambda kv: int(kv[0])) if v is not None)
        return f"  {where} [{result['param']}] {name}: {result['scored']} pairs, MAE {by}"
    lead = int(result.get("horizon_s", 0)) // 3600
    return (f"  {where} [{result['param']}] {name}: {result['scored']} pairs, "
            f"MAE {result.get('mae')} at {lead}h")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.prediction.scoring",
        description="Score stored forecasts and their baselines against the picks.")
    parser.add_argument("--param", default="muf",
                        help="comma separated: muf,lof (default: %(default)s)")
    parser.add_argument("--method", default="contour",
                        help="which estimator's picks are the truth (default: %(default)s)")
    parser.add_argument("--tx", default=None)
    parser.add_argument("--rx", default=None)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--as-of", default=None,
                        help="end the scored window here instead of now, to "
                             "rescore an archive (ISO-8601 UTC)")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--interval", type=float, default=21600,
                        help="seconds between passes (default: %(default)s)")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    params = tuple(p.strip() for p in args.param.split(",") if p.strip())

    while True:
        started = time.monotonic()
        with db.session(args.db) as conn:
            results = run_once(conn, params, args.method, args.window_days,
                               args.tx, args.rx, args.as_of)

        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        print(f"[{stamp}] scored {len(results)} subjects in "
              f"{time.monotonic() - started:.1f}s")
        for result in results:
            print(describe(result))
        sys.stdout.flush()

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
