"""Fitting a model on this instrument's own measurements.

    python -m services.prediction.train --param muf --tx NIC3 --rx Yoshkar-Ola --lead 24

Every model this service has run so far is a legacy import: fitted somewhere
else, on somebody else's circuit, against a *modelled* target -- which is why
the schema will not let one become the operational forecast, and why all four
lose to persistence at 24 h. This module is the other end of that. What it
produces is `origin='trained'`, `target_src='measured'`, bound to the circuit
it learned, and therefore promotable.

**This is the only module in the repository that calls ``fit``.** The rule it
is the exception to is worth restating: the research code this service replaces
loads a saved model and refits it before predicting, so its "forecasts" come
from a model trained seconds earlier on the data it is about to be judged
against. ``tests/test_prediction_infer.py`` pins the inference path shut by
making ``fit`` raise. Training is a separate process, in a separate container,
writing a separate artifact -- and it never predicts anything into ``forecast``.

Three decisions carry the whole thing, and each is a way of not fooling
yourself:

**Inputs are the tracked grid; the target is a measured pick.** Features have
to exist at regular instants, which is what ``dataset.tracked`` is for. Truth
does not: a Kalman-filled point is an estimate, and fitting to it teaches a
model to reproduce the filter. So ``y`` comes from ``scoring.truth`` -- the
same picks the leaderboard judges against -- and a feature row with no real
pick within half a step of the instant it predicts is dropped rather than
filled.

**Band-edge picks are excluded from the fit and kept in the score.** A
``limited`` MUF is a lower bound, not a measurement; regressing onto it teaches
the model the sweep ceiling. ``scoring.summarise`` already counts bounds
one-sidedly and apart, so the holdout number here is directly comparable with
what the leaderboard will report later.

**The holdout is the last N days, never a random split.** Every feature is a
lagged function of the same series, so a shuffled split puts the answer in the
training set. The resulting MAE looks excellent and means nothing.

Nothing here activates a model. A trained model is eligible for promotion --
it satisfies both schema CHECKs -- and promotion stays a deliberate act behind
the control token, because it changes what every consumer of ``/forecast``
receives with nothing in the logs having asked for it.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..api import db
from . import (artifacts, dataset, importer, legacy_features, registry,
               scoring, store)

#: What may be fitted. `huber` first because it is what the archive's models
#: are and because a MUF series has outliers -- a mistracked trace is a real
#: number in the wrong place, and least squares chases it.
ESTIMATORS = ("huber", "ridge", "xgboost")
DEFAULT_ESTIMATOR = "huber"

#: A four-hour rolling window at five-minute sampling. The archive's models use
#: 48 too, which makes the first trained model directly comparable with the
#: imports it is meant to replace.
DEFAULT_WINDOWS = (48,)
DEFAULT_STATS = ("mean", "std")

#: How much of the record is held back. Two days rather than a fraction: the
#: holdout has to be long enough to contain a diurnal cycle at every lead this
#: service offers, and a percentage of a short archive is not.
DEFAULT_HOLDOUT_DAYS = 2.0

DEFAULT_METHOD = "contour"

#: Rows below which a fit is not a fit. Enforced alongside a per-feature rule
#: (ten rows per column), because eighteen features on two hundred rows and
#: eighteen features on eighteen rows fail for different reasons and only one
#: of them is obvious from the count.
MIN_TRAIN_ROWS = 200
ROWS_PER_FEATURE = 10

#: The components a decomposition can contribute, and the guard `build` applies
#: to them: the trend is a centred filter, so at a short lag its features would
#: carry values from after the instant being predicted.
COMPONENTS = ("trend", "seasonal", "residual")


class TrainError(RuntimeError):
    """The training run was refused, or could not be assembled."""


# --------------------------------------------------------------------------
# The recipe
# --------------------------------------------------------------------------

def feature_names(alias: str, lag: int, *, raw: bool,
                  components: tuple[str, ...], windows: tuple[int, ...],
                  stats: tuple[str, ...],
                  time_predictors: tuple[str, ...]) -> tuple[str, ...]:
    """The model's columns, in a fixed order.

    The order is the contract: it is what ``feature_names_in_`` records, what
    the frame is built in, and what ``legacy_features.parse`` recovers on
    import. The archive's models carry ``set`` iteration order from the source
    project, which is neither sorted nor stable across runs -- reordering does
    not raise, it multiplies the wrong coefficient by the wrong number. Ours
    are deterministic, and `tests/test_prediction_train.py` round-trips them.
    """
    names: list[str] = []
    if raw:
        names.append(f"{alias}_lag_{lag}")
    for component in sorted(components):
        names.append(f"{alias}_{component}_lag_{lag}")
    for window in sorted(windows):
        for stat in (s for s in legacy_features.STATS if s in stats):
            names.append(f"{alias}_rolling_{window}_{stat}_lag_{lag}")
    names.extend(time_predictors)
    return tuple(names)


def recipe_for(plan: dict) -> legacy_features.Recipe:
    """A :class:`legacy_features.Recipe` built rather than parsed.

    ``period_assumed=False``: this is the one case in the service where the
    decomposition period is known instead of guessed, and the registry should
    not go on claiming an assumption that was not made.
    """
    alias, lag = plan["alias"], plan["lag"]
    windows = tuple(plan["windows"])
    stats = tuple(plan["stats"])
    components = tuple(plan["components"])
    time_predictors = tuple(plan["time"])
    return legacy_features.Recipe(
        alias=alias, lag=lag, windows=windows, stats=stats,
        components=components, raw=plan["raw"],
        time_predictors=time_predictors,
        period=plan["period"], period_assumed=False,
        features=feature_names(alias, lag, raw=plan["raw"],
                               components=components, windows=windows,
                               stats=stats, time_predictors=time_predictors),
    )


# --------------------------------------------------------------------------
# Vetting a request
# --------------------------------------------------------------------------

def _positive_ints(value, field: str, ceiling: int) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TrainError(f"{field} must be a list of whole numbers")
    out = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise TrainError(f"{field} must be whole numbers; got {item!r}") from exc
        if number < 1 or number > ceiling:
            raise TrainError(f"{field} must be between 1 and {ceiling}; got {number}")
        out.append(number)
    return tuple(sorted(set(out)))


def _subset(value, allowed: tuple[str, ...], field: str,
            default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TrainError(f"{field} must be a list")
    chosen = [str(v) for v in value]
    unknown = [v for v in chosen if v not in allowed]
    if unknown:
        raise TrainError(f"unknown {field}: {unknown}; expected some of {list(allowed)}")
    return tuple(v for v in allowed if v in chosen)


def vet(spec: dict) -> dict:
    """Normalise and check a training request. Raises :class:`TrainError`.

    Called at the api door as well as in the worker. The point is that a
    request that cannot work is refused while the operator is still looking at
    the form; what genuinely cannot be checked here -- whether this circuit has
    enough history for this lead -- needs the tracked series, and is refused by
    :func:`assemble` with the arithmetic in the message.
    """
    if not isinstance(spec, dict):
        raise TrainError("a training request is a JSON object")

    param = str(spec.get("param", "")).strip().lower()
    if param not in dataset.PARAMS:
        raise TrainError(f"param must be one of {sorted(dataset.PARAMS)}")

    tx = str(spec.get("tx", "") or "").strip()
    rx = str(spec.get("rx", "") or "").strip()
    if not tx or not rx:
        raise TrainError(
            "a trained model is bound to a circuit, so both tx and rx are "
            "required. An unbound model cannot be promoted -- the schema "
            "refuses it -- which would make the run pointless.")

    method = str(spec.get("method") or DEFAULT_METHOD).strip()
    estimator = str(spec.get("estimator") or DEFAULT_ESTIMATOR).strip().lower()
    if estimator not in ESTIMATORS:
        raise TrainError(f"estimator must be one of {list(ESTIMATORS)}")

    step_s = dataset.DEFAULT_STEP_S
    if spec.get("lag") is not None:
        lag = _positive_ints(spec["lag"], "lag", 20000)[0]
    elif spec.get("lead_h") is not None:
        try:
            hours = float(spec["lead_h"])
        except (TypeError, ValueError) as exc:
            raise TrainError("lead_h must be a number of hours") from exc
        lag = int(round(hours * 3600 / step_s))
        if lag < 1:
            raise TrainError(
                f"a lead of {hours:g} h is under one {step_s} s sample. The "
                f"shortest lead this grid can express is "
                f"{step_s / 3600:.2f} h.")
    else:
        raise TrainError("say how far ahead to predict: lead_h, or lag in samples")

    # `or` would be the shorter spelling and would swallow a zero, turning a
    # value this function exists to refuse into the default.
    supplied = spec.get("period")
    period = (legacy_features.DEFAULT_DECOMPOSITION_PERIOD
              if supplied is None else int(supplied))
    if period < 2:
        raise TrainError("period must be at least 2 samples")

    windows = _positive_ints(spec.get("windows", list(DEFAULT_WINDOWS)),
                             "windows", 20000)
    stats = _subset(spec.get("stats", list(DEFAULT_STATS)),
                    legacy_features.STATS, "stats")
    components = _subset(spec.get("components"), COMPONENTS, "components")
    time_predictors = _subset(spec.get("time"),
                              tuple(legacy_features.TIME_PREDICTORS), "time")
    raw = bool(spec.get("raw", True))

    if windows and not stats:
        raise TrainError("rolling windows were asked for but no stats to take over them")
    if stats and not windows:
        raise TrainError("rolling stats were asked for but no window to take them over")
    if not (raw or components or windows):
        raise TrainError(
            "this recipe has no lagged features at all, only time columns. A "
            "model over calendar variables alone is not a forecast of the "
            "ionosphere.")
    if components and lag <= period // 2:
        raise TrainError(
            f"lag {lag} is not larger than half the decomposition period "
            f"({period // 2}), so the trend features would carry values from "
            f"after the instant being predicted. Drop the components, or "
            f"predict further ahead than {period // 2 * step_s / 3600:.1f} h.")

    supplied = spec.get("holdout_days")
    try:
        holdout_days = (DEFAULT_HOLDOUT_DAYS if supplied is None
                        else float(supplied))
    except (TypeError, ValueError) as exc:
        raise TrainError("holdout_days must be a number") from exc
    if holdout_days <= 0:
        raise TrainError(
            "holdout_days must be positive: a model reported on the data it "
            "was fitted to has no reported accuracy at all.")

    normalised = {
        "alias": str(spec.get("alias") or param),
        "lag": lag,
        "windows": list(windows),
        "stats": list(stats),
        "components": list(components),
        "time": list(time_predictors),
        "raw": raw,
        "period": period,
        "estimator": estimator,
        "holdout_days": holdout_days,
        "start": spec.get("start") or None,
        "end": spec.get("end") or None,
        "name": (str(spec["name"]).strip() if spec.get("name") else None),
        "note": (str(spec["note"]).strip() if spec.get("note") else None),
    }

    return {"param": param, "tx": tx, "rx": rx, "method": method,
            "estimator": estimator, "lag": lag,
            "lead_h": lag * step_s / 3600, "spec": normalised}


def plan_from_job(job: dict) -> dict:
    """One flat argument set for :func:`run`, from a `train_job` row."""
    spec = dict(job.get("spec") or {})
    spec.update({"param": job["param"], "tx": job["tx"], "rx": job["rx"],
                 "method": job["method"]})
    checked = vet(spec)
    return {**checked["spec"], "param": checked["param"], "tx": checked["tx"],
            "rx": checked["rx"], "method": checked["method"],
            "lag": checked["lag"], "lead_h": checked["lead_h"]}


# --------------------------------------------------------------------------
# Assembling the training set
# --------------------------------------------------------------------------

def assemble(conn, plan: dict) -> dict:
    """The design matrix, the measured target, and the picks that judge both.

    Returns ``X`` (every feature row that a real pick backs), ``y`` aligned to
    it, ``censored`` marking the band-edge picks among them, and the tracked
    series and raw picks the two came from.
    """
    param, tx, rx = plan["param"], plan["tx"], plan["rx"]

    try:
        series = dataset.tracked(conn, param, tx, rx, plan["method"],
                                 start=plan.get("start"), end=plan.get("end"))
    except ValueError as exc:
        raise TrainError(str(exc)) from exc

    recipe = recipe_for(plan)
    largest_window = max(plan["windows"], default=0)
    needed = plan["lag"] + max(largest_window, plan["period"] if plan["components"] else 0)
    if len(series.frame) < max(dataset.MIN_SAMPLES, needed):
        raise TrainError(
            f"{tx} -> {rx} has {len(series.frame)} grid points "
            f"({len(series.frame) * dataset.DEFAULT_STEP_S / 86400:.1f} days). "
            f"A lag-{plan['lag']} model with a {largest_window}-sample window "
            f"needs {needed} of them before it can build a single row, and "
            f"the minimum useful window is {dataset.MIN_SAMPLES}. This is a "
            f"data limit, not a fault: predict a shorter lead, or wait for "
            f"more archive.")

    try:
        frame = legacy_features.build(series.values, recipe, alias=recipe.alias)
    except legacy_features.RecipeError as exc:
        raise TrainError(str(exc)) from exc
    if frame.empty:
        raise TrainError(
            f"no feature row could be built for {tx} -> {rx}: the rolling "
            f"windows never fill within the {len(series.frame)} grid points "
            f"available.")

    observed = scoring.truth(conn, param, tx, rx, plan["method"],
                             start=plan.get("start"), end=plan.get("end"))
    if observed.empty:
        raise TrainError(
            f"no measured {param} picks for {tx} -> {rx} [{plan['method']}]. "
            f"Features can be built from the tracked grid; a target cannot.")

    # A feature row is kept only when a real pick sits within half a step of
    # the instant it predicts. The tracker would happily supply a value there
    # -- that is the point of it -- and fitting to that value teaches the model
    # the filter rather than the ionosphere.
    position = scoring._nearest(observed.index, pd.DatetimeIndex(frame.index))
    keep = position >= 0
    if not keep.any():
        raise TrainError(
            f"none of the {len(frame)} feature rows lands within "
            f"{scoring.MATCH_TOLERANCE} of a measured pick. The features and "
            f"the picks do not overlap in time.")

    position = position[keep]
    return {
        "X": frame[keep],
        "y": observed["value"].to_numpy(dtype=float)[position],
        "censored": observed["censored"].to_numpy(dtype=bool)[position],
        "observed": observed,
        "series": series,
        "recipe": recipe,
    }


def split(index: pd.DatetimeIndex, holdout_days: float) -> pd.Timestamp:
    """The instant the holdout begins: the last ``holdout_days`` of the record.

    Chronological, and it is the only split this module offers. A random split
    over lagged features puts each row's own future in the training set, and
    the MAE that comes back is a measurement of leakage.
    """
    return index.max() - pd.Timedelta(days=float(holdout_days))


def _estimator(name: str):
    """The estimator, wrapped in whatever it needs to be honest.

    The linear ones get a scaler: `HuberRegressor`'s epsilon is a threshold on
    a standardised residual and its `alpha` penalises coefficients, so both
    mean something different per column when the columns are megahertz,
    megahertz-squared-ish rolling standard deviations, and a month number.
    Trees do not care, so xgboost is left bare -- and left recognisable, so
    `artifacts._framework_of` reports `xgboost` rather than `sklearn.pipeline`.
    """
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9,
                            objective="reg:absoluteerror", n_jobs=2)

    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    inner = HuberRegressor(max_iter=500) if name == "huber" else Ridge(alpha=1.0)
    return Pipeline([("scale", StandardScaler()), ("model", inner)])


def default_name(plan: dict) -> str:
    lead = plan["lag"] * dataset.DEFAULT_STEP_S
    return (f"{plan['estimator']}-{plan['param']}-{_lead_label(lead)}-"
            f"{db.utcnow()[:10]}")


def _lead_label(seconds: float) -> str:
    hours = seconds / 3600
    if hours < 48:
        return f"{hours:g}h"
    return f"{hours / 24:g}d"


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def run(conn, plan: dict, by: str | None = None) -> dict:
    """Fit, evaluate on a held-out tail, store, and register. Never activates."""
    import joblib

    parts = assemble(conn, plan)
    frame, y, censored = parts["X"], parts["y"], parts["censored"]
    recipe = parts["recipe"]
    index = pd.DatetimeIndex(frame.index)

    cut = split(index, plan["holdout_days"])
    is_train = index < cut
    fit_rows = is_train & ~censored

    n_features = len(recipe.features)
    floor = max(MIN_TRAIN_ROWS, ROWS_PER_FEATURE * n_features)
    if int(fit_rows.sum()) < floor:
        raise TrainError(
            f"only {int(fit_rows.sum())} measured, uncensored rows fall before "
            f"the holdout cut at {cut:%Y-%m-%d %H:%M}. Fitting {n_features} "
            f"features needs at least {floor} of them "
            f"({ROWS_PER_FEATURE} per column, and never fewer than "
            f"{MIN_TRAIN_ROWS}). There are {len(frame)} paired rows in total, "
            f"{int(censored.sum())} of them band-edge bounds and "
            f"{int((~is_train).sum())} in the holdout. Shorten the holdout, "
            f"take a shorter lead, or wait for more archive.")
    if not (~is_train).any():
        raise TrainError(
            f"the holdout window of {plan['holdout_days']:g} days covers no "
            f"rows: the record ends at {index.max():%Y-%m-%d %H:%M} and starts "
            f"at {index.min():%Y-%m-%d %H:%M}.")

    estimator = _estimator(plan["estimator"])
    estimator.fit(frame[fit_rows], y[fit_rows])

    # Judged exactly as the leaderboard will judge it: same picks, same
    # tolerance, same one-sided treatment of band-edge bounds. A holdout number
    # computed a different way would not be comparable with the thing it is
    # supposed to be compared with.
    held = frame[~is_train]
    predicted = pd.Series(np.asarray(artifacts.predict(estimator, held),
                                     dtype=float), index=held.index)
    holdout = scoring.summarise(scoring.pair(predicted, parts["observed"]),
                                plan["param"])

    horizon_s = int(plan["lag"] * dataset.DEFAULT_STEP_S)
    baseline = _persistence(conn, plan, parts["observed"],
                            pd.DatetimeIndex(held.index), horizon_s, cut)

    name = plan.get("name") or default_name(plan)
    with tempfile.TemporaryDirectory() as scratch:
        artifact = Path(scratch) / f"{name}.sav"
        joblib.dump(estimator, artifact)
        digest = artifacts.sha256(artifact)
        stored = store.put(artifact, digest)

    try:
        model = importer.import_artifact(
            stored, param=plan["param"], name=name,
            tx=plan["tx"], rx=plan["rx"], origin="trained",
            target_src="measured", period=plan["period"],
            period_assumed=False, note=plan.get("note"),
            trained_from=index[is_train].min().strftime(db.TIME_FORMAT),
            trained_to=index[is_train].max().strftime(db.TIME_FORMAT),
            conn=conn,
        )
    except (artifacts.ArtifactError, legacy_features.RecipeError,
            registry.RegistryError) as exc:
        if not _already_registered(conn, digest):
            store.unlink(digest)
        raise TrainError(f"the model fitted but could not be registered: {exc}") from exc

    registry.set_metrics(conn, model["id"], {"holdout": {
        "horizon_s": horizon_s,
        "from": cut.strftime(db.TIME_FORMAT),
        "to": index.max().strftime(db.TIME_FORMAT),
        "days": plan["holdout_days"],
        "n_train": int(fit_rows.sum()),
        "estimator": plan["estimator"],
        **holdout,
        "persistence": baseline,
    }})

    return {
        "model": registry.get(conn, model["id"]),
        "holdout": holdout, "persistence": baseline,
        "horizon_s": horizon_s, "cut": cut,
        "n_train": int(fit_rows.sum()), "n_paired": len(frame),
        "n_censored": int(censored.sum()),
        "series": str(parts["series"]),
    }


def _persistence(conn, plan: dict, observed: pd.DataFrame,
                 at: pd.DatetimeIndex, horizon_s: int,
                 cut: pd.Timestamp) -> dict | None:
    """Persistence over the same holdout, so the number has something to beat.

    A model that cannot beat "the value one lead ago" is not worth promoting,
    and finding that out at fit time is cheaper than finding it out on the
    leaderboard a week later. Computed with `scoring`'s own baseline so it is
    the same comparison, not a second implementation of it.
    """
    try:
        series = scoring.baseline_series(conn, "persistence", plan["param"],
                                         plan["tx"], plan["rx"], observed, at,
                                         horizon_s, cut)
    except scoring.ScoringError:
        return None
    if series.empty:
        return None
    return scoring.summarise(scoring.pair(series, observed), plan["param"])


def _already_registered(conn, digest: str) -> bool:
    row = db.one(conn, "SELECT COUNT(*) AS n FROM model_registry WHERE sha256 = ?",
                 (digest,))
    return bool((row or {}).get("n"))


def describe(result: dict) -> str:
    """One line, in the style `infer.describe` and `scoring.describe` set."""
    model = result["model"]
    holdout = result["holdout"] or {}
    baseline = result["persistence"] or {}
    mae = holdout.get("mae")
    theirs = baseline.get("mae")
    verdict = ""
    if mae is not None and theirs is not None:
        verdict = (f", {'beats' if mae < theirs else 'loses to'} persistence "
                   f"({theirs:.2f})")
    return (f"#{model['id']} {model['name']}: fitted on {result['n_train']} "
            f"measured rows, holdout MAE "
            f"{'--' if mae is None else f'{mae:.2f}'} MHz over "
            f"{holdout.get('n', 0)} pairs{verdict}. Registered for comparison "
            f"until somebody activates it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.prediction.train",
        description="Fit a forecasting model on measured picks and register it.")
    parser.add_argument("--param", required=True, choices=("muf", "lof"))
    parser.add_argument("--tx", required=True)
    parser.add_argument("--rx", required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--lead", type=float, default=None,
                        help="lead time in hours; or use --lag")
    parser.add_argument("--lag", type=int, default=None,
                        help=f"lead in samples of {dataset.DEFAULT_STEP_S} s")
    parser.add_argument("--estimator", default=DEFAULT_ESTIMATOR, choices=ESTIMATORS)
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--stats", default=",".join(DEFAULT_STATS))
    parser.add_argument("--components", default="")
    parser.add_argument("--time", default="")
    parser.add_argument("--no-raw", action="store_true",
                        help="drop the plain lagged value from the features")
    parser.add_argument("--period", type=int,
                        default=legacy_features.DEFAULT_DECOMPOSITION_PERIOD)
    parser.add_argument("--holdout-days", type=float, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    def _split(value: str) -> list:
        return [v.strip() for v in value.split(",") if v.strip()]

    spec = {
        "param": args.param, "tx": args.tx, "rx": args.rx,
        "method": args.method, "estimator": args.estimator,
        "lead_h": args.lead, "lag": args.lag,
        "windows": [int(w) for w in _split(args.windows)],
        "stats": _split(args.stats), "components": _split(args.components),
        "time": _split(args.time), "raw": not args.no_raw,
        "period": args.period, "holdout_days": args.holdout_days,
        "start": args.start, "end": args.end,
        "name": args.name, "note": args.note,
    }

    try:
        checked = vet(spec)
    except TrainError as exc:
        print(f"training refused: {exc}", file=sys.stderr)
        return 1

    plan = {**checked["spec"], "param": checked["param"], "tx": checked["tx"],
            "rx": checked["rx"], "method": checked["method"],
            "lag": checked["lag"], "lead_h": checked["lead_h"]}

    with db.session(args.db) as conn:
        try:
            result = run(conn, plan)
        except (TrainError, store.StoreError) as exc:
            print(f"training failed: {exc}", file=sys.stderr)
            return 1

    print(f"  {result['series']}")
    print(describe(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
