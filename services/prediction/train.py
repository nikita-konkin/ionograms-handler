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
ESTIMATORS = ("huber", "ridge", "xgboost", "voting", "stacking")
DEFAULT_ESTIMATOR = "huber"

#: The two that are not estimators but committees of them, ported from the
#: `muf` project's ``voting_stacking_models``. Both exist because the three
#: single estimators fail differently: huber shrugs off a mistracked trace and
#: cannot bend, xgboost bends and will happily learn the tracker's artefacts,
#: ridge sits between them. Averaging decorrelated errors is the one free lunch
#: in forecasting, and it is worth having before reaching for a bigger model.
ENSEMBLES = ("voting", "stacking")

#: What may sit inside a committee. Deliberately the single estimators and
#: nothing else: every member is something this module can already fit alone,
#: so a committee's holdout number is comparable with its own members' and an
#: operator can find out whether the ensemble earned its cost.
MEMBERS = ("huber", "ridge", "xgboost")
DEFAULT_MEMBERS = ("huber", "ridge", "xgboost")

#: The tail of the *training* half held back to weight the voters -- never the
#: holdout, which is the judge and must not be consulted by anything being
#: judged. `muf` weighted its voters by the mass of their coefficients, which
#: this module cannot do honestly: see :func:`_voting_weights`.
INNER_VALIDATION_FRACTION = 0.2
MIN_INNER_VALIDATION_ROWS = 30

#: Folds for the stack's internal cross-validation. See :func:`_ensemble` for
#: why they are not, and cannot be, chronological.
STACKING_FOLDS = 5

#: **`muf`'s vertical recipe, verbatim.** One hour, four hours and a day at
#: five-minute sampling, each with four statistics -- twelve rolling columns.
#: `muf/data_handler/muf_data_handler.py` builds exactly these
#: (`create_rolling_features_fnc(df_total, 'muf', windows=[12, 48, 288],
#: stats=['mean', 'std', 'min', 'max'])`), and the archive's own artifacts name
#: all twelve.
#:
#: This defaulted to `(48,)` and `("mean", "std")` until 2026-08-27 -- three
#: columns against the imports' eighteen. Nothing chose that; it was the
#: smallest recipe that ran, and it made every trained model a thinner thing
#: than the import it is supposed to replace on the same leaderboard. The 288
#: window is the one that mattered most on the fixture: it is what tells the
#: model whether yesterday as a whole was high or low, rather than only what
#: the last four hours did.
DEFAULT_WINDOWS = (12, 48, 288)
DEFAULT_STATS = ("mean", "std", "min", "max")

#: The additive decomposition, which `muf` also has on by default
#: (`use_residual_trend_seasonal_features = True`, `period=288` on the vertical
#: path). Three more columns, and the archive's artifacts carry all three.
#:
#: Applied only where the lag can carry it -- see :func:`vet`. The trend is a
#: centred filter over `period` samples, so at a lag under half a period these
#: would be built from after the instant being predicted and `build` refuses.
#: A default that refuses would turn a short lead into an error rather than
#: into a shorter recipe.
DEFAULT_COMPONENTS = ("trend", "seasonal", "residual")

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

#: How many times each column is shuffled for the permutation importance. Five
#: is enough to average out a single unlucky shuffle on a few hundred holdout
#: rows without turning a diagnostic into the expensive part of a run.
PERMUTATION_REPEATS = 5

#: Below this many holdout rows a permutation importance is noise, and a noisy
#: number in a table reads exactly like a real one.
MIN_PERMUTATION_ROWS = 60

#: Points kept from a learning curve. The curve is a shape to read, not a
#: series to compute with, and 400 rounds is that shape at fifty samples.
CURVE_POINTS = 50

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

    members = _subset(spec.get("members"), MEMBERS, "members",
                      default=DEFAULT_MEMBERS)
    if estimator in ENSEMBLES:
        if len(members) < 2:
            raise TrainError(
                f"{estimator} is a committee, and a committee of "
                f"{len(members)} is just {members[0] if members else 'nothing'}"
                f". Name at least two members, or fit that one estimator "
                f"directly and save the cost.")
    elif spec.get("members") is not None:
        # Accepting it would mean storing a `members` list on a model that has
        # none, which is the kind of quiet this service keeps refusing to ship.
        raise TrainError(
            f"members only means something for {list(ENSEMBLES)}; "
            f"{estimator} is a single estimator.")

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
    # Defaulted, but only where `build` will accept them: half of `period`
    # samples is the floor, and below it the trend column would carry values
    # from after the instant being predicted.
    # Sorted, because `legacy_features.parse` recovers them sorted and the
    # stored recipe has to equal what a reader of the artifact reconstructs.
    # `_subset` returns them in COMPONENTS order, which is the decomposition's
    # natural order and not alphabetical -- readable, and one round trip away
    # from a recipe that silently disagrees with its own model.
    components = tuple(sorted(
        _subset(spec.get("components"), COMPONENTS, "components",
                default=(DEFAULT_COMPONENTS if lag > period // 2 else ()))))
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
        # Absent, not empty, for a single estimator. `plan_from_job` re-vets
        # the stored spec, and a `members: []` sitting in it would come back
        # through the refusal above as "members only means something for..."
        # -- a job refused on the strength of a key this function wrote.
        **({"members": list(members)} if estimator in ENSEMBLES else {}),
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


def _voting_weights(members: list, frame, y) -> tuple[list[float], dict]:
    """How much say each voter gets, from an inner chronological validation.

    **This is where the port diverges from `muf`, deliberately.** That project
    weighted its voters by the mass of their fitted parameters -- the sum of
    ``feature_importances_`` for a tree, the sum of ``abs(coef_)`` for a linear
    model, normalised across members. Three things are wrong with it here, and
    the first is fatal:

    * The linear members are ``Pipeline`` objects, because :func:`_estimator`
      scales them. A pipeline exposes neither attribute, so the original code's
      ``hasattr`` chain falls through to its ``importance = 1.0`` default and
      every linear voter silently gets an identical weight. The scheme would
      not fail, it would quietly stop being the scheme.
    * ``sum(feature_importances_)`` is 1.0 for any fitted gradient booster, by
      construction -- the importances are normalised. So the tree's weight is a
      constant that carries no information about the tree.
    * Coefficient mass measures scale, not skill. A model whose columns happen
      to be small numbers gets large coefficients and, under that rule, more
      of the vote for it.

    So the *intent* is kept -- members earn their weight rather than splitting
    it evenly -- and the measure is changed to the one thing that actually
    ranks forecasters: error on data they did not see. The last
    ``INNER_VALIDATION_FRACTION`` of the training half is held back, each
    member is fitted on what precedes it, and weights are inverse MAE,
    normalised. Chronological, like every other split in this module.

    The inner block is carved out of the *training* rows only. Weighting on the
    holdout would be fitting to the judge, and the MAE this run reports would
    be describing a model that had already read the answer.
    """
    from sklearn.base import clone

    names = [name for name, _ in members]
    equal = [1.0 / len(members)] * len(members)

    total_rows = len(frame)
    cut = int(round(total_rows * (1.0 - INNER_VALIDATION_FRACTION)))
    if cut < MIN_INNER_VALIDATION_ROWS or total_rows - cut < MIN_INNER_VALIDATION_ROWS:
        return equal, {
            "basis": "equal",
            "why": (f"{total_rows} training rows split "
                    f"{cut}/{total_rows - cut} leaves an inner validation "
                    f"block under {MIN_INNER_VALIDATION_ROWS} rows, which "
                    f"would rank the members on noise"),
            "members": names,
        }

    inner_x, inner_y = frame.iloc[:cut], y[:cut]
    valid_x, valid_y = frame.iloc[cut:], y[cut:]

    errors: list[float] = []
    for _, estimator in members:
        try:
            fitted = clone(estimator).fit(inner_x, inner_y)
            predicted = np.asarray(artifacts.predict(fitted, valid_x),
                                   dtype=float)
            mae = float(np.mean(np.abs(predicted - valid_y)))
        except Exception:            # a member that cannot fit gets no vote
            mae = float("inf")
        errors.append(mae if np.isfinite(mae) else float("inf"))

    inverse = [0.0 if not np.isfinite(e) else 1.0 / max(e, 1e-6)
               for e in errors]
    if sum(inverse) <= 0:
        return equal, {"basis": "equal",
                       "why": "no member produced a finite error",
                       "members": names}

    weights = [value / sum(inverse) for value in inverse]
    return weights, {
        "basis": "inverse-mae",
        "n_inner_train": cut,
        "n_inner_validation": total_rows - cut,
        "members": names,
        "mae": {name: (None if not np.isfinite(e) else round(e, 4))
                for name, e in zip(names, errors)},
        "weight": {name: round(w, 4) for name, w in zip(names, weights)},
    }


def _ensemble(name: str, member_names: list, frame, y) -> tuple:
    """A committee of :func:`_estimator` members, and what to record about it.

    ``voting`` is a weighted average of the members' predictions; ``stacking``
    fits a small random forest on top of their out-of-fold predictions, which
    is what `muf` used and is kept.

    **The stack's internal cross-validation is not chronological, and cannot
    be.** ``StackingRegressor`` builds the meta-learner's training matrix with
    ``cross_val_predict``, which requires the folds to partition the rows --
    every row predicted exactly once. Forward-chaining folds never partition
    anything: the earliest block has no past to be trained on, so it can be a
    training set or excluded, never a test fold. sklearn refuses a
    ``TimeSeriesSplit`` here with "cross_val_predict only works for
    partitions", and it is right to.

    So the meta-learner sees out-of-fold predictions from members that were
    fitted partly on later data. That is leakage, and it is confined: it can
    make the *blend* better than it deserves to be, and it cannot touch the
    number this run reports, because the entire stack is fitted on rows before
    the cut and scored on rows after it. If a stack beats its own members on
    that holdout, the win is real; if it wins by a suspiciously wide margin,
    this paragraph is the first place to look.
    """
    from sklearn.ensemble import (RandomForestRegressor, StackingRegressor,
                                  VotingRegressor)

    members = [(member, _estimator(member)) for member in member_names]

    if name == "voting":
        weights, detail = _voting_weights(members, frame, y)
        return (VotingRegressor(estimators=members, weights=list(weights)),
                {"kind": "voting", **detail})

    return (StackingRegressor(
                estimators=members,
                final_estimator=RandomForestRegressor(n_estimators=50,
                                                      random_state=42),
                cv=STACKING_FOLDS, n_jobs=1),
            {"kind": "stacking", "members": list(member_names),
             "final_estimator": "random_forest(50)",
             "cv_folds": STACKING_FOLDS, "cv_chronological": False})


def _column_weights(fitted) -> tuple:
    """One fitted estimator's per-column weights, and what they are.

    Unwraps the `Pipeline` :func:`_estimator` puts linear models in, which is
    the same wrapper whose opacity broke `muf`'s voting scheme -- see
    :func:`_voting_weights`. Here it is unwrapped rather than worked around,
    because the question is different.
    """
    inner = fitted
    if hasattr(fitted, "named_steps"):
        inner = fitted.named_steps.get("model", fitted)
    if hasattr(inner, "feature_importances_"):
        return np.asarray(inner.feature_importances_, dtype=float), "gain"
    if hasattr(inner, "coef_"):
        return np.abs(np.asarray(inner.coef_, dtype=float).ravel()), "coefficient"
    return None, None


def _influence(estimator, names: list[str]) -> dict | None:
    """Which columns the fitted model actually leans on, as shares.

    Recorded here because this is the only container that ever holds a fitted
    estimator. The api serves the model page and deliberately cannot
    deserialise an artifact -- that is what `capability` is about -- so if this
    is not written at fit time the console can only ever list column names.

    Two sources, normalised to shares of one so a table can put them in the
    same column:

    * a linear model's ``coef_``, absolute. Comparable across columns *because*
      :func:`_estimator` scales the inputs: a coefficient on a standardised
      column is the response to one standard deviation of it, so megahertz and
      a rolling standard deviation can be read against each other.
    * a booster's ``feature_importances_``, already normalised by construction.

    **This is not the question `_voting_weights` refuses to answer with
    coefficient mass**, and the difference matters. Ranking *columns within*
    one fitted model is exactly what a scaled coefficient is for. Ranking
    *models against each other* by the size of their coefficients measures
    scale rather than skill, which is why that function weights by inverse MAE
    instead. Same numbers, different question, opposite verdict.

    A committee is reported per member rather than blended. Two members
    disagreeing about which column matters is a fact worth seeing, and an
    average of a booster's gain and a linear model's coefficient is a number
    with no units and no meaning.
    """
    def share(weights) -> dict | None:
        if weights is None or len(weights) != len(names):
            return None
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            return None
        return {name: round(float(w) / total, 5)
                for name, w in zip(names, weights)}

    weights, basis = _column_weights(estimator)
    direct = share(weights)
    if direct is not None:
        return {"basis": basis, "share": direct}

    members = {}
    for member, fitted in (getattr(estimator, "named_estimators_", None)
                           or {}).items():
        weights, basis = _column_weights(fitted)
        part = share(weights)
        if part is not None:
            members[member] = {"basis": basis, "share": part}
    return {"basis": "per-member", "members": members} if members else None


def _permutation_importance(estimator, frame, y) -> dict | None:
    """How much the holdout error worsens when one column is shuffled.

    The answer to what :func:`_influence` cannot answer. Gain and coefficient
    mass are *model-internal*: they say what the fitted model leans on, and
    where the columns are near-duplicates of each other -- which these are,
    `muf_lag_288` against `muf_rolling_12_mean_lag_288` at 0.967 on the rig --
    the credit is split among them roughly arbitrarily. A low share is then not
    evidence a column carries nothing.

    Shuffling asks a different question: break the link between this column and
    the target, on rows the model has never seen, and see how much worse the
    error gets. It measures effect on error rather than internal weight, it is
    the same quantity for every estimator so a committee gets one comparable
    number instead of three incomparable ones, and it is reported in **MHz of
    MAE**, which is the unit everything else on the page is in.

    It does not solve collinearity either, and nothing does: with two
    near-identical columns the model can lean on whichever survives the
    shuffle, so both can look unimportant. Two measures disagreeing is
    informative, which is why this sits beside the shares rather than
    replacing them.

    A value can be **negative** -- shuffling made the holdout better. That is
    noise, or a column the model would be better off without, and it is
    reported as it came out rather than clamped to zero.

    Scored as plain MAE over the uncensored holdout rows, not through
    `scoring.pair`. The differences are what this is about, and they are
    unaffected; the baseline here will not equal the headline MAE, which is
    matched to picks by tolerance and one-sided at a band edge.
    """
    if len(frame) < MIN_PERMUTATION_ROWS:
        return None

    def mae(data) -> float:
        return float(np.mean(np.abs(
            np.asarray(artifacts.predict(estimator, data), dtype=float) - y)))

    try:
        base = mae(frame)
        work = frame.copy()
        rng = np.random.default_rng(42)
        delta = {}
        for name in frame.columns:
            original = work[name].to_numpy(copy=True)
            scores = [mae(work.assign(**{name: rng.permutation(original)}))
                      for _ in range(PERMUTATION_REPEATS)]
            delta[name] = round(float(np.mean(scores)) - base, 4)
    except Exception as exc:        # a diagnostic never takes the run down
        return {"basis": "unavailable", "why": f"{type(exc).__name__}: {exc}"}

    return {
        "basis": "permutation-mae",
        "baseline_mae": round(base, 4),
        "repeats": PERMUTATION_REPEATS,
        "n_rows": int(len(frame)),
        "delta": delta,
    }


def _learning_curve(frame, y) -> dict | None:
    """Per-round training and validation loss, where there are rounds at all.

    **Only the booster has a learning curve.** Ridge has a closed form and
    Huber converges to one; neither has a loss that evolves over iterations a
    reader could watch, so asking for their "learning loss" is asking for a
    quantity that does not exist. Where the plan involves xgboost -- alone or
    as a committee member -- there are 400 rounds and the shape of those two
    curves is the standard diagnostic: still falling together at the end means
    the model is under-trained, and a validation curve that turns up while the
    training curve keeps falling means it is memorising.

    **This is a probe, not the shipped model.** Getting a validation curve
    means holding rows back, and a model fitted on less than all of its
    training data is not the model this run is supposed to deliver. So a
    second booster is fitted on the inner split purely to record the curve and
    is then thrown away; the estimator that gets stored has seen every training
    row. The cost is one extra fit, which is the price of the curve being about
    something.

    The split is the same inner chronological one :func:`_voting_weights`
    uses, and for the same reason: carving it out of the *holdout* would be
    fitting to the judge.

    One thing this will not show, and it is worth being blunt about it because
    it is usually the question being asked: a curve like this is a single
    number per round, averaged over every hour of the day. A model that fits
    the sunlit hours and misses the nightly minimum has a perfectly healthy
    curve. That is what :func:`scoring.diurnal` is for.
    """
    total = len(frame)
    cut = int(round(total * (1.0 - INNER_VALIDATION_FRACTION)))
    if cut < MIN_INNER_VALIDATION_ROWS or total - cut < MIN_INNER_VALIDATION_ROWS:
        return None

    probe = _estimator("xgboost")
    probe.set_params(eval_metric="mae")
    try:
        probe.fit(frame.iloc[:cut], y[:cut],
                  eval_set=[(frame.iloc[:cut], y[:cut]),
                            (frame.iloc[cut:], y[cut:])],
                  verbose=False)
        history = probe.evals_result()
        train = [float(v) for v in history["validation_0"]["mae"]]
        valid = [float(v) for v in history["validation_1"]["mae"]]
    except Exception as exc:        # a curve is a diagnostic, never the run
        return {"basis": "unavailable", "why": f"{type(exc).__name__}: {exc}"}

    if not valid:
        return None

    # Thinned for storage, with the last round always kept: 400 rounds is a
    # readable curve at 50 points and four times the JSON at 400.
    step = max(1, len(valid) // CURVE_POINTS)
    keep = sorted(set(list(range(0, len(valid), step)) + [len(valid) - 1]))
    best = int(np.argmin(valid))

    return {
        "basis": "xgboost-rounds",
        "metric": "mae",
        "n_inner_train": cut,
        "n_inner_validation": total - cut,
        "n_rounds": len(valid),
        "round": [k + 1 for k in keep],
        "train": [round(train[k], 4) for k in keep],
        "validation": [round(valid[k], 4) for k in keep],
        "best_round": best + 1,
        "best_validation": round(valid[best], 4),
        "final_validation": round(valid[-1], 4),
    }


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

    ensemble: dict | None = None
    if plan["estimator"] in ENSEMBLES:
        members = list(plan.get("members") or DEFAULT_MEMBERS)
        estimator, ensemble = _ensemble(plan["estimator"], members,
                                        frame[fit_rows], y[fit_rows])
    else:
        estimator = _estimator(plan["estimator"])

    # Before the real fit, because the probe wants the same rows and reading
    # them off a fitted estimator is not possible for a committee -- the
    # members are cloned inside it.
    involves_xgboost = (plan["estimator"] == "xgboost"
                        or "xgboost" in list(plan.get("members") or []))
    curve = (_learning_curve(frame[fit_rows], y[fit_rows])
             if involves_xgboost else None)

    estimator.fit(frame[fit_rows], y[fit_rows])
    influence = _influence(estimator, list(recipe.features))

    # Judged exactly as the leaderboard will judge it: same picks, same
    # tolerance, same one-sided treatment of band-edge bounds. A holdout number
    # computed a different way would not be comparable with the thing it is
    # supposed to be compared with.
    held = frame[~is_train]
    predicted = pd.Series(np.asarray(artifacts.predict(estimator, held),
                                     dtype=float), index=held.index)
    pairs = scoring.pair(predicted, parts["observed"])
    holdout = scoring.summarise(pairs, plan["param"])

    # The same error split by hour, on the holdout and on the rows the model
    # was fitted on. The pair is the diagnostic, not either half: a night error
    # that is large in both says the model *cannot* represent the nightly
    # minimum with the columns it was given, and more archive will not help --
    # that is a features problem. Large only on the holdout says it learned a
    # night that has since moved.
    diurnal = {
        "holdout": scoring.diurnal(pairs, plan["param"]),
        "train": scoring.diurnal(scoring.Pairs(
            valid_at=index[fit_rows],
            predicted=np.asarray(artifacts.predict(estimator, frame[fit_rows]),
                                 dtype=float),
            observed=y[fit_rows],
            censored=np.zeros(int(fit_rows.sum()), dtype=bool),
        ), plan["param"]),
    }

    # On the holdout, uncensored: a band-edge bound is a one-sided
    # observation and shuffling a column to see how far a two-sided error
    # moves would be measuring the bound, not the column.
    scored_rows = (~is_train) & (~censored)
    permutation = _permutation_importance(estimator, frame[scored_rows],
                                          y[scored_rows])

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
        **({"ensemble": ensemble} if ensemble else {}),
        **holdout,
        "persistence": baseline,
    }, "diurnal": diurnal,
        **({"learning": curve} if curve else {}),
        **({"influence": influence} if influence else {}),
        **({"permutation": permutation} if permutation else {})})

    return {
        "model": registry.get(conn, model["id"]),
        "holdout": holdout, "persistence": baseline,
        "horizon_s": horizon_s, "cut": cut,
        "diurnal": diurnal, "learning": curve, "influence": influence,
        "permutation": permutation,
        "n_train": int(fit_rows.sum()), "n_paired": len(frame),
        "n_censored": int(censored.sum()),
        "ensemble": ensemble,
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
    committee = ""
    ensemble = result.get("ensemble") or {}
    if ensemble:
        weights = ensemble.get("weight")
        committee = (
            f" {ensemble['kind'].title()} of "
            + (", ".join(f"{name} {value:.2f}"
                         for name, value in weights.items()) if weights
               else ", ".join(ensemble.get("members", [])))
            + ".")

    # The hour that went worst, because a headline MAE cannot say "it is fine
    # all day and hopeless at 02 UTC" and that is the failure this service
    # actually keeps producing.
    worst = ""
    hours = (result.get("diurnal") or {}).get("holdout") or []
    ranked = [h for h in hours if h.get("mae") is not None]
    if ranked and mae is not None:
        peak = max(ranked, key=lambda h: h["mae"])
        if peak["mae"] > 1.5 * mae:
            worst = (f" Worst hour {peak['hour']:02d} UTC at "
                     f"{peak['mae']:.2f} MHz "
                     f"({peak['bias']:+.2f} bias).")

    curve = result.get("learning") or {}
    training = ""
    if curve.get("basis") == "xgboost-rounds":
        stopped = curve["best_round"] < 0.8 * curve["n_rounds"]
        training = (f" Booster best at round {curve['best_round']}"
                    f"/{curve['n_rounds']}"
                    + (", so it is over-trained past that." if stopped
                       else ", still improving at the last round."))

    return (f"#{model['id']} {model['name']}: fitted on {result['n_train']} "
            f"measured rows, holdout MAE "
            f"{'--' if mae is None else f'{mae:.2f}'} MHz over "
            f"{holdout.get('n', 0)} pairs{verdict}.{committee}{worst}"
            f"{training} Registered for comparison until somebody activates "
            f"it.")


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
    parser.add_argument("--members", default="",
                        help=f"for {'/'.join(ENSEMBLES)}: which of "
                             f"{','.join(MEMBERS)} sit on the committee "
                             f"(default: all three)")
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
        "members": _split(args.members) or None,
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
