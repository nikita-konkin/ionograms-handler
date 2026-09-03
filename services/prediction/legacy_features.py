"""Rebuilding the exact feature vector a saved model was fitted on.

A model artifact names its columns and nothing else. ``MUF(3000)F2_rolling_48_std_lag_288``
says there is a 48-sample rolling standard deviation, lagged 288 samples, of a
column called ``MUF(3000)F2`` -- so the *recipe* is recoverable from the
contract, one regex over the names, and this module rebuilds it from whatever
series it is given.

**The alias is the hazard, and it is why this is a module and not three lines
inline.** These models want a column literally named ``MUF(3000)F2_lag_288``.
Our series is tracked MUF from a circuit on the other side of the planet from
the one those names refer to. Renaming a column to make the model accept it is
a two-character edit and is exactly how a model of somebody else's ionosphere
quietly becomes "the forecast" -- so :func:`build` takes the alias as a
required argument rather than inferring it, the registry stores it, and the
frame carries it in ``attrs`` for anything downstream that wants to check.

**Lagging is what makes this a forecast rather than a fit.** Every predictor
carries ``_lag_N``, so the row that predicts time *T* is built from the series
at *T - N*. This module therefore never needs a value it does not already have:
it computes the transforms on observed data and shifts the resulting index
*forward*, which is the same arithmetic the source project does with
``shift(N)`` and considerably harder to get subtly wrong.

One thing the names do not carry: the **seasonal decomposition period**. The
source project uses 288 -- one day at five-minute sampling -- and that is the
default here, recorded in the recipe as an assumption rather than a reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _day_fraction(idx: pd.DatetimeIndex) -> np.ndarray:
    """Position within the UTC day, in [0, 1). The same quantity
    ``scoring.harmonic_design`` fits its diurnal terms over, so the two agree
    about what "time of day" means rather than each having its own idea."""
    return ((idx.hour * 3600 + idx.minute * 60 + idx.second)
            / 86400.0).to_numpy(dtype=float)


#: Time predictors, which are built from the index and are *not* lagged.
#:
#: The first block matches the vertical branch of the source project's
#: ``create_time_predictors_fnc``; the horizontal branch adds calendar columns
#: that none of the vertical-format artifacts use. **``hour`` and ``minute``
#: are integers with a cliff in them**: 23 and 0 are adjacent instants and
#: maximally distant numbers. A tree can split its way around that; the linear
#: members cannot, and a straight line in ``hour`` cannot describe a diurnal
#: cycle at all. They are kept because saved artifacts name them and this
#: module's job is to rebuild what a model asks for.
#:
#: The second block is the fix, and it is what a new recipe should ask for.
#: ``daily_*`` are the first and second harmonics of the *fraction of the day*,
#: not of the integer hour -- a 24-step staircase would still be a staircase --
#: and they are continuous across midnight. ``yearly_*`` do the same for the
#: seasonal term that ``month``, ``quarter``, ``weekofyear`` and ``dayofyear``
#: currently spell four collinear ways, three of which are constant over any
#: training window this instrument has yet produced.
TIME_PREDICTORS = {
    "hour":       lambda idx: idx.hour,
    "minute":     lambda idx: idx.minute,
    "dayofweek":  lambda idx: idx.dayofweek,
    "quarter":    lambda idx: idx.quarter,
    "month":      lambda idx: idx.month,
    "dayofyear":  lambda idx: idx.dayofyear,
    "weekofyear": lambda idx: idx.isocalendar().week.astype(int),

    "daily_sin":  lambda idx: np.sin(2 * np.pi * _day_fraction(idx)),
    "daily_cos":  lambda idx: np.cos(2 * np.pi * _day_fraction(idx)),
    "daily_sin2": lambda idx: np.sin(4 * np.pi * _day_fraction(idx)),
    "daily_cos2": lambda idx: np.cos(4 * np.pi * _day_fraction(idx)),
    "yearly_sin": lambda idx: np.sin(2 * np.pi * (idx.dayofyear - 1) / 365.25),
    "yearly_cos": lambda idx: np.cos(2 * np.pi * (idx.dayofyear - 1) / 365.25),
}

#: The diurnal terms, as a recipe can ask for them in one word. Not a default:
#: a saved artifact names its own columns and gets exactly those, and changing
#: what an existing recipe means is how a model quietly stops being the model
#: that was scored.
DIURNAL = ("daily_sin", "daily_cos", "daily_sin2", "daily_cos2")

#: The seasonal pair, kept apart from the diurnal one because **it needs
#: months of archive and the diurnal terms need a day**.
#:
#: Over a week of training rows these two move through about 1% of their range
#: -- day 229 to day 236 of the year -- so they are very nearly constant, and
#: a nearly-constant column is the most dangerous thing you can hand a
#: regression. It cannot help in training and it will happily absorb weight as
#: a second intercept, which is exactly what happened the first time they were
#: offered by default: on 2026-08-27, fitted over seven days, `yearly_cos` and
#: `yearly_sin` together took **31%** of an xgboost model's gain and **60%** of
#: a Huber model's coefficient mass. In production day-of-year keeps moving,
#: those columns drift into values never seen in training, and the weight
#: parked on them goes with it.
#:
#: The console's own feature table is what surfaced this, which is the whole
#: argument for printing per-column weights rather than a feature count.
SEASONAL = ("yearly_sin", "yearly_cos")

#: Both, for a recipe fitted over enough archive to earn the seasonal pair.
CYCLICAL = DIURNAL + SEASONAL

#: Decomposition period when the contract does not say, which it never does.
DEFAULT_DECOMPOSITION_PERIOD = 288

#: One feature name. The alias is non-greedy so that `X_trend_lag_9` parses as
#: alias `X` with component `trend`, not as alias `X_trend` with none.
FEATURE_RE = re.compile(
    r"^(?P<alias>.+?)_"
    r"(?:rolling_(?P<window>\d+)_(?P<stat>mean|std|min|max)_"
    r"|(?P<component>trend|seasonal|residual)_)?"
    r"lag_(?P<lag>\d+)$"
)

STATS = ("mean", "std", "min", "max")


class RecipeError(ValueError):
    """The feature names do not describe a recipe this module can rebuild."""


@dataclass(frozen=True)
class Recipe:
    """How to rebuild a model's inputs from one series."""

    alias: str
    lag: int
    windows: tuple[int, ...] = ()
    stats: tuple[str, ...] = ()
    components: tuple[str, ...] = ()          # trend | seasonal | residual
    raw: bool = False                         # the plain lagged target
    time_predictors: tuple[str, ...] = ()
    period: int = DEFAULT_DECOMPOSITION_PERIOD
    period_assumed: bool = True
    features: tuple[str, ...] = ()            # verbatim, in model order

    def as_dict(self) -> dict:
        return {
            "alias": self.alias, "lag": self.lag,
            "windows": list(self.windows), "stats": list(self.stats),
            "components": list(self.components), "raw": self.raw,
            "time_predictors": list(self.time_predictors),
            "period": self.period, "period_assumed": self.period_assumed,
        }


def parse(features: Iterable[str],
          period: int = DEFAULT_DECOMPOSITION_PERIOD,
          assumed: bool = True) -> Recipe:
    """Recover a :class:`Recipe` from a model's ordered feature names.

    Raises :class:`RecipeError` rather than guessing when a name does not
    parse, or when two aliases or two lags appear. A model whose input contract
    is only partly understood is not runnable -- the half that parsed would
    still produce a number.

    ``assumed`` is what the returned recipe records about ``period``. It
    defaults to True because the period is genuinely unrecoverable from an
    artifact -- that is this module's standing complaint about the archive's
    models. A model this service trained is the one case where it is known, so
    :mod:`~services.prediction.train` passes False and the registry stops
    claiming an assumption it did not make.
    """
    features = tuple(features)
    if not features:
        raise RecipeError("the artifact carries no feature names")

    aliases: set[str] = set()
    lags: set[int] = set()
    windows: set[int] = set()
    stats: set[str] = set()
    components: set[str] = set()
    time_cols: list[str] = []
    raw = False

    for name in features:
        if name in TIME_PREDICTORS:
            time_cols.append(name)
            continue

        match = FEATURE_RE.match(name)
        if not match:
            raise RecipeError(
                f"cannot parse the feature name {name!r}. Expected "
                f"'<alias>_lag_<n>', '<alias>_<trend|seasonal|residual>_lag_<n>', "
                f"'<alias>_rolling_<window>_<stat>_lag_<n>', or one of the time "
                f"predictors {sorted(TIME_PREDICTORS)}."
            )

        aliases.add(match["alias"])
        lags.add(int(match["lag"]))
        if match["window"]:
            windows.add(int(match["window"]))
            stats.add(match["stat"])
        elif match["component"]:
            components.add(match["component"])
        else:
            raw = True

    if len(aliases) != 1:
        raise RecipeError(
            f"the features name {len(aliases)} different source columns "
            f"({sorted(aliases)}). This module rebuilds one series; a model "
            f"over several needs each one supplied explicitly."
        )
    if len(lags) != 1:
        raise RecipeError(
            f"the features carry {len(lags)} different lags ({sorted(lags)}). "
            f"A single-horizon model has one; several means this artifact "
            f"predicts a window and needs the horizontal-format builder."
        )

    return Recipe(
        alias=aliases.pop(), lag=lags.pop(),
        windows=tuple(sorted(windows)), stats=tuple(s for s in STATS if s in stats),
        components=tuple(sorted(components)), raw=raw,
        time_predictors=tuple(time_cols),
        period=period, period_assumed=assumed,
        features=features,
    )


def describe_feature(name: str) -> dict:
    """One column name, taken apart. The reading half of :func:`parse`.

    `parse` needs the whole ordered set to recover a recipe and refuses
    anything it cannot rebuild, which is right for a contract and useless for
    a table: a console wants to say what *this* column is without judging
    whether the set as a whole is runnable. So this answers per name, and an
    unparseable one comes back as ``kind: "unknown"`` rather than raising --
    a model with an odd column should still render its other seventeen.
    """
    if name in TIME_PREDICTORS:
        return {"name": name, "kind": "cyclical" if name in CYCLICAL else "time",
                "alias": None, "lag": None, "window": None,
                "stat": None, "component": None}

    match = FEATURE_RE.match(name)
    if not match:
        return {"name": name, "kind": "unknown", "alias": None, "lag": None,
                "window": None, "stat": None, "component": None}

    if match["window"]:
        kind = "rolling"
    elif match["component"]:
        kind = "component"
    else:
        kind = "raw"
    return {
        "name": name, "kind": kind, "alias": match["alias"],
        "lag": int(match["lag"]),
        "window": int(match["window"]) if match["window"] else None,
        "stat": match["stat"], "component": match["component"],
    }


def _decompose(series: pd.Series, period: int) -> pd.DataFrame:
    """Additive trend/seasonal/residual, as the source project computes them.

    ``extrapolate_trend="freq"`` matches the source and fills the ends the
    centred moving average cannot reach.

    **The trend is a centred filter**, so a value at *t* is built from
    *t ± period/2*. That is only safe here because the lag applied afterwards
    exceeds half the period -- 288 against 144 for the default. :func:`build`
    enforces it rather than trusting it.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model="additive", period=period,
                                extrapolate_trend="freq")
    return pd.DataFrame({
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }, index=series.index)


def build(series: pd.Series, recipe: Recipe, *, alias: str) -> pd.DataFrame:
    """Build the model's input frame, indexed by the time each row predicts.

    Args:
        series: the observed series, on a regular grid, ascending. In this
            service that is the Kalman-tracked MUF or LOF, not raw picks.
        recipe: from :func:`parse`.
        alias: the column name the model's features are built around. Required,
            never inferred -- see the module docstring. Pass
            ``recipe.alias`` deliberately, having decided that feeding this
            series to this model is meaningful.

    Returns a frame whose columns are exactly ``recipe.features``, in that
    order, and whose index is the instants those rows predict. Rows that cannot
    be built (the rolling windows are not yet full) are dropped.
    """
    if alias != recipe.alias:
        raise RecipeError(
            f"this model's features are built around {recipe.alias!r} but the "
            f"alias supplied is {alias!r}. The two must match: the model "
            f"resolves its inputs by name."
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        raise RecipeError("series must be indexed by time")
    if not series.index.is_monotonic_increasing:
        raise RecipeError("series index must be ascending")

    if recipe.components and recipe.lag <= recipe.period // 2:
        raise RecipeError(
            f"lag {recipe.lag} is not larger than half the decomposition "
            f"period ({recipe.period // 2}). The trend is a centred filter, so "
            f"at this lag the features would carry values from after the "
            f"instant being predicted."
        )

    step = _step(series.index)
    parts: dict[str, pd.Series] = {}

    if recipe.raw:
        parts[f"{alias}_lag_{recipe.lag}"] = series

    for component in recipe.components:
        decomposed = _decompose(series, recipe.period)
        for name in recipe.components:
            parts[f"{alias}_{name}_lag_{recipe.lag}"] = decomposed[name]
        break                                   # decompose once, not per name

    for window in recipe.windows:
        rolling = series.rolling(window)
        for stat in recipe.stats:
            parts[f"{alias}_rolling_{window}_{stat}_lag_{recipe.lag}"] = \
                getattr(rolling, stat)()

    frame = pd.DataFrame(parts, index=series.index)

    # The row computed at t predicts t + lag*step. Shifting the index forward
    # is the same operation as shifting the values back, and it leaves the
    # frame indexed by the instant each row is *about*.
    frame.index = frame.index + recipe.lag * step

    for name in recipe.time_predictors:
        frame[name] = np.asarray(TIME_PREDICTORS[name](frame.index))

    missing = [c for c in recipe.features if c not in frame.columns]
    if missing:
        raise RecipeError(f"could not build {missing}")

    frame = frame[list(recipe.features)].dropna()
    frame.attrs["alias"] = alias
    frame.attrs["lag"] = recipe.lag
    frame.attrs["step_s"] = step.total_seconds()
    return frame


def _step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """The grid spacing, which must be regular for a lag to mean anything.

    Two separate tests, because they catch different faults and the cheap one
    misses the dangerous one.

    **Any single gap is fatal**, which a proportion cannot express. ``build``
    shifts the index forward by ``lag * step``, so it is counting *samples* and
    calling the answer a duration. One missing hour in a month of data is 0.1%
    of the steps -- far under any sane tolerance -- and it silently misaligns
    every row after it by an hour. So the widest step is checked against the
    median directly.

    **Jitter is tolerated**, because sounding instants are not exactly on the
    grid: a few seconds either side of a five-minute cadence is the recorder,
    not a gap.
    """
    if len(index) < 2:
        raise RecipeError("need at least two samples to establish the grid step")

    deltas = np.diff(index.to_numpy()).astype("timedelta64[s]").astype(int)
    step = int(np.median(deltas))
    if step <= 0:
        raise RecipeError("series has a non-positive time step")

    # A tenth of a step, or one second, whichever is larger.
    tolerance = max(1, step // 10)
    off_grid = np.abs(deltas - step) > tolerance

    if off_grid.any():
        widest = int(deltas.max())
        where = index[int(np.argmax(deltas))]
        raise RecipeError(
            f"the series is not on a regular grid: {off_grid.sum()} of "
            f"{len(deltas)} steps differ from the median {step} s by more than "
            f"{tolerance} s, the widest being {widest} s at {where}. Lagged "
            f"features count samples and report the answer as a duration, so a "
            f"single gap misaligns every row after it. Resample through "
            f"`muf track`, which fills gaps and says which points it filled."
        )

    return pd.Timedelta(seconds=step)
