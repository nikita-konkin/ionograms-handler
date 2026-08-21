"""Recovering a model's feature recipe from its column names, and rebuilding it.

The archive's models name their inputs `MUF(3000)F2_rolling_48_std_lag_288`,
which is a complete recipe if you read it carefully and a source of silently
wrong numbers if you read it loosely. These tests pin both halves: what the
names mean, and what the rebuilt frame is allowed to contain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.prediction import legacy_features as lf

pytest.importorskip("statsmodels")

ALIAS = "MUF(3000)F2"
LAG = 288


def archive_features() -> list[str]:
    """The 18 names a real archive artifact carries, in a shuffled order.

    Shuffled on purpose: the real files are in `set` iteration order, not
    sorted, and anything that assumes sorting works on the fixture and fails
    on the archive.
    """
    names = [f"{ALIAS}_lag_{LAG}"]
    names += [f"{ALIAS}_{c}_lag_{LAG}" for c in ("trend", "seasonal", "residual")]
    names += [f"{ALIAS}_rolling_{w}_{s}_lag_{LAG}"
              for w in (12, 48, 288) for s in ("mean", "std", "min", "max")]
    names += ["hour", "minute"]
    rng = np.random.default_rng(3)
    rng.shuffle(names)
    return names


@pytest.fixture
def series():
    """Three days of a diurnal MUF on the five-minute grid."""
    index = pd.date_range("2026-08-18", periods=288 * 3, freq="5min")
    t = np.arange(len(index))
    return pd.Series(18 + 7 * np.sin(2 * np.pi * (t - 60) / 288), index=index)


def test_the_recipe_is_read_off_the_names():
    recipe = lf.parse(archive_features())
    assert recipe.alias == ALIAS
    assert recipe.lag == LAG
    assert recipe.windows == (12, 48, 288)
    assert recipe.stats == ("mean", "std", "min", "max")
    assert set(recipe.components) == {"trend", "seasonal", "residual"}
    assert recipe.raw is True
    # A set: `time_predictors` records *which* time columns to build, and the
    # order they come out in is fixed by `recipe.features`, not by this list.
    assert set(recipe.time_predictors) == {"hour", "minute"}


def test_the_decomposition_period_is_flagged_as_an_assumption():
    """It is the one part of the recipe the names do not carry."""
    recipe = lf.parse(archive_features())
    assert recipe.period_assumed is True
    assert recipe.period == lf.DEFAULT_DECOMPOSITION_PERIOD


def test_an_alias_containing_underscores_and_brackets_survives():
    recipe = lf.parse([f"{ALIAS}_trend_lag_9", f"{ALIAS}_lag_9"])
    assert recipe.alias == ALIAS


def test_an_unparseable_name_is_refused_not_ignored():
    with pytest.raises(lf.RecipeError, match="cannot parse"):
        lf.parse([f"{ALIAS}_lag_9", "something_else_entirely"])


def test_two_source_columns_are_refused():
    with pytest.raises(lf.RecipeError, match="different source columns"):
        lf.parse(["A_lag_9", "B_lag_9"])


def test_two_lags_are_refused():
    with pytest.raises(lf.RecipeError, match="different lags"):
        lf.parse(["A_lag_9", "A_lag_10"])


def test_the_built_frame_matches_the_contract_exactly(series):
    names = archive_features()
    recipe = lf.parse(names)
    frame = lf.build(series, recipe, alias=ALIAS)
    assert list(frame.columns) == names          # order, not just membership
    assert not frame.isna().any().any()


def test_rows_are_indexed_by_the_instant_they_predict(series):
    """The row built from t is about t + lag, which is what makes it a forecast."""
    recipe = lf.parse(archive_features())
    frame = lf.build(series, recipe, alias=ALIAS)
    step = pd.Timedelta(minutes=5)
    assert frame.index.max() == series.index.max() + LAG * step
    assert frame.attrs["lag"] == LAG
    assert frame.attrs["alias"] == ALIAS


def test_it_predicts_past_the_end_of_the_observations(series):
    recipe = lf.parse(archive_features())
    frame = lf.build(series, recipe, alias=ALIAS)
    beyond = frame.index[frame.index > series.index.max()]
    assert len(beyond) > 0, "a lagged model must reach past the last observation"


def test_a_mismatched_alias_is_refused(series):
    """Renaming a column to make a model accept it is the provenance hazard
    this whole module exists to make deliberate."""
    recipe = lf.parse(archive_features())
    with pytest.raises(lf.RecipeError, match="alias supplied"):
        lf.build(series, recipe, alias="muf")


def test_a_single_gap_is_refused(series):
    """One gap misaligns every row after it, and is far under any tolerance
    expressed as a proportion of steps."""
    gapped = series.drop(series.index[100:180])
    recipe = lf.parse(archive_features())
    with pytest.raises(lf.RecipeError, match="not on a regular grid"):
        lf.build(gapped, recipe, alias=ALIAS)


def test_one_missing_sample_is_refused(series):
    recipe = lf.parse(archive_features())
    with pytest.raises(lf.RecipeError, match="not on a regular grid"):
        lf.build(series.drop(series.index[500]), recipe, alias=ALIAS)


def test_a_few_seconds_of_jitter_is_tolerated(series):
    """Sounding instants are not exactly on the grid, and that is not a gap."""
    rng = np.random.default_rng(1)
    jittered = series.copy()
    jittered.index = series.index + pd.to_timedelta(
        rng.integers(-8, 9, len(series)), unit="s")
    jittered = jittered.sort_index()
    frame = lf.build(jittered, lf.parse(archive_features()), alias=ALIAS)
    assert len(frame) > 0


def test_a_lag_shorter_than_half_the_period_is_refused(series):
    """`seasonal_decompose`'s trend is a centred filter, so at a short lag the
    features would carry values from after the instant being predicted."""
    recipe = lf.Recipe(alias=ALIAS, lag=100, components=("trend",), period=288,
                       features=(f"{ALIAS}_trend_lag_100",))
    with pytest.raises(lf.RecipeError, match="centred filter"):
        lf.build(series, recipe, alias=ALIAS)
