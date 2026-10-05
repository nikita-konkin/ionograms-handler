"""What a forecast does when the station goes quiet.

NIC3 -> Yoshkar-Ola had no picks from 2026-09-01 to 09-20. The tracker that
turns picks into a model's input grid is a constant-velocity Kalman smoother,
whose uncertainty across a gap grows as the gap cubed: it bridged the twenty
days with a straight line carrying a 3039 MHz sigma, the model forecast from
that line, and the Series page drew a 3 GHz band that flattened every
measurement into the floor of the panel.

These pin the four parts of the answer: the tracker leaves a hole longer than
`dataset.MAX_BRIDGE_HOURS` empty, the features decompose each unbroken stretch on its
own, no forecast row rests on an outage, and each row says how much of its
input was measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.api import db
from services.prediction import dataset, importer, infer, legacy_features

pytest.importorskip("joblib")
pytest.importorskip("statsmodels")
sklearn_linear = pytest.importorskip("sklearn.linear_model")

from conftest import LAG, feature_names  # noqa: E402
from test_prediction_train import seed  # noqa: E402

TX, RX = "NIC3", "Yoshkar-Ola"
DAY = 288


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as connection:
        yield connection


# --------------------------------------------------------------------------
# The tracker
# --------------------------------------------------------------------------

def test_a_long_outage_is_left_empty_not_bridged(conn):
    index = seed(conn, days=8, gaps=slice(3 * DAY, 5 * DAY))
    series = dataset.tracked(conn, "muf", TX, RX, "contour")
    frame = series.frame

    # Empty end to end -- including the hours just past each edge, which a
    # one-sided fill would have extrapolated.
    hole = frame.loc[index[3 * DAY]:index[5 * DAY - 1]]
    assert hole["value"].isna().all() and hole["sigma"].isna().all()
    assert series.n_unfilled == len(hole)

    # What survives is believable: nothing like the 3039 MHz of September.
    assert frame["sigma"].max() < 6.0
    assert frame.loc[:index[3 * DAY - 1], "value"].notna().all()


def test_a_short_gap_is_still_bridged(conn):
    """Most holes are minutes to a few hours; the features need them filled."""
    seed(conn, days=4, gaps=slice(DAY + 10, DAY + 10 + 48))     # 4 h
    frame = dataset.tracked(conn, "muf", TX, RX, "contour").frame
    assert frame["value"].notna().all()


def test_the_grid_ends_at_the_last_pick_not_the_last_sounding(conn):
    """A scheduled slot that has gone silent keeps producing soundings with no
    pick; extending the grid over them is pure extrapolation, which is what
    drew the rising band at the right of the September plot."""
    index = seed(conn, days=4, gaps=slice(3 * DAY, 4 * DAY))
    frame = dataset.tracked(conn, "muf", TX, RX, "contour").frame
    assert frame.index.max() <= index[3 * DAY - 1]


# --------------------------------------------------------------------------
# The features
# --------------------------------------------------------------------------

def test_each_stretch_is_decomposed_on_its_own():
    t = np.arange(8 * DAY)
    values = 18 + 7 * np.sin(2 * np.pi * t / DAY)
    series = pd.Series(values, index=pd.date_range("2026-08-10", periods=len(t),
                                                   freq="5min"))
    holed = series.copy()
    holed.iloc[3 * DAY:3 * DAY + 100] = np.nan

    got = legacy_features._decompose(holed, DAY)
    assert got.iloc[3 * DAY:3 * DAY + 100].isna().all().all()

    # The stretch after the hole decomposes exactly as it would alone: nothing
    # from before the hole reaches it.
    after = holed.iloc[3 * DAY + 100:]
    alone = legacy_features._decompose(after, DAY)
    np.testing.assert_allclose(got.iloc[3 * DAY + 100:].to_numpy(),
                               alone.to_numpy())


def test_a_stretch_too_short_to_decompose_gives_no_features():
    series = pd.Series(np.ones(5 * DAY),
                       index=pd.date_range("2026-08-10", periods=5 * DAY,
                                           freq="5min"))
    series.iloc[DAY:DAY + 10] = np.nan       # leaves a 1-day run at the start
    got = legacy_features._decompose(series, DAY)
    assert got.iloc[:DAY].isna().all().all()
    assert got.iloc[DAY + 10:].notna().all().all()


# --------------------------------------------------------------------------
# The forecast
# --------------------------------------------------------------------------

@pytest.fixture
def model_row(conn, tmp_path):
    import joblib

    names = feature_names()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, len(names))), columns=names)
    path = tmp_path / "ridge.sav"
    joblib.dump(sklearn_linear.Ridge().fit(frame, frame.iloc[:, 0]), path)
    return importer.import_artifact(path, param="muf", conn=conn,
                                    origin="legacy")


def test_no_forecast_rests_on_an_outage_and_each_says_how_measured(conn,
                                                                   model_row):
    index = seed(conn, days=12, gaps=slice(4 * DAY, 6 * DAY))
    result = infer.run_model(conn, model_row, TX, RX, method="contour")
    assert result["written"] > 0

    rows = db.rows(conn, "SELECT valid_at, sigma, input_measured FROM forecast")
    built_from = (pd.to_datetime([r["valid_at"] for r in rows])
                  - pd.Timedelta(minutes=5 * LAG)).tz_localize(None)
    outage = (built_from > index[4 * DAY] + pd.Timedelta(hours=4)) \
        & (built_from < index[6 * DAY] - pd.Timedelta(hours=4))
    assert not outage.any(), "a row was built from inside the outage"

    measured = np.array([r["input_measured"] for r in rows], dtype=float)
    assert np.isfinite(measured).all()
    assert ((measured >= 0) & (measured <= 1)).all()
    sigma = np.array([r["sigma"] for r in rows], dtype=float)
    assert np.nanmax(sigma) < 6.0


def test_the_series_page_breaks_a_forecast_at_its_holes():
    from services.api.web_routes import _broken_at_gaps

    base = {"value": 10.0, "sigma": 1.0, "input_measured": 1.0, "name": "m",
            "id": 1, "active": 1, "target_alias": None, "trained_from": None,
            "trained_to": None}
    stamps = ["2026-09-01T00:00:00Z", "2026-09-01T00:05:00Z",
              "2026-09-01T00:10:00Z", "2026-09-20T00:00:00Z",
              "2026-09-20T00:05:00Z"]
    rows = [{**base, "valid_at": s} for s in stamps]
    out = _broken_at_gaps(rows)
    assert len(out) == len(rows) + 1
    hole = out[3]
    assert hole["value"] is None and hole["sigma"] is None
    assert "2026-09-01T00:10" < hole["valid_at"] < "2026-09-20"
