"""Importing a model, and running it without retraining it.

The load-bearing test here is `test_inference_never_fits`. The code this
service replaces loads a saved model and immediately refits it, so its
"predictions" come from a model trained seconds earlier on the data it is being
scored against. That is invisible from the outside -- the numbers look good,
which is the problem -- so it is pinned by making `fit` raise rather than by
reading the code and believing it.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from services.api import db
from services.prediction import artifacts, importer, infer, registry

joblib = pytest.importorskip("joblib")
sklearn_linear = pytest.importorskip("sklearn.linear_model")
pytest.importorskip("statsmodels")

# The alias, the lag and the recipe that builds column names from them are
# in conftest: the other prediction module asserts against the same two
# constants, and a feature list that disagreed between the two would have
# one of them testing an artifact the other cannot load.
from conftest import ALIAS, LAG, feature_names  # noqa: E402


@pytest.fixture
def artifact(tmp_path):
    names = feature_names()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, len(names))), columns=names)
    model = sklearn_linear.Ridge().fit(frame, frame.iloc[:, 0] * 2 + 5)
    path = tmp_path / "ridge_mae-0.4321_evals-0.sav"
    joblib.dump(model, path)
    return path


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as c:
        seed_soundings(c)
        yield c


def seed_soundings(conn, days: int = 4, tx: str = "NIC", rx: str = "DOB"):
    """Four days of five-minute soundings with a diurnal MUF and a clean LOF."""
    index = pd.date_range("2026-08-10", periods=288 * days, freq="5min")
    t = np.arange(len(index))
    muf = 18 + 7 * np.sin(2 * np.pi * (t - 60) / 288)
    lof = 9 + 4 * np.sin(2 * np.pi * (t - 80) / 288)
    for position, stamp in enumerate(index):
        cursor = conn.execute(
            "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"s{position}.h5", f"s{position}.h5",
             stamp.strftime("%Y-%m-%d %H:%M:%S"), tx, rx, db.utcnow()),
        )
        conn.execute(
            "INSERT INTO extraction (sounding_id, method, muf, lof, snr, run, "
            "limited, loflim) VALUES (?,?,?,?,?,?,?,?)",
            (cursor.lastrowid, "contour", float(muf[position]),
             float(lof[position]), 55.0, 30, 0, 0),
        )
    conn.commit()


def register(conn, artifact, **kw):
    row = importer.import_artifact(artifact, param="muf", conn=conn, **kw)
    return row


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------

def test_a_legacy_import_is_comparison_only_by_default(conn, artifact):
    """The rule holds by default rather than by remembering to pass a flag."""
    row = register(conn, artifact, origin="legacy")
    assert row["target_src"] == "modelled"
    assert row["state"] == "comparison"
    with pytest.raises(registry.RegistryError):
        registry.activate(conn, row["id"])


def test_the_recipe_and_alias_are_recorded_at_import(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    assert row["target_alias"] == ALIAS
    assert row["feature_recipe"]["lag"] == LAG
    assert row["feature_recipe"]["period_assumed"] is True
    assert row["features"] == feature_names()


def test_the_golden_pair_is_stored_at_import(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    assert row["golden_output"] is not None
    assert len(row["golden_input"]) == len(row["features"])


def test_an_artifact_that_does_not_name_its_inputs_is_refused(conn, tmp_path):
    """Without names the column order is unknowable, and the wrong order
    returns a plausible number rather than an error."""
    model = sklearn_linear.Ridge().fit(np.zeros((10, 3)), np.zeros(10))
    path = tmp_path / "nameless.sav"
    joblib.dump(model, path)
    with pytest.raises(artifacts.ArtifactError, match="does not record the names"):
        register(conn, path, origin="legacy")


def test_names_can_be_supplied_for_an_artifact_that_lacks_them(conn, tmp_path):
    model = sklearn_linear.Ridge().fit(np.zeros((10, len(feature_names()))),
                                       np.zeros(10))
    path = tmp_path / "nameless.sav"
    joblib.dump(model, path)
    row = register(conn, path, origin="legacy", features=feature_names())
    assert row["features"] == feature_names()


def test_a_wrong_number_of_supplied_names_is_refused(conn, tmp_path):
    model = sklearn_linear.Ridge().fit(np.zeros((10, 4)), np.zeros(10))
    path = tmp_path / "nameless.sav"
    joblib.dump(model, path)
    with pytest.raises(artifacts.ArtifactError, match="expects 4 inputs"):
        register(conn, path, origin="legacy", features=feature_names())


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def test_inference_never_fits(conn, artifact, monkeypatch):
    """The regression test for the defect this service exists to not repeat.

    `xgb_evaluate` and `xgb_test` in the research project both load a saved
    model and call `.fit` before predicting. If that ever creeps in here, this
    fails loudly instead of the forecast quietly becoming a hindcast.
    """
    row = register(conn, artifact, origin="legacy")

    def explode(*args, **kwargs):
        raise AssertionError("inference must never call fit()")

    estimator, _ = artifacts.load(artifact)
    monkeypatch.setattr(type(estimator), "fit", explode)

    result = infer.run_model(conn, row, "NIC", "DOB", method="contour")
    assert result["written"] > 0


def test_a_run_writes_forecast_rows_with_their_quality(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    result = infer.run_model(conn, row, "NIC", "DOB", method="contour")

    stored = db.rows(conn, "SELECT * FROM forecast ORDER BY valid_at")
    assert len(stored) == result["written"] > 0
    quality = json.loads(stored[0]["quality"])
    assert quality["golden"] == "ok"
    assert quality["alias"] == ALIAS
    assert quality["method"] == "contour"


def test_the_horizon_is_lead_time_not_distance_from_the_run(conn, artifact):
    """A backtest over an old archive must not report negative horizons, and
    the same prediction must not change scoring bucket with the day it ran."""
    row = register(conn, artifact, origin="legacy")
    infer.run_model(conn, row, "NIC", "DOB", method="contour")

    horizons = {r["horizon_s"] for r in db.rows(conn, "SELECT horizon_s FROM forecast")}
    assert horizons == {LAG * 300}


def test_a_backtest_says_so(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    result = infer.run_model(conn, row, "NIC", "DOB", method="contour")
    assert result["backtest"] is True
    stored = db.rows(conn, "SELECT quality FROM forecast LIMIT 1")
    assert json.loads(stored[0]["quality"])["backtest"] is True


def test_a_cold_start_reports_and_does_not_raise(conn, artifact):
    """A fresh deployment has no trained model. That is normal, not a fault."""
    register(conn, artifact, origin="legacy")       # registered, not active
    results = infer.run_once(conn, ("muf",), method="contour")
    assert results
    assert all(r["written"] == 0 for r in results)
    assert all(r["detail"] == "no active model" for r in results)


def test_an_active_model_runs_in_the_ordinary_pass(conn, artifact):
    row = register(conn, artifact, origin="trained", tx="NIC", rx="DOB",
                   target_src="measured")
    registry.activate(conn, row["id"], by="test")

    results = infer.run_once(conn, ("muf",), method="contour")
    ran = [r for r in results if r.get("written")]
    assert len(ran) == 1
    assert ran[0]["tx"] == "NIC"


def test_reissuing_replaces_rather_than_duplicates(conn, artifact):
    """Two runs at the same instant are the same issue, not two."""
    row = register(conn, artifact, origin="legacy")
    stamp = db.utcnow()
    infer.run_model(conn, row, "NIC", "DOB", "contour", issued_at=stamp)
    first = db.rows(conn, "SELECT COUNT(*) AS n FROM forecast")[0]["n"]
    infer.run_model(conn, row, "NIC", "DOB", "contour", issued_at=stamp)
    assert db.rows(conn, "SELECT COUNT(*) AS n FROM forecast")[0]["n"] == first


def test_a_failing_golden_check_stops_the_run(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    conn.execute("UPDATE model_registry SET golden_output = ? WHERE id = ?",
                 (row["golden_output"] + 5.0, row["id"]))
    conn.commit()
    reloaded = registry.get(conn, row["id"])

    with pytest.raises(artifacts.ArtifactError, match="golden check failed"):
        infer.run_model(conn, reloaded, "NIC", "DOB", method="contour")


def test_skew_can_be_allowed_and_every_row_records_it(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    conn.execute("UPDATE model_registry SET golden_output = ? WHERE id = ?",
                 (row["golden_output"] + 5.0, row["id"]))
    conn.commit()
    reloaded = registry.get(conn, row["id"])

    infer.run_model(conn, reloaded, "NIC", "DOB", method="contour",
                    allow_skew=True)
    for stored in db.rows(conn, "SELECT quality FROM forecast"):
        quality = json.loads(stored["quality"])
        assert quality["golden"] == "failed"
        assert quality["version_skew"] is True


def test_a_circuit_with_no_data_is_reported_not_raised(conn, artifact):
    row = register(conn, artifact, origin="legacy")
    with pytest.raises(ValueError, match="no muf rows"):
        infer.run_model(conn, row, "NOWHERE", "DOB", method="contour")


def test_nan_predictions_land_as_null(conn, artifact, monkeypatch):
    """A NaN in a REAL column compares false against everything, so
    `WHERE value IS NOT NULL` would return rows with no value."""
    row = register(conn, artifact, origin="legacy")
    monkeypatch.setattr(infer.artifacts, "predict",
                        lambda estimator, frame: np.full(len(frame), np.nan))
    infer.run_model(conn, row, "NIC", "DOB", method="contour")
    total = db.rows(conn, "SELECT COUNT(*) AS n FROM forecast")[0]["n"]
    nulls = db.rows(conn,
                    "SELECT COUNT(*) AS n FROM forecast WHERE value IS NULL")[0]["n"]
    assert total > 0 and nulls == total


# --------------------------------------------------------------------------
# Writing to a database this process does not own
# --------------------------------------------------------------------------
#
# The failure these pin cost an afternoon on the local rig: `infer` ran as its
# own uid, the `api-data` volume was owned by the api's, and the container came
# up, read every row, answered every query and then crash-looped on its first
# write with "attempt to write a readonly database" -- about a file that is not
# read-only.

def test_a_writable_database_probes_clean_and_is_left_alone(conn):
    conn.execute("PRAGMA user_version = 7")
    assert infer.writable(conn) is None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 7


def test_a_database_this_process_cannot_write_is_named_as_such(tmp_path):
    path = tmp_path / "ro.sqlite3"
    with db.session(path):
        pass
    readonly = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    assert "readonly" in (infer.writable(readonly) or "")


def test_the_run_stops_at_the_start_rather_than_at_the_first_write(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    """Fail fast and name the remedy. The alternative is a traceback hours
    later from whichever line happened to write first."""
    monkeypatch.setattr(infer, "writable",
                        lambda conn: "attempt to write a readonly database")
    assert infer.main(["--once", "--db", str(tmp_path / "t.sqlite3")]) == 1
    err = capsys.readouterr().err
    assert "cannot write" in err
    assert "10001" in err, "the message has to name the uid to run as"


def test_a_scoring_failure_does_not_discard_the_forecasts(tmp_path, monkeypatch,
                                                          capsys):
    """Scoring runs after the forecasts are written. A database locked by the
    api mid-scan at that moment must not take the pass down with it."""
    def explode(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(infer.scoring, "run_once", explode)
    path = tmp_path / "t.sqlite3"
    with db.session(path) as c:
        seed_soundings(c)

    assert infer.main(["--once", "--method", "contour", "--db", str(path)]) == 0
    assert "scoring skipped: OperationalError" in capsys.readouterr().err
