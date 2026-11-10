"""Rolling windows must not shrink while features warm up or data goes stale."""

import json

import numpy as np
import pandas as pd
import pytest
from test_prediction_train import RX, TX, plan, seed

from services.api import db
from services.prediction import infer, registry, rolling, train


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STORE", str(tmp_path / "models"))
    with db.session(tmp_path / "rolling.sqlite3") as connection:
        yield connection


def spec():
    return {
        "param": "muf",
        "tx": TX,
        "rx": RX,
        "method": "contour",
        "lead_h": 24,
        "estimator": "ridge",
        "components": [],
        "windows": [12],
        "stats": ["mean"],
    }


def test_exact_13_days_of_targets_plus_one_day_of_validation(conn):
    index = seed(conn, days=18)
    anchor = index[-1] + pd.Timedelta(minutes=5)
    requested, start, cut = rolling.window_plan(spec(), anchor)
    result = train.run(
        conn, requested, training_start=start, validation_start=cut, diagnostics=False
    )
    assert result["n_train"] == 13 * 288
    assert result["holdout"]["n"] == 288
    assert pd.Timestamp(result["model"]["trained_from"]).tz_localize(None) == start
    assert pd.Timestamp(result["model"]["trained_to"]).tz_localize(
        None
    ) == cut - pd.Timedelta(minutes=5)
    assert result["cut"] == cut
    assert result["model"]["active"] == 0


def test_unchanged_data_is_skipped_across_restarts(conn, tmp_path):
    index = seed(conn, days=18)
    first = rolling.run_once(conn, spec(), now=index[-1], state_dir=tmp_path)
    again = rolling.run_once(conn, spec(), now=index[-1], state_dir=tmp_path)
    assert first["state"] == "trained"
    assert again["state"] == "unchanged"
    assert first["model_id"] == again["model_id"]
    assert len(registry.models(conn)) == 1


def test_new_step_moves_both_windows_by_five_minutes(conn, tmp_path):
    index = seed(conn, days=18)
    a = rolling.run_once(conn, spec(), now=index[-2], state_dir=tmp_path)
    b = rolling.run_once(conn, spec(), now=index[-1], state_dir=tmp_path)
    for key in ("anchor", "train_start", "validation_start"):
        assert pd.Timestamp(b[key]) - pd.Timestamp(a[key]) == pd.Timedelta(minutes=5)


def test_stale_or_missing_data_does_not_fit(conn, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not train on missing or stale data")

    monkeypatch.setattr(train, "run", forbidden)
    assert rolling.run_once(conn, spec(), state_dir=tmp_path)["state"] == "waiting"
    index = seed(conn, days=18)
    assert (
        rolling.run_once(
            conn, spec(), now=index[-1] + pd.Timedelta(hours=1), state_dir=tmp_path
        )["state"]
        == "waiting"
    )


def test_lock_prevents_overlapping_fit(tmp_path):
    path = tmp_path / "worker.lock"
    with rolling.exclusive(path) as a, rolling.exclusive(path) as b:
        assert a and not b
    with rolling.exclusive(path) as c:
        assert c


def test_forecast_failure_keeps_incumbent(conn, tmp_path, monkeypatch):
    index = seed(conn, days=18)
    old = train.run(conn, plan(estimator="ridge"))["model"]
    registry.activate(conn, old["id"])
    monkeypatch.setattr(infer, "run_model", lambda *a, **kw: {"written": 0})
    with pytest.raises(train.TrainError, match="incumbent kept"):
        rolling.run_once(conn, spec(), activate=True, now=index[-1], state_dir=tmp_path)
    assert registry.active(conn, "muf", TX, RX)["id"] == old["id"]
    assert not list(tmp_path.glob("*.json"))


def test_success_writes_forecast_before_activation(conn, tmp_path, monkeypatch):
    index = seed(conn, days=18)
    seen = []

    def predict(connection, model, *args, **kwargs):
        assert registry.active(connection, "muf", TX, RX) is None
        seen.append(model["id"])
        return {"written": 1}

    monkeypatch.setattr(infer, "run_model", predict)
    result = rolling.run_once(
        conn, spec(), activate=True, now=index[-1], state_dir=tmp_path
    )
    assert registry.active(conn, "muf", TX, RX)["id"] == seen[0]
    assert result["activated"]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["activated"]


def test_insufficient_validation_never_activates(conn, tmp_path, monkeypatch):
    index = seed(conn, days=18)
    monkeypatch.setattr(
        train, "run", lambda *a, **kw: {"holdout": {"n": 3, "mae": np.nan}}
    )
    with pytest.raises(train.TrainError, match="validation requires"):
        rolling.run_once(conn, spec(), activate=True, now=index[-1], state_dir=tmp_path)
    assert registry.active(conn, "muf", TX, RX) is None


def test_rolling_forecasts_do_not_duplicate_archive_backtests(conn):
    index = seed(conn, days=18)
    requested, start, cut = rolling.window_plan(
        spec(), index[-1] + pd.Timedelta(minutes=5)
    )
    model = train.run(
        conn, requested, training_start=start, validation_start=cut, diagnostics=False
    )["model"]
    issued = index[-1].strftime(db.TIME_FORMAT)
    forecast = infer.run_model(
        conn, model, TX, RX, issued_at=issued, valid_after=index[-1]
    )
    rows = db.rows(
        conn, "SELECT valid_at FROM forecast WHERE model_id = ?", (model["id"],)
    )
    assert forecast["written"] == 288
    assert all(pd.Timestamp(r["valid_at"]).tz_localize(None) > index[-1] for r in rows)


def test_warmup_must_not_enter_the_fit_even_if_its_values_change(conn):
    index = seed(conn, days=18)
    requested, start, cut = rolling.window_plan(
        spec(), index[-1] + pd.Timedelta(minutes=5)
    )
    parts = train.assemble(conn, requested)
    assert parts["X"].index.min() < start  # context can build earlier rows
    result = train.run(
        conn, requested, training_start=start, validation_start=cut, diagnostics=False
    )
    assert result["n_train"] == 13 * 288
    assert result["holdout"]["n"] == 288
