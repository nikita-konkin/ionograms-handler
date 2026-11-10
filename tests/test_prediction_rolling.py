"""Rolling windows must not shrink while features warm up or data goes stale."""

import json
from pathlib import Path

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


def test_the_status_names_the_newest_rolling_fit(conn, tmp_path):
    index = seed(conn, days=18)
    rolling.run_once(conn, spec(), now=index[-1], state_dir=tmp_path)
    rolling.run_once(conn, spec(), now=index[-1] + pd.Timedelta(minutes=10),
                     state_dir=tmp_path)
    found = rolling.status(conn, "muf", TX, RX, now=index[-1])
    assert found["name"] == registry.models(conn)[0]["name"], \
        "the status names an older fit"


def test_the_status_carries_the_validation_error(conn, tmp_path):
    index = seed(conn, days=18)
    trained = rolling.run_once(conn, spec(), now=index[-1], state_dir=tmp_path)
    found = rolling.status(conn, "muf", TX, RX, now=index[-1])
    assert found["mae"] == trained["validation"]["mae"], \
        "the status lost the validation error"


def test_the_status_says_the_worker_waits_on_old_data(conn, tmp_path):
    index = seed(conn, days=18)
    rolling.run_once(conn, spec(), now=index[-1], state_dir=tmp_path)
    found = rolling.status(conn, "muf", TX, RX,
                           now=index[-1] + pd.Timedelta(minutes=47))
    assert found["waiting"] is True, "a stalled worker was reported as fitting"


def test_there_is_no_status_without_a_rolling_fit(conn):
    seed(conn, days=2)
    assert rolling.status(conn, "muf", TX, RX) is None, \
        "a status appeared for a circuit nobody rolls"


def judged(verdict):
    """`train.run` as it is, with the comparison to the live forecast fixed."""
    real = train.run

    def run(*args, **kwargs):
        return {**real(*args, **kwargs), "vs_live": verdict}

    return run


WORSE = {"distinguishable": True, "delta": 0.4, "mae": 1.9,
         "baseline_mae": 1.5, "n": 200}


def test_a_fit_worse_than_the_live_forecast_is_not_activated(conn, tmp_path,
                                                             monkeypatch):
    index = seed(conn, days=18)
    monkeypatch.setattr(train, "run", judged(WORSE))
    monkeypatch.setattr(infer, "run_model", lambda *a, **kw: {"written": 1})
    rolling.run_once(conn, spec(), activate=True, now=index[-1],
                     state_dir=tmp_path)
    assert registry.active(conn, "muf", TX, RX) is None, \
        "a fit worse than the live forecast went live"


def test_a_refused_fit_is_not_retried_for_the_same_step(conn, tmp_path,
                                                        monkeypatch):
    index = seed(conn, days=18)
    monkeypatch.setattr(train, "run", judged(WORSE))
    monkeypatch.setattr(infer, "run_model", lambda *a, **kw: {"written": 1})
    rolling.run_once(conn, spec(), activate=True, now=index[-1],
                     state_dir=tmp_path)
    again = rolling.run_once(conn, spec(), activate=True, now=index[-1],
                             state_dir=tmp_path)
    assert again["state"] == "unchanged", "a refused step was fitted again"


def test_a_fit_level_with_the_live_forecast_is_activated(conn, tmp_path,
                                                         monkeypatch):
    index = seed(conn, days=18)
    monkeypatch.setattr(train, "run", judged({**WORSE, "distinguishable": False}))
    monkeypatch.setattr(infer, "run_model", lambda *a, **kw: {"written": 1})
    rolling.run_once(conn, spec(), activate=True, now=index[-1],
                     state_dir=tmp_path)
    assert registry.active(conn, "muf", TX, RX) is not None, \
        "a fit no worse than the live forecast was held back"


def stored(conn, name: str, active: bool = False) -> dict:
    """A registered model whose artifact really is in the store."""
    import hashlib

    from services.prediction import store

    body = name.encode() * 8
    digest = hashlib.sha256(body).hexdigest()
    source = store.root().parent / f"{name}.sav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(body)
    path = store.put(source, digest)
    model = registry.register(
        conn, name=name, param="muf", tx=TX, rx=RX, origin="trained",
        framework="sklearn", loader="joblib", capability="slim",
        artifact=str(path), sha256=digest, features=["a"],
        target_src="measured")
    if active:
        registry.activate(conn, model, by="test")
    return registry.get(conn, model)


def test_pruning_removes_the_oldest_rolling_artifacts(conn):
    old = stored(conn, "rolling-20261110T210000")
    for minute in (5, 10, 15):
        stored(conn, f"rolling-20261110T21{minute:02d}00")
    rolling.prune(conn, "muf", TX, RX, keep=2)
    assert not Path(old["artifact"]).exists(), "an old fit kept its artifact"


def test_pruning_keeps_the_newest_rolling_artifacts(conn):
    for minute in (0, 5, 10):
        newest = stored(conn, f"rolling-20261110T21{minute:02d}00")
    rolling.prune(conn, "muf", TX, RX, keep=2)
    assert Path(newest["artifact"]).exists(), "the newest fit lost its artifact"


def test_pruning_keeps_the_live_model(conn):
    live = stored(conn, "rolling-20261110T200000", active=True)
    for minute in (0, 5, 10):
        stored(conn, f"rolling-20261110T21{minute:02d}00")
    rolling.prune(conn, "muf", TX, RX, keep=1)
    assert Path(live["artifact"]).exists(), "the live model lost its artifact"


def test_pruning_spares_models_trained_by_hand(conn):
    manual = stored(conn, "voting-muf-24h-2026-10-31")
    for minute in (0, 5, 10):
        stored(conn, f"rolling-20261110T21{minute:02d}00")
    rolling.prune(conn, "muf", TX, RX, keep=1)
    assert Path(manual["artifact"]).exists(), "a hand-trained model was pruned"


def test_pruning_keeps_the_registry_rows_and_their_forecasts(conn):
    old = stored(conn, "rolling-20261110T210000")
    conn.execute("INSERT INTO forecast (model_id, param, tx, rx, issued_at, "
                 "valid_at, horizon_s, value) VALUES (?, 'muf', ?, ?, "
                 "'2026-11-10T21:00:00Z', '2026-11-11T21:00:00Z', 86400, 11.0)",
                 (old["id"], TX, RX))
    conn.commit()
    for minute in (5, 10):
        stored(conn, f"rolling-20261110T21{minute:02d}00")
    rolling.prune(conn, "muf", TX, RX, keep=1)
    assert conn.execute("SELECT COUNT(*) FROM forecast WHERE model_id = ?",
                        (old["id"],)).fetchone()[0] == 1, \
        "pruning deleted the forecasts the published record is built from"
