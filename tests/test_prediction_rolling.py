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


def forecast(conn, model: dict, issued: str) -> None:
    conn.execute("INSERT INTO forecast (model_id, param, tx, rx, issued_at, "
                 "valid_at, horizon_s, value) VALUES (?, 'muf', ?, ?, ?, "
                 "'2026-11-11T21:00:00Z', 86400, 11.0)",
                 (model["id"], TX, RX, issued))
    conn.commit()


def test_pruning_deletes_superseded_refits(conn):
    old = stored(conn, "rolling-20261110T210000", active=True)
    stored(conn, "rolling-20261110T210500", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert registry.get(conn, old["id"]) is None, "a superseded refit was kept"


def test_pruning_reaps_the_artifacts_of_deleted_refits(conn):
    old = stored(conn, "rolling-20261110T210000", active=True)
    stored(conn, "rolling-20261110T210500", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert not Path(old["artifact"]).exists(), "a deleted refit kept its file"


def test_pruning_keeps_the_live_model(conn):
    stored(conn, "rolling-20261110T210500", active=True)
    live = stored(conn, "rolling-20261110T200000", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert registry.get(conn, live["id"]) is not None, \
        "the live model was deleted"


def test_pruning_keeps_the_last_live_fit_of_a_past_month(conn):
    stored(conn, "rolling-20261030T120000", active=True)
    last = stored(conn, "rolling-20261031T235500", active=True)
    stored(conn, "rolling-20261110T210000", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert registry.get(conn, last["id"]) is not None, \
        "the month lost the fit that stands for it"


def test_pruning_keeps_one_fit_per_month(conn):
    for name in ("rolling-20261030T120000", "rolling-20261031T235500",
                 "rolling-20261110T205500", "rolling-20261110T210000"):
        stored(conn, name, active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert [model["name"] for model in registry.models(conn)] == [
        "rolling-20261110T210000", "rolling-20261031T235500"], \
        "pruning did not leave one fit per month"


def test_pruning_moves_a_deleted_fits_forecasts_onto_its_months_model(conn):
    old = stored(conn, "rolling-20261030T120000", active=True)
    forecast(conn, old, "2026-10-30T12:00:00Z")
    last = stored(conn, "rolling-20261031T235500", active=True)
    stored(conn, "rolling-20261110T210000", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert conn.execute("SELECT COUNT(*) FROM forecast WHERE model_id = ?",
                        (last["id"],)).fetchone()[0] == 1, \
        "a deleted fit's forecasts were not moved onto its month's model"


def test_pruning_drops_the_forecasts_of_fits_never_live(conn):
    candidate = stored(conn, "rolling-20261110T205500")
    forecast(conn, candidate, "2026-11-10T20:55:00Z")
    stored(conn, "rolling-20261110T210000", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert conn.execute("SELECT COUNT(*) FROM forecast").fetchone()[0] == 0, \
        "a fit that never went live left forecasts behind"


def test_pruning_spares_models_trained_by_hand(conn):
    manual = stored(conn, "voting-muf-24h-2026-10-31")
    for minute in (0, 5, 10):
        stored(conn, f"rolling-20261110T21{minute:02d}00", active=True)
    rolling.prune(conn, "muf", TX, RX)
    assert registry.get(conn, manual["id"]) is not None, \
        "a hand-trained model was pruned"


def daily(conn, tmp_path, at, period=2, activate=False):
    return rolling.run_once(conn, spec(), activate=activate,
                            now=pd.Timestamp(at), state_dir=tmp_path,
                            period=period)


def test_a_fit_trains_on_the_days_it_is_given():
    _, start, cut = rolling.window_plan(spec(), pd.Timestamp("2026-08-28"),
                                        days=6)
    assert cut - start == pd.Timedelta(days=6), \
        "the fit did not train on the days it was given"


def test_a_daily_fit_is_anchored_at_midnight(conn, tmp_path):
    seed(conn, days=18)
    found = daily(conn, tmp_path, "2026-08-28 00:10")
    assert found["validation_end"] == "2026-08-28 00:00:00", \
        "a daily fit was not anchored at midnight"


def test_a_daily_fit_waits_for_the_previous_day_to_be_measured(conn, tmp_path):
    seed(conn, days=18)
    conn.execute("DELETE FROM extraction WHERE sounding_id IN (SELECT id FROM "
                 "sounding WHERE datetime > '2026-08-27 20:00:00')")
    conn.commit()
    assert daily(conn, tmp_path, "2026-08-28 00:10")["state"] == "waiting", \
        "a daily fit ran on a day that was not measured to its end"


def test_a_daily_model_is_not_refitted_before_its_period_ends(conn, tmp_path):
    seed(conn, days=19)
    daily(conn, tmp_path, "2026-08-27 00:10")
    daily(conn, tmp_path, "2026-08-28 00:10")
    assert len(registry.models(conn)) == 1, \
        "a daily model was refitted before its period ended"


def test_a_daily_model_is_refitted_once_its_period_ends(conn, tmp_path):
    seed(conn, days=19)
    for at in ("2026-08-26 00:10", "2026-08-27 00:10", "2026-08-28 00:10"):
        daily(conn, tmp_path, at)
    assert len(registry.models(conn)) == 2, \
        "a daily model was not refitted once its period ended"


def test_a_daily_issue_covers_only_the_day_ahead(conn, tmp_path):
    seed(conn, days=19)
    daily(conn, tmp_path, "2026-08-28 00:10", activate=True)
    row = db.one(conn, "SELECT MAX(valid_at) AS last FROM forecast")
    assert row["last"] == "2026-08-28T23:55:00Z", \
        "a daily issue reached past the day ahead"


def test_a_daily_issue_is_made_once_a_day(conn, tmp_path):
    seed(conn, days=19)
    daily(conn, tmp_path, "2026-08-28 00:10", activate=True)
    again = daily(conn, tmp_path, "2026-08-28 00:20", activate=True)
    assert again["state"] == "unchanged", "a day was issued twice"


def test_the_next_day_is_issued_by_the_same_daily_model(conn, tmp_path):
    seed(conn, days=19)
    first = daily(conn, tmp_path, "2026-08-27 00:10", activate=True)
    second = daily(conn, tmp_path, "2026-08-28 00:10", activate=True)
    assert (second["state"], second["model_id"], bool(second["forecast"])) \
        == ("issued", first["model_id"], True), \
        "the next day was not issued by the model fitted for the period"


def test_the_status_carries_the_mode_of_the_fit(conn, tmp_path):
    seed(conn, days=18)
    daily(conn, tmp_path, "2026-08-28 00:10")
    found = rolling.status(conn, "muf", TX, RX,
                           now=pd.Timestamp("2026-08-28 00:10"))
    assert "every 2 days" in found["note"], "the status hides the daily mode"


def test_a_refused_daily_fit_leaves_the_day_to_the_live_model(conn, tmp_path,
                                                              monkeypatch):
    seed(conn, days=19)
    first = daily(conn, tmp_path, "2026-08-27 00:10", period=1, activate=True)
    monkeypatch.setattr(train, "run", judged(WORSE))
    second = daily(conn, tmp_path, "2026-08-28 00:10", period=1, activate=True)
    assert second["forecast"]["model"] == registry.get(
        conn, first["model_id"])["name"], \
        "the live model did not issue the day its successor was refused"
