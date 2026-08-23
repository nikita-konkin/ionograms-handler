"""Fitting a model on this instrument's own measurements.

Three of these tests are the reason the module exists at all, and each pins a
way of not fooling yourself:

* `test_the_target_is_measured_never_tracked` -- the Kalman filter will supply
  a value at every grid instant, and fitting to those teaches a model the
  filter. The target comes from the picks.
* `test_band_edge_picks_are_kept_out_of_the_fit` -- a `limited` MUF is a lower
  bound. Regressing onto it teaches the model the sweep ceiling, hardest at
  midday, which is exactly where an operator needs the number.
* `test_the_holdout_is_the_tail_and_never_a_shuffle` -- every feature is a
  lagged function of the target, so a random split puts each row's own future
  in the training set and the MAE that comes back measures leakage.

The rest are the contract: that what comes out is registered as `trained` and
`measured`, bound to its circuit, *not* activated, with the feature order it
was fitted in recoverable from the artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.api import db
from services.prediction import (artifacts, dataset, legacy_features, queues,
                                 registry, store, train, trainer)

pytest.importorskip("joblib")
pytest.importorskip("sklearn.linear_model")
pytest.importorskip("statsmodels")

TX, RX = "NIC3", "Yoshkar-Ola"


def seed(conn, days: int = 8, tx: str = TX, rx: str = RX,
         censor: slice | None = None, gaps: slice | None = None):
    """Five-minute soundings with a diurnal MUF, optionally censored or absent.

    `censor` marks a span as band-edge picks; `gaps` leaves the extraction row
    out entirely, so the tracker has to fill those instants.
    """
    index = pd.date_range("2026-08-10", periods=288 * days, freq="5min")
    t = np.arange(len(index))
    rng = np.random.default_rng(11)
    muf = 18 + 7 * np.sin(2 * np.pi * (t - 60) / 288) + rng.normal(0, 0.25, len(t))
    censored = np.zeros(len(t), dtype=bool)
    if censor is not None:
        censored[censor] = True
    missing = np.zeros(len(t), dtype=bool)
    if gaps is not None:
        missing[gaps] = True

    for position, stamp in enumerate(index):
        cursor = conn.execute(
            "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"s{position}.h5", f"s{position}.h5",
             stamp.strftime("%Y-%m-%d %H:%M:%S"), tx, rx, db.utcnow()))
        if missing[position]:
            continue
        conn.execute(
            "INSERT INTO extraction (sounding_id, method, muf, lof, snr, run, "
            "limited, loflim) VALUES (?,?,?,?,?,?,?,?)",
            (cursor.lastrowid, "contour", float(muf[position]), 9.0, 55.0, 30,
             int(censored[position]), 0))
    conn.commit()
    return index


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STORE", str(tmp_path / "models"))
    with db.session(tmp_path / "t.sqlite3") as connection:
        yield connection


def plan(**overrides) -> dict:
    spec = {"param": "muf", "tx": TX, "rx": RX, "lead_h": 24,
            "estimator": "huber", "holdout_days": 2, **overrides}
    checked = train.vet(spec)
    return {**checked["spec"], "param": checked["param"], "tx": checked["tx"],
            "rx": checked["rx"], "method": checked["method"],
            "lag": checked["lag"], "lead_h": checked["lead_h"]}


# --------------------------------------------------------------------------
# Vetting
# --------------------------------------------------------------------------

def test_a_lead_in_hours_becomes_a_lag_in_samples():
    checked = train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24})
    assert checked["lag"] == 24 * 3600 // dataset.DEFAULT_STEP_S == 288
    assert checked["lead_h"] == 24


def test_a_model_with_no_circuit_is_refused_at_the_door():
    """It could never be promoted, so training it would be pointless."""
    with pytest.raises(train.TrainError, match="bound to a circuit"):
        train.vet({"param": "muf", "tx": "", "rx": RX, "lead_h": 24})


def test_a_lead_shorter_than_one_sample_is_refused_with_the_arithmetic():
    with pytest.raises(train.TrainError, match="under one 300 s sample"):
        train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 0.01})


def test_decomposition_components_are_refused_at_a_short_lag():
    """`build` raises on this too; refusing here means the operator finds out
    while looking at the form rather than a minute later in a worker log."""
    with pytest.raises(train.TrainError, match="after the instant being predicted"):
        train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 1,
                   "components": ["trend"]})


def test_a_recipe_with_no_lagged_features_is_refused():
    with pytest.raises(train.TrainError, match="not a forecast of the ionosphere"):
        train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24,
                   "raw": False, "windows": [], "stats": [], "time": ["hour"]})


def test_a_zero_holdout_is_refused():
    with pytest.raises(train.TrainError, match="no reported accuracy at all"):
        train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24,
                   "holdout_days": 0})


def test_an_unknown_estimator_names_the_ones_there_are():
    with pytest.raises(train.TrainError, match="huber"):
        train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24,
                   "estimator": "randomforest"})


# --------------------------------------------------------------------------
# The recipe
# --------------------------------------------------------------------------

def test_the_feature_order_round_trips_through_parse():
    """The order is the contract.

    It is what `feature_names_in_` records, what the frame is built in, and
    what `legacy_features.parse` recovers on import. Reordering columns does
    not raise -- it multiplies the wrong coefficient by the wrong number.
    """
    recipe = train.recipe_for(plan(windows=[12, 48], stats=["std", "mean"],
                                   time=["hour", "minute"]))
    recovered = legacy_features.parse(recipe.features, period=recipe.period,
                                      assumed=False)

    assert recovered == recipe
    assert recipe.features[0] == "muf_lag_288", "the raw lag leads the frame"
    assert recipe.features[1:5] == (
        "muf_rolling_12_mean_lag_288", "muf_rolling_12_std_lag_288",
        "muf_rolling_48_mean_lag_288", "muf_rolling_48_std_lag_288")
    assert recipe.features[-2:] == ("hour", "minute")


def test_a_trained_recipe_does_not_claim_the_period_was_assumed():
    """It is assumed for every legacy artifact and known for exactly this one."""
    assert train.recipe_for(plan()).period_assumed is False
    assert legacy_features.parse(("muf_lag_288",)).period_assumed is True


# --------------------------------------------------------------------------
# The three that matter
# --------------------------------------------------------------------------

def test_the_target_is_measured_never_tracked(conn):
    """A grid instant with no pick behind it is not a training row.

    The tracker will supply a value there -- that is what it is for -- and it
    is an estimate. Fitting to estimates teaches the model the filter.
    """
    seed(conn, days=8, gaps=slice(288 * 4, 288 * 5))
    parts = train.assemble(conn, plan())

    series = parts["series"]
    assert series.n_filled > 0, "the fixture did not actually leave a gap"
    assert len(series.frame) > len(parts["X"]), \
        "every grid instant became a training row, gaps included"

    # Every kept row is backed by a pick within the match tolerance.
    picks = parts["observed"].index
    distance = np.abs(pd.DatetimeIndex(parts["X"].index)
                      - picks[picks.get_indexer(parts["X"].index, method="nearest")])
    assert (distance <= pd.Timedelta(seconds=dataset.DEFAULT_STEP_S / 2)).all()


def test_band_edge_picks_are_kept_out_of_the_fit_and_kept_in_the_score(conn):
    """A `limited` MUF is a lower bound, not a measurement.

    It is excluded from the regression and counted separately in the holdout,
    exactly as `scoring.summarise` counts it on the leaderboard -- so the
    number here is comparable with the number there.
    """
    seed(conn, days=10, censor=slice(288 * 7, 288 * 8))
    parts = train.assemble(conn, plan(holdout_days=2))
    assert parts["censored"].any(), "the fixture censored nothing that paired"

    result = train.run(conn, plan(holdout_days=2))
    assert result["n_censored"] > 0
    assert result["n_train"] < len(parts["X"]), "bounds were fitted to"
    assert result["holdout"]["n_censored"] > 0
    assert result["holdout"]["mae_censored"] is not None


def test_the_holdout_is_the_tail_and_never_a_shuffle(conn):
    seed(conn, days=10)
    parts = train.assemble(conn, plan(holdout_days=3))
    index = pd.DatetimeIndex(parts["X"].index)

    cut = train.split(index, 3)

    assert cut == index.max() - pd.Timedelta(days=3)
    # Contiguous on both sides: no training row sits after a holdout row.
    assert index[index < cut].max() < index[index >= cut].min()


# --------------------------------------------------------------------------
# What comes out
# --------------------------------------------------------------------------

def test_a_trained_model_is_measured_bound_and_not_activated(conn):
    seed(conn, days=8)
    result = train.run(conn, plan())
    model = result["model"]

    assert model["origin"] == "trained"
    assert model["target_src"] == "measured"
    assert (model["tx"], model["rx"]) == (TX, RX)
    assert model["active"] == 0, "training promoted a model on its own"
    assert registry.active(conn, "muf", TX, RX) is None

    # Eligible, though: it satisfies both promotion CHECKs, which is the whole
    # difference between this and every legacy import on the rig.
    registry.activate(conn, model["id"], by="test")
    assert registry.active(conn, "muf", TX, RX)["id"] == model["id"]


def test_the_artifact_lands_in_the_store_under_its_own_digest(conn):
    seed(conn, days=8)
    model = train.run(conn, plan())["model"]

    assert str(store.path_for(model["sha256"])) == model["artifact"]
    assert store.verify(model["sha256"])
    assert artifacts.sha256(model["artifact"]) == model["sha256"]


def test_the_training_window_is_recorded(conn):
    """`trained_from`/`trained_to` have been in the schema since it was written
    and nothing had ever set them."""
    seed(conn, days=8)
    model = train.run(conn, plan())["model"]

    assert model["trained_from"] and model["trained_to"]
    assert model["trained_from"] < model["trained_to"]
    # The window is the *fitted* span, so it ends before the holdout begins.
    assert model["trained_to"] < model["imported_at"]


def test_the_holdout_numbers_survive_the_first_scoring_pass(conn):
    """`scoring` and `train` share the metrics column and neither owns it.

    A replacing write would mean the first scoring pass silently erased the
    only record of what the model was accepted on.
    """
    seed(conn, days=8)
    model = train.run(conn, plan())["model"]
    assert model["metrics"]["holdout"]["mae"] is not None

    registry.set_metrics(conn, model["id"], {"by_horizon": {"86400": 0.42}})

    metrics = registry.get(conn, model["id"])["metrics"]
    assert metrics["by_horizon"] == {"86400": 0.42}
    assert metrics["holdout"]["mae"] is not None


def test_the_artifact_records_the_columns_it_was_fitted_on(conn):
    """The round trip that makes the model runnable at all."""
    seed(conn, days=8)
    model = train.run(conn, plan())["model"]

    estimator, contract = artifacts.load(model["artifact"])
    assert list(contract.features) == model["features"]
    assert contract.capability == "slim", "a linear model needs no training image"
    recovered = legacy_features.parse(contract.features, assumed=False)
    assert recovered.lag == 288 and recovered.alias == "muf"


def test_persistence_is_measured_over_the_same_holdout(conn):
    """A model that cannot beat "the value one lead ago" is not worth promoting,
    and finding that out at fit time is cheaper than a week later."""
    seed(conn, days=10)
    result = train.run(conn, plan())

    assert result["persistence"] is not None
    assert result["persistence"]["n"] > 0
    assert "persistence" in train.describe(result)


# --------------------------------------------------------------------------
# Refusals that are data limits, not faults
# --------------------------------------------------------------------------

def test_too_short_an_archive_is_refused_with_the_arithmetic(conn):
    seed(conn, days=3)
    with pytest.raises(train.TrainError) as raised:
        train.run(conn, plan(lead_h=120))

    message = str(raised.value)
    assert "grid points" in message and "days" in message
    assert "data limit, not a fault" in message


def test_too_few_rows_before_the_cut_is_refused_and_says_what_to_change(conn):
    seed(conn, days=6)
    with pytest.raises(train.TrainError) as raised:
        train.run(conn, plan(holdout_days=4.5))

    message = str(raised.value)
    assert "holdout cut" in message
    assert "Shorten the holdout" in message


def test_a_circuit_with_no_picks_at_all_is_refused(conn):
    seed(conn, days=8)
    with pytest.raises(train.TrainError, match="no muf rows"):
        train.run(conn, plan(tx="SGO", rx="DOB"))


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------

def test_the_trainer_claims_one_job_and_settles_it(conn):
    seed(conn, days=8)
    checked = train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24})
    queues.add_job(conn, param=checked["param"], tx=checked["tx"],
                   rx=checked["rx"], method=checked["method"],
                   spec=checked["spec"], by="test")

    settled = trainer.run_once(conn)

    assert len(settled) == 1
    assert settled[0]["state"] == "done", settled[0]["detail"]
    assert settled[0]["model_id"] is not None
    assert "holdout MAE" in settled[0]["detail"]


def test_a_job_that_cannot_work_fails_with_a_sentence_not_a_crash(conn):
    seed(conn, days=3)
    checked = train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 120})
    queues.add_job(conn, param=checked["param"], tx=checked["tx"],
                   rx=checked["rx"], method=checked["method"],
                   spec=checked["spec"])

    settled = trainer.run_once(conn)[0]

    assert settled["state"] == "failed"
    assert "data limit, not a fault" in settled["detail"]
    assert registry.models(conn) == []


def test_claiming_is_atomic_so_one_job_cannot_be_taken_twice(conn):
    seed(conn, days=8)
    checked = train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24})
    queues.add_job(conn, param="muf", tx=TX, rx=RX, spec=checked["spec"])

    first = queues.claim_job(conn)
    second = queues.claim_job(conn)

    assert first is not None and first["state"] == "running"
    assert second is None, "the same queued job was claimed twice"


def test_a_running_job_cannot_be_cancelled_out_from_under_the_fit(conn):
    checked = train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24})
    job = queues.add_job(conn, param="muf", tx=TX, rx=RX, spec=checked["spec"])

    assert queues.cancel_job(conn, job["id"])["state"] == "cancelled"

    job = queues.add_job(conn, param="muf", tx=TX, rx=RX, spec=checked["spec"])
    queues.claim_job(conn)
    with pytest.raises(queues.QueueError, match="runs to its end"):
        queues.cancel_job(conn, job["id"])
