"""The forecast surface: reading models and forecasts, and promoting a model.

Promotion is the one operation in this service that changes a published product
without touching a radio, which is exactly why it is tested at the route level
as well as in the registry: the question here is whether the *scope* is right,
not whether the SQL is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.api import auth, db, main, net
from services.api import series as series_mod
from services.prediction import importer, infer, registry

joblib = pytest.importorskip("joblib")
sklearn_linear = pytest.importorskip("sklearn.linear_model")
pytest.importorskip("statsmodels")

ALIAS = "MUF(3000)F2"
LAG = 288
CTL = {"Authorization": "Bearer ctl"}


def feature_names() -> list[str]:
    names = [f"{ALIAS}_lag_{LAG}"]
    names += [f"{ALIAS}_{c}_lag_{LAG}" for c in ("trend", "seasonal", "residual")]
    names += [f"{ALIAS}_rolling_{w}_{s}_lag_{LAG}"
              for w in (12, 48) for s in ("mean", "std")]
    names += ["hour", "minute"]
    return names


@pytest.fixture
def artifact(tmp_path):
    names = feature_names()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, len(names))), columns=names)
    model = sklearn_linear.Ridge().fit(frame, frame.iloc[:, 0] * 2 + 5)
    path = tmp_path / "ridge.sav"
    joblib.dump(model, path)
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    net.reset()
    monkeypatch.setattr(series_mod, "MODEL", False)
    series_mod.clear()
    with TestClient(main.app) as c:
        yield c


def seed(conn, tx="NIC", rx="DOB", days=4):
    index = pd.date_range("2026-08-10", periods=288 * days, freq="5min")
    t = np.arange(len(index))
    muf = 18 + 7 * np.sin(2 * np.pi * (t - 60) / 288)
    for position, stamp in enumerate(index):
        cursor = conn.execute(
            "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"{tx}{position}.h5", f"{tx}{position}.h5",
             stamp.strftime("%Y-%m-%d %H:%M:%S"), tx, rx, db.utcnow()),
        )
        conn.execute(
            "INSERT INTO extraction (sounding_id, method, muf, snr, run, limited) "
            "VALUES (?,?,?,?,?,?)",
            (cursor.lastrowid, "contour", float(muf[position]), 55.0, 30, 0),
        )
    conn.commit()


def measured(conn, name="xgb-live", tx="NIC", rx="DOB"):
    return registry.register(
        conn, name=name, param="muf", tx=tx, rx=rx, origin="trained",
        framework="xgboost", loader="joblib", capability="slim",
        artifact=f"/models/{name}.joblib", sha256=name, features=["a"],
        target_src="measured")


def test_models_lists_what_is_registered(client, artifact):
    conn = client.app.state.db
    seed(conn)
    importer.import_artifact(artifact, param="muf", origin="legacy", conn=conn)

    body = client.get("/models").json()
    assert body["count"] == 1
    row = body["models"][0]
    assert row["state"] == "comparison"
    assert row["target_src"] == "modelled"
    assert row["golden"] == "recorded"


def test_the_stored_golden_input_is_not_published(client, artifact):
    """It is an implementation detail of the check; the verdict is not."""
    conn = client.app.state.db
    importer.import_artifact(artifact, param="muf", origin="legacy", conn=conn)
    row = client.get("/models").json()["models"][0]
    assert "golden_input" not in row
    assert "golden_output" not in row


def test_forecast_returns_active_models_only_by_default(client, artifact):
    """Every model may write forecasts -- that is how comparison works -- so an
    unfiltered query would interleave the live curve with every candidate."""
    conn = client.app.state.db
    seed(conn)
    legacy = importer.import_artifact(artifact, param="muf", origin="legacy",
                                      conn=conn)
    infer.run_model(conn, legacy, "NIC", "DOB", method="contour")

    assert client.get("/forecast").json()["count"] == 0
    named = client.get(f"/forecast?model={legacy['id']}").json()
    assert named["count"] > 0
    assert named["points"][0]["model_name"] == legacy["name"]


def test_forecast_points_carry_their_quality_decoded(client, artifact):
    conn = client.app.state.db
    seed(conn)
    legacy = importer.import_artifact(artifact, param="muf", origin="legacy",
                                      conn=conn)
    infer.run_model(conn, legacy, "NIC", "DOB", method="contour")
    point = client.get(f"/forecast?model={legacy['id']}").json()["points"][0]
    assert point["quality"]["golden"] == "ok"
    assert point["quality"]["alias"] == ALIAS
    assert point["horizon_s"] == LAG * 300


def test_activation_needs_the_control_token(client, artifact):
    conn = client.app.state.db
    model = measured(conn)
    assert client.post(f"/models/{model}/activate").status_code == 401
    assert client.post(f"/models/{model}/activate", headers=CTL).status_code == 200


def test_a_read_token_does_not_grant_promotion(client, artifact, monkeypatch):
    monkeypatch.setattr(auth, "READ_TOKEN", "ro")
    conn = client.app.state.db
    model = measured(conn)
    response = client.post(f"/models/{model}/activate",
                           headers={"Authorization": "Bearer ro"})
    assert response.status_code == 401


def test_promoting_a_comparison_model_is_refused_with_a_reason(client, artifact):
    conn = client.app.state.db
    legacy = importer.import_artifact(artifact, param="muf", origin="legacy",
                                      conn=conn)
    response = client.post(f"/models/{legacy['id']}/activate", headers=CTL)
    assert response.status_code == 409
    assert "modelled" in response.json()["detail"]


def test_promotion_demotes_the_incumbent_and_says_so(client):
    conn = client.app.state.db
    first, second = measured(conn, "xgb-a"), measured(conn, "xgb-b")
    client.post(f"/models/{first}/activate", headers=CTL)

    body = client.post(f"/models/{second}/activate", headers=CTL).json()
    assert body["activated"]["name"] == "xgb-b"
    assert body["deactivated"]["name"] == "xgb-a"
    assert "replacing xgb-a" in body["detail"]


def test_retiring_keeps_the_row(client):
    conn = client.app.state.db
    model = measured(conn)
    client.post(f"/models/{model}/activate", headers=CTL)
    assert client.post(f"/models/{model}/retire", headers=CTL).status_code == 200
    assert client.get("/models").json()["models"][0]["active"] == 0


def test_an_active_models_forecast_is_the_default_one(client, artifact):
    conn = client.app.state.db
    seed(conn)
    row = importer.import_artifact(artifact, param="muf", origin="trained",
                                   tx="NIC", rx="DOB", target_src="measured",
                                   conn=conn)
    client.post(f"/models/{row['id']}/activate", headers=CTL)
    infer.run_model(conn, registry.get(conn, row["id"]), "NIC", "DOB",
                    method="contour")
    assert client.get("/forecast").json()["count"] > 0


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def test_a_leaderboard_needs_a_circuit(client):
    """MAE in MHz on a 2400 km path is not comparable with one on 700 km, so
    there is deliberately no combined view to mistake for a ranking."""
    body = client.get("/scores?param=muf")
    assert body.status_code == 400
    assert "per circuit" in body.json()["detail"]


def test_scores_hands_back_the_baselines_beside_the_models(client, artifact):
    """A caller cannot fetch a model's MAE without also being handed what that
    MAE has to beat."""
    from services.prediction import scoring

    conn = client.app.state.db
    seed(conn, days=6)
    row = importer.import_artifact(artifact, param="muf", origin="trained",
                                   tx="NIC", rx="DOB", target_src="measured",
                                   conn=conn)
    infer.run_model(conn, registry.get(conn, row["id"]), "NIC", "DOB",
                    method="contour")
    last = db.one(conn, "SELECT MAX(datetime) AS t FROM sounding")["t"]
    scoring.run_once(conn, ("muf",), "contour", window_days=30,
                     now=pd.Timestamp(last).strftime(db.TIME_FORMAT))

    body = client.get("/scores?param=muf&tx=NIC&rx=DOB").json()
    kinds = {entry["kind"] for entry in body["leaderboard"]}
    assert kinds == {"model", "baseline"}
    names = {e["name"] for e in body["leaderboard"] if e["kind"] == "baseline"}
    assert names == set(scoring.BASELINES)


def test_an_unavailable_baseline_states_why_rather_than_vanishing(client):
    from services.prediction import scoring

    conn = client.app.state.db
    seed(conn, days=6)
    last = db.one(conn, "SELECT MAX(datetime) AS t FROM sounding")["t"]
    scoring.run_once(conn, ("muf",), "contour", window_days=30,
                     now=pd.Timestamp(last).strftime(db.TIME_FORMAT))

    rows = client.get("/scores?param=muf&flat=1").json()["scores"]
    iri = [r for r in rows if r["subject"] == "baseline:iri"]
    assert iri, "a baseline with nothing to say must still have a row"
    assert "no IRI rows stored" in iri[0]["detail"]["unavailable"]


def test_the_forecast_page_draws_the_leaderboard(client, artifact):
    from services.prediction import scoring

    conn = client.app.state.db
    seed(conn, days=6)
    row = importer.import_artifact(artifact, param="muf", origin="trained",
                                   tx="NIC", rx="DOB", target_src="measured",
                                   conn=conn)
    infer.run_model(conn, registry.get(conn, row["id"]), "NIC", "DOB",
                    method="contour")
    last = db.one(conn, "SELECT MAX(datetime) AS t FROM sounding")["t"]
    scoring.run_once(conn, ("muf",), "contour", window_days=30,
                     now=pd.Timestamp(last).strftime(db.TIME_FORMAT))

    page = client.get("/ui/forecast")
    assert page.status_code == 200
    assert "leaderboard" in page.text
    assert "persistence" in page.text
    assert "recurrence-27d" in page.text
    assert "Not yet computed" not in page.text


def test_the_forecast_page_survives_having_nothing_scored(client):
    """The fresh-deployment case, which is the one an operator sees first."""
    seed(client.app.state.db)
    page = client.get("/ui/forecast")
    assert page.status_code == 200
    assert "Nothing scored for this circuit yet" in page.text


def test_the_series_page_draws_the_live_forecast_beside_the_measurement(client,
                                                                        artifact):
    """The point of the trace: what the model said, on the same axis as what
    arrived, including the part that reaches past the last sounding."""
    conn = client.app.state.db
    seed(conn, days=6)
    row = importer.import_artifact(artifact, param="muf", origin="trained",
                                   tx="NIC", rx="DOB", target_src="measured",
                                   conn=conn)
    client.post(f"/models/{row['id']}/activate", headers=CTL)
    infer.run_model(conn, registry.get(conn, row["id"]), "NIC", "DOB",
                    method="contour")

    page = client.get("/ui/series?method=contour&circuit=NIC+-%3E+DOB&model=off")
    assert page.status_code == 200
    assert 'data-family="forecast"' in page.text
    assert row["name"] in page.text
    # The Z is stripped: the forecast table stamps with it and the soundings do
    # not, and two axes an offset apart would draw the forecast lagging its own
    # measurement by the local timezone.
    assert "Z\"" not in page.text.split('"forecast"')[0][-4000:]


def test_a_comparison_models_forecast_is_not_drawn_on_the_series_page(client,
                                                                     artifact):
    """Same rule as `/forecast`: drawing a legacy import in the operational
    style is how one quietly becomes the forecast."""
    conn = client.app.state.db
    seed(conn, days=6)
    legacy = importer.import_artifact(artifact, param="muf", origin="legacy",
                                      conn=conn)
    infer.run_model(conn, legacy, "NIC", "DOB", method="contour")

    page = client.get("/ui/series?method=contour&circuit=NIC+-%3E+DOB&model=off")
    assert page.status_code == 200
    assert 'data-family="forecast"' not in page.text
