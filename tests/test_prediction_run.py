"""Asking for a forecast from the console.

Activating a model does not produce one -- `infer` does, on its next pass, and
that pass is six hours apart by default. Before this queue the only way to
close that gap was a shell on the host, which left the console covering every
step of a model's life except the one that produces the curve.

The wall is the same as the other two queues and is tested the same way:
`test_the_api_issues_nothing_itself` asserts that a queued pass writes no
`forecast` rows until a worker runs it. The api loads no artifact, here or
anywhere -- `test_prediction_upload.py::test_the_api_never_unpickles` reads the
syntax tree of `services/api/` and now covers this route too.
"""

from __future__ import annotations

import pytest

from services.api import db
from services.prediction import infer, queues, registry, train

pytest.importorskip("joblib")
pytest.importorskip("sklearn.linear_model")
pytest.importorskip("statsmodels")

from test_prediction_train import RX, TX, seed  # noqa: E402

AUTH = {"Authorization": "Bearer ctl"}


@pytest.fixture
def rig(client, tmp_path, monkeypatch):
    """The api over an archive with one trained, activated model."""
    monkeypatch.setenv("MODEL_STORE", str(tmp_path / "models"))
    conn = client.app.state.db
    seed(conn, days=8)

    checked = train.vet({"param": "muf", "tx": TX, "rx": RX, "lead_h": 24})
    plan = {**checked["spec"], "param": checked["param"], "tx": checked["tx"],
            "rx": checked["rx"], "method": checked["method"],
            "lag": checked["lag"], "lead_h": checked["lead_h"]}
    model = train.run(conn, plan)["model"]
    return {"client": client, "conn": conn, "model": model}


def ask(rig, headers=AUTH, **body):
    payload = {"param": "muf", "tx": TX, "rx": RX, **body}
    return rig["client"].post("/models/run", json=payload, headers=headers)


def forecast_rows(conn) -> int:
    return db.one(conn, "SELECT COUNT(*) AS n FROM forecast")["n"]


# --------------------------------------------------------------------------
# The door
# --------------------------------------------------------------------------

def test_queueing_a_pass_needs_the_control_token(rig):
    """It writes what every consumer of /forecast reads next."""
    registry.activate(rig["conn"], rig["model"]["id"])
    assert ask(rig, headers={}).status_code == 401
    assert queues.runs(rig["conn"]) == []


def test_a_circuit_with_nothing_live_is_refused_with_the_reason(rig):
    """"Queued, then done, wrote 0 rows" is a worse answer than a sentence."""
    response = ask(rig)

    assert response.status_code == 409
    assert "no MUF model is live" in response.json()["detail"]
    assert queues.runs(rig["conn"]) == []


def test_a_pass_needs_a_circuit(rig):
    registry.activate(rig["conn"], rig["model"]["id"])
    assert ask(rig, tx="").status_code == 400
    assert ask(rig, param="foF2").status_code == 400


def test_naming_a_model_that_does_not_exist_is_a_404(rig):
    assert ask(rig, model_id=999).status_code == 404


# --------------------------------------------------------------------------
# The wall
# --------------------------------------------------------------------------

def test_the_api_issues_nothing_itself(rig):
    """The load-bearing one, and the same property the other two queues have.

    A queued pass has written no forecast rows, because issuing one means
    loading the artifact and the api does not do that.
    """
    registry.activate(rig["conn"], rig["model"]["id"])
    assert forecast_rows(rig["conn"]) == 0

    response = ask(rig)

    assert response.status_code == 200, response.text
    assert response.json()["run"]["state"] == "queued"
    assert forecast_rows(rig["conn"]) == 0, "the api issued a forecast itself"


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------

def test_draining_the_queue_issues_the_forecast(rig):
    registry.activate(rig["conn"], rig["model"]["id"])
    run = ask(rig).json()["run"]

    settled = infer.drain(rig["conn"])

    assert len(settled) == 1
    assert settled[0]["id"] == run["id"]
    assert settled[0]["state"] == "done", settled[0]["detail"]
    assert settled[0]["written"] > 0
    assert forecast_rows(rig["conn"]) == settled[0]["written"]


def test_a_pass_over_an_archive_that_ends_in_the_past_says_backtest(rig):
    """Not an error, and not hidden.

    A lagged model run over a finished archive predicts instants that have
    already happened. `infer.run_model` labels that, and the label has to reach
    the page or the operator reads a forecast as being about the future.
    """
    registry.activate(rig["conn"], rig["model"]["id"])
    ask(rig)

    settled = infer.drain(rig["conn"])[0]

    assert settled["backtest"] == 1
    assert "backtest" in settled["detail"]


def test_a_named_model_runs_as_a_comparison_without_being_promoted(rig):
    """The shell's `infer --model <id>` as a button, with the same rule.

    Nothing is activated by running it, so a comparison stays a comparison.
    """
    model = rig["model"]
    response = ask(rig, model_id=model["id"])

    assert response.status_code == 200
    assert "comparison" in response.json()["detail"]

    settled = infer.drain(rig["conn"])[0]

    assert settled["state"] == "done", settled["detail"]
    assert settled["written"] > 0
    assert registry.get(rig["conn"], model["id"])["active"] == 0


def test_a_pass_that_cannot_run_fails_with_a_sentence_not_a_crash(rig):
    """A model whose artifact has gone is an outcome, not a dead worker."""
    from pathlib import Path

    registry.activate(rig["conn"], rig["model"]["id"])
    artifact = Path(rig["model"]["artifact"])
    artifact.chmod(0o644)
    artifact.unlink()
    ask(rig)

    settled = infer.drain(rig["conn"])[0]

    assert settled["state"] == "failed"
    assert "artifact not found" in settled["detail"]
    assert forecast_rows(rig["conn"]) == 0


def test_claiming_is_atomic_so_one_pass_cannot_run_twice(rig):
    registry.activate(rig["conn"], rig["model"]["id"])
    ask(rig)

    first = queues.claim_run(rig["conn"])
    second = queues.claim_run(rig["conn"])

    assert first is not None and first["state"] == "running"
    assert second is None, "the same queued pass was claimed twice"


def test_the_sleep_is_sliced_so_a_request_is_not_stuck_behind_the_interval(
        rig, tmp_path, monkeypatch):
    """The whole point of the change.

    `infer` used to sleep the entire interval in one call, so a button would
    have waited up to six hours. `_wait` now serves the queue between slices
    and still returns when the interval is up.
    """
    import time

    registry.activate(rig["conn"], rig["model"]["id"])
    run = ask(rig).json()["run"]

    monkeypatch.setattr(infer, "POLL_S", 0.05)
    started = time.monotonic()
    infer._wait(0.2, str(tmp_path / "api.sqlite3"), False)
    elapsed = time.monotonic() - started

    assert elapsed < 5, "the interval was not the thing being waited on"
    assert queues.run(rig["conn"], run["id"])["state"] == "done"


# --------------------------------------------------------------------------
# Narrowing, and reading it back
# --------------------------------------------------------------------------

def test_a_circuit_that_was_not_asked_for_is_not_run(rig):
    """`--tx` on its own used to be accepted and then ignored.

    That reads as "the circuit has no data" rather than "the flag did
    nothing", which is the worst kind of quiet.
    """
    seed(rig["conn"], days=8, tx="SGO", rx="DOB")
    registry.activate(rig["conn"], rig["model"]["id"])

    results = infer.run_once(rig["conn"], ("muf",), "contour",
                             tx=TX, rx=RX)

    assert results, "the requested circuit was skipped too"
    assert {(r["tx"], r["rx"]) for r in results} == {(TX, RX)}


def test_the_console_can_list_passes_without_a_token(rig):
    registry.activate(rig["conn"], rig["model"]["id"])
    ask(rig)

    body = rig["client"].get("/models/runs").json()

    assert body["count"] == 1
    assert body["runs"][0]["state"] == "queued"


def test_a_queued_pass_can_be_cancelled_and_a_running_one_cannot(rig):
    registry.activate(rig["conn"], rig["model"]["id"])
    run = ask(rig).json()["run"]

    response = rig["client"].delete(f"/models/runs/{run['id']}", headers=AUTH)
    assert response.status_code == 200
    assert queues.run(rig["conn"], run["id"])["state"] == "cancelled"

    run = ask(rig).json()["run"]
    queues.claim_run(rig["conn"])
    response = rig["client"].delete(f"/models/runs/{run['id']}", headers=AUTH)
    assert response.status_code == 409
    assert "runs to its end" in response.json()["detail"]


def test_cancelling_needs_the_control_token(rig):
    registry.activate(rig["conn"], rig["model"]["id"])
    run = ask(rig).json()["run"]

    assert rig["client"].delete(f"/models/runs/{run['id']}").status_code == 401
    assert queues.run(rig["conn"], run["id"])["state"] == "queued"
