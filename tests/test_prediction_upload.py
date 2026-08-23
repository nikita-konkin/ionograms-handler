"""Adding a model from the console, and the wall that keeps it honest.

`services/prediction/importer.py` refuses an HTTP import route because loading
a `.sav` runs code out of the file. The console has an upload button anyway,
and the way both are true is that the surface is split rather than opened: the
api hashes bytes and writes them to a quarantine volume without opening them,
and a worker that listens on nothing is what opens them.

**`test_the_api_registers_nothing_itself` is the load-bearing test here**, in
the way `test_inference_never_fits` is in `test_prediction_infer`. It asserts
that a successful upload leaves the registry empty. Two syntax-tree checks
back it up, because a property enforced only by a docstring is a property
somebody deletes in a refactor and nobody notices.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.api import db
from services.prediction import artifacts, queues, registrar, registry, store

joblib = pytest.importorskip("joblib")
sklearn_linear = pytest.importorskip("sklearn.linear_model")
pytest.importorskip("statsmodels")

from conftest import ALIAS, LAG, feature_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUTH = {"Authorization": "Bearer ctl"}


@pytest.fixture
def artifact_bytes(tmp_path):
    """A real joblib pickle that `importer.import_artifact` will accept."""
    names = feature_names()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, len(names))), columns=names)
    model = sklearn_linear.Ridge().fit(frame, frame.iloc[:, 0] * 2 + 5)
    path = tmp_path / "huber_mae-0.2456_evals-0.sav"
    joblib.dump(model, path)
    return path.read_bytes()


@pytest.fixture
def nameless_bytes(tmp_path):
    """Fitted on a bare array, so it records no `feature_names_in_`."""
    model = sklearn_linear.Ridge().fit(np.zeros((10, 3)), np.zeros(10))
    path = tmp_path / "nameless.sav"
    joblib.dump(model, path)
    return path.read_bytes()


@pytest.fixture
def rig(client, tmp_path, monkeypatch):
    """The api, a quarantine directory, and a model store, all under tmp."""
    from services.api import control_routes

    uploads = tmp_path / "uploads"
    monkeypatch.setattr(control_routes, "UPLOAD_DIR", uploads)
    monkeypatch.setenv("MODEL_STORE", str(tmp_path / "models"))
    return {"client": client, "uploads": uploads,
            "models": tmp_path / "models",
            "conn": client.app.state.db}


def upload(rig, payload: bytes, filename="huber_mae-0.2456_evals-0.sav",
           headers=AUTH, **query):
    params = {"filename": filename, "param": "muf", **query}
    return rig["client"].post("/models/upload", params=params,
                              content=payload, headers=headers)


# --------------------------------------------------------------------------
# The door
# --------------------------------------------------------------------------

def test_upload_needs_the_control_token(rig, artifact_bytes):
    """Uploading a pickle is not a read, whatever else it is."""
    assert upload(rig, artifact_bytes, headers={}).status_code == 401
    assert upload(rig, artifact_bytes,
                  headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert not rig["uploads"].exists() or not list(rig["uploads"].glob("*"))


def test_something_that_is_not_an_artifact_is_refused_by_its_first_bytes(rig):
    response = upload(rig, b"this is a CSV, not a model\n", filename="picks.csv")

    assert response.status_code == 415
    assert "does not begin like a model artifact" in response.json()["detail"]
    assert list(rig["uploads"].glob("*")) == [], "a refused body was still written"


def test_an_empty_body_is_refused_rather_than_stored(rig):
    assert upload(rig, b"").status_code == 400
    assert list(rig["uploads"].glob("*")) == []


def test_a_body_over_the_cap_is_refused_and_the_partial_file_removed(
        rig, monkeypatch, artifact_bytes):
    from services.api import control_routes

    monkeypatch.setattr(control_routes, "MAX_UPLOAD_BYTES", 32)
    response = upload(rig, artifact_bytes)

    assert response.status_code == 413
    assert "MODEL_UPLOAD_MAX_BYTES" in response.json()["detail"]
    assert list(rig["uploads"].glob("*")) == [], "the partial write survived"


# --------------------------------------------------------------------------
# The wall
# --------------------------------------------------------------------------

def test_the_api_registers_nothing_itself(rig, artifact_bytes):
    """The load-bearing one. A successful upload leaves the registry empty.

    The api hashed the bytes, checked four of them and wrote the rest to a
    volume. It did not unpickle anything, so nothing it could learn from the
    artifact -- the framework, the feature names, the golden output -- is in
    the row it wrote. That is the whole design, and if this assertion ever
    needs relaxing, the design changed.
    """
    response = upload(rig, artifact_bytes)
    assert response.status_code == 200, response.text

    row = response.json()["upload"]
    assert row["state"] == "pending"
    assert registry.models(rig["conn"]) == []

    blob = rig["uploads"] / row["sha256"]
    assert blob.is_file()
    assert blob.read_bytes() == artifact_bytes
    assert artifacts.sha256(blob) == row["sha256"]
    # Nothing is in the object store either: storing is the worker's first act,
    # not the api's last.
    assert not (rig["models"] / "objects").exists()


def _sources(package: str):
    for path in sorted((ROOT / "services" / package).rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), str(path))


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(ROOT)}:{node.lineno}"


#: Modules that can turn bytes into running code.
UNPICKLERS = {"joblib", "pickle", "cloudpickle", "dill"}

#: Names that load an artifact, wherever they are reached from.
LOADERS = {"load", "loads", "load_verified"}


def test_the_api_never_unpickles(rig):
    """A source check, in the spirit of `test_h5_read_pattern`.

    The property is stated in three docstrings and enforced by none of them.
    Here it is mechanical: nothing under `services/api/` may load an artifact,
    so a route that grew the ability has to come past this test on its way in.

    Read from the syntax tree rather than by grep, because the words appear
    legitimately in prose -- the 415 refusal tells the operator that a joblib
    pickle is what it wanted -- and a guard that fires on its own error message
    is a guard somebody deletes.
    """
    offenders = []
    for path, tree in _sources("api"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in UNPICKLERS:
                        offenders.append(f"{_where(path, node)}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")
                if module and module[0] in UNPICKLERS:
                    offenders.append(f"{_where(path, node)}: imports from {node.module}")
                if module and module[-1] == "artifacts":
                    for alias in node.names:
                        if alias.name in LOADERS:
                            offenders.append(
                                f"{_where(path, node)}: imports artifacts.{alias.name}")
            elif isinstance(node, ast.Attribute) and node.attr in LOADERS:
                base = node.value
                if isinstance(base, ast.Name) and base.id in UNPICKLERS | {"artifacts"}:
                    offenders.append(f"{_where(path, node)}: calls {base.id}.{node.attr}")
    assert not offenders, (
        "the api must not be able to load a model artifact; the registrar is "
        "what opens uploaded files:\n" + "\n".join(offenders))


def test_only_the_trainer_fits(rig):
    """`fit` belongs to exactly one module.

    The code this service replaces refits a saved model on load, so its
    forecasts come from a model trained seconds earlier.
    `test_prediction_infer.test_inference_never_fits` pins the inference path
    shut at runtime; this pins the boundary at the source, now that one module
    is finally allowed to call it.
    """
    allowed = {"train.py"}
    fitting = {"fit", "fit_transform", "fit_predict", "partial_fit"}
    offenders = []
    for package in ("api", "prediction", "agent"):
        for path, tree in _sources(package):
            if path.name in allowed:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in fitting):
                    offenders.append(f"{_where(path, node.func)}: .{node.func.attr}()")
    assert not offenders, (
        "only services/prediction/train.py may fit a model:\n" + "\n".join(offenders))


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------

def test_the_registrar_settles_a_pending_upload(rig, artifact_bytes):
    row = upload(rig, artifact_bytes, tx="NIC3", rx="Yoshkar-Ola",
                 origin="imported").json()["upload"]

    settled = registrar.run_once(rig["conn"], rig["uploads"])

    assert len(settled) == 1
    assert settled[0]["state"] == "registered", settled[0]["detail"]
    model = registry.get(rig["conn"], settled[0]["model_id"])
    assert model["sha256"] == row["sha256"]
    assert model["tx"] == "NIC3" and model["rx"] == "Yoshkar-Ola"
    assert model["features"] == feature_names()
    assert model["target_alias"] == ALIAS
    assert model["feature_recipe"]["lag"] == LAG


def test_the_registered_artifact_lives_in_the_store_under_its_digest(
        rig, artifact_bytes):
    row = upload(rig, artifact_bytes).json()["upload"]
    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    model = registry.get(rig["conn"], settled["model_id"])
    assert Path(model["artifact"]) == store.path_for(row["sha256"])
    assert store.verify(row["sha256"])
    assert Path(model["artifact"]).stat().st_mode & 0o777 == 0o444
    # And the quarantined copy is gone: the store holds the bytes now.
    assert not (rig["uploads"] / row["sha256"]).exists()


def test_the_registered_name_is_the_file_the_operator_uploaded(rig, artifact_bytes):
    """Not the digest.

    `import_artifact` defaults a model's name to the artifact's file stem, and
    in the store that stem is sixty-four hex characters -- correct as an
    address and unreadable in a list.
    """
    upload(rig, artifact_bytes, filename="huber_lag288_mae-0.2456.sav")
    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    assert registry.get(rig["conn"],
                        settled["model_id"])["name"] == "huber_lag288_mae-0.2456"


def test_an_artifact_that_names_no_inputs_is_refused_with_its_own_sentence(
        rig, nameless_bytes):
    """The refusal comes from the importer, unchanged, and reaches the page."""
    row = upload(rig, nameless_bytes, filename="nameless.sav").json()["upload"]

    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    assert settled["state"] == "refused"
    assert "does not record the names of its inputs" in settled["detail"]
    assert registry.models(rig["conn"]) == []
    # Nothing is left in the store, and the quarantined bytes are kept: the
    # usual fix is to register the same file again with an explicit list.
    assert not store.has(row["sha256"])
    assert (rig["uploads"] / row["sha256"]).is_file()


def test_a_missing_quarantine_file_is_a_refusal_not_a_crash(rig, artifact_bytes):
    row = upload(rig, artifact_bytes).json()["upload"]
    (rig["uploads"] / row["sha256"]).unlink()

    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    assert settled["state"] == "refused"
    assert "no longer at" in settled["detail"]


def test_bytes_that_changed_after_upload_are_refused_unopened(rig, artifact_bytes):
    """The api hashed a stream; this confirms the volume still holds it."""
    row = upload(rig, artifact_bytes).json()["upload"]
    blob = rig["uploads"] / row["sha256"]
    blob.write_bytes(b"\x80\x04" + artifact_bytes)

    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    assert settled["state"] == "refused"
    assert "changed after it was uploaded" in settled["detail"]
    assert registry.models(rig["conn"]) == []


def test_the_same_bytes_twice_converge_on_one_object(rig, artifact_bytes):
    first = upload(rig, artifact_bytes, tx="NIC3", rx="Yoshkar-Ola").json()["upload"]
    second = upload(rig, artifact_bytes, tx="NIC3", rx="Yoshkar-Ola").json()["upload"]
    assert first["sha256"] == second["sha256"]

    settled = registrar.run_once(rig["conn"], rig["uploads"])

    assert [s["state"] for s in settled] == ["registered", "registered"]
    # One object, and one registry row: the registry's identity is
    # (name, param, tx, rx, sha256), and both uploads agree on all five.
    assert len(list((rig["models"] / "objects").glob("*/*"))) == 1
    assert len(registry.models(rig["conn"])) == 1
    assert settled[0]["model_id"] == settled[1]["model_id"]


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------

def test_the_console_can_list_uploads_without_a_token(rig, artifact_bytes):
    upload(rig, artifact_bytes)
    body = rig["client"].get("/models/uploads").json()

    assert body["count"] == 1
    assert body["uploads"][0]["state"] == "pending"


def test_the_artifact_can_be_pulled_back_out_with_its_digest(rig, artifact_bytes):
    row = upload(rig, artifact_bytes).json()["upload"]
    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    response = rig["client"].get(f"/models/{settled['model_id']}/artifact")

    assert response.status_code == 200
    assert response.content == artifact_bytes
    assert response.headers["x-artifact-sha256"] == row["sha256"]


def test_pulling_a_model_whose_volume_is_gone_says_so(rig, artifact_bytes):
    upload(rig, artifact_bytes)
    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]
    model = registry.get(rig["conn"], settled["model_id"])
    path = Path(model["artifact"])
    path.unlink()

    response = rig["client"].get(f"/models/{settled['model_id']}/artifact")

    assert response.status_code == 410
    assert model["sha256"] in response.json()["detail"]


def test_forgetting_an_upload_reaps_its_bytes(rig, nameless_bytes):
    row = upload(rig, nameless_bytes, filename="nameless.sav").json()["upload"]
    registrar.run_once(rig["conn"], rig["uploads"])
    assert (rig["uploads"] / row["sha256"]).is_file()

    response = rig["client"].delete(f"/models/uploads/{row['id']}", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["reaped"] is True
    assert not (rig["uploads"] / row["sha256"]).exists()
    assert queues.uploads(rig["conn"]) == []


def test_forgetting_needs_the_control_token(rig, artifact_bytes):
    row = upload(rig, artifact_bytes).json()["upload"]
    assert rig["client"].delete(f"/models/uploads/{row['id']}").status_code == 401
    assert queues.upload(rig["conn"], row["id"]) is not None


def test_a_refused_upload_keeps_its_bytes_for_a_second_attempt(rig, nameless_bytes):
    """Re-registering with an explicit feature list is the usual fix.

    Making the operator upload the same file again to do that would be
    gratuitous, so a refused row holds its quarantined copy.
    """
    row = upload(rig, nameless_bytes, filename="nameless.sav").json()["upload"]
    registrar.run_once(rig["conn"], rig["uploads"])

    assert queues.blob_is_referenced(rig["conn"], row["sha256"])
    assert (rig["uploads"] / row["sha256"]).is_file()


def test_an_upload_is_bound_to_a_circuit_only_if_the_operator_said_so(
        rig, artifact_bytes):
    """Unbound is a legitimate state -- and it is the one that cannot be
    promoted, which the page has to be able to say."""
    upload(rig, artifact_bytes, origin="legacy")
    settled = registrar.run_once(rig["conn"], rig["uploads"])[0]

    model = registry.get(rig["conn"], settled["model_id"])
    assert model["tx"] is None and model["rx"] is None
    assert model["target_src"] == "modelled", "legacy must default to comparison"
    assert model["state"] == "comparison"
    with pytest.raises(registry.RegistryError):
        registry.activate(rig["conn"], model["id"])
