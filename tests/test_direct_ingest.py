"""The direct path must keep working when archive discovery cannot run."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.agent import upload
from services.api import archives, auth, db, direct, ingest, sim_live

NAME = "lfm_ionogram-NIC2-yoshkar-ola-ch0-000-1770163210.00.h5"
URL = "/stations/laptop/files/" + NAME
HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECT_INGEST", "1")
    monkeypatch.setenv("DIRECT_ROOT", str(tmp_path / "local"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "db.sqlite3")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "test-token")
    conn = db.init(db.connect())
    app = FastAPI()
    app.state.db = conn
    app.include_router(direct.router)
    with TestClient(app) as client:
        yield client, conn
    conn.close()


def put(client, body, url=URL, **headers):
    return client.put(url, content=body, headers={
        **HEADERS, "X-Content-SHA256": hashlib.sha256(body).hexdigest(), **headers})


def test_upload_auth_disabled_and_bad_content(hub, monkeypatch):
    client, _ = hub
    assert client.put(URL, content=b"anything").status_code == 401
    assert put(client, b"not hdf5").status_code == 400
    assert put(client, direct.SIGNATURE, **{"X-Content-SHA256": "0" * 64}) \
        .status_code == 400
    assert not list(direct.root().glob("*/*.pending"))
    assert not list(direct.root().glob("*/*.upload"))
    monkeypatch.setenv("DIRECT_INGEST", "0")
    assert put(client, direct.SIGNATURE).status_code == 503


def test_limits_and_names(hub, monkeypatch):
    client, _ = hub
    assert put(client, direct.SIGNATURE, url=URL.replace("laptop", "bad:station")) \
        .status_code == 400
    assert put(client, direct.SIGNATURE, url=URL.replace(NAME, "file.h5.exe")) \
        .status_code == 400
    monkeypatch.setattr(direct, "MAX_BYTES", 8)
    assert put(client, direct.SIGNATURE + b"too large").status_code == 413
    assert put(client, direct.SIGNATURE, **{"Content-Length": "10"}).status_code == 413


def test_lost_ack_retry_and_conflicting_content(hub):
    client, _ = hub
    first = put(client, direct.SIGNATURE)
    assert first.status_code == 202
    assert put(client, direct.SIGNATURE).json() == first.json()
    assert len(list(direct.root().glob("*/*.pending"))) == 1
    assert put(client, direct.SIGNATURE + b"changed").status_code == 409
    assert (direct.root() / "laptop" / NAME).read_bytes() == direct.SIGNATURE


def test_retry_repairs_a_receipt_after_interrupted_publication(hub):
    client, _ = hub
    put(client, direct.SIGNATURE)
    receipt = direct.root() / "laptop" / (NAME + ".json")
    receipt.unlink()
    assert put(client, direct.SIGNATURE).status_code == 202
    assert client.get(URL, headers=HEADERS).json()["status"] == "queued"


def test_muted_circuit_is_not_resurrected_by_upload(hub, tmp_path):
    client, conn = hub
    conn.execute("INSERT INTO muted_circuit (tx, rx, muted_at, muted_by) "
                 "VALUES ('nic2', 'yoshkar-ola', '2026-01-01', 'test')")
    conn.commit()
    put(client, direct.SIGNATURE)
    direct.advance(conn, tmp_path, process=lambda p: {
        "datetime": "2026-02-04 00:05:10", "tx": "NIC2", "rx": "yoshkar-ola"})
    assert client.get(URL, headers=HEADERS).json()["status"] == "muted"
    assert db.one(conn, "SELECT count(*) AS n FROM sounding")["n"] == 0


def test_real_hdf5_ingest_is_independent_and_keeps_local_path(
        hub, make_chirp_h5, tmp_path, monkeypatch):
    client, conn = hub
    path = make_chirp_h5(np.ones((24, 1024)), name=NAME,
                         txname="NIC2", station_name="yoshkar-ola",
                         freqs_hz=np.linspace(7e6, 12e6, 24))
    def forbidden(*args, **kw):
        raise AssertionError("direct ingestion touched archive discovery")
    monkeypatch.setattr(archives, "scan", forbidden)
    # No /archive directory exists. The genuine reader and contour extractor
    # must still produce a database row using the received file alone.
    nas = tmp_path / "absent-nas"
    assert put(client, path.read_bytes()).status_code == 202
    assert direct.advance(conn, nas) == 1
    receipt = client.get(URL, headers=HEADERS).json()
    assert receipt["status"] == "ingested", receipt
    row = db.one(conn, "SELECT * FROM sounding WHERE id=?", (receipt["sounding_id"],))
    local = Path(row["path"])
    assert local.is_file() and local.is_relative_to(direct.root())
    assert db.one(conn, "SELECT method FROM extraction")["method"] == "contour"
    # A later NAS ingestion adds methods without redirecting the simulator
    # back to a slow or unavailable network file or adding a duplicate row.
    record = direct._extract(str(local))
    record["muf_algo"] = 12.0
    ingest.ingest_row(conn, record, nas / NAME, nas, ("algo",))
    conn.commit()
    assert db.one(conn, "SELECT count(*) AS n FROM sounding")["n"] == 1
    assert db.one(conn, "SELECT path FROM sounding")["path"] == local.as_posix()
    assert put(client, path.read_bytes()).json()["status"] == "ingested"
    assert direct.advance(conn, nas, process=forbidden) == 0


def test_failed_ingestion_survives_restart_and_retries(hub, tmp_path):
    client, conn = hub
    put(client, direct.SIGNATURE)  # valid signature, truncated HDF5
    direct.advance(conn, tmp_path, process=direct._extract)
    receipt = client.get(URL, headers=HEADERS).json()
    assert receipt["status"] == "queued" and receipt["error"]
    assert db.one(conn, "SELECT count(*) AS n FROM sounding")["n"] == 0
    pending = next(direct.root().glob("*/*.pending"))
    receipt["retry_at"] = 0
    pending.write_text(json.dumps(receipt))
    def process(_):
        return {"file": NAME, "datetime": "2026-02-04 00:00:10",
                "tx": "NIC2", "rx": "yoshkar-ola", "muf_contour": 12.0}
    other = db.connect()
    try:
        assert direct.advance(other, tmp_path, process=process) == 1
    finally:
        other.close()
    assert client.get(URL, headers=HEADERS).json()["status"] == "ingested"
    assert not pending.exists()


def test_new_direct_row_reaches_the_live_simulator(hub, tmp_path):
    client, conn = hub
    ingest.ingest_row(conn, {"file": "earlier.h5", "datetime": "2026-02-04 00:00:10",
                            "tx": "NIC2", "rx": "yoshkar-ola"},
                      tmp_path / "earlier.h5", tmp_path, ())
    conn.commit()
    sim_live.follow(conn, "NIC2", "yoshkar-ola")
    put(client, direct.SIGNATURE)
    direct.advance(conn, tmp_path, process=lambda p: {
        "datetime": "2026-02-04 00:05:10", "tx": "NIC2", "rx": "yoshkar-ola"})
    calls = []
    def turn(ledger, path, epoch, tx, rx, cache):
        calls.append(Path(path))
        return ledger, []
    assert sim_live.advance(conn, tmp_path / "unavailable", tmp_path / "cache",
                            turn=turn) == 1
    assert calls == [direct.root() / "laptop" / NAME]


def test_sender_stability_retry_checkpoint_and_unchanged_file(tmp_path):
    source = tmp_path / "station"
    source.mkdir()
    config = SimpleNamespace(server_url="http://hub", token="secret", station="laptop")
    path = source / NAME
    path.write_bytes(direct.SIGNATURE)
    os.utime(path, (990, 990))
    state = tmp_path / "state.json"
    calls = []
    def send(*args):
        calls.append(args[1])
    assert upload.run_once(config, source, state, now=1000, upload=send)["sent"] == 0
    def fail(*args):
        raise OSError("connection lost")
    assert upload.run_once(config, source, state, now=1100, upload=fail)["errors"]
    assert not json.loads(state.read_text())["sent"]
    assert upload.run_once(config, source, state, now=1120, upload=send)["sent"] == 0
    assert upload.run_once(config, source, state, now=1170, upload=send)["sent"] == 1
    assert upload.run_once(config, source, state, now=1200, upload=send)["sent"] == 0
    assert calls == [path]
    config.server_url = "http://replacement-hub"
    assert upload.run_once(config, source, state, now=1200, upload=send)["sent"] == 1


def test_sender_streams_to_the_real_receiver(hub, tmp_path):
    client, _ = hub
    path = tmp_path / NAME
    path.write_bytes(direct.SIGNATURE)
    config = SimpleNamespace(server_url="http://hub", token="test-token",
                             station="laptop")
    def opener(request, timeout):
        assert not isinstance(request.data, bytes), "large capture buffered in memory"
        response = client.put(URL, content=request.data,
                              headers=dict(request.header_items()))
        assert response.status_code == 202, response.text
        from io import BytesIO
        return BytesIO(response.content)
    receipt = upload.send(config, path, upload.fingerprint(path), opener=opener)
    assert receipt["status"] == "queued"


def test_sender_never_acknowledges_a_file_changed_while_reading(tmp_path):
    path = tmp_path / NAME
    path.write_bytes(direct.SIGNATURE)
    stamp = upload.fingerprint(path)
    path.write_bytes(direct.SIGNATURE + b"changed")
    config = SimpleNamespace(server_url="http://hub", token="test-token",
                             station="laptop")
    with pytest.raises(ValueError, match="changed while hashing"):
        upload.send(config, path, stamp)
