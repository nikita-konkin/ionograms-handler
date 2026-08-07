"""The api service.

The load-bearing tests are about the two things that would be dangerous to get
wrong: control must never fall open when its secret is missing, and the
tri-state health metric must survive the round trip through SQL. A boolean
column would turn "could not measure" into "failing" somewhere between the
station and the screen, and the whole health design rests on those being
different.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

from services.api import auth, db                  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as c:
        yield c


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")

    from services.api import main

    with TestClient(main.app) as c:
        yield c


CTL = {"Authorization": "Bearer ctl"}


def report(station="SIM", healthy=False):
    return {
        "station": station, "timestamp": 1785888000.0, "agent_version": "0.1.0",
        "healthy": healthy,
        "metrics": [
            {"name": "unit:chirp-rx.service", "value": None, "ok": None,
             "detail": "systemctl: not found"},
            {"name": "disk_free_fraction", "value": 0.42, "ok": True, "detail": ""},
            {"name": "newest_product_age_s", "value": 9000.0, "ok": False,
             "detail": "threshold 900s"},
        ],
    }


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def test_control_is_disabled_when_its_secret_is_missing(tmp_path, monkeypatch):
    """A missing secret must never be the same as a granted one.

    Getting this backwards means a stranger can stop acquisition, so it fails
    closed with an explanation rather than falling open.
    """
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "")
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "a.sqlite3")
    from services.api import main

    with TestClient(main.app) as c:
        r = c.post("/stations/health", json=report())
    assert r.status_code == 503
    assert "not an open door" in r.json()["detail"]


def test_a_wrong_control_token_is_refused(client):
    r = client.post("/stations/health", json=report(),
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_read_is_open_when_no_read_token_is_set(client):
    assert client.get("/stations").status_code == 200


def test_read_token_does_not_grant_control(client, monkeypatch):
    """arch 4.3: public read must not share a scope with the stop button."""
    monkeypatch.setattr(auth, "READ_TOKEN", "rd")
    assert client.get("/stations", headers={"Authorization": "Bearer rd"}).status_code == 200
    r = client.post("/stations/SIM/commands", json={"name": "stop"},
                    headers={"Authorization": "Bearer rd"})
    assert r.status_code == 401


# --------------------------------------------------------------------------
# Health ingest
# --------------------------------------------------------------------------

def test_a_pushed_report_appears_with_its_metrics(client):
    assert client.post("/stations/health", json=report(), headers=CTL).status_code == 200

    body = client.get("/stations/SIM/health").json()
    assert body["healthy"] is False
    names = {m["name"]: m for m in body["metrics"]}
    assert names["disk_free_fraction"]["ok"] is True
    assert names["newest_product_age_s"]["ok"] is False


def test_unknown_survives_the_round_trip_as_none(client):
    """The one that would be silently wrong if `ok` were a boolean column."""
    client.post("/stations/health", json=report(), headers=CTL)
    body = client.get("/stations/SIM/health").json()
    unit = next(m for m in body["metrics"] if m["name"].startswith("unit:"))

    assert unit["ok"] is None, "unknown must not collapse to False"


def test_a_document_without_a_station_is_refused(client):
    r = client.post("/stations/health", json={"metrics": []}, headers=CTL)
    assert r.status_code == 400


def test_the_raw_document_is_kept_verbatim(client, tmp_path):
    """So a newer agent's extra fields are not lost to this server's schema."""
    payload = report()
    payload["metrics"][0]["future_field"] = "kept"
    client.post("/stations/health", json=payload, headers=CTL)

    with db.session(tmp_path / "api.sqlite3") as conn:
        stored = json.loads(db.rows(conn, "SELECT document FROM health_report")[0]["document"])
    assert stored["metrics"][0]["future_field"] == "kept"


def test_history_is_kept_not_overwritten(client):
    for _ in range(3):
        client.post("/stations/health", json=report(), headers=CTL)
    body = client.get("/stations/SIM/health/history").json()

    assert len(body["history"]) == 3, "silence is the alert; history is the evidence"


def test_the_station_list_reports_an_age(client):
    client.post("/stations/health", json=report(), headers=CTL)
    entry = client.get("/stations").json()["stations"][0]

    assert entry["station"] == "SIM"
    assert entry["age_s"] is not None and entry["age_s"] >= 0
    assert entry["failing"] == 1 and entry["unknown"] == 1 and entry["ok"] == 1


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def test_a_command_is_queued_delivered_once_and_acked(client):
    queued = client.post("/stations/SIM/commands", json={"name": "restart"},
                         headers=CTL).json()
    command_id = queued["id"]

    first = client.get("/stations/SIM/commands", headers=CTL).json()["commands"]
    assert [c["name"] for c in first] == ["restart"]

    second = client.get("/stations/SIM/commands", headers=CTL).json()["commands"]
    assert second == [], "a delivered command must not be handed out again"

    client.post(f"/stations/SIM/commands/{command_id}/ack", headers=CTL,
                json={"results": [{"command": "restart chirp.target", "ok": True}]})
    shown = client.get("/stations/SIM/health").json()
    assert shown["commands"][0]["state"] == "acked"
    assert shown["commands"][0]["ok"] is True


def test_only_process_verbs_are_queueable_from_the_web(client):
    """control.py allows more; the web surface deliberately does not."""
    for name in ("set_config", "isolate", "mask", "logs"):
        r = client.post("/stations/SIM/commands", json={"name": name}, headers=CTL)
        assert r.status_code == 400, name


def test_a_failed_command_is_recorded_as_failed(client):
    queued = client.post("/stations/SIM/commands", json={"name": "stop"},
                         headers=CTL).json()
    client.get("/stations/SIM/commands", headers=CTL)
    client.post(f"/stations/SIM/commands/{queued['id']}/ack", headers=CTL,
                json={"results": [{"command": "stop", "ok": False,
                                   "detail": "timed out"}]})

    shown = client.get("/stations/SIM/health").json()["commands"][0]
    assert shown["state"] == "acked" and shown["ok"] is False


def test_acking_an_unknown_command_is_accepted(client):
    """The agent has already acted; refusing the ack makes it retry."""
    r = client.post("/stations/SIM/commands/nope/ack", headers=CTL,
                    json={"results": []})
    assert r.status_code == 200 and r.json()["known"] is False


# --------------------------------------------------------------------------
# The real agent against the real server
# --------------------------------------------------------------------------

def test_the_agent_and_the_server_agree_on_every_path(client, tmp_path):
    """The integration that the endpoint-path mismatch would have broken.

    architecture.md 4.3 proposed `/health/report`; the agent posts to
    `/stations/health`. This drives the actual client against the actual
    routes, so a future edit to either cannot quietly desynchronise them.
    """
    from services.agent import runner
    from services.agent.config import StationConfig

    config = StationConfig(station="SIM", server_url="", token="ctl",
                           chirp_config=tmp_path / "none.ini",
                           output_dir=tmp_path / "products",
                           ringbuffer_dir=tmp_path,
                           reference_tx={})
    config = StationConfig(**{**config.as_dict(),
                              "chirp_config": config.chirp_config,
                              "output_dir": config.output_dir,
                              "ringbuffer_dir": config.ringbuffer_dir,
                              "server_url": "http://testserver",
                              "token": "ctl"})

    def opener(request, timeout=None):
        method = request.get_method()
        url = request.full_url.replace("http://testserver", "")
        body = json.loads(request.data.decode()) if request.data else None
        headers = {k: v for k, v in request.header_items()}
        response = (client.get(url, headers=headers) if method == "GET"
                    else client.post(url, json=body, headers=headers))

        class Wrapped:
            def read(self_inner):
                return response.content

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        assert response.status_code == 200, f"{method} {url} -> {response.status_code}"
        return Wrapped()

    client.post("/stations/SIM/commands", json={"name": "restart"}, headers=CTL)
    result = runner.run_once(config, opener=opener, include_epoch=False)

    assert result.pushed, result.errors
    assert result.commands_run == 1
    assert not result.errors

    shown = client.get("/stations/SIM/health").json()
    assert shown["metrics"], "the real collector produced a document"
    assert shown["commands"][0]["state"] == "acked"


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

def test_a_successful_row_is_not_read_as_an_error(conn, tmp_path):
    """pandas fills the absent `error` cell of a *successful* row with NaN, and
    `bool(nan)` is True. Testing it directly marks every good sounding as
    failed and ingests nothing -- which is exactly what happened."""
    import numpy as np

    from services.api import ingest

    row = {"file": "s.lfs", "error": np.nan, "datetime": "2026-02-04T00:00:10",
           "tx": "cyprus1", "rx": "yoshkar-ola", "muf_algo": 12.2,
           "run_algo": 7, "limited_algo": False}
    (tmp_path / "s.lfs").write_bytes(b"x")

    got = ingest.ingest_row(conn, row, tmp_path / "s.lfs", tmp_path, ("algo",))
    assert got is not None
    assert db.rows(conn, "SELECT muf FROM extraction")[0]["muf"] == 12.2


def test_a_real_error_row_is_skipped(conn, tmp_path):
    from services.api import ingest

    row = {"file": "bad.lfs", "error": "ValueError: not an LFS file",
           "datetime": "2026-02-04T00:00:10"}
    (tmp_path / "bad.lfs").write_bytes(b"x")

    assert ingest.ingest_row(conn, row, tmp_path / "bad.lfs", tmp_path, ("algo",)) is None
    assert db.rows(conn, "SELECT * FROM sounding") == []


def test_a_missing_pick_is_null_not_nan(conn, tmp_path):
    """`WHERE muf IS NOT NULL` must mean what it says. A NaN in a REAL column
    compares false against everything including itself, so a stored NaN would
    come back from that query as a row with no MUF."""
    import numpy as np

    from services.api import ingest

    (tmp_path / "s.lfs").write_bytes(b"x")
    ingest.ingest_row(conn, {"file": "s.lfs", "datetime": "2026-02-04T00:00:10",
                             "muf_algo": np.nan, "run_algo": np.nan},
                      tmp_path / "s.lfs", tmp_path, ("algo",))

    assert db.rows(conn, "SELECT muf FROM extraction WHERE muf IS NOT NULL") == []


def test_ingest_is_idempotent_on_file_and_method(conn, tmp_path):
    """Re-ingesting is the first thing anyone does when a run looks wrong."""
    from services.api import ingest

    (tmp_path / "s.lfs").write_bytes(b"x")
    row = {"file": "s.lfs", "datetime": "2026-02-04T00:00:10", "muf_algo": 12.2}
    ingest.ingest_row(conn, row, tmp_path / "s.lfs", tmp_path, ("algo",))
    ingest.ingest_row(conn, {**row, "muf_algo": 13.9}, tmp_path / "s.lfs",
                      tmp_path, ("algo",))

    assert len(db.rows(conn, "SELECT * FROM sounding")) == 1
    extractions = db.rows(conn, "SELECT muf FROM extraction")
    assert len(extractions) == 1 and extractions[0]["muf"] == 13.9


def test_the_stored_path_is_relative_to_the_archive_root(conn, tmp_path):
    """The same database is read from the host and from inside a container,
    which mount the archive at different paths."""
    from services.api import ingest

    day = tmp_path / "2026-02-04"
    day.mkdir()
    (day / "s.lfs").write_bytes(b"x")
    ingest.ingest_row(conn, {"file": "s.lfs", "datetime": "2026-02-04T00:00:10"},
                      day / "s.lfs", tmp_path, ("algo",))

    stored = db.rows(conn, "SELECT path FROM sounding")[0]["path"]
    assert stored == "2026-02-04/s.lfs"
