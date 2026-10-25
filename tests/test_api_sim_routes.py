"""The hub's simulator page: a sounding's console, built on demand."""

from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from services.api import auth, db, main, sim  # noqa: E402

CTL = {"Authorization": "Bearer ctl"}


@pytest.fixture
def made(monkeypatch):
    calls = []

    def build(path):
        calls.append(path.name)
        if path.name.startswith("bad"):
            raise ValueError(f"no trace in {path.name}")
        return {"source": path.name, "utc": "2026-02-04 12:00 UTC", "modes": []}

    monkeypatch.setattr(sim, "_execute", build)
    return calls


@pytest.fixture
def client(tmp_path, monkeypatch, made):
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("ARCHIVE_SCAN_INTERVAL_S", "0")
    with TestClient(main.app) as c:
        yield c


def _sounding(client, tmp_path, name="cyprus1_20260204_120010.lfs") -> int:
    (tmp_path / name).write_bytes(b"\0" * 64)
    conn = client.app.state.db
    cursor = conn.execute(
        "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at, "
        "tx_lat, tx_lon, rx_lat, rx_lon) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (name, name, "2026-02-04 12:00:10", "cyprus1", "yoshkar-ola", db.utcnow(),
         35.0, 34.0, 56.38, 47.53))
    conn.commit()
    return cursor.lastrowid


def _settled(client, sounding_id: int, timeout_s: float = 10.0) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        got = client.get(f"/sim/{sounding_id}").json()
        if got["state"] not in ("queued", "running"):
            return got
        time.sleep(0.02)
    raise AssertionError(f"the build of {sounding_id} never settled")


def test_a_sounding_not_yet_run_is_absent(client, tmp_path):
    assert client.get(f"/sim/{_sounding(client, tmp_path)}").json()["state"] == "absent"


def test_running_the_simulator_needs_the_control_token(client, tmp_path):
    assert client.post(f"/sim/{_sounding(client, tmp_path)}").status_code == 401


def test_a_run_ends_with_the_console_built(client, tmp_path):
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    assert _settled(client, k)["state"] == "done"


def test_the_console_carries_the_soundings_data(client, tmp_path):
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    _settled(client, k)
    page = client.get(f"/sim/{k}/console?lang=en&embed=1").text
    assert "const DATA = [{" in page and "cyprus1_20260204_120010.lfs" in page, \
        "the console page came without the sounding's data"


def test_a_console_not_built_is_not_found(client, tmp_path):
    assert client.get(f"/sim/{_sounding(client, tmp_path)}/console").status_code == 404


def test_a_failed_run_says_why(client, tmp_path):
    k = _sounding(client, tmp_path, "bad_20260204_120010.lfs")
    client.post(f"/sim/{k}", headers=CTL)
    assert "no trace in bad_20260204_120010.lfs" in _settled(client, k)["error"]


def test_a_sounding_built_once_is_not_built_again(client, tmp_path, made):
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    _settled(client, k)
    client.post(f"/sim/{k}", headers=CTL)
    _settled(client, k)
    assert made == ["cyprus1_20260204_120010.lfs"], "a cached console was built again"


def test_a_console_from_another_build_of_the_hub_is_built_again(client, tmp_path,
                                                                 monkeypatch):
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    _settled(client, k)
    monkeypatch.setattr(sim, "VERSION", "a later image")
    assert client.get(f"/sim/{k}").json()["state"] == "absent"


def test_an_unknown_sounding_cannot_be_run(client):
    assert client.post("/sim/4040", headers=CTL).status_code == 404


def test_the_page_lists_the_soundings_to_run(client, tmp_path):
    _sounding(client, tmp_path)
    assert "cyprus1_20260204_120010.lfs" in client.get("/ui/sim").text


def test_the_page_marks_a_sounding_already_built(client, tmp_path):
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    _settled(client, k)
    assert f'value="{k}" data-built' in client.get("/ui/sim").text, \
        "the page did not mark a built console"


def test_the_navigation_leads_to_the_simulator(client):
    assert 'href="/ui/sim"' in client.get("/ui").text
