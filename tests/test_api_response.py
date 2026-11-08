"""A stored sounding's frequency response: services.api.response and its route.

One synthetic recording serves the module: 10-14 MHz in 40 s, one law mode
the extractor finds, as tests/test_sim_sounding.py observes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
h5py = pytest.importorskip("h5py")
from fastapi.testclient import TestClient  # noqa: E402

from muf.sim import fast, presets, sounding  # noqa: E402
from muf.sim.state import ChannelState, Sweep  # noqa: E402
from services.api import auth, db, main, net, response  # noqa: E402


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    mode = presets.law_mode(muf_mhz=14.0, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.0)
    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)
    state = ChannelState(presets.cyprus1_circuit(), 1770206410.0, modes=[mode],
                         sweep=sweep, source="test")
    return sounding.write_lfs(state, tmp_path_factory.mktemp("rec") / "sim.lfs", seed=5)


@pytest.fixture
def client(tmp_path, monkeypatch, recording):
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    monkeypatch.setenv("ARCHIVE_ROOT", str(recording.parent))
    monkeypatch.setenv("ARCHIVE_SCAN_INTERVAL_S", "0")
    response.forget()
    with TestClient(main.app) as c:
        yield c


def _row(client, name: str) -> int:
    conn = client.app.state.db
    cursor = conn.execute(
        "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
        "VALUES (?,?,?,?,?,?)",
        (name, name, "2026-02-04 12:00:10", "cyprus1", "yoshkar-ola", db.utcnow()))
    conn.commit()
    return cursor.lastrowid


def _power_only(folder: Path) -> Path:
    path = folder / "power-only.h5"
    with h5py.File(path, "w") as fh:
        fh["S"] = [[1.0, 2.0], [3.0, 4.0]]
    return path


def test_a_recording_keeps_the_phase(recording):
    assert response.keeps_phase(recording), "an .lfs was said to keep no phase"


def test_a_product_without_z_or_a_window_keeps_no_phase(tmp_path):
    assert not response.keeps_phase(_power_only(tmp_path)), \
        "a power-only product was said to keep the phase"


def test_the_route_reads_the_first_mode_by_default(client):
    got = client.get(f"/soundings/{_row(client, 'sim.lfs')}/response",
                     params={"lo": 11.0, "hi": 12.0})
    assert got.json()["response"]["mode"] == "1F", "the first mode was not read"


def test_the_route_passes_the_settings_on(client):
    got = client.get(f"/soundings/{_row(client, 'sim.lfs')}/response",
                     params={"lo": 11.0, "hi": 12.0, "sub": 40.96, "overlap": 0.9,
                             "held": True, "shift": 10.0, "half": 50.0}).json()
    seen = (got["response"]["window"], got["response"]["gate"]["held"],
            got["response"]["gate"]["shift_us"], got["response"]["gate"]["half_us"])
    assert seen == (16384, True, 10.0, 50.0), "the settings were not passed on"


def test_the_route_sends_the_modes_curves(client):
    got = client.get(f"/soundings/{_row(client, 'sim.lfs')}/response",
                     params={"lo": 11.0, "hi": 12.0}).json()
    assert len(got["modes"][0]["curve"]["freq_mhz"]) > 10, "no curve came with the mode"


def test_the_route_sends_an_ionogram_about_the_modes(client):
    got = client.get(f"/soundings/{_row(client, 'sim.lfs')}/response",
                     params={"lo": 11.0, "hi": 12.0}).json()
    assert got["ionogram"]["cols"] * got["ionogram"]["rows"] > 0, \
        "no ionogram came with the response"


def test_the_route_thins_a_long_series(client):
    got = client.get(f"/soundings/{_row(client, 'sim.lfs')}/response",
                     params={"lo": 11.0, "hi": 12.0, "overlap": 1.0}).json()
    assert len(got["response"]["sliding"]["freq_mhz"]) <= response.POINTS, \
        "a sliding series at every sample was sent whole"


def test_a_second_request_reads_the_recording_once(client, monkeypatch):
    calls = []
    real = fast.record_of
    monkeypatch.setattr(fast, "record_of", lambda *a, **k: calls.append(a) or real(*a))
    sounding_id = _row(client, "sim.lfs")
    for sub in (20.48, 40.96):
        client.get(f"/soundings/{sounding_id}/response",
                   params={"lo": 11.0, "hi": 12.0, "sub": sub})
    assert len(calls) == 1, "the recording was read again for a second setting"


def test_an_unknown_mode_is_unprocessable(client):
    got = client.get(f"/soundings/{_row(client, 'sim.lfs')}/response",
                     params={"mode": "7F"})
    assert got.status_code == 422, "an unknown mode was not refused"


def test_a_file_without_the_phase_is_unprocessable(client, recording):
    _power_only(recording.parent)
    got = client.get(f"/soundings/{_row(client, 'power-only.h5')}/response")
    assert got.status_code == 422, "a power-only product was not refused"


def test_an_unknown_sounding_is_not_found(client):
    assert client.get("/soundings/987654/response").status_code == 404, \
        "an unknown sounding was found"


def test_the_page_of_a_recording_loads_its_script(client):
    page = client.get(f"/ui/sounding/{_row(client, 'sim.lfs')}/response").text
    assert "/static/response.js" in page, "the response page did not load its script"


def test_the_page_of_a_power_only_product_says_it_keeps_no_phase(client, recording):
    _power_only(recording.parent)
    page = client.get(f"/ui/sounding/{_row(client, 'power-only.h5')}/response").text
    assert "keeps no phase" in page, "the page did not say the product keeps no phase"


def test_the_page_of_an_unknown_sounding_is_not_found(client):
    assert client.get("/ui/sounding/987654/response").status_code == 404, \
        "the response page of an unknown sounding was found"


def test_the_sounding_page_links_a_recording_to_its_response(client):
    sounding_id = _row(client, "sim.lfs")
    page = client.get(f"/ui/sounding/{sounding_id}", params={"plot": "rendered"}).text
    assert f"/ui/sounding/{sounding_id}/response" in page, \
        "the sounding page did not link to the response"


def test_a_power_only_sounding_page_has_no_response_link(client, recording):
    _power_only(recording.parent)
    sounding_id = _row(client, "power-only.h5")
    page = client.get(f"/ui/sounding/{sounding_id}", params={"plot": "rendered"}).text
    assert f"/ui/sounding/{sounding_id}/response" not in page, \
        "a power-only product was linked to a response"
