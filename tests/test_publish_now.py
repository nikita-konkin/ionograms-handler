"""The public page's live pass: the sounding in progress, every few seconds.

What `now.json` may hold is the point, as for the rest of the page: only
allow-listed, unmuted circuits, only allow-listed fields, never half a file,
and a station that cannot be read costs its own circuits, not the pass.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from services.api import db
from services.publish import now as live

NOW = datetime(2026, 8, 13, 9, 30, 12, tzinfo=timezone.utc)
T0 = 1786613100.0


def _station(conn, station: str, *receivers: str) -> None:
    conn.execute("INSERT INTO health_report (station, received_at, document)"
                 " VALUES (?, ?, '{}')",
                 (station, "2026-08-13 09:30:00"))
    for rx in receivers:
        conn.execute("INSERT INTO station_receiver (station, rx) VALUES (?, ?)",
                     (station, rx))
    conn.commit()


def _sounding(tx: str, done: int, *, columns: int = 40, rows: int = 12,
              t0: float = T0) -> dict:
    rng = np.random.default_rng(7)
    data = {str(i): base64.b64encode(
        rng.integers(0, 160, rows, dtype=np.uint8).tobytes()).decode()
        for i in range(done)}
    return {"type": "sounding", "tx": tx, "t0": t0, "f_lo_mhz": 4.0,
            "f_step_mhz": 0.25, "columns": columns, "range_lo_km": 2000.0,
            "range_step_km": 7.32, "rows": rows, "km_per_ms": 299.79,
            "data": data}


def _snapshot(*soundings: dict) -> dict:
    return {"station": "dob-station", "status": {}, "last": {}, "daily": {},
            "echo": {}, "current": {s["tx"]: s for s in soundings}}


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "api.sqlite3"
    with db.session(path) as c:
        _station(c, "dob-station", "DOB")
    c = db.connect(path)
    yield c
    c.close()


def _pass(conn, site, snapshots, circuits="NIC->DOB, SGO->DOB"):
    def fetch(station):
        found = snapshots.get(station)
        if isinstance(found, Exception):
            raise found
        return found
    return live.write_now(conn, site, fetch=fetch,
                          circuits=live.parse_circuits(circuits), now=NOW)


def _read(site: Path) -> dict:
    return json.loads((site / "now.json").read_text(encoding="utf-8"))


def test_a_circuit_being_sounded_is_listed(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert [c["slug"] for c in _read(tmp_path)["circuits"]] == ["nic-dob"], \
        "the circuit being sounded was not listed"


def test_a_circuit_not_allowed_is_left_out(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("Chilton", 10))})
    assert _read(tmp_path)["circuits"] == [], "a circuit off the allow-list was shown"


def test_a_muted_circuit_is_left_out(conn, tmp_path):
    db.mute_circuit(conn, "NIC", "DOB")
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert _read(tmp_path)["circuits"] == [], "a muted circuit was shown"


def test_the_progress_is_the_columns_so_far(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10, columns=40))})
    assert _read(tmp_path)["circuits"][0]["progress"] == 0.25, \
        "the progress was not the share of columns arrived"


def test_the_frequency_reached_is_the_last_column(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert _read(tmp_path)["circuits"][0]["f_now_mhz"] == 6.25, \
        "the frequency reached was not the last column's"


def test_a_circuit_carries_only_the_allowed_fields(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert set(_read(tmp_path)["circuits"][0]) == set(live.FIELDS), \
        "a circuit carried a field that is not allow-listed"


def test_the_station_name_never_leaves(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert "dob-station" not in (tmp_path / "now.json").read_text(encoding="utf-8"), \
        "the station's own id reached the public folder"


def test_the_preview_is_written_where_now_json_points(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    image = _read(tmp_path)["circuits"][0]["img"]
    assert (tmp_path / image).read_bytes()[8:12] == b"WEBP", \
        "the preview now.json names is not a WebP on disk"


def test_a_preview_no_longer_named_is_pruned(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    first = _read(tmp_path)["circuits"][0]["img"]
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 12))})
    assert not (tmp_path / first).exists(), "an old preview was left behind"


def test_a_station_that_cannot_be_read_costs_only_its_own_circuits(conn, tmp_path):
    _station(conn, "sgo-station", "DOB")
    _pass(conn, tmp_path, {"dob-station": OSError("api down"),
                           "sgo-station": _snapshot(_sounding("SGO", 5))})
    assert [c["tx"] for c in _read(tmp_path)["circuits"]] == ["SGO"], \
        "one unreadable station took the other's circuits with it"


def test_nothing_sounding_still_says_when_it_looked(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot()})
    assert _read(tmp_path)["generated_at"] == "2026-08-13T09:30:12Z", \
        "an idle pass did not say when it looked"


def test_the_sounding_start_is_given_in_utc(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert _read(tmp_path)["circuits"][0]["started"] == \
        datetime.fromtimestamp(T0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), \
        "the start was not the sounding's t0 in UTC"


def test_no_temporary_file_is_left(conn, tmp_path):
    _pass(conn, tmp_path, {"dob-station": _snapshot(_sounding("NIC", 10))})
    assert not list(tmp_path.rglob(".tmp-*")), "a half-written file was left"


def test_one_publisher_run_also_writes_the_live_pass(conn, tmp_path, monkeypatch):
    from services.publish import publisher

    monkeypatch.setattr(publisher, "build_site", lambda *a, **k: {
        "circuits": 0, "rendered": 0, "skipped": 0, "pruned": 0, "errors": [],
        "site_files": 0})
    monkeypatch.setattr(live, "fetcher", lambda api, token="": (
        lambda station: _snapshot(_sounding("NIC", 10))))
    conn.close()
    publisher.main(["--db", str(tmp_path / "api.sqlite3"),
                    "--out", str(tmp_path / "site"),
                    "--circuits", "NIC->DOB", "--api", "http://api:8000"])
    assert (tmp_path / "site" / "now.json").exists(), \
        "a run with an api did not write the live pass"
