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


def test_both_formats_reach_the_database_distinguishable(conn, tmp_path,
                                                         make_lfs, make_chirp_h5):
    """Ingesting a real run of each format must leave `sounding.format` set.

    It was NULL for every row: `ingest` read `row.get("format")` and the
    pipeline never put one there, so `/ui/soundings` showed `?` for all of
    them. A database holding both a recording and a v2 product -- which is
    what a parallel run produces -- could not be told apart at all.
    """
    import numpy as np

    from muf import pipeline
    from muf.pipeline import Options
    from services.api import ingest

    from conftest import synth_iq

    lfs = make_lfs(synth_iq(n_freq=200, window=512, echo_range_km=2700.0,
                            half_span_km=60_000.0, echo_last_bin=120))
    chirp = make_chirp_h5(np.full((4, 64), 100.0))

    for path in (lfs, chirp):
        row = pipeline.process_file(path, Options(window=512, methods=("algo",)))
        assert ingest.ingest_row(conn, row, path, tmp_path, ("algo",)) is not None

    stored = {r["file"]: r for r in
              db.rows(conn, "SELECT file, format, window FROM sounding")}

    assert stored[lfs.name]["format"] == "lfs"
    assert stored[chirp.name]["format"] == "chirp2"
    # Every row carries a window, so "re-derivable?" is answerable from the
    # database alone rather than by reopening the file.
    assert all(r["window"] for r in stored.values())


# --------------------------------------------------------------------------
# Watcher
# --------------------------------------------------------------------------

def test_the_watcher_offers_only_what_is_not_already_held(conn, tmp_path,
                                                          make_lfs, make_chirp_h5):
    """The whole point: a recurring check must cost a scan, not a re-derive.

    Pointing `ingest` at the archive root on a timer would re-run the pipeline
    over the entire history every pass, at a cost that grows with the archive
    rather than with what arrived.
    """
    import numpy as np

    from muf import pipeline
    from muf.pipeline import Options
    from services.api import ingest, watch

    from conftest import synth_iq

    lfs = make_lfs(synth_iq(n_freq=200, window=512, echo_range_km=2700.0,
                            half_span_km=60_000.0, echo_last_bin=120))
    chirp = make_chirp_h5(np.full((4, 64), 100.0))
    methods = ("algo",)

    new, found, fresh = watch.find_new([tmp_path], conn, methods, min_age_s=0)
    assert found == 2 and {p.name for p in new} == {lfs.name, chirp.name}

    row = pipeline.process_file(lfs, Options(window=512, methods=methods))
    ingest.ingest_row(conn, row, lfs, tmp_path, methods)

    new, found, _ = watch.find_new([tmp_path], conn, methods, min_age_s=0)
    assert found == 2, "still two on disk"
    assert [p.name for p in new] == [chirp.name], "the ingested one is not offered again"


def test_a_method_added_later_brings_old_soundings_back(conn, tmp_path, make_lfs):
    """`already_done` is keyed on (file, method), like `ingest`'s upsert.

    Widening --methods must not silently leave the existing rows short of the
    new estimator, with no way to notice but a column of nulls.
    """
    from muf import pipeline
    from muf.pipeline import Options
    from services.api import ingest, watch

    from conftest import synth_iq

    lfs = make_lfs(synth_iq(n_freq=200, window=512, echo_range_km=2700.0,
                            half_span_km=60_000.0, echo_last_bin=120))
    row = pipeline.process_file(lfs, Options(window=512, methods=("algo",)))
    ingest.ingest_row(conn, row, lfs, tmp_path, ("algo",))

    assert watch.find_new([tmp_path], conn, ("algo",), min_age_s=0)[0] == []
    still = watch.find_new([tmp_path], conn, ("algo", "kmeans"), min_age_s=0)[0]
    assert [p.name for p in still] == [lfs.name]


def test_a_file_still_arriving_is_left_for_the_next_pass(conn, tmp_path, make_lfs):
    """A sounding mid-write or mid-sync reads as a short sweep, and a short
    sweep does not fail -- it ingests with `sweep_complete` false and stays
    that way. Waiting one pass is cheaper than the wrong row."""
    from services.api import watch

    from conftest import synth_iq

    make_lfs(synth_iq(n_freq=200, window=512, echo_range_km=2700.0,
                      half_span_km=60_000.0, echo_last_bin=120))

    new, found, fresh = watch.find_new([tmp_path], conn, ("algo",), min_age_s=3600)
    assert found == 1 and new == [] and fresh == 1


def test_a_tree_with_no_soundings_is_skipped_not_fatal(conn, tmp_path):
    """Archives hold detection trees, digisonde products and empty days beside
    the ionograms. One of those must not stop the scan."""
    from services.api import watch

    (tmp_path / "empty").mkdir()
    new, found, _ = watch.find_new([tmp_path / "empty"], conn, ("algo",), min_age_s=0)
    assert new == [] and found == 0


# --------------------------------------------------------------------------
# Web: filtering and neighbours
# --------------------------------------------------------------------------

def _mk(conn, tmp_path, name, when, *, tx="cyprus1", fmt="lfs", muf=12.0):
    """Insert a sounding directly. `conn` must be the database the client is
    serving -- see `api_db`; the `conn` fixture is a different file."""
    from services.api import ingest
    (tmp_path / name).write_bytes(b"x")
    ingest.ingest_row(conn, {"file": name, "datetime": when, "tx": tx, "rx": "rx",
                             "format": fmt, "muf_algo": muf},
                      tmp_path / name, tmp_path, ("algo",))


@pytest.fixture
def api_db(client, tmp_path):
    """A writable handle on the database `client` reads.

    The `conn` fixture points at a different file, so seeding through it makes
    every assertion here pass or fail for the wrong reason.
    """
    with db.session(tmp_path / "api.sqlite3") as conn:
        yield conn


def test_soundings_filters_narrow_the_table(client, api_db, tmp_path):
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 00:00:00", tx="cyprus1", fmt="lfs")
    _mk(conn, tmp_path, "b.h5", "2026-08-09 00:00:00", tx="unkown", fmt="chirp2")
    _mk(conn, tmp_path, "c.h5", "2026-08-09 01:00:00", tx="unkown", fmt="chirp2",
        muf=float("nan"))
    conn.commit()

    assert "3 matching" in client.get("/ui/soundings").text
    assert "1 matching" in client.get("/ui/soundings?tx=cyprus1").text
    assert "2 matching" in client.get("/ui/soundings?fmt=chirp2").text
    assert "1 matching" in client.get("/ui/soundings?picks=none").text
    assert "2 matching" in client.get(
        "/ui/soundings?from=2026-08-09&to=2026-08-09T23:59:59").text


def test_an_unknown_sort_key_falls_back_rather_than_reaching_sql(client, api_db,
                                                                 tmp_path):
    """`sort` names a column, so it cannot be a bound parameter. It is mapped
    through a whitelist; anything else must be ignored, not interpolated."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 00:00:00")
    conn.commit()

    hostile = client.get("/ui/soundings?sort=s.id;DROP TABLE sounding--")
    assert hostile.status_code == 200
    assert "1 matching" in hostile.text
    assert db.rows(conn, "SELECT * FROM sounding"), "table survived"


def test_neighbours_follow_time_not_id(client, api_db, tmp_path):
    """Ids are ingest order, which stops matching time the moment a day is
    back-filled. Stepping through a day has to follow the clock."""
    conn = api_db
    _mk(conn, tmp_path, "late.lfs", "2026-02-04 02:00:00")     # ingested first
    _mk(conn, tmp_path, "early.lfs", "2026-02-04 01:00:00")
    conn.commit()

    ids = {r["file"]: r["id"] for r in db.rows(conn, "SELECT id, file FROM sounding")}
    assert ids["late.lfs"] < ids["early.lfs"], "ingest order is not time order here"

    page = client.get(f"/ui/sounding/{ids['early.lfs']}").text
    assert f"/ui/sounding/{ids['late.lfs']}" in page, "next is the later time"

    ends = client.get(f"/ui/sounding/{ids['late.lfs']}").text
    assert "latest sounding" in ends, "the last one offers no next"


# --------------------------------------------------------------------------
# Time bounds
# --------------------------------------------------------------------------

def test_a_bare_date_covers_the_whole_day_at_either_end():
    """`datetime` is stored with a space separator and compared as text.

    A bare date is a prefix of every timestamp on that day, and a prefix sorts
    first -- so it works as a lower bound and silently truncates as an upper
    one. `to=2026-08-09` used to drop all of the 9th.
    """
    assert db.time_bound("2026-08-09") == "2026-08-09 00:00:00"
    assert db.time_bound("2026-08-09", end=True) == "2026-08-09 23:59:59.999999"


def test_an_iso_t_bound_does_not_exclude_its_own_day():
    """Space is 0x20 and `T` is 0x54, so `'2026-08-09 00:00' >= '2026-08-09T00:00'`
    is false and `from=2026-08-09T00:00:00` returned nothing at all."""
    assert db.time_bound("2026-08-09T04:00:00") == "2026-08-09 04:00:00"
    assert db.time_bound("") is None and db.time_bound(None) is None


def test_a_whole_second_upper_bound_covers_its_own_microseconds():
    """`'…23:59:59.999999'` is longer than `'…23:59:59'` with the same prefix,
    so it compares greater and `to=…T23:59:59` dropped the whole last second.
    Timestamps here carry microseconds, so that second is never empty."""
    assert db.time_bound("2026-08-09 23:59:59", end=True) == "2026-08-09 23:59:59.999999"
    assert db.time_bound("2026-08-09 23:59:59") == "2026-08-09 23:59:59", "lower bound unchanged"
    assert db.time_bound("2026-08-09 23:59:59.5", end=True) == "2026-08-09 23:59:59.5"


def test_the_bounds_actually_select_that_day(client, api_db, tmp_path):
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-08-08 23:59:59.999")
    _mk(conn, tmp_path, "b.lfs", "2026-08-09 00:00:00.009633")   # microseconds
    _mk(conn, tmp_path, "c.lfs", "2026-08-09 23:59:59.999999")
    _mk(conn, tmp_path, "d.lfs", "2026-08-10 00:00:00")
    conn.commit()

    def n(query):
        body = client.get(f"/series/muf?method=algo&{query}").json()
        return body["count"]

    assert n("") == 4
    assert n("from=2026-08-09&to=2026-08-09") == 2, "a bare date is one whole day"
    assert n("from=2026-08-09T00:00:00&to=2026-08-09T23:59:59") == 2
    assert n("from=2026-08-08&to=2026-08-10") == 4


# --------------------------------------------------------------------------
# Web: circuits
# --------------------------------------------------------------------------

def test_the_series_shows_one_circuit_by_default(client, api_db, tmp_path):
    """MUF is a property of a path. Two circuits on one axis describe neither,
    so the default is the circuit with the most picks, not all of them."""
    conn = api_db
    for i in range(3):
        _mk(conn, tmp_path, f"cy{i}.lfs", f"2026-02-04 0{i}:00:00",
            tx="cyprus1", muf=20.0)
    _mk(conn, tmp_path, "dob.h5", "2026-08-09 00:00:00", tx="unkown", fmt="chirp2",
        muf=12.0)
    conn.commit()

    page = client.get("/ui/series?method=algo").text
    assert "3 point(s)" in page, "defaults to the busiest circuit, not the union"

    both = client.get("/ui/series?method=algo&circuit=all").text
    assert "4 point(s)" in both

    named = client.get("/ui/series?method=algo&circuit=unkown -> rx").text
    assert "1 point(s)" in named


def test_an_unknown_circuit_falls_back_rather_than_drawing_nothing(client, api_db,
                                                                   tmp_path):
    """A stale bookmark or a hand-edited query must not produce an empty chart
    that looks like "no data" -- the reason would be invisible."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 00:00:00", tx="cyprus1")
    conn.commit()

    page = client.get("/ui/series?method=algo&circuit=nowhere -> nohow")
    assert page.status_code == 200
    assert "1 point(s)" in page.text


def test_a_circuit_no_estimator_picked_is_not_offered(client, api_db, tmp_path):
    """SGO -> DOB is 381 soundings and no picks at all. Offering it as a choice
    would draw an empty chart with no way to tell why."""
    conn = api_db
    _mk(conn, tmp_path, "good.lfs", "2026-02-04 00:00:00", tx="cyprus1")
    _mk(conn, tmp_path, "dead.h5", "2026-08-04 00:00:00", tx="SGO", fmt="chirp2",
        muf=float("nan"))
    conn.commit()

    page = client.get("/ui/series?method=algo").text
    assert "cyprus1 -&gt; rx" in page or "cyprus1 -> rx" in page
    assert "SGO" not in page, "a circuit with no picks is not a choice"
