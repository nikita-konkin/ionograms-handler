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
import re
import shutil
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

from muf.geometry import Point                     # noqa: E402
from muf.reference import ReferenceSeries          # noqa: E402
from muf.reference import indices                  # noqa: E402
from services.api import acquisition as acq        # noqa: E402
from services.api.locale import en                 # noqa: E402
from services.api import auth, db, main, net       # noqa: E402
from services.api import series as series_mod      # noqa: E402
from services.api import web_routes                # noqa: E402


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as c:
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


#: One complete schedule entry. All five keys, because all five are read off
#: it by `calc_ionograms.py` with a bare subscript -- see
#: `services/api/acquisition.REQUIRED_ENTRY_KEYS`.
SCHEDULE = [{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
             "id": 1, "transmit_name": "NIC"}]


def test_the_web_queues_process_verbs_and_the_mode_edit_only(client):
    """`control.py` allows more than this. The widening to `set_config` is
    deliberate -- the sounding mode is what the emitter census on /ui/sources
    is *for* -- but it stops there."""
    for name in ("isolate", "mask", "logs", "enable"):
        r = client.post("/stations/SIM/commands", json={"name": name}, headers=CTL)
        assert r.status_code == 400, name

    ok = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"mode": "scheduled",
                               "sounder_timings": SCHEDULE}}})
    assert ok.status_code == 200


def test_settings_outside_the_web_list_are_refused(client):
    """`output_dir` decides where a week of data lands and a typo is
    unrecoverable from here. The agent would accept it; the web does not."""
    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config", "params": {"changes": {"output_dir": "/tmp/x"}}})
    assert r.status_code == 400
    assert "output_dir" in r.json()["detail"]


def test_leaving_search_mode_without_a_schedule_is_refused(client):
    """The failure the agent exists to prevent: a scheduled station with no
    schedule records nothing while every process reports healthy. Refused at
    the server too, so the operator sees it immediately."""
    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config", "params": {"changes": {"mode": "scheduled"}}})
    assert r.status_code == 400
    assert "record nothing" in r.json()["detail"]

    partial = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"mode": "scheduled",
                               "sounder_timings": [{"chirp-rate": 100e3}]}}})
    assert partial.status_code == 400
    assert "missing" in partial.json()["detail"]


def test_going_back_to_search_needs_no_schedule(client):
    """Search mode records whatever sweeps past, so it has nothing to be
    missing. The asymmetry is the point."""
    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config", "params": {"changes": {"mode": "search"}}})
    assert r.status_code == 200


def test_an_unknown_mode_is_refused(client):
    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config", "params": {"changes": {"mode": "turbo"}}})
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"]


def test_a_schedule_sent_as_json_text_is_accepted(client):
    """A browser form sends a string; the ini stores a nested list. Both are
    schedules and both have to parse, or the UI cannot post what it renders."""
    import json as _json

    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"mode": "scheduled",
                               "sounder_timings": _json.dumps([SCHEDULE])}}})
    assert r.status_code == 200


def test_a_failed_command_is_recorded_as_failed(client):
    queued = client.post("/stations/SIM/commands", json={"name": "stop"},
                         headers=CTL).json()
    client.get("/stations/SIM/commands", headers=CTL)
    client.post(f"/stations/SIM/commands/{queued['id']}/ack", headers=CTL,
                json={"results": [{"command": "stop", "ok": False,
                                   "detail": "timed out"}]})

    shown = client.get("/stations/SIM/health").json()["commands"][0]
    assert shown["state"] == "acked" and shown["ok"] is False
    assert shown["detail"] == "timed out"


def test_the_console_shows_why_a_command_failed(client):
    """It used to show the *request* in a column headed "result".

    Every refusal on DOB rendered as a bare red `failed` with the parameters
    echoed back, so "no systemd target configured for this station" -- which
    says exactly what to do -- was in the database and nowhere else. An
    operator cannot act on a colour.
    """
    why = ("no systemd target configured for this station (`target` is empty "
           "in the agent config)")
    queued = client.post("/stations/SIM/commands", json={"name": "restart"},
                         headers=CTL).json()
    client.get("/stations/SIM/commands", headers=CTL)
    client.post(f"/stations/SIM/commands/{queued['id']}/ack", headers=CTL,
                json={"results": [{"command": "restart", "ok": False,
                                   "detail": why}]})

    page = client.get("/ui").text
    assert why in page, "the agent's reason never reached the page"


def test_acking_an_unknown_command_is_accepted(client):
    """The agent has already acted; refusing the ack makes it retry."""
    r = client.post("/stations/SIM/commands/nope/ack", headers=CTL,
                    json={"results": []})
    assert r.status_code == 200 and r.json()["known"] is False


# --------------------------------------------------------------------------
# Station previews
# --------------------------------------------------------------------------

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"x" * 30)


def push_preview(client, station="SIM", tx="cyprus1", image=PNG, **meta):
    import base64

    body = {"tx": tx, "image_b64": base64.b64encode(image).decode("ascii")}
    body.update(meta)
    return client.post(f"/stations/{station}/preview", json=body, headers=CTL)


def test_a_preview_needs_the_control_scope(client):
    """A picture of what a station is receiving is a station write."""
    import base64

    r = client.post("/stations/SIM/preview",
                    json={"tx": "x", "image_b64": base64.b64encode(PNG).decode()})
    assert r.status_code == 401


def test_a_preview_replaces_the_one_before_it(client):
    """One row per circuit. This table is a live view, not a record."""
    assert push_preview(client, t0=100.0).status_code == 200
    assert push_preview(client, t0=200.0, image=PNG + b"newer").status_code == 200

    with db.session(db.DEFAULT_DB) as conn:
        stored = db.previews(conn, "SIM")
        assert len(stored) == 1
        assert stored[0]["t0"] == 200.0
        assert db.preview_image(conn, "SIM", "cyprus1") == PNG + b"newer"

    got = client.get("/preview/SIM/cyprus1.png")
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.content == PNG + b"newer"


def test_an_oversize_preview_is_refused(client):
    """The first endpoint here where an unbounded body fills a disk."""
    from services.api import agent_routes

    r = push_preview(client, image=b"\x89PNG\r\n\x1a\n"
                     + b"x" * agent_routes.MAX_PREVIEW_BYTES)
    assert r.status_code == 413


def test_something_that_is_not_a_png_is_refused(client):
    """Stored bytes are served back under `image/png`, so they must be one.

    Without this the endpoint hands a browser arbitrary content under a
    content type of the server's choosing.
    """
    assert push_preview(client, image=b"GIF89a" + b"x" * 40).status_code == 400
    assert push_preview(client, tx="").status_code == 400


def test_an_absent_preview_is_a_404_not_an_empty_image(client):
    assert client.get("/preview/SIM/nobody.png").status_code == 404


def test_forgetting_a_receiver_takes_its_pictures_too(client):
    """Otherwise the pictures outlive the panel they hung on."""
    push_preview(client)
    with db.session(db.DEFAULT_DB) as conn:
        went = db.forget_station(conn, "SIM")
        assert went["previews"] == 1
        assert db.previews(conn, "SIM") == []


def test_a_circuit_that_stops_reporting_is_swept(conn):
    """A renamed or retired transmitter must not leave a picture up forever."""
    db.save_preview(conn, "SIM", "gone", PNG, t0=1.0)
    stale = (db.datetime.now(db.timezone.utc)
             - db.timedelta(days=db.PREVIEW_RETENTION_DAYS + 1))
    conn.execute("UPDATE station_preview SET received_at = ?",
                 (stale.strftime(db.TIME_FORMAT),))
    conn.commit()

    db.save_preview(conn, "SIM", "here", PNG, t0=2.0)

    assert [p["tx"] for p in db.previews(conn, "SIM")] == ["here"]


def test_the_console_links_the_picture_and_never_carries_it(client):
    """The console reloads whole every 15 s; an inlined image would go with it."""
    client.post("/stations/health", json=report(), headers=CTL)
    push_preview(client, t0=1785846834.0, width=128, height=96,
                 freq_lo_hz=5.5e5, freq_hi_hz=1.6e7,
                 range_lo_m=2.3e6, range_hi_m=5.0e6)

    html = client.get("/ui").text

    assert '/preview/SIM/cyprus1.png?t=1785846834' in html, (
        "the URL carries `t0` or the picture freezes for an hour behind "
        "`max-age`, and without `max-age` it re-transfers four times a minute")
    assert "base64" not in html
    assert "0.6–16.0 MHz" in html


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


def test_a_product_on_the_stations_disk_reaches_the_console(client, tmp_path,
                                                            make_chirp_h5):
    """The whole path, with nothing faked but the network.

    A real product is decoded by the real agent, pushed through the real route
    and rendered by the real template. Every piece of this is exercised
    separately elsewhere; what only this test can catch is the two halves
    disagreeing about a name -- the transmitter that keys the row, the URL and
    the dedup are three places the same string has to arrive.
    """
    pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    from services.agent import runner
    from services.agent.config import StationConfig

    products = tmp_path / "products"
    products.mkdir()
    power = numpy.full((16, 600), 1.0)
    power[:, 290] = 300.0
    shutil.move(str(make_chirp_h5(power, txname="cyprus1",
                                  station_name="SIM")),
                str(products / "lfm_ionogram-cyprus1-SIM-ch000-007-"
                                "1770163210.01.h5"))

    config = StationConfig(station="SIM", server_url="http://testserver",
                           token="ctl", chirp_config=tmp_path / "none.ini",
                           output_dir=products, ringbuffer_dir=tmp_path,
                           reference_tx={})

    def opener(request, timeout=None):
        url = request.full_url.replace("http://testserver", "")
        body = json.loads(request.data.decode()) if request.data else None
        headers = {k: v for k, v in request.header_items()}
        response = (client.get(url, headers=headers)
                    if request.get_method() == "GET"
                    else client.post(url, json=body, headers=headers))
        assert response.status_code == 200, f"{url} -> {response.text[:200]}"

        class Wrapped:
            def read(self_inner):
                return response.content

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return Wrapped()

    state = runner.PreviewState()
    first = runner.run_once(config, opener=opener, include_epoch=False,
                            previews=state)
    assert first.previews_sent == 1, first.errors

    again = runner.run_once(config, opener=opener, include_epoch=False,
                            previews=state)
    assert again.previews_sent == 0, "an unchanged product must cost nothing"

    image = client.get("/preview/SIM/cyprus1.png")
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert '/preview/SIM/cyprus1.png?t=1770163210' in client.get("/ui").text


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

    new, found, fresh, _ = watch.find_new([tmp_path], conn, methods, min_age_s=0)
    assert found == 2 and {p.name for p in new} == {lfs.name, chirp.name}

    row = pipeline.process_file(lfs, Options(window=512, methods=methods))
    ingest.ingest_row(conn, row, lfs, tmp_path, methods)

    new, found, *_ = watch.find_new([tmp_path], conn, methods, min_age_s=0)
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

    new, found, fresh, _ = watch.find_new([tmp_path], conn, ("algo",), min_age_s=3600)
    assert found == 1 and new == [] and fresh == 1


def test_a_future_dated_file_is_ingested_not_withheld_forever(conn, tmp_path, make_lfs):
    """DOB's archive moved to CIFS and the NAS clock ran 5 h 43 m fast.

    `now - mtime` went negative for every product, negative is below any
    threshold, and the watcher reported the whole archive as "too fresh" on
    every pass while ingesting none of it. A file whose stamp we did not write
    tells us nothing about whether it finished writing, and withholding it is
    permanent where taking it early is self-correcting.
    """
    import os

    from services.api import watch

    from conftest import synth_iq

    lfs = make_lfs(synth_iq(n_freq=200, window=512, echo_range_km=2700.0,
                            half_span_km=60_000.0, echo_last_bin=120))
    ahead = time.time() + 20565          # the measured NAS skew
    os.utime(lfs, (ahead, ahead))

    new, found, fresh, skewed = watch.find_new([tmp_path], conn, ("algo",),
                                               min_age_s=3600)
    assert found == 1
    assert [p.name for p in new] == [lfs.name], "must not be withheld"
    assert fresh == 0, "not fresh -- mis-stamped, and the difference matters"
    assert skewed == 1

    assert "FUTURE-DATED" in watch.describe(
        {"found": 1, "new": 1, "too_fresh": 0, "future_dated": 1,
         "held_back": 0, "loaded": 1, "skipped": 0})


def test_a_tree_with_no_soundings_is_skipped_not_fatal(conn, tmp_path):
    """Archives hold detection trees, digisonde products and empty days beside
    the ionograms. One of those must not stop the scan."""
    from services.api import watch

    (tmp_path / "empty").mkdir()
    new, found, *_ = watch.find_new([tmp_path / "empty"], conn, ("algo",), min_age_s=0)
    assert new == [] and found == 0


# --------------------------------------------------------------------------
# Web: filtering and neighbours
# --------------------------------------------------------------------------

def _mk(conn, tmp_path, name, when, *, tx="cyprus1", fmt="lfs", muf=12.0, **extra):
    """Insert a sounding directly. `conn` must be the database the client is
    serving -- see `api_db`; the `conn` fixture is a different file.

    ``extra`` goes into the pipeline row verbatim, which is how a test asks for
    the columns this helper has no argument for -- ``lof_algo``, the
    coordinates, ``freq_stop``. Spelled as the pipeline spells them rather than
    as the schema does, because that is the row `ingest_row` actually reads.
    """
    from services.api import ingest
    (tmp_path / name).write_bytes(b"x")
    ingest.ingest_row(conn, {"file": name, "datetime": when, "tx": tx, "rx": "rx",
                             "format": fmt, "muf_algo": muf, **extra},
                      tmp_path / name, tmp_path, ("algo",))


#: A circuit with both ends on the map: Nicosia to Yoshkar-Ola, the path the
#: 2026-02-04 archive was recorded over. Needed by anything that models,
#: because a control point cannot be found without coordinates.
GEOMETRY = {"tx_lat": 35.0, "tx_lon": 34.0, "rx_lat": 56.38, "rx_lon": 47.53,
            "path_km": 2588.4}


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
    assert "latest sounding" in ends.lower(), "the last one offers no next"


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


# --------------------------------------------------------------------------
# Series: four parameters, and a model beside them
# --------------------------------------------------------------------------

def test_a_sounding_with_only_a_lof_is_still_a_point(client, api_db, tmp_path):
    """A trace that faded out before it reached the ceiling has a real LOF and
    no MUF. Under `muf IS NOT NULL` the whole day vanished from the chart,
    which reads as "nothing was recorded" rather than "the top was never seen"
    -- and on this archive it is 120 soundings of Juliusruh -> DOB with one
    MUF between them."""
    conn = api_db
    _mk(conn, tmp_path, "lofonly.lfs", "2026-02-04 00:00:00",
        muf=float("nan"), lof_algo=8.4)
    conn.commit()

    page = client.get("/ui/series?method=algo").text
    assert "1 point(s)" in page
    frame = _frame(page)
    assert frame["circuits"][0]["lof"] == [8.4]
    assert frame["circuits"][0]["muf"] == [None]


def test_the_frame_carries_lof_and_an_equivalent_fof2(client, api_db, tmp_path):
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0,
        lof_algo=9.0, **GEOMETRY)
    conn.commit()

    circuit = _frame(client.get("/ui/series?method=algo").text)["circuits"][0]
    assert circuit["muf"] == [20.0] and circuit["lof"] == [9.0]
    # 2588 km at 300 km gives an M-factor near 3.2, so a 20 MHz oblique MUF is
    # a little over 6 MHz vertical. Asserted as a band rather than a constant:
    # the point is that it is the inverse of the obliquity, not that the
    # secant law is reproduced here to five figures.
    assert 5.5 < circuit["fof2"][0] < 7.0


def test_the_equivalent_fof2_is_taken_over_one_hop(client, api_db, tmp_path):
    """The same convention `iri.predict` converts by.

    A path beyond `MAX_SINGLE_HOP_KM` reflects more than once, and the
    obliquity is set by one hop's ground distance. Inverting the measurement
    over the whole path would put the measured foF2 and the modelled one on
    different geometries -- and the two curves sitting side by side on this
    page is the entire reason either is drawn.
    """
    from muf import geometry

    long_path = 6000.0
    conn = api_db
    _mk(conn, tmp_path, "far.lfs", "2026-02-04 06:00:00", muf=20.0,
        **{**GEOMETRY, "path_km": long_path})
    conn.commit()

    circuit = _frame(client.get("/ui/series?method=algo").text)["circuits"][0]
    assert circuit["hops"] == 2
    hop = geometry.muf_to_fof2(20.0, long_path / 2, series_mod.EQUIVALENT_HMF2_KM)
    whole = geometry.muf_to_fof2(20.0, long_path, series_mod.EQUIVALENT_HMF2_KM)
    assert circuit["fof2"][0] == pytest.approx(hop)
    assert circuit["fof2"][0] != pytest.approx(whole), "the two must differ here"


def test_a_circuit_without_coordinates_says_so_rather_than_drawing_nothing(
        client, api_db, tmp_path, monkeypatch):
    """`unkown -> DOB` is 647 soundings with no transmitter position.

    The model cannot find a control point without both ends, and a panel that
    was simply empty would read as "IRI agrees with nothing" rather than as
    "IRI was never asked".
    """
    monkeypatch.setattr(series_mod, "MODEL", True)
    conn = api_db
    _mk(conn, tmp_path, "nowhere.lfs", "2026-02-04 06:00:00", muf=20.0)
    conn.commit()

    page = client.get("/ui/series?method=algo").text
    assert "no coordinates stored for this circuit" in page
    circuit = _frame(page)["circuits"][0]
    assert circuit["model"]["error"]
    assert circuit["fof2"] == [None], "no geometry, no obliquity to invert"


def test_the_page_survives_a_model_that_is_not_installed(client, api_db,
                                                         tmp_path, monkeypatch):
    """A missing optional dependency is a normal condition, not a 500."""
    from muf.reference import iri

    monkeypatch.setattr(series_mod, "MODEL", True)
    monkeypatch.setattr(iri, "available", lambda: False)
    series_mod.clear()

    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    conn.commit()

    page = client.get("/ui/series?method=algo")
    assert page.status_code == 200
    assert "not installed" in page.text
    assert "1 point(s)" in page.text, "the measurement is still drawn"


def test_the_model_is_driven_by_each_days_own_sun(monkeypatch):
    """One IRI call per day, not one per window.

    `iri.predict` reads its solar driver from the *first* timestamp it is
    given. Handed a window holding February and August it would model both
    with February's sun and report a single F10.7 that looks like a fact. The
    page's day pills make multi-day windows normal, so the split is here.
    """
    import pandas as pd

    from muf.reference import iri

    calls = []

    def fake(tx, rx, times, **_):
        index = pd.DatetimeIndex(pd.to_datetime(list(times)))
        calls.append(index)
        return ReferenceSeries(
            name="iri", muf=pd.Series(20.0, index=index),
            detail=pd.DataFrame({"fof2": pd.Series(6.0, index=index),
                                 "hmf2": pd.Series(300.0, index=index)}),
            source=f"fake, F10.7={len(calls) * 10}")

    monkeypatch.setattr(iri, "available", lambda: True)
    monkeypatch.setattr(iri, "predict", fake)
    series_mod.clear()

    got = series_mod.model_for(Point(35, 34), Point(56, 47),
                               ["2026-02-04T00:00:00.000",
                                "2026-02-04T12:00:00.000",
                                "2026-08-09T00:00:00.000"])
    assert len(calls) == 2, "one call per day"
    assert [len(c) for c in calls] == [2, 1]
    assert got["muf"] == [20.0, 20.0, 20.0]
    assert "more driver(s)" in got["source"], "two drivers are not one driver"


def test_a_day_the_model_fails_on_does_not_take_the_others_with_it(monkeypatch):
    import pandas as pd

    from muf.reference import iri

    def fake(tx, rx, times, **_):
        index = pd.DatetimeIndex(pd.to_datetime(list(times)))
        if index[0].month == 8:
            return ReferenceSeries(name="iri", error="no solar driver")
        return ReferenceSeries(name="iri", muf=pd.Series(20.0, index=index),
                               detail=pd.DataFrame({"fof2": pd.Series(6.0, index=index)}),
                               source="fake")

    monkeypatch.setattr(iri, "available", lambda: True)
    monkeypatch.setattr(iri, "predict", fake)
    series_mod.clear()

    got = series_mod.model_for(Point(35, 34), Point(56, 47),
                               ["2026-02-04T00:00:00.000",
                                "2026-08-09T00:00:00.000"])
    assert got["muf"] == [20.0, None]
    assert "1 day(s) unmodelled" in got["note"]
    assert got["hmf2"] == [None, None], "a detail column that is absent is None"


def test_the_model_declines_a_window_it_cannot_afford(monkeypatch):
    """The cost is one evaluation per day, paid while the request is open. A
    page that quietly takes a minute is worse than one that says why not."""
    from muf.reference import iri

    monkeypatch.setattr(iri, "available", lambda: True)
    monkeypatch.setattr(iri, "predict", lambda *a, **k: pytest.fail("asked anyway"))
    monkeypatch.setattr(series_mod, "MAX_MODEL_DAYS", 1)
    series_mod.clear()

    got = series_mod.model_for(Point(35, 34), Point(56, 47),
                               ["2026-02-04T00:00:00.000",
                                "2026-02-05T00:00:00.000"])
    assert "spans 2 days" in got["error"]


def test_a_lower_bound_is_counted_but_not_scored(client):
    """A `limited` MUF says the ionosphere supported *at least* that much.

    Scoring it as a residual reports the recorder's band ceiling as a
    modelling error, and on a ceiling-limited circuit that is most of the
    daytime. It is excluded from the statistics and counted beside them --
    a bias over four of forty points is a different claim from one over forty.
    """
    got = series_mod.compare(measured=[20.0, 22.0, 30.0], modelled=[19.0, 21.0, 10.0],
                             limited=[0, 0, 1])
    assert got["n"] == 2 and got["excluded"] == 1
    assert got["bias"] == pytest.approx(1.0), "the bound did not move it"
    assert got["rms"] == pytest.approx(1.0)

    none = series_mod.compare([20.0], [None], [0])
    assert none["n"] == 0 and "bias" not in none


def test_the_frame_is_json_a_browser_will_parse(client, api_db, tmp_path):
    """Every array crosses into the page through `|tojson`.

    Python writes a bare `NaN` for a float NaN and `JSON.parse` refuses it, so
    one absent pick would blank the whole plot with the reason visible only in
    a console nobody has open. `allow_nan=False` is that failure, here.
    """
    rows = [{"id": 1, "datetime": "2026-02-04 00:00:00.009633",
             "tx": "cyprus1", "rx": "rx", "muf": float("nan"),
             "lof": None, "limited": None, "loflim": None,
             "muf_smooth": float("nan"), "freq_stop": 30.0, **GEOMETRY}]
    got = series_mod.frame(rows, model="off")

    json.dumps(got, allow_nan=False)                 # the whole assertion
    circuit = got["circuits"][0]
    assert circuit["muf"] == [None] and circuit["fof2"] == [None]
    assert circuit["t"] == ["2026-02-04T00:00:00.009"], "milliseconds, and a T"


# --------------------------------------------------------------------------
# Drawing a forecast against the measurement it is meant to predict
#
# The plot could already draw a forecast trace; what it could not do is reach
# a model that is not live. Every legacy import is `comparison` and can never
# be promoted (registry.activate refuses a modelled target), so on a rig where
# the only models are legacy the overlay was permanently empty -- which is the
# state the local OrbStack rig was in with 9,352 forecast rows in the table.
# --------------------------------------------------------------------------

def _register(conn, name, *, param="muf", tx="cyprus1", rx="rx", active=0,
              trained=None):
    """A registered model. `trained` is the (from, to) span a fit recorded.

    Left null by default, because the default here is a legacy import and an
    import genuinely has no recorded window -- see
    `test_an_imported_model_gets_no_band_because_it_has_no_recorded_window`.
    """
    cur = conn.execute(
        "INSERT INTO model_registry (name, param, tx, rx, origin, framework, "
        "loader, capability, artifact, sha256, features, target_alias, "
        "target_src, imported_at, active) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, param, tx, rx, "legacy", "sklearn", "joblib", "slim",
         f"/models/{name}.sav", name, '["muf_lag_288"]', "muf",
         "modelled" if not active else "measured", db.utcnow(), active))
    model_id = int(cur.lastrowid)
    if trained:
        conn.execute("UPDATE model_registry SET origin = 'trained', "
                     "target_src = 'measured', trained_from = ?, "
                     "trained_to = ? WHERE id = ?", (*trained, model_id))
    return model_id


def _forecast(conn, model_id, *, param="muf", tx="cyprus1", rx="rx",
              issued="2026-02-04T12:00:00Z", stamps=(), value=21.0):
    for stamp in stamps:
        conn.execute(
            "INSERT INTO forecast (model_id, param, tx, rx, issued_at, "
            "valid_at, horizon_s, value, sigma) VALUES (?,?,?,?,?,?,?,?,?)",
            (model_id, param, tx, rx, issued, stamp, 86400, value, 0.4))


def test_a_comparison_model_is_not_drawn_unless_it_is_named(client, api_db,
                                                            tmp_path):
    """The default is still active-only, exactly as `/forecast` behaves."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "huber_lag288", active=0)
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    frame = _frame(client.get("/ui/series?method=algo").text)
    assert frame["any_forecast"] is False, (
        "a comparison model was drawn without being asked for; the operational "
        "curve and a candidate would then be indistinguishable")


def test_naming_a_model_draws_it_beside_the_picks(client, api_db, tmp_path):
    """The comparison this page exists to make."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "huber_lag288", active=0)
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z",
                                   "2026-02-05T07:00:00Z"])
    conn.commit()

    frame = _frame(client.get(f"/ui/series?method=algo&forecast={model}").text)
    assert frame["any_forecast"] is True
    drawn = frame["circuits"][0]["forecast"]
    assert len(drawn) == 1
    assert drawn[0]["model"] == "huber_lag288"
    assert drawn[0]["value"] == [21.0, 21.0]
    # Its own x axis, not the soundings'. The forecast reaches a day past the
    # single pick, and that overhang is the only part worth looking at.
    #
    # No trailing Z: `_iso_stamp` strips it so the forecast and the picks
    # land on one plotly axis. The soundings are naive UTC, and a mixed
    # axis draws the forecast lagging its own measurement by the local
    # timezone.
    assert drawn[0]["t"] == ["2026-02-05T06:00:00", "2026-02-05T07:00:00"]


def test_a_named_comparison_says_it_is_one(client, api_db, tmp_path):
    """`comparison` is what makes the plot dot the line instead of dashing it.

    A candidate drawn in the operational style is the failure this whole
    distinction exists to prevent.
    """
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "huber_lag288", active=0)
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    page = client.get(f"/ui/series?method=algo&forecast={model}").text
    assert _frame(page)["circuits"][0]["forecast"][0]["comparison"] is True


def test_an_active_model_is_not_labelled_a_comparison(client, api_db, tmp_path):
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "operational", active=1)
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    # Drawn with no `forecast=` at all, because it is live.
    frame = _frame(client.get("/ui/series?method=algo").text)
    assert frame["circuits"][0]["forecast"][0]["comparison"] is False


def test_only_models_that_wrote_rows_are_offered(client, api_db, tmp_path):
    """A chooser entry that draws nothing is indistinguishable from a bug."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    ran = _register(conn, "has_run")
    _register(conn, "never_run")
    _forecast(conn, ran, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    page = client.get("/ui/series?method=algo").text
    assert "has_run" in page
    assert "never_run" not in page


def test_a_stale_forecast_link_renders_the_page_anyway(client, api_db,
                                                       tmp_path):
    """An id that no longer exists is a dead bookmark, not a 404.

    Models are retired and re-imported; a link in somebody's notes outliving
    one is ordinary, and losing the whole page over it is not.
    """
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    conn.commit()

    response = client.get("/ui/series?method=algo&forecast=9999")
    assert response.status_code == 200
    assert _frame(response.text)["any_forecast"] is False


def test_the_chosen_model_survives_a_day_or_method_click(client, api_db,
                                                         tmp_path):
    """Losing the overlay on the next click is how a comparison gets abandoned."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "huber_lag288")
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    page = client.get(f"/ui/series?method=algo&forecast={model}").text
    links = re.findall(r'href="(/ui/series\?[^"]+)"', page)
    days = [href for href in links if "from=" in href]
    assert days, "no day links on the page to check"
    assert all(f"forecast={model}" in href for href in days), (
        "a day link dropped the chosen model")


# --------------------------------------------------------------------------
# What the model learned on
#
# A lagged model run over a finished archive predicts instants that have
# already happened -- `infer` calls that a backtest. Where those instants fall
# inside the span the model was *fitted* on, the curve is not a prediction at
# all: it is the model reciting rows it has already seen, and it will look
# superb. The plot shades that span so the two halves of the same line can be
# told apart by eye; these pin the data it shades from.
# --------------------------------------------------------------------------

def test_a_trained_model_carries_the_span_it_was_fitted_on(client, api_db,
                                                           tmp_path):
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "huber-muf-24h", active=1,
                      trained=("2026-01-20T00:00:00Z", "2026-02-03T00:00:00Z"))
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    frame = _frame(client.get("/ui/series?method=algo").text)
    drawn = frame["circuits"][0]["forecast"][0]

    # Stripped of the trailing Z, exactly as `t` is: the band and the curve
    # share an axis with the naive-UTC soundings, and a mixed axis would put
    # the shading a whole timezone away from the hours it describes.
    assert drawn["trained_from"] == "2026-01-20T00:00:00"
    assert drawn["trained_to"] == "2026-02-03T00:00:00"
    assert frame["any_trained"] is True


def test_an_imported_model_gets_no_band_because_it_has_no_recorded_window(
        client, api_db, tmp_path):
    """Null, not a zero-width band, and the difference is the whole point.

    Every legacy import was fitted somewhere else on somebody else's circuit.
    "Trained on nothing" and "we do not know what it was trained on" are
    different claims, and only one of them is true here.
    """
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    model = _register(conn, "legacy_import", active=1)
    _forecast(conn, model, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    frame = _frame(client.get("/ui/series?method=algo").text)
    drawn = frame["circuits"][0]["forecast"][0]

    assert drawn["trained_from"] is None
    assert drawn["trained_to"] is None
    assert frame["any_trained"] is False


def test_the_band_is_explained_only_when_there_is_a_band(client, api_db,
                                                         tmp_path):
    """A sentence about shading, printed beside a plot with none, sends the
    reader hunting for something that is not there."""
    conn = api_db
    _mk(conn, tmp_path, "a.lfs", "2026-02-04 06:00:00", muf=20.0, **GEOMETRY)
    imported = _register(conn, "legacy_import")
    _forecast(conn, imported, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()
    note = en.MESSAGES["series.trained_note"][:40]

    assert note not in client.get(
        f"/ui/series?method=algo&forecast={imported}").text

    trained = _register(conn, "huber-muf-24h",
                        trained=("2026-01-20T00:00:00Z",
                                 "2026-02-03T00:00:00Z"))
    _forecast(conn, trained, stamps=["2026-02-05T06:00:00Z"])
    conn.commit()

    assert note in client.get(
        f"/ui/series?method=algo&forecast={trained}").text


def _frame(page: str) -> dict:
    """The JSON the page hands plotly, out of the rendered HTML."""
    import re

    found = re.search(r'<script id="series-frame"[^>]*>(.*?)</script>',
                      page, re.S)
    assert found, "the page carries no frame"
    return json.loads(found.group(1))


# --------------------------------------------------------------------------
# Sources: search mode's output is scheduled mode's input
# --------------------------------------------------------------------------

def test_the_census_offers_each_emitter_as_a_schedule_entry(tmp_path,
                                                            make_detection_h5):
    """The join between the two sounding modes. `control.py` refuses to leave
    search mode without a `sounder_timings` list, so search has to be able to
    produce one."""
    from services.api import sources

    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    got = sources.census(tmp_path, max_days=2, min_count=2)
    assert got["count"] >= 1, got
    entry = got["emitters"][0]["timing_entry"]
    assert set(entry) == {"chirp-rate", "rep", "chirpt"}, entry
    assert entry["chirp-rate"] > 0 and entry["rep"] > 0


def test_days_are_ranked_by_date_not_by_string(tmp_path):
    """Both halves of this were live on the real archive.

    `2026.02.09` sorts above `2026-08-10` because '.' is 0x2E and '-' is 0x2D,
    so a reverse lexical sort served February as "what is on air today"; and
    `ionozond_data2` beats every digit, taking a slot without being a day at
    all. Between them, no August directory was ever opened.
    """
    from services.api import sources

    for name in ("2026-08-09", "2026-08-10", "2026.02.04", "2026.02.09",
                 "20260807", "ionozond_data2", "logs"):
        (tmp_path / name).mkdir()

    picked = [p.name for p in sources._day_directories(tmp_path, 3)]
    assert picked == ["2026-08-10", "2026-08-09", "20260807"], picked


def test_a_flat_archive_still_scans_the_root(tmp_path):
    """No dated subdirectory means the products are here, not below."""
    from services.api import sources

    (tmp_path / "ionozond_data2").mkdir()
    assert sources._day_directories(tmp_path, 3) == [tmp_path]


def test_interference_is_rejected_by_shape_not_by_strength():
    """The loudest group on DOB was the least real.

    500 kHz/s, median SNR 68 -- above cyprus1 -- claiming every one of the 300
    seconds in the cycle with a fractional offset scattering +/-274 ms. A
    transmitter is quiet in most seconds and arrives at the same instant
    within the ones it uses; strength says nothing either way.
    """
    from types import SimpleNamespace

    from services.api import sources

    def verdict(sd_ms, slots, count):
        return sources._rejection(
            SimpleNamespace(fraction_sd_s=sd_ms / 1e3, count=count,
                            observed_seconds=list(range(slots))),
            300.0, sources.DEFAULT_MAX_SCATTER_S,
            sources.DEFAULT_MAX_SLOT_FRACTION, sources.DEFAULT_MIN_REPEATS)

    assert verdict(0.91, 6, 235) is None, "the tightest real emitter must survive"
    assert verdict(2.27, 37, 31637) is None, "cyprus1's group, 12% of the cycle"
    assert "274 ms" in verdict(273.93, 300, 11468)
    assert "scatters" in verdict(20.54, 150, 2844)
    # Tight but everywhere: rejected on occupancy alone, so a narrow-scatter
    # detector artefact cannot slip through by being consistent.
    assert "sparse" in verdict(0.5, 200, 4000)


def test_each_slot_heard_once_is_coincidence_not_a_schedule():
    """Thirteen groups on DOB scored exactly 1.0 detection per slot.

    A transmitter on a 300 s cycle hands back the same second every cycle --
    the real ones scored 24, 39 and 855. Nothing landed between 1.5 and 24, so
    this is the cleanest of the three cuts. It is also independent of scatter:
    the 500 kHz/s interference repeated 38 times per slot and still isn't a
    transmitter, while these are phase-tight and still aren't.
    """
    from types import SimpleNamespace

    from services.api import sources

    def verdict(sd_ms, slots, count):
        return sources._rejection(
            SimpleNamespace(fraction_sd_s=sd_ms / 1e3, count=count,
                            observed_seconds=list(range(slots))),
            300.0, sources.DEFAULT_MAX_SCATTER_S,
            sources.DEFAULT_MAX_SLOT_FRACTION, sources.DEFAULT_MIN_REPEATS)

    assert "coincidence" in verdict(3.80, 12, 12)     # phase-tight, once each
    assert "coincidence" in verdict(0.08, 3, 3)       # tightest of all, n=3
    assert "coincidence" in verdict(4.72, 34, 52)     # 1.5 per slot
    assert verdict(1.58, 6, 144) is None              # 24 per slot -- real


def test_rejects_are_reported_not_hidden(tmp_path, make_detection_h5):
    """A schedule page that silently drops rows cannot be checked, and the
    operator is the one who knows whether the discard was what they came for."""
    from services.api import sources

    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    # Via occupancy, not scatter: a synthetic emitter has *exactly* zero
    # scatter, and zero is not greater than any threshold.
    got = sources.census(tmp_path, max_days=2, min_count=2,
                         max_slot_fraction=0.0)
    assert got["emitters"] == []
    assert got["rejected"], "everything was dropped and nothing said so"
    assert "rejected_because" in got["rejected"][0]


def test_the_census_names_no_transmitter(tmp_path, make_detection_h5):
    """Nothing in a detection identifies who sent it. A guessed name would
    reach the product file name and then the database, looking like
    knowledge -- so `transmit_name` is left for the operator."""
    from services.api import sources

    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    for emitter in sources.census(tmp_path, min_count=2)["emitters"]:
        assert "transmit_name" not in emitter["timing_entry"]


def test_an_archive_with_no_detections_is_empty_not_an_error(tmp_path):
    """A station that has never run search mode, or whose detection files have
    not synced, is a normal state and not a failure."""
    from services.api import sources

    got = sources.census(tmp_path)
    assert got == {"count": 0, "kind": "none", "cycle_s": got["cycle_s"],
                   "emitters": [], "cost": got["cost"]}
    assert got["cost"]["files"] == 0


def test_the_sources_page_and_endpoint_agree(cold_census, client, tmp_path,
                                             make_detection_h5):
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    client.app.state.archive_root = tmp_path
    # Neither surface reads the archive on the request path, so warm the cache
    # first -- otherwise both agree on "still building" and prove nothing.
    warmed = cold_census.census(tmp_path, min_count=2)
    assert warmed["count"] > 0

    api = client.get("/sources?min_count=2").json()
    page = client.get("/ui/sources?min_count=2")
    assert page.status_code == 200
    assert api["count"] == warmed["count"], "served something else"
    assert f"{api['count']} emitter(s)" in page.text


# --------------------------------------------------------------------------
# What made the page take minutes
# --------------------------------------------------------------------------
#
# One HDF5 open per detection file, on every page load, with nothing cached.
# ~1850 opens for three days of DOB: 0.6 s on a local SSD and two to three
# minutes on the network archive the server reads. These files are written
# once and never touched, so the fix is to stop re-reading them -- and the
# risk a cache introduces is a page that is fast and wrong, which is what
# these tests are about.

@pytest.fixture
def cold_census():
    """A census with an empty cache, restored afterwards."""
    from services.api import sources

    sources._MEMO.clear()
    sources._LAST.clear()
    # A refresh spawned by an earlier test may still be in flight, and its flag
    # is what decides whether the next one is allowed to start.
    sources._REFRESHING = False
    yield sources
    sources._MEMO.clear()
    sources._LAST.clear()
    sources._REFRESHING = False


def test_an_unchanged_archive_is_not_read_twice(cold_census, tmp_path,
                                                make_detection_h5):
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    first = cold_census.census(tmp_path, min_count=2)
    assert first["cost"]["opened"] == first["cost"]["files"] > 0

    second = cold_census.census(tmp_path, min_count=2)
    assert second["cost"]["opened"] == 0
    assert second["cost"]["unchanged"] is True
    assert second["emitters"] == first["emitters"]


def test_a_new_detection_file_is_picked_up(cold_census, tmp_path,
                                           make_detection_h5):
    """The whole point of a page called "transmitters heard" is that it
    changes when the station hears something new."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    cold_census.census(tmp_path, min_count=2)

    grown = day / "chirp-ch0-100-44664265260000000-1999999999.h5"
    shutil.copy(next(day.glob("chirp-*.h5")), grown)

    after = cold_census.census(tmp_path, min_count=2)
    assert after["cost"]["unchanged"] is False
    assert after["cost"]["opened"] == 1, "re-read the whole archive for one file"
    assert after["cost"]["cached"] == after["cost"]["files"] - 1


def test_a_deleted_file_leaves_the_census(cold_census, tmp_path,
                                          make_detection_h5):
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    before = cold_census.census(tmp_path, min_count=2)

    for path in list(day.glob("chirp-*.h5")):
        path.unlink()
    after = cold_census.census(tmp_path, min_count=2)

    assert before["emitters"], "nothing to lose, so nothing was proven"
    assert after["emitters"] == []
    assert after["cost"]["files"] == 0


def test_the_tuning_parameters_are_part_of_the_key(cold_census, tmp_path,
                                                   make_detection_h5):
    """`?min_count=` on the URL must not be answered from a run that used a
    different one -- the same archive has several right answers."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    strict = cold_census.census(tmp_path, min_count=2, max_slot_fraction=0.0)
    loose = cold_census.census(tmp_path, min_count=2)
    assert strict["emitters"] == []
    assert loose["emitters"], "answered from the stricter run's cache"


def test_a_file_caught_mid_write_is_retried_when_it_grows(cold_census,
                                                          tmp_path,
                                                          make_detection_h5):
    """The one case where a path's content really does change.

    Detection files are written by a running detector, so a scan will
    occasionally catch one truncated. Skipping it is right; remembering the
    skip forever is not, because the completed file has the same name.
    """
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    good = next(day.glob("chirp-*.h5"))
    body = good.read_bytes()

    truncated = day / "chirp-ch0-100-44664265260000000-1999999999.h5"
    truncated.write_bytes(body[:len(body) // 3])
    with pytest.warns(UserWarning, match="unreadable"):
        first = cold_census.census(tmp_path, min_count=2)
    n_first = first["cost"]["records"]

    truncated.write_bytes(body)                    # the writer finished
    after = cold_census.census(tmp_path, min_count=2)
    assert after["cost"]["opened"] == 1
    assert after["cost"]["records"] == n_first + 1, "still skipping a good file"


def test_one_unreadable_file_does_not_disable_the_short_circuit(cold_census,
                                                                tmp_path,
                                                                make_detection_h5):
    """It used to, and a live archive always has one.

    The test was "did the last census skip anything at all", so a single
    truncated file among 1846 -- the normal state of a directory a detector is
    writing into -- turned every later page load into a full re-read and
    re-group, for the rest of the process's life. The cache was off exactly
    where it was needed. Only the files that actually failed are re-stat-ed
    now, and a file that failed and has not moved since leaves the cache good.
    """
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    body = next(day.glob("chirp-*.h5")).read_bytes()

    truncated = day / "chirp-ch0-100-44664265260000000-1999999999.h5"
    truncated.write_bytes(body[:len(body) // 3])
    with pytest.warns(UserWarning, match="unreadable"):
        first = cold_census.census(tmp_path, min_count=2)
    assert first["cost"]["opened"] > 0

    again = cold_census.census(tmp_path, min_count=2)
    assert again["cost"]["unchanged"] is True, "one bad file re-read the archive"
    assert again["cost"]["opened"] == 0
    assert again["emitters"] == first["emitters"]


# --------------------------------------------------------------------------
# The ceiling: an archive too big to census the way the page asks
# --------------------------------------------------------------------------
#
# The day bound assumed a day was a bounded amount of work. On 2026-08-15 DOB's
# newest three days held 172,056 files, 45,602 of them the `chirp-*.h5` this
# reads first -- 93x what the cache above was measured against. The warm-up
# took the census lock and never came back, so every request queued behind a
# read that would have taken hours: the page did not answer slowly, it did not
# answer. These tests are about answering about part of the archive, loudly,
# instead of starting a read that cannot finish.

def test_a_census_reads_the_newest_files_and_says_it_capped(cold_census,
                                                            tmp_path,
                                                            make_detection_h5):
    """The recent end is the end the page is about."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=8, into=day)            # 24 files
    every = sorted(p.name for p in day.glob("chirp-*.h5"))

    with pytest.warns(UserWarning, match="ceiling"):
        got = cold_census.census(tmp_path, min_count=2, max_files=10)

    cost = got["cost"]
    assert cost["found"] == len(every) == 24
    assert (cost["files"], cost["capped"], cost["budget"]) == (10, 14, 10)
    read = sorted(Path(k).name for k in cold_census._MEMO)
    assert read == every[-10:], "trimmed the recent end, not the old one"


def test_the_newest_files_are_the_newest_and_not_the_last_alphabetically(
        cold_census, tmp_path):
    """The `i0` field has no fixed width, so sorting on the name is not
    sorting on time.

    A real DOB name is `chirp-<channel>-<rate>-<i0>-<unix>.h5` where `i0` is a
    sample index: `9000` and `44664265260000000` sort `9` after `4`, putting an
    older file last. Under a ceiling that keeps the tail, that is silently the
    wrong 2000 files -- ordered by channel and sample index, with time as a
    tiebreak.
    """
    from services.api.sources import _file_time

    old = tmp_path / "chirp-ch0-100-9000-1785888000.h5"
    new = tmp_path / "chirp-ch0-100-44664265260000000-1785899000.h5"
    assert sorted([old, new], key=lambda p: p.name) == [new, old], \
        "the name sort no longer inverts these, so this test proves nothing"
    assert sorted([old, new], key=_file_time) == [old, new]

    junk = tmp_path / "chirp-ch0-100-0-partial.h5"
    assert sorted([new, junk], key=_file_time)[0] == junk, "dropped a good file"


def test_one_directory_pass_answers_for_all_three_products(tmp_path,
                                                            make_detection_h5):
    """Three finders meant three walks of the same tree. On a station that
    writes no `par-*.h5`, the first of them visited all 45,602 `chirp-*.h5`
    to return an empty list, and then the second visited them again."""
    from muf import io_detect

    make_detection_h5("chirp", cycles=2, into=tmp_path)
    make_detection_h5("cdetections", cycles=2, into=tmp_path)
    (tmp_path / "lfm_ionogram-DOB-007-1785888000.h5").write_bytes(b"not ours")

    got = io_detect.find_products(tmp_path)
    assert set(got) == set(io_detect.PRODUCT_PREFIXES)
    assert got["par"] == []
    assert sorted(got["chirp"]) == sorted(io_detect.find_detections(tmp_path))
    assert sorted(got["cdetections"]) == sorted(
        io_detect.find_cdetections(tmp_path))
    assert not any("lfm_ionogram" in str(p)
                   for paths in got.values() for p in paths)


def test_the_budget_is_spent_on_the_newest_day_first(cold_census, tmp_path,
                                                     make_detection_h5):
    """Degrading a whole day is better than half-reading two.

    Both days are wanted, but if only some files can be opened they should be
    today's: an emitter that stopped yesterday is not what "what is on air"
    means, and a schedule is built from the current one.
    """
    old, new = tmp_path / "2026-08-08", tmp_path / "2026-08-09"
    old.mkdir(), new.mkdir()
    make_detection_h5("chirp", cycles=4, into=old, base_epoch=1785801600.0)
    make_detection_h5("chirp", cycles=4, into=new, base_epoch=1785888000.0)

    with pytest.warns(UserWarning, match="ceiling"):
        got = cold_census.census(tmp_path, max_days=2, min_count=2,
                                 max_files=12)

    assert (got["cost"]["found"], got["cost"]["capped"]) == (24, 12)
    assert {Path(k).parent.name for k in cold_census._MEMO} == {"2026-08-09"}


def test_the_ceiling_trims_time_and_not_quality(cold_census, tmp_path,
                                                make_detection_h5):
    """The cheap files are cheap because they are the detector's raw
    candidates. Falling back to them under load is how the census loses the
    transmitter the page exists to find -- 84 `cdetections-*.h5` hold what
    45,602 `chirp-*.h5` do on DOB, and on one real day they produced a
    100 kHz/s "emitter" with 26,137 detections that the occupancy filter
    throws away. So the ceiling drops files, never the product."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=8, into=day)            # 24 files
    make_detection_h5("cdetections", cycles=8, into=day)      # 1 file, same data

    with pytest.warns(UserWarning, match="ceiling"):
        got = cold_census.census(tmp_path, min_count=2, max_files=6)

    assert got["kind"] == "detection", "fell back to the consolidated files"
    assert got["cost"]["found"] == 24, "counted a product it did not read"
    assert not [k for k in cold_census._MEMO if "cdetections" in k]


def test_an_archive_within_the_ceiling_is_not_marked_capped(cold_census,
                                                            tmp_path,
                                                            make_detection_h5):
    """The notice has to mean something, so the ordinary archive must not
    raise it."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    got = cold_census.census(tmp_path, min_count=2)
    assert got["cost"]["capped"] == 0
    assert got["cost"]["found"] == got["cost"]["files"] == 18


def test_the_page_says_the_census_only_read_part_of_the_archive(
        cold_census, client, tmp_path, monkeypatch, make_detection_h5):
    """A capped answer that looks like a whole one is the worst of the three.

    The operator reads this page to decide what to sound; "no such emitter"
    and "not in the part I read" are different answers and only one of them
    is a reason to stop looking.
    """
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=8, into=day)
    client.app.state.archive_root = tmp_path

    real = cold_census.census
    monkeypatch.setattr(
        web_routes.sources_mod, "census",
        lambda root, **kw: real(root, **dict(kw, max_files=10, block=True)))
    with pytest.warns(UserWarning, match="ceiling"):
        page = client.get("/ui/sources?min_count=2")

    assert page.status_code == 200
    assert "newest 10 of 24 file(s)" in page.text
    assert "capped" in page.text


# --------------------------------------------------------------------------
# The request path never touches the archive
# --------------------------------------------------------------------------
#
# Measured on the deployed server: one `os.scandir` of `/archive/2026-08-15`
# returned 46,436 entries in **293.8 s**. That is 6.3 ms per directory entry --
# a network round trip each -- so listing three days costs a quarter of an hour
# before the first file is opened, and no ceiling on files opened can help. The
# page therefore answers from the last completed census and says how old it is.

def test_a_request_never_waits_for_the_archive(cold_census, tmp_path,
                                               monkeypatch, make_detection_h5):
    """The scan is the cost, so `block=False` must not reach it at all."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    from muf import io_detect
    monkeypatch.setattr(io_detect, "find_products",
                        lambda target: pytest.fail("scanned on a request"))
    # As if a refresh were already in flight, so the request path is the only
    # thing running and a background thread cannot muddy the assertion.
    monkeypatch.setattr(cold_census, "_REFRESHING", True)

    got = cold_census.census(tmp_path, min_count=2, block=False)
    assert got["building"] is True
    assert got["emitters"] == [] and got["cost"]["files"] == 0


def test_nothing_yet_is_not_the_same_as_nothing_heard(cold_census, tmp_path,
                                                      make_detection_h5):
    """An empty page and an unfinished census look identical, and only one of
    them means the station is not hearing anything."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    building = cold_census.census(tmp_path, min_count=2, block=False)
    assert building.get("building") is True

    cold_census.census(tmp_path, min_count=2)               # the real thing
    served = cold_census.census(tmp_path, min_count=2, block=False)
    assert not served.get("building")
    assert served["emitters"], "served nothing after a completed census"
    assert served["age_s"] is not None


def test_a_stale_census_is_served_rather_than_a_new_one_awaited(
        cold_census, tmp_path, make_detection_h5):
    """Past `max_age_s` the answer is still the old one -- the refresh happens
    behind it. Blocking would hand the fifteen-minute scan to whoever opened
    the page, which is the whole failure this replaces."""
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    fresh = cold_census.census(tmp_path, min_count=2)

    cold_census._LAST["at"] = time.time() - 10_000          # long past due
    served = cold_census.census(tmp_path, min_count=2, block=False,
                                max_age_s=60)
    assert served["emitters"] == fresh["emitters"], "dropped a usable answer"
    assert served["age_s"] > 60


def test_only_one_background_refresh_runs_at_a_time(cold_census):
    """Two requests can both find the archive idle. If both spawn, the second
    repeats a scan the first is already doing -- half an hour of a mount that
    takes 294 s to list one day."""
    import threading

    running, release, second = threading.Event(), threading.Event(), []

    def slow():
        running.set()
        release.wait(5)

    assert cold_census._start_refresh(slow) is True
    assert running.wait(5), "the refresh never started"
    try:
        assert cold_census._start_refresh(lambda: second.append(1)) is False
    finally:
        release.set()
    assert second == [], "a second scan of the same archive was queued"


def test_the_page_says_a_census_is_still_being_built(cold_census, client,
                                                     tmp_path,
                                                     make_detection_h5):
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)
    client.app.state.archive_root = tmp_path

    page = client.get("/ui/sources?min_count=2")
    assert page.status_code == 200
    assert "first census running" in page.text.lower()
    assert "No census has finished yet" in page.text


def test_the_warm_up_answers_the_question_the_page_asks(cold_census, tmp_path,
                                                        make_detection_h5):
    """A warm-up keyed differently from the page is a warm-up for nothing.

    The short-circuit is fingerprinted on the tuning parameters, so warming
    with `max_days=7` would fill the per-file memo and still leave the first
    visitor doing the grouping. `sources.warm` and the route both take their
    defaults from the same constants for that reason.
    """
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    warmed = cold_census.warm(tmp_path)
    assert warmed["cost"]["opened"] > 0

    served = cold_census.census(tmp_path)
    assert served["cost"]["unchanged"] is True
    assert served["cost"]["opened"] == 0
    assert served["emitters"] == warmed["emitters"]


def test_the_archive_is_read_at_startup_not_by_the_first_visitor(
        cold_census, tmp_path, monkeypatch, make_detection_h5):
    """234 seconds, once, is the whole cost of this cache being in memory.

    It is paid on every container start, and left to the request path it is
    paid by a person looking at a blank tab -- which is indistinguishable from
    the page being broken, and was read as exactly that.
    """
    day = tmp_path / "2026-08-09"
    day.mkdir()
    make_detection_h5("chirp", cycles=6, into=day)

    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setattr(main, "WARM_CENSUS", True)

    with TestClient(main.app) as c:
        c.app.state.census_warm.join(timeout=60)
        assert not c.app.state.census_warm.is_alive(), "warm-up never finished"
        served = c.get("/ui/sources")

    assert served.status_code == 200
    assert cold_census._LAST["census"]["cost"]["opened"] > 0, "read nothing"
    assert "nothing re-opened" in served.text


def test_a_missing_archive_does_not_stop_the_api_starting(cold_census,
                                                          tmp_path):
    """The warm-up runs at boot, so anything it raises takes the server down.

    An unreadable archive is a page that cannot be rendered; it is not a
    reason for the health views and the command queue to be unreachable too.
    """
    got = cold_census.warm(tmp_path / "not-here")
    assert got["cost"]["files"] == 0 and got["emitters"] == []


def test_the_build_is_reported_so_a_deploy_can_be_checked(client,
                                                          monkeypatch):
    """`version` is hand-edited and has read 0.1.0 through every deploy.

    Without a stamp that moves on its own, "is the fix on the server?" can
    only be answered by looking for the fix's effects -- which on a slow page
    is two four-minute page loads.
    """
    got = client.get("/healthz").json()
    assert got["ok"] is True
    assert got["build"] == "source", "an unstamped checkout is not a build"

    monkeypatch.setattr(main, "BUILD_SHA", "3097398")
    monkeypatch.setattr(main, "BUILD_TIME", "2026-08-13T18:40:00Z")
    stamped = client.get("/healthz").json()
    assert stamped["build"] == "3097398"
    assert stamped["built_at"] == "2026-08-13T18:40:00Z"


def test_a_census_row_with_a_missing_column_is_still_json():
    """`NaN` is not JSON, and a census row is an ordinary place to find one.

    Python writes it as the bare token `NaN`, which `json.dumps` emits by
    default and no strict parser accepts. Found in the browser: a group whose
    detections carried no SNR field gave `"snr_median": NaN`, `JSON.parse`
    threw on it, and the identify button for that row did nothing -- because
    the whole row travels to the page as one JSON attribute. `/sources`
    was returning a document that says it is JSON and is not.
    """
    from muf.io_detect import Emitter
    from services.api import sources

    row = sources._as_row(Emitter(
        rate=100e3, fraction_s=0.5,
        fraction_sd_s=float("nan"),   # one slot: no scatter to compute
        count=9, observed_seconds=(235,), cycle_s=300.0,
        first_seen=1.0, last_seen=3601.0,
        snr_median=float("nan")))     # detections carried no SNR field

    assert row["snr_median"] is None
    assert row["fraction_sd_s"] is None
    assert json.loads(json.dumps(row, allow_nan=False))["rate"] == 100e3


# --------------------------------------------------------------------------
# Verified transmitters, and the schedule composed from them
# --------------------------------------------------------------------------
#
# The gap this closes: an emitter census is anonymous, and `calc_ionograms.py`
# is not. Its rank loop reads `st[s_idx]["id"]` and `st[s_idx]["transmit_name"]`
# with a bare subscript, and both end up in the product --
# `lfm_ionogram-{tx}-{rx}-{ch}-{id:03d}-{t0}.h5` and `ho["txname"]` -- which
# this pipeline reads back as `sounding.tx` and resolves against
# `muf/stations.py` for the geometry and the band ceiling. So the schedule
# cannot be built from the census: somebody has to say who these are.

def _identify(client, station="SIM", code="NIC", **kw):
    body = {"code": code,
            "timings": [{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0}]}
    body.update(kw)
    return client.post(f"/stations/{station}/transmitters", headers=CTL, json=body)


def test_an_identification_round_trips_with_its_evidence(client):
    r = _identify(client, name="Nicosia", note="operator judgement, 2026-08-05",
                  evidence={"rate": 100e3, "count": 855, "snr_median": 41.2})
    assert r.status_code == 200

    listed = client.get("/stations/SIM/transmitters").json()["transmitters"]
    assert [t["code"] for t in listed] == ["NIC"]
    assert listed[0]["evidence"]["count"] == 855
    assert listed[0]["note"].startswith("operator judgement")
    assert listed[0]["verified_at"]


def test_the_two_missing_keys_are_filled_in_from_the_record(client):
    """Not from the caller. They are the record's identity, in the ini's words.

    Letting a form set `transmit_name` independently of `code` is how a
    schedule comes to name a transmitter the database has never heard of.
    """
    entry = _identify(client, code="NIC").json()["transmitter"]["timings"][0]
    assert entry["transmit_name"] == "NIC"
    assert entry["id"] == 1
    assert set(entry) >= set(acq.REQUIRED_ENTRY_KEYS)


def test_a_dash_in_a_code_is_refused_because_of_the_file_name(client):
    """`lfm_ionogram-{tx}-{rx}-{ch}-{cid}-{t0}.h5`, parsed back by a regex on
    dashes (`muf/io_chirp.py:188`). A dash inside the transmitter name does not
    fail to parse -- it parses into the *next* field, so the tail of the name
    becomes the receiver and everything after it shifts by one."""
    r = _identify(client, code="yoshkar-ola")
    assert r.status_code == 400
    assert "no dash" in r.json()["detail"]

    for bad in ("", "  ", "a/b", "with space", "x" * 25):
        assert _identify(client, code=bad).status_code == 400, bad


def test_a_transmitter_needs_at_least_one_slot(client):
    assert _identify(client, timings=[]).status_code == 400
    assert _identify(client, timings=[{"chirp-rate": 100e3}]).status_code == 400
    assert _identify(client, timings=[{"chirp-rate": 100e3, "rep": 0.0,
                                       "chirpt": 1.0}]).status_code == 400


def test_a_code_the_registry_cannot_resolve_is_saved_with_a_warning(client):
    """Not refused: a newly heard emitter has to be nameable before anyone
    knows where it is, and refusing here would make identifying one impossible.

    But `io_chirp._coords_for` answers NaN for an unknown name by design, so
    the cost is silent and hours away -- a full-span range gate, a NULL
    path_km, no measured band ceiling, and finally IRI reporting a foF2 at
    `nanS nanW` on a sounding page, with nothing tying it back to the name that
    was typed. That is exactly how NIC1 and NIC3 were lost on 2026-08-16.
    """
    body = _identify(client, code="TGO7").json()
    assert body["ok"] is True
    assert body["transmitter"]["code"] == "TGO7"
    assert "nanS nanW" in body["warning"]
    assert "muf/stations.py" in body["warning"]


def test_a_code_the_registry_knows_warns_about_nothing(client):
    """Including the per-slot aliases, which is the shape the warning exists to
    make people reach for: NIC1 is another slot of NIC, not another site."""
    for code in ("NIC", "NIC1", "cyprus1"):
        assert "warning" not in _identify(client, code=code).json(), code


def test_re_identifying_replaces_and_keeps_the_number(client):
    """Narrowing the slots after another day of census is the normal case.

    The `sounder_id` must survive it: it is `%03d` in the file name of every
    product already on disk.
    """
    first = _identify(client).json()["transmitter"]
    second = _identify(client, timings=[
        {"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0},
        {"chirp-rate": 100e3, "rep": 300.0, "chirpt": 265.0}]).json()["transmitter"]

    assert second["sounder_id"] == first["sounder_id"]
    assert len(second["timings"]) == 2
    assert len(client.get("/stations/SIM/transmitters").json()["transmitters"]) == 1


def test_ids_are_not_handed_out_twice_even_after_a_forget(client):
    """Products on disk carry the id. Reusing it makes two sites one number."""
    _identify(client, code="NIC")
    _identify(client, code="SGO")
    assert client.delete("/stations/SIM/transmitters/NIC", headers=CTL).status_code == 200

    again = _identify(client, code="TGO").json()["transmitter"]
    assert again["sounder_id"] == 3

    missing = client.delete("/stations/SIM/transmitters/NIC", headers=CTL)
    assert missing.status_code == 404


def test_an_identification_is_per_receiver(client):
    """A slot second is a reception second: transmit time plus travel time plus
    that receiver's epoch offset. One transmitter, two receivers, two numbers --
    the same reason the band ceiling is keyed by receiver."""
    _identify(client, station="DOB", code="NIC")
    _identify(client, station="KHO", code="NIC",
              timings=[{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 237.5}])

    dob = client.get("/stations/DOB/transmitters").json()["transmitters"]
    kho = client.get("/stations/KHO/transmitters").json()["transmitters"]
    assert dob[0]["timings"][0]["chirpt"] == 235.0
    assert kho[0]["timings"][0]["chirpt"] == 237.5


# --------------------------------------------------------------------------
# Forgetting a receiver
# --------------------------------------------------------------------------

def _retired(client, api_db, station="DOB"):
    """A station that reported once and never will again, e.g. renamed."""
    client.post("/stations/health", json=report(station=station), headers=CTL)
    api_db.execute("UPDATE health_report SET received_at = '2026-08-01T00:00:00Z'"
                   " WHERE station = ?", (station,))
    api_db.commit()


def test_a_renamed_receiver_can_be_forgotten(client, api_db):
    """Its panel is STALE for good, and a red that is always red is not read."""
    _retired(client, api_db)
    assert "DOB" in client.get("/ui").text

    r = client.delete("/stations/DOB", headers=CTL)
    assert r.status_code == 200 and r.json()["health_reports"] == 1

    assert client.get("/stations").json()["stations"] == []
    assert "DOB" not in client.get("/ui").text
    # The cascade is only as good as the PRAGMA, so it is asserted rather than
    # assumed: orphaned metrics are invisible and would never be noticed.
    assert api_db.execute("SELECT COUNT(*) FROM health_metric").fetchone()[0] == 0


def test_forgetting_a_receiver_keeps_what_is_provenance(client, api_db):
    """The identifications outlive the panel.

    A transmitter row is a human judgement with its evidence attached, and
    products already on disk carry the name it assigned. Deleting that to tidy
    a console would be destroying a record to fix a colour.
    """
    _identify(client, station="DOB", code="NIC")
    _retired(client, api_db)

    body = client.delete("/stations/DOB", headers=CTL).json()

    assert body["kept"]["transmitters"] == 1
    assert "1 identification(s)" in body["note"]
    listed = client.get("/stations/DOB/transmitters").json()["transmitters"]
    assert [t["code"] for t in listed] == ["NIC"]


def test_a_receiver_that_is_still_pushing_is_not_forgotten(client, api_db):
    """Clearing a live station loses when reports stopped and gains nothing:
    the panel returns on the next push. The name is the agent's to change."""
    client.post("/stations/health", json=report(station="SIM"), headers=CTL)

    refused = client.delete("/stations/SIM", headers=CTL)
    assert refused.status_code == 409
    assert "still pushing under that name" in refused.json()["detail"]

    forced = client.delete("/stations/SIM?force=true", headers=CTL)
    assert forced.status_code == 200


def test_the_forget_button_is_offered_only_on_a_stale_panel(client, api_db):
    """The server refuses a live receiver at the same threshold, so the button
    appears exactly where pressing it would be accepted."""
    client.post("/stations/health", json=report(station="SIM"), headers=CTL)
    assert "forget this receiver" not in client.get("/ui").text

    _retired(client, api_db, station="DOB")
    assert client.get("/ui").text.lower().count("forget this receiver") == 1


def test_forgetting_a_receiver_is_control_scope(client, api_db):
    _retired(client, api_db)
    assert client.delete("/stations/DOB").status_code == 401
    assert client.delete("/stations/NOSUCH", headers=CTL).status_code == 404
    assert "DOB" in client.get("/ui").text, "a refused delete must delete nothing"


def test_a_schedule_names_transmitters_rather_than_carrying_numbers(client):
    _identify(client, code="NIC")
    r = client.post("/stations/SIM/schedule", headers=CTL, json={"codes": ["TGO"]})
    assert r.status_code == 400
    assert "no verified transmitter named TGO" in r.json()["detail"]
    assert "NIC" in r.json()["detail"], "say what there is, not just what there isn't"

    assert client.post("/stations/SIM/schedule", headers=CTL,
                       json={"codes": []}).status_code == 400


def test_the_schedule_is_one_rank_group_per_transmitter(client):
    """And it says so, because the launcher's `-np` has to match it."""
    _identify(client, code="NIC", timings=[
        {"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0},
        {"chirp-rate": 100e3, "rep": 300.0, "chirpt": 265.0}])
    _identify(client, code="SGO",
              timings=[{"chirp-rate": 500.0084e3, "rep": 60.0, "chirpt": 54.0}])

    body = client.post("/stations/SIM/schedule", headers=CTL,
                       json={"codes": ["NIC", "SGO"]}).json()
    assert body["ranks"] == 2 and body["entries"] == 3
    assert "-np 2" in body["note"]
    assert [len(g) for g in body["sounder_timings"]] == [2, 1]
    for group in body["sounder_timings"]:
        for entry in group:
            assert set(entry) >= set(acq.REQUIRED_ENTRY_KEYS), entry


def test_the_schedule_is_queued_as_a_vetted_set_config(client):
    _identify(client, code="NIC")
    client.post("/stations/SIM/schedule", headers=CTL, json={"codes": ["NIC"]})

    pulled = client.get("/stations/SIM/commands", headers=CTL).json()["commands"]
    assert pulled[0]["name"] == "set_config"
    changes = pulled[0]["params"]["changes"]
    assert changes["mode"] == "scheduled"
    assert json.loads(changes["sounder_timings"])[0][0]["transmit_name"] == "NIC"


def test_writing_a_transmitter_needs_the_control_scope(client):
    """It stops no radio, but it names the files the station will write."""
    assert client.post("/stations/SIM/transmitters", json={
        "code": "NIC",
        "timings": [{"chirp-rate": 1e5, "rep": 300.0, "chirpt": 1.0}]
    }).status_code == 401
    assert client.post("/stations/SIM/schedule",
                       json={"codes": ["NIC"]}).status_code == 401
    assert client.delete("/stations/SIM/transmitters/NIC").status_code == 401


# --------------------------------------------------------------------------
# Configuration epochs, and the live view built on them
# --------------------------------------------------------------------------

def _journal(mode="false", timings="[[]]"):
    return {"command": "apply_config", "ok": True,
            "detail": "mode: 'true' -> 'false'",
            "journal": {"station": "SIM", "requires_restart": True,
                        "changes": {"mode": {"from": "true", "to": mode},
                                    "sounder_timings": {"from": "[]",
                                                        "to": timings}}}}


def test_an_epoch_opens_when_the_station_acknowledges_not_when_it_is_queued(client):
    """A queued command has changed nothing.

    Opening the epoch at enqueue would attribute every sounding recorded while
    the station was unreachable to a configuration it was not running.
    """
    _identify(client, code="NIC")
    command_id = client.post("/stations/SIM/schedule", headers=CTL,
                             json={"codes": ["NIC"]}).json()["id"]
    assert client.get("/stations/SIM/schedule").json()["mode"] is None

    client.get("/stations/SIM/commands", headers=CTL)          # delivered
    assert client.get("/stations/SIM/schedule").json()["mode"] is None

    timings = json.dumps([[{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
                            "id": 1, "transmit_name": "NIC"}]])
    client.post(f"/stations/SIM/commands/{command_id}/ack", headers=CTL,
                json={"results": [_journal(timings=timings)]})

    live = client.get("/stations/SIM/schedule").json()
    assert live["mode"] == "scheduled"
    assert live["slots"][0]["transmitter"] == "NIC"
    assert live["epoch"]["changed_by"] == "web"


def test_the_epoch_opens_even_when_the_restart_failed(client):
    """The file on the station has already changed.

    `apply_and_restart` is stop, write, start. If the start fails the command
    is not ok -- but anything recorded from now on was recorded under the new
    configuration, and attributing it to the old one is the error that lasts.
    """
    command_id = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"mode": "search"}}}).json()["id"]
    client.post(f"/stations/SIM/commands/{command_id}/ack", headers=CTL, json={
        "results": [{"command": "stop chirp.target", "ok": True},
                    _journal(mode="true"),
                    {"command": "start chirp.target", "ok": False,
                     "detail": "Job for chirp-rx.service failed"}]})

    assert client.get("/stations/SIM/schedule").json()["mode"] == "search"


def test_a_failed_write_opens_no_epoch(client):
    command_id = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"mode": "search"}}}).json()["id"]
    client.post(f"/stations/SIM/commands/{command_id}/ack", headers=CTL, json={
        "results": [{"command": "apply_config", "ok": False,
                     "detail": "sounder_timings is not valid JSON"}]})

    assert client.get("/stations/SIM/schedule").json()["mode"] is None


def test_only_one_epoch_is_ever_open(client, tmp_path):
    """Two open epochs is not an untidy table, it is a station with two current
    configurations and a guess behind every attribution after it."""
    for _ in range(3):
        cid = client.post("/stations/SIM/commands", headers=CTL, json={
            "name": "set_config", "params": {"changes": {"mode": "search"}}}).json()["id"]
        client.post(f"/stations/SIM/commands/{cid}/ack", headers=CTL,
                    json={"results": [_journal(mode="true")]})

    with db.session(tmp_path / "api.sqlite3") as conn:
        rows = db.rows(conn, "SELECT valid_from, valid_to FROM config_epoch"
                             " WHERE station = 'SIM' ORDER BY id")
        assert len(rows) == 3
        assert sum(r["valid_to"] is None for r in rows) == 1
        assert rows[-1]["valid_to"] is None


def test_a_station_with_no_epoch_says_so_rather_than_reporting_empty(client):
    live = client.get("/stations/SIM/schedule").json()
    assert live["slots"] == []
    assert "no schedule recorded" in live["unknown"]


# --------------------------------------------------------------------------
# Is it acquiring, which the schedule cannot say
# --------------------------------------------------------------------------
#
# The slot arithmetic is the ini read against a clock: it stays true with the
# recorder dead. These tests pin the separate question -- whether anything is
# actually being recorded -- and the order in which the evidence is weighed.

def _unit(name, ok):
    return {"name": f"unit:{name}", "value": "active" if ok else "failed",
            "ok": ok, "detail": ""}


def _age(seconds, ok):
    return {"name": "newest_product_age_s", "value": seconds, "ok": ok,
            "detail": "threshold 900s"}


def _state(metrics, age_s=30.0):
    return acq.running_state(metrics, age_s=age_s,
                             stale_after=acq.STALE_AFTER_S)["state"]


def test_products_arriving_is_what_acquiring_means():
    """DOB has no unit states at all -- `dombas.sh` supervises it, not systemd.

    An indicator built on unit states would read "unknown" forever on the one
    station being watched, so the product age has to be able to answer alone.
    """
    assert _state([_age(45.0, True)]) == "running"


def test_a_dead_recorder_outranks_a_product_that_is_still_fresh():
    """A product age inside its threshold can be fifteen minutes old.

    The unit state is a fact about now, so it wins: the last sweep landing
    recently is not evidence that the next one will.
    """
    assert _state([_unit("chirp-rx.service", False), _age(45.0, True)]) == "stopped"


def test_green_units_with_nothing_arriving_are_silent_not_running():
    """This is what the real outage looked like.

    Every unit active for two days with `/dev/shm` at 100%: the ringbuffer was
    never trimmed and the recording was full of holes, while systemd was
    perfectly happy. "Silent" is its own state so that it cannot be read as a
    milder shade of either green or red.
    """
    assert _state([_unit("chirp-rx.service", True),
                   _unit("chirp-ionograms.service", True),
                   _age(9000.0, False)]) == "silent"


def test_a_failed_upload_is_not_a_stopped_station():
    """`chirp-sync` pushes dashboard images; the station sounds without it.

    Reporting every listed unit would give eleven false reds, which is how an
    indicator gets ignored. It still shows FAIL in the metrics table.
    """
    assert _state([_unit("chirp-sync.service", False),
                   _unit("chirp-archive-sync.service", False),
                   _age(45.0, True)]) == "running"


def test_a_stale_report_is_unknown_not_stopped():
    """Absence of evidence. Silence is the alert, but it is not a diagnosis --
    the two must not share a colour."""
    assert _state([_age(45.0, True)], age_s=4000.0) == "unknown"
    assert _state([_age(45.0, True)], age_s=None) == "unknown"


def test_a_report_that_measured_nothing_says_so():
    assert _state([{"name": "disk_free_fraction", "value": 0.42, "ok": True,
                    "detail": ""}]) == "unknown"


def test_the_console_will_not_call_a_slot_a_sounding_when_nothing_arrives(client):
    """The default report is 9000 s since the last product: silent.

    The schedule still says a chirp is due this second, and the page has to
    show both without the arithmetic being mistaken for an observation.
    """
    client.post("/stations/health", headers=CTL, json=report(station="DOB"))
    _identify(client, station="DOB", code="NIC")
    command_id = client.post("/stations/DOB/schedule", headers=CTL,
                             json={"codes": ["NIC"]}).json()["id"]
    timings = json.dumps([[{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
                            "id": 1, "transmit_name": "NIC"}]])
    client.post(f"/stations/DOB/commands/{command_id}/ack", headers=CTL,
                json={"results": [_journal(timings=timings)]})

    live = client.get("/stations/DOB/schedule").json()
    assert live["running"]["state"] == "silent"

    page = client.get("/ui").text
    assert "NO PRODUCTS" in page
    assert "SOUNDING" not in page


def test_the_console_shows_what_is_sounding_and_what_arrived(client, tmp_path,
                                                             monkeypatch):
    """The panel exists because unit states cannot answer it: every process can
    be active while the schedule points at a transmitter that stopped."""
    from services.api import ingest

    client.post("/stations/health", headers=CTL, json=report(station="DOB"))
    _identify(client, station="DOB", code="NIC")
    command_id = client.post("/stations/DOB/schedule", headers=CTL,
                             json={"codes": ["NIC"]}).json()["id"]
    timings = json.dumps([[{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
                            "id": 1, "transmit_name": "NIC"}]])
    client.post(f"/stations/DOB/commands/{command_id}/ack", headers=CTL,
                json={"results": [_journal(timings=timings)]})

    # One ingested product, which is where the sweep length comes from: the
    # band is measured off what the receiver actually produced, not configured.
    path = tmp_path / "lfm_ionogram-NIC-DOB-ch000-001-1785888235.00.h5"
    path.write_bytes(b"x")
    with db.session(tmp_path / "api.sqlite3") as conn:
        ingest.ingest_row(conn, {
            "file": path.name, "datetime": "2026-08-06 00:03:55",
            "tx": "NIC", "rx": "DOB", "freq_start": 0.0, "freq_stop": 24.825,
            "muf_algo": 18.4}, path, tmp_path, ("algo",))
        conn.commit()          # `ingest_row` leaves the commit to its batch

    live = client.get("/stations/DOB/schedule").json()
    assert live["span_mhz"] == pytest.approx(24.825)
    assert live["slots"][0]["sweep_s"] == pytest.approx(248.25)
    assert live["arrivals"][0]["tx"] == "NIC"
    assert live["arrivals"][0]["age_s"] > 0

    page = client.get("/ui")
    assert page.status_code == 200
    assert "acquisition" in page.text
    assert "NIC" in page.text


def test_the_console_control_does_not_hinge_on_a_native_dialog(client):
    """A false from ``window.confirm`` is indistinguishable from a dead button.

    The stop button used to be gated behind one, and a dialog the browser
    suppressed -- or that this page's own 15 s refresh cancelled mid-read --
    made ``send`` return with no request and no message. The report that found
    it was "I press STOP but nothing changes", which is precisely what that
    looks like from the chair. The question now lives in the page, answered by
    a second press, and every outcome writes a line the operator can read.
    """
    client.post("/stations/health", headers=CTL, json=report(station="DOB"))
    page = client.get("/ui").text

    assert 'id="say-DOB"' in page, "each station needs somewhere to be answered"
    assert "confirm(" not in page
    # The refresh must stand down while a stop is armed, or it eats the
    # question the way it used to cancel the dialog.
    assert "if (!armed && !planning) location.reload()" in page
    # And queued must not read as done: nothing happens on the station until
    # its agent pulls the row, which may be never.
    assert "pending until the station" in page


def test_the_console_carries_the_chooser_the_start_button_needs(client):
    """Choosing who to sound and starting the sounding are one decision.

    They were two pages: the transmitters lived on /ui/sources, behind a census
    that reads the archive, and the start button lived here. So the operator
    picked a schedule on one page, then went looking for the button on another
    -- and /ui/sources is the slow page, which is a poor place to keep a
    control. The list comes from the database, not from the census, so the
    console carries the chooser without inheriting the archive read.
    """
    client.post("/stations/health", headers=CTL, json=report(station="DOB"))
    _identify(client, station="DOB", code="NIC", name="Nicosia")
    _identify(client, station="DOB", code="SGO", name="Sodankyla")
    command_id = client.post("/stations/DOB/schedule", headers=CTL,
                             json={"codes": ["NIC"]}).json()["id"]

    def box(page, code):
        """The one checkbox for `code`, as the template lays it out."""
        marker = f'class="use" data-station="DOB"\n                   value="{code}"'
        assert marker in page, f"{code} is verified here, so it must be tickable"
        return page.split(marker, 1)[1].split("onchange", 1)[0]

    page = client.get("/ui").text
    assert "sounding plan" in page
    assert 'id="mode-DOB"' in page and 'id="planSay-DOB"' in page

    # Queued is not configured. The schedule above has not been acknowledged,
    # so the station is not running it and nothing may be pre-ticked yet.
    assert "checked" not in box(page, "NIC"), "a pending command is not a state"
    assert "checked" not in box(page, "SGO")
    assert "&mdash; not recorded &mdash;" in page.lower(), \
        "no mode was ever acknowledged, so none may be pre-selected"

    timings = json.dumps([[{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
                            "id": 1, "transmit_name": "NIC"}]])
    client.post(f"/stations/DOB/commands/{command_id}/ack", headers=CTL,
                json={"results": [_journal(mode="scheduled", timings=timings)]})

    acked = client.get("/ui").text
    assert "checked" in box(acked, "NIC"), "the acknowledged schedule names NIC"
    assert "checked" not in box(acked, "SGO"), "SGO is verified, not scheduled"
    assert "&mdash; not recorded &mdash;" not in acked, "the mode is known now"


def test_the_console_will_not_apply_a_mode_nobody_recorded(client):
    """The select's fallback used to be the first option, `search`.

    That put a mode this server never observed one click from being applied to
    a live receiver -- and search mode records whatever sweeps past, so the
    mistake is not visible until the products stop matching the schedule. The
    unrecorded case selects a sentinel with no value instead, and both the
    preview and `applyPlan` refuse it.
    """
    client.post("/stations/health", headers=CTL, json=report(station="DOB"))
    page = client.get("/ui").text

    assert '<option value="" selected>' in page
    assert "if (!mode)" in page, "the empty value has to be refused, not sent"
    assert "no record of the mode" in page


# --------------------------------------------------------------------------
# Reachability: can this host still refresh the solar indices IRI runs on?
# --------------------------------------------------------------------------

def _reachable(*hosts_ok):
    """A probe stub. ``_reachable(True, False)`` reaches the first host only."""
    answers = list(hosts_ok)

    def probe(url, *, timeout=0.0):
        ok = answers.pop(0)
        return (True, 12.0, "HTTP 200") if ok else (False, 0.0, "refused")

    return probe


def test_reachability_is_all_or_some_or_none(tmp_path):
    n = len(net.hosts())
    assert n >= 2, "the fold between online and degraded needs two hosts"

    assert net.check(probe_fn=_reachable(*[True] * n),
                     cache_dir=tmp_path).state == "online"
    assert net.check(probe_fn=_reachable(*[False] * n),
                     cache_dir=tmp_path).state == "offline"

    partial = net.check(probe_fn=_reachable(True, *[False] * (n - 1)),
                        cache_dir=tmp_path)
    assert partial.state == "degraded"
    # A partial outage has to name what is down: "degraded" alone does not
    # tell an operator whether the missing host is the one carrying F10.7.
    assert any(p.host in partial.detail for p in partial.probes if not p.ok)


def test_a_probe_never_raises_whatever_the_host_does():
    """The connectivity light must not be able to take the page down."""
    ok, ms, detail = net.probe("http://127.0.0.1:1/nothing", timeout=0.5)

    assert ok is False
    assert ms is not None
    assert detail                      # it says *why*, not just "failed"


def test_an_http_error_still_means_reachable(monkeypatch):
    """404 is the server answering. The route is open; the file moved.

    Reporting that as offline would send an operator to the network team over
    a renamed file, which is the wrong half of the system.
    """
    import urllib.error

    def refuse(*a, **k):                        # noqa: ANN002, ANN003
        raise urllib.error.HTTPError("https://example.invalid/x", 404,
                                     "Not Found", {}, None)

    monkeypatch.setattr(net.urllib.request, "urlopen", refuse)
    ok, _, detail = net.probe("https://example.invalid/x")

    assert ok is True
    assert "404" in detail and "reachable" in detail


def test_cache_age_is_reported_beside_reachability(tmp_path):
    """Unreachable-with-a-cache and reachable-without-one are opposite faults."""
    key = indices.SOURCES[0]
    # A plausible document, not a stub: a file under `MIN_USEFUL_BYTES` is
    # reported as absent on purpose, because that is what an empty one written
    # by a failed download actually is. See `tests/test_indices_cache.py`.
    (tmp_path / key.filename).write_text("cached\n" * 64)

    got = net.check(probe_fn=_reachable(*[False] * len(net.hosts())),
                    cache_dir=tmp_path)

    assert got.state == "offline"
    assert got.cache[key.key] is not None and got.cache[key.key] < 60
    assert got.cache[indices.SOURCES[-1].key] is None      # never fetched


def test_an_emptied_cache_is_reported_as_absent_not_as_fresh(tmp_path):
    """The indicator said "47 h old, all six sources" while the file behind it
    was zero bytes -- the write that destroyed it had refreshed the mtime the
    indicator reads. Reachability was fine, the cache looked fine, and the
    model had silently lost its solar driver."""
    for source in indices.SOURCES:
        (tmp_path / source.filename).write_text("")

    got = net.check(probe_fn=_reachable(*[False] * len(net.hosts())),
                    cache_dir=tmp_path)

    assert all(age is None for age in got.cache.values()), got.cache


def test_a_reading_nobody_refreshed_decays_to_unknown(monkeypatch):
    """A dead checker must not leave a permanently green light.

    The thread is a daemon and daemons die. Showing its last reading for ever
    would turn "nothing is checking" into "everything is fine", which is the
    exact failure this module was written to make visible.
    """
    net.reset()
    try:
        net.refresh(probe_fn=_reachable(*[True] * len(net.hosts())))
        assert net.current().state == "online"

        monkeypatch.setattr(net, "STALE_AFTER_S", -1.0)
        assert net.current().state == "unknown"
        assert "no reading" in net.current().detail
    finally:
        net.reset()


def test_the_net_route_never_probes(client, monkeypatch):
    """`/net` reports the last pass. It must not issue outbound requests.

    Otherwise anyone who can reach this port can make the server call three
    third parties, and every caller pays the timeout.
    """
    def explode(*a, **k):                       # noqa: ANN002, ANN003
        raise AssertionError("the route probed the network")

    monkeypatch.setattr(net, "probe", explode)
    net.reset()

    body = client.get("/net").json()
    assert body["state"] == "unknown"           # nothing has run in this test
    assert len(body["hosts"]) == 0
    assert body["age_s"] is None


def test_the_console_says_whether_the_indices_can_be_refreshed(client):
    """The point of the panel: the IRI numbers have an upstream, and it shows."""
    net.refresh(probe_fn=_reachable(*[True] * len(net.hosts())))
    try:
        page = client.get("/ui").text
        assert "INTERNET OK" in page
        for host, _, _ in net.hosts():
            assert host in page

        net.refresh(probe_fn=_reachable(*[False] * len(net.hosts())))
        page = client.get("/ui").text
        assert "NO INTERNET" in page
    finally:
        net.reset()

    # With no reading at all the pill is grey, not red: "we have not asked" is
    # not "the answer is no", the same tri-state the metrics table uses.
    assert "INTERNET?" in client.get("/ui").text


# --------------------------------------------------------------------------
# SAO: the scaling behind the download, the panel and the interactive plot
# --------------------------------------------------------------------------

@pytest.fixture
def scaled(client, tmp_path, monkeypatch, make_chirp_h5):
    """One synthetic product, ingested, with a sounding id. Returns (id, path).

    IRI is off. It is the one part of a scaling that can reach the internet --
    PyIRI needs a solar driver, and an unwarmed index cache means a fetch --
    and a unit suite must not depend on somebody else's uptime, exactly as it
    must not for the reachability checker.
    """
    import numpy as np

    from services.api import ingest, sao

    monkeypatch.setattr(sao, "MODEL", False)
    sao.clear()

    # A ridge at one range across the band: the simplest thing the detectors
    # can scale, and enough to give the trace list something in it.
    power = np.full((48, 96), 100.0)
    power[:, 40] = 4000.0
    path = make_chirp_h5(power)

    conn = client.app.state.db
    monkeypatch.setattr(client.app.state, "archive_root", tmp_path)
    row = {"file": path.name, "datetime": "2026-02-04T00:00:10",
           "tx": "synthtx", "rx": "synthrx"}
    sounding_id = ingest.ingest_row(conn, row, path, tmp_path, ("algo",))
    conn.commit()
    yield sounding_id, path
    sao.clear()


def test_the_raster_extent_runs_half_a_cell_past_the_samples(scaled):
    """`pcolormesh(shading="nearest")` centres a cell on each sample.

    So the image covers half a cell more than the axis at each end. Half a bin
    out and every overlaid circle sits beside its echo instead of on it --
    invisible at full extent, obvious the moment anyone zooms in, which is the
    whole reason the plot is interactive.
    """
    from services.api import sao

    _, path = scaled
    ion = sao.load_ion(path, gate=None)
    f_lo, f_hi, r_lo, r_hi = sao.raster_extent(ion)

    df = float(ion.freq[1] - ion.freq[0])
    assert f_lo == pytest.approx(float(ion.freq[0]) - df / 2)
    assert f_hi == pytest.approx(float(ion.freq[-1]) + df / 2)
    assert r_lo < float(min(ion.vrange)) and r_hi > float(max(ion.vrange))


def test_full_is_a_word_the_gate_understands(scaled):
    """The page's own toggle sends `gate=full`, meaning "do not gate".

    Read as a range pair it raises ValueError -- a 500 on a link the interface
    offers, which is the worst kind because nothing else has to go wrong.
    """
    from services.api import sao

    _, path = scaled
    full = sao.load_ion(path, gate="full")
    plain = sao.load_ion(path, gate=None)
    assert full.vrange.size == plain.vrange.size


def test_a_scaling_is_reused_until_the_file_changes(scaled):
    """Detection products are write-once, so path plus mtime identifies one."""
    from services.api import sao

    _, path = scaled
    first = sao.build(path, gate=None)
    assert sao.build(path, gate=None) is first

    path.touch()
    assert sao.build(path, gate=None) is not first


def test_every_trace_gets_its_own_legend_entry(scaled):
    """A legend of five entries reading "unlabelled" in one colour is not one.

    `Branch` comes back empty for most traces on an oblique circuit -- the hop
    identification labels some and not others -- so the fallback has to
    distinguish them, and the frequency span is what does it.
    """
    from services.api import sao

    _, path = scaled
    frame = sao.plot_data(sao.build(path, gate=None), "algo")
    names = [t["name"] for t in frame["traces"]]
    assert len(names) == len(set(names)), names
    assert all("MHz" in name for name in names)


def test_the_plot_frame_carries_points_not_the_raster(scaled):
    """486 x 3999 cells is about 11 MB of JSON against 164 KB as a PNG.

    Only the scaled points cross the wire; the image is placed behind them in
    data coordinates. If a heatmap ever creeps back in, this catches it.
    """
    from services.api import sao

    _, path = scaled
    frame = sao.plot_data(sao.build(path, gate=None), "algo")
    assert set(frame) == {"extent", "traces", "marks", "relative", "scale"}
    for trace in frame["traces"]:
        assert len(trace["freq"]) == len(trace["vrange"])
        assert len(trace["freq"]) < 5000
    # The colour key is 32 stops, not 1.9 million cells: it describes the
    # raster rather than reproducing it.
    assert len(frame["scale"]["stops"]) == 32


# --------------------------------------------------------------------------
# The raster's colour key
# --------------------------------------------------------------------------

def test_the_colour_bar_is_the_map_the_raster_was_drawn_with():
    """A bar built from a different colormap than the image it explains is
    worse than no bar, because it still reads as authoritative."""
    from muf import render

    from services.api import sao

    axis = sao.colour_axis()

    assert axis["vmin"] == render.DEFAULT_VMIN_DB
    assert axis["vmax"] == render.DEFAULT_VMAX_DB
    assert axis["stops"] == render.colour_scale(render.DEFAULT_CMAP)


def test_the_colour_bar_follows_a_change_of_colormap(monkeypatch):
    """Nothing transcribes the scale, so changing the map moves the bar too."""
    from muf import render

    from services.api import sao

    monkeypatch.setattr(render, "DEFAULT_CMAP", "viridis")

    assert sao.colour_axis()["stops"] == render.colour_scale("viridis")


def test_the_scale_spans_the_map_from_end_to_end():
    from muf import render

    stops = render.colour_scale()

    assert stops[0][0] == 0.0 and stops[-1][0] == 1.0
    assert all(a[0] < b[0] for a, b in zip(stops, stops[1:]))
    assert all(stop[1].startswith("rgb(") for stop in stops)


def test_a_sounding_with_no_record_still_gets_its_colour_key(scaled):
    """The raster is drawn whether or not an estimator scaled it, and a field
    in false colour with no key is what this exists to fix."""
    from services.api import sao

    _, path = scaled
    frame = sao.plot_data(sao.build(path, gate=None), "no-such-method")

    assert frame["traces"] == []
    assert frame["scale"]["label"] == "SNR (dB)"


def test_the_page_draws_without_a_bar_rather_than_inventing_one(monkeypatch):
    """No matplotlib, no honest scale. Guessing one would be a confident
    wrong answer about what the picture means."""
    from muf import render

    from services.api import sao

    def refuse(*a, **k):                        # noqa: ANN002, ANN003
        raise ModuleNotFoundError("no matplotlib")

    monkeypatch.setattr(render, "colour_scale", refuse)

    assert sao.colour_axis() is None


def test_the_sao_download_holds_one_record_per_estimator(client, scaled):
    """The spec keeps independent scalings apart (sec. 1.3.4).

    Three estimators disagreeing on one trace is the closest thing this
    pipeline has to an error bar; merging them into one record throws it away.
    """
    import xml.etree.ElementTree as ET

    sounding_id, _ = scaled
    got = client.get(f"/soundings/{sounding_id}/sao.xml?gate=full")
    assert got.status_code == 200
    assert got.headers["content-type"].startswith("application/xml")

    root = ET.fromstring(got.text)
    assert root.tag == "SAORecordList"
    methods = [r.findtext("SystemInfo/AutoScaler", "") for r in root]
    assert len(methods) == 3
    for method in ("algo", "kmeans", "contour"):
        assert any(f"({method})" in text for text in methods)


def test_a_sounding_whose_file_is_gone_says_so(client, scaled):
    """410, naming the path and ARCHIVE_ROOT. A 500 here reads as a bug in the
    server when it is a mismatch between the database and the disk."""
    sounding_id, path = scaled
    path.unlink()

    for url in (f"/soundings/{sounding_id}/sao.xml",
                f"/ionogram/{sounding_id}.png"):
        got = client.get(url)
        assert got.status_code == 410, url
        assert "ARCHIVE_ROOT" in got.json()["detail"]

    assert client.get("/soundings/99999/sao.xml").status_code == 404


def test_the_bare_raster_is_the_backing_image_not_a_styling_option(client, scaled):
    """`bare=true` must drop the axes, not merely hide them.

    The interactive plot places this in data coordinates and draws its own
    axes over it. Any margin matplotlib leaves is a shift between the picture
    and the numbers.
    """
    import io as io_module

    from PIL import Image

    sounding_id, _ = scaled
    bare = client.get(f"/ionogram/{sounding_id}.png?bare=true&dpi=60")
    full = client.get(f"/ionogram/{sounding_id}.png?dpi=60&muf=false")
    assert bare.status_code == full.status_code == 200
    assert bare.headers["content-type"] == "image/png"

    def corners(body):
        image = Image.open(io_module.BytesIO(body)).convert("RGB")
        w, h = image.size
        return [image.getpixel(xy) for xy in
                ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]

    # Not the pixel dimensions: `figsize` is fixed, so both come back 960
    # wide and only the *content* differs. Every corner of the bare one is
    # raster; the axed one has the figure's white paper under its title and
    # its colourbar.
    assert all(pixel != (255, 255, 255) for pixel in corners(bare.content))
    assert any(pixel == (255, 255, 255) for pixel in corners(full.content))


def test_the_sounding_page_draws_the_scaling_itself(client, scaled):
    """The page ships the points and the library, and asks for the raster.

    Not a CDN: the station this serves has been off the internet for a week at
    a time, and a plot that cannot draw a local file without a third party is
    one that fails exactly when someone needs it.
    """
    sounding_id, _ = scaled
    page = client.get(f"/ui/sounding/{sounding_id}?gate=full").text

    assert '<script src="/static/plotly.min.js">' in page
    assert 'id="sao-frame"' in page
    assert f"/ionogram/{sounding_id}.png?bare=true" in page
    assert "cdn" not in page.lower()

    body = json.loads(page.split('type="application/json">')[1].split("</script>")[0])
    assert body["extent"]["f_hi"] > body["extent"]["f_lo"]

    assert client.get("/static/plotly.min.js").status_code == 200


def test_a_scaling_that_fails_does_not_take_the_page_with_it(client, scaled,
                                                             monkeypatch):
    """The row, the neighbours and the stored extractions are still worth
    reading. A page that 500s because one panel could not be drawn hides all
    of them, and hides the reason too."""
    from services.api import sao

    def refuse(*a, **k):                        # noqa: ANN002, ANN003
        raise OSError("truncated product")

    monkeypatch.setattr(sao, "build", refuse)
    sounding_id, _ = scaled
    got = client.get(f"/ui/sounding/{sounding_id}")

    assert got.status_code == 200
    assert "truncated product" in got.text
    assert "extractions" in got.text


# --------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------

@pytest.fixture
def many_soundings(api_db, tmp_path):
    """Enough rows for a listing to be worth compressing."""
    for minute in range(60):
        _mk(api_db, tmp_path, f"bulk_{minute}.lfs",
            f"2026-02-04 00:{minute:02d}:00", **GEOMETRY)
    api_db.commit()
    return api_db


def test_a_large_page_is_compressed(client, many_soundings):
    """The station's link is slow and its pages are tables of numbers."""
    got = client.get("/soundings?limit=500",
                     headers={"Accept-Encoding": "gzip"})

    assert got.status_code == 200
    assert got.headers.get("content-encoding") == "gzip"


def test_a_client_that_cannot_unpack_gzip_still_gets_its_answer(
        client, many_soundings):
    got = client.get("/soundings?limit=500", headers={"Accept-Encoding": ""})

    assert got.status_code == 200
    assert "content-encoding" not in got.headers
    assert got.json()["count"] == 60


def test_a_short_answer_is_sent_as_it_is(client):
    """Compressing a health check makes the response bigger, not smaller."""
    got = client.get("/healthz", headers={"Accept-Encoding": "gzip"})

    assert got.status_code == 200
    assert "content-encoding" not in got.headers


# --- set_band over the wire --------------------------------------------------

def test_the_band_is_queued_as_one_composite_not_five_keys(client):
    """BACKLOG sec. 30. The web may ask for a band; it may not ask for the
    five ini values the band is made of."""
    ok = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"set_band": {"band_start_mhz": 7.5}}}})
    assert ok.status_code == 200

    pulled = client.get("/stations/SIM/commands", headers=CTL).json()["commands"]
    assert pulled[0]["params"]["changes"]["set_band"] == {"band_start_mhz": 7.5}


@pytest.mark.parametrize("key", [
    "center_freq", "min_freq", "max_freq",
    "minimum_analysis_frequency", "maximum_analysis_frequency",
    "manual_freq_extent",
])
def test_the_coupled_band_keys_are_not_individually_settable(client, key):
    """Each of these alone is a station that reports healthy and records
    nothing usable. `center_freq` alone is the 2026-08-19 failure exactly."""
    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config", "params": {"changes": {key: "20e6"}}})
    assert r.status_code == 400
    assert key in r.json()["detail"]


@pytest.mark.parametrize("payload,expect", [
    ({"analysis_min_mhz": 8.0}, "band_start_mhz"),
    ({"band_start_mhz": "wide"}, "not a number"),
    ({"band_start_mhz": -1}, "below zero"),
    ({"band_start_mhz": 7.5, "sample_rate": 30e6}, "sample_rate"),
    ({"band_start_mhz": 7.5, "analysis_min_mhz": 20, "analysis_max_mhz": 10},
     "not above"),
    ("7.5-32.5", "object"),
])
def test_a_malformed_band_is_refused_while_the_operator_is_watching(
        client, payload, expect):
    r = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config", "params": {"changes": {"set_band": payload}}})
    assert r.status_code == 400
    assert expect in r.json()["detail"]


def test_the_web_does_not_re_check_what_needs_the_station(client):
    """A window outside the digitised band depends on `sample_rate` in the
    station's own ini, which the server does not have. It queues here and the
    agent refuses it there -- deliberately, so there is one authority for it
    rather than two that can disagree."""
    ok = client.post("/stations/SIM/commands", headers=CTL, json={
        "name": "set_config",
        "params": {"changes": {"set_band": {"band_start_mhz": 0.0,
                                            "analysis_max_mhz": 30.0}}}})
    assert ok.status_code == 200


# --- the band panel ----------------------------------------------------------

def band_report(station="SIM", *, centre=20e6, lo=7.5e6, hi=32.5e6,
                patched=True):
    """A health report carrying the band, as `health.band` builds it."""
    body = report(station, healthy=True)
    body["metrics"] += [
        {"name": "center_freq_hz", "value": centre, "ok": True, "detail": ""},
        {"name": "band_start_hz", "value": centre - 12.5e6, "ok": True,
         "detail": ""},
        {"name": "band_stop_hz", "value": centre + 12.5e6, "ok": True,
         "detail": ""},
        {"name": "analysis_min_hz", "value": lo, "ok": True, "detail": ""},
        {"name": "analysis_max_hz", "value": hi, "ok": True, "detail": ""},
        {"name": "recorder_reads_ini",
         "value": True if patched else None,
         "ok": True if patched else None,
         "detail": "carries --center-freq (patch 0014)" if patched else
                   "/x/rx_uhd_ext_gps has no --center-freq option. "
                   "Apply patch 0014 and rebuild first."},
    ]
    return body


def _sounding(client, tmp_path, *, lo, hi, when="2026-08-19 16:24:05"):
    """One ingested product, which is where the observed band comes from."""
    from services.api import ingest

    path = tmp_path / "lfm_ionogram-NIC3-SIM-ch000-004-1787145845.00.h5"
    path.write_bytes(b"x")
    conn = client.app.state.db
    ingest.ingest_row(conn, {
        "file": path.name, "datetime": when, "tx": "NIC3", "rx": "SIM",
        "freq_start": lo, "freq_stop": hi}, path, tmp_path, ())
    conn.commit()          # `ingest_row` leaves the commit to its batch


def test_the_console_shows_the_configured_band(client):
    client.post("/stations/health", json=band_report(), headers=CTL)
    page = client.get("/ui").text
    assert "receive band" in page
    assert "7.500&ndash;32.500 MHz" in page or "7.500–32.500 MHz" in page


def test_a_station_on_an_old_recorder_gets_a_read_only_panel(client):
    """Phase 1 is not deployed there, so the band must not be offered. The
    agent would refuse it anyway; this is so the operator is not invited to
    try."""
    client.post("/stations/health", json=band_report(patched=False),
                headers=CTL)
    page = client.get("/ui").text
    assert "read-only" in page.lower()
    assert "patch 0014" in page
    assert 'id="bandStart-SIM"' not in page, (
        "the band fields must not be rendered for a recorder that cannot "
        "read the ini")


def test_a_patched_station_gets_the_fields(client):
    client.post("/stations/health", json=band_report(), headers=CTL)
    page = client.get("/ui").text
    assert 'id="bandStart-SIM"' in page
    assert "applyBand('SIM')" in page


def test_configured_and_observed_disagreeing_is_shown(client, tmp_path):
    """The 2026-08-19 failure, as the console would have caught it: the ini
    says 7.5-32.5 while the recorder is still on the 2.5-27.5 it was compiled
    with. Every unit green, products fresh, wrong spectrum."""
    client.post("/stations/health", json=band_report(), headers=CTL)
    _sounding(client, tmp_path, lo=2.55, hi=27.30)
    page = client.get("/ui").text
    assert "configured and observed disagree" in page.lower()
    assert "wrong spectrum" in page


def test_the_expected_edge_trim_is_not_called_a_disagreement(client, tmp_path):
    """7.5-32.5 configured is stored as 7.55-32.30, because
    `calc_ionograms.py:327` selects with strict inequalities. That is the
    normal case and must not cry wolf."""
    client.post("/stations/health", json=band_report(), headers=CTL)
    _sounding(client, tmp_path, lo=7.55, hi=32.30)
    page = client.get("/ui").text
    assert "configured and observed disagree" not in page
    assert "agree to" in page


def test_the_observed_row_says_so_when_there_are_no_products(client):
    client.post("/stations/health", json=band_report(), headers=CTL)
    assert "no products yet" in client.get("/ui").text.lower()


def test_a_station_that_never_reported_a_band_says_that(client):
    """Not the same as a band of zero. An older agent reports no band keys at
    all, and the panel must say so rather than render 0.000-0.000 MHz."""
    client.post("/stations/health", json=report(), headers=CTL)
    page = client.get("/ui").text
    assert "not reported" in page
    assert "0.000&ndash;0.000" not in page


def test_an_agent_that_reports_no_band_at_all_fails_closed(client):
    """The third state, and the one every station is in until the agent that
    reports `recorder_reads_ini` is deployed.

    Not "patched" and not "unpatched" but *no answer*, which must be read-only
    for the same reason `recorder_reads_the_ini` refuses on "cannot tell": a
    band written against a recorder that ignores it produces fresh products
    from the wrong spectrum and no error anywhere. Silence is not consent.
    """
    client.post("/stations/health", json=report(), headers=CTL)
    page = client.get("/ui").text
    assert "has not reported whether its recorder reads the ini" in page
    assert 'id="bandStart-SIM"' not in page
    assert "applyBand('SIM')" not in page


def test_nothing_is_called_a_disagreement_without_a_configured_band(client,
                                                                    tmp_path):
    """Products alone cannot disagree with anything. An old agent plus a full
    archive must not produce a red warning about a band it never reported."""
    client.post("/stations/health", json=report(), headers=CTL)
    _sounding(client, tmp_path, lo=7.55, hi=32.30)
    assert "configured and observed disagree" not in client.get("/ui").text
