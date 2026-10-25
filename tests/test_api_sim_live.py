"""services.api.sim_live: the simulator's forecasts on the soundings the hub stores."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from muf.sim import presets, sounding
from muf.sim.state import ChannelState, Sweep
from services.api import db, sim_live

#: A short sweep, 10-14 MHz in 40 s, so a sounding is 1.6 M samples.
SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)


def _conn(tmp_path: Path):
    return db.init(db.connect(tmp_path / "api.sqlite3"))


def _sounding(conn, when: str, *, name: str | None = None, tx: str = "NIC2",
              rx: str = "yoshkar-ola") -> int:
    name = name or f"{tx}_{when[:10].replace('-', '')}_{when[11:].replace(':', '')}.h5"
    cursor = conn.execute(
        "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
        "VALUES (?,?,?,?,?,?)", (name, name, when, tx, rx, db.utcnow()))
    conn.commit()
    return cursor.lastrowid


def _turner(calls: list, *, fail: bool = False):
    def turn(ledger, path, epoch, tx, rx, cache):
        calls.append(Path(path).name)
        if fail:
            raise RuntimeError(f"the live forecast on {Path(path).name}: broken")
        return ({**ledger, "times": [*ledger["times"], epoch]},
                [{"schema": "muf.sim.live.event", "version": 1, "event": "issued",
                  "at": "2026-10-24T12:00:00Z", "from": Path(path).name,
                  "lead_s": 600.0}])
    return turn


def _advance(conn, tmp_path: Path, calls: list, **kw) -> int:
    fail = kw.pop("fail", False)
    return sim_live.advance(conn, tmp_path, tmp_path / "cache",
                        turn=_turner(calls, fail=fail), **kw)


def _kinds(conn) -> list[str]:
    return [r["kind"] for r in db.rows(conn, "SELECT kind FROM live_event ORDER BY id")]


def _follow(conn) -> dict:
    return sim_live.follow(conn, "NIC2", "yoshkar-ola")


def test_a_circuit_not_followed_is_left_alone(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, calls)
    assert db.one(conn, "SELECT COUNT(*) AS n FROM live_circuit")["n"] == 0


def test_a_followed_circuit_begins_at_its_newest_sounding(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _sounding(conn, "2026-10-24 12:10:15")
    _follow(conn)
    _advance(conn, tmp_path, calls)
    assert calls == [], "the archive's past was replayed"


def test_a_followed_circuit_knows_when_its_last_soundings_were(tmp_path):
    conn = _conn(tmp_path)
    _sounding(conn, "2026-10-24 12:00:15")
    _sounding(conn, "2026-10-24 12:10:15")
    _follow(conn)
    ledger = json.loads(db.one(conn, "SELECT ledger FROM live_circuit")["ledger"])
    assert len(ledger["times"]) == 2, "a followed circuit began without its interval"


def test_a_circuit_with_no_soundings_cannot_be_followed(tmp_path):
    with pytest.raises(LookupError, match="no soundings of NIC2 -> yoshkar-ola"):
        _follow(_conn(tmp_path))


def test_a_stopped_circuit_is_left_alone(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    sim_live.stop(conn, "NIC2", "yoshkar-ola")
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, calls)
    assert calls == [], "a stopped circuit was still forecast"


def test_a_circuit_followed_again_begins_at_its_newest_sounding(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    sim_live.stop(conn, "NIC2", "yoshkar-ola")
    _sounding(conn, "2026-10-24 12:10:15")
    _sounding(conn, "2026-10-24 12:20:15")
    _follow(conn)
    _advance(conn, tmp_path, calls)
    assert calls == [], "following again replayed the soundings missed"


def test_the_soundings_after_the_newest_are_taken_in_time_order(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    _sounding(conn, "2026-10-24 12:20:15")
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, calls)
    assert calls == ["NIC2_20261024_121015.h5", "NIC2_20261024_122015.h5"]


def test_a_sounding_older_than_the_last_taken_is_passed_over(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    _sounding(conn, "2026-10-24 12:20:15")
    _advance(conn, tmp_path, calls)
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, calls)
    assert calls == ["NIC2_20261024_122015.h5"], "a late sounding was forecast live"


def test_a_turn_that_fails_is_stored_as_an_unobserved_event(tmp_path):
    conn = _conn(tmp_path)
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, [], fail=True)
    assert _kinds(conn) == ["unobserved"], "a failed turn left no event"


def test_a_failed_turn_does_not_hold_the_circuit_up(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, calls, fail=True)
    _sounding(conn, "2026-10-24 12:20:15")
    _advance(conn, tmp_path, calls)
    assert calls == ["NIC2_20261024_121015.h5", "NIC2_20261024_122015.h5"], \
        "the circuit was held up by a failed turn"


def test_an_event_names_the_sounding_it_came_from(tmp_path):
    conn = _conn(tmp_path)
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    k = _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, [])
    assert db.one(conn, "SELECT source_id FROM live_event")["source_id"] == k


def test_a_muted_circuit_is_left_alone(tmp_path):
    conn, calls = _conn(tmp_path), []
    _sounding(conn, "2026-10-24 12:00:15")
    _follow(conn)
    db.mute_circuit(conn, "NIC2", "yoshkar-ola")
    _sounding(conn, "2026-10-24 12:10:15")
    _advance(conn, tmp_path, calls)
    assert calls == [], "a muted circuit was forecast"


def test_events_older_than_the_days_kept_are_forgotten(tmp_path):
    conn = _conn(tmp_path)
    _sounding(conn, "2026-09-01 12:00:15")
    _follow(conn)
    _sounding(conn, "2026-09-01 12:10:15")
    _advance(conn, tmp_path, [])
    _sounding(conn, "2026-10-24 12:00:15")
    _advance(conn, tmp_path, [], keep_days=30.0)
    assert len(_kinds(conn)) == 1, "an event older than the days kept was kept"


def _lfs(folder: Path, epoch: float, k: int) -> Path:
    law = presets.law_mode(muf_mhz=13.6 + 0.07 * k, f_lo_mhz=10.5, cn0_dbhz=45.0,
                           spread_hz=0.5)
    state = ChannelState(presets.cyprus1_circuit(), epoch, [law], sweep=SHORT,
                         source="test")
    return sounding.write_lfs(state, folder / sounding.default_name(state), seed=k)


def test_a_real_turn_scores_a_forecast_at_its_sounding(tmp_path):
    conn, root = _conn(tmp_path), tmp_path / "archive"
    root.mkdir()
    t0 = 1770206410.0
    paths = [_lfs(root, t0 + 300.0 * k, k) for k in range(4)]
    stamps = ["2026-02-04 12:00:10", "2026-02-04 12:05:10", "2026-02-04 12:10:10",
              "2026-02-04 12:15:10"]
    for path, when in zip(paths[:2], stamps[:2]):
        _sounding(conn, when, name=path.name, tx="cyprus1")
    sim_live.follow(conn, "cyprus1", "yoshkar-ola")
    for path, when in zip(paths[2:], stamps[2:]):
        _sounding(conn, when, name=path.name, tx="cyprus1")
    sim_live.advance(conn, root, tmp_path / "cache", turn=sim_live._turn)
    assert _kinds(conn).count("scored") == 1, f"no forecast was scored: {_kinds(conn)}"


# --- the page -----------------------------------------------------------------

#: 2026-10-24 12:00:15 UTC.
T = 1792843215.0


def _utc(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _held(conn, pending=()) -> None:
    conn.execute("INSERT INTO live_circuit (tx, rx, taken_at, ledger, updated_at) "
                 "VALUES (?,?,?,?,?)",
                 ("NIC2", "yoshkar-ola", "2026-10-24 12:00:15",
                  json.dumps({"pending": list(pending), "times": [T - 600.0, T]}),
                  db.utcnow()))
    conn.commit()


def _scored(conn, epoch: float, muf_error: float, *, sounding_id=None,
            source_id=None, tx: str = "NIC2", suspect: bool = False) -> None:
    bounded = {"forecast": 20.0, "truth": 20.0 - muf_error, "error": muf_error,
               "censored_truth": False, "censored_forecast": False}
    body = {"schema": "muf.sim.live.event", "version": 1, "event": "scored",
            "at": _utc(epoch), "from": "a.h5", "to": "b.h5", "lead_s": 600.0,
            "issued_at": _utc(epoch - 600.0), "valid_at": _utc(epoch),
            "clock": {"suspect": suspect, "jump_us": 1520.0 if suspect else 40.0,
                      "offsets_us": [480.0, 2000.0 if suspect else 520.0],
                      "why": "the echo moved 1520 us between the soundings"
                             if suspect else ""},
            "pair": {"muf_mhz": bounded, "lof_mhz": bounded,
                     "modes": {"matched": 1, "missed": 0, "false": 0}},
            "modes": [{"status": "matched", "delay_rms_us": 10.0, "cn0_rms_db": 1.0,
                       "f_hi_error_mhz": 0.1, "f_hi_censored_truth": False,
                       "f_hi_censored_forecast": False, "overlap_mhz": 1.0}]}
    conn.execute("INSERT INTO live_event (tx, rx, epoch, kind, lead_s, sounding_id, "
                 "source_id, body) VALUES (?,?,?,?,?,?,?,?)",
                 (tx, "yoshkar-ola", epoch, "scored", 600.0, sounding_id,
                  source_id, json.dumps(body)))
    conn.commit()


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from services.api import auth, main

    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("ARCHIVE_SCAN_INTERVAL_S", "0")
    monkeypatch.setenv("LIVE_INTERVAL_S", "0")
    with TestClient(main.app) as c:
        yield c


def test_the_view_sums_up_the_last_hours_pairs(tmp_path):
    conn = _conn(tmp_path)
    _held(conn)
    for epoch, error in ((T - 4000.0, 3.0), (T - 600.0, 1.0), (T, 0.5)):
        _scored(conn, epoch, error)
    assert sim_live.view(conn, "NIC2", "yoshkar-ola")["figures"]["hour"][600.0]["n"] \
        == 2


def test_the_view_shows_each_leads_latest_score(tmp_path):
    conn = _conn(tmp_path)
    _held(conn)
    _scored(conn, T - 600.0, 1.0)
    _scored(conn, T, 0.5)
    latest = sim_live.view(conn, "NIC2", "yoshkar-ola")["latest"][600.0]
    assert latest["event"]["pair"]["muf_mhz"]["error"] == 0.5


def test_the_view_shows_the_forecasts_waiting(tmp_path):
    from muf.sim import forecast
    from muf.sim import live as loop

    law = presets.law_mode(muf_mhz=13.6, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.5)
    state = ChannelState(presets.cyprus1_circuit(), T, [law], sweep=SHORT, source="a")
    conn = _conn(tmp_path)
    _held(conn, [loop.Pending(Path("a.h5"), forecast.persist(state, 1200.0)).to_dict()])
    assert [p["lead_s"]
            for p in sim_live.view(conn, "NIC2", "yoshkar-ola")["pending"]] \
        == [1200.0], "the forecast waiting was not shown"


def test_the_live_page_shows_the_circuit(client):
    _held(client.app.state.db)
    assert "NIC2" in client.get("/ui/sim/live").text, "the circuit is not on the page"


def test_the_live_page_shows_both_ionograms_of_a_score(client):
    _held(client.app.state.db)
    _scored(client.app.state.db, T, 0.5, sounding_id=7, source_id=6)
    page = client.get("/ui/sim/live").text
    assert all(f"/ionogram/{k}.png" in page for k in (6, 7)), \
        "the page does not show the forecast's and the truth's ionograms"


def test_the_live_page_says_when_the_loop_is_off(client):
    assert "LIVE_INTERVAL_S" in client.get("/ui/sim/live").text, \
        "the page did not say the loop is off"


def test_the_simulator_page_leads_to_the_live_page(client):
    assert 'href="/ui/sim/live"' in client.get("/ui/sim").text


CTL = {"Authorization": "Bearer ctl"}


def test_following_a_circuit_needs_the_control_token(client):
    _sounding(client.app.state.db, "2026-10-24 12:00:15")
    assert client.post("/sim/live/NIC2/yoshkar-ola").status_code == 401


def test_a_circuit_is_followed_from_its_route(client):
    _sounding(client.app.state.db, "2026-10-24 12:00:15")
    assert client.post("/sim/live/NIC2/yoshkar-ola", headers=CTL).json()["followed"], \
        "the circuit was not followed"


def test_an_unknown_circuit_cannot_be_followed(client):
    assert client.post("/sim/live/NIC9/nowhere", headers=CTL).status_code == 404


def test_a_circuit_is_stopped_from_its_route(client):
    _sounding(client.app.state.db, "2026-10-24 12:00:15")
    client.post("/sim/live/NIC2/yoshkar-ola", headers=CTL)
    client.delete("/sim/live/NIC2/yoshkar-ola", headers=CTL)
    assert db.one(client.app.state.db, "SELECT COUNT(*) AS n FROM live_circuit")["n"] \
        == 0, "the circuit was still followed"


def test_the_live_page_offers_a_circuit_not_followed(client):
    _sounding(client.app.state.db, "2026-10-24 12:00:15", tx="NIC1")
    page = client.get("/ui/sim/live?tx=NIC1&rx=yoshkar-ola").text
    assert 'id="liveFollow"' in page, "the page did not offer to follow the circuit"


def test_the_view_leaves_clock_suspect_pairs_out_of_its_figures(tmp_path):
    conn = _conn(tmp_path)
    _held(conn)
    _scored(conn, T - 600.0, 1.0)
    _scored(conn, T, 9.0, suspect=True)
    assert sim_live.view(conn, "NIC2", "yoshkar-ola")["figures"]["hour"][600.0]["n"] \
        == 1, "a clock-suspect pair was counted in the view's figures"


def test_the_view_keeps_a_clock_suspect_latest_score_in_its_own_figures(tmp_path):
    conn = _conn(tmp_path)
    _held(conn)
    _scored(conn, T, 9.0, suspect=True)
    assert sim_live.view(conn, "NIC2", "yoshkar-ola")["latest"][600.0]["figures"][
        "n"] == 1, "the latest score's own figures left it out"


def test_the_view_marks_the_clock_suspect_events(tmp_path):
    conn = _conn(tmp_path)
    _held(conn)
    _scored(conn, T - 600.0, 1.0)
    _scored(conn, T, 9.0, suspect=True)
    recent = sim_live.view(conn, "NIC2", "yoshkar-ola")["recent"]
    assert [r["suspect"] for r in recent] == [True, False], \
        "the view did not mark the clock-suspect events"


def test_the_live_page_says_a_score_was_set_apart(client):
    _held(client.app.state.db)
    _scored(client.app.state.db, T, 9.0, suspect=True)
    assert "set apart from the running figures" in client.get("/ui/sim/live").text, \
        "the page did not say the score was set apart"


def test_the_live_page_counts_the_pairs_set_apart(client):
    _held(client.app.state.db)
    _scored(client.app.state.db, T, 9.0, suspect=True)
    assert "Set apart" in client.get("/ui/sim/live").text, \
        "the page did not count the pairs set apart"


def test_an_exported_event_says_whether_the_clock_broke_it(tmp_path):
    conn = _conn(tmp_path)
    _scored(conn, T, 9.0, suspect=True)
    event = sim_live.events(conn, "NIC2", "yoshkar-ola")[0]
    assert event["clock"]["suspect"] is True, "the export lost the clock's verdict"


def test_the_events_after_an_id_are_exported_oldest_first(tmp_path):
    conn = _conn(tmp_path)
    for epoch in (T - 1200.0, T - 600.0, T):
        _scored(conn, epoch, 1.0)
    assert [e["id"] for e in sim_live.events(conn, "NIC2", "yoshkar-ola", after=1)] \
        == [2, 3], "the export did not carry on after the id it was given"


def test_an_exported_event_keeps_its_forecast_and_its_truth(tmp_path):
    conn = _conn(tmp_path)
    _scored(conn, T, 0.5)
    pair = sim_live.events(conn, "NIC2", "yoshkar-ola")[0]["pair"]
    assert (pair["muf_mhz"]["forecast"], pair["muf_mhz"]["truth"]) == (20.0, 19.5), \
        "the export lost the forecast's or the truth's value"


def test_an_exported_event_names_its_circuit(tmp_path):
    conn = _conn(tmp_path)
    _scored(conn, T, 0.5)
    event = sim_live.events(conn, "NIC2", "yoshkar-ola")[0]
    assert (event["tx"], event["rx"]) == ("NIC2", "yoshkar-ola"), \
        "the exported event does not name its circuit"


def test_the_export_leaves_other_circuits_out(tmp_path):
    conn = _conn(tmp_path)
    _scored(conn, T - 600.0, 1.0, tx="NIC1")
    _scored(conn, T, 0.5)
    assert [e["tx"] for e in sim_live.events(conn, "NIC2", "yoshkar-ola")] \
        == ["NIC2"], "another circuit's events were exported"


def test_the_export_stops_at_its_limit(tmp_path):
    conn = _conn(tmp_path)
    for epoch in (T - 1200.0, T - 600.0, T):
        _scored(conn, epoch, 1.0)
    assert len(sim_live.events(conn, "NIC2", "yoshkar-ola", limit=2)) == 2, \
        "the export went past its limit"


def test_the_events_route_answers_in_json_lines(client):
    _scored(client.app.state.db, T, 0.5)
    line = client.get("/sim/live/NIC2/yoshkar-ola/events").text.splitlines()[0]
    assert json.loads(line)["event"] == "scored", "the route's line is not the event"


def test_the_events_route_says_it_answers_in_json_lines(client):
    _scored(client.app.state.db, T, 0.5)
    assert client.get("/sim/live/NIC2/yoshkar-ola/events").headers["content-type"] \
        .startswith("application/x-ndjson"), "the route did not say it is JSON Lines"


def test_the_events_route_carries_on_after_an_id(client):
    _scored(client.app.state.db, T - 600.0, 1.0)
    _scored(client.app.state.db, T, 0.5)
    lines = client.get("/sim/live/NIC2/yoshkar-ola/events?after=1").text.splitlines()
    assert [json.loads(line)["id"] for line in lines] == [2], \
        "the route did not carry on after the id it was given"


def test_the_events_route_refuses_an_unbounded_page(client):
    assert client.get("/sim/live/NIC2/yoshkar-ola/events?limit=100000").status_code \
        == 422, "the route took a page past its bound"
