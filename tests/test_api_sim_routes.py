"""The hub's simulator page: a sounding's console, built on demand."""

from __future__ import annotations

import contextlib
import datetime as dt
import functools
import json
import random
import threading
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from services.api import auth, db, main, sim, sim_period  # noqa: E402

CTL = {"Authorization": "Bearer ctl"}


@pytest.fixture
def made(monkeypatch):
    calls = []

    def build(path, progress):
        calls.append(path.name)
        progress(1, 2)
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


def _reporting(client, sounding_id: int, timeout_s: float = 10.0) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        got = client.get(f"/sim/{sounding_id}").json()
        if got.get("progress"):
            return got["progress"]
        time.sleep(0.02)
    raise AssertionError(f"the build of {sounding_id} never reported its progress")


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


def test_a_running_build_says_how_far_it_has_got(client, tmp_path, monkeypatch):
    done, total, release = random.randint(1, 12), random.randint(13, 20), \
        threading.Event()

    def build(path, progress):
        progress(done, total)
        release.wait(10.0)
        return {"source": path.name, "modes": []}

    monkeypatch.setattr(sim, "_execute", build)
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    seen = _reporting(client, k)
    release.set()
    _settled(client, k)
    assert (seen["done"], seen["total"]) == (done, total), \
        "a running build did not say how far it had got"


def test_a_finished_build_leaves_no_progress_behind(client, tmp_path):
    k = _sounding(client, tmp_path)
    client.post(f"/sim/{k}", headers=CTL)
    _settled(client, k)
    assert list(client.app.state.sim_cache.glob("*.progress")) == [], \
        "a finished build left its progress behind"


def test_the_page_has_a_progress_bar(client, tmp_path):
    _sounding(client, tmp_path)
    assert 'id="simBar"' in client.get("/ui/sim").text, \
        "the page has no progress bar"


def test_a_progress_held_open_does_not_stop_later_builds(client, tmp_path, monkeypatch):
    with contextlib.ExitStack() as held:

        def build(path, progress):
            progress(1, 2)
            held.enter_context(open(progress.path, encoding="utf-8"))
            return {"source": path.name, "modes": []}

        monkeypatch.setattr(sim, "_execute", build)
        first = _sounding(client, tmp_path)
        client.post(f"/sim/{first}", headers=CTL)
        _settled(client, first)
        later = _sounding(client, tmp_path,
                          f"cyprus1_20260204_12{random.randint(10, 59)}10.lfs")
        client.post(f"/sim/{later}", headers=CTL)
        state = _settled(client, later)["state"]
    assert state == "done", "a progress held open stopped later builds"


@pytest.fixture
def scored(monkeypatch):
    def execute(paths, out, lead_s, progress):
        progress(1, 2)
        Path(out).mkdir(parents=True)
        (Path(out) / "summary.json").write_text(json.dumps(
            {"pairs": 0, "set_apart": 0, "stretches": 0, "reason": "no pairs scored"}),
            encoding="utf-8")

    monkeypatch.setattr(sim_period, "_execute", execute)


def _day() -> dt.date:
    return dt.date(2031, 1, 1) + dt.timedelta(days=random.randint(0, 3000))


def _circuit(client, day: dt.date, count: int = 3, step_s: float = 600.0,
             files=None) -> None:
    conn = client.app.state.db
    start = dt.datetime.combine(day, dt.time(random.randint(0, 3), 5, 15))
    for k in range(count):
        when = start + dt.timedelta(seconds=step_s * k)
        name = files[k] if files else f"NIC2-yoshkar-ola-{when:%Y%m%d%H%M%S}.h5"
        conn.execute("INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
                     "VALUES (?,?,?,?,?,?)",
                     (name, name, f"{when:%Y-%m-%d %H:%M:%S}", "NIC2", "yoshkar-ola",
                      db.utcnow()))
    conn.commit()


def _ask(day: dt.date, **kw) -> dict:
    return {"tx": "NIC2", "rx": "yoshkar-ola", "first": day.isoformat(),
            "last": day.isoformat(), **kw}


def _period_settled(client, run_id: int, timeout_s: float = 10.0) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        got = client.get(f"/sim/periods/{run_id}").json()
        if got["state"] not in ("queued", "running"):
            return got
        time.sleep(0.02)
    raise AssertionError(f"period run {run_id} never settled")


def _scored_period(client, day: dt.date) -> int:
    _circuit(client, day)
    run_id = client.post("/sim/periods", json=_ask(day), headers=CTL).json()["id"]
    _period_settled(client, run_id)
    return run_id


def test_scoring_a_period_needs_the_control_token(client, scored):
    day = _day()
    _circuit(client, day)
    assert client.post("/sim/periods", json=_ask(day)).status_code == 401, \
        "a period was scored without the control token"


def test_a_period_is_scored_on_request(client, scored):
    assert _period_settled(client, _scored_period(client, _day()))["state"] == "done", \
        "a period asked for was not scored"


def test_a_period_with_a_malformed_day_is_refused(client, scored):
    day = _day()
    _circuit(client, day)
    assert client.post("/sim/periods", json=_ask(day, first=random.choice(
        ("tomorrow", "2031-13-01", "", None))), headers=CTL).status_code == 422, \
        "a malformed day was not refused"


def test_a_period_longer_than_the_limit_is_refused(client, scored):
    day = _day()
    _circuit(client, day)
    last = day + dt.timedelta(days=sim_period.MAX_DAYS + random.randint(0, 30))
    assert client.post("/sim/periods", json=_ask(day, last=last.isoformat()),
                       headers=CTL).status_code == 422, \
        "a period longer than the limit was not refused"


def test_a_period_with_no_soundings_is_not_found(client, scored):
    day = _day()
    _circuit(client, day)
    assert client.post("/sim/periods",
                       json=_ask(day + dt.timedelta(days=random.randint(2, 9))),
                       headers=CTL).status_code == 404, \
        "a period with no soundings was found"


def test_the_periods_are_listed(client, scored):
    run_id = _scored_period(client, _day())
    assert [r["id"] for r in client.get("/sim/periods").json()] == [run_id], \
        "the period runs were not listed"


def test_an_unknown_period_is_not_found(client):
    assert client.get(f"/sim/periods/{random.randint(1, 999)}").status_code == 404, \
        "a period never asked for was found"


def test_a_finished_periods_summary_can_be_downloaded(client, scored):
    run_id = _scored_period(client, _day())
    assert client.get(f"/sim/periods/{run_id}/summary.json").json()["pairs"] == 0, \
        "a finished period's summary could not be downloaded"


def test_a_file_not_offered_is_not_found(client, scored):
    run_id = _scored_period(client, _day())
    assert client.get(f"/sim/periods/{run_id}/job.json").status_code == 404, \
        "a file not offered for download was served"


def test_the_simulator_page_leads_to_the_periods(client, tmp_path):
    _sounding(client, tmp_path)
    assert 'href="/ui/sim/periods"' in client.get("/ui/sim").text, \
        "the simulator page does not lead to the periods"


def test_the_periods_page_offers_the_circuits(client):
    _circuit(client, _day())
    assert "NIC2 → yoshkar-ola" in client.get("/ui/sim/periods").text, \
        "the periods page did not offer the circuit"


def _finished_run(client, tmp_path, monkeypatch) -> int:
    from muf.sim import presets, sounding, summary
    from muf.sim.state import ChannelState, Sweep

    monkeypatch.setattr(summary, "metrics",
                        functools.partial(summary.metrics, minimum=1))
    monkeypatch.setattr(sim_period, "_execute", sim_period._score)
    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)
    day = _day()
    start = dt.datetime.combine(day, dt.time(9, 0), tzinfo=dt.timezone.utc).timestamp()
    files = []
    for k in range(3):
        law = presets.law_mode(muf_mhz=12.5 + random.random(), f_lo_mhz=10.5,
                               cn0_dbhz=45.0, spread_hz=0.5)
        state = ChannelState(presets.cyprus1_circuit(), start + 600.0 * k, [law],
                             sweep=sweep, source="test")
        files.append(sounding.write_lfs(state, tmp_path / sounding.default_name(state),
                                        seed=k).name)
    _circuit(client, day, files=files)
    run_id = client.post("/sim/periods", json=_ask(day), headers=CTL).json()["id"]
    _period_settled(client, run_id, timeout_s=120.0)
    return run_id


def _notes(page: str, attribute: str) -> set[str]:
    import re
    return set(re.findall(attribute + r'="#?note-([a-z_]+)"', page))


def test_the_periods_page_shows_a_finished_runs_figures(client, tmp_path, monkeypatch):
    run_id = _finished_run(client, tmp_path, monkeypatch)
    mae = sim_period.summary(client.app.state.sim_cache,
                             run_id)["all"]["muf_mhz"]["mae"]
    assert f"{mae:.2f}" in client.get(f"/ui/sim/periods?id={run_id}").text, \
        "the periods page did not show the run's MUF error"


def test_every_footnote_mark_on_the_periods_page_leads_to_a_note(client, tmp_path,
                                                                 monkeypatch):
    run_id = _finished_run(client, tmp_path, monkeypatch)
    page = client.get(f"/ui/sim/periods?id={run_id}").text
    assert set() != _notes(page, "href") <= _notes(page, "id"), \
        "a footnote mark on the periods page leads to no note, or none is shown"


def test_every_note_on_the_periods_page_is_marked_where_it_applies(client, tmp_path,
                                                                   monkeypatch):
    run_id = _finished_run(client, tmp_path, monkeypatch)
    page = client.get(f"/ui/sim/periods?id={run_id}").text
    assert set() != _notes(page, "id") == _notes(page, "href"), \
        "a note on the periods page is marked nowhere, or no notes are shown"


def test_the_notes_give_the_usable_threshold_the_run_was_scored_with(client, tmp_path,
                                                                     monkeypatch):
    run_id = _finished_run(client, tmp_path, monkeypatch)
    params = sim_period.summary(client.app.state.sim_cache, run_id)["params"]
    page = client.get(f"/ui/sim/periods?id={run_id}").text
    note = page[page.index('id="note-carriers"'):].split("</li>", 1)[0]
    assert f"{params['snr_min_db']:g} dB in {params['bandwidth_hz']:g} Hz" in note, \
        "the notes do not give the usable threshold the run was scored with"


def test_the_russian_periods_page_asks_for_no_missing_string(client, tmp_path,
                                                             monkeypatch):
    from services.api import i18n

    run_id = _finished_run(client, tmp_path, monkeypatch)
    i18n.MISSES.clear()
    client.get(f"/ui/sim/periods?id={run_id}&lang=ru")
    assert not i18n.MISSES, f"the Russian periods page misses {sorted(i18n.MISSES)}"
