"""services.api.sim_period: a period of the archive scored on the hub."""

from __future__ import annotations

import datetime as dt
import json
import random
import threading
import time
from pathlib import Path

import pytest

from services.api import db, sim_period


def _day() -> dt.date:
    return dt.date(2031, 1, 1) + dt.timedelta(days=random.randint(0, 3000))


def _hub(tmp_path: Path, day: dt.date, count: int, step_s: float, *,
         tx: str = "NIC2", rx: str = "yoshkar-ola", conn=None):
    conn = conn or db.init(db.connect(tmp_path / "hub.sqlite3"))
    start = dt.datetime.combine(day, dt.time(random.randint(0, 3), 5, 15))
    for k in range(count):
        when = start + dt.timedelta(seconds=step_s * k)
        name = f"{tx}-{rx}-{when:%Y%m%d%H%M%S}.h5"
        conn.execute("INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
                     "VALUES (?,?,?,?,?,?)",
                     (name, name, f"{when:%Y-%m-%d %H:%M:%S}", tx, rx, db.utcnow()))
    conn.commit()
    return conn


@pytest.fixture
def scored(monkeypatch):
    calls = []

    def execute(paths, out, lead_s, progress):
        calls.append({"paths": [Path(p).name for p in paths], "lead_s": lead_s})
        progress(1, 2)
        Path(out).mkdir(parents=True)
        (Path(out) / "summary.json").write_text(json.dumps({"pairs": len(paths) - 1}),
                                                encoding="utf-8")

    monkeypatch.setattr(sim_period, "_execute", execute)
    return calls


def _settled(cache: Path, run_id: int, timeout_s: float = 10.0) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        got = sim_period.status(cache, run_id)
        if got["state"] not in ("queued", "running"):
            return got
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never settled")


def _submit(conn, tmp_path: Path, day: dt.date, **kw) -> dict:
    options = {"tx": "NIC2", "rx": "yoshkar-ola", "first": day, "last": day}
    options.update(kw)
    return sim_period.submit(conn, tmp_path / "sim", tmp_path, **options)


def test_a_period_longer_than_the_limit_is_refused(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    with pytest.raises(ValueError, match="at most"):
        _submit(conn, tmp_path, day,
                last=day + dt.timedelta(days=sim_period.MAX_DAYS))


def test_a_last_day_before_the_first_is_refused(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    with pytest.raises(ValueError, match="before"):
        _submit(conn, tmp_path, day, last=day - dt.timedelta(days=random.randint(1, 5)))


def test_a_period_with_none_of_the_circuits_soundings_is_not_found(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    with pytest.raises(LookupError, match="no soundings"):
        _submit(conn, tmp_path, day + dt.timedelta(days=random.randint(2, 9)))


def test_a_period_over_the_sounding_limit_is_refused(tmp_path, scored, monkeypatch):
    day = _day()
    limit = random.randint(2, 5)
    conn = _hub(tmp_path, day, limit + 1, 600.0)
    monkeypatch.setattr(sim_period, "MAX_SOUNDINGS", limit)
    with pytest.raises(ValueError, match="at most"):
        _submit(conn, tmp_path, day)


def test_a_run_reads_only_the_circuits_soundings_in_the_period(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 4, 600.0)
    for other in (day - dt.timedelta(days=1), day + dt.timedelta(days=1)):
        _hub(tmp_path, other, 2, 600.0, conn=conn)
    _hub(tmp_path, day, 2, 600.0, tx="SGO", rx="DOB", conn=conn)
    wanted = sorted(r["file"] for r in db.rows(
        conn, "SELECT file FROM sounding WHERE tx = 'NIC2' AND datetime LIKE ?",
        (f"{day}%",)))
    _settled(tmp_path / "sim", _submit(conn, tmp_path, day)["id"])
    assert sorted(scored[0]["paths"]) == wanted, \
        "the run read soundings outside the circuit or the period"


def test_the_lead_defaults_to_the_soundings_interval(tmp_path, scored):
    day = _day()
    step = random.choice((300.0, 600.0, 900.0))
    conn = _hub(tmp_path, day, 5, step)
    _settled(tmp_path / "sim", _submit(conn, tmp_path, day)["id"])
    assert scored[0]["lead_s"] == step, "the lead was not the soundings' interval"


def test_a_lead_asked_for_is_kept(tmp_path, scored):
    day = _day()
    lead = random.randint(2, 9) * 600.0
    conn = _hub(tmp_path, day, 5, 600.0)
    _settled(tmp_path / "sim", _submit(conn, tmp_path, day, lead_s=lead)["id"])
    assert scored[0]["lead_s"] == lead, "the lead asked for was not kept"


def test_a_lead_of_nothing_is_refused(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    with pytest.raises(ValueError, match="lead"):
        _submit(conn, tmp_path, day, lead_s=-random.random() * 600.0)


def test_a_finished_run_reads_as_done(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    assert _settled(tmp_path / "sim", _submit(conn, tmp_path, day)["id"])["state"] \
        == "done", "a finished run did not read as done"


def test_a_failed_run_says_why(tmp_path, monkeypatch):
    day = _day()
    why = f"no trace in {random.randint(1, 999)} soundings"

    def execute(paths, out, lead_s, progress):
        raise RuntimeError(why)

    monkeypatch.setattr(sim_period, "_execute", execute)
    conn = _hub(tmp_path, day, 3, 600.0)
    assert why in _settled(tmp_path / "sim",
                           _submit(conn, tmp_path, day)["id"])["error"], \
        "a failed run did not say why"


def test_a_running_run_says_how_far_it_has_got(tmp_path, monkeypatch):
    day = _day()
    done, total, release = random.randint(1, 50), random.randint(51, 99), \
        threading.Event()

    def execute(paths, out, lead_s, progress):
        progress(done, total)
        release.wait(10.0)

    monkeypatch.setattr(sim_period, "_execute", execute)
    conn = _hub(tmp_path, day, 3, 600.0)
    run_id = _submit(conn, tmp_path, day)["id"]
    end, seen = time.monotonic() + 10.0, None
    while seen is None and time.monotonic() < end:
        seen = sim_period.status(tmp_path / "sim", run_id).get("progress")
        time.sleep(0.02)
    release.set()
    _settled(tmp_path / "sim", run_id)
    assert (seen["done"], seen["total"]) == (done, total), \
        "a running run did not say how far it had got"


def test_a_run_that_left_no_result_reads_as_failed(tmp_path, monkeypatch):
    day = _day()
    monkeypatch.setattr(sim_period, "_execute",
                        lambda paths, out, lead_s, progress: None)
    conn = _hub(tmp_path, day, 3, 600.0)
    assert _settled(tmp_path / "sim", _submit(conn, tmp_path, day)["id"])["state"] \
        == "failed", "a run that left no result did not read as failed"


def test_the_runs_are_listed_newest_first(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    made = [_submit(conn, tmp_path, day)["id"] for _ in range(random.randint(2, 4))]
    for run_id in made:
        _settled(tmp_path / "sim", run_id)
    assert [r["id"] for r in sim_period.runs(tmp_path / "sim")] == made[::-1], \
        "the runs were not listed newest first"


def test_a_finished_runs_summary_can_be_downloaded(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    run_id = _submit(conn, tmp_path, day)["id"]
    _settled(tmp_path / "sim", run_id)
    assert sim_period.download(tmp_path / "sim", run_id, "summary.json").is_file(), \
        "a finished run's summary could not be downloaded"


def test_a_file_not_offered_cannot_be_downloaded(tmp_path, scored):
    day = _day()
    conn = _hub(tmp_path, day, 3, 600.0)
    run_id = _submit(conn, tmp_path, day)["id"]
    _settled(tmp_path / "sim", run_id)
    with pytest.raises(LookupError):
        sim_period.download(tmp_path / "sim", run_id, random.choice(
            ("../job.json", "job.json", "states", "..\\..\\hub.sqlite3")))


def test_a_scored_period_keeps_its_summary_and_not_its_states(tmp_path):
    from muf.sim import presets, sounding
    from muf.sim.state import ChannelState, Sweep

    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0,
                  sample_rate_hz=40e3)
    start = 1950000000.0 + random.randint(0, 10**6)
    paths = []
    for k in range(3):
        law = presets.law_mode(muf_mhz=13.6 + 0.07 * k, f_lo_mhz=10.5, cn0_dbhz=45.0,
                               spread_hz=0.5)
        state = ChannelState(presets.cyprus1_circuit(), start + 300.0 * k, [law],
                             sweep=sweep, source="test")
        paths.append(str(sounding.write_lfs(
            state, tmp_path / sounding.default_name(state), seed=k)))
    sim_period._score(paths, str(tmp_path / "run"), 300.0, lambda done, total: None)
    kept = {p.name for p in (tmp_path / "run").iterdir()}
    assert "summary.json" in kept and "states" not in kept, \
        "a scored period did not keep its summary without its states"


def _live(conn, day: dt.date, count: int, *, step_s: float = 600.0) -> list[dict]:
    """A followed circuit's live events over ``count`` soundings of ``day``,
    stored as the loop stores them."""
    from muf.sim import live, presets, series
    from muf.sim.state import ChannelState, Sweep

    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0,
                  sample_rate_hz=40e3)
    start = dt.datetime.combine(day, dt.time(random.randint(0, 3), 5, 15),
                                tzinfo=dt.timezone.utc).timestamp()
    ledger, events = live.Ledger(), []
    for k in range(count):
        epoch = start + step_s * k
        law = presets.law_mode(muf_mhz=13.6 + 0.1 * k, f_lo_mhz=10.5, cn0_dbhz=45.0,
                               spread_hz=0.5)
        state = ChannelState(presets.cyprus1_circuit(), epoch, [law], sweep=sweep,
                             source=f"s{k}.h5")
        ledger, got = live.step(ledger, series.Session(Path(f"s{k}.h5"), epoch, "NIC2",
                                                       "yoshkar-ola"),
                                state, lambda pending, s=state: s)
        for e in got:
            conn.execute("INSERT INTO live_event (tx, rx, epoch, kind, lead_s, "
                         "sounding_id, source_id, body) VALUES (?,?,?,?,?,?,?,?)",
                         ("NIC2", "yoshkar-ola", epoch, e["event"], e.get("lead_s"),
                          k + 1, None, json.dumps(e)))
        events.extend(got)
    conn.commit()
    return events


def _live_conn(tmp_path: Path):
    return db.init(db.connect(tmp_path / "hub.sqlite3"))


def test_a_period_of_live_forecasts_is_summed_up(tmp_path):
    conn, day = _live_conn(tmp_path), _day()
    events = _live(conn, day, 6)
    run = _submit(conn, tmp_path, day, source="live", lead_s=600.0)
    _settled(tmp_path / "sim", run["id"], timeout_s=120.0)
    scored = sum(e["event"] == "scored" and e["lead_s"] == 600.0 for e in events)
    assert sim_period.summary(tmp_path / "sim", run["id"])["pairs"] == scored > 0, \
        "a period of live forecasts was not summed up over its pairs"


def test_a_period_with_no_live_forecasts_is_not_found(tmp_path):
    conn, day = _live_conn(tmp_path), _day()
    _live(conn, day, 6)
    with pytest.raises(LookupError):
        _submit(conn, tmp_path, day + dt.timedelta(days=1), source="live")


@pytest.fixture
def summed(monkeypatch):
    """A live run summed up at once, so that no run is left for the worker when
    a test ends, to be taken for the next test's."""
    def total(out):
        (Path(out) / "summary.json").write_text(json.dumps({"pairs": 0}),
                                                encoding="utf-8")

    monkeypatch.setattr(sim_period, "_sum", total)


def test_a_live_period_defaults_to_the_shortest_lead_kept(tmp_path, summed):
    conn, day = _live_conn(tmp_path), _day()
    _live(conn, day, 6)
    run = _submit(conn, tmp_path, day, source="live")
    _settled(tmp_path / "sim", run["id"])
    assert run["lead_s"] == 600.0, \
        "a live period did not default to the shortest lead kept"


def test_a_live_period_at_a_lead_never_forecast_is_not_found(tmp_path):
    conn, day = _live_conn(tmp_path), _day()
    _live(conn, day, 6)
    with pytest.raises(LookupError):
        _submit(conn, tmp_path, day, source="live", lead_s=900.0)


def test_a_live_run_says_it_was_live(tmp_path, summed):
    conn, day = _live_conn(tmp_path), _day()
    _live(conn, day, 6)
    run = _submit(conn, tmp_path, day, source="live")
    _settled(tmp_path / "sim", run["id"])
    assert run["source"] == "live", "a live run does not say it was live"


def test_an_unknown_source_is_refused(tmp_path, scored):
    conn, day = _live_conn(tmp_path), _day()
    _hub(tmp_path, day, 3, 600.0, conn=conn)
    with pytest.raises(ValueError):
        _submit(conn, tmp_path, day, source="forecast")


def test_the_live_circuits_are_offered_with_the_leads_kept(tmp_path):
    conn, day = _live_conn(tmp_path), _day()
    _live(conn, day, 6)
    found = sim_period.live_circuits(conn)
    assert [(c["tx"], c["rx"], c["leads"]) for c in found] == [
        ("NIC2", "yoshkar-ola", [600.0, 1200.0])], \
        "the live circuits were not offered with the leads kept"
