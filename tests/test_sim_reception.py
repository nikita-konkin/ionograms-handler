import base64
import collections
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from muf.sim import presets
from muf.sim.state import ChannelState, Mode, Sweep
from services.api import db, live, sim_reception
from tests.test_api_sim_live import client  # noqa: F401

T0 = 1770206410.0
HEAD = {"type": "sounding", "tx": "cyprus1", "t0": T0,
        "columns": 80, "rows": 64, "f_lo_mhz": 10.0, "f_step_mhz": .05,
        "range_lo_km": 2400.0, "range_step_km": 10.0, "km_per_ms": 300.0}


def state(lead=300, epoch=T0):
    return {"epoch": epoch, "metrics": {"lead_s": lead}}


def test_matching_is_by_valid_time_and_shortest_issued_lead():
    ledger = {"pending": [{"forecast": s} for s in
                          [state(600), state(300), state(10, T0 + 20)]]}
    assert sim_reception.select(ledger, T0) == state(300)
    assert sim_reception.select(ledger, T0 + 300) is None
    assert sim_reception.select({"pending": [{"forecast": state(-1)}]}, T0) is None


def test_large_grid_is_bounded_without_changing_extent():
    head = dict(HEAD, columns=2033, rows=4001)
    grid = sim_reception.grid(head)
    assert grid["columns"] <= 800 and grid["rows"] <= 256
    for n, step in (("columns", "f_step_mhz"), ("rows", "range_step_km")):
        assert grid[n] * grid[step] == pytest.approx(head[n] * head[step])


def test_browsers_share_builds_and_cannot_grow_the_queue(monkeypatch):
    monkeypatch.setattr(sim_reception, "_cache", collections.OrderedDict())
    monkeypatch.setattr(sim_reception, "_queue", collections.deque())
    monkeypatch.setattr(sim_reception, "_worker", object())
    assert sim_reception.image(state(), HEAD)["status"] == "building"
    assert sim_reception.image(state(), HEAD)["status"] == "building"
    assert len(sim_reception._queue) == 1
    for i in range(1, sim_reception.LIMIT):
        sim_reception.image(state(epoch=T0 + i * 300), HEAD)
    assert sim_reception.image(state(epoch=T0 + 9999), HEAD)["status"] == "busy"
    assert len(sim_reception._queue) == sim_reception.LIMIT
    first = next(iter(sim_reception._cache))
    sim_reception._cache[first] = {"status": "ready", "cells": "AAAA"}
    assert sim_reception.image(state(), HEAD)["cells"] == "AAAA"


def test_real_synthesis_places_the_forecast_echo_on_live_axes():
    forecast = ChannelState(presets.cyprus1_circuit(), T0,
                            [Mode("flat", [10.5, 13.5], [9000, 9000], 45.0)],
                            sweep=Sweep(start_mhz=10, rate_hz_s=100e3,
                                        duration_s=40, sample_rate_hz=40e3))
    result = sim_reception.render(forecast.to_dict(), HEAD)
    cells = np.frombuffer(base64.b64decode(result["cells"]), np.uint8).reshape(80, 64)
    peak = HEAD["range_lo_km"] + np.argmax(cells[20:60].mean(axis=0)) * 10
    assert peak == pytest.approx(2700, abs=20)
    assert cells[20:60].max() > cells[:5].max()


def test_a_synthetic_trace_is_as_thin_as_the_live_sweep_resolves():
    head = dict(HEAD, columns=120, rows=96, f_lo_mhz=10.0, f_step_mhz=0.04096,
                range_lo_km=2400.0, range_step_km=7.32421875)
    forecast = ChannelState(presets.cyprus1_circuit(), T0,
                            [Mode("flat", [10.4, 14.6], [9137, 9137], 47.0)],
                            sweep=Sweep(start_mhz=10, rate_hz_s=100e3,
                                        duration_s=60, sample_rate_hz=40e3))
    cells = np.frombuffer(base64.b64decode(
        sim_reception.render(forecast.to_dict(), head)["cells"]),
        np.uint8).reshape(120, 96)[20:100] / 4
    width = np.median([np.sum(c > c.max() - 6) for c in cells])
    assert width <= 3, \
        f"a flat echo spreads over {width:g} live rows, not the 3 its window resolves"


def test_station_mapping_and_no_forecast_do_not_show_an_unrelated_image(tmp_path,
                                                                     monkeypatch):
    with db.session(tmp_path / "a.sqlite3") as conn:
        hub = live.Hub()
        hub.publish("receiver-a", [HEAD])
        initial = sim_reception.view(conn, hub, "cyprus1", "yoshkar-ola")
        assert initial["station"] is None
        conn.execute("INSERT INTO station_receiver(station,rx) VALUES (?,?)",
                     ("receiver-a", "yoshkar-ola"))
        view = sim_reception.view(conn, hub, "cyprus1", "yoshkar-ola")
        assert view["station"] == "receiver-a" and view["status"] == "no_forecast"
        conn.execute("INSERT INTO live_circuit(tx,rx,taken_at,ledger,updated_at) "
                     "VALUES (?,?,?,?,?)", ("cyprus1", "yoshkar-ola", "2026-02-04",
                     json.dumps({"pending": [{"forecast": state()}]}), db.utcnow()))
        calls = []
        def image(s, h):
            calls.append((s, h))
            return {"status": "building"}
        monkeypatch.setattr(sim_reception, "image", image)
        view = sim_reception.view(conn, hub, "cyprus1", "yoshkar-ola")
        assert view["status"] == "building" and view["issued_at"] == T0 - 300
        assert "data" not in calls[0][1], "received powers must not enter the forecast"
        hub.publish("receiver-a", [dict(HEAD, t0=T0 + 300)])
        assert sim_reception.view(conn, hub, "cyprus1", "yoshkar-ola")["status"] \
            == "no_forecast"
        assert len(calls) == 1


def test_page_has_two_live_panes_without_periodic_navigation(client):  # noqa: F811
    from services.api import main

    conn = main.app.state.db
    conn.execute("INSERT INTO sounding(file,path,datetime,tx,rx,ingested_at) "
                 "VALUES ('a.h5','a.h5','2026-02-04','cyprus1','yoshkar-ola',?)",
                 (db.utcnow(),))
    conn.commit()
    for lang in ("en", "ru"):
        html = client.get("/ui/sim/live?lang=" + lang).text
        assert 'id="simPrediction"' in html and 'id="simReceiving"' in html
        assert 'setTimeout(() => location.reload()' not in html
        assert '/static/sim-reception.js' in html
    assert client.get("/sim/live/cyprus1/yoshkar-ola/reception").json()["status"] \
        == "choose_station"


def test_live_columns_keep_gaps_and_ignore_other_sweep_updates():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = r"""
const assert=require('node:assert/strict');
const {picture,column}=require('./services/api/static/sim-reception.js');
const p=picture({t0:1,columns:3,rows:2});
column(p,{t0:1,index:2,db:Buffer.from([40,80]).toString('base64')});
column(p,{t0:2,index:0,db:Buffer.from([99,99]).toString('base64')});
column(p,{t0:1,index:2,db:Buffer.from([44,88]).toString('base64')});
assert.equal(p.columns.size,1);
assert.equal(p.columns.has(0),false);
assert.deepEqual([...p.columns.get(2)],[44,88]);
"""
    subprocess.run([node, "-e", script], check=True,
                   cwd=Path(__file__).resolve().parents[1])
