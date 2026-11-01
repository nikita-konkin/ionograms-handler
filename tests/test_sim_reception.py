import base64
import collections
import json
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from muf import echo_window
from muf.sim import observe, presets, sounding
from muf.sim.state import ChannelState, Mode, Sweep
from services.agent.live import Capture
from services.api import db, live, sim_reception
from tests.test_api_sim_live import client  # noqa: F401

T0 = 1770206410.0
HEAD = {"type": "sounding", "tx": "cyprus1", "t0": T0,
        "columns": 80, "rows": 64, "f_lo_mhz": 10.0, "f_step_mhz": .05,
        "range_lo_km": 2400.0, "range_step_km": 10.0, "km_per_ms": 300.0}


def state(lead=300, epoch=T0):
    return {"epoch": epoch, "metrics": {"lead_s": lead}}


def kept(tmp_path, lead=300, epoch=T0):
    product = tmp_path / f"{epoch:.0f}.{lead}s.sim.echo.h5"
    product.write_bytes(b"")
    return {"forecast": state(lead, epoch), "simulated": {"product": str(product)}}


def _forecast(duration=60.0, modes=None):
    return ChannelState(presets.cyprus1_circuit(), T0,
                        modes or [Mode("flat", [10.4, 14.6], [9137.0, 9137.0], 47.0),
                                  Mode("slope", [10.6, 13.9], [9300.0, 9700.0], 44.0)],
                        sweep=Sweep(start_mhz=10, rate_hz_s=100e3,
                                    duration_s=duration, sample_rate_hz=40e3))


def _issued(tmp_path, forecast):
    lfs = sounding.write_lfs(forecast, tmp_path / sounding.default_name(forecast),
                             seed=random.randrange(1000))
    product = echo_window.write(lfs, out_dir=tmp_path).target
    return lfs, {"forecast": forecast.to_dict(),
                 "simulated": {"product": str(product), "state": forecast.to_dict()}}


def _received(lfs):
    capture = Capture(lfs)
    return capture.head, np.array(
        [np.frombuffer(base64.b64decode(c["db"]), np.uint8)
         for c in capture.columns() if c["type"] == "column"])


def _drawn(result):
    head = result["head"]
    return np.frombuffer(base64.b64decode(result["cells"]), np.uint8).reshape(
        head["columns"], head["rows"])


def test_matching_is_by_valid_time_and_shortest_issued_lead():
    ledger = {"pending": [{"forecast": s} for s in
                          [state(600), state(300), state(10, T0 + 20)]]}
    assert sim_reception.select(ledger, T0)["forecast"] == state(300)
    assert sim_reception.select(ledger, T0 + 300) is None
    assert sim_reception.select({"pending": [{"forecast": state(-1)}]}, T0) is None


def test_identical_samples_give_the_cells_the_live_sweep_gives(tmp_path):
    lfs, _ = _issued(tmp_path, _forecast())
    head, received = _received(lfs)
    ion = observe.load(lfs, window=16384, gate_km=sim_reception.gate_of(head))
    assert np.array_equal(sim_reception.cells(ion, head), received), \
        "identical samples did not give the live sweep's cells"


def test_the_issued_product_is_drawn_within_a_quarter_db_of_its_samples(tmp_path):
    lfs, pending = _issued(tmp_path, _forecast())
    head, received = _received(lfs)
    difference = _drawn(sim_reception.render(pending, head)).astype(int) - received
    assert np.abs(difference).max() <= 1, \
        "the issued product was not drawn within a quarter dB of its samples"


def test_the_pane_draws_the_issued_product_without_synthesising_it(tmp_path,
                                                                   monkeypatch):
    lfs, pending = _issued(tmp_path, _forecast(duration=40.0))
    head, _ = _received(lfs)

    def refused(*args, **kwargs):
        raise AssertionError("the pane synthesised the forecast again")

    monkeypatch.setattr(sounding, "write_lfs", refused)
    assert sim_reception.render(pending, head)["cells"], "the pane drew nothing"


def test_a_forecast_whose_product_is_gone_says_so(tmp_path):
    pending = kept(tmp_path)
    Path(pending["simulated"]["product"]).unlink()
    assert sim_reception.image(pending, HEAD)["status"] == "gone", \
        "a forecast whose product is gone did not say so"


def test_a_forecast_issued_before_products_were_kept_says_its_product_is_gone():
    assert sim_reception.image({"forecast": state()}, HEAD)["status"] == "gone", \
        "a forecast with no product did not say so"


def test_a_head_taller_than_the_page_is_folded_as_the_page_folds_it():
    rows = random.randrange(257, 4000)
    assert sim_reception.grid(dict(HEAD, rows=rows))["rows"] == 256, \
        "the head was not folded to the page's rows"


def test_a_range_row_lands_where_the_page_puts_the_received_sweeps():
    rows, row = 400, random.randrange(400)
    head = dict(HEAD, columns=10, rows=rows, range_step_km=1.0)
    db = np.full((10, rows), 30.0)
    db[:, row] = 60.0
    ion = SimpleNamespace(freq=10.0 + 0.05 * (np.arange(10) + 0.5),
                          vrange=2400.0 + np.arange(rows) * 1.0, db=db)
    assert int(np.argmax(sim_reception.cells(ion, head)[0])) == row * 256 // rows, \
        "the row did not land where the page puts the received sweep's"


def test_large_grid_is_bounded_without_changing_extent():
    head = dict(HEAD, columns=2033, rows=4001)
    grid = sim_reception.grid(head)
    assert grid["columns"] <= 800 and grid["rows"] <= 256
    for n, step in (("columns", "f_step_mhz"), ("rows", "range_step_km")):
        assert grid[n] * grid[step] == pytest.approx(head[n] * head[step])


def test_browsers_share_builds_and_cannot_grow_the_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(sim_reception, "_cache", collections.OrderedDict())
    monkeypatch.setattr(sim_reception, "_queue", collections.deque())
    monkeypatch.setattr(sim_reception, "_worker", object())
    assert sim_reception.image(kept(tmp_path), HEAD)["status"] == "building"
    assert sim_reception.image(kept(tmp_path), HEAD)["status"] == "building"
    assert len(sim_reception._queue) == 1
    for i in range(1, sim_reception.LIMIT):
        sim_reception.image(kept(tmp_path, epoch=T0 + i * 300), HEAD)
    assert sim_reception.image(kept(tmp_path, epoch=T0 + 9999), HEAD)["status"] \
        == "busy"
    assert len(sim_reception._queue) == sim_reception.LIMIT
    first = next(iter(sim_reception._cache))
    sim_reception._cache[first] = {"status": "ready", "cells": "AAAA"}
    assert sim_reception.image(kept(tmp_path), HEAD)["cells"] == "AAAA"


def test_real_synthesis_places_the_forecast_echo_on_live_axes(tmp_path):
    forecast = _forecast(duration=40.0,
                         modes=[Mode("flat", [10.5, 13.5], [9000, 9000], 45.0)])
    result = sim_reception.render(_issued(tmp_path, forecast)[1], HEAD)
    cells = np.frombuffer(base64.b64decode(result["cells"]), np.uint8).reshape(80, 64)
    peak = HEAD["range_lo_km"] + np.argmax(cells[20:60].mean(axis=0)) * 10
    assert peak == pytest.approx(2700, abs=20)
    assert cells[20:60].max() > cells[:5].max()


def test_a_synthetic_trace_is_as_thin_as_the_live_sweep_resolves(tmp_path):
    head = dict(HEAD, columns=120, rows=96, f_lo_mhz=10.0, f_step_mhz=0.04096,
                range_lo_km=2400.0, range_step_km=7.32421875)
    forecast = _forecast(modes=[Mode("flat", [10.4, 14.6], [9137, 9137], 47.0)])
    cells = np.frombuffer(base64.b64decode(
        sim_reception.render(_issued(tmp_path, forecast)[1], head)["cells"]),
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
                     json.dumps({"pending": [kept(tmp_path)]}), db.utcnow()))
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
