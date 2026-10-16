"""The live routes: the agent posts batches, browsers read a snapshot or a stream."""

from __future__ import annotations

import base64
import random

import numpy as np
import pytest

pytest.importorskip("fastapi")


def batch(rows=None, columns=None):
    rows = rows or random.randint(20, 300)
    head = {"type": "sounding", "tx": "cyprus1", "t0": 1792161010 + 300 * random.randint(0, 99),
            "f_lo_mhz": 7.5, "f_step_mhz": 0.04096, "columns": columns or random.randint(10, 610),
            "range_lo_km": 2551.3, "range_step_km": 7.32, "rows": rows, "km_per_ms": 300.0}
    cells = np.random.randint(0, 256, rows, dtype=np.uint8)
    return [head, {"type": "column", "tx": "cyprus1", "t0": head["t0"], "index": 0,
                   "db": base64.b64encode(cells.tobytes()).decode("ascii")}]


def agent():
    return {"Authorization": "Bearer ctl"}


def test_agent_token_may_push_live_events(client):
    assert client.post("/stations/YOS/live", json={"events": batch()}, headers=agent()).status_code == 200, "the agent token could not push live events"


def test_live_push_without_a_token_is_refused(client):
    assert client.post("/stations/YOS/live", json={"events": batch()}).status_code == 401, "an anonymous caller pushed live events"


def test_malformed_batch_is_refused_as_a_bad_request(client):
    events = batch()
    events[1]["index"] = events[0]["columns"] + random.randint(0, 9)
    assert client.post("/stations/YOS/live", json={"events": events}, headers=agent()).status_code == 400, "a column past its sounding was accepted"


def test_batch_without_an_event_list_is_refused(client):
    assert client.post("/stations/YOS/live", json={"events": "column"}, headers=agent()).status_code == 400, "a batch without a list of events was accepted"


def test_snapshot_shows_a_pushed_column(client):
    events = batch()
    client.post("/stations/YOS/live", json={"events": events}, headers=agent())
    assert client.get("/stations/YOS/live").json()["current"]["cyprus1"]["data"]["0"] == events[1]["db"], "the snapshot did not show the column just pushed"


def test_event_stream_opens_with_a_snapshot_of_the_station(client):
    client.post("/stations/YOS/live", json={"events": batch()}, headers=agent())
    client.app.state.live.close()
    assert client.get("/stations/YOS/live/events").text.startswith('event: snapshot\ndata: {"station":"YOS"'), "the event stream did not open with the station's snapshot"


def test_event_stream_is_served_as_server_sent_events(client):
    client.app.state.live.close()
    assert client.get("/stations/YOS/live/events").headers["content-type"].startswith("text/event-stream"), "the event stream is not text/event-stream"


def test_live_page_names_its_station(client):
    assert 'data-station="YOS"' in client.get("/ui/live?station=YOS").text, "the live page does not name the station it watches"


def test_live_page_defaults_to_a_station_that_is_live(client):
    client.post("/stations/NOR/live", json={"events": batch()}, headers=agent())
    assert 'data-station="NOR"' in client.get("/ui/live").text, "the live page did not pick the station that is live"


def test_what_the_agent_sends_the_server_accepts(client, make_lfs, tmp_path):
    from services.agent import live as feed
    from services.agent.config import StationConfig

    (tmp_path / "2026.02.04").mkdir()
    windows = random.randint(2, 6)
    noise = np.random.normal(0, 1, (2, windows * 16384))
    make_lfs(noise[0] + 1j * noise[1], name="2026.02.04/cyprus1_20260204_000010.lfs")

    def post(events):
        client.post("/stations/YOS/live", json={"events": events}, headers=agent()).raise_for_status()

    feed.Feed(StationConfig(station="YOS", live_dir=tmp_path), lambda: ["STATUS cyprus1 capturing 0.0100 0"],
              post, lambda: 1770163210 + 30).step()
    assert len(client.get("/stations/YOS/live").json()["current"]["cyprus1"]["data"]) == windows, "the server did not take every column the agent sent"
