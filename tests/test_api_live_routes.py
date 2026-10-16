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


def queue(client, station, count):
    for _ in range(count):
        client.post("/stations/%s/commands" % station, json={"name": random.choice(["start", "stop"])}, headers=agent()).raise_for_status()


def test_live_reply_counts_the_commands_queued_for_its_station(client):
    count = random.randint(1, 5)
    queue(client, "YOS", count)
    assert client.post("/stations/YOS/live", json={"events": batch()}, headers=agent()).json()["queued"] == count, "the live reply did not count the commands queued for the station"


def test_live_reply_does_not_count_another_stations_commands(client):
    queue(client, "NOR", random.randint(1, 5))
    assert client.post("/stations/YOS/live", json={"events": batch()}, headers=agent()).json()["queued"] == 0, "the live reply counted commands queued for another station"


def test_live_reply_leaves_the_commands_it_counts_for_the_agent(client):
    count = random.randint(1, 5)
    queue(client, "YOS", count)
    client.post("/stations/YOS/live", json={"events": batch()}, headers=agent())
    assert len(client.get("/stations/YOS/commands", headers=agent()).json()["commands"]) == count, "the live reply took commands it was only meant to count"


def test_live_reply_does_not_count_commands_already_taken(client):
    queue(client, "YOS", random.randint(1, 5))
    client.get("/stations/YOS/commands", headers=agent())
    assert client.post("/stations/YOS/live", json={"events": batch()}, headers=agent()).json()["queued"] == 0, "the live reply counted commands the agent had already taken"


def test_empty_live_push_still_counts_the_queue(client):
    count = random.randint(1, 5)
    queue(client, "YOS", count)
    assert client.post("/stations/YOS/live", json={"events": []}, headers=agent()).json()["queued"] == count, "a live push with no events did not count the queue"


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


def test_live_page_offers_start_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" data-command="start">' in client.get("/ui/live?station=YOS").text, "the live page offers no Start gated on the station capability"


def test_live_page_offers_stop_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" data-command="stop">' in client.get("/ui/live?station=YOS").text, "the live page offers no Stop gated on the station capability"


def test_live_page_disables_its_controls_for_whoever_may_not(client):
    assert "gate('station')" in client.get("/ui/live?station=YOS").text, "the live page never gates its controls on the station capability"


def test_live_page_queues_its_commands_through_the_command_route(client):
    assert "control('/stations/' + encodeURIComponent(STATION) + '/commands', 'POST'" in client.get("/ui/live?station=YOS").text, "the live page does not queue its commands through POST /stations/{id}/commands"


def test_live_page_labels_start_in_russian(client):
    assert "Пуск</button>" in client.get("/ui/live?station=YOS&lang=ru").text, "the live page's Start is not labelled in Russian"


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


def test_live_page_says_a_stop_waits_for_the_sounding_in_progress(client):
    assert "T('live.js.stop_queued'" in client.get("/ui/live?station=YOS").text, "the live page does not say a queued stop waits for the sounding in progress"
