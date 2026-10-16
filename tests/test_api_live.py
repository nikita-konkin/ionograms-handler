"""The live hub: what a station is sounding now, held in memory and fanned out."""

from __future__ import annotations

import asyncio
import base64
import json
import random

import numpy as np
import pytest

from services.api import live


def sounding(tx="cyprus1", t0=None, columns=None, rows=None):
    return {
        "type": "sounding", "tx": tx,
        "t0": t0 if t0 is not None else 1792161010 + 300 * random.randint(0, 999),
        "f_lo_mhz": 7.5, "f_step_mhz": 0.04096,
        "columns": columns or random.randint(40, 610),
        "range_lo_km": 2550.0 + random.random() * 100, "range_step_km": 7.32,
        "rows": rows or random.randint(30, 330), "km_per_ms": 300.0,
    }


def column(head, index, cells=None):
    cells = cells if cells is not None else np.random.randint(0, 256, head["rows"], dtype=np.uint8)
    return {"type": "column", "tx": head["tx"], "t0": head["t0"], "index": index,
            "db": base64.b64encode(np.asarray(cells, dtype=np.uint8).tobytes()).decode("ascii")}


def status(tx, state, **fields):
    return dict({"type": "status", "tx": tx, "state": state}, **fields)


def frames(hub, station, act):
    async def collect():
        out = []
        stream = live.stream(hub, station, disconnected=lambda: asyncio.sleep(0, False), keepalive=5.0)
        out.append(await asyncio.wait_for(stream.__anext__(), 5.0))
        act()
        hub.close()
        async for frame in stream:
            out.append(frame)
        return out
    return asyncio.run(asyncio.wait_for(collect(), 10.0))


def test_column_reaches_a_browser_watching_its_station():
    hub = live.Hub()
    head = sounding()
    event = column(head, random.randrange(head["columns"]))
    got = frames(hub, "YOS", lambda: hub.publish("YOS", [head, event]))
    assert any(event["db"] in f and f.startswith("event: column") for f in got), "a browser watching the station never saw its column"


def test_stream_opens_with_a_snapshot():
    hub = live.Hub()
    got = frames(hub, "YOS", lambda: None)
    assert got[0].startswith("event: snapshot\n"), "the stream did not open with a snapshot"


def test_stations_do_not_see_each_other():
    hub = live.Hub()
    head = sounding()
    got = frames(hub, "YOS", lambda: hub.publish("DOB", [head, column(head, 0)]))
    assert not any(f.startswith("event: column") for f in got), "a browser saw a column from another station"


def test_snapshot_holds_the_columns_received_so_far():
    hub = live.Hub()
    head = sounding()
    picked = random.sample(range(head["columns"]), 3)
    hub.publish("YOS", [head] + [column(head, i) for i in picked])
    assert sorted(int(k) for k in hub.snapshot("YOS")["current"]["cyprus1"]["data"]) == sorted(picked), "the snapshot lost columns it had received"


def whole(head):
    return [head] + [column(head, i) for i in range(head["columns"])]


def test_sounding_with_every_column_becomes_the_last_one():
    hub = live.Hub()
    head = sounding(columns=random.randint(2, 9))
    hub.publish("YOS", whole(head))
    assert hub.snapshot("YOS")["last"]["cyprus1"]["t0"] == head["t0"], "a complete sounding did not become the last one"


def test_complete_sounding_leaves_nothing_current():
    hub = live.Hub()
    hub.publish("YOS", whole(sounding(columns=random.randint(2, 9))))
    assert "cyprus1" not in hub.snapshot("YOS")["current"], "a complete sounding stayed current"


def test_complete_sounding_adds_a_daily_summary():
    hub = live.Hub()
    hub.publish("YOS", whole(sounding(columns=random.randint(2, 9))))
    assert len(hub.snapshot("YOS")["daily"]["cyprus1"]) == 1, "a complete sounding added no daily summary"


def test_ending_status_alone_does_not_end_a_sounding():
    hub = live.Hub()
    head = sounding()
    hub.publish("YOS", [head, column(head, 0), status("cyprus1", random.choice(["writing", "clean", "degraded", "failed"]))])
    assert hub.snapshot("YOS")["current"]["cyprus1"]["t0"] == head["t0"], "a status line ended a sounding whose columns were still arriving"


def test_columns_after_an_ending_status_in_one_batch_are_taken():
    hub = live.Hub()
    head = sounding(columns=3)
    hub.publish("YOS", [head, column(head, 0), status("cyprus1", "writing"), column(head, 1), column(head, 2)])
    assert hub.snapshot("YOS")["last"]["cyprus1"]["t0"] == head["t0"], "columns sent after the writing status were lost"


def test_column_repeated_after_its_sounding_completed_is_ignored():
    hub = live.Hub()
    head = sounding(columns=2)
    hub.publish("YOS", whole(head) + [column(head, 1)])
    assert hub.snapshot("YOS")["current"] == {}, "a repeated column revived a sounding that had completed"


def test_summary_snr_is_the_strongest_cell_of_each_column():
    hub = live.Hub()
    head = sounding(columns=7)
    cells = np.random.randint(0, 256, (7, head["rows"]), dtype=np.uint8)
    hub.publish("YOS", [head] + [column(head, i, cells[i]) for i in range(7)])
    snr = np.frombuffer(base64.b64decode(hub.snapshot("YOS")["daily"]["cyprus1"][0]["snr"]), np.uint8)
    assert np.array_equal(snr, cells.max(axis=1)), "the summary SNR is not the strongest cell of each column"


def test_summary_pdp_averages_power_over_the_columns():
    hub = live.Hub()
    head = sounding(columns=5)
    cells = np.random.randint(0, 200, (5, head["rows"]), dtype=np.uint8)
    hub.publish("YOS", [head] + [column(head, i, cells[i]) for i in range(5)])
    pdp = np.frombuffer(base64.b64decode(hub.snapshot("YOS")["daily"]["cyprus1"][0]["pdp"]), np.uint8)
    expected = np.round(10 * np.log10(np.mean(10 ** (cells * 0.25 / 10), axis=0)) / 0.25)
    assert np.abs(pdp.astype(int) - expected).max() <= 1, "the summary PDP is not the mean power over the columns"


def test_new_sounding_closes_the_previous_one():
    hub = live.Hub()
    first = sounding(t0=1792161010)
    hub.publish("YOS", [first, column(first, 0), sounding(t0=1792161310)])
    assert hub.snapshot("YOS")["last"]["cyprus1"]["t0"] == 1792161010, "a new sounding did not close the one before it"


def test_repeated_sounding_event_keeps_its_columns():
    hub = live.Hub()
    head = sounding()
    hub.publish("YOS", [head, column(head, 0), column(head, 1)])
    hub.publish("YOS", [dict(head)])
    assert len(hub.snapshot("YOS")["current"]["cyprus1"]["data"]) == 2, "repeating the sounding event dropped its columns"


def test_column_of_an_unknown_sounding_is_refused():
    hub = live.Hub()
    with pytest.raises(live.Refused):
        hub.publish("YOS", [column(sounding(), 0)])


def test_column_of_the_wrong_length_is_refused():
    hub = live.Hub()
    head = sounding()
    with pytest.raises(live.Refused):
        hub.publish("YOS", [head, column(head, 0, np.zeros(head["rows"] + 1, np.uint8))])


def test_column_past_the_end_of_its_sounding_is_refused():
    hub = live.Hub()
    head = sounding()
    with pytest.raises(live.Refused):
        hub.publish("YOS", [head, column(head, head["columns"])])


def test_unknown_event_type_is_refused():
    hub = live.Hub()
    with pytest.raises(live.Refused):
        hub.publish("YOS", [{"type": "verdict", "tx": "cyprus1"}])


def test_refused_batch_changes_nothing():
    hub = live.Hub()
    head = sounding()
    try:
        hub.publish("YOS", [head, column(head, 0, np.zeros(1 + head["rows"], np.uint8))])
    except live.Refused:
        pass
    assert hub.snapshot("YOS")["current"] == {}, "a refused batch was half applied"


def test_status_keeps_only_the_fields_it_knows():
    hub = live.Hub()
    hub.publish("YOS", [status("cyprus1", "capturing", progress=0.37, overflows=0, script="<b>")])
    assert "script" not in hub.snapshot("YOS")["status"]["cyprus1"], "a status kept a field nobody defined"


def test_daily_summaries_stop_at_a_day():
    hub = live.Hub()
    for k in range(live.DAY + random.randint(1, 9)):
        hub.publish("YOS", whole(sounding(t0=1792161010 + 300 * k, columns=2, rows=3)))
    assert len(hub.snapshot("YOS")["daily"]["cyprus1"]) == live.DAY, "the daily summaries outgrew a day"


def test_slow_browser_is_cut_off():
    async def run():
        hub = live.Hub()
        reader = hub.subscribe("YOS")
        head = sounding(columns=live.BACKLOG + 5, rows=2)
        hub.publish("YOS", [head] + [column(head, i) for i in range(live.BACKLOG + 2)])
        return reader.open
    assert not asyncio.run(run()), "a browser that stopped reading was never cut off"


def test_snapshot_is_plain_json():
    hub = live.Hub()
    head = sounding()
    hub.publish("YOS", [head, column(head, 0), status("cyprus1", "clean"), sounding(t0=head["t0"] + 300)])
    assert json.loads(json.dumps(hub.snapshot("YOS")))["station"] == "YOS", "the snapshot does not survive JSON"


def test_sounding_resent_after_it_completed_is_not_counted_twice():
    hub = live.Hub()
    head = sounding(columns=random.randint(2, 6))
    hub.publish("YOS", whole(head))
    hub.publish("YOS", whole(dict(head)))
    assert len(hub.snapshot("YOS")["daily"]["cyprus1"]) == 1, "a sounding resent after it completed was summarised twice"
