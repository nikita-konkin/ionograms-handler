"""The agent's live feed: a growing capture and the sounder's STATUS lines."""

from __future__ import annotations

import base64
import os
import random
import sys
import time

import numpy as np

from muf import calibrate, io_lfs, spectro
from services.agent import live
from services.agent.client import TransportError
from services.agent.config import StationConfig


def capture(make_lfs, tmp_path, windows, name="cyprus1_20260204_000010.lfs",
            day="2026.02.04", iq=None, **header):
    (tmp_path / day).mkdir(exist_ok=True)
    noise = np.random.normal(0, 1, (2, windows * 16384))
    return make_lfs(iq if iq is not None else noise[0] + 1j * noise[1],
                    name=day + "/" + name, **header)


def grow(path, windows):
    noise = np.random.normal(0, 1, (2, windows * 16384))
    with open(path, "ab") as fh:
        fh.write((noise[0] + 1j * noise[1]).astype(np.complex64).tobytes())


def cells(event):
    return np.frombuffer(base64.b64decode(event["db"]), np.uint8)


def test_capturing_line_carries_its_progress():
    progress = round(random.random(), 4)
    assert live.status_event("STATUS cyprus1 capturing %.4f 2" % progress)[
        "progress"] == progress, "a capturing line lost its progress"


def test_waiting_line_carries_the_end_of_its_window():
    t0 = random.randint(1792161000, 1792169999)
    assert live.status_event("STATUS magadan waiting %d %d" % (t0, t0 + 50))[
        "t1"] == t0 + 50, "a waiting line lost the end of its window"


def test_clean_line_carries_the_samples_received():
    got = random.randint(9000000, 10000000)
    assert live.status_event("STATUS cyprus1 clean 0 %d" % got)["got"] == got, (
        "a clean line lost the samples received")


def test_line_without_the_status_prefix_is_ignored():
    assert live.status_event("  sidecar cyprus1_20260204_000010.lfp 213 kB") is None, (
        "an ordinary log line was taken for a status")


def test_capture_names_its_transmitter_after_the_file(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 1, name="magadan_20261016_143000.lfs")
    assert live.Capture(path).head["tx"] == "magadan", (
        "the capture did not take its transmitter from the file name")


def test_capture_spans_the_whole_sweep(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 1)
    assert live.Capture(path).head["columns"] == 250 * 40000 // 16384, (
        "the capture does not span the sweep its header declares")


def test_capture_yields_only_whole_windows(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 2)
    with open(path, "ab") as fh:
        fh.write(np.zeros(random.randint(1, 16383), np.complex64).tobytes())
    assert len(live.Capture(path).columns()) == 2, (
        "the capture yielded a column from a partial window")


def test_capture_picks_up_windows_written_later(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 1)
    follow = live.Capture(path)
    follow.columns()
    more = random.randint(2, 5)
    grow(path, more)
    assert [e["index"] for e in follow.columns()] == list(range(1, 1 + more)), (
        "windows written later did not arrive in order")


def echo(offset, windows):
    n = np.arange(windows * 16384)
    noise = np.random.normal(0, 1, (2, n.size))
    return noise[0] + 1j * noise[1] + 40 * np.exp(-2j * np.pi * offset * n / 16384)


def peak(follow):
    return follow.head["range_lo_km"] + follow.head["range_step_km"] * int(
        np.argmax(cells(follow.columns()[1])))


def test_echo_lands_on_the_row_of_its_range(make_lfs, tmp_path):
    step = 120000.0 / 16384
    target = random.uniform(2700.0, 4000.0)
    found = peak(live.Capture(capture(make_lfs, tmp_path, 3,
                                      iq=echo(round(target / step), 3))))
    assert abs(found - round(target / step) * step) < step / 2, (
        "an echo at %.1f km came back at %.1f km"
        % (round(target / step) * step, found))


def test_live_column_agrees_with_the_archive_ionogram(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 4)
    gate = calibrate.geometry_gate(io_lfs.read_header(path))
    archived = np.clip(spectro.compute(path, window=16384, gate_km=gate).db[:, ::-1]
                       - spectro.NOISE_FLOOR_DB, 0, 63.75)
    sent = np.array([cells(e) * 0.25 for e in live.Capture(path).columns()])
    assert np.abs(sent - archived).max() <= 0.125 + 1e-4, (
        "the live columns differ from the archive's ionogram")


def magadan(make_lfs, tmp_path, windows, iq=None, **header):
    return capture(make_lfs, tmp_path, windows, name="magadan_20260204_000010.lfs",
                   iq=iq, **{"cf": 16500000, "dur": 50, "rate": 500000, "rmax": 5000,
                      "tx_latitude": 59.56, "tx_longitude": 150.80, **header})


def test_echo_on_a_long_fast_path_lands_within_one_row_of_its_range(make_lfs, tmp_path):
    step = 24000.0 / 16384
    target = round(random.uniform(5450.0, 6250.0) / step)
    follow = live.Capture(magadan(make_lfs, tmp_path, 3, iq=echo(target, 3)))
    found = peak(follow)
    assert abs(found - target * step) < follow.head["range_step_km"], (
        "an echo at %.1f km on a long fast path came back at %.1f km"
        % (target * step, found))


def test_no_capture_sends_more_rows_than_the_limit(make_lfs, tmp_path):
    follow = live.Capture(magadan(make_lfs, tmp_path, 2,
                                  rate=random.choice([100000, 250000, 500000]),
                                  tx_latitude=random.uniform(45.0, 70.0),
                                  tx_longitude=random.uniform(60.0, 170.0)))
    assert max(cells(e).size for e in follow.columns()) <= live.ROWS, (
        "a capture sent more rows than the limit")


def test_a_folded_column_carries_the_rows_its_sounding_declares(make_lfs, tmp_path):
    follow = live.Capture(magadan(make_lfs, tmp_path, 1))
    assert cells(follow.columns()[0]).size == follow.head["rows"], (
        "a folded column disagrees with its sounding about how many rows it has")


def test_a_capture_at_100_khz_per_second_is_not_folded(make_lfs, tmp_path):
    follow = live.Capture(capture(make_lfs, tmp_path, 1))
    assert abs(follow.head["range_step_km"] - 120000.0 / 16384) < 1e-9, (
        "a cyprus1 capture at 100 kHz/s was folded")


def test_a_capture_without_usable_coordinates_is_gated_as_recorded(make_lfs, tmp_path):
    follow = live.Capture(capture(make_lfs, tmp_path, 1, tx_latitude=56.38,
                                  tx_longitude=47.53))
    miss = abs(follow.head["range_lo_km"] - calibrate.FALLBACK_MIN_RANGE_KM)
    assert miss < follow.head["range_step_km"], (
        "a capture without a path was not gated as its header recorded")


def test_newest_capture_is_the_one_written_last(make_lfs, tmp_path):
    older = capture(make_lfs, tmp_path, 1, name="cyprus1_20260204_000010.lfs")
    newer = capture(make_lfs, tmp_path, 1, name="magadan_20260205_000500.lfs",
                    day="2026.02.05")
    os.utime(older, (time.time(), time.time()))
    os.utime(newer, (time.time() - 600, time.time() - 600))
    assert live.newest(tmp_path) == older, (
        "the newest capture was not the one written last")


def feed(tmp_path, post, clock, lines=()):
    config = StationConfig(station="YOS", live_dir=tmp_path)
    pending = list(lines)
    return live.Feed(config, lambda: [pending.pop(0)] if pending else [], post, clock)


def test_feed_sends_the_sounding_before_its_columns(make_lfs, tmp_path):
    capture(make_lfs, tmp_path, 3)
    sent = []
    feed(tmp_path, sent.append, lambda: 1770163210 + 20).step()
    assert sent[0][0]["type"] == "sounding", (
        "the feed sent columns ahead of their sounding")


def test_feed_does_not_replay_a_capture_that_ended_long_ago(make_lfs, tmp_path):
    capture(make_lfs, tmp_path, 3)
    sent = []
    feed(tmp_path, sent.append,
         lambda: 1770163210 + 250 + random.randint(120, 9000)).step()
    assert not any(e["type"] == "column" for batch in sent for e in batch), (
        "the feed replayed a capture that ended long ago")


def test_feed_forwards_status_lines(make_lfs, tmp_path):
    sent = []
    feed(tmp_path, sent.append, time.time, ["STATUS cyprus1 writing"]).step()
    assert sent[0][0] == {"type": "status", "tx": "cyprus1", "state": "writing"}, (
        "the feed did not forward a status line")


def test_failed_post_keeps_its_columns_for_the_next(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 2)
    sent = []
    outcomes = [TransportError("server away")]

    def post(events):
        if outcomes:
            raise outcomes.pop()
        sent.append(events)

    beat = feed(tmp_path, post, lambda: 1770163210 + 20)
    beat.step()
    grow(path, 1)
    beat.step()
    assert [e["index"] for e in sent[0] if e["type"] == "column"] == [0, 1, 2], (
        "columns from a failed post were lost")


def test_new_capture_drops_columns_of_the_old_one(make_lfs, tmp_path):
    old = capture(make_lfs, tmp_path, 2)
    outcomes = [TransportError("server away")]
    sent = []

    def post(events):
        if outcomes:
            raise outcomes.pop()
        sent.append(events)

    now = [1770163210 + 20]
    beat = feed(tmp_path, post, lambda: now[0])
    beat.step()
    capture(make_lfs, tmp_path, 1, name="cyprus1_20260204_000510.lfs",
            start_epoch=1770163510)
    os.utime(old, (time.time() - 60, time.time() - 60))
    now[0] = 1770163510 + 20
    beat.step()
    assert {e["t0"] for e in sent[0] if e["type"] == "column"} <= {1770163510}, (
        "columns of a finished capture were sent with the next one")


def test_feed_runs_a_command_pass_when_the_server_says_commands_wait(tmp_path):
    passes = []
    live.Feed(StationConfig(station="YOS", live_dir=tmp_path),
              lambda: ["STATUS cyprus1 writing"],
              lambda events: {"ok": True, "queued": random.randint(1, 9)}, time.time,
              lambda: passes.append(1)).step()
    assert passes == [1], "the feed ran no command pass while commands waited"


def test_feed_runs_no_command_pass_when_none_wait(tmp_path):
    passes = []
    live.Feed(StationConfig(station="YOS", live_dir=tmp_path),
              lambda: ["STATUS cyprus1 writing"],
              lambda events: {"ok": True, "queued": 0}, time.time,
              lambda: passes.append(1)).step()
    assert passes == [], "the feed ran a command pass with nothing queued"


def test_feed_runs_no_command_pass_when_its_post_fails(tmp_path):
    passes = []

    def post(events):
        raise TransportError("server away")

    live.Feed(StationConfig(station="YOS", live_dir=tmp_path),
              lambda: ["STATUS cyprus1 writing"],
              post, time.time, lambda: passes.append(1)).step()
    assert passes == [], "the feed ran a command pass on a post that never arrived"


def test_feed_asks_the_server_even_with_nothing_to_send(tmp_path):
    sent = []
    feed(tmp_path, sent.append, time.time).step()
    assert sent == [[]], (
        "the feed stayed silent with nothing to send, so a queued start could wait a "
        "minute")


def test_journal_lines_reach_the_feed():
    journal = live.Journal([sys.executable, "-c",
                            "print('STATUS cyprus1 writing', flush=True)"])
    got = []
    deadline = time.time() + 10
    while not got and time.time() < deadline:
        got = journal.lines()
        time.sleep(0.05)
    assert got == ["STATUS cyprus1 writing"], (
        "a line the follow command printed never reached the feed")


def test_feed_keeps_earlier_status_lines_ahead_of_the_sounding(make_lfs, tmp_path):
    capture(make_lfs, tmp_path, 2)
    sent = []
    feed(tmp_path, sent.append, lambda: 1770163210 + 20,
         ["STATUS cyprus1 waiting 1770163210 1770163460"]).step()
    assert [e["type"] for e in sent[0][:3]] == ["status", "sounding", "column"], (
        "the sounding was moved ahead of a status line read before it")


def test_batch_the_server_refused_is_not_sent_again(make_lfs, tmp_path):
    capture(make_lfs, tmp_path, 1)
    outcomes = [TransportError("HTTP 400 Bad Request", status=400)]
    sent = []

    def post(events):
        if outcomes:
            raise outcomes.pop()
        sent.append(events)

    beat = feed(tmp_path, post, lambda: 1770163210 + 20,
                ["STATUS cyprus1 capturing 0.0016 0",
                 "STATUS cyprus1 capturing 0.0040 0"])
    beat.step()
    beat.step()
    assert not any(e.get("progress") == 0.0016 for batch in sent for e in batch), (
        "a batch the server refused was sent again")


def test_columns_of_a_capture_renamed_away_keep_their_sounding(make_lfs, tmp_path):
    path = capture(make_lfs, tmp_path, 2)
    outcomes = [TransportError("server away")]
    sent = []

    def post(events):
        if outcomes:
            raise outcomes.pop()
        sent.append(events)

    beat = feed(tmp_path, post, lambda: 1770163210 + 20)
    beat.step()
    os.replace(path, str(path) + ".partial")
    beat.step()
    assert sent[0][0]["type"] == "sounding", (
        "columns of a capture renamed away were sent without their sounding")


def finished(make_lfs, tmp_path, iq=None):
    """A whole capture of 24 windows: ten seconds of sweep at 40 kHz."""
    return live.Capture(capture(make_lfs, tmp_path, 24, iq=iq, dur=10))


def reported(follow):
    return [e for e in follow.columns() if e["type"] == "echo"]


def test_a_finished_capture_says_where_its_echo_is(make_lfs, tmp_path):
    step = 120000.0 / 16384
    target = round(random.uniform(1200.0, 1900.0) / step)
    found = reported(finished(make_lfs, tmp_path, iq=echo(target, 24)))
    at = target * step
    assert found and found[0]["lo_km"] - step <= at <= found[0]["hi_km"] + step, (
        "a finished capture did not say its echo at %.0f km was there" % at)


def test_an_echo_outside_the_window_is_still_found(make_lfs, tmp_path):
    step = 120000.0 / 16384
    target = round(random.uniform(1200.0, 1900.0) / step)
    follow = finished(make_lfs, tmp_path, iq=echo(target, 24))
    found = reported(follow)
    assert found and found[0]["hi_km"] < follow.head["range_lo_km"], (
        "an echo below the window was not reported below it")


def test_a_capture_of_noise_says_no_echo_stands_out(make_lfs, tmp_path):
    assert reported(finished(make_lfs, tmp_path)) == [], (
        "a capture of noise reported an echo")


def test_an_unfinished_capture_says_nothing_of_its_echo_yet(make_lfs, tmp_path):
    step = 120000.0 / 16384
    follow = live.Capture(capture(make_lfs, tmp_path, 2,
                                  iq=echo(round(1500.0 / step), 2)))
    assert reported(follow) == [], (
        "a capture still being written reported its echo")
