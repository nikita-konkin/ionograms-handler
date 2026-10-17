"""The agent's backfill: summaries of soundings archived before anyone watched."""

from __future__ import annotations

import base64
import random
import time
from pathlib import Path

import numpy as np
import pytest

from services.agent import backfill, live
from services.agent.config import StationConfig

pytest.importorskip("h5py")


def product(make_chirp_h5, root, t0=None, tx="NIC1", freqs=None):
    t0 = t0 if t0 is not None else 1792108810.0 + 300 * random.randint(0, 999)
    path = make_chirp_h5(
        np.random.exponential(1.0, (freqs or random.randint(20, 60), 600)),
        t0=t0, txname=tx, station_name="yoshkar-ola")
    day = Path(root) / time.strftime("%Y.%m.%d", time.gmtime(t0))
    day.mkdir(parents=True, exist_ok=True)
    return path.rename(day / path.name)


def cells(text):
    return np.frombuffer(base64.b64decode(text), np.uint8)


def test_summary_names_the_transmitter_of_its_product(make_chirp_h5, tmp_path):
    tx = random.choice(["NIC1", "NIC2", "SGO", "Magadan"])
    path = product(make_chirp_h5, tmp_path / "a", tx=tx)
    assert backfill.summarise(path)["tx"] == tx, (
        "the summary does not name the transmitter of its product")


def test_summary_starts_when_its_product_starts(make_chirp_h5, tmp_path):
    t0 = 1792108810.0 + 300 * random.randint(0, 999)
    path = product(make_chirp_h5, tmp_path / "a", t0=t0)
    assert backfill.summarise(path)["t0"] == round(t0), (
        "the summary does not start when its product starts")


def test_summary_snr_has_one_cell_per_frequency(make_chirp_h5, tmp_path):
    n = random.randint(20, 60)
    summary = backfill.summarise(product(make_chirp_h5, tmp_path / "a", freqs=n))
    assert len(cells(summary["snr"])) == summary["columns"] == n, (
        "the summary SNR does not have one cell per frequency of the product")


def test_summary_pdp_has_one_cell_per_row(make_chirp_h5, tmp_path):
    summary = backfill.summarise(product(make_chirp_h5, tmp_path / "a"))
    assert len(cells(summary["pdp"])) == summary["rows"] <= live.ROWS, (
        "the summary PDP does not have one cell per folded range row")


def test_summary_is_accepted_by_the_server(make_chirp_h5, tmp_path):
    from services.api import live as hub
    kept = []
    hub.Hub(keep=lambda station, summary: kept.append(summary)).publish(
        "YOS", [backfill.summarise(product(make_chirp_h5, tmp_path / "a"))])
    assert len(kept) == 1, "the server did not take a summary the agent made"


def test_products_take_a_sounding_from_the_first_root_that_has_it(make_chirp_h5,
                                                                  tmp_path):
    t0 = 1792108810.0 + 300 * random.randint(0, 999)
    first = product(make_chirp_h5, tmp_path / "local", t0=t0)
    product(make_chirp_h5, tmp_path / "nas", t0=t0)
    found = backfill.products([tmp_path / "local", tmp_path / "nas"])
    assert list(found.values()) == [first], (
        "a sounding in two roots was not taken from the first one")


def test_products_skip_a_root_that_is_missing(make_chirp_h5, tmp_path):
    path = product(make_chirp_h5, tmp_path / "local")
    found = backfill.products([tmp_path / "gone", tmp_path / "local"])
    assert list(found.values()) == [path], "a missing root stopped the survey"


def test_wanted_takes_a_day_the_server_has_fewer_of():
    t0 = 1792195200 + 300 * random.randint(0, 200)
    local = {("NIC1", t0 + 300 * k): Path("p%d" % k) for k in range(3)}
    day = time.strftime("%Y-%m-%d", time.gmtime(t0))
    assert len(backfill.wanted(local, {"NIC1": {day: 2}})) == 3, (
        "a day the server holds fewer summaries of was not sent")


def test_wanted_skips_a_day_the_server_has_all_of():
    t0 = 1792195200 + 300 * random.randint(0, 200)
    local = {("NIC1", t0 + 300 * k): Path("p%d" % k) for k in range(3)}
    day = time.strftime("%Y-%m-%d", time.gmtime(t0))
    assert backfill.wanted(local, {"NIC1": {day: 3}}) == [], (
        "a day the server already holds was sent again")


def test_wanted_sends_the_newest_day_first():
    t0 = 1792195200
    local = {("NIC1", t0 - 86400 * k): Path("day%d" % k) for k in range(4)}
    assert backfill.wanted(local, {})[0] == Path("day0"), (
        "the newest day was not sent first")


def test_wanted_leaves_out_a_product_that_failed():
    t0 = 1792195200 + 300 * random.randint(0, 200)
    local = {("NIC1", t0): Path("broken")}
    assert backfill.wanted(local, {}, failed={Path("broken")}) == [], (
        "a product that could not be read was offered again")


def test_filling_skips_a_product_that_cannot_be_read(make_chirp_h5, tmp_path):
    good = product(make_chirp_h5, tmp_path / "a")
    broken = good.with_name(good.name.replace("NIC1", "NIC2"))
    broken.write_bytes(b"not an archive")
    worker = backfill.Backfill(StationConfig(live_dir=tmp_path / "a"), dict)
    worker.fill()
    assert [s["tx"] for s in worker.take(10)] == ["NIC1"], (
        "an unreadable product stopped the backfill or was sent")


def test_feed_posts_backfill_apart_from_the_live_batch(tmp_path):
    posts = []

    class Ready:
        def take(self, n):
            return [{"type": "summary", "tx": "NIC1"}]

    feed = live.Feed(StationConfig(live_dir=tmp_path), list, posts.append,
                     backfill=Ready())
    feed.step()
    assert posts == [[], [{"type": "summary", "tx": "NIC1"}]], (
        "the backfill did not go in a post of its own after the live batch")
