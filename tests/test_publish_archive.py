"""The public archive: past soundings as echo windows, byte for byte (#151).

A window leaves the hub unchanged, named for its sounding's time and never
rewritten. Only circuits on the archive list and not muted appear, a pass
copies a bounded batch, and a bad file costs only itself.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import make_header_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muf import echo_window
from services.api import db
from services.publish import archive as public
from services.publish import publisher

DAY = "archive/cyprus1-yoshkar-ola/2026/02/04"


def _window(path, *, lo=2486.5, hi=4112.25, edge=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("echo_window")
        group.attrs["range_lo_km"] = lo
        group.attrs["range_hi_km"] = hi
        if edge is not None:
            group.attrs["trace_at_edge"] = np.bool_(edge)
        group.create_dataset("z", data=np.arange(37, dtype=np.complex64))
    return path


def _row(conn, root, path, when, *, tx="cyprus1", rx="yoshkar-ola",
         kind="echo_window"):
    conn.execute(
        "INSERT INTO sounding(file,path,format,datetime,tx,rx,ingested_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (path.name, path.relative_to(root).as_posix(), kind, when, tx, rx,
         db.utcnow()))
    conn.commit()


@pytest.fixture
def hub(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    with db.session(tmp_path / "api.sqlite3") as conn:
        yield conn, root, tmp_path / "site"


def _capture(hub, stamp="20260204_000010", when="2026-02-04 00:00:10", **kw):
    conn, root, _ = hub
    row = {k: kw.pop(k) for k in ("tx", "rx") if k in kw}
    path = _window(root / "2026.02.04" / f"cyprus1_{stamp}.echo.h5", **kw)
    _row(conn, root, path, when, **row)
    return path


def _pass(hub, circuits="cyprus1->yoshkar-ola", **kw):
    conn, root, site = hub
    return public.write_archive(conn, site,
                                circuits=publisher.parse_circuits(circuits),
                                archive_root=root, **kw)


def _index(site, folder=DAY):
    return json.loads((site / folder / "index.json").read_text(encoding="utf-8"))


def test_a_window_is_published_byte_for_byte(hub):
    source = _capture(hub)
    _pass(hub)
    assert (hub[2] / DAY / "20260204T000010.echo.h5").read_bytes() == \
        source.read_bytes(), "the published window differs from the hub's"


def test_the_day_index_describes_the_window(hub):
    source = _capture(hub, lo=2611.75, hi=3907.5, edge=True)
    _pass(hub)
    assert _index(hub[2])["soundings"] == [{
        "t": "2026-02-04T00:00:10Z", "file": "20260204T000010.echo.h5",
        "bytes": source.stat().st_size, "range_km": [2611.75, 3907.5],
        "trace_at_edge": True,
    }], "the day index does not describe the window"


def test_a_window_cut_from_a_capture_is_described(hub):
    conn, root, site = hub
    rng = np.random.default_rng(11)
    samples = (rng.normal(size=200000) + 1j * rng.normal(size=200000)
               + 30 * np.exp(-2j * np.pi * (2900 / 3) * np.arange(200000) / 40000))
    capture = root / "synth_20260204_000010.lfs"
    capture.write_bytes(make_header_bytes()
                        + samples.astype(np.complex64).tobytes())
    target = echo_window.write(capture).target
    _row(conn, root, target, "2026-02-04 00:00:10")
    _pass(hub)
    assert _index(site)["soundings"][0]["bytes"] == target.stat().st_size, (
        "a window cut from a capture was not described")


def test_a_window_without_the_edge_test_says_so(hub):
    _capture(hub)
    _pass(hub)
    assert _index(hub[2])["soundings"][0]["trace_at_edge"] is None, \
        "a window without the edge test does not say so"


def test_a_sounding_without_a_window_is_listed_without_a_file(hub):
    conn, root, site = hub
    _capture(hub)
    raw = root / "2026.02.04" / "cyprus1_20260204_000510.lfs"
    raw.write_bytes(b"raw")
    _row(conn, root, raw, "2026-02-04 00:05:10", kind="lfs")
    _pass(hub)
    assert _index(site)["soundings"][1] == {
        "t": "2026-02-04T00:05:10Z", "file": None, "bytes": None,
        "range_km": None, "trace_at_edge": None,
    }, "a sounding without a window is not listed as such"


def test_a_raw_capture_is_never_published(hub):
    conn, root, site = hub
    _capture(hub)
    raw = root / "2026.02.04" / "cyprus1_20260204_000510.lfs"
    raw.write_bytes(b"raw")
    _row(conn, root, raw, "2026-02-04 00:05:10", kind="lfs")
    _pass(hub)
    assert sorted(p.name for p in (site / DAY).iterdir()) == [
        "20260204T000010.echo.h5", "index.json",
    ], "something other than the window and its index was published"


def test_a_day_without_any_window_is_not_published(hub):
    conn, root, site = hub
    raw = root / "cyprus1_20260122_101500.lfs"
    raw.write_bytes(b"raw")
    _row(conn, root, raw, "2026-01-22 10:15:00", kind="lfs")
    _pass(hub)
    assert not (site / "archive").exists(), \
        "a day without any window was published"


def test_a_circuit_off_the_archive_list_is_not_published(hub):
    _capture(hub, tx="NIC7")
    _pass(hub)
    assert not (hub[2] / "archive").exists(), \
        "a circuit off the archive list was published"


def test_a_muted_circuit_is_not_published(hub):
    _capture(hub)
    db.mute_circuit(hub[0], "cyprus1", "yoshkar-ola")
    _pass(hub)
    assert not (hub[2] / "archive").exists(), "a muted circuit was published"


def test_two_spellings_of_a_circuit_share_one_folder(hub):
    _capture(hub)
    _capture(hub, "20260204_000510", "2026-02-04 00:05:10", tx="Cyprus1",
             rx="Yoshkar-Ola")
    _pass(hub, circuits="CYPRUS1->yoshkar-OLA")
    assert [e["file"] for e in _index(hub[2])["soundings"]] == [
        "20260204T000010.echo.h5", "20260204T000510.echo.h5",
    ], "two spellings of a circuit did not share one folder"


def test_the_catalogue_counts_each_day(hub):
    first = _capture(hub)
    second = _capture(hub, "20260205_231500", "2026-02-05 23:15:00.250000")
    _pass(hub)
    assert json.loads((hub[2] / "archive" / "index.json").read_text(
        encoding="utf-8")) == {"circuits": [{
            "slug": "cyprus1-yoshkar-ola", "tx": "cyprus1", "rx": "yoshkar-ola",
            "tx_name": publisher.station_name("cyprus1"),
            "rx_name": publisher.station_name("yoshkar-ola"),
            "days": [
                {"day": "2026-02-04", "soundings": 1, "windows": 1,
                 "bytes": first.stat().st_size},
                {"day": "2026-02-05", "soundings": 1, "windows": 1,
                 "bytes": second.stat().st_size},
            ],
        }]}, "the catalogue does not count each day"


def test_a_published_window_is_never_rewritten(hub):
    source = _capture(hub)
    before = source.read_bytes()
    _pass(hub)
    _window(source, lo=1.5, hi=2.5)
    _pass(hub)
    assert (hub[2] / DAY / "20260204T000010.echo.h5").read_bytes() == before, \
        "a published window was rewritten"


def test_a_published_window_outlives_a_lost_source(hub):
    source = _capture(hub, edge=False)
    _pass(hub)
    source.unlink()
    _pass(hub)
    assert _index(hub[2])["soundings"][0]["trace_at_edge"] is False, \
        "a published window did not outlive its lost source"


def test_a_lost_window_costs_only_itself(hub):
    _capture(hub).unlink()
    _capture(hub, "20260204_000510", "2026-02-04 00:05:10")
    _pass(hub)
    assert [e["file"] for e in _index(hub[2])["soundings"]] == [
        None, "20260204T000510.echo.h5",
    ], "a lost window cost more than itself"


def test_a_lost_window_is_reported(hub):
    _capture(hub).unlink()
    assert "20260204T000010.echo.h5" in _pass(hub)["errors"][0], \
        "a lost window was not reported"


def test_a_file_that_is_not_a_window_is_not_published(hub):
    _capture(hub).write_bytes(os.urandom(211))
    _pass(hub)
    assert not (hub[2] / "archive").exists(), \
        "a file that is not a window was published"


def test_a_pass_copies_no_more_than_its_batch(hub):
    for minute in (0, 5, 10, 15, 20):
        _capture(hub, f"20260204_00{minute:02d}10",
                 f"2026-02-04 00:{minute:02d}:10")
    _pass(hub, limit=2)
    assert len(list((hub[2] / DAY).glob("*.echo.h5"))) == 2, \
        "a pass did not stop at its batch"


def test_a_pass_reports_what_it_held_back(hub):
    for minute in (0, 5, 10, 15, 20):
        _capture(hub, f"20260204_00{minute:02d}10",
                 f"2026-02-04 00:{minute:02d}:10")
    assert _pass(hub, limit=2)["held"] == 3, \
        "a pass did not report what it held back"


def test_the_held_windows_follow_on_later_passes(hub):
    for minute in (0, 5, 10, 15, 20):
        _capture(hub, f"20260204_00{minute:02d}10",
                 f"2026-02-04 00:{minute:02d}:10")
    for _ in range(3):
        _pass(hub, limit=2)
    assert len(list((hub[2] / DAY).glob("*.echo.h5"))) == 5, \
        "the held windows did not follow on later passes"


def test_a_circuit_that_leaves_the_list_is_pruned(hub):
    _capture(hub)
    _pass(hub)
    _pass(hub, circuits="NIC7->yoshkar-ola")
    assert not (hub[2] / "archive").exists(), \
        "a circuit that left the list was not pruned"


def test_no_temporary_files_are_left_behind(hub):
    _capture(hub)
    _capture(hub, "20260204_000510", "2026-02-04 00:05:10").write_bytes(b"junk")
    _pass(hub)
    assert not list(hub[2].rglob(f"{publisher.TMP_PREFIX}*")), \
        "a temporary file was left behind"


def test_the_full_pass_publishes_the_archive(hub):
    conn, root, site = hub
    _capture(hub)
    publisher.build_site(conn, site, circuits=[("nic9", "dob")],
                         method="contour", days=30, archive_root=root,
                         archive_circuits=[("cyprus1", "yoshkar-ola")])
    assert (site / DAY / "20260204T000010.echo.h5").is_file(), \
        "the full pass did not publish the archive"


def test_the_log_line_counts_the_archive(hub):
    conn, root, site = hub
    for minute in (0, 5, 10):
        _capture(hub, f"20260204_00{minute:02d}10",
                 f"2026-02-04 00:{minute:02d}:10")
    result = publisher.build_site(conn, site, circuits=[("nic9", "dob")],
                                  method="contour", days=30, archive_root=root,
                                  archive_circuits=[("cyprus1", "yoshkar-ola")])
    assert "3 windows archived" in publisher.describe(result), \
        "the log line does not count the archive"


def test_main_reads_the_archive_list(hub, tmp_path, monkeypatch):
    conn, root, site = hub
    _capture(hub)
    conn.close()
    monkeypatch.setenv("PUBLISH_ARCHIVE_CIRCUITS", "Cyprus1->Yoshkar-Ola")
    publisher.main(["--db", str(tmp_path / "api.sqlite3"), "--out", str(site),
                    "--circuits", "NIC9->DOB", "--archive-root", str(root),
                    "--api", ""])
    assert (site / DAY / "20260204T000010.echo.h5").is_file(), \
        "main did not read the archive list"


def test_the_vhost_keeps_a_window_for_good():
    vhost = (Path(__file__).resolve().parents[1] / "deploy"
             / "public-site.nginx.conf").read_text(encoding="utf-8")
    assert re.search(r"location /archive/ \{[^}]*immutable", vhost), (
        "the vhost does not keep a window for good")
