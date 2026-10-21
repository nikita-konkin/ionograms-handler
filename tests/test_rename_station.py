"""Renaming a station's agent moves what the server keeps under that name.

Yoshkar-Ola has two receivers whose agents reported as `Yoshkar-Ola` and
`yoshkar-ola` (issue #27). They become `yoshkar-ola-1` and `yoshkar-ola-2`,
while the product files keep the `rx` they were written with -- so the rename
moves every row keyed by the agent's name, and records that the old `rx` is
the new station's own.
"""

from __future__ import annotations

import importlib.util
import random
import string
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.api import db  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rename_station", ROOT / "tools/rename_station.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules["rename_station"] = tool
_spec.loader.exec_module(tool)


def name():
    return "".join(random.choices(string.ascii_letters, k=random.randint(4, 9)))


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as c:
        yield c


def reported(conn, station):
    conn.execute("INSERT INTO health_report (station, received_at, document)"
                 " VALUES (?, ?, '{}')", (station, db.utcnow()))


def summarised(conn, station, tx, t0, source="live"):
    conn.execute("INSERT INTO live_summary (station, tx, t0, source, f_lo_mhz,"
                 " f_step_mhz, columns, range_lo_km, range_step_km, rows,"
                 " km_per_ms, snr, pdp) VALUES (?, ?, ?, ?, 7.5, 0.04, 1, 2500,"
                 " 7.3, 1, 300, 'AA==', 'AA==')", (station, tx, t0, source))


def identified(conn, station, code, sounder_id):
    conn.execute("INSERT INTO transmitter (station, code, sounder_id, timings,"
                 " verified_at) VALUES (?, ?, ?, '[]', ?)",
                 (station, code, sounder_id, db.utcnow()))


def count(conn, table, station):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE station = ?",
                        (station,)).fetchone()[0]


def test_health_reports_arrive_under_the_new_name(conn):
    old, new = name(), name()
    n = random.randint(2, 5)
    for _ in range(n):
        reported(conn, old)
    db.rename_station(conn, old, new)
    assert count(conn, "health_report", new) == n, (
        "health reports did not arrive under the new station name")


def test_live_summaries_arrive_under_the_new_name(conn):
    old, new = name(), name()
    summarised(conn, old, name(), random.randint(1, 10**9))
    db.rename_station(conn, old, new)
    assert count(conn, "live_summary", new) == 1, (
        "live summaries did not arrive under the new station name")


def test_identified_transmitters_arrive_under_the_new_name(conn):
    old, new = name(), name()
    identified(conn, old, name(), random.randint(1, 99))
    db.rename_station(conn, old, new)
    assert count(conn, "transmitter", new) == 1, (
        "identified transmitters did not arrive under the new station name")


@pytest.mark.parametrize("table", db.STATION_TABLES)
def test_nothing_stays_under_the_old_name(conn, table):
    old = name()
    reported(conn, old)
    conn.execute("INSERT INTO config_epoch (station, valid_from) VALUES (?, ?)",
                 (old, db.utcnow()))
    conn.execute("INSERT INTO sounder_block (station, received_at, block)"
                 " VALUES (?, ?, '{}')", (old, db.utcnow()))
    conn.execute("INSERT INTO sounder_literals (station, config_hash, literals,"
                 " first_seen) VALUES (?, 'h', '{}', ?)", (old, db.utcnow()))
    conn.execute("INSERT INTO command (id, station, name, issued_at)"
                 " VALUES (?, ?, 'stop', ?)", (name(), old, db.utcnow()))
    conn.execute("INSERT INTO station_preview (station, tx, received_at, image)"
                 " VALUES (?, 'cyprus1', ?, x'00')", (old, db.utcnow()))
    summarised(conn, old, "cyprus1", random.randint(1, 10**9))
    identified(conn, old, name(), random.randint(1, 99))
    db.rename_station(conn, old, name())
    assert count(conn, table, old) == 0, (
        "rows of %s stayed under the old station name" % table)


def test_the_old_receiver_name_belongs_to_the_new_station(conn):
    old, new = name(), name()
    reported(conn, old)
    db.rename_station(conn, old, new)
    assert old in db.receivers(conn, new), (
        "the new station does not own the receiver name its files carry")


def test_a_second_rename_carries_the_first_receiver_name_along(conn):
    first, second, third = name(), name(), name()
    reported(conn, first)
    db.rename_station(conn, first, second)
    db.rename_station(conn, second, third)
    assert first in db.receivers(conn, third), (
        "a second rename dropped the receiver name the first one recorded")


def test_a_station_never_renamed_owns_only_its_own_name(conn):
    station = name()
    assert db.receivers(conn, station) == (station,), (
        "a station never renamed does not own exactly its own name")


def test_a_summary_the_new_name_already_holds_wins_over_the_old_copy(conn):
    old, new, tx, t0 = name(), name(), name(), random.randint(1, 10**9)
    summarised(conn, old, tx, t0, source="archive")
    summarised(conn, new, tx, t0, source="live")
    db.rename_station(conn, old, new)
    assert conn.execute("SELECT source FROM live_summary WHERE station = ?",
                        (new,)).fetchone()[0] == "live", (
        "the old name's copy of a sounding replaced the one the new name holds")


def test_a_transmitter_both_names_identified_stops_the_rename(conn):
    old, new, code = name(), name(), name()
    identified(conn, old, code, 1)
    identified(conn, new, code, 2)
    with pytest.raises(ValueError):
        db.rename_station(conn, old, new)


def test_a_stopped_rename_writes_nothing(conn):
    old, new, code = name(), name(), name()
    reported(conn, old)
    identified(conn, old, code, 1)
    identified(conn, new, code, 2)
    with pytest.raises(ValueError):
        db.rename_station(conn, old, new)
    assert count(conn, "health_report", old) == 1, (
        "a rename that was stopped still moved rows")


def test_a_name_with_nothing_under_it_is_refused(conn):
    with pytest.raises(ValueError):
        db.rename_station(conn, name(), name())


def test_a_station_cannot_be_renamed_to_itself(conn):
    station = name()
    reported(conn, station)
    with pytest.raises(ValueError):
        db.rename_station(conn, station, station)


def test_the_tool_writes_nothing_without_apply(tmp_path):
    path, old = tmp_path / "t.sqlite3", name()
    with db.session(path) as c:
        reported(c, old)
        c.commit()
    tool.main(["--db", str(path), "--from", old, "--to", name()])
    with db.session(path) as c:
        assert count(c, "health_report", old) == 1, (
            "the tool renamed a station without --apply")


def test_the_tool_renames_with_apply(tmp_path):
    path, old, new = tmp_path / "t.sqlite3", name(), name()
    with db.session(path) as c:
        reported(c, old)
        c.commit()
    tool.main(["--db", str(path), "--from", old, "--to", new, "--apply"])
    with db.session(path) as c:
        assert count(c, "health_report", new) == 1, (
            "the tool did not rename the station with --apply")


def test_the_tool_keeps_a_copy_of_the_database_before_applying(tmp_path):
    path, old = tmp_path / "t.sqlite3", name()
    with db.session(path) as c:
        reported(c, old)
        c.commit()
    tool.main(["--db", str(path), "--from", old, "--to", name(), "--apply"])
    assert len(list(tmp_path.glob("t.sqlite3.bak-*"))) == 1, (
        "the tool applied a rename without keeping a copy first")
