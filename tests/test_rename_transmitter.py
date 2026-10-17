"""Renaming an ionozond transmitter: the schedule, every copy of its products, the rows.

`tools/rename_transmitter.py` exists because the Cyprus groups on the
Yoshkar-Ola laptop were named after slots they do not record, and their names
resolve to no coordinates. The name is written into a product five ways, and a
rename that misses one leaves files that disagree with each other, or a
database that re-ingests them under the old name.
"""

from __future__ import annotations

import importlib.util
import os
import random
import struct
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.api import db  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rename_transmitter", ROOT / "tools/rename_transmitter.py")
rename = importlib.util.module_from_spec(_spec)
sys.modules["rename_transmitter"] = rename
_spec.loader.exec_module(rename)

#: A start time whose phase in the 300 s cycle is 10, and one whose phase is 0.
AT_10, AT_0 = 1790783410, 1790783400

SCHEDULE = (b"[NIC1_235]\nactive=false\nchirpt=0\nrep=300\n\n"
            b"[NIC4_280]\nactive=true\nchirpt=10\nrep=600\n\n"
            b"[Sodankyl%U00E4]\nactive=false\nchirpt=121.0015\nrep=300\n\n"
            b"[yoshkar-ola]\nactive=false\nchirpt=0\nrx=true\n")


def rules(*texts):
    return [rename.rule(t) for t in texts]


def stamp(t0):
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime(t0))


def lfs_header(name: str) -> bytes:
    return (struct.pack("<4sf4sH64s", b"LFSG", 1.0, b"fmt ", 498,
                        name.encode("latin-1")).ljust(512, b"\0"))


def capture(folder, name, t0, ext=".lfs"):
    path = folder / ("%s_%s%s" % (name, stamp(t0), ext))
    if ext == ".lfs":
        path.write_bytes(lfs_header(name) + os.urandom(64))
    else:
        head = bytearray(0x100)
        head[0:4] = b"LFPR"
        head[0x30:0x30 + len(name.encode("utf-8"))] = name.encode("utf-8")
        path.write_bytes(bytes(head) + os.urandom(32))
    old = time.time() - 7200
    os.utime(path, (old, old))
    return path


def archive(folder, name, t0, echo=False):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = folder / ("lfm_ionogram-%s-yoshkar-ola-ch0-000-%d.00.h5" % (name, t0))
    with h5py.File(path, "w") as fh:
        fh["txname"] = name
        fh["station_name"] = "yoshkar-ola"
        fh["t0"] = float(t0)
        if echo:
            group = fh.create_group("echo_window")
            group.attrs["tx_name"] = name
            group.attrs["source_name"] = "%s_%s.lfs" % (name, stamp(t0))
            group.create_dataset("lfs_header", data=np.frombuffer(
                lfs_header(name), dtype=np.uint8))
    old = time.time() - 7200
    os.utime(path, (old, old))
    return path


def moved(roots, *texts, apply=True, settle=600.0):
    return rename.files([Path(r) for r in roots], rules(*texts), apply=apply,
                        settle=settle)


def h5(path, key):
    import h5py
    with h5py.File(path, "r") as fh:
        value = fh[key][()]
    return value.decode("utf-8") if isinstance(value, bytes) else value


def test_a_rule_without_a_phase_takes_every_start_time():
    assert rename.rule("Sodankylä=SGO").phase is None, (
        "a rule without a phase was given one")


def test_a_rule_reads_its_phase():
    assert rename.rule("NIC4_280@10=NIC1").phase == 10, "a rule lost its phase"


@pytest.mark.parametrize("text", ["NIC4_280", "NIC4_280=NIC-1", "=NIC1",
                                  "NIC4_280@x=NIC1", "NIC4_280@300=NIC1"])
def test_a_malformed_rule_is_refused(text):
    with pytest.raises(SystemExit):
        rename.rule(text)


def test_a_rule_matches_the_name_as_ingest_garbles_it():
    assert rename.rule("Sodankyl\u00e4=SGO").matches("Sodankyl\ufffd\ufffd", AT_0), (
        "the rule did not match the name as the server's ingest stored it")


def test_a_rule_leaves_a_start_time_at_another_phase():
    assert not rename.rule("NIC4_280@10=NIC1").matches("NIC4_280", AT_0), (
        "a rule matched a product at another phase")


def test_the_schedule_rename_changes_only_the_group_header():
    text, _ = rename.schedule(SCHEDULE, rules("NIC4_280@10=NIC1"), [])
    assert text == SCHEDULE.replace(b"[NIC4_280]", b"[NIC1]"), (
        "the schedule rename changed more than the group header")


def test_the_schedule_rename_finds_a_qt_escaped_group_by_its_name():
    text, _ = rename.schedule(SCHEDULE, rules("Sodankyl\u00e4=SGO"), [])
    assert b"[SGO]\nactive=false\nchirpt=121.0015" in text, (
        "the Qt-escaped group was not found by its name")


def test_the_schedule_rename_finds_a_raw_utf8_group_by_its_name():
    text, _ = rename.schedule(SCHEDULE.replace(b"%U00E4", "\u00e4".encode()),
                              rules("Sodankyl\u00e4=SGO"), [])
    assert b"[SGO]\nactive=false\nchirpt=121.0015" in text, (
        "the raw UTF-8 group was not found by its name")


def test_the_schedule_removes_a_group_with_its_keys():
    text, _ = rename.schedule(SCHEDULE, [], ["NIC1_235"])
    assert text == SCHEDULE.replace(
        b"[NIC1_235]\nactive=false\nchirpt=0\nrep=300\n\n", b""), (
        "the group was not removed whole")


def test_the_schedule_refuses_a_group_whose_timing_the_rule_does_not_describe():
    with pytest.raises(SystemExit, match="NIC4_280"):
        rename.schedule(SCHEDULE, rules("NIC4_280@15=NIC1"), [])


def test_the_schedule_refuses_a_name_that_is_already_a_group():
    with pytest.raises(SystemExit, match="NIC1_235"):
        rename.schedule(SCHEDULE, rules("NIC4_280@10=NIC1_235"), [])


def test_the_schedule_refuses_a_group_it_cannot_find():
    with pytest.raises(SystemExit, match="NIC_900"):
        rename.schedule(SCHEDULE, rules("NIC_900@15=NIC2"), [])


def test_an_archive_at_the_rules_phase_is_renamed(tmp_path):
    archive(tmp_path, "NIC4_280", AT_10)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert (tmp_path / ("lfm_ionogram-NIC1-yoshkar-ola-ch0-000-%d.00.h5" % AT_10)
            ).exists(), "the archive at the rule's phase was not renamed"


def test_a_renamed_archive_carries_the_new_name_inside(tmp_path):
    archive(tmp_path, "NIC4_280", AT_10)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert h5(tmp_path / ("lfm_ionogram-NIC1-yoshkar-ola-ch0-000-%d.00.h5" % AT_10),
              "txname") == "NIC1", "the renamed archive kept the old txname"


def test_an_archive_at_another_phase_keeps_its_name(tmp_path):
    path = archive(tmp_path, "NIC4_280", AT_0)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert path.exists(), "an archive at another phase was renamed"


def test_the_echo_window_names_the_new_transmitter(tmp_path):
    import h5py
    archive(tmp_path, "NIC4_280", AT_10, echo=True)
    moved([tmp_path], "NIC4_280@10=NIC1")
    with h5py.File(tmp_path / ("lfm_ionogram-NIC1-yoshkar-ola-ch0-000-%d.00.h5"
                               % AT_10), "r") as fh:
        source = fh["echo_window"].attrs["source_name"]
    assert source == "NIC1_%s.lfs" % stamp(AT_10), (
        "the echo window still names the old capture")


def test_the_echo_windows_copy_of_the_capture_header_is_rewritten(tmp_path):
    import h5py
    archive(tmp_path, "NIC4_280", AT_10, echo=True)
    moved([tmp_path], "NIC4_280@10=NIC1")
    with h5py.File(tmp_path / ("lfm_ionogram-NIC1-yoshkar-ola-ch0-000-%d.00.h5"
                               % AT_10), "r") as fh:
        header = bytes(fh["echo_window/lfs_header"][()])
    assert header == lfs_header("NIC1"), "the stored capture header kept the old name"


def test_a_capture_is_renamed_with_its_header(tmp_path):
    capture(tmp_path, "NIC4_280", AT_10)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert (tmp_path / ("NIC1_%s.lfs" % stamp(AT_10))).read_bytes()[:512] == (
        lfs_header("NIC1")), "the renamed capture kept the old name in its header"


def test_a_sidecar_is_renamed_with_its_header(tmp_path):
    capture(tmp_path, "NIC4_280", AT_10, ext=".lfp")
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert (tmp_path / ("NIC1_%s.lfp" % stamp(AT_10))).read_bytes()[0x30:0x35] == (
        b"NIC1\0"), "the renamed sidecar kept the old name in its header"


def test_products_in_day_folders_are_found(tmp_path):
    day = tmp_path / "2026.09.30"
    day.mkdir()
    capture(day, "NIC4_280", AT_10, ext=".lfp")
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert (day / ("NIC1_%s.lfp" % stamp(AT_10))).exists(), (
        "a product in a day folder was not renamed")


def test_a_non_ascii_archive_is_renamed(tmp_path):
    archive(tmp_path, "Sodankyl\u00e4", AT_0)
    moved([tmp_path], "Sodankyl\u00e4=SGO")
    assert (tmp_path / ("lfm_ionogram-SGO-yoshkar-ola-ch0-000-%d.00.h5" % AT_0)
            ).exists(), "the non-ASCII archive was not renamed"


def test_a_non_ascii_capture_header_is_rewritten(tmp_path):
    capture(tmp_path, "Sodankyl\u00e4", AT_0)
    moved([tmp_path], "Sodankyl\u00e4=SGO")
    assert (tmp_path / ("SGO_%s.lfs" % stamp(AT_0))).read_bytes()[:512] == (
        lfs_header("SGO")), "the non-ASCII capture kept its old header"


def test_a_product_still_being_written_is_left_alone(tmp_path):
    path = capture(tmp_path, "NIC4_280", AT_10)
    os.utime(path, None)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert path.exists(), "a product still being written was renamed"


def test_a_header_naming_another_transmitter_is_left_alone(tmp_path):
    path = capture(tmp_path, "NIC4_280", AT_10)
    path.write_bytes(lfs_header("Magadan") + os.urandom(64))
    os.utime(path, (time.time() - 7200,) * 2)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert path.exists(), "a capture whose header names another transmitter was renamed"


def test_a_product_whose_new_name_is_taken_is_left_alone(tmp_path):
    path = capture(tmp_path, "NIC4_280", AT_10)
    capture(tmp_path, "NIC1", AT_10)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert path.exists(), "a product was renamed over one already there"


def test_a_rewritten_product_left_under_its_old_name_is_finished(tmp_path):
    path = capture(tmp_path, "NIC4_280", AT_10)
    path.write_bytes(lfs_header("NIC1") + os.urandom(64))
    os.utime(path, (time.time() - 7200,) * 2)
    moved([tmp_path], "NIC4_280@10=NIC1")
    assert (tmp_path / ("NIC1_%s.lfs" % stamp(AT_10))).exists(), (
        "a product rewritten before an interruption was not renamed")


def test_a_dry_run_moves_nothing(tmp_path):
    path = archive(tmp_path, "NIC4_280", AT_10)
    before = path.read_bytes()
    moved([tmp_path], "NIC4_280@10=NIC1", apply=False)
    assert path.read_bytes() == before, "a dry run changed a product"


def test_the_survey_counts_what_it_would_move(tmp_path):
    for n in range(random.randint(2, 6)):
        capture(tmp_path, "NIC4_280", AT_10 + 600 * n, ext=".lfp")
    found = moved([tmp_path], "NIC4_280@10=NIC1", apply=False)
    assert len(found.moves) == len(list(tmp_path.iterdir())), (
        "the survey did not count every product it would move")


NIC = (35.18557, 33.38228)
SGO = (67.36369337350563, 26.634311805059543)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "ionograms.sqlite3"
    conn = db.init(db.connect(path))
    for n, (tx, t0) in enumerate([("NIC4_280", AT_10), ("NIC4_280", AT_10 + 600),
                                  ("NIC4_280", AT_0), ("Sodankyl\u00e4", AT_0),
                                  ("Magadan", AT_10)]):
        name = "lfm_ionogram-%s-yoshkar-ola-ch0-000-%d.00.h5" % (tx, t0)
        tx = tx.encode().decode("ascii", "replace")
        conn.execute(
            "INSERT INTO sounding (file, path, format, datetime, tx, rx, path_type, "
            "rx_lat, rx_lon, ingested_at) VALUES (?, ?, 'chirp2', ?, ?, "
            "'yoshkar-ola', 'oblique', 56.38, 47.53, '2026-10-01')",
            (name, "ionozond_5tb/ionozond_data/2026.09.30/" + name,
             time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0)), tx))
        conn.execute("INSERT INTO extraction (sounding_id, method, muf, hops) "
                     "VALUES (?, 'kmeans', 18.5, NULL)", (n + 1,))
    conn.execute("INSERT INTO station_preview (station, tx, received_at, image) "
                 "VALUES ('yoshkar-ola', 'NIC4_280', '2026-10-01', x'89')")
    conn.commit()
    conn.close()
    return path


def renamed(path, *texts, apply=True):
    conn = db.connect(path)
    try:
        plan = rename.survey(conn, rules(*texts), "yoshkar-ola")
        if apply:
            rename.commit(conn, plan)
        return plan
    finally:
        conn.close()


def rows(path, sql, params=()):
    conn = db.connect(path)
    try:
        return [tuple(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def test_the_rows_at_the_rules_phase_take_the_new_name(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT COUNT(*) FROM sounding WHERE tx = 'NIC1'") == [
        (2,)], "the rows at the rule's phase did not take the new name"


def test_a_row_at_another_phase_keeps_its_name(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT COUNT(*) FROM sounding WHERE tx = 'NIC4_280'") == [
        (1,)], "a row at another phase was renamed"


def test_a_renamed_row_names_the_renamed_file(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT path FROM sounding WHERE tx = 'NIC1' ORDER BY id"
                )[0] == ("ionozond_5tb/ionozond_data/2026.09.30/lfm_ionogram-NIC1-"
                         "yoshkar-ola-ch0-000-%d.00.h5" % AT_10,), (
        "a renamed row still names the old file")


def test_a_renamed_row_takes_the_registrys_coordinates(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT tx_lat, tx_lon FROM sounding WHERE tx = 'NIC1'"
                )[0] == NIC, "a renamed row did not take the registry's coordinates"


def test_a_renamed_row_gets_a_path_length(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT path_km IS NOT NULL FROM sounding WHERE tx = 'NIC1'"
                )[0] == (1,), "a renamed row got no path length"


def test_a_renamed_rows_extraction_gets_a_hop_count(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT e.hops IS NOT NULL FROM extraction e JOIN sounding s"
                          " ON s.id = e.sounding_id WHERE s.tx = 'NIC1'")[0] == (1,), (
        "a renamed row's extraction got no hop count")


def test_a_name_ingest_garbled_is_renamed_in_the_database(database):
    renamed(database, "Sodankyl\u00e4=SGO")
    assert rows(database, "SELECT tx_lat, tx_lon FROM sounding WHERE tx = 'SGO'") == [
        SGO], "the name ingest garbled was not renamed in the database"


def test_a_row_garbled_alike_but_filed_under_another_name_is_left(database):
    conn = db.connect(database)
    conn.execute("INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
                 "VALUES (?, 'x', '2026-09-30', ?, 'yoshkar-ola', '2026-10-17')",
                 ("lfm_ionogram-Sodankyl\u00f6-yoshkar-ola-ch0-000-%d.00.h5" % AT_0,
                  "Sodankyl\ufffd\ufffd"))
    conn.commit()
    conn.close()
    renamed(database, "Sodankyl\u00e4=SGO")
    assert rows(database, "SELECT COUNT(*) FROM sounding WHERE tx = 'SGO'") == [(1,)], (
        "a row filed under another name was renamed for its garbled tx")


def test_a_row_ingested_under_the_new_name_meanwhile_is_dropped(database):
    conn = db.connect(database)
    name = "lfm_ionogram-NIC1-yoshkar-ola-ch0-000-%d.00.h5" % AT_10
    conn.execute("INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at)"
                 " VALUES (?, ?, '2026-09-30', 'NIC1', 'yoshkar-ola', '2026-10-17')",
                 (name, "x/" + name))
    conn.commit()
    conn.close()
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT COUNT(*) FROM sounding WHERE file = ?", (name,)) == [
        (1,)], "the row ingested under the new name was not dropped"


def test_the_live_preview_takes_the_new_name(database):
    renamed(database, "NIC4_280@10=NIC1")
    assert rows(database, "SELECT tx FROM station_preview") == [("NIC1",)], (
        "the live preview kept the old name")


def test_a_dry_run_writes_no_row(database):
    renamed(database, "NIC4_280@10=NIC1", apply=False)
    assert rows(database, "SELECT COUNT(*) FROM sounding WHERE tx = 'NIC1'") == [
        (0,)], "a dry run renamed a row"


def test_a_name_the_registry_cannot_place_is_refused(database):
    with pytest.raises(SystemExit, match="NIC_235"):
        renamed(database, "NIC4_280@10=NIC_235", apply=False)
