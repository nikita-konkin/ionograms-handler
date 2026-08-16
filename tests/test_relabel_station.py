"""Renaming a receiver is a geometry change, and the tests are about that.

`tools/relabel_station.py` exists because the acquisition laptop wrote
`station_name=DOB` for a site that is Yoshkar-Ola, 2400 km away. The rename
half of that is trivial SQL. The half worth testing is everything the name
decides -- coordinates, path length, hop count -- and the two ways a careless
rename does damage that an UPDATE statement cannot undo: releasing stale
commands onto a live station, and colliding with rows the new name already has.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.api import db  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "relabel_station", ROOT / "tools/relabel_station.py")
relabel = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], and a module that is not there yet raises.
sys.modules["relabel_station"] = relabel
_spec.loader.exec_module(relabel)


#: Nicosia, from muf/stations.py. The transmitter every test here sounds.
NIC = (35.18557, 33.38228)
#: What the registry resolves each receiver name to.
DOMBAS = (62.073, 9.111)
YOSHKAR_OLA = (56.38, 47.53)


@pytest.fixture
def database(tmp_path):
    """A small archive labelled DOB, shaped like the real one."""
    path = tmp_path / "ionograms.sqlite3"
    conn = db.init(db.connect(path))

    for n in range(3):
        conn.execute(
            "INSERT INTO sounding (file, path, format, datetime, tx, rx, "
            "path_type, tx_lat, tx_lon, rx_lat, rx_lon, path_km, "
            "gate_lo, gate_hi, ingested_at) VALUES "
            "(?, ?, 'chirp2', ?, 'NIC', 'DOB', 'oblique', ?, ?, ?, ?, "
            "3435.95, 2329.6, 5000.0, '2026-08-16')",
            (f"lfm_ionogram-NIC-DOB-ch000-002-17708000{n:02d}.00.h5",
             f"2026/08/16/f{n}.h5", f"2026-08-16 12:0{n}:00.0", *NIC, *DOMBAS))
        conn.execute("INSERT INTO extraction (sounding_id, method, muf, hops) "
                     "VALUES (?, 'kmeans', 18.5, 1)", (n + 1,))
        conn.execute("INSERT INTO reference (sounding_id, source, param, value) "
                     "VALUES (?, 'iri', 'fof2', 5.5)", (n + 1,))

    conn.execute("INSERT INTO health_report (station, received_at, document) "
                 "VALUES ('DOB', '2026-08-16T11:00:00Z', '{}')")
    conn.execute("INSERT INTO config_epoch (station, valid_from) "
                 "VALUES ('DOB', '2026-08-16T11:00:00Z')")
    conn.execute(
        "INSERT INTO command (id, station, name, issued_at, delivered_at, acked_at) "
        "VALUES ('cmd-done', 'DOB', 'restart', '2026-08-16T11:00:00Z', "
        "'2026-08-16T11:00:30Z', '2026-08-16T11:01:00Z')")
    conn.commit()
    conn.close()
    return path


def survey(path, old="DOB", new="Yoshkar-Ola"):
    conn = db.connect(path)
    try:
        return relabel.survey(conn, old, new)
    finally:
        conn.close()


def run(path, argv):
    return relabel.main(["--db", str(path)] + argv)


def rows(path, sql, params=()):
    conn = db.connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# The geometry, which is the point
# --------------------------------------------------------------------------

def test_the_receiver_coordinates_and_path_length_move_with_the_name(database):
    """The rename is not the change; this is. Left at DOB's coordinates every
    foF2 on the circuit is divided by the wrong M-factor."""
    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])

    row = rows(database, "SELECT rx, rx_lat, rx_lon, path_km FROM sounding")[0]
    assert row["rx"] == "Yoshkar-Ola"
    assert (row["rx_lat"], row["rx_lon"]) == YOSHKAR_OLA
    # Nicosia -> Yoshkar-Ola, not the 3435.95 km of Nicosia -> Dombas.
    assert row["path_km"] == pytest.approx(2587.8, abs=0.5)


def test_a_name_the_registry_does_not_know_is_refused(database):
    """The failure mode this tool repairs is a name that resolves to nothing.
    Producing one would be the same bug in a new place."""
    with pytest.raises(SystemExit) as excinfo:
        survey(database, new="Nowhere-In-Particular")
    assert "muf/stations.py" in str(excinfo.value)


def test_the_path_length_is_recomputed_from_the_rows_own_transmitter(database):
    """Not from the registry: a transmitter it does not know still has stored
    coordinates, and those are what the sounding was ingested with."""
    conn = db.connect(database)
    conn.execute("UPDATE sounding SET tx = 'not-in-any-table', "
                 "tx_lat = 60.0, tx_lon = 30.0 WHERE id = 1")
    conn.commit()
    conn.close()

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    row = rows(database, "SELECT path_km FROM sounding WHERE id = 1")[0]
    assert row["path_km"] == pytest.approx(1099.4, abs=0.5)


def test_a_sounding_with_no_transmitter_coordinates_keeps_a_null_path(database):
    """NULL means "not known". A path length invented from one endpoint would
    be a number, which is worse."""
    conn = db.connect(database)
    conn.execute("UPDATE sounding SET tx_lat = NULL, tx_lon = NULL WHERE id = 1")
    conn.commit()
    conn.close()

    plan = survey(database)
    assert plan.no_tx_coords == 1

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    assert rows(database, "SELECT path_km FROM sounding WHERE id = 1")[0]["path_km"] is None


def test_hop_count_follows_the_new_path_length(database):
    """`m_factor` converts one hop, so `hops` is what a MUF is divided at. A
    path that crosses 4000 km in either direction changes it, and a stored 1
    against a two-hop path understates foF2 by a third."""
    conn = db.connect(database)
    # 3900 km to DOB is one hop; 4852 km to Yoshkar-Ola is two, either side of
    # the 4000 km MAX_SINGLE_HOP_KM this rename moves the path across.
    conn.execute("UPDATE sounding SET tx_lat = 20.0, tx_lon = 80.0, "
                 "path_km = 3900.0 WHERE id = 1")
    conn.commit()
    conn.close()

    plan = survey(database)
    assert plan.hop_changes == 1

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    assert rows(database, "SELECT hops FROM extraction WHERE sounding_id = 1")[0]["hops"] == 2
    # The circuits that did not change keep the hop count they were given.
    assert rows(database, "SELECT hops FROM extraction WHERE sounding_id = 2")[0]["hops"] == 1


def test_measurements_are_not_touched(database):
    """A MUF is the highest frequency that came back and a virtual range is a
    delay. Neither knows where the receiver is; the foF2 derived from them is
    derived at read time from path_km, which is why fixing path_km is enough."""
    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    assert rows(database, "SELECT muf FROM extraction")[0]["muf"] == 18.5


def test_the_extraction_gate_is_left_describing_the_run_that_happened(database):
    """gate_lo/gate_hi record the window the estimators actually searched. The
    corrected geometry would have chosen a different one, but it did not, and a
    row claiming otherwise describes a run that never took place."""
    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    row = rows(database, "SELECT gate_lo, gate_hi FROM sounding")[0]
    assert (row["gate_lo"], row["gate_hi"]) == (2329.6, 5000.0)


# --------------------------------------------------------------------------
# The two ways it does damage
# --------------------------------------------------------------------------

def test_an_undelivered_command_is_left_behind_by_default(database):
    """Renaming it hands it to the live station on its next pull. A set_config
    queued days ago, against a schedule since replaced, is not something a
    rename should be able to fire."""
    conn = db.connect(database)
    conn.execute("INSERT INTO command (id, station, name, issued_at) "
                 "VALUES ('cmd-queued', 'DOB', 'set_config', '2026-08-10T09:00:00Z')")
    conn.commit()
    conn.close()

    plan = survey(database)
    assert [c[0] for c in plan.undelivered] == ["cmd-queued"]

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    stations = {r["id"]: r["station"] for r in rows(database, "SELECT id, station FROM command")}
    assert stations["cmd-queued"] == "DOB", "an undeliverable command stays undeliverable"
    assert stations["cmd-done"] == "Yoshkar-Ola", "history moves with the station"


def test_release_pending_is_what_hands_it_over(database):
    conn = db.connect(database)
    conn.execute("INSERT INTO command (id, station, name, issued_at) "
                 "VALUES ('cmd-queued', 'DOB', 'set_config', '2026-08-10T09:00:00Z')")
    conn.commit()
    conn.close()

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply",
                   "--release-pending"])
    stations = {r["id"]: r["station"] for r in rows(database, "SELECT id, station FROM command")}
    assert stations["cmd-queued"] == "Yoshkar-Ola"


def test_a_transmitter_collision_blocks_the_whole_rename(database):
    """`transmitter` is UNIQUE (station, code) and UNIQUE (station, sounder_id),
    and the new name already has rows whenever an operator has been identifying
    transmitters under it -- which is exactly when this tool gets run. Finding
    out at COMMIT would roll back a survey that reported success."""
    conn = db.connect(database)
    for station, code, sid in (("DOB", "NIC", 2), ("Yoshkar-Ola", "NIC", 2)):
        conn.execute(
            "INSERT INTO transmitter (station, code, sounder_id, timings, "
            "verified_at) VALUES (?, ?, ?, '[]', '2026-08-16T11:50:00Z')",
            (station, code, sid))
    conn.commit()
    conn.close()

    plan = survey(database)
    assert len(plan.collisions) == 1

    with pytest.raises(SystemExit, match="on-conflict"):
        run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    assert rows(database, "SELECT COUNT(*) c FROM sounding WHERE rx = 'DOB'")[0]["c"] == 3


def test_keep_old_drops_only_the_rows_that_actually_collide(database):
    """"Keep the old one" is about the duplicate, not about everything the new
    name has identified. NIC3 was censused under Yoshkar-Ola and DOB never
    sounded it; it has no business being collateral."""
    conn = db.connect(database)
    for station, code, sid in (("DOB", "NIC", 2), ("Yoshkar-Ola", "NIC", 2),
                               ("Yoshkar-Ola", "NIC3", 4)):
        conn.execute(
            "INSERT INTO transmitter (station, code, sounder_id, timings, "
            "verified_at) VALUES (?, ?, ?, '[]', '2026-08-16T11:50:00Z')",
            (station, code, sid))
    conn.commit()
    conn.close()

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply",
                   "--on-conflict", "keep-old"])
    kept = sorted(r["code"] for r in rows(database, "SELECT code FROM transmitter"))
    assert kept == ["NIC", "NIC3"]


def test_keep_new_drops_the_old_stations_duplicate(database):
    conn = db.connect(database)
    for station, code, sid, note in (("DOB", "NIC", 2, "old"),
                                     ("Yoshkar-Ola", "NIC", 2, "new")):
        conn.execute(
            "INSERT INTO transmitter (station, code, sounder_id, timings, "
            "verified_at, note) VALUES (?, ?, ?, '[]', '2026-08-16T11:50:00Z', ?)",
            (station, code, sid, note))
    conn.commit()
    conn.close()

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply",
                   "--on-conflict", "keep-new"])
    kept = rows(database, "SELECT station, note FROM transmitter")
    assert [(r["station"], r["note"]) for r in kept] == [("Yoshkar-Ola", "new")]


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------

def test_a_dry_run_writes_nothing(database):
    """The default, because the first run of this is always exploratory."""
    assert run(database, ["--from", "DOB", "--to", "Yoshkar-Ola"]) == 0
    assert rows(database, "SELECT COUNT(*) c FROM sounding WHERE rx = 'DOB'")[0]["c"] == 3
    assert rows(database, "SELECT rx_lat FROM sounding")[0]["rx_lat"] == DOMBAS[0]


def test_applying_leaves_a_backup_that_opens(database):
    """Taken with SQLite's backup API rather than a file copy: the database
    runs in WAL mode and a `cp` under a concurrent writer can land
    mid-transaction."""
    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])

    backups = list(database.parent.glob(f"{database.name}.bak-*"))
    assert len(backups) == 1

    conn = sqlite3.connect(str(backups[0]))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM sounding WHERE rx = 'DOB'").fetchone()[0] == 3
    finally:
        conn.close()


def test_modelled_values_survive_unless_asked_for(database):
    """IRI is evaluated at the path's control point, which moves ~1000 km here,
    and the GIRO station chosen by proximity moves with it. Stale, but deleting
    a user's data as a side effect of a rename is not this tool's call."""
    plan = survey(database)
    assert plan.reference_rows == 3

    run(database, ["--from", "DOB", "--to", "Yoshkar-Ola", "--apply"])
    assert rows(database, "SELECT COUNT(*) c FROM reference")[0]["c"] == 3

    run(database, ["--from", "Yoshkar-Ola", "--to", "DOB", "--apply",
                   "--drop-reference"])
    assert rows(database, "SELECT COUNT(*) c FROM reference")[0]["c"] == 0


def test_a_name_nothing_carries_is_a_no_op(database):
    assert run(database, ["--from", "SGO", "--to", "Yoshkar-Ola", "--apply"]) == 0
    assert rows(database, "SELECT COUNT(*) c FROM sounding WHERE rx = 'DOB'")[0]["c"] == 3


def test_the_files_still_carrying_the_old_name_are_counted(database):
    """The name is in the h5's own `station_name`, and `io_chirp.read_header`
    prefers that over everything. Re-ingesting a pre-rename product puts the
    old name back on that row -- correct behaviour, and the size of it is worth
    printing rather than discovering."""
    assert survey(database).stale_filenames == 3
