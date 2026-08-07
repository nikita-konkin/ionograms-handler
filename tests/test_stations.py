"""Station coordinate registry (``architecture.md`` sec. 2.3 item 3).

A wrong coordinate here does not raise. It produces a path length that looks
reasonable and a virtual height that is quietly wrong, so the tests that earn
their place are the ones about provenance and about names that must *not*
resolve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from muf import loader, stations
from muf.stations import LFS, V2, Registry, Station

#: The pinned chirpsounder2 clone, when it is on this machine. The built-in
#: table was transcribed from it by hand, and a hand transcription of 16
#: coordinates to 14 significant figures is exactly the kind of thing that
#: acquires a typo.
CLONE_SERVER_INI = Path(
    os.environ.get("MUF_TEST_CHIRP_CLONE",
                   Path(__file__).resolve().parents[2] / "chirpsounder2")
) / "examples" / "marieluise" / "server.ini"


def test_the_built_in_table_holds_both_sources():
    registry = stations.default_registry()
    sources = {s.source for s in registry.stations()}

    assert sources == {V2, LFS}
    assert len(registry) == 17
    assert sum(s.source == V2 for s in registry.stations()) == 16
    assert sum(s.source == LFS for s in registry.stations()) == 1


def test_every_station_records_where_its_numbers_came_from():
    for station in stations.default_registry().stations():
        assert station.source, f"{station.code} has no provenance"
        assert -90.0 <= station.latitude <= 90.0
        assert -180.0 <= station.longitude <= 180.0


def test_lookup_folds_case_and_whitespace():
    """v2 writes DOB, .lfs writes cyprus1, a hand-edited config writes anything."""
    registry = stations.default_registry()
    for spelling in ("DOB", "dob", "Dob", "  DOB  "):
        assert registry[spelling] == pytest.approx((62.073, 9.111))


def test_the_unidentified_marker_never_resolves():
    """Every unidentified emitter in an archive shares this string.

    Resolving it would give a whole night of distinct transmitters one
    position, and the resulting path lengths would all look plausible.
    """
    registry = stations.default_registry()
    assert registry.station(stations.UNIDENTIFIED) is None
    assert registry.station("unkown") is None
    assert registry.station("UNKOWN") is None
    with pytest.raises(KeyError):
        registry["unkown"]


def test_an_unknown_name_is_a_miss_not_an_exception():
    registry = stations.default_registry()
    assert registry.station("no-such-site") is None
    assert registry.station("") is None


def test_cyprus1_is_nic():
    """Two names for one site, resolved by operator judgement on 2026-08-05.

    The archive could not decide it: after removing the DOB receiver's
    0.9557 s epoch error the four slots spread over 106 km, against the 40 km
    separating the candidates. The five-decimal position was preferred over
    the header's round 35.0/34.0.
    """
    registry = stations.default_registry()
    cyprus1, nic = registry.station("cyprus1"), registry.station("NIC")

    assert cyprus1 is nic
    assert registry["cyprus1"] == pytest.approx((35.18557, 33.38228))
    assert "cyprus1" in nic.note
    # one site, not two -- the alias must not inflate the count
    assert len(registry) == 17
    assert "cyprus1" not in list(registry)


def test_the_superseded_cyprus1_position_is_kept_not_deleted():
    """It is still embedded in every .lfs header; someone will notice."""
    assert stations.CYPRUS1_LFS_COORDINATES == (35.0, 34.0)


def test_lfs_soundings_are_unaffected_by_the_registry(make_lfs):
    """`io_lfs` reads coordinates from each file's own header, never from here.

    So a .lfs sounding of cyprus1 keeps reporting 35.0/34.0 and the 2588.4 km
    path that `signal-chain.md` records as measured. Choosing NIC's position
    moves that path by 0.6 km and this pipeline never applies it anyway; it is
    v2 products, which carry a name and nothing else, that the table governs.
    """
    rng = np.random.default_rng(0)
    iq = (rng.normal(size=8192 * 4) + 1j * rng.normal(size=8192 * 4)).astype("complex64")
    header = loader.read_header(make_lfs(iq, name="c.lfs"))

    assert header.tx_latitude == pytest.approx(35.0)
    assert header.tx_longitude == pytest.approx(34.0)


def test_aliases_resolve_without_being_counted_twice():
    registry = Registry([
        Station("AAA", "A", 1.0, 2.0, "test", aliases=("aaa-old", "A1")),
    ])
    assert registry["AAA"] == registry["aaa-old"] == registry["A1"]
    assert len(registry) == 1
    assert list(registry) == ["AAA"]


def test_merge_lets_the_local_file_win():
    """A station is authoritative about its own coordinates."""
    base = stations.default_registry()
    override = Registry([Station("DOB", "Dombas, resurveyed", 62.1, 9.2, "local")])
    merged = base.merged_with(override)

    assert merged["DOB"] == pytest.approx((62.1, 9.2))
    assert merged["SGO"] == base["SGO"]      # everything else survives
    assert len(merged) == len(base)
    assert base["DOB"] == pytest.approx((62.073, 9.111)), "merge mutated the original"


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def test_reads_a_json_registry(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(json.dumps({
        "XYZ": {"name": "Test Site", "lat": 10.5, "lon": -20.25,
                "aliases": ["xyz-1"], "note": "synthetic"},
        "ABC": {"name": "Other", "latitude": 1.0, "longitude": 2.0},
    }), encoding="utf-8")

    registry = stations.from_json(path)
    assert len(registry) == 2
    assert registry["XYZ"] == pytest.approx((10.5, -20.25))
    assert registry["xyz-1"] == pytest.approx((10.5, -20.25))
    assert registry["ABC"] == pytest.approx((1.0, 2.0))
    assert registry.station("XYZ").source == "stations.json"


def test_a_registry_entry_without_coordinates_is_an_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"XYZ": {"name": "no position"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="has no lat/lon"):
        stations.from_json(path)


def test_reads_chirpsounder2s_own_server_ini(tmp_path):
    """So a deployment can point at its live config instead of transcribing."""
    path = tmp_path / "server.ini"
    path.write_text(
        "[stations]\n"
        'station_info = {"QQQ": {"name": "Q", "lat": 5.0, "lon": 6.0}}\n',
        encoding="utf-8")

    registry = stations.from_server_ini(path)
    assert registry["QQQ"] == pytest.approx((5.0, 6.0))


def test_a_server_ini_without_stations_says_so(tmp_path):
    path = tmp_path / "server.ini"
    path.write_text("[lfm]\nrate = 100e3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no \\[stations\\] station_info"):
        stations.from_server_ini(path)


@pytest.mark.skipif(not CLONE_SERVER_INI.is_file(),
                    reason=f"pinned clone not present: {CLONE_SERVER_INI}")
def test_the_transcription_matches_the_clone_it_came_from():
    """16 coordinates copied by hand at 14 significant figures.

    This is the test that catches a digit dropped in transcription, which
    nothing downstream would. It compares against the clone when it happens to
    be on this machine, and skips otherwise -- the built-in table must never
    *depend* on the clone at runtime (sec. 2.2).
    """
    upstream = stations.from_server_ini(CLONE_SERVER_INI)
    builtin = stations.default_registry()

    for station in upstream.stations():
        mine = builtin.station(station.code)
        assert mine is not None, f"{station.code} missing from the built-in table"
        assert mine.latitude == station.latitude, station.code
        assert mine.longitude == station.longitude, station.code
        assert mine.source == V2


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def test_loader_defaults_to_the_built_in_table():
    """`None` means the built-in table here, and no table in `io_chirp`."""
    assert len(loader.resolve_stations(None)) == 17
    assert loader.resolve_stations({}) == {}


def test_a_v2_product_gets_geometry_without_being_asked(make_chirp_h5):
    """The whole point of the item: v2 carries names, not coordinates."""
    path = make_chirp_h5(np.full((4, 64), 100.0), txname="SGO", station_name="DOB")
    header = loader.read_header(path)

    assert header.has_coordinates
    assert header.tx_latitude == pytest.approx(67.36369, abs=1e-4)
    assert header.rx_longitude == pytest.approx(9.111, abs=1e-4)

    from muf.geometry import path_of
    assert path_of(header)[2] == pytest.approx(1013.4, abs=1.0)


def test_an_unidentified_product_still_has_no_geometry(make_chirp_h5):
    from muf.io_chirp import UNIDENTIFIED_TX

    path = make_chirp_h5(np.full((4, 64), 100.0), txname=UNIDENTIFIED_TX,
                         station_name="DOB")
    header = loader.read_header(path)

    assert not header.has_coordinates
    assert np.isnan(header.tx_latitude)
    assert header.rx_latitude == pytest.approx(62.073)   # the receiver is known


def test_an_empty_registry_disables_lookup(make_chirp_h5):
    path = make_chirp_h5(np.full((4, 64), 100.0), txname="SGO", station_name="DOB")
    assert not loader.read_header(path, stations={}).has_coordinates


def test_describe_shows_provenance_and_the_cyprus_note():
    text = stations.describe()
    assert V2 in text and LFS in text
    assert "cyprus1" in text
    assert "17 stations" in text
