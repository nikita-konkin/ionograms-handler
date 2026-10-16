"""chirp_config.py, generated as the console does and read as the sounder does."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from services.agent.qsettings import Ini
from services.agent.sounderconfig import UnreadableConfig, build, literals, parsed

DATA = Path(__file__).resolve().parent / "ionozond"


def golden(name: str) -> Ini:
    return Ini((DATA / name).read_bytes())


def two(names) -> Ini:
    text = "".join("[%s]\nactive=true\nrx=false\ncf=%d\nrep=300\nlat=35\nlon=34\n\n"
                   % (name, 20000 + n) for n, name in enumerate(names))
    return Ini(text.encode("ascii"))


def test_the_generator_matches_the_original_binary_byte_for_byte():
    want = (DATA / "golden_chirp_config.py.txt").read_bytes().decode("utf-8")
    got = build(golden("golden_schedule.ini"), golden("golden_config.ini"))
    assert got == want, "the generator does not reproduce the original binary output"


def test_two_active_stations_are_separated_by_a_comma():
    names = random.sample(["NIC4_280", "NIC_900", "Magadan", "Norilsk", "cyprus1"], 2)
    text = build(two(sorted(names)), Ini(b""))
    assert text.count("}],") == 1, "two active stations were not separated"


def test_an_inactive_station_is_left_out():
    schedule = golden("golden_schedule.ini").with_value("cyprus1", "active", "false")
    text = build(schedule, golden("fixture_config.ini"))
    assert "'name':'cyprus1'" not in text, "an inactive station reached chirp_config.py"


def test_the_receiver_becomes_rx_station():
    text = build(golden("golden_schedule.ini"), golden("fixture_config.ini"))
    assert "rx_station = {'name':'yoshkar-ola','lat':56.38,'lon':47.53}" in text, (
        "the receiver did not become rx_station")


def test_a_quoted_data_dir_is_written_as_a_python_string():
    text = build(golden("console_schedule.ini"), golden("console_config.ini"))
    assert 'data_dir = "/home/electronic/IGs/"\n' in text, (
        "data_dir was not written as a Python string")


def test_what_the_fixtures_generate_is_what_the_sounder_reads():
    config = parsed(build(golden("fixture_schedule.ini"), golden("fixture_config.ini")))
    assert [s["name"] for s in config["sounders"][0]] == ["cyprus1"], (
        "the sounder would not read the generated file's sounders")


def test_the_sounder_derives_if_rate_from_the_sample_rate_and_decimation():
    config = parsed(build(golden("fixture_schedule.ini"), golden("fixture_config.ini")))
    assert config["if_rate"] == 25000e3 / 625, "if_rate was not derived"


def test_the_originals_empty_tb_is_unreadable_to_the_sounder():
    with pytest.raises(UnreadableConfig):
        parsed((DATA / "golden_chirp_config.py.txt").read_bytes().decode("utf-8"))


def test_a_schedule_with_no_active_station_is_unreadable_to_the_sounder():
    schedule = golden("fixture_schedule.ini").with_value("cyprus1", "active", "false")
    with pytest.raises(UnreadableConfig):
        parsed(build(schedule, golden("fixture_config.ini")))


def test_sounders_that_are_not_a_literal_are_unreadable_to_the_sounder():
    with pytest.raises(UnreadableConfig):
        parsed("sounders = [\n[{'name':'a'}]\n[{'name':'b'}]\n]\n")


def test_an_empty_file_is_unreadable_to_the_sounder():
    with pytest.raises(UnreadableConfig):
        parsed(" \n")


def test_the_literals_of_a_file_without_an_active_station_can_still_be_read():
    schedule = golden("fixture_schedule.ini").with_value("cyprus1", "active", "false")
    config = literals(build(schedule, golden("fixture_config.ini")))
    assert config["sounders"] == [], "a file without an active station was not read"


def test_the_literals_of_a_file_python_cannot_parse_are_unreadable():
    with pytest.raises(UnreadableConfig):
        literals("tb = \nsounders = []\n")
