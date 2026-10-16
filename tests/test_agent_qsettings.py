"""The ionozond console's QSettings files, read and edited as the console's bytes."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from services.agent.qsettings import Ini, IniError

DATA = Path(__file__).resolve().parent / "ionozond"
FILES = ["golden_config.ini", "golden_schedule.ini", "fixture_config.ini",
         "fixture_schedule.ini", "console_config.ini", "console_schedule.ini"]
PLAIN = re.compile(r"^[A-Za-z0-9_.+-]+$")


def pairs(ini: Ini) -> list:
    return [(group, key) for group in ["", *ini.groups()] for key in ini.keys(group)]


def restored(ini: Ini) -> Ini:
    for group, key in pairs(ini):
        value = ini.value(group, key)
        if PLAIN.match(value):
            other = "0" if value != "0" else "1"
            ini = ini.with_value(group, key, other).with_value(group, key, value)
    return ini


@pytest.mark.parametrize("name", FILES)
def test_every_plain_key_edited_and_restored_leaves_the_file_byte_identical(name):
    data = (DATA / name).read_bytes()
    assert restored(Ini(data)).data == data, (
        "%s did not survive a round trip byte for byte" % name)


@pytest.mark.parametrize("name", FILES)
def test_every_value_the_console_wrote_can_be_read(name):
    ini = Ini((DATA / name).read_bytes())
    values = [ini.value(group, key) for group, key in pairs(ini)]
    assert len(values) == len(pairs(ini)), "%s has a value that cannot be read" % name


def test_an_edit_changes_only_the_line_of_its_key():
    data = (DATA / "golden_schedule.ini").read_bytes()
    edited = Ini(data).with_value("cyprus1", "chirpt", str(random.randint(11, 299)))
    changed = [n for n, (a, b) in enumerate(zip(data.split(b"\n"),
                                                edited.data.split(b"\n"))) if a != b]
    assert changed == [3], "an edit of one key touched other lines"


def test_an_edit_keeps_the_files_crlf_line_endings():
    data = b"[cyprus1]\r\nactive=true\r\nrep=300\r\n\r\n[pr]\r\nrep=720\r\n"
    edited = Ini(data).with_value("cyprus1", "rep", "600")
    assert edited.data == data.replace(b"rep=300", b"rep=600"), (
        "an edit did not keep the file's CRLF line endings")


def test_a_quoted_value_reads_without_its_quotes():
    ini = Ini((DATA / "console_config.ini").read_bytes())
    assert ini.value("", "data_dir") == "/home/electronic/IGs/", (
        "a quoted value kept its quotes")


def test_an_escaped_quote_inside_quotes_reads_as_a_quote():
    assert Ini(b'[General]\nname="a\\"b"\n').value("", "name") == 'a"b', (
        "an escaped quote was not read as a quote")


def test_a_hex_escape_reads_as_its_character():
    assert Ini(b"[General]\nname=\\x41\\x0444\n").value("", "name") == "Aф", (
        "a hex escape was not read as its character")


def test_a_comma_makes_a_list_that_cannot_be_read_as_one_value():
    with pytest.raises(IniError):
        Ini(b"[pr]\nlat=56,38\n").value("pr", "lat")


def test_an_at_sign_value_cannot_be_read_as_text():
    with pytest.raises(IniError):
        Ini(b"[pr]\ngeometry=@ByteArray(abc)\n").value("pr", "geometry")


def test_a_doubled_at_sign_reads_as_one():
    assert Ini(b"[pr]\nnote=@@home\n").value("pr", "note") == "@home", (
        "a doubled at sign was not read as one")


def test_a_semicolon_ends_the_value():
    assert Ini(b"[pr]\nrep=720 ; every 12 min\n").value("pr", "rep") == "720", (
        "a semicolon did not end the value")


def test_a_comment_line_is_not_a_key():
    assert Ini(b"[pr]\n;rep=1\nchirpt=380\n").keys("pr") == ["chirpt"], (
        "a comment line was read as a key")


def test_the_general_section_is_the_root_group():
    assert Ini(b"[General]\ndec=625\n[Variations]\nx=1\n").keys("") == ["dec"], (
        "the General section was not read as the root group")


def test_a_percent_general_section_is_a_group_named_general():
    assert Ini(b"[%General]\ndec=625\n").groups() == ["General"], (
        "a %General section was not a group named General")


def test_groups_come_sorted_as_qsettings_lists_them():
    ini = Ini(b"[yoshkar-ola]\nrx=true\n[Norilsk]\nrx=false\n[cyprus1]\nrx=false\n")
    assert ini.groups() == ["Norilsk", "cyprus1", "yoshkar-ola"], (
        "groups did not come in QSettings' order")


def test_keys_come_sorted_as_qsettings_lists_them():
    ini = Ini((DATA / "golden_schedule.ini").read_bytes())
    assert ini.keys("cyprus1") == sorted(ini.keys("cyprus1")), (
        "keys did not come in QSettings' order")


def test_the_last_of_two_equal_keys_wins():
    assert Ini(b"[pr]\nrep=720\nrep=360\n").value("pr", "rep") == "360", (
        "the last of two equal keys did not win")


def test_a_key_given_twice_cannot_be_edited():
    with pytest.raises(IniError):
        Ini(b"[pr]\nrep=720\nrep=360\n").with_value("pr", "rep", "300")


def test_a_new_key_is_read_back_from_its_group():
    value = str(random.choice([2048, 4096, 8192]))
    ini = Ini((DATA / "golden_schedule.ini").read_bytes()).with_value(
        "cyprus1", "fft_count", value)
    assert ini.value("cyprus1", "fft_count") == value, "a new key was not read back"


def test_a_new_key_leaves_every_other_group_alone():
    data = (DATA / "golden_schedule.ini").read_bytes()
    ini = Ini(data).with_value("cyprus1", "fft_count", "4096")
    assert ini.data.endswith(data[data.index(b"[yoshkar-ola]"):]), (
        "a new key disturbed another group")


def test_a_new_key_after_a_last_line_without_a_break_keeps_that_line():
    ini = Ini(b"[pr]\nrep=720").with_value("pr", "chirpt", "380")
    assert ini.value("pr", "rep") == "720", (
        "a key added after an unterminated last line damaged that line")


def test_a_new_group_is_read_back():
    ini = Ini((DATA / "golden_schedule.ini").read_bytes()).with_value(
        "magadan", "cf", "16500")
    assert ini.value("magadan", "cf") == "16500", "a new group was not read back"


def test_a_root_key_in_a_file_without_general_is_read_back():
    ini = Ini(b"[Variations]\nx=1\n").with_value("", "keep_lfs", "true")
    assert ini.value("", "keep_lfs") == "true", (
        "a root key added to a file without [General] was not read back")


def test_a_root_key_added_to_a_file_without_general_leaves_its_groups():
    ini = Ini(b"[Variations]\nx=1\n").with_value("", "keep_lfs", "true")
    assert ini.value("Variations", "x") == "1", (
        "a root key added to a file without [General] disturbed its groups")


@pytest.mark.parametrize("value", ["56,38", "a b", '"q"', "Норильск", "", "x;y"])
def test_a_value_that_needs_escaping_is_refused(value):
    with pytest.raises(IniError):
        Ini(b"[pr]\nrep=720\n").with_value("pr", "rep", value)


def test_a_missing_key_is_refused_when_read():
    with pytest.raises(KeyError):
        Ini(b"[pr]\nrep=720\n").value("pr", "chirpt")
