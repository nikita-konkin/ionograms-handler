"""The ionozond site settings: all shown from the root file, a few set from the web."""

from __future__ import annotations

import os
import random
from dataclasses import replace

import pytest
from test_agent_ionozond import laptop, systemd

from services.agent import client, ionozond, runner, siteconfig


def station(tmp_path, overrides: bytes = b""):
    config = laptop(tmp_path)
    with open(str(config.sounder_defaults), "ab") as fh:
        fh.write(b"PRUNE_KEEP_DAYS=7\nPRUNE_FREE_GB=50\nNAS_BWLIMIT=0\n"
                 b"NAS_KEEP_H5_DAYS=30\nIONOZOND_ARCHIVE="
                 + tmp_path.as_posix().encode() + b"\n")
    web = tmp_path / "web" / "web.env"
    if overrides:
        web.parent.mkdir()
        web.write_bytes(overrides)
    return replace(config, site_overrides=web)


def change(config, **values) -> dict:
    return {"base": siteconfig.report(config)["hash"], "site": values}


def test_every_key_of_the_root_file_is_shown(tmp_path):
    keys = siteconfig.report(station(tmp_path))["keys"]
    assert {"RADIO_ADDR", "JUMBO", "IONOZOND_CONFIG", "PRUNE_KEEP_DAYS"} <= set(keys), (
        "a key of /etc/default/ionozond was not shown")


def test_a_key_from_the_root_file_says_so(tmp_path):
    config = station(tmp_path)
    assert siteconfig.report(config)["keys"]["JUMBO"]["from"] == str(
        config.sounder_defaults), "a key from the root file did not name it"


def test_an_overridden_key_says_it_comes_from_the_override_file(tmp_path):
    config = station(tmp_path, b"PRUNE_FREE_GB=80\n")
    assert siteconfig.report(config)["keys"]["PRUNE_FREE_GB"] == {
        "value": "80", "from": str(config.site_overrides)}, (
        "an overridden key did not come from the override file")


def test_the_report_names_the_keys_the_web_may_edit(tmp_path):
    assert siteconfig.report(station(tmp_path))["editable"] == [
        "NAS_BWLIMIT", "PRUNE_FREE_GB", "PRUNE_KEEP_DAYS"], (
        "the report named other keys as editable")


def test_the_report_gives_the_archive_volume_size(tmp_path):
    assert siteconfig.report(station(tmp_path))["archive_gb"] > 0, (
        "the report gave no size for the archive's volume")


def test_a_site_change_writes_the_override_file(tmp_path):
    config = station(tmp_path)
    days = random.randint(1, 365)
    siteconfig.apply(config, change(config, PRUNE_KEEP_DAYS=str(days)))
    assert "PRUNE_KEEP_DAYS=%d\n" % days in config.site_overrides.read_text(
        encoding="utf-8"), "the site change did not reach the override file"


def test_a_site_change_leaves_the_root_file_alone(tmp_path):
    config = station(tmp_path)
    before = config.sounder_defaults.read_bytes()
    siteconfig.apply(config, change(config, NAS_BWLIMIT="500"))
    assert config.sounder_defaults.read_bytes() == before, (
        "a site change edited /etc/default/ionozond")


@pytest.mark.parametrize("key", ["RADIO_ADDR", "IFACE", "JUMBO", "IONOZOND_ARCHIVE",
                                 "NAS_DEST", "NAS_SSH_KEY", "NAS_KEEP_H5_DAYS", "PATH",
                                 "LD_PRELOAD"])
def test_a_key_outside_the_allow_list_is_refused(tmp_path, key):
    config = station(tmp_path)
    assert siteconfig.apply(config, change(config, **{key: "1"}))[0].ok is False, (
        "a key outside the allow list was written")


def test_a_refused_key_writes_no_override_file(tmp_path):
    config = station(tmp_path)
    siteconfig.apply(config, change(config, PATH="/tmp"))
    assert not config.site_overrides.exists(), (
        "a refused site change still wrote the override file")


@pytest.mark.parametrize("key,value", [
    ("PRUNE_KEEP_DAYS", "0"), ("PRUNE_KEEP_DAYS", "366"), ("PRUNE_KEEP_DAYS", "x"),
    ("PRUNE_FREE_GB", "-1"), ("PRUNE_FREE_GB", "999999999"), ("NAS_BWLIMIT", "-5"),
    ("NAS_BWLIMIT", "1e3"), ("PRUNE_KEEP_DAYS", "7\nPATH=/tmp")])
def test_a_value_out_of_range_is_refused(tmp_path, key, value):
    config = station(tmp_path)
    assert siteconfig.apply(config, change(config, **{key: value}))[0].ok is False, (
        "a value out of range was written")


def test_an_empty_value_drops_the_override(tmp_path):
    config = station(tmp_path, b"PRUNE_FREE_GB=80\n")
    siteconfig.apply(config, change(config, PRUNE_FREE_GB=None))
    assert siteconfig.report(config)["keys"]["PRUNE_FREE_GB"]["value"] == "50", (
        "an empty value did not fall back to the root file")


def test_a_site_change_against_a_moved_base_is_refused(tmp_path):
    config = station(tmp_path)
    shown = dict(change(config, NAS_BWLIMIT="100"), base="0" * 64)
    assert siteconfig.apply(config, shown)[0].ok is False, (
        "a site change against a moved base was written")


def test_a_failed_write_leaves_the_old_override_file(tmp_path, monkeypatch):
    config = station(tmp_path, b"PRUNE_FREE_GB=80\n")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(
        OSError("disk full")))
    siteconfig.apply(config, change(config, PRUNE_FREE_GB="90"))
    assert config.site_overrides.read_bytes() == b"PRUNE_FREE_GB=80\n", (
        "a failed write did not leave the old override file")


def test_a_failed_write_is_acknowledged_as_a_failure(tmp_path, monkeypatch):
    config = station(tmp_path)
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(
        OSError("disk full")))
    assert siteconfig.apply(config, change(config, NAS_BWLIMIT="1"))[0].ok is False, (
        "a failed write was acknowledged as written")


def test_a_hand_added_line_in_the_override_file_survives_a_change(tmp_path):
    config = station(tmp_path, b"# kept\nNAS_BWLIMIT=10\n")
    siteconfig.apply(config, change(config, PRUNE_FREE_GB="70"))
    assert config.site_overrides.read_text(encoding="utf-8").startswith(
        "# kept\nNAS_BWLIMIT=10\n"), "a site change dropped a line it did not edit"


def test_the_agent_dispatches_a_site_change(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(siteconfig, "apply", lambda config, params: seen.append(
        params) or [])
    runner._dispatch(station(tmp_path), client.Command(
        "1", "set_ionozond_site", {"base": "x"}))
    assert seen == [{"base": "x"}], "set_ionozond_site did not reach the site settings"


@pytest.mark.parametrize("unit", ["ionozond-nas.service", "ionozond-prune.service"])
def test_run_now_starts_a_job_unit_without_waiting(tmp_path, monkeypatch, unit):
    calls = systemd(monkeypatch)
    config = replace(station(tmp_path), no_block=False)
    runner._dispatch(config, client.Command("1", "start", {"unit": unit}))
    assert calls == [("start", unit, True)], (
        "run now did not start the job unit with --no-block")


@pytest.mark.parametrize("unit", ["ionozond-sounder.service", "ssh.service",
                                  "chirp-agent.service", ""])
def test_run_now_refuses_any_other_unit(tmp_path, monkeypatch, unit):
    calls = systemd(monkeypatch)
    runner._dispatch(station(tmp_path), client.Command("1", "start", {"unit": unit}))
    assert calls == [], "run now started a unit that is not a job unit"


def test_a_plain_start_still_starts_the_sounder(tmp_path, monkeypatch):
    calls = systemd(monkeypatch)
    runner._dispatch(station(tmp_path), client.Command("1", "start", {}))
    assert [c[1] for c in calls] == ["ionozond-sounder.service"], (
        "a start without a unit no longer started the sounder")


def test_the_report_carries_the_site_settings(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = station(tmp_path, b"NAS_BWLIMIT=10\n")
    assert ionozond.state(config)["site"]["keys"]["NAS_BWLIMIT"]["value"] == "10", (
        "the schedule report did not carry the site settings")


def test_the_report_carries_the_nas_destination(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = station(tmp_path)
    path = config.console_settings / "config.ini"
    path.write_bytes(path.read_bytes().replace(
        b"[General]\n", b"[General]\nnas_dest=\"/mnt/ionozond_5tb/ionozond_data\"\n"))
    assert ionozond.state(config)["nas_dest"] == "/mnt/ionozond_5tb/ionozond_data", (
        "the schedule report did not carry the NAS destination")


def test_control_is_untouched_by_a_site_change(tmp_path, monkeypatch):
    calls = systemd(monkeypatch)
    config = station(tmp_path)
    siteconfig.apply(config, change(config, NAS_BWLIMIT="5"))
    assert calls == [], "a site change restarted something"

