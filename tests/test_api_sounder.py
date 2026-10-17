"""The ionozond schedule on the server: vetted, shown, and journalled."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from services.agent import ionozond
from services.agent.qsettings import Ini
from services.agent.sounderconfig import build
from services.api import db

CTL = {"Authorization": "Bearer ctl"}
CONFIG = Path(__file__).resolve().parent / "ionozond" / "fixture_config.ini"
SCHEDULE = (b"[cyprus1]\nactive=true\nrep=300\nchirpt=10\nrate=100\ndur=250\ncf=20000\n"
            b"lat=35\nlon=34\nrx=false\n\n"
            b"[norilsk]\nactive=false\nrep=300\nchirpt=180\nrate=500\ndur=50\n"
            b"cf=16500\nlat=69.2\nlon=88.13\nrx=false\nfft_count=4096\n\n"
            b"[yoshkar-ola]\nactive=false\nrep=60\nchirpt=0\nrate=50\ndur=260\n"
            b"cf=13500\nlat=56.38\nlon=47.53\nrx=true\n")


def reported(schedule: bytes = SCHEDULE, **extra) -> dict:
    read = Ini(CONFIG.read_bytes())
    return dict(ionozond.described(Ini(schedule), read, build(Ini(schedule), read)),
                timeout_stop_s=300.0, pending=False, **extra)


def switched(field: str, value: str) -> bytes:
    return Ini(SCHEDULE).with_value("norilsk", field, value).data


def health(client, sounder=None, station="yoshkar-ola"):
    return client.post("/stations/health", headers=CTL, json={
        "station": station, "timestamp": time.time(), "agent_version": "0.1.0",
        "healthy": True, "metrics": [],
        "sounder_config": reported() if sounder is None else sounder})


def edit(sounder: dict, change) -> dict:
    wanted = copy.deepcopy(sounder["document"])
    change(wanted)
    return dict(wanted, base=sounder["base"])


def queue(client, params, station="yoshkar-ola"):
    return client.post("/stations/%s/commands" % station, headers=CTL,
                       json={"name": "set_ionozond", "params": params})


def epochs(client, station="yoshkar-ola") -> list:
    return db.rows(client.app.state.db, "SELECT * FROM config_epoch WHERE station = ?"
                                        " ORDER BY id", (station,))


def test_a_change_for_a_station_without_a_health_loop_is_refused(client):
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    assert r.status_code == 409, "a change was queued with no schedule to vet against"


def test_the_refusal_names_the_missing_health_loop(client):
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    assert "health loop" in r.json()["detail"], (
        "the refusal did not say the station's health loop is missing")


def test_a_valid_change_is_queued(client):
    health(client)
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt=str(random.randint(11, 40)))))
    assert r.status_code == 200, "a valid schedule change was not queued"


def test_a_change_against_a_stale_base_is_refused(client):
    health(client)
    r = queue(client, dict(edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")), base="0" * 64))
    assert r.status_code == 400, "a change against a stale schedule was queued"


def test_a_change_the_checker_calls_wrong_is_refused(client):
    health(client)
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        rep="0")))
    assert "hangs" in r.json()["detail"], "a schedule with rep 0 was queued"


def test_a_change_to_a_drifted_station_is_refused(client):
    health(client, reported(drift=True))
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    assert "regenerate" in r.json()["detail"], "a drifted station took a change"


def test_a_check_returns_the_collision_it_finds(client):
    health(client)
    r = client.post("/stations/yoshkar-ola/ionozond/check", headers=CTL, json=edit(
        reported(), lambda d: d["schedule"]["norilsk"].update(active="true",
                                                             chirpt="100")))
    assert any(i["station"] == "norilsk" and i["level"] == "warning"
               for i in r.json()["issues"]), "the check did not return the collision"


def test_a_check_queues_nothing(client):
    health(client)
    client.post("/stations/yoshkar-ola/ionozond/check", headers=CTL, json=edit(
        reported(), lambda d: d["schedule"]["cyprus1"].update(chirpt="20")))
    assert db.count_pending(client.app.state.db, "yoshkar-ola") == 0, (
        "a check queued a command")


def test_a_check_of_a_wrong_schedule_says_it_cannot_go(client):
    health(client)
    r = client.post("/stations/yoshkar-ola/ionozond/check", headers=CTL, json=edit(
        reported(), lambda d: d["schedule"]["cyprus1"].update(rep="0")))
    assert r.json()["ok"] is False, "a check of a schedule with rep 0 said it could go"


def test_a_check_lists_the_next_captures_of_the_edit(client):
    health(client)
    r = client.post("/stations/yoshkar-ola/ionozond/check", headers=CTL, json=edit(
        reported(), lambda d: (d["schedule"]["norilsk"].update(active="true"),
                               d["schedule"]["cyprus1"].update(active="false"))))
    assert {c["tx"] for c in r.json()["next"]} == {"norilsk"}, (
        "the check did not list the captures of the edited schedule")


def test_the_schedule_in_force_can_be_read(client):
    health(client)
    r = client.get("/stations/yoshkar-ola/ionozond")
    assert r.json()["sounder_config"]["base"] == reported()["base"], (
        "the schedule in force could not be read")


def test_the_schedule_in_force_comes_with_its_next_captures(client):
    health(client)
    r = client.get("/stations/yoshkar-ola/ionozond")
    assert [c["tx"] for c in r.json()["next"]][:1] == ["cyprus1"], (
        "the schedule in force came without its next captures")


def test_a_station_with_no_schedule_reported_has_none_to_read(client):
    client.post("/stations/health", headers=CTL, json={
        "station": "DOB", "timestamp": time.time(), "metrics": []})
    assert client.get("/stations/DOB/ionozond").status_code == 404, (
        "a station that reports no schedule had one to read")


def test_the_first_report_opens_no_epoch(client):
    health(client)
    assert epochs(client) == [], "the first report of a schedule opened an epoch"


def test_switching_a_transmitter_on_at_the_station_opens_an_epoch(client):
    health(client)
    health(client, reported(switched("active", "true")))
    assert [e["changed_by"] for e in epochs(client)] == ["station"], (
        "a transmitter switched on at the station opened no epoch")


def test_the_station_epoch_records_what_the_sounder_now_reads(client):
    health(client)
    health(client, reported(switched("active", "true")))
    assert "sounders.norilsk.cf" in epochs(client)[0]["changes"], (
        "the station's epoch does not record what the sounder now reads")


def test_editing_a_switched_off_transmitter_at_the_station_opens_no_epoch(client):
    health(client)
    health(client, reported(switched("chirpt", "190")))
    assert epochs(client) == [], (
        "an edit the sounder cannot see, made at the station, opened an epoch")


def test_a_change_the_queue_made_opens_no_second_epoch(client):
    health(client)
    after = reported(switched("active", "true"))
    command = queue(client, edit(reported(), lambda d: d["schedule"]["norilsk"].update(
        active="true"))).json()["id"]
    client.get("/stations/yoshkar-ola/commands", headers=CTL)
    client.post("/stations/yoshkar-ola/commands/%s/ack" % command, headers=CTL, json={
        "results": [{"command": "set_ionozond", "ok": True, "detail": "written",
                     "journal": {"changes": {"sounders.norilsk.cf": {
                         "from": None, "to": 16500e3}},
                         "config_hash": after["sounder_hash"]}}]})
    health(client, after)
    assert len(epochs(client)) == 1, "a change the queue made opened a second epoch"


def test_a_change_reported_before_its_acknowledgement_opens_no_station_epoch(client):
    health(client)
    queue(client, edit(reported(), lambda d: d["schedule"]["norilsk"].update(
        active="true")))
    client.get("/stations/yoshkar-ola/commands", headers=CTL)
    health(client, reported(switched("active", "true")))
    assert epochs(client) == [], (
        "a change still awaiting its acknowledgement opened a station epoch")


def test_a_change_after_an_earlier_queued_one_still_opens_an_epoch(client):
    health(client)
    after = reported(switched("active", "true"))
    command = queue(client, edit(reported(), lambda d: d["schedule"]["norilsk"].update(
        active="true"))).json()["id"]
    client.get("/stations/yoshkar-ola/commands", headers=CTL)
    client.post("/stations/yoshkar-ola/commands/%s/ack" % command, headers=CTL, json={
        "results": [{"command": "set_ionozond", "ok": True, "detail": "written",
                     "journal": {"changes": {"sounders.norilsk.cf": {
                         "from": None, "to": 16500e3}},
                         "config_hash": after["sounder_hash"]}}]})
    health(client, after)
    health(client)
    health(client, after)
    assert len(epochs(client)) == 3, (
        "a station change back to an old schedule was taken for the queued one")


def test_a_malformed_schedule_report_is_still_stored(client):
    health(client, ["not", "a", "schedule"])
    r = health(client, "nor this")
    assert r.status_code == 200, "a malformed schedule report lost the health push"


def test_a_change_that_is_not_an_object_is_refused(client):
    health(client)
    assert queue(client, ["cyprus1"]).status_code == 400, (
        "a set_ionozond whose params are not an object was not refused")


def drifted() -> dict:
    return reported(drift=True)


def site(**values) -> dict:
    return {"keys": {key: {"value": value, "from": "/etc/default/ionozond"}
                     for key, value in values.items()},
            "editable": ["NAS_BWLIMIT", "PRUNE_FREE_GB", "PRUNE_KEEP_DAYS"],
            "hash": "%064x" % random.getrandbits(256),
            "archive_gb": random.uniform(500.0, 4000.0)}


def command(client, name, params, station="yoshkar-ola"):
    return client.post("/stations/%s/commands" % station, headers=CTL,
                       json={"name": name, "params": params})


def test_a_regeneration_of_a_drifted_station_is_queued(client):
    health(client, drifted())
    r = command(client, "regenerate_ionozond", {
        "base": drifted()["base"], "sounder_hash": drifted()["sounder_hash"]})
    assert r.status_code == 200, "a regeneration of a drifted station was not queued"


def test_a_regeneration_of_an_undrifted_station_is_refused(client):
    health(client)
    r = command(client, "regenerate_ionozond", {
        "base": reported()["base"], "sounder_hash": reported()["sounder_hash"]})
    assert "nothing to regenerate" in r.json()["detail"], (
        "a regeneration of an undrifted station was queued")


def test_a_regeneration_against_a_moved_sounder_hash_is_refused(client):
    health(client, drifted())
    r = command(client, "regenerate_ionozond", {
        "base": drifted()["base"], "sounder_hash": "%064x" % random.getrandbits(256)})
    assert r.status_code == 400, "a regeneration against a moved file was queued"


def test_a_regeneration_without_a_health_loop_is_refused(client):
    r = command(client, "regenerate_ionozond", {"base": "x", "sounder_hash": "y"})
    assert r.status_code == 409, "a regeneration was queued with nothing to vet against"


def test_a_site_change_is_queued(client):
    shown = site(PRUNE_KEEP_DAYS="7")
    health(client, reported(site=shown))
    r = command(client, "set_ionozond_site", {"base": shown["hash"], "site": {
        "PRUNE_KEEP_DAYS": str(random.randint(1, 365))}})
    assert r.status_code == 200, "a valid site change was not queued"


@pytest.mark.parametrize("key,value", [("JUMBO", "8000"), ("PATH", "/tmp"),
                                       ("PRUNE_KEEP_DAYS", "0"),
                                       ("PRUNE_FREE_GB", "999999999")])
def test_a_site_change_the_station_would_refuse_is_refused(client, key, value):
    shown = site(PRUNE_KEEP_DAYS="7")
    health(client, reported(site=shown))
    r = command(client, "set_ionozond_site", {"base": shown["hash"],
                                              "site": {key: value}})
    assert r.status_code == 400, "a site change the station would refuse was queued"


def test_a_site_change_against_a_moved_base_is_refused(client):
    health(client, reported(site=site(NAS_BWLIMIT="0")))
    r = command(client, "set_ionozond_site", {
        "base": "%064x" % random.getrandbits(256), "site": {"NAS_BWLIMIT": "100"}})
    assert r.status_code == 400, "a site change against a moved base was queued"


def test_a_site_change_for_an_agent_without_site_settings_is_refused(client):
    health(client)
    r = command(client, "set_ionozond_site", {"base": "x",
                                              "site": {"NAS_BWLIMIT": "100"}})
    assert "update the agent" in r.json()["detail"], (
        "a site change was queued for an agent that reports no site settings")


@pytest.mark.parametrize("unit", ["ionozond-nas.service", "ionozond-prune.service"])
def test_run_now_is_queued_for_a_job_unit(client, unit):
    assert command(client, "start", {"unit": unit}).status_code == 200, (
        "run now was not queued for a job unit")


@pytest.mark.parametrize("unit", ["ionozond-sounder.service", "ssh.service", "",
                                  ["ionozond-nas.service"]])
def test_run_now_is_refused_for_any_other_unit(client, unit):
    assert command(client, "start", {"unit": unit}).status_code == 400, (
        "run now was queued for a unit that is not a job unit")


def test_a_plain_start_is_still_queued(client):
    assert command(client, "start", {}).status_code == 200, (
        "a start without a unit was no longer queued")
