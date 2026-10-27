"""The ionozond schedule on the server: vetted, shown, and journalled."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from services.agent import ionozond, schedulecheck
from services.agent.qsettings import Ini
from services.agent.sounderconfig import build
from services.api import db, sounder

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


def waited(client) -> None:
    conn = client.app.state.db
    conn.execute("UPDATE command SET delivered_at = '2026-01-01T00:00:00Z'"
                 " WHERE delivered_at IS NOT NULL")
    conn.commit()


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


def test_an_unacknowledged_change_opens_an_epoch_after_the_wait(client):
    health(client)
    health(client, reported(switched("active", "true")))
    queue(client, edit(reported(switched("active", "true")), lambda d: d["schedule"][
        "cyprus1"].update(chirpt=str(random.randint(11, 40)))))
    client.get("/stations/yoshkar-ola/commands", headers=CTL)
    changed = reported(Ini(switched("active", "true")).with_value(
        "cyprus1", "chirpt", "45").data)
    health(client, changed)
    waited(client)
    health(client, changed)
    assert epochs(client)[-1]["config_hash"] == changed["sounder_hash"], (
        "a change whose acknowledgement never came opened no epoch")


def test_a_report_of_the_hash_the_epoch_holds_opens_none(client):
    health(client)
    health(client, reported(switched("active", "true")))
    health(client, reported(switched("active", "true")))
    health(client, reported(switched("active", "true")))
    assert len(epochs(client)) == 1, "a report of the epoch's own hash opened another"


def test_a_regeneration_awaiting_its_acknowledgement_opens_no_station_epoch(client):
    health(client, drifted())
    command(client, "regenerate_ionozond", {
        "base": drifted()["base"], "sounder_hash": drifted()["sounder_hash"]})
    client.get("/stations/yoshkar-ola/commands", headers=CTL)
    health(client, reported(switched("active", "true")))
    assert epochs(client) == [], (
        "a regeneration still awaiting its acknowledgement opened a station epoch")


def test_the_epoch_for_an_unacknowledged_change_records_what_changed(client):
    health(client)
    health(client, reported(switched("active", "true")))
    queue(client, edit(reported(switched("active", "true")), lambda d: d["schedule"][
        "cyprus1"].update(chirpt=str(random.randint(11, 40)))))
    client.get("/stations/yoshkar-ola/commands", headers=CTL)
    changed = reported(Ini(switched("active", "true")).with_value(
        "cyprus1", "chirpt", "45").data)
    health(client, changed)
    waited(client)
    health(client, changed)
    assert "cyprus1" in epochs(client)[-1]["changes"], (
        "the epoch for an unacknowledged change did not record what changed")


def test_a_short_report_keeps_the_last_whole_block(client):
    health(client)
    health(client, ionozond.summary(reported()))
    r = client.get("/stations/yoshkar-ola/ionozond")
    assert r.json()["sounder_config"]["document"] == reported()["document"], (
        "a short report lost the last whole block")


def test_a_change_is_vetted_against_the_kept_block(client):
    health(client)
    health(client, ionozond.summary(reported()))
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt=str(random.randint(11, 40)))))
    assert r.status_code == 200, "a change was not vetted against the kept block"


def test_a_short_report_that_disagrees_with_the_kept_block_refuses_a_change(client):
    health(client)
    health(client, dict(ionozond.summary(reported()),
                        sounder_hash="%064x" % random.getrandbits(256)))
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    assert r.status_code == 409, "a change was vetted against an outdated block"


def test_a_short_report_with_no_kept_block_refuses_a_change(client):
    health(client, ionozond.summary(reported()))
    r = queue(client, edit(reported(), lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    assert r.status_code == 409, "a change was vetted with no whole block kept"


def test_the_schedule_read_says_when_its_block_is_outdated(client):
    health(client)
    health(client, dict(ionozond.summary(reported()), pending=True))
    r = client.get("/stations/yoshkar-ola/ionozond")
    assert r.json()["sounder_config"]["stale"] is True, (
        "the schedule read did not say its block is outdated")


def test_a_station_change_after_short_reports_records_what_changed(client):
    health(client)
    health(client, ionozond.summary(reported()))
    health(client, reported(switched("active", "true")))
    assert "sounders.norilsk.cf" in epochs(client)[0]["changes"], (
        "a change after short reports did not record what the sounder now reads")


def test_forgetting_a_station_drops_its_kept_block(client):
    health(client)
    db.forget_station(client.app.state.db, "yoshkar-ola")
    assert db.block(client.app.state.db, "yoshkar-ola") is None, (
        "forgetting a station kept its schedule block")


def slot(**change) -> dict:
    return dict({"chirp-rate": 100000, "rep": 300, "chirpt": 235}, **change)


def seed(client, **change):
    db.save_transmitter(client.app.state.db, "Yoshkar-Ola", "NIC1", [slot()])
    return client.post("/stations/yoshkar-ola/ionozond/seed", headers=CTL, json=dict(
        {"from": "Yoshkar-Ola", "code": "NIC1", "slot": 0, "low_edge_khz": 7500},
        **change))


def test_seeding_converts_a_verified_slot_into_the_group_the_hand_made_schedule_has():
    assert sounder.seed("NIC1", slot(), 7500, reported())["group"]["chirpt"] == "10", (
        "NIC1 verified at 235 s did not become the hand-made group's chirpt of 10 s")


def test_seeding_moves_chirpt_to_the_band_edge_within_the_cycle():
    rep, rate = random.choice([60, 120, 300, 600]), random.choice([50, 100, 125, 500])
    chirpt, edge = random.randint(0, rep - 1), random.randint(1000, 20000)
    found = sounder.seed("NIC1", slot(**{"chirp-rate": rate * 1000, "rep": rep,
                                         "chirpt": chirpt}), edge, reported())
    assert float(found["group"]["chirpt"]) == pytest.approx(
        (chirpt + edge / rate) % rep, abs=1e-3), (
        "chirpt was not moved to the moment the sweep reaches the band edge")


def test_seeding_puts_cf_half_the_sample_rate_above_the_low_edge():
    edge = random.randint(2000, 15000)
    assert float(sounder.seed("NIC1", slot(), edge, reported())["group"]["cf"]) == (
        edge + 12500), "cf is not half the 25 MHz sample rate above the low edge"


def test_seeding_gives_the_rate_in_khz_per_second():
    rate = random.randint(10, 900)
    assert float(sounder.seed("NIC1", slot(**{"chirp-rate": rate * 1000}), 7500,
                              reported())["group"]["rate"]) == rate, (
        "the chirp rate was not converted from Hz/s to kHz/s")


def test_seeding_keeps_the_verified_cycle():
    rep = random.choice([60, 120, 300, 600, 900])
    assert float(sounder.seed("NIC1", slot(rep=rep), 7500, reported())["group"][
        "rep"]) == rep, "the seeded group did not keep the verified rep"


def test_seeding_listens_across_the_whole_receive_band():
    assert sounder.seed("NIC1", slot(**{"chirp-rate": 500000}), 7500, reported())[
        "group"]["dur"] == "50", "25 MHz at 500 kHz/s was not a 50 s sounding"


def test_seeding_caps_the_sounding_so_it_fits_the_stop_timeout():
    limit = random.randint(40, 200)
    dur = float(sounder.seed("NIC1", slot(), 7500, dict(
        reported(), timeout_stop_s=float(limit)))["group"]["dur"])
    assert dur + schedulecheck.processing_estimate_s(dur) <= limit, (
        "a seeded sounding and its products outlast the unit's TimeoutStopSec")


def test_seeding_says_when_it_capped_the_sounding():
    assert sounder.seed("NIC1", slot(), 7500, dict(reported(), timeout_stop_s=60.0))[
        "capped"] is True, "a sounding cut short to fit TimeoutStopSec was not said so"


def test_seeding_leaves_the_group_switched_off():
    assert sounder.seed("NIC1", slot(), 7500, reported())["group"][
        "active"] == "false", (
        "a seeded group was switched on before the checker showed its collisions")


def test_seeding_names_the_group_after_the_code():
    code = "X%d" % random.randint(1, 9999)
    assert sounder.seed(code, slot(), 7500, reported())["name"] == code, (
        "the seeded group was not named after the verified code")


def test_seeding_places_the_transmitter_from_the_station_registry():
    assert float(sounder.seed("NIC1", slot(), 7500, reported())["group"]["lat"]) == (
        35.18557), "NIC1 was not placed at Nicosia from muf/stations.py"


def test_seeding_says_when_the_registry_cannot_place_the_code():
    assert sounder.seed("QZX%d" % random.randint(1, 999), slot(), 7500, reported())[
        "located"] is False, "a code the registry cannot resolve was called located"


def test_an_unplaced_transmitter_is_seeded_without_coordinates():
    assert sounder.seed("QZX%d" % random.randint(1, 999), slot(), 7500, reported())[
        "group"]["lon"] == "", "a code the registry cannot resolve got a longitude"


def test_the_seed_route_converts_a_slot_verified_at_another_station(client):
    health(client)
    assert seed(client).json()["group"]["chirpt"] == "10", (
        "the seed route did not convert NIC1, verified at Yoshkar-Ola, to chirpt 10")


def test_the_seed_route_refuses_a_transmitter_nobody_verified(client):
    health(client)
    assert "QZX1" in seed(client, code="QZX1").json()["detail"], (
        "the refusal of a transmitter nobody verified did not name it")


def test_the_seed_route_refuses_a_slot_the_transmitter_lacks(client):
    health(client)
    index = random.randint(1, 9)
    assert "slot %d" % index in seed(client, slot=index).json()["detail"], (
        "the refusal of a slot the verified transmitter lacks did not name it")


def test_the_seed_route_refuses_a_negative_low_edge(client):
    health(client)
    assert seed(client, low_edge_khz=-random.randint(1, 9000)).status_code == 400, (
        "a band below 0 Hz was seeded")


def test_the_seed_route_refuses_a_low_edge_that_is_no_number(client):
    health(client)
    assert seed(client, low_edge_khz="seven").status_code == 400, (
        "a low edge that is no number was seeded")


def test_the_seed_route_refuses_a_station_without_a_health_loop(client):
    assert seed(client).status_code == 409, (
        "a group was seeded with no reported sample rate to size it by")


def test_the_seed_route_asks_for_the_station_capability(client):
    health(client)
    db.save_transmitter(client.app.state.db, "Yoshkar-Ola", "NIC1", [slot()])
    assert client.post("/stations/yoshkar-ola/ionozond/seed", json={
        "from": "Yoshkar-Ola", "code": "NIC1", "slot": 0,
        "low_edge_khz": 7500}).status_code == 401, (
        "an anonymous caller could seed a group")


# --- planned against recorded --------------------------------------------------

def summarised(conn, t0, tx="cyprus1", station="yoshkar-ola"):
    db.keep_summary(conn, station, {
        "tx": tx, "t0": t0, "source": "live", "f_lo_mhz": 7.5, "f_step_mhz": 0.04,
        "columns": 1, "range_lo_km": 2500.0, "range_step_km": 7.3, "rows": 1,
        "km_per_ms": 300.0, "snr": "AA==", "pdp": "AA=="})


def tallied(client, starts, now, config=None, epoch=None, station="yoshkar-ola"):
    """The tally after a summary at each of ``starts``: cyprus1 is planned at
    second 10 of every 300 s, for 250 s."""
    conn = client.app.state.db
    for t0 in starts:
        summarised(conn, t0)
    found = sounder.tally(conn, station, config or reported(), epoch, now)
    return {row["tx"]: row for row in found["rows"]}, found


def midday() -> float:
    return 86400.0 * random.randint(20700, 20800) + 43200.0


def planned(now, since=None) -> list:
    since = now - sounder.TALLY_S if since is None else since
    return [c["start"] for c in ionozond.planned(reported()["document"], since, now)
            if c["start"] + c["dur"] + sounder.DUE_S <= now]


def test_a_tally_counts_every_planned_capture_that_was_recorded(client):
    now = midday()
    starts = planned(now)
    rows, _ = tallied(client, [int(t) for t in starts], now)
    assert (rows["cyprus1"]["planned"], rows["cyprus1"]["recorded"]) == (
        len(starts), len(starts)), (
        "a day of captures recorded as planned did not count as recorded")


def test_a_tally_counts_a_planned_capture_with_no_summary_as_unseen(client):
    now = midday()
    starts = planned(now)
    gone = random.randrange(len(starts))
    rows, _ = tallied(client, [int(t) for k, t in enumerate(starts) if k != gone], now)
    assert (rows["cyprus1"]["unseen"], rows["cyprus1"]["unseen_at"]) == (
        1, [starts[gone]]), "a capture with no summary was not counted unseen"


def test_a_tally_counts_a_summary_off_the_schedule_as_unplanned(client):
    now = midday()
    rows, _ = tallied(client, [int(planned(now)[0]) + 150], now)
    assert rows["cyprus1"]["unplanned"] == 1, (
        "a summary halfway between two planned captures was not unplanned")


def test_a_tally_does_not_count_a_capture_still_running(client):
    running = midday() + 10
    rows, _ = tallied(client, [], running + random.randint(0, 200))
    assert running not in rows["cyprus1"]["unseen_at"], (
        "a capture still being recorded was counted unseen")


def test_a_tally_says_how_far_recorded_starts_are_from_planned_ones(client):
    now, late = midday(), random.randint(1, int(sounder.MATCH_S))
    rows, _ = tallied(client, [int(t) + late for t in planned(now)], now)
    assert rows["cyprus1"]["offset_s"] == late, (
        "the tally did not say how late the recorded soundings started")


def test_a_tally_begins_when_the_schedule_last_changed(client):
    now, hours = midday(), random.randint(1, 6)
    changed = now - 3600 * hours
    epoch = {"valid_from": time.strftime(db.TIME_FORMAT, time.gmtime(changed))}
    rows, found = tallied(client, [], now, epoch=epoch)
    assert (found["since"], rows["cyprus1"]["planned"]) == (
        changed, len(planned(now, changed))), (
        "the tally counted captures from before the schedule last changed")


def test_a_tally_ignores_soundings_from_before_the_schedule_changed(client):
    now = midday()
    changed = now - 3600
    epoch = {"valid_from": time.strftime(db.TIME_FORMAT, time.gmtime(changed))}
    rows, _ = tallied(client, [int(changed) - 300 * random.randint(1, 9) - 150], now,
                      epoch=epoch)
    assert rows["cyprus1"]["unplanned"] == 0, (
        "a sounding from before the schedule changed counted as unplanned")


def test_a_tally_keeps_to_its_station(client):
    now = midday()
    for t0 in planned(now):
        summarised(client.app.state.db, int(t0), station="yoshkar-ola-1")
    rows, _ = tallied(client, [], now)
    assert rows["cyprus1"]["recorded"] == 0, (
        "the tally counted another station's soundings")


@pytest.mark.parametrize("flag", ["drift", "pending", "stale"])
def test_a_tally_against_a_schedule_the_sounder_does_not_run_says_so(client, flag):
    _, found = tallied(client, [], midday(), config=dict(reported(), **{flag: True}))
    assert found["comparable"] is False, (
        "a tally against a %s schedule claimed to be comparable" % flag)


def test_the_schedule_route_carries_the_tally(client):
    health(client)
    found = client.get("/stations/yoshkar-ola/ionozond").json()["tally"]
    assert "cyprus1" in {row["tx"] for row in found["rows"]}, (
        "the schedule route carried no planned-and-recorded counts")
