"""A schedule change on an ionozond station: checked, written, taken up, reported."""

from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

import pytest

from services.agent import client, control, health, ionozond, runner
from services.agent.config import StationConfig
from services.agent.qsettings import Ini
from services.agent.sounderconfig import build

DATA = Path(__file__).resolve().parent / "ionozond"

SCHEDULE = (b"[cyprus1]\nactive=true\nrep=300\nchirpt=10\nrate=100\ndur=250\ncf=20000\n"
            b"lat=35\nlon=34\nrx=false\n\n"
            b"[norilsk]\nactive=false\nrep=300\nchirpt=180\nrate=500\ndur=50\n"
            b"cf=16500\nlat=69.2\nlon=88.13\nrx=false\nfft_count=4096\n\n"
            b"[yoshkar-ola]\nactive=false\nrep=60\nchirpt=0\nrate=50\ndur=260\n"
            b"cf=13500\nlat=56.38\nlon=47.53\nrx=true\n")


def laptop(tmp_path) -> StationConfig:
    console = tmp_path / "dsChirp"
    console.mkdir()
    sounder = tmp_path / "chirp_config.py"
    (console / "schedule.ini").write_bytes(SCHEDULE)
    (console / "config.ini").write_bytes(
        (DATA / "fixture_config.ini").read_bytes().replace(
            b"config_file=/tmp/out/chirp_config.py",
            b"config_file=" + sounder.as_posix().encode()))
    defaults = tmp_path / "ionozond"
    defaults.write_bytes(b"RADIO_ADDR=192.168.10.2\nJUMBO=4000\nIONOZOND_CONFIG="
                         + sounder.as_posix().encode() + b"\n")
    regenerate(console, sounder)
    then = time.time() - 7200
    os.utime(sounder, (then, then))
    return StationConfig(station="yoshkar-ola", chirp_config=tmp_path / "absent.ini",
                         live_dir=tmp_path, target="ionozond-sounder.service",
                         units=("ionozond-sounder.service",), no_block=True,
                         sounder_defaults=defaults, console_settings=console)


def regenerate(console: Path, sounder: Path) -> None:
    sounder.write_bytes(build(Ini.read(console / "schedule.ini"),
                              Ini.read(console / "config.ini")).encode("utf-8"))


def systemd(monkeypatch, state="active", accept=True, timeout="5min",
            started=None) -> list:
    calls = []
    props = {"ActiveState": state, "TimeoutStopUSec": timeout,
             "ActiveEnterTimestampMonotonic": str(int(
                 (time.monotonic() - 3600 if started is None else started) * 1e6))}
    monkeypatch.setattr(health, "_show", lambda unit, prop: (
        (0, props[prop]) if prop in props else (1, "")))
    monkeypatch.setattr(control, "systemctl", lambda verb, target, **kw: (
        calls.append((verb, target, kw.get("no_block")))
        or control.CommandResult("%s %s" % (verb, target), accept,
                                 "" if accept else "Interactive authentication "
                                                   "required.")))
    return calls


def request(config, change=None) -> dict:
    now = ionozond.state(config)
    document = copy.deepcopy(now["document"])
    if change:
        change(document)
    return dict(document, base=now["base"])


def files(config) -> dict:
    return {p.name: p.read_bytes() for p in
            (config.console_settings / "schedule.ini",
             config.console_settings / "config.ini",
             config.live_dir / "chirp_config.py")}


def test_the_editable_document_holds_a_switched_off_transmitter(tmp_path, monkeypatch):
    systemd(monkeypatch)
    assert "norilsk" in ionozond.state(laptop(tmp_path))["document"]["schedule"], (
        "a switched-off transmitter is missing from the editable document")


def test_the_editable_document_leaves_out_the_receiver(tmp_path, monkeypatch):
    systemd(monkeypatch)
    assert "yoshkar-ola" not in ionozond.state(laptop(tmp_path))["document"][
        "schedule"], "the receiver can be edited as a transmitter"


def test_the_receiver_is_reported_beside_the_document(tmp_path, monkeypatch):
    systemd(monkeypatch)
    receivers = ionozond.state(laptop(tmp_path))["receivers"]
    assert [r["name"] for r in receivers] == ["yoshkar-ola"], (
        "the receiver was not reported")


def test_a_freshly_generated_file_has_not_drifted(tmp_path, monkeypatch):
    systemd(monkeypatch)
    assert ionozond.state(laptop(tmp_path))["drift"] is False, (
        "a file generated from the console's settings read as drifted")


def test_a_file_unlike_what_the_settings_generate_has_drifted(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    path = config.live_dir / "chirp_config.py"
    path.write_bytes(path.read_bytes().replace(b"'chirpt':10", b"'chirpt':20"))
    assert ionozond.state(config)["drift"] is True, (
        "a chirp_config.py the settings do not generate did not read as drifted")


def test_editing_a_switched_off_transmitter_changes_the_document_hash(
        tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    before = ionozond.state(config)["base"]
    ini = config.console_settings / "schedule.ini"
    ini.write_bytes(Ini.read(ini).with_value("norilsk", "chirpt", "190").data)
    assert ionozond.state(config)["base"] != before, (
        "an edit of a switched-off transmitter left the document hash alone")


def test_editing_a_switched_off_transmitter_leaves_the_sounder_hash(
        tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    before = ionozond.state(config)["sounder_hash"]
    ini = config.console_settings / "schedule.ini"
    ini.write_bytes(Ini.read(ini).with_value("norilsk", "chirpt", "190").data)
    regenerate(config.console_settings, config.live_dir / "chirp_config.py")
    assert ionozond.state(config)["sounder_hash"] == before, (
        "an edit the sounder cannot see changed the sounder hash")


def test_switching_a_transmitter_on_changes_the_sounder_hash(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    before = ionozond.state(config)["sounder_hash"]
    ini = config.console_settings / "schedule.ini"
    ini.write_bytes(Ini.read(ini).with_value("norilsk", "active", "true").data)
    regenerate(config.console_settings, config.live_dir / "chirp_config.py")
    assert ionozond.state(config)["sounder_hash"] != before, (
        "switching a transmitter on left the sounder hash alone")


def test_a_file_written_after_the_sounder_started_is_pending(tmp_path, monkeypatch):
    config = laptop(tmp_path)
    os.utime(config.live_dir / "chirp_config.py", None)
    systemd(monkeypatch, started=time.monotonic() - 600)
    assert ionozond.state(config)["pending"] is True, (
        "a file written after the sounder started was not pending")


def test_a_file_older_than_the_running_sounder_is_not_pending(tmp_path, monkeypatch):
    systemd(monkeypatch, started=time.monotonic() - 60)
    assert ionozond.state(laptop(tmp_path))["pending"] is False, (
        "a file the running sounder already read was pending")


def test_the_unit_stop_timeout_is_read_in_seconds(tmp_path, monkeypatch):
    systemd(monkeypatch, timeout="4min 30s")
    assert ionozond.state(laptop(tmp_path))["timeout_stop_s"] == 270.0, (
        "the unit's stop timeout was not read in seconds")


def test_a_change_writes_its_value_into_the_schedule(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    chirpt = str(random.randint(11, 40))
    ionozond.apply(config, request(config, lambda d: d["schedule"]["cyprus1"].update(
        chirpt=chirpt)))
    ini = Ini.read(config.console_settings / "schedule.ini")
    assert ini.value("cyprus1", "chirpt") == chirpt, (
        "the change did not reach schedule.ini")


def test_a_change_regenerates_chirp_config_as_the_console_would(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    ionozond.apply(config, request(config, lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    want = build(Ini.read(config.console_settings / "schedule.ini"),
                 Ini.read(config.console_settings / "config.ini"))
    assert (config.live_dir / "chirp_config.py").read_text(encoding="utf-8") == want, (
        "chirp_config.py was not regenerated from the edited settings")


def test_a_change_touches_only_the_line_it_edits(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    ionozond.apply(config, request(config, lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    after = (config.console_settings / "schedule.ini").read_bytes()
    assert after == SCHEDULE.replace(b"chirpt=10", b"chirpt=20"), (
        "a change touched more of schedule.ini than its own line")


def test_a_change_asks_for_a_try_restart_without_waiting(tmp_path, monkeypatch):
    calls = systemd(monkeypatch)
    config = laptop(tmp_path)
    ionozond.apply(config, request(config, lambda d: d["general"].update(
        keep_lfs="true")))
    assert calls == [("try-restart", "ionozond-sounder.service", True)], (
        "the change was not taken up by a try-restart without waiting")


def test_a_refused_restart_puts_every_file_back(tmp_path, monkeypatch):
    systemd(monkeypatch, accept=False)
    config = laptop(tmp_path)
    before = files(config)
    ionozond.apply(config, request(config, lambda d: d["schedule"]["norilsk"].update(
        active="true")))
    assert files(config) == before, "a refused restart left the new files in place"


def test_a_refused_restart_journals_nothing(tmp_path, monkeypatch):
    systemd(monkeypatch, accept=False)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "norilsk"].update(active="true")))
    assert not any(r.journal for r in results), "a rolled-back change was journalled"


def test_editing_a_switched_off_transmitter_opens_no_epoch(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "norilsk"].update(chirpt="190")))
    assert "changes" not in results[0].journal, (
        "an edit the sounder cannot see would open an epoch")


def test_switching_a_transmitter_on_opens_an_epoch(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "norilsk"].update(active="true")))
    assert "sounders.norilsk.cf" in results[0].journal["changes"], (
        "switching a transmitter on would open no epoch")


def test_the_journal_carries_the_hash_of_what_the_sounder_reads(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "cyprus1"].update(chirpt="20")))
    now = ionozond.state(config)["sounder_hash"]
    assert results[0].journal["config_hash"] == now, (
        "the journal does not carry the hash of what the sounder now reads")


def test_a_change_against_a_stale_document_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    wanted = request(config, lambda d: d["schedule"]["cyprus1"].update(chirpt="20"))
    ini = config.console_settings / "schedule.ini"
    ini.write_bytes(Ini.read(ini).with_value("norilsk", "chirpt", "190").data)
    assert ionozond.apply(config, wanted)[0].ok is False, (
        "a change made against a schedule that has since changed was applied")


def test_a_change_on_a_drifted_station_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    wanted = request(config, lambda d: d["schedule"]["cyprus1"].update(chirpt="20"))
    path = config.live_dir / "chirp_config.py"
    path.write_bytes(path.read_bytes().replace(b"'chirpt':10", b"'chirpt':30"))
    assert "regenerate" in ionozond.apply(config, wanted)[0].detail, (
        "a drifted station took a change without saying the console must regenerate")


def test_a_schedule_the_checker_calls_wrong_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    before = files(config)
    ionozond.apply(config, request(config, lambda d: d["schedule"]["cyprus1"].update(
        rep="0")))
    assert files(config) == before, "a schedule with rep 0 was written"


def test_a_sounding_that_outlasts_the_stop_timeout_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch, timeout="2min")
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "cyprus1"].update(chirpt="20")))
    assert "TimeoutStopSec" in results[0].detail, (
        "a 250 s sounding against a 120 s stop timeout was not refused")


def test_a_missing_transmitter_is_refused_not_deleted(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"].pop(
        "norilsk")))
    assert results[0].ok is False, "a transmitter left out of the document was deleted"


def test_a_new_transmitter_with_a_dash_in_its_name_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    entry = {"active": "false", "cf": "16500", "rate": "500", "dur": "50",
             "rep": "300", "chirpt": "0", "lat": "60", "lon": "150.7"}
    results = ionozond.apply(config, request(config, lambda d: d["schedule"].update(
        {"magadan-2": entry})))
    assert results[0].ok is False, "a transmitter name with a dash was accepted"


def test_a_new_transmitter_is_added_as_a_transmitter(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    entry = {"active": "false", "cf": "16500", "rate": "500", "dur": "50",
             "rep": "300", "chirpt": "0", "lat": "60", "lon": "150.7"}
    ionozond.apply(config, request(config, lambda d: d["schedule"].update(
        {"magadan": entry})))
    ini = Ini.read(config.console_settings / "schedule.ini")
    assert ini.value("magadan", "rx") == "false", "a new transmitter was not added"


def test_a_new_transmitter_without_every_field_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"].update(
        {"magadan": {"active": "false", "cf": "16500"}})))
    assert results[0].ok is False, "a transmitter without its timing was added"


def test_the_receiver_cannot_be_edited(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"].update(
        {"yoshkar-ola": {"lat": "56.4"}})))
    assert results[0].ok is False, "the receiver was edited as a transmitter"


def test_a_field_outside_the_list_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["general"].update(
        nas_dest="user@host:/volume1")))
    assert results[0].ok is False, "nas_dest was set from the web"


def test_a_console_writing_another_file_refuses_the_change(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    ini = config.console_settings / "config.ini"
    ini.write_bytes(ini.read_bytes().replace(
        b"config_file=" + (config.live_dir / "chirp_config.py").as_posix().encode(),
        b"config_file=/tmp/out/chirp_config.py"))
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "cyprus1"].update(chirpt="20")))
    assert results[0].ok is False, (
        "a change was written while the console configures another file")


def test_a_lock_the_console_holds_refuses_the_change(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    monkeypatch.setattr(ionozond, "_alive", lambda pid: True)
    lock = config.console_settings / "schedule.ini.lock"
    lock.write_bytes(b"4242\ndsChirp\nhost\n")
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "cyprus1"].update(chirpt="20")))
    assert results[0].ok is False, "a change was written under the console's lock"


def test_a_lock_left_by_a_dead_console_is_cleared(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    monkeypatch.setattr(ionozond, "_alive", lambda pid: False)
    lock = config.console_settings / "schedule.ini.lock"
    lock.write_bytes(b"4242\ndsChirp\nhost\n")
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "cyprus1"].update(chirpt="20")))
    assert results[0].ok is True, "a lock left by a dead console blocked the change"


def test_the_agent_releases_its_locks_after_writing(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    ionozond.apply(config, request(config, lambda d: d["schedule"]["cyprus1"].update(
        chirpt="20")))
    assert not list(config.console_settings.glob("*.lock")), (
        "the agent left its lock files behind")


def test_the_checkers_warnings_travel_in_the_journal(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    results = ionozond.apply(config, request(config, lambda d: d["schedule"][
        "norilsk"].update(active="true", chirpt="100")))
    assert any("norilsk" in w for w in results[0].journal["warnings"]), (
        "the checker's warnings did not reach the journal")


def test_a_change_that_changes_nothing_is_refused(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    assert ionozond.apply(config, request(config))[0].ok is False, (
        "a change that changes nothing was applied")


def test_the_agent_dispatches_set_ionozond_to_the_sounder(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ionozond, "apply", lambda config, params: seen.append(
        params) or [])
    command = client.Command("1", "set_ionozond", {"base": "x"})
    runner._dispatch(laptop(tmp_path), command)
    assert seen == [{"base": "x"}], "set_ionozond did not reach the sounder's apply"


def test_set_config_on_an_ionozond_station_is_refused_by_name(tmp_path):
    results = runner._dispatch(laptop(tmp_path), client.Command(
        "1", "set_config", {"changes": {"mode": "search"}}))
    assert "set_ionozond" in results[0].detail, (
        "set_config on an ionozond station was not pointed at set_ionozond")


def test_an_ionozond_health_report_carries_the_schedule_in_force(tmp_path, monkeypatch):
    systemd(monkeypatch)
    monkeypatch.setattr(health, "_run", lambda args, timeout=5.0: (
        (0, "active") if args[:2] == ["systemctl", "is-active"]
        else (127, "%s: not found" % args[0])))
    report = health.collect(laptop(tmp_path))
    assert "sounder_config" in report.to_json(), (
        "an ionozond health report does not carry the schedule in force")


@pytest.mark.parametrize("text,seconds", [("5min", 300.0), ("1min 30s", 90.0),
                                          ("300ms", 0.3), ("1h", 3600.0)])
def test_a_systemd_time_span_reads_in_seconds(text, seconds):
    assert ionozond._seconds(text) == seconds, "%s was not read in seconds" % text


def test_an_infinite_time_span_has_no_seconds():
    assert ionozond._seconds("infinity") is None, "an infinite span read as a number"


def test_a_quoted_value_in_the_defaults_file_reads_without_quotes(tmp_path):
    path = tmp_path / "ionozond"
    path.write_bytes(b'# site\nIONOZOND_CONFIG="/home/samsung/chirp_config.py"\n')
    named = ionozond._defaults(path)["IONOZOND_CONFIG"]
    assert named == "/home/samsung/chirp_config.py", (
        "a quoted value in the defaults file kept its quotes")


def described(schedule: bytes = SCHEDULE) -> dict:
    read = Ini((DATA / "fixture_config.ini").read_bytes())
    return ionozond.described(Ini(schedule), read, build(Ini(schedule), read))


def test_a_described_station_has_not_drifted():
    assert described()["drift"] is False, "a freshly generated station read as drifted"


def test_vetting_refuses_a_drifted_station():
    reported = dict(described(), drift=True)
    with pytest.raises(control.ControlError, match="regenerate"):
        ionozond.vetted(dict(reported["document"], base=reported["base"]), reported)


def test_vetting_refuses_a_stale_base():
    reported = described()
    with pytest.raises(control.ControlError, match="changed on the station"):
        ionozond.vetted(dict(reported["document"], base="0" * 64), dict(
            reported, timeout_stop_s=300.0))


def test_vetting_returns_the_checkers_warnings():
    reported = dict(described(), timeout_stop_s=300.0)
    wanted = copy.deepcopy(reported["document"])
    wanted["schedule"]["norilsk"].update(active="true", chirpt="100")
    found = ionozond.vetted(dict(wanted, base=reported["base"]), reported)[1]
    assert any(i.station == "norilsk" and "loses" in i.text for i in found), (
        "vetting did not return the collision the checker finds")


def test_the_next_captures_are_of_switched_on_transmitters_only():
    midnight = 86400.0 * random.randint(20000, 21000)
    names = {c["tx"] for c in ionozond.captures(described()["document"], midnight, 6)}
    assert names == {"cyprus1"}, "a switched-off transmitter was given captures"


def test_the_first_capture_after_midnight_is_at_its_chirptime():
    midnight = 86400.0 * random.randint(20000, 21000)
    first = ionozond.captures(described()["document"], midnight, 1)[0]
    assert first["start"] == midnight + 10, (
        "the first capture after midnight was not at its chirptime")
