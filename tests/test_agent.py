"""The station agent: health push, narrow control, logs.

``architecture.md`` sec. 2.5 and 5.4. The tests worth having are about the
promises that keep a station recoverable: a collector that cannot measure
something must say so rather than report zero, a stop must never become a
kill, and a config edit must never land half-written.
"""

from __future__ import annotations

import configparser
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from services.agent import client, control, health, logs, runner
from services.agent.config import StationConfig

#: Read from the module so the tests move with the threshold, not against it.
STALE = health.STALE_PRODUCT_S


@pytest.fixture
def station(tmp_path) -> StationConfig:
    ini = tmp_path / "my_station.ini"
    ini.write_text(
        "[config]\n"
        "sample_rate = 25e6\n"
        'output_dir = "%s"\n'
        "[lfm]\n"
        "serendipitous = true\n"
        "sounder_timings = []\n"
        "max_range_extent = 4000e3\n"
        "save_raw_voltage = false\n" % tmp_path.as_posix(),
        encoding="utf-8")
    (tmp_path / "data").mkdir()
    return StationConfig(station="TST", chirp_config=ini,
                         output_dir=tmp_path / "data",
                         ringbuffer_dir=tmp_path)


def fake_runner(returncode=0, stdout="", stderr=""):
    def _run(args, **kw):
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)
    return _run


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

def test_a_metric_that_cannot_be_measured_is_unknown_not_zero(tmp_path):
    """The distinction the whole module is built on.

    "Zero soundings in the last hour" and "could not tell how many soundings"
    need different responses, and conflating them pages someone for the wrong
    thing.
    """
    config = StationConfig(output_dir=tmp_path / "nope",
                           chirp_config=tmp_path / "nope.ini")
    metric = health.newest_product_age(config)

    assert metric.value is None
    assert metric.ok is None            # not False
    assert "no such directory" in metric.detail


def test_no_products_at_all_is_a_failure_not_an_unknown(station):
    """An empty output directory is measurable, and the answer is bad."""
    metric = health.newest_product_age(station)
    assert metric.ok is False
    assert "no products" in metric.detail


def test_a_future_dated_product_is_unknown_not_ok(station):
    """DOB reported `ok` at -20420 s, because every negative age beats 900.

    The clock had slipped 4.8 hours behind the products on disk, so the one
    metric watching for acquisition stopping was passing unconditionally --
    and would have kept passing with the recorder dead.
    """
    product = Path(station.output_dir) / "lfm_ionogram-DOB-000.h5"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"")
    os.utime(product, (time.time() + 20420, time.time() + 20420))

    metric = health.newest_product_age(station)
    assert metric.ok is None                     # not True, and not False
    assert "in the future" in metric.detail
    assert "system_clock_s" in metric.detail


def test_small_clock_skew_is_still_measured(station):
    """A second or two ahead is ordinary mtime skew, not a broken clock."""
    product = Path(station.output_dir) / "lfm_ionogram-DOB-001.h5"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"")
    os.utime(product, (time.time() + 2, time.time() + 2))

    metric = health.newest_product_age(station)
    assert metric.ok is True
    assert metric.value < 0


def _sounding(station, t0, mtime=None):
    """A product named the way the recorder names them, with its own t0."""
    path = (Path(station.output_dir)
            / f"lfm_ionogram-unkown-DOB-ch0-000-{t0:.2f}.h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_product_age_reads_the_filename_not_the_stamp(station):
    """DOB's mtimes ran 5 h 36 m ahead while the data time was correct.

    t0 is on the recorder's GPS-disciplined epoch; mtime belongs to whichever
    clock touched the file last -- an RTC stuck in 2021, a fast CIFS server,
    or the stamps left behind after that server was fixed. Only one of the two
    answers "when did we last hear the ionosphere".
    """
    now = time.time()
    _sounding(station, t0=now - 600, mtime=now + 20565)

    metric = health.newest_product_age(station)
    assert metric.ok is True, "a correct 10-minute-old sounding is not a failure"
    assert 590 < metric.value < 615
    assert "filename" in metric.detail


def test_a_sounding_past_the_threshold_still_fails(station):
    """Reading t0 must not cost the detection this metric exists for."""
    now = time.time()
    _sounding(station, t0=now - 4 * STALE, mtime=now)

    metric = health.newest_product_age(station)
    assert metric.ok is False
    assert metric.value > STALE


def test_pipeline_latency_alone_does_not_trip_the_threshold(station):
    """DOB emits products ~960 s after t0, which the old 900 s would fail."""
    now = time.time()
    _sounding(station, t0=now - 1260, mtime=now)      # latency + one 300 s cycle

    assert health.newest_product_age(station).ok is True


def _future_product(station, name="lfm_ionogram-DOB-002.h5", ahead=20565):
    path = Path(station.output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    os.utime(path, (time.time() + ahead, time.time() + ahead))
    return path


def test_a_network_archive_is_not_evidence_about_our_clock(station, monkeypatch):
    """DOB's products moved to CIFS, and the metric blamed the host anyway.

    The NAS was 5 h 43 m fast while the laptop sat 47 ms from its NTP server.
    Worse than the wrong words: the early return also skipped the NTP check,
    so the one question that *is* about this host went unasked.
    """
    _future_product(station)
    monkeypatch.setattr(health, "_fstype_of", lambda path: "cifs")
    monkeypatch.setattr(health, "_ntp_synchronised", lambda: True)

    metric = health.system_clock(station)
    assert metric.ok is True                     # not False
    assert "RTC lost time" not in metric.detail
    assert "NTP synchronised" in metric.detail   # the check actually ran
    assert "cifs" in metric.detail


def test_a_local_archive_ahead_of_the_clock_still_convicts_it(station, monkeypatch):
    """The original inference is sound when we stamped the files ourselves."""
    _future_product(station)
    monkeypatch.setattr(health, "_fstype_of", lambda path: "ext4")

    metric = health.system_clock(station)
    assert metric.ok is False
    assert "RTC lost time" in metric.detail


def test_product_age_names_the_share_rather_than_our_clock(station, monkeypatch):
    _future_product(station)
    monkeypatch.setattr(health, "_fstype_of", lambda path: "cifs")

    metric = health.newest_product_age(station)
    assert metric.ok is None
    assert "cifs" in metric.detail


def test_fstype_of_resolves_a_real_path():
    """Longest-prefix match against /proc/self/mounts, where there is one."""
    if not Path("/proc/self/mounts").exists():
        pytest.skip("no /proc/self/mounts on this platform")
    assert health._fstype_of(Path("/")) is not None


# --------------------------------------------------------------------------
# Prune: deleting only what is provably archived
# --------------------------------------------------------------------------

@pytest.fixture
def staging(tmp_path):
    """A local staging dir and a remote archive, both with one product."""
    from services.agent import prune

    local, remote = tmp_path / "local", tmp_path / "remote"
    (local / "2026-08-11").mkdir(parents=True)
    (remote / "2026-08-11").mkdir(parents=True)
    name = "2026-08-11/lfm_ionogram-unkown-DOB-ch0-000-1786364945.01.h5"
    (local / name).write_bytes(b"x" * 512)
    (remote / name).write_bytes(b"x" * 512)
    old = time.time() - 10 * 86400
    os.utime(local / name, (old, old))
    return prune, local, remote, name


def test_an_unmounted_archive_prunes_nothing(staging):
    """An unmounted share is an empty directory, and every `exists()` on a
    path under it fails -- so each file would be 'unverified' and kept. But an
    empty *remote root* is the signal that the mount is gone, and the guard is
    there so a future refactor cannot turn 'nothing to compare against' into
    'nothing worth keeping'."""
    prune, local, _, name = staging
    empty = local.parent / "not-mounted"
    empty.mkdir()

    result = prune.run(local, empty, min_age_s=0)
    assert result["mounted"] is False
    assert result["removed"] == 0
    assert (local / name).exists()
    assert "guard working" in prune.describe(result)


def test_a_verified_copy_is_removed(staging):
    prune, local, remote, name = staging
    result = prune.run(local, remote, min_age_s=86400)

    assert result["removed"] == 1
    assert not (local / name).exists()
    assert (remote / name).exists(), "the archive copy must survive"


def test_a_size_mismatch_is_not_a_copy(staging):
    """Truncated by an interrupted transfer. Timestamps cannot be trusted --
    SMB stamps come from the file server -- so size is the check."""
    prune, local, remote, name = staging
    (remote / name).write_bytes(b"x" * 100)

    result = prune.run(local, remote, min_age_s=86400)
    assert result["removed"] == 0
    assert (local / name).exists()


def test_a_missing_copy_is_not_a_copy(staging):
    prune, local, remote, name = staging
    (remote / name).unlink()

    assert prune.run(local, remote, min_age_s=86400)["removed"] == 0
    assert (local / name).exists()


def test_recent_products_are_left_alone(staging):
    """They may still be mid-write, and the mirror may not have reached them."""
    prune, local, remote, name = staging
    os.utime(local / name, None)                  # now

    assert prune.run(local, remote, min_age_s=86400)["removed"] == 0
    assert (local / name).exists()


def test_dry_run_deletes_nothing_but_reports(staging):
    prune, local, remote, name = staging
    result = prune.run(local, remote, min_age_s=86400, dry_run=True)

    assert result["removed"] == 1
    assert (local / name).exists(), "dry run must not touch the disk"
    assert "would remove" in prune.describe(result, dry_run=True)


def test_only_products_are_considered(staging):
    """The staging tree holds logs and partial transfers too. A prune by age
    alone would take them; this one matches product names."""
    prune, local, remote, _ = staging
    old = time.time() - 10 * 86400
    for junk in ("2026-08-11/notes.txt", "2026-08-11/.rsync-partial"):
        (local / junk).write_bytes(b"keep me")
        os.utime(local / junk, (old, old))

    prune.run(local, remote, min_age_s=86400)
    assert (local / "2026-08-11/notes.txt").exists()
    assert (local / "2026-08-11/.rsync-partial").exists()


def test_collect_never_raises_on_a_machine_that_is_not_a_station(tmp_path):
    config = StationConfig(chirp_config=tmp_path / "absent.ini",
                           output_dir=tmp_path / "absent",
                           ringbuffer_dir=tmp_path / "absent")
    report = health.collect(config)

    assert report.metrics, "a document must be produced whatever failed"
    assert report.to_json()


def test_unknown_metrics_do_not_make_a_station_unhealthy(station):
    """A missing `systemctl` must not page anyone at 03:00."""
    report = health.HealthReport(station="TST", timestamp=0.0, metrics=[
        health.Metric.unknown("unit:x", "no systemctl"),
        health.Metric("disk_free_fraction", 0.5, ok=True),
    ])
    assert report.healthy
    assert report.unknown and not report.failing


def test_one_definite_failure_is_enough(station):
    report = health.HealthReport(station="TST", timestamp=0.0, metrics=[
        health.Metric("disk_free_fraction", 0.01, ok=False),
        health.Metric("uptime_s", 99.0, ok=True),
    ])
    assert not report.healthy


def test_a_clock_five_years_slow_is_caught_with_no_data_on_disk(monkeypatch, station):
    """2026-08-06: the RTC had lost five years and the recorder stamped a run
    2021-04-02. `epoch_offset` cannot see this -- a clock that wrong means no
    recent products, so it reports "no timing solutions" and says nothing."""
    monkeypatch.setattr(health.time, "time", lambda: 1617339242.0)
    metric = health.system_clock(station)

    assert metric.ok is False
    assert "2021-04-02" in metric.detail
    assert "unset, not slow" in metric.detail


def test_a_clock_behind_files_on_disk_needs_no_hardcoded_date(monkeypatch, station):
    """The check that keeps working after CLOCK_SANITY_FLOOR_S goes stale."""
    product = station.output_dir / "lfm_ionogram-1.h5"
    product.write_bytes(b"x")
    future = health.CLOCK_SANITY_FLOOR_S + 400 * 86400.0
    os.utime(product, (future, future))
    monkeypatch.setattr(health.time, "time",
                        lambda: health.CLOCK_SANITY_FLOOR_S + 10.0)

    metric = health.system_clock(station)
    assert metric.ok is False
    assert "days behind products already on disk" in metric.detail


def test_a_plausible_clock_with_no_ntp_is_still_a_failure(monkeypatch, station):
    """Nothing else holds the epoch: rx_uhd_ext_gps copies the host clock into
    set_time_next_pps and never reads the GPSDO's gps_time sensor."""
    monkeypatch.setattr(health.time, "time",
                        lambda: health.CLOCK_SANITY_FLOOR_S + 86400.0)
    monkeypatch.setattr(health, "_ntp_synchronised", lambda: False)

    metric = health.system_clock(station)
    assert metric.ok is False
    assert "not synchronised" in metric.detail


def test_no_timedatectl_is_unknown_not_a_failure(monkeypatch, station):
    monkeypatch.setattr(health.time, "time",
                        lambda: health.CLOCK_SANITY_FLOOR_S + 86400.0)
    monkeypatch.setattr(health, "_ntp_synchronised", lambda: None)
    assert health.system_clock(station).ok is None


def test_epoch_metric_needs_a_reference_and_says_when_it_lacks_one(station):
    bare = StationConfig(output_dir=station.output_dir, reference_tx={})
    assert health.epoch_offset(bare).ok is None


def test_epoch_metric_is_unknown_when_the_slots_disagree(monkeypatch, station):
    """A solve with 146 ms of scatter is not a measurement.

    It happens when an archive spans a clock change, or when the wrong
    transmitter is named. Reporting the mean anyway would publish a number
    with no meaning.
    """
    class FakeOffset:
        seconds, residual_sd_s, n_slots, n_samples = -0.5, 0.146, 4, 600
        range_uncertainty_km = 43902.0

    from muf import io_detect
    monkeypatch.setattr(io_detect, "solve_epoch_offset",
                        lambda *a, **k: FakeOffset())
    monkeypatch.setattr(io_detect, "read_timing", lambda p: object())
    (station.output_dir / "par-ch0-1.0000.h5").write_bytes(b"x")

    metric = health.epoch_offset(station, max_age_s=1e9)
    assert metric.ok is None
    assert "disagree" in metric.detail


# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------

def test_only_allowed_systemctl_verbs(station):
    """`isolate` and `mask` have no business being reachable over a network."""
    with pytest.raises(control.ControlError, match="not allowed"):
        control.systemctl("isolate", station.target)
    with pytest.raises(control.ControlError, match="not allowed"):
        control.systemctl("mask", station.target)


@pytest.mark.parametrize("target", ["", "   "])
def test_an_unset_target_refuses_instead_of_running_systemctl(station, target):
    """A script-supervised station must not be 'restarted' by systemd.

    DOB runs `dombas.sh`, which owns the recorder as its own child, so
    `systemctl restart chirp.target` there does not restart anything -- it
    starts a *second* recorder against the same USRP, and two streamers means
    a power cycle by hand. The runner must never be reached.
    """
    calls = []

    def runner(args, **kw):                       # pragma: no cover - must not run
        calls.append(args)
        raise AssertionError(f"systemctl was invoked: {args}")

    result = control.systemctl("restart", target, runner=runner)
    assert calls == []
    assert result.ok is False
    assert "no systemd target configured" in result.detail


@pytest.mark.parametrize("shape,label", [
    ([{"chirp-rate": 1e5, "rep": 300.0, "chirpt": 235.0}], "flat"),
    ([[{"chirp-rate": 1e5, "rep": 300.0, "chirpt": 235.0}]], "one MPI rank"),
    ([[{"chirp-rate": 1e5, "rep": 300.0, "chirpt": 235.0}],
      [{"chirp-rate": 1.25e5, "rep": 30.0, "chirpt": 15.0}]], "two ranks"),
])
def test_both_schedule_shapes_are_accepted(shape, label):
    """`/ui/sources` builds the per-rank shape; the validator knew only flat.

    It reached `set(entry)` with a list and raised TypeError: unhashable type
    'dict' -- not a ControlError, so the operator saw the command fail with a
    Python internal and nothing to act on.
    """
    parser = configparser.ConfigParser()
    parser.add_section("lfm")
    control._validate(parser, {"mode": "scheduled",
                               "sounder_timings": json.dumps(shape)})


@pytest.mark.parametrize("shape,expect", [
    ([[{"rep": 300.0, "chirpt": 235.0}]], "chirp-rate"),
    ([["not an object"]], "not an object"),
    ([[]], "list of entries"),
])
def test_a_malformed_schedule_says_what_is_wrong(shape, expect):
    parser = configparser.ConfigParser()
    parser.add_section("lfm")
    with pytest.raises(control.ControlError, match=expect):
        control._validate(parser, {"mode": "scheduled",
                                   "sounder_timings": json.dumps(shape)})


def test_a_stop_that_times_out_warns_about_the_radio(station):
    """systemd escalates to SIGKILL, and a killed USRP needs a site visit."""
    def timeout_runner(args, **kw):
        raise subprocess.TimeoutExpired(args, 1.0)

    result = control.stop(station, runner=timeout_runner, timeout=1.0)
    assert not result.ok
    assert "wedge" in result.detail or "may have been killed" in result.detail


def test_only_allowed_config_keys(station):
    with pytest.raises(control.ControlError, match="not editable remotely"):
        control.apply_config(station, {"center_freq": "5e6"})


def test_mode_names_map_to_the_flag(station):
    control.apply_config(station, {"mode": "search"})
    assert control.read_config(station.chirp_config).get("lfm", "serendipitous") == "true"


def test_scheduled_mode_without_a_schedule_is_refused(station):
    """The combination that records nothing and reports healthy."""
    with pytest.raises(control.ControlError, match="record nothing"):
        control.apply_config(station, {"mode": "scheduled"})


def test_scheduled_mode_with_a_schedule_is_accepted(station):
    timings = json.dumps([{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
                           "transmit_name": "NIC"}])
    result = control.apply_config(station, {"mode": "scheduled",
                                            "sounder_timings": timings})
    assert result.ok
    parser = control.read_config(station.chirp_config)
    assert parser.get("lfm", "serendipitous") == "false"


def test_an_incomplete_schedule_entry_is_refused(station):
    bad = json.dumps([{"chirp-rate": 100e3}])          # no rep, no chirpt
    with pytest.raises(control.ControlError, match="missing"):
        control.apply_config(station, {"mode": "scheduled", "sounder_timings": bad})


def test_an_output_dir_that_cannot_exist_is_refused(station, tmp_path):
    with pytest.raises(control.ControlError, match="write nowhere"):
        control.apply_config(station,
                             {"output_dir": str(tmp_path / "no" / "such" / "tree")})


def test_a_change_is_backed_up_and_journaled(station):
    result = control.apply_config(station, {"max_range_extent": "2000e3"})

    assert result.ok
    assert result.journal["changes"]["max_range_extent"]["from"] == "4000e3"
    assert result.journal["changes"]["max_range_extent"]["to"] == "2000e3"
    assert result.journal["requires_restart"] is True
    backups = list(Path(station.chirp_config).parent.glob("*.bak"))
    assert len(backups) == 1


def test_the_config_is_never_left_half_written(station, monkeypatch):
    """A truncated .ini is an outage, not a misconfiguration."""
    original = Path(station.chirp_config).read_text(encoding="utf-8")

    def explode(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(control.os, "replace", explode)
    with pytest.raises(OSError):
        control.apply_config(station, {"max_range_extent": "1"}, backup=False)

    assert Path(station.chirp_config).read_text(encoding="utf-8") == original
    leftovers = [p for p in Path(station.chirp_config).parent.glob("*.tmp")]
    assert not leftovers, "temporary file was not cleaned up"


def test_apply_and_restart_stops_before_it_edits(station):
    """The config is read at process start, so stopping first makes the window
    in which old processes could run under new settings exactly zero."""
    calls = []

    def recording_runner(args, **kw):
        calls.append(args[1])
        return subprocess.CompletedProcess(args, 0, "", "")

    results = control.apply_and_restart(station, {"max_range_extent": "1000e3"},
                                        runner=recording_runner)
    assert calls == ["stop", "start"]
    assert [r.command.split()[0] for r in results] == ["stop", "apply_config", "start"]
    assert all(r.ok for r in results)


def test_a_refused_edit_brings_acquisition_back_up(station):
    """Never leave the station down because a parameter was invalid."""
    calls = []

    def recording_runner(args, **kw):
        calls.append(args[1])
        return subprocess.CompletedProcess(args, 0, "", "")

    results = control.apply_and_restart(station, {"mode": "scheduled"},
                                        runner=recording_runner)
    assert calls == ["stop", "start"]
    assert not results[1].ok
    assert results[-1].ok


def test_a_failed_stop_does_not_edit_the_config(station):
    original = Path(station.chirp_config).read_text(encoding="utf-8")
    results = control.apply_and_restart(
        station, {"max_range_extent": "1"},
        runner=fake_runner(returncode=1, stderr="unit not found"))

    assert not results[0].ok
    assert not results[1].ok and "not attempted" in results[1].detail
    assert Path(station.chirp_config).read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

def test_signatures_recognise_this_stations_real_failures():
    """Each line here was pasted from an actual incident."""
    lines = [
        "Unable to set the thread priority",
        "got no data in recv 0", "got no data in recv 0",
        " * F5F86F: false", "WARNING: One or more devices not locked.",
        "No UHD Devices Found",
        "no DigitalRF data bounds available for channel ch0",
    ]
    found = dict(logs.scan_signatures(lines))

    assert any("real-time" in k for k in found)
    assert any("fell behind" in k for k in found)
    assert any("not locked" in k for k in found)
    assert any("wedged" in k for k in found)
    assert any("not writing" in k for k in found)


def test_signatures_recognise_the_2026_08_06_restart():
    """Three faults in one paste, none of which any metric would have named."""
    lines = [
        "error message = 'No space left on device', buf = 0x7f797f0bb3e8",
        "Problem detected, dataset_samples_written = 0 after  0 samples_written",
        "Fatal Digital RF write error on channel 0 at sample index 0",
        "./examples/marieluise/dombas.sh: 14: source: not found",
        "Python version: Python 3.5.2",
    ]
    found = dict(logs.scan_signatures(lines))

    assert any("volume is full" in k for k in found)
    assert any("wrote zero samples" in k for k in found)
    assert any("venv did not activate" in k for k in found)


def test_the_patched_recorders_epoch_verdict_is_a_signature():
    """patches/0001 prints these; triage must name them without the log."""
    found = dict(logs.scan_signatures([
        "EPOCH CHECK FAILED: USRP epoch is -1 s from GPS; every range ...",
        "WARNING: setting USRP epoch from host clock; gps_time unavailable",
    ]))
    assert any("300,000 km per second" in k for k in found)
    assert any("fell back to the host clock" in k for k in found)


def test_the_patched_recorders_success_line_is_not_a_signature():
    assert not logs.scan_signatures([
        "Epoch check OK: USRP last pps == GPSDO gps_time",
        "GPSDO gps_time: 1770400123  (host clock is 0 s from GPS)",
    ])


def test_a_plausible_recorder_epoch_is_not_a_signature():
    """`PC time now:` is printed on every start; only the value makes it news."""
    sane = int(logs_clock_floor() + 86400)
    assert not logs.scan_signatures([f"PC time now: {sane} + 0.93 sec"])


def test_an_implausible_recorder_epoch_names_the_date():
    found = dict(logs.scan_signatures(["PC time now: 1617339242 + 0.931616 sec"]))
    assert any("2021-04-02" in k for k in found)
    assert any("mis-timestamped" in k for k in found)


def logs_clock_floor() -> float:
    return health.CLOCK_SANITY_FLOOR_S


def test_the_count_is_the_diagnosis_not_the_presence():
    """One dropped packet is noise; four hundred is a misconfiguration."""
    found = dict(logs.scan_signatures(["got no data in recv 0"] * 400))
    assert next(iter(found.values())) == 400


def test_line_counts_are_clamped(station):
    """A log endpoint that can return a gigabyte is a self-inflicted DoS."""
    huge = "\n".join(f"line {i}" for i in range(logs.MAX_LINES * 3))
    chunk = logs.read_unit("x", lines=10 ** 9,
                           runner=fake_runner(stdout=huge))

    assert len(chunk.lines) <= logs.MAX_LINES
    assert chunk.truncated


def test_a_missing_journalctl_is_reported_not_raised(station):
    def absent(args, **kw):
        raise FileNotFoundError(args[0])

    chunk = logs.read_unit("x", runner=absent)
    assert chunk.lines == [] and "not found" in chunk.error


def test_triage_answers_which_failure_without_the_text(station, monkeypatch):
    monkeypatch.setattr(
        logs, "read_unit",
        lambda unit, **kw: logs.LogChunk(
            unit, lines=[], signatures=[("host fell behind the USRP", 12)]))
    found = logs.triage(station)

    assert found["signatures"][0]["count"] == 12 * len(station.units)
    assert "lines" not in json.dumps(found)


# --------------------------------------------------------------------------
# Transport and loop
# --------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_push_carries_the_token_and_the_document(station):
    station = StationConfig(**{**station.as_dict(),
                               "chirp_config": station.chirp_config,
                               "output_dir": station.output_dir,
                               "ringbuffer_dir": station.ringbuffer_dir,
                               "server_url": "http://server/api",
                               "token": "secret"})
    seen = {}

    def opener(request, timeout=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data.decode())
        return FakeResponse({"ok": True})

    report = health.collect(station, include_epoch=False)
    client.push_health(station, report, opener=opener)

    assert seen["url"].endswith("/stations/health")
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["station"] == station.station


def test_a_server_that_is_down_does_not_stop_the_pass(station):
    down = StationConfig(station="TST", chirp_config=station.chirp_config,
                         output_dir=station.output_dir,
                         ringbuffer_dir=station.ringbuffer_dir,
                         server_url="http://server/api")

    def refuse(request, timeout=None):
        raise OSError("connection refused")

    result = runner.run_once(down, opener=refuse, include_epoch=False)

    assert result.report is not None, "health must still be collected"
    assert not result.pushed
    assert any("push:" in e for e in result.errors)


def test_an_unknown_command_is_answered_not_ignored(station):
    results = runner._dispatch(station, client.Command("1", "self-destruct", {}))
    assert not results[0].ok
    assert "unknown command" in results[0].detail


@pytest.mark.parametrize("name,verb", [
    ("start", "start"), ("stop_sounding", "stop"), ("restart", "restart"),
])
def test_process_commands_map_to_the_target(station, monkeypatch, name, verb):
    seen = {}
    monkeypatch.setattr(control, "systemctl",
                        lambda v, t, **kw: seen.update(verb=v, target=t)
                        or control.CommandResult(v, True))
    runner._dispatch(station, client.Command("1", name, {}))
    assert seen == {"verb": verb, "target": station.target}


def test_a_command_is_acknowledged_even_when_it_fails(station):
    live = StationConfig(station="TST", chirp_config=station.chirp_config,
                         output_dir=station.output_dir,
                         ringbuffer_dir=station.ringbuffer_dir,
                         server_url="http://server/api")
    acked = []

    def opener(request, timeout=None):
        url = request.full_url
        if url.endswith("/commands"):
            return FakeResponse({"commands": [
                {"id": "c1", "name": "set_config",
                 "params": {"changes": {"mode": "scheduled"}}}]})
        if url.endswith("/ack"):
            acked.append(json.loads(request.data.decode()))
        return FakeResponse({"ok": True})

    result = runner.run_once(live, opener=opener, include_epoch=False)

    assert result.commands_run == 1
    assert len(acked) == 1, "an unacknowledged command is redelivered forever"
    assert any(not r["ok"] for r in acked[0]["results"])


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def test_the_token_can_come_from_the_environment(tmp_path, monkeypatch):
    """`deploy/station-sim.json` is committed, so the secret must not live in
    it. The compose rig passes CONTROL_TOKEN through as AGENT_TOKEN instead,
    and the real station should be deployed the same way."""
    path = tmp_path / "station.json"
    path.write_text(json.dumps({"station": "SIM", "token": ""}), encoding="utf-8")
    monkeypatch.setenv("AGENT_CONFIG", str(path))
    monkeypatch.setenv("AGENT_TOKEN", "from-the-environment")

    assert StationConfig.from_env().token == "from-the-environment"


def test_an_absent_token_env_leaves_the_file_alone(tmp_path, monkeypatch):
    """Unset must mean "not specified", not "clear it". The station already
    treats a missing secret as "report to stdout and take no commands", and an
    empty override silently disarming a configured agent would be the same
    class of mistake as an unset CONTROL_TOKEN opening control."""
    path = tmp_path / "station.json"
    path.write_text(json.dumps({"station": "SIM", "token": "from-the-file"}),
                    encoding="utf-8")
    monkeypatch.setenv("AGENT_CONFIG", str(path))

    monkeypatch.delenv("AGENT_TOKEN", raising=False)
    assert StationConfig.from_env().token == "from-the-file"

    monkeypatch.setenv("AGENT_TOKEN", "")
    assert StationConfig.from_env().token == "from-the-file"


def test_the_token_is_redacted_when_the_config_is_reported(tmp_path):
    """`as_dict` feeds the health document, which crosses the wire."""
    assert StationConfig(token="s3cret").as_dict()["token"] == "***"
    assert StationConfig(token="").as_dict()["token"] == ""


def test_a_missing_config_names_itself_and_agent_config(tmp_path):
    """Under Restart=always this message is read on the twentieth repeat."""
    missing = tmp_path / "nowhere" / "agent.json"
    with pytest.raises(FileNotFoundError) as caught:
        StationConfig.from_json(missing)
    text = str(caught.value)
    assert str(missing) in text
    assert "AGENT_CONFIG" in text


def test_malformed_config_json_names_the_file(tmp_path):
    """`json.JSONDecodeError` gives a line and column but never the path."""
    broken = tmp_path / "agent.json"
    broken.write_text('{"station": "DOB",}', encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        StationConfig.from_json(broken)
    assert str(broken) in str(caught.value)
