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
import re
import subprocess
import time
from dataclasses import replace
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


def _fake_systemctl(monkeypatch, states: dict):
    """`systemctl is-active <unit>` answering from a table."""
    monkeypatch.setattr(health, "_run",
                        lambda args, **kw: (0, states[args[-1]]))


UNITS = ("chirp-rx.service", "chirp-digisonde@Dourbes.service")


def test_an_inactive_digisonde_receiver_is_not_a_failure(monkeypatch, station):
    """Four stopped digisonde receivers made DOB read UNHEALTHY.

    Each is an oblique reception of a remote vertical sounder -- an extra
    circuit, not the instrument -- so their state is worth reporting and is
    not evidence about this station. The state itself still travels: "down
    since Tuesday" is a different sentence from "not watched".
    """
    _fake_systemctl(monkeypatch, {"chirp-rx.service": "active",
                                  "chirp-digisonde@Dourbes.service": "inactive"})
    metrics = {m.name: m for m in health.unit_states(replace(station, units=UNITS))}

    optional = metrics["unit:chirp-digisonde@Dourbes.service"]
    assert optional.ok is None and optional.value == "inactive"
    assert metrics["unit:chirp-rx.service"].ok is True
    assert health.HealthReport("TST", 0.0, list(metrics.values())).healthy


def test_a_required_unit_that_is_inactive_still_fails(monkeypatch, station):
    """The exemption is by name, not a general softening of the check."""
    _fake_systemctl(monkeypatch, {"chirp-rx.service": "inactive",
                                  "chirp-digisonde@Dourbes.service": "active"})
    metrics = health.unit_states(replace(station, units=UNITS))

    assert not health.HealthReport("TST", 0.0, metrics).healthy
    assert [m.name for m in metrics if m.ok is False] == ["unit:chirp-rx.service"]


def test_the_exemption_can_be_turned_off_per_station(monkeypatch, station):
    """A station that does depend on its digisonde feeds says so."""
    _fake_systemctl(monkeypatch, {"chirp-digisonde@Dourbes.service": "failed"})
    config = replace(station, units=UNITS[1:], optional_units=())
    metric = health.unit_states(config)[0]

    assert metric.ok is False and metric.value == "failed"


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


#: A complete entry. Five keys, because `calc_ionograms.py` reads five.
ENTRY = {"chirp-rate": 1e5, "rep": 300.0, "chirpt": 235.0,
         "id": 1, "transmit_name": "NIC"}


@pytest.mark.parametrize("shape,label", [
    ([ENTRY], "flat"),
    ([[ENTRY]], "one MPI rank"),
    ([[ENTRY], [dict(ENTRY, **{"chirp-rate": 1.25e5, "rep": 30.0,
                               "chirpt": 15.0, "id": 2,
                               "transmit_name": "SGO"})]], "two ranks"),
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
    ([[{k: v for k, v in ENTRY.items() if k != "chirp-rate"}]], "chirp-rate"),
    ([["not an object"]], "not an object"),
    ([[]], "list of entries"),
])
def test_a_malformed_schedule_says_what_is_wrong(shape, expect):
    parser = configparser.ConfigParser()
    parser.add_section("lfm")
    with pytest.raises(control.ControlError, match=expect):
        control._validate(parser, {"mode": "scheduled",
                                   "sounder_timings": json.dumps(shape)})


def _launcher(tmp_path, line: str) -> Path:
    """A launcher script whose only interesting line is the one under test."""
    path = tmp_path / "dombas.sh"
    path.write_text("#!/bin/sh\necho detect_chirps.py\n"
                    "$MPIRUN -np 4 python3 -u detect_chirps.py --config \"$C\" &\n"
                    f"{line}\n"
                    "echo plot_ionograms.py\n")
    return path


def _schedule(n: int) -> str:
    return json.dumps([[dict(ENTRY, id=i + 1, transmit_name=f"TX{i}")]
                       for i in range(n)])


@pytest.mark.parametrize("line,ranks,expect", [
    ('$MPIRUN -np 2 python3 -u calc_ionograms.py --config "$C" &', 2, None),
    ('$MPIRUN -np 2 python3 -u calc_ionograms.py --config "$C" &', 3,
     "never be sounded"),
    ('$MPIRUN -np 3 python3 -u calc_ionograms.py --config "$C" &', 2,
     "IndexError"),
    # No mpirun at all: one process, so rank 0 and nothing else.
    ('python3 -u calc_ionograms.py --config "$C" &', 2, "never be sounded"),
    ('python3 -u calc_ionograms.py --config "$C" &', 1, None),
])
def test_np_must_match_the_schedule(tmp_path, station, line, ranks, expect):
    """The silent half of a schedule change, and the reason for patch 0009.

    `calc_ionograms.py:452` indexes `sounder_timings` by MPI rank with no
    guard. Set a three-rank schedule through the UI while the launcher still
    says `-np 2` and the third transmitter is never sounded -- no error, and
    the log reads as healthy because for ranks 0 and 1 it is.
    """
    station = replace(station, launcher=_launcher(tmp_path, line))
    changes = {"mode": "scheduled", "sounder_timings": _schedule(ranks)}
    if expect is None:
        assert control.apply_config(station, changes).ok
    else:
        with pytest.raises(control.ControlError, match=expect):
            control.apply_config(station, changes)


def test_a_derived_np_disables_the_check(tmp_path, station):
    """Patch 0009 makes `-np` a variable, which is the point: nothing to check.

    A literal count is a claim the agent can falsify. `-np "$NP_IONO"` is the
    launcher promising to compute it from the same `sounder_timings` the agent
    just wrote, so any schedule length is fine and guessing otherwise would
    refuse correct commands.
    """
    station = replace(station, launcher=_launcher(
        tmp_path, '$MPIRUN -np "$NP_IONO" python3 -u calc_ionograms.py &'))
    assert control.apply_config(
        station, {"mode": "scheduled", "sounder_timings": _schedule(5)}).ok


@pytest.mark.parametrize("launcher_line,label", [
    ('# $MPIRUN -np 1 python3 -u calc_ionograms.py --config "$C" &', "commented out"),
    ('echo "nothing to see"', "no calc_ionograms line"),
])
def test_an_unreadable_np_does_not_block_a_schedule(tmp_path, station,
                                                    launcher_line, label):
    """Refusing on "I could not tell" would be worse than not checking.

    This guard exists to catch one specific mistake. A launcher it cannot
    parse is not evidence of that mistake, and a station whose acquisition is
    started some other way must still be configurable.
    """
    station = replace(station, launcher=_launcher(tmp_path, launcher_line))
    assert control.apply_config(
        station, {"mode": "scheduled", "sounder_timings": _schedule(2)}).ok


def test_a_missing_launcher_does_not_block_a_schedule(tmp_path, station):
    station = replace(station, launcher=tmp_path / "nope" / "dombas.sh")
    assert control.apply_config(
        station, {"mode": "scheduled", "sounder_timings": _schedule(2)}).ok


def _unit(tmp_path, exec_start: str) -> Path:
    """The real shape of `chirp-ionograms.service`, Description= included."""
    path = tmp_path / "chirp-ionograms.service"
    path.write_text(
        "[Unit]\n"
        "Description=chirpsounder2 calc_ionograms.py\n"
        "PartOf=chirp.target\n\n"
        "[Service]\n"
        "User=ionouser\n"
        "# -np must equal len(sounder_timings): calc_ionograms.py:452 does\n"
        "# `st = conf.sounder_timings[rank]` with no guard.\n"
        f"{exec_start}\n"
        "Restart=always\n")
    return path


def test_the_launcher_may_be_a_systemd_unit(tmp_path, station):
    """After the migration the unit is what starts it, so it is the launcher.

    A unit names the program in `Description=` before it launches it in
    `ExecStart=`, and that first mention carries no `-np`. Read naively it
    answers 1 for a unit that starts two ranks, and then refuses a correct
    two-transmitter schedule with a mismatch that does not exist -- which is
    what happened on the station on 2026-08-16, the day `launcher` was
    repointed at the unit.
    """
    station = replace(station, launcher=_unit(
        tmp_path,
        "ExecStart=/usr/bin/mpirun --bind-to none -np 2 /opt/python3 "
        "calc_ionograms.py --config /home/ionouser/my_station.ini"))
    assert control.apply_config(
        station, {"mode": "scheduled", "sounder_timings": _schedule(2)}).ok

    with pytest.raises(control.ControlError, match="never be sounded"):
        control.apply_config(
            station, {"mode": "scheduled", "sounder_timings": _schedule(3)})


def test_a_unit_whose_exec_wraps_across_lines_is_still_read(tmp_path, station):
    """`ExecStart=` is routinely wrapped, which splits `-np` from the script.

    Both halves have to be joined before the scan or the `-np` is on a line
    with no `calc_ionograms.py` in it and the check reads the wrong number --
    the same failure as `Description=`, arrived at from the other side.
    """
    station = replace(station, launcher=_unit(
        tmp_path,
        "ExecStart=/usr/bin/mpirun --bind-to none -np 2 \\\n"
        "    /opt/python3 calc_ionograms.py \\\n"
        "    --config /home/ionouser/my_station.ini"))
    with pytest.raises(control.ControlError, match="IndexError"):
        control.apply_config(
            station, {"mode": "scheduled", "sounder_timings": _schedule(1)})


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
    timings = json.dumps([ENTRY])
    result = control.apply_config(station, {"mode": "scheduled",
                                            "sounder_timings": timings})
    assert result.ok
    parser = control.read_config(station.chirp_config)
    assert parser.get("lfm", "serendipitous") == "false"


def test_an_incomplete_schedule_entry_is_refused(station):
    bad = json.dumps([{"chirp-rate": 100e3}])          # no rep, no chirpt
    with pytest.raises(control.ControlError, match="missing"):
        control.apply_config(station, {"mode": "scheduled", "sounder_timings": bad})


@pytest.mark.parametrize("dropped", ["id", "transmit_name"])
def test_the_two_keys_nothing_was_writing_are_required(station, dropped):
    """`calc_ionograms.py:446-447` reads both with a bare subscript.

    Neither was in the entry `/ui/sources` composed, and neither is in
    `chirp_config.py`'s own default -- while every `.ini` in the clone's
    `examples/marieluise` carries both. A schedule short of them is not a
    smaller schedule: the rank that receives it dies of KeyError at its first
    slot, the other ranks carry on, and the log looks normal.
    """
    entry = {k: v for k, v in ENTRY.items() if k != dropped}
    with pytest.raises(control.ControlError, match=dropped):
        control.apply_config(station, {"mode": "scheduled",
                                       "sounder_timings": json.dumps([entry])})


def test_an_empty_transmit_name_is_refused(station):
    """It is the product's file name, not a label.

    `lfm_ionogram--DOB-ch000-001-1770163210.00.h5` parses back with an empty
    transmitter and no error anywhere.
    """
    entry = dict(ENTRY, transmit_name="  ")
    with pytest.raises(control.ControlError, match="empty transmit_name"):
        control.apply_config(station, {"mode": "scheduled",
                                       "sounder_timings": json.dumps([entry])})


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
# Preview
# --------------------------------------------------------------------------

def decode_png(data: bytes):
    """An independent 4-bit palette PNG reader, in about fifteen lines.

    Independent on purpose: a round-trip through
    ``services.agent.preview.encode_png``'s own logic would prove only that it
    agrees with itself, and the encoder is hand-rolled precisely because there
    is no library on the station to check it against.
    """
    import struct
    import zlib

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, i = {}, 8
    while i < len(data):
        (size,) = struct.unpack(">I", data[i:i + 4])
        kind = data[i + 4:i + 8]
        chunks.setdefault(kind, b"")
        chunks[kind] += data[i + 8:i + 8 + size]
        i += 12 + size
    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (4, 3), "expected a 4-bit palette PNG"
    palette = [tuple(chunks[b"PLTE"][k:k + 3])
               for k in range(0, len(chunks[b"PLTE"]), 3)]

    raw = zlib.decompress(chunks[b"IDAT"])
    stride = (width + 1) // 2
    rows = []
    for y in range(height):
        line = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        assert line[0] == 0, "filter type 0 only"
        nibbles = []
        for byte in line[1:]:
            nibbles += [byte >> 4, byte & 0xF]
        rows.append(nibbles[:width])
    return rows, palette


def test_the_png_encoder_writes_a_png(tmp_path):
    """Every pixel back, unchanged, through a decoder that shares no code."""
    from services.agent import preview

    want = [[(x + y) % 16 for x in range(7)] for y in range(5)]   # odd width
    rows, palette = decode_png(
        preview.encode_png([bytes(r) for r in want], 7))

    assert rows == want
    assert palette == [tuple(c) for c in preview.JET_16]


def test_the_preview_constants_still_match_muf():
    """The duplication this module admits to, pinned.

    `preview` cannot import `muf` -- the agent is stdlib-only and `muf` is a
    server-side package -- so the arithmetic is copied. Copied constants drift;
    this is what stops them.
    """
    io_chirp = pytest.importorskip("muf.io_chirp")
    render = pytest.importorskip("muf.render")
    spectro = pytest.importorskip("muf.spectro")
    from services.agent import preview

    assert preview.NOISE_COEF == spectro.NOISE_COEF
    assert preview.C_M_S == io_chirp.C_M_S
    assert preview.UNIDENTIFIED_TX == io_chirp.UNIDENTIFIED_TX
    assert preview.MAX_VIRTUAL_RANGE_KM == io_chirp.MAX_VIRTUAL_RANGE_KM
    assert (preview.VMIN_DB, preview.VMAX_DB) == (render.DEFAULT_VMIN_DB,
                                                  render.DEFAULT_VMAX_DB)
    assert preview.SNR_OFFSET_DB == io_chirp.SNR_OFFSET_DB


def test_the_preview_is_not_mirrored_top_to_bottom(make_chirp_h5):
    """The one bug in here that would still look like an ionogram.

    v2's range axis ascends and this pipeline's descends, with the largest
    virtual range at the top of the picture where `render` draws it. Reverse it
    wrongly and the thumbnail is upside down and entirely plausible.

    The echo is put in the *near* half of the stored axis, so an unreversed
    read puts it in the wrong half of the image rather than merely off-centre.
    """
    np = pytest.importorskip("numpy")
    from services.agent import preview

    n_freq, fftlen = 32, 3000          # 40 km bins, so 71 fit under the extent
    power = np.full((n_freq, fftlen), 1.0)
    power[:, fftlen // 2 - 20] = 400.0     # ascending axis: below centre is near
    path = make_chirp_h5(power, max_range_extent_km=1400.0)
    shot = preview.build(path, size=(32, 24))

    assert not shot.cropped, "this product is narrow enough to be shown whole"
    rows, _ = decode_png(shot.png)
    brightest = max(range(len(rows)), key=lambda y: max(rows[y]))
    assert brightest > len(rows) // 2, (
        "a near echo belongs near the bottom, where `render` puts the small "
        "ranges -- finding it in the upper half means the reversal was skipped")


def test_the_preview_uses_the_same_db_scale_as_the_full_render(make_chirp_h5):
    """A bright cell in the thumbnail has to mean what it means in the big one."""
    np = pytest.importorskip("numpy")
    io_chirp = pytest.importorskip("muf.io_chirp")
    from services.agent import preview

    rng = np.random.default_rng(11)
    power = rng.gamma(2.0, 1.0, (32, 3000)) + 0.5
    power[:, 1520] = 900.0
    path = make_chirp_h5(power, max_range_extent_km=1400.0)

    ion = io_chirp.load(path)
    want = preview._block_max(preview._block_max(ion.db, 16, 0), 12, 1)
    want = np.clip(np.rint((want - preview.VMIN_DB) * 15.0
                           / (preview.VMAX_DB - preview.VMIN_DB)), 0, 15).T

    rows, _ = decode_png(preview.build(path, size=(16, 12)).png)
    assert np.array_equal(np.array(rows), want.astype(int)), (
        "the thumbnail's levels must come from the same dB the renderer uses")


def test_a_zero_byte_product_costs_a_preview_and_nothing_else(station):
    """Products are opened while the recorder is still writing them.

    Every existing health test writes zero-byte products; a decoder that raised
    out of the pass would take all of them with it.
    """
    pytest.importorskip("h5py")
    (station.output_dir / "lfm_ionogram-A-TST-ch0-001-1785846834.00.h5").touch()
    live = replace(station, server_url="http://server/api")
    pushed = []

    def opener(request, timeout=None):
        pushed.append(request.full_url)
        return FakeResponse({"ok": True})

    result = runner.run_once(live, opener=opener, include_epoch=False,
                             previews=runner.PreviewState())

    assert result.pushed, "health must still go out"
    assert result.previews_sent == 0
    assert not any(u.endswith("/preview") for u in pushed)
    assert any("preview A:" in e for e in result.errors)


def test_the_walk_groups_products_by_transmitter(station):
    """One rglob answers both `newest_product_age` and the previews.

    The transmitter is in the filename, so grouping by circuit opens nothing
    and stats nothing -- which is what makes doing this every 60 s on a FUSE
    volume reasonable.
    """
    for name in ("lfm_ionogram-SGO-TST-ch0-001-1785846000.00.h5",
                 "lfm_ionogram-SGO-TST-ch0-001-1785849000.00.h5",
                 "lfm_ionogram-cyprus1-TST-ch0-002-1785847000.00.h5"):
        (station.output_dir / name).touch()

    by_tx, mtime = health.scan_products(station.output_dir)

    assert set(by_tx) == {"SGO", "cyprus1"}
    assert by_tx["SGO"][1] == 1785849000.0, "newest per circuit, not first seen"
    assert mtime is None, "no name needed the mtime fallback"
    # And the metric still reads the newest of all of them.
    assert health.newest_product_age(station, (by_tx, mtime)).detail.startswith(
        "sounding start time")


def test_an_unchanged_product_is_not_sent_twice():
    """An idle circuit must cost nothing at all."""
    from services.agent import preview

    newest = {"SGO": (Path("a.h5"), 100.0)}
    picked, cursor = preview.due(newest, {})
    assert [p[0] for p in picked] == ["SGO"]

    picked, cursor = preview.due(newest, {"SGO": 100.0}, cursor=cursor)
    assert picked == []

    picked, _ = preview.due({"SGO": (Path("b.h5"), 200.0)}, {"SGO": 100.0})
    assert [p[2] for p in picked] == [200.0], "a newer product must go"


def test_more_new_products_than_the_budget_spread_over_passes():
    """A station handed eleven circuits at once must not make one long pass.

    Round-robin rather than newest-first: sorting by `t0` would let a busy
    circuit starve a quiet one forever, and the quiet one is usually the
    interesting one.
    """
    from services.agent import preview

    newest = {f"tx{i}": (Path(f"{i}.h5"), float(i)) for i in range(11)}
    sent, cursor, seen = {}, 0, []
    for _ in range(3):
        picked, cursor = preview.due(newest, sent, cursor=cursor)
        assert len(picked) <= preview.MAX_PER_PASS
        for tx, _path, t0 in picked:
            sent[tx] = t0
            seen.append(tx)

    assert len(seen) == len(set(seen)) == 11, (
        "three passes at four each must cover eleven circuits exactly once")


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

def test_one_output_dir_is_written_in_three_places():
    """`agent.json` and the two archive units must name the same directory.

    `chirp-archive-sync.service` says it in its own comment -- "three places,
    one path" -- and for eleven days two of them said one thing and the third
    said `/media/ionouser/DATA3/ionozond_data2`, a volume the station had
    stopped writing to. Nothing reported it, because every symptom of this
    points somewhere else: `newest_product_age_s` answers `no such directory`
    for a recorder that is producing normally, the station preview finds
    nothing to encode and the console explains that the agent must be too old,
    and `chirp-archive-sync` mirrors an empty tree and exits 0.

    The authority is `output_dir` in `my_station.ini`, which is on the station
    and not in this repository, so this cannot check the truth -- only that the
    three copies of it here have not drifted apart, which is the failure that
    actually happened.
    """
    root = Path(__file__).resolve().parent.parent
    example = root / "deploy/station-dob.json.example"
    wanted = json.loads(example.read_text(encoding="utf-8"))["output_dir"]

    for unit in ("chirp-archive-sync.service", "chirp-archive-prune.service"):
        text = (root / "services/agent/systemd" / unit).read_text(encoding="utf-8")
        found = re.findall(r"^Environment=ARCHIVE_LOCAL=(.+)$", text, re.M)
        assert found == [wanted], (
            f"{unit} stages from {found} while {example.name} says {wanted!r}. "
            "Whichever is wrong is silent: the mirror reports success over an "
            "empty directory, and the agent reports a station that has stopped "
            "producing. Fix both against `output_dir` in my_station.ini."
        )

    assert StationConfig().output_dir == Path(wanted), (
        "the dataclass default is what a station with no output_dir in its "
        "agent.json gets, so it is a fourth copy of this path"
    )


def test_the_station_example_asks_for_no_digisonde_receiver():
    """A `chirp-digisonde@` instance in this file is how the 45% sample loss
    comes back, and it has come back once already.

    `receive_digisonde.py` is not a downloader -- it demodulates off air from
    the ringbuffer at 25 MS/s, so each instance costs what `detect_chirps`
    costs. Patch 0007 removed five of them from `dombas.sh` and took the
    recorder from ~969 dropped events/s to zero over an hour; the unavoidable
    pipeline already needs 4.4 of 8 cores and the receivers want ~3.4 more.

    This file is not documentation. It is copied to the station as
    `agent.json`, and its unit list is read as the set of units to enable --
    which is exactly what happened on 2026-08-16, four instances enabled off
    `_units_when_migrated`, restoring the fault patch 0007 exists to fix. On a
    systemd station nothing reports it: `chirp-drop-watch` reads a `thor.log`
    that systemd no longer writes and answers zero drops for ever.

    Every string in the file is checked, not just the two unit lists, because
    the prose is what gets copied when the JSON does not. What counts as naming
    one is an instance name straight after the ``@`` -- the file is allowed,
    and expected, to talk *about* `chirp-digisonde@` in order to say no.
    """
    example = Path(__file__).resolve().parent.parent / "deploy/station-dob.json.example"
    config = json.loads(example.read_text(encoding="utf-8"))

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from strings(value)

    instance = re.compile(r"chirp-digisonde@\w")
    named = [s for s in strings(config) if instance.search(s)]
    assert named == [], (
        f"{example.name} names {named}. Enabling a digisonde receiver costs "
        "the recorder ~969 dropped events/s and nothing on the station "
        "reports it -- see patches/0007-dombas-move-digisonde-and-drop-"
        "plotters.patch. If a station really wants them back, start with "
        "CPUAffinity pinning and measure, do not list them here."
    )


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


# --- set_band ----------------------------------------------------------------
#
# BACKLOG sec. 30, phase 2. The band is five ini keys that must agree with each
# other and with the compiled recorder, and on 2026-08-19 they disagreed twice
# in one day -- both times with every unit green, every log clean, and no
# product worth keeping. These tests are that day, executable.

#: The station's real band configuration as of 2026-08-19, so a test that
#: passes here describes something that ran.
BAND_INI_TEXT = (
    "[config]\n"
    "sample_rate = 25000000\n"
    "center_freq = 20e6\n"
    "[lfm]\n"
    "decimation = 625\n"
    "downconversion_block_samples = 4000\n"
    "frequency_resolution = 50e3\n"
    "max_freq = 32.5e6\n"
    "min_freq = 7.5e6\n"
    "manual_freq_extent = true\n"
    "maximum_analysis_frequency = 32.5e6\n"
    "minimum_analysis_frequency = 7.5e6\n"
    'sounder_timings = [[{"chirp-rate": 100000, "rep": 300, "chirpt": 245,'
    ' "id": 4, "transmit_name": "NIC3"}]]\n'
    "serendipitous = false\n"
)


def _band_parser(text: str = BAND_INI_TEXT) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read_string(text)
    return parser


def _recorder(tmp_path, *, patched: bool) -> Path:
    """A stand-in binary that does or does not carry patch 0014's option."""
    path = tmp_path / ("rx_uhd_ext_gps" if patched else "rx_uhd_old")
    body = b"\x7fELF..." + (b"center-freq" if patched else b"gps-lock-timeout")
    path.write_bytes(body + b"\x00usrp_args\x00")
    return path


def test_the_running_band_is_what_the_planner_derives():
    """band_start 7.5 MHz must reproduce the config the station is running.

    If this drifts, the panel would offer to "change" the band to something
    other than what is already there, and the operator's first click would be
    an unintended change.
    """
    plan = control.plan_band(_band_parser(), band_start_mhz=7.5,
                             check_recorder=False)
    assert plan.changes["center_freq"] == "20e6"
    assert plan.changes["minimum_analysis_frequency"] == "7.5e6"
    assert plan.changes["maximum_analysis_frequency"] == "32.5e6"
    assert plan.changes["min_freq"] == "7.5e6"
    assert plan.changes["max_freq"] == "32.5e6"
    assert plan.sweep_seconds == {"NIC3": 250.0}


def test_the_lo_is_the_middle_of_the_band_not_its_edge():
    """The one arithmetic error that would blind the station outright."""
    plan = control.plan_band(_band_parser(), band_start_mhz=0.0,
                             check_recorder=False)
    assert plan.changes["center_freq"] == "12.5e6"      # v2's default
    assert (plan.band_start_hz, plan.band_stop_hz) == (0.0, 25e6)


def test_analysis_outside_the_digitised_band_is_refused():
    """2026-08-18, exactly: `maximum_analysis_frequency = 30e6` against a
    0-25 MHz passband. It produced no error and no signal above 25 MHz."""
    with pytest.raises(control.ControlError, match="never sampled"):
        control.plan_band(_band_parser(), band_start_mhz=0.0,
                          analysis_max_mhz=30.0, check_recorder=False)


def test_a_sweep_that_outruns_the_repetition_period_is_refused():
    """A slow transmitter over a wide span: the next sounding starts before
    this one has been read out. Needs a rate low enough to reach the check --
    at 100 kHz/s a full 25 MHz passband sweeps in 250 s and always fits."""
    text = BAND_INI_TEXT.replace('"chirp-rate": 100000', '"chirp-rate": 50000')
    with pytest.raises(control.ControlError, match="repetition period"):
        control.plan_band(_band_parser(text), band_start_mhz=7.5,
                          check_recorder=False)


def test_the_ringbuffer_budget_warns_and_does_not_refuse():
    """`r` is measured and it moves, so this degrades rather than fails --
    and it fires on the band the station is running today, which is sec. 3's
    8.37% loss showing up where an operator can see it before pressing."""
    plan = control.plan_band(_band_parser(), band_start_mhz=7.5,
                             check_recorder=False)
    assert plan.sweep_seconds["NIC3"] > plan.budget_seconds
    assert any("ringbuffer budget" in w for w in plan.warnings)
    assert plan.changes, "a warning must not suppress the plan"


def test_a_narrower_window_comes_in_under_the_budget():
    """The remedy the panel exists to make visible: analyse less, lose none."""
    plan = control.plan_band(_band_parser(), band_start_mhz=7.5,
                             analysis_min_mhz=7.5, analysis_max_mhz=27.5,
                             check_recorder=False)
    assert plan.sweep_seconds["NIC3"] == 200.0
    assert not plan.warnings


def test_an_off_grid_floor_is_snapped_and_said_so():
    """Patch 0013 skips whole blocks to reach `minimum_analysis_frequency`.
    Off-grid, the downconverter starts mid-block and the frequency axis shifts
    under every product -- silently. 4000 x 625 samples at 100 kHz/s over
    25 MS/s is a 10 kHz grid."""
    plan = control.plan_band(_band_parser(), band_start_mhz=7.5,
                             analysis_min_mhz=7.503, check_recorder=False)
    assert plan.changes["minimum_analysis_frequency"] == "7.5e6"
    assert any("snapped" in note for note in plan.notes)


def test_an_on_grid_floor_is_left_alone():
    plan = control.plan_band(_band_parser(), band_start_mhz=7.5,
                             analysis_min_mhz=7.51, check_recorder=False)
    assert plan.changes["minimum_analysis_frequency"] == "7.51e6"
    assert not plan.notes


def test_min_freq_binds_only_with_manual_freq_extent():
    """`calc_ionograms.py:326` reads min_freq/max_freq only when this is true.
    Written without it, the narrowing is accepted and silently ignored."""
    text = BAND_INI_TEXT.replace("manual_freq_extent = true",
                                 "manual_freq_extent = false")
    plan = control.plan_band(_band_parser(text), band_start_mhz=7.5,
                             analysis_max_mhz=27.5, check_recorder=False)
    assert plan.changes["manual_freq_extent"] == "true"


@pytest.mark.parametrize("kw,expect", [
    (dict(band_start_mhz=-1.0), "below zero"),
    (dict(band_start_mhz=7.5, analysis_min_mhz=-0.1), "below zero"),
    (dict(band_start_mhz=7.5, analysis_min_mhz=20.0, analysis_max_mhz=10.0),
     "not above"),
])
def test_the_arithmetic_refusals(kw, expect):
    with pytest.raises(control.ControlError, match=expect):
        control.plan_band(_band_parser(), check_recorder=False, **kw)


# --- the precondition on the binary -----------------------------------------

def test_a_recorder_without_center_freq_is_refused(tmp_path):
    """The failure this whole verb exists to prevent. An unpatched recorder
    tunes to its compiled `set_rx_freq` and ignores the ini, so every product
    is dechirped by the difference with no error anywhere."""
    ok, reason = control.recorder_reads_the_ini(_recorder(tmp_path, patched=False))
    assert not ok
    assert "patch 0014" in reason


def test_a_patched_recorder_is_accepted(tmp_path):
    ok, _ = control.recorder_reads_the_ini(_recorder(tmp_path, patched=True))
    assert ok


@pytest.mark.parametrize("binary", [None, "/nonexistent/rx_uhd_ext_gps"])
def test_an_unreadable_recorder_refuses_rather_than_assumes(binary):
    """"Cannot tell" must fail closed: the failure it guards against is
    silent and costs every sounding until someone checks the LO by hand."""
    ok, _ = control.recorder_reads_the_ini(binary)
    assert not ok


def test_the_precondition_runs_before_anything_is_written(tmp_path):
    with pytest.raises(control.ControlError, match="patch 0014"):
        control.plan_band(_band_parser(), band_start_mhz=7.5,
                          recorder_binary=_recorder(tmp_path, patched=False))


# --- the composite, through apply_config ------------------------------------

@pytest.fixture
def band_station(tmp_path) -> StationConfig:
    ini = tmp_path / "my_station.ini"
    ini.write_text(BAND_INI_TEXT, encoding="utf-8")
    return StationConfig(station="TST", chirp_config=ini,
                         recorder_binary=_recorder(tmp_path, patched=True),
                         output_dir=tmp_path, ringbuffer_dir=tmp_path)


def test_the_five_coupled_keys_have_no_individual_door():
    """Five keys that must agree is the shape that failed. The only way in is
    the composite, which checks them together."""
    for key in control.BAND_INI:
        assert key not in control.EDITABLE, (
            f"{key} is individually editable; it can then be changed alone, "
            f"which is what set_band exists to prevent")


def test_set_band_writes_every_coupled_key_at_once(band_station):
    result = control.apply_config(
        band_station, {"set_band": {"band_start_mhz": 2.5}})
    assert result.ok
    parser = control.read_config(band_station.chirp_config)
    assert parser.get("config", "center_freq") == "15e6"
    assert parser.get("lfm", "minimum_analysis_frequency") == "2.5e6"
    assert parser.get("lfm", "maximum_analysis_frequency") == "27.5e6"
    assert parser.get("lfm", "min_freq") == "2.5e6"
    assert parser.get("lfm", "max_freq") == "27.5e6"
    assert parser.get("lfm", "manual_freq_extent") == "true"
    assert result.journal["band"]["summary"].startswith("digitise 2.500-27.500")


def test_a_refused_band_writes_nothing(band_station):
    before = band_station.chirp_config.read_text()
    with pytest.raises(control.ControlError, match="never sampled"):
        control.apply_config(band_station, {
            "set_band": {"band_start_mhz": 0.0, "analysis_max_mhz": 30.0}})
    assert band_station.chirp_config.read_text() == before


def test_a_band_is_checked_against_a_schedule_sent_in_the_same_command(
        band_station):
    """Both halves of a retune arrive together or the check is on stale data:
    a slower transmitter installed in the same call must be the one the sweep
    is measured against."""
    slow = json.dumps([[dict(ENTRY, **{"chirp-rate": 50000.0,
                                       "transmit_name": "SLOW"})]])
    with pytest.raises(control.ControlError, match="repetition period"):
        control.apply_config(band_station, {
            "mode": "scheduled", "sounder_timings": slow,
            "set_band": {"band_start_mhz": 7.5}})


def test_set_band_refuses_fields_it_does_not_own(band_station):
    """The sample rate has a hardcoded twin at rx_uhd_ext_gps.cpp:173 and must
    divide the N2x0's 100 MHz clock. It stays compiled in."""
    with pytest.raises(control.ControlError, match="sample_rate"):
        control.apply_config(band_station, {
            "set_band": {"band_start_mhz": 7.5, "sample_rate": 30e6}})


def test_set_band_requires_a_band_start(band_station):
    with pytest.raises(control.ControlError, match="band_start_mhz is required"):
        control.apply_config(band_station, {"set_band": {"analysis_min_mhz": 8.0}})


def test_an_unpatched_station_cannot_change_its_band(tmp_path, band_station):
    station = replace(band_station,
                      recorder_binary=_recorder(tmp_path, patched=False))
    with pytest.raises(control.ControlError, match="rebuild"):
        control.apply_config(station, {"set_band": {"band_start_mhz": 2.5}})
