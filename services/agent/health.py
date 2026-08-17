"""Health metrics for one station, collected into an atomic JSON document.

``architecture.md`` sec. 2.5, "Health -- read-only, ship first". The disk-full
failure mode is what this exists to prevent, and the rest of the list is what
this particular station has actually been broken by.

Two rules shape every collector here:

**A collector never raises.** A station reporting nine metrics and one error
is far more useful than a station reporting nothing because ``systemctl`` was
missing. Each returns its own failure as a value, and :func:`collect` always
produces a document.

**A metric that cannot be measured is ``None``, not zero.** "Zero soundings in
the last hour" and "could not tell how many soundings" are different
situations with different responses, and a monitoring system that conflates
them pages you for the wrong thing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import StationConfig

#: Anything older than this and the station is not producing, whatever the
#: processes say.
#:
#: This measures the *sounding's* age, not the file's, so it has to cover the
#: cycle plus however long the pipeline takes to emit the product. Three
#: periods at DOB's 300 s cycle is 900 s and would false-alarm continuously:
#: measured latency there is ~960 s, so the newest sounding is routinely older
#: than 900 s while everything works perfectly.
STALE_PRODUCT_S = 1800.0

#: ``lfm_ionogram-{tx}-{rx}-{ch}-{cid}-{t0:.2f}.h5``. The trailing field is the
#: sounding's start time, from the recorder's GPS-disciplined epoch -- the same
#: number ``muf.io_chirp._NAME_RE`` reads. Duplicated as a bare regex rather
#: than imported because this package is stdlib-only: it runs under the
#: station's Python with no numpy and no h5py.
_PRODUCT_T0_RE = re.compile(r"-(\d{9,12}(?:\.\d+)?)\.h5$")


def _t0_from_name(name: str) -> float | None:
    """The sounding's own start time, or None if the name does not carry one."""
    match = _PRODUCT_T0_RE.search(name)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:                                        # pragma: no cover
        return None

#: How far a product may be stamped ahead of the clock before the age becomes
#: unmeasurable rather than merely small. A few seconds is ordinary skew --
#: mtime granularity, a write finishing after the stat, a network filesystem.
#: DOB's -20420 s was not skew.
FUTURE_PRODUCT_TOLERANCE_S = 5.0

#: Ringbuffer occupancy past this is the hour-before-failure signal. Observed
#: at 94 % on 2026-08-05 with `ringbuffer_max_age_min` too high.
RINGBUFFER_WARN_FRACTION = 0.85

#: Free space below this on the data volume and the station has days, not
#: weeks. `save_raw_voltage` turns days into hours -- see sec. 3.4.
DISK_WARN_FRACTION = 0.10

#: Scatter across the reference transmitter's slots, past which the epoch
#: solve is not a measurement. A sound one at DOB agreed to 0.08 ms across
#: four slots eleven hours apart, and the phase drifts 0.19 ms/hour, so 10 ms
#: is generous. Exceeding it means the slots disagree -- an archive holding
#: two eras, or the wrong transmitter named -- and the honest report is
#: "unknown", not a number.
EPOCH_SCATTER_LIMIT_S = 10e-3

#: A system clock reading earlier than this is not slow, it is unset. The
#: season this archive begins; nothing legitimate predates it. Systemd keeps
#: the same kind of constant in `/usr/lib/clock-epoch` for the same reason.
#: 2026-01-01T00:00:00Z.
CLOCK_SANITY_FLOOR_S = 1_767_225_600.0


@dataclass
class Metric:
    """One measurement, its verdict, and why it could not be taken."""

    name: str
    value: Any = None
    ok: bool | None = None          # None: unknown, not "fine"
    detail: str = ""

    @classmethod
    def unknown(cls, name: str, why: str) -> "Metric":
        return cls(name=name, value=None, ok=None, detail=why)


def _run(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run a command, never raise. Returns ``(returncode, output)``."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode, (proc.stdout or proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, f"{args[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{args[0]}: timed out after {timeout}s"
    except Exception as exc:                                  # pragma: no cover
        return 1, f"{args[0]}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Collectors
# --------------------------------------------------------------------------

def unit_states(config: StationConfig) -> list[Metric]:
    """Is each process of ``chirp.target`` running.

    ``systemctl is-active`` rather than a process-table scan: the units are
    the supervision boundary, and a bare ``pgrep`` cannot tell a service that
    exited cleanly from one that was never started.

    A unit matching ``config.optional_units`` reports its state and never a
    failure. "Not running" and "wrong" are different claims, and for the
    digisonde receivers only the first one is true -- see
    :data:`~services.agent.config.DEFAULT_OPTIONAL_UNITS`.
    """
    out = []
    for unit in config.units:
        code, text = _run(["systemctl", "is-active", unit])
        if text in ("", "unknown") or code == 127:
            out.append(Metric.unknown(f"unit:{unit}", text or "no systemctl"))
        elif text == "active":
            out.append(Metric(f"unit:{unit}", text, ok=True))
        elif any(part in unit for part in config.optional_units):
            # Not `Metric.unknown`: the state was measured and is worth
            # showing. What is unknown is whether anyone minds.
            out.append(Metric(f"unit:{unit}", text, ok=None,
                              detail="optional unit -- the station acquires "
                                     "its own path without it"))
        else:
            out.append(Metric(f"unit:{unit}", text, ok=False))
    return out


def newest_product_age(config: StationConfig) -> Metric:
    """Age of the newest sounding. Soundings stopping is not the same as a
    process dying, and this is the metric that separates them.

    **From the filename, not from mtime.** The trailing field of
    ``lfm_ionogram-...-{t0}.h5`` is the sounding's start time on the recorder's
    GPS-disciplined epoch. A file's mtime is written by whichever clock touched
    it last, and on DOB that has been wrong three separate ways: an RTC that
    booted at 2021-04-02, a CIFS server running 5 h 36 m fast, and the stamps
    that survived on disk after the server was corrected. Data time is the
    thing being asked about anyway -- "when did we last hear the ionosphere",
    not "when did a file appear" -- so reading it directly is both more robust
    and, on a network share with a large archive, far cheaper than stat()ing
    every product.

    mtime remains the fallback for names that carry no epoch.

    A *negative* age is not a fresh product but a broken measurement, and it is
    reported unknown rather than failing: with mtime that means a clock
    disagreement `system_clock_s` already covers, and claiming "products have
    stopped" would be a different and unproven thing to say. It was measured on
    DOB at -20420 s and reported `ok`, because `age < threshold` is trivially
    true for every negative number -- so the one metric watching for
    acquisition stopping was passing unconditionally, and would have gone on
    passing with the recorder dead.
    """
    root = Path(config.output_dir)
    if not root.is_dir():
        return Metric.unknown("newest_product_age_s", f"{root}: no such directory")

    newest_t0, newest_mtime = None, None
    try:
        for path in root.rglob("lfm_ionogram-*.h5"):
            t0 = _t0_from_name(path.name)
            if t0 is not None:
                newest_t0 = t0 if newest_t0 is None else max(newest_t0, t0)
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue                  # vanished mid-scan; the rest still count
            newest_mtime = (mtime if newest_mtime is None
                            else max(newest_mtime, mtime))
    except OSError as exc:
        return Metric.unknown("newest_product_age_s", f"{type(exc).__name__}: {exc}")

    if newest_t0 is not None:
        newest, source = newest_t0, "sounding start time from the filename"
    elif newest_mtime is not None:
        newest, source = newest_mtime, "file mtime (no epoch in the filename)"
    else:
        return Metric("newest_product_age_s", None, ok=False,
                      detail="no products under the output directory at all")

    age = time.time() - newest
    if age < -FUTURE_PRODUCT_TOLERANCE_S:
        if newest_t0 is not None:
            whose = ("the recorder's epoch is ahead of this clock -- one of "
                     "the two is wrong, see system_clock_s and epoch_offset_s")
        else:
            fstype = _fstype_of(root)
            whose = (f"{fstype} share's server clock is ahead"
                     if fstype in REMOTE_FSTYPES else "see system_clock_s")
        return Metric.unknown(
            "newest_product_age_s",
            f"newest product is {-age:.0f}s in the future, so age cannot be "
            f"measured -- {whose}")
    return Metric("newest_product_age_s", round(age, 1), ok=(age < STALE_PRODUCT_S),
                  detail=f"{source}; threshold {STALE_PRODUCT_S:.0f}s")


def disk_free(config: StationConfig) -> list[Metric]:
    """Free space on the data volume and on the ringbuffer.

    Two separate failures. A full data volume stops the archive; a full
    ringbuffer stops acquisition within seconds and is usually a
    `ringbuffer_max_age_min` set too high rather than a disk problem.
    """
    out = []
    for name, path, warn in (
        ("disk_free_fraction", Path(config.output_dir), DISK_WARN_FRACTION),
        ("ringbuffer_free_fraction", Path(config.ringbuffer_dir),
         1.0 - RINGBUFFER_WARN_FRACTION),
    ):
        if not path.exists():
            out.append(Metric.unknown(name, f"{path}: no such path"))
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            out.append(Metric.unknown(name, f"{type(exc).__name__}: {exc}"))
            continue
        fraction = usage.free / usage.total if usage.total else 0.0
        out.append(Metric(name, round(fraction, 4), ok=(fraction > warn),
                          detail=f"{usage.free / 1e9:.1f} GB free of "
                                 f"{usage.total / 1e9:.1f} GB"))
    return out


def sample_rate(config: StationConfig) -> Metric:
    """Configured sample rate against what is expected.

    A silent misconfiguration: nothing fails, the products are simply on a
    different range scale than every other day in the archive.
    """
    if config.expected_sample_rate is None:
        return Metric.unknown("sample_rate_hz", "no expectation configured")
    import configparser

    path = Path(config.chirp_config)
    if not path.is_file():
        return Metric.unknown("sample_rate_hz", f"{path}: no such file")
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
        value = float(json.loads(parser.get("config", "sample_rate")))
    except Exception as exc:
        return Metric.unknown("sample_rate_hz", f"{type(exc).__name__}: {exc}")
    ok = abs(value - config.expected_sample_rate) < 1.0
    return Metric("sample_rate_hz", value, ok=ok,
                  detail=f"expected {config.expected_sample_rate:.0f}")


def uptime_s() -> Metric:
    """Seconds since boot, for the startup grace period."""
    try:
        with open("/proc/uptime", "r", encoding="ascii") as fh:
            return Metric("uptime_s", round(float(fh.read().split()[0]), 1), ok=True)
    except Exception as exc:
        return Metric.unknown("uptime_s", f"{type(exc).__name__}: {exc}")


#: Filesystems whose timestamps are written by somebody else's clock. A file on
#: one of these is stamped by the server, so "the newest file is ahead of me"
#: says nothing whatever about my clock. On DOB it accused a host sitting 47 ms
#: from its NTP server, because the archive moved to a CIFS share on a NAS that
#: was 5 h 43 m fast.
REMOTE_FSTYPES = frozenset({
    "cifs", "smbfs", "smb3", "nfs", "nfs4", "afs", "ncpfs", "9p",
    "fuse.sshfs", "fuse.s3fs", "fuse.rclone", "davfs",
})


def _fstype_of(path: Path) -> str | None:
    """Filesystem type carrying ``path``, or None if it cannot be read.

    By longest matching mount point in ``/proc/self/mounts``. Not ``st_dev``:
    two mounts of the *same* share get different device numbers, so st_dev
    answers "are these the same mount" when the question is "is the clock
    behind this filesystem mine".
    """
    try:
        target = str(Path(path).resolve())
        with open("/proc/self/mounts", encoding="utf-8") as handle:
            entries = [(parts[1].replace("\\040", " "), parts[2])
                       for parts in (line.split() for line in handle)
                       if len(parts) >= 3]
    except OSError:
        return None
    best, best_type = "", None
    for mount, fstype in entries:
        if (target == mount or target.startswith(mount.rstrip("/") + "/")) \
                and len(mount) >= len(best):
            best, best_type = mount, fstype
    return best_type


def _ntp_synchronised() -> bool | None:
    """Does the host believe its clock is disciplined. ``None`` if unaskable.

    Parsed from bare ``timedatectl`` rather than ``timedatectl show``, which
    only exists from systemd 239 and the acquisition laptop runs 229. The
    wording moved too: "NTP synchronized" on the old one, "System clock
    synchronized" on the new.
    """
    code, text = _run(["timedatectl"])
    if code != 0:
        return None
    match = re.search(r"(?:NTP|System clock)\s+synchroniz\w*:\s*(yes|no)",
                      text, re.IGNORECASE)
    return match.group(1).lower() == "yes" if match else None


def system_clock(config: StationConfig) -> Metric:
    """Is the host clock plausible at all, before anything is asked of it.

    The precondition for every other timing metric, and the one that has to be
    answerable with no data on disk. Stock ``rx_uhd_ext_gps`` takes the PPS
    *edge* from the GPSDO and the *second number* from this clock
    (``rx_uhd_ext_gps.cpp:433``, ``set_time_next_pps(pc_secs + 1)``); it waits
    for ``gps_locked``, prints it, and never reads the ``gps_time`` sensor that
    would make the epoch exact. So the host clock's error lands whole in every
    sample timestamp. That is what the 0.956 s offset was, and on 2026-08-06
    the same line stamped a run 2021-04-02 because the RTC had lost five years
    and NTP had not yet stepped it.

    ``patches/0001`` removes that dependency by reading ``gps_time``, and DOB
    now runs a build that does. This metric stays regardless: the patch falls
    back to the host clock whenever the GPSDO is absent or unlocked, which is
    precisely when nobody is watching.

    :func:`epoch_offset` cannot cover this. It needs recent ``par-*.h5``, and a
    clock this wrong means there are none -- it answers "no timing solutions"
    and the operator learns nothing. This one answers at boot, from nothing.
    """
    now = time.time()
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

    if now < CLOCK_SANITY_FLOOR_S:
        floor = time.strftime("%Y-%m-%d", time.gmtime(CLOCK_SANITY_FLOOR_S))
        return Metric("system_clock_s", round(now, 1), ok=False,
                      detail=f"clock reads {when}, before {floor} -- it is "
                             f"unset, not slow. Every recorded sample will "
                             f"carry this epoch. Step it before starting "
                             f"acquisition, then `hwclock -w`")

    # A clock behind files already written is a definite failure needing no
    # hardcoded date: those files were stamped by a clock that ran later.
    newest = None
    try:
        root = Path(config.output_dir)
        if root.is_dir():
            for path in root.rglob("*.h5"):
                mtime = path.stat().st_mtime
                if newest is None or mtime > newest:
                    newest = mtime
    except OSError:
        newest = None

    # "Files ahead of me" only convicts my clock if my clock wrote them. On a
    # network share the server stamps them, and the inference is not merely
    # weaker -- it is about a different machine. Returning here on a CIFS
    # archive also skipped the NTP check below, which is the only part of this
    # function that is about *this* host.
    note = ""
    if newest is not None and now < newest - 60.0:
        fstype = _fstype_of(Path(config.output_dir))
        if fstype in REMOTE_FSTYPES:
            # Deliberately not an instruction. A skew that is already fixed
            # persists in the stamps of files written before the fix, and it
            # takes exactly as long to age out as the skew was large -- 5.6 h
            # on DOB. Telling the operator to go and fix a server they have
            # just fixed is how a note stops being read. Whether the gap
            # shrinks is the thing that distinguishes the two cases, so say
            # that instead and let them look twice.
            note = (f"; the newest product's mtime is "
                    f"{(newest - now) / 3600.0:.1f} h ahead, but the archive "
                    f"is {fstype} -- those stamps come from the file server, "
                    f"not from here. Stamps written before a clock fix age out "
                    f"on their own; a gap that does not shrink is a server "
                    f"clock still to correct")
        else:
            behind = (newest - now) / 86400.0
            return Metric("system_clock_s", round(now, 1), ok=False,
                          detail=f"clock reads {when}, {behind:.1f} days behind "
                                 f"products already on disk -- the RTC lost time "
                                 f"and NTP has not stepped it")

    synced = _ntp_synchronised()
    if synced is None:
        return Metric("system_clock_s", round(now, 1), ok=None,
                      detail=f"{when}; plausible, but NTP state is unreadable "
                             f"(no timedatectl) so nothing is holding it{note}")
    if not synced:
        return Metric("system_clock_s", round(now, 1), ok=False,
                      detail=f"{when} is plausible but NTP is not "
                             f"synchronised; nothing is holding the epoch and "
                             f"the recorder copies it into every timestamp{note}")
    return Metric("system_clock_s", round(now, 1), ok=True,
                  detail=f"{when}, NTP synchronised{note}")


def epoch_offset(config: StationConfig, max_age_s: float = 6 * 3600.0) -> Metric:
    """Receiver clock against a transmitter of known position and schedule.

    The metric this station earned. On 2026-08-05 its epoch was 0.956 s out;
    every product looked perfect -- stable to 0.5 ms, self-consistent
    schedule, plausible ionograms -- and every range was nonsense. Nothing
    internal to the files could reveal it, and nothing did for two days.

    Reported in seconds. Anything past a millisecond is 300 km of range error;
    past half a second the transmit *second* is wrong too and identification
    fails along with ranging.
    """
    spec = config.reference_tx or {}
    if not spec:
        return Metric.unknown("epoch_offset_s", "no reference transmitter configured")
    try:
        from muf import io_detect
    except Exception as exc:
        return Metric.unknown("epoch_offset_s", f"muf unavailable: {exc}")

    root = Path(config.output_dir)
    if not root.is_dir():
        return Metric.unknown("epoch_offset_s", f"{root}: no such directory")

    cutoff = time.time() - max_age_s
    try:
        recent = [p for p in root.rglob("par-*.h5") if p.stat().st_mtime > cutoff]
    except OSError as exc:
        return Metric.unknown("epoch_offset_s", f"{type(exc).__name__}: {exc}")
    if not recent:
        return Metric.unknown(
            "epoch_offset_s",
            f"no timing solutions in the last {max_age_s / 3600:.0f} h")

    try:
        solutions = []
        for path in recent:
            try:
                solutions.append(io_detect.read_timing(path))
            except Exception:
                continue
        offset = io_detect.solve_epoch_offset(
            solutions,
            rate=float(spec["rate"]),
            transmit_seconds=tuple(spec["transmit_seconds"]),
            distance_km=float(spec["distance_km"]),
            cycle_s=float(spec.get("cycle_s", 300.0)),
            reference=str(spec.get("name", "reference")),
            window_s=float(spec.get("window_s", 1.5)),
        )
    except ValueError as exc:
        # Not an error: the reference simply was not transmitting, or was not
        # heard. Saying "unknown" is right; saying "0.0" would be a lie.
        return Metric.unknown("epoch_offset_s", str(exc))
    except Exception as exc:                                  # pragma: no cover
        return Metric.unknown("epoch_offset_s", f"{type(exc).__name__}: {exc}")

    if offset.residual_sd_s > EPOCH_SCATTER_LIMIT_S:
        return Metric.unknown(
            "epoch_offset_s",
            f"slots disagree by +/-{offset.residual_sd_s * 1e3:.0f} ms "
            f"({offset.range_uncertainty_km:.0f} km) across {offset.n_slots} "
            f"slots -- not one transmitter on one clock. Either the archive "
            f"spans a clock change, or {spec.get('name', 'the reference')} is "
            f"not what is being heard.")

    # Split into the two failures it actually causes. Reporting the raw
    # product of offset and c gives 286,440 km for the real DOB fault -- a
    # number larger than the planet, which tells a human nothing. Whole
    # seconds break transmitter *identification*; the remainder breaks range.
    whole = round(offset.seconds)
    remainder = offset.seconds - whole
    parts = []
    if whole:
        parts.append(f"{whole:+d} whole second(s) -- transmit seconds are "
                     f"misidentified")
    parts.append(f"{remainder * 1e3:+.1f} ms = {abs(remainder) * 299792.458:.0f} km "
                 f"of range error")

    return Metric(
        "epoch_offset_s", round(offset.seconds, 6),
        ok=abs(offset.seconds) < 1e-3,
        detail=(f"{spec.get('name', 'reference')}, {offset.n_slots} slots, "
                f"{offset.n_samples} samples, +/-{offset.residual_sd_s * 1e3:.2f} ms "
                f"({offset.range_uncertainty_km:.0f} km); " + "; ".join(parts)),
    )


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

@dataclass
class HealthReport:
    station: str
    timestamp: float
    metrics: list[Metric] = field(default_factory=list)
    agent_version: str = "0.1.0"

    @property
    def in_startup_grace(self) -> bool:
        up = self.metric("uptime_s")
        return bool(up and up.value is not None and up.value < self._grace)

    def metric(self, name: str) -> Metric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    @property
    def failing(self) -> list[Metric]:
        return [m for m in self.metrics if m.ok is False]

    @property
    def unknown(self) -> list[Metric]:
        return [m for m in self.metrics if m.ok is None]

    @property
    def healthy(self) -> bool:
        """False only on a *definite* failure.

        An unknown metric is not a failure -- it is a gap in observation, and
        treating the two alike means a missing `systemctl` pages someone at
        03:00 for a station that is fine.
        """
        return not self.failing

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps({
            "station": self.station,
            "timestamp": self.timestamp,
            "agent_version": self.agent_version,
            "healthy": self.healthy,
            "metrics": [asdict(m) for m in self.metrics],
        }, indent=indent, sort_keys=False)

    _grace: float = 300.0


def collect(config: StationConfig | None = None, *,
            include_epoch: bool = True) -> HealthReport:
    """Every metric, in one document. Never raises."""
    config = config or StationConfig()
    metrics: list[Metric] = []
    metrics.extend(unit_states(config))
    metrics.append(newest_product_age(config))
    metrics.extend(disk_free(config))
    metrics.append(sample_rate(config))
    metrics.append(uptime_s())
    metrics.append(system_clock(config))
    if include_epoch:
        metrics.append(epoch_offset(config))

    report = HealthReport(station=config.station, timestamp=time.time(),
                          metrics=metrics)
    report._grace = config.startup_grace_s
    return report
