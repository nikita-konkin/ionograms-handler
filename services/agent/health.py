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

import configparser
import json
import math
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

#: The whole name, when the transmitter is wanted too. Same shape as
#: ``muf.io_chirp._NAME_RE`` and duplicated for the same reason as above.
_PRODUCT_NAME_RE = re.compile(
    r"^lfm_ionogram-(?P<tx>.+?)-(?P<rx>.+?)-(?P<ch>.+?)-(?P<cid>\d+)"
    r"-(?P<t0>[\d.]+)\.h5$"
)


def _t0_from_name(name: str) -> float | None:
    """The sounding's own start time, or None if the name does not carry one."""
    match = _PRODUCT_T0_RE.search(name)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:                                        # pragma: no cover
        return None


def scan_products(root: Path) -> tuple[dict[str, tuple[Path, float]], float | None]:
    """One walk of ``root``: newest product per transmitter, plus the mtime
    fallback for names that carry no epoch.

    Split out of :func:`newest_product_age` because the preview collector wants
    the same files and the *walk* is the expensive half of both. On DOB
    ``output_dir`` is a USB volume behind a FUSE daemon that competes with the
    recorder for CPU, so walking it twice to answer two questions about one set
    of names would be paying that cost for nothing.

    The transmitter comes out of the filename, which means grouping by circuit
    costs no extra I/O and does not open a single product.
    """
    by_tx: dict[str, tuple[Path, float]] = {}
    newest_mtime = None
    for path in root.rglob("lfm_ionogram-*.h5"):
        t0 = _t0_from_name(path.name)
        if t0 is not None:
            match = _PRODUCT_NAME_RE.match(path.name)
            # A name with an epoch but no parseable transmitter still counts as
            # a product -- it is what the `unkown` marker looks like -- so it is
            # grouped under the empty name rather than dropped.
            tx = match.group("tx") if match is not None else ""
            known = by_tx.get(tx)
            if known is None or t0 > known[1]:
                by_tx[tx] = (path, t0)
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue                      # vanished mid-scan; the rest still count
        newest_mtime = (mtime if newest_mtime is None
                        else max(newest_mtime, mtime))
    return by_tx, newest_mtime

#: How far a product may be stamped ahead of the clock before the age becomes
#: unmeasurable rather than merely small. A few seconds is ordinary skew --
#: mtime granularity, a write finishing after the stat, a network filesystem.
#: DOB's -20420 s was not skew.
FUTURE_PRODUCT_TOLERANCE_S = 5.0

#: Ringbuffer occupancy past this is the hour-before-failure signal. Observed
#: at 94 % on 2026-08-05 with `ringbuffer_max_age_min` too high.
#:
#: **Fallback only**, used when the ring's own cap cannot be read. A ring
#: sized at 14000MB of a 16.8 GB tmpfs leaves 16.7% free when it is working
#: perfectly, against the 15% this allows -- 1.7 points, about 285 MB. The
#: threshold and the buffer size were set independently and nearly collide,
#: so on that station this reads as a station one bad minute from red,
#: forever. See :data:`RINGBUFFER_HEADROOM_FLOOR`.
RINGBUFFER_WARN_FRACTION = 0.85

#: Of the space the ring's own ``-z`` deliberately leaves free, how much may
#: be gone before that is a failure.
#:
#: 1.0 is a ring exactly at its cap, which is success -- ``drf`` printing
#: ``99% size`` on a steady cadence is what a healthy trimmer looks like.
#: Below 1.0 something is in the ring's headroom: the trimmer has stopped,
#: or another tenant is on the same tmpfs.
#:
#: Half, not something tighter, because this metric cannot be an early
#: warning and pretending otherwise would be the same mistake again. At
#: 25 MS/s the recorder writes 100 MB/s, so a dead trimmer crosses 2.8 GB of
#: headroom in **28 seconds** -- less than one push interval. What this
#: catches is the aftermath. The leading indicator is the bounded ``% size``
#: line in ``journalctl -u chirp-ringbuffer``, which stops or reaches 100%.
RINGBUFFER_HEADROOM_FLOOR = 0.5

#: How long a healthy archive timer may stay silent before that is a failure
#: rather than a wait. The sync timer runs every 5 min and the prune hourly,
#: so two hours clears both by a wide margin and still catches a timer that
#: was disabled or never enabled after a unit edit.
JOB_SILENT_S = 7200.0

#: Free space below this on the data volume and the station has days, not
#: weeks. `save_raw_voltage` turns days into hours -- see sec. 3.4.
DISK_WARN_FRACTION = 0.10

#: Metres per second, for turning an unmodelled path excess into the delay it
#: costs. Named rather than inlined because two places now need the same one.
C_KM_S = 299792.458

#: Earth radius used for the hop geometry. Matches `muf.geometry`, which this
#: cannot import: the agent runs on the station's 3.8 interpreter with no
#: scientific stack, and a health metric must never depend on one.
EARTH_RADIUS_KM = 6371.0

#: How far *early* an arrival may be before the reading is a clock fault.
#:
#: This is the tight side, and it is tight because the physics is one-sided.
#: `solve_epoch_offset` models ``tau = distance_km / c`` -- a straight line
#: along the ground, with no ionosphere in it. A real signal reflects off a
#: layer 300-450 km up, so its path is always *longer* and it can only arrive
#: **late**. An arrival earlier than the ground line is not a short path; there
#: is no such thing. It is the receiver's clock running slow.
#:
#: The fault this station actually had was -2.108 ms, so half a millisecond
#: catches it with room to spare while clearing the +/-0.07 ms scatter of a
#: sound solve by seven times over.
EPOCH_EARLY_LIMIT_S = 0.5e-3


def hop_excess_s(distance_km: float, hops: int, virtual_height_km: float) -> float:
    """How much later than the ground line a reflected path arrives.

    Spherical, not the flat-Earth ``sqrt(d^2 + 4h^2)``: at ~1300 km per hop the
    two differ by 13%, and the difference lands directly in a threshold. Each
    hop is two equal slant legs to a mirror at ``virtual_height_km``; the leg
    is the third side of a triangle with the Earth's centre.
    """
    if hops < 1 or distance_km <= 0 or virtual_height_km <= 0:
        return 0.0
    half_ground = (distance_km / hops) / 2.0
    phi = half_ground / EARTH_RADIUS_KM
    r, rh = EARTH_RADIUS_KM, EARTH_RADIUS_KM + virtual_height_km
    leg = math.sqrt(max(0.0, r * r + rh * rh - 2 * r * rh * math.cos(phi)))
    return max(0.0, 2.0 * hops * leg - distance_km) / C_KM_S

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
    def unknown(cls, name: str, why: str) -> Metric:
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


#: Suffixes systemd loads. A `.bak-2026-09-14` beside a unit is ignored by
#: systemd and must be ignored here too, or every `sed -i.bak` would report
#: drift against a file nothing reads.
UNIT_SUFFIXES = (".service", ".timer", ".target", ".mount", ".automount",
                 ".socket", ".path")


def _directives(text: str) -> list[str]:
    """The lines systemd acts on: comments and blank lines dropped.

    Comments are most of the bulk of these files and they are *supposed* to
    change -- a repo that gains a paragraph of reasoning has not drifted. What
    must not differ is the settings, and both incidents here were one line:
    an ``Environment=`` naming a different destination.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", ";")):
            kept.append(stripped)
    return kept


def units_match_repo(config: StationConfig) -> Metric:
    """Do the installed unit files still say what the repo's copies say.

    Nothing else checks this. `systemctl` reads `/etc/systemd/system` and no
    document reads the repo, so the two drift silently and in both directions:
    the installed `chirp-archive-sync.service` was edited in place on
    2026-08-22 and went unnoticed for four days, and on 2026-09-14 the repo
    copies turned out to be pointed at a different NAS entirely, one routine
    `sudo cp` away from redirecting the archive with every unit still green.

    Only units present in **both** places are compared. A unit in the repo and
    not installed is the ordinary state of a station that does not run all of
    them -- the digisonde receivers, the mount units before they are installed
    -- and reporting that as drift would be the false-red this whole config
    was shaped to avoid.
    """
    name = "units_match_repo"
    source = Path(config.unit_source_dir) if config.unit_source_dir else None
    if source is None or not source.is_dir():
        return Metric.unknown(name, f"{source}: no repo unit directory to compare")

    installed = Path(config.unit_install_dir)
    checked, differ, unreadable = 0, [], 0
    for path in sorted(source.iterdir()):
        if path.suffix not in UNIT_SUFFIXES:
            continue
        target = installed / path.name
        if not target.is_file():
            continue
        try:
            mine = _directives(path.read_text(encoding="utf-8", errors="replace"))
            theirs = _directives(target.read_text(encoding="utf-8",
                                                  errors="replace"))
        except OSError:
            unreadable += 1
            continue
        checked += 1
        if mine != theirs:
            differ.append(path.name)

    if not checked:
        return Metric.unknown(
            name,
            f"none of the repo's units are installed in {installed}"
            + (f" ({unreadable} unreadable)" if unreadable else ""))
    if not differ:
        return Metric(name, checked, ok=True,
                      detail=f"{checked} installed unit(s) match {source}")

    # Deliberately does not say "run sudo cp". Which copy is right is not
    # knowable from here, and on 2026-09-14 it was the installed one.
    return Metric(
        name, None, ok=False,
        detail=(f"{', '.join(differ)} differ from {source} in a directive, not "
                f"a comment. systemd obeys {installed}; the repo copy is what a "
                f"deploy would install over it. Diff them and decide which is "
                f"right before copying either way -- the repo copy has named "
                f"the wrong NAS before now."))

def _show(unit: str, prop: str) -> tuple[int, str]:
    """One ``systemctl show`` property, on any systemd this station may have.

    Deliberately **not** ``--value``, which arrived in systemd 230 (2016).
    The HP ZBook answers ``unknown option --value``, and the consequence was
    worse than a metric going missing: :func:`_run` hands back the error text,
    which is *non-empty*, so a caller testing ``if not result`` sails straight
    past the failure and reports the error string as the property's value.
    Both archive jobs then read red, with a garbage ``Result``, on a station
    whose jobs were running perfectly -- the "false reds" failure the station
    config was written to avoid.

    The default ``Property=value`` output predates ``--value`` by years and is
    what every version prints, so ask for that and drop the prefix. A
    non-zero exit returns no value at all rather than its own error text,
    which is what makes the callers' ``not result`` check mean what it says.
    """
    code, text = _run(["systemctl", "show", unit, f"--property={prop}"])
    if code != 0:
        return code, ""
    head, sep, value = text.partition("=")
    if not sep or head.strip() != prop:
        return 1, ""
    return 0, value.strip()


def _unit_environment(unit: str) -> dict[str, str]:
    """``Environment=`` of a unit, as systemd resolved it.

    Read from the running unit rather than from this station's config, and
    that is the point: the two are allowed to disagree and did. The installed
    ``chirp-archive-sync.service`` had been edited in place to a different
    ``ARCHIVE_REMOTE`` than the repo's copy, so every document describing the
    station named a path the station had not written to since 22 August.
    Whatever the console reports here is what rsync will actually use.
    """
    code, text = _show(unit, "Environment")
    if code != 0 or not text:
        return {}
    found = {}
    for item in text.split():
        key, sep, value = item.partition("=")
        if sep:
            found[key] = value
    return found


def _age_of(usec: str) -> float | None:
    """Seconds since a systemd ``*USec`` timestamp, or None if it never ran."""
    try:
        stamp = int(usec)
    except (TypeError, ValueError):
        return None
    # systemd reports 0 for "never", and the monotonic variants are relative
    # to boot rather than the epoch -- neither is an instant we can subtract.
    return time.time() - stamp / 1e6 if stamp > 0 else None


def job_states(config: StationConfig) -> list[Metric]:
    """The archive jobs: did the last run succeed, and is the timer still firing.

    Two failures, and only the first one looks like a failure.

    A job that **fails** leaves ``Result=exit-code`` and is red here. That is
    the 2026-08-26 case: rsync exiting 11 every five minutes because the
    destination share was full, for thirteen hours, with nothing on any page
    saying so.

    A timer that is **disabled** is the quieter one. Nothing turns red --
    there is no failed run to report, because there are no runs. The archive
    simply stops growing, and the first symptom is a forecast that has run out
    of measurements days later. So the timer's own state and the age of its
    last trigger are reported beside the job, and a timer that has not fired
    in more than :data:`JOB_SILENT_S` is a definite failure rather than an
    unknown.

    ``rsync`` exit codes 23 and 24 -- partial transfer, vanished source file --
    are ordinary here and the unit already lists them in
    ``SuccessExitStatus``, so systemd reports them as success and this reads
    them the same way. That is deliberate: on a station deleting detection
    snippets minutes after writing them, a vanished file is the normal case.
    """
    out: list[Metric] = []
    for unit in config.job_units:
        code, result = _show(unit, "Result")
        if code != 0 or not result:
            out.append(Metric.unknown(
                f"job:{unit}",
                "systemctl could not be asked" if code == 127
                else "systemctl reported no Result for this unit"))
            continue

        _, status = _show(unit, "ExecMainStatus")
        _, when = _show(unit, "ExecMainExitTimestamp")
        _, condition = _show(unit, "ConditionResult")

        env = _unit_environment(unit)
        remote = env.get("ARCHIVE_REMOTE", "")
        where = f" -> {remote}" if remote else ""
        ran = f", last ran {when}" if when else ", never run"

        # `Result` is `success` on a unit that has never run a single time --
        # it is systemd's default, not a verdict -- and `success` is also what
        # a start skipped by ConditionPathIsMountPoint leaves behind. Both
        # rendered green, one of them under the sentence "last run succeeded,
        # never run". Each is handled before `result` is trusted.
        if condition == "no":
            # The most urgent of the three, and previously invisible in both
            # directions: a skipped start is not a failed run, so `Result`
            # stays at whatever the last real run left, and the timer keeps
            # firing on schedule so the silence check never trips either. The
            # archive simply stops moving.
            return_detail = (
                f"the last start was SKIPPED -- a condition was not met, "
                f"almost always the archive share not being mounted{where}. "
                f"systemd does not count that as a failure, so this job can "
                f"skip every pass for days with the timer firing normally"
                f"{ran}. `systemctl show {unit} --property=ConditionResult` "
                f"and `findmnt` on the share say which.")
            out.append(Metric(f"job:{unit}", "skipped", ok=False,
                              detail=return_detail))
        elif not when:
            out.append(Metric.unknown(
                f"job:{unit}",
                f"never run{where}. `Result={result}` is the value systemd "
                f"carries before a unit has run at all, so it is not evidence "
                f"of anything -- green here would have been a claim nothing "
                f"measured. The timer metric beside this one says whether it "
                f"is about to."))
        elif result == "success":
            out.append(Metric(f"job:{unit}", result, ok=True,
                              detail=f"last run succeeded{ran}{where}"))
        else:
            out.append(Metric(f"job:{unit}", result, ok=False,
                              detail=f"last run failed with exit status "
                                     f"{status or '?'}{ran}{where}. "
                                     f"`journalctl -u {unit} -n 20` says why."))

        out.append(_timer_metric(unit))
    return out


def _timer_metric(unit: str) -> Metric:
    """The timer behind a job unit, by systemd's own naming convention."""
    timer = unit.rsplit(".", 1)[0] + ".timer"
    code, state = _show(timer, "ActiveState")
    if code != 0 or not state or state == "unknown":
        return Metric.unknown(
            f"timer:{timer}",
            "systemctl could not be asked" if code == 127
            else state or "systemctl reported no ActiveState for this timer")

    _, last = _show(timer, "LastTriggerUSec")
    age = _age_of(last)

    if state != "active":
        return Metric(f"timer:{timer}", state, ok=False,
                      detail="the timer is not running, so this job will "
                             "never fire again. Nothing will turn red as the "
                             "archive falls behind -- there are no failed "
                             "runs when there are no runs.")
    if age is None:
        return Metric(f"timer:{timer}", state, ok=None,
                      detail="active, but it has not fired yet")
    if age > JOB_SILENT_S:
        return Metric(f"timer:{timer}", state, ok=False,
                      detail=f"active but last fired {age / 3600:.1f} h ago, "
                             f"past the {JOB_SILENT_S / 3600:.0f} h this "
                             f"station expects")
    return Metric(f"timer:{timer}", state, ok=True,
                  detail=f"last fired {age / 60:.0f} min ago")


def archive_remote_free(config: StationConfig) -> list[Metric]:
    """Free space where the archive is being *sent*, not only where it is staged.

    ``disk_free`` measures ``output_dir`` and the ringbuffer -- both local,
    both on this station. Neither says anything about the volume rsync writes
    to, and on 2026-08-26 that was the one that had filled: 7.2 TB, 100% used,
    zero bytes free, while the station's own disks were comfortable and the
    console was green.

    The path comes from the unit's ``ARCHIVE_REMOTE``, so this measures the
    destination in use rather than a configured guess. An unmounted share is
    reported unknown rather than full: an empty mountpoint directory on the
    root filesystem would otherwise report the root's free space and read as
    healthy, which is precisely the wrong answer.
    """
    seen: set[str] = set()
    out: list[Metric] = []
    for unit in config.job_units:
        remote = _unit_environment(unit).get("ARCHIVE_REMOTE", "")
        if not remote or remote in seen:
            continue
        seen.add(remote)
        path = Path(remote)
        name = "archive_remote_free_fraction"

        if not path.exists():
            out.append(Metric.unknown(name, f"{path}: not present -- the "
                                            f"share may not be mounted"))
            continue
        # The same guard `prune.py` makes before deleting anything: an
        # unmounted share is an empty directory that every existence check
        # passes.
        if path.is_dir() and not any(path.iterdir()):
            out.append(Metric.unknown(name, f"{path}: empty -- an unmounted "
                                            f"share looks exactly like this"))
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            out.append(Metric.unknown(name, f"{type(exc).__name__}: {exc}"))
            continue

        fraction = usage.free / usage.total if usage.total else 0.0
        out.append(Metric(name, round(fraction, 4),
                          ok=(fraction > DISK_WARN_FRACTION),
                          detail=f"{usage.free / 1e9:.1f} GB free of "
                                 f"{usage.total / 1e9:.1f} GB at {path}"))
    return out


def _norm(raw) -> str:
    """A path from a unit file or an ini, in a form two of them can be compared.

    Empty stays empty rather than becoming ``"."``: an unset variable means
    "no answer", and `os.path.normpath("")` would turn that into a real path
    that could then be reported as disagreeing with something.
    """
    text = str(raw).strip().strip('"').strip("'").strip()
    return os.path.normpath(os.path.expanduser(text)) if text else ""


def effective_output_dir(config: StationConfig) -> str:
    """Where acquisition actually writes: the ini when it says, else the config.

    The precedence `StationConfig.output_dir` documents -- read from
    ``chirp_config`` when present, this being the fallback -- made explicit,
    because the answer to "is the staging path consistent" depends entirely on
    which of the two is the live one. `set_config` edits the ini, so on any
    station the api has ever configured, the ini is the live one.
    """
    parser = configparser.ConfigParser()
    try:
        if parser.read(Path(config.chirp_config)):
            found = _norm(parser.get("config", "output_dir", fallback=""))
            if found:
                return found
    except (configparser.Error, OSError, UnicodeDecodeError):
        pass
    return _norm(str(config.output_dir))



def product_root(config: StationConfig) -> Path:
    """Where products actually are, for anything that goes looking for them.

    :func:`effective_output_dir` as a path, and the thing every collector below
    measures rather than the config's own ``output_dir`` field. Those two are
    the same on a station whose ini and agent.json agree, and when they do not,
    the ini wins because that is the file acquisition reads.

    This is the difference between the three symptoms of a drifted path and
    their cause. `test_one_output_dir_is_written_in_three_places` lists them:
    ``newest_product_age_s`` answering "no such directory" for a recorder that
    is producing normally, the preview finding nothing to encode, and the
    mirror reporting success over an empty tree. All three are this function
    reading one folder while the recorder writes to another.
    """
    return Path(effective_output_dir(config))

def archive_local(config: StationConfig) -> Path | None:
    """The staging folder the copy jobs are running with, or ``None``.

    Read out of the running units for the same reason as
    :func:`_unit_environment`: the installed unit and the repo's copy are
    allowed to disagree, and have.

    ``None`` is "no answer, do not check" -- no job units, no systemctl, or no
    ``ARCHIVE_LOCAL`` in any of them. A caller must not read it as a mismatch:
    a guard that fires when it cannot see is one that gets switched off.
    """
    for unit in config.job_units:
        found = _norm(_unit_environment(unit).get("ARCHIVE_LOCAL", ""))
        if found:
            return Path(found)
    return None


def archive_paths_agree(config: StationConfig) -> Metric:
    """One staging path, or the reason products are about to pile up unseen.

    `chirp-archive-sync.service` states the requirement itself -- *"Keep this
    identical to `output_dir` in agent.json and to ARCHIVE_LOCAL in
    chirp-archive-prune.service: three places, one path"* -- and until now
    nothing checked it. The same class of drift went unnoticed for four days
    on DOB, which is why `_unit_environment` reads the running unit at all.

    **Drift here is not a stale document, it is a full disk.** The mirror
    never deletes, deliberately, so the prune is the only thing that reclaims
    the staging volume. A folder the jobs do not know about is copied by
    nothing and reclaimed by nothing: it does not merely stop reaching the
    server, it fills the disk acquisition is running on, while every unit
    stays ``active`` and every other metric stays green.

    Compares the *effective* ``output_dir`` -- see
    :func:`effective_output_dir` -- against every job unit's
    ``ARCHIVE_LOCAL``, and the units against each other.
    """
    name = "archive_paths_agree"
    sources: dict = {}

    def note(path: str, who: str) -> None:
        if path:
            sources.setdefault(path, []).append(who)

    note(effective_output_dir(config), "output_dir")
    for unit in config.job_units:
        note(_norm(_unit_environment(unit).get("ARCHIVE_LOCAL", "")),
             f"{unit} ARCHIVE_LOCAL")

    total = sum(len(who) for who in sources.values())
    if total < 2:
        return Metric.unknown(
            name, "fewer than two staging paths could be read, so there is "
                  "nothing to compare -- systemctl missing, or no job unit "
                  "sets ARCHIVE_LOCAL")
    if len(sources) == 1:
        agreed = next(iter(sources))
        return Metric(name, agreed, ok=True,
                      detail=f"{total} sources name {agreed}")

    where = "; ".join(f"{' + '.join(who)} -> {path}"
                      for path, who in sorted(sources.items()))
    return Metric(name, None, ok=False,
                  detail=f"the staging folder is named {len(sources)} "
                         f"different ways: {where}. Products written where "
                         f"the archive jobs are not looking are copied by "
                         f"nothing and reclaimed by nothing, and the staging "
                         f"volume fills while every unit stays active.")


def newest_product_age(config: StationConfig, scan=None) -> Metric:
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

    ``scan`` is a :func:`scan_products` result the caller already has, so a
    pass that also builds previews walks the archive once rather than twice.
    """
    root = product_root(config)
    if not root.is_dir():
        return Metric.unknown("newest_product_age_s", f"{root}: no such directory")

    try:
        by_tx, newest_mtime = scan_products(root) if scan is None else scan
    except OSError as exc:
        return Metric.unknown("newest_product_age_s", f"{type(exc).__name__}: {exc}")
    newest_t0 = max((t0 for _, t0 in by_tx.values()), default=None)

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


#: ``-z 14000MB`` in the unit's ExecStart. Decimal multipliers, which is what
#: ``drf`` means by MB and what the size-to-seconds arithmetic in
#: ``chirp-ringbuffer.service`` assumes: 12000MB / 100 MB/s = 120 s, measured
#: at 119 s.
#: The separator is optional because ``-z14000MB`` is valid short-option
#: spelling, and a size this cannot read falls back to the flat floor
#: silently -- a miss here is a wrong verdict, not an error.
_SIZE_ARG = re.compile(r"(?:-z|--size)[=\s]*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\b")
_SIZE_SCALE = {"": 1, "K": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12}


def ringbuffer_cap_bytes(config: StationConfig) -> int | None:
    """The ring's own ``-z`` size, read from the running unit.

    From ``systemctl show``, not from the repo's copy of the unit, for the
    reason :func:`_unit_environment` documents: the installed file is the one
    that is true, and the two have diverged before without anyone noticing for
    days.

    ``None`` when there is no unit to ask, the property cannot be read, or no
    size argument is in it -- all ordinary on a station whose ringbuffer is
    started by a script rather than by systemd.
    """
    unit = (config.ringbuffer_unit or "").strip()
    if not unit:
        return None
    code, text = _show(unit, "ExecStart")
    if code != 0 or not text:
        return None
    found = _SIZE_ARG.search(text)
    if not found:
        return None
    try:
        return int(float(found.group(1)) * _SIZE_SCALE[found.group(2)])
    except (ValueError, KeyError, OverflowError):          # pragma: no cover
        return None


def _volume(name: str, path: Path):
    """``shutil.disk_usage``, or the Metric explaining why there is none."""
    if not path.exists():
        return Metric.unknown(name, f"{path}: no such path"), None
    try:
        return None, shutil.disk_usage(path)
    except OSError as exc:
        return Metric.unknown(name, f"{type(exc).__name__}: {exc}"), None


def disk_free(config: StationConfig) -> list[Metric]:
    """Free space on the data volume and on the ringbuffer.

    Two separate failures. A full data volume stops the archive; a full
    ringbuffer stops acquisition within seconds and is usually a
    `ringbuffer_max_age_min` set too high rather than a disk problem.
    """
    out = []

    failed, usage = _volume("disk_free_fraction", product_root(config))
    if failed is not None:
        out.append(failed)
    else:
        fraction = usage.free / usage.total if usage.total else 0.0
        out.append(Metric("disk_free_fraction", round(fraction, 4),
                          ok=(fraction > DISK_WARN_FRACTION),
                          detail=f"{usage.free / 1e9:.1f} GB free of "
                                 f"{usage.total / 1e9:.1f} GB"))

    out.append(_ringbuffer_metric(config))
    return out


def _ringbuffer_metric(config: StationConfig) -> Metric:
    """The ringbuffer volume, judged against the ring's own cap where it can be.

    The value stays free-of-total, unchanged: `BACKLOG.md` sec. 20 tells an
    operator to watch this number across a rotation and `chirp-rx.service`
    points at it by name, so its meaning is not ours to redefine. What changes
    is the verdict.

    A ring at its cap is success, not a warning -- the whole point of ``-z`` is
    that the buffer stays full and old samples fall off the back. Judged on
    free-of-total it looks like a volume 84% consumed, which on this station
    sits 1.7 points from a red it never deserved. Judged against the headroom
    the cap deliberately leaves, the same buffer reads 100% and moves only when
    something is actually in that headroom.
    """
    name = "ringbuffer_free_fraction"
    failed, usage = _volume(name, Path(config.ringbuffer_dir))
    if failed is not None:
        return failed

    fraction = usage.free / usage.total if usage.total else 0.0
    where = f"{usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB"
    cap = ringbuffer_cap_bytes(config)

    # A cap at or past the volume is a misconfiguration, not a reading: there
    # is no headroom to measure, and dividing by it would report the fullest
    # possible buffer as fine.
    if cap is None or not 0 < cap < usage.total:
        why = ("no -z size on " + (config.ringbuffer_unit or "a ringbuffer unit")
               if cap is None
               else f"-z {cap / 1e9:.1f} GB is not smaller than the volume")
        floor = 1.0 - RINGBUFFER_WARN_FRACTION
        return Metric(name, round(fraction, 4), ok=(fraction > floor),
                      detail=f"{where}; judged against a flat {floor:.0%} "
                             f"floor because {why}")

    designed = usage.total - cap
    headroom = usage.free / designed
    return Metric(
        name, round(fraction, 4),
        ok=(headroom > RINGBUFFER_HEADROOM_FLOOR),
        detail=(f"{where}; ring capped at {cap / 1e9:.1f} GB, so "
                f"{designed / 1e9:.1f} GB is headroom by design and "
                f"{headroom:.0%} of it is free "
                f"(floor {RINGBUFFER_HEADROOM_FLOOR:.0%})"))


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


def band(config: StationConfig) -> list[Metric]:
    """The band the station is configured for, and whether it can be changed.

    Reported rather than judged: none of these is a fault on its own, so they
    carry ``ok=True`` when they could be read and ``unknown`` when they could
    not. Nothing here can turn a station red -- see the console's "only a
    definite failure makes a station unhealthy".

    They exist because the console has no other way to show the configured
    band beside the **observed** one. A recorder tuned somewhere other than
    `center_freq` produces perfectly healthy-looking products from the wrong
    spectrum, and printing "configured 7.5-32.5, observed 7.55-32.30" is the
    one thing that would have caught 2026-08-19 within a cycle.

    ``recorder_reads_ini`` is the capability, not a verdict: a station on a
    pre-0014 binary is fine as long as nobody changes the band, so it is
    reported and never failed. It is what the console greys the panel on.
    """
    import configparser


    out: list[Metric] = []
    path = Path(config.chirp_config)
    parser = configparser.ConfigParser()
    try:
        if not path.is_file():
            raise FileNotFoundError(f"{path}: no such file")
        parser.read(path)
        centre = float(json.loads(parser.get("config", "center_freq")))
        rate = float(json.loads(parser.get("config", "sample_rate")))
        lo = float(json.loads(parser.get("lfm", "minimum_analysis_frequency")))
        hi = float(json.loads(parser.get("lfm", "maximum_analysis_frequency")))
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
        return [Metric.unknown(name, why) for name in
                ("center_freq_hz", "band_start_hz", "band_stop_hz",
                 "analysis_min_hz", "analysis_max_hz")] + [
            _recorder_metric(config)]

    out.append(Metric("center_freq_hz", centre, ok=True,
                      detail=f"{centre / 1e6:.3f} MHz"))
    out.append(Metric("band_start_hz", centre - rate / 2.0, ok=True,
                      detail="bottom of the digitised passband"))
    out.append(Metric("band_stop_hz", centre + rate / 2.0, ok=True,
                      detail="top of the digitised passband"))
    out.append(Metric("analysis_min_hz", lo, ok=True,
                      detail=f"{lo / 1e6:.3f} MHz"))
    out.append(Metric("analysis_max_hz", hi, ok=True,
                      detail=f"{hi / 1e6:.3f} MHz"))
    out.append(_recorder_metric(config))
    return out


def _recorder_metric(config: StationConfig) -> Metric:
    from . import control

    ok, reason = control.recorder_reads_the_ini(
        getattr(config, "recorder_binary", None))
    if ok:
        return Metric("recorder_reads_ini", True, ok=True,
                      detail="carries --center-freq (patch 0014)")
    # `unknown`, not False: an unpatched recorder is a limitation, not a
    # fault, and a station that has never needed a band change is healthy.
    return Metric.unknown("recorder_reads_ini", reason)


def uptime_s() -> Metric:
    """Seconds since boot, for the startup grace period."""
    try:
        with open("/proc/uptime", encoding="ascii") as fh:
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
        root = product_root(config)
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
        fstype = _fstype_of(product_root(config))
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

    root = product_root(config)
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

    # The window is asymmetric because the physics is. `solve_epoch_offset`
    # models tau = distance/c, a straight line along the ground with no
    # ionosphere in it, so a real signal -- reflecting off a layer 300-450 km
    # up -- can only ever arrive LATE. Early is not a short path; there is no
    # such thing. It is a slow clock, which is what -2.108 ms was here.
    #
    # Measured 2026-09-14, and this is the reading that forced the change: a
    # symmetric abs() < 1 ms called +1.28 ms a fault. It is not one. The
    # recorder takes its epoch from the GPSDO's gps_time on a PPS edge (patch
    # 0001 prints `[source: GPSDO gps_time]`, and it does), the host clock's
    # own 0.77 ms bias never reaches a sample, and 1.28 ms is two hops over
    # 2588 km at ~330 km virtual height. Ordinary evening F2 geometry, red
    # every night.
    hops = int(spec.get("max_hops", 2) or 2)
    height = float(spec.get("max_virtual_height_km", 450.0) or 450.0)
    late = hop_excess_s(float(spec["distance_km"]), hops, height)

    parts = []
    if whole:
        parts.append(f"{whole:+d} whole second(s) -- transmit seconds are "
                     f"misidentified")
    km = abs(remainder) * C_KM_S
    if -EPOCH_EARLY_LIMIT_S < offset.seconds < late:
        parts.append(f"{remainder * 1e3:+.1f} ms = {km:.0f} km, inside the "
                     f"{late * 1e3:.2f} ms this path can add by reflecting")
    else:
        parts.append(f"{remainder * 1e3:+.1f} ms = {km:.0f} km of range error")

    return Metric(
        "epoch_offset_s", round(offset.seconds, 6),
        ok=(-EPOCH_EARLY_LIMIT_S < offset.seconds < late),
        detail=(f"{spec.get('name', 'reference')}, {offset.n_slots} slots, "
                f"{offset.n_samples} samples, +/-{offset.residual_sd_s * 1e3:.2f} ms "
                f"({offset.range_uncertainty_km:.0f} km); " + "; ".join(parts)
                + f". Window {-EPOCH_EARLY_LIMIT_S * 1e3:+.1f} to "
                  f"{late * 1e3:+.2f} ms -- {hops} hop(s) over "
                  f"{float(spec['distance_km']):.0f} km at {height:.0f} km "
                  f"virtual height, which the solve does not model"),
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
            include_epoch: bool = True, scan=None) -> HealthReport:
    """Every metric, in one document. Never raises.

    ``scan`` is passed straight to :func:`newest_product_age`; see there.
    """
    config = config or StationConfig()
    metrics: list[Metric] = []
    metrics.extend(unit_states(config))
    metrics.extend(job_states(config))
    metrics.append(newest_product_age(config, scan))
    metrics.extend(disk_free(config))
    metrics.extend(archive_remote_free(config))
    metrics.append(archive_paths_agree(config))
    metrics.append(units_match_repo(config))
    metrics.append(sample_rate(config))
    metrics.extend(band(config))
    metrics.append(uptime_s())
    metrics.append(system_clock(config))
    if include_epoch:
        metrics.append(epoch_offset(config))

    report = HealthReport(station=config.station, timestamp=time.time(),
                          metrics=metrics)
    report._grace = config.startup_grace_s
    return report
