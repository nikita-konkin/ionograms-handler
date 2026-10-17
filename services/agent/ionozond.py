"""The ionozond sounder's schedule and parameters, changed from the web console.

Phase 3 of the live console (issue #4,
``docs/2026-10-16-ionozond-parameters-and-schedule.md``). The sounder reads
``chirp_config.py`` once, at start; the Qt console regenerates that file from
``schedule.ini`` and ``config.ini`` whenever one of its dialogs closes. So a
web edit goes into those two files and ``chirp_config.py`` is regenerated as
the console would, or the next dialog closed on the laptop would undo it.

Two fingerprints, for two questions:

* ``base`` hashes the **editable document** -- every transmitter in
  ``schedule.ini`` but the receiver, switched on or not, and the
  parameters the web may set. A queued edit carries the ``base`` it was
  made against, and is refused if the schedule has changed since.
* ``sounder_hash`` hashes **what the sounder reads**: the literals of
  ``chirp_config.py``. It alone drives the configuration history; editing a
  transmitter that is switched off changes nothing the sounder reads, and
  opens no epoch.

**Drift.** On every health pass and before every edit the console's files are
regenerated in memory and compared with ``chirp_config.py`` as the sounder
would read it. When they differ, someone edited one without the other, and an
edit is refused until :func:`regenerate` writes the file again. Since phase 4
(issue #8) the console is retired, so drift means a hand edit of an ini file.

**Taking it up.** The write is followed by ``systemctl --no-block
try-restart``: a running sounder finishes the sounding in progress on SIGTERM
and starts again on the new file, and a stopped one stays stopped. A refused
restart puts every file back, since the running sounder still has the old one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import control, health, schedulecheck, siteconfig
from .config import StationConfig
from .qsettings import PLAIN, Ini, IniError
from .sounderconfig import UnreadableConfig, build, literals, parsed

NAME = "set_ionozond"
REGENERATE = "regenerate_ionozond"

#: The fields of a transmitter the web may set, and those a new one must have.
FIELDS = ("active", "cf", "chirpt", "dur", "fft_count", "lat", "lon", "rate", "rep")
REQUIRED = ("active", "cf", "chirpt", "dur", "lat", "lon", "rate", "rep")

#: The parameters the web may set, from ``[General]`` in ``config.ini``.
#: ``nas_dest``, ``data_dir``, the receive chain and the processing internals
#: are left out on purpose: they are hand edits plus Regenerate. See the
#: phase 3 design, section 3, and the phase 4 design, section 2.
GENERAL = ("fft_count_index", "h5_archive", "keep_lfs", "keep_lfs_hours",
           "nas_keep_h5_days", "save_raw_voltage")
FLAGS = ("active", "h5_archive", "keep_lfs", "save_raw_voltage")

#: A new transmitter's name: ``services/api/control_routes.py``'s ``CODE_RE``.
#: It becomes the dash-delimited ``txname`` of the ``.h5`` archive, so no dash.
#: What a short report carries. The health loop sends the whole block when
#: any of them changes, and every :data:`WHOLE_S` regardless; the server keeps
#: the last whole one. See the phase 4 design, section 10.
SUMMARY = ("base", "sounder_hash", "site_hash", "pending", "drift", "problems")
WHOLE_S = 3600.0

CODE_RE = re.compile(r"^[A-Za-z0-9_.]{1,24}$")

SPAN = re.compile(r"(\d+(?:\.\d+)?)\s*(us|ms|min|s|h|d)?")
SPAN_UNITS = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0,
              "d": 86400.0, None: 1.0}


@dataclass(frozen=True)
class Files:
    """The three files a schedule change touches, as found on the station."""

    schedule: Path
    config: Path
    sounder: Path


@dataclass(frozen=True)
class Plan:
    """A checked change: the bytes before and after, and what to journal."""

    files: Files
    before: tuple
    after: tuple
    journal: dict

    @property
    def paths(self) -> tuple:
        return (self.files.schedule, self.files.config, self.files.sounder)


def state(config: StationConfig) -> dict:
    """The schedule in force, for the health report. Never raises.

    A health push that a schedule problem could silence would turn a bad edit
    into a station that looks dead, so every failure here is a reported
    problem instead.
    """
    try:
        return _state(config)
    except Exception as exc:
        return {"problems": ["%s: %s" % (type(exc).__name__, exc)]}


def apply(config: StationConfig, params: dict) -> list:
    """Check, write and take up one ``set_ionozond``; every outcome acknowledged."""
    return _carried(config, NAME, lambda files: _plan(config, files, params))


def regenerate(config: StationConfig, params: dict) -> list:
    """Write ``chirp_config.py`` again from the ini files, after a hand edit.

    ``params`` carries the ``base`` and ``sounder_hash`` the operator was shown,
    so a file that changed since is refused rather than overwritten.
    """
    return _carried(config, REGENERATE, lambda files: _regenerated(
        config, files, params))


def document(schedule: Ini, config: Ini) -> dict:
    """What the web may edit: every transmitter but the receiver, and the parameters."""
    return {
        "schedule": {group: {key: schedule.value(group, key) for key in FIELDS
                             if key in schedule.keys(group)}
                     for group in schedule.groups() if not _receiver(schedule, group)},
        "general": {key: config.value("", key) for key in GENERAL
                    if key in config.keys("")},
    }


def receivers(schedule: Ini) -> list:
    """The receiving station: the site, not a setting, as the console has it."""
    return [dict({key: schedule.value(group, key) for key in FIELDS
                  if key in schedule.keys(group)}, name=group)
            for group in schedule.groups() if _receiver(schedule, group)]


def settings(config: Ini) -> dict:
    """What the checks need from ``config.ini``, read as ``readCheckConfig`` does."""
    keys = config.keys("")
    return {
        "sample_rate_khz": _qt(config, "sample_rate", 25000.0, float),
        "sample_rate_index": (_qt(config, "sample_rate_index", -1, int)
                              if "sample_rate_index" in keys else -1),
        "dec": _qt(config, "dec", 625.0, float),
        "fft_count_index": _qt(config, "fft_count_index", 5, int),
        "obj_w": _qt(config, "obj_size_horizontal", 9, int),
        "obj_h": _qt(config, "obj_size_vertical", 3, int),
        "obj_level": _qt(config, "obj_level", 11.0, float),
    }


def issues(wanted: dict, sites: list, read: dict,
           timeout_stop_s: float | None) -> list:
    """Every issue with a document, the console's checks and the stop timeout's."""
    entries = [_entry(name, fields, False)
               for name, fields in wanted["schedule"].items()]
    entries += [_entry(site["name"], site, True) for site in sites]
    entries.sort(key=lambda e: e.name)
    index = _int(wanted["general"].get("fft_count_index"), read["fft_count_index"])
    found = schedulecheck.check(entries, schedulecheck.Settings(
        sample_rate_khz=read["sample_rate_khz"],
        sample_rate_index=read["sample_rate_index"], dec=read["dec"],
        fft=(schedulecheck.FFT_COUNT_LIST[index]
             if 0 <= index < len(schedulecheck.FFT_COUNT_LIST) else 16384),
        obj_w=read["obj_w"], obj_h=read["obj_h"], obj_level=read["obj_level"]))
    if timeout_stop_s is not None:
        for e in entries:
            dur = _float(e.dur)
            if not e.active or dur is None:
                continue
            needed = dur + schedulecheck.processing_estimate_s(dur)
            if needed > timeout_stop_s:
                found.append(schedulecheck.Issue(
                    schedulecheck.ERROR, e.name,
                    "a %s s sounding and about %.0f s to write its products outlast "
                    "the unit's %.0f s TimeoutStopSec: a stop during it would end "
                    "in SIGKILL and a truncated capture"
                    % (e.dur, needed - dur, timeout_stop_s)))
    return sorted(found, key=lambda i: i.level)


def merged(current: dict, wanted: dict, sites: list) -> dict:
    """``current`` with ``wanted`` laid over it, or why it cannot be.

    Every transmitter must be listed: one left out is refused rather than
    deleted, since the web offers no deletion. A value left as it was is kept
    as it was, even one the console would not write today, so the checks see
    the schedule exactly as the station holds it.
    """
    unknown = sorted(set(wanted) - {"base", "schedule", "general"})
    schedule, general = wanted.get("schedule"), wanted.get("general", {})
    if unknown or not isinstance(schedule, dict) or not isinstance(general, dict):
        raise control.ControlError(
            "set_ionozond takes 'base', a 'schedule' object of transmitters and a "
            "'general' object of parameters%s"
            % ("; not %s" % ", ".join(unknown) if unknown else ""))
    missing = sorted(set(current["schedule"]) - set(schedule))
    if missing:
        raise control.ControlError(
            "every transmitter must be listed, and %s is not. The web does not "
            "delete a transmitter: switch it off instead." % ", ".join(missing))
    out = copy.deepcopy(current)
    names = [site["name"] for site in sites]
    for name, fields in schedule.items():
        if name in names:
            raise control.ControlError(
                "%s is the receiver: the site, not a setting, so it is changed on "
                "the station" % name)
        if not isinstance(fields, dict):
            raise control.ControlError("transmitter %s must be an object of fields"
                                       % name)
        extra = sorted(set(fields) - set(FIELDS))
        if extra:
            raise control.ControlError("%s: %s cannot be set from the web; allowed: %s"
                                       % (name, ", ".join(extra), ", ".join(FIELDS)))
        if name not in current["schedule"]:
            if not CODE_RE.match(name):
                raise control.ControlError(
                    "%r is not a transmitter name the archive can carry: one to 24 "
                    "Latin letters, digits, underscores or dots, and no dash" % name)
            lacking = sorted(set(REQUIRED) - set(fields))
            if lacking:
                raise control.ControlError("new transmitter %s needs %s"
                                           % (name, ", ".join(lacking)))
            out["schedule"][name] = {}
        for key, value in fields.items():
            if current["schedule"].get(name, {}).get(key) != value:
                out["schedule"][name][key] = _written_as(key, value)
    for key, value in general.items():
        if key not in GENERAL:
            raise control.ControlError("%s cannot be set from the web; allowed: %s"
                                       % (key, ", ".join(GENERAL)))
        if current["general"].get(key) != value:
            out["general"][key] = _written_as(key, value)
    return out


def edited(schedule: Ini, config: Ini, current: dict, wanted: dict) -> tuple:
    """The console's two files with ``wanted`` written into them, line by line."""
    for name, fields in wanted["schedule"].items():
        for key in sorted(fields):
            if current["schedule"].get(name, {}).get(key) != fields[key]:
                schedule = schedule.with_value(name, key, fields[key])
        if name not in current["schedule"]:
            schedule = schedule.with_value(name, "rx", "false")
    for key, value in wanted["general"].items():
        if current["general"].get(key) != value:
            config = config.with_value("", key, value)
    return schedule, config


def flat(read: dict) -> dict:
    """The sounder's literals as one level of keys, for hashing and for diffs."""
    out = {}
    for key, value in read.items():
        if key == "sounders":
            for thread in value:
                for sounder in thread:
                    for field, item in sounder.items():
                        if field != "name":
                            out["sounders.%s.%s" % (sounder.get("name"), field)] = item
        elif key == "rx_station" and isinstance(value, dict):
            for field, item in value.items():
                out["rx_station.%s" % field] = item
        else:
            out[key] = value
    return out


def diff(before: dict, after: dict) -> dict:
    """Every key whose value differs between two flat dictionaries, from and to."""
    return {key: {"from": before.get(key), "to": after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)}


def edits(current: dict, wanted: dict) -> dict:
    """What an edit changes in the editable document, one line per field."""
    return diff(_flattened(current), _flattened(wanted))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def described(schedule: Ini, read: Ini, text: str) -> dict:
    """The console's two files and ``chirp_config.py``, as the health report shows them.

    ``document`` and ``base`` are what the web edits; ``sounder`` and
    ``sounder_hash`` are what the sounder reads; ``drift`` says whether the
    second is still what the first generates.
    """
    current = document(schedule, read)
    out = {"document": current, "base": fingerprint(current),
           "receivers": receivers(schedule), "settings": settings(read),
           "problems": []}
    try:
        out["sounder"] = flat(literals(text))
        out["sounder_hash"] = fingerprint(out["sounder"])
    except UnreadableConfig as exc:
        out["sounder"], out["sounder_hash"] = None, None
        out["problems"].append(str(exc))
    out["drift"] = _drifted(schedule, read, text)
    if out["drift"]:
        try:
            out["regenerated"] = flat(literals(build(schedule, read)))
        except UnreadableConfig as exc:
            out["problems"].append("regenerating would give an unreadable file: %s"
                                   % exc)
    return out


def vetted(params: dict, reported: dict) -> tuple:
    """A ``set_ionozond`` laid over what a station reported, and its issues.

    The one check, made by the server while the operator is looking and again
    by the agent against the files as they are when the command arrives.
    Refused outright on drift, on a stale ``base``, on a malformed document and
    on a change that changes nothing; the issues are for the caller to weigh.
    """
    if reported.get("drift"):
        raise control.ControlError(
            "chirp_config.py no longer matches schedule.ini and config.ini, so "
            "one was changed without the other. Regenerate it from the ini files, "
            "with Regenerate on /ui/live or `python3 -m services.agent ionozond "
            "regenerate` on the laptop, then make the change again.")
    if params.get("base") != reported.get("base"):
        raise control.ControlError(
            "the schedule changed on the station after this edit was made; reload "
            "it and make the change again")
    sites = reported.get("receivers") or []
    wanted = merged(reported["document"], params, sites)
    if wanted == reported["document"]:
        raise control.ControlError("this change changes nothing")
    return wanted, issues(wanted, sites, reported["settings"],
                          reported.get("timeout_stop_s"))


def summary(block: dict) -> dict:
    """A report's block reduced to what says whether it changed, marked short."""
    return dict({key: block.get(key) for key in SUMMARY}, short=True)


def regenerable(params: dict, reported: dict) -> None:
    """Refuse a ``regenerate_ionozond`` that would not write what the operator saw.

    The one check, made by the server against the last report and again by the
    agent against the files: the ``base`` and ``sounder_hash`` shown must still
    stand, there must be drift to undo, and the console's checks must pass.
    """
    if (params.get("base") != reported.get("base")
            or params.get("sounder_hash") != reported.get("sounder_hash")):
        raise control.ControlError(
            "the settings or chirp_config.py changed on the station after this was "
            "shown; reload and look again")
    if not reported.get("drift"):
        raise control.ControlError(
            "chirp_config.py already holds what schedule.ini and config.ini "
            "generate: there is nothing to regenerate")
    errors = [i for i in issues(reported["document"], reported.get("receivers") or [],
                                reported["settings"], reported.get("timeout_stop_s"))
              if i.level == schedulecheck.ERROR]
    if errors:
        raise control.ControlError("refused: " + "; ".join(_line(i) for i in errors))


def captures(wanted: dict, now: float, count: int) -> list:
    """The next ``count`` captures a document would make, as the sounder picks them."""
    names, stations = [], []
    for name in sorted(wanted["schedule"]):
        fields = wanted["schedule"][name]
        rep, chirpt, dur = (_float(fields.get(k)) for k in ("rep", "chirpt", "dur"))
        if (fields.get("active", "").lower() in ("", "0", "false")
                or None in (rep, chirpt, dur) or rep <= 0):
            continue
        names.append(name)
        stations.append(schedulecheck.SimStation(
            rep, chirpt, dur, schedulecheck.processing_estimate_s(dur)))
    return [{"tx": names[c.station], "start": c.start}
            for c in schedulecheck.upcoming(stations, now, count)]


def _state(config: StationConfig) -> dict:
    files = _files(config)
    schedule, read = Ini.read(files.schedule), Ini.read(files.config)
    out = described(schedule, read, files.sounder.read_text(encoding="utf-8"))
    out["config_path"] = str(files.sounder)
    mismatch = _mismatch(read, files)
    if mismatch:
        out["problems"].append(mismatch)
    since = health._show(config.target, "ActiveEnterTimestampMonotonic")[1]
    out.update(_timing(config, files, since))
    try:
        out["pending"] = _pending(config, "%s:%s" % (_boot(), since), out)
    except OSError as exc:
        out["pending"] = None
        out["problems"].append("could not keep %s: %s" % (config.sounder_seen, exc))
    out["nas_dest"] = (read.value("", "nas_dest") if "nas_dest" in read.keys("")
                       else None)
    try:
        out["site"] = siteconfig.report(config)
    except (control.ControlError, OSError) as exc:
        out["site"] = None
        out["problems"].append("could not read the site settings: %s" % exc)
    out["site_hash"] = (out["site"] or {}).get("hash")
    return out


def _timing(config: StationConfig, files: Files, since: str) -> dict:
    loaded = (time.time() - (time.monotonic() - int(since) / 1e6)
              if since.isdigit() and int(since) > 0 else None)
    return {"written_at": files.sounder.stat().st_mtime, "loaded_at": loaded,
            "active_state": health._show(config.target, "ActiveState")[1] or None,
            "timeout_stop_s": _seconds(
                health._show(config.target, "TimeoutStopUSec")[1])}


def _pending(config: StationConfig, run: str, report: dict) -> bool | None:
    """Whether the running sounder read other than ``chirp_config.py`` now says.

    Not the file's date: the console rewrites the file, unchanged, every time
    one of its dialogs closes with OK, and a rolled-back edit rewrites it too.
    So the agent keeps the file's fingerprint as it first sees it during each
    run of the sounder, and compares with that. A file first seen only after it
    changed during the run leaves nothing to compare with, and the answer is
    None rather than a guess either way.
    """
    if (report["active_state"] != "active" or report["loaded_at"] is None
            or report["sounder_hash"] is None):
        return False
    path = Path(os.path.expanduser(str(config.sounder_seen)))
    seen = _seen(path)
    if seen.get("run") == run:
        return report["sounder_hash"] != seen.get("hash")
    if report["written_at"] > report["loaded_at"]:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    control._atomic_write_bytes(path, json.dumps(
        {"run": run, "hash": report["sounder_hash"]}).encode("utf-8"))
    return False


def _seen(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _boot() -> str:
    """This boot's id: a start time counted from boot repeats across boots."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii").strip()
    except OSError:
        return ""


def _plan(config: StationConfig, files: Files, params: dict) -> Plan:
    schedule, read = Ini.read(files.schedule), Ini.read(files.config)
    raw = files.sounder.read_bytes()
    text = raw.decode("utf-8")
    mismatch = _mismatch(read, files)
    if mismatch:
        raise control.ControlError(mismatch)
    reported = dict(described(schedule, read, text), timeout_stop_s=_seconds(
        health._show(config.target, "TimeoutStopUSec")[1]))
    current = reported["document"]
    wanted, found = vetted(params, reported)
    errors = [i for i in found if i.level == schedulecheck.ERROR]
    if errors:
        raise control.ControlError("refused: " + "; ".join(_line(i) for i in errors))
    schedule_after, read_after = edited(schedule, read, current, wanted)
    generated = build(schedule_after, read_after)
    after = flat(parsed(generated))
    try:
        before = flat(literals(text))
    except UnreadableConfig:
        before = {}
    journal = {
        "station": config.station, "config_path": str(files.sounder),
        "edits": edits(current, wanted),
        "document_hash": fingerprint(wanted), "config_hash": fingerprint(after),
        "warnings": [_line(i) for i in found if i.level == schedulecheck.WARNING],
        "applied_at": time.time(), "requires_restart": True,
    }
    if fingerprint(after) != fingerprint(before):
        journal["changes"] = diff(before, after)
    return Plan(files, (schedule.data, read.data, raw),
                (schedule_after.data, read_after.data, generated.encode("utf-8")),
                journal)


def _carried(config: StationConfig, name: str, planned) -> list:
    """Lock, plan, write and ``try-restart``; roll back if the restart is refused."""
    try:
        files = _files(config)
        with _locked((files.schedule, files.config)):
            plan = planned(files)
            active = health._show(config.target, "ActiveState")[1] or "unknown"
            _written(plan)
            restart = control.systemctl("try-restart", config.target, no_block=True)
            if not restart.ok:
                _restored(plan)
                return [control.CommandResult(
                    name, False, "rolled back: the restart was refused, and the "
                                 "sounder still runs the configuration it started "
                                 "with. %s" % restart.detail), restart]
    except (control.ControlError, IniError, UnreadableConfig, OSError) as exc:
        return [control.CommandResult(name, False, str(exc))]
    return [control.CommandResult(name, True, _said(plan, active),
                                  journal=plan.journal), restart]


def _regenerated(config: StationConfig, files: Files, params: dict) -> Plan:
    schedule, read = Ini.read(files.schedule), Ini.read(files.config)
    raw = files.sounder.read_bytes()
    text = raw.decode("utf-8")
    refusal = _mismatch(read, files) or _folders(config, read)
    if refusal:
        raise control.ControlError(refusal)
    current = dict(described(schedule, read, text), timeout_stop_s=_seconds(
        health._show(config.target, "TimeoutStopUSec")[1]))
    regenerable(params, current)
    generated = build(schedule, read)
    after = flat(parsed(generated))
    before = current["sounder"] or {}
    journal = {
        "station": config.station, "config_path": str(files.sounder),
        "edits": {}, "regenerated": True, "document_hash": current["base"],
        "config_hash": fingerprint(after), "warnings": [],
        "applied_at": time.time(), "requires_restart": True,
    }
    if fingerprint(after) != fingerprint(before):
        journal["changes"] = diff(before, after)
    return Plan(files, (schedule.data, read.data, raw),
                (schedule.data, read.data, generated.encode("utf-8")), journal)


def _folders(config: StationConfig, read: Ini) -> str:
    """Why ``data_dir`` or ``live_dir`` disagrees with ``IONOZOND_ARCHIVE``, or ""."""
    archive = _defaults(Path(config.sounder_defaults)).get("IONOZOND_ARCHIVE", "")
    if not archive:
        return ""
    data_dir = read.value("", "data_dir") if "data_dir" in read.keys("") else ""
    live_dir = str(config.live_dir) if config.live_dir is not None else ""
    if all(_folder(path) == _folder(archive) for path in (data_dir, live_dir)
           if path):
        return ""
    return ("data_dir in config.ini is %s and the agent's live_dir is %s, but the "
            "units hand the sounder IONOZOND_ARCHIVE=%s. The sounder, the live "
            "feed, the prune unit and the NAS unit must all name one folder: "
            "correct config.ini or the agent's config first"
            % (data_dir or "unset", live_dir or "unset", archive))


def _folder(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.expanduser(path)))


def _written(plan: Plan) -> None:
    """Back up, then replace each changed file, ``chirp_config.py`` last."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    changed = [(p, a, b) for p, a, b in zip(plan.paths, plan.before, plan.after)
               if a != b]
    for path, _, _ in changed:
        shutil.copy2(str(path), str(path.with_name("%s.%s.bak" % (path.name, stamp))))
    done = []
    try:
        for path, before, after in changed:
            control._atomic_write_bytes(path, after)
            done.append((path, before))
    except BaseException:
        for path, before in done:
            control._atomic_write_bytes(path, before)
        raise


def _restored(plan: Plan) -> None:
    for path, before, after in zip(plan.paths, plan.before, plan.after):
        if before != after:
            control._atomic_write_bytes(path, before)


def _said(plan: Plan, active: str) -> str:
    edits = "; ".join("%s: %r -> %r" % (key, change["from"], change["to"])
                      for key, change in plan.journal["edits"].items())
    if plan.journal.get("regenerated"):
        edits = "chirp_config.py, regenerated from schedule.ini and config.ini (%s)" % (
            ", ".join(plan.journal.get("changes", {})) or "no value the sounder reads")
    then = ("the sounder restarts on it once the sounding in progress ends"
            if active == "active" else
            "the sounder is %s, so it reads the change at its next start" % active)
    if "changes" not in plan.journal:
        then = "nothing the sounder reads has changed, so it carries on as it is"
    warned = plan.journal["warnings"]
    return "written: %s. %s%s" % (edits, then[0].upper() + then[1:], (
        ". Warnings: " + "; ".join(warned)) if warned else ".")


@contextmanager
def _locked(paths):
    """Hold the lock files ``QSettings`` takes, so the console cannot save meanwhile."""
    held = []
    try:
        for path in paths:
            held.append(_lock(Path(path)))
        yield
    finally:
        for lock in held:
            try:
                os.unlink(str(lock))
            except OSError:
                pass


def _lock(path: Path) -> Path:
    lock = path.with_name(path.name + ".lock")
    for attempt in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid = _pid(lock)
            if attempt == 0 and pid is not None and not _alive(pid):
                os.unlink(str(lock))
                continue
            raise control.ControlError(
                "the console holds %s: it is saving its settings. Try again in a "
                "moment." % lock) from None
        with os.fdopen(fd, "w") as fh:
            fh.write("%d\nionograms-handler-agent\n%s\n"
                     % (os.getpid(), socket.gethostname()))
        return lock
    raise control.ControlError("could not take %s" % lock)


def _pid(lock: Path) -> int | None:
    try:
        first = lock.read_text(encoding="latin-1").split("\n", 1)[0].strip()
    except OSError:
        return None
    return int(first) if first.isdigit() else None


def _alive(pid: int) -> bool:
    """Is a process running, read from ``/proc``; assumed so where there is none."""
    proc = Path("/proc")
    return (proc / str(pid)).exists() if proc.is_dir() else True


def _files(config: StationConfig) -> Files:
    path = Path(config.sounder_defaults)
    sounder = _defaults(path).get("IONOZOND_CONFIG", "")
    if not sounder:
        raise control.ControlError(
            "%s names no IONOZOND_CONFIG, so there is no telling which "
            "chirp_config.py the sounder reads" % path)
    console = Path(os.path.expanduser(str(config.console_settings)))
    return Files(console / "schedule.ini", console / "config.ini", Path(sounder))


def _defaults(path: Path) -> dict:
    try:
        return siteconfig.environment(path)
    except FileNotFoundError:
        raise control.ControlError(
            "%s is missing: it is the unit's EnvironmentFile, and names the "
            "chirp_config.py the sounder reads" % path) from None


def _mismatch(read: Ini, files: Files) -> str:
    if "config_file" not in read.keys(""):
        return ""
    named = read.value("", "config_file")
    if os.path.normpath(named) == os.path.normpath(str(files.sounder)):
        return ""
    return ("the console writes %s but the sounder's unit reads %s, so each would "
            "configure a sounder the other cannot see; point the console's "
            "configuration file at %s first" % (named, files.sounder, files.sounder))


def _drifted(schedule: Ini, read: Ini, text: str) -> bool:
    generated = build(schedule, read)
    try:
        return literals(generated) != literals(text)
    except UnreadableConfig:
        return generated != text


def _receiver(schedule: Ini, group: str) -> bool:
    try:
        return schedule.value(group, "rx").lower() == "true"
    except KeyError:
        return False


def _entry(name: str, fields: dict, rx: bool) -> schedulecheck.Entry:
    return schedulecheck.Entry(
        name=name, active=fields.get("active", "").lower() not in ("", "0", "false"),
        rx=rx, rep=fields.get("rep", ""), chirpt=fields.get("chirpt", ""),
        rate=fields.get("rate", ""), dur=fields.get("dur", ""),
        cf=fields.get("cf", ""), lat=fields.get("lat", ""), lon=fields.get("lon", ""),
        fft=_int(fields.get("fft_count"), 0))


def _written_as(key: str, value) -> str:
    """A value from the web as the console's settings carry it, or why it cannot be."""
    if key in FLAGS:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower()
        raise control.ControlError("%s must be true or false, not %r" % (key, value))
    if isinstance(value, bool):
        raise control.ControlError("%s must be a number, not %r" % (key, value))
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, float) and math.isfinite(value):
        text = str(int(value)) if value.is_integer() else repr(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise control.ControlError("%s must be a number, not %r" % (key, value))
    if not PLAIN.match(text) or "e" in text.lower():
        raise control.ControlError("%s = %r is not a number the console's settings "
                                   "can carry" % (key, value))
    if key == "fft_count" and _int(text, -1) not in [0, *schedulecheck.FFT_COUNT_LIST]:
        raise control.ControlError("fft_count must be 0, for the general FFT, or one "
                                   "of %s" % schedulecheck.FFT_COUNT_LIST)
    if key == "fft_count_index" and not 0 <= _int(text, -1) < 8:
        raise control.ControlError("fft_count_index must be 0 to 7")
    if key == "keep_lfs_hours" and (_float(text) is None or _float(text) < 0):
        raise control.ControlError("keep_lfs_hours must be zero or more")
    if key == "nas_keep_h5_days" and not (
            _int(text, 0) == -1 or 1 <= _int(text, 0) <= 3650):
        raise control.ControlError(
            "nas_keep_h5_days must be -1, to keep every .h5 locally, or 1 to 3650 "
            "days")
    return text


def _flattened(wanted: dict) -> dict:
    out = {"%s.%s" % (name, key): value
           for name, fields in wanted["schedule"].items()
           for key, value in fields.items()}
    out.update(wanted["general"])
    return out


def _line(issue: schedulecheck.Issue) -> str:
    return "%s: %s" % (issue.station, issue.text) if issue.station else issue.text


def _qt(read: Ini, key: str, default, kind):
    """``QSettings::value(key, default).toDouble()``: absent is the default, junk 0."""
    try:
        text = read.value("", key)
    except KeyError:
        return default
    except IniError:
        return kind(0)
    try:
        return kind(float(text)) if kind is int else kind(text)
    except ValueError:
        return kind(0)


def _float(text) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int(text, default: int) -> int:
    try:
        return int(str(text).strip())
    except ValueError:
        return default


def _seconds(text: str) -> float | None:
    """A systemd time span, ``5min`` or ``1min 30s``, in seconds; None for infinity."""
    text = (text or "").strip()
    if not text or text == "infinity":
        return None
    parts = SPAN.findall(text)
    if not parts:
        return None
    return sum(float(number) * SPAN_UNITS[unit or None] for number, unit in parts)
