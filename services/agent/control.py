"""Start, stop and reconfigure acquisition.

``architecture.md`` sec. 2.5, "Control -- narrow, authenticated, journaled".
Deliberately small: everything here is either a ``systemctl`` verb on one
target or an edit to one ``.ini``. Anything not on that list is a change made
on the station.

Three properties this module exists to guarantee, each of which the station
has already been bitten by:

**Ordered, graceful shutdown.** ``systemctl stop chirp.target`` takes the
units down in dependency order, and the recorder's unit must send ``SIGINT``.
A USRP killed mid-stream keeps transmitting to a host that is gone and wedges:
it stops answering ARP and discovery, and no software on the host can recover
it -- only removing power. That is a site visit, so it is worth a unit file
getting it right.

**No half-applied configuration.** A parameter change is write-then-restart,
and the write is atomic. A truncated ``.ini`` is not a bad setting, it is an
acquisition outage that needs someone to notice and repair the file.

**No combination that records nothing while reporting healthy.** Switching to
scheduled mode without a schedule, or pointing ``output_dir`` at a path that
does not exist, both produce a station whose processes are all "active" and
whose archive stops growing. Those are validated before the write, and
refused whole.
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import StationConfig


class ControlError(RuntimeError):
    """A command that could not be carried out, with a reason for the operator."""


@dataclass
class CommandResult:
    """What happened, in a form that can be acknowledged back to the server."""

    command: str
    ok: bool
    detail: str = ""
    timestamp: float = field(default_factory=time.time)
    #: Populated for parameter changes: what to write into ``config_epoch``.
    journal: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {"command": self.command, "ok": self.ok, "detail": self.detail,
                "timestamp": self.timestamp, "journal": self.journal}


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------

#: Verbs the agent will pass to systemctl. An allow-list, not a filter: the
#: command name arrives over the network, and `systemctl` has verbs like
#: `mask` and `isolate` that have no business being reachable remotely.
ALLOWED_VERBS = ("start", "stop", "restart", "status", "is-active")


#: Set ``AGENT_FAKE_SYSTEMCTL=1`` to log verbs instead of running them.
#:
#: For the Docker test rig, where the agent runs in a container that has no
#: systemd -- ``systemctl`` is simply absent, so every control command would
#: report "not found" and the command path could never be demonstrated end to
#: end. Named for what it is, off unless explicitly asked for, and it prints
#: every call so a fake success can never be mistaken for a real one.
FAKE_SYSTEMCTL_ENV = "AGENT_FAKE_SYSTEMCTL"


def _fake_runner(args, **kwargs):
    import sys

    print(f"agent: FAKE systemctl {' '.join(args[1:])} "
          f"(AGENT_FAKE_SYSTEMCTL is set; nothing was actually done)",
          file=sys.stderr)
    return subprocess.CompletedProcess(args, 0, "", "")


def systemctl(verb: str, target: str, *, timeout: float = 120.0,
              runner=None) -> CommandResult:
    """One systemctl verb against one unit or target.

    ``runner`` exists so a test can drive this without systemd; it defaults to
    :func:`subprocess.run` and is never something a caller supplies in
    production.
    """
    if verb not in ALLOWED_VERBS:
        raise ControlError(
            f"verb {verb!r} is not allowed; choose from {', '.join(ALLOWED_VERBS)}")

    if runner is None and os.environ.get(FAKE_SYSTEMCTL_ENV, "").strip() not in ("", "0"):
        runner = _fake_runner
    runner = runner or subprocess.run
    args = ["systemctl", verb, target]
    try:
        proc = runner(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return CommandResult(f"{verb} {target}", False, "systemctl: not found")
    except subprocess.TimeoutExpired:
        # A stop that times out is the dangerous case: systemd will escalate to
        # SIGKILL and the radio can wedge. Say so rather than reporting a bare
        # timeout.
        extra = (" -- the recorder may have been killed rather than stopped; "
                 "check that the USRP still answers"
                 if verb in ("stop", "restart") else "")
        return CommandResult(f"{verb} {target}", False,
                             f"timed out after {timeout}s{extra}")
    except Exception as exc:                                  # pragma: no cover
        return CommandResult(f"{verb} {target}", False,
                             f"{type(exc).__name__}: {exc}")

    text = (proc.stdout or "") + (proc.stderr or "")
    return CommandResult(f"{verb} {target}", proc.returncode == 0, text.strip())


def start(config: StationConfig, **kw) -> CommandResult:
    return systemctl("start", config.target, **kw)


def stop(config: StationConfig, **kw) -> CommandResult:
    return systemctl("stop", config.target, **kw)


def restart(config: StationConfig, **kw) -> CommandResult:
    return systemctl("restart", config.target, **kw)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Settings the agent will change, mapped to their ``.ini`` location. An
#: allow-list for the same reason as `ALLOWED_VERBS`: `chirp_config.py` reads
#: dozens of keys, most of which would need a considered decision and a site
#: visit if they went wrong.
EDITABLE = {
    "mode": ("lfm", "serendipitous"),
    "sounder_timings": ("lfm", "sounder_timings"),
    "output_dir": ("config", "output_dir"),
    "max_range_extent": ("lfm", "max_range_extent"),
    "save_raw_voltage": ("lfm", "save_raw_voltage"),
}

#: `mode` is friendlier than a bare boolean, and the mapping is the one place
#: the two vocabularies meet. See architecture.md sec. 2.5.
MODES = {"search": "true", "serendipitous": "true",
         "scheduled": "false", "schedule": "false"}


def read_config(path: str | Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    read = parser.read(Path(path))
    if not read:
        raise ControlError(f"{path}: could not be read")
    return parser


def _validate(parser: configparser.ConfigParser, changes: dict) -> None:
    """Refuse combinations that record nothing while looking healthy."""
    def value_of(key, section, option):
        if key in changes:
            return changes[key]
        return parser.get(section, option, fallback=None)

    mode = value_of("mode", "lfm", "serendipitous")
    if mode is not None and str(mode).lower() in ("false", "scheduled", "schedule"):
        timings = value_of("sounder_timings", "lfm", "sounder_timings")
        try:
            parsed = json.loads(timings) if timings else []
        except json.JSONDecodeError as exc:
            raise ControlError(f"sounder_timings is not valid JSON: {exc}") from None
        if not parsed:
            raise ControlError(
                "scheduled mode with an empty sounder_timings would record "
                "nothing while every process reported healthy. Supply the "
                "schedule in the same command, or stay in search mode.")
        for entry in parsed:
            missing = {"chirp-rate", "rep", "chirpt"} - set(entry)
            if missing:
                raise ControlError(
                    f"sounder_timings entry {entry} is missing {sorted(missing)}")

    if "output_dir" in changes:
        target = Path(str(changes["output_dir"]).strip('"'))
        if not target.parent.exists():
            raise ControlError(
                f"{target}: parent directory does not exist. Acquisition would "
                f"start, report healthy, and write nowhere.")

    if "save_raw_voltage" in changes:
        if str(changes["save_raw_voltage"]).lower() not in ("true", "false"):
            raise ControlError("save_raw_voltage must be true or false")


def _atomic_write(path: Path, parser: configparser.ConfigParser) -> None:
    """Write via a temporary file in the same directory, then replace.

    Same directory so the replace is on one filesystem and therefore atomic.
    A half-written config is an outage, not a misconfiguration.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name,
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            parser.write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply_config(config: StationConfig, changes: dict, *,
                 backup: bool = True) -> CommandResult:
    """Validate, back up, write atomically, and describe what to journal.

    Does **not** restart: :func:`apply_and_restart` does, so that a caller who
    wants to batch several changes into one restart can. v2 reads its config
    only at process start, so nothing takes effect until something restarts.
    """
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise ControlError(
            f"not editable remotely: {sorted(unknown)}; "
            f"allowed: {sorted(EDITABLE)}")
    if not changes:
        raise ControlError("no changes given")

    path = Path(config.chirp_config)
    parser = read_config(path)

    normalized = dict(changes)
    if "mode" in normalized:
        raw = str(normalized["mode"]).lower()
        if raw not in MODES:
            raise ControlError(
                f"mode {normalized['mode']!r} unknown; "
                f"choose from {sorted(set(MODES))}")
        normalized["mode"] = MODES[raw]

    _validate(parser, normalized)

    before = {}
    for key, value in normalized.items():
        section, option = EDITABLE[key]
        if not parser.has_section(section):
            parser.add_section(section)
        before[key] = parser.get(section, option, fallback=None)
        parser.set(section, option, str(value))

    if backup:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        shutil.copy2(path, path.with_suffix(path.suffix + f".{stamp}.bak"))

    _atomic_write(path, parser)

    return CommandResult(
        command="apply_config", ok=True,
        detail="; ".join(f"{k}: {before[k]!r} -> {v!r}"
                         for k, v in normalized.items()),
        journal={
            "station": config.station,
            "config_path": str(path),
            "changes": {k: {"from": before[k], "to": str(v)}
                        for k, v in normalized.items()},
            "applied_at": time.time(),
            "requires_restart": True,
        },
    )


def apply_and_restart(config: StationConfig, changes: dict, **kw) -> list[CommandResult]:
    """The whole sequence, so a half-applied change cannot exist.

    Stop, write, start -- rather than restart-after-write -- because the
    config is read at process start and stopping first makes the window in
    which the old processes could write products under the new configuration
    exactly zero.
    """
    results = [stop(config, **kw)]
    if not results[0].ok:
        results.append(CommandResult(
            "apply_config", False,
            "not attempted: the stop failed, and editing the config of a "
            "running acquisition would take effect at an unpredictable moment"))
        return results

    try:
        results.append(apply_config(config, changes))
    except ControlError as exc:
        results.append(CommandResult("apply_config", False, str(exc)))
        # Bring it back up on the old config rather than leaving it down.
        results.append(start(config, **kw))
        return results

    results.append(start(config, **kw))
    return results
