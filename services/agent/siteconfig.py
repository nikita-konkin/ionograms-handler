"""The ionozond site settings: ``/etc/default/ionozond``, and the few the web sets.

Phase 4 of the live console (issue #8,
``docs/2026-10-17-ionozond-retiring-the-console.md``, sections 4 and 5). Every
key of the root file is shown. Three are storage policy, and a web edit writes
them into ``web.env``, a file the agent owns, which drop-ins load after the
root file in the prune and NAS units. systemd lets a later environment file
win, so a key there overrides the root file's, and deleting ``web.env`` undoes
every web edit. The root file itself is never written.

An environment file can set ``PATH`` or ``LD_PRELOAD``, and ``web.env`` is
writable by the agent's user. So only units that run as that user, with no
root-run step, load it, and the agent writes only the allow-listed keys.

Run now is here too: it starts one of those two units once, and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from . import control
from .config import StationConfig

NAME = "set_ionozond_site"
RUN = "start"

#: The keys a web edit may write, with the range each takes. ``None`` stands
#: for the archive volume's size, which ``prune_lfs.py`` counts in units of
#: 10**9 bytes.
EDITABLE = {"NAS_BWLIMIT": (0, 999999999), "PRUNE_FREE_GB": (0, None),
            "PRUNE_KEEP_DAYS": (1, 365)}
#: The oneshots Run now may start: the only units that load ``web.env``.
JOBS = ("ionozond-nas.service", "ionozond-prune.service")

NUMBER = re.compile(r"[0-9]{1,9}")


def report(config: StationConfig) -> dict:
    """Every key the prune and NAS units are handed, and the file it came from."""
    root, web = Path(config.sounder_defaults), _overrides(config)
    keys = {key: {"value": value, "from": str(root)}
            for key, value in environment(root).items()}
    keys.update({key: {"value": value, "from": str(web)}
                 for key, value in _optional(web).items()})
    return {"keys": keys, "editable": sorted(EDITABLE),
            "hash": hashlib.sha256(json.dumps(keys, sort_keys=True).encode(
                "utf-8")).hexdigest(),
            "archive_gb": _size(keys.get("IONOZOND_ARCHIVE", {}).get("value", ""))}


def apply(config: StationConfig, params: dict) -> list:
    """Check and write one ``set_ionozond_site``; every outcome acknowledged."""
    path = _overrides(config)
    try:
        problems = vetted(params, report(config))
        if problems:
            return [control.CommandResult(NAME, False, "refused: " + "; ".join(
                problems))]
        before = _text(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        control._atomic_write_bytes(path, _edited(before, params["site"]).encode(
            "utf-8"))
    except (control.ControlError, OSError) as exc:
        return [control.CommandResult(NAME, False, "%s was not written: %s"
                                      % (path, exc))]
    said = ", ".join("%s=%s" % (key, _value(value)) if _value(value) else
                     "%s dropped" % key
                     for key, value in sorted(params["site"].items()))
    return [control.CommandResult(NAME, True, "%s: %s; the prune and NAS units read "
                                              "it at their next run" % (path, said))]


def vetted(params: dict, reported: dict) -> list:
    """What stands in the way of a ``set_ionozond_site``, against a site report."""
    site = params.get("site")
    if not isinstance(site, dict) or not site:
        return ["a site change names no keys"]
    if params.get("base") != reported.get("hash"):
        return ["the site settings changed on the station after this was shown; "
                "reload and look again"]
    return [problem for problem in (_checked(key, value, reported)
                                    for key, value in sorted(site.items())) if problem]


def run(unit) -> control.CommandResult:
    """Start one prune or NAS run now, acknowledged once systemd has queued it.

    Always ``--no-block``, whatever the station's ``no_block`` says. A blocking
    start of a oneshot waits for the whole run, which the NAS unit allows an
    hour, and the agent gives up after 120 s: the run would go on while its
    acknowledgement said it had failed.
    """
    if unit not in JOBS:
        return control.CommandResult(RUN, False, "Run now starts %s only, not %r"
                                     % (" or ".join(JOBS), unit))
    return control.systemctl(RUN, unit, no_block=True)


def environment(path: Path) -> dict:
    """``KEY=value`` lines, as systemd's ``EnvironmentFile=`` reads the simple ones."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key = _key(line)
        if key is None:
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def _checked(key: str, value, reported: dict) -> str:
    if key not in EDITABLE:
        return "%s cannot be set from the web, only %s" % (key, ", ".join(
            sorted(EDITABLE)))
    text = _value(value)
    if not text:
        return ""
    if not NUMBER.fullmatch(text):
        return "%s must be a whole number, not %r" % (key, value)
    low, high = EDITABLE[key]
    if high is None and reported.get("archive_gb") is None:
        return "%s cannot be checked: the archive volume's size is unknown" % key
    high = int(reported["archive_gb"]) if high is None else high
    if not low <= int(text) <= high:
        return "%s must be %d to %d, not %s" % (key, low, high, text)
    return ""


def _value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value if isinstance(value, str) else repr(value)


def _edited(text: str, site: dict) -> str:
    wanted = {key: _value(value) for key, value in site.items()}
    lines, seen = [], set()
    for line in text.splitlines():
        key = _key(line)
        if key not in wanted:
            lines.append(line)
        elif key not in seen and wanted[key]:
            lines.append("%s=%s" % (key, wanted[key]))
        seen.add(key)
    lines.extend("%s=%s" % (key, wanted[key]) for key in sorted(wanted)
                 if key not in seen and wanted[key])
    return "".join(line + "\n" for line in lines)


def _key(line: str):
    line = line.strip()
    if not line or line.startswith(("#", ";")) or "=" not in line:
        return None
    return line.split("=", 1)[0].strip()


def _overrides(config: StationConfig) -> Path:
    return Path(os.path.expanduser(str(config.site_overrides)))


def _optional(path: Path) -> dict:
    try:
        return environment(path)
    except FileNotFoundError:
        return {}


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _size(path: str):
    if not path:
        return None
    try:
        return shutil.disk_usage(os.path.expanduser(path)).total / 1e9
    except OSError:
        return None
