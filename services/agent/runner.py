"""The loop: collect, push, pull, execute, acknowledge.

One pass is :func:`run_once` and it is the whole agent. :func:`run_forever`
adds nothing but sleep, which keeps the interesting part testable without a
clock.

**Nothing in a pass may kill the loop.** A station that stops reporting
because the server was briefly unreachable has turned a network blip into an
outage that looks identical to a dead station -- and under sec. 5.4 silence is
the alert, so it would page someone. Every step catches its own failures and
the pass continues.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import client, control, health, logs, preview
from .config import StationConfig


@dataclass
class PassResult:
    """What one pass did, for logging and for tests."""

    pushed: bool = False
    healthy: bool | None = None
    commands_run: int = 0
    previews_sent: int = 0
    errors: list[str] = field(default_factory=list)
    report: Any = None


@dataclass
class PreviewState:
    """What the server already has, carried between passes.

    In memory and nowhere else. An agent restart re-sends every thumbnail once,
    which costs four passes' worth of work and gets the console a fresh picture
    -- both better outcomes than a state file on the acquisition laptop that
    could disagree with the server.
    """

    sent: dict = field(default_factory=dict)      # tx -> t0 the server holds
    cursor: int = 0


def _dispatch(config: StationConfig, command: client.Command) -> list:
    """Map one pulled command onto the control surface.

    An unknown name is answered, not ignored: the server needs to learn that
    this agent is too old for the command rather than waiting for an
    acknowledgement that never comes.
    """
    name = command.name.replace("-", "_").lower()
    params = command.params or {}

    if name in ("start", "start_sounding"):
        return [control.start(config)]
    if name in ("stop", "stop_sounding"):
        return [control.stop(config)]
    if name in ("restart", "restart_sounding"):
        return [control.restart(config)]

    if name in ("set_config", "configure", "set_params"):
        changes = params.get("changes", params)
        restart = bool(params.get("restart", True))
        try:
            if restart:
                return control.apply_and_restart(config, changes)
            return [control.apply_config(config, changes)]
        except control.ControlError as exc:
            return [control.CommandResult(name, False, str(exc))]

    if name in ("logs", "get_logs"):
        chunks = logs.read_all(
            config,
            lines=int(params.get("lines", 200)),
            since=params.get("since"),
            priority=params.get("priority"),
            grep=params.get("grep"),
        )
        return [control.CommandResult(
            name, True, json.dumps([c.to_dict() for c in chunks]))]

    if name in ("triage", "diagnose"):
        return [control.CommandResult(
            name, True, json.dumps(logs.triage(config)))]

    return [control.CommandResult(
        name, False, f"unknown command {command.name!r}; this agent "
                     f"understands start, stop, restart, set_config, logs, triage")]


def _push_previews(config: StationConfig, scan, state: PreviewState,
                   result: PassResult, opener: Callable | None) -> None:
    """Encode and send the thumbnails that changed. Never raises.

    Sent *after* the health push, deliberately: the health document is the
    thing sec. 5.4 makes silence mean something about, and a picture must never
    be able to delay or displace it.
    """
    picked, state.cursor = preview.due(scan[0], state.sent,
                                       cursor=state.cursor)
    for tx, path, t0 in picked:
        try:
            shot = preview.build(path, config.preview_size)
        except preview.PreviewUnavailable as exc:
            # numpy or h5py is missing. Nothing is wrong with the station, and
            # saying so once per pass per circuit would drown the log.
            result.errors.append(f"preview: unavailable ({exc})")
            return
        except Exception as exc:
            # A product opened mid-write is the ordinary case here, not an
            # incident: it will be complete by the next pass, and `sent` is
            # left alone so the next pass tries again.
            result.errors.append(f"preview {tx}: {type(exc).__name__}: {exc}")
            continue
        # The product's own `txname` is authoritative for *labelling* it -- a
        # station renamed in config makes the two disagree, and the file is the
        # record of what was actually written. The filename is what the scan
        # grouped by, so it stays the key here: dedup keyed on a name that only
        # appears after the decode would re-send every pass whenever they
        # differ, which is exactly the case the fallback exists for.
        shot.tx = shot.tx or tx
        try:
            client.push_preview(config, shot, opener=opener)
        except client.TransportError as exc:
            result.errors.append(f"preview {tx}: {exc}")
            continue
        state.sent[tx] = t0
        result.previews_sent += 1


def run_once(config: StationConfig, *, opener: Callable | None = None,
             include_epoch: bool = True,
             previews: PreviewState | None = None) -> PassResult:
    """Collect health, push it, then pull and execute any pending commands.

    ``previews`` carries what the server already has; without one every pass
    re-sends, which is what a single ``--once`` run should do.
    """
    result = PassResult()

    scan = None
    if config.preview:
        # One walk of the archive for both the age metric and the previews. On
        # DOB `output_dir` sits behind a FUSE daemon competing with the
        # recorder for CPU, so walking it twice to answer two questions about
        # the same set of filenames would be paying that cost for nothing.
        try:
            root = Path(config.output_dir)
            scan = health.scan_products(root) if root.is_dir() else ({}, None)
        except OSError as exc:
            result.errors.append(f"scan: {type(exc).__name__}: {exc}")

    try:
        report = health.collect(config, include_epoch=include_epoch, scan=scan)
        result.report = report
        result.healthy = report.healthy
    except Exception as exc:                                  # pragma: no cover
        result.errors.append(f"collect: {type(exc).__name__}: {exc}")
        return result

    if config.server_url:
        try:
            client.push_health(config, report, opener=opener)
            result.pushed = True
        except client.TransportError as exc:
            result.errors.append(f"push: {exc}")

        if scan is not None:
            _push_previews(config, scan, previews or PreviewState(),
                           result, opener)

        try:
            commands = client.pull_commands(config, opener=opener)
        except client.TransportError as exc:
            result.errors.append(f"pull: {exc}")
            commands = []

        for command in commands:
            try:
                results = _dispatch(config, command)
            except Exception as exc:
                results = [control.CommandResult(
                    command.name, False, f"{type(exc).__name__}: {exc}")]
            result.commands_run += 1
            try:
                client.acknowledge(config, command, results, opener=opener)
            except client.TransportError as exc:
                result.errors.append(f"ack {command.id}: {exc}")

    return result


def run_forever(config: StationConfig, *, opener: Callable | None = None,
                sleep: Callable[[float], None] = time.sleep,
                max_passes: int | None = None) -> None:
    """``run_once`` on an interval. ``max_passes`` exists so a test can stop."""
    passes = 0
    previews = PreviewState()
    while max_passes is None or passes < max_passes:
        started = time.time()
        result = run_once(config, opener=opener, previews=previews)
        for problem in result.errors:
            print(f"agent: {problem}", file=sys.stderr)
        if result.report is not None and not result.healthy:
            for metric in result.report.failing:
                print(f"agent: FAIL {metric.name}={metric.value} {metric.detail}",
                      file=sys.stderr)
        passes += 1
        if max_passes is not None and passes >= max_passes:
            break
        elapsed = time.time() - started
        sleep(max(0.0, config.push_interval_s - elapsed))
