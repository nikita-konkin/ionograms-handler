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
from typing import Any, Callable

from . import client, control, health, ionozond, logs, preview, siteconfig
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
        if "unit" in params:
            return [siteconfig.run(params["unit"])]
        return [control.start(config, no_block=config.no_block)]
    if name in ("stop", "stop_sounding"):
        return [control.stop(config, no_block=config.no_block)]
    if name in ("restart", "restart_sounding"):
        return [control.restart(config, no_block=config.no_block)]

    if name == "set_ionozond":
        return ionozond.apply(config, params)
    if name == "regenerate_ionozond":
        return ionozond.regenerate(config, params)
    if name == "set_ionozond_site":
        return siteconfig.apply(config, params)

    if name in ("set_config", "configure", "set_params"):
        if health.runs_ionozond(config):
            return [control.CommandResult(
                name, False, "this station runs ionozond, whose schedule and "
                             "parameters change with set_ionozond, not set_config")]
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
                     f"understands start, stop, restart, set_config, set_ionozond, "
                     f"regenerate_ionozond, set_ionozond_site, "
                     f"logs, triage")]


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


def obey(config: StationConfig, result: PassResult, *,
         opener: Callable | None = None) -> None:
    """Pull pending commands, carry each out, and acknowledge it. Never raises.

    One pass of the control half of sec. 5.4, shared by the health loop and the
    live feed: the feed runs it as soon as its reply says something is queued,
    so a start or stop pressed in the console lands in seconds rather than on
    this loop's next minute.
    """
    try:
        commands = client.pull_commands(config, opener=opener)
    except client.TransportError as exc:
        result.errors.append(f"pull: {exc}")
        return
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


@dataclass
class SounderState:
    """The ionozond block the server last took whole, carried between passes.

    In memory, as :class:`PreviewState` is: a restarted agent sends the block
    whole once more, and the server takes it.
    """

    summary: dict = field(default_factory=dict)
    at: float = 0.0


def run_once(config: StationConfig, *, opener: Callable | None = None,
             include_epoch: bool = True,
             previews: PreviewState | None = None,
             sounder: SounderState | None = None) -> PassResult:
    """Collect health, push it, then pull and execute any pending commands.

    ``previews`` carries what the server already has; without one every pass
    re-sends, which is what a single ``--once`` run should do. ``sounder``
    does the same for the ionozond block: without one it always goes whole.
    """
    result = PassResult()

    scan = None
    if config.preview:
        # One walk of the archive for both the age metric and the previews. On
        # DOB `output_dir` sits behind a FUSE daemon competing with the
        # recorder for CPU, so walking it twice to answer two questions about
        # the same set of filenames would be paying that cost for nothing.
        try:
            root = health.product_root(config)
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

    whole = report.sounder_config
    if whole is not None and sounder is not None and (
            ionozond.summary(whole) == sounder.summary
            and time.time() - sounder.at < ionozond.WHOLE_S):
        report.sounder_config = ionozond.summary(whole)

    if config.server_url:
        try:
            client.push_health(config, report, opener=opener)
            result.pushed = True
        except client.TransportError as exc:
            result.errors.append(f"push: {exc}")
        # `whole` is None on a station that does not run ionozond, and then
        # there is no block to remember.
        if (result.pushed and sounder is not None and whole is not None
                and report.sounder_config is whole):
            sounder.summary, sounder.at = ionozond.summary(whole), time.time()

        if scan is not None:
            _push_previews(config, scan, previews or PreviewState(),
                           result, opener)

        obey(config, result, opener=opener)

    return result


def run_forever(config: StationConfig, *, opener: Callable | None = None,
                sleep: Callable[[float], None] = time.sleep,
                max_passes: int | None = None) -> None:
    """``run_once`` on an interval. ``max_passes`` exists so a test can stop."""
    passes = 0
    previews, sounder = PreviewState(), SounderState()
    while max_passes is None or passes < max_passes:
        started = time.time()
        result = run_once(config, opener=opener, previews=previews,
                          sounder=sounder)
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
