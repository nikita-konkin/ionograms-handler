"""``python -m services.agent <command>`` -- the station agent.

Every subcommand also works standing alone on the station, which is the point:
the same code that reports to a server is what an operator runs over AnyDesk
when the server is the thing that is broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import control, health, logs, runner
from .config import StationConfig


def _config(args) -> StationConfig:
    return (StationConfig.from_json(args.config) if args.config
            else StationConfig.from_env())


def cmd_health(args) -> int:
    config = _config(args)
    report = health.collect(config, include_epoch=not args.no_epoch)
    if args.json:
        print(report.to_json(indent=2))
        return 0 if report.healthy else 1

    verdict = "HEALTHY" if report.healthy else "UNHEALTHY"
    print(f"{config.station}  {verdict}"
          f"  ({len(report.failing)} failing, {len(report.unknown)} unknown)")
    for metric in report.metrics:
        mark = {True: "ok ", False: "FAIL", None: "  ?"}[metric.ok]
        print(f"  {mark:4} {metric.name:30} {str(metric.value)[:20]:>20}"
              f"  {metric.detail}")
    return 0 if report.healthy else 1


def cmd_start(args) -> int:
    return 0 if control.start(_config(args)).ok else 1


def cmd_stop(args) -> int:
    result = control.stop(_config(args))
    print(result.detail or result.command, file=sys.stderr)
    return 0 if result.ok else 1


def cmd_restart(args) -> int:
    result = control.restart(_config(args))
    print(result.detail or result.command, file=sys.stderr)
    return 0 if result.ok else 1


def cmd_set(args) -> int:
    config = _config(args)
    changes = {}
    for item in args.change:
        if "=" not in item:
            print(f"expected key=value, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        changes[key.strip()] = value.strip()

    try:
        results = (control.apply_and_restart(config, changes) if args.restart
                   else [control.apply_config(config, changes)])
    except control.ControlError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    ok = True
    for result in results:
        print(f"{'ok  ' if result.ok else 'FAIL'} {result.command}: {result.detail}")
        if result.journal:
            print("  journal: " + json.dumps(result.journal, sort_keys=True))
        ok = ok and result.ok
    return 0 if ok else 1


def cmd_logs(args) -> int:
    config = _config(args)
    units = [args.unit] if args.unit else list(config.units)
    for unit in units:
        chunk = logs.read_unit(unit, lines=args.lines, since=args.since,
                               priority=args.priority, grep=args.grep)
        print(f"--- {unit}"
              + (f"  ({chunk.error})" if chunk.error else "")
              + ("  [truncated]" if chunk.truncated else ""))
        for line in chunk.lines:
            print(f"  {line}")
        for meaning, count in chunk.signatures:
            print(f"  >> x{count}  {meaning}")
    return 0


def cmd_triage(args) -> int:
    config = _config(args)
    found = logs.triage(config, lines=args.lines)
    if args.json:
        print(json.dumps(found, indent=2))
        return 0
    if not found["signatures"] and not found["errors"]:
        print("no known failure signatures in the recent journal")
    for entry in found["signatures"]:
        print(f"x{entry['count']:<6} {entry['meaning']}")
    for unit, error in found["errors"].items():
        print(f"  ?    {unit}: {error}")
    return 0


def cmd_run(args) -> int:
    config = _config(args)
    if not config.server_url:
        print("no server_url configured; running one pass to stdout",
              file=sys.stderr)
        return cmd_health(args)
    runner.run_forever(config, max_passes=args.passes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="services.agent",
        description="Station agent: health push, narrow control, logs.")
    parser.add_argument("--config", type=Path, default=None,
                        help="agent config JSON (default: $AGENT_CONFIG, "
                             "then built-in defaults)")
    sub = parser.add_subparsers(dest="command", required=True)

    h = sub.add_parser("health", help="collect and print every metric")
    h.add_argument("--json", action="store_true")
    h.add_argument("--no-epoch", action="store_true",
                   help="skip the reference-transmitter clock check, which "
                        "reads recent par-*.h5 and is the slow one")
    h.set_defaults(func=cmd_health)

    sub.add_parser("start", help="start acquisition").set_defaults(func=cmd_start)
    sub.add_parser("stop", help="stop acquisition (ordered, graceful)"
                   ).set_defaults(func=cmd_stop)
    sub.add_parser("restart", help="restart acquisition").set_defaults(func=cmd_restart)

    s = sub.add_parser("set", help="change acquisition parameters",
                       description="Validated, backed up, written atomically, "
                                   "and journaled. Editable keys: "
                                   + ", ".join(sorted(control.EDITABLE)))
    s.add_argument("change", nargs="+", metavar="KEY=VALUE")
    s.add_argument("--no-restart", dest="restart", action="store_false",
                   help="write the config but do not restart; nothing takes "
                        "effect until something does")
    s.set_defaults(func=cmd_set)

    lg = sub.add_parser("logs", help="recent journal for the acquisition units")
    lg.add_argument("--unit", default=None)
    lg.add_argument("--lines", type=int, default=200)
    lg.add_argument("--since", default=None, help="systemd syntax, e.g. '10 min ago'")
    lg.add_argument("--priority", default=None, help="e.g. warning, err")
    lg.add_argument("--grep", default=None)
    lg.set_defaults(func=cmd_logs)

    t = sub.add_parser("triage", help="which known failures are in the logs")
    t.add_argument("--lines", type=int, default=500)
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_triage)

    r = sub.add_parser("run", help="the push/pull loop")
    r.add_argument("--passes", type=int, default=None,
                   help="stop after N passes (default: forever)")
    r.add_argument("--json", action="store_true")
    r.add_argument("--no-epoch", action="store_true")
    r.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
