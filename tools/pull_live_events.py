"""Pull the hub's live forecasts into JSON Lines files, one per circuit.

The hub keeps every live event -- each forecast issued, each one scored
against its truth with both values side by side, each one that lapsed -- for
LIVE_KEEP_DAYS (30), then deletes it (`services/api/sim_live.py`). This copies
a circuit's events to OUT/TX__RX.jsonl and carries on after the last id the
file holds, so a pull run more often than that keeps everything, and a pull
run twice writes nothing twice.

    python tools/pull_live_events.py --hub http://10.0.0.130:8002 \\
        --out out/article/hub NIC2:yoshkar-ola NIC1:yoshkar-ola

A READ_TOKEN in the environment is sent as a bearer token, for a hub that
sets one. Each line is the event `muf sim live --out` writes, with the hub's
`id`, `tx` and `rx` added.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

#: Events asked for at once; the route answers with up to 10 000.
PAGE = 1000

#: Seconds one page may take to arrive.
TIMEOUT_S = 60.0


def get(url: str) -> list[str]:
    """The lines a GET of ``url`` answers with."""
    request = urllib.request.Request(url)
    if os.environ.get("READ_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['READ_TOKEN']}")
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return [line for line in response.read().decode("utf-8").splitlines() if line]


def target(out: Path, tx: str, rx: str) -> Path:
    """The file a circuit's events are kept in."""
    return Path(out) / f"{tx}__{rx}.jsonl"


def last(path: Path) -> int:
    """The id of the last event ``path`` holds; 0 for none."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    held = [line for line in lines if line.strip()]
    return int(json.loads(held[-1])["id"]) if held else 0


def pull(hub: str, tx: str, rx: str, path: Path, *,
         get: Callable[[str], list[str]] = get, page: int = PAGE) -> int:
    """Append the circuit's events after the last ``path`` holds; how many."""
    after, added = last(path), 0
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        lines = get(f"{hub.rstrip('/')}/sim/live/{urllib.parse.quote(tx)}/"
                    f"{urllib.parse.quote(rx)}/events?after={after}&limit={page}")
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.writelines(line + "\n" for line in lines)
        added += len(lines)
        if len(lines) < page:
            return added
        after = int(json.loads(lines[-1])["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("circuits", nargs="+", metavar="TX:RX")
    parser.add_argument("--hub", required=True, help="the hub's address")
    parser.add_argument("--out", type=Path, required=True,
                        help="folder the circuits' files are kept in")
    args = parser.parse_args(argv)
    for circuit in args.circuits:
        tx, rx = circuit.split(":", 1)
        path = target(args.out, tx, rx)
        print(f"{circuit}: {pull(args.hub, tx, rx, path)} new events in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
