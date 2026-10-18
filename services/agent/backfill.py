"""The live view's history, from soundings the station archived before anyone watched.

The server learns a sounding's summary from its live columns, so a strip on
``/ui/live`` began only when the live feed did, and every server restart
used to empty it. The station keeps its products, though: an
``lfm_ionogram-*.h5`` for every sounding, locally for ``nas_keep_h5_days`` and
on the NAS after that. This worker reads them, summarises each one as the
server summarises a live sounding (:mod:`services.agent.summary`), and hands
the summaries to the live feed to post (issue #12).

**What to send** is decided a day at a time. The server says how many
summaries it holds for each transmitter and UTC day; a day where it holds
fewer than the station has products is sent whole, newest day first. The
server keeps a live summary over an archived one, so resending a day costs
bandwidth and nothing else. A product that cannot be read is remembered and
not offered again until the process restarts.

**It never delays the live view.** The reading runs on its own thread, and
what it has ready waits in a bounded queue that the feed drains a few at a
time, in a post of its own after each live batch.
"""

from __future__ import annotations

import collections
import os
import queue
import sys
import time
from pathlib import Path
from typing import Callable

from muf import io_chirp

from . import client, product, summary
from .config import StationConfig

#: Summaries ready and waiting for the feed. The worker blocks when it is
#: full, so it reads no faster than the feed posts.
READY = 200


def summarise(path: Path) -> dict:
    """One archived product as a ``summary`` event, in the live columns' units."""
    head, cells = product.read(path)
    return dict(head, type="summary", snr=summary.snr(cells), pdp=summary.pdp(cells))


def products(roots: list) -> dict:
    """Every product under the roots, by transmitter and start; the first root wins."""
    found: dict = {}
    for root in roots:
        for folder, _, names in sorted(os.walk(str(root))):
            for name in sorted(names):
                match = io_chirp._NAME_RE.match(name)
                if match:
                    found.setdefault((match["tx"], round(float(match["t0"]))),
                                     Path(folder) / name)
    return found


def wanted(local: dict, days: dict, failed=frozenset()) -> list:
    """The products of every day the server holds fewer summaries of, newest first."""
    grouped = collections.defaultdict(list)
    for (tx, t0), path in local.items():
        if path not in failed:
            grouped[(tx, time.strftime("%Y-%m-%d", time.gmtime(t0)))].append((t0, path))
    out = []
    for (tx, day), items in grouped.items():
        if days.get(tx, {}).get(day, 0) < len(items):
            out.extend(items)
    return [path for _, path in sorted(out, key=lambda item: item[0], reverse=True)]


class Backfill:
    """Reads the station's products into summaries, one survey at a time."""

    def __init__(self, config: StationConfig, fetch: Callable[[], dict]):
        self.config = config
        self.fetch = fetch
        self.ready: queue.Queue = queue.Queue(maxsize=READY)
        self.failed: set = set()

    def fill(self) -> None:
        """One survey: ask what the server holds, then summarise what it lacks."""
        roots = [r for r in (self.config.live_dir, *self.config.backfill_dirs) if r]
        for path in wanted(products(roots), self.fetch(), self.failed):
            try:
                self.ready.put(summarise(path))
            except (OSError, ValueError, KeyError) as exc:
                self.failed.add(path)
                print("backfill: cannot summarise %s: %s" % (path, exc),
                      file=sys.stderr, flush=True)

    def take(self, most: int) -> list:
        out = []
        while len(out) < most:
            try:
                out.append(self.ready.get_nowait())
            except queue.Empty:
                break
        return out

    def run(self, sleep: Callable[[float], None] = time.sleep) -> None:
        while True:
            try:
                self.fill()
            except client.TransportError as exc:
                print("backfill: %s" % exc, file=sys.stderr, flush=True)
            sleep(self.config.backfill_every_s)
