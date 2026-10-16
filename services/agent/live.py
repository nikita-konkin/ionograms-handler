"""The live view, from the station's side: a sounding as it is captured.

An ionozond sounder writes each capture to ``<live_dir>/<YYYY.MM.DD>/<tx>_<date>_<time>.lfs``
while it records, and prints ``STATUS`` lines to the journal as it goes. This
process follows both and posts what it sees to the server every couple of
seconds (``architecture.md`` sec. 5.5): the station still never listens, and
the sounder is not changed to make it possible.

A column is one FFT window of the capture formed exactly as ``muf.spectro``
forms it, gated to the header's range window, and sent as one byte per range
row at a quarter of a dB per count above the noise floor. At the default 16384
samples that is one column every 0.41 s and about half a kilobyte a second.

Runs as its own process, under its own unit and at low priority, rather than
inside the health loop: a capture that fails to parse must not be able to stop
the health push, whose silence is the alert.

It is also the fast half of the command path. Every reply says how many
commands wait for this station, and when any do the feed pulls, carries out
and acknowledges them at once, as the health loop does once a minute. So a
start or stop pressed in the console lands within one interval, and the
station still only ever asks.
"""

from __future__ import annotations

import base64
import collections
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from muf import calibrate, io_lfs, spectro

from . import client, runner
from .config import StationConfig
from .logs import status_event

#: dB per count in a column's bytes, and so the largest value one can carry is
#: 63.75 dB above the noise.
DB_STEP = 0.25

#: Events held while the server cannot be reached. A sounding is 610 columns,
#: so this keeps the rest of a sounding through an outage of a few minutes and
#: nothing older.
BACKLOG = 4096

#: How long after its end a capture found at start-up is still worth sending.
#: An older one is history, and history is the archive's job.
FRESH_S = 60.0

def newest(root: Path) -> Optional[Path]:
    """The capture written last, from the two newest day folders only.

    Two, not one, so the first capture after midnight is found while the last
    one before it is still the newest file; and not all of them, because the
    archive behind ``root`` may hold years.
    """
    days = sorted(p for p in Path(root).glob("????.??.??") if p.is_dir())[-2:]
    found = [p for day in days for p in day.glob("*.lfs")]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


class Capture:
    """One capture file as it grows: its axes once, then each new window as a column."""

    def __init__(self, path: Path, window: int = 16384):
        header = io_lfs.read_header(path)
        total = int(header.dur * header.sample_rate / header.dec) // window
        self.path = Path(path)
        self.window = window
        self.cal = calibrate.build(header, n_freq=max(total, 1), window=window)
        self.taper = spectro.hanning(window)
        self.done = 0
        self.ends = header.start_epoch + header.dur
        self.head = {
            "type": "sounding",
            "tx": self.path.name.rsplit("_", 2)[0],
            "t0": int(header.start_epoch),
            "f_lo_mhz": float(self.cal.freq_start),
            "f_step_mhz": float(calibrate.sweep_step_mhz(header, window)),
            "columns": total,
            "range_lo_km": float(self.cal.vrange[-1]),
            "range_step_km": float(self.cal.range_step),
            "rows": int(self.cal.n_range),
            "km_per_ms": calibrate.C_KM_S / 1e3 * 2.0 / header.div_coef,
        }

    @property
    def finished(self) -> bool:
        return self.done >= self.head["columns"]

    def columns(self) -> list:
        if self.finished:
            return []
        ready = min(self.head["columns"], io_lfs.n_samples(self.path) // self.window)
        if ready <= self.done:
            return []
        iq = np.fromfile(self.path, dtype=np.complex64,
                         offset=io_lfs.HEADER_SIZE + self.done * self.window * 8,
                         count=(ready - self.done) * self.window)
        out = []
        for k in range(ready - self.done):
            row = spectro.column(iq[k * self.window:(k + 1) * self.window], self.taper,
                                 self.cal.n_range_full, self.cal.gate_idx)
            db = spectro.to_db(row)[::-1] - spectro.NOISE_FLOOR_DB
            cells = np.clip(np.round(db / DB_STEP), 0, 255).astype(np.uint8)
            out.append({"type": "column", "tx": self.head["tx"], "t0": self.head["t0"],
                        "index": self.done + k,
                        "db": base64.b64encode(cells.tobytes()).decode("ascii")})
        self.done = ready
        return out


class Journal:
    """The sounder's lines, read from a follow command on a background thread.

    The command is ``journalctl -f`` on the sounder's unit in service; a test
    gives it anything that prints. It is started again whenever it has ended,
    so a journal rotation or a restarted journald costs a gap and not the feed.
    """

    def __init__(self, command: list):
        self.command = command
        self.queue: queue.Queue = queue.Queue()
        self.process = None
        self._start()

    def lines(self) -> list:
        if self.process is not None and self.process.poll() is not None and self.queue.empty():
            self._start()
        out = []
        while not self.queue.empty():
            out.append(self.queue.get_nowait())
        return out

    def _start(self) -> None:
        try:
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError:
            self.process = None
            return
        threading.Thread(target=self._pump, args=(self.process,), daemon=True).start()

    def _pump(self, process) -> None:
        for line in process.stdout:
            self.queue.put(line.rstrip("\r\n"))


class Feed:
    """The live process: follows the sounder, posts what it sees, runs what waits."""

    def __init__(self, config: StationConfig, lines: Callable[[], list],
                 post: Callable[[list], object], clock: Callable[[], float] = time.time,
                 obey: Optional[Callable[[], None]] = None):
        self.config = config
        self.lines = lines
        self.post = post
        self.clock = clock
        self.obey = obey or (lambda: _obey(config))
        self.capture: Optional[Capture] = None
        self.backlog: collections.deque = collections.deque(maxlen=BACKLOG)

    def step(self) -> None:
        """Gather what is new and post it, each column preceded by its sounding.

        A post that fails keeps its events for the next one. The sounding goes
        into every post that carries its columns, rather than once, so that a
        server restarted mid-sounding learns its shape from the next batch; and
        it goes immediately before the first of them, so nothing read earlier
        is reordered behind it.

        A step with nothing to send still posts, empty. Its reply is how the
        feed learns that commands wait, and a stopped sounder -- the one
        waiting for Start -- has nothing else to say.
        """
        self.backlog.extend(self._events())
        batch = list(self.backlog)
        first = next((i for i, e in enumerate(batch) if e["type"] == "column"), None)
        if first is not None and self.capture is not None:
            batch.insert(first, self.capture.head)
        try:
            reply = self.post(batch)
        except client.TransportError as exc:
            refused = exc.status is not None and 400 <= exc.status < 500
            print("live: %s%s" % (exc, ", dropping %d events" % len(batch) if refused else ""),
                  file=sys.stderr, flush=True)
            if refused:
                self.backlog.clear()
            return
        self.backlog.clear()
        queued = reply.get("queued") if isinstance(reply, dict) else 0
        if isinstance(queued, int) and queued > 0:
            self.obey()

    def _events(self) -> list:
        events = [e for e in map(status_event, self.lines()) if e is not None]
        path = newest(self.config.live_dir) if self.config.live_dir else None
        if path is not None and (self.capture is None or path != self.capture.path):
            self._follow(path)
        if self.capture is None:
            return events
        try:
            return events + self.capture.columns()
        except OSError:
            self.capture.done = self.capture.head["columns"]
            return events

    def _follow(self, path: Path) -> None:
        try:
            found = Capture(path, self.config.live_window)
        except (OSError, ValueError):
            return
        if found.ends < self.clock() - FRESH_S:
            found.done = found.head["columns"]
        self.capture = found
        kept = [e for e in self.backlog if e["type"] != "column" or e["t0"] == found.head["t0"]]
        self.backlog.clear()
        self.backlog.extend(kept)


def _obey(config: StationConfig) -> None:
    """One command pass, saying on stderr what went wrong in it."""
    result = runner.PassResult()
    runner.obey(config, result)
    for problem in result.errors:
        print("live: %s" % problem, file=sys.stderr, flush=True)


def run(config: StationConfig, sleep: Callable[[float], None] = time.sleep) -> None:
    """Follow the sounder for as long as the process lives."""
    journal = Journal(["journalctl", "-u", config.live_unit, "-f", "-n", "0", "-o", "cat"])
    feed = Feed(config, journal.lines, lambda events: client.push_live(config, events))
    while True:
        started = time.time()
        feed.step()
        sleep(max(0.0, config.live_interval_s - (time.time() - started)))
