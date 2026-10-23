"""The live view: what each station is sounding right now.

The station agent posts batches of events as a capture grows
(``services/agent/live.py``); this module keeps the latest of them in memory
and fans every one out to the browsers watching that station, over
Server-Sent Events. ``architecture.md`` sec. 5.5 has the event vocabulary.

**Memory, not the database.** A live column is worth something for the four
minutes its sounding lasts and nothing afterwards -- the archived product is
the record. What survives a sounding here is its summary, one SNR value per
frequency step and one PDP value per range row, kept for a day so the console
can draw the daily course without a query. Every summary is also handed to
``keep``, which the app points at the database, so the course outlives a
restart; and the agent sends summaries of soundings it has archived, which go
to ``keep`` alone (issue #12).

**One vocabulary in both directions.** What the agent posts is what a browser
receives, plus the ``summary`` this module derives when a sounding ends. The
hub is a relay with a memory, not a translator.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import collections
import json
import math
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from services.agent import summary

#: Summaries kept per transmitter: a day at the 300 s repetition period.
DAY = 288

#: Frames a browser may fall behind by before it is cut off. It reconnects and
#: starts again from a snapshot, which is cheaper for everyone than a server
#: holding an unbounded queue for a tab left open in the background.
BACKLOG = 2048

#: dB per count in a column's bytes, as the agent encodes them.
DB_STEP = 0.25

MAX_COLUMNS = 20000
MAX_ROWS = 4096
MAX_TX = 64

STATUS_FIELDS = ("progress", "overflows", "t0", "t1", "got")
SOUNDING_FIELDS = ("t0", "f_lo_mhz", "f_step_mhz", "range_lo_km",
                   "range_step_km", "km_per_ms")
#: Where an echo stood out over a sounding's whole range, which the agent says
#: after its last column: the rows it sent cover only what the path allows,
#: so an echo a station's clock has moved out of them is otherwise invisible.
ECHO_FIELDS = ("t0", "lo_km", "hi_km", "sigma")
AXES = ("tx", *SOUNDING_FIELDS, "columns", "rows")


class Refused(ValueError):
    """An event the hub will not take, and why."""


class Sounding:
    """One sounding of one transmitter, as far as its columns have arrived."""

    def __init__(self, head: dict):
        self.head = head
        self.data: dict[str, str] = {}

    def view(self) -> dict:
        return dict(self.head, data=dict(self.data))

    def summary(self) -> dict:
        cells = np.zeros((self.head["columns"], self.head["rows"]), np.float64)
        for index, text in self.data.items():
            cells[int(index)] = np.frombuffer(base64.b64decode(text), np.uint8)
        taken = [int(i) for i in self.data]
        return dict({k: self.head[k] for k in AXES}, type="summary", source="live",
                    snr=summary.snr(cells), pdp=summary.pdp(cells[taken]))


class Board:
    """Everything known live about one station."""

    def __init__(self):
        self.status: dict[str, dict] = {}
        self.current: dict[str, Sounding] = {}
        self.last: dict[str, Sounding] = {}
        self.daily: dict[str, collections.deque] = {}
        self.echo: dict[str, dict] = {}


class Reader:
    """One browser's place in the stream."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.open = True

    def offer(self, frame: str) -> None:
        if not self.open:
            return
        if self.queue.qsize() < BACKLOG:
            self.queue.put_nowait(frame)
            return
        while not self.queue.empty():
            self.queue.get_nowait()
        self.end()

    def end(self) -> None:
        if self.open:
            self.open = False
            self.queue.put_nowait(None)


class Hub:
    """Live state of every station, and the browsers watching each one."""

    def __init__(self,
                 keep: Callable[[str, dict], None] = lambda station, summary: None):
        self.keep = keep
        self.boards: dict[str, Board] = collections.defaultdict(Board)
        self.readers: dict[str, set[Reader]] = collections.defaultdict(set)
        self.closed = False

    def publish(self, station: str, events: list) -> int:
        board = self.boards[station]
        clean = _checked(events, board)
        for event in clean:
            if event["type"] == "summary":
                self.keep(station, event)
                continue
            for out in _apply(board, event):
                if out["type"] == "summary":
                    self.keep(station, out)
                self._send(station, out)
        return len(clean)

    def snapshot(self, station: str) -> dict:
        board = self.boards.get(station) or Board()
        return {"station": station,
                "status": dict(board.status),
                "current": {tx: s.view() for tx, s in board.current.items()},
                "last": {tx: s.view() for tx, s in board.last.items()},
                "daily": {tx: list(d) for tx, d in board.daily.items()},
                "echo": dict(board.echo)}

    def subscribe(self, station: str) -> Reader:
        reader = Reader()
        self.readers[station].add(reader)
        if self.closed:
            reader.end()
        return reader

    def unsubscribe(self, station: str, reader: Reader) -> None:
        self.readers[station].discard(reader)

    def close(self) -> None:
        self.closed = True
        for readers in self.readers.values():
            for reader in readers:
                reader.end()

    def _send(self, station: str, event: dict) -> None:
        text = frame(event["type"], event)
        for reader in list(self.readers.get(station, ())):
            reader.offer(text)
            if not reader.open:
                self.readers[station].discard(reader)


def frame(kind: str, payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":"))
    return "event: %s\ndata: %s\n\n" % (kind, body)


async def stream(hub: Hub, station: str,
                 disconnected: Callable[[], Awaitable[bool]],
                 keepalive: float = 15.0) -> AsyncIterator[str]:
    """One browser's Server-Sent Events: a snapshot, then every event as it comes.

    Ends when the hub closes, when the browser has fallen too far behind, or
    when a keepalive finds the browser gone. A comment line every
    ``keepalive`` seconds keeps proxies from timing the connection out between
    soundings, when nothing else is said for a minute at a time.
    """
    reader = hub.subscribe(station)
    try:
        yield frame("snapshot", hub.snapshot(station))
        while True:
            try:
                item = await asyncio.wait_for(reader.queue.get(), keepalive)
            except asyncio.TimeoutError:
                if await disconnected():
                    return
                yield ": keepalive\n\n"
                continue
            if item is None:
                return
            yield item
    finally:
        hub.unsubscribe(station, reader)


def _apply(board: Board, event: dict) -> list[dict]:
    """Apply one checked event, and say what the browsers should hear.

    **A sounding ends on its own last column, or when the next one starts --
    never on a status line.** ``STATUS ... writing`` carries no ``t0``, and the
    agent reads it from the journal and the columns from the disk, so in one
    batch it can arrive ahead of the final columns it follows in time. Ending
    on it once dropped those columns into a sounding that no longer existed.
    """
    tx = event["tx"]
    if event["type"] == "status":
        board.status[tx] = event
        return [event]
    if event["type"] == "echo":
        board.echo[tx] = event
        return [event]
    held = board.current.get(tx)
    if event["type"] == "sounding":
        if any(s is not None and s.head["t0"] == event["t0"]
               for s in (held, board.last.get(tx))):
            return []
        out = _finish(board, tx)
        board.current[tx] = Sounding(event)
        return [*out, event]
    if held is None or held.head["t0"] != event["t0"]:
        return []
    held.data[str(event["index"])] = event["db"]
    if len(held.data) < held.head["columns"]:
        return [event]
    return [event, *_finish(board, tx)]


def _finish(board: Board, tx: str) -> list[dict]:
    done = board.current.pop(tx, None)
    if done is None:
        return []
    board.last[tx] = done
    if not done.data:
        return []
    summary = done.summary()
    board.daily.setdefault(tx, collections.deque(maxlen=DAY)).append(summary)
    return [summary]


def _checked(events: list, board: Board) -> list[dict]:
    """Every event in the batch, cleaned, or Refused for the first bad one.

    All of a batch or none of it: a column is checked against the sounding it
    belongs to, which may be declared earlier in the same batch, so the shapes
    held so far are tracked here rather than read from a board half updated.
    """
    shapes = {(s.head["tx"], s.head["t0"]): s.head
              for s in list(board.current.values()) + list(board.last.values())}
    out = []
    for n, event in enumerate(events):
        if not isinstance(event, dict):
            raise Refused("event %d is not an object" % n)
        kind = event.get("type")
        tx = event.get("tx")
        if not isinstance(tx, str) or not tx.strip() or len(tx) > MAX_TX:
            raise Refused("event %d has no usable `tx`" % n)
        if kind == "status":
            out.append(_status(n, event))
        elif kind == "sounding":
            head = _sounding(n, event)
            shapes[(tx, head["t0"])] = head
            out.append(head)
        elif kind == "column":
            out.append(_column(n, event, shapes.get((tx, event.get("t0")))))
        elif kind == "summary":
            out.append(_summary(n, event))
        elif kind == "echo":
            out.append(_echo(n, event))
        else:
            raise Refused("event %d has unknown type %r" % (n, kind))
    return out


def _status(n: int, event: dict) -> dict:
    state = event.get("state")
    if not isinstance(state, str) or not state.isalpha() or len(state) > 32:
        raise Refused("status %d has no usable `state`" % n)
    clean = {"type": "status", "tx": event["tx"], "state": state}
    for name in STATUS_FIELDS:
        if name in event:
            clean[name] = _finite(n, event, name)
    return clean


def _sounding(n: int, event: dict) -> dict:
    head = {"type": "sounding", "tx": event["tx"]}
    for name in SOUNDING_FIELDS:
        head[name] = _finite(n, event, name)
    for name, top in (("columns", MAX_COLUMNS), ("rows", MAX_ROWS)):
        value = event.get(name)
        if (not isinstance(value, int) or isinstance(value, bool)
                or not 0 < value <= top):
            raise Refused("sounding %d has `%s` outside 1..%d" % (n, name, top))
        head[name] = value
    if min(head["f_step_mhz"], head["range_step_km"], head["km_per_ms"]) <= 0:
        raise Refused("sounding %d has a step that is not positive" % n)
    return head


def _echo(n: int, event: dict) -> dict:
    clean = {"type": "echo", "tx": event["tx"]}
    for name in ECHO_FIELDS:
        clean[name] = _finite(n, event, name)
    if clean["lo_km"] > clean["hi_km"]:
        raise Refused("echo %d ends before it starts" % n)
    return clean


def _summary(n: int, event: dict) -> dict:
    """A sounding the agent summarised from its archive, with the axes it has."""
    head = _sounding(n, event)
    clean = dict({k: head[k] for k in AXES}, type="summary", source="archive")
    for name, size in (("snr", head["columns"]), ("pdp", head["rows"])):
        try:
            got = len(base64.b64decode(event.get(name), validate=True))
        except (TypeError, ValueError, binascii.Error):
            raise Refused("summary %d has `%s` that is not base64"
                          % (n, name)) from None
        if got != size:
            raise Refused("summary %d has %d `%s` cells for %d" % (n, got, name, size))
        clean[name] = event[name]
    return clean


def _column(n: int, event: dict, head: dict | None) -> dict:
    if head is None:
        raise Refused("column %d belongs to no sounding this station has sent" % n)
    index = event.get("index")
    if (not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < head["columns"]):
        raise Refused("column %d has index %r outside 0..%d"
                      % (n, index, head["columns"] - 1))
    text = event.get("db")
    try:
        size = len(base64.b64decode(text, validate=True))
    except (TypeError, ValueError, binascii.Error):
        raise Refused("column %d has `db` that is not base64" % n) from None
    if size != head["rows"]:
        raise Refused("column %d has %d cells for %d rows" % (n, size, head["rows"]))
    return {"type": "column", "tx": event["tx"], "t0": head["t0"], "index": index,
            "db": text}


def _finite(n: int, event: dict, name: str) -> float:
    value = event.get(name)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise Refused("event %d has `%s` that is not a finite number" % (n, name))
    return value

