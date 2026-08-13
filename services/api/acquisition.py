"""What the station is sounding this minute.

A schedule in the ini is a list of ``{chirp-rate, rep, chirpt}`` objects and
tells you nothing on its own; the question an operator actually has is *"is it
working right now"*, and answering it means putting the schedule against a
clock. That is all this module does -- arithmetic on a schedule and a
timestamp, no database, no config -- so it can be tested against a fixed
instant rather than against whatever time it happens to be.

**A slot second is a reception second.** ``chirpt`` is when this receiver
hears the chirp start: the transmit second plus the one-way travel time plus
the receiver's own epoch offset. The station listens at that second, so for
scheduling it is the right number, but it is not a transmit time -- the same
caveat that travels with the emitter census in :mod:`services.api.sources`.

**Nothing here says a product exists.** The schedule says when the station
*intends* to listen. Whether anything was recorded is a separate measurement,
which is why :func:`state` carries the arrivals alongside the slots rather
than deriving one from the other: a schedule that looks perfect while no file
has landed for six hours is exactly the failure worth seeing on one screen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Rate/slot agreement, in Hz/s and seconds, for calling a scheduled entry the
#: same thing as a verified transmitter. Both numbers come from the same
#: census originally, so this is guarding against float round-tripping through
#: JSON and a hand edit of the last digit, not against a real difference.
MATCH_RATE_HZ = 1.0
MATCH_SLOT_S = 0.5


def rank_groups(value) -> list[list[dict]]:
    """``sounder_timings`` as one list of entries per MPI rank.

    The ini stores it nested -- ``[[{...}], [{...}]]``, one group per rank of
    ``calc_ionograms.py`` -- and a browser form or a hand edit may send the
    flat form for a single rank. Both are read; a flat list is one rank.

    The grouping is not cosmetic. ``calc_ionograms.py:452`` does
    ``st = conf.sounder_timings[rank]`` with no guard, so the number of groups
    *is* the ``-np`` the launcher must use (patch 0009 derives it), and
    flattening the shape away here would lose the only copy of that number.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    if all(isinstance(item, list) for item in value):
        return [[e for e in group if isinstance(e, dict)] for group in value]
    return [[e for e in value if isinstance(e, dict)]]


def entries(value) -> list[dict]:
    """Every entry in a schedule, rank grouping discarded."""
    return [entry for group in rank_groups(value) for entry in group]


#: Every key ``calc_ionograms.py`` reads off a schedule entry, all five with a
#: bare subscript and no default:
#:
#: * ``chirp-rate``, ``rep``, ``chirpt`` -- when to listen and how fast the
#:   sweep runs (lines 444-446);
#: * ``id`` and ``transmit_name`` -- lines 446-447, and both end up in the
#:   product: ``lfm_ionogram-<transmit_name>-<station>-<ch>-<id>-<t0>.h5``.
#:
#: A three-key entry is not a smaller schedule, it is ``KeyError: 'id'`` on
#: the rank that gets it, at the first slot, with the other ranks carrying on.
#: The emitter census cannot supply the last two -- nothing in a detection
#: identifies a transmitter -- which is why a schedule is composed from
#: *identified* transmitters rather than straight from the census.
REQUIRED_ENTRY_KEYS = ("chirp-rate", "rep", "chirpt", "id", "transmit_name")


def problems(groups: list[list[dict]]) -> list[str]:
    """Everything wrong with a schedule, in the operator's words.

    All of them, not the first: a form that reports one fault per submission
    turns a five-field mistake into five round trips.
    """
    found = []
    if not groups or not any(groups):
        found.append("the schedule is empty; a scheduled station with no "
                     "schedule records nothing while reporting healthy")
    seen_ids: dict = {}
    for rank, group in enumerate(groups):
        for entry in group:
            missing = [k for k in REQUIRED_ENTRY_KEYS if k not in entry]
            if missing:
                found.append(
                    f"rank {rank} entry {entry} is missing {missing}; "
                    f"calc_ionograms.py reads all of "
                    f"{list(REQUIRED_ENTRY_KEYS)} without a default")
                continue
            if not str(entry["transmit_name"]).strip():
                found.append(f"rank {rank} entry has an empty transmit_name; "
                             f"it becomes the product's file name")
            key = entry["id"]
            if key in seen_ids and seen_ids[key] != entry["transmit_name"]:
                found.append(
                    f"id {key} is used by both {seen_ids[key]!r} and "
                    f"{entry['transmit_name']!r}; the id is part of the "
                    f"product file name, so their files would collide")
            seen_ids[key] = entry["transmit_name"]
            if _entry_floats(entry) is None:
                found.append(f"rank {rank} entry {entry} has no usable "
                             f"chirp-rate/rep/chirpt")
    return found


def compose(records: list[dict]) -> list[list[dict]]:
    """A `sounder_timings` value from verified transmitters, one rank each.

    One MPI rank group per transmitter, which is upstream's own arrangement --
    "we allocate one MPI process for each sounder to be on the safe side"
    (``calc_ionograms.analyze_realtime``) -- and the shape patch 0009 counts
    to derive the launcher's ``-np``. A transmitter with four slots is four
    entries in *one* group: the rank picks whichever of them comes soonest,
    so the slots of one site share a process, and two sites never do.
    """
    return [list(record.get("timings") or []) for record in records
            if record.get("timings")]


@dataclass
class Slot:
    """One scheduled entry, placed against a clock."""

    rank: int
    rate_hz_s: float
    rep_s: float
    chirpt_s: float
    #: Verified transmitter this entry matches, if any. ``None`` is not an
    #: error -- a schedule may legitimately carry an entry nobody has named --
    #: but it is what the identify workflow on ``/ui/sources`` is for.
    transmitter: str | None = None
    sweep_s: float | None = None
    #: Unix seconds of the most recent start at or before "now", and the next.
    started_at: float = 0.0
    next_start: float = 0.0
    in_progress: bool = False
    seconds_in: float = 0.0
    seconds_until: float = 0.0
    #: When the sweep now running finishes, if its length is known.
    ends_at: float | None = None

    @property
    def who(self) -> str:
        return self.transmitter or "unidentified"

    @property
    def label(self) -> str:
        return (f"{self.who} {self.rate_hz_s / 1e3:.1f} kHz/s @ "
                f"{self.chirpt_s:g}s/{self.rep_s:g}s")

    @property
    def next_start_utc(self) -> str:
        return _iso(self.next_start)

    @property
    def ends_at_utc(self) -> str | None:
        return _iso(self.ends_at) if self.ends_at else None

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        out["label"] = self.label
        out["next_start_utc"] = self.next_start_utc
        out["ends_at_utc"] = self.ends_at_utc
        return out


def sweep_seconds(rate_hz_s: float, span_mhz: float | None) -> float | None:
    """How long one sweep takes, or None when the band is not known.

    ``None`` rather than a guess. The sweep length decides whether a slot is
    reported as *sounding now* or as *between slots*, and a station whose band
    this server has never seen a product from should say it cannot tell rather
    than assert either.
    """
    if not rate_hz_s or span_mhz is None or span_mhz <= 0:
        return None
    return float(span_mhz) * 1e6 / float(rate_hz_s)


def _iso(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _entry_floats(entry: dict) -> tuple[float, float, float] | None:
    try:
        rate = float(entry["chirp-rate"])
        rep = float(entry["rep"])
        chirpt = float(entry["chirpt"])
    except (KeyError, TypeError, ValueError):
        return None
    # A zero cycle is not a schedule, and it is a division by zero one line
    # further down; an entry that cannot be placed on a clock is dropped rather
    # than shown at an invented time.
    return (rate, rep, chirpt) if rep > 0 else None


def place(entry: dict, now: float, *, rank: int = 0,
          span_mhz: float | None = None,
          transmitter: str | None = None) -> Slot | None:
    """One schedule entry against one instant.

    The station listens whenever ``unix_time % rep == chirpt``, so the most
    recent start is found by flooring rather than by scanning -- which is what
    makes this correct across the cycle boundary, where "the next slot" is in
    the following cycle and a naive ``chirpt - (now % rep)`` goes negative.
    """
    parsed = _entry_floats(entry)
    if parsed is None:
        return None
    rate, rep, chirpt = parsed
    chirpt %= rep                       # a slot past the end of its own cycle

    started = ((now - chirpt) // rep) * rep + chirpt
    sweep = sweep_seconds(rate, span_mhz)
    elapsed = now - started
    running = sweep is not None and elapsed < sweep
    return Slot(
        rank=rank, rate_hz_s=rate, rep_s=rep, chirpt_s=chirpt,
        transmitter=transmitter, sweep_s=sweep,
        started_at=started, next_start=started + rep,
        in_progress=running, seconds_in=elapsed,
        seconds_until=started + rep - now,
        ends_at=(started + sweep) if running else None,
    )


def name_for(entry: dict, verified: list[dict]) -> str | None:
    """What to call this entry: its own ``transmit_name``, else a match.

    The entry's own name wins, because that is the string the station will
    write into the product file. Falling back to a match on the numbers is for
    schedules written before any of this existed -- the ini on a station
    configured by hand -- where the name is the one thing missing.

    The fallback is matched on rate *and* slot second, both, because neither
    alone identifies anyone: two transmitters can share a chirp rate --
    100 kHz/s is the common one -- and a slot second is only meaningful within
    a cycle.
    """
    declared = str(entry.get("transmit_name") or "").strip()
    if declared:
        return declared

    parsed = _entry_floats(entry)
    if parsed is None:
        return None
    rate, rep, chirpt = parsed
    chirpt %= rep
    for record in verified:
        for candidate in record.get("timings") or []:
            other = _entry_floats(candidate)
            if other is None:
                continue
            if (abs(other[0] - rate) <= MATCH_RATE_HZ
                    and abs(other[2] % other[1] - chirpt) <= MATCH_SLOT_S):
                return record.get("code")
    return None


@dataclass
class Acquisition:
    """The whole answer for one station at one instant."""

    station: str
    now: float
    mode: str | None = None
    epoch: dict | None = None
    slots: list[Slot] = field(default_factory=list)
    arrivals: list[dict] = field(default_factory=list)
    span_mhz: float | None = None
    #: Why the schedule is unknown, when it is. An empty schedule and an
    #: unrecorded one are different situations: the first records nothing, the
    #: second is a station this server has simply never seen configured.
    unknown: str | None = None

    @property
    def sounding_now(self) -> list[Slot]:
        return [s for s in self.slots if s.in_progress]

    @property
    def next_slot(self) -> Slot | None:
        return min(self.slots, key=lambda s: s.seconds_until, default=None)

    @property
    def contended(self) -> list[int]:
        """Ranks whose slots overlap, so one of them is not being recorded.

        A rank is one process and it sounds one chirp at a time: it picks the
        slot with the shortest wait and is then busy for the whole sweep. Two
        slots of one rank less than a sweep apart cannot both be taken -- the
        rank reaches one and the other is skipped that cycle, quietly, at
        whatever fraction of the intended rate that works out to.

        This can only be seen against a clock, which is why it lives here and
        not in :func:`problems`: the sweep length is measured off the products,
        not declared anywhere in the schedule. On a 100 kHz/s circuit into a
        24.8 MHz receiver the sweep is 248 s of a 300 s cycle, so two slots of
        one transmitter overlap unless they are most of a cycle apart.
        """
        busy: dict[int, int] = {}
        for slot in self.sounding_now:
            busy[slot.rank] = busy.get(slot.rank, 0) + 1
        return sorted(rank for rank, n in busy.items() if n > 1)

    def as_dict(self) -> dict:
        return {
            "station": self.station, "now_utc": _iso(self.now),
            "mode": self.mode, "unknown": self.unknown,
            "span_mhz": self.span_mhz,
            "epoch": self.epoch,
            "slots": [s.as_dict() for s in self.slots],
            "sounding_now": [s.label for s in self.sounding_now],
            "contended_ranks": self.contended,
            "next": self.next_slot.as_dict() if self.next_slot else None,
            "arrivals": self.arrivals,
        }


def describe(station: str, *, timings, mode: str | None = None,
             epoch: dict | None = None, verified: list[dict] | None = None,
             span_mhz: float | None = None, arrivals: list[dict] | None = None,
             now: float | None = None) -> Acquisition:
    """Put a schedule, a band and some arrivals against a clock."""
    now = time_now() if now is None else now
    verified = verified or []
    state = Acquisition(station=station, now=now, mode=mode, epoch=epoch,
                        span_mhz=span_mhz, arrivals=list(arrivals or []))

    groups = rank_groups(timings)
    if not groups or not any(groups):
        state.unknown = (
            "no schedule recorded for this station. Nothing is wrong with the "
            "station -- this server writes a config_epoch when a set_config "
            "command is acknowledged, so a station configured by hand, or "
            "before the agent was deployed, has none.")
        return state

    for rank, group in enumerate(groups):
        for entry in group:
            slot = place(entry, now, rank=rank, span_mhz=span_mhz,
                         transmitter=name_for(entry, verified))
            if slot is not None:
                state.slots.append(slot)
    state.slots.sort(key=lambda s: (s.rank, s.chirpt_s))
    return state


def time_now() -> float:
    import time

    return time.time()


# --------------------------------------------------------------------------
# Reading the pieces out of the database
# --------------------------------------------------------------------------

def _parse_stamp(text: str | None) -> float | None:
    """``sounding.datetime`` (space-separated, naive UTC) to unix seconds."""
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text.strip().replace("Z", ""))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def arrivals(conn, station: str, limit: int = 6,
             now: float | None = None) -> list[dict]:
    """The last few products filed under this receiver, with their age.

    Filed under, not recorded by: this reads the ingested archive, so a
    station whose transfer has stalled shows old arrivals while the recorder
    is fine. That is a true statement about the archive and a useful one -- the
    agent's own ``newest_product_age_s`` metric is the one that speaks for the
    station -- but the two must not be read as the same measurement.
    """
    from . import db

    now = time_now() if now is None else now
    out = []
    for row in db.rows(conn,
                       "SELECT id, datetime, tx, rx, freq_start, freq_stop,"
                       " sweep_fraction FROM sounding WHERE rx = ?"
                       " ORDER BY datetime DESC LIMIT ?", (station, limit)):
        stamp = _parse_stamp(row["datetime"])
        row["age_s"] = None if stamp is None else round(now - stamp, 1)
        out.append(row)
    return out


def observed_span_mhz(conn, station: str, limit: int = 50) -> float | None:
    """The sweep width this receiver's recent products actually cover.

    Measured rather than configured, because the ini does not hold it in one
    place -- the band comes from the recorder's centre frequency and rate --
    and because a measured span is the one that matches the products being
    produced. The median of the last few, so one truncated sweep does not
    decide how long a sweep takes.
    """
    from . import db

    spans = [r["span"] for r in db.rows(
        conn, "SELECT (freq_stop - freq_start) AS span FROM sounding"
              " WHERE rx = ? AND freq_stop IS NOT NULL AND freq_start IS NOT NULL"
              " ORDER BY datetime DESC LIMIT ?", (station, limit))
        if r["span"] and r["span"] > 0]
    if not spans:
        return None
    spans.sort()
    return float(spans[len(spans) // 2])


def current(conn, station: str, *, now: float | None = None,
            limit: int = 6) -> Acquisition:
    """Everything the console panel needs for one station."""
    from . import db

    epoch = db.open_epoch(conn, station)
    changes = (epoch or {}).get("changes") or {}

    def latest(key):
        """The value a journalled change moved *to*, however it was recorded."""
        value = changes.get(key)
        if isinstance(value, dict):            # {"from": ..., "to": ...}
            return value.get("to")
        return value

    mode = latest("mode")
    if mode in ("true", "True"):               # the ini's own vocabulary
        mode = "search"
    elif mode in ("false", "False"):
        mode = "scheduled"

    return describe(
        station,
        timings=latest("sounder_timings"),
        mode=mode,
        epoch=None if epoch is None else {
            "id": epoch["id"], "valid_from": epoch["valid_from"],
            "changed_by": epoch["changed_by"], "note": epoch["note"]},
        verified=db.transmitters(conn, station),
        span_mhz=observed_span_mhz(conn, station),
        arrivals=arrivals(conn, station, limit=limit, now=now),
        now=now,
    )
