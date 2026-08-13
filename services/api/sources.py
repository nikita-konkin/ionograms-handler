"""Transmitters the station has heard, as candidates for a schedule.

This is the join between the station's two sounding modes, and it is why
search mode exists at all. In **search** (serendipitous) mode the station
records everything that sweeps past and ``find_timings.py`` infers who was
transmitting; in **scheduled** mode it downconverts a fixed list of
transmitters at times it is told. The output of the first is the input to the
second: an :class:`~muf.io_detect.Emitter` carries a chirp rate, a repeat
cycle and a slot second, which is exactly a ``sounder_timings`` entry.

``control.py`` already refuses to leave search mode without that list --
"scheduled mode with an empty ``sounder_timings`` would record nothing while
every process reported healthy" -- so an operator needs somewhere to get one.
Before this, that meant running ``muf detect`` on the station by hand and
transcribing the numbers.

**The seconds are as received, not as transmitted.** ``muf detect`` says so
every time it prints, and the same caveat travels with these records: the slot
a transmitter appears in is its transmit second plus the one-way travel time
plus whatever this receiver's epoch offset is. For scheduling that is the
right number anyway -- the station wants to know when to *listen* -- but it is
not a transmit time and it is not a range.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict
from pathlib import Path

#: Cap on directories scanned in one request. A census reads every detection
#: file under the target, and an archive holds thousands; the endpoint is meant
#: for "what is on air today", not for a survey.
DEFAULT_MAX_DAYS = 3

#: Scatter of the fractional-second offset, past which a group is not one
#: transmitter. A real chirp starts at the same point within its second every
#: time -- the tight groups on DOB sit at +/-0.9 and +/-1.6 ms across hundreds
#: of detections. Noise does not: the worst group in that archive claimed
#: +/-274 ms, which is most of a second of "when".
DEFAULT_MAX_SCATTER_S = 5e-3

#: Share of the cycle's seconds a single emitter may occupy. One transmitter
#: does not transmit in half the seconds of its own repeat period. On DOB the
#: 500 kHz/s group claimed *all three hundred* -- every second, 0 to 299 --
#: which is a detector firing on broadband interference, not a schedule.
DEFAULT_MAX_SLOT_FRACTION = 0.25

#: Times a slot must be heard again before it is a schedule and not a
#: coincidence. `count` is detection records; `observed_seconds` is the
#: *distinct* seconds they fell in, so their ratio is how often the emitter
#: came back to the same second.
#:
#: A transmitter on a 300 s cycle hands you the same slot ~240 times in 20
#: hours. Thirteen groups on DOB scored exactly 1.0 -- every slot seen once and
#: never again -- which is what unrelated detections look like when the phase
#: grouping collects them. The real ones scored 24, 39 and 855; nothing landed
#: between 1.5 and 24, so this threshold is in open space.
DEFAULT_MIN_REPEATS = 3.0


#: A directory name that is a date: ``2026-08-10``, ``2026.02.04``, ``20260810``.
_DAY_RE = re.compile(r"^(\d{4})[-._]?(\d{2})[-._]?(\d{2})$")


def _day_key(name: str) -> tuple[int, int, int] | None:
    match = _DAY_RE.match(name)
    return (int(match.group(1)), int(match.group(2)),
            int(match.group(3))) if match else None


def _day_directories(root: Path, max_days: int) -> list[Path]:
    """Newest dated subdirectories, or the root itself if it holds files.

    Sorted by the date the name *means*, not by the string. A reverse lexical
    sort is wrong twice over on a real archive, and both were live here:

    * Two conventions coexist. ``.`` is 0x2E and ``-`` is 0x2D, so every
      ``2026.02.*`` sorts ahead of every ``2026-08-*`` -- the census reported
      February as "what is on air today" and never opened an August day.
    * Not every subdirectory is a day. ``ionozond_data2`` beats any digit and
      took a slot outright.

    A directory whose name is not a date is not a day, and is skipped rather
    than ranked. If none of them are dates the tree is flat, and the root is
    scanned as before.
    """
    if not root.exists():
        return []
    dated = [(key, p) for p in root.iterdir() if p.is_dir()
             for key in (_day_key(p.name),) if key is not None]
    if not dated:
        return [root]
    dated.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in dated[:max_days]]


def census(archive_root: str | os.PathLike, *,
           max_days: int = DEFAULT_MAX_DAYS,
           cycle_s: float | None = None,
           min_count: int = 3,
           max_scatter_s: float = DEFAULT_MAX_SCATTER_S,
           max_slot_fraction: float = DEFAULT_MAX_SLOT_FRACTION,
           min_repeats: float = DEFAULT_MIN_REPEATS) -> dict:
    """Repeating emitters under ``archive_root``, newest days first.

    Reads whichever detection product the tree actually has, in the order
    ``muf detect`` does: timing solutions first, then raw detections, then the
    consolidated summaries -- which are often the only ones left on a synced
    archive, because ``chirp-*.h5`` lived in the ringbuffer and rotated away.
    """
    from muf import io_detect

    cycle = cycle_s or io_detect.DEFAULT_CYCLE_S
    root = Path(archive_root)

    records, kind = [], "none"
    for day in _day_directories(root, max_days):
        for loader, name in ((io_detect.load_timings, "timing solution"),
                             (io_detect.load_detections, "detection"),
                             (io_detect.load_cdetections,
                              "consolidated detection")):
            try:
                found = loader(day)
            except Exception:
                found = []
            if found:
                records.extend(found)
                kind = name
                break

    if not records:
        return {"count": 0, "kind": "none", "cycle_s": cycle, "emitters": []}

    emitters = io_detect.census(records, cycle_s=cycle, min_count=min_count)
    kept, rejected = [], []
    for emitter in emitters:
        why = _rejection(emitter, cycle, max_scatter_s, max_slot_fraction,
                         min_repeats)
        (rejected if why else kept).append(
            (emitter, why) if why else emitter)
    return {
        "count": len(kept),
        "kind": kind,
        "cycle_s": cycle,
        "emitters": [_as_row(e) for e in kept],
        # Kept, not silently dropped. A schedule page that hides its rejects
        # cannot be checked, and the operator is the one who knows whether the
        # thing it threw away was the transmitter they came for.
        "rejected": [dict(_as_row(e), rejected_because=why)
                     for e, why in rejected],
    }


def _repeats_per_slot(emitter) -> float:
    """How often the emitter came back to the same second."""
    slots = len(emitter.observed_seconds)
    return (emitter.count / slots) if slots else 0.0


def _rejection(emitter, cycle_s: float, max_scatter_s: float,
               max_slot_fraction: float,
               min_repeats: float = DEFAULT_MIN_REPEATS) -> str | None:
    """Why this group is not a transmitter, or None if it might be.

    Both tests are about self-consistency rather than strength, because
    strength is exactly what fools you here: the 500 kHz/s group on DOB had a
    median SNR of 68 -- higher than cyprus1 -- while occupying every second of
    the cycle. Loud and everywhere is interference; a transmitter is quiet in
    291 seconds out of 300 and always arrives at the same instant within the
    nine.
    """
    scatter = getattr(emitter, "fraction_sd_s", 0.0) or 0.0
    if scatter > max_scatter_s:
        return (f"fractional offset scatters by +/-{scatter * 1e3:.0f} ms; a "
                f"transmitter holds it under {max_scatter_s * 1e3:.0f} ms")
    slots = len(emitter.observed_seconds)
    share = slots / cycle_s if cycle_s else 0.0
    if share > max_slot_fraction:
        return (f"occupies {slots} of {cycle_s:.0f} seconds "
                f"({share * 100:.0f}%); a schedule is sparse")
    repeats = _repeats_per_slot(emitter)
    if repeats < min_repeats:
        return (f"each slot heard {repeats:.1f} time(s); a transmitter on a "
                f"{cycle_s:.0f} s cycle returns to its slot every cycle, so "
                f"this is coincidence, not a schedule")
    return None


def _finite(value):
    """``NaN``/``inf`` to ``None``, because neither is JSON.

    Python writes them as the bare tokens ``NaN`` and ``Infinity``, which
    `json.dumps` emits by default and no strict parser accepts --
    ``JSON.parse`` in a browser throws on the first one. A census row is not
    exotic input either: a group whose detections carry no SNR field has a
    ``NaN`` median, and a single-slot group has no scatter to compute.

    So the whole row was unreadable in the browser, and `/sources` returned a
    document that says it is JSON and is not, because of one absent field in
    one column nobody was reading.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_row(emitter) -> dict:
    """One emitter, plus the `sounder_timings` entry it would become."""
    row = {key: _finite(value) for key, value in asdict(emitter).items()}
    row["span_hours"] = round(emitter.span_hours, 2)
    row["observed_seconds"] = list(emitter.observed_seconds)
    row["repeats_per_slot"] = round(_repeats_per_slot(emitter), 1)
    # The entry `control.set_config` would write. `transmit_name` is left for
    # the operator: nothing in a detection identifies the transmitter, and a
    # guessed name would end up in the product file name and then in the
    # database, looking like knowledge.
    row["timing_entry"] = {
        "chirp-rate": emitter.rate,
        "rep": emitter.cycle_s,
        "chirpt": (float(emitter.observed_seconds[0])
                   if emitter.observed_seconds else 0.0),
    }
    return row
