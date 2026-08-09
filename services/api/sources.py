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

import os
from dataclasses import asdict
from pathlib import Path

#: Cap on directories scanned in one request. A census reads every detection
#: file under the target, and an archive holds thousands; the endpoint is meant
#: for "what is on air today", not for a survey.
DEFAULT_MAX_DAYS = 3


def _day_directories(root: Path, max_days: int) -> list[Path]:
    """Newest dated subdirectories, or the root itself if it holds files."""
    if not root.exists():
        return []
    days = sorted((p for p in root.iterdir() if p.is_dir()),
                  key=lambda p: p.name, reverse=True)
    return days[:max_days] if days else [root]


def census(archive_root: str | os.PathLike, *,
           max_days: int = DEFAULT_MAX_DAYS,
           cycle_s: float | None = None,
           min_count: int = 3) -> dict:
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
    return {
        "count": len(emitters),
        "kind": kind,
        "cycle_s": cycle,
        "emitters": [_as_row(e) for e in emitters],
    }


def _as_row(emitter) -> dict:
    """One emitter, plus the `sounder_timings` entry it would become."""
    row = asdict(emitter)
    row["span_hours"] = round(emitter.span_hours, 2)
    row["observed_seconds"] = list(emitter.observed_seconds)
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
