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
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

#: Cap on directories scanned in one request. A census reads every detection
#: file under the target, and an archive holds thousands; the endpoint is meant
#: for "what is on air today", not for a survey.
DEFAULT_MAX_DAYS = 3

#: Detections a group needs before it is reported at all. Named rather than
#: written twice, because it is half of the cache key: the page's default and
#: the startup warm-up have to ask the *same* question or the warm pass leaves
#: the first visitor paying for the archive anyway.
DEFAULT_MIN_COUNT = 3

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


#: Parsed records per detection file, keyed by path. **A chirpsounder2
#: detection product is written once and never touched again** -- its name
#: carries the unix second it belongs to -- so the path names one immutable
#: set of records and re-opening it cannot tell us anything new.
#:
#: This exists because the census was costing one HDF5 open per file on every
#: page load. On DOB that is ~1850 opens for three days -- 0.6 s on a local
#: SSD, and **two to three minutes** on the network archive the server reads,
#: which is what a page taking "a few minutes" was. Nothing about the
#: arithmetic was slow; it was opening the same files over and over.
#:
#: Values are ``(identity, records)``, where identity is the ``stat`` taken
#: when the file was read -- kept only for the retry path below, and never
#: consulted on a hit, so a warm census performs no ``stat`` calls either.
_MEMO: "OrderedDict[str, tuple]" = OrderedDict()

#: Entries kept, oldest evicted first. A day is ~1500 detection files and each
#: entry holds a handful of floats, so this covers a fortnight of archive in a
#: few MB. Bounded because the process is long-lived and an archive is not.
_MEMO_MAX = 40_000

#: One census at a time. Two operators opening the page on a cold cache would
#: otherwise each pay the full read, competing for the same disk -- the second
#: one waits and then finds the answer already in hand.
_CENSUS_LOCK = threading.Lock()

#: Last result, with the fingerprint of the files it was computed from.
_LAST: dict = {}


def _identity(path: Path) -> tuple | None:
    """`(mtime, size)`, or None if the file went away mid-scan."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_cached(paths, reader, expand=None) -> tuple[list, dict]:
    """Records for ``paths``, opening only the files not already read.

    Returns ``(records, counts)`` with ``opened``, ``cached`` and
    ``unreadable``.

    A file that will not parse is remembered as **empty**. A detector caught
    mid-write is normal and one truncated file must not lose the census, but
    neither should it be re-opened on every page load for the rest of the
    archive's life. That is the one case where a path's content does change,
    so it is also the only case that pays for a ``stat``: an entry that parsed
    to nothing is re-checked, and re-read if the file has grown since.

    Unreadable files are counted and warned about, as the ``io_detect``
    loaders this replaces did. A census that quietly drops a third of the
    archive looks exactly like a quiet one.
    """
    records = []
    counts = {"opened": 0, "cached": 0, "unreadable": 0}
    for path in paths:
        key = str(path)
        entry = _MEMO.get(key)
        if entry is not None:
            was, got = entry
            if got:                       # parsed once, immutable: trust it
                _MEMO.move_to_end(key)
                counts["cached"] += 1
                records.extend(got)
                continue
            if was is not None and was == _identity(path):
                counts["cached"] += 1     # still the same broken file
                counts["unreadable"] += 1
                continue
        try:
            parsed = reader(path)
            got = list(expand(path, parsed)) if expand else [parsed]
        except Exception:
            got = []
        if not got:
            counts["unreadable"] += 1
        _MEMO[key] = (_identity(path) if not got else None, got)
        _MEMO.move_to_end(key)
        counts["opened"] += 1
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)
        records.extend(got)
    return records, counts


def _cdetection_rows(path, rows):
    """One `cdetections-*.h5` array as `Detection` records. See io_detect."""
    from muf.io_detect import Detection

    for chirp_time, _i0_seconds, f0, rate, snr in rows:
        yield Detection(path=path, channel="", chirp_time=float(chirp_time),
                        rate=float(rate), f0=float(f0), snr=float(snr),
                        i0=-1, n_samples=-1, sample_rate=float("nan"))


def census(archive_root: str | os.PathLike, *,
           max_days: int = DEFAULT_MAX_DAYS,
           cycle_s: float | None = None,
           min_count: int = DEFAULT_MIN_COUNT,
           max_scatter_s: float = DEFAULT_MAX_SCATTER_S,
           max_slot_fraction: float = DEFAULT_MAX_SLOT_FRACTION,
           min_repeats: float = DEFAULT_MIN_REPEATS) -> dict:
    """Repeating emitters under ``archive_root``, newest days first.

    Reads whichever detection product the tree actually has, in the order
    ``muf detect`` does: timing solutions first, then raw detections, then the
    consolidated summaries -- which are often the only ones left on a synced
    archive, because ``chirp-*.h5`` lived in the ringbuffer and rotated away.

    **The preference order is about quality, not cost.** The consolidated files
    are by far the cheapest to read -- 96 of them hold what 1500 ``chirp-*.h5``
    do -- but they are the detector's raw candidates, not its conclusions. On
    one real day they produced a 100 kHz/s "emitter" with 26,137 detections
    spread across nearly every second of the cycle, which the occupancy filter
    then rejects as interference: reading the cheap files first would lose the
    transmitter the page exists to find. So the expensive files stay first and
    the cost is paid by caching instead.

    Every file read is remembered, and the scan is fingerprinted on the names
    the directory listing already yields, so a second call over an archive that
    has not changed opens nothing and stats nothing. The returned ``cost``
    block reports what it did, because a page that is slow for a reason the
    operator cannot see is a page that gets guessed about.
    """
    from muf import io_detect

    cycle = cycle_s or io_detect.DEFAULT_CYCLE_S
    root = Path(archive_root)
    started = time.perf_counter()

    # Scan first, read second. A directory listing is one round trip per
    # directory; an HDF5 open is several per *file*. Doing the cheap half
    # first is what lets an unchanged archive answer without opening anything.
    scans, matched = [], 0
    for day in _day_directories(root, max_days):
        for finder, reader, expand, name in (
                (io_detect.find_timings, io_detect.read_timing, None,
                 "timing solution"),
                (io_detect.find_detections, io_detect.read_detection, None,
                 "detection"),
                (io_detect.find_cdetections, io_detect.read_cdetections,
                 _cdetection_rows, "consolidated detection")):
            try:
                paths = finder(day)
            except Exception:
                paths = []
            if paths:
                scans.append((paths, reader, expand, name))
                matched += len(paths)
                break

    # Fingerprinted on the file *names*, which the scan above already has for
    # free -- not on their contents, and not on a `stat` per file. Every one
    # of these names carries the second it belongs to, so a name that was
    # there last time refers to the same recording. Stat-ing 1846 files to
    # prove that would cost a round trip each on the archive this is slow on,
    # which is most of what we are trying to avoid.
    fingerprint = (str(root), max_days, cycle, min_count, max_scatter_s,
                   max_slot_fraction, min_repeats,
                   tuple(sorted(str(p) for paths, *_ in scans for p in paths)))

    with _CENSUS_LOCK:
        # The names matching is only proof that nothing changed if every file
        # behind them was read. A detector caught mid-write keeps its name
        # when it finishes, so an archive with a skipped file goes down the
        # read path again -- which is nearly free, since every good file is a
        # cache hit and only the skipped ones are looked at.
        settled = not _LAST.get("census", {}).get("cost", {}).get("unreadable")
        if settled and _LAST.get("fingerprint") == fingerprint:
            out = dict(_LAST["census"])
            # This call's cost, not the cost of the call that filled the
            # cache. Reporting the earlier one would make the page claim it
            # had just opened 1846 files when it opened none.
            out["cost"] = dict(out["cost"], unchanged=True,
                               opened=0, cached=matched,
                               seconds=round(time.perf_counter() - started, 2))
            return out

        records, kind = [], "none"
        totals = {"opened": 0, "cached": 0, "unreadable": 0}
        for paths, reader, expand, name in scans:
            got, counts = _read_cached(paths, reader, expand)
            for field, n in counts.items():
                totals[field] += n
            if got:
                records.extend(got)
                kind = name
        if totals["unreadable"]:
            warnings.warn(
                f"skipped {totals['unreadable']} unreadable detection file(s) "
                f"under {root}", stacklevel=2)

        cost = dict(
            totals,
            days=[d.name for d in _day_directories(root, max_days)],
            files=matched, records=len(records), unchanged=False,
            seconds=round(time.perf_counter() - started, 2),
        )

        if not records:
            out = {"count": 0, "kind": "none", "cycle_s": cycle,
                   "emitters": [], "cost": cost}
            _LAST.update(fingerprint=fingerprint, census=out)
            return out

        emitters = io_detect.census(records, cycle_s=cycle,
                                    min_count=min_count)
        kept, rejected = [], []
        for emitter in emitters:
            why = _rejection(emitter, cycle, max_scatter_s, max_slot_fraction,
                             min_repeats)
            (rejected if why else kept).append(
                (emitter, why) if why else emitter)
        cost["seconds"] = round(time.perf_counter() - started, 2)
        out = {
            "count": len(kept),
            "kind": kind,
            "cycle_s": cycle,
            "emitters": [_as_row(e) for e in kept],
            # Kept, not silently dropped. A schedule page that hides its
            # rejects cannot be checked, and the operator is the one who knows
            # whether the thing it threw away was the transmitter they came
            # for.
            "rejected": [dict(_as_row(e), rejected_because=why)
                         for e, why in rejected],
            "cost": cost,
        }
        _LAST.update(fingerprint=fingerprint, census=out)
        return out


def warm(archive_root, *, max_days: int = DEFAULT_MAX_DAYS,
         min_count: int = DEFAULT_MIN_COUNT) -> dict | None:
    """Pay the cold read here, so the first visitor does not.

    The cache lives in this process's memory, so every container start hands
    the whole archive to whoever opens the page first -- 234 s on the work
    server, which is indistinguishable from the page being broken. Started in
    the background at boot, the same read happens while nothing is waiting on
    it.

    Called with the parameters the page defaults to, because the short-circuit
    is keyed on them: warming a *different* question fills the per-file memo
    but still leaves the first request doing the grouping.

    Never raises. A missing or unreadable archive must not stop the api from
    starting -- the console's other pages work without it, and a census that
    cannot run will report its own failure on the page that needs it.
    """
    try:
        return census(archive_root, max_days=max_days, min_count=min_count)
    except Exception as exc:                                  # noqa: BLE001
        warnings.warn(f"census warm-up failed: {exc!r}", stacklevel=2)
        return None


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
