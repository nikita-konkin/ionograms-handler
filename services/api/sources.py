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

#: Detection files one census may open. The scan below is bounded by days, and
#: the assumption underneath that was that a day is a bounded amount of work:
#: the archive this was written against held 1846 files across three days, and
#: a cold census of it cost 234 s.
#:
#: DOB is not that archive. On 2026-08-15 its newest three days held 172,056
#: files, of which 45,602 were the ``chirp-*.h5`` this reads first -- 93x the
#: design point, and at 50-100 ms an open on a network archive, hours. The
#: warm-up started, took the census lock, and never came back; every request
#: queued behind it, so the page did not answer slowly, it did not answer.
#:
#: So the day bound is not enough and there is a file bound too. Days are read
#: newest first and the budget is spent in that order, which degrades the way
#: the page is used: today stays whole, and it is the oldest day that gets
#: trimmed. What is trimmed is *time*, not quality -- the newest files of the
#: preferred product, never a fallback to a cheaper one, for the reason in
#: `census`'s docstring. 2000 keeps the cost near the 234 s this was built for.
DEFAULT_MAX_FILES = 2000

#: How old a served census may be before a new one is started. The refresh runs
#: in the background and the page is answered from the previous result, because
#: on the archive this actually runs against there is no such thing as a census
#: on the request path: **one `os.scandir` of one day's directory measured
#: 293.8 s for 46,436 entries** -- 6.3 ms per directory entry, which is a
#: network round trip each, not a disk. Three days is a quarter of an hour
#: before the first file is opened.
#:
#: So the page serves what the last completed census found and says how old it
#: is. A transmitter schedule does not change minute to minute; a page that
#: never renders does.
#:
#: Half an hour, because a refresh costs about five minutes of listing even
#: after the scan was cut to one pass per day. Ten minutes would leave the
#: server scanning half the time it is up, for an answer that changes when a
#: transmitter comes on air -- which is not a per-minute event.
DEFAULT_MAX_AGE_S = 1800.0


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

#: One background refresh at a time, and never two queued behind each other.
#: The lock alone is not enough: two requests can both find it free, both
#: spawn, and the second then repeats a fifteen-minute scan the first just
#: finished.
_REFRESH_LOCK = threading.Lock()
_REFRESHING = False


def _start_refresh(work) -> bool:
    """Run ``work`` in a daemon thread unless a refresh is already running."""
    global _REFRESHING
    with _REFRESH_LOCK:
        if _REFRESHING:
            return False
        _REFRESHING = True

    def run() -> None:
        global _REFRESHING
        try:
            work()
        except Exception as exc:                              # noqa: BLE001
            warnings.warn(f"census refresh failed: {exc!r}", stacklevel=2)
        finally:
            with _REFRESH_LOCK:
                _REFRESHING = False

    threading.Thread(target=run, daemon=True, name="census-refresh").start()
    return True


def _identity(path: Path) -> tuple | None:
    """`(mtime, size)`, or None if the file went away mid-scan."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _file_time(path: Path) -> tuple[int, float, str]:
    """Sort key putting the newest detection file last, without a ``stat``.

    Every chirpsounder2 detection name ends in the instant it belongs to:
    ``chirp-<channel>-<rate>-<i0>-<unix>.h5``, ``par-<channel>-<unix>.h5``,
    ``cdetections-<station>-<unix>.h5``. Sorting on the *whole name* looks
    equivalent and is not -- ``i0`` is a sample index of no fixed width, so
    ``chirp-ch0-100-9000-...`` sorts after ``chirp-ch0-100-44664265260000000-...``
    on the leading ``9`` and the order becomes one over channel and sample
    index, with time as a tiebreak. That is the wrong 2000 files.

    A name that does not end in a number sorts first, so an unrecognised one
    is dropped before a good one rather than displacing it.
    """
    tail = path.stem.rsplit("-", 1)[-1]
    try:
        return (1, float(tail), path.name)
    except ValueError:
        return (0, 0.0, path.name)


def _read_cached(paths, reader, expand=None) -> tuple[list, dict, list]:
    """Records for ``paths``, opening only the files not already read.

    Returns ``(records, counts, skipped)`` with ``opened``, ``cached`` and
    ``unreadable`` counted, and the paths that would not parse listed --
    :func:`census` keeps those to decide whether a later call may trust its
    cached answer.

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
    skipped: list = []
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
                skipped.append(path)
                continue
        try:
            parsed = reader(path)
            got = list(expand(path, parsed)) if expand else [parsed]
        except Exception:
            got = []
        if not got:
            counts["unreadable"] += 1
            skipped.append(path)
        _MEMO[key] = (_identity(path) if not got else None, got)
        _MEMO.move_to_end(key)
        counts["opened"] += 1
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)
        records.extend(got)
    return records, counts, skipped


def _served(params: tuple, max_age_s: float | None) -> dict | None:
    """The last census, if it answers *this* question and is young enough.

    Keyed on the tuning parameters alone, not on the file list: proving the
    file list has not changed costs the scan, which is the thing that cannot
    be afforded on a request. Age is what stands in for it.
    """
    if _LAST.get("params") != params or "census" not in _LAST:
        return None
    age = time.time() - _LAST.get("at", 0.0)
    if max_age_s is not None and age > max_age_s:
        return None
    out = dict(_LAST["census"])
    out["age_s"] = round(age, 1)
    out["refreshing"] = _REFRESHING
    out["cost"] = dict(out["cost"], unchanged=True, opened=0, seconds=0.0)
    return out


def _building(cycle: float, max_files: int) -> dict:
    """No census has finished yet. Distinct from "no transmitters heard"."""
    return {
        "count": 0, "kind": "none", "cycle_s": cycle, "emitters": [],
        "rejected": [], "building": True, "age_s": None, "refreshing": True,
        "cost": {"opened": 0, "cached": 0, "unreadable": 0, "days": [],
                 "files": 0, "found": 0, "capped": 0, "budget": max_files,
                 "records": 0, "unchanged": False, "seconds": 0.0,
                 "building": True},
    }


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
           min_repeats: float = DEFAULT_MIN_REPEATS,
           max_files: int = DEFAULT_MAX_FILES,
           block: bool = True,
           max_age_s: float | None = None) -> dict:
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

    At most ``max_files`` are opened. Past that the census reads the newest
    files it found and reports ``capped``, rather than beginning a read it
    cannot finish -- see `DEFAULT_MAX_FILES`. It trims time and not quality:
    the preferred product is kept and its oldest files are dropped, because
    falling back to the cheap ones is what loses the transmitter.

    ``block=False`` never touches the archive. It answers from the last
    completed census, starting a background refresh if that answer is older
    than ``max_age_s``, and reports ``building`` when there is nothing yet.
    **That is what a request should use**, because the scan alone -- before any
    file is opened -- measured 293.8 s per day on the archive this serves; see
    `DEFAULT_MAX_AGE_S`. ``block=True`` is the real thing, for the warm-up, the
    background refresh, and the command line.
    """
    from muf import io_detect

    cycle = cycle_s or io_detect.DEFAULT_CYCLE_S
    root = Path(archive_root)
    started = time.perf_counter()
    params = (str(root), max_days, cycle, min_count, max_scatter_s,
              max_slot_fraction, min_repeats, max_files)

    if not block:
        served = _served(params, max_age_s)
        if served is not None:
            return served
        _start_refresh(lambda: census(
            archive_root, max_days=max_days, cycle_s=cycle_s,
            min_count=min_count, max_scatter_s=max_scatter_s,
            max_slot_fraction=max_slot_fraction, min_repeats=min_repeats,
            max_files=max_files, block=True))
        # Stale is still an answer; nothing yet is not, and says so rather
        # than rendering an empty archive as "no transmitters heard".
        return _served(params, None) or _building(cycle, max_files)

    # Scan first, read second. A directory listing is one round trip per
    # directory; an HDF5 open is several per *file*. Doing the cheap half
    # first is what lets an unchanged archive answer without opening anything.
    #
    # One pass per day, not one per product. Asking the three finders in turn
    # walked the tree three times, and on a station that writes no `par-*.h5`
    # the first of those walks visited every `chirp-*.h5` in the day to
    # discover that -- 46,436 entries to return an empty list.
    scans, matched = [], 0
    for day in _day_directories(root, max_days):
        try:
            products = io_detect.find_products(day)
        except Exception:
            continue
        for key, reader, expand, name in (
                ("par", io_detect.read_timing, None, "timing solution"),
                ("chirp", io_detect.read_detection, None, "detection"),
                ("cdetections", io_detect.read_cdetections,
                 _cdetection_rows, "consolidated detection")):
            paths = products.get(key) or []
            if paths:
                scans.append((paths, reader, expand, name))
                matched += len(paths)
                break

    # The budget, spent newest day first. A trimmed day still answers the
    # question the page asks -- 2000 files is ~12 cycles of 300 s, and
    # `min_repeats` wants 3.
    found, capped = matched, 0
    if max_files and matched > max_files:
        budget, kept_scans = max_files, []
        for paths, reader, expand, name in scans:       # newest day first
            keep = sorted(paths, key=_file_time)[len(paths) - budget:] \
                if len(paths) > budget else list(paths)
            capped += len(paths) - len(keep)
            budget -= len(keep)
            if keep:
                kept_scans.append((keep, reader, expand, name))
        scans, matched = kept_scans, found - capped

    # Fingerprinted on the file *names*, which the scan above already has for
    # free -- not on their contents, and not on a `stat` per file. Every one
    # of these names carries the second it belongs to, so a name that was
    # there last time refers to the same recording. Stat-ing 1846 files to
    # prove that would cost a round trip each on the archive this is slow on,
    # which is most of what we are trying to avoid.
    fingerprint = params + (
        tuple(sorted(str(p) for paths, *_ in scans for p in paths)),)

    with _CENSUS_LOCK:
        # The names matching is only proof that nothing changed if every file
        # behind them was read. A detector caught mid-write keeps its name
        # when it finishes, so a skipped file has to be looked at again.
        #
        # Only the skipped ones, though: this used to test "did the last
        # census skip anything at all", which one truncated file in an archive
        # of 1846 turned into a full re-read and re-group on every page load,
        # for the rest of the process's life. That is exactly the state a live
        # archive sits in -- the detector is always writing something -- so the
        # short-circuit was off precisely when it was needed. Re-stat the
        # handful that failed and trust the cache if none of them moved.
        settled = all(_MEMO.get(str(p), (None, None))[0] == _identity(p)
                      for p in _LAST.get("skipped", ()))
        if settled and _LAST.get("fingerprint") == fingerprint:
            out = dict(_LAST["census"])
            # This call's cost, not the cost of the call that filled the
            # cache. Reporting the earlier one would make the page claim it
            # had just opened 1846 files when it opened none.
            out["cost"] = dict(out["cost"], unchanged=True,
                               opened=0, cached=matched,
                               found=found, capped=capped,
                               seconds=round(time.perf_counter() - started, 2))
            return out

        records, kind, skipped = [], "none", []
        totals = {"opened": 0, "cached": 0, "unreadable": 0}
        for paths, reader, expand, name in scans:
            got, counts, could_not = _read_cached(paths, reader, expand)
            for field, n in counts.items():
                totals[field] += n
            skipped.extend(could_not)
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
            files=matched, found=found, capped=capped, budget=max_files,
            records=len(records), unchanged=False,
            seconds=round(time.perf_counter() - started, 2),
        )
        if capped:
            # Loud, because the answer is now about part of the archive and
            # the operator is the one who knows whether that is enough.
            warnings.warn(
                f"census read the newest {matched} of {found} detection "
                f"file(s) under {root}: the {max_files}-file ceiling",
                stacklevel=2)

        if not records:
            out = {"count": 0, "kind": "none", "cycle_s": cycle,
                   "emitters": [], "cost": cost}
            _LAST.update(fingerprint=fingerprint, census=out, skipped=skipped,
                         params=params, at=time.time())
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
        _LAST.update(fingerprint=fingerprint, census=out, skipped=skipped,
                         params=params, at=time.time())
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
