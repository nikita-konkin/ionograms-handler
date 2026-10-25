"""A forecast scored over a run of soundings: ``muf sim series``.

One circuit's soundings, in time order, are paired -- each with the first
sounding a lead later, within a tolerance -- and every pair is scored as
``muf sim forecast`` scores one (:mod:`muf.sim.forecast`). A pair's rows are
written the moment it is done, its line in ``pairs.jsonl`` last, so an
interrupted run loses nothing and a resumed one starts where it stopped.

Each sounding's state is observed once and kept in ``states/``, keyed by the
file, the code and the parameters it was observed with; a key that no longer
matches is observed again. The truth of a pair is its sounding's own state
when the two soundings share a range axis, as every ``.lfs`` of a circuit
does, and is observed on the forecast's axis otherwise. Whether the
receiver's clock broke the pair is read off the two soundings' own states
(:mod:`muf.sim.clock`), and goes in its line; ``muf sim summary`` sets those
pairs apart.

Every k-th sounding also runs the closed loop's level 1 on its own state
(:func:`muf.sim.check.closed_loop`): how far ``observe`` itself is off, the
limit a forecast's error is read against. Its synthetic copies are those the
forecast's ionogram was compared with, sounded once.

What cannot be scored -- a file that does not read, a state that is not
observed, no sounding a lead later, a pair that fails -- goes to
``skipped.jsonl`` with its reason, and the run goes on.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .. import geometry, loader, spectro
from ..reference import chapman
from . import check, clock, link, observe
from . import forecast as fc
from .state import ChannelState

RUN_SCHEMA = "muf.sim.forecast.run"
SKIPPED_SCHEMA = "muf.sim.forecast.skipped"
RECOVERY_SCHEMA = "muf.sim.forecast.recovery"
VERSION = 1

#: The row files a pair writes, in the order it writes them: ``pairs`` last,
#: so that a pair is done exactly when its line is in ``pairs.jsonl``.
ROWS = ("modes", "carriers", "ionogram", "recovery", "pairs")

LEAD_S = 300.0

#: How far from the lead the next sounding may be, s.
TOLERANCE_S = 30.0

#: Synthetic soundings of each forecast its ionogram is compared with.
COPIES = 4

#: Every this many soundings, the closed loop's level 1 on the state.
RECOVERY_EVERY = 12

#: Synthetic soundings that loop reads back.
RECOVERY_COPIES = 4

#: How each sounding is observed, unless told otherwise.
OBSERVE = {"method": "contour", "window": spectro.DEFAULT_WINDOW,
           "knot_mhz": observe.KNOT_MHZ, "doppler": observe.DEFAULT_DOPPLER_PRIOR}

#: The fields of a recovery error, as :class:`muf.sim.check.ModeError` has them.
RECOVERY_ERRORS = ("overlap_mhz", "delay_bias_us", "delay_rms_us",
                   "gdd_bias_us_per_mhz", "gdd_rms_us_per_mhz", "cn0_bias_db",
                   "cn0_rms_db", "f_lo_error_mhz", "f_hi_error_mhz")


@dataclass(frozen=True)
class Session:
    """One sounding of a run: its file, when it began, and its circuit."""

    path: Path
    epoch: float
    tx: str
    rx: str


def _skipped(source: str, to: str | None, stage: str, reason: str) -> dict:
    return {"schema": SKIPPED_SCHEMA, "version": VERSION, "from": source, "to": to,
            "stage": stage, "reason": reason}


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def sessions(paths: Iterable, *,
             stations: Mapping | None = None) -> tuple[list[Session], list[dict]]:
    """The soundings under ``paths``, in time order, and those whose header
    does not read."""
    files: list[Path] = []
    for item in paths:
        item = Path(item)
        files.extend(loader.find_soundings(item) if item.is_dir() else [item])
    found, skipped = [], []
    for path in sorted(set(files)):
        try:
            header = loader.read_header(path, stations=stations)
            found.append(Session(path, observe.epoch_of(header), header.tx_name,
                                 header.rx_name))
        except Exception as exc:
            skipped.append(_skipped(path.name, None, "header",
                                    f"header unreadable: {_failure(exc)}"))
    return sorted(found, key=lambda s: s.epoch), skipped


def circuit_of(found: Sequence[Session], circuit: str | None) -> list[Session]:
    """The soundings of ``circuit``, ``tx:rx``; with None, of the only one."""
    names = sorted({f"{s.tx}:{s.rx}" for s in found})
    if circuit is None:
        if len(names) > 1:
            raise ValueError(f"soundings of {len(names)} circuits "
                             f"({', '.join(names)}): choose one as tx:rx")
        return list(found)
    kept = [s for s in found if f"{s.tx}:{s.rx}" == circuit]
    if not kept:
        raise ValueError(f"no soundings of circuit {circuit}, only of "
                         f"{', '.join(names) or 'none'}")
    return kept


def pairs(found: Sequence[Session], *, lead_s: float = LEAD_S,
          tolerance_s: float = TOLERANCE_S
          ) -> tuple[list[tuple[Session, Session]], list[dict]]:
    """Each sounding with the first one ``lead_s`` later, within ``tolerance_s``.

    A sounding with none -- the next one missing, or the run's step not the
    lead -- makes no pair, and is listed with the reason.
    """
    out, skipped = [], []
    for i, a in enumerate(found):
        b = None
        for later in found[i + 1:]:
            dt = later.epoch - a.epoch
            if dt > lead_s + tolerance_s:
                break
            if abs(dt - lead_s) <= tolerance_s:
                b = later
                break
        if b is None:
            reason = f"no sounding {lead_s:g} +- {tolerance_s:g} s later"
            skipped.append(_skipped(a.path.name, None, "pair", reason))
        else:
            out.append((a, b))
    return out, skipped


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """The commit muf runs from, and a digest of any change to it not committed."""
    here = str(Path(__file__).resolve().parents[1])
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        diff = subprocess.run(["git", "-C", here, "diff", "HEAD", "--", "."],
                              capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if sha.returncode != 0:
        return "unknown"
    out = sha.stdout.strip()
    if diff.stdout:
        out += "+" + hashlib.sha1(diff.stdout).hexdigest()[:12]
    return out


def _grid(ion) -> list:
    """An ionogram's axes, enough to tell whether two are on one grid."""
    v, f = np.asarray(ion.vrange, dtype=float), np.asarray(ion.freq, dtype=float)
    return [round(float(v[0]), 6), round(float(v[-1]), 6), int(v.size),
            round(float(f[0]), 6), round(float(f[-1]), 6), int(f.size)]


def _gate(grid: Sequence) -> tuple[float, float]:
    """:func:`muf.sim.check.gate_of`, from a grid's range axis."""
    lo, hi, n = min(grid[0], grid[1]), max(grid[0], grid[1]), grid[2]
    pad = 0.25 * (hi - lo) / max(n - 1, 1)
    return lo - pad, hi + pad


def _files(cache: Path, path: Path) -> tuple[Path, Path]:
    return cache / f"{path.name}.json", cache / f"{path.name}.key.json"


def state_of(path: str | Path, cache: str | Path, params: Mapping | None = None, *,
             stations: Mapping | None = None, stations_id: str | None = None
             ) -> ChannelState:
    """The state of the sounding at ``path``, from ``cache`` when its key matches.

    The key is the file's path, size and modification time, the code's
    version (:func:`code_version`) and the observing ``params`` -- extractor,
    window, knots and Doppler prior, :data:`OBSERVE` filling in the rest --
    with ``stations_id`` naming the station table. A state observed afresh is
    cached with its key and its ionogram's grid.
    """
    path, cache = Path(path), Path(cache)
    params = {**OBSERVE, **(params or {})}
    state_file, key_file = _files(cache, path)
    stat = path.stat()
    key = {"path": str(path.resolve()), "size": stat.st_size,
           "mtime_ns": stat.st_mtime_ns, "muf": code_version(), "observe": params,
           "stations": stations_id}
    if state_file.exists() and key_file.exists():
        stored = json.loads(key_file.read_text(encoding="utf-8"))
        if stored.get("key") == key:
            return ChannelState.load(state_file)
    ion = observe.load(path, window=params["window"], stations=stations)
    state = observe.from_ionogram(ion, method=params["method"], source=path.name,
                                  doppler=params["doppler"],
                                  knot_mhz=params["knot_mhz"])
    cache.mkdir(parents=True, exist_ok=True)
    state.save(state_file)
    key_file.write_text(json.dumps({"key": key, "grid": _grid(ion)}, indent=1) + "\n",
                        encoding="utf-8")
    return state


def _cached_grid(cache: Path, path: Path) -> list:
    return json.loads(_files(cache, path)[1].read_text(encoding="utf-8"))["grid"]


def truth_of(origin: str | Path, path: str | Path, cache: str | Path,
             params: Mapping | None = None, *, stations: Mapping | None = None
             ) -> ChannelState:
    """The state of the sounding at ``path`` that a forecast from ``origin`` is
    scored against.

    Its own, from ``cache``, when the two soundings share a grid; observed on
    ``origin``'s range otherwise, so the two states are read off one span.
    Both soundings must be in ``cache`` (:func:`state_of`).
    """
    a, b, cache = Path(origin), Path(path), Path(cache)
    params = {**OBSERVE, **(params or {})}
    grid = _cached_grid(cache, a)
    if grid == _cached_grid(cache, b):
        return ChannelState.load(_files(cache, b)[0])
    ion = observe.load(b, window=params["window"], stations=stations,
                       gate_km=_gate(grid))
    return observe.from_ionogram(ion, method=params["method"], source=b.name,
                                 doppler=params["doppler"], knot_mhz=params["knot_mhz"])


def zenith_deg(state: ChannelState) -> float | None:
    """The sun's zenith angle at the circuit's control point at the state's
    time, as the ``harmonic`` baseline reads it; None with no coordinates."""
    c = state.circuit
    if None in (c.tx_lat, c.tx_lon, c.rx_lat, c.rx_lon):
        return None
    points = geometry.control_points(geometry.Point(c.tx_lat, c.tx_lon),
                                     geometry.Point(c.rx_lat, c.rx_lon))
    if not points:
        return None
    cos = chapman.solar_zenith_cos(state.datetime, points[0])
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _recovery_rows(report: dict, state: ChannelState, to: str) -> list[dict]:
    muf = _finite(state.metrics.get("muf_mhz"))
    out = []
    for run in report["runs"]:
        got = _finite(run["muf_mhz"])
        for e in run["modes"]:
            out.append({"schema": RECOVERY_SCHEMA, "version": VERSION,
                        "from": state.source, "to": to, "seed": run["seed"],
                        "mode": e["name"], "matched": e["matched"],
                        **{k: _finite(e[k]) for k in RECOVERY_ERRORS},
                        "muf_error_mhz": (None if got is None or muf is None
                                          else got - muf)})
    return out


def _empty() -> dict[str, list]:
    return {name: [] for name in (*ROWS, "skipped")}


def _observe_job(job: dict) -> dict | None:
    """One sounding's state into the cache; a skipped row if it fails."""
    path = Path(job["path"])
    try:
        state_of(path, job["cache"], job["observe"], stations=job["stations"],
                 stations_id=job["stations_id"])
    except Exception as exc:
        return _skipped(path.name, None, "state", f"not observed: {_failure(exc)}")
    return None


def _score_job(job: dict) -> dict[str, list]:
    """One pair scored: its rows by file, or a skipped row if it fails."""
    a, b, cache = Path(job["a"]), Path(job["b"]), Path(job["cache"])
    params, stations = job["observe"], job["stations"]
    rows = _empty()
    try:
        start = time.perf_counter()
        origin = ChannelState.load(_files(cache, a)[0])
        truth = truth_of(a, b, cache, params, stations=stations)
        for state, whose in ((origin, "forecast's"), (truth, "truth's")):
            if not state.modes:
                rows["skipped"].append(_skipped(a.name, b.name, "pair",
                                                f"no trace in the {whose} sounding"))
                return rows
        predicted = fc.persist(origin, job["lead_s"])
        score = fc.score(predicted, truth, carriers=job["carriers"],
                         bandwidth_hz=job["bandwidth_hz"], snr_min_db=job["snr_min_db"])
        copies, recovery = job["copies"], job["recovery"]
        grid = (fc.on_one_grid(a, b, window=params["window"], stations=stations)
                if copies else None)
        made = ([] if grid is None
                else fc.sounded(predicted, max(copies, recovery),
                                gate_km=check.gate_of(grid[1])))
        if copies:
            rows["ionogram"].append(fc.image_row(predicted, truth, grid, made[:copies]))
        if recovery:
            report = check.closed_loop(origin, realisations=recovery,
                                       method=params["method"],
                                       synthetic=made[:recovery])
            rows["recovery"] = _recovery_rows(report, origin, b.name)
        score.pair.update(method=params["method"], zenith_deg=zenith_deg(predicted),
                          clock=clock.verdict(origin, ChannelState.load(
                              _files(cache, b)[0])),
                          seconds=round(time.perf_counter() - start, 2))
        rows.update(modes=score.modes, carriers=score.carriers, pairs=[score.pair])
    except Exception as exc:
        rows = _empty()
        rows["skipped"].append(_skipped(a.name, b.name, "pair", _failure(exc)))
    return rows


def _map(fn: Callable, jobs: list, workers: int):
    """``fn`` over ``jobs``, results as they come: in process, or on a pool.

    On a failure, here or where the results go, the jobs not yet begun are
    dropped rather than run for nothing.
    """
    if workers <= 1 or len(jobs) <= 1:
        yield from (fn(job) for job in jobs)
        return
    from ..pipeline import pool_context

    with ProcessPoolExecutor(max_workers=workers, mp_context=pool_context()) as pool:
        try:
            for done in as_completed([pool.submit(fn, job) for job in jobs]):
                yield done.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def read_rows(path: Path) -> list[dict]:
    """The rows of a JSON Lines file; none if it is not there."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class _Writer:
    """Appends a run's rows, a pair's line in ``pairs.jsonl`` after the rest,
    and never the same skipped row twice."""

    def __init__(self, out: Path, skipped: Iterable[dict]) -> None:
        self.out = out
        self.seen = {self._key(r) for r in skipped}
        self.counts = dict.fromkeys((*ROWS, "skipped"), 0)

    @staticmethod
    def _key(row: dict) -> tuple:
        return row["from"], row["to"], row["stage"], row["reason"]

    def _append(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with (self.out / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.counts[name] += len(rows)

    def skip(self, rows: Iterable[dict]) -> None:
        fresh = [r for r in rows if self._key(r) not in self.seen]
        self.seen.update(self._key(r) for r in fresh)
        self._append("skipped", fresh)

    def write(self, rows: dict[str, list]) -> None:
        self.skip(rows["skipped"])
        for name in ROWS:
            self._append(name, rows[name])


def _resume(out: Path) -> tuple[set, list[dict]]:
    """What a run in ``out`` finished, with its rows of unfinished pairs dropped."""
    done = {(r["from"], r["to"]) for r in read_rows(out / "pairs.jsonl")}
    for name in ROWS[:-1]:
        path = out / f"{name}.jsonl"
        rows = read_rows(path)
        kept = [r for r in rows if (r["from"], r["to"]) in done]
        if len(kept) != len(rows):
            path.write_text("".join(json.dumps(r, allow_nan=False) + "\n"
                                    for r in kept), encoding="utf-8")
    skipped = read_rows(out / "skipped.jsonl")
    done |= {(r["from"], r["to"]) for r in skipped if r["to"] is not None}
    return done, skipped


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(out: Path, params: dict, chosen: Sequence[Session], resume: bool) -> dict:
    """``run.json``: checked against a run resumed, begun for a new one."""
    path = out / "run.json"
    if resume:
        if not path.exists():
            raise ValueError(f"{out} holds no run to resume")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["params"] != params:
            differ = sorted(k for k in params if record["params"].get(k) != params[k])
            raise ValueError(f"the parameters differ from the run in {out}: "
                             f"{', '.join(differ)}")
        return record
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty: resume the run in it, or choose "
                         f"another folder")
    out.mkdir(parents=True, exist_ok=True)
    suffixes = sorted({s.path.suffix for s in chosen})
    return {"schema": RUN_SCHEMA, "version": VERSION, "params": params,
            "muf": code_version(), "python": platform.python_version(),
            "numpy": np.__version__,
            "files": [{"path": str(s.path), "size": s.path.stat().st_size}
                      for s in chosen],
            "formats": {x: sum(1 for s in chosen if s.path.suffix == x)
                        for x in suffixes},
            "runs": []}


def _save(out: Path, record: dict) -> None:
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n",
                                  encoding="utf-8")


def run_series(paths: Iterable, out: str | Path, *, circuit: str | None = None,
               lead_s: float = LEAD_S, tolerance_s: float = TOLERANCE_S,
               copies: int = COPIES, recovery_every: int = RECOVERY_EVERY,
               recovery_copies: int = RECOVERY_COPIES,
               carriers: Sequence[float] | None = None,
               bandwidth_hz: float = link.BANDWIDTH_HZ,
               snr_min_db: float = link.SNR_MIN_DB, method: str = OBSERVE["method"],
               window: int = OBSERVE["window"], knot_mhz: float = OBSERVE["knot_mhz"],
               doppler: str | float = OBSERVE["doppler"],
               stations: Mapping | None = None, stations_file: str | None = None,
               jobs: int = 1, resume: bool = False,
               progress: Callable[[int, int], None] | None = None) -> Path:
    """Score persistence over the soundings under ``paths``; the rows go to ``out``.

    ``stations_file`` names the table ``stations`` was read from, for the
    record and the state cache's key. ``jobs`` processes score pairs side by
    side. ``resume`` carries on the run already in ``out``, with the same
    parameters, scoring only what it did not finish. ``progress`` hears
    (pairs done, pairs to do) as each one is.
    """
    out = Path(out)
    begun = time.perf_counter()
    found, unreadable = sessions(paths, stations=stations)
    chosen = circuit_of(found, circuit)
    observing = {"method": method, "window": int(window), "knot_mhz": float(knot_mhz),
                 "doppler": doppler}
    params = {"circuit": circuit, "lead_s": float(lead_s),
              "tolerance_s": float(tolerance_s), "copies": int(copies),
              "recovery_every": int(recovery_every),
              "recovery_copies": int(recovery_copies),
              "carriers": None if carriers is None else [float(f) for f in carriers],
              "bandwidth_hz": float(bandwidth_hz), "snr_min_db": float(snr_min_db),
              "observe": observing, "stations": stations_file}
    record = _record(out, params, chosen, resume)
    done, skipped = _resume(out) if resume else (set(), [])
    record["runs"].append({"started": _utc_now(), "muf": code_version(), "jobs": jobs})
    _save(out, record)
    writer = _Writer(out, skipped)
    writer.skip(unreadable)
    cache = out / "states"
    common = {"cache": str(cache), "observe": observing, "stations": stations,
              "stations_id": stations_file}
    failed = {r["from"] for r in skipped if r["stage"] == "state"}
    states = [{**common, "path": str(s.path)} for s in chosen
              if s.path.name not in failed]
    for row in _map(_observe_job, states, jobs):
        if row is not None:
            writer.skip([row])
            failed.add(row["from"])
    found_pairs, unpaired = pairs(chosen, lead_s=lead_s, tolerance_s=tolerance_s)
    writer.skip(unpaired)
    order = {s.path: k for k, s in enumerate(chosen)}
    todo = []
    for a, b in found_pairs:
        if (a.path.name, b.path.name) in done:
            continue
        lost = [s.path.name for s in (a, b) if s.path.name in failed]
        if lost:
            writer.skip([_skipped(a.path.name, b.path.name, "pair",
                                  f"no state of {', '.join(lost)}")])
            continue
        due = recovery_every > 0 and order[a.path] % recovery_every == 0
        todo.append({**common, "a": str(a.path), "b": str(b.path),
                     "lead_s": float(lead_s), "copies": int(copies),
                     "recovery": int(recovery_copies) if due else 0,
                     "carriers": params["carriers"],
                     "bandwidth_hz": float(bandwidth_hz),
                     "snr_min_db": float(snr_min_db)})
    for k, rows in enumerate(_map(_score_job, todo, jobs), 1):
        writer.write(rows)
        if progress is not None:
            progress(k, len(todo))
    record["runs"][-1].update(finished=_utc_now(),
                              seconds=round(time.perf_counter() - begun, 1),
                              written=dict(writer.counts), cpus=os.cpu_count())
    _save(out, record)
    return out
