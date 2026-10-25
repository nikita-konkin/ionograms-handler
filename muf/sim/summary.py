"""A run's rows summed up: ``muf sim summary``.

Every figure is recomputed from the rows a run wrote (:mod:`muf.sim.series`),
never from the soundings, and carries a 95 % interval from a **moving block
bootstrap**. Neighbouring five-minute errors are far from independent -- a
MUF climbing through a sunrise is wrong the same way for an hour -- and
resampling pairs one at a time would narrow the interval several times over.
Blocks of :data:`BLOCK` pairs are drawn within the run's unbroken stretches,
never across a gap; :data:`BLOCKS` show how much the length matters. Every
figure of a run is read off the same resamples, drawn from rng 0, so the same
rows always give the same summary.

The figures are the spec's: n, bias, MAE, RMSE, the median and the 90th and
95th percentiles of |e|, and the robust spread 1.4826 MAD; none from under
:data:`MIN_PAIRS` pairs, which says so instead. A mode pair weighs as the band
the two modes share. Percentiles get intervals over the whole run only --
they are the costly ones -- and the robust spread none.

The run is cut by the hour and the day of the valid time, by the sun at the
circuit's control point (day, the transition of zenith angles 85-100 deg,
night), by the lead actually achieved, and the carriers by their frequency
over the truth's MUF.

The pairs the receiver's clock broke (:mod:`muf.sim.clock`) are set apart,
their rows with them, and counted: their errors are the clock's.

``q``, the closed loop's RMS error over the forecast's, says whether the
forecast's error is the channel's change at all: with no change it is
1/sqrt 2, the forecast's error then two measurements' noise. Over
:data:`UNRESOLVED_Q` the change is not resolved.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import clock, forecast, link
from .series import RECOVERY_ERRORS, read_rows

SUMMARY_SCHEMA = "muf.sim.forecast.summary"
VERSION = 1

#: Pairs to a block: an hour of five-minute pairs.
BLOCK = 12

#: Block lengths the main intervals are also drawn with.
BLOCKS = (6, 12, 24)

REPS = 4000

#: Pairs a figure needs, as ``MIN_COMPARISON_PAIRS`` in
#: ``services/prediction/scoring.py``.
MIN_PAIRS = 30

LEVEL = 0.95

QUANTILES = {"median": 0.5, "p90": 0.9, "p95": 0.95}

#: MAD to standard deviation, for normal errors.
ROBUST = 1.4826

#: q over which a characteristic's change is not resolved.
UNRESOLVED_Q = 0.6

#: Edges of the carriers' frequency over the truth's MUF.
FMUF_EDGES = (0.7, 0.9, 0.97)

#: Solar zenith angles, deg, between which a pair is in the transition.
SUN_DEG = (85.0, 100.0)

#: Seconds the achieved leads are cut by.
LEAD_BIN_S = 10.0

CLASSES = ("all", "main", "high", "other")

#: A mode's top: censored at the band's edge as a MUF is.
TOP = "f_hi_error_mhz"

#: The cuts :func:`summarise` makes.
BY = ("hour", "day", "sun", "lead", "fmuf")

#: Cells of the resampled weights a percentile interval holds in memory at once.
CHUNK = 4_000_000

#: Figures whose intervals are also drawn with each of :data:`BLOCKS`.
HEADLINE = (("muf_mhz", "mae"), ("modes.all.delay_rms_us", "rmse"),
            ("modes.all.cn0_rms_db", "rmse"),
            ("carriers.figures.first_delay_us", "mae"),
            ("carriers.figures.cn0_dbhz", "mae"))


# --- the block bootstrap -------------------------------------------------------

def segments(times: Sequence[float], *, gap_s: float) -> list[tuple[int, int]]:
    """Index ranges of the unbroken stretches of ``times``, broken where the
    step exceeds ``gap_s``."""
    out, lo = [], 0
    for k in range(1, len(times)):
        if times[k] - times[k - 1] > gap_s:
            out.append((lo, k))
            lo = k
    if times:
        out.append((lo, len(times)))
    return out


def resamples(stretches: Sequence[tuple[int, int]], *, block: int, reps: int,
              rng: np.random.Generator) -> np.ndarray:
    """How many times each pair is drawn, ``reps`` by pairs: a moving block
    bootstrap within ``stretches``.

    Every block of ``block`` pairs inside a stretch is equally likely, and a
    stretch shorter than that is one block. Blocks are drawn until they hold
    as many pairs as the run, the last one cut short.
    """
    n = stretches[-1][1] if stretches else 0
    starts, lengths = [], []
    for lo, hi in stretches:
        if hi - lo <= block:
            starts.append(lo)
            lengths.append(hi - lo)
        else:
            starts.extend(range(lo, hi - block + 1))
            lengths.extend([block] * (hi - lo - block + 1))
    starts_a, lengths_a = np.asarray(starts), np.asarray(lengths)
    need = math.ceil(n / lengths_a.min()) + 1
    out = np.zeros((reps, n), dtype=np.int32)
    for r in range(reps):
        pick = rng.integers(0, starts_a.size, size=need)
        lens = lengths_a[pick]
        stop = int(np.searchsorted(np.cumsum(lens), n))
        pick, lens = pick[:stop + 1], lens[:stop + 1].copy()
        lens[-1] -= int(lens.sum()) - n
        before = np.cumsum(lens) - lens
        index = (np.arange(n) - np.repeat(before, lens)
                 + np.repeat(starts_a[pick], lens))
        out[r] = np.bincount(index, minlength=n)
    return out


def _interval(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    tail = 50.0 * (1.0 - LEVEL)
    return [float(np.percentile(values, tail)),
            float(np.percentile(values, 100 - tail))]


def _quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(w[order])
    return float(x[order][min(int(np.searchsorted(cum, q * cum[-1])), x.size - 1)])


def _quantile_intervals(a: np.ndarray, w: np.ndarray, pair: np.ndarray,
                        drawn: np.ndarray) -> dict:
    """Intervals of :data:`QUANTILES` of ``a``, each resample weighing a row by
    its weight times its pair's draws."""
    order = np.argsort(a, kind="stable")
    a, w, pair = a[order], w[order], pair[order]
    chunk = max(1, CHUNK // a.size)
    found = {name: [] for name in QUANTILES}
    for lo in range(0, drawn.shape[0], chunk):
        cum = np.cumsum(drawn[lo:lo + chunk][:, pair] * w, axis=1)
        total = cum[:, -1:]
        for name, q in QUANTILES.items():
            k = np.minimum((cum < q * total).sum(axis=1), a.size - 1)
            found[name].append(np.where(total[:, 0] > 0, a[k], np.nan))
    return {name: _interval(np.concatenate(v)) for name, v in found.items()}


def metrics(e, w, pair, drawn: np.ndarray, *, minimum: int = MIN_PAIRS,
            quantiles: bool = True) -> dict:
    """The spec's figures of errors ``e``, weighed by ``w``, with intervals.

    ``pair`` is each error's pair, an index into the columns of ``drawn``
    (:func:`resamples`). Errors that are not finite are left out.
    """
    e, w, pair = (np.asarray(x, dtype=float) for x in (e, w, pair))
    keep = np.isfinite(e) & (w > 0)
    e, w, pair = e[keep], w[keep], pair[keep].astype(int)
    out = {"n": int(e.size), "pairs": int(np.unique(pair).size)}
    if out["pairs"] < minimum:
        out["reason"] = f"{out['pairs']} pairs, under {minimum}"
        return out
    a = np.abs(e)
    total = float(w.sum())
    out.update(bias=float(np.sum(w * e) / total), mae=float(np.sum(w * a) / total),
               rmse=math.sqrt(float(np.sum(w * e * e) / total)))
    out.update({name: _quantile(a, w, q) for name, q in QUANTILES.items()})
    centre = _quantile(e, w, 0.5)
    out["robust_sd"] = ROBUST * _quantile(np.abs(e - centre), w, 0.5)
    sums = np.stack([np.bincount(pair, weights=x, minlength=drawn.shape[1])
                     for x in (w, w * e, w * a, w * e * e)], axis=1)
    got = drawn @ sums
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = {"bias": _interval(got[:, 1] / got[:, 0]),
              "mae": _interval(got[:, 2] / got[:, 0]),
              "rmse": _interval(np.sqrt(got[:, 3] / got[:, 0]))}
    if quantiles:
        ci.update(_quantile_intervals(a, w, pair, drawn))
    out["ci"] = ci
    return out


def _events(outcomes: Iterable[tuple[int, bool, bool]], drawn: np.ndarray) -> dict:
    """A 2x2 table over ``(pair, forecast, truth)`` outcomes, its scores
    (:func:`muf.sim.forecast.contingency`) and their intervals."""
    outcomes = list(outcomes)
    table = forecast.contingency((f, t) for _, f, t in outcomes)
    counts = np.zeros((drawn.shape[1], 4))
    for pair, said, had in outcomes:
        counts[pair, (0 if had else 1) if said else (2 if had else 3)] += 1
    got = drawn @ counts
    a, b, c, d = got.T
    with np.errstate(invalid="ignore", divide="ignore"):
        table["ci"] = {"pod": _interval(a / (a + c)), "far": _interval(b / (a + b)),
                       "csi": _interval(a / (a + b + c)),
                       "base_rate": _interval((a + c) / (a + b + c + d))}
    return table


# --- the rows -------------------------------------------------------------------

def _epoch(utc: str) -> float:
    return _dt.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.timezone.utc).timestamp()


def _content(row: dict) -> str:
    return json.dumps(row, sort_keys=True)


class _Run:
    """A run's rows, each tied to its pair's place in time order.

    Rows are put in one order whatever order the workers wrote them in, so
    that every sum adds the same numbers the same way and two runs of the
    same pairs give the same summary to the last bit. A clock-suspect pair
    has no place, and its rows go with it.
    """

    def __init__(self, folder: Path) -> None:
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        self.params = record["params"]
        rows = read_rows(folder / "pairs.jsonl")
        self.pairs = sorted((r for r in rows if not clock.suspect(r)),
                            key=lambda r: (r["issued_at"], r["from"]))
        self.set_apart = len(rows) - len(self.pairs)
        index = {(r["from"], r["to"]): k for k, r in enumerate(self.pairs)}

        def placed(name: str) -> list[tuple[int, dict]]:
            return sorted(((index[(r["from"], r["to"])], r)
                           for r in read_rows(folder / f"{name}.jsonl")
                           if (r["from"], r["to"]) in index),
                          key=lambda item: (item[0], _content(item[1])))

        self.modes = placed("modes")
        self.carriers = placed("carriers")
        self.ionogram = [(k, r) for k, r in placed("ionogram") if "skipped" not in r]
        self.recovery = sorted(read_rows(folder / "recovery.jsonl"), key=_content)
        self.skipped = sorted(read_rows(folder / "skipped.jsonl"), key=_content)
        self.times = [_epoch(r["issued_at"]) for r in self.pairs]


def _bounded(run: _Run, key: str, chosen: np.ndarray, drawn: np.ndarray,
             quantiles: bool) -> dict:
    """A MUF or LOF: the figures over pairs censored on neither side, and the
    censored ones counted, a censored truth's one-sided error averaged."""
    clean, bounds, truth, made = [], [], 0, 0
    for k in np.flatnonzero(chosen):
        entry = run.pairs[k][key]
        if entry["error"] is None:
            continue
        truth += entry["censored_truth"]
        made += entry["censored_forecast"]
        if entry["censored_truth"] and entry["error_bound"] is not None:
            bounds.append(entry["error_bound"])
        if not (entry["censored_truth"] or entry["censored_forecast"]):
            clean.append((k, entry["error"]))
    out = metrics([e for _, e in clean], np.ones(len(clean)), [k for k, _ in clean],
                  drawn, quantiles=quantiles)
    out["censored"] = {"truth": truth, "forecast": made,
                       "mae_bound": float(np.mean(bounds)) if bounds else None}
    return out


def _mode_counts(run: _Run, chosen: np.ndarray, drawn: np.ndarray,
                 quantiles: bool) -> dict:
    picked = [(k, run.pairs[k]["modes"]) for k in np.flatnonzero(chosen)]
    transitions = Counter(f"{m['forecast']}->{m['truth']}" for _, m in picked)
    hits = sum(m["matched"] for _, m in picked)
    misses = sum(m["missed"] for _, m in picked)
    false = sum(m["false"] for _, m in picked)
    return {"transitions": dict(sorted(transitions.items())),
            "count_error": metrics([m["forecast"] - m["truth"] for _, m in picked],
                                   np.ones(len(picked)), [k for k, _ in picked],
                                   drawn, quantiles=quantiles),
            "hits": hits, "misses": misses, "false_alarms": false,
            "pod": hits / (hits + misses) if hits + misses else None,
            "far": false / (hits + false) if hits + false else None,
            "csi": hits / (hits + misses + false) if hits + misses + false else None,
            "excluded_prior_fields": sum(run.pairs[k]["excluded_prior_fields"]
                                         for k, _ in picked)}


def _modes(run: _Run, chosen: np.ndarray, drawn: np.ndarray, quantiles: bool) -> dict:
    """The mode pairs' errors by class, a mode top censored on either side
    counted apart from its error, as a MUF's is."""
    out = {}
    for cls in CLASSES:
        rows = [(k, r) for k, r in run.modes if chosen[k] and r["status"] == "matched"
                and cls in ("all", r["class"])]
        out[cls] = {}
        for name in forecast.MODE_ERRORS:
            kept = [(k, r) for k, r in rows if name != TOP or not
                    (r["f_hi_censored_truth"] or r["f_hi_censored_forecast"])]
            out[cls][name] = metrics([r[name] if r[name] is not None else np.nan
                                      for _, r in kept],
                                     [r["overlap_mhz"] or 0.0 for _, r in kept],
                                     [k for k, _ in kept], drawn, quantiles=quantiles)
        out[cls][TOP]["censored"] = len(rows) - out[cls][TOP]["n"]
    return out


def _carriers(rows: Sequence[tuple[int, dict]], drawn: np.ndarray,
              quantiles: bool) -> dict:
    figures = {}
    for name in link.FIGURES:
        scored = [(k, r["figures"][name]["error"]) for k, r in rows
                  if r["figures"][name]["error"] is not None]
        figures[name] = metrics([e for _, e in scored], np.ones(len(scored)),
                                [k for k, _ in scored], drawn, quantiles=quantiles)
        figures[name]["excluded"] = sum(name in r["excluded"] for _, r in rows)
    return {"rows": len(rows), "figures": figures,
            "events": {event: _events(((k, r["events"][event]["forecast"],
                                        r["events"][event]["truth"]) for k, r in rows),
                                      drawn)
                       for event in forecast.EVENTS}}


def _z_mean(pairs: list[int], r: list[float], drawn: np.ndarray) -> dict:
    """A mean correlation, averaged in Fisher's z and its interval with it."""
    m = metrics(np.arctanh(np.clip(r, -0.999999, 0.999999)), np.ones(len(r)), pairs,
                drawn, quantiles=False)
    if "reason" in m:
        return {"pairs": m["pairs"], "reason": m["reason"]}
    lo, hi = m["ci"]["bias"]
    return {"pairs": m["pairs"], "mean": math.tanh(m["bias"]),
            "ci": [math.tanh(lo), math.tanh(hi)]}


def _ionogram(run: _Run, chosen: np.ndarray, drawn: np.ndarray) -> dict:
    rows = [(k, r) for k, r in run.ionogram if chosen[k]]
    out = {"pairs": len(rows)}
    if not rows:
        return out
    for key in ("rows", "smoothed"):
        entry = {}
        for kind in ("ion", "state", "expected", "ceiling"):
            have = [(k, r[key][kind]) for k, r in rows if r[key].get(kind) is not None]
            entry[kind] = _z_mean([k for k, _ in have], [v for _, v in have], drawn)
        for kind in ("dz_ion", "dz_ceiling"):
            have = [(k, r[key][kind]) for k, r in rows if r[key].get(kind) is not None]
            entry[kind] = metrics([v for _, v in have], np.ones(len(have)),
                                  [k for k, _ in have], drawn, quantiles=False)
        ahead = [(k, float(r[key]["dz_ion"] >= 0.0)) for k, r in rows
                 if r[key].get("dz_ion") is not None]
        share = metrics([v for _, v in ahead], np.ones(len(ahead)),
                        [k for k, _ in ahead], drawn, quantiles=False)
        entry["state_at_least_ion"] = {k: share[k] for k in ("pairs", "bias", "ci",
                                                              "reason") if k in share}
        judged = [r[key]["inside"] for _, r in rows if "inside" in r[key]]
        entry["outside"] = (sum(not x for x in judged) / len(judged) if judged
                            else None)
        out[key] = entry
    levels = [(k, r["level_db"]) for k, r in rows if "level_db" in r]
    for kind in ("bias", "rms"):
        out[f"level_{kind}_db"] = metrics([v[kind] for _, v in levels],
                                          np.ones(len(levels)),
                                          [k for k, _ in levels], drawn,
                                          quantiles=False)
    return out


def _section(run: _Run, chosen: np.ndarray, drawn: np.ndarray, *,
             quantiles: bool = False) -> dict:
    return {"pairs": int(chosen.sum()),
            "muf_mhz": _bounded(run, "muf_mhz", chosen, drawn, quantiles),
            "mode_muf_mhz": _bounded(run, "mode_muf_mhz", chosen, drawn, quantiles),
            "lof_mhz": _bounded(run, "lof_mhz", chosen, drawn, quantiles),
            "mode_counts": _mode_counts(run, chosen, drawn, quantiles),
            "modes": _modes(run, chosen, drawn, quantiles),
            "carriers": _carriers([(k, r) for k, r in run.carriers if chosen[k]],
                                  drawn, quantiles),
            "carriers_outside_band": sum(run.pairs[k]["carriers"]["outside_band"]
                                         for k in np.flatnonzero(chosen)),
            "ionogram": _ionogram(run, chosen, drawn)}


def _sun(zenith: float | None) -> str:
    if zenith is None:
        return "unknown"
    if zenith < SUN_DEG[0]:
        return "day"
    return "transition" if zenith <= SUN_DEG[1] else "night"


def _fmuf(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    lo, mid, hi = FMUF_EDGES
    if ratio < lo:
        return f"<{lo:g}"
    if ratio < mid:
        return f"{lo:g}-{mid:g}"
    return f"{mid:g}-{hi:g}" if ratio < hi else f">={hi:g}"


def _q(run: _Run, whole: dict) -> dict:
    """Each error the closed loop also measures: its RMS there over the
    forecast's RMSE, the same rows weighed the same way."""
    out = {}
    for name in [*RECOVERY_ERRORS[1:], "muf_error_mhz"]:
        rec = [(r.get(name), r.get("overlap_mhz") or 1.0) for r in run.recovery
               if r.get(name) is not None]
        if not rec:
            continue
        x, w = np.array([v for v, _ in rec]), np.array([v for _, v in rec])
        if name == "muf_error_mhz":
            key, rmse = "muf_mhz", whole["muf_mhz"].get("rmse")
        else:
            key, rmse = name, whole["modes"]["all"][name].get("rmse")
        if not rmse:
            continue
        q = math.sqrt(float(np.sum(w * x * x) / w.sum())) / rmse
        out[key] = {"q": q, "recovery_rows": len(rec), "unresolved": q > UNRESOLVED_Q}
    return out


def _dig(section: dict, path: str) -> dict:
    for key in path.split("."):
        section = section[key]
    return section


def summarise(folder: str | Path, *, by: Sequence[str] = BY, block: int = BLOCK,
              reps: int = REPS) -> dict:
    """The run in ``folder`` summed up: ``summary.json``'s content.

    ``by`` names the cuts, of :data:`BY` (``mode`` is accepted and always
    made: every section sums the mode classes apart). ``set_apart`` counts
    the clock-suspect pairs left out.
    """
    unknown = set(by) - set(BY) - {"mode"}
    if unknown:
        raise ValueError(f"no cut by {', '.join(sorted(unknown))}; there are "
                         f"{', '.join((*BY, 'mode'))}")
    run = _Run(Path(folder))
    params = run.params
    gap = float(params.get("lead_s", 300.0)) + float(params.get("tolerance_s", 30.0))
    stretches = segments(run.times, gap_s=gap)
    draws = {b: resamples(stretches, block=b, reps=reps, rng=np.random.default_rng(0))
             .astype(np.float64) for b in sorted({*BLOCKS, block})} if run.pairs else {}
    out = {"schema": SUMMARY_SCHEMA, "version": VERSION, "pairs": len(run.pairs),
           "set_apart": run.set_apart, "stretches": len(stretches), "block": block,
           "reps": reps, "level": LEVEL, "min_pairs": MIN_PAIRS, "params": params}
    if not run.pairs:
        return {**out, "reason": "no pairs scored"}
    drawn = draws[block]
    everything = np.ones(len(run.pairs), dtype=bool)
    out["all"] = _section(run, everything, drawn, quantiles=True)
    out["q"] = _q(run, out["all"])
    out["sensitivity"] = {}
    for path, figure in HEADLINE:
        name = path.split(".")[-1]
        rows = ([(k, r["muf_mhz"]["error"]) for k, r in enumerate(run.pairs)
                 if r["muf_mhz"]["error"] is not None
                 and not r["muf_mhz"]["censored_truth"]
                 and not r["muf_mhz"]["censored_forecast"]] if path == "muf_mhz" else
                [(k, r[name]) for k, r in run.modes if r["status"] == "matched"
                 and r[name] is not None] if path.startswith("modes") else
                [(k, r["figures"][name]["error"]) for k, r in run.carriers
                 if r["figures"][name]["error"] is not None])
        weights = ([r["overlap_mhz"] for k, r in run.modes if r["status"] == "matched"
                    and r[name] is not None] if path.startswith("modes")
                   else np.ones(len(rows)))
        out["sensitivity"][f"{path}.{figure}"] = {
            str(b): metrics([e for _, e in rows], weights, [k for k, _ in rows], d,
                            quantiles=False).get("ci", {}).get(figure)
            for b, d in draws.items()}
    keys = {"hour": [r["valid_at"][11:13] for r in run.pairs],
            "day": [r["valid_at"][:10] for r in run.pairs],
            "sun": [_sun(r.get("zenith_deg")) for r in run.pairs],
            "lead": [f"{LEAD_BIN_S * round(r['lead_actual_s'] / LEAD_BIN_S):.0f}"
                     for r in run.pairs]}
    out["by"] = {}
    for cut in (c for c in BY if c in by and c in keys):
        labels = np.array(keys[cut])
        out["by"][cut] = {label: _section(run, labels == label, drawn)
                          for label in sorted(set(keys[cut]))}
    if "fmuf" in by:
        out["by"]["fmuf"] = {
            label: _carriers([(k, r) for k, r in run.carriers
                              if _fmuf(r["f_over_muf"]) == label], drawn, False)
            for label in sorted({_fmuf(r["f_over_muf"]) for _, r in run.carriers})}
    out["skipped"] = {"by_stage": dict(Counter(r["stage"] for r in run.skipped)),
                      "reasons": dict(Counter(r["reason"] for r in run.skipped)
                                      .most_common(20))}
    return out


# --- for a terminal -------------------------------------------------------------

def _figure(m: dict, key: str, unit: str, digits: str = ".2f") -> str:
    if key not in m:
        return f"- ({m.get('reason', 'none')})"
    ci = (m.get("ci") or {}).get(key)
    span = f" [{format(ci[0], digits)}, {format(ci[1], digits)}]" if ci else ""
    return f"{format(m[key], digits)}{span} {unit}"


def _score(t: dict) -> str:
    def one(key):
        ci = t["ci"].get(key)
        span = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        return "-" if t[key] is None else f"{t[key]:.2f}{span}"

    return f"POD {one('pod')}  FAR {one('far')}  CSI {one('csi')}  " \
           f"(in the truth {one('base_rate')})"


def format_summary(s: dict) -> str:
    """The headline figures, a few lines; the rest is ``summary.json``."""
    head = (f"{s['pairs']} pairs in {s['stretches']} unbroken stretches; "
            f"{s['level']:.0%} intervals from {s['reps']} resamples in blocks of "
            f"{s['block']} pairs"
            + (f"; {s['set_apart']} clock-suspect pairs set apart"
               if s.get("set_apart") else ""))
    if "all" not in s:
        return head + f"\n  {s.get('reason', '')}"
    a = s["all"]
    lines = [head]
    for key, label in (("muf_mhz", "MUF"), ("lof_mhz", "LOF")):
        m = a[key]
        lines.append(f"  {label:<12} MAE {_figure(m, 'mae', 'MHz', '.3f')}, RMSE "
                     f"{_figure(m, 'rmse', 'MHz', '.3f')}, bias "
                     f"{_figure(m, 'bias', 'MHz', '+.3f')}; censored: truth "
                     f"{m['censored']['truth']}, forecast {m['censored']['forecast']}")
    modes = a["modes"]["all"]
    for key, label, unit in (("delay_rms_us", "delay", "us"),
                             ("cn0_rms_db", "C/N0", "dB"),
                             ("f_hi_error_mhz", "mode top", "MHz")):
        q = s["q"].get(key)
        tail = (f"; q {q['q']:.2f}" + (" -- change not resolved" if q["unresolved"]
                                       else "")) if q else ""
        lines.append(f"  {label:<12} RMSE {_figure(modes[key], 'rmse', unit)}, median "
                     f"|e| {_figure(modes[key], 'median', unit)}{tail}")
    mc = a["mode_counts"]
    lines.append(f"  modes        matched {mc['hits']}, missed {mc['misses']}, false "
                 f"{mc['false_alarms']}")
    c = a["carriers"]
    lines.append(f"  carriers     a mode seen: {_score(c['events']['mode'])}")
    lines.append(f"               usable: {_score(c['events']['usable'])}")
    for key, label, unit in (("first_delay_us", "first delay", "us"),
                             ("delay_spread_us", "delay spread", "us"),
                             ("cn0_dbhz", "C/N0", "dB")):
        lines.append(f"               {label} MAE "
                     f"{_figure(c['figures'][key], 'mae', unit)}")
    ion = a["ionogram"]
    for key in ("rows", "smoothed"):
        if key not in ion:
            continue
        e = ion[key]
        parts = [f"{kind} {e[kind]['mean']:.3f}" for kind in
                 ("ion", "state", "expected", "ceiling") if "mean" in e[kind]]
        lines.append(f"  ionogram     {key}: " + ", ".join(parts))
    return "\n".join(lines)
