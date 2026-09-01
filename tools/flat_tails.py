"""How often is the reported MUF sitting on a flat line rather than a trace?

Sounding `020` of the 2026-09-01 adjudication was rejected by eye, and the
reason turned out to be a feature at a constant virtual range, flat to within
one 2 km range bin across five megahertz, sitting above the real trace. Nothing
in the pipeline forbids that: the 150 km/MHz limit in `muf.pick` and in
`muf.extractors.viterbi` bounds how *fast* range may change with frequency, and
**zero slope satisfies every slope constraint there is**. Both estimators are
free to walk along such a line to its end and report that as the MUF.

A real oblique trace cannot do this. Group range exceeds the ground distance by
an amount that grows as the wave penetrates higher, and it rises steeply
approaching the junction frequency -- the nose is the defining feature of the
MUF. A delay that does not change at all over megahertz is not ionospheric
propagation; it is something arriving at a fixed delay.

So this asks three questions of the archive, in order, because the third only
matters if the first two say it does:

1. Do flat features exist, and how flat -- measured as km of range spread per
   MHz of frequency span, against the curvature real segments show?
2. Is a flat feature *persistent* -- at the same range across soundings, hours
   and transmitters? Ionospheric structure moves with the ionosphere. A fixed
   delay that does not move is instrumental, and that is the decisive test.
3. Do `algo` and `dp` land on it? A persistent artefact nothing reports is a
   curiosity; one that carries the MUF is a measurement error in the archive.

Deliberately extractor-independent where it can be: the range profile is the
brightest cell above threshold at each frequency, not any estimator's mask, so
"what is in the data" and "what the estimators did with it" stay separable.

    python tools/flat_tails.py --archive .../ionozond_data2 --out out/flat
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muf import loader                                    # noqa: E402
from muf.extractors import algorithmic, viterbi           # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "adjudicate", Path(__file__).resolve().parent / "adjudicate.py")
adjudicate = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("adjudicate", adjudicate)
_spec.loader.exec_module(adjudicate)

#: The level a cell must beat to count as detected. The same constant the
#: estimators use, so "what is lit" means the same thing here as there.
THRESHOLD_DB = viterbi.DEFAULT_THRESHOLD_DB

#: A segment is *flat* below this much range change per MHz. Set against what
#: real segments show rather than picked: the curved segments in the four
#: soundings inspected by hand run 35-130 km/MHz, the flat ones 0-6.7. The gap
#: is wide enough that the exact cut does not matter, which is the only reason
#: a single number is defensible here.
FLAT_KM_PER_MHZ = 10.0

#: And it must be long enough that flatness means something. Below this a real
#: trace is legitimately flat -- the low-ray leg is, away from the nose.
MIN_FLAT_MHZ = 1.0

#: Frequency bins may be this far apart and still belong to one feature, and
#: the range may step this far between them. Fades are short and the trace is
#: continuous; a bigger jump is a different feature.
MAX_GAP_BINS = 2
MAX_STEP_KM = 20.0

#: Segments shorter than this are noise, not features.
MIN_SEGMENT_BINS = 5


def range_profile(ion):
    """Brightest lit range per frequency bin. No extractor involved."""
    lit = ion.db >= THRESHOLD_DB
    has = lit.any(axis=1)
    idx = np.argmax(np.where(lit, ion.db, -np.inf), axis=1)
    return has, np.where(has, ion.vrange[idx], np.nan)


def segments(ion):
    """Connected features, as (freq_bins, freq_span_mhz, range_spread_km)."""
    has, rng = range_profile(ion)
    out, cur = [], []
    for f in np.flatnonzero(has):
        if cur and (f - cur[-1] <= MAX_GAP_BINS
                    and abs(rng[f] - rng[cur[-1]]) <= MAX_STEP_KM):
            cur.append(f)
            continue
        if len(cur) >= MIN_SEGMENT_BINS:
            out.append(cur)
        cur = [f]
    if len(cur) >= MIN_SEGMENT_BINS:
        out.append(cur)

    return [dict(bins=np.array(c),
                 f_lo=float(ion.freq[c[0]]), f_hi=float(ion.freq[c[-1]]),
                 span=float(ion.freq[c[-1]] - ion.freq[c[0]]),
                 r_med=float(np.median(rng[c])),
                 spread=float(np.ptp(rng[c]))) for c in out]


def _flatness(seg):
    return seg["spread"] / seg["span"] if seg["span"] > 0 else float("inf")


def survey_one(path):
    ion = loader.load(str(path))
    segs = segments(ion)
    if not segs:
        return None

    flat = [s for s in segs
            if s["span"] >= MIN_FLAT_MHZ and _flatness(s) < FLAT_KM_PER_MHZ]
    curved = [s for s in segs if _flatness(s) >= FLAT_KM_PER_MHZ]

    a, d = algorithmic.extract(ion), viterbi.extract(ion)
    longest = max(flat, key=lambda s: s["span"]) if flat else None
    top_curved = max((s["f_hi"] for s in curved), default=float("nan"))

    row = dict(
        n_seg=len(segs), n_flat=len(flat),
        flat_span=round(longest["span"], 3) if longest else 0.0,
        flat_range=round(longest["r_med"], 1) if longest else "",
        flat_f_hi=round(longest["f_hi"], 3) if longest else "",
        flat_km_per_mhz=round(_flatness(longest), 2) if longest else "",
        top_curved_mhz=round(top_curved, 3) if curved else "",
        algo_mhz=round(a.muf_mhz, 3) if a.ok else "",
        dp_mhz=round(d.muf_mhz, 3) if d.ok else "",
        algo_range=round(a.vrange_km, 1) if a.ok else "",
        dp_range=round(d.vrange_km, 1) if d.ok else "",
    )
    # Did an estimator land on the flat feature? Judged by range, not by
    # frequency: the flat line and the real trace are separated in range by
    # more than the tolerance, which is what makes this answerable at all.
    for name in ("algo", "dp"):
        r = row[f"{name}_range"]
        row[f"{name}_on_flat"] = int(
            longest is not None and r != ""
            and abs(r - longest["r_med"]) <= MAX_STEP_KM)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default="out/flat")
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, read, failed = [], 0, 0

    for path, when, tx in adjudicate._archive_files(Path(args.archive)):
        if read >= args.limit:
            break
        read += 1
        try:
            row = survey_one(path)
        except Exception:
            failed += 1
            continue
        if row is None:
            continue
        row.update(when=when.strftime("%Y-%m-%d %H:%M:%S"), hour=when.hour,
                   tx=tx, path=str(path))
        rows.append(row)
        if len(rows) % 250 == 0:
            print(f"  {len(rows)} of {read} read", flush=True)

    if not rows:
        print("nothing readable in the archive")
        return 1

    csv_path = out / "flat_tails.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    has_flat = [r for r in rows if r["n_flat"]]
    print(f"\n{n} soundings with detections ({read} read, {failed} unreadable)")
    print(f"  a flat feature >= {MIN_FLAT_MHZ} MHz: "
          f"{len(has_flat)} ({100 * len(has_flat) / n:.1f}%)")
    if has_flat:
        spans = np.array([r["flat_span"] for r in has_flat])
        rngs = np.array([r["flat_range"] for r in has_flat], dtype=float)
        print(f"  flat span:  median {np.median(spans):.2f} MHz, "
              f"max {spans.max():.2f}")
        print(f"  flat range: median {np.median(rngs):+.0f} km, "
              f"iqr {np.percentile(rngs, 25):+.0f}..{np.percentile(rngs, 75):+.0f}")
        for name in ("algo", "dp"):
            on = sum(r[f"{name}_on_flat"] for r in has_flat)
            print(f"  {name:5s} reported a MUF on the flat feature: "
                  f"{on} of {len(has_flat)} ({100 * on / len(has_flat):.1f}%)")
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
