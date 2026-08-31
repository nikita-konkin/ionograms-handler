"""Settle `dp` against `algo` by eye, on the soundings where they disagree.

Every comparison in `docs/2026-08-30-segmentation-quality.md` scores the
extractors against each other, and that cannot say which is right. It shows up
as `dp` being simultaneously the closest thing to `contour` (MAE 0.078, the
tightest pair in the table) and the furthest from the consensus (0.132 against
`algo`'s 0.107) -- the signature of a circular measurement rather than a
finding. GIRO was going to break the circle and cannot: the only station within
the correlation limit of this path's control point publishes nothing.

What is left is a person looking at the picture. This makes that cheap.

**Fifty soundings, not sixteen thousand.** NOIRE-Net's number is for *training*
a network. Adjudicating a binary question needs far less, because the
disagreement is concentrated -- 30.7% of soundings at 20 UTC against 1.4% at
15 UTC -- and one-directional, every one of 379 with `algo` reading lower. A
few dozen soundings drawn from the hours where the two estimators actually
differ carry nearly all the information there is.

**Blinded, because the alternative is worthless.** The renderer draws two
candidate MUF markers as `A` and `B`, assigned by coin flip per sounding, with
no method names and no frequencies. Anyone who knows which marker is the new
extractor cannot un-know it, and this is exactly the kind of judgement that
bends: the question "is the trace still going at 21 MHz" has a real answer, and
also has a comfortable one. `score` un-blinds afterwards from the manifest.

**A preference, and optionally a measurement.** Picking the better marker takes
a few seconds and answers the question that was asked. The `muf_mhz` column is
there for the soundings where the honest answer is "neither" -- those are worth
recording, because a case both estimators get wrong is not evidence for either
and would otherwise be lost as a coin flip.

Usage:

    python tools/adjudicate.py select --archive $ARCHIVE --out review/ -n 50
    python tools/adjudicate.py render --out review/
    ... open review/*.png, fill the `verdict` column of review/verdicts.csv ...
    python tools/adjudicate.py score --out review/
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muf import loader                                    # noqa: E402
from muf.extractors import algorithmic, viterbi           # noqa: E402

#: Hours where the two estimators disagree, from the 2026-08-30 survey: the
#: rate peaks at 30.7% at 20 UTC and bottoms at 1.4% at 15 UTC. Sampling the
#: quiet hours would spend a person's afternoon confirming agreement.
TERMINATOR_HOURS = tuple(range(18, 24)) + (0, 1)

#: Below this the two are picking the same feature and there is nothing to
#: judge -- a marker pair a person cannot separate is a wasted sounding.
MIN_DISAGREEMENT_MHZ = 0.5

#: `tx` is the first segment and cannot contain a hyphen; the receiver can and
#: does -- "Yoshkar-Ola". A non-greedy `tx` with a hyphen-free `rx` splits this
#: the wrong way round and yields tx="NIC1-Yoshkar", rx="Ola".
FILENAME = re.compile(r"^lfm_ionogram-(?P<tx>[^-]+)-(?P<rx>.+?)-ch\d+-\d+-"
                      r"(?P<t0>\d+)\.\d+\.h5$")


def _archive_files(root: Path):
    """Every ionogram in the archive, with the instant its name encodes."""
    for day in sorted(os.listdir(root)):
        folder = root / day
        if not folder.is_dir() or " " in day:      # Nextcloud conflict copies
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            match = FILENAME.match(name)
            if not match:
                continue
            when = dt.datetime.fromtimestamp(int(match.group("t0")), dt.UTC)
            yield folder / name, when, match.group("tx")


def select(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    candidates = [(p, w, tx) for p, w, tx in _archive_files(Path(args.archive))
                  if w.hour in TERMINATOR_HOURS
                  and (args.tx is None or tx.startswith(args.tx))]
    rng.shuffle(candidates)
    print(f"{len(candidates)} soundings in hours "
          f"{TERMINATOR_HOURS[0]}-{TERMINATOR_HOURS[-1]} UTC; "
          f"reading until {args.n} disagree by >= {MIN_DISAGREEMENT_MHZ} MHz")

    rows, looked = [], 0
    for path, when, tx in candidates:
        if len(rows) >= args.n or looked >= args.limit:
            break
        looked += 1
        try:
            ion = loader.load(str(path))
        except Exception:
            continue
        a = algorithmic.extract(ion)
        d = viterbi.extract(ion)
        if not (a.ok and d.ok):
            continue
        if abs(a.muf_mhz - d.muf_mhz) < MIN_DISAGREEMENT_MHZ:
            continue
        # Coin flip per sounding: which estimator is drawn as `A`.
        a_is_algo = rng.random() < 0.5
        rows.append({
            "id": f"{len(rows):03d}",
            "path": str(path),
            "when": when.strftime("%Y-%m-%d %H:%M:%S"),
            "tx": tx,
            "algo_mhz": round(a.muf_mhz, 3),
            "dp_mhz": round(d.muf_mhz, 3),
            "gap_mhz": round(abs(a.muf_mhz - d.muf_mhz), 3),
            "A": "algo" if a_is_algo else "dp",
            "B": "dp" if a_is_algo else "algo",
        })
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} kept of {looked} read", flush=True)

    if not rows:
        print("nothing selected -- widen --limit or lower MIN_DISAGREEMENT_MHZ")
        return 1

    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # The sheet the person fills in. Deliberately carries *no* estimator
    # columns: it is the only file they need open, so it must not leak the
    # assignment the manifest holds.
    verdicts = out / "verdicts.csv"
    if verdicts.exists():
        print(f"{verdicts} exists; leaving it alone")
    else:
        with open(verdicts, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "verdict", "muf_mhz", "note"])
            for row in rows:
                writer.writerow([row["id"], "", "", ""])

    print(f"\nwrote {len(rows)} to {manifest}")
    print(f"      {verdicts}  <- fill the `verdict` column: A, B, or neither")
    print(f"      median gap {np.median([r['gap_mhz'] for r in rows]):.2f} MHz")
    return 0


def _read_manifest(out: Path) -> list[dict]:
    with open(out / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    rows = _read_manifest(out)
    images = out / "png"
    images.mkdir(exist_ok=True)

    for row in rows:
        try:
            ion = loader.load(row["path"])
        except Exception as exc:
            print(f"  {row['id']}: unreadable ({exc})")
            continue

        marks = {"A": float(row[row["A"] + "_mhz"]),
                 "B": float(row[row["B"] + "_mhz"])}

        fig, ax = plt.subplots(figsize=(15, 6.5))
        ax.pcolormesh(ion.freq, ion.vrange, ion.db.T, shading="nearest",
                      cmap="jet", vmin=35.0, vmax=65.0)

        # Zoom to where the signal is. The stored range axis spans +/-3999 km
        # about the expected arrival, and an echo occupies a few hundred of
        # them -- drawn whole, the trace is a sliver and the judgement being
        # asked for is not readable.
        lit = ion.db.max(axis=0) > 50.0
        if lit.any():
            rows_lit = ion.vrange[lit]
            pad = max(400.0, 0.35 * (rows_lit.max() - rows_lit.min()))
            ax.set_ylim(rows_lit.min() - pad, rows_lit.max() + pad)

        low, high = ax.get_ylim()
        for label, mhz in marks.items():
            ax.axvline(mhz, color="white", lw=1.8, alpha=0.95)
            # Inside the axes, not above them: at axes fraction 1.005 these
            # overprinted the title and each other.
            ax.annotate(label, xy=(mhz, high), xytext=(4, -6),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=17, fontweight="bold", color="white")
        ax.set_xlabel("Frequency (MHz)", fontsize=12)
        ax.set_ylabel("Virtual range (km)", fontsize=12)
        # Identity and time only. No MUF values, no method names: see the
        # module docstring on why the blinding is the point.
        ax.set_title(f"{row['id']}   {row['tx']}   {row['when']}Z", fontsize=13)
        ax.grid(axis="x", color="white", alpha=0.3, lw=0.5)

        fig.tight_layout()
        fig.savefig(images / f"{row['id']}.png", dpi=110)
        plt.close(fig)
        print(f"  {row['id']}", flush=True)

    print(f"\n{len(rows)} images in {images}")
    print("For each: which marker better sits at the high-frequency end of the")
    print("trace? Put A or B in verdicts.csv. If both are wrong, write")
    print("'neither' and, if you can read it off, the real MUF in muf_mhz.")
    return 0


def score(args) -> int:
    out = Path(args.out)
    rows = {r["id"]: r for r in _read_manifest(out)}
    with open(out / "verdicts.csv", encoding="utf-8") as fh:
        verdicts = [v for v in csv.DictReader(fh) if v["verdict"].strip()]

    if not verdicts:
        print("no verdicts filled in yet")
        return 1

    wins = {"algo": 0, "dp": 0}
    neither, errors = [], {"algo": [], "dp": []}
    for v in verdicts:
        row = rows.get(v["id"])
        if row is None:
            continue
        choice = v["verdict"].strip().upper()
        if choice in ("A", "B"):
            wins[row[choice]] += 1
        else:
            neither.append(v["id"])
        if v["muf_mhz"].strip():
            truth = float(v["muf_mhz"])
            errors["algo"].append(abs(float(row["algo_mhz"]) - truth))
            errors["dp"].append(abs(float(row["dp_mhz"]) - truth))

    judged = wins["algo"] + wins["dp"]
    print(f"{len(verdicts)} judged, {judged} picked a marker, "
          f"{len(neither)} said neither\n")
    for name in ("algo", "dp"):
        share = 100.0 * wins[name] / judged if judged else float("nan")
        print(f"  {name:<5} preferred on {wins[name]:3d} of {judged}  ({share:.0f}%)")

    if judged:
        # Exact binomial against a coin, which is the null a preference test
        # has to beat. No scipy: the two-sided tail is a short sum.
        from math import comb
        k, n = max(wins.values()), judged
        tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
        verdict = ("distinguishable from a coin flip" if p < 0.05
                   else "NOT distinguishable from a coin flip")
        print(f"\n  two-sided exact binomial p = {p:.3f} -- {verdict}")
        if p >= 0.05 and n < 30:
            need = 30 - n
            print(f"  {n} is a small sample; ~{need} more would sharpen it")

    if errors["algo"]:
        print(f"\n  against the {len(errors['algo'])} hand-scaled MUFs:")
        for name in ("algo", "dp"):
            print(f"    {name:<5} MAE {np.mean(errors[name]):.3f} MHz")

    if neither:
        print(f"\n  both wrong on: {', '.join(neither)}")
        print("  worth reading before trusting either estimator at the "
              "terminator.")

    summary = {"wins": wins, "neither": neither,
               "n_hand_scaled": len(errors["algo"])}
    (out / "score.json").write_text(json.dumps(summary, indent=1),
                                    encoding="utf-8")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="choose the soundings to judge")
    s.add_argument("--archive", required=True)
    s.add_argument("--out", default="review")
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--limit", type=int, default=600,
                   help="stop after reading this many, for slow archives")
    s.add_argument("--tx", default=None, help="only this transmitter prefix")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=select)

    r = sub.add_parser("render", help="draw the blinded images")
    r.add_argument("--out", default="review")
    r.set_defaults(func=render)

    c = sub.add_parser("score", help="un-blind and report")
    c.add_argument("--out", default="review")
    c.set_defaults(func=score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
