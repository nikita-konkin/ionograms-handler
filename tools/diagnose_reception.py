"""Why did reception fail? Locate the echo on the *whole* range axis.

The default gate keeps 2329.6-5000 km of a +/-60,000 km axis (`signal-chain.md`
sec. 5.3). That is the right window when the acquisition timing is right, and a
trap when it is not: because this is stretch processing -- the chirp replica is
mixed in at acquisition and only the beat signal is recorded -- a timing error
maps straight to apparent range,

    range offset = c * dt        1 ms of timing error = 300 km

so roughly 9 ms of clock drift walks the echo out of a 2670 km gate. The
symptom is indistinguishable from "no signal": the estimators find nothing,
because nothing is inside the window they were told to look in.

This tool re-forms each ionogram across the full range axis and asks three
questions the estimators cannot:

1. Is there a coherent trace *anywhere* on the axis?
2. If so, is it inside the nominal gate, and if not, by how much?
3. Did the recording contain the whole sweep?

It deliberately does **not** use the estimators. They all narrow to the gated
tile and hand a presence array to `pick`; running them here would mean asking
the code under suspicion whether anything is wrong. Everything below works on
the raw thresholded power array instead.

What it cannot do: if the transmitter changed its chirp *rate*, or its timing
moved by more than the axis half-span, the beat signal fell outside the 40 kHz
recording bandwidth and no reprocessing recovers it. Absence of a trace here is
consistent both with "nothing was transmitted" and with "we de-chirped against
the wrong parameters".

Usage:

    python tools/diagnose_reception.py data/2026.02.04 --out out/diag
    python tools/diagnose_reception.py data/2026.02.0[456] --stride 4
    python tools/diagnose_reception.py data --stride 12 --expect-range 2739
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Running a script puts tools/ on sys.path, not the repo root, so `muf` would
# only import from an editable install. Prepending the root makes the tool work
# either way, and makes the repo copy win -- which is what you want when the
# thing you are diagnosing is the repo copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muf import calibrate, spectro          # noqa: E402
from muf.io_lfs import HEADER_SIZE, read_header   # noqa: E402

#: The threshold every estimator shares, on the dB scale `to_db` produces.
#: 30 dB is where median equalization puts the noise floor, so this is a 13 dB
#: signal-to-noise -- see `signal-chain.md` sec. 5.2.
DETECTION_DB = spectro.NOISE_FLOOR_DB + 13.0

#: Below this many above-threshold cells in the best range bin, there is no
#: trace anywhere on the axis -- not a displaced one, not a weak one. A real
#: low-ray leg is flat in range and persists over many frequency bins, so it
#: concentrates; broadband interference spreads across one frequency row and
#: does not.
MIN_TRACE_CELLS = 20

#: Fraction of soundings that must show a symptom before it is called a
#: pattern rather than noise.
PATTERN_FRACTION = 0.20

#: Spread of the displacement, below which it reads as a fixed offset rather
#: than drift. One gate width is far too coarse to matter; 300 km is 1 ms.
FIXED_OFFSET_SPREAD_KM = 300.0


@dataclass
class Scan:
    """One sounding: where its energy sits, and whether it was fully recorded."""

    file: str
    datetime: str
    tx: str
    rx: str
    path_type: str
    div_coef: float           # how range maps to delay on this path

    sweep_fraction: float
    sweep_complete: bool

    gate_lo: float
    gate_hi: float

    peak_range_km: float      # range bin holding the most above-threshold cells
    peak_cells: int           # how many, in that bin
    total_cells: int          # above-threshold cells over the whole axis
    frac_in_gate: float       # of those, the fraction inside the nominal gate

    offset_km: float          # peak_range_km minus the reference range
    timing_ms: float          # the timing error that offset would imply

    detected: bool            # peak_cells >= MIN_TRACE_CELLS
    in_gate: bool


def timing_error_ms(offset_km: float, div_coef: float) -> float:
    """The acquisition timing error a range displacement would imply.

    Range maps to delay through ``2 / div_coef``: an oblique echo's group range
    is ``c * delay``, a vertical sounding's is half that (`io_lfs.div_coef`).
    So 300 km is 1 ms oblique, 150 km is 1 ms vertical.
    """
    delay_s = offset_km * div_coef / (2 * calibrate.C_KM_S)
    return delay_s * 1e3


def n_freq_from_size(path: Path, window: int) -> int:
    """Window count from the file size, without reading the samples.

    A truncated recording still declares the full sweep in its header, so this
    is what makes the fast pre-scan possible: `sweep_fraction` needs only the
    header and the payload length.
    """
    n_samples = (path.stat().st_size - HEADER_SIZE) // 8   # complex64
    return max(0, n_samples // window)


def scan_one(path: Path, window: int, reference_km: float | None) -> Scan | None:
    """Form the ungated ionogram and locate its energy. None if unreadable."""
    try:
        header = read_header(path)
        n_freq = n_freq_from_size(path, window)
        if n_freq == 0:
            print(f"  skip {path.name}: shorter than one {window}-sample window",
                  file=sys.stderr)
            return None

        # The whole axis, both signs. gate_indices clamps to the array, so
        # asking for +/- the half-span returns every bin -- and a negative
        # apparent range is exactly what a timing error of the other sign
        # produces, so it has to stay in view.
        half = calibrate.range_half_span(header)
        ion = spectro.compute(path, window=window, gate_km=(-half, half),
                              header=header)
    except Exception as exc:                      # a bad file must not stop the scan
        print(f"  skip {path.name}: {exc}", file=sys.stderr)
        return None

    gate_lo, gate_hi = calibrate.default_gate(header)
    vrange = ion.cal.vrange
    above = ion.db >= DETECTION_DB

    # Count per range bin, across frequency. A trace is persistent in frequency
    # at near-constant range, so it concentrates in a bin; interference is
    # broad in range within a single frequency row and does not.
    per_range = above.sum(axis=0)
    peak_idx = int(np.argmax(per_range))
    peak_cells = int(per_range[peak_idx])
    peak_range = float(vrange[peak_idx])

    total = int(per_range.sum())
    inside = (vrange >= gate_lo) & (vrange <= gate_hi)
    frac_in_gate = float(per_range[inside].sum() / total) if total else 0.0

    ref = reference_km if reference_km is not None else peak_range
    offset = peak_range - ref

    return Scan(
        file=path.name,
        datetime=header.datetime.isoformat(),
        tx=header.tx_name,
        rx=header.rx_name,
        path_type=header.path_type,
        div_coef=header.div_coef,
        sweep_fraction=round(ion.cal.sweep_fraction, 4),
        sweep_complete=ion.cal.sweep_complete,
        gate_lo=round(gate_lo, 1),
        gate_hi=round(gate_hi, 1),
        peak_range_km=round(peak_range, 1),
        peak_cells=peak_cells,
        total_cells=total,
        frac_in_gate=round(frac_in_gate, 4),
        offset_km=round(offset, 1),
        timing_ms=round(timing_error_ms(offset, header.div_coef), 3),
        detected=peak_cells >= MIN_TRACE_CELLS,
        in_gate=gate_lo <= peak_range <= gate_hi,
    )


def rescore(scans: list[Scan], reference_km: float) -> None:
    """Recompute offsets once a reference range is known. In place."""
    for s in scans:
        s.offset_km = round(s.peak_range_km - reference_km, 1)
        s.timing_ms = round(timing_error_ms(s.offset_km, s.div_coef), 3)


def verdict(scans: list[Scan], reference_km: float, ref_source: str) -> list[str]:
    """What the numbers say. Ordered so the first hit is the likeliest cause."""
    n = len(scans)
    out = [
        f"{n} soundings, {scans[0].tx} -> {scans[0].rx}",
        f"reference range {reference_km:.0f} km ({ref_source})",
        f"nominal gate {scans[0].gate_lo:.0f}-{scans[0].gate_hi:.0f} km",
        "",
    ]

    detected = [s for s in scans if s.detected]
    silent = n - len(detected)
    displaced = [s for s in detected if not s.in_gate]
    truncated = [s for s in scans if not s.sweep_complete]

    out.append(f"  trace found anywhere on the axis : {len(detected):5d}  ({len(detected)/n:.0%})")
    out.append(f"  no coherent trace at all        : {silent:5d}  ({silent/n:.0%})")
    out.append(f"  found, but outside the gate     : {len(displaced):5d}  ({len(displaced)/n:.0%})")
    out.append(f"  recording truncated             : {len(truncated):5d}  ({len(truncated)/n:.0%})")
    out.append("")

    found = False

    if displaced and len(displaced) / max(1, len(detected)) >= PATTERN_FRACTION:
        found = True
        offs = np.array([s.offset_km for s in displaced])
        med, spread = float(np.median(offs)), float(np.percentile(offs, 90) - np.percentile(offs, 10))
        ms = timing_error_ms(med, displaced[0].div_coef)
        out.append("TIMING -- the echo is on the axis but outside the gate.")
        out.append(f"  median displacement {med:+.0f} km  (10-90 spread {spread:.0f} km)")
        out.append(f"  implies an acquisition timing error of {ms:+.1f} ms")
        if spread <= FIXED_OFFSET_SPREAD_KM:
            out.append("  The displacement is consistent, so this reads as a fixed")
            out.append("  offset -- a changed transmitter schedule, or a constant")
            out.append("  lag in the local replica. Check the console's Lag line")
            out.append("  and whether NTP is enabled (`signal-chain.md` sec. 7.3).")
        else:
            out.append("  The displacement varies, so this reads as clock drift")
            out.append("  rather than a fixed offset. NTP appears disabled in the")
            out.append("  console -- that is the first thing to check.")
        lo = min(s.peak_range_km for s in displaced)
        hi = max(s.peak_range_km for s in displaced)
        pad = max(200.0, 0.1 * (hi - lo))
        out.append("  Either way the recordings are salvageable: widen the gate")
        out.append(f"  (`muf run --gate {lo - pad:.0f},{hi + pad:.0f}`) and re-extract.")
        out.append("")

    if silent / n >= 0.5:
        found = True
        out.append("NO SIGNAL -- most soundings have no coherent trace anywhere on")
        out.append("  the axis, displaced or otherwise. That is consistent with a")
        out.append("  local fault (antenna, feed, USRP, front-end), with the")
        out.append("  transmitter being off air, or with the transmitter having")
        out.append("  changed its chirp *rate* -- in the last case the beat signal")
        out.append("  fell outside the 40 kHz recording bandwidth and these files")
        out.append("  hold nothing recoverable. Migrating to v2 would make that")
        out.append("  case diagnosable in future; it does not recover these.")
        out.append("")

    if len(truncated) / n >= 0.05:
        found = True
        frac = np.array([s.sweep_fraction for s in truncated])
        out.append("TRUNCATION -- recordings are stopping before the sweep ends.")
        out.append(f"  median {np.median(frac):.0%} of the intended sweep, "
                   f"shortest {frac.min():.0%}")
        out.append("  This is `BACKLOG.md` sec. 4 (17% of 2026-02-05). If the rate has")
        out.append("  risen since, the dropouts and the current fault are the same")
        out.append("  problem, and it has been developing for months.")
        out.append("  MUF from a truncated sweep is capped by where the recording")
        out.append("  stopped, not by the ionosphere.")
        out.append("")

    if not found:
        out.append("NOTHING ANOMALOUS -- no symptom reaches the reporting")
        out.append("  threshold, and the trace sits where the gate expects it.")
        out.append("  Reception is not failing at the acquisition-timing level")
        out.append("  over this set. Look at the extraction stage, or narrow the")
        out.append("  window: a fault confined to a few hours does not clear the")
        out.append("  thresholds once it is averaged over days.")
        out.append("")

    return out


# --- plotting ----------------------------------------------------------------
#
# Three stacked panels on one shared time axis rather than twin y-scales: the
# quantities have unrelated units, and overlaying them on two scales would
# invent a correlation. Colours are two slots from a validated palette --
# categorical slot 1 for the data, the status `critical` step for out-of-gate --
# with marker shape and the legend carrying the same distinction, so nothing
# depends on hue alone.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
CRITICAL = "#d03b3b"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.yaxis.label.set_color(INK_MUTED)


def plot(scans: list[Scan], reference_km: float, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from datetime import datetime as _dt

    # date2num rather than datetime64: the timestamps carry a UTC offset, which
    # numpy deprecated, and float dates index cleanly under a boolean mask.
    t = mdates.date2num([_dt.fromisoformat(s.datetime) for s in scans])
    ok = np.array([s.in_gate and s.detected for s in scans])
    gate_lo, gate_hi = scans[0].gate_lo, scans[0].gate_hi

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    # 1 -- where the energy actually is, against where the gate looks.
    #
    # The axis spans +/-60,000 km, so one sounding whose argmax landed on noise
    # would auto-scale the gate down to a hairline. The view is pinned to the
    # gate and a few gate-widths either side -- enough to show any displacement
    # a timing error could plausibly cause -- and anything past that is drawn at
    # the boundary rather than dropped, so nothing hides off-scale.
    ax = axes[0]
    y = np.array([s.peak_range_km for s in scans])
    det = y[np.array([s.detected for s in scans])]

    width = gate_hi - gate_lo
    lo_v, hi_v = gate_lo - 1.5 * width, gate_hi + 1.5 * width
    if det.size:
        med = float(np.median(det))
        if not lo_v <= med <= hi_v:        # a genuinely shifted day: recentre,
            lo_v = min(med - 1.5 * width, gate_lo - 0.2 * width)   # keeping the
            hi_v = max(med + 1.5 * width, gate_hi + 0.2 * width)   # gate in view

    ax.axhspan(gate_lo, gate_hi, color=GRID, zorder=0)
    ax.axhline(reference_km, color=AXIS, linewidth=1.0, zorder=1)
    ax.annotate(f"expected {reference_km:.0f} km",
                xy=(0.004, reference_km), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8, color=INK_MUTED)

    below, above = y < lo_v, y > hi_v
    shown = ~(below | above)
    # Labels only where there are marks: an empty scatter still claims a legend
    # entry, which would advertise a category the plot does not contain.
    ax.scatter(t[ok & shown], y[ok & shown], s=18, c=SERIES, marker="o",
               linewidths=0, zorder=3,
               label="in gate" if (ok & shown).any() else None)
    ax.scatter(t[~ok & shown], y[~ok & shown], s=26, c=CRITICAL, marker="x",
               linewidths=1.4, zorder=4,
               label="outside gate / no trace" if (~ok & shown).any() else None)
    for mask, edge, marker in ((below, lo_v, "v"), (above, hi_v, "^")):
        if mask.any():
            ax.scatter(t[mask], np.full(int(mask.sum()), edge), s=26,
                       c=CRITICAL, marker=marker, linewidths=0,
                       label=f"off scale ({int(mask.sum())})", zorder=4)

    ax.set_ylim(lo_v, hi_v)
    _style(ax)
    ax.set_ylabel("peak echo range  (km)")
    ax.set_title("Where the echo actually sits -- shaded band is the nominal gate",
                 color=INK, fontsize=11, loc="left")
    if (~ok).any():
        leg = ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
        for txt in leg.get_texts():
            txt.set_color(INK_MUTED)

    # 2 -- how much of the detected energy the gate is actually keeping.
    ax = axes[1]
    ax.scatter(t, [s.frac_in_gate for s in scans], s=18, c=SERIES, marker="o",
               linewidths=0, zorder=3)
    _style(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("fraction in gate")
    ax.set_title("Share of above-threshold energy the nominal gate retains",
                 color=INK, fontsize=11, loc="left")

    # 3 -- whether the recording contained the sweep at all.
    ax = axes[2]
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.scatter(t, [s.sweep_fraction for s in scans], s=18, c=SERIES, marker="o",
               linewidths=0, zorder=3)
    _style(ax)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("sweep fraction")
    ax.set_title("How much of the intended sweep each recording contains",
                 color=INK, fontsize=11, loc="left")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# --- entry point -------------------------------------------------------------

def iter_lfs(paths: list[str], stride: int) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.lfs")))
        elif p.exists():
            files.append(p)
        else:
            print(f"  skip {raw}: not found", file=sys.stderr)
    return sorted(set(files))[::stride]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Locate the echo on the full range axis to tell a timing "
                    "fault from an absent signal.")
    ap.add_argument("paths", nargs="+", help=".lfs files, or directories of them")
    ap.add_argument("--out", default="out/diag",
                    help="output stem; writes <stem>.csv and <stem>.png")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth sounding (the full axis costs ~40 MB "
                         "and ~0.3 s each; 288/day makes 4 or 12 reasonable)")
    ap.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW)
    ap.add_argument("--expect-range", type=float, default=None,
                    help="reference range in km. Default: the median peak range "
                         "of soundings that landed in the gate, which "
                         "self-calibrates against a known-good subset.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    files = iter_lfs(args.paths, args.stride)
    if not files:
        print("no .lfs files found", file=sys.stderr)
        return 2
    print(f"scanning {len(files)} soundings (stride {args.stride})...")

    scans: list[Scan] = []
    for i, path in enumerate(files, 1):
        s = scan_one(path, args.window, args.expect_range)
        if s is not None:
            scans.append(s)
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", end="\r", flush=True)
    print()

    if not scans:
        print("nothing readable", file=sys.stderr)
        return 2

    # Self-calibrate the reference from whatever landed in the gate, so a
    # partially-failing day measures its own displacement.
    if args.expect_range is not None:
        reference, ref_source = args.expect_range, "given"
    else:
        good = [s.peak_range_km for s in scans if s.in_gate and s.detected]
        if good:
            reference, ref_source = float(np.median(good)), \
                f"median of {len(good)} in-gate soundings"
        else:
            reference = (scans[0].gate_lo + scans[0].gate_hi) / 2
            ref_source = "gate midpoint -- nothing landed in the gate"
    rescore(scans, reference)

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(scans[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in scans)

    report = verdict(scans, reference, ref_source)
    print()
    print("\n".join(report))
    print(f"  table  {csv_path}")

    if not args.no_plot:
        png_path = stem.with_suffix(".png")
        plot(scans, reference, png_path)
        print(f"  plot   {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
