"""Command line interface: ``python -m muf <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__, compare, extractors, pipeline, render, spectro, track
from .export import saoxml
from .extractors import ALL_METHODS, DEFAULT_METHODS
from .reference import ALL_REFERENCES
from .io_lfs import find_lfs, read_header
from .pipeline import Options


def _gate(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    try:
        lo, hi = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--gate wants 'low,high' in km, got {text!r}"
        ) from None
    if lo >= hi:
        raise argparse.ArgumentTypeError(f"--gate low must be below high, got {text!r}")
    return lo, hi


def _methods(text: str | None) -> tuple[str, ...]:
    if not text:
        return DEFAULT_METHODS
    if text == "all":
        return ALL_METHODS
    chosen = tuple(extractors.canonical(part.strip())
                   for part in text.split(",") if part.strip())
    unknown = [m for m in chosen if m not in ALL_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown method(s) {', '.join(unknown)}; "
            f"choose from {', '.join(ALL_METHODS)} or 'all'"
        )
    return chosen


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW,
                        help="FFT window length (default: %(default)s)")
    parser.add_argument("--zero-periods", type=int,
                        default=spectro.DEFAULT_ZERO_PERIODS,
                        help="zero-padding periods. Subdivides range bins "
                             "without improving resolution (default: %(default)s)")
    parser.add_argument("--gate", type=str, default=None, metavar="LO,HI",
                        help="virtual-range gate in km (default: from the file header)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="cache gated spectrograms here to make re-runs fast")
    parser.add_argument("--band-floor", type=float, default=None, metavar="MHZ",
                        help="lowest frequency the transmitter actually "
                             "radiates, for flagging LOF that ran off the "
                             "bottom of the band (default: the sweep start, "
                             "which `muf lof` will tell you is often wrong)")


def _options(args) -> Options:
    method_options: dict[str, dict] = {}
    if getattr(args, "legacy_algo", False):
        method_options["algo"] = {"legacy": True}
    if getattr(args, "threshold_db", None) is not None:
        method_options.setdefault("contour", {})["threshold_db"] = args.threshold_db
    if getattr(args, "k", None) is not None:
        method_options.setdefault("kmeans", {})["k"] = args.k

    return Options(
        window=args.window,
        zero_periods=args.zero_periods,
        gate_km=_gate(args.gate),
        methods=getattr(args, "methods", DEFAULT_METHODS),
        min_run=getattr(args, "min_run", None),
        percentile=getattr(args, "percentile", None),
        cache_dir=args.cache_dir,
        method_options=method_options or None,
        band_floor_mhz=getattr(args, "band_floor", None),
    )


# --- commands ----------------------------------------------------------------

def cmd_run(args) -> int:
    options = _options(args)
    paths = find_lfs(args.target)
    print(f"{len(paths)} sounding(s) from {len(args.target)} target(s); "
          f"methods: {', '.join(options.methods)}", file=sys.stderr)

    frame = pipeline.process_many(
        args.target, options, jobs=args.jobs, progress=not args.quiet
    )

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "parquet" if args.format == "parquet" else "csv"
    days = pipeline.days_in(frame)

    # One file per day, whatever the targets were: a day is the natural unit
    # for everything downstream, and a table spanning several would break the
    # daily curve and the comparison report.
    for day, sub in pipeline.split_by_day(frame):
        out_path = pipeline.write(sub, args.out / f"{day}.{suffix}", args.format)
        print(f"wrote {out_path}  ({len(sub)} soundings)", file=sys.stderr)
        _summarise(sub, indent="  ")

    if args.combined and len(days) > 1:
        span = f"{days[0]}_{days[-1]}"
        combined = pipeline.write(frame, args.out / f"{span}.{suffix}", args.format)
        print(f"wrote {combined}  ({len(frame)} soundings over {len(days)} days)",
              file=sys.stderr)

    if len(days) > 1:
        print(f"\n{len(days)} days: {days[0]} to {days[-1]}", file=sys.stderr)
        _summarise(frame, indent="  ")

    if args.plot:
        _plot_each(paths, options, args.out / "ionograms")

    if args.daily:
        for day, sub in pipeline.split_by_day(frame):
            _write_daily(sub, args.out, day, args.format, args.plot)

    return 0


def _summarise(frame, indent: str = "  ") -> None:
    for method in pipeline.methods_in(frame):
        values = compare.usable(frame, method)
        got = int(values.notna().sum())
        if got:
            print(f"{indent}{method:8s} {got:4d}/{len(frame)} picked   "
                  f"{values.min():.2f}-{values.max():.2f} MHz", file=sys.stderr)
        else:
            print(f"{indent}{method:8s} no picks", file=sys.stderr)


def _plot_each(paths, options: Options, out_dir: Path) -> None:
    print(f"rendering {len(paths)} ionogram(s) to {out_dir}", file=sys.stderr)
    for path in paths:
        ion = spectro.compute_cached(
            path, options.window, options.zero_periods,
            options.gate_km, options.cache_dir,
        )
        results = extractors.run(ion, methods=options.methods,
                                 **options.per_method())
        render.plot(ion, out_dir / f"{path.stem}.png", results)


def _write_daily(frame, out_dir: Path, day, fmt: str, plot: bool) -> None:
    suffix = "parquet" if fmt == "parquet" else "csv"
    for method in pipeline.methods_in(frame):
        try:
            curve = pipeline.daily(frame, method=method)
        except (ValueError, KeyError) as exc:
            print(f"  daily {method}: skipped ({exc})", file=sys.stderr)
            continue
        path = pipeline.write(curve, out_dir / f"{day}_daily_{method}.{suffix}", fmt)
        print(f"wrote {path}", file=sys.stderr)
        if plot:
            render.plot_daily(curve, out_dir / f"{day}_daily_{method}.png", frame)


def cmd_plot(args) -> int:
    options = _options(args)
    paths = find_lfs(args.target)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(paths)} ionogram(s)", file=sys.stderr)

    for path in paths:
        ion = spectro.compute_cached(
            path, options.window, options.zero_periods,
            options.gate_km, options.cache_dir,
        )
        results = None
        if not args.no_axes and not args.no_muf:
            results = extractors.run(ion, methods=options.methods,
                                     **options.per_method())

        segments = reconstruction = None
        if args.trace and results:
            segments, reconstruction = _analyse_trace(ion, results, args.trace)

        out_path = args.out / f"{path.stem}.png"
        render.plot(ion, out_path, results, axes=not args.no_axes, dpi=args.dpi,
                    segments=segments, reconstruction=reconstruction)
        print(f"  {out_path}", file=sys.stderr)
    return 0


def _analyse_trace(ion, results, method: str):
    """Segment and reconstruct one estimator's trace, for the plot overlay."""
    from muf import fit, geometry, trace

    result = results.get(method)
    if result is None or not result.ok:
        return None, None

    _, _, ground = geometry.path_of(ion.header)
    freq, vrange, weight = trace.extract_points(ion, result)
    return trace.analyse(freq, vrange, ground, weight=weight)


def cmd_daily(args) -> int:
    frame = _read_tables(args.table)
    methods = [args.method] if args.method else pipeline.methods_in(frame)
    days = pipeline.days_in(frame)
    label = f"{days[0]}" if len(days) == 1 else f"{days[0]}_{days[-1]}"
    out_dir = args.out or _default_out_dir(args.table[0])

    if len(days) > 1:
        print(f"{len(days)} days: {days[0]} to {days[-1]}", file=sys.stderr)

    for method in methods:
        try:
            curve = pipeline.daily(frame, method=method, smooth=not args.no_smooth)
        except (ValueError, KeyError) as exc:
            print(f"  {method}: skipped ({exc})", file=sys.stderr)
            continue
        for skipped in curve.attrs.get("skipped_days", []):
            print(f"  {method}: skipped {skipped}", file=sys.stderr)

        path = pipeline.write(curve, out_dir / f"{label}_daily_{method}.csv")
        print(f"wrote {path}  ({len(curve)} points)", file=sys.stderr)
        if args.plot:
            image = render.plot_daily(curve, out_dir / f"{label}_daily_{method}.png",
                                      frame)
            print(f"wrote {image}", file=sys.stderr)
    return 0


def cmd_track(args) -> int:
    """Kalman-track the MUF through time: fill gaps, reject outliers."""
    from muf import track as tracking

    frame = _read_tables(args.table)
    methods = [args.method] if args.method else pipeline.methods_in(frame)
    out_dir = args.out or _default_out_dir(args.table[0])

    for method in methods:
        pieces = []
        for day, sub in pipeline.split_by_day(frame):
            try:
                result = tracking.track_results(
                    sub, method=method,
                    process_noise=args.process_noise,
                    gate_sigma=args.gate_sigma,
                )
            except (ValueError, KeyError) as exc:
                print(f"  {method} {day}: skipped ({exc})", file=sys.stderr)
                continue
            print(f"  {method} {day}: {result}", file=sys.stderr)
            pieces.append(result.frame)

        if not pieces:
            continue
        curve = pd.concat(pieces, ignore_index=True)
        days = pipeline.days_in(frame)
        label = f"{days[0]}" if len(days) == 1 else f"{days[0]}_{days[-1]}"
        path = pipeline.write(curve, out_dir / f"{label}_track_{method}.csv")
        print(f"wrote {path}  ({len(curve)} points)", file=sys.stderr)

        if args.plot:
            image = render.plot_track(curve, out_dir / f"{label}_track_{method}.png",
                                      frame, method)
            print(f"wrote {image}", file=sys.stderr)
    return 0


def cmd_compare(args) -> int:
    frame = _read_tables(args.table)

    reference = None
    if args.ref:
        reference = compare.read_reference(args.ref)
        print(f"reference: {len(reference)} points from {args.ref}", file=sys.stderr)

    exclude = []
    for span in args.exclude or []:
        try:
            start, stop = span.split("..")
        except ValueError:
            raise SystemExit(f"--exclude wants 'START..STOP', got {span!r}")
        exclude.append((start, stop))

    models = None
    if args.ref_model:
        models = ALL_REFERENCES if args.ref_model == "all" else [
            m.strip() for m in args.ref_model.split(",") if m.strip()
        ]

    text, _, _ = compare.report(
        frame, reference=reference,
        reference_name=Path(args.ref).stem if args.ref else "reference",
        exclude=exclude, reference_models=models,
    )

    first = Path(args.table[0])
    days = pipeline.days_in(frame)
    default_name = (f"{days[0]}_{days[-1]}_compare.md" if len(days) > 1
                    else f"{days[0]}_compare.md" if days
                    else "compare.md")
    out_path = args.out or (first.parent / default_name if first.is_dir() or
                            len(args.table) > 1
                            else first.with_name(first.stem + "_compare.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


def cmd_export(args) -> int:
    """Write each sounding as a SAO.XML 5.0 record list."""
    options = _options(args)
    paths = find_lfs(args.target)
    print(f"exporting {len(paths)} sounding(s) to {args.out}", file=sys.stderr)

    models = _iri_values(paths, args) if args.iri else {}

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in paths:
        try:
            root = saoxml.export_file(path, options, ursi_code=args.ursi_code,
                                      station_name=args.station,
                                      model=models.get(path))
        except Exception as exc:
            print(f"  {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        saoxml.write(root, args.out / f"{path.stem}.xml")
        written += 1

    print(f"wrote {written}/{len(paths)} record list(s)", file=sys.stderr)
    return 0 if written else 1


def _iri_values(paths, args) -> dict[Path, saoxml.ModelValues]:
    """IRI foF2, hmF2 and path MUF per sounding, in one evaluation per path.

    PyIRI takes an array of hours for a date, so a 288-sounding day costs one
    model call. Asking per sounding instead costs 0.42 s each -- two minutes for
    a day against 0.16 s -- which is why the timestamps are collected up front
    rather than resolved inside the export loop.

    Soundings are grouped by transmitter/receiver pair: the control point and
    the path length differ per circuit, so one series cannot serve two.
    """
    from collections import defaultdict

    from . import geometry
    from .reference import iri

    if not iri.available():
        print(f"  --iri: {iri.INSTALL_HINT}", file=sys.stderr)
        return {}

    circuits: dict[tuple, list] = defaultdict(list)
    for path in paths:
        try:
            header = read_header(path)
        except Exception as exc:
            print(f"  {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        tx, rx, _ = geometry.path_of(header)
        circuits[(tx, rx)].append((path, header.datetime))

    out: dict[Path, saoxml.ModelValues] = {}
    for (tx, rx), items in circuits.items():
        # No cache_dir: --cache-dir holds gated spectrograms, and the solar
        # indices keep their own default, the same one `compare --ref-model`
        # uses. Sharing the two would put index CSVs among the .npz tiles.
        series = iri.predict(tx, rx, [when for _, when in items],
                             offline=args.offline)
        if series.error:
            print(f"  --iri: {series.error}", file=sys.stderr)
            continue
        print(f"  --iri: {series.source}", file=sys.stderr)

        fof2 = series.detail["fof2"].to_numpy(dtype=float)
        hmf2 = series.detail["hmf2"].to_numpy(dtype=float)
        muf = series.muf.to_numpy(dtype=float)
        control = geometry.midpoint(tx, rx)
        options = (f"control point {control.lat:.2f}N {control.lon:.2f}E, "
                   f"D={geometry.great_circle_km(tx, rx):.0f}km")
        for i, (path, _) in enumerate(items):
            values = saoxml.ModelValues(
                name=series.name.upper(), muf_mhz=muf[i],
                fof2_mhz=fof2[i], hmf2_km=hmf2[i], options=options,
            )
            if values.any_finite:
                out[path] = values

    print(f"  --iri: {len(out)}/{len(paths)} sounding(s) modelled",
          file=sys.stderr)
    return out


def cmd_plot_sao(args) -> int:
    """Render exported SAO.XML records, over their ionograms if available."""
    paths = _find_xml(args.target)
    if not paths:
        raise SystemExit(f"no .xml files in {', '.join(str(t) for t in args.target)}")

    recordings = _recordings_by_stem(args.ionogram)
    if args.ionogram:
        print(f"{len(recordings)} recording(s) to draw under the records",
              file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(paths)} SAO file(s)", file=sys.stderr)

    options = _options(args)
    drawn = missing = 0
    for path in paths:
        try:
            records = saoxml.read(path)
        except Exception as exc:
            print(f"  {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        ion = None
        if recordings:
            source = recordings.get(path.stem)
            if source is None:
                missing += 1
            else:
                ion = spectro.compute_cached(
                    source, options.window, options.zero_periods,
                    options.gate_km, options.cache_dir,
                )

        wanted = [r for r in records
                  if not args.method or r.method == args.method]
        for record in wanted:
            # One record per estimator, so the method has to be in the name or
            # the last one written would silently win.
            suffix = f"_{record.method}" if record.method and len(wanted) > 1 else ""
            out_path = args.out / f"{path.stem}{suffix}.png"
            render.plot_sao(record, out_path, ion=ion, dpi=args.dpi,
                            full_band=args.full_band, trace=_trace_choice(args))
            print(f"  {out_path}", file=sys.stderr)
            drawn += 1

    if missing:
        print(f"{missing} record(s) had no matching recording; drew the "
              f"scaled trace instead", file=sys.stderr)
    print(f"drew {drawn} record(s)", file=sys.stderr)
    return 0 if drawn else 1


def _trace_choice(args) -> bool | None:
    """``--trace``/``--no-trace``, or None to let the renderer decide."""
    if args.trace:
        return True
    if args.no_trace:
        return False
    return None


def _recordings_by_stem(targets) -> dict[str, Path]:
    """Map ``.lfs`` stem -> path, for pairing a record with its sounding.

    The stem is the join key because `export` names each XML after the
    recording it came from, so nothing else has to be threaded through.
    """
    if not targets:
        return {}
    return {path.stem: path for path in find_lfs(targets)}


def _find_xml(targets) -> list[Path]:
    found: list[Path] = []
    for item in (Path(t) for t in targets):
        if item.is_dir():
            found.extend(sorted(item.rglob("*.xml")))
        elif item.exists():
            found.append(item)
        else:
            raise SystemExit(f"no such file or directory: {item}")
    return found


def cmd_lof(args) -> int:
    """Measure the band floor, and summarise LOF over a set of soundings."""
    from . import lof as lof_module

    options = _options(args)
    paths = find_lfs(args.target)
    print(f"{len(paths)} sounding(s)", file=sys.stderr)

    ions = []
    for path in paths:
        try:
            ions.append(spectro.compute_cached(
                path, options.window, options.zero_periods,
                options.gate_km, options.cache_dir,
            ))
        except Exception as exc:
            print(f"  {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not ions:
        raise SystemExit("no soundings could be read")

    # A hardware edge is constant while absorption swings through the day, so
    # the floor only shows up across many soundings -- which is the whole
    # reason this is a command rather than a per-file column.
    level = args.level if args.level is not None else lof_module.DEFAULT_LADDER[0]
    floor = lof_module.measure_band_floor(ions, level)
    declared = ions[0].cal.freq_start
    print("", file=sys.stderr)
    print(f"sweep declares      {declared:6.2f} MHz", file=sys.stderr)
    print(f"measured band floor {floor:6.2f} MHz   "
          f"({'matches' if abs(floor - declared) < 0.2 else 'DIFFERS'})",
          file=sys.stderr)
    if abs(floor - declared) >= 0.2:
        print(f"  pass --band-floor {floor:.2f} so a LOF down there is flagged "
              f"as an upper bound rather than reported as a measurement",
              file=sys.stderr)

    rows = []
    for path, ion in zip(paths, ions):
        rungs = lof_module.ladder(ion, band_floor_mhz=args.band_floor or floor)
        row = {"datetime": ion.header.datetime, "file": path.name}
        for level, rung in sorted(rungs.items()):
            row[f"lof{int(level)}"] = rung.lof_mhz
            row[f"lof{int(level)}_at_floor"] = rung.at_band_floor
        rows.append(row)

    frame = pd.DataFrame(rows)
    for level in lof_module.DEFAULT_LADDER:
        column = frame[f"lof{int(level)}"]
        got = int(column.notna().sum())
        at_floor = int(frame[f"lof{int(level)}_at_floor"].sum())
        if got:
            print(f"  {level:.0f} dB: {got:4d}/{len(frame)} scaled  "
                  f"{column.min():5.2f}-{column.max():5.2f} MHz  "
                  f"{at_floor} at the floor", file=sys.stderr)
        else:
            print(f"  {level:.0f} dB: nothing scaled", file=sys.stderr)

    if args.out:
        path = pipeline.write(frame, args.out)
        print(f"wrote {path}", file=sys.stderr)
    return 0


def cmd_info(args) -> int:
    """Print header and derived geometry for a sounding -- no processing."""
    from . import calibrate

    for path in find_lfs(args.target)[: args.limit]:
        header = read_header(path)
        cal = calibrate.build(
            header,
            n_freq=max(1, (path.stat().st_size - 512) // 8 // args.window),
            window=args.window, zero_periods=args.zero_periods,
        )
        ground = calibrate.ground_range_km(header)
        print(f"{path.name}")
        print(f"  {header}")
        print(f"  tx {header.tx_latitude:.3f},{header.tx_longitude:.3f}  "
              f"rx {header.rx_latitude:.3f},{header.rx_longitude:.3f}"
              + (f"  ground path {ground:.0f} km" if ground else ""))
        print(f"  sweep      {cal.freq_start:.2f}-{cal.freq_stop:.2f} MHz, "
              f"{cal.n_freq} bins @ {cal.freq_step_mhz*1e3:.2f} kHz")
        print(f"  range axis +/-{cal.half_span:.0f} km, "
              f"{cal.range_step:.2f} km/bin, resolution {cal.resolution_km:.2f} km")
        print(f"  gate       {cal.gate_km[0]:.0f}-{cal.gate_km[1]:.0f} km "
              f"-> bins {cal.gate_idx[0]}..{cal.gate_idx[1]} "
              f"({cal.n_range} of {cal.n_range_full}, "
              f"{cal.n_range_full/cal.n_range:.0f}x reduction)")
        print()
    return 0


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"no such results table: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "datetime" not in frame:
        raise SystemExit(f"{path} has no 'datetime' column -- is it a results table?")
    return frame


def _default_out_dir(target: Path) -> Path:
    """Write results beside their inputs: into a directory, or next to a file."""
    target = Path(target)
    return target if target.is_dir() else target.parent


def _is_results_table(frame: pd.DataFrame) -> bool:
    """True for a per-sounding table from `run`.

    ``file`` is the discriminator: one row per ``.lfs``. Daily curves have a
    ``datetime`` and a ``muf_smooth`` column, so looking only for those would
    pull them in and invent a method called "smooth".
    """
    return ("datetime" in frame and "file" in frame
            and any(str(c).startswith("muf_") for c in frame.columns))


def _read_tables(paths) -> pd.DataFrame:
    """Read one or more results tables into a single frame, sorted by time.

    A directory is searched for results tables; daily curves, comparison
    reports and anything else living alongside them are skipped. Duplicate
    soundings -- which appear when both the per-day tables and a `--combined`
    table are present -- are dropped.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    explicit: list[Path] = []
    discovered: list[Path] = []
    for item in (Path(p) for p in paths):
        if item.is_dir():
            discovered.extend(sorted(item.glob("*.csv")))
            discovered.extend(sorted(item.glob("*.parquet")))
        else:
            explicit.append(item)

    frames = []
    for path in explicit:
        frame = _read_table(path)          # named directly: complain if unusable
        if not _is_results_table(frame):
            raise SystemExit(
                f"{path} is not a per-sounding results table (no 'file' column). "
                f"Point `compare` at the output of `muf run`."
            )
        frames.append(frame)

    for path in discovered:
        try:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" \
                else pd.read_csv(path)
        except Exception:
            continue
        if _is_results_table(frame):
            frames.append(frame)

    if not frames:
        raise SystemExit(
            f"no results tables found in {', '.join(str(p) for p in paths)}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    return (combined.sort_values("datetime")
            .drop_duplicates(subset=["datetime", "file"])
            .reset_index(drop=True))


# --- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muf",
        description="Extract MUF from .lfs chirp-ionosonde recordings.",
    )
    parser.add_argument("--version", action="version", version=f"muf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run = sub.add_parser("run", help="extract MUF from soundings")
    run.add_argument("target", type=Path, nargs="+",
                     help=".lfs files or directories; several days may be given "
                          "at once, and results are written one file per day")
    run.add_argument("--out", type=Path, default=Path("out"),
                     help="output directory (default: %(default)s)")
    run.add_argument("--methods", type=_methods, default=DEFAULT_METHODS,
                     help=f"comma-separated, or 'all' "
                          f"(default: {','.join(DEFAULT_METHODS)})")
    run.add_argument("--jobs", type=int, default=1,
                     help="parallel workers; 0 uses all but one core")
    run.add_argument("--format", choices=("csv", "parquet"), default="csv")
    run.add_argument("--min-run", type=int, default=None,
                     help="consecutive frequency bins required for a pick")
    run.add_argument("--percentile", type=float, default=None,
                     help="trim the top of the detection distribution")
    run.add_argument("--threshold-db", type=float, default=None,
                     help="signal threshold for the contour method")
    run.add_argument("-k", type=int, default=None, help="clusters for kmeans")
    run.add_argument("--legacy-algo", action="store_true",
                     help="reproduce the pre-fix algorithmic decision rule")
    run.add_argument("--plot", action="store_true", help="also render ionograms")
    run.add_argument("--daily", action="store_true",
                     help="also write the interpolated daily curve")
    run.add_argument("--combined", action="store_true",
                     help="additionally write one table spanning every day")
    run.add_argument("--quiet", action="store_true")
    _common(run)
    run.set_defaults(func=cmd_run)

    # plot
    plot = sub.add_parser("plot", help="render ionograms")
    plot.add_argument("target", type=Path, nargs="+")
    plot.add_argument("--out", type=Path, default=Path("imgs"))
    plot.add_argument("--methods", type=_methods, default=DEFAULT_METHODS)
    plot.add_argument("--no-axes", action="store_true",
                      help="bare raster, no axes or annotation")
    plot.add_argument("--no-muf", action="store_true",
                      help="do not run the estimators or mark their picks")
    plot.add_argument("--trace", nargs="?", const="algo", default=None,
                      metavar="METHOD",
                      help="overlay the detected trace, split by propagation "
                           "mode, with the reconstructed curve "
                           "(default method: algo)")
    plot.add_argument("--dpi", type=int, default=render.DEFAULT_DPI)
    _common(plot)
    plot.set_defaults(func=cmd_plot)

    # daily
    daily = sub.add_parser("daily", help="interpolate and smooth results")
    daily.add_argument("table", type=Path, nargs="+",
                       help="results table(s) from `run`, or a directory of them")
    daily.add_argument("--method", default=None, help="default: every method present")
    daily.add_argument("--out", type=Path, default=None)
    daily.add_argument("--no-smooth", action="store_true")
    daily.add_argument("--plot", action="store_true")
    daily.set_defaults(func=cmd_daily)

    # track
    trk = sub.add_parser("track",
                         help="track MUF through time: fill gaps, reject outliers")
    trk.add_argument("table", type=Path, nargs="+",
                     help="results table(s) from `run`, or a directory of them")
    trk.add_argument("--method", default=None, help="default: every method present")
    trk.add_argument("--out", type=Path, default=None)
    trk.add_argument("--process-noise", type=float,
                     default=track.DEFAULT_PROCESS_NOISE_MHZ_PER_HOUR,
                     help="how fast MUF may change, MHz/hour (default: %(default)s)")
    trk.add_argument("--gate-sigma", type=float, default=track.DEFAULT_GATE_SIGMA,
                     help="reject picks beyond this many sigma (default: %(default)s)")
    trk.add_argument("--plot", action="store_true")
    trk.set_defaults(func=cmd_track)

    # compare
    comp = sub.add_parser("compare", help="compare methods, and a reference series")
    comp.add_argument("table", type=Path, nargs="+",
                      help="results table(s) from `run`, or a directory of them")
    comp.add_argument("--ref", type=Path, default=None,
                      help="reference CSV, e.g. MUF_cyprus1_20220320.csv")
    comp.add_argument("--ref-model", type=str, default=None,
                      metavar="NAMES",
                      help=f"external reference models to evaluate: "
                           f"{','.join(ALL_REFERENCES)}, or 'all'")
    comp.add_argument("--exclude", action="append", metavar="START..STOP",
                      help="drop a time span; repeatable")
    comp.add_argument("--out", type=Path, default=None)
    comp.set_defaults(func=cmd_compare)

    # export
    exp = sub.add_parser("export",
                         help="write soundings as SAO.XML 5.0 (URSI/INAG)")
    exp.add_argument("target", type=Path, nargs="+")
    exp.add_argument("--out", type=Path, default=Path("sao"))
    exp.add_argument("--methods", type=_methods, default=DEFAULT_METHODS,
                     help="one SAORecord per method, per the spec's "
                          "separate-storage rule (default: %(default)s)")
    exp.add_argument("--ursi-code", type=str, default="",
                     help="URSI station code, if one has been issued for "
                          "this path (default: empty)")
    exp.add_argument("--station", type=str, default=None,
                     help="StationName attribute (default: the receiver name)")
    exp.add_argument("--iri", action="store_true",
                     help="add IRI's MUF, foF2 and hmF2 as <Modeled> "
                          "characteristics, beside the measured ones")
    exp.add_argument("--offline", action="store_true",
                     help="with --iri, use cached solar indices only")
    _common(exp)
    exp.set_defaults(func=cmd_export)

    # plot-sao
    psao = sub.add_parser("plot-sao",
                          help="render exported SAO.XML records as images")
    psao.add_argument("target", type=Path, nargs="+",
                      help=".xml files or directories of them, from `export`")
    psao.add_argument("--out", type=Path, default=Path("imgs/sao"))
    psao.add_argument("--ionogram", type=Path, nargs="+", default=None,
                      metavar="TARGET",
                      help=".lfs files or directories; each record is drawn "
                           "over the sounding it was scaled from, matched by "
                           "file name")
    psao.add_argument("--method", default=None,
                      help="draw only the record from this estimator "
                           "(default: every record in the file)")
    psao.add_argument("--full-band", action="store_true",
                      help="show the whole sweep instead of framing the trace")
    psao.add_argument("--trace", action="store_true",
                      help="overlay the scaled trace points on the ionogram")
    psao.add_argument("--no-trace", action="store_true",
                      help="omit the trace points even without an ionogram")
    psao.add_argument("--dpi", type=int, default=render.DEFAULT_DPI)
    _common(psao)
    psao.set_defaults(func=cmd_plot_sao)

    # lof
    low = sub.add_parser("lof",
                         help="measure the band floor and summarise LOF")
    low.add_argument("target", type=Path, nargs="+")
    low.add_argument("--level", type=float, default=None,
                     help="detection level for the floor measurement, dB "
                          "(default: the first ladder rung)")
    low.add_argument("--out", type=Path, default=None,
                     help="write the per-sounding ladder to this CSV")
    _common(low)
    low.set_defaults(func=cmd_lof)

    # info
    info = sub.add_parser("info", help="show header and derived geometry")
    info.add_argument("target", type=Path, nargs="+")
    info.add_argument("--limit", type=int, default=3)
    info.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW)
    info.add_argument("--zero-periods", type=int,
                      default=spectro.DEFAULT_ZERO_PERIODS)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
