"""Driving the estimators over files and days.

One spectrogram is formed per sounding and shared by every estimator, so adding
a method costs almost nothing beyond the method itself -- the FFTs dominate.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import extractors, fit, lof as lof_module, pick as pick_module, spectro, trace
from .extractors import DEFAULT_METHODS
from . import loader
from .loader import find_soundings, read_header

#: A pick within this many frequency bins of the top of the sweep means the
#: trace ran off the end of the band: the true MUF is at or above the highest
#: sounded frequency and the instrument cannot see it. Recorded rather than
#: reported as a measurement.
#:
#: Counted in bins rather than MHz because estimators can legitimately land a
#: bin or two short of the edge -- the algorithmic one anchors on the middle of
#: a three-bin run, so it can never reach the final bin -- and a fixed MHz
#: tolerance would mean different things at different window lengths.
BAND_EDGE_BINS = 3


@dataclass
class Options:
    """Everything that affects a sounding's result."""

    window: int = spectro.DEFAULT_WINDOW
    zero_periods: int = spectro.DEFAULT_ZERO_PERIODS
    gate_km: tuple[float, float] | None = None
    methods: tuple[str, ...] = DEFAULT_METHODS
    min_run: int | None = None
    percentile: float | None = None
    cache_dir: Path | None = None
    method_options: dict[str, dict] | None = None
    fit: bool = True
    segment: bool = True
    lof: bool = True
    #: Lowest frequency the circuit actually radiates. None means the sweep
    #: start, which is right only when the transmitter really does reach it --
    #: see muf.lof.measure_band_floor.
    band_floor_mhz: float | None = None
    #: Override extension-based format selection. None means dispatch on the
    #: suffix, which is right unless a recording was renamed.
    format: str | None = None
    #: Station coordinate registry, for v2 products whose header carries only
    #: a transmitter *name*. None leaves the geometry unavailable rather than
    #: guessed -- see io_chirp.ChirpHeader.has_coordinates.
    stations: dict | None = None

    def per_method(self) -> dict[str, dict]:
        """Method keyword arguments, with the shared picker settings folded in."""
        shared: dict[str, object] = {}
        if self.min_run is not None:
            shared["min_run"] = self.min_run
        if self.percentile is not None:
            shared["percentile"] = self.percentile

        out: dict[str, dict] = {}
        for name in self.methods:
            opts = dict(shared)
            opts.update((self.method_options or {}).get(name, {}))
            out[name] = opts
        return out


def _path_km(header) -> float:
    from .geometry import path_of

    return path_of(header)[2]


def process_file(path: str | Path, options: Options | None = None) -> dict:
    """Run every selected estimator over one sounding. Returns one result row."""
    options = options or Options()
    path = Path(path)

    row: dict[str, object] = {"file": path.name}
    try:
        header = read_header(path, format=options.format,
                             stations=options.stations)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row.update(
        datetime=header.datetime.replace(tzinfo=None),
        tx=header.tx_name,
        rx=header.rx_name,
        path_type=header.path_type,
        # Carried so `muf compare --ref-model` can locate the path's control
        # point without going back to the .lfs files.
        tx_lat=round(header.tx_latitude, 4),
        tx_lon=round(header.tx_longitude, 4),
        rx_lat=round(header.rx_latitude, 4),
        rx_lon=round(header.rx_longitude, 4),
        path_km=round(_path_km(header), 1),
    )

    try:
        ion = loader.load(
            path,
            window=options.window,
            zero_periods=options.zero_periods,
            gate_km=options.gate_km,
            cache_dir=options.cache_dir,
            format=options.format,
            header=header,
            stations=options.stations,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row.update(
        freq_start=round(ion.cal.freq_start, 4),
        freq_stop=round(ion.cal.freq_stop, 4),
        gate_lo=round(ion.cal.gate_km[0], 1),
        gate_hi=round(ion.cal.gate_km[1], 1),
        # A recording cut short still declares the full sweep in its header.
        # Its MUF is capped by where the recording stopped, so it is a lower
        # bound like a band-limited pick, and is flagged the same way.
        sweep_complete=ion.cal.sweep_complete,
        sweep_fraction=round(ion.cal.sweep_fraction, 4),
    )

    results = extractors.run(ion, methods=options.methods, **options.per_method())
    band_edge = ion.cal.freq_stop - BAND_EDGE_BINS * ion.cal.freq_step_mhz

    for name, result in results.items():
        pick = result.pick
        row[f"muf_{name}"] = pick.muf_mhz
        row[f"vrange_{name}"] = pick.vrange_km
        row[f"ndet_{name}"] = pick.n_detections
        row[f"run_{name}"] = pick.run_len
        row[f"snr_{name}"] = pick.snr_db

        # Fitting the nose costs a polyfit over a couple of hundred points, so
        # it is always computed: the residual is a useful quality signal even
        # when the vertex itself is not trustworthy.
        # Segmenting reveals when the "trace" is really several propagation
        # modes stitched together -- which the extractors cannot see, since
        # their continuity rule looks only along frequency.
        segments = None
        if options.segment and result.ok:
            freq, vrange, weight = trace.extract_points(ion, result)
            segments = trace.merge_branches(trace.identify_hops(
                trace.group_tracks(freq, vrange, weight), row["path_km"]))
            primary = trace.primary_segment(segments)
            row[f"nseg_{name}"] = len({s.group for s in segments})
            row[f"hops_{name}"] = primary.hops if primary else None
            row[f"branch_{name}"] = (
                sum(1 for s in segments if s.branch == "high") > 0)
            row[f"scatter_{name}"] = round(
                trace.trace_scatter_km(primary.vrange) if primary else np.nan, 1)

        if options.fit:
            # Fit both branches of the nose when they were found: with the
            # vertex bracketed by data rather than extrapolated off one side,
            # the fit is a far better estimator.
            if segments is not None:
                nose = fit.fit_nose(*trace.nose_points(segments))
            else:
                nose = fit.fit_result(ion, result)
            row[f"fit_{name}"] = nose.muf_mhz if nose.ok else np.nan
            row[f"fitres_{name}"] = nose.rms_residual_mhz
            row[f"fitex_{name}"] = nose.extrapolation_mhz if nose.ok else np.nan
        # The trace reaching the top of the sweep means the MUF is at or above
        # it; the value is a lower bound, not a measurement.
        row[f"limited_{name}"] = bool(pick.ok and pick.muf_mhz >= band_edge)

        # The low-frequency end of this estimator's own trace, so LOF and MUF
        # describe the same detected set and can be compared per method.
        if options.lof:
            low = lof_module.pick_lof(
                result.presence, ion.freq, power_db=ion.db, vrange=ion.vrange,
                min_run=options.min_run or pick_module.DEFAULT_MIN_RUN,
                band_floor_mhz=options.band_floor_mhz,
            )
            row[f"lof_{name}"] = low.lof_mhz
            row[f"lofsnr_{name}"] = low.snr_db
            # At the floor the true LOF is below the band: an upper bound, the
            # mirror image of limited_.
            row[f"loflim_{name}"] = bool(low.at_band_floor)

        if result.error:
            row[f"err_{name}"] = result.error

    # Estimator-independent, straight off the ionogram: these stay comparable
    # when the estimators change, which the per-method columns do not.
    if options.lof:
        for level, low in lof_module.ladder(
            ion, min_run=options.min_run or pick_module.DEFAULT_MIN_RUN,
            band_floor_mhz=options.band_floor_mhz,
        ).items():
            row[f"lof{int(round(level))}"] = low.lof_mhz

    return row


def _worker(args: tuple[Path, Options]) -> dict:
    return process_file(*args)


def process_many(
    target,
    options: Options | None = None,
    jobs: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the pipeline over files, directories, or several of either.

    Several days can be given at once; the result is one table covering them
    all, sorted by time. Use :func:`split_by_day` to write it out per day.
    """
    options = options or Options()
    paths = find_soundings(target, format=options.format)
    if not paths:
        raise FileNotFoundError(f"no soundings under {target}")

    if jobs <= 0:
        jobs = max(1, (os.cpu_count() or 2) - 1)

    rows: list[dict] = []
    if jobs == 1:
        iterator = (process_file(p, options) for p in paths)
        rows = list(_maybe_progress(iterator, len(paths), progress))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            iterator = pool.map(_worker, [(p, options) for p in paths])
            rows = list(_maybe_progress(iterator, len(paths), progress))

    frame = pd.DataFrame(rows)
    if "datetime" in frame:
        frame = frame.sort_values("datetime").reset_index(drop=True)
    return frame


def _maybe_progress(iterator, total: int, enabled: bool):
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm
    except ImportError:
        return iterator
    return tqdm(iterator, total=total, unit="sounding")


# --- daily aggregation -------------------------------------------------------
#
# Reimplemented rather than imported: muf_interpolation.py and
# data_handler/muf_data_handler.py both import psycopg2 at module scope, so
# neither can be imported without a database driver installed. The logic they
# contain is small enough that duplicating it costs less than the coupling.

DEFAULT_GRID_MINUTES = 5
DEFAULT_SAVGOL_WINDOW = 19
DEFAULT_SAVGOL_ORDER = 1


def days_in(frame: pd.DataFrame) -> list:
    """The distinct dates present in a results table, in order.

    A sounding whose header could not be read has no ``datetime`` -- the row
    carries ``error`` instead -- so the column holds NaT and sorting it raises.
    One truncated file must not abort the day it was found in: a recorder
    killed mid-write leaves exactly one, and the other 318 soundings are fine.
    """
    if "datetime" not in frame or frame.empty:
        return []
    dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.date
    return sorted(d for d in dates.dropna().unique())


def split_by_day(frame: pd.DataFrame):
    """Yield ``(date, sub_frame)`` for each day in a results table.

    Rows with no usable timestamp belong to no day and are dropped here; they
    are still counted in the run summary, and ``error`` says why.
    """
    if "datetime" not in frame or frame.empty:
        return
    dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.date
    for day in sorted(d for d in dates.dropna().unique()):
        yield day, frame[dates == day].reset_index(drop=True)


def daily(
    frame: pd.DataFrame,
    method: str | None = None,
    grid_minutes: int = DEFAULT_GRID_MINUTES,
    smooth: bool = True,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_order: int = DEFAULT_SAVGOL_ORDER,
    drop_limited: bool = True,
) -> pd.DataFrame:
    """Resample MUF onto a regular grid and smooth it.

    Akima interpolation then linear fill, matching ``muf_interpolation.py:35``;
    Savitzky-Golay smoothing matching ``muf_data_handler.py:74``.

    A table spanning several days is handled a day at a time and the results
    concatenated, so a missing day leaves a gap rather than being interpolated
    across. Each row carries the ``date`` it belongs to.

    Band-limited picks are dropped by default: they are lower bounds, and
    treating them as measurements flattens the midday peak.
    """
    method = method or _first_method(frame)
    column = f"muf_{method}"
    if column not in frame:
        raise KeyError(f"no results for method {method!r} in this table")

    pieces, failures = [], []
    for day, sub in split_by_day(frame):
        try:
            pieces.append(_daily_one(sub, method, column, grid_minutes, smooth,
                                     savgol_window, savgol_order, drop_limited))
        except ValueError as exc:
            failures.append(f"{day}: {exc}")

    if not pieces:
        raise ValueError(
            f"method {method!r} produced no usable values"
            + (" (" + "; ".join(failures) + ")" if failures else "")
        )

    out = pd.concat(pieces, ignore_index=True)
    out.attrs["skipped_days"] = failures
    return out


def _daily_one(
    frame: pd.DataFrame, method: str, column: str, grid_minutes: int,
    smooth: bool, savgol_window: int, savgol_order: int, drop_limited: bool,
) -> pd.DataFrame:
    """One day's curve. See :func:`daily`."""
    from scipy.signal import savgol_filter

    from .compare import flag

    series = frame[["datetime", column]].copy()
    if drop_limited and f"limited_{method}" in frame:
        series = series[~flag(frame, f"limited_{method}").to_numpy()]
    series = series.dropna()
    if series.empty:
        raise ValueError("no usable values")

    series["datetime"] = pd.to_datetime(series["datetime"])
    series = series.set_index("datetime").sort_index()

    day = series.index[0].normalize()
    grid = pd.date_range(day, day + pd.Timedelta(days=1),
                         freq=f"{grid_minutes}min", inclusive="left")

    joined = series[column].reindex(series.index.union(grid))
    joined = joined.interpolate(method="akima", limit_direction="both")
    joined = joined.interpolate(method="linear", limit_direction="both")
    out = pd.DataFrame({"muf": joined.reindex(grid).round(3)}, index=grid)
    out.index.name = "datetime"

    if smooth and len(out) > savgol_window:
        window = savgol_window + (savgol_window % 2 == 0)   # must be odd
        out["muf_smooth"] = savgol_filter(
            out["muf"].to_numpy(), window, savgol_order
        ).round(3)

    out = out.reset_index()
    out.insert(1, "date", day.date())
    out["method"] = method
    return out


def _first_method(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if isinstance(column, str) and column.startswith("muf_"):
            return column[4:]
    raise KeyError("this table has no muf_* columns")


def methods_in(frame: pd.DataFrame) -> list[str]:
    """Method names present in a results table."""
    return [c[4:] for c in frame.columns
            if isinstance(c, str) and c.startswith("muf_")]


def write(frame: pd.DataFrame, path: str | Path, fmt: str = "csv") -> Path:
    """Write a results table as CSV or Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
    elif fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return path
