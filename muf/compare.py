"""Comparing estimators against each other and against a reference series.

The proper version of ``MUF_clustering/stat.py``, which aligned an Excel of
clustering results against a CSV of DSP results and printed RMSE / MAE / R2.
Differences from that script:

* every pair of methods is compared, not one hardcoded pair;
* the date exclusions it hardcoded (``stat.py:57-107``) become a parameter;
* ``stat.py:119`` labels a column ``MAE`` that actually holds the row-wise
  *mean* of the two series -- that column is not a metric and is dropped;
* band-limited picks are excluded, since they are lower bounds rather than
  measurements and would otherwise flatter agreement at the midday peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Agreement:
    """How closely two series of MUF values agree."""

    left: str
    right: str
    n: int
    rmse: float
    mae: float
    bias: float          # mean(left - right); sign says which reads higher
    r2: float
    max_abs: float

    def as_row(self) -> dict:
        return {
            "a": self.left, "b": self.right, "n": self.n,
            "rmse_mhz": round(self.rmse, 4),
            "mae_mhz": round(self.mae, 4),
            "bias_mhz": round(self.bias, 4),
            "r2": round(self.r2, 4),
            "max_abs_mhz": round(self.max_abs, 4),
        }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small table as Markdown.

    pandas' ``to_markdown`` needs ``tabulate``; that is a lot of dependency for
    a pipe-separated table.
    """
    columns = [str(c) for c in frame.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row]
            for row in frame.itertuples(index=False)]
    widths = [
        max([len(columns[i])] + [len(r[i]) for r in rows])
        for i in range(len(columns))
    ]

    def line(cells) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    return "\n".join(
        [line(columns), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        + [line(r) for r in rows]
    )


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination, treating ``actual`` as the reference."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def agreement(left: pd.Series, right: pd.Series, name_a: str, name_b: str) -> Agreement:
    """Compare two aligned series, ignoring rows either one is missing."""
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna()
    if joined.empty:
        return Agreement(name_a, name_b, 0, *([float("nan")] * 5))

    a = joined["a"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    diff = a - b
    return Agreement(
        left=name_a, right=name_b, n=len(joined),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        mae=float(np.mean(np.abs(diff))),
        bias=float(np.mean(diff)),
        r2=_r2(b, a),
        max_abs=float(np.max(np.abs(diff))),
    )


def flag(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read a boolean column robustly.

    A round trip through CSV turns ``True``/``False`` into strings, and blanks
    into NaN, so the column comes back as object dtype. Coercing explicitly
    avoids depending on pandas' deprecated implicit downcasting.
    """
    if name not in frame:
        return pd.Series(False, index=frame.index)
    column = frame[name]
    if column.dtype == bool:
        return column
    return column.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def usable(frame: pd.DataFrame, method: str, drop_limited: bool = True) -> pd.Series:
    """A method's MUF series, with band-limited picks removed."""
    values = frame[f"muf_{method}"].astype(float)
    if drop_limited and f"limited_{method}" in frame:
        values = values.mask(flag(frame, f"limited_{method}"))
    return values


def compare_methods(
    frame: pd.DataFrame,
    methods: list[str] | None = None,
    drop_limited: bool = True,
) -> pd.DataFrame:
    """Pairwise agreement between every method in a results table."""
    from .pipeline import methods_in

    methods = methods or methods_in(frame)
    rows = [
        agreement(
            usable(frame, a, drop_limited), usable(frame, b, drop_limited), a, b
        ).as_row()
        for a, b in combinations(methods, 2)
    ]
    return pd.DataFrame(rows)


def summarise_methods(
    frame: pd.DataFrame,
    methods: list[str] | None = None,
    drop_limited: bool = True,
) -> pd.DataFrame:
    """Per-method coverage and range."""
    from .pipeline import methods_in

    methods = methods or methods_in(frame)
    rows = []
    for name in methods:
        values = usable(frame, name, drop_limited)
        limited = int(flag(frame, f"limited_{name}").sum())
        errors = int(frame[f"err_{name}"].notna().sum()) \
            if f"err_{name}" in frame else 0
        rows.append({
            "method": name,
            "n_soundings": len(frame),
            "n_picked": int(values.notna().sum()),
            "coverage_pct": round(100 * values.notna().mean(), 1),
            "n_band_limited": limited,
            "n_errors": errors,
            "muf_min": round(float(values.min()), 2) if values.notna().any() else np.nan,
            "muf_median": round(float(values.median()), 2) if values.notna().any() else np.nan,
            "muf_max": round(float(values.max()), 2) if values.notna().any() else np.nan,
            "median_run": (round(float(frame[f"run_{name}"].median()), 1)
                           if f"run_{name}" in frame else np.nan),
        })
    return pd.DataFrame(rows)


def read_reference(path: str | Path) -> pd.Series:
    """Read a historical reference series.

    Handles the ``MUF_cyprus1_<date>.csv`` layout written by ``MUF.py:198``:
    space-delimited, no header, columns ``MUF time vrange``. Also accepts a
    normal CSV with ``time``/``datetime`` and ``muf`` columns.
    """
    path = Path(path)
    head = path.read_text(encoding="utf-8", errors="replace").lstrip()[:200].lower()

    if "muf" in head.split("\n")[0] or "," in head.split("\n")[0]:
        frame = pd.read_csv(path)
        time_col = next(
            (c for c in frame.columns if c.lower() in ("time", "datetime")), None
        )
        muf_col = next((c for c in frame.columns if c.lower().startswith("muf")), None)
        if time_col is None or muf_col is None:
            raise ValueError(f"{path}: expected time and muf columns")
        times, values = frame[time_col], frame[muf_col]
    else:
        frame = pd.read_csv(path, sep=r"\s+", header=None,
                            names=["muf", "time", "vrange"], usecols=[0, 1, 2])
        times, values = frame["time"], frame["muf"]

    index = pd.to_datetime(times, format="mixed", errors="coerce")
    series = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=index)
    return series[series.index.notna()].sort_index()


def align_reference(frame: pd.DataFrame, reference: pd.Series) -> pd.Series:
    """Align a reference series onto a results table by time of day.

    Time of day rather than full timestamp, so a reference from a different
    date can still be compared -- which is the usual case when checking today's
    run against a historical day.
    """
    ours = pd.to_datetime(frame["datetime"])
    key = ours.dt.strftime("%H:%M:%S")
    lookup = pd.Series(reference.to_numpy(),
                       index=pd.Index(reference.index).strftime("%H:%M:%S"))
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    return pd.Series(key.map(lookup).to_numpy(), index=frame.index, dtype=float)


def add_reference_models(
    frame: pd.DataFrame,
    models,
    primary: str | None = None,
    **options,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate external reference models and add them as ``muf_<name>`` columns.

    Returns the widened frame and the names that produced data. Models that are
    unavailable are reported and skipped rather than failing the run.
    """
    from . import reference as refmod
    from .geometry import Point
    from .pipeline import methods_in

    if "tx_lat" in frame and "rx_lat" in frame:
        tx = Point(float(frame["tx_lat"].iloc[0]), float(frame["tx_lon"].iloc[0]))
        rx = Point(float(frame["rx_lat"].iloc[0]), float(frame["rx_lon"].iloc[0]))
    else:
        raise KeyError(
            "this results table has no tx_lat/tx_lon/rx_lat/rx_lon columns, so "
            "the path geometry is unknown -- re-run `muf run` to regenerate it"
        )

    times = pd.to_datetime(frame["datetime"])
    primary = primary or (methods_in(frame) or ["algo"])[0]

    per_model = dict(options)
    per_model.setdefault("chapman", {})["observed"] = usable(frame, primary)

    frame = frame.copy()
    added, problems = [], []
    for name, series in refmod.run(models, tx, rx, times, **per_model).items():
        if not series.ok:
            problems.append(f"{name}: {series.error}")
            continue
        frame[f"muf_{name}"] = series.muf.to_numpy()
        frame.attrs.setdefault("reference_sources", {})[name] = series.source
        added.append(name)

    frame.attrs["reference_problems"] = problems
    return frame, added


def band_limited_by_reference(
    frame: pd.DataFrame, model: str, margin_mhz: float = 0.0
) -> pd.Series:
    """Soundings where a reference puts the MUF above what the sounder can see.

    The pipeline's own ``limited_`` flag only fires when a pick lands at the top
    of the band. A trace that fades below the top for signal-strength reasons
    looks like a valid measurement even when the true MUF is out of band; only
    an external reference can tell the difference.

    "Out of band" means above what the *circuit* returns, so this prefers the
    ``band_ceiling`` column the pipeline records and falls back to ``freq_stop``
    only for frames written before that column existed. The header value can sit
    well above anything the receiver ever saw -- 24.825 against 24.55 on DOB's
    Cyprus path -- and using it understates how often a reference is out of
    reach.
    """
    column = f"muf_{model}"
    if column not in frame:
        return pd.Series(False, index=frame.index)
    for anchor in ("band_ceiling", "freq_stop"):
        if anchor in frame:
            return frame[column] > (frame[anchor] + margin_mhz)
    return pd.Series(False, index=frame.index)


def report(
    frame: pd.DataFrame,
    reference: pd.Series | None = None,
    reference_name: str = "reference",
    exclude: list[tuple[str, str]] | None = None,
    drop_limited: bool = True,
    reference_models: list[str] | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Build a Markdown comparison report.

    Returns ``(markdown, summary_table, pairwise_table)``.
    """
    from .pipeline import methods_in

    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    for start, stop in exclude or []:
        span = (frame["datetime"] >= pd.Timestamp(start)) & \
               (frame["datetime"] <= pd.Timestamp(stop))
        frame = frame[~span]

    methods = methods_in(frame)

    model_names: list[str] = []
    if reference_models:
        frame, model_names = add_reference_models(frame, reference_models)
        methods = methods + model_names

    if reference is not None:
        frame["muf_" + reference_name] = align_reference(frame, reference)
        methods = methods + [reference_name]

    summary = summarise_methods(frame, methods, drop_limited)
    pairwise = compare_methods(frame, methods, drop_limited)

    days = sorted(frame["datetime"].dt.date.unique())
    span = f"{days[0]}" if len(days) == 1 else f"{days[0]} to {days[-1]}"
    when = (f"{frame['datetime'].min():%H:%M} to {frame['datetime'].max():%H:%M} UTC"
            if len(days) == 1
            else f"{len(days)} days, "
                 f"{frame['datetime'].min():%Y-%m-%d %H:%M} to "
                 f"{frame['datetime'].max():%Y-%m-%d %H:%M} UTC")
    lines = [
        f"# MUF method comparison - {span}",
        "",
        f"{len(frame)} soundings, {when}.",
        "",
        "## Coverage",
        "",
        _markdown_table(summary),
        "",
        "## Pairwise agreement",
        "",
        "`bias` is mean(a - b): positive means *a* reads higher than *b*.",
        "",
        _markdown_table(pairwise),
        "",
    ]

    for name in model_names:
        source = frame.attrs.get("reference_sources", {}).get(name, "")
        lines += [f"*{name}*: {source}", ""]

    for problem in frame.attrs.get("reference_problems", []):
        lines += [f"*unavailable* -- {problem}", ""]

    # An external reference can reveal soundings whose true MUF was above the
    # sweep, which the pipeline's own flag cannot see.
    for name in model_names:
        over = band_limited_by_reference(frame, name)
        if not over.any():
            continue
        hours = sorted({int(h) for h in pd.to_datetime(frame["datetime"])[over].dt.hour})
        own = int(sum(flag(frame, f"limited_{m}").sum() for m in methods_in(frame)))
        lines += [
            "## Out-of-band soundings",
            "",
            f"`{name}` puts the MUF above the {frame['freq_stop'].iloc[0]:.1f} MHz "
            f"sweep in **{int(over.sum())} of {len(frame)}** soundings "
            f"(hours {', '.join(f'{h:02d}' for h in hours)} UTC). The pipeline's "
            f"own band-limit flag caught {own}: it fires only when a pick lands "
            f"at the top of the sweep, so a trace fading below it for "
            f"signal-strength reasons still looks like a measurement. Values in "
            f"those hours are lower bounds.",
            "",
        ]

    if drop_limited and any(f"limited_{m}" in frame for m in methods):
        limited_total = int(sum(flag(frame, f"limited_{m}").sum() for m in methods))
        if limited_total:
            lines += [
                "## Band-limited soundings",
                "",
                f"{limited_total} picks landed at the top of the sweep "
                f"({frame['freq_stop'].iloc[0]:.1f} MHz) and were excluded: the "
                "MUF was at or above the highest sounded frequency, so those "
                "values are lower bounds rather than measurements.",
                "",
            ]

    return "\n".join(lines), summary, pairwise
