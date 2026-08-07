"""Loading ``muf`` pipeline output into the database.

``architecture.md`` sec. 5.2 describes the schema as "the normalized form of
the 68-column run CSV", and that is exactly what this does: one ``sounding``
row per file, one ``extraction`` row per (sounding, method).

**Idempotent on ``(file, method)``.** Re-ingesting a directory must be
harmless, because it is the first thing anyone does when a run looks wrong.
The sounding is upserted on its file name and extractions are replaced, so a
re-run with different estimator settings updates in place rather than
doubling the series.

Runs the pipeline in-process rather than shelling out to ``muf run`` and
parsing CSV. The CSV is a presentation format -- floats already rounded, NaN
already stringified -- and going through it would mean the database inherits
the rounding.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

from . import db

#: Columns copied straight from a pipeline row into `sounding`.
SOUNDING_COLUMNS = (
    "tx", "rx", "path_type", "tx_lat", "tx_lon", "rx_lat", "rx_lon", "path_km",
    "freq_start", "freq_stop", "gate_lo", "gate_hi", "sweep_fraction",
)

#: ``(database column, pipeline suffix)`` for each per-method value.
EXTRACTION_COLUMNS = (
    ("muf", "muf"), ("lof", "lof"), ("vrange", "vrange"), ("snr", "snr"),
    ("ndet", "ndet"), ("run", "run"), ("nseg", "nseg"), ("hops", "hops"),
    ("branch", "branch"), ("scatter", "scatter"),
    ("fit", "fit"), ("fitres", "fitres"), ("fitex", "fitex"),
    ("limited", "limited"), ("loflim", "loflim"),
)


def _clean(value):
    """NaN and NaT become NULL.

    A NaN stored in a REAL column compares false against everything including
    itself, so ``WHERE muf IS NOT NULL`` would return rows with no MUF. NULL is
    what "no pick" means in SQL, and the distinction is the whole reason the
    pipeline reports NaN rather than zero.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool,)):
        return int(value)
    # pandas NaT, which is neither None nor a float NaN but means the same
    # thing: no value. Detected by name so this module does not import pandas
    # for one check.
    if value.__class__.__name__ == "NaTType":
        return None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
            if isinstance(value, float) and math.isnan(value):
                return None
            if isinstance(value, bool):
                return int(value)
    except ImportError:                                       # pragma: no cover
        pass
    return value


def ingest_row(conn: sqlite3.Connection, row: dict, path: Path,
               archive_root: Path, methods) -> int | None:
    """One pipeline row into ``sounding`` + ``extraction``. Returns the id."""
    # `_clean`, not a bare truth test: pandas fills the absent `error` cell of
    # a *successful* row with NaN, and `bool(nan)` is True. Testing it directly
    # marks every good sounding as failed and ingests nothing.
    if _clean(row.get("error")):
        return None
    when = _clean(row.get("datetime"))
    if when is None or (hasattr(when, "__class__")
                        and when.__class__.__name__ == "NaTType"):
        return None

    try:
        relative = path.resolve().relative_to(archive_root.resolve()).as_posix()
    except ValueError:
        # Outside the configured root. Store the absolute path rather than
        # refusing: the renderer will still find it on this host, and a
        # half-ingested archive is worse than a non-portable row.
        relative = path.resolve().as_posix()

    values = {name: _clean(row.get(name)) for name in SOUNDING_COLUMNS}
    conn.execute(
        "INSERT INTO sounding (file, path, format, window, datetime,"
        " sweep_complete, ingested_at, " + ", ".join(SOUNDING_COLUMNS) + ")"
        " VALUES (?, ?, ?, ?, ?, ?, ?, " + ", ".join("?" * len(SOUNDING_COLUMNS)) + ")"
        " ON CONFLICT(file) DO UPDATE SET"
        " path=excluded.path, format=excluded.format, window=excluded.window,"
        " datetime=excluded.datetime, sweep_complete=excluded.sweep_complete,"
        " ingested_at=excluded.ingested_at, "
        + ", ".join(f"{c}=excluded.{c}" for c in SOUNDING_COLUMNS),
        (row["file"], relative, row.get("format"), _clean(row.get("window")),
         str(when), _clean(row.get("sweep_complete")), db.utcnow(),
         *[values[name] for name in SOUNDING_COLUMNS]),
    )
    found = db.one(conn, "SELECT id FROM sounding WHERE file = ?", (row["file"],))
    sounding_id = int(found["id"])

    for method in methods:
        if f"muf_{method}" not in row:
            continue
        payload = [_clean(row.get(f"{suffix}_{method}"))
                   for _, suffix in EXTRACTION_COLUMNS]
        conn.execute(
            "INSERT OR REPLACE INTO extraction (sounding_id, method, "
            + ", ".join(column for column, _ in EXTRACTION_COLUMNS)
            + ") VALUES (?, ?, " + ", ".join("?" * len(EXTRACTION_COLUMNS)) + ")",
            (sounding_id, method, *payload),
        )
    return sounding_id


def ingest(targets, conn: sqlite3.Connection, options=None,
           archive_root: Path | None = None, jobs: int = 1,
           progress: bool = True) -> dict:
    """Run the pipeline over ``targets`` and load the result. Returns counts."""
    from muf import pipeline

    options = options or pipeline.Options()
    archive_root = Path(archive_root or db.ARCHIVE_ROOT)

    frame = pipeline.process_many(targets, options, jobs=jobs, progress=progress)
    by_name = {}
    for target in targets:
        for path in pipeline.find_soundings([target], format=options.format):
            by_name[path.name] = path

    loaded = skipped = 0
    for record in frame.to_dict("records"):
        path = by_name.get(record.get("file"))
        if path is None:
            skipped += 1
            continue
        if ingest_row(conn, record, path, archive_root, options.methods) is None:
            skipped += 1
        else:
            loaded += 1
    conn.commit()
    return {"loaded": loaded, "skipped": skipped, "total": len(frame)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="services.api.ingest",
        description="Run the muf pipeline over an archive and load the "
                    "results into the api database.")
    parser.add_argument("target", nargs="+", type=Path)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--archive-root", type=Path, default=None,
                        help="sounding.path is stored relative to this so the "
                             "same database works on the host and in a "
                             "container (default: $ARCHIVE_ROOT)")
    parser.add_argument("--methods", default=None,
                        help="comma separated (default: the pipeline default)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from muf import pipeline
    from muf.extractors import DEFAULT_METHODS

    methods = (tuple(m.strip() for m in args.methods.split(",") if m.strip())
               if args.methods else DEFAULT_METHODS)
    options = pipeline.Options(methods=methods)

    with db.session(args.db) as conn:
        counts = ingest(args.target, conn, options,
                        archive_root=args.archive_root or db.ARCHIVE_ROOT,
                        jobs=args.jobs, progress=not args.quiet)
    print(f"loaded {counts['loaded']} sounding(s), skipped {counts['skipped']}"
          f" of {counts['total']}", file=sys.stderr)
    return 0 if counts["loaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
