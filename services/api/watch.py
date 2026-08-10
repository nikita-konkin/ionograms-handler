"""Ingest whatever is new in the archive, and nothing else.

``services.api.ingest`` re-runs the pipeline over every sounding it is handed.
That is the right behaviour for a deliberate reload, and the wrong one for a
recurring check: pointing it at the archive root once an hour would re-derive
the whole history each time, and the cost grows with the archive rather than
with what arrived.

This narrows the target list first. It enumerates what is on disk, asks the
database what it already holds, and hands ``ingest`` only the difference --
so the usual result is "nothing to do" at the cost of a directory scan and one
query.

**Idempotent on ``(file, method)``**, the same key ``ingest`` upserts on, and
for the same reason: re-running after a crash, a partial sync or a config
change must be harmless. A sounding is considered done when every requested
method has an ``extraction`` row for it, so adding a method to ``--methods``
brings the older soundings back into scope without a manual reload.

Run it from cron for one pass, or with ``--interval`` to stay resident::

    python -m services.api.watch /archive --db /data/ionograms.sqlite3 \\
        --archive-root /archive --methods algo,kmeans,contour --interval 900
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from . import db

#: Skip anything modified more recently than this. A file still being written
#: by the recorder, or still arriving over a sync, reads as a truncated
#: sounding -- and a truncated sounding does not fail loudly. It ingests as a
#: short sweep, which `sweep_complete` records but nothing rejects. Waiting one
#: minute costs one cycle and removes the whole class.
DEFAULT_MIN_AGE_S = 60.0

#: A file stamped further ahead of us than this is not recent, it is
#: mis-stamped, and `now - mtime` says nothing about whether writing finished.
#:
#: DOB's archive moved to a CIFS share whose NAS clock ran 5 h 43 m fast. Every
#: product's age came out negative, negative beats any threshold, and the
#: watcher skipped the entire archive on every pass -- reporting it as "too
#: fresh", which is the most reassuring possible word for "nothing will ever be
#: ingested". Timestamps on a network share belong to the file server; this
#: watcher must not assume they belong to it.
FUTURE_MTIME_TOLERANCE_S = 5.0

#: How long to wait for a writer to release the database before giving up.
#: The api reads on every request and SQLite locks the whole file, so a
#: default-timeout connection loses this race often enough to matter.
BUSY_TIMEOUT_MS = 30_000


def connect(path: Path | None) -> sqlite3.Connection:
    conn = db.connect(path)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return db.init(conn)


def already_done(conn: sqlite3.Connection, methods: tuple[str, ...]) -> set[str]:
    """File names holding an extraction for *every* requested method.

    Anything short of that is incomplete and worth revisiting -- a run that
    died midway, or a method added since. `sounding.file` is the basename,
    which is what `ingest` keys on.
    """
    wanted = set(methods)
    seen: dict[str, set[str]] = {}
    for row in db.rows(conn, "SELECT s.file AS file, e.method AS method"
                             " FROM sounding s"
                             " JOIN extraction e ON e.sounding_id = s.id"):
        seen.setdefault(row["file"], set()).add(row["method"])
    return {name for name, got in seen.items() if wanted <= got}


def find_new(targets, conn, methods, min_age_s: float, now: float | None = None):
    """Soundings on disk that the database does not already hold.

    Returns ``(new, n_found, n_too_fresh, n_skewed)``. Targets holding no
    soundings at all are skipped rather than fatal: an archive normally
    contains detection trees, digisonde products and empty days beside the
    ionograms, and one of those must not stop the scan.
    """
    from muf import loader

    now = time.time() if now is None else now
    done = already_done(conn, methods)

    found, fresh, skewed, new = 0, 0, 0, []
    for target in targets:
        try:
            paths = loader.find_soundings(target)
        except FileNotFoundError:
            continue                      # nothing this reader recognises
        for path in paths:
            found += 1
            if path.name in done:
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue                  # vanished mid-scan; next cycle
            if age < -FUTURE_MTIME_TOLERANCE_S:
                # Withholding is the worse guess here. A mis-stamped file held
                # back is held back forever, while one taken mid-write fails to
                # parse and simply returns on the next pass -- which is exactly
                # what `skipped` already exists to absorb.
                skewed += 1
            elif age < min_age_s:
                fresh += 1
                continue
            new.append(path)
    new.sort(key=lambda p: p.name)
    return new, found, fresh, skewed


def run_once(targets, conn, *, methods, archive_root, jobs=1, batch=0,
             min_age_s=DEFAULT_MIN_AGE_S, dry_run=False, quiet=False) -> dict:
    from muf import pipeline

    from . import ingest as ingest_mod

    new, found, fresh, skewed = find_new(targets, conn, methods, min_age_s)
    held_back = 0
    if batch and len(new) > batch:
        held_back = len(new) - batch
        new = new[:batch]

    result = {"found": found, "new": len(new), "too_fresh": fresh,
              "future_dated": skewed, "held_back": held_back,
              "loaded": 0, "skipped": 0}
    if not new or dry_run:
        return result

    options = pipeline.Options(methods=methods)
    counts = ingest_mod.ingest(new, conn, options, archive_root=archive_root,
                               jobs=jobs, progress=not quiet)
    result["loaded"] = counts["loaded"]
    result["skipped"] = counts["skipped"]
    return result


def describe(result: dict) -> str:
    bits = [f"{result['found']} on disk", f"{result['new']} new"]
    if result["too_fresh"]:
        bits.append(f"{result['too_fresh']} too fresh")
    if result.get("future_dated"):
        # Ingested anyway, but say so every pass: it means the archive's clock
        # is not ours, and a count that never falls is a file server to fix.
        bits.append(f"{result['future_dated']} FUTURE-DATED (archive clock is "
                    f"ahead of ours)")
    if result["held_back"]:
        bits.append(f"{result['held_back']} held for the next pass")
    if result["new"]:
        bits.append(f"loaded {result['loaded']}")
        if result["skipped"]:
            # Unreadable or still-partial files are not recorded, so they come
            # back next pass. That is deliberate -- a half-synced file becomes
            # readable on its own -- but a count that never falls means a file
            # that will never load, and it is worth going to look.
            bits.append(f"SKIPPED {result['skipped']}")
    return ", ".join(bits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="services.api.watch",
        description="Ingest soundings the database does not already hold.")
    parser.add_argument("target", nargs="+", type=Path)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--archive-root", type=Path, default=None,
                        help="sounding.path is stored relative to this "
                             "(default: $ARCHIVE_ROOT)")
    parser.add_argument("--methods", default="algo,kmeans,contour",
                        help="comma separated (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="0 uses every core (default: %(default)s)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds between passes; 0 runs once and exits, "
                             "which is what cron wants (default: %(default)s)")
    parser.add_argument("--batch", type=int, default=0,
                        help="most soundings to ingest per pass, so a first "
                             "run over a large archive does not hold the "
                             "database for hours. 0 means no cap")
    parser.add_argument("--min-age", type=float, default=DEFAULT_MIN_AGE_S,
                        metavar="SECONDS",
                        help="skip files modified more recently than this, so "
                             "a sounding still being written or synced is not "
                             "read short (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be ingested, change nothing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods:
        print("no methods requested", file=sys.stderr)
        return 2

    archive_root = args.archive_root or db.ARCHIVE_ROOT
    conn = connect(args.db)

    while True:
        started = time.time()
        try:
            result = run_once(args.target, conn, methods=methods,
                              archive_root=archive_root, jobs=args.jobs,
                              batch=args.batch, min_age_s=args.min_age,
                              dry_run=args.dry_run, quiet=args.quiet)
        except Exception as exc:                        # noqa: BLE001
            # A pass that raises must not kill a resident watcher -- the usual
            # causes (a locked database, a half-written file, a sync that
            # removed a directory mid-scan) all clear by themselves.
            print(f"{db.utcnow()}  pass failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            if args.interval <= 0:
                return 1
        else:
            if not args.quiet or result["new"]:
                prefix = "would ingest: " if args.dry_run else ""
                print(f"{db.utcnow()}  {prefix}{describe(result)}", flush=True)

        if args.interval <= 0:
            return 0
        time.sleep(max(1.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
