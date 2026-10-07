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


def stored_paths(conn: sqlite3.Connection) -> dict[str, str]:
    """``sounding.file -> sounding.path`` for every row."""
    return {row["file"]: row["path"]
            for row in db.rows(conn, "SELECT file, path FROM sounding")}


def _lost(stored: str, archive_root: Path) -> bool:
    """True if a stored path no longer opens. An unreachable share counts.

    "Host is down" is what a dead CIFS mount answers, and it is exactly the
    case this is for -- `Path.exists` raises on it rather than saying False.
    """
    full = Path(stored)
    if not full.is_absolute():
        full = archive_root / full
    try:
        return not full.exists()
    except OSError:
        return True


#: v2's product name, up to the transmitter and receiver:
#: ``lfm_ionogram-{txname}-{station_name}-{ch}-{cid:03d}-{t0:.2f}.h5``.
V2_PREFIX = "lfm_ionogram-"


def muted_by_name(conn: sqlite3.Connection):
    """``matcher(filename) -> bool`` for v2 products a mute rule covers.

    **Why at the listing, and not only at the write.** `ingest_row` declines a
    muted row, and a declined file never gets a row -- so `already_done` never
    counts it, and every pass offers it again. That alone is only waste. With
    ``--batch`` it is a stall: `find_new` sorts by name, the batch takes the
    front of the list, and on Yoshkar-Ola the front is `agent1`, `chilton`,
    `db049`, `juliusruh` ... -- all muted. Every pass spent its whole budget
    re-reading and re-declining the same 200 files and reported "loaded 0,
    1702 held for the next pass" for as long as anyone let it (2026-09-27).

    **Prefixes, not a parse.** Station names contain hyphens -- `Yoshkar-Ola`
    -- and so does the separator, so splitting ``lfm_ionogram-unkown-Yoshkar-
    Ola-ch0-...`` is ambiguous; `io_chirp._NAME_RE` reads the receiver as
    `Yoshkar`. Asking "does the name start with this rule's circuit" has no
    ambiguity to resolve. Folded, like the rules themselves.

    The filename is a proxy for the file's own attributes, which `ingest_row`
    trusts over it. The two agree unless a station was renamed in its config
    between writes; `.lfs` names carry no circuit and are left to the write.
    """
    prefixes = []
    for rule in db.rows(conn, "SELECT tx, rx FROM muted_circuit"):
        head = f"{V2_PREFIX}{rule['tx']}-"
        prefixes.append(head + (f"{rule['rx']}-" if rule["rx"] else ""))
    prefixes = tuple(prefixes)

    def matcher(name: str) -> bool:
        return bool(prefixes) and name.lower().startswith(prefixes)

    return matcher


def find_new(targets, conn, methods, min_age_s: float, now: float | None = None,
             *, format: str | None = None, archive_root: Path | None = None):
    """Soundings on disk that the database does not already hold.

    Returns ``(new, n_found, n_too_fresh, n_skewed, n_muted, n_relinked)``.

    **Relinking.** Given ``archive_root``, a file already indexed under another
    path whose stored path no longer opens has its row pointed here instead,
    and counts as relinked rather than new -- its extractions are kept, only
    the path the pages open changes. Without this a moved archive could never
    heal: `already_done` keys on the basename, so the copy is skipped forever
    while the row points at the old place. That was tesla's ~10,000 August
    soundings, indexed from ``ionozond_data2`` (now disabled; its share went
    down 2026-09-09) and drawn as FileNotFoundError ever since, though the
    station's sync had put the same files under ``ionozond_5tb/ionozond_data2``.
    A row whose stored path still opens is never moved, so a file kept in two
    places stays where it was first indexed.

    A v2 product
    whose circuit is muted is counted and left out -- see `muted_by_name` for
    why that cannot wait for the write. Targets holding no
    soundings at all are skipped rather than fatal: an archive normally
    contains detection trees, digisonde products and empty days beside the
    ionograms, and one of those must not stop the scan.

    ``format`` narrows what counts as a sounding at all, straight through to
    `loader.find_soundings`. ``None`` -- every caller's default -- means both,
    which is what a folder of mixed products should give a plain CLI run.

    It matters more than a filter usually does. Anything this returns gets a
    row, and `already_done` keys on the basename, so a file with no row is
    new *forever*: delete its row and the next pass puts it straight back.
    Narrowing here is therefore the only thing that can make a removal stick,
    which is why `archive.format` is carried down to it rather than merely
    checked when a folder is registered.
    """
    from muf import loader

    now = time.time() if now is None else now
    done = already_done(conn, methods)
    muted = muted_by_name(conn)
    stored = stored_paths(conn) if archive_root is not None else {}

    found, fresh, skewed, silenced, new, moves = 0, 0, 0, 0, [], []
    for target in targets:
        try:
            paths = loader.find_soundings(target, format=format)
        except FileNotFoundError:
            continue                      # nothing this reader recognises
        for path in paths:
            found += 1
            if path.name in done:
                if path.name in stored:
                    here = _relative(path, archive_root)
                    if (here != stored[path.name]
                            and _lost(stored[path.name], archive_root)):
                        moves.append((here, path.name))
                continue
            if muted(path.name):
                silenced += 1
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
    if moves:
        conn.executemany("UPDATE sounding SET path = ? WHERE file = ?", moves)
        conn.commit()
    return new, found, fresh, skewed, silenced, len(moves)


def _relative(path: Path, archive_root: Path) -> str:
    """The path as `ingest` would store it. Lexical first: a scan meets every
    file it already holds, and resolving each one on a network share costs a
    round trip per path component."""
    try:
        return path.relative_to(archive_root).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(archive_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_once(targets, conn, *, methods, archive_root, jobs=1, batch=0,
             min_age_s=DEFAULT_MIN_AGE_S, dry_run=False, quiet=False,
             format: str | None = None) -> dict:
    from muf import pipeline

    from . import ingest as ingest_mod

    new, found, fresh, skewed, silenced, relinked = find_new(
        targets, conn, methods, min_age_s, format=format,
        archive_root=Path(archive_root))
    held_back = 0
    if batch and len(new) > batch:
        held_back = len(new) - batch
        new = new[:batch]

    result = {"found": found, "new": len(new), "too_fresh": fresh,
              "future_dated": skewed, "muted": silenced, "relinked": relinked,
              "held_back": held_back, "loaded": 0, "skipped": 0}
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
    if result.get("muted"):
        # Said every pass for the same reason as FUTURE-DATED: a mute rule
        # that covers more than intended is otherwise invisible from here.
        bits.append(f"{result['muted']} muted")
    if result.get("relinked"):
        bits.append(f"{result['relinked']} relinked (moved here from a folder "
                    f"that no longer has them)")
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
        except Exception as exc:
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
