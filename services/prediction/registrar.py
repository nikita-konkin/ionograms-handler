"""The worker that opens what the console uploaded.

    python -m services.prediction.registrar --interval 10

``importer.py`` refuses an HTTP import route because registering a model means
running code out of a file, and the prediction path is meant to have no inbound
surface. That refusal is kept here rather than reversed. The api takes bytes,
hashes them, and writes them to a quarantine volume without ever opening them;
this process -- which listens on nothing, is reachable from nothing, and exists
only to drain a table -- is what opens them.

So the console gets its button and the artifact is still loaded by a container
that no request can reach. The two halves are in different images for the same
reason they are in different processes.

**Order of operations matters, and it is: store, then register.** The artifact
is placed in the content-addressed store *before* the registry row is written,
so a row never names an object that is not there. If the import then fails, the
object is removed again -- a refused artifact leaves nothing behind.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..api import db
from . import artifacts, importer, legacy_features, queues, registry, store
from .infer import writable

#: Matches the api's `MODEL_UPLOADS`. The two must agree; they are set from the
#: same compose file and both default to the same path.
UPLOAD_DIR = Path(os.environ.get("MODEL_UPLOADS", "/uploads"))

#: Short by the standards of this service's other loops, because a person is
#: watching. `infer` sleeps six hours between passes; this one settles an
#: upload while the operator still has the page open.
DEFAULT_INTERVAL_S = 10.0


def settle(conn: sqlite3.Connection, row: dict,
           uploads: Path = UPLOAD_DIR) -> dict:
    """Register one uploaded artifact, or record why it could not be.

    Returns the settled ``model_upload`` row. Never raises for a bad artifact:
    a refusal is an outcome to be shown to the operator, not a crash that takes
    the worker down and leaves the queue unattended.
    """
    digest = row["sha256"]
    blob = uploads / digest

    if not blob.is_file():
        return queues.settle_upload(
            conn, row["id"], queues.REFUSED,
            f"the quarantined bytes for {row['filename']} are no longer at "
            f"{blob}. Upload it again -- nothing was registered.")

    # Re-hashed rather than trusted. The api hashed a stream as it wrote it;
    # this confirms what is on the volume now is what it hashed then.
    actual = artifacts.sha256(blob)
    if actual != digest:
        return queues.settle_upload(
            conn, row["id"], queues.REFUSED,
            f"{row['filename']} changed after it was uploaded: the api "
            f"recorded sha256 {digest[:12]} and the file now hashes to "
            f"{actual[:12]}. Nothing was loaded.")

    stored = None
    try:
        stored = store.put(blob, digest)
        model = importer.import_artifact(
            stored,
            param=row["param"],
            # Never the default. `import_artifact` falls back to the file stem,
            # and the file stem here is a 64-character hash -- correct as an
            # address and unreadable as a name.
            name=row["name"] or Path(row["filename"]).stem,
            tx=row["tx"], rx=row["rx"], origin=row["origin"],
            target_src=row["target_src"],
            period=row["period"] or legacy_features.DEFAULT_DECOMPOSITION_PERIOD,
            note=row["note"], conn=conn,
        )
    except (artifacts.ArtifactError, legacy_features.RecipeError,
            registry.RegistryError, store.StoreError, ValueError) as exc:
        if stored is not None and not _object_is_registered(conn, digest):
            store.unlink(digest)
        return queues.settle_upload(conn, row["id"], queues.REFUSED, str(exc))

    settled = queues.settle_upload(conn, row["id"], queues.REGISTERED,
                                   importer.describe(model), model["id"])
    _reap(conn, digest, uploads)
    return settled


def _object_is_registered(conn: sqlite3.Connection, digest: str) -> bool:
    """Whether any registry row already names these bytes.

    Checked before removing an object after a failed import: the same file may
    have been registered earlier under a different binding, and deleting the
    artifact out from under that row would break a model that works.
    """
    row = db.one(conn, "SELECT COUNT(*) AS n FROM model_registry WHERE sha256 = ?",
                 (digest,))
    return bool((row or {}).get("n"))


def _reap(conn: sqlite3.Connection, digest: str, uploads: Path) -> bool:
    """Remove the quarantined copy once nothing unsettled needs it.

    A refused upload keeps its bytes: the usual fix is to register the same
    file again with an explicit feature list, and making the operator upload it
    a second time to do that would be gratuitous.
    """
    if queues.blob_is_referenced(conn, digest):
        return False
    blob = uploads / digest
    if blob.exists():
        blob.unlink()
        return True
    return False


def run_once(conn: sqlite3.Connection, uploads: Path = UPLOAD_DIR,
             limit: int = 20) -> list[dict]:
    return [settle(conn, row, uploads)
            for row in queues.pending_uploads(conn, limit=limit)]


def describe(row: dict) -> str:
    """One line an operator can read, in the style `infer.describe` set."""
    if row["state"] == queues.REGISTERED:
        return f"  registered {row['filename']}: {row['detail']}"
    return f"  refused {row['filename']}: {row['detail']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.prediction.registrar",
        description="Register artifacts uploaded from the console.")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help="seconds between passes (default: %(default)s)")
    parser.add_argument("--uploads", type=Path, default=UPLOAD_DIR,
                        help="the quarantine directory (default: %(default)s)")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    if not args.once:
        print(f"registrar: watching {args.uploads} every {args.interval:g}s; "
              f"{store.describe()}", flush=True)

    while True:
        with db.session(args.db) as conn:
            refused = writable(conn)
            if refused is not None:
                print(f"cannot write {args.db or 'the database'}: {refused}\n"
                      f"  This process is uid {os.getuid()}. A shared SQLite "
                      f"database needs the *directory* writable too, for its "
                      f"-wal and -shm files, so every service that opens it "
                      f"must run as the uid that owns the data volume "
                      f"(the api's, 10001).", file=sys.stderr)
                return 1
            try:
                settled = run_once(conn, args.uploads)
            except sqlite3.Error as exc:
                # The api holds the write lock for minutes during an archive
                # scan. A pass that cannot start is a pass to retry, not a
                # reason to exit and leave the queue unattended.
                print(f"registrar pass skipped: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                settled = []

        if settled:
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            print(f"[{stamp}] settled {len(settled)} upload"
                  f"{'' if len(settled) == 1 else 's'}")
            for row in settled:
                print(describe(row))
            sys.stdout.flush()

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
