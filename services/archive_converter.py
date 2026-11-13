"""Run archive replacement jobs queued through the web console.

    python -m services.archive_converter --interval 5

One process lock for the database, released by the OS on crashes. A restarted
worker resumes the active job and its per-file verification journal.
"""

import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .api import conversion, db


@contextmanager
def worker_lock(path):
    with Path(path).open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.db or db.DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    with worker_lock(path.with_suffix(".conversion.lock")):
        while True:
            with db.session(path) as conn:
                conn.execute("PRAGMA busy_timeout = 30000")
                result = conversion.run_next(conn)
                if result:
                    print(
                        f"conversion {result['id']}: {result['state']} "
                        f"{result['done']}/{result['total']} replaced; "
                        f"{result['failed']} failed",
                        flush=True,
                    )
            if args.once:
                return 0
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
