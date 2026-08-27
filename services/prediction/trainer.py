"""The worker that fits what the console asked for.

    python -m services.prediction.trainer --interval 60

Same argument as ``registrar.py`` and the same shape: the api writes a row and
holds no ability to run it; a container that listens on nothing drains the
queue. Registering a model means running code out of a file, and training one
means running code that fits -- neither belongs in the process that answers
HTTP, and separating them is cheaper than defending them.

Separate from the registrar rather than folded into it, for one reason worth
stating: a fit takes minutes and an upload settles in seconds. One worker
draining both queues would leave an operator watching a spinner for the length
of somebody else's training run.

**One job at a time, claimed with a conditional UPDATE.** ``queues.claim_job``
marks a row ``running`` in the same statement that selects it, so two workers
cannot both take it. There is one worker today; the guarantee costs a line and
removes the need to remember that.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

from ..api import db
from . import queues, store, train
from .infer import writable

#: A minute, against the registrar's ten seconds. Nobody watches a fit tick by
#: the second, and the queue is drained in order rather than in parallel.
DEFAULT_INTERVAL_S = 60.0


def settle(conn: sqlite3.Connection, job: dict) -> dict:
    """Run one claimed job to its end. Returns the settled row.

    A refusal is an outcome, not a crash: `TrainError` carries a sentence the
    operator can act on and it is stored as one. Anything else is stored as its
    type and message rather than being allowed to take the worker down, because
    a worker that exits leaves every later job queued behind a fault that has
    already happened.
    """
    # Re-vetting the stored spec is separated from running it, because the two
    # failures mean opposite things and until 2026-08-27 they read the same.
    #
    # `control_routes.queue_training` vets before it inserts, so a row in
    # `train_job` was accepted by *some* api. If this worker's `vet` then
    # refuses the same spec, the request was never the problem -- the two
    # builds disagree about what is valid. That is not hypothetical: `api` and
    # `watch` are watchtower-labelled and update themselves while `trainer`,
    # `registrar` and `infer` are not, so an updated api offering a new
    # estimator or a new feature column while a months-old trainer rejects it
    # is the normal failure mode of this deployment. It happened on 2026-08-26
    # with `voting`/`stacking` and again on 2026-08-27 with the cyclical time
    # columns, and both times the message described the running code perfectly
    # and gave no hint that the running code was stale.
    try:
        plan = train.plan_from_job(job)
    except train.TrainError as exc:
        return queues.settle_job(
            conn, job["id"], queues.FAILED,
            f"{exc}\n\nThis spec was accepted when it was queued, so the api "
            f"that queued it and this worker (build {db.build_id()}) do not "
            f"agree about what is valid -- which usually means this worker is "
            f"the older of the two. `trainer`, `registrar` and `infer` do not "
            f"update themselves: pull and recreate them, then queue it again.")

    try:
        result = train.run(conn, plan)
    except train.TrainError as exc:
        return queues.settle_job(conn, job["id"], queues.FAILED, str(exc))
    except (store.StoreError, ValueError, KeyError, ArithmeticError) as exc:
        return queues.settle_job(conn, job["id"], queues.FAILED,
                                 f"{type(exc).__name__}: {exc}")
    except MemoryError as exc:                       # pragma: no cover - host
        return queues.settle_job(
            conn, job["id"], queues.FAILED,
            f"ran out of memory fitting this model ({exc}). A shorter window "
            f"or a smaller estimator will fit; the container's limit will not "
            f"change on its own.")

    return queues.settle_job(conn, job["id"], queues.DONE,
                             train.describe(result), result["model"]["id"])


def run_once(conn: sqlite3.Connection, limit: int = 1) -> list[dict]:
    """Claim and run up to ``limit`` jobs. One by default."""
    settled = []
    for _ in range(max(int(limit), 1)):
        job = queues.claim_job(conn)
        if job is None:
            break
        settled.append(settle(conn, job))
    return settled


def describe(row: dict) -> str:
    verb = "trained" if row["state"] == queues.DONE else "failed"
    return (f"  {verb} job {row['id']} [{row['param']}] "
            f"{row['tx']} -> {row['rx']}: {row['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.prediction.trainer",
        description="Fit the models the console asked for.")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help="seconds between passes (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="jobs to take per pass (default: %(default)s)")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    if not args.once:
        print(f"trainer: draining train_job every {args.interval:g}s; "
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
                settled = run_once(conn, args.jobs)
            except sqlite3.Error as exc:
                # The api holds the write lock for minutes during an archive
                # scan. A pass that cannot start is a pass to retry.
                print(f"trainer pass skipped: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                settled = []

        if settled:
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            print(f"[{stamp}] settled {len(settled)} training job"
                  f"{'' if len(settled) == 1 else 's'}")
            for row in settled:
                print(describe(row))
            sys.stdout.flush()

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
