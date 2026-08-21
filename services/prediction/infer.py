"""Running a model forward and storing what it said.

    python -m services.prediction.infer --once --param muf
    python -m services.prediction.infer --interval 21600 --param muf,lof

**Nothing in this module calls ``fit``, and that is the point of the module.**
The code it replaces did: ``xgb_evaluate`` and ``xgb_test`` in the research
project both ``joblib.load`` a saved model and immediately refit it on the
training window, so the artifact contributes hyperparameters and the numbers
come from a model trained moments earlier. In a notebook that is a defensible
shortcut. In a service it is indistinguishable from inference until someone
asks why the "forecast" tracks the training data so well.
``tests/test_prediction_infer.py`` makes ``fit`` raise and runs a whole pass to
keep it that way.

Two products come out of the same code path and should not be confused for one
another. A **nowcast** extends the tracked series a little past the last
sounding and is nearly free. A **forecast** runs days ahead and is a much
harder problem. They differ here only in the model that is active and the
window asked for, so they are separated by model name and horizon rather than
by pretending one is the other.

A run that finds no active model logs and exits zero. A prediction service with
nothing trained yet is the normal state of a fresh deployment, not a fault, and
a container that crash-loops over it is noise in every log the operator reads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from ..api import db
from . import artifacts, dataset, legacy_features, registry, scoring

#: Default cadence: four times a day. The long-horizon forecast has nothing to
#: say more often than the drivers move, and issuing it hourly multiplies the
#: `forecast` table by 24 for no new information.
DEFAULT_INTERVAL_S = 21600


def _clean(value):
    """NaN becomes NULL, exactly as `services.api.ingest` does it.

    A NaN in a REAL column compares false against everything including itself,
    so `WHERE value IS NOT NULL` would return rows with no value. NULL is what
    "the model declined" means in SQL.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def writable(conn: sqlite3.Connection) -> str | None:
    """``None`` if this process can write the database, else why it cannot.

    The probe writes ``user_version`` back to the value it already holds: a
    real write, to the database header, that changes nothing. ``BEGIN
    IMMEDIATE`` looks like the tidier probe and is not one -- SQLite defers
    acquiring the write lock, so it succeeds against a database it cannot
    write and reports the problem only at the first statement that matters.

    Worth a dedicated check because the failure it catches is unreadable
    otherwise. A container running as a different uid from the one that owns
    the data volume can open the database, read every row and answer every
    query -- SQLite only needs the *directory* writable when it comes to
    create the `-wal` and `-shm` siblings, which is at the first write. The
    error then says "attempt to write a readonly database" about a file that
    is plainly not read-only, hours after start, from whichever line happened
    to write first.
    """
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.execute(f"PRAGMA user_version = {int(version)}")
    except sqlite3.OperationalError as exc:
        return str(exc)
    return None


def build_features(series: dataset.Series, model: dict) -> pd.DataFrame:
    """The model's input frame, built from a tracked series.

    The alias is passed explicitly from the registry row rather than inferred
    from the model, because that is the decision worth making deliberately:
    it is the moment a series measured on one circuit is handed to a model
    whose features are named after another.
    """
    recipe = legacy_features.parse(
        model["features"],
        period=(model.get("feature_recipe") or {}).get(
            "period", legacy_features.DEFAULT_DECOMPOSITION_PERIOD),
    )
    return legacy_features.build(series.values, recipe,
                                 alias=model["target_alias"] or recipe.alias)


def run_model(conn: sqlite3.Connection, model: dict, tx: str, rx: str,
              method: str = "contour", issued_at: str | None = None,
              allow_skew: bool = False) -> dict:
    """Run one model against one circuit and write its forecast rows."""
    issued_at = issued_at or db.utcnow()

    estimator, contract, quality = artifacts.load_verified(
        model["artifact"], model.get("golden_input"), model.get("golden_output"),
        allow_skew=allow_skew,
    )

    series = dataset.tracked(conn, model["param"], tx, rx, method)
    frame = build_features(series, model)
    if frame.empty:
        return {"model": model["name"], "tx": tx, "rx": rx, "written": 0,
                "detail": "not enough history to build a single feature row"}

    predictions = artifacts.predict(estimator, frame)

    quality = dict(quality)
    quality["method"] = method
    quality["n_measured"] = series.n_measured
    quality["n_filled"] = series.n_filled
    quality["alias"] = model["target_alias"]
    quality_json = json.dumps(quality)

    issued = pd.Timestamp(issued_at.replace("Z", "+00:00")).tz_convert(None) \
        if issued_at.endswith("Z") else pd.Timestamp(issued_at)

    # The tracker's sigma at the instant each row was built from is the closest
    # honest uncertainty available: these models emit a point estimate and no
    # interval of their own. Recorded as the input uncertainty it is, not
    # dressed up as the model's.
    lag_step = pd.Timedelta(seconds=frame.attrs["step_s"] * frame.attrs["lag"])
    source_sigma = series.frame["sigma"].reindex(frame.index - lag_step)

    # **Horizon is lead time, not wall-clock distance from the run.** A lagged
    # model predicts an instant from data one lag earlier, so its lead time is
    # the lag -- the same 24 h whether it runs live or over a 2023 archive.
    # Measuring `valid_at - issued_at` instead makes every backtest report
    # negative horizons and puts the same prediction in a different scoring
    # bucket depending on the day someone ran it, which would make the
    # by-horizon leaderboard meaningless.
    horizon_s = int(lag_step.total_seconds())
    backtest = bool(frame.index.max() < issued)
    if backtest:
        quality["backtest"] = True
        quality_json = json.dumps(quality)

    rows = []
    for position, (valid_at, value) in enumerate(zip(frame.index, predictions)):
        rows.append((
            model["id"], model["param"], tx, rx, issued_at,
            valid_at.strftime(db.TIME_FORMAT),
            horizon_s,
            _clean(float(value)),
            _clean(float(source_sigma.iloc[position])),
            None, None, quality_json,
        ))

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO forecast (model_id, param, tx, rx, "
            "issued_at, valid_at, horizon_s, value, sigma, lo, hi, quality) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    return {
        "model": model["name"], "tx": tx, "rx": rx, "param": model["param"],
        "written": len(rows), "issued_at": issued_at,
        "horizon_h": round(horizon_s / 3600, 1),
        "valid": (frame.index.min().strftime("%Y-%m-%d %H:%M"),
                  frame.index.max().strftime("%Y-%m-%d %H:%M")),
        "backtest": backtest,
        "golden": quality["golden"],
    }


def run_once(conn: sqlite3.Connection, params: tuple[str, ...] = ("muf",),
             method: str = "contour", model_id: int | None = None,
             tx: str | None = None, rx: str | None = None,
             allow_skew: bool = False) -> list[dict]:
    """One pass: every active model, or one named model for a comparison run."""
    results: list[dict] = []
    issued_at = db.utcnow()

    if model_id is not None:
        model = registry.get(conn, model_id)
        if model is None:
            raise SystemExit(f"no model with id {model_id}")
        if not (tx and rx):
            raise SystemExit(
                "a named model needs --tx and --rx: the model says what it "
                "expects, not which circuit's data to feed it.")
        return [run_model(conn, model, tx, rx, method, issued_at, allow_skew)]

    for param in params:
        for circuit in dataset.circuits(conn, param, method):
            model = registry.active(conn, param, circuit["tx"], circuit["rx"])
            if model is None:
                results.append({
                    "param": param, "tx": circuit["tx"], "rx": circuit["rx"],
                    "written": 0, "detail": "no active model",
                })
                continue
            try:
                results.append(run_model(conn, model, circuit["tx"],
                                         circuit["rx"], method, issued_at,
                                         allow_skew))
            except (ValueError, artifacts.ArtifactError,
                    legacy_features.RecipeError) as exc:
                results.append({
                    "param": param, "tx": circuit["tx"], "rx": circuit["rx"],
                    "model": model["name"], "written": 0, "detail": str(exc),
                })
    return results


def describe(result: dict) -> str:
    """One line per circuit, in the style `services.api.watch.describe` set."""
    where = f"{result.get('tx')}->{result.get('rx')}"
    if not result.get("written"):
        return f"  {where} [{result.get('param','?')}]: {result.get('detail','nothing')}"
    skew = "" if result.get("golden") == "ok" else f" golden={result.get('golden')}"
    kind = "backtest" if result.get("backtest") else "forecast"
    start, stop = result["valid"]
    return (f"  {where} [{result['param']}]: {result['written']} rows from "
            f"{result['model']}, {kind} at +{result['horizon_h']} h lead, "
            f"valid {start} .. {stop}{skew}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.prediction.infer",
        description="Run trained models forward and store their forecasts.")
    parser.add_argument("--param", default="muf",
                        help="comma separated: muf,lof (default: %(default)s)")
    parser.add_argument("--method", default="contour",
                        help="which estimator's series to track (default: %(default)s)")
    parser.add_argument("--model", type=int, default=None,
                        help="run one registered model by id, for comparison. "
                             "Needs --tx and --rx.")
    parser.add_argument("--tx", default=None)
    parser.add_argument("--rx", default=None)
    parser.add_argument("--once", action="store_true",
                        help="one pass, then exit")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help="seconds between passes (default: %(default)s)")
    parser.add_argument("--allow-version-skew", action="store_true",
                        help="run a model whose golden check fails, recording "
                             "the skew on every forecast it produces")
    parser.add_argument("--no-score", action="store_true",
                        help="skip the scoring pass that normally follows")
    parser.add_argument("--score-window-days", type=int,
                        default=scoring.DEFAULT_WINDOW_DAYS,
                        help="how much history each scoring pass covers "
                             "(default: %(default)s)")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    params = tuple(p.strip() for p in args.param.split(",") if p.strip())

    while True:
        started = time.monotonic()
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
                results = run_once(conn, params, args.method, args.model,
                                   args.tx, args.rx, args.allow_version_skew)
            except artifacts.ArtifactError as exc:
                print(f"inference failed: {exc}", file=sys.stderr)
                return 1

            # Scoring rides on the same pass, because the issues that just
            # came due are exactly the ones nobody would otherwise remember to
            # score. It is deliberately not allowed to fail the run: a
            # forecast that was written and not yet judged is a normal state,
            # and losing the forecast because the judging broke is not.
            scored = []
            if not args.no_score:
                try:
                    scored = scoring.run_once(conn, params, args.method,
                                              args.score_window_days,
                                              args.tx, args.rx)
                except (ValueError, KeyError, sqlite3.Error,
                        scoring.ScoringError) as exc:
                    # `sqlite3.Error` included deliberately. Scoring runs after
                    # the forecasts are already written, and a locked database
                    # at that moment -- the api mid-scan holds the write lock
                    # for minutes -- would otherwise discard a pass that had
                    # already done its job.
                    print(f"scoring skipped: {type(exc).__name__}: {exc}",
                          file=sys.stderr)

        written = sum(r.get("written", 0) for r in results)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        print(f"[{stamp}] {written} forecast rows over {len(results)} circuits "
              f"in {time.monotonic() - started:.1f}s")
        for result in results:
            print(describe(result))
        if scored:
            print(f"  scored {len(scored)} subjects over the last "
                  f"{args.score_window_days} days")
            for result in scored:
                print(scoring.describe(result))
        sys.stdout.flush()

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
