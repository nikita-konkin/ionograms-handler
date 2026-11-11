"""Serial rolling fits, triggered by completed measurement steps or by days.

Training targets occupy [anchor-1d-N, anchor-1d); the final day is held out.
Earlier measurements are feature context only. No validation row is refitted.
Run in the training image, with the model store writable, never in the API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..api import db
from . import dataset, infer, registry, scoring, store, train

#: Fitting pauses while the newest measurement is older than this, so an
#: archive that stopped arriving is not refitted over and over.
STALE = pd.Timedelta(minutes=30)

#: Calendar days of training targets before the validation day.
DAYS = 13


def window_plan(
    spec: dict, anchor: pd.Timestamp, days: int = DAYS, note: str | None = None
) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    """``days`` calendar days of targets, then one day of validation that
    ends at ``anchor``, plus sufficient context for feature warmup."""
    plan = train.plan_from_job({**spec, "spec": {**spec, "holdout_days": 1}})
    cut = anchor - pd.Timedelta(days=1)
    start = cut - pd.Timedelta(days=days)
    warmup = (
        plan["lag"]
        + max(
            max(plan["windows"], default=0),
            2 * plan["period"] if plan["components"] else 0,
        )
        + 2
    )
    plan.update(
        start=str(start - pd.Timedelta(seconds=warmup * dataset.DEFAULT_STEP_S)),
        end=str(anchor - pd.Timedelta(seconds=dataset.DEFAULT_STEP_S)),
        name="rolling-" + anchor.strftime("%Y%m%dT%H%M%S"),
        note=note,
    )
    return plan, start, cut


@contextmanager
def exclusive(path: Path):
    """An OS lock releases on crashes and prevents overlapping worker replicas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_once(
    conn,
    spec: dict,
    *,
    activate: bool = False,
    now=None,
    state_dir: Path | None = None,
    days: int = DAYS,
    period: int = 0,
) -> dict:
    """Fit only the newest available step; no catch-up queue after an outage.

    With a ``period`` of days the worker is daily instead: it fits at 00:00
    UTC once the day before is measured, refits every ``period`` days, and
    every day issues the 24 hours that follow midnight and nothing beyond.
    """
    key = hashlib.sha256(
        json.dumps({**spec, "days": days, "period": period}, sort_keys=True)
        .encode()
    ).hexdigest()[:24]
    directory = Path(state_dir) if state_dir is not None else store.root() / "rolling"
    with exclusive(directory / (key + ".lock")) as locked:
        if not locked:
            return {"state": "busy"}
        state_file = directory / (key + ".json")
        saved = json.loads(state_file.read_text()) if state_file.exists() else {}
        now = scoring._naive(now or db.utcnow())
        # Limit the query at wall clock time: future-dated or incomplete data
        # must never move the training window forward.
        observed = dataset.observations(
            conn, spec["param"], spec["tx"], spec["rx"], spec["method"], end=str(now)
        )
        observed = observed[observed["value"].notna()]
        if observed.empty:
            return {"state": "waiting", "detail": "no measurements"}
        newest = pd.Timestamp(observed["datetime"].max())
        anchor = newest.floor("5min") + pd.Timedelta(minutes=5)
        if period:
            result = _daily(conn, spec, saved, now, anchor, activate=activate,
                            days=days, period=period)
        else:
            result = _step(conn, spec, saved, now, newest, activate=activate,
                           days=days)
        if result["state"] in ("trained", "refused", "issued"):
            saved = {key: value for key, value in result.items()
                     if key not in ("state", "forecast")}
            temporary = state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved), encoding="utf-8")
            temporary.replace(state_file)
        return result


def _step(conn, spec, saved, now, newest, *, activate, days) -> dict:
    """A fit on every new five-minute step, while measurements keep arriving."""
    anchor = newest.floor("5min") + pd.Timedelta(minutes=5)
    if saved.get("anchor") == str(anchor) and (
        not activate or saved.get("activated") or saved.get("refused")
    ):
        return {"state": "unchanged", **saved}
    # Do not train repeatedly on an old archive while waiting for ingest.
    if now - newest > STALE:
        return {
            "state": "waiting",
            "detail": "latest measurement is over 30 minutes old",
        }
    return _fit(conn, spec, anchor, now, activate=activate, days=days,
                by=f"rolling-{days}d-1d",
                note=f"Refitted on every step on {days} days, validated on "
                     f"the day after.")


def _daily(conn, spec, saved, now, anchor, *, activate, days, period) -> dict:
    """Fits at midnight every ``period`` days, one issue for each day."""
    today = now.floor("D")
    if anchor < today:
        return {"state": "waiting",
                "detail": "the day before is not measured to its end yet"}
    if saved.get("day") == str(today):
        return {"state": "unchanged", **saved}
    until = today + pd.Timedelta(days=1)
    if not saved.get("block") or today >= pd.Timestamp(saved["block"]) \
            + pd.Timedelta(days=period):
        result = _fit(conn, spec, today, now, activate=activate, days=days,
                      by=f"daily-{days}d-every-{period}d", until=until,
                      note=f"Refitted at 00:00 UTC every {period} days on "
                           f"{days} days, validated on the day after; each "
                           f"day issues the next 24 hours.")
        result["block"] = str(today)
    else:
        result = {**saved, "state": "issued", "forecast": None}
    if activate and result["forecast"] is None:
        # The incumbent issues the day: either the period's own fit, or the
        # live model a refused fit left in place.
        model = registry.active(conn, spec["param"], spec["tx"], spec["rx"])
        if model is not None:
            result["forecast"] = infer.run_model(
                conn, model, spec["tx"], spec["rx"], spec["method"],
                valid_after=now, valid_before=until)
    result["day"] = str(today)
    return result


def _fit(conn, spec, anchor, now, *, activate, days, by, note,
         until=None) -> dict:
    """Fit the days before ``anchor`` and, unless the gate refuses it, issue
    its forecast up to ``until`` and put it live."""
    plan, start, cut = window_plan(spec, anchor, days, note)
    began = time.monotonic()
    result = train.run(
        conn, plan, training_start=start, validation_start=cut, diagnostics=False
    )
    metrics = result["holdout"]
    if (
        metrics["n"] < 60
        or metrics["mae"] is None
        or not math.isfinite(metrics["mae"])
    ):
        raise train.TrainError(
            "rolling validation requires 60 uncensored pairs and finite MAE"
        )
    model = result["model"]
    # Validity is not enough to go live: a fit the paired test finds worse
    # than the live forecast, on the same validation day, stays a
    # candidate and the incumbent keeps publishing.
    versus = result.get("vs_live") or {}
    refused = (
        activate
        and bool(versus.get("distinguishable"))
        and versus.get("delta", 0) > 0
    )
    # Write forecasts before promotion, so a failed prediction leaves the
    # old active model and its public curve intact.
    forecast = None
    if activate and not refused:
        forecast = infer.run_model(
            conn,
            model,
            spec["tx"],
            spec["rx"],
            spec["method"],
            valid_after=now,
            valid_before=until,
        )
        if not forecast["written"]:
            raise train.TrainError("candidate produced no forecast; incumbent kept")
        registry.activate(conn, model["id"], by=by)
        prune(conn, spec["param"], spec["tx"], spec["rx"])
    return {
        "state": "refused" if refused else "trained",
        "anchor": str(anchor),
        "model_id": model["id"],
        "activated": activate and not refused,
        "refused": (
            f"worse than the live forecast on the validation day: MAE "
            f"{versus['mae']:.2f} v {versus['baseline_mae']:.2f} MHz over "
            f"{versus['n']} pairs"
            if refused
            else None
        ),
        "vs_live": versus or None,
        "train_start": str(start),
        "validation_start": str(cut),
        "validation_end": str(anchor),
        "validation": metrics,
        "elapsed_s": round(time.monotonic() - began, 3),
        "forecast": forecast,
    }


def prune(conn, param: str, tx: str, rx: str) -> int:
    """Delete superseded rolling fits; return how many.

    The live model stays, and so does one fit a month: the last that went
    live in it. A deleted fit's forecasts move onto its month's fit first, so
    the record as issued, which is built from them, loses nothing. A fit that
    never went live has no place in that record and goes with its forecasts.
    """
    rows = db.rows(
        conn,
        "SELECT id, name, active, activated_at FROM model_registry WHERE name "
        "LIKE 'rolling-%' AND param = ? AND tx = ? AND rx = ? "
        "ORDER BY name DESC",
        (param, tx, rx),
    )
    heirs: dict[str, int] = {}
    for row in rows:
        if row["active"] or row["activated_at"]:
            heirs.setdefault(row["name"][8:14], row["id"])
    kept = set(heirs.values()) | {row["id"] for row in rows if row["active"]}
    removed = 0
    for row in rows:
        if row["id"] in kept:
            continue
        if row["activated_at"]:
            with conn:
                conn.execute(
                    "UPDATE OR REPLACE forecast SET model_id = ? "
                    "WHERE model_id = ?",
                    (heirs[row["name"][8:14]], row["id"]),
                )
        registry.delete(conn, row["id"])
        removed += 1
    return removed


def status(conn, param: str, tx: str, rx: str, method: str = "contour",
           now=None) -> dict | None:
    """The newest rolling fit for a circuit, and whether the worker can fit.

    Read from the registry and the picks, not from the worker, so the console
    can show it without reaching into the training container. None when no
    rolling fit was ever registered for the circuit.
    """
    row = db.one(conn, "SELECT id FROM model_registry WHERE name LIKE "
                       "'rolling-%' AND param = ? AND tx = ? AND rx = ? "
                       "ORDER BY id DESC LIMIT 1", (param, tx, rx))
    if row is None:
        return None
    model = registry.get(conn, row["id"])
    holdout = (model.get("metrics") or {}).get("holdout") or {}
    now = scoring._naive(now or db.utcnow())
    newest = dataset.observations(conn, param, tx, rx, method,
                                  start=str(now - pd.Timedelta(days=2)),
                                  end=str(now))
    newest = newest[newest["value"].notna()]
    at = pd.Timestamp(newest["datetime"].max()) if len(newest) else None
    return {"name": model["name"], "fitted_at": model["imported_at"],
            "note": model.get("note"),
            "live": bool(model["active"]), "mae": holdout.get("mae"),
            "from": holdout.get("from"), "to": holdout.get("to"),
            "trained_from": model.get("trained_from"),
            "trained_to": model.get("trained_to"),
            "newest": None if at is None else at.strftime(db.TIME_FORMAT),
            "age_min": (None if at is None
                        else int((now - at).total_seconds() // 60)),
            "waiting": at is None or now - at > STALE,
            "stale_min": int(STALE.total_seconds() // 60)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param", default="muf", choices=("muf", "lof"))
    parser.add_argument("--tx", required=True)
    parser.add_argument("--rx", required=True)
    parser.add_argument("--method", default="contour")
    parser.add_argument("--lead", type=float, default=24)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--train-days", type=int, default=DAYS)
    parser.add_argument(
        "--period-days", type=int, default=0,
        help="fit at 00:00 UTC every this many days and issue one day at a "
             "time; 0 refits on every new step")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive and finite")
    if args.train_days < 1:
        parser.error("train days must be at least one")
    if args.period_days < 0:
        parser.error("period days must not be negative")
    spec = {
        "param": args.param,
        "tx": args.tx,
        "rx": args.rx,
        "method": args.method,
        "lead_h": args.lead,
        "estimator": "voting",
        # seasonal_decompose estimates seasonal means over the entire record,
        # including the held-out day. Causal rolling/time features avoid that.
        "components": [],
        "time": ["daily_sin", "daily_cos", "daily_sin2", "daily_cos2"],
    }
    # Vet before entering the long-running loop.
    window_plan(spec, pd.Timestamp("2026-01-01"))
    while True:
        began = time.monotonic()
        try:
            with db.session(args.db) as conn:
                result = run_once(conn, spec, activate=args.activate,
                                  days=args.train_days,
                                  period=args.period_days)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            print(
                f"rolling pass failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.once:
                return 1
        if args.once:
            return 0
        # Fits are serial. If one overruns, skip missed ticks and process the
        # newest data after a short pause instead of building an obsolete queue.
        time.sleep(max(1, args.interval - (time.monotonic() - began)))


if __name__ == "__main__":
    raise SystemExit(main())
