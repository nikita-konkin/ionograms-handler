"""Serial rolling fits, triggered by completed measurement steps.

Training targets occupy [anchor-14d, anchor-1d); the final day is held out.
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

def window_plan(
    spec: dict, anchor: pd.Timestamp
) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    """14 calendar days of targets plus sufficient context for feature warmup."""
    plan = train.plan_from_job({**spec, "spec": {**spec, "holdout_days": 1}})
    cut = anchor - pd.Timedelta(days=1)
    start = cut - pd.Timedelta(days=13)
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
    conn, spec: dict, *, activate: bool = False, now=None, state_dir: Path | None = None
) -> dict:
    """Fit only the newest available step; no catch-up queue after an outage."""
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
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
        stamp = str(anchor)
        if saved.get("anchor") == stamp and (
            not activate or saved.get("activated") or saved.get("refused")
        ):
            return {"state": "unchanged", **saved}
        # Do not train repeatedly on an old archive while waiting for ingest.
        if now - newest > STALE:
            return {
                "state": "waiting",
                "detail": "latest measurement is over 30 minutes old",
            }
        plan, start, cut = window_plan(spec, anchor)
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
            )
            if not forecast["written"]:
                raise train.TrainError("candidate produced no forecast; incumbent kept")
            registry.activate(conn, model["id"], by="rolling-13d-1d")
            prune(conn, spec["param"], spec["tx"], spec["rx"])
        saved = {
            "anchor": stamp,
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
            "validation_end": stamp,
            "validation": metrics,
            "elapsed_s": round(time.monotonic() - began, 3),
        }
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved), encoding="utf-8")
        temporary.replace(state_file)
        return {
            "state": "refused" if refused else "trained",
            **saved,
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
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive and finite")
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
                result = run_once(conn, spec, activate=args.activate)
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
