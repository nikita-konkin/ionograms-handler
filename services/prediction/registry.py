"""The model registry: what is registered, what is live, and how it changes.

One row binds one artifact to one circuit and one parameter. Promotion --
making a model *the* forecast for a circuit -- is the only operation here that
changes what anybody downstream sees, so it is the only one that is
transactional, audited, and refused rather than corrected.

The rules it enforces live in ``schema.sql`` as constraints, not here as
checks, and this module's job is to turn their failures into sentences an
operator can act on. That split is deliberate: a rule enforced in application
code is a rule that a second writer, a migration or a psql session can walk
straight through.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..api import db

#: Columns a caller may set at registration. `active` is absent on purpose --
#: promotion goes through :func:`activate`, which is transactional.
REGISTERABLE = (
    "name", "param", "tx", "rx", "origin", "framework", "loader", "capability",
    "artifact", "sha256", "features", "target_alias", "feature_recipe", "env",
    "golden_input", "golden_output", "target_src", "note",
    "trained_from", "trained_to",
)


class RegistryError(RuntimeError):
    """A registry operation was refused."""


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value)


def register(conn: sqlite3.Connection, **fields) -> int:
    """Insert a model row. Returns its id.

    Idempotent on ``(name, param, tx, rx, sha256)``: re-importing the same file
    for the same binding updates the row rather than creating a second one, so
    running the importer twice is harmless -- which is the first thing anyone
    does when an import looks wrong.
    """
    unknown = set(fields) - set(REGISTERABLE)
    if unknown:
        raise RegistryError(f"not registerable: {sorted(unknown)}")

    for key in ("features", "feature_recipe", "env", "golden_input"):
        if key in fields:
            fields[key] = _json(fields[key])

    fields.setdefault("imported_at", db.utcnow())
    columns = list(fields)
    placeholders = ",".join("?" * len(columns))
    updates = ",".join(f"{c}=excluded.{c}" for c in columns if c != "sha256")

    try:
        conn.execute(
            f"INSERT INTO model_registry ({','.join(columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (name, param, COALESCE(tx, ''), COALESCE(rx, ''), "
            f"sha256) DO UPDATE SET {updates}",
            tuple(fields.values()),
        )
    except sqlite3.IntegrityError as exc:
        raise RegistryError(f"could not register: {exc}") from exc
    conn.commit()

    # `lastrowid` is the inserted row on an INSERT and the *updated* row on an
    # upsert, but SQLite does not promise the latter across versions -- so the
    # identity is re-read rather than inferred. `IS` rather than `=` because
    # tx/rx are NULL for an unbound model and `= NULL` matches nothing.
    row = db.one(
        conn,
        "SELECT id FROM model_registry WHERE name=? AND param=? AND sha256=? "
        "AND tx IS ? AND rx IS ?",
        (fields["name"], fields["param"], fields["sha256"],
         fields.get("tx"), fields.get("rx")),
    )
    if row is None:                               # pragma: no cover - defensive
        raise RegistryError("registered a model but could not read it back")
    return int(row["id"])


def get(conn: sqlite3.Connection, model_id: int) -> dict | None:
    row = db.one(conn, "SELECT * FROM model_registry WHERE id = ?", (model_id,))
    return _decode(row) if row else None


def models(conn: sqlite3.Connection, param: str | None = None,
           tx: str | None = None, rx: str | None = None) -> list[dict]:
    """Every registered model, newest first, optionally narrowed."""
    sql = ["SELECT * FROM model_registry WHERE 1=1"]
    params: list = []
    for column, value in (("param", param), ("tx", tx), ("rx", rx)):
        if value:
            sql.append(f"AND {column} = ?")
            params.append(value)
    sql.append("ORDER BY active DESC, imported_at DESC")
    return [_decode(r) for r in db.rows(conn, " ".join(sql), tuple(params))]


def active(conn: sqlite3.Connection, param: str, tx: str, rx: str) -> dict | None:
    """The live model for one circuit and parameter, or None."""
    row = db.one(
        conn,
        "SELECT * FROM model_registry "
        "WHERE active = 1 AND param = ? AND tx = ? AND rx = ?",
        (param, tx, rx),
    )
    return _decode(row) if row else None


def activate(conn: sqlite3.Connection, model_id: int,
             by: str | None = None) -> dict:
    """Make a model the live forecast for its circuit, demoting the incumbent.

    One transaction: there is no instant at which two models are live for a
    circuit, and none at which zero are because the demotion landed and the
    promotion did not.

    Refusals come from the schema, and there are three worth naming:

    * a model whose target was **modelled** rather than measured -- it may be
      compared for ever and never promoted;
    * a model **bound to no circuit**, which is every legacy import;
    * a **second** model for a circuit that already has one, which cannot
      happen because the demotion above precedes it.
    """
    model = get(conn, model_id)
    if model is None:
        raise RegistryError(f"no model with id {model_id}")

    if model["target_src"] != "measured":
        raise RegistryError(
            f"{model['name']} was fitted against a {model['target_src']} "
            f"target, so it can be scored and compared but never made the "
            f"operational forecast. Training a model on modelled values and "
            f"then validating against them is circular, which is why the "
            f"schema refuses this rather than the service warning about it."
        )
    if not model["tx"] or not model["rx"]:
        raise RegistryError(
            f"{model['name']} is registered unbound (no circuit), so there is "
            f"no forecast it could be. Re-import it with --tx and --rx naming "
            f"the circuit it is for."
        )

    incumbent = active(conn, model["param"], model["tx"], model["rx"])

    try:
        with conn:                                  # one transaction
            if incumbent and incumbent["id"] != model_id:
                conn.execute(
                    "UPDATE model_registry SET active = 0 WHERE id = ?",
                    (incumbent["id"],),
                )
            conn.execute(
                "UPDATE model_registry SET active = 1, activated_at = ?, "
                "activated_by = ? WHERE id = ?",
                (db.utcnow(), by, model_id),
            )
    except sqlite3.IntegrityError as exc:
        raise RegistryError(f"could not activate {model['name']}: {exc}") from exc

    return {
        "activated": get(conn, model_id),
        "deactivated": incumbent if incumbent and incumbent["id"] != model_id else None,
    }


def retire(conn: sqlite3.Connection, model_id: int) -> dict | None:
    """Take a model out of service. Its rows and its forecasts stay.

    Retiring is not deleting: the forecasts it issued are what its scores are
    computed from, and re-activating it is how a promotion is rolled back.
    """
    with conn:
        conn.execute("UPDATE model_registry SET active = 0 WHERE id = ?", (model_id,))
    return get(conn, model_id)


def set_metrics(conn: sqlite3.Connection, model_id: int, metrics: dict) -> None:
    """Merge into the metrics column. Two writers share it.

    `scoring` writes how a model does in production; `train` writes how it did
    on its holdout at fit time, before it has ever issued a forecast. A
    replacing write would mean the first scoring pass silently erased the only
    record of what the model was accepted on, so the top-level keys are merged
    and each writer owns its own.
    """
    row = db.one(conn, "SELECT metrics FROM model_registry WHERE id = ?",
                 (model_id,))
    existing: dict = {}
    if row and row.get("metrics"):
        try:
            loaded = json.loads(row["metrics"])
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded

    existing.update(metrics)
    with conn:
        conn.execute("UPDATE model_registry SET metrics = ? WHERE id = ?",
                     (json.dumps(existing), model_id))


def _decode(row: dict) -> dict:
    """JSON columns back into objects, so callers never json.loads by hand."""
    out = dict(row)
    for key in ("features", "feature_recipe", "env", "golden_input", "metrics"):
        value = out.get(key)
        if isinstance(value, str) and value:
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    out["state"] = state_of(out)
    return out


def state_of(row: dict) -> str:
    """The lifecycle state a row is in, as the console labels it."""
    if row.get("active"):
        return "active"
    if row.get("target_src") != "measured":
        return "comparison"
    if row.get("metrics"):
        return "scored"
    if row.get("golden_output") is not None:
        return "validated"
    return "registered"
