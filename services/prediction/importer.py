"""Registering a trained model file so it can be run.

    python -m services.prediction.importer <artifact> --param muf [--tx NIC --rx DOB]

One command for all three kinds of artifact -- the legacy files from the
research project, the output of a training run, and anything an operator drops
on the models volume by hand -- because they differ only in what they can prove
about themselves, and the difference is recorded rather than assumed.

**Nothing that answers HTTP ever calls this.** Registering a model means
reading a file from a shared volume and running code out of it, and giving the
prediction service an inbound surface is what the architecture spends real
effort avoiding. That has not changed now that the console has an upload
button: the api hashes the bytes, checks four magic bytes and writes them to a
quarantine volume *without opening them*, and
``services/prediction/registrar.py`` -- a container that listens on nothing --
is what calls the function below. ``tests/test_prediction_upload.py`` greps
``services/api/`` to keep it that way.

This module is still also a command, run as a one-off container beside
``train``, which is what a scripted bulk import wants.

What it refuses, and why refusing beats guessing:

* **An artifact that does not name its inputs.** Without ``feature_names_in_``
  the column order is unknowable, and the wrong order does not raise -- it
  multiplies the wrong coefficient by the wrong number and returns a plausible
  megahertz. ``--features`` supplies the list explicitly for the rare artifact
  that genuinely lacks one.
* **A legacy artifact being claimed as measured.** ``--target-src`` defaults to
  ``modelled`` for ``--origin legacy``, so the comparison-only rule holds by
  default rather than by remembering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..api import db
from . import artifacts, legacy_features, registry, store


def describe(row: dict) -> str:
    """One line an operator can read, in the style `watch.describe` set."""
    where = f"{row['tx']} -> {row['rx']}" if row.get("tx") else "unbound"
    features = row.get("features") or []
    env = row.get("env") or {}
    version = f" {env.get('sklearn')}" if env.get("sklearn") else ""
    return (f"#{row['id']} {row['name']} [{row['param']}] {where} · "
            f"{row['origin']} · {row['framework']}{version} · "
            f"{len(features)} features · {row['capability']} · {row['state']}")


def import_artifact(path: Path, *, param: str, name: str | None = None,
                    tx: str | None = None, rx: str | None = None,
                    origin: str = "legacy", target_src: str | None = None,
                    features: list[str] | None = None,
                    period: int = legacy_features.DEFAULT_DECOMPOSITION_PERIOD,
                    period_assumed: bool = True,
                    note: str | None = None,
                    trained_from: str | None = None,
                    trained_to: str | None = None,
                    conn=None) -> dict:
    """Load, introspect, prove it runs, and write the registry row."""
    path = Path(path)
    estimator, contract = artifacts.load(path)

    names = tuple(features) if features else contract.features
    if not names:
        raise artifacts.ArtifactError(
            f"{path} does not record the names of its inputs, so the order to "
            f"feed them in is unknown -- and the wrong order returns a wrong "
            f"number rather than an error. Supply them with "
            f"--features <file.json>, listing them in the order the model was "
            f"fitted on."
        )
    if contract.n_features and len(names) != contract.n_features:
        raise artifacts.ArtifactError(
            f"{path} expects {contract.n_features} inputs but {len(names)} "
            f"names were supplied."
        )

    contract = artifacts.Contract(
        framework=contract.framework, loader=contract.loader,
        features=tuple(names), n_features=len(names), env=contract.env,
    )

    recipe = legacy_features.parse(names, period=period, assumed=period_assumed)

    # Prove it runs, and record what it said, before anything is written.
    row = artifacts.golden_row(contract)
    output = artifacts.golden_check(estimator, contract, row, None)

    if target_src is None:
        target_src = "modelled" if origin == "legacy" else "measured"

    fields = dict(
        name=name or path.stem, param=param, tx=tx, rx=rx, origin=origin,
        framework=contract.framework, loader=contract.loader,
        capability=contract.capability,
        artifact=str(path), sha256=artifacts.sha256(path),
        features=list(names), target_alias=recipe.alias,
        feature_recipe=recipe.as_dict(), env=contract.env,
        golden_input=row, golden_output=output,
        target_src=target_src, note=note,
        trained_from=trained_from, trained_to=trained_to,
    )

    owned = conn is not None
    if owned:
        model_id = registry.register(conn, **fields)
        return registry.get(conn, model_id)
    with db.session() as connection:
        model_id = registry.register(connection, **fields)
        return registry.get(connection, model_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.prediction.importer",
        description="Register a trained model so the inference service can run it.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--param", required=True, choices=("muf", "lof"))
    parser.add_argument("--name", default=None,
                        help="default: the file stem")
    parser.add_argument("--tx", default=None,
                        help="transmitter; with --rx, binds the model to a circuit")
    parser.add_argument("--rx", default=None, help="receiver")
    parser.add_argument("--origin", default="legacy",
                        choices=("legacy", "trained", "imported"))
    parser.add_argument("--target-src", default=None, choices=("measured", "modelled"),
                        help="default: modelled for legacy, measured otherwise. "
                             "A modelled target can never be promoted.")
    parser.add_argument("--features", type=Path, default=None,
                        help="JSON list, for an artifact that does not name its "
                             "own inputs. In fitting order.")
    parser.add_argument("--period", type=int,
                        default=legacy_features.DEFAULT_DECOMPOSITION_PERIOD,
                        help="seasonal decomposition period in samples "
                             "(default: %(default)s). Not recoverable from the "
                             "artifact -- this is the one part of the recipe "
                             "that is assumed rather than read.")
    parser.add_argument("--note", default=None)
    parser.add_argument("--store", action="store_true",
                        help="copy the artifact into the content-addressed "
                             "store first, and register the stored object "
                             "rather than the path given. Off by default: a "
                             "row that already names a path keeps working, "
                             "and rewriting one under a running `infer` is a "
                             "worse failure than an inconsistency.")
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args(argv)

    names = json.loads(args.features.read_text()) if args.features else None

    artifact = args.artifact
    if args.store:
        try:
            artifact = store.put(artifact, artifacts.sha256(artifact))
        except (store.StoreError, OSError) as exc:
            print(f"import failed: {exc}", file=sys.stderr)
            return 1
        # The stem becomes the digest once it is in the store, so a name that
        # was going to be defaulted has to be pinned to the original file.
        args.name = args.name or Path(args.artifact).stem

    try:
        with db.session(args.db) as conn:
            row = import_artifact(
                artifact, param=args.param, name=args.name,
                tx=args.tx, rx=args.rx, origin=args.origin,
                target_src=args.target_src, features=names,
                period=args.period, note=args.note, conn=conn,
            )
    except (artifacts.ArtifactError, legacy_features.RecipeError,
            registry.RegistryError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1

    print(describe(row))
    if row["target_src"] != "measured":
        print("  registered for comparison only: a modelled target cannot be "
              "promoted to the operational forecast.")
    if row.get("feature_recipe", {}).get("period_assumed"):
        print(f"  decomposition period assumed to be "
              f"{row['feature_recipe']['period']} samples; the artifact does "
              f"not record it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
