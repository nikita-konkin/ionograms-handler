"""Loading a trained model file, and establishing what it expects.

A model artifact on disk is not self-evidently runnable. Three things have to
be recovered before it can be trusted with a number:

1. **What it is** -- sklearn, xgboost, Keras or torch, each with a different
   loader and a different way of describing its own input.
2. **What it expects** -- the ordered feature vector. sklearn records this as
   ``feature_names_in_`` when it was fitted on a DataFrame, which the source
   project's models were, so the contract can be *read* rather than
   reconstructed. Where it is absent the artifact is not importable, and
   guessing is how confident wrong numbers get produced.
3. **Whether it still behaves the way it did** -- see :func:`golden_check`.

**Modelled on ``muf.extractors.cnn``**, which solves the same problem for the
autoencoder: find the file, cache the load because loading dominates per-call
cost, and raise errors that name the remedy rather than the symptom. Its Keras
2-versus-3 warning applies verbatim here, because the source project writes
both formats -- ``nn_tuner.py`` saves ``save_format="tf"`` (a Keras 2
SavedModel directory, which Keras 3 cannot read) while newer files are the
Keras 3 ``.keras`` zip.
"""

from __future__ import annotations

import hashlib
import json
import warnings
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Frameworks the slim inference image can load. Anything else needs the
#: training image, and saying so at import is much kinder than an ImportError
#: from three frames inside a loader.
SLIM_FRAMEWORKS = ("sklearn", "xgboost")

#: How far a reloaded model's golden prediction may drift before it is treated
#: as a different model. Tight on purpose: this is the same estimator on the
#: same input, so the only honest tolerance is floating-point noise. A real
#: behaviour change from a library upgrade moves it far more than this.
GOLDEN_TOLERANCE = 1e-6

#: Pickle protocol 2 and above open with this. joblib writes a plain pickle
#: stream unless asked to compress, so the sniff is the pickle opcode.
PICKLE_MAGIC = b"\x80"

#: Zip. Both a Keras 3 ``.keras`` file and a torch ``.pt`` are zips, so the
#: magic alone does not settle it and the extension breaks the tie.
ZIP_MAGIC = b"PK\x03\x04"


class ArtifactError(RuntimeError):
    """The artifact cannot be loaded, or cannot be trusted once loaded."""


@dataclass(frozen=True)
class Contract:
    """What a model expects on its input, and what produced it.

    ``features`` is **ordered and verbatim**. sklearn's ``feature_names_in_``
    carries whatever order the fitting frame had -- in the source project that
    is ``set`` iteration order from ``data_lag_and_lag_diff``, which is neither
    sorted nor stable across runs. Reordering the columns does not raise; it
    quietly multiplies the wrong coefficient by the wrong number.
    """

    framework: str                     # sklearn | xgboost | keras | torch
    loader: str                        # joblib | keras | torch
    features: tuple[str, ...]
    n_features: int
    env: dict[str, Any] = field(default_factory=dict)

    @property
    def capability(self) -> str:
        """``slim`` when the inference image can load it, else ``deep``."""
        return "slim" if self.framework in SLIM_FRAMEWORKS else "deep"

    def as_json(self) -> str:
        return json.dumps(list(self.features))


def sha256(path: Path) -> str:
    """Content hash of the artifact. Identity, because the path is not.

    A models volume is shared and writable by the training job; a file at a
    given path is not necessarily the file that was registered there.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sniff(path: Path) -> tuple[str, str]:
    """Return ``(framework, loader)`` for an artifact, by content then name.

    Content first because extensions in this archive are not reliable -- the
    same joblib pickle appears as ``.sav`` and as ``.joblib`` depending on
    which script wrote it. The framework is refined after loading, since a
    pickle can hold an sklearn estimator or an XGBRegressor and only the
    unpickled object says which.
    """
    path = Path(path)

    if path.is_dir():
        if (path / "saved_model.pb").exists():
            raise ArtifactError(
                f"{path} is a Keras 2 SavedModel directory (it contains "
                f"saved_model.pb). Keras 3, which ships with TensorFlow 2.16+, "
                f"cannot read it: install `tf-keras` and set "
                f"TF_USE_LEGACY_KERAS=1 in the training image, or re-save it "
                f"as a .keras file from an environment that can still load it."
            )
        raise ArtifactError(f"{path} is a directory with no saved_model.pb in it")

    if not path.exists():
        raise ArtifactError(f"artifact not found: {path}")

    head = path.open("rb").read(4)

    if head.startswith(PICKLE_MAGIC):
        # Framework is provisional: refined in `load` from the actual object.
        return "sklearn", "joblib"

    if head.startswith(ZIP_MAGIC):
        if path.suffix == ".keras":
            return "keras", "keras"
        if path.suffix in (".pt", ".pth"):
            return "torch", "torch"
        raise ArtifactError(
            f"{path} is a zip archive but its extension ({path.suffix or 'none'}) "
            f"does not say whether it is a Keras .keras file or a torch "
            f"checkpoint. Rename it to the format it actually is."
        )

    raise ArtifactError(
        f"{path} is not a format this service recognises "
        f"(first bytes: {head!r}). Expected a joblib pickle, a Keras .keras "
        f"file, or a Keras 2 SavedModel directory."
    )


def _sklearn_origin_version(caught) -> str | None:
    """The sklearn version an estimator was pickled from, if it differs.

    Not readable from the loaded object: sklearn stores ``_sklearn_version`` in
    the pickled state and pops it during ``__setstate__``, so
    ``getattr(estimator, "_sklearn_version")`` is ``None`` afterwards. What it
    does instead is warn, and the warning carries the version -- so the warning
    is the only place the number survives.
    """
    for entry in caught:
        original = getattr(entry.message, "original_sklearn_version", None)
        if original:
            return str(original)
    return None


def _describe(estimator: Any) -> str:
    return f"{type(estimator).__module__}.{type(estimator).__name__}"


def _framework_of(estimator: Any) -> str:
    module = type(estimator).__module__ or ""
    if module.startswith("xgboost"):
        return "xgboost"
    if module.startswith("sklearn"):
        return "sklearn"
    if module.startswith("keras") or module.startswith("tensorflow"):
        return "keras"
    return module.split(".", 1)[0] or "unknown"


def _feature_names(estimator: Any) -> tuple[str, ...]:
    """Ordered input feature names, or empty when the artifact does not say."""
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return tuple(str(n) for n in names)

    # XGBoost keeps them on the booster rather than the wrapper.
    booster = getattr(estimator, "get_booster", None)
    if callable(booster):
        try:
            from_booster = booster().feature_names
        except Exception:                      # an unfitted or odd booster
            from_booster = None
        if from_booster:
            return tuple(str(n) for n in from_booster)

    return ()


def _load_pickle(path: Path) -> tuple[Any, str | None]:
    import joblib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator = joblib.load(path)
        except ModuleNotFoundError as exc:
            raise ArtifactError(
                f"{path} unpickles into a module this image does not have "
                f"({exc.name}). A joblib artifact carries no dependencies, only "
                f"references to them -- load it in the image that can supply "
                f"{exc.name}, or re-export it."
            ) from exc
        except Exception as exc:
            raise ArtifactError(f"could not unpickle {path}: {exc}") from exc
    return estimator, _sklearn_origin_version(caught)


def _keras_module():
    """A Keras able to load these files, or an error naming the image to use.

    ``muf.extractors.cnn`` makes the same import dance for the autoencoder and
    for the same reason: the legacy files predate Keras 3, and which module can
    read them depends on an environment variable rather than on the file.
    """
    import os

    if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
        try:
            import tf_keras
            return tf_keras
        except ImportError:
            pass
    try:
        import keras
        return keras
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.keras
    except ImportError as exc:
        raise ArtifactError(
            "this image has no Keras, so a Keras model cannot be loaded here. "
            "Such models are registered with capability 'deep' -- run them in "
            "the training image (ionograms-train), not the inference one "
            "(ionograms-infer), which carries sklearn and xgboost only."
        ) from exc


def _load_keras(path: Path) -> Any:
    keras = _keras_module()
    try:
        return keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise ArtifactError(
            f"could not load {path}: {exc}. If this file was written by "
            f"Keras 2, Keras 3 cannot read it -- `pip install tf-keras` and "
            f"set TF_USE_LEGACY_KERAS=1."
        ) from exc


@lru_cache(maxsize=8)
def _cached(path_str: str, mtime: float, size: int):
    """Load and cache. Keyed on path plus mtime plus size, never path alone.

    Same reasoning as the renderer's memoisation: the models volume is written
    by the training job, so a path is not an identity. mtime and size make a
    replaced file a cache miss rather than a stale hit.
    """
    path = Path(path_str)
    framework, loader = sniff(path)

    if loader == "joblib":
        estimator, origin = _load_pickle(path)
        framework = _framework_of(estimator)
        env: dict[str, Any] = {}
        if origin:
            env["sklearn"] = origin
        else:
            try:
                import sklearn
                env["sklearn"] = sklearn.__version__
            except ImportError:
                pass
    elif loader == "keras":
        estimator = _load_keras(path)
        env = {}
    else:
        raise ArtifactError(f"no loader implemented for {loader} artifacts")

    names = _feature_names(estimator)
    n_features = int(getattr(estimator, "n_features_in_", 0) or len(names))

    return estimator, Contract(
        framework=framework, loader=loader,
        features=names, n_features=n_features, env=env,
    )


def load(path: str | Path) -> tuple[Any, Contract]:
    """Load an artifact and describe its input contract.

    Returns the estimator and its :class:`Contract`. Does **not** verify the
    golden prediction -- callers that are about to produce numbers should use
    :func:`load_verified` instead.
    """
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"artifact not found: {path}")
    stat = path.stat()
    return _cached(str(path), stat.st_mtime, stat.st_size)


def predict(estimator: Any, frame) -> Any:
    """Run a model. The one place inference happens, and it never fits.

    Takes a DataFrame rather than an array so that sklearn checks the feature
    names itself: passing a bare ndarray to an estimator fitted with names
    warns and proceeds, which turns a column-order bug into a silent one.
    """
    return estimator.predict(frame)


def golden_row(contract: Contract) -> list[float]:
    """A deterministic synthetic input row, for :func:`golden_check`.

    Deliberately not random and not stored from real data: it has to be
    reproducible from the contract alone, and it must not embed a measurement
    whose meaning could later be misread as one.
    """
    return [10.0 + i for i in range(max(contract.n_features, len(contract.features)))]


def golden_check(estimator: Any, contract: Contract,
                 row: list[float], expected: float,
                 tolerance: float = GOLDEN_TOLERANCE) -> float:
    """Re-run the stored input and confirm the stored output comes back.

    This is the real compatibility test, and it exists because the version
    comparison it replaces cannot answer the question. The archive's estimators
    were pickled by scikit-learn 1.4.2 and this image carries a later one;
    sklearn warns about that on every load, but a warning says the versions
    differ, not that the *behaviour* does. Most such loads are fine. The ones
    that are not produce plausible numbers.

    Raises :class:`ArtifactError` on drift; returns the recomputed value.
    """
    import pandas as pd

    frame = pd.DataFrame([row], columns=list(contract.features) or None)
    got = float(predict(estimator, frame)[0])
    if expected is None:
        return got
    if abs(got - float(expected)) > tolerance:
        raise ArtifactError(
            f"golden check failed: this model predicted {expected!r} from the "
            f"stored row when it was imported and predicts {got!r} now, a "
            f"difference of {abs(got - float(expected)):.6g}. The library "
            f"versions under it have changed behaviour. Re-import it in this "
            f"environment after confirming the numbers are still right, or "
            f"pass --allow-version-skew to run it anyway and have every "
            f"forecast record that it was run across a skew."
        )
    return got


def load_verified(path: str | Path, row: list[float] | None,
                  expected: float | None,
                  allow_skew: bool = False) -> tuple[Any, Contract, dict]:
    """Load, then prove the model still behaves as it did at import.

    Returns ``(estimator, contract, quality)`` where ``quality`` records what
    was checked -- it goes onto every forecast row the model produces, so a
    number run across a known skew says so wherever it is read.
    """
    estimator, contract = load(path)
    quality: dict[str, Any] = {"golden": "skipped"}

    if row is None or expected is None:
        quality["golden"] = "absent"
        return estimator, contract, quality

    try:
        golden_check(estimator, contract, row, expected)
        quality["golden"] = "ok"
    except ArtifactError:
        if not allow_skew:
            raise
        quality["golden"] = "failed"
        quality["version_skew"] = True

    return estimator, contract, quality
