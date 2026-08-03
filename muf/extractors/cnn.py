"""The autoencoder estimator -- experimental.

``MUF_clustering`` contains a small convolutional autoencoder
(``autoencoder_model_filtered_v2.h5``, built by ``myCNN_0.02.py``) trained to
map noisy ionogram images to cleaned ones. It is a *denoiser*, not a MUF
estimator: it produces an image, so a segmentation step still has to read the
MUF off its output. That is how it is wired here -- denoise, then hand the
result to :mod:`muf.extractors.contour`.

Two caveats, both real:

1. The bundled model expects ``(256, 208, 1)`` and was trained on rendered PNGs
   of a *different* geometry (a 2500-4000 km render at a different window
   length). Feeding it a resized gated dB tile is out-of-distribution. Its
   output should be treated as indicative until the model is retrained.

2. The file was written by Keras 2.x in 2022. Keras 3 -- what ships with
   TensorFlow 2.16+ -- will not load that format. Install ``tf-keras`` and set
   ``TF_USE_LEGACY_KERAS=1``, or retrain.

Retraining at the correct geometry is the way out of both, and the pipeline can
supply the training pairs itself: the gated dB tile is the input, and the
agreement of the other three estimators gives the target mask. That trainer is
**not written yet** -- until it is, this estimator only runs against a model you
supply with ``model_path``.

This estimator is excluded from the defaults for those reasons.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..pick import DEFAULT_MIN_RUN, pick_muf
from ..spectro import Ionogram
from . import MufResult
from .contour import DEFAULT_THRESHOLD_DB, contour_mask, segment

#: Input geometry the bundled model was built for (myCNN_0.02.py:51-52).
INPUT_SHAPE = (256, 208)

#: Where to look for a model when none is given. The second is the model
#: carried over from MUF_clustering, if it has been copied alongside the
#: package.
DEFAULT_MODEL_PATHS = (
    Path("models/muf_autoencoder.keras"),
    Path("models/autoencoder_model_filtered_v2.h5"),
)

# The dB window mapped onto the model's 0..1 input range. Matches the display
# range the source scripts used (vmin_dB=20, vmax_dB=75).
NORM_MIN_DB = 20.0
NORM_MAX_DB = 75.0


def _import_keras():
    """Return a Keras module able to load the model, or raise with guidance."""
    if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
        try:
            import tf_keras  # type: ignore
            return tf_keras
        except ImportError:
            pass
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "the cnn method needs TensorFlow: pip install tensorflow"
        ) from exc
    return tf.keras


def find_model(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        return path
    for candidate in DEFAULT_MODEL_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no autoencoder found. Looked for "
        + ", ".join(str(p) for p in DEFAULT_MODEL_PATHS)
        + ". Copy autoencoder_model_filtered_v2.h5 from MUF_clustering into "
        "models/ (it is a Keras 2.x file and needs `pip install tf-keras` plus "
        "TF_USE_LEGACY_KERAS=1), or pass model_path= to point at your own. Note "
        "the bundled model was trained on a different image geometry, so its "
        "output is indicative only."
    )


@lru_cache(maxsize=4)
def load_model(path: str):
    """Load and cache a model. Cached because loading dominates per-file cost."""
    keras = _import_keras()
    try:
        return keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise RuntimeError(
            f"could not load {path}: {exc}. If this is the 2022 "
            f"autoencoder_model_filtered_v2.h5, it is a Keras 2.x file and "
            f"Keras 3 cannot read it -- `pip install tf-keras` and set "
            f"TF_USE_LEGACY_KERAS=1."
        ) from exc


def to_model_input(db: np.ndarray) -> np.ndarray:
    """Resize and normalise a dB tile to the model's input tensor."""
    import cv2

    normed = np.clip((db - NORM_MIN_DB) / (NORM_MAX_DB - NORM_MIN_DB), 0.0, 1.0)
    # cv2.resize takes (width, height); the tile is [n_freq, n_range].
    resized = cv2.resize(
        normed.astype(np.float32),
        (INPUT_SHAPE[1], INPUT_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )
    return resized[np.newaxis, :, :, np.newaxis]


def from_model_output(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize the denoised output back to the tile's shape, in dB."""
    import cv2

    image = np.asarray(prediction).reshape(INPUT_SHAPE)
    restored = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return restored * (NORM_MAX_DB - NORM_MIN_DB) + NORM_MIN_DB


def extract(
    ion: Ionogram,
    model_path: str | Path | None = None,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_height: int = 2,
    min_run: int = DEFAULT_MIN_RUN,
    percentile: float = 100.0,
) -> MufResult:
    """Denoise with the autoencoder, then segment its output.

    Raises if TensorFlow or a usable model is missing -- the pipeline records
    that as this estimator's error and carries on with the others.
    """
    model = load_model(str(find_model(model_path)))

    prediction = model.predict(to_model_input(ion.db), verbose=0)
    denoised = from_model_output(prediction, ion.shape)

    binary = segment(denoised, threshold_db=threshold_db)
    mask_img, _ = contour_mask(binary, select="all", min_height=min_height)
    mask = mask_img.T
    presence = mask.any(axis=1)

    pick = pick_muf(
        presence, ion.freq,
        power_db=ion.db, vrange=ion.vrange,
        min_run=min_run, percentile=percentile,
    )
    return MufResult(method="cnn", pick=pick, presence=presence, mask=mask)
