"""The clustering estimator: K-means over the ionogram's power values.

Carries across the method developed in ``MUF_clustering`` -- most recently
``ionogr_clustering_0.026.py`` (Nov 2025) and ``kmeans_clustering.py`` -- with
one change: it clusters the dB array directly instead of the RGB pixels of a
rendered PNG.

That change removes a whole class of error. The image-based scripts recovered
frequency from pixel position via a hardcoded ``ion_col_num = 1220`` and a
virtual-range span of 3500..2500 km that did not match what the renderer
actually produced (``MUF.py:333`` renders 2500..4000 km), and clustered colours
that a colormap had already quantised. Here the axes come from the file header.
Notably ``ion_col_num = 1220`` was right by construction -- it is
``len(iq) // window`` for this instrument -- but only for this window length.

Two cluster-selection rules are offered. ``centroid`` is the default and is the
natural one on a numeric array: the trace is simply the high-power end.
``smallest`` reproduces ``ionogr_clustering_0.026.py:180``, which sorts clusters
by pixel count and keeps the smallest, exploiting the fact that the trace is
sparse while the noise background is not.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..pick import DEFAULT_MIN_RUN, pick_muf
from ..spectro import Ionogram
from . import MufResult

DEFAULT_K = 11

#: Upper bound on retained clusters. On a scalar dB array the trace normally
#: collapses into a single cluster -- the image-based scripts needed 5 because
#: the colormap split it across several colour clusters -- so this rarely binds
#: and exists to stop a pathological sounding retaining half the array.
DEFAULT_N_RETAIN = 3

#: A cluster counts as trace when its centroid sits this far above the
#: ionogram's median. Median equalization pins that median near 25.6 dB, so the
#: effective cut lands around 43 dB -- the same physical level the other two
#: estimators use, reached adaptively rather than by fiat.
DEFAULT_MARGIN_DB = 15.0

#: Fitting on every cell is unnecessary -- the value distribution is what
#: matters and it is well sampled by a subset. Predict still labels every cell.
DEFAULT_FIT_SAMPLE = 20_000

DEFAULT_RANDOM_STATE = 0


def cluster(
    db: np.ndarray,
    k: int = DEFAULT_K,
    fit_sample: int = DEFAULT_FIT_SAMPLE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Label every cell of ``db``. Returns ``(labels [n_freq, n_range], centroids [k])``."""
    values = db.reshape(-1, 1).astype(np.float64)

    if 0 < fit_sample < len(values):
        rng = np.random.default_rng(random_state)
        fit_on = values[rng.choice(len(values), fit_sample, replace=False)]
    else:
        fit_on = values

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(fit_on)
    labels = model.predict(values).reshape(db.shape)
    return labels, model.cluster_centers_.ravel()


def select_clusters(
    labels: np.ndarray,
    centroids: np.ndarray,
    n_retain: int = DEFAULT_N_RETAIN,
    rule: str = "centroid",
    min_db: float | None = None,
) -> np.ndarray:
    """Indices of the clusters judged to hold the trace."""
    k = len(centroids)
    n_retain = max(1, min(n_retain, k))

    if rule == "centroid":
        order = np.argsort(centroids)[::-1]          # brightest first
        chosen = order[:n_retain]
    elif rule == "smallest":
        counts = np.bincount(labels.ravel(), minlength=k)
        order = np.argsort(counts)                   # sparsest first
        chosen = order[:n_retain]
    else:
        raise ValueError(f"unknown cluster selection rule {rule!r}")

    if min_db is not None:
        # No cluster bright enough means no trace. Returning the brightest
        # anyway would invent a detection out of noise -- which is exactly what
        # happens on a sounding where the transmitter was off, and it is how
        # this method used to report a MUF at the top of the band for a
        # recording containing nothing at all.
        chosen = chosen[centroids[chosen] >= min_db]
    return chosen


def extract(
    ion: Ionogram,
    k: int = DEFAULT_K,
    n_retain: int = DEFAULT_N_RETAIN,
    rule: str = "centroid",
    min_db: float | None = None,
    margin_db: float = DEFAULT_MARGIN_DB,
    min_run: int = DEFAULT_MIN_RUN,
    max_range_slope: float | None = None,
    percentile: float = 100.0,
    fit_sample: int = DEFAULT_FIT_SAMPLE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MufResult:
    """Estimate MUF by clustering the ionogram's dB values.

    Args:
        ion: the gated ionogram.
        k: number of clusters.
        n_retain: cap on clusters kept as trace.
        rule: ``centroid`` (brightest clusters) or ``smallest`` (sparsest,
            reproducing ``ionogr_clustering_0.026.py``).
        min_db: absolute floor on a retained cluster's centroid. When None it is
            derived per sounding as ``median(db) + margin_db``.
        margin_db: how far above the ionogram's median a cluster must sit.
        min_run, percentile: passed to the shared picker.
    """
    labels, centroids = cluster(ion.db, k=k, fit_sample=fit_sample,
                               random_state=random_state)
    if min_db is None and rule == "centroid":
        min_db = float(np.median(ion.db)) + margin_db
    chosen = select_clusters(labels, centroids, n_retain=n_retain,
                             rule=rule, min_db=min_db)

    mask = np.isin(labels, chosen)
    presence = mask.any(axis=1)

    pick = pick_muf(
        presence, ion.freq,
        power_db=ion.db, vrange=ion.vrange,
        min_run=min_run, percentile=percentile,
        max_range_slope=max_range_slope,
    )
    return MufResult(method="kmeans", pick=pick, presence=presence, mask=mask)
