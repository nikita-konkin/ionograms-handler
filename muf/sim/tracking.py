"""Causal continuity of hump cuts between observations of one circuit.

History guides segmentation only: every range and level is measured afresh.
It is deliberately separate from forecast scoring and never sees future data.
"""

from __future__ import annotations

import numpy as np

from .. import trace

MAX_AGE_S = 900.0
MAX_RANGE_KM = trace.DEFAULT_MAX_JUMP_KM


def split(segments, previous=()):
    """Return branches and JSON history, matching parents before splitting.

    Only mutual nearest neighbours with at least half of the shorter band's
    overlap may inherit history. Ambiguous equal-distance matches start over.
    """
    curves = [trace.reconstruct(s) for s in segments]
    costs = np.full((len(segments), len(previous)), np.inf)
    for i, curve in enumerate(curves):
        if not curve.ok:
            continue
        for j, old in enumerate(previous):
            f, r = np.asarray(old["freq"]), np.asarray(old["range"])
            lo, hi = max(curve.freq[0], f[0]), min(curve.freq[-1], f[-1])
            span = min(np.ptp(curve.freq), np.ptp(f))
            if hi - lo < max(0.5, 0.5 * span):
                continue
            sample = np.linspace(lo, hi, 64)
            delta = np.median(np.abs(np.interp(sample, curve.freq, curve.vrange)
                                     - np.interp(sample, f, r)))
            if delta < MAX_RANGE_KM:
                costs[i, j] = delta
    branches, history, names = [], [], {}
    for i, (segment, curve) in enumerate(zip(segments, curves)):
        old = None
        if previous and np.isfinite(costs[i]).any():
            j = int(np.argmin(costs[i]))
            score = costs[i, j]
            if (np.sum(costs[i] <= score + 1.0) == 1
                    and np.sum(costs[:, j] <= score + 1.0) == 1):
                old = previous[j]
        cuts = []
        pieces = trace._branches(segment, trace.DEFAULT_MIN_SEGMENT_POINTS,
                                 () if old is None else old["cuts"], cuts)
        if curve.ok:
            sample = np.linspace(curve.freq[0], curve.freq[-1], 64)
            entry = {"freq": sample.tolist(),
                     "range": np.interp(sample, curve.freq, curve.vrange).tolist(),
                     "cuts": sorted(cuts),
                     "names": [None] * len(pieces)}
            history.append(entry)
            for k, piece in enumerate(pieces):
                # Reuse names only when the topology is unchanged.
                prior_name = (old["names"][k] if old is not None
                              and len(old.get("names", [])) == len(pieces) else None)
                names[id(piece)] = (entry, k, prior_name)
        branches.extend(pieces)
    return branches, history, names
