"""Restored MUF samples for the gaps in the public series (#160).

The page shows a restored sample in a color of its own and the publisher keeps
it in a block of its own, so a guess is never mistaken for a measurement.

The rules come from a study on a month of published ``nic2-yoshkar-ola`` data
(2026-10-22 to 2026-11-21), which hid observed stretches, restored them and
compared the result with what had been measured:

* up to an hour, the straight line between the edges of the gap is as good as
  anything tried (mean error 0.72 MHz);
* from an hour to 18 hours, a Gaussian process with a 24-hour periodic kernel
  is the best of eight methods (1.24 MHz at 4 hours, 1.44 MHz at 18 hours,
  against 5.5 and 7.8 MHz for the line at 12 and 18 hours), and its 95% band
  held 95% of the hidden values;
* longer gaps were not tested and stay empty.

The kernel is the one fitted in that study and is not refitted on each pass:
a fixed kernel restores the same gap the same way every time.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import final

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, WhiteKernel

GAP = timedelta(minutes=25)
LINE = timedelta(hours=1)
LONGEST = timedelta(hours=18)
STEP = timedelta(minutes=10)
CONTEXT = timedelta(days=2)
NEAR = timedelta(hours=6)
KERNEL = (ConstantKernel(0.824 ** 2, "fixed")
          * ExpSineSquared(0.449, 1.0, "fixed", "fixed")
          * RBF(30.0, "fixed")
          + ConstantKernel(0.225 ** 2, "fixed") * RBF(0.0252, "fixed")
          + WhiteKernel(0.0115, "fixed"))


@final
class Restoration:
    """The restored MUF samples of one measured series.

    A gap is a stretch longer than 25 minutes between two measured values,
    whether the soundings are missing or carry no MUF. Its samples are spaced
    as evenly as the ten-minute cadence allows, strictly between its edges.
    """

    def __init__(self, t: list[str], muf: list):
        self._t = t
        self._muf = muf

    def block(self) -> dict:
        """The restored samples as the page reads them: times, values, and
        the one-sigma uncertainty, which is None on the straight line."""
        seen = [(datetime.fromisoformat(x).replace(tzinfo=timezone.utc), float(v))
                for x, v in zip(self._t, self._muf) if v is not None]
        when = np.array([w.timestamp() for w, _ in seen])
        value = np.array([v for _, v in seen])
        out = {"t": [], "muf": [], "sigma": []}
        for i in range(len(seen) - 1):
            span = seen[i + 1][0] - seen[i][0]
            if span <= GAP or span > LONGEST:
                continue
            count = max(1, round(span / STEP) - 1)
            at = when[i] + np.arange(1, count + 1) * span.total_seconds() / (count + 1)
            if span <= LINE:
                guess, sigma = np.interp(at, when, value), [None] * count
            else:
                guess, spread = self._process(when, value, when[i], when[i + 1], at)
                sigma = [round(float(s), 3) for s in spread]
            out["t"] += [datetime.fromtimestamp(round(a), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                         for a in at]
            out["muf"] += [round(float(g), 3) for g in guess]
            out["sigma"] += sigma
        return out

    def _process(self, when, value, left, right, at):
        """The Gaussian process over two days on either side of the gap; the
        far part thinned to every third sample, which costs nothing measurable
        in accuracy and keeps each fit to a fraction of a second."""
        near = (when >= left - NEAR.total_seconds()) & (when <= right + NEAR.total_seconds())
        keep = ((when >= left - CONTEXT.total_seconds()) & (when <= right + CONTEXT.total_seconds())
                & (near | (np.arange(len(when)) % 3 == 0)))
        days = (when[keep] - left) / 86400
        model = GaussianProcessRegressor(KERNEL, optimizer=None, normalize_y=True)
        model.fit(days[:, None], value[keep])
        return model.predict(((at - left) / 86400)[:, None], return_std=True)
