"""The tracked series as it was known at an instant (#166).

`dataset.tracked` smooths backward over the whole window, so each value knows
the soundings after it. `dataset.known` gives the grid up to an instant as the
tracker knew it then, which is what a live issue at that instant reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.api import db
from services.prediction import dataset

pytest.importorskip("statsmodels")

from test_prediction_train import seed

TX, RX = "NIC3", "Yoshkar-Ola"


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as connection:
        yield connection


def _shift_after(conn, instant, offset):
    conn.execute(
        "UPDATE extraction SET muf = muf + ? WHERE sounding_id IN "
        "(SELECT id FROM sounding WHERE datetime > ?)",
        (offset, instant.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()


def _known(conn, instant):
    return next(iter(dataset.known(conn, "muf", TX, RX, "contour", [instant])))


def test_the_series_known_at_an_instant_ignores_soundings_after_it(conn):
    rng = np.random.default_rng()
    index = seed(conn, days=3)
    instant = index[int(rng.integers(300, 700))]
    before = _known(conn, instant)
    _shift_after(conn, instant, float(rng.uniform(3, 9)))
    assert before.iloc[36:].equals(_known(conn, instant).iloc[36:]), (
        "a sounding after the instant changed the series known at it")


def test_the_series_known_at_an_instant_ends_at_it(conn):
    rng = np.random.default_rng()
    index = seed(conn, days=3)
    instant = index[int(rng.integers(300, 700))]
    assert _known(conn, instant).index[-1] == instant, (
        "the series known at an instant does not end at it")


def test_the_series_known_at_the_last_grid_point_is_the_tracked_series(conn):
    seed(conn, days=3, gaps=slice(400, 440))
    whole = dataset.tracked(conn, "muf", TX, RX, "contour").values
    known = _known(conn, whole.index[-1])
    assert np.allclose(known.to_numpy(), whole.to_numpy(), atol=1e-3, equal_nan=True), (
        "the series known at the last grid point differs from the tracked one")
