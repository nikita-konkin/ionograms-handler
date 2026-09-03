"""Scoring a forecast, and the baselines it has to beat.

The two things worth pinning here are what counts as truth and what a censored
pick costs. Both are ways to produce a number that looks like accuracy and is
not: scoring against the tracker's filled points rewards a model for agreeing
with a Kalman filter through a gap where nothing was measured, and charging
full error against a band-edge bound penalises a model hardest at midday, which
is exactly where the number is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.api import db
from services.prediction import dataset, registry, scoring

TX, RX = "NIC", "DOB"
START = pd.Timestamp("2026-07-01")


def seed(conn, days: int = 40, step_min: int = 30, censor_from: int | None = None):
    """A diurnal MUF and LOF at a fixed cadence, optionally censored at the top.

    Coarser than the five-minute model grid on purpose: these tests match
    predictions to picks at the pick instants, and a long history matters more
    here than a dense one -- the 27-day recurrence baseline reaches further
    back than any other part of the service.
    """
    periods = int(days * 24 * 60 / step_min)
    index = pd.date_range(START, periods=periods, freq=f"{step_min}min")
    per_day = int(24 * 60 / step_min)
    t = np.arange(periods)
    muf = 18 + 7 * np.sin(2 * np.pi * (t - per_day / 4) / per_day)
    lof = 9 + 4 * np.sin(2 * np.pi * (t - per_day / 3) / per_day)

    for position, stamp in enumerate(index):
        limited = int(censor_from is not None and position >= censor_from)
        cursor = conn.execute(
            "INSERT INTO sounding (file, path, datetime, tx, rx, "
            "tx_lat, tx_lon, rx_lat, rx_lon, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"s{position}.h5", f"s{position}.h5",
             stamp.strftime("%Y-%m-%d %H:%M:%S"), TX, RX,
             35.1, 33.3, 56.6, 47.9, db.utcnow()),
        )
        conn.execute(
            "INSERT INTO extraction (sounding_id, method, muf, lof, snr, run, "
            "limited, loflim) VALUES (?,?,?,?,?,?,?,?)",
            (cursor.lastrowid, "contour", float(muf[position]),
             float(lof[position]), 55.0, 30, limited, 0),
        )
    conn.commit()
    return index, pd.Series(muf, index=index), pd.Series(lof, index=index)


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "s.sqlite3") as c:
        yield c


def add_model(conn, name="test-model", param="muf", target_src="measured"):
    return registry.register(
        conn, name=name, param=param, tx=TX, rx=RX, origin="trained",
        framework="sklearn", loader="joblib", capability="slim",
        artifact=f"/models/{name}.sav", sha256=name * 4,
        features=["a"], target_src=target_src,
    )


def add_forecasts(conn, model_id, values: pd.Series, horizon_s=86400,
                  param="muf", issued_at="2026-08-09T00:00:00Z"):
    conn.executemany(
        "INSERT OR REPLACE INTO forecast (model_id, param, tx, rx, issued_at, "
        "valid_at, horizon_s, value) VALUES (?,?,?,?,?,?,?,?)",
        [(model_id, param, TX, RX, issued_at,
          stamp.strftime(db.TIME_FORMAT), horizon_s, float(value))
         for stamp, value in values.items()],
    )
    conn.commit()


def just_after(index) -> str:
    """The end of the scored window, one step past the last pick.

    Every test seeds its own stretch of history, so "now" has to be tied to
    that stretch rather than to the wall clock -- a fixed date here would score
    an empty window the moment the seed length changed.
    """
    return (index[-1] + pd.Timedelta(minutes=1)).strftime(db.TIME_FORMAT)


# --------------------------------------------------------------------------
# Truth
# --------------------------------------------------------------------------

def test_truth_is_measured_never_tracked(conn):
    """A gap in the picks is a gap in the scoring, not a filled grid point."""
    index, muf, _ = seed(conn, days=3)
    conn.execute(
        "DELETE FROM extraction WHERE sounding_id IN "
        "(SELECT id FROM sounding WHERE datetime >= ? AND datetime < ?)",
        (index[20].strftime("%Y-%m-%d %H:%M:%S"),
         index[40].strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    assert len(observed) == len(index) - 20
    # The tracker would have filled every one of those instants.
    tracked = dataset.tracked(conn, "muf", TX, RX, "contour")
    assert tracked.n_filled > 0
    assert not observed.index.isin(index[20:40]).any()


def test_a_perfect_forecast_scores_zero(conn):
    _, muf, _ = seed(conn, days=3)
    pairs = scoring.pair(muf, scoring.truth(conn, "muf", TX, RX, "contour"))
    assert len(pairs) == len(muf)
    assert scoring.summarise(pairs, "muf")["mae"] == 0.0


def test_a_prediction_far_from_any_pick_is_dropped_not_guessed(conn):
    _, muf, _ = seed(conn, days=3)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    stray = pd.Series([20.0], index=[START + pd.Timedelta(days=400)])
    assert len(scoring.pair(stray, observed)) == 0


# --------------------------------------------------------------------------
# Censoring
# --------------------------------------------------------------------------

def test_a_muf_forecast_above_a_lower_bound_costs_nothing(conn):
    """`limited` means the sweep ran out, not that the MUF was that low."""
    seed(conn, days=3, censor_from=0)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    over = pd.Series(observed["value"].to_numpy() + 5.0, index=observed.index)
    error = scoring.absolute_error(scoring.pair(over, observed), "muf")
    assert np.allclose(error, 0.0)


def test_a_muf_forecast_below_a_lower_bound_still_costs(conn):
    seed(conn, days=3, censor_from=0)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    under = pd.Series(observed["value"].to_numpy() - 2.0, index=observed.index)
    error = scoring.absolute_error(scoring.pair(under, observed), "muf")
    assert np.allclose(error, 2.0)


def test_the_censored_sign_flips_for_lof(conn):
    """A `loflim` pick is an upper bound: below it is free, above it is not."""
    index, _, lof = seed(conn, days=3)
    conn.execute("UPDATE extraction SET loflim = 1")
    conn.commit()
    observed = scoring.truth(conn, "lof", TX, RX, "contour")
    under = pd.Series(observed["value"].to_numpy() - 3.0, index=observed.index)
    over = pd.Series(observed["value"].to_numpy() + 3.0, index=observed.index)
    assert np.allclose(scoring.absolute_error(scoring.pair(under, observed), "lof"), 0.0)
    assert np.allclose(scoring.absolute_error(scoring.pair(over, observed), "lof"), 3.0)


def test_bounds_are_counted_apart_from_the_headline_number(conn):
    """Half the picks censored must not dilute the MAE of the other half."""
    index, muf, _ = seed(conn, days=3)
    half = len(index) // 2
    conn.execute(
        "UPDATE extraction SET limited = 1 WHERE sounding_id IN "
        "(SELECT id FROM sounding WHERE datetime >= ?)",
        (index[half].strftime("%Y-%m-%d %H:%M:%S"),))
    conn.commit()

    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    over = pd.Series(observed["value"].to_numpy() + 4.0, index=observed.index)
    result = scoring.summarise(scoring.pair(over, observed), "muf")

    assert result["n"] == half
    assert result["mae"] == 4.0            # the measured half, undiluted
    assert result["n_censored"] == len(index) - half
    assert result["mae_censored"] == 0.0   # consistent with every bound


# --------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lead_s, expected", [
    (3600, 3600), (300, 3600), (21600, 21600), (86400, 86400),
    (82800, 86400), (604800, 604800),
])
def test_a_lead_is_filed_under_the_nearest_horizon(lead_s, expected):
    assert scoring.bucket(lead_s) == expected


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def test_persistence_is_offset_by_the_lead(conn):
    """At a 24 h lead it is yesterday's value at the same UTC minute."""
    index, muf, _ = seed(conn, days=5)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index[observed.index >= START + pd.Timedelta(days=2)])
    predicted = scoring.baseline_series(
        conn, "persistence", "muf", TX, RX, observed, at, 86400, START)
    yesterday = muf.reindex(at - pd.Timedelta(days=1))
    assert np.allclose(predicted.to_numpy(), yesterday.to_numpy())


def test_persistence_at_a_short_lead_is_the_recent_value(conn):
    """Not a fixed day: at a 1 h lead the comparison is what was known 1 h ago."""
    index, muf, _ = seed(conn, days=3)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index[10:])
    predicted = scoring.baseline_series(
        conn, "persistence", "muf", TX, RX, observed, at, 3600, START)
    assert np.allclose(predicted.to_numpy(),
                       muf.reindex(at - pd.Timedelta(hours=1)).to_numpy())


def test_recurrence_reaches_back_a_solar_rotation(conn):
    index, muf, _ = seed(conn, days=40)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index[observed.index >= START + scoring.RECURRENCE])
    predicted = scoring.baseline_series(
        conn, "recurrence-27d", "muf", TX, RX, observed, at, 86400, START)
    assert len(predicted) == len(at)
    assert np.allclose(predicted.to_numpy(),
                       muf.reindex(at - scoring.RECURRENCE).to_numpy())


def test_recurrence_drops_instants_with_no_history_behind_them(conn):
    """Nothing 27 days back means no pair, not an interpolated stand-in."""
    _, muf, _ = seed(conn, days=5)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index)
    predicted = scoring.baseline_series(
        conn, "recurrence-27d", "muf", TX, RX, observed, at, 86400, START)
    assert len(predicted) == 0


def test_the_harmonic_baseline_refuses_to_fit_on_what_it_is_scored_on(conn):
    """Fitting on the scored window would put an oracle on the leaderboard."""
    seed(conn, days=5)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index)
    with pytest.raises(scoring.ScoringError, match="training window earlier"):
        scoring.baseline_series(conn, "harmonic", "muf", TX, RX, observed,
                                at, 86400, START)


def test_the_harmonic_baseline_tracks_a_diurnal_series(conn):
    index, muf, _ = seed(conn, days=10)
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    split = START + pd.Timedelta(days=5)
    at = pd.DatetimeIndex(observed.index[observed.index >= split])
    predicted = scoring.baseline_series(
        conn, "harmonic", "muf", TX, RX, observed, at, 86400, split)
    result = scoring.summarise(scoring.pair(predicted, observed), "muf")
    # A clean sine fitted by two harmonics: good, and nowhere near perfect,
    # which is what a baseline is supposed to be.
    assert result["mae"] < 1.0


def test_the_harmonic_baseline_needs_coordinates(conn):
    seed(conn, days=10)
    conn.execute("UPDATE sounding SET tx_lat = NULL, rx_lat = NULL")
    conn.commit()
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index)
    with pytest.raises(scoring.ScoringError, match="no coordinates"):
        scoring.baseline_series(conn, "harmonic", "muf", TX, RX, observed, at,
                                86400, START + pd.Timedelta(days=5))


def test_iri_says_why_it_has_nothing_for_lof(conn):
    seed(conn, days=3)
    with pytest.raises(scoring.ScoringError, match="absorption floor"):
        scoring.iri(conn, "lof", TX, RX)


def test_iri_is_read_from_the_stored_reference_rows(conn):
    index, muf, _ = seed(conn, days=3)
    conn.execute(
        "INSERT INTO reference (sounding_id, source, param, value) "
        "SELECT id, 'iri', 'muf', 17.0 FROM sounding")
    conn.commit()
    observed = scoring.truth(conn, "muf", TX, RX, "contour")
    at = pd.DatetimeIndex(observed.index)
    predicted = scoring.baseline_series(conn, "iri", "muf", TX, RX, observed,
                                        at, 86400, START)
    assert len(predicted) == len(at)
    assert np.allclose(predicted.to_numpy(), 17.0)


def test_an_unavailable_baseline_is_stored_with_its_reason(conn):
    """A missing row reads as neglect; a row saying why is an answer."""
    index, _, _ = seed(conn, days=3)
    scoring.score_baselines(conn, "muf", TX, RX, "contour", (86400,),
                            window_days=2, now=just_after(index))
    rows = {r["subject"]: r for r in scoring.scores(conn, "muf", TX, RX)}
    assert "baseline:iri" in rows
    assert rows["baseline:iri"]["n"] == 0
    assert "no IRI rows stored" in rows["baseline:iri"]["detail"]["unavailable"]


# --------------------------------------------------------------------------
# Scoring a model
# --------------------------------------------------------------------------

def test_scoring_a_model_writes_rows_and_marks_it_scored(conn):
    index, muf, _ = seed(conn, days=12)
    model_id = add_model(conn)
    assert registry.state_of(registry.get(conn, model_id)) != "scored"

    window = index[-96:]
    add_forecasts(conn, model_id, muf.reindex(window) + 0.5)
    result = scoring.score_model(conn, registry.get(conn, model_id), TX, RX,
                                 "contour", window_days=30, now=just_after(index))

    assert result["scored"] > 0
    assert result["mae"]["86400"] == 0.5
    assert registry.state_of(registry.get(conn, model_id)) == "scored"
    row = [r for r in scoring.scores(conn, "muf", TX, RX)
           if r["subject"] == f"model:{model_id}"][0]
    assert row["horizon_s"] == 86400
    assert row["bias"] == 0.5


def test_a_model_with_nothing_due_yet_is_reported_not_scored(conn):
    index, _, _ = seed(conn, days=3)
    model_id = add_model(conn)
    future = pd.Series([20.0], index=[index[-1] + pd.Timedelta(days=365)])
    add_forecasts(conn, model_id, future)
    result = scoring.score_model(conn, registry.get(conn, model_id), TX, RX,
                                 "contour", now=just_after(index))
    assert result["scored"] == 0
    assert "valid_at has passed" in result["detail"]


def test_the_latest_issue_wins_where_two_cover_the_same_instant(conn):
    """Two issues at the same lead: the one made with more information counts."""
    index, muf, _ = seed(conn, days=12)
    model_id = add_model(conn)
    window = index[-96:]
    add_forecasts(conn, model_id, muf.reindex(window) + 4.0,
                  issued_at="2026-07-01T00:00:00Z")
    add_forecasts(conn, model_id, muf.reindex(window) + 0.25,
                  issued_at="2026-07-02T00:00:00Z")
    result = scoring.score_model(conn, registry.get(conn, model_id), TX, RX,
                                 "contour", now=just_after(index))
    assert result["mae"]["86400"] == 0.25


def test_a_pass_scores_every_model_and_the_baselines_beside_it(conn):
    index, muf, _ = seed(conn, days=40)
    model_id = add_model(conn)
    window = index[-96 * 8:]
    add_forecasts(conn, model_id, muf.reindex(window) + 0.5)

    scoring.run_once(conn, ("muf",), "contour", window_days=8,
                     now=just_after(index))
    board = scoring.leaderboard(conn, "muf", TX, RX)

    kinds = [entry["kind"] for entry in board]
    assert kinds.count("model") == 1
    assert kinds.count("baseline") == len(scoring.BASELINES)
    # Models sort before baselines, so the operator reads the thing being
    # judged first and what it is judged against underneath.
    assert kinds == sorted(kinds, key=lambda k: 0 if k == "model" else 1)
    assert board[0]["name"] == "test-model"
    assert board[0]["state"] == "scored"


def test_the_leaderboard_names_models_rather_than_their_ids(conn):
    seed(conn, days=5)
    model_id = add_model(conn, name="xgb-censored-r7")
    scoring.store(conn, f"model:{model_id}", "muf", TX, RX, 86400,
                  {"n": 10, "mae": 0.9})
    entry = scoring.leaderboard(conn, "muf", TX, RX)[0]
    assert entry["name"] == "xgb-censored-r7"
    assert entry["origin"] == "trained"
    assert entry["model_id"] == model_id


def test_a_baseline_reports_once_not_once_per_horizon(conn):
    """Four horizons repeating the same "no IRI here" line four times is noise
    in every log an operator reads."""
    index, _, _ = seed(conn, days=5)
    results = scoring.score_baselines(conn, "muf", TX, RX, "contour",
                                      scoring.HORIZONS, window_days=2,
                                      now=just_after(index))
    assert len(results) == len(scoring.BASELINES)
    assert set(results[0]["mae"]) == {str(h) for h in scoring.HORIZONS}
    iri = [r for r in results if r["name"] == "iri"][0]
    assert "absorption floor" not in (iri["detail"] or "")   # muf, so it is a
    assert "no IRI rows stored" in iri["detail"]             # storage problem


def test_a_baseline_that_finds_no_pairs_says_why(conn):
    """27 days back into a five-day archive: a gap, not a mystery."""
    index, _, _ = seed(conn, days=5)
    scoring.score_baselines(conn, "muf", TX, RX, "contour", (86400,),
                            window_days=2, now=just_after(index))
    row = [r for r in scoring.scores(conn, "muf", TX, RX)
           if r["subject"] == "baseline:recurrence-27d"][0]
    assert row["n"] == 0
    assert "does not go far enough" in row["detail"]["unavailable"]


def test_a_model_beaten_by_a_baseline_is_surfaced_not_demoted(conn):
    """The crossing is reported; the model stays live until a human says
    otherwise. A service that demoted on its own would change a published
    product with nothing in the logs having asked for it."""
    index, muf, _ = seed(conn, days=12)
    model_id = add_model(conn)
    registry.activate(conn, model_id)

    window = index[-96 * 3:]
    add_forecasts(conn, model_id, muf.reindex(window) + 5.0)
    scoring.run_once(conn, ("muf",), "contour", window_days=10,
                     now=just_after(index))

    board = scoring.leaderboard(conn, "muf", TX, RX)
    crossed = scoring.drift(board)

    assert crossed, "a model five MHz out must not outrank every baseline"
    # The *best* baseline is named, not the first one that happened to beat it:
    # what an operator needs to know is what the model is losing to.
    board_best = min(e["mae"]["86400"] for e in board
                     if e["kind"] == "baseline" and e["mae"].get("86400") is not None)
    assert crossed[0]["baseline_mae"] == board_best
    assert crossed[0]["baseline"] in scoring.BASELINES
    assert crossed[0]["model_mae"] > crossed[0]["baseline_mae"]
    assert registry.get(conn, model_id)["active"] == 1


def test_nothing_is_flagged_when_the_model_wins(conn):
    index, muf, _ = seed(conn, days=12)
    model_id = add_model(conn)
    registry.activate(conn, model_id)
    window = index[-96 * 3:]
    add_forecasts(conn, model_id, muf.reindex(window))     # exact
    scoring.run_once(conn, ("muf",), "contour", window_days=10,
                     now=just_after(index))
    assert scoring.drift(scoring.leaderboard(conn, "muf", TX, RX)) == []


def test_a_comparison_model_is_never_flagged_as_drifting(conn):
    """Only what is live can be overtaken; a legacy import is on the board to
    be compared, not to be an operational forecast."""
    index, muf, _ = seed(conn, days=12)
    model_id = add_model(conn, name="brisbane-huber", target_src="modelled")
    add_forecasts(conn, model_id, muf.reindex(index[-96 * 3:]) + 5.0)
    scoring.run_once(conn, ("muf",), "contour", window_days=10,
                     now=just_after(index))
    assert scoring.drift(scoring.leaderboard(conn, "muf", TX, RX)) == []


def test_diurnal_finds_a_night_the_headline_mae_hides():
    """The whole point of the split.

    Two models with the same overall MAE: one uniformly mediocre, one perfect
    by day and badly high at night. Only the second has a cause worth chasing,
    and only the split can tell them apart.
    """
    when = pd.date_range("2026-08-01", periods=24 * 8, freq="h")
    night = np.isin(when.hour, [0, 1, 2, 3, 4])
    observed = np.where(night, 13.0, 24.0)

    def summarised(predicted):
        pairs = scoring.Pairs(valid_at=when, predicted=predicted,
                              observed=observed,
                              censored=np.zeros(len(when), dtype=bool))
        return scoring.summarise(pairs, "muf"), scoring.diurnal(pairs, "muf")

    flat, flat_hours = summarised(observed + 0.5)
    night_high = observed + np.where(night, 2.4, 0.0)
    peaked, peaked_hours = summarised(night_high)

    assert flat["mae"] == pytest.approx(peaked["mae"], abs=0.01), \
        "the fixture does not actually hide the difference"

    assert {h["mae"] for h in flat_hours} == {0.5}
    worst = max(peaked_hours, key=lambda h: h["mae"])
    assert worst["hour"] in (0, 1, 2, 3, 4)
    assert worst["mae"] == pytest.approx(2.4)
    # Sitting above a trough it cannot reach: the bias carries the sign.
    assert worst["bias"] == pytest.approx(2.4)


def test_diurnal_reports_only_the_hours_that_have_pairs():
    when = pd.DatetimeIndex(["2026-08-01T06:00", "2026-08-01T06:30",
                             "2026-08-01T18:00"])
    pairs = scoring.Pairs(valid_at=when, predicted=np.array([1.0, 2.0, 3.0]),
                          observed=np.array([1.0, 1.0, 1.0]),
                          censored=np.zeros(3, dtype=bool))

    hours = scoring.diurnal(pairs, "muf")

    assert [h["hour"] for h in hours] == [6, 18]
    assert [h["n"] for h in hours] == [2, 1]
    assert hours[0]["mae"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Is the difference real, or is it the window?
#
# The failure these exist for: on 2026-08-27 a voting model read 1.10 MHz on
# one day and 1.58 on the next and looked like a regression, while persistence
# -- which involves no model at all -- moved 1.07 to 1.47 over the same two
# windows. Most of the change was the ionosphere. Only a paired comparison and
# a ratio can tell those apart.
# --------------------------------------------------------------------------

def _pairs(when, predicted, observed):
    return scoring.Pairs(valid_at=pd.DatetimeIndex(when),
                         predicted=np.asarray(predicted, dtype=float),
                         observed=np.asarray(observed, dtype=float),
                         censored=np.zeros(len(when), dtype=bool))


def test_a_real_difference_is_called_distinguishable():
    when = pd.date_range("2026-08-01", periods=300, freq="h")
    rng = np.random.default_rng(1)
    truth = 20 + rng.normal(0, 1.0, len(when))
    good = _pairs(when, truth + rng.normal(0, 0.2, len(when)), truth)
    bad = _pairs(when, truth + rng.normal(0, 2.0, len(when)), truth)

    result = scoring.compare(good, bad, "muf")

    assert result["distinguishable"] is True
    assert result["delta"] < 0, "the better forecast has the negative delta"
    assert result["delta_hi"] < 0, "the interval clears zero"
    assert result["skill"] > 0.5


def test_a_tenth_of_a_megahertz_on_a_short_window_is_not_a_finding():
    """The case the console kept reporting as 'loses to persistence'."""
    when = pd.date_range("2026-08-01", periods=140, freq="h")
    rng = np.random.default_rng(2)
    truth = 20 + rng.normal(0, 2.0, len(when))
    ours = _pairs(when, truth + rng.normal(0, 1.5, len(when)), truth)
    theirs = _pairs(when, truth + rng.normal(0, 1.45, len(when)), truth)

    result = scoring.compare(ours, theirs, "muf")

    assert result["distinguishable"] is False
    assert result["delta_lo"] < 0 < result["delta_hi"], "the interval straddles zero"


def test_skill_survives_a_window_that_got_harder():
    """The ratio is what makes two holdout windows comparable.

    Same model quality throughout -- errors scaled by the same factor as the
    baseline's -- so the absolute MAE doubles and the skill does not move.
    """
    when = pd.date_range("2026-08-01", periods=300, freq="h")
    rng = np.random.default_rng(3)
    truth = np.zeros(len(when))
    ours = rng.normal(0, 1.0, len(when))
    theirs = rng.normal(0, 1.25, len(when))

    quiet = scoring.compare(_pairs(when, ours, truth),
                            _pairs(when, theirs, truth), "muf")
    stormy = scoring.compare(_pairs(when, ours * 2, truth),
                             _pairs(when, theirs * 2, truth), "muf")

    # Tolerances are set by the 4-decimal rounding the stored values carry,
    # not by the arithmetic, which is exact.
    assert stormy['mae'] == pytest.approx(2 * quiet['mae'], abs=1e-3)
    assert stormy['skill'] == pytest.approx(quiet['skill'], abs=1e-3)


def test_a_comparison_uses_only_the_instants_both_forecasts_covered():
    when = pd.date_range("2026-08-01", periods=200, freq="h")
    truth = np.full(len(when), 20.0)
    ours = _pairs(when, truth + 0.5, truth)
    theirs = _pairs(when[:120], truth[:120] + 1.0, truth[:120])

    result = scoring.compare(ours, theirs, "muf")

    assert result["n"] == 120


def test_too_few_shared_instants_get_no_comparison_at_all():
    when = pd.date_range("2026-08-01", periods=10, freq="h")
    truth = np.full(len(when), 20.0)

    assert scoring.compare(_pairs(when, truth + 0.5, truth),
                           _pairs(when, truth + 1.0, truth), "muf") is None


def test_the_interval_does_not_move_between_two_reads():
    """A table that changes on reload is a table nobody can quote."""
    when = pd.date_range("2026-08-01", periods=200, freq="h")
    rng = np.random.default_rng(4)
    truth = 20 + rng.normal(0, 1, len(when))
    ours = _pairs(when, truth + rng.normal(0, 1, len(when)), truth)
    theirs = _pairs(when, truth + rng.normal(0, 1, len(when)), truth)

    assert scoring.compare(ours, theirs, "muf") == \
        scoring.compare(ours, theirs, "muf")
