"""The low-frequency end: LOF, the threshold ladder, and the band floor.

Correctness comes from synthetic soundings, as it does for the MUF: the echo is
injected between two known frequency bins and the picker has to find the lower
one. Real recordings have no ground truth, so they are used here only to check
that the number behaves like absorption -- which is the whole claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import extractors, lof, spectro
from muf.export import saoxml

from conftest import synth_iq

WINDOW = 512
N_FREQ = 200
HALF_SPAN = 60_000.0
ECHO_RANGE = 2700.0
FIRST_BIN = 40
LAST_BIN = 150


@pytest.fixture
def sounding(make_lfs):
    """An echo present only between bins 40 and 150 -- a known LOF and MUF."""
    iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=ECHO_RANGE,
                  half_span_km=HALF_SPAN, echo_first_bin=FIRST_BIN,
                  echo_last_bin=LAST_BIN)
    path = make_lfs(iq, name="lof.lfs", dur=2)
    return spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))


# --- the pick ----------------------------------------------------------------

def test_lof_finds_the_bottom_of_the_injected_echo(sounding):
    """Within one frequency bin of where the echo was switched on."""
    pick = lof.lof_at(sounding, 43.0)

    assert pick.ok
    assert pick.lof_mhz == pytest.approx(sounding.freq[FIRST_BIN],
                                         abs=2 * sounding.cal.freq_step_mhz)


def test_lof_sits_below_the_muf(sounding):
    result = extractors.get("contour")(sounding)
    low = lof.pick_lof(result.presence, sounding.freq)

    assert low.ok and result.pick.ok
    assert low.lof_mhz < result.pick.muf_mhz


def test_the_threshold_travels_with_the_value(sounding):
    """A LOF without its detection level is not reproducible."""
    pick = lof.lof_at(sounding, 47.5)
    assert pick.threshold_db == 47.5


def test_no_echo_gives_no_pick(make_lfs):
    iq = (np.random.default_rng(3).normal(size=(N_FREQ * WINDOW, 2))
          .astype(np.float32).view(np.complex64).ravel())
    ion = spectro.compute(make_lfs(iq, name="quiet.lfs", dur=2),
                          window=WINDOW, gate_km=(2000.0, 5000.0))

    assert not lof.lof_at(ion, 43.0).ok


def test_presence_mismatch_is_rejected(sounding):
    with pytest.raises(ValueError):
        lof.pick_lof(np.zeros(5, dtype=bool), sounding.freq)


# --- the ladder --------------------------------------------------------------

def test_ladder_is_monotonic_in_threshold(sounding):
    """A stricter level can only move the LOF up.

    Detections above 50 dB are a subset of those above 43, so every run at the
    higher level sits inside one at the lower. If this ever inverts, the ladder
    is measuring something other than a signal level.
    """
    rungs = lof.ladder(sounding)
    values = [rungs[level].lof_mhz for level in sorted(rungs)
              if rungs[level].ok]

    assert len(values) >= 2
    assert values == sorted(values)


def test_ladder_covers_the_requested_levels(sounding):
    rungs = lof.ladder(sounding, thresholds=(43.0, 55.0))
    assert set(rungs) == {43.0, 55.0}
    assert all(r.threshold_db == level for level, r in rungs.items())


# --- the band floor ----------------------------------------------------------

def test_a_lof_at_the_floor_is_flagged(sounding):
    """The echo starts at bin 40; declaring the floor there makes it a bound."""
    floor = float(sounding.freq[FIRST_BIN])
    assert lof.lof_at(sounding, 43.0, band_floor_mhz=floor).at_band_floor
    # Well below it, the same LOF is a measurement.
    assert not lof.lof_at(sounding, 43.0,
                          band_floor_mhz=floor - 2.0).at_band_floor


def test_the_default_floor_is_the_sweep_start(sounding):
    """Which is the documented wrong answer whenever the transmitter starts higher."""
    pick = lof.lof_at(sounding, 43.0)
    assert not pick.at_band_floor          # bin 40 is well above the sweep start


def test_measure_band_floor_recovers_the_lowest_radiated_frequency(sounding):
    """A hardware edge is constant, so the minimum over soundings finds it."""
    floor = lof.measure_band_floor([sounding, sounding, sounding], 43.0)
    assert floor == pytest.approx(sounding.freq[FIRST_BIN],
                                  abs=2 * sounding.cal.freq_step_mhz)


def test_measure_band_floor_with_nothing_to_measure():
    assert np.isnan(lof.measure_band_floor([], 43.0))


# --- the band ceiling ---------------------------------------------------------

@pytest.fixture
def closed_band(make_lfs):
    """A worse hour on the same circuit: the echo gives out 50 bins lower."""
    iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=ECHO_RANGE,
                  half_span_km=HALF_SPAN, echo_first_bin=FIRST_BIN,
                  echo_last_bin=LAST_BIN - 50)
    path = make_lfs(iq, name="closed.lfs", dur=2)
    return spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))


def test_measure_band_ceiling_recovers_the_highest_returned_frequency(sounding):
    """The echo stops at bin 150 of 200. The sweep does not, and that is the bug."""
    ceiling = lof.measure_band_ceiling([sounding, sounding, sounding], 43.0)

    assert ceiling == pytest.approx(sounding.freq[LAST_BIN],
                                    abs=2 * sounding.cal.freq_step_mhz)
    # Far enough below the declared stop that BAND_EDGE_BINS cannot bridge the
    # gap -- the DOB Cyprus case, where anchoring on the header put the band
    # edge above anything the receiver ever saw. In bins, because this fixture
    # sweeps 0.26 MHz where DOB sweeps 17.
    assert ceiling < (sounding.cal.freq_stop
                      - 20 * sounding.cal.freq_step_mhz)


def test_measure_band_ceiling_with_nothing_to_measure():
    assert np.isnan(lof.measure_band_ceiling([], 43.0))


def test_a_mostly_closed_band_does_not_drag_the_ceiling_down(sounding, closed_band):
    """A high quantile, because the ionosphere only censors the top downwards.

    Three bad hours and one good one describe a circuit that reaches the good
    hour's top; the same statistic the floor uses would report the bad one and
    censor every daytime pick thereafter.
    """
    ions = [closed_band, closed_band, closed_band, sounding]
    ceiling = lof.measure_band_ceiling(ions, 43.0)

    reached = float(sounding.freq[LAST_BIN])
    closed = float(closed_band.freq[LAST_BIN - 50])
    assert abs(ceiling - reached) < abs(ceiling - closed)

    # The floor over the same set is the mirror image and stays at the bottom:
    # both echoes start at FIRST_BIN, so nothing here moves it.
    floor = lof.measure_band_floor(ions, 43.0)
    assert floor == pytest.approx(sounding.freq[FIRST_BIN],
                                  abs=2 * sounding.cal.freq_step_mhz)


# --- LUF ---------------------------------------------------------------------

def test_luf_is_lof_at_the_offset_level(sounding):
    """Same mechanism, different claim -- and the claim is the reason for the name.

    The claim is also why the argument differs: ``lof_at`` takes a raw ionogram
    level, ``luf_at_snr`` a true signal-to-noise, which is 30 dB lower for the
    same test.
    """
    luf = lof.luf_at_snr(sounding, 20.0)
    assert luf.lof_mhz == lof.lof_at(sounding, 50.0).lof_mhz
    assert luf.threshold_db == pytest.approx(50.0)


# --- the record ---------------------------------------------------------------

def _record(ion, tmp_path, band_floor_mhz=None):
    result = extractors.get("contour")(ion)
    low = lof.pick_lof(result.presence, ion.freq, power_db=ion.db,
                       vrange=ion.vrange, band_floor_mhz=band_floor_mhz)
    rungs = lof.ladder(ion, band_floor_mhz=band_floor_mhz)
    root = saoxml.build_document([
        saoxml.build_record(ion, result, lof=low, lof_ladder=rungs)])
    saoxml.write(root, tmp_path / "lof.xml")
    return saoxml.read(tmp_path / "lof.xml")[0]


def test_record_carries_lof_and_the_ladder(sounding, tmp_path):
    record = _record(sounding, tmp_path)

    assert record.characteristic("LOF") is not None
    for level in lof.DEFAULT_LADDER:
        assert record.characteristic(f"LOF@{level:.0f}dB") is not None


def test_lof_is_not_reachable_as_the_muf(sounding, tmp_path):
    record = _record(sounding, tmp_path)
    assert record.muf.name == "MUF"
    assert record.muf.value > record.characteristic("LOF").value


def test_a_floored_lof_earns_the_less_than_letter(sounding, tmp_path):
    """E, the mirror of the D already used for a band-limited MUF."""
    floor = float(sounding.freq[FIRST_BIN])
    record = _record(sounding, tmp_path, band_floor_mhz=floor)

    assert record.characteristic("LOF").letter == saoxml.QL_LESS_THAN


def test_an_unfloored_lof_earns_no_letter(sounding, tmp_path):
    record = _record(sounding, tmp_path, band_floor_mhz=1.0)
    assert record.characteristic("LOF").letter == ""


def test_render_draws_the_window(sounding, tmp_path):
    from muf import render

    record = _record(sounding, tmp_path)
    out = render.plot_sao(record, tmp_path / "window.png", ion=sounding)
    assert out.exists() and out.stat().st_size > 0


# --- behaviour on real recordings ---------------------------------------------

def test_lof_tracks_solar_illumination(real_dir):
    """The claim that makes LOF worth scaling: it follows D-region absorption.

    Not a tolerance on a value -- there is no ground truth -- but on the
    correlation with the cosine of the solar zenith angle at the path midpoint,
    which absorption is driven by. Measured at +0.86 over 2026-02-04.
    """
    from muf import geometry
    from muf.io_lfs import find_lfs
    from muf.reference import chapman

    paths = find_lfs([real_dir])[::8]
    if len(paths) < 12:
        pytest.skip(f"only {len(paths)} soundings available")

    cosines, values = [], []
    for path in paths:
        ion = spectro.compute_cached(path, cache_dir=".muf_cache")
        pick = lof.lof_at(ion, 43.0)
        if not pick.ok:
            continue
        tx, rx, _ = geometry.path_of(ion.header)
        cosines.append(chapman.solar_zenith_cos(ion.header.datetime,
                                                geometry.midpoint(tx, rx)))
        values.append(pick.lof_mhz)

    assert len(values) >= 10
    assert np.corrcoef(cosines, values)[0, 1] > 0.6


def test_luf_takes_a_true_snr_not_a_raw_ionogram_level(sounding):
    """The dB scale is offset by 30: noise reads 30 dB, not 0.

    A caller asking for a 13 dB signal-to-noise means the 43 dB level the
    estimators share. Taking the argument raw would test against -17 dB of
    signal, i.e. against noise, which is what this conversion prevents.
    """
    assert spectro.NOISE_FLOOR_DB == 30.0
    assert (lof.luf_at_snr(sounding, 13.0).threshold_db
            == pytest.approx(43.0))
    assert (lof.luf_at_snr(sounding, 13.0).lof_mhz
            == lof.lof_at(sounding, 43.0).lof_mhz)


def test_a_requirement_below_the_noise_peak_saturates(sounding):
    """Documented rather than rejected: the answer is the bottom of the sweep."""
    saturated = lof.luf_at_snr(sounding, lof.MIN_MEANINGFUL_SNR_DB - 15.0)
    real = lof.luf_at_snr(sounding, 13.0)

    assert saturated.ok and real.ok
    assert saturated.lof_mhz < real.lof_mhz


# --- on the rendered ionogram -------------------------------------------------
#
# The rendered plot drew every estimator's MUF and none of their LOFs, so the
# picture said less than the table beside it. These pin what it draws now, and
# -- the part that matters -- that the number under the line is the same one
# `pipeline` would store.

def test_the_rendered_plot_draws_each_estimators_own_lof(sounding):
    """Per method, not once for the ionogram.

    `algo`'s LOF and `algo`'s MUF are the two ends of one detected set. A
    single band-wide LOF drawn beside three MUFs would invite reading a spread
    between quantities that were never measured the same way.
    """
    from muf import render

    results = extractors.run(sounding)
    drawn = render._lofs_of(sounding, results)

    assert drawn, "no estimator produced a LOF on a sounding with a clear echo"
    assert set(drawn) <= set(results)
    for name, low in drawn.items():
        assert low.ok
        # Below the MUF the same estimator picked, which is the one ordering
        # that has to hold for the pair to mean anything.
        assert low.lof_mhz < results[name].pick.muf_mhz


def test_the_drawn_lof_is_the_one_the_pipeline_would_store(sounding):
    """The picture must not disagree with the table.

    `pipeline.process_file` builds its `lof_<method>` column from
    `pick_lof(result.presence, ...)` with the defaults; so does the renderer.
    If those two ever drift apart the image is quietly wrong rather than
    broken, which is the failure mode `_render`'s gating comment is about.
    """
    from muf import render

    results = extractors.run(sounding)
    drawn = render._lofs_of(sounding, results)

    for name, low in drawn.items():
        expected = lof.pick_lof(results[name].presence, sounding.freq,
                                power_db=sounding.db, vrange=sounding.vrange)
        assert low.lof_mhz == pytest.approx(expected.lof_mhz, abs=1e-12)


def test_an_estimator_that_found_nothing_contributes_no_line(sounding):
    """A failed estimator is a missing line, not a NaN drawn at the origin."""
    from muf import render

    results = extractors.run(sounding)
    broken = next(iter(results))
    results[broken].error = "detector fell over"

    drawn = render._lofs_of(sounding, results)

    assert broken not in drawn


def test_the_lof_lines_can_be_turned_off(sounding, tmp_path):
    from muf import render

    results = extractors.run(sounding)
    with_lof = render.plot(sounding, tmp_path / "with.png", results)
    without = render.plot(sounding, tmp_path / "without.png", results, lof=False)

    assert with_lof.stat().st_size > 0 and without.stat().st_size > 0
    assert with_lof.read_bytes() != without.read_bytes(), (
        "lof=False drew the same image, so the lines were never conditional")
