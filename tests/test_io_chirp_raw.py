"""Re-deriving a v2 product at a different FFT window, from its stored ``z``.

``architecture.md`` sec. 3.4 records that ``.h5`` soundings cannot be
reprocessed at another window because the waveform is gone. That is true of the
default configuration and false when ``save_raw_voltage = true``, which is the
whole reason to turn it on.

The load-bearing test is :func:`test_reprocessing_at_the_original_window_round_trips`.
:func:`muf.io_chirp.v2_spectrogram` is a hand port of upstream's function --
13x oversampled, Hann-windowed, on the conjugate -- and nothing else in the
suite would notice if the port drifted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from muf import io_chirp, loader, spectro

WINDOW = 256
STEP = 64


def test_a_product_without_z_says_so(make_chirp_h5):
    path = make_chirp_h5(np.full((4, 64), 100.0))

    assert not loader.read_header(path).has_raw_voltage
    with pytest.raises(ValueError, match="no 'z' dataset"):
        io_chirp.read_raw_voltage(path)


def test_a_product_with_z_advertises_it(make_chirp_z_h5):
    path = make_chirp_z_h5(window=WINDOW, step=STEP)

    header = loader.read_header(path)
    assert header.has_raw_voltage
    z = io_chirp.read_raw_voltage(path)
    assert z.dtype == np.complex64 and z.size > WINDOW


def test_reprocessing_at_the_original_window_round_trips(make_chirp_z_h5):
    """The port of `calc_ionograms.spectrogram`, checked against itself.

    The fixture derives the stored `SNR` *from* the waveform using
    `v2_spectrogram`, so re-deriving at the same window must return the same
    array. If the 13x oversampling, the Hann window, the conjugate or the
    fftshift ever drift out of step with upstream, this is what fails.

    Compared only where v2 kept the cell: it sparsifies below
    `storage_snr_threshold` and those cells read back as the row median, which
    a reprocess has no reason to reproduce. See the next test.
    """
    path = make_chirp_z_h5(window=WINDOW, step=STEP, echo_bin=WINDOW // 2 + 20)

    stored = io_chirp.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        redone = io_chirp.reprocess(path, window=WINDOW)

    assert redone.power.shape == stored.power.shape
    np.testing.assert_allclose(redone.freq, stored.freq, rtol=1e-9)
    np.testing.assert_allclose(redone.vrange, stored.vrange, rtol=1e-9)

    # Above the sparsification floor and below the float16 ceiling: v2 stores
    # SNR as float16, whose maximum 65504 lands at exactly 73.73 dB, so a
    # stronger echo is clipped on the way to disk.
    kept = (stored.db > 26.0) & (stored.db < 73.0)
    assert kept.sum() > 20, "fixture produced nothing in the comparable band"
    np.testing.assert_allclose(redone.db[kept], stored.db[kept], atol=0.05)


def test_reprocessing_recovers_what_v2_threw_away(make_chirp_z_h5):
    """A side benefit worth knowing about, not just a difference to tolerate.

    v2 stores NaN for every cell below `storage_snr_threshold`, and `io_chirp`
    fills those with the row median -- so a stored product has a hard floor at
    25.571 dB and no noise texture below it. Re-deriving from `z` gets the real
    distribution back, which matters for anything estimating a noise level
    rather than crossing a threshold.
    """
    path = make_chirp_z_h5(window=WINDOW, step=STEP)
    stored = io_chirp.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        redone = io_chirp.reprocess(path, window=WINDOW)

    assert float(stored.db.min()) == pytest.approx(25.571, abs=1e-3)
    assert float(redone.db.min()) < 20.0
    # and the median cell still lands on the shared convention
    assert float(np.median(redone.db)) == pytest.approx(25.571, abs=1.0)


def test_reprocessing_recovers_what_float16_clipped(make_chirp_z_h5):
    """The other thing storage costs: SNR is float16, which saturates at 73.73 dB.

    A strong echo is clipped on the way to disk and reads back at the ceiling.
    Re-deriving from `z` gets the real amplitude, which is why 73.734 in a
    stored product should be treated as "at least this", not as a measurement.
    """
    path = make_chirp_z_h5(window=WINDOW, step=STEP,
                           echo_bin=WINDOW // 2 + 20, echo_amplitude=300.0)
    stored = io_chirp.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        redone = io_chirp.reprocess(path, window=WINDOW)

    assert float(stored.db.max()) == pytest.approx(73.734, abs=1e-2)
    assert float(redone.db.max()) > float(stored.db.max())


def test_a_finer_window_halves_the_range_bin(make_chirp_z_h5):
    """The trade `--window` makes on the .lfs path, now available here."""
    path = make_chirp_z_h5(window=WINDOW, step=STEP)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coarse = io_chirp.reprocess(path, window=WINDOW)
        fine = io_chirp.reprocess(path, window=WINDOW * 2)

    assert fine.cal.range_step == pytest.approx(coarse.cal.range_step / 2, rel=1e-6)
    assert fine.vrange.size == coarse.vrange.size * 2
    # and it costs frequency rows, because fewer whole windows fit
    assert fine.freq.size < coarse.freq.size


def test_the_frequency_axis_is_recomputed_not_reused(make_chirp_z_h5):
    """`freqs = rate * arange(n_spec) * step / sr`, and n_spec depends on window.

    Reusing the stored vector would leave every row mislabelled by a growing
    amount -- an ionogram that plots and reads MUF off the wrong frequency.
    """
    path = make_chirp_z_h5(window=WINDOW, step=STEP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fine = io_chirp.reprocess(path, window=WINDOW * 2)

    stored_freqs = io_chirp.load(path).freq
    assert fine.freq.size != stored_freqs.size
    # the step is a property of the sweep, so it must be unchanged
    assert np.diff(fine.freq)[0] == pytest.approx(np.diff(stored_freqs)[0], rel=1e-9)
    assert fine.freq[0] == pytest.approx(stored_freqs[0], abs=1e-9)


def test_an_echo_stays_at_the_same_range_across_windows(make_chirp_z_h5):
    """A constant beat is a constant range; the bin index changes, the km must not."""
    path = make_chirp_z_h5(window=WINDOW, step=STEP,
                           echo_bin=WINDOW // 2 + 20, echo_amplitude=40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coarse = io_chirp.reprocess(path, window=WINDOW)
        fine = io_chirp.reprocess(path, window=WINDOW * 2)

    def peak_km(ion):
        return float(ion.vrange[int(np.argmax(ion.power.sum(axis=0)))])

    assert peak_km(fine) == pytest.approx(peak_km(coarse),
                                          abs=2 * coarse.cal.range_step)


def test_reprocessing_keeps_the_shared_db_convention(make_chirp_z_h5):
    """43 dB has to keep meaning the same thing after a reprocess.

    The *median* cell is the anchor, not the minimum: a reprocessed product
    has no sparsification floor, so its minimum is real noise scatter rather
    than the row median. `NOISE_COEF` is what both paths divide by, and a
    median-noise cell must land where it lands on the `.lfs` path.
    """
    path = make_chirp_z_h5(window=WINDOW, step=STEP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        redone = io_chirp.reprocess(path, window=WINDOW)

    floor = spectro.to_db(np.array([1.0 / spectro.NOISE_COEF], dtype=np.float32))[0]
    assert float(np.median(redone.db)) == pytest.approx(floor, abs=1.0)


def test_a_window_too_large_for_the_waveform_explains_itself(make_chirp_z_h5):
    path = make_chirp_z_h5(window=WINDOW, step=STEP)
    z = io_chirp.read_raw_voltage(path)
    with pytest.raises(ValueError, match="leaves no complete row"):
        io_chirp.reprocess(path, window=z.size * 2)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def test_loader_honours_window_when_z_is_present(make_chirp_z_h5):
    """The warning must not fire on a product that really can be re-derived."""
    path = make_chirp_z_h5(window=WINDOW, step=STEP)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=".*range axis left relative.*")
        fine = loader.load(path, window=WINDOW * 2)

    assert fine.window == WINDOW * 2
    assert fine.cal.range_step < loader.load(path).cal.range_step


def test_loader_still_warns_when_z_is_absent(make_chirp_h5):
    path = make_chirp_h5(np.full((8, 64), 100.0))
    with pytest.warns(UserWarning, match="no `z` dataset"):
        loader.load(path, window=4096)


def test_the_warning_says_how_to_fix_it(make_chirp_h5):
    """A warning about a config flag should name the config flag."""
    path = make_chirp_h5(np.full((8, 64), 100.0))
    with pytest.warns(UserWarning, match="save_raw_voltage = true"):
        loader.load(path, window=4096)


def test_zero_periods_is_still_meaningless_with_z(make_chirp_z_h5):
    """v2 does not zero-pad, so there is nothing to divide out either way."""
    path = make_chirp_z_h5(window=WINDOW, step=STEP)
    with pytest.warns(UserWarning, match="does not zero-pad"):
        loader.load(path, window=WINDOW * 2, zero_periods=7)
