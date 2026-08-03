"""Segmenting a trace into modes and reconstructing one of them."""

from __future__ import annotations

import numpy as np
import pytest

from muf import trace

GROUND_KM = 2588.0        # Cyprus -> Yoshkar-Ola


def one_mode(f0=10.0, f1=20.0, h0=2700.0, rise=60.0, n=60, scatter=0.0, seed=0):
    """A single rising trace, optionally with scatter."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(f0, f1, n)
    vrange = h0 + rise * ((freq - f0) / (f1 - f0)) ** 2
    if scatter:
        vrange = vrange + rng.normal(0, scatter, n)
    return freq, vrange


# --- segmentation ------------------------------------------------------------

def test_single_mode_stays_whole():
    segments = trace.segment(*one_mode())
    assert len(segments) == 1
    assert segments[0].n_points == 60


def test_splits_at_a_mode_boundary():
    """The real case: range drops sharply where another mode takes over."""
    f1, h1 = one_mode(10.0, 17.0, 2850.0, 30.0, n=40)
    f2, h2 = one_mode(23.0, 29.0, 2680.0, 40.0, n=40)
    segments = trace.segment(np.concatenate([f1, f2]), np.concatenate([h1, h2]))

    assert len(segments) == 2
    assert segments[0].median_range > segments[1].median_range
    assert segments[0].freq.max() < segments[1].freq.min()


def test_noise_alone_does_not_split():
    """A noisy trace must stay whole; the threshold scales with its scatter.

    Before this was scaled, the noisiest soundings (37 km) were split on noise.
    """
    segments = trace.segment(*one_mode(scatter=35.0, n=80, seed=3))
    assert len(segments) == 1


def test_a_single_wild_point_does_not_split():
    freq, vrange = one_mode(n=60)
    vrange[30] += 400.0                       # one absurd outlier
    assert len(trace.segment(freq, vrange)) == 1


def test_short_fragments_are_dropped():
    f1, h1 = one_mode(10.0, 11.0, 2850.0, 5.0, n=3)      # too short to keep
    f2, h2 = one_mode(20.0, 28.0, 2680.0, 40.0, n=40)
    segments = trace.segment(np.concatenate([f1, f2]), np.concatenate([h1, h2]))

    assert len(segments) == 1
    assert segments[0].n_points == 40


def test_empty_input():
    assert trace.segment(np.empty(0), np.empty(0)) == []


def test_segments_come_out_in_frequency_order():
    freq, vrange = one_mode(n=40)
    shuffled = np.argsort(np.random.default_rng(0).random(40))
    segments = trace.segment(freq[shuffled], vrange[shuffled])

    assert len(segments) == 1
    assert np.all(np.diff(segments[0].freq) > 0)


def test_scatter_measures_noise_not_slope():
    """A steep clean trace must read as low scatter, not high."""
    _, steep = one_mode(rise=400.0, n=60)
    _, noisy = one_mode(rise=0.0, n=60, scatter=30.0, seed=1)

    assert trace.trace_scatter_km(steep) < 10
    assert trace.trace_scatter_km(noisy) > 15


# --- hop identification ------------------------------------------------------

def test_hop_range_grows_with_hops_and_height():
    assert trace.hop_range_km(GROUND_KM, 1, 300) < trace.hop_range_km(GROUND_KM, 2, 300)
    assert trace.hop_range_km(GROUND_KM, 1, 250) < trace.hop_range_km(GROUND_KM, 1, 400)


def test_hop_range_reduces_to_the_ground_path():
    """At zero height the path is just the ground distance."""
    assert trace.hop_range_km(GROUND_KM, 1, 0.0) == pytest.approx(GROUND_KM)


def test_hop_range_rejects_zero_hops():
    with pytest.raises(ValueError):
        trace.hop_range_km(GROUND_KM, 0, 300)


def test_labels_a_clear_one_hop_segment():
    """Low enough in range that only one hop count fits.

    2650 km sits right on the 1-hop family (2636-2774) and 124 km below the
    nearest 2-hop possibility, so the margin rule accepts it.
    """
    freq, vrange = one_mode(h0=2650.0, rise=5.0)
    labelled = trace.identify_hops(trace.segment(freq, vrange), GROUND_KM)

    assert labelled[0].hops == 1
    assert labelled[0].height_km is not None


def test_leaves_an_ambiguous_segment_unlabelled():
    """Between the 1-hop and 2-hop families, "don't know" is the right answer.

    Over 2588 km the families touch near 2774 km, so a segment sitting there
    must not be assigned on a few kilometres of difference.
    """
    freq, vrange = one_mode(h0=2770.0, rise=5.0)
    labelled = trace.identify_hops(trace.segment(freq, vrange), GROUND_KM)

    assert labelled[0].hops is None


def test_leaves_an_implausible_segment_unlabelled():
    freq, vrange = one_mode(h0=12000.0, rise=10.0)
    labelled = trace.identify_hops(trace.segment(freq, vrange), GROUND_KM)

    assert labelled[0].hops is None


def test_primary_segment_is_the_highest_in_frequency():
    f1, h1 = one_mode(10.0, 17.0, 2850.0, 30.0, n=40)
    f2, h2 = one_mode(23.0, 29.0, 2680.0, 40.0, n=40)
    segments = trace.segment(np.concatenate([f1, f2]), np.concatenate([h1, h2]))

    assert trace.primary_segment(segments).freq.max() == pytest.approx(29.0)
    assert trace.primary_segment([]) is None


# --- reconstruction ----------------------------------------------------------

def test_reconstruction_fills_gaps():
    """Output is denser than the sparse input it came from."""
    freq, vrange = one_mode(n=40)
    keep = np.sort(np.random.default_rng(0).choice(40, 20, replace=False))
    segments = trace.segment(freq[keep], vrange[keep])
    result = trace.reconstruct(segments[0])

    assert result.ok
    assert len(result.freq) > segments[0].n_points
    assert np.all(np.diff(result.freq) > 0)


def test_reconstruction_recovers_the_underlying_curve():
    freq, vrange = one_mode(n=80, scatter=15.0, seed=5)
    _, truth = one_mode(n=80)
    result = trace.reconstruct(trace.segment(freq, vrange)[0])

    assert result.ok
    fitted = np.interp(freq, result.freq, result.vrange)
    assert np.sqrt(np.mean((fitted - truth) ** 2)) < 15.0     # beats the noise


def test_reconstruction_stays_inside_its_segment():
    """It must never extrapolate past the data it was given."""
    freq, vrange = one_mode(12.0, 22.0)
    result = trace.reconstruct(trace.segment(freq, vrange)[0])

    assert result.freq.min() >= 12.0 - 1e-6
    assert result.freq.max() <= 22.0 + 1e-6


def test_reconstruction_declines_on_too_few_points():
    freq, vrange = one_mode(n=4)
    result = trace.reconstruct(trace.Segment(freq, vrange))

    assert not result.ok
    assert "points" in result.reason
    assert np.isnan(result.muf_mhz)


def test_weights_are_honoured():
    """Points marked confident should pull the curve more than doubtful ones."""
    freq, vrange = one_mode(n=60)
    vrange[30] += 100.0

    trusted = trace.reconstruct(trace.Segment(freq, vrange,
                                              weight=np.full(60, 1.0)))
    doubted = np.full(60, 1.0)
    doubted[30] = 0.001
    ignored = trace.reconstruct(trace.Segment(freq, vrange, weight=doubted))

    at_trusted = np.interp(freq[30], trusted.freq, trusted.vrange)
    at_ignored = np.interp(freq[30], ignored.freq, ignored.vrange)
    assert at_trusted > at_ignored


# --- following modes that overlap in frequency -------------------------------

def test_group_tracks_separates_modes_at_the_same_frequency():
    """Two echoes coexisting across the same band must come out as two tracks.

    This is the case a frequency-ordered split cannot handle: the points
    alternate between the modes, so walking in frequency sees a trace jumping
    up and down rather than two steady ones.
    """
    freq = np.repeat(np.linspace(10.0, 25.0, 40), 2)
    vrange = np.empty(80)
    vrange[0::2] = 2680.0                    # one mode
    vrange[1::2] = 2850.0                    # another, 170 km above
    tracks = trace.group_tracks(freq, vrange)

    assert len(tracks) == 2
    lows, highs = sorted(t.median_range for t in tracks)
    assert lows == pytest.approx(2680.0, abs=5)
    assert highs == pytest.approx(2850.0, abs=5)
    # Each track spans the whole band, rather than being chopped up.
    for t in tracks:
        assert t.freq.min() == pytest.approx(10.0)
        assert t.freq.max() == pytest.approx(25.0)


def test_group_tracks_bridges_a_long_fade():
    """A real trace fading for several MHz is one track, not two.

    Observed here: a continuous echo with a 7 MHz hole in the middle.
    """
    left = np.linspace(13.0, 16.0, 20)
    right = np.linspace(23.0, 29.0, 30)
    freq = np.concatenate([left, right])
    vrange = np.full(50, 2690.0)

    assert len(trace.group_tracks(freq, vrange, max_gap_mhz=8.0)) == 1
    assert len(trace.group_tracks(freq, vrange, max_gap_mhz=2.0)) == 2


def test_group_tracks_will_not_bridge_a_range_jump():
    """The range window is what keeps different modes apart."""
    freq = np.linspace(10.0, 25.0, 40)
    vrange = np.where(freq < 17.0, 2680.0, 2950.0)      # 270 km apart
    tracks = trace.group_tracks(freq, vrange)

    assert len(tracks) == 2


def test_group_tracks_needs_no_weights():
    freq = np.linspace(10.0, 20.0, 30)
    tracks = trace.group_tracks(freq, np.full(30, 2700.0))
    assert len(tracks) == 1


def test_group_tracks_on_nothing():
    assert trace.group_tracks(np.empty(0), np.empty(0)) == []


def test_extract_points_finds_several_modes_per_bin(real_file):
    """More points than frequency bins means overlapping modes were separated."""
    from muf import extractors, spectro

    ion = spectro.compute(real_file)
    result = extractors.get("algo")(ion)
    freq, vrange, weight = trace.extract_points(ion, result)

    assert len(freq) == len(vrange) == len(weight)
    assert len(freq) >= int(result.presence.sum())
    assert np.all(np.isfinite(vrange))


def test_extract_points_without_a_mask(real_file):
    """Falls back to one point per bin when the estimator supplies no mask."""
    from muf import extractors, spectro

    ion = spectro.compute(real_file)
    result = extractors.get("algo")(ion)
    result.mask = None

    freq, vrange, weight = trace.extract_points(ion, result)
    assert len(freq) == int(result.presence.sum())


# --- low-ray / high-ray branches ---------------------------------------------

def _nose(muf=20.0, vertex=2760.0, n=60, low_span=8.0, high_span=2.0):
    """Both branches of one nose: a gentle rising ray and a steep falling one.

    Below the MUF two rays arrive at every frequency; they converge at the nose.
    """
    low_f = np.linspace(muf - low_span, muf, n)
    low_h = vertex - 60.0 * ((muf - low_f) / low_span)          # rises gently
    high_f = np.linspace(muf - high_span, muf, n // 2)
    high_h = vertex + 90.0 * ((muf - high_f) / high_span)       # falls steeply
    return (np.concatenate([low_f, high_f]),
            np.concatenate([low_h, high_h]))


def test_branches_are_paired_into_one_nose():
    freq, vrange = _nose()
    tracks = trace.merge_branches(trace.group_tracks(freq, vrange))

    assert len(tracks) == 2
    assert {t.branch for t in tracks} == {"low", "high"}
    assert len({t.group for t in tracks}) == 1        # one nose


def test_primary_is_the_low_ray_not_the_high_ray():
    """The high ray reaches marginally higher in frequency but is a fragment.

    Picking on frequency alone chose a 33-point steep fragment over the
    104-point main trace on real soundings.
    """
    freq, vrange = _nose()
    tracks = trace.merge_branches(trace.group_tracks(freq, vrange))
    primary = trace.primary_segment(tracks)

    assert primary.branch == "low"
    assert primary.n_points == max(t.n_points for t in tracks)
    assert trace.slope_km_per_mhz(primary) > 0


def test_unrelated_modes_are_not_paired():
    """A steep track far from the nose must stay separate.

    Well clear in range as well as frequency, so track-following keeps it apart
    in the first place and the pairing step is what is under test.
    """
    freq, vrange = _nose(muf=26.0)
    other_f = np.linspace(16.6, 17.2, 12)
    other_h = np.linspace(3250.0, 3200.0, 12)       # falling, ~450 km above
    tracks = trace.merge_branches(trace.group_tracks(
        np.concatenate([freq, other_f]), np.concatenate([vrange, other_h])))

    assert len({t.group for t in tracks}) >= 2
    odd = [t for t in tracks if t.freq.max() < 18]
    assert odd and odd[0].branch is None


def test_nose_points_returns_both_branches():
    freq, vrange = _nose()
    tracks = trace.merge_branches(trace.group_tracks(freq, vrange))
    nf, nv = trace.nose_points(tracks)

    assert len(nf) == sum(t.n_points for t in tracks)
    assert np.all(np.diff(nf) >= 0)

    # Both sides of the vertex are represented. Checked a little below the top,
    # since the two branches converge at the nose by definition.
    top = nf.max()
    window = (nf > top - 1.5) & (nf < top - 0.3)
    assert window.sum() > 4
    assert nv[window].max() - nv[window].min() > 30


def test_nose_points_lets_the_fit_bracket_the_vertex():
    """The point of merging: the vertex sits inside the data, not beyond it."""
    from muf import fit

    freq, vrange = _nose(muf=20.0)
    tracks = trace.merge_branches(trace.group_tracks(freq, vrange))
    result = fit.fit_nose(*trace.nose_points(tracks))

    assert result.ok
    assert result.muf_mhz == pytest.approx(20.0, abs=0.4)
    assert abs(result.extrapolation_mhz) < 1.0


def test_slope_sign_distinguishes_the_branches():
    freq, vrange = _nose()
    tracks = trace.group_tracks(freq, vrange)
    slopes = sorted(trace.slope_km_per_mhz(t) for t in tracks)

    assert slopes[0] < 0 < slopes[-1]


# --- reconstruction: support and shape ----------------------------------------

def test_reconstruction_leaves_wide_gaps_empty():
    """A spline will arc hundreds of km across a hole; do not draw that."""
    left = np.linspace(9.0, 10.0, 20)
    right = np.linspace(13.0, 15.0, 40)
    freq = np.concatenate([left, right])
    vrange = np.concatenate([np.full(20, 2900.0), np.linspace(2760, 2800, 40)])

    result = trace.reconstruct(trace.Segment(freq, vrange),
                               max_support_gap_mhz=1.0)
    assert result.ok
    # Nothing emitted in the middle of the 3 MHz hole.
    assert not ((result.freq > 11.2) & (result.freq < 12.0)).any()


def test_support_gap_can_be_relaxed():
    left = np.linspace(9.0, 10.0, 20)
    right = np.linspace(13.0, 15.0, 40)
    freq = np.concatenate([left, right])
    vrange = np.concatenate([np.full(20, 2900.0), np.linspace(2760, 2800, 40)])

    wide = trace.reconstruct(trace.Segment(freq, vrange),
                             max_support_gap_mhz=5.0)
    assert ((wide.freq > 11.2) & (wide.freq < 12.0)).any()


@pytest.mark.parametrize("method", ["spline", "pchip", "makima"])
def test_reconstruction_methods_agree_on_clean_data(method):
    freq, vrange = one_mode(n=80)
    result = trace.reconstruct(trace.Segment(freq, vrange), method=method)

    assert result.ok
    fitted = np.interp(freq, result.freq, result.vrange)
    assert np.max(np.abs(fitted - vrange)) < 25.0


def test_shape_preserving_methods_do_not_overshoot():
    """pchip cannot exceed the data's range; a smoothing spline can."""
    freq, vrange = one_mode(n=60, scatter=20.0, seed=11)
    result = trace.reconstruct(trace.Segment(freq, vrange), method="pchip")

    assert result.ok
    assert result.vrange.min() >= vrange.min() - 1e-6
    assert result.vrange.max() <= vrange.max() + 1e-6


def test_unknown_reconstruction_method():
    freq, vrange = one_mode(n=40)
    with pytest.raises(ValueError, match="unknown reconstruction method"):
        trace.reconstruct(trace.Segment(freq, vrange), method="telepathy")


# --- the whole pipeline ------------------------------------------------------

def test_analyse_returns_segments_and_the_primary_reconstruction():
    f1, h1 = one_mode(10.0, 17.0, 2850.0, 30.0, n=40)
    f2, h2 = one_mode(23.0, 29.0, 2680.0, 40.0, n=40)
    segments, result = trace.analyse(np.concatenate([f1, f2]),
                                     np.concatenate([h1, h2]), GROUND_KM)

    assert len(segments) == 2
    assert result.ok
    assert result.freq.min() >= 23.0 - 1e-6      # the high-frequency segment


def test_analyse_on_nothing():
    segments, result = trace.analyse(np.empty(0), np.empty(0), GROUND_KM)
    assert segments == []
    assert result is None


def test_real_sounding_is_segmented(real_file):
    from muf import extractors, fit, geometry, spectro

    ion = spectro.compute(real_file)
    result = extractors.get("algo")(ion)
    _, _, ground = geometry.path_of(ion.header)

    freq, vrange = fit.trace_points(ion, result)
    segments, reconstruction = trace.analyse(
        freq, vrange, ground, weight=trace.trace_weights(ion, result))

    assert segments
    assert reconstruction is not None and reconstruction.ok
    # The reconstruction reaches the MUF the extractor picked.
    assert reconstruction.freq.max() == pytest.approx(result.pick.muf_mhz, abs=0.1)
    assert reconstruction.rms_residual_km < 60
