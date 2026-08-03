"""Exporting soundings as SAO.XML 5.0.

Most of these assert what the record must *not* contain. The format's whole
risk for this instrument is claiming vertical-sounding quantities it cannot
measure, so the tests police the boundary between what was measured, what was
modelled, and what is absent.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import pytest

from muf import extractors, fit, geometry, spectro, trace
from muf.export import saoxml
from muf.pipeline import Options

from conftest import synth_iq


WINDOW = 512
N_FREQ = 200
HALF_SPAN = 60_000.0


#: 200 windows of 512 at 40 kHz is 2.56 s, so a declared ``dur`` of 2 s makes
#: the recording cover its whole nominal sweep. Without this the synthetic file
#: is a truncated sweep and every pick correctly earns the "D" letter, which
#: would mask the other cases.
COMPLETE_SWEEP = dict(dur=2)


def _sounding(make_lfs, echo_last_bin=120, **header):
    iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
                  half_span_km=HALF_SPAN, echo_last_bin=echo_last_bin)
    path = make_lfs(iq, name=f"synth{echo_last_bin}.lfs", **header)
    return spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))


@pytest.fixture
def sounding(make_lfs):
    """A clear echo, ending well short of the band edge, on a complete sweep."""
    return _sounding(make_lfs, **COMPLETE_SWEEP)


def record_of(ion, method: str = "algo"):
    result = extractors.get(method)(ion)
    segments = nose = None
    if result.ok:
        freq, vrange, weight = trace.extract_points(ion, result)
        _, _, path_km = geometry.path_of(ion.header)
        segments = trace.merge_branches(trace.identify_hops(
            trace.group_tracks(freq, vrange, weight), path_km))
        nose = fit.fit_nose(*trace.nose_points(segments))
    return saoxml.build_record(ion, result, segments, nose)


# --- structure ---------------------------------------------------------------

def test_record_has_the_required_attributes(sounding):
    """Table 4 lists these as required; all must be present, even when empty."""
    record = record_of(sounding)

    for name in ("FormatVersion", "StartTimeUTC", "URSICode", "StationName",
                 "GeoLatitude", "GeoLongitude", "Source", "SourceType",
                 "ScalerType"):
        assert name in record.attrib, f"missing required attribute {name}"
    assert record.get("FormatVersion") == "5.0"


def test_ursi_code_is_present_but_empty_by_default(sounding):
    """No registry entry exists for this path. Say so, do not omit the field."""
    assert record_of(sounding).get("URSICode") == ""


def test_ursi_code_is_used_when_given(sounding):
    result = extractors.get("algo")(sounding)
    record = saoxml.build_record(sounding, result, ursi_code="XX000")

    assert record.get("URSICode") == "XX000"


def test_element_order_follows_the_spec(sounding):
    """Table 4 orders SystemInfo, CharacteristicList, TraceList, ProfileList."""
    record = record_of(sounding)
    order = [child.tag for child in record]

    assert order[0] == "SystemInfo"
    assert order[1] == "CharacteristicList"
    assert "ProfileList" not in order          # no inversion exists yet
    if "TraceList" in order:
        assert order.index("TraceList") > order.index("CharacteristicList")


def test_characteristic_list_is_always_present(sounding):
    """It is the one required sub-element (Table 4)."""
    assert record_of(sounding).find("CharacteristicList") is not None


def test_document_is_well_formed_xml(sounding):
    text = saoxml.to_string(saoxml.build_document([record_of(sounding)]))

    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    root = ET.fromstring(text)
    assert root.tag == "SAORecordList"
    assert len(root) == 1


# --- what must not be claimed ------------------------------------------------

def test_muf_is_custom_not_a_ursi_characteristic(sounding):
    """UAG-23A 1.50: Standard MUF(3000) is not the operational MUF of a circuit.

    Emitting ours under URSI ID 03 or 07 would tell every consumer it was a
    transmission-curve conversion from a vertical critical frequency.
    """
    chars = record_of(sounding).find("CharacteristicList")

    assert chars.find("URSI") is None
    ids = {c.get("ID") for c in chars.findall("URSI")}
    assert not ({"03", "07"} & ids)

    muf = [c for c in chars.findall("Custom") if c.get("Name") == "MUF"]
    assert len(muf) == 1
    assert muf[0].get("Units") == "MHz"
    assert float(muf[0].get("Val")) > 0


def test_no_vertical_characteristics_are_invented(sounding):
    """foF2 and friends are not measurable on a fixed oblique path."""
    chars = record_of(sounding).find("CharacteristicList")
    measured = {c.get("Name") for c in chars.findall("Custom")}

    for vertical in ("foF2", "foF1", "foE", "fxI", "h'F", "h'E", "hmF2", "yF2"):
        assert vertical not in measured


def test_equivalent_fof2_is_modelled_and_states_its_assumption(sounding):
    """The secant-law back-conversion is a model result, so it goes in <Modeled>."""
    chars = record_of(sounding).find("CharacteristicList")
    modelled = chars.findall("Modeled")

    assert [m.get("Name") for m in modelled] == ["foF2"]
    options = modelled[0].get("ModelOptions")
    assert "hmF2" in options and "D=" in options
    assert modelled[0].get("ModelName") == "secant-law"


def test_traces_carry_no_polarization(sounding):
    """This receiver has no polarimetry; O/X was tested twice and rejected."""
    traces = record_of(sounding).find("TraceList")
    if traces is None:
        pytest.skip("no segments on this sounding")

    for element in traces.findall("Trace"):
        assert "Polarization" not in element.attrib
        assert element.get("Type") == "non-standard"


# --- traces ------------------------------------------------------------------

def test_trace_lists_are_positionally_consistent(sounding):
    """FrequencyList, RangeList and amplitudes correspond one-to-one."""
    traces = record_of(sounding).find("TraceList")
    if traces is None:
        pytest.skip("no segments on this sounding")

    for element in traces.findall("Trace"):
        freq = element.find("FrequencyList").text.split()
        vrange = element.find("RangeList").text.split()
        assert len(freq) == len(vrange) == int(element.get("Num"))

        amplitude = element.find("TraceValueList")
        if amplitude is not None:
            assert len(amplitude.text.split()) == len(freq)


def test_range_list_is_labelled_group_range_not_virtual_height(sounding):
    """On an oblique path these are path lengths; calling them heights misleads."""
    traces = record_of(sounding).find("TraceList")
    if traces is None:
        pytest.skip("no segments on this sounding")

    for element in traces.findall("Trace"):
        description = element.find("RangeList").get("Description", "")
        assert "Group Range" in description
        assert "height" not in description.lower()


def test_hop_order_is_omitted_when_unknown():
    """Multiple must not appear when identify_hops declined to label."""
    segment = trace.Segment(
        freq=np.linspace(10.0, 12.0, 20),
        vrange=np.full(20, 2700.0),
        hops=None,
    )
    element = saoxml._trace_list([segment]).find("Trace")

    assert "Multiple" not in element.attrib


def test_hop_order_is_reported_when_known():
    segment = trace.Segment(
        freq=np.linspace(10.0, 12.0, 20),
        vrange=np.full(20, 2700.0),
        hops=2,
        branch="low",
    )
    element = saoxml._trace_list([segment]).find("Trace")

    assert element.get("Multiple") == "2"
    assert element.get("Branch") == "low"


# --- qualifying letters ------------------------------------------------------

def test_band_limited_pick_earns_the_greater_than_letter(sounding):
    """UAG-23A 3.1: D is used when only a limiting value is observed."""
    result = extractors.get("algo")(sounding)
    result.pick.__dict__["muf_mhz"] = sounding.cal.freq_stop

    assert saoxml.qualifying_letter(sounding, result) == saoxml.QL_GREATER_THAN


def test_incomplete_sweep_earns_the_greater_than_letter(make_lfs):
    """A truncated recording caps the MUF just as the band edge does.

    The header still declares the full 250 s sweep while the file holds 2.56 s
    of it, which is exactly how real truncated recordings present.
    """
    truncated = _sounding(make_lfs)
    result = extractors.get("algo")(truncated)

    assert not truncated.cal.sweep_complete
    assert saoxml.qualifying_letter(truncated, result) == saoxml.QL_GREATER_THAN


def test_weak_pick_earns_the_uncertain_letter(sounding):
    """UAG-23A 3.1: U covers a trace obscured by noise or interference."""
    result = extractors.get("algo")(sounding)
    result.pick.__dict__["snr_db"] = saoxml.UNCERTAIN_SNR_DB - 1.0

    assert saoxml.qualifying_letter(sounding, result) == saoxml.QL_UNCERTAIN


def test_ragged_trace_earns_the_uncertain_letter(sounding):
    """Scatter beyond a couple of range bins is wander, not quantisation."""
    result = extractors.get("algo")(sounding)
    step = sounding.cal.range_step
    rng = np.random.default_rng(0)
    ragged = trace.Segment(
        freq=np.linspace(10.0, 12.0, 40),
        vrange=2700.0 + rng.normal(0.0, 4.0 * step, 40),
    )

    assert saoxml.qualifying_letter(sounding, result, ragged) == saoxml.QL_UNCERTAIN


def test_scatter_threshold_scales_with_the_range_bin(sounding):
    """Expressed in bins, so --window cannot silently change its meaning."""
    result = extractors.get("algo")(sounding)
    step = sounding.cal.range_step
    rng = np.random.default_rng(0)
    # The same shape of noise at a fraction of a bin is quantisation, not wander,
    # and must not be flagged -- which is what makes the threshold scale-relative.
    steady = trace.Segment(
        freq=np.linspace(10.0, 12.0, 40),
        vrange=2700.0 + rng.normal(0.0, 0.3 * step, 40),
    )

    assert saoxml.qualifying_letter(sounding, result, steady) == ""


def test_clean_pick_earns_no_letter(sounding):
    """No letter means no difficulty -- do not decorate a good measurement."""
    result = extractors.get("algo")(sounding)
    if not result.ok:
        pytest.skip("no pick on this sounding")

    assert saoxml.qualifying_letter(sounding, result) == ""


def test_letter_rides_on_the_custom_muf(sounding):
    """<Custom> has no QL attribute in Table 9, so it goes on as a custom one."""
    result = extractors.get("algo")(sounding)
    result.pick.__dict__["muf_mhz"] = sounding.cal.freq_stop
    record = saoxml.build_record(sounding, result)

    muf = [c for c in record.find("CharacteristicList").findall("Custom")
           if c.get("Name") == "MUF"]
    assert muf[0].get("QL") == saoxml.QL_GREATER_THAN


# --- declining gracefully ----------------------------------------------------

def test_record_without_a_pick_still_validates(sounding):
    """A failed sounding is still a record: metadata without characteristics."""
    from muf.extractors import MufResult
    from muf.pick import NO_PICK

    empty = MufResult(method="algo", pick=NO_PICK,
                      presence=np.zeros(sounding.cal.n_freq, dtype=bool))
    record = saoxml.build_record(sounding, empty)

    assert record.get("URSICode") == ""
    assert record.find("CharacteristicList") is not None
    assert len(record.find("CharacteristicList")) == 0
    assert record.find("TraceList") is None


# --- against real soundings --------------------------------------------------

def test_real_sounding_exports(real_file):
    root = saoxml.export_file(real_file, Options(methods=("algo", "contour")))

    assert root.tag == "SAORecordList"
    # Section 1.3.4: one record per interpretation, not one merged record.
    assert len(root) == 2
    scalers = [r.find("SystemInfo").find("AutoScaler").text for r in root]
    assert any("algo" in s for s in scalers)
    assert any("contour" in s for s in scalers)

    text = saoxml.to_string(root)
    assert ET.fromstring(text) is not None


def test_real_sounding_reports_the_known_muf(real_file):
    """The 03:00 UTC sounding is the package's pinned reference: 12.2 MHz."""
    root = saoxml.export_file(real_file, Options(methods=("algo",)))
    chars = root[0].find("CharacteristicList")
    muf = [c for c in chars.findall("Custom") if c.get("Name") == "MUF"][0]

    assert float(muf.get("Val")) == pytest.approx(12.2, abs=0.3)


def test_real_path_geometry_is_recorded(real_file):
    root = saoxml.export_file(real_file, Options(methods=("algo",)))
    path = root[0].find("SystemInfo").find("ObliquePath")

    assert path.get("TransmitterName") == "cyprus1"
    assert path.get("ReceiverName") == "yoshkar-ola"
    assert float(path.get("GreatCircleDistance")) == pytest.approx(2588, abs=50)


# --- reading back ------------------------------------------------------------
#
# An export nobody can read is a private file with angle brackets in it. These
# assert the round trip through `read`, since that is the only evidence the
# record is self-describing rather than merely well formed.

def _round_trip(ion, tmp_path, method="algo"):
    root = saoxml.build_document([record_of(ion, method)])
    saoxml.write(root, tmp_path / "record.xml")
    return saoxml.read(tmp_path / "record.xml")


def test_read_returns_one_record_per_estimator(sounding, tmp_path):
    root = saoxml.build_document(
        saoxml.records_for(sounding, Options(methods=("algo", "contour")))
    )
    saoxml.write(root, tmp_path / "both.xml")

    records = saoxml.read(tmp_path / "both.xml")
    assert [r.method for r in records] == ["algo", "contour"]


def test_read_recovers_identity_and_path(sounding, tmp_path):
    record = _round_trip(sounding, tmp_path)[0]

    assert record.time == sounding.header.datetime
    assert record.path_type == "oblique"
    assert record.path["TransmitterName"] == sounding.header.tx_name
    _, _, path_km = geometry.path_of(sounding.header)
    assert record.number("GreatCircleDistance", "path") == pytest.approx(
        path_km, abs=0.1)


def test_read_recovers_the_muf(sounding, tmp_path):
    record = _round_trip(sounding, tmp_path)[0]
    written = extractors.get("algo")(sounding)

    assert record.muf.value == pytest.approx(written.pick.muf_mhz, abs=1e-3)
    assert record.muf.units == "MHz"
    # The file's own digits, so a reader can echo them without adding any.
    assert record.muf.text == f"{written.pick.muf_mhz:.3f}"


def test_read_keeps_modelled_apart_from_measured(sounding, tmp_path):
    record = _round_trip(sounding, tmp_path)[0]

    assert record.muf.modelled is False
    fof2 = record.characteristic("foF2")
    assert fof2 is not None and fof2.modelled is True


def test_read_recovers_trace_points(sounding, tmp_path):
    record = _round_trip(sounding, tmp_path)[0]
    assert record.traces

    for item in record.traces:
        assert item.freq.size == item.vrange.size == item.n_points
        if item.amplitude is not None:
            assert item.amplitude.size == item.n_points
        assert np.all(np.diff(item.freq) >= 0)


def test_read_of_a_truncated_pick_carries_its_letter(make_lfs, tmp_path):
    """The "D" of a band-limited pick has to survive the round trip.

    A lower bound that reads back as a measurement is worse than no export.
    """
    ion = _sounding(make_lfs, echo_last_bin=N_FREQ - 1, **COMPLETE_SWEEP)
    record = _round_trip(ion, tmp_path)[0]

    assert record.muf.letter == saoxml.QL_GREATER_THAN


def test_read_ignores_unknown_elements(sounding, tmp_path):
    """Section 1.3.1: readers skip what they do not recognise."""
    root = saoxml.build_document([record_of(sounding)])
    ET.SubElement(root[0], "SomethingFromAnotherSounder", Value="1")
    saoxml.write(root, tmp_path / "extra.xml")

    record = saoxml.read(tmp_path / "extra.xml")[0]
    assert record.muf is not None
    assert record.traces


def test_read_survives_a_record_with_no_pick(make_lfs, tmp_path):
    """Noise only: no MUF, no traces, but still a readable record."""
    iq = (np.random.default_rng(4).normal(size=(N_FREQ * WINDOW, 2))
          .astype(np.float32).view(np.complex64).ravel())
    path = make_lfs(iq, name="noise.lfs", **COMPLETE_SWEEP)
    ion = spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))

    record = _round_trip(ion, tmp_path)[0]
    assert record.station == ion.header.rx_name
    assert record.muf is None or np.isfinite(record.muf.value)


# --- rendering ---------------------------------------------------------------

def test_render_draws_a_record(sounding, tmp_path):
    from muf import render

    record = _round_trip(sounding, tmp_path)[0]
    out = render.plot_sao(record, tmp_path / "record.png")

    assert out.exists() and out.stat().st_size > 0


def test_render_says_so_when_nothing_was_scaled(make_lfs, tmp_path):
    """A record whose estimator declined is valid, and must not draw blank.

    Blank axes read as a broken renderer; the picture has to distinguish "no
    trace was found" from "the drawing failed".
    """
    from muf import render

    iq = (np.random.default_rng(9).normal(size=(N_FREQ * WINDOW, 2))
          .astype(np.float32).view(np.complex64).ravel())
    path = make_lfs(iq, name="empty.lfs", **COMPLETE_SWEEP)
    ion = spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))

    record = _round_trip(ion, tmp_path)[0]
    assert not record.traces

    out = render.plot_sao(record, tmp_path / "empty.png")
    assert out.exists() and out.stat().st_size > 0


def test_trace_points_default_off_once_there_is_a_raster(sounding):
    """The evidence outranks the interpretation.

    Branch labels come from segmentation; the raster is what was measured. So
    the points appear only when there is no ionogram to speak for itself,
    unless asked for either way.
    """
    from muf import render

    assert render.draw_trace_points(None) is True
    assert render.draw_trace_points(sounding) is False
    assert render.draw_trace_points(sounding, True) is True
    assert render.draw_trace_points(None, False) is False


def test_render_puts_the_record_over_its_ionogram(sounding, tmp_path):
    from muf import render

    record = _round_trip(sounding, tmp_path)[0]
    out = render.plot_sao(record, tmp_path / "over.png", ion=sounding)

    assert out.exists() and out.stat().st_size > 0


def test_render_over_a_raster_spans_the_whole_sweep(sounding, tmp_path):
    """Cropped to the echoes it stops being an ionogram.

    Framing to the trace is right for the bare point plot, where there is
    nothing else to see; over a raster it throws away the sweep.
    """
    import matplotlib.pyplot as plt
    from muf import render

    record = _round_trip(sounding, tmp_path)[0]
    render.plot_sao(record, tmp_path / "span.png", ion=sounding)

    # plot_sao closes its own figure, so check the limits it asked for by
    # rebuilding the frame on a throwaway axes.
    fig, ax = plt.subplots()
    drawn = [t for t in record.traces if t.freq.size]
    render._sao_frame(ax, record, sounding, drawn, False, [], None)
    assert ax.get_xlim() == pytest.approx(
        (sounding.cal.freq_start, sounding.cal.freq_stop))
    plt.close(fig)


def test_render_needs_no_lfs(real_file, tmp_path):
    """The point of the format: draw from the XML alone.

    The .lfs is read once to make the file, then never opened again -- the
    renderer is handed a path to XML, as a stranger would receive it.
    """
    from muf import render

    saoxml.write(saoxml.export_file(real_file, Options(methods=("algo",))),
                 tmp_path / "published.xml")
    record = saoxml.read(tmp_path / "published.xml")[0]

    out = render.plot_sao(record, tmp_path / "published.png")
    assert out.exists() and out.stat().st_size > 0
    assert record.muf.value == pytest.approx(12.2, abs=0.3)


# --- modelled values beside the measured ones ---------------------------------
#
# <Modeled> is the format's own answer to "a number this instrument did not
# produce". The tests police the boundary: a model value must never be
# reachable as though it were a measurement.

IRI = saoxml.ModelValues(name="IRI", muf_mhz=9.904, fof2_mhz=3.355,
                         hmf2_km=331.8, options="control point 45.88N 39.45E")


def _with_model(ion, tmp_path, model=IRI, method="algo"):
    result = extractors.get(method)(ion)
    root = saoxml.build_document(
        [saoxml.build_record(ion, result, model=model)])
    saoxml.write(root, tmp_path / "modelled.xml")
    return saoxml.read(tmp_path / "modelled.xml")[0]


def test_model_values_are_written_as_modeled(sounding, tmp_path):
    record = _with_model(sounding, tmp_path)

    for name, expected in (("MUF", 9.904), ("foF2", 3.355), ("hmF2", 331.8)):
        item = record.characteristic(name, "IRI")
        assert item is not None, f"no modelled {name}"
        assert item.kind == "Modeled"
        assert item.modelled is True
        assert item.value == pytest.approx(expected, abs=1e-3)


def test_a_modelled_muf_is_not_reachable_as_the_measured_one(sounding, tmp_path):
    """`record.muf` must mean the instrument's answer, always."""
    record = _with_model(sounding, tmp_path)
    measured = extractors.get("algo")(sounding).pick.muf_mhz

    assert record.muf.model == ""
    assert record.muf.value == pytest.approx(measured, abs=1e-3)
    assert record.muf.value != pytest.approx(IRI.muf_mhz, abs=1e-3)


def test_two_fof2_values_coexist_and_stay_distinguishable(sounding, tmp_path):
    """One from the secant law, one from IRI. Both are foF2; neither is 'the' one."""
    record = _with_model(sounding, tmp_path)

    models = {c.model for c in record.characteristics if c.name == "foF2"}
    assert models == {"secant-law", "IRI"}
    assert record.characteristic("foF2", "IRI").value == pytest.approx(3.355, abs=1e-3)
    assert (record.characteristic("foF2", "secant-law").value
            != pytest.approx(3.355, abs=1e-3))


def test_non_finite_model_values_are_omitted(sounding, tmp_path):
    """A model that could not answer writes nothing, rather than a NaN."""
    partial = saoxml.ModelValues(name="IRI", muf_mhz=float("nan"),
                                 fof2_mhz=3.3, hmf2_km=float("nan"))
    record = _with_model(sounding, tmp_path, partial)

    assert record.characteristic("MUF", "IRI") is None
    assert record.characteristic("hmF2", "IRI") is None
    assert record.characteristic("foF2", "IRI") is not None


def test_no_model_means_no_extra_characteristics(sounding, tmp_path):
    record = _with_model(sounding, tmp_path, model=None)
    assert not [c for c in record.characteristics if c.model == "IRI"]


def test_render_separates_modelled_rows(sounding, tmp_path):
    from muf import render

    record = _with_model(sounding, tmp_path)
    out = render.plot_sao(record, tmp_path / "modelled.png", ion=sounding)
    assert out.exists() and out.stat().st_size > 0
