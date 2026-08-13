"""Exporting soundings as SAO.XML 5.0.

SAO.XML 5.0 is the URSI/INAG interchange format for ionogram-derived data
(Reinisch, Galkin and Khmyrov, UMLCAR, 25 September 2008, DTD revision 5.0.1f).
It is what GIRO stations publish and what DIDBase ingests, and it supersedes the
flat SAO 4 format.

**Why SAO.XML and not SAO 4.** SAO 4 is a fixed-format file whose 80-slot index
hardwires each numbered group to a vertical layer and magnetoionic mode -- groups
7-11 are the F2 O-trace, 22-25 the F2 X-trace, 51-53 the true-height profile.
There is no slot for a bistatic path and no extension mechanism, so it cannot
represent this instrument at all. SAO.XML 5.0 can, through three documented
extension points:

``<Trace Type="non-standard">``
    The ``Type`` attribute is specified as "standard or non-standard". An oblique
    trace is non-standard.
``<Custom Name= Units= Val= Description=>``
    A user-defined characteristic, for the MUF.
``<SystemInfo>``
    "Custom attributes and elements ... are allowed"; the spec's own examples are
    a ``UMLStationID`` attribute and a ``<DigisondePreface>`` element. The
    transmitter/receiver pair goes here.

Extension is safe by contract: section 1.3.1 requires readers to skip unknown
elements and attributes.

**The MUF is emitted as <Custom>, never as URSI ID 03 or 07.** URSI's
``M(3000)F2`` and ``MUF(3000)`` are *derived* quantities -- a standard
transmission curve is slid against a vertical trace until tangent and the
frequency read off. UAG-23A section 1.50 is explicit that this is not the same
measurement as ours:

    MUF factors were originally introduced as conversion factors for oblique
    propagation computations. ... This definition corresponds to a rather
    simplified propagation model and it is now known that this Standard MUF is
    not necessarily identical with the Operational MUF of a radio circuit.

This instrument measures the operational MUF of an actual circuit, directly.
Putting that number in ID 07 would tell every consumer it was a nomogram
conversion from a vertical critical frequency, which it is not.

**Qualifying letters are assigned from UAG-23A section 3.1**, which turns out to
have had standard notation for our flags since 1972:

``D`` -- "Greater than"
    Used "when only limiting values are observed": the trace reached the top of
    the sweep, or the recording was cut short. Our band-limited lower bound.
``U`` -- "Uncertain or doubtful numerical value"
    For a trace "obscured by interference, noise, instrumental defects, spread
    echoes". Our low-SNR and high-scatter picks.

``<URSI>`` carries these as the ``QL`` attribute; ``<Custom>`` (Table 9) does
not define one, so on a custom characteristic the letter rides as a custom
attribute, which 1.3.1 permits.

**"D" inherits a known blind spot.** It fires when the pick lands at the top of
the sweep, so it misses the case where the true MUF is above the band but the
trace faded below it first. Exporting all of 2026-02-04 produces 791 records
carrying a MUF, of which exactly 25 earn "D" -- the same 25 the ``limited_``
column catches, out of the 82 soundings IRI places above the 32.5 MHz ceiling.
They split 13 kmeans / 11 contour / 1 algo: the letter needs a pick at the top
of the sweep, and the algorithmic estimator's three-in-a-row rule rarely gets
there. 07:00 UTC reads 31.94 MHz with no letter at all. Midday records are lower
bounds whether or not they say so. See BACKLOG section 3: the export does not
make this worse, but it does not fix it either.

**What is deliberately absent.**

*Polarization.* ``<Trace>`` normally carries ``Polarization="O"`` or ``"X"``.
This receiver has no polarimetry and the O/X hypothesis was tested and rejected
twice on these soundings, so the attribute is omitted rather than guessed. The
custom ``Branch`` attribute carries what we do know -- low ray or high ray.

*``<ProfileList>``.* Requires an electron-density inversion, which does not
exist yet. ``<Profile Type=>`` does enumerate ``"off-vertical"``, so the format
is ready when the inversion is.

*``<FrequencyStepping>`` and ``<RangeStepping>``.* These are standard
``<SystemInfo>`` sub-elements, but their contents are specified in the spec's
Appendix B, which was not in the copy consulted. Rather than guess a schema the
sweep parameters go in a custom ``<Sweep>`` element.

*``URSICode``.* A required ``<SAORecord>`` attribute, assigned by the station
registry. This path has none, so it defaults to empty; pass ``ursi_code`` if one
is ever issued. The file is well formed either way, but publishing to DIDBase is
an arrangement with UMLCAR rather than a code change.

**One record per estimator.** Section 1.3.4 specifies "separate storage", one
SAO.XML record per interpretation of an ionogram, rather than merging scalers.
Each estimator therefore gets its own ``<SAORecord>`` naming itself in
``<AutoScaler>``, which is exactly how this package's several methods should be
reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .. import (__version__, extractors, fit as fit_module, geometry,
                lof as lof_module, spectro, trace)
from ..pipeline import Options, band_edge_mhz, circuit_ceiling

#: Value of the ``FormatVersion`` attribute this module writes.
FORMAT_VERSION = "5.0"

#: ``<SAORecord>`` ``Source``/``SourceType``. "Ionosonde" is one of the values
#: the spec lists for ``Source``; ``SourceType`` is free text naming the model.
SOURCE = "Ionosonde"
SOURCE_TYPE = "Chirp oblique sounder"

#: UAG-23A section 3.1 qualifying letters. Only these two can be assigned
#: without a human scaler; the rest describe judgements a person makes.
QL_GREATER_THAN = "D"
QL_UNCERTAIN = "U"

#: "Less than", the mirror of D, for a LOF that ran off the bottom of the band.
#:
#: URSI's descriptive letter B -- "measurement influenced by, or impossible
#: because of, absorption near fmin" -- is the obvious companion and is
#: deliberately *not* emitted: it needs a criterion for when the propagation
#: window has closed, and the window measured over 2026-02-04 is a smooth
#: 1.78-18.35 MHz with no break to put one at. Inventing a threshold here is the
#: mistake UNCERTAIN_SNR_DB already had to be rescued from. See BACKLOG 14.
QL_LESS_THAN = "E"

#: Below this peak SNR the pick earns UAG-23A's "U": the echo is barely above
#: what the detector itself demands, which is the "obscured by noise" condition
#: the letter describes.
#:
#: Tied to the algorithmic extractor's own linear-power threshold of 20, i.e.
#: 13 dB, with headroom -- not to a percentile of any particular day, which
#: would manufacture a flag whenever one sounding happened to be weakest. On
#: 2026-02-04 the whole day sits between 44.6 and 67.3 dB (n=64), so nothing
#: earns "U". That is the correct answer for a day of strong echoes.
UNCERTAIN_SNR_DB = 20.0

#: Scatter about the local trend, in *range bins*, above which the trace is too
#: ragged to call a clean measurement. Expressed in bins so it tracks --window
#: and --zero-periods instead of silently changing meaning with them.
#:
#: Two bins: below one bin the scatter is quantisation, not wander. Measured on
#: primary segments over 2026-02-04, scatter runs 0.0-15.3 km (median 3.8)
#: against a 14.6 km bin, so again nothing earns "U" on this day.
UNCERTAIN_SCATTER_BINS = 2.0

#: ``<Trace Layer=>`` is a required attribute, but this instrument cannot
#: resolve which layer formed an echo -- there is no vertical trace to compare
#: against. F2 is the standard assumption for the MUF-carrying mode over a path
#: this long; it is an assumption, not a measurement, and is overridable.
DEFAULT_LAYER = "F2"

#: Assumed peak height for the secant-law back-conversion reported as a
#: ``<Modeled>`` foF2. Stated in ``ModelOptions`` so the assumption travels with
#: the number.
EQUIVALENT_HMF2_KM = geometry.DEFAULT_HMF2_KM


# --- formatting --------------------------------------------------------------

def _numbers(values, fmt: str, per_line: int = 15) -> str:
    """Space-separated values, wrapped, in the layout the spec's samples use."""
    items = [fmt.format(float(v)) for v in values]
    lines = [" ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "\n" + "\n".join(lines) + "\n"


def _attrs(**kwargs) -> dict[str, str]:
    """Drop empty values so optional attributes simply do not appear."""
    return {k: str(v) for k, v in kwargs.items() if v not in (None, "")}


# --- quality -----------------------------------------------------------------

def qualifying_letter(ion, result, primary=None,
                      band_ceiling_mhz: float | None = None) -> str:
    """The UAG-23A section 3.1 letter this pick has earned, or ``""``.

    ``D`` takes precedence over ``U``: a value known only as a lower bound is a
    limiting value first and an uncertain one second.

    ``band_ceiling_mhz`` is the highest frequency the circuit actually returns;
    the default of the declared sweep stop withholds ``D`` from every clipped
    pick on a path that gives out below it. Shares
    :func:`muf.pipeline.band_edge_mhz` with the ``limited_`` columns so the two
    cannot drift apart -- they did, and the anchor bug outlived being noticed.
    """
    if not result.ok:
        return ""

    cal = ion.cal
    if result.pick.muf_mhz >= band_edge_mhz(cal, band_ceiling_mhz) \
            or not cal.sweep_complete:
        return QL_GREATER_THAN

    if np.isfinite(result.pick.snr_db) and result.pick.snr_db < UNCERTAIN_SNR_DB:
        return QL_UNCERTAIN
    if primary is not None and primary.n_points:
        limit = UNCERTAIN_SCATTER_BINS * cal.range_step
        if trace.trace_scatter_km(primary.vrange) > limit:
            return QL_UNCERTAIN
    return ""


# --- document parts ----------------------------------------------------------

def _system_info(ion, method: str) -> ET.Element:
    header, cal = ion.header, ion.cal
    tx, rx, path_km = geometry.path_of(header)

    info = ET.Element("SystemInfo")
    ET.SubElement(info, "AutoScaler").text = f"muf {__version__} ({method})"

    ET.SubElement(info, "ObliquePath", _attrs(
        TransmitterName=header.tx_name,
        TransmitterLatitude=f"{tx.lat:.4f}",
        TransmitterLongitude=f"{tx.lon:.4f}",
        ReceiverName=header.rx_name,
        ReceiverLatitude=f"{rx.lat:.4f}",
        ReceiverLongitude=f"{rx.lon:.4f}",
        GreatCircleDistance=f"{path_km:.1f}",
        Units="km",
    ))

    ET.SubElement(info, "Sweep", _attrs(
        StartFrequency=f"{cal.freq_start:.4f}",
        StopFrequency=f"{cal.freq_stop:.4f}",
        NominalStopFrequency=f"{cal.freq_stop_nominal:.4f}",
        FrequencyStep=f"{cal.freq_step_mhz:.6f}",
        FrequencyUnits="MHz",
        RangeGateLow=f"{cal.gate_km[0]:.1f}",
        RangeGateHigh=f"{cal.gate_km[1]:.1f}",
        RangeStep=f"{cal.range_step:.4f}",
        RangeResolution=f"{cal.resolution_km:.4f}",
        RangeUnits="km",
        Complete=str(bool(cal.sweep_complete)).lower(),
        Fraction=f"{cal.sweep_fraction:.4f}",
    ))

    info.append(_acquisition(header))

    comments = [
        "Oblique-incidence chirp sounding over a fixed transmitter-receiver "
        "path. Ranges are group range along the path, not virtual height. "
        "Vertical characteristics (foF2, foE, h'F, hmF2) are not measurable on "
        "this geometry and are absent rather than blank. No polarimetry, so "
        "traces carry no Polarization attribute."
    ]
    if _range_is_relative(header):
        comments.append(
            "RANGES ARE RELATIVE. " + _relative_reason(header) + " Range "
            "*differences* in this record are correct; the zero is not, so "
            "group range and any height derived from it must not be read as "
            "absolute. See RangeList/@Reference on every trace."
        )
    ET.SubElement(info, "Comments").text = "\n".join(comments)
    return info


def _range_is_relative(header) -> bool:
    """Whether this sounding's range axis has a trustworthy zero.

    ``.lfs`` headers have no such concept and answer False, which is right:
    their range zero comes from a scheduled transmit time, not from a fitted
    detection.
    """
    return bool(getattr(header, "range_is_relative", False))


def _relative_reason(header) -> str:
    return str(getattr(header, "range_relative_reason", "") or
               "the range offset could not be established")


def _acquisition(header) -> ET.Element:
    """Acquisition parameters, the ones that decide what the numbers mean.

    ``<Sweep>`` describes the axes; this describes the radio that produced
    them. The chirp rate above all -- ``range = c * f_beat / rate`` is the
    whole measurement, and a record that omits it cannot be checked, let
    alone reproduced. Everything is read with ``getattr`` so one writer serves
    both header types: v1 has no ``t0`` or git provenance, v2 has no
    ``whiten``, and an attribute that does not apply is omitted rather than
    written blank.
    """
    element = ET.Element("Acquisition", _attrs(
        Format=str(getattr(header, "format", "") or ""),
        ChirpRate=_fmt(getattr(header, "rate", None), "{:.4f}"),
        ChirpRateUnits="Hz/s",
        SampleRate=_fmt(getattr(header, "sample_rate", None), "{:.1f}"),
        Decimation=_fmt(getattr(header, "dec", None), "{:.0f}"),
        SampleRateUnits="Hz",
        Channel=str(getattr(header, "channel", "") or "") or None,
        # v2's t0 carries the sub-second sweep start, and its fractional part
        # *is* the one-way travel time (io_chirp.range_offset_km). StartTimeUTC
        # is written to whole seconds for compatibility, so without this the
        # record cannot be re-derived.
        SweepStartEpoch=_fmt(getattr(header, "t0", None), "{:.6f}"),
        SweepStartEpochUnits="s",
        NoiseFloorMedian=_fmt(getattr(header, "noise_floor_median", None),
                              "{:.3f}"),
        Detections=_fmt(getattr(header, "num_detections", None), "{:.0f}"),
        # Recorded by the v1 console, never applied by us -- you cannot whiten
        # twice. Exported because it decides whether the median-based noise
        # floor is well founded: measured over the archive, `whiten=1` files
        # sit within 0.5 % of an exponential power spectrum and `whiten=0`
        # ones 45 % above it, so dB values from the two are not directly
        # comparable. See docs/signal-chain.md sec. 5.1.
        Whitening=(None if getattr(header, "whiten", None) is None
                   else str(bool(header.whiten)).lower()),
        WhiteningLength=_fmt(getattr(header, "whiten_len", None), "{:.0f}"),
    ))

    # Provenance: which acquisition code wrote the product. `git_dirty` is
    # written even when False -- a record that does not say is different from
    # one that says the tree was clean.
    version = getattr(header, "software_version", "")
    commit = getattr(header, "git_commit", "")
    dirty = getattr(header, "git_dirty", None)
    if version or commit:
        ET.SubElement(element, "Recorder", _attrs(
            Software=str(version) or None,
            Commit=str(commit) or None,
            Dirty=None if dirty is None else str(bool(dirty)).lower(),
        ))

    ET.SubElement(element, "RangeReference", _attrs(
        Value="relative" if _range_is_relative(header) else "absolute",
        Reason=_relative_reason(header) if _range_is_relative(header) else None,
    ))
    return element


def _fmt(value, spec: str) -> str | None:
    """Format a number, or ``None`` so ``_attrs`` drops the attribute.

    A missing acquisition parameter must be absent, never zero: "the chirp
    rate was not recorded" and "the chirp rate was 0 Hz/s" would otherwise
    read the same, and the second is impossible.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return spec.format(number) if np.isfinite(number) else None


@dataclass(frozen=True)
class ModelValues:
    """An external model's answer for the same sounding.

    Carried into the record as ``<Modeled>`` characteristics, which is what the
    element is for: a value the model asserts, next to the one the instrument
    measured, each labelled with where it came from. A reader that wants only
    measurements filters on the tag; a reader drawing the record gets the
    comparison for free.

    Nothing here is used to correct, weight or gate the measurement. IRI's MUF
    for this path on this day disagrees with the measured one by several MHz at
    times, and which is right is not settled by writing them side by side.
    """

    name: str
    muf_mhz: float = float("nan")
    fof2_mhz: float = float("nan")
    hmf2_km: float = float("nan")
    options: str = ""

    @property
    def any_finite(self) -> bool:
        return any(np.isfinite(v) for v in
                   (self.muf_mhz, self.fof2_mhz, self.hmf2_km))


#: ``(characteristic name, attribute, units, format)`` for each modelled
#: quantity. hmF2 is included because the secant-law foF2 above assumes 300 km
#: -- with a modelled hmF2 beside it a reader can see how far off that is.
_MODELLED = (
    ("MUF", "muf_mhz", "MHz", "{:.3f}"),
    ("foF2", "fof2_mhz", "MHz", "{:.3f}"),
    ("hmF2", "hmf2_km", "km", "{:.1f}"),
)


def _lof_characteristics(chars, ion, low, rungs) -> None:
    """LOF entries: the estimator's own, then the threshold ladder.

    Every one carries its detection level. A LOF without its threshold is not
    reproducible -- it is the frequency where the trace crossed *some* line, and
    moving the line moves the answer by several MHz.
    """
    if low is not None and low.ok:
        ET.SubElement(chars, "Custom", _attrs(
            Name="LOF", Units="MHz", Val=f"{low.lof_mhz:.3f}", SigFig="5",
            ThresholdDb=(f"{low.threshold_db:.1f}"
                         if np.isfinite(low.threshold_db) else None),
            Description=(
                "Lowest observed frequency: the bottom of this estimator's own "
                "trace. Not the ITU-R P.533 LUF, which needs a required "
                "signal-to-noise and a monthly median (P.533-13 section 9)."
            ),
            QL=QL_LESS_THAN if low.at_band_floor else "",
        ))
        if np.isfinite(low.snr_db):
            ET.SubElement(chars, "Custom", _attrs(
                Name="LOFSignalToNoise", Units="dB", Val=f"{low.snr_db:.1f}",
                SigFig="3",
                Description="Peak power at the LOF bin over the noise floor",
            ))

    for level, rung in sorted((rungs or {}).items()):
        if not rung.ok:
            continue
        ET.SubElement(chars, "Custom", _attrs(
            Name=f"LOF@{level:.0f}dB", Units="MHz", Val=f"{rung.lof_mhz:.3f}",
            SigFig="5", ThresholdDb=f"{level:.1f}",
            Description=(
                "Lowest frequency carrying a continuous run of echoes above "
                "this level, measured from the ionogram rather than from any "
                "estimator's trace. The spread across the ladder is how "
                "steeply the signal rolls off into D-region absorption."
            ),
            QL=QL_LESS_THAN if rung.at_band_floor else "",
        ))


def _characteristics(ion, result, nose, letter: str,
                     model: ModelValues | None = None,
                     lof=None, lof_ladder=None) -> ET.Element:
    _, _, path_km = geometry.path_of(ion.header)
    chars = ET.Element("CharacteristicList")

    pick = result.pick
    if pick.ok:
        ET.SubElement(chars, "Custom", _attrs(
            Name="MUF", Units="MHz", Val=f"{pick.muf_mhz:.3f}", SigFig="5",
            Description=(
                f"Operational MUF measured for the "
                f"{ion.header.tx_name}-{ion.header.rx_name} path, "
                f"D={path_km:.0f} km. Not URSI MUF(3000): that is a "
                f"transmission-curve conversion from a vertical critical "
                f"frequency (UAG-23A 1.50)."
            ),
            QL=letter,
        ))
        ET.SubElement(chars, "Custom", _attrs(
            Name="MUFGroupRange", Units="km", Val=f"{pick.vrange_km:.1f}",
            SigFig="5", Description="Group range of the echo at the MUF",
        ))
        if np.isfinite(pick.snr_db):
            ET.SubElement(chars, "Custom", _attrs(
                Name="MUFSignalToNoise", Units="dB", Val=f"{pick.snr_db:.1f}",
                SigFig="3",
                Description="Peak power at the picked bin over the noise floor",
            ))

        equivalent = geometry.muf_to_fof2(pick.muf_mhz, path_km,
                                          EQUIVALENT_HMF2_KM)
        ET.SubElement(chars, "Modeled", _attrs(
            Name="foF2", Units="MHz", Val=f"{equivalent:.3f}",
            ModelName="secant-law",
            ModelOptions=f"hmF2={EQUIVALENT_HMF2_KM:.0f}km,D={path_km:.0f}km",
        ))

    if nose is not None and nose.ok:
        ET.SubElement(chars, "Custom", _attrs(
            Name="MUFNoseFit", Units="MHz", Val=f"{nose.muf_mhz:.3f}",
            SigFig="5", Bound=f"{nose.rms_residual_mhz:.3f}",
            BoundaryType="1sigma",
            Description=(
                "MUF from the vertex of a parabola fitted to both branches of "
                "the nose; bound is the RMS residual of that fit"
            ),
            QL=letter,
        ))

    _lof_characteristics(chars, ion, lof, lof_ladder)

    if model is not None:
        for name, attribute, units, fmt in _MODELLED:
            value = getattr(model, attribute)
            if not np.isfinite(value):
                continue
            ET.SubElement(chars, "Modeled", _attrs(
                Name=name, Units=units, Val=fmt.format(value),
                ModelName=model.name, ModelOptions=model.options,
            ))
    return chars


def _trace_list(segments, layer: str = DEFAULT_LAYER, header=None) -> ET.Element | None:
    if not segments:
        return None

    relative = _range_is_relative(header) if header is not None else False
    traces = ET.Element("TraceList", Num=str(len(segments)))
    for item in segments:
        element = ET.SubElement(traces, "Trace", _attrs(
            Type="non-standard",
            Layer=layer,
            Multiple=item.hops,
            Num=item.n_points,
            Branch=item.branch,
            NoseGroup=item.group,
            ReflectionHeight=(f"{item.height_km:.0f}"
                              if item.height_km is not None else None),
        ))
        ET.SubElement(element, "FrequencyList", _attrs(
            Type="float", SigFig="5", Units="MHz", Description="Nominal Frequency",
        )).text = _numbers(item.freq, "{:.3f}")
        # `Reference` is not optional decoration. A relative axis published as
        # plain group range is a number a consumer will use, and at DOB the
        # zero has been wrong by as much as 286,000 km. Differences survive;
        # the origin does not.
        ET.SubElement(element, "RangeList", _attrs(
            Type="float", SigFig="6", Units="km",
            Reference="relative" if relative else "absolute",
            Description=(
                "Group range offsets along the oblique path, relative to an "
                "unestablished zero -- differences are correct, the origin is "
                "not" if relative else
                "Group Range along the oblique path"),
        )).text = _numbers(item.vrange, "{:.1f}")
        if item.weight is not None and len(item.weight):
            ET.SubElement(element, "TraceValueList", _attrs(
                Name="Amplitude", Type="integer", SigFig="3", Units="dB",
                NoValue="0",
                Description="Relative Amplitude over the equalized noise floor",
            )).text = _numbers(np.rint(item.weight), "{:.0f}")
    return traces


# --- records -----------------------------------------------------------------

def build_record(
    ion,
    result,
    segments=None,
    nose=None,
    ursi_code: str = "",
    station_name: str | None = None,
    layer: str = DEFAULT_LAYER,
    model: ModelValues | None = None,
    lof=None,
    lof_ladder=None,
    band_ceiling_mhz: float | None = None,
) -> ET.Element:
    """One ``<SAORecord>`` for one estimator's reading of one sounding."""
    header = ion.header
    _, rx, _ = geometry.path_of(header)
    primary = trace.primary_segment(segments) if segments else None
    letter = qualifying_letter(ion, result, primary, band_ceiling_mhz)

    record = ET.Element("SAORecord", {
        "FormatVersion": FORMAT_VERSION,
        "StartTimeUTC": header.datetime.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        # Required by Table 4, so it is written even when empty: a blank code
        # says "this path has no registry entry", where omitting the attribute
        # would say "forgot".
        "URSICode": ursi_code,
        "StationName": station_name or header.rx_name,
        "GeoLatitude": f"{rx.lat:.4f}",
        "GeoLongitude": f"{rx.lon:.4f}",
        "Source": SOURCE,
        "SourceType": SOURCE_TYPE,
        "ScalerType": "Auto",
        "PathType": header.path_type,
    })

    record.append(_system_info(ion, result.method))
    record.append(_characteristics(ion, result, nose, letter, model,
                                   lof, lof_ladder))
    traces = _trace_list(segments, layer, header)
    if traces is not None:
        record.append(traces)
    return record


def build_document(records) -> ET.Element:
    """Wrap records in the ``<SAORecordList>`` root."""
    root = ET.Element("SAORecordList")
    for record in records:
        root.append(record)
    return root


def records_for(ion, options: Options | None = None, **kwargs) -> list[ET.Element]:
    """One record per estimator, following the spec's separate-storage rule."""
    options = options or Options()
    results = extractors.run(ion, methods=options.methods, **options.per_method())
    # setdefault, not assignment: an explicit kwarg from the caller wins.
    # `circuit_ceiling` resolves the flag against the registry, so the SAO `D`
    # letter and the CSV `limited_` column see the same number for this circuit.
    kwargs.setdefault("band_ceiling_mhz", circuit_ceiling(ion.header, options))

    # The ladder is estimator-independent, so it is computed once and shared:
    # every record in the file gets the same rungs, which is the point of it.
    rungs = None
    if options.lof:
        rungs = lof_module.ladder(ion, band_floor_mhz=options.band_floor_mhz)

    out = []
    for result in results.values():
        segments = nose = low = None
        if options.lof:
            low = lof_module.pick_lof(
                result.presence, ion.freq, power_db=ion.db, vrange=ion.vrange,
                band_floor_mhz=options.band_floor_mhz,
            )
        if result.ok:
            freq, vrange, weight = trace.extract_points(ion, result)
            _, _, path_km = geometry.path_of(ion.header)
            segments = trace.merge_branches(trace.identify_hops(
                trace.group_tracks(freq, vrange, weight), path_km))
            nose = fit_module.fit_nose(*trace.nose_points(segments))
        out.append(build_record(ion, result, segments, nose,
                                lof=low, lof_ladder=rungs, **kwargs))
    return out


def export_file(path, options: Options | None = None, **kwargs) -> ET.Element:
    """Read one sounding of either format and return its ``<SAORecordList>``.

    Goes through :mod:`muf.loader` rather than :func:`spectro.compute_cached`,
    so ``.h5`` exports at all -- calling the ``.lfs`` reader directly made this
    the last command that could not (``architecture.md`` sec. 3.2).
    """
    from .. import interference, loader

    options = options or Options()
    ion = loader.load(
        Path(path), options.window, options.zero_periods,
        options.gate_km, options.cache_dir,
        format=options.format, stations=options.stations,
    )
    ion, _ = interference.apply(ion, options)
    return build_document(records_for(ion, options, **kwargs))


def to_string(root: ET.Element) -> str:
    """Serialise, indented, with an XML declaration."""
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def write(root: ET.Element, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_string(root), encoding="utf-8")
    return out_path


# --- reading -----------------------------------------------------------------
#
# Reading back is what makes the export worth writing: a record that only this
# package can interpret is a private file with angle brackets. These types hold
# what a *consumer* can recover from the XML alone -- no .lfs, no spectrogram,
# no knowledge of how the trace was found.

@dataclass(frozen=True)
class Characteristic:
    """One entry of a ``<CharacteristicList>``."""

    name: str
    value: float
    units: str = ""
    #: ``Val`` exactly as written. Kept so a reader can echo the file's own
    #: precision instead of inventing digits: the MUF is published as "12.200"
    #: and the SNR as "54.6", and reformatting both to three decimals would
    #: claim a milli-dB the instrument never measured.
    text: str = ""
    #: Element tag it came from: ``URSI``, ``Modeled`` or ``Custom``. Worth
    #: keeping -- a modelled value and a measured one should never be shown as
    #: though they carried the same authority.
    kind: str = "Custom"
    #: ``ModelName`` for a ``<Modeled>`` value: which model asserted it. Two
    #: entries can share a ``Name`` -- an IRI foF2 and a secant-law one are both
    #: called foF2 -- so this is what tells them apart.
    model: str = ""
    #: UAG-23A qualifying letter, ``""`` when the value is unflagged.
    letter: str = ""
    bound: float | None = None
    description: str = ""

    @property
    def modelled(self) -> bool:
        return self.kind == "Modeled"


@dataclass(frozen=True)
class Trace:
    """One ``<Trace>``: a polyline of echoes with optional amplitudes."""

    freq: np.ndarray
    vrange: np.ndarray
    amplitude: np.ndarray | None = None
    layer: str = ""
    branch: str = ""
    group: int | None = None
    hops: int | None = None
    height_km: float | None = None
    #: ``RangeList/@Reference``. ``"relative"`` means `vrange` differences are
    #: correct and the origin is not, so these must not be read as group
    #: range. Defaults to absolute, which is what every record written before
    #: the attribute existed meant.
    range_reference: str = "absolute"

    @property
    def range_is_relative(self) -> bool:
        return self.range_reference == "relative"

    @property
    def n_points(self) -> int:
        return int(self.freq.size)

    @property
    def label(self) -> str:
        parts = [self.branch or "unlabelled"]
        if self.hops:
            parts.append(f"{self.hops}-hop")
        return " ".join(parts)


@dataclass(frozen=True)
class Record:
    """One ``<SAORecord>``, as much of it as a reader needs to draw it."""

    time: datetime | None
    station: str = ""
    ursi_code: str = ""
    latitude: float | None = None
    longitude: float | None = None
    path_type: str = ""
    scaler: str = ""
    path: dict[str, str] = field(default_factory=dict)
    sweep: dict[str, str] = field(default_factory=dict)
    #: ``<Acquisition>`` attributes: chirp rate, sample rate, sweep start
    #: epoch, recorder provenance. Empty for a record written before it
    #: existed, which is why every reader must treat a missing key as unknown
    #: rather than defaulting it.
    acquisition: dict[str, str] = field(default_factory=dict)
    characteristics: list[Characteristic] = field(default_factory=list)
    traces: list[Trace] = field(default_factory=list)

    @property
    def range_is_relative(self) -> bool:
        """True when any trace in this record has no trustworthy range zero."""
        return any(t.range_is_relative for t in self.traces)

    @property
    def chirp_rate(self) -> float | None:
        """Hz/s. ``range = c * f_beat / rate``, so nothing checks out without it."""
        return _float(self.acquisition.get("ChirpRate"))

    @property
    def method(self) -> str:
        """The estimator named in ``<AutoScaler>``, e.g. ``algo``."""
        match = re.search(r"\(([^)]+)\)\s*$", self.scaler)
        return match.group(1) if match else ""

    def characteristic(self, name: str,
                       model: str | None = None) -> Characteristic | None:
        """First characteristic with this name, optionally from one model.

        ``model`` matters once a record carries both an IRI foF2 and a
        secant-law one: without it, "the foF2" is whichever the writer put
        first. ``model=""`` selects the measured value specifically.
        """
        for item in self.characteristics:
            if item.name == name and (model is None or item.model == model):
                return item
        return None

    @property
    def muf(self) -> Characteristic | None:
        """The *measured* MUF. A modelled one is fetched by name and model."""
        return self.characteristic("MUF", model="")

    def number(self, key: str, source: str = "sweep") -> float | None:
        """A numeric ``<Sweep>`` or ``<ObliquePath>`` attribute, or None."""
        return _float((self.sweep if source == "sweep" else self.path).get(key))


def _float(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _int(text) -> int | None:
    value = _float(text)
    return int(value) if value is not None else None


def _numbers_in(element) -> np.ndarray:
    """Parse one whitespace-separated number list back to an array."""
    if element is None or not (element.text or "").strip():
        return np.empty(0, dtype=float)
    return np.fromstring(element.text.replace("\n", " "), sep=" ")


def _parse_time(text) -> datetime | None:
    """``StartTimeUTC`` back to an aware datetime.

    The trailing ``Z`` is the whole point of the attribute's name, so the
    result carries UTC rather than being naive -- a naive timestamp compares
    equal to the same wall clock in any zone, which is how a sounding ends up
    filed under the wrong hour.
    """
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_characteristic(element) -> Characteristic | None:
    value = _float(element.get("Val"))
    if value is None:
        return None
    return Characteristic(
        name=element.get("Name", element.get("ID", "")),
        value=value,
        units=element.get("Units", ""),
        text=element.get("Val", ""),
        model=element.get("ModelName", ""),
        kind=element.tag,
        letter=element.get("QL", ""),
        bound=_float(element.get("Bound")),
        description=element.get("Description", ""),
    )


def _parse_trace(element) -> Trace:
    amplitude = None
    for values in element.findall("TraceValueList"):
        if values.get("Name") == "Amplitude":
            amplitude = _numbers_in(values)
    return Trace(
        freq=_numbers_in(element.find("FrequencyList")),
        vrange=_numbers_in(element.find("RangeList")),
        amplitude=amplitude,
        layer=element.get("Layer", ""),
        branch=element.get("Branch", ""),
        group=_int(element.get("NoseGroup")),
        hops=_int(element.get("Multiple")),
        height_km=_float(element.get("ReflectionHeight")),
        range_reference=(ranges.get("Reference", "absolute")
                         if (ranges := element.find("RangeList")) is not None
                         else "absolute"),
    )


def _parse_record(element) -> Record:
    info = element.find("SystemInfo")
    scaler = path = sweep = acquisition = None
    if info is not None:
        scaler = info.findtext("AutoScaler", "")
        path = info.find("ObliquePath")
        sweep = info.find("Sweep")
        acquisition = info.find("Acquisition")

    characteristics = []
    for group in element.findall("CharacteristicList"):
        for item in group:
            parsed = _parse_characteristic(item)
            if parsed is not None:
                characteristics.append(parsed)

    return Record(
        time=_parse_time(element.get("StartTimeUTC")),
        station=element.get("StationName", ""),
        ursi_code=element.get("URSICode", ""),
        latitude=_float(element.get("GeoLatitude")),
        longitude=_float(element.get("GeoLongitude")),
        path_type=element.get("PathType", ""),
        scaler=(scaler or "").strip(),
        path=dict(path.attrib) if path is not None else {},
        sweep=dict(sweep.attrib) if sweep is not None else {},
        acquisition=(dict(acquisition.attrib)
                     if acquisition is not None else {}),
        characteristics=characteristics,
        traces=[_parse_trace(t) for t in element.iter("Trace")],
    )


def read(path) -> list[Record]:
    """Parse a SAO.XML file into records -- one per estimator, in file order."""
    root = ET.parse(Path(path)).getroot()
    if root.tag == "SAORecord":
        return [_parse_record(root)]
    return [_parse_record(item) for item in root.iter("SAORecord")]
