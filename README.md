# ionograms-handler

Extracting **MUF** (Maximum Usable Frequency) from oblique-incidence chirp
ionosonde recordings.

A transmitter sweeps the HF band; a receiver several thousand kilometres away
records the signal reflected from the ionosphere. Fourier-transforming the
de-chirped signal turns delay into **virtual range**, so a power array over
frequency x virtual range is an **ionogram**. The right-hand edge of the
reflection trace is the MUF: the highest frequency the ionosphere still returns,
and therefore the practical ceiling for an HF link over that path.

```
        virtual                                            MUF
        range                                               |
          ^     .:::::..                                    v
     3000 |   .:::::::::::.......                     ....::|
          |  ::::::::::::::::::::::::::::::::::::::::::::::.|
     2500 +--+---------+---------+---------+---------+-------+--> frequency
             8        12        18        24        30      32.5 MHz
```

This repository holds the whole path: read `.lfs` IQ recordings, form the
ionogram, estimate the MUF by several independent methods, track it through the
day, and check it against sources outside the pipeline.

---

## Setup

Everything needed is already installed on the development machine (Python
3.12.10, numpy 1.26.4, scipy 1.13.0, scikit-learn 1.7.2, opencv 4.12,
pandas 2.2.2, matplotlib 3.8.4, PyIRI 0.1.5). To set it up elsewhere:

```bash
cd N:/ionograms-handler
pip install -e .              # gives you a `muf` command, runnable anywhere
```

Or without installing, run it from the repository directory with
`python -m muf ...` — the two are interchangeable everywhere below.

Optional extras, none needed for MUF extraction itself:

```bash
pip install -e ".[iri]"       # the IRI reference model
pip install -e ".[parquet]"   # --format parquet
pip install -e ".[test]"      # pytest
pip install -e ".[cnn]"       # the experimental CNN estimator (TensorFlow)
pip install -e ".[db]"        # only for the pre-existing data_handler/ scripts
```

Check it works — this reads one file header and computes nothing:

```bash
muf info F:/MyData/ND/lfs/2026.02.04 --limit 1
```

You should see the path, the sweep bounds, and the range gate with its reduction
factor. If that prints, the pipeline is ready.

## Quick start

```bash
# what is in a recording?
muf info F:/MyData/ND/lfs/2026.02.04

# what does this circuit's band floor look like? (do this once per path)
muf lof F:/MyData/ND/lfs/2026.02.04

# extract MUF and LOF from a day of soundings
muf run F:/MyData/ND/lfs/2026.02.04 --out out --jobs 0 --daily --band-floor 8.0

# smooth through time: fill gaps, reject outliers
muf track out --method algo --plot

# how well do the methods agree, and do they agree with IRI?
muf compare out --ref-model iri

# draw the ionograms
muf plot F:/MyData/ND/lfs/2026.02.04 --out imgs

# publish in the URSI/INAG interchange format, then draw what was published
muf export F:/MyData/ND/lfs/2026.02.04 --out sao
muf plot-sao sao --ionogram F:/MyData/ND/lfs/2026.02.04 --out imgs/sao
```

The usual sequence is **run → track → compare**: extract per sounding, smooth
through time, then check against the references. `--out` can be anywhere; keep
it off the drive holding the recordings if that one is tight.

Recordings live wherever there is room for them — any path works.

Several days can be processed in one go. Results are always written **one file
per day**, whatever the targets were, since a day is the unit everything
downstream expects:

```bash
# two folders, or a parent containing many days
muf run F:/MyData/ND/lfs/2026.02.04 F:/MyData/ND/lfs/2026.02.05 --out out
muf run F:/MyData/ND/lfs --out out --jobs 0 --daily

# `daily`, `track` and `compare` take several tables, or a directory of them
muf daily   out --method algo
muf track   out --method algo --plot
muf compare out --ref-model iri
```

`--combined` additionally writes one table spanning every day. `daily` handles
each day separately and concatenates, so a missing day stays a gap rather than
being interpolated across; every row carries the `date` it belongs to.

A full day (288 soundings, ~23 GB of IQ) takes about 25 seconds on 8 cores.

---

## How it works

```
 .lfs IQ                muf/io_lfs.py       512-byte header + complex64 samples
    |
    v
 windowed FFT           muf/spectro.py      Hann window, |FFT|^2, fftshift
    |
    v
 noise equalization     muf/spectro.py      divide each row by 4*ln2 * median
    |
    v
 RANGE GATE             muf/calibrate.py    keep only physically possible ranges
    |                                       -- applied inside the FFT loop
    v
 gated ionogram         ~1 MB float32       one array, shared by every estimator
    |
    +--> algorithmic ---+
    +--> k-means -------+---> muf/pick.py ---> MUF, virtual range, quality
    +--> contour -------+          |
    +--> autoencoder ---+          |
                                   +--> muf/fit.py    nose fit: quality metric
                                   |                  and guarded extrapolation
                                   +--> muf/track.py  Kalman over the day:
                                   |                  gaps, outliers, sigma
                                   +--> muf/reference IRI, GIRO, Chapman
```

The gate is the load-bearing step. The range axis spans +/-60,000 km, but on a
2,588 km path the echo can only sit between roughly 2,300 and 5,000 km. Keeping
just that slice discards 97.5% of the array before anything looks at it: **205
bins instead of 8,192, a 40x reduction**, and it removes most of what the
estimators would otherwise mistake for signal.

### The estimators

All four consume the same array and hand a per-frequency "trace present here"
array to one shared decision rule (`muf/pick.py`), so they stay comparable.

| Method | How it finds the trace | From |
|---|---|---|
| `algo` | three vertically adjacent above-threshold cells whose range-neighbours are also lit | `stuffr.filter2_np_nb_MUF`, vectorised |
| `kmeans` | K-means over dB values; keep clusters whose centroid stands above the noise | `MUF_clustering/ionogr_clustering_0.026.py` |
| `contour` | dB threshold, morphological open/dilate, external contours, then intersected back with the cells that were above threshold | `MUF_clustering/segment_ionogram.py` |
| `cnn` | autoencoder denoises, then `contour` reads the result | `MUF_clustering/myCNN_0.02.py` (experimental) |

The shared rule requires the trace to persist over several consecutive
frequency bins before it will call something the MUF. Every source method ended
at "the right-most bright thing", which one interference spike can defeat.

#### Why `contour` and not `thresh`

It was called `thresh` until the threshold turned out to be the one thing it
does *not* do differently: it uses the same 43 dB level as `algo`, by
construction, so the name pointed at the shared part. The contour analysis is
what distinguishes it. `--methods thresh` still works and resolves to
`contour`; existing result tables keep their `muf_thresh` columns, since
renaming those retroactively would make two runs of the same data disagree
about what they measured.

Renaming it meant reading it, which turned up two defects. Both came from
treating morphology as detection rather than as selection.

**The 3×3 opening erased thin traces.** Opening removes anything narrower than
the kernel in *either* axis, and the flat low-ray leg of an oblique trace is
often one range bin tall. At 14:00 UTC it discarded **91%** of the
above-threshold cells (827 → 72); across the day it kept only 9–72%. That is
why `contour` used to start at 10.42 MHz on the 03:00 sounding where `algo`
started at 9.50. The kernel is now `(1, 3)` — frequency only — which still
removes the speckle it exists for (8–31% of cells) while a one-bin trace
survives. A real echo persists across frequency; noise does not, and that is
the axis the test belongs on.

**Dilation and `cv2.FILLED` invented detections.** Only 32–46% of the cells in
the old mask were ever above threshold; the rest were the dilation skirt and
the filled interior of a contour's outline. **8–18% of the runs** handed to
`trace.extract_points` contained no above-threshold cell at all, so their
reported group range was the power-weighted centroid of a *gap* — visibly
floating above the trace it claimed to describe. The retained-contour mask is
now intersected back with the cells that were actually above threshold.

Measured over every fourth sounding of 2026-02-04 (n=72):

| | old | fixed |
|---|---|---|
| soundings with a detection | 64 | **66** |
| trace points emitted | 8,728 | **21,099** |
| points on cells never above threshold | **11.2%** | **0%** |

Coverage in the table above reads *lower* after the fix (90.3% → 88.5%) and
that is the fix working: the trace now reaches the top of the sweep where the
opening used to cut it short, so five more soundings are correctly classified
as **band-limited lower bounds** rather than reported as measurements. They
moved from `n_picked` to `n_band_limited`, not into the bin.

### Results on 2026-02-04

288 soundings, Cyprus -> Yoshkar-Ola, 7.5-32.5 MHz:

| Method | Coverage | Band-limited | MUF range |
|---|---|---|---|
| `algo` | 90.6% | 1 | 9.4-32.1 MHz |
| `contour` | 88.5% | 11 | 10.7-32.1 MHz |
| `kmeans` | 86.8% | 13 | 11.0-32.4 MHz |

| Pair | RMSE | MAE | R² |
|---|---|---|---|
| kmeans vs contour | 0.87 | **0.17** | **0.986** |
| algo vs contour | 1.35 | 0.41 | 0.967 |
| algo vs kmeans | 1.66 | 0.57 | 0.950 |

The diurnal curve behaves as it should: ~12 MHz at night, climbing through
sunrise to ~32 MHz around midday, falling back after sunset.

**`contour` changed sides when its mask was fixed** (see below). It used to
track `algo` at 0.23 MHz MAE and sit 0.41 from `kmeans`; those numbers are now
exactly swapped. The reading: the old 3×3 opening left only the strongest,
most compact cells — the same conservative core `algo`'s three-in-a-row rule
finds — so the agreement was partly an artefact of both methods being clipped
the same way. `contour` and `kmeans` now both see the full above-threshold
extent, and agree to 0.17 MHz over 247 soundings. `algo` is the distinct one.

That is an observation, not a proof: there is no ground truth here, so which
pair is *right* is not settled by their agreeing. What did improve
unambiguously is the trace — see below.

### The low end: LOF

The MUF is where the ionosphere stops returning the signal. The other edge is
where D-region absorption stops it getting through, and it carries its own
information — it tracks solar illumination directly.

```bash
# measure the band floor and summarise LOF over a set of soundings
muf lof F:/MyData/ND/lfs/2026.02.04

# write the per-sounding ladder, and use a different detection level
muf lof F:/MyData/ND/lfs/2026.02.04 --out out/lof.csv --level 50

# then use the floor: LOF below it is flagged as an upper bound
muf run     F:/MyData/ND/lfs/2026.02.04 --band-floor 8.0 --out out
muf export  F:/MyData/ND/lfs/2026.02.04 --band-floor 8.0 --out sao
muf plot-sao sao --ionogram F:/MyData/ND/lfs/2026.02.04 --out imgs/sao
```

The last one draws the propagation window: a cyan dashed line at the LOF, a
white one at the MUF, and the panel listing the whole ladder with its letters.

From the package, without a results table:

```python
from muf import lof, spectro

ion = spectro.compute_cached("…/cyprus1_20260204_210010.lfs")

for level, rung in sorted(lof.ladder(ion, band_floor_mhz=8.0).items()):
    print(f"{level:.0f} dB  {rung}")

# the one form of LUF this instrument supports: P.533 section 9 applied to a
# *measured* S/N, for this circuit and this receiver, one sounding at a time
print(lof.luf_at_snr(ion, required_snr_db=13.0, band_floor_mhz=8.0))
print(lof.luf_at_snr(ion, required_snr_db=27.0, band_floor_mhz=8.0))
```

```console
43 dB  LOF 8.02 MHz at 43 dB (at the band floor: upper bound)
50 dB  LOF 8.02 MHz at 50 dB (at the band floor: upper bound)
57 dB  LOF 9.29 MHz at 57 dB
LOF 8.02 MHz at 43 dB (at the band floor: upper bound)
LOF 9.29 MHz at 57 dB
```

`required_snr_db` is a **true power ratio**; the ladder levels are raw ionogram
dB, which sit 30 dB higher. `spectro.to_db` divides by 1e-3, and median
equalization puts the noise floor at 1.0 in linear power, so **noise reads 30 dB,
not 0** — the 43 dB detection level every estimator shares is a 13 dB
signal-to-noise, which is what the historical linear threshold of 20 always
meant. `luf_at_snr` adds the offset for you (`spectro.NOISE_FLOOR_DB`), which is
why 13 and 27 above land on the 43 and 57 dB rungs.

Below about **10 dB true S/N** the answer saturates at the bottom of the sweep:
the peak in the range gate of pure noise already reads 4–8 dB — the largest of
~205 exponentially distributed samples — so a requirement under that is met
everywhere. `lof.MIN_MEANINGFUL_SNR_DB` records the limit.

**It is LOF, not LUF, and the distinction is the whole design.** ITU-R P.533-13
§9 defines the lowest usable frequency as "the lowest frequency, expressed to
the nearest 0.1 MHz, at which a **required signal-to-noise ratio** is achieved by
the **monthly median** signal-to-noise." A required S/N is a property of a
modulation and a service; a monthly median is a statistic over many soundings.
Neither is a measurement of one ionogram. P.533's field-strength chain makes the
dependence explicit — eq (17) is `E_w = 136.6 + P_t + G_t + 20 log f − L_b`,
carrying transmitter power and antenna gain, which this instrument does not
know.

What an oblique ionosonde scales is the **lowest observed frequency**. The URSI
INAG reference on oblique sounding with signal levels ([Blagov, UAG-104](https://www.ursi.org/files/CommissionWebsites/INAG/uag-104/text/blagov.html))
names exactly three parameters: "the maximum observed frequency (MOF), the
maximum useable frequency (MUF), the lowest observed frequency (LOF)". So this
reports LOF, and reports a LUF only in the one form the data supports —
`lof.luf_at_snr`, which is P.533 §9 applied to a *measured* rather than a
predicted S/N, valid for this circuit and this receiver and one sounding at a
time. Both departures are in the docstring and both belong in any citation.

**It works.** Over 45 soundings of 2026-02-04, LOF at 43 dB correlates with the
cosine of the solar zenith angle at the path midpoint at **r = +0.864** — around
16 MHz through the day, at the band floor at night. That is the D-region
signature, and it is a regression test.

**Every LOF carries its detection threshold.** A LOF is the frequency where the
trace crossed *some* line, and moving the line moves the answer by several MHz —
at 12:00 UTC, 13.84 MHz at 43 dB, 23.10 at 50, 28.83 at 57. This is the
century-old criticism of `fmin` as an equipment parameter. The response is not
to pick one and hide it: `lof43`/`lof50`/`lof57` are reported as a ladder, the
threshold rides in the SAO record's `ThresholdDb` attribute, and the spread
across the ladder is what steep and shallow absorption look like.

**The band floor is not the sweep start.** These recordings declare a sweep from
7.5 MHz, but below 8.0 the peak in the range gate is 34–38 dB *at every hour,
day and night, with no diurnal variation at all* — a flat noise floor, so a
transmitter or filter edge rather than absorption, which would move with the
sun. `muf lof` measures it:

```console
$ muf lof F:/MyData/ND/lfs/2026.02.04
288 sounding(s)

sweep declares        7.50 MHz
measured band floor   8.00 MHz   (DIFFERS)
  pass --band-floor 8.00 so a LOF down there is flagged as an upper bound rather than reported as a measurement
  43 dB:  266/288 scaled   8.00-28.42 MHz  102 at the floor
  50 dB:  260/288 scaled   8.00-29.08 MHz  52 at the floor
  57 dB:  183/288 scaled   8.02-30.63 MHz  4 at the floor
```

**102 of 266** LOF values at 43 dB are therefore upper bounds, not measurements —
the mirror image of the band-limited MUF problem, and it earns the mirror-image
URSI qualifying letter **E** ("less than") to the `D` already used at the top of
the band. Worth checking against the transmitter schedule, and it means the
`<Sweep StartFrequency="7.5000">` published in SAO.XML is the nominal sweep
rather than the radiated band.

**IRI cannot supply a reference LOF.** See "External references" below.

### Tracking MUF through time

Scaling each sounding in isolation throws away the strongest constraint there
is: the ionosphere does not change much in five minutes.

```bash
muf track out --method algo --plot
```

`muf/track.py` runs a constant-velocity Kalman filter with a
Rauch-Tung-Striebel backward pass, weighting each sounding by its own pick
quality (`run_`, `snr_`). On 2026-02-04:

| | raw | tracked |
|---|---|---|
| largest jump between consecutive soundings | **12.06 MHz** | **0.91 MHz** |
| mean step | 0.562 MHz | 0.176 MHz |

252 measured, 27 gaps filled, 9 outliers rejected. Every point carries a
standard deviation -- 0.17 MHz where measured, 0.51 MHz where filled -- so
downstream work can weight or threshold on it. Band-limited picks are excluded
rather than allowed to anchor the state.

### Fitting the trace

`muf/fit.py` fits a parabola to the nose of the trace and reads MUF from the
vertex, following OIASA (Ippolito et al., 2018). Reported per sounding as
`fit_`, `fitres_` and `fitex_`.

**It is fed both branches of the nose**, which is what makes it work. Below the
MUF a low ray and a high ray arrive at every frequency and converge at the nose;
with both present the vertex is bracketed by data rather than extrapolated off
one side:

| against the extractors' pick | one branch | both branches |
|---|---|---|
| bias | −0.12 MHz | **−0.05 MHz** |
| MAE | 0.74 MHz | **0.37 MHz** |
| median residual | 0.31 MHz | **0.18 MHz** |

It is **also an outlier detector**. Where the vertex disagrees with the pick by
more than 3 MHz, the pick is wrong:

```python
suspect = (frame.fit_algo - frame.muf_algo).abs() > 3.0
```

Every such flag on 2026-02-04 was independently rejected by `muf track` on
purely temporal grounds — two mechanisms sharing nothing, agreeing on both the
diagnosis and the correction. It flags less often than it once did, because it
now declines on marginal soundings rather than guessing; `track` is the primary
outlier mechanism.

Two things it does **not** do, both tested:

- **Not a reliability filter — a finding that no longer replicates.** The
  original measurement, against the pre-fix `thresh` mask, was that filtering on
  `fitres < 0.3` made agreement *worse* (MAE 0.302 against 0.245 unfiltered),
  because bad picks tend to have excellent residuals: the trace fits a clean
  parabola, the pick just landed on the wrong part of it. Against the fixed
  `contour` mask the same filter now *helps* — 0.251 MHz over 177 soundings
  against 0.410 unfiltered over 251. Both numbers are agreement between two
  estimators, not accuracy, and the reference moved underneath the test, so
  neither result establishes anything about `fitres_` yet. Treat this as open;
  see BACKLOG §13.
- **Does not recover band-limited MUF.** Of the soundings flagged `limited_`,
  none produce a usable extrapolation: when the trace runs to the top of the
  sweep the nose was never reached, so the fit correctly declines.

### Trace segmentation and reconstruction

An extractor returns the frequency bins where it found signal. That set is not a
curve: on this instrument it covers only **18-48%** of the frequency span it
reaches across, carries **12-37 km** of scatter, and — the part that matters —
is usually **more than one propagation mode stitched together**.

`muf/trace.py` measures the gaps and finds two clear classes:

| change in range across a gap | what it is |
|---|---|
| +15 to +37 km | a fade inside one trace — safe to bridge |
| **−130 to −180 km** | **a mode boundary** — bridging it would be nonsense |

Virtual range *falling* as frequency rises is backwards for a single trace, so a
large drop marks where another mode takes over. A 06:00 sounding has a 5.5 MHz
gap with the range dropping 176 km across it. The extractors' continuity rule
runs straight through those, because it only looks along frequency.

**On 2026-02-04, 87% of soundings carry more than one propagation mode**, two
being the median. That is unsurprising for an oblique path — 1F2, 2F2, sporadic
E and others arrive together — but the extractors see none of it.

`nseg_` turns out to be a good proxy for how strong a sounding is, in the
direction opposite to intuition:

| modes resolved | n | `algo` vs `contour` MAE |
|---|---|---|
| 1 | 77 | 0.281 MHz |
| 2 | 117 | 0.633 MHz |
| 3 | 46 | 0.132 MHz |
| 4 | 9 | 0.112 MHz |
| 5+ | 2 | **0.061 MHz** |

From three modes upward the reading holds: a sounding rich enough to resolve
several modes is well above noise, and the estimators converge. Below that it
does not. The monotonic version of this table was measured against the pre-fix
`thresh` mask, where one-mode soundings were the worst (0.412 MHz); with the
fixed `contour` mask the counts shift heavily toward the low end — 77 one-mode
soundings against 34 before — and **two** modes is now the worst case at 0.633
MHz, twice one-mode. A plausible reading is that two segments is where the
splitter is most often wrong, either cutting one mode in half or merging two;
that is a hypothesis about the segmenter, not a measurement of it. See
BACKLOG §13.

**To see it**, overlay it on the ionogram — the quickest way to judge whether
the extraction is doing the right thing:

```bash
muf plot F:/MyData/ND/lfs/2026.02.04/cyprus1_20260204_060010.lfs --trace --out imgs
```

Each mode is drawn in its own colour with its point count and hop label, the
reconstructed curve in red with its RMS residual, and the y-axis focuses on the
echoes rather than the whole gate. `--trace kmeans` overlays a different
estimator's trace.

**To use it directly:**

```python
from muf import extractors, geometry, spectro, trace

ion = spectro.compute("…/cyprus1_20260204_060010.lfs")
res = extractors.get("algo")(ion)
_, _, ground = geometry.path_of(ion.header)

freq, vrange, weight = trace.extract_points(ion, res)   # several modes per bin
segments, curve = trace.analyse(freq, vrange, ground, weight=weight)

curve.freq, curve.vrange        # continuous h(f) at native resolution
curve.rms_residual_km           # how well the spline describes the points
```

**Modes usually overlap in frequency**, so the points are grouped by *following
each mode across frequency* (`group_tracks`) rather than by splitting a
frequency-ordered sequence (`segment`). Two echoes at 2680 and 2850 km coexist
across the same band; collapsing each frequency bin to one range makes the
extraction alternate between them, which then looks like a trace jumping about.
`extract_points` emits one point per contiguous run in the detection mask, so
both survive.

**Low and high rays are paired, not treated as separate modes.** Track-following
sees them as two tracks, since they are separated in range — the low ray rising
gently (+2 to +17 km/MHz), the high ray falling steeply (−24 to −76). But they
are one mode, converging at the nose. `merge_branches` pairs them when their
high-frequency ends meet: on real noses the ends agree to 8–38 km, while
different modes differ by 160–225 km, so the two cases separate cleanly. Without
this the "primary" track came out as the *high ray* — a 93-point steep fragment
chosen over the 336-point main trace.

**Reconstruction is only emitted where measurements support it** (within 1 MHz
of a real point by default), and `method=` selects `spline` (weighted
smoothing), or `pchip`/`makima`, which bin to medians first and cannot
overshoot. A smoothing spline left to cross a 2.1 MHz gap arced 200 km above the
data on a real sounding.

A 06:00 sounding resolves into a 1-hop echo followed from 13.1 to 28.9 MHz
across a 7 MHz fade, a second mode at 2812 km, and a third at 2754 km — the
primary reconstructed at **8.8 km RMS residual**. A clean 09:00 sounding gives a
single 1-hop trace over 18.5–32.0 MHz at **1.8 km**. Every sounding reports
`nseg_`, `hops_` and `scatter_` per method, so mixed traces are visible in the
results table without opening a plot.

**Hop labelling is weak on this path, deliberately so.** Over 2,588 km a 1-hop
echo spans 2636–2774 km across plausible reflection heights and a 2-hop echo
2774–3182: the families touch, so most segments are genuinely ambiguous and are
left unlabelled rather than assigned on a few kilometres. It sharpens on longer
paths. Segmentation does not depend on it.

### External references

Three estimators agreeing shows consistency, not accuracy -- they share a
spectrogram, a gate, a threshold and a picker, so a common bias would not
appear. `muf/reference/` compares against sources outside the pipeline:

```bash
muf compare out --ref-model all
```

| Reference | What it is | Needs |
|---|---|---|
| `giro` | real ionosonde measurements, converted through the secant law | network; a station near the control point |
| `iri` | the International Reference Ionosphere | `pip install PyIRI` |
| `chapman` | transparent solar-zenith model; **shape only**, amplitude fitted | nothing |
| `minimuf` | not implemented -- see `muf/reference/minimuf.py` | verified coefficients |

For Cyprus -> Yoshkar-Ola the control point is 45.88N 39.45E and **RV149 Rostov**
sits 148 km away, well inside the F2 correlation scale.

**What IRI showed.** Where IRI puts the MUF inside the 32.5 MHz sweep, the
pipeline agrees to **+0.55 MHz** (n=204). Where IRI puts it *above* the sweep --
82 of 288 soundings, every one between 06:00 and 13:00 UTC -- the pipeline reads
**5.25 MHz low**, because the instrument cannot see a MUF above its own band.
The built-in `limited_` flag caught only 15 of those: it fires when a pick lands
at the top of the sweep, so a trace fading below it for signal-strength reasons
still looks like a measurement. **Midday values are lower bounds.** No amount of
internal cross-checking would have revealed this.

**IRI cannot do the same job for LOF, and nothing here can yet.** IRI outputs
electron density and F-region characteristics; it has no absorption term and no
`fmin`. PyIRI 0.1.5 exposes `IRI_density_1day`, `EDP_builder`,
`reconstruct_density_from_parameters` and `edp_to_vtec` — no collision
frequency, which is what absorption is made of. Deriving LOF from a density
profile means integrating ν·N over the D region, and ν needs a neutral
atmosphere model (NRLMSISE-00), not IRI. IRI's D region is also its weakest
part: night-time profiles extend only down to ~80 km, and MF-radar comparisons
show large discrepancies. The scale of the collision term is not a detail —
[ray-tracing work on IONORT-ISP-WC](https://arxiv.org/pdf/2506.24098) reports an
E-region LOF moving from 4.2 MHz without collisions to 10.0 MHz with them, so
essentially the entire signal lives in the part IRI does not model.

The right reference model for LOF is **ITU-R P.533-13 eq (20)**, an empirical
absorption term needing only R₁₂, solar zenith angle, gyrofrequency and foE:

```
L_i = (1 + 0.0067·R₁₂)·sec i · Σⱼ [ AT_jnoon /(f + f_Lj)² ] · [F(χⱼ)/F(χ_jnoon)] · φₙ(f_v/foE_j)
```

Every input is reachable — `reference/indices.py` has R₁₂, `chapman.py` has χ,
IRI gives foE — except `AT_jnoon` and `φₙ`, which the Recommendation publishes
as *figures* rather than formulas. Implementing it means digitising Figures 1–3.
Recorded in BACKLOG §15 rather than guessed at.

### Publishing: SAO.XML 5.0

`muf export` writes soundings in the URSI/INAG interchange format that GIRO
stations publish and DIDBase ingests — [SAO.XML 5.0](https://ulcar.uml.edu/SAOXML/SAO.XML%205.0%20specification%20v1.0.pdf)
(Reinisch, Galkin and Khmyrov, UMLCAR, 2008). This is what turns the segmented
trace from a plotting byproduct into a product: the segments become a
`<TraceList>`.

```bash
# one sounding, one file: sao/cyprus1_20260204_030010.xml
muf export F:/MyData/ND/lfs/2026.02.04/cyprus1_20260204_030010.lfs --out sao

# a whole day, or several
muf export F:/MyData/ND/lfs/2026.02.04 --out sao
muf export F:/MyData/ND/lfs --out sao

# one <SAORecord> per estimator, per the spec's separate-storage rule (1.3.4)
muf export F:/MyData/ND/lfs/2026.02.04 --methods algo,kmeans,contour --out sao

# if a URSI station code is ever issued for this path
muf export F:/MyData/ND/lfs/2026.02.04 --ursi-code XX000 --station "Cyprus-YO" --out sao

# same gate and window flags as `run`, and the same cache
muf export F:/MyData/ND/lfs/2026.02.04 --gate 2000,5000 --cache-dir .cache --out sao

# add IRI's MUF, foF2 and hmF2 beside the measured values
muf export F:/MyData/ND/lfs/2026.02.04 --iri --out sao
```

### Modelled values beside measured ones

`--iri` adds the International Reference Ionosphere's answer for the same path
and instant as `<Modeled>` characteristics — which is exactly what that element
is for. The record then carries both, each labelled with where it came from:

```xml
<Custom  Name="MUF"  Units="MHz" Val="12.200" QL=""/>
<Modeled Name="foF2" Units="MHz" Val="3.879" ModelName="secant-law"
         ModelOptions="hmF2=300km,D=2588km"/>
<Modeled Name="MUF"  Units="MHz" Val="9.904" ModelName="IRI"
         ModelOptions="control point 45.88N 39.45E, D=2588km"/>
<Modeled Name="foF2" Units="MHz" Val="3.355" ModelName="IRI" …/>
<Modeled Name="hmF2" Units="km"  Val="331.8" ModelName="IRI" …/>
```

Two entries are called `foF2` and neither is "the" one: the secant-law value is
this instrument's own MUF converted back to a vertical critical frequency under
a 300 km assumption, the IRI value is an independent model's. `ModelName` is
what tells them apart, so `Characteristic.model` carries it and
`record.characteristic("foF2", "IRI")` selects one. **`record.muf` always means
the measured MUF** — a modelled value is never reachable as though the
instrument had produced it, and there is a test that says so.

`hmF2` is included because the secant-law conversion assumes 300 km; with a
modelled hmF2 on the next line a reader can see how far off that is. On the
03:00 sounding IRI says 331.8 km.

The same record carries the low-frequency end. `LOF` is the bottom of *this
estimator's* trace, so it has no `ThresholdDb` — the estimator is named in
`<AutoScaler>` and its rule is not one dB level (`algo` tests linear power,
`kmeans` has no threshold at all). The `LOF@…dB` rungs are estimator-independent
and each states its own level. `QL="E"` marks the ones that reached the band
floor:

```console
$ muf export …/cyprus1_20260204_210010.lfs --methods contour --iri --band-floor 8.0 --out sao
```

| characteristic | value | letter |
|---|---|---|
| `MUF` | 14.658 MHz | |
| `MUFNoseFit` | 14.515 ± 0.119 MHz | |
| `LOF` | 8.022 MHz | **E** |
| `LOF@43dB` | 8.022 MHz | **E** |
| `LOF@50dB` | 8.022 MHz | **E** |
| `LOF@57dB` | 9.292 MHz | |
| `MUF` (IRI) | 10.126 MHz | |

A night-time sounding: the propagation window runs 8.0–14.7 MHz, and the two
lower rungs are pinned at the floor, so the real window is wider at the bottom
than the record can say. The 57 dB rung at 9.292 MHz is a genuine measurement —
the only one of the three.

Nothing modelled is used to correct, weight or gate a measurement. Over the
whole of 2026-02-04, measured − IRI runs **−16.1 to +4.9 MHz, median +0.74, MAE
3.10** across 766 unflagged records: they agree in the middle and diverge badly
at the extremes, which is the out-of-band problem again (IRI puts 82 soundings
above the 32.5 MHz ceiling). Writing them side by side does not settle which is
right.

One model evaluation covers a whole day: PyIRI takes an array of hours per
date, so `--iri` collects every timestamp up front rather than asking per
sounding — 0.16 s against two minutes. Soundings are grouped by
transmitter/receiver pair, since the control point differs per circuit.
`--offline` uses cached solar indices only; without it the run fetches SILSO and
NOAA and continues without IRI if that fails. Note the driver caveat: R12 does
not exist until six months after the fact, so for a 2026 date IRI is indicative
rather than authoritative.

One XML file per sounding, each holding a `<SAORecordList>`. Two methods give
two records in that list:

```console
$ muf export …/cyprus1_20260204_030010.lfs --methods algo,contour --out sao
exporting 1 sounding(s) to sao
wrote 1/1 record list(s)
```

What comes out, trimmed:

```xml
<SAORecordList>
  <SAORecord FormatVersion="5.0" StartTimeUTC="2026-02-04T03:00:10.000Z" URSICode=""
             StationName="yoshkar-ola" GeoLatitude="56.3800" GeoLongitude="47.5300"
             Source="Ionosonde" SourceType="Chirp oblique sounder" ScalerType="Auto"
             PathType="oblique">
    <SystemInfo>
      <AutoScaler>muf 0.1.0 (algo)</AutoScaler>
      <ObliquePath TransmitterName="cyprus1" TransmitterLatitude="35.0000"
                   ReceiverName="yoshkar-ola" ReceiverLatitude="56.3800"
                   GreatCircleDistance="2588.4" Units="km"/>
      <Sweep StartFrequency="7.5000" StopFrequency="32.4856" FrequencyStep="0.020480"
             RangeGateLow="2329.6" RangeGateHigh="5000.0" Complete="true"/>
    </SystemInfo>
    <CharacteristicList>
      <Custom Name="MUF" Units="MHz" Val="12.200"
              Description="Operational MUF … D=2588 km. Not URSI MUF(3000)…"/>
      <Custom Name="MUFGroupRange" Units="km" Val="2739.0"/>
      <Modeled Name="foF2" Units="MHz" Val="3.879" ModelName="secant-law"
               ModelOptions="hmF2=300km,D=2588km"/>
      <Custom Name="MUFNoseFit" Units="MHz" Val="13.065" Bound="0.181"
              BoundaryType="1sigma"/>
      <Custom Name="LOF" Units="MHz" Val="8.022" QL="E" Description="…"/>
      <Custom Name="LOF@43dB" Units="MHz" Val="8.022" ThresholdDb="43.0" QL="E"/>
      <Custom Name="LOF@50dB" Units="MHz" Val="8.022" ThresholdDb="50.0" QL="E"/>
      <Custom Name="LOF@57dB" Units="MHz" Val="9.292" ThresholdDb="57.0"/>
    </CharacteristicList>
    <TraceList Num="3">
      <Trace Type="non-standard" Layer="F2" Num="34" Branch="low" NoseGroup="0">
        <FrequencyList Type="float" Units="MHz" Description="Nominal Frequency">
9.497 9.640 9.886 9.906 9.988 10.009 …
        </FrequencyList>
        <RangeList Type="float" Units="km" Description="Group Range along the oblique path">
2710.0 2710.0 2710.0 2710.0 2710.0 2710.0 …
        </RangeList>
        <TraceValueList Name="Amplitude" Type="integer" Units="dB" NoValue="0"
                        Description="Relative Amplitude over the equalized noise floor">
50 52 51 50 50 51 …
        </TraceValueList>
      </Trace>
    </TraceList>
  </SAORecord>
</SAORecordList>
```

Reading one back needs nothing but the standard library:

```python
from xml.etree import ElementTree as ET

root = ET.parse("sao/cyprus1_20260204_030010.xml").getroot()
for record in root:
    scaler = record.findtext("SystemInfo/AutoScaler")
    muf = record.find("CharacteristicList/Custom[@Name='MUF']")
    print(scaler, muf.get("Val"), "MHz", muf.get("QL") or "")

    for trace in record.iterfind("TraceList/Trace"):
        freq = [float(v) for v in trace.findtext("FrequencyList").split()]
        vrange = [float(v) for v in trace.findtext("RangeList").split()]
        print(f"  {trace.get('Branch') or 'unlabelled'}: "
              f"{len(freq)} points, {freq[0]:.2f}-{freq[-1]:.2f} MHz")
```

```console
muf 0.1.0 (algo) 12.200 MHz
  low: 34 points, 9.50-12.12 MHz
  unlabelled: 20 points, 9.85-11.07 MHz
  high: 22 points, 11.01-12.20 MHz
muf 0.1.0 (contour) 12.221 MHz
  low: 89 points, 9.44-12.10 MHz
  high: 56 points, 9.82-12.22 MHz
```

The two estimators agree on the MUF to 0.02 MHz and disagree on the trace:
`contour` finds 145 points to `algo`'s 76 over the same nose, and leaves none
of them unlabelled where `algo` cannot place 20. That difference is exactly
what the `<TraceList>` preserves and a bare MUF number throws away.

An empty `QL` means no qualifying letter — the measurement had no difficulty
worth annotating. `D` or `U` there would mean a lower bound or a doubtful value
(see above).

Or from the package, without going through a file:

```python
from muf.export import saoxml
from muf.pipeline import Options

root = saoxml.export_file("…/cyprus1_20260204_030010.lfs", Options(methods=("algo",)))
print(saoxml.to_string(root))
saoxml.write(root, "sao/one.xml")
```

`saoxml.read` parses a file back into `Record` objects, which is the same route
`muf plot-sao` takes:

```python
for record in saoxml.read("sao/cyprus1_20260204_030010.xml"):
    print(record.time, record.method, record.muf.text, record.muf.units)
    print(" ", " | ".join(f"{t.label} {t.n_points}" for t in record.traces))
```

```console
2026-02-04 03:00:10+00:00 algo 12.200 MHz
  low 34 | unlabelled 20 | high 22
2026-02-04 03:00:10+00:00 contour 12.221 MHz
  low 1-hop 89 | high 56
```

`Characteristic.text` is `Val` exactly as written, so a reader can echo the
file's own precision instead of inventing digits: the MUF is published as
`12.200` and the SNR as `54.6`, and reformatting both to three decimals would
claim a milli-dB nothing measured.

Reading the propagation window out of a published record:

```python
from muf.export import saoxml

record = saoxml.read("sao/cyprus1_20260204_210010.xml")[0]
muf, lof = record.muf, record.characteristic("LOF", model="")
print(f"{lof.text}-{muf.text} MHz", "| LOF is an upper bound" if lof.letter == "E" else "")
for level in (43, 50, 57):
    rung = record.characteristic(f"LOF@{level}dB")
    print(f"  {level} dB: {rung.text} MHz {rung.letter}")
```

```console
8.022-14.658 MHz | LOF is an upper bound
  43 dB: 8.022 MHz E
  50 dB: 8.022 MHz E
  57 dB: 9.292 MHz 
```

`record.muf` is always the **measured** MUF — `characteristic("MUF", "IRI")`
fetches the modelled one, and `characteristic("LOF", model="")` is spelled with
the empty model for the same reason: it says "the measured LOF, not a modelled
one", and there is a test that keeps it that way.

### Rendering a record

`muf plot-sao` puts a record's scaled values in a Digisonde-style panel beside
the ionogram they were scaled from:

```bash
# the usual view: SAO formatting over the sounding itself
muf plot-sao sao --ionogram F:/MyData/ND/lfs/2026.02.04 --out imgs/sao

# every record in every file; one estimator only
muf plot-sao sao --out imgs/sao
muf plot-sao sao/cyprus1_20260204_030010.xml --method algo
```

```console
$ muf plot-sao sao --ionogram F:/MyData/ND/lfs/2026.02.04 --out imgs/sao
288 recording(s) to draw under the records
rendering 1 SAO file(s)
  imgs/sao/cyprus1_20260204_030010.png
drew 1 record(s)
```

XML and recording are paired by file name, since `export` names each record
after the sounding it came from. Records with no matching recording still draw —
they fall back to the scaled trace — and the count is reported.

```
┌──────────────────────────┬────────────────────────────────────────┐
│ cyprus1 → yoshkar-ola    │  2900 ┤     ▓▓▒░                       │
│ 2026-02-04 03:00:10 UTC  │       │    ▓█▓▒░  ╎                    │
│ oblique, D = 2588 km     │  2800 ┤   ▓██▓▒   ╎                    │
│ muf 0.1.0 (algo)         │       │  ░▓███▓░  ╎  ← the raster is   │
│                          │  2700 ┤ ▒▓████▓▒░ ╎    what was        │
│ MUF           12.200 MHz │       │           ╎    measured        │
│ MUFGroupRange  2739.0 km │  2600 ┤           ╎                    │
│ MUFSignalToNoise 54.6 dB │       └──┬────┬───╎┬────┬────┬────┬──  │
│ MUFNoseFit 13.065 ±0.181 │         10   11  ╎12   15   20   30    │
│                          │              MUF ╯  (dashed, white)    │
│ foF2 (secant-law)* 3.879 │                                        │
│ MUF (IRI)*         9.904 │   measured and modelled are separated  │
│ foF2 (IRI)*        3.355 │   by a blank line, and every modelled  │
│ hmF2 (IRI)*        331.8 │   row names the model that asserted it │
│ * modelled, not measured │                                        │
└──────────────────────────┴────────────────────────────────────────┘
```

The frequency axis always spans the whole sweep here — cropped to the echoes it
stops being an ionogram — while the range axis narrows to the echoes, exactly as
`muf plot` does. `--full-band` opens it back to the full gate.

**The trace points are off by default over a raster.** Branch labels come from
segmentation, which is the least settled step in this pipeline; the raster is
simply what was measured. Drawing the interpretation on top of the evidence by
default would put the two on equal footing. `--trace` overlays them anyway — open
rings in white, amber and violet, chosen to read on jet — and `--no-trace`
suppresses them even without a raster.

Without `--ionogram` the record draws from the XML alone: no `.lfs`, no
spectrogram, no knowledge of how the trace was found. That is the test of
whether the export is self-describing rather than merely well formed, and it
means anyone who receives the file can see it.

```
cyprus1 → yoshkar-ola      2860 ┤        ○
2026-02-04 03:00:10 UTC         │      ○○   ○         ← unlabelled
oblique, D = 2588 km       2800 ┤     ○  ○ ○○        ╎
muf 0.1.0 (algo)                │        ○  ●● ●     ╎  ← high ray
                           2760 ┤             ● ●●●  ╎
MUF            12.200 MHz       │                ●● ╎▓
MUFGroupRange   2739.0 km  2720 ┤   ● ●●● ● ●●●●●●●●╎▓  ← low ray
MUFSignalToNoise  54.6 dB       │ ●●                ╎▓
foF2*            3.879 MHz 2700 ┼──┬────┬────┬────┬─╎▓──
MUFNoseFit  13.065 ± 0.181     9.5  10.5  11.5  12.5 MHz
                                                   ╎▓
* modelled, not measured            MUF 12.2 ──────╯▓ nose fit 13.065 ± 0.181
```

Points are coloured by branch and sized by echo amplitude. A Digisonde colours
its traces by polarization — red for O, green for X — so the same visual channel
here carries the distinction this receiver *can* make, low ray against high ray,
and the legend says which; borrowing red/green would be read as O/X. The dashed
line is the picked MUF, the thin line and band the nose fit with its 1σ
residual, and a right-hand axis restates group range as excess over the 2588 km
ground path — the oblique equivalent of reading a virtual height.

Rendering both records of a file puts the estimators side by side. On the
03:00 sounding both resolve the nose into low and high branches converging at
12.2 MHz, but they extrapolate it in opposite directions: `algo` fits its
vertex at 13.065 ± 0.181 MHz from 76 points, `contour` at 11.900 ± 0.164 from
145. The picked MUF differs by 0.02 MHz and the fitted one by 1.17 — the
`<TraceList>` is what lets a reader see why.

A record whose estimator found nothing is still a valid record, and renders as
"nothing scaled" rather than as empty axes — blank axes read as a broken
renderer.

**Third-party viewers.** [SAO Explorer](https://ulcar.uml.edu/installers/) is
UMLCAR's own SAO.XML viewer and the reference implementation. These records have
not been opened in it, and two things would want checking before relying on it:
every trace is `Type="non-standard"` with no `Polarization`, and the MUF is a
`<Custom>` characteristic rather than a URSI ID, so a viewer built around
vertical Digisonde records may draw the traces but leave its scaled-values panel
empty. That is the format working as specified — §1.3.1 has readers skip what
they do not recognise — but it is why `plot-sao` exists.

**Not SAO 4.** Its 80-slot index hardwires each group to a vertical layer and
mode — groups 7–11 are the F2 O-trace, 22–25 the F2 X-trace. There is no slot
for a bistatic path and no extension mechanism. SAO.XML 5.0 has three:
`<Trace Type="non-standard">`, `<Custom>` characteristics, and custom elements
under `<SystemInfo>`, with §1.3.1 requiring readers to skip what they do not
recognise.

**The MUF is a `<Custom>`, never URSI ID 03 or 07.** UAG-23A §1.50 says the
MUF-factor method gives a *Standard MUF* from "a rather simplified propagation
model" and that "it is now known that this Standard MUF is not necessarily
identical with the Operational MUF of a radio circuit". URSI's `MUF(3000)` is a
transmission-curve conversion from a vertical critical frequency; this
instrument measures the operational MUF of an actual circuit. Filing ours under
ID 07 would misdescribe it.

**Qualifying letters come from UAG-23A §3.1**, which has had standard notation
for our flags since 1972: **D** ("greater than", used "when only limiting values
are observed") for `limited_` and for truncated sweeps, **U** ("uncertain or
doubtful", for a trace "obscured by interference, noise, instrumental defects")
for weak or ragged picks.

Exporting all 288 soundings of 2026-02-04 gives 791 records carrying a MUF, of
which **25 earn `D` and none earn `U`**. Both numbers deserve a caveat.

**D inherits the midday blind spot** described above. Twenty-five is exactly the
`limited_` count, and it splits very unevenly — `kmeans` 13, `contour` 11,
`algo` 1 — because the letter fires only when a pick lands at the top of the
sweep, and `algo`'s three-in-a-row rule almost never gets there. 07:00 UTC reads
31.943 MHz with no letter at all, against a band ceiling of 32.5. The export
publishes those midday records as measurements when they are lower bounds; see
BACKLOG §3. (Before the `contour` mask fix these counts were 786 and 15: seeing
further along the trace turns measurements into honest lower bounds, which is an
improvement that looks like a regression in any coverage table.)

**U never firing is the right answer here.** Its thresholds are absolute — SNR
within 7 dB of what the detector itself demands, scatter beyond two range bins —
rather than a percentile of the day, and this day's picks run 44.6–67.3 dB with
0.0–15.3 km of scatter against a 14.6 km bin. A percentile would have
manufactured a doubt flag for whichever sounding happened to be weakest.

**Deliberately absent**, rather than blank: `Polarization` (no polarimetry — the
O/X hypothesis was tested twice and rejected; the custom `Branch` attribute
carries low-ray/high-ray instead), all vertical characteristics, and
`<ProfileList>` (needs the inversion in the backlog; `<Profile Type>` does
enumerate `"off-vertical"`, so the format is ready when we are).

Two honest gaps. `URSICode` is required and assigned by the station registry;
this path has none, so it is written empty — publishing to DIDBase is an
arrangement with UMLCAR, not a code change. And `Layer="F2"` is an assumption,
not a measurement, since there is no vertical trace to compare against.

Neither UAG-23A nor Wakai (1987) covers this geometry: UAG-23A §2.7's "oblique"
means off-vertical echoes contaminating a *vertical* ionogram from tilted
layers, and Wakai is 160 worked vertical mid-latitude examples. There is no URSI
convention for scaling oblique ionograms to ignore — we are extending a format
that invites it.

One record per estimator, following §1.3.4's separate-storage rule: methods are
different interpretations of one ionogram, not something to merge.

---

## Commands

| Command | Purpose |
|---|---|
| `muf run TARGET` | extract MUF; writes `out/<date>.csv` |
| `muf plot TARGET` | render ionograms (`--no-axes` for bare rasters) |
| `muf daily TABLE` | interpolate onto a 5-minute grid and smooth |
| `muf track TABLE` | Kalman-track through time: fill gaps, reject outliers |
| `muf compare TABLE` | agreement between methods, `--ref` series and `--ref-model` |
| `muf export TARGET` | write soundings as SAO.XML 5.0 (URSI/INAG interchange) |
| `muf plot-sao TARGET` | draw exported SAO.XML records, over their ionograms |
| `muf lof TARGET` | measure the band floor and summarise LOF |
| `muf info TARGET` | header and derived geometry, no processing |

`run` · `--methods algo,kmeans,contour` (or `all`) · `--gate LO,HI` · `--window` ·
`--zero-periods` · `--min-run` · `--threshold-db` · `-k` · `--jobs 0` ·
`--cache-dir` · `--format parquet` · `--legacy-algo` · `--daily` · `--combined`
· `--plot`

`track` · `--process-noise` (MHz/hour the MUF may change) · `--gate-sigma`
(reject picks beyond this many sigma)

`compare` · `--ref FILE` (a historical CSV) · `--ref-model iri,giro,chapman`
(or `all`) · `--exclude START..STOP`

`export` · `--methods` (one record per method) · `--ursi-code` · `--station` ·
`--iri` (add IRI's MUF/foF2/hmF2 as `<Modeled>`) · `--offline` (cached solar
indices only)

`plot-sao` · `--ionogram TARGET` (draw each record over its own sounding,
matched by file name) · `--method` (one estimator's record) · `--trace` /
`--no-trace` (overlay the scaled points, or never) · `--full-band` · `--dpi` ·
the `run` gate and window flags, so the raster matches what was exported

`lof` · `--level` (detection level for the floor measurement, raw ionogram dB)
· `--out FILE` (per-sounding ladder as CSV)

`--band-floor MHZ` is shared by `run`, `plot`, `export` and `lof`: the lowest
frequency the transmitter actually radiates, which is not the sweep start.

`--cache-dir` stores the gated array per sounding, so re-running with different
estimator settings skips the FFTs entirely — the useful mode when tuning.

### Output

One row per sounding, 67 columns. Per method: `muf_`, `vrange_`, `ndet_`,
`run_`, `snr_`, `limited_`, plus `fit_`, `fitres_`, `fitex_` from the nose fit,
`nseg_`, `hops_`, `scatter_` from segmentation — `nseg_ > 1` means the trace
mixes more than one propagation mode — and `lof_`, `lofsnr_`, `loflim_` from the
low-frequency end. Shared columns carry the path geometry (`tx_lat`, `rx_lon`,
`path_km`), the sweep (`freq_start`, `freq_stop`), the gate, `sweep_complete` /
`sweep_fraction`, and the estimator-independent LOF ladder `lof43`, `lof50`,
`lof57`.

```console
$ muf run F:/MyData/ND/lfs/2026.02.04 --out out --jobs 0 --band-floor 8.0
wrote out/2026-02-04.csv  (288 soundings)
  algo      261/288 picked   9.44-32.05 MHz
  kmeans    250/288 picked   11.01-32.39 MHz
  contour   255/288 picked   10.68-32.11 MHz
```

```python
import pandas as pd

f = pd.read_csv("out/2026-02-04.csv")
print(f[["datetime", "lof43", "lof50", "lof57", "muf_contour"]].iloc[[0, 60, 110, 180]])
print("LOF at the band floor:", int(f["loflim_contour"].sum()))
```

```console
                datetime     lof43     lof50     lof57  muf_contour
0    2026-02-04 00:00:10   8.84144   9.43536  11.70864     12.67120
60   2026-02-04 05:00:10   9.21008   9.66064       NaN     21.60048
110  2026-02-04 09:10:10  17.83216  20.26928  28.37936     31.84048
180  2026-02-04 15:00:10  11.64720  16.72624       NaN     21.96912
LOF at the band floor: 98
```

The ladder reads as absorption: at 00:00 the three rungs sit within 3 MHz of
each other, at 09:10 they span 17.8 to 28.4 — a much steeper rolloff into a much
heavier D layer. A `NaN` on the top rung means no continuous run cleared 57 dB
anywhere, not that the sounding failed.

`limited_<method>` marks a value that is a **lower bound rather than a
measurement**: the pick reached the top of the sweep, so the MUF is at or above
the highest sounded frequency. `daily`, `track` and `compare` all exclude those.

`loflim_<method>` is its mirror at the other end — the LOF reached the band
floor, so the true value is *below* the lowest radiated frequency and the number
is an upper bound. It fires on 98 of 266 soundings here, which is why
`--band-floor` matters: without it the floor defaults to the sweep start and
none of them are flagged.

`sweep_complete = False` marks a recording that stopped before the sweep
finished. It is *not* an exclusion on its own, deliberately: the ceiling such a
recording imposes is compared against the frequency it actually reached, so a
truncated sounding whose MUF falls below that ceiling is a perfectly good
measurement, while one that runs into it is already caught by `limited_`.

`muf track` writes `datetime, method, muf, rate_mhz_per_hour, sigma, measured,
rejected`; `muf daily` writes `datetime, date, muf, muf_smooth, method`.

`muf export` writes one `<file stem>.xml` per sounding rather than a table —
the results table has room for a MUF per method but not for the trace behind
it, which is the whole point of the SAO.XML record.

---

## Layout

```
muf/                    the pipeline
  io_lfs.py             .lfs header and IQ
  calibrate.py          header -> frequency and virtual-range axes, range gate
  spectro.py            gated spectrogram, noise equalization, caching
  geometry.py           great-circle path, control point, secant law
  pick.py               the shared MUF decision rule
  extractors/           algorithmic.py, kmeans.py, contour.py, cnn.py
  fit.py                parabola fitted to the nose of the trace
  trace.py              mode segmentation and spline reconstruction
  track.py              Kalman filter + RTS smoother over time
  lof.py                the low-frequency end: LOF, the ladder, the band floor
  reference/            iri.py, giro.py, chapman.py, minimuf.py, indices.py
  export/               saoxml.py -- SAO.XML 5.0 write and read, for interchange
  pipeline.py           per-file and per-day driving, parallelism
  compare.py            agreement metrics and reports
  render.py, cli.py
tests/                  285 tests; `python -m pytest tests -q`
```

Tests that need real recordings find them via `MUF_TEST_DATA`, and skip when it
does not resolve:

```bash
set MUF_TEST_DATA=F:\MyData\ND\lfs\2026.02.04
python -m pytest tests -q
```

Correctness is established on **synthetic** soundings: IQ is built with an echo
at a chosen virtual range that stops at a chosen frequency, and each estimator
has to recover both. Real recordings have no ground truth, so they are used for
regression pinning and for physical plausibility instead.

The original scripts (`MUF.py`, `stuffr.py`, `muf_interpolation.py`,
`data_handler/`, …) are untouched and still present for reference. They need
`psycopg2` and a reachable PostgreSQL server; the `muf/` pipeline needs neither.

---

## Changes in this merge

The clustering work previously lived in a separate `MUF_clustering` folder and
ran on rendered PNG ionograms, recovering frequency from pixel position. It now
runs on the numeric array, and both halves live here.

**Corrections**

- **The virtual-range axis was inverted.** The echo in
  `cyprus1_20260204_000010.lfs` sits at fftshifted bin 3909. Under the
  ascending axis used at `MUF.py:116` that is **-2732 km**, which no echo can
  occupy; under `R - idx*step` it is **+2739 km**, correct for a 2,588 km path.
  The plot looked right only because it was reversed with `[::-1]` at draw
  time, and `MUF.py:297`'s `if vrng < 0: vrng = R + vrng` was patching over the
  sign error. The axis is now defined once, in `calibrate.py`.
- **`rx_longitude` was reading the latitude.** `lfs_header.py:108` seeks offset
  150, which is `rx_latitude`'s; the longitude is at 154. Yoshkar-Ola now reads
  47.53E rather than 56.38.
- **Three defects in the algorithmic estimator**, documented in
  `extractors/algorithmic.py`: an uninitialised buffer, an `np.append` whose
  result was discarded, and an `np.amax(axis=0)` that assembled its returned
  `(frequency, range, row)` triple from *different* detections — so the
  reported range was not the range at the reported MUF. `--legacy-algo`
  reproduces the old decision where that is possible.
- **K-means invented MUFs out of noise.** On a recording containing no echo it
  still reported a value at the top of the band, because the selection rule fell
  back to "keep the brightest cluster" when nothing passed the threshold. It now
  returns no pick.
- **Pixel-position calibration is gone.** The scripts assumed 1,220 columns and
  a 3500-2500 km height span; the renderer actually produced 2500-4000 km. Axes
  now come from the file header. (`ion_col_num = 1220` was right by accident —
  it is `len(iq) // 8192` for this instrument.)
- **Truncated recordings are no longer stretched.** A recording cut short still
  declares the full sweep in its header — 10 files in `2026.02.05` hold 347
  windows instead of 1,220 while claiming `dur=250`. Mapping the axis onto the
  nominal 32.5 MHz endpoint would place their last bin there when the
  transmitter had only reached 14.6 MHz, inflating MUF by up to 2.2x. The
  frequency axis is now derived from the chirp rate and elapsed time, which is
  what physically sets it; `sweep_complete` and `sweep_fraction` record the
  shortfall.

**Additions**

- Range gating from the header, applied inside the FFT loop — 40x less data,
  and the biggest single accuracy gain.
- The algorithmic estimator vectorised: a triple-nested Python loop over ~110M
  cells becomes a handful of array operations.
- One shared, tunable MUF decision rule with a continuity requirement, replacing
  three partial ad-hoc versions.
- Band-limited and truncated soundings detected and excluded from statistics.
- **Temporal tracking** (`muf track`) — Kalman filter with RTS smoothing, which
  fills gaps, rejects outliers and attaches an uncertainty to every point.
- **Trace fitting** (`muf/fit.py`) — an outlier detector that agrees 100% with
  `track`'s independent rejections, and repairs the values it flags.
- **Mode segmentation and reconstruction** (`muf/trace.py`) — splits a trace at
  propagation-mode boundaries and fits a weighted smoothing spline to one mode,
  turning a sparse scattered point set into a continuous `h(f)`. Revealed that
  87% of soundings carry more than one propagation mode.
- **External references** (`muf/reference/`) — IRI, GIRO and a transparent
  solar-zenith model, with solar indices fetched and cached from SILSO and NOAA.
  This is what revealed the out-of-band problem above.
- **SAO.XML 5.0 export** (`muf export`) — the URSI/INAG interchange format GIRO
  publishes, so the segmented trace becomes an archivable product rather than a
  plot overlay. The MUF is emitted as a `<Custom>` characteristic, not URSI
  `MUF(3000)`, because UAG-23A §1.50 states those are different quantities; the
  band-limited flag becomes UAG-23A's qualifying letter `D`. `muf plot-sao`
  reads a record back and draws it — over its own ionogram when the `.lfs` is
  to hand, and from the XML alone when it is not, which is what distinguishes a
  published format from a private file with angle brackets. `--iri` adds the
  reference model's MUF, foF2 and hmF2 as `<Modeled>` characteristics, so the
  panel shows measured and modelled side by side without ever conflating them.
- **The low-frequency end** (`muf/lof.py`, `muf lof`) — LOF at the estimator's
  own trace and at a ladder of detection thresholds, with the threshold carried
  in every result. Correlates with the cosine of the solar zenith angle at
  r = +0.86, which is the D-region absorption signature. Found that the
  transmitter's real band floor is 8.0 MHz against a declared sweep start of
  7.5, so 102 of 266 LOF values are upper bounds and earn URSI's `E`.
- **Several days at once**, written one file per day; `daily`, `track` and
  `compare` accept multiple tables or a directory of them.
- `--jobs` parallelism and a gated-array cache.
- `pyproject.toml` (installable, gives a `muf` command), `requirements.txt`,
  and `data/`/`*.lfs` added to `.gitignore` — the recordings
  were untracked but *not* ignored, so a single `git add .` would have committed
  tens of gigabytes.

See `BACKLOG.md` for what is deliberately not done: the archive-format analysis,
the machine-learning options and the literature behind them, and the historical
database values that need re-deriving.

**Note on `--zero-periods`**

Zero-padding subdivides range bins without resolving anything further: the true
resolution is set by the bandwidth swept during one window (14.65 km here) and
equals the *unpadded* bin spacing. The old `-z 10` therefore cost 11x memory and
FFT time for finer sampling of the same information. The default is now 0, and
sub-bin precision comes from parabolic interpolation of the peak instead.
