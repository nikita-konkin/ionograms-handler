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

| Method    | How it finds the trace                                                                                                     | From                                          |
|-----------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `algo`    | three vertically adjacent above-threshold cells whose range-neighbours are also lit                                        | `stuffr.filter2_np_nb_MUF`, vectorised        |
| `kmeans`  | K-means over dB values; keep clusters whose centroid stands above the noise                                                | `MUF_clustering/ionogr_clustering_0.026.py`   |
| `contour` | dB threshold, morphological open/dilate, external contours, then intersected back with the cells that were above threshold | `MUF_clustering/segment_ionogram.py`          |
| `cnn`     | autoencoder denoises, then `contour` reads the result                                                                      | `MUF_clustering/myCNN_0.02.py` (experimental) |

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

**Dilation and** `cv2.FILLED` **invented detections.** Only 32–46% of the cells in
the old mask were ever above threshold; the rest were the dilation skirt and
the filled interior of a contour's outline. **8–18% of the runs** handed to
`trace.extract_points` contained no above-threshold cell at all, so their
reported group range was the power-weighted centroid of a *gap* — visibly
floating above the trace it claimed to describe. The retained-contour mask is
now intersected back with the cells that were actually above threshold.

Measured over every fourth sounding of 2026-02-04 (n=72):

|                                       | old       | fixed      |
|---------------------------------------|-----------|------------|
| soundings with a detection            | 64        | **66**     |
| trace points emitted                  | 8,728     | **21,099** |
| points on cells never above threshold | **11.2%** | **0%**     |

Coverage in the table above reads *lower* after the fix (90.3% → 88.5%) and
that is the fix working: the trace now reaches the top of the sweep where the
opening used to cut it short, so five more soundings are correctly classified
as **band-limited lower bounds** rather than reported as measurements. They
moved from `n_picked` to `n_band_limited`, not into the bin.

### Results on 2026-02-04

288 soundings, Cyprus -> Yoshkar-Ola, 7.5-32.5 MHz:

| Method    | Coverage | Band-limited | MUF range     |
|-----------|----------|--------------|---------------|
| `algo`    | 90.6%    | 1            | 9.4-32.1 MHz  |
| `contour` | 88.5%    | 11           | 10.7-32.1 MHz |
| `kmeans`  | 86.8%    | 13           | 11.0-32.4 MHz |

| Pair              | RMSE | MAE      | R²        |
|-------------------|------|----------|-----------|
| kmeans vs contour | 0.87 | **0.17** | **0.986** |
| algo vs contour   | 1.35 | 0.41     | 0.967     |
| algo vs kmeans    | 1.66 | 0.57     | 0.950     |

The diurnal curve behaves as it should: ~12 MHz at night, climbing through
sunrise to ~32 MHz around midday, falling back after sunset.

`contour` **changed sides when its mask was fixed** (see below). It used to
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

sweep declares          7.50 - 32.49 MHz
measured band floor     8.00 MHz   (DIFFERS)
  pass --band-floor 8.00 so a LOF down there is flagged as an upper bound rather than reported as a measurement
measured band ceiling  32.48 MHz   (matches)
  43 dB:  266/288 scaled   8.00-28.42 MHz  102 at the floor
  50 dB:  260/288 scaled   8.00-29.08 MHz  52 at the floor
  57 dB:  183/288 scaled   8.02-30.63 MHz  4 at the floor
```

(The floor and ladder figures are from the full-day archive; a 133-sounding
subset covering 00:00–11:00 UTC gives 8.01 and 32.48, so neither edge is an
artefact of which hours you feed it.)

**102 of 266** LOF values at 43 dB are therefore upper bounds, not measurements —
the mirror image of the band-limited MUF problem, and it earns the mirror-image
URSI qualifying letter **E** ("less than") to the `D` already used at the top of
the band. Worth checking against the transmitter schedule, and it means the
`<Sweep StartFrequency="7.5000">` published in SAO.XML is the nominal sweep
rather than the radiated band.

**The ceiling is the same measurement at the other end**, and on this circuit it
reports `matches` — 32.48 against a declared 32.49, so this path really does
reach the top of its sweep and the midday censoring here is the sweep's fault,
not the path's. That is not universal, which is the point of measuring it: DOB's
Cyprus circuit declares 24.825 MHz and returns nothing above **24.53**, so its
band edge sat above anything the receiver could see and `limited_` fired on 0 of
216 picks. Pass `--band-ceiling` when the two differ.

Note it is read off the end of the last qualifying *run*, not the highest lit
bin. The floor can use a bare threshold because the bottom of the band is
genuinely dead; the top never is — narrowband interferers reach the sweep stop —
and a single-bin rule returns 24.80 on that DOB archive, measuring the
interference instead of the circuit.

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

|                                            | raw           | tracked      |
|--------------------------------------------|---------------|--------------|
| largest jump between consecutive soundings | **12.06 MHz** | **0.91 MHz** |
| mean step                                  | 0.562 MHz     | 0.176 MHz    |

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
|------------------------------|------------|---------------|
| bias                         | −0.12 MHz  | **−0.05 MHz** |
| MAE                          | 0.74 MHz   | **0.37 MHz**  |
| median residual              | 0.31 MHz   | **0.18 MHz**  |

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

- **Not a reliability filter — a finding that no longer replicates.** The original measurement, against the pre-fix `thresh` mask, was that filtering on `fitres < 0.3` made agreement *worse* (MAE 0.302 against 0.245 unfiltered), because bad picks tend to have excellent residuals: the trace fits a clean parabola, the pick just landed on the wrong part of it. Against the fixed `contour` mask the same filter now *helps* — 0.251 MHz over 177 soundings against 0.410 unfiltered over 251. Both numbers are agreement between two estimators, not accuracy, and the reference moved underneath the test, so neither result establishes anything about `fitres_` yet. Treat this as open; see BACKLOG §13.
- **Does not recover band-limited MUF.** Of the soundings flagged `limited_`, none produce a usable extrapolation: when the trace runs to the top of the sweep the nose was never reached, so the fit correctly declines.

### Trace segmentation and reconstruction

An extractor returns the frequency bins where it found signal. That set is not a
curve: on this instrument it covers only **18-48%** of the frequency span it
reaches across, carries **12-37 km** of scatter, and — the part that matters —
is usually **more than one propagation mode stitched together**.

`muf/trace.py` measures the gaps and finds two clear classes:

| change in range across a gap | what it is                                          |
|------------------------------|-----------------------------------------------------|
| +15 to +37 km                | a fade inside one trace — safe to bridge            |
| **−130 to −180 km**          | **a mode boundary** — bridging it would be nonsense |

Virtual range *falling* as frequency rises is backwards for a single trace, so a
large drop marks where another mode takes over. A 06:00 sounding has a 5.5 MHz
gap with the range dropping 176 km across it. The extractors' continuity rule
runs straight through those, because it only looks along frequency.

**On 2026-02-04, 87% of soundings carry more than one propagation mode**, two
being the median. That is unsurprising for an oblique path — 1F2, 2F2, sporadic
E and others arrive together — but the extractors see none of it.

`nseg_` turns out to be a good proxy for how strong a sounding is, in the
direction opposite to intuition:

| modes resolved | n   | algo vs contour MAE |
|----------------|-----|---------------------|
| 1              | 77  | 0.281 MHz           |
| 2              | 117 | 0.633 MHz           |
| 3              | 46  | 0.132 MHz           |
| 4              | 9   | 0.112 MHz           |
| 5+             | 2   | **0.061 MHz**       |

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

| Reference | What it is                                                       | Needs                                     |
|-----------|------------------------------------------------------------------|-------------------------------------------|
| `giro`    | real ionosonde measurements, converted through the secant law    | network; a station near the control point |
| `iri`     | the International Reference Ionosphere                           | `pip install PyIRI`                       |
| `chapman` | transparent solar-zenith model; **shape only**, amplitude fitted | nothing                                   |
| `minimuf` | not implemented -- see `muf/reference/minimuf.py`                | verified coefficients                     |

For Cyprus -> Yoshkar-Ola the control point is 45.99N 39.09E and **RV149 Rostov**
sits 146 km away, well inside the F2 correlation scale.

**The control point is the circuit's, not the transmitter's.** It comes from
each sounding's own two ends, so the same Nicosia transmitter is modelled over
45.99N 39.09E when Yoshkar-Ola hears it and over 49.22N 24.59E when Dombås
does — 1100 km and, at 09:00 UTC in February, a different ionosphere. Above
4000 km a path takes more than one hop: it then has **two** control points,
2000 km in from each end (ITU-R P.533), is limited by the worse of them, and
the vertical-to-oblique factor applies per hop rather than to the whole
distance. That factor peaks at 3840 km and falls away after — beyond it the
whole-path number describes a reflection below the horizon, so an 8000 km path
converted whole reads a fifth low. Nothing in the archive is over 4000 km yet.

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
         ModelOptions="hmF2=300km,hop=2588km"/>
<Modeled Name="MUF"  Units="MHz" Val="9.904" ModelName="IRI"
         ModelOptions="control point 45.99N 39.09E, D=2588km"/>
<Modeled Name="foF2" Units="MHz" Val="3.355" ModelName="IRI" …/>
<Modeled Name="hmF2" Units="km"  Val="331.8" ModelName="IRI" …/>
```

Two entries are called `foF2` and neither is "the" one: the secant-law value is
this instrument's own MUF converted back to a vertical critical frequency under
a 300 km assumption, the IRI value is an independent model's. `ModelName` is
what tells them apart, so `Characteristic.model` carries it and
`record.characteristic("foF2", "IRI")` selects one. `record.muf` **always means** — a modelled value is never reachable as though the
instrument had produced it, and there is a test that says so.

The secant-law conversion is taken over **one hop**: `hop=` is the ground
distance of a single reflection, which is what sets the obliquity, and it
equals the whole path only while that path stays under `MAX_SINGLE_HOP_KM`.
Past it `ModelOptions` names both distances and the hop count —
`hmF2=300km,hop=2919km,D=5837km,2 hops` — because a bare `D=` cannot be read as
one or the other. Same convention as `iri.predict` and `/ui/series`, so the
download and the page report one foF2 for a sounding rather than two.

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

| characteristic | value              | letter |
|----------------|--------------------|--------|
| `MUF`          | 14.658 MHz         |        |
| `MUFNoseFit`   | 14.515 ± 0.119 MHz |        |
| `LOF`          | 8.022 MHz          | **E**  |
| `LOF@43dB`     | 8.022 MHz          | **E**  |
| `LOF@50dB`     | 8.022 MHz          | **E**  |
| `LOF@57dB`     | 9.292 MHz          |        |
| `MUF` (IRI)    | 10.126 MHz         |        |

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
      <ObliquePath TransmitterName="cyprus1" TransmitterLatitude="35.1856"
                   ReceiverName="yoshkar-ola" ReceiverLatitude="56.3800"
                   GreatCircleDistance="2587.8" Units="km"/>
      <Sweep StartFrequency="7.5000" StopFrequency="32.4856" FrequencyStep="0.020480"
             RangeGateLow="2329.0" RangeGateHigh="5000.0" Complete="true"/>
      <Acquisition Format="chirp2" ChirpRate="100000.0000" ChirpRateUnits="Hz/s"
                   SampleRate="40000.0" Decimation="1" Channel="ch0"
                   SweepStartEpoch="1785888234.055033" NoiseFloorMedian="3.590">
        <Recorder Software="0.2.0" Commit="0d2712553063" Dirty="true"/>
        <RangeReference Value="relative" Reason="transmitter is v2's 'unkown'
                        marker, so the 16499 km implied by t0 rests on a timing
                        solution nothing has cross-checked"/>
      </Acquisition>
    </SystemInfo>
    <CharacteristicList>
      <Custom Name="MUF" Units="MHz" Val="12.200"
              Description="Operational MUF … D=2588 km. Not URSI MUF(3000)…"/>
      <Custom Name="MUFGroupRange" Units="km" Val="2739.0"/>
      <Modeled Name="foF2" Units="MHz" Val="3.879" ModelName="secant-law"
               ModelOptions="hmF2=300km,hop=2588km"/>
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

**The MUF is a** `<Custom>`**, never URSI ID 03 or 07.** UAG-23A §1.50 says the
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
which **25 earn** `D` **and none earn** `U`. Both numbers deserve a caveat.

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

| Command               | Purpose                                                                     |
|-----------------------|-----------------------------------------------------------------------------|
| `muf run TARGET`      | extract MUF; writes `out/<date>.csv`                                        |
| `muf plot TARGET`     | render ionograms (`--no-axes` for bare rasters)                             |
| `muf daily TABLE`     | interpolate onto a 5-minute grid and smooth                                 |
| `muf track TABLE`     | Kalman-track through time: fill gaps, reject outliers                       |
| `muf compare TABLE`   | agreement between methods, `--ref` series and `--ref-model`                 |
| `muf export TARGET`   | write soundings as SAO.XML 5.0 (URSI/INAG interchange)                      |
| `muf plot-sao TARGET` | draw exported SAO.XML records, over their ionograms                         |
| `muf lof TARGET`      | measure the band floor and summarise LOF                                    |
| `muf info TARGET`     | header and derived geometry, no processing                                  |
| `muf detect TARGET`   | census a chirpsounder2 detection tree: which transmitters, on what schedule |
| `muf stations`        | print the station coordinate registry                                       |

### Which commands read which format

`run`, `plot`, `export`, `lof`, `plot-sao` and `info` read **three** formats:
`.lfs` recordings, chirpsounder2 `lfm_ionogram-*.h5` products, and
`digisonde_ionogram-*.h5` (`architecture.md` §3.2, `muf.loader`). A tree holding
all of them — which is what a DOB day looks like — works without saying
anything:

```bash
muf run /media/.../ionozond_data2/2026-08-05 --out out --jobs 4
```

```
wrote out/2026-08-05.csv  (318 soundings)
  algo      149/318 picked   9.80-24.45 MHz
  kmeans    198/318 picked   12.05-24.50 MHz
  contour   185/318 picked   6.90-24.50 MHz
```

**The extension does not decide, and cannot.** Two of the three are `.h5`, in
one directory, alongside three kinds of detection file. Dispatch is by name —
`digisonde_ionogram-` against `lfm_ionogram-` — which is the writer's own
prefix rather than a guess, and `io_digisonde.read_header` then confirms it
against the file's `type` dataset before reading anything.

`--input-format {lfs,chirp2,digisonde}` overrides that, for a file that was
renamed. It is separate from `run`'s `--format {csv,parquet}`, which is the
*output* table format.

#### Digisonde products are somebody else's sounder

`receive_digisonde.py` does not download these. It receives the transmissions
**off air with the station's own USRP**, decoding the complementary phase codes
a Digisonde transmits — so each one is an *oblique* reception of a **vertical**
sounder a few hundred kilometres away, and a free extra circuit with a named,
registered transmitter. At DOB there are four: Juliusruh (864 km), Ramfjordmoen
(951 km), Chilton (1325 km) and Dourbes (1360 km).

Three things differ from a chirp product, and `muf/io_digisonde.py` documents
each at the decision:

- `SNR` **is** `(2, n_freq, n_range)` — two polarizations. Physically these are O and X, but nothing in the product records which channel is which, so the reader never claims: they are channel 0 and 1, summed by default, exactly as upstream's own plot shows them. `io_digisonde.load(path, pol=0)` selects one.
- **NaN means "below threshold", not "missing".** `receive_digisonde.py:535` writes `SNR[SNR < snr_threshold] = nan`, so ~90% of a real array is NaN by construction. Those cells are read back as the noise level rather than propagated into estimators that would each have to special-case them.
- **The stored range axis starts at zero.** The absolute axis is that plus `offset_us × c` — 600 km at the usual setting — which is how upstream plots it. That offset is *configured*, not measured, so `DigisondeHeader.range_is_configured` says so: differences are right, and the zero is only as good as the ini. Same distinction as `ChirpHeader.range_is_relative`, reached from the other side.

The power scale is deliberately `io_chirp`'s. Both instruments define SNR as
`(P − median)/median`, so both go through `snr_to_power` and the 43 dB level
every estimator shares means the same thing in either. Measured on a real
Juliusruh→DOB sounding, the noise floor lands at **25.6 dB** — the same as a
chirp product.

**The default gate comes from the path, not the pulse timing.** A digisonde
product stores the whole unambiguous window — `c` times the inter-pulse
period, 2997 km at the usual 10 ms — which is a property of the transmitter's
timing and says nothing about this circuit. `calibrate.geometry_gate` asks the
geometry instead: the near edge is one hop at the lowest plausible mirror
height, the far edge `DEFAULT_MAX_HOPS` at the highest, both from
`trace.hop_range_km` so a gate and a hop label cannot disagree about what a
range means.

| path              | stored      | geometry gate | kept             |
|-------------------|-------------|---------------|------------------|
| Juliusruh, 864 km | 600–3597 km | 898–3222 km   | 775 of 1000 bins |
| Chilton, 1325 km  | 600–3597 km | 1317–3380 km  | 687 of 1000 bins |

On the 864 km path nothing can arrive before ~998 km, and the near third of
the stored window is where the interference sits. Gating it removed two
spurious bright patches and moved the picked MUF from 2.96 to 3.44 MHz, onto
the edge of the actual echo. `--gate` still overrides; without usable
coordinates the stored extent is kept rather than cropped on a guess.

**The range-consistency rule, and why these files yield nothing.** Across all
four stations the picks landed at 3.05–3.06 MHz — identical to 0.01 MHz on
paths of 864 to 1360 km, at latitudes from 51°N to 69°N, with no diurnal
movement over ten hours — while the pick *range* wandered randomly over
1700 km. Four circuits cannot share a MUF that precisely, and a real echo's
range does not jump. `min_run` asks whether neighbouring frequencies are lit,
which a crowded band satisfies by accident; `max_range_slope` asks whether
they agree about **where**, which it cannot.

The test splits a run wherever the brightest range jumps by more than
`DEFAULT_MAX_RANGE_SLOPE` (150 km/MHz) per frequency step, then re-applies
`min_run` so a long stretch of interference cannot survive as several short
ones. It is **off by default** and on only for digisonde, because switching it
on for `.lfs` would move every result already published.

It is not a novel test. ARTIST 5 — Galkin and Reinisch, UMLCAR, the authors of
the SAO.XML spec `muf export` writes — groups echoes by "the proximity and
**good continuation** principles of the Gestalt perception", having found that
tags alone are unreliable ("even polarization tags can be wrong"). Ding et al.
state the same criterion as "the continuity of the slope of the single layer
trace and rejection of impractical changes in slope when the ionogram is
traversed in the frequency axis".

Measured against a real trace, it is not aggressive:

|                               | raw run | after the range test |
|-------------------------------|---------|----------------------|
| `cyprus1_20260204_030010.lfs` | 39 bins | **22**               |
| `cyprus1_20260204_031010.lfs` | 31 bins | **28**               |
| Juliusruh digisonde           | 14 bins | **2**                |
| Ramfjordmoen digisonde        | 6 bins  | **1**                |

A genuine trace keeps 60–70% of its run, well above `min_run = 5`. The
digisonde runs collapse to 1–2 bins, and **all 334 soundings now yield no
pick** — which is the correct answer, not a failure: nothing in them is a
range-consistent trace above 43 dB. Heavy rejection at these latitudes is also
what the reference implementation reports. ARTIST 5 excludes "only ~5%" of
records at "mid-latitude, low interference observatories" but "up to 70% … at
polar stations during severe spread F conditions", and DOB sits at 62°N with
Ramfjordmoen at 69.6°N.

**The paired vertical foF2 settles it.** Each of these stations scales its own
*vertical* ionogram and publishes it, so `muf.reference.giro.history` fetches
what the transmitter measured directly overhead at the same instant, and
`geometry.fof2_to_muf` converts it to the oblique MUF this receiver should see:

| station   | path    | its own foF2, 00–11:30Z | implied oblique MUF |
|-----------|---------|-------------------------|---------------------|
| Juliusruh | 864 km  | 1.85 → 6.25 MHz         | **3.4 → 11.5 MHz**  |
| Dourbes   | 1360 km | 2.85 → 8.00 MHz         | **6.7 → 18.8 MHz**  |
| Tromsø    | 951 km  | 3.50 → 5.12 MHz         | **7.2 → 10.5 MHz**  |

So the ionosphere was varying strongly — the implied MUF roughly triples
through the morning — while every pick sat flat at 3.05 MHz. The picks were
never measuring it. (Chilton has no real-time feed; DIDBase carries it, the
mirror does not.)

**Lowering the detection threshold does not recover the trace, and was
measured rather than assumed.** On a 09:00Z Juliusruh sounding, where the
station's own foF2 implies a MUF near 10.3 MHz:

| threshold | lit bins | longest run | after the range test |
|-----------|----------|-------------|----------------------|
| 43 dB     | 18       | 8           | **0**                |
| 37 dB     | 34       | 8           | **0**                |
| 34 dB     | 57       | 10          | **0**                |
| 28 dB     | 172      | 34          | **0**                |

28 dB is barely above the 25.6 dB noise median, and still nothing
range-consistent appears. `--reject-interference` removes 5 rows and changes
nothing. The reason is visible in the array: at 2.96 MHz **350 lit range cells
span 1902 km** — a signal filling the whole range window, which is what an
unsynchronised emitter does after pulse compression. Loosening the range
tolerance to 60–250 km readmits exactly that blob, which is why it stays tight.

So the threshold is deliberately unchanged. It is already tunable per method
when there is reason to —
`Options(method_options={"contour": {"threshold_db": 34}})` — and on this data
there is not. What is missing is signal, not sensitivity: the Digisonde
radiates weakly at the low elevation angles an 864 km hop needs, and nothing
from it clears the noise on a range-consistent path.

`muf detect` is the exception: it reads chirpsounder2 **detection** files
(`par-*.h5`, `chirp-*.h5`, `cdetections-*.h5`) and nothing else. `daily`,
`track` and `compare` read result tables, not soundings.

Two flags change meaning across formats:

- `--window` **and** `--zero-periods` **are ignored for** `.h5`**, with a warning.** v2 fixed the FFT window when it wrote the product and the raw IQ is not in the file, so the sounding cannot be re-derived at another one (§3.4). They warn rather than silently no-op, because a run whose `--window` was quietly dropped produces a table indistinguishable from one where it was applied.
- `--cache-dir` **does nothing for** `.h5`**.** The cache exists to skip FFTs and there are none: reading a v2 product is 26 ms against seconds for `.lfs`, so an entry would cost 8 MB on disk to save 26 ms. Cache keys still carry a format tag so a `sounding.lfs` and a `sounding.h5` cannot overwrite each other.

### Shared processing flags

These four appear on `run`, `plot`, `export` and `lof`, and describe how the
ionogram is formed rather than how it is read. They are meaningless for `.h5`
products, where v2 fixed the window at archive time.

| Flag                    | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--window N`            | FFT window length, default 8192. Sets range resolution and, with `--zero-periods`, the range bin count.                                                                                                                                                                                                                                                                                                                |
| `--zero-periods N`      | Zero-padding periods. Subdivides range bins **without improving resolution** — it interpolates, it does not resolve.                                                                                                                                                                                                                                                                                                   |
| `--gate LO,HI`          | Virtual-range gate in km. Default comes from the file header's geometry. Narrowing it is the main lever on runtime.                                                                                                                                                                                                                                                                                                    |
| `--reject-interference` | Flatten frequency rows whose above-43 dB energy occupies more than 800 km of range. A burst has no delay and smears across every range bin; an echo is narrow in range and continuous in frequency. Off by default because it changes results — and measured over three archives it changed a MUF about once in twenty soundings, so treat it as a diagnostic first. See `muf/interference.py` for the measured yield. |
| `--gate auto`           | `plot` **only.** Fit the range window to where the echo actually is, per sounding. On DOB's search-mode products the stored axis is ±3998 km and the trace occupies a few hundred, so the default plot is a hairline in an empty field; this is a ~15× vertical zoom. Soundings with no range concentration are left at full extent and counted at the end, rather than cropped to an invented window.                 |
| `--cache-dir DIR`       | Cache the gated array per sounding, so re-running with different estimator settings skips the FFTs entirely. The useful mode when tuning.                                                                                                                                                                                                                                                                              |

`--stations FILE` is shared the same way. v2 products carry `txname` and
`station_name` as bare strings and **no coordinates**, so without a registry
`geometry.path_of` has nothing to work with. The built-in table is used by
default — 16 sites transcribed from chirpsounder2's own
`examples/marieluise/server.ini`, plus `yoshkar-ola` and `cyprus1`, which only
this pipeline knows. A file given here is merged *over* it and wins every
collision: a station is more authoritative about its own coordinates than a
table copied from someone else's example config. JSON, or point it straight at
a live `server.ini`.

`--band-floor MHZ` is likewise shared by `run`, `plot`, `export` and `lof`: the
lowest frequency the transmitter actually radiates, which is **not** the sweep
start, and without which a LOF that ran off the bottom of the band is
indistinguishable from a real one.

`--band-ceiling MHZ` is its counterpart at the top, shared by the same commands:
the highest frequency the circuit actually returns echoes at, which is **not**
the sweep stop wherever the path gives out first. It is what the `limited_`
columns and the SAO `D` letter are measured against, and the run records the
value it used in a `band_ceiling` column — two runs over the same files with
different ceilings produce identical MUF values and different censoring, so the
column is the only thing that says which one you are holding. `muf lof` measures
both edges and prints the flag to pass.

**Usually you should not need the flag.** The ceiling lives in the station
registry, keyed by receiver, and `NIC` already carries 24.53 MHz into `DOB`, so
a DOB archive is censored correctly with no arguments. Resolution order is
flag, then registry, then the sweep stop — the flag wins because it is what an
operator types while working out a circuit that is not registered yet, and a
stale entry must not override that. In a JSON registry:

```json
{"NIC": {"lat": 35.18557, "lon": 33.38228,
         "band_ceiling_mhz": {"DOB": 24.53}}}
```

Keyed by receiver because a ceiling belongs to a **circuit**, not a
transmitter: it is where this path's signal drops below the detection level,
which depends on the receiving antenna, its noise environment, the path length
and the sweep that receiver runs. Nicosia measures 24.53 into Dombås and 32.48
into Yoshkar-Ola, where it matches the declared stop. A bare number instead of
a mapping is rejected rather than applied to every receiver — it reads like the
obvious spelling, and it would silently censor picks on circuits nobody has
measured.

### Per-command flags

`run` — extract MUF and LOF; writes `out/<date>.csv`

| Flag                     | Meaning                                                                                                                                                                         |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--out DIR`              | Output directory, default `out`.                                                                                                                                                |
| `--methods LIST`         | `algo,kmeans,contour`, or `all`. Comma-separated.                                                                                                                               |
| `--min-run N`            | Consecutive frequency bins required before a pick is believed. The single most effective guard against calling a carrier a trace — an echo spans frequencies, RFI spans ranges. |
| `--percentile P`         | Percentile used by the percentile-based estimators.                                                                                                                             |
| `--threshold-db DB`      | Detection threshold for `contour`, in the shared 43 dB convention.                                                                                                              |
| `-k N`                   | Cluster count for `kmeans`.                                                                                                                                                     |
| `--legacy-algo`          | Reproduce the pre-fix `algo` decision rule, for comparison against old results only.                                                                                            |
| `--jobs N`               | Parallel workers; `0` uses all but one core.                                                                                                                                    |
| `--format {csv,parquet}` | Output table format.                                                                                                                                                            |
| `--plot`                 | Also render each ionogram.                                                                                                                                                      |
| `--daily`                | Also write the interpolated daily curve.                                                                                                                                        |
| `--combined`             | Additionally write one table spanning every day, on top of the per-day files.                                                                                                   |
| `--quiet`                | Suppress the progress bar.                                                                                                                                                      |

`plot` — render ionograms

| Flag               | Meaning                                                                                  |
|--------------------|------------------------------------------------------------------------------------------|
| `--out DIR`        | Output directory.                                                                        |
| `--methods LIST`   | Which estimators' picks to mark.                                                         |
| `--no-axes`        | Bare raster, no axes or annotation.                                                      |
| `--no-muf`         | Do not run the estimators or mark their picks.                                           |
| `--trace [METHOD]` | Overlay the detected trace, split by propagation mode. Takes an optional estimator name. |
| `--dpi N`          | Output resolution.                                                                       |

`daily` — interpolate onto a 5-minute grid and smooth

| Flag            | Meaning                                     |
|-----------------|---------------------------------------------|
| `--method NAME` | Default: every method present in the table. |
| `--out DIR`     | Output directory.                           |
| `--no-smooth`   | Interpolate but do not smooth.              |
| `--plot`        | Also draw the curve.                        |

`track` — Kalman-track through time: fill gaps, reject outliers

| Flag                           | Meaning                                                                                                            |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `--method NAME`                | Default: every method present.                                                                                     |
| `--out DIR`                    | Output directory.                                                                                                  |
| `--process-noise MHZ_PER_HOUR` | How fast the MUF is allowed to change. Too small and the filter lags a sunrise; too large and it follows outliers. |
| `--gate-sigma N`               | Reject picks further than this many sigma from the prediction.                                                     |
| `--plot`                       | Also draw the tracked series.                                                                                      |

`compare` — agreement between methods, and against references

| Flag                    | Meaning                                                          |
|-------------------------|------------------------------------------------------------------|
| `--ref FILE`            | A historical CSV, e.g. `MUF_cyprus1_20220320.csv`.               |
| `--ref-model NAMES`     | External models to evaluate: `iri`, `giro`, `chapman`, or `all`. |
| `--exclude START..STOP` | Exclude a time span, for a known outage.                         |
| `--out DIR`             | Output directory.                                                |

`export` — write soundings as SAO.XML 5.0 (URSI/INAG interchange)

| Flag               | Meaning                                                                     |
|--------------------|-----------------------------------------------------------------------------|
| `--out DIR`        | Output directory.                                                           |
| `--methods LIST`   | One `<SAORecord>` per method, per the spec's separate-storage rule (1.3.4). |
| `--ursi-code CODE` | URSI station code, if one has been issued for this path.                    |
| `--station NAME`   | `StationName` attribute; default is the receiver name.                      |
| `--iri`            | Add IRI's MUF, foF2 and hmF2 as `<Modeled>` beside the measured values.     |
| `--offline`        | With `--iri`, use cached solar indices only — no network.                   |

`plot-sao` — draw exported SAO.XML records

| Flag                     | Meaning                                                       |
|--------------------------|---------------------------------------------------------------|
| `--out DIR`              | Output directory.                                             |
| `--ionogram TARGET...`   | Draw each record over its own sounding, matched by file name. |
| `--method NAME`          | Draw only the record from this estimator.                     |
| `--full-band`            | Show the whole sweep instead of framing the trace.            |
| `--trace` / `--no-trace` | Overlay the scaled trace points, or never.                    |
| `--dpi N`                | Output resolution.                                            |

Takes the shared gate and window flags too, so the raster matches what was
exported rather than a differently-gated version of the same sounding.

`lof` — measure the band floor and summarise LOF

| Flag         | Meaning                                                        |
|--------------|----------------------------------------------------------------|
| `--level DB` | Detection level for the floor measurement, in raw ionogram dB. |
| `--out FILE` | Write the per-sounding ladder to this CSV.                     |

`info` — header and derived geometry, no processing

| Flag                             | Meaning                                                              |
|----------------------------------|----------------------------------------------------------------------|
| `--limit N`                      | How many soundings to describe, default 3.                           |
| `--window N`, `--zero-periods N` | Only affect the derived axes that are printed; nothing is processed. |

`stations` — print the station coordinate registry

| Flag              | Meaning                                                                        |
|-------------------|--------------------------------------------------------------------------------|
| `--stations FILE` | Merge a JSON registry or a chirpsounder2 `server.ini` over the built-in table. |

```
$ muf stations
17 stations
          code   latitude   longitude  name [source]
...
NIC              35.18557    33.38228  Nicosia, Cyprus [chirpsounder2 server.ini]
               note: also the .lfs archive's 'cyprus1'; its header says 35.0/34.0, 59.9 km away, superseded by these five decimals
yoshkar-ola      56.38000    47.53000  Yoshkar-Ola [.lfs header]
               note: the .lfs receiver; absent from v2's table
```

Every entry carries its provenance, because a wrong coordinate does not raise —
it yields a plausible path length and a wrong virtual height. Nothing in the
table is inferred from measurements: a range measured through the ionosphere
carries a virtual-height excess of a few percent and, on an uncalibrated
receiver, an epoch error of any size at all.

`cyprus1` **is an alias of** `NIC`, resolving to v2's five-decimal position.
That is one site under two names — the `.lfs` archive's and v2's — and it is
the only entry whose position was *chosen* rather than copied. The data could
not decide it: after removing the DOB receiver's 0.9557 s epoch error the four
cyprus1 slots gave 3398, 3420, 3422 and 3504 km of virtual range, a 106 km
spread against the 40 km separating the candidates. The round `35.0/34.0` from
the `.lfs` header was rejected in favour of `35.18557/33.38228` on the shape of
the numbers, and kept in the module as `stations.CYPRUS1_LFS_COORDINATES`
rather than deleted, because it is still embedded in every `.lfs` file.

What that choice moves:

```
cyprus1 -> yoshkar-ola   2588.4 -> 2587.8 km    (-0.6 km)
cyprus1 -> DOB           3476.3 -> 3435.9 km   (-40.3 km, 20 range bins)
```

`.lfs` **soundings follow the table too**, as of 2026-08-14. An `.lfs` header
carries its own lat/lon, so there the registry is a *correction* rather than a
lookup — without one the file's numbers are used verbatim, and `stations={}`
still means no table — but the correction is applied, and the path
`signal-chain.md` records went from 2588.4 km to 2587.8 km with it. The
alternative was one Nicosia transmitter existing as two objects to anything
keying on `(tx, rx)`, which the 0.6 km was never the point of.

Which end a coordinate came from is answerable after the fact:
`LfsHeader.from_registry` names the corrected ends, `()` when the file was read
as written. It says `("tx", "rx")` for the archive's soundings even though
`yoshkar-ola` does not move — that entry was transcribed into the table *from*
an `.lfs` header, and the provenance of the value in play is still the table.

A database ingested before the change keeps the old numbers in
`sounding.tx_lat/tx_lon/path_km`: those columns are written at ingest, not read
from the file per request. Re-running `ingest` over the same archive corrects
them in place — it upserts on `file`. The cached spectrogram tiles need no such
treatment; they hold the tile, not the geometry, and the header is re-read on
every load.

v2's marker for an unidentified transmitter — `unkown`, upstream's spelling —
never resolves. Every unidentified emitter in an archive shares that string, so
a match would give a whole night of distinct transmitters one position.

`detect` — census a chirpsounder2 detection tree

Reads `par-*.h5` timing solutions (or `chirp-*.h5` with `--raw`) and groups
them into transmitters by chirp rate and arrival phase.

| Flag                         | Meaning                                                                                                                                                                                      |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--cycle SECONDS`            | Schedule cycle, default 300. Matches the Twente chirp list's `300:235` notation — period 300 s, starts at second 235.                                                                        |
| `--min-count N`              | How many sightings make an emitter rather than a false alarm, default 3. A search-mode tree is mostly noise, and a one-off detection carries no schedule. `--min-count 1` shows them anyway. |
| `--raw`                      | Census `chirp-*.h5` detections instead of `par-*.h5`. Noisier, and the only option before `find_timings` has run.                                                                            |
| `--reference-km KM`          | Great-circle distance to a transmitter you can identify independently. **Supplying it is what enables transmit seconds and ranges to be printed at all.**                                    |
| `--reference-slots LIST`     | Its published transmit seconds, comma separated. Default `235,240,245` — cyprus1 per the Twente list.                                                                                        |
| `--reference-rate HZ_PER_S`  | Its chirp rate, default `100e3`.                                                                                                                                                             |
| `--reference-name NAME`      | Label for the report.                                                                                                                                                                        |
| `--reference-window SECONDS` | How far from a published slot a sighting may land and still count, default 1.5. Widen past 1 s only when the epoch error is known to exceed it.                                              |

Without the `--reference-*` flags, `detect` prints only what the files say —
seconds **as received** and arrival phase — and says so. That reticence is the
point, and it is explained under [Why `detect` will not print a range](#why-detect-will-not-print-a-range).

### Why `detect` will not print a range

Every range in the chirp world comes from one identity: transmitters start
their sweep on a whole second, so whatever is left over in `chirp_time` is the
travel time. That holds only if the *receiver* agrees about where the second
begins.

At Dombås on 2026-08-05 it did not. The recorder's epoch was **0.956 s slow**,
and cyprus1 — 3436 km away — reported at 16,700 km. Nothing in the files showed
it: `chirp_time` was stable to 0.5 ms across eleven hours, the schedule was
self-consistent, the ionograms plotted. It took an external schedule (the
[Twente chirp list](http://websdr.ewi.utwente.nl:8901/chirps/), which publishes
cyprus1's transmit seconds) to see it at all.

So the default output names nothing it cannot justify:

```bash
muf detect /media/.../ionozond_data2/2026-08-05
```

```
  280 timing solution(s), 3 repeating emitter(s) on a 300 s cycle
       rate                received at     n  phase ms   sd ms      snr  span h
       100k   53,234,239,244,290,29...   254     56.05    0.60     25.6    10.3
       100k                        119    20     77.54    0.22     26.4     6.4
       125k                     74,194     6     92.28    0.18     19.7     6.6
```

Give it a transmitter whose distance and published schedule you know, and it
solves the receiver's clock first:

```bash
muf detect /media/.../ionozond_data2/2026-08-05 \
  --reference-km 3436 --reference-slots 235,240,245,300 \
  --reference-rate 100e3 --reference-name cyprus1 --reference-window 2.0
```

```
  epoch offset -0.95546 s from cyprus1 (4 slots, 249 samples, +/-0.08 ms = 25 km)
  NOTE: past half a second, so every received second above is a whole second early or late.
       rate             transmitted at     n   range km
       100k   0,54,235,240,245,291,292   254       3450
       100k                        120    20       9894
       125k                     75,195     6      14312
```

Note what changed: the received seconds `234,239,244,299` became the
transmitted seconds `235,240,245,0`, matching the Twente listing exactly, and
cyprus1 moved from 16,700 km to 3450 km against a true 3436. **An epoch error
past half a second shifts the whole schedule by a whole second while leaving it
perfectly self-consistent**, which is why the correction cannot be derived from
inside the archive and why `--reference-window` must be widened to admit it.

`--reference-window 2.0` is needed only because this receiver was that far out.
On a healthy one, leave it at the default.

### Plotting v2 products

Same command as for `.lfs`:

```bash
muf plot /media/.../ionozond_data2/2026-08-05 --out png
```

Everything downstream of `Ionogram` was already format-blind — the extractors,
`pick`, `render` and `export` all take the gated array and its axes — so
`muf.loader` picking the right reader is the whole of the change. `io_chirp`
puts the dB scale on the same 25.571 dB noise floor as the `.lfs` path, so the
shared 43 dB threshold keeps its meaning without recalibration, and the
frequency axis, range axis and gate come from the file's own datasets rather
than from header arithmetic.

From Python, if you want the array rather than a PNG:

```python
from muf import loader, extractors, render
ion = loader.load("lfm_ionogram-unkown-DOB-ch0-000-1785905639.06.h5")
render.plot(ion, "out.png", extractors.run(ion))
```

**Search-mode products keep a relative range axis.** Loading one warns and
`ChirpHeader.range_is_relative` is True, for the reason in the previous
section: the zero is unknown, the differences are not. MUF is a frequency and
is unaffected — the DOB archive gives 149–198 picks out of 318 depending on
estimator, spanning 6.9–24.5 MHz. Virtual height is not recoverable until the
receiver's epoch is calibrated, and `muf info` says so per file:

```
$ muf info lfm_ionogram-unkown-DOB-ch0-000-1785905639.06.h5
  unkown->DOB (oblique) 2026-08-05 04:53:59Z
  RANGE IS RELATIVE -- transmitter is v2's 'unkown' marker, so the 17003 km
                       implied by t0 rests on a timing solution nothing has cross-checked
  sweep      0.52-24.83 MHz, 486 bins @ 50.00 kHz
  range axis +/-59958 km, 2.00 km/bin, resolution 2.00 km
  gate       -3998-3998 km -> bins 27980..31978 (3999 of 59958, 15x reduction)
```

That `3999 of 59958` is v2's own gating, not `muf`'s: at this rate and sample
rate the full stretch-processed axis spans ±59958 km, and v2 stored the
±4000 km window around the expected echo. A `--gate` narrower than what v2 kept
is applied on top; a wider one warns and is ignored, because the rest was
discarded at acquisition and cannot be recovered.

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

`muf track` writes `datetime, method, muf, rate_mhz_per_hour, sigma, measured, rejected`; `muf daily` writes `datetime, date, muf, muf_smooth, method`.

`muf export` writes one `<file stem>.xml` per sounding rather than a table —
the results table has room for a MUF per method but not for the trace behind
it, which is the whole point of the SAO.XML record.

---

## Layout

```
muf/                    the pipeline
  io_lfs.py             .lfs header and IQ
  io_chirp.py           chirpsounder2 lfm_ionogram-*.h5
  io_digisonde.py       digisonde_ionogram-*.h5 -- another station's sounder
  loader.py             format dispatch across the three
  paths.py              the finders' shared dedupe -- no per-file syscalls
  calibrate.py          header -> frequency and virtual-range axes, range gate
  spectro.py            gated spectrogram, noise equalization, caching
  geometry.py           great-circle path, control points, hops, secant law
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
  interference.py       rejecting burst rows that cannot be echoes
  render.py, cli.py
services/
  agent/                runs ON the station: health push, control, logs
    health.py           the metrics; unknown is never zero, see below
    control.py          systemctl verbs and ini edits, allow-listed
    systemd/            unit files -- the migration off dombas.sh
  api/                  runs on the server: health ingest, read API, web UI
    watch.py            incremental ingest on a timer
    sources.py          emitters heard -> candidates for a schedule
    acquisition.py      which slot is being sounded, and when the next is due
    sao.py              one scaling per sounding: XML, panel, interactive plot
    net.py              can this host still reach the solar index servers?
    static/             plotly.min.js, vendored -- the one asset, no build step
patches/                diffs against the pinned chirpsounder2 clone
deploy/                 Docker Compose test rig -- see deploy/README.md
tests/                  854 tests; `python -m pytest tests -q`
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
`data_handler/`, …) were removed from the tree in `f94f561`; nothing imported
them, and they needed `psycopg2` and a reachable PostgreSQL server that the
`muf/` pipeline needs neither of. They remain readable in git history, which is
where every citation of them below resolves:

```bash
git show f94f561^:MUF.py
```

---

## Running the server

A temporary Docker rig brings up the api, the web console and a simulated
station. Full instructions, including how to point the real acquisition laptop
at it, are in [`deploy/README.md`](deploy/README.md).

```bash
cp deploy/.env.example deploy/.env          # set CONTROL_TOKEN
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up --build
```

Then <http://127.0.0.1:8000/ui>, or whatever `PORT` you set — the shipped
`.env.example` uses **8002**, because 8000 is already taken on the work server.
The console is empty until an archive is
ingested:

```bash
python -m services.api.ingest F:/MyData/ND/lfs/ionozond_data2/2026-08-05 \
    --db data/ionograms.sqlite3 --archive-root F:/MyData/ND/lfs \
    --methods algo,kmeans,contour
```

Two things about it worth knowing before you rely on it. `CONTROL_TOKEN`
**unset disables control rather than opening it** — a missing secret must never
be the same as a granted one, since these endpoints can stop a radio. And the
service **binds to** `127.0.0.1` **by default**; reaching it from the sounding
laptop means naming a LAN address on purpose, or tunnelling, which
`deploy/README.md` covers.

The console's **start / stop / restart buttons queue a command; they do not
execute one.** The press writes a row, the station's agent collects it on its
next pull, and until then the row reads `pending` — so on a server whose agent
is not running, a stop stays pending for good and the recorder keeps recording.
The page says as much when it queues one. `stop` asks twice, in the page rather
than in a browser dialog, and the 15 s auto-refresh stands down while it waits.
It stands down for a half-composed schedule too — the **sounding plan** panel
sits on the same page, so who to sound and whether to start it are chosen in
one place, and a refresh mid-choice would put every tick back the way the
server has it.

It is deliberately throwaway: SQLite, plain HTTP, no migrations. See
`architecture.md` §6 M2.5 for what it settled and what M3/M4 should not
inherit from it.

### On the acquisition laptop

The agent is stdlib-only and Python 3.7-clean, so the sounder's own virtualenv
runs it. Two files, both outside the checkout so `git pull` cannot touch them
and `git add -A` cannot publish them:

|                            |                                                                                     |
|----------------------------|-------------------------------------------------------------------------------------|
| `~/agent.json`             | station name, `server_url`, paths, `units` — from `deploy/station-dob.json.example` |
| `/etc/default/chirp-agent` | one line, `AGENT_TOKEN=<the server's CONTROL_TOKEN>`, root-owned `0600`             |

`chirp-agent.service` is deliberately **not** `PartOf=chirp.target`: it has to
survive a stop of acquisition, or "stop sounding" would be the last command the
server could ever issue. It also names the venv interpreter explicitly —
`/usr/bin/python3` on that laptop is 3.5, and under `Restart=always` a wrong
interpreter repeats a SyntaxError every 30 s forever. `services/agent/__init__`
is 3.5-syntax on purpose so it can say so instead.

**A station whose acquisition is run by a script, not systemd, sets** `units` **to**`[]` **and** `target` **to** `""`**.** `systemctl is-active` cannot see another
supervisor's children, so listing units there is eleven permanent red lights;
and `systemctl restart chirp.target` would not restart that script's recorder,
it would start a *second* one against the same USRP. With no target the agent
refuses the verb and says why. Both go back to normal after the migration in
`services/agent/systemd/`.

### What health measures, and what it refuses to claim

Every collector returns its own failure as a value, and a metric that cannot be
measured is `None`, never zero — "no soundings in the last hour" and "could not
tell" need different responses.

- `newest_product_age_s` **reads the sounding's** `t0` **from the filename**, not the file's mtime. mtime belongs to whichever clock touched the file last, and on one station that has been wrong three ways: an RTC that booted at 2021-04-02, a CIFS server running 5 h 36 m fast, and the stamps that outlived the fix. `t0` is on the recorder's GPS-disciplined epoch. It is also what the question actually means — when did we last hear the ionosphere, not when did a file appear — and it avoids `stat()`ing a large archive over a network share. The threshold covers pipeline latency, which is ~960 s on DOB.
- **A file newer than the clock is unknown, not fresh.** `age < threshold` is trivially true for every negative number, so the one metric watching for acquisition stopping once passed unconditionally at −20420 s and would have gone on passing with the recorder dead.
- **"Files newer than me" only convicts *my* clock if my clock wrote them.** On a network archive `system_clock_s` reports the NTP state instead and carries the skew as a note — it once accused a host sitting 47 ms from its NTP server, and worse, returned before the NTP check ran at all.
- **The digisonde receivers are watched but cannot fail the station.** Each `chirp-digisonde@…` instance is an oblique reception of a *remote vertical* sounder — an extra circuit, and one this station is normally better off without: they are ringbuffer consumers at 25 MS/s and the cause of the 45 % sample loss, which is why none should be enabled here at all. Stopped is therefore the *correct* state, and painting four permanent reds for it is how a status column stops being read. Their state is still reported, as `?` rather than `FAIL`, because "Dourbes has been down since Tuesday" is worth reading and "not running" and "wrong" are different claims. `optional_units` in `~/agent.json` is the list, matched as a substring of the unit name; set it to `[]` on a receiver that really does depend on one. The exemption is a safety net for a station whose unit list has not been cleaned up yet — it makes an enabled receiver quiet, not cheap.
- `epoch_offset_s` **needs** `par-*.h5`, which only search mode produces. It is the only external check on the recorder's clock — it caught a 0.956 s offset that displaced every echo by 286,000 km while every product stayed self-consistent — so consider leaving `find_timings.py` running even in scheduled mode, and keep `chirp-timings.service` in `units` to match.

### The picture the station sends of itself

Every metric above answers *whether* the station is producing. None of them
answers **what is in the products**, and a station can be green on all of them
while the pictures are pure interference. So the agent also encodes a 128×96
thumbnail of the newest product **from each transmitter it is hearing** and
pushes it with the health report; the console shows the row in the station
panel, above the arrivals table.

It has to come from the station because nothing else can. `arrivals` measures
the **archive**, which reaches the server only on `chirp-archive-sync`'s timer
and is routinely hours behind — the note under that table has always said so.
This is the one thing in the console that is current.

**It is affordable because a v2** `.h5` **holds the ionogram already computed.**
There is no FFT here, only a read, a decimation and a PNG: measured at 2.5 ms
for an ordinary DOB product and 18 ms for a search-mode one, once per 60 s
push, against a full server-side render's ~200 ms of matplotlib. At most four
products are decoded per pass and an unchanged one is skipped entirely, so an
idle circuit costs nothing at all.

Four decisions in `services/agent/preview.py` are not free choices:

- **The range axis is reversed.** v2 stores it ascending, this pipeline uses descending virtual range, and `render` puts the largest range at the top. Skip the reversal and the thumbnail is upside down and entirely plausible.
- **Decimation keeps the maximum of each block**, not the mean and not a stride. A trace is one bright cell among noise; at the ~30× reduction a search-mode product needs, averaging dilutes it below the noise floor and point-sampling misses it. This is the difference between a picture with a trace in it and a picture of noise.
- **The dB scale is the renderer's**, 20–75 dB through the same `jet`, so brightness means one thing across both. NaN — v2's "below the storage threshold" — lands at 25.6 dB, well under the 43 dB a detection needs, so the sparsification cannot invent a trace.
- **The PNG is written with** `zlib` **and** `struct`, about twenty lines, because the acquisition laptop has no Pillow and adding a dependency there to make a 3 KB picture would be a poor trade. Four bits per pixel and a 16-entry palette: the colours of the full render for the size of greyscale.

Server-side it is a separate endpoint and a separate table, **not** a field in
the health document. That document is stored verbatim forever, written 1440×
per station per day, and read back in full twice per station on every 15 s
console refresh — putting images in it would be tens of megabytes a day of
permanent storage, re-read constantly by code that throws it away.
`station_preview` instead holds one row per `(station, transmitter)`,
overwritten in place and swept after seven days, so it is bounded by the number
of circuits rather than by time. The `<img>` URL carries the sounding's `t0`:
the console reloads whole every 15 s, so with a plain URL the picture would
freeze behind `max-age` and without one it would re-transfer four times a
minute.

Needs numpy and h5py, which the station has because chirpsounder2 does. Their
absence degrades to *no preview* and says so — never to a failed health pass.
Set `"preview": false` in `~/agent.json` to send none.

### From search mode to a schedule

`/ui/sources` is the join between the two sounding modes: search records
whatever sweeps past and infers who was transmitting, and each row is a
candidate for the `sounder_timings` list that would put the station on it.

It takes **two steps, not one**, and the reason is that a detection is
anonymous. `calc_ionograms.py:444` reads five keys off every entry with a bare
subscript and no default — `chirp-rate`, `rep`, `chirpt`, `id` and
`transmit_name` — and a census can supply the first three. Nothing in a
detection says who sent it. So a row is **identified** first and scheduled
after:

1. **Identify**, on `/ui/sources`. Pick a census row, give it a code, and it is saved as a verified transmitter for that receiver, with the census row it was read off kept as evidence. This is the same judgement that resolved `cyprus1` to `NIC`, and it is written down instead of being remembered.
2. **Schedule**, on `/ui`. Tick verified transmitters and apply. The page composes the `sounder_timings` list **by name**, never by row number, and posts it as one `set_config` command with the mode.

The two steps are on two pages on purpose. Identifying is archive work and
belongs where the census is; choosing who to sound and pressing start are one
decision, so the chooser sits on the console beside the start button rather
than a page away. The console's list comes from the database — the identify
step's output — so it carries the chooser without inheriting the archive read.
A mode the server has never seen acknowledged pre-selects **nothing**: the box
reads `— not recorded —` and apply refuses it, because defaulting it to the
first option puts an unobserved mode one click from a live receiver.

The name is load-bearing rather than cosmetic. `calc_ionograms.py:344` writes
it into the product's *file name* and into `txname`, which this pipeline reads
back as the transmitter's identity and uses to look up its coordinates and its
band ceiling. **No dash**: `io_chirp.py:188` parses that file name
dash-delimited, so a dash inside the code does not fail to parse, it shifts
every field after it.

An identification belongs to a **circuit, not to a transmitter**. A slot second
is a reception second — transmit second plus one-way travel plus this
receiver's epoch offset — so the same transmitter heard at two receivers has
two different `chirpt` values. Verified transmitters are keyed by receiver, for
the same reason `band_ceiling_mhz` is.

Each transmitter gets its own MPI rank group, which is upstream's own
arrangement, and the page states the `-np` the launcher then needs. The agent
refuses a schedule whose rank count does not match what patch 0009 derives.

`/ui` answers the other half — whether it is working *this minute*. Per
station: the mode, which slot is being sounded right now, when the next one
starts, and the last few products with their age. A sweep is timed from the
band the station's recent products actually cover and the entry's chirp rate,
so it is measured rather than configured; with nothing ingested the slot times
are still exact and no slot is claimed to be in progress. The panel also flags
**rank oversubscription** — two slots of one rank whose sweeps overlap. A rank
is one process, so it takes the nearer slot and the other is skipped that cycle
with nothing in the log to say so.

That schedule is arithmetic on the ini, and **it stays true with the recorder
dead** — so the panel leads with a separate indicator for whether anything is
actually being recorded, and refuses to call a slot a sounding unless it is:

| pill            | means                                                            |
|-----------------|------------------------------------------------------------------|
| `ACQUIRING`     | products are arriving, or the supervisor says the recorder is up |
| `NOT ACQUIRING` | a process acquisition needs is definitely down                   |
| `NO PRODUCTS`   | nothing is being produced while nothing reports itself dead      |
| `ACQUIRING?`    | no report current enough to answer with                          |

`newest_product_age_s` leads, because it measures the acquisition rather than
the supervisor and because DOB reports **no unit states at all** — `dombas.sh`
supervises it, not systemd, so a unit-based indicator would read unknown
forever on the one station being watched. A definitely-dead unit still outranks
it: that is a fact about now, while a product age inside its threshold can be
fifteen minutes old. Only `chirp-rx` and `chirp-ionograms` count — a failed
`chirp-sync` upload is a red row in the metrics table, not a stopped station.
`NO PRODUCTS` is its own state rather than a shade of red because it is what
the real outage looked like: every unit green for two days with `/dev/shm` at
100%, the ringbuffer never trimmed and the recording full of holes. A stale
report is `ACQUIRING?`, never `NOT ACQUIRING` — silence is the alert, but it is
the absence of evidence, and the two must not share a colour.

A search-mode archive is mostly interference, so three filters run first, all
on **shape rather than strength** — the loudest group in one real archive had a
higher median SNR than cyprus1 and was pure noise:

| test                  | default | what it catches                                   |
|-----------------------|---------|---------------------------------------------------|
| arrival-phase scatter | 5 ms    | 500 kHz/s "emitter" whose phase wandered ±274 ms  |
| share of the cycle    | 25%     | a group claiming all 300 seconds of a 300 s cycle |
| detections per slot   | 3       | thirteen groups that saw each second exactly once |

Rejects are **listed with their reason**, not dropped silently: if the row you
came for is among them, the thresholds are wrong, not the transmitter.

**The page reads the archive, and that is what it costs.** A census opens one
HDF5 file per detection — ~1850 of them for three days on DOB. That is 0.6 s
on a local SSD and **two to three minutes** on a network archive, where every
open is a round trip; it is the whole reason the page once took minutes. These
products are written once and never touched again, so every file read is
remembered and the next load opens only what has arrived since. The page says
which days it scanned, how many files it read, how many it had to open, and
how long it took, so a slow load can be attributed instead of guessed at. If
it stays slow, either the process restarted or the archive root has no dated
subdirectories — in which case `_day_directories` falls back to the root and
the scan walks the whole tree.

A warm load should then open nothing at all, and for a long time it did not.
Two things spent the saving. The finders keyed their dedupe on
`Path.resolve()`, which is a `realpath` **system call per file** — 38 ms for
1368 files locally against 0.4 ms for `abspath`, and a round trip each on the
network archive, for a listing that itself cost 3 ms; `muf/paths.py` now keys
on the absolute path. And the "nothing changed" short-circuit was disabled by
*any* unreadable file, which is the normal state of a directory a detector is
writing into — one truncated file out of 1846 forced a full re-read and
re-group on every page load for the life of the process. Only the files that
failed are re-checked now. Warm census 43 ms → 9.1 ms, warm page 45 ms → 10 ms
on the local checkout, and the work removed is precisely the per-file round
trips that dominate on the server.

That cache lives in the process, so **every restart hands the cold read to
whoever opens the page first** — 234 s on the work server, which looks exactly
like a broken page and was read as one. The api therefore does that read itself
at startup, in a background thread, with the parameters the page defaults to;
`CENSUS_WARM=0` turns it off where the archive is huge or absent. The startup
log says what it cost:

```console
api 0.1.0 3097398  db=/data/ionograms.sqlite3  archive=/archive
  census warm: reading up to 2000 detection file(s) under /archive
  census warm: 1846 file(s), 5 emitter(s) in 0.6s
```

The first of those two lines is printed before the read starts, and there is a
third for a read that failed, because the alternative was silence. **The census
also refuses to start a read it cannot finish.** Bounding it by days assumed a
day was a bounded amount of work, and DOB is not that archive: on 2026-08-15 its
newest three days held **172,056 files**, 45,602 of them the `chirp-*.h5` the
census reads first — 93× the 1846 this was measured against, and hours of round
trips. The warm-up took the census lock and never returned it, so the page did
not answer slowly, it did not answer. At most `DEFAULT_MAX_FILES` (2000) are
opened now. The budget is spent **newest day first**, so today stays whole and
the oldest day is the one that loses files, and what is trimmed is *time* and
not quality — the same detection product, never a fall back to the cheap
consolidated files, for the reason above. A capped census says so in `cost`
(`found`, `capped`, `budget`), in a warning, and in a notice on the page naming
how much of the archive it read: "no such emitter" and "not in the part I read"
are different answers, and only one of them is a reason to stop looking.

**And then the ceiling turned out not to be enough, because the cost is not in
the files.** Deployed, the warm-up still never finished. One `os.scandir` of
`/archive/2026-08-15` on the work server returned its 46,436 entries in
**293.8 s** — 6.3 ms per directory *entry*, which is a network round trip
each, not a disk. Listing three days costs a quarter of an hour before the
first file is opened, and no bound on files opened touches that.

Two things followed. The scan stopped walking the tree once per product:
`find_timings`, `find_detections` and `find_cdetections` each walked it, so a
station that writes no `par-*.h5` — DOB writes none — paid a full pass over
every `chirp-*.h5` in the day to learn that, and then paid it again.
`io_detect.find_products` does one `os.walk` and buckets on the prefix before
the first `-`.

More importantly, **a request no longer touches the archive at all.** Both
`/sources` and `/ui/sources` call the census with `block=False`: it answers
from the last completed one, reports `age_s`, and starts a background refresh
when that passes `DEFAULT_MAX_AGE_S` (30 min — a refresh costs about
five minutes of listing, so ten would leave the server scanning half the time
it is up) — one refresh at a time, because
two requests can both find the archive idle and the second would repeat a
fifteen-minute scan the first was already doing. With nothing computed yet it
returns `building`, and the page says so in those words: an unfinished census
and a station that is hearing nothing render identically otherwise, and only
one of them is a reason to go and look at the radio.

The blocking census is still there and is still the real thing — the startup
warm-up, the background refresh and the command line all use it.

That line's `3097398` is the commit the image was built from, stamped in by
`deploy/Dockerfile.api` and served at `/healthz` as `build`. `version` is a
hand-edited string and has read `0.1.0` through every deploy; this one moves on
its own, which is the difference between asking a server what it is running and
inferring it from whether the last fix appears to work:

```console
$ curl -s http://server:8002/healthz
{"ok":true,"version":"0.1.0","build":"3097398","built_at":"2026-08-13T18:41:02Z"}
```

It reads `"source"` from a checkout and `"unknown"` from an image built without
the build arg — neither pretends to be a commit.

### Reading one sounding

`/ui/sounding/{id}` draws the scaled trace over the ionogram and lets you turn
parts of it off. Circles are per-trace: click a legend entry to hide one
segment, double-click to isolate it, or use the **scaled points**, **raster**
and **MUF / LOF** boxes to strip the picture down to whichever layer you are
arguing about. The `scaling:` pills switch between `algo`, `kmeans` and
`contour`; **download SAO.XML** gives you all three at once.

**The raster stays a PNG, and only the points are data.** 486 × 3999 cells is
1.9 million numbers — about 11 MB as JSON against 164 KB as an image — while
the trace is a few hundred points. So the image is served bare
(`/ionogram/{id}.png?bare=true`: no axes, no title, no colourbar, no overlay)
and placed *in data coordinates* beneath scatter traces that carry their own
hover readout. Its extent runs half a cell past the first and last sample,
because `pcolormesh(shading="nearest")` centres a cell on each one; half a bin
out and every circle sits beside its echo instead of on it, which is invisible
until someone zooms in.

**plotly.js is vendored, not fetched.** `services/api/static/plotly.min.js` is
the 1.0 MB `plotly-basic` build, served from the image at `/static/`. DOB has
been off the internet for a week at a time, and a plot that needs a CDN to draw
a file already on disk fails exactly when someone needs it. There is still no
build step.

The panel beside it is the record itself, read back out of the XML rather than
recomputed — measured characteristics with their UAG-23A qualifying letters,
then `<Modeled>` rows labelled with the model that asserted them. On NIC→DOB
the measured MUF is 24.450 `D` (censored by the recorder's band ceiling, so a
lower bound) against IRI's 22.275; the two are written side by side and neither
corrects the other.

Scaling costs about a second cold and nothing after: detection products are
write-once, so path plus mtime identifies a scaling for good and the last 24
are kept. Stepping through a day with ← and → therefore stays instant. If the
build fails the page still renders — the row, the neighbours and the stored
extractions are worth reading, and the failure is named rather than turned into
a 500. `SAO_MODEL=0` drops the IRI column on a host that should not be making
outbound requests at all.

### Reading a series

`/ui/series` is the same question over a day instead of over one sounding, and
it draws four things rather than one: **MUF**, **LOF**, an equivalent **foF2**,
and **IRI** beside them with the residual underneath. Same vendored plotly as
the sounding page — drag to zoom, click a legend entry to drop a curve, click a
point to open that sounding.

**Hue is the parameter, shape is the source.** Markers are measured, lines are
modelled, so the blue markers and the blue dashed line are this circuit's MUF
and IRI's, and the gap between them is what the residual panel plots. Overlay
several circuits with `circuit=all` and hue becomes the *circuit* instead —
two paths' MUFs in one colour is exactly the comparison the page must not
invite — with each modelled at its own control point. The five family
checkboxes move whole parameters at once, because four parameters over five
circuits is a legend of twenty-odd entries and "show me the LOF" should not be
twenty clicks.

**Hollow markers are bounds at both ends of the band.** A hollow MUF marker sat
at the top of the sweep, so the real MUF is at or above it; a hollow LOF marker
sat at the band floor, so the real one is below. Neither is filtered out —
dropping either would bend the curve towards the middle of the band — but both
are **excluded from the residual statistics**, and the count of what was
excluded is printed beside the count that was used. A pick pinned to the
ceiling says the ionosphere supported *at least* that much; scoring it as a
residual reports the recorder's band ceiling as a modelling error, which on a
ceiling-limited circuit is most of the daytime.

**foF2 here is not a measurement.** An oblique sounder never sees vertical
incidence. It is the measured MUF put back through the secant law at
hmF2 = 300 km **over one hop** — the same convention `iri.predict` converts by,
so the measured curve and the modelled one sit on the same geometry. Over
`MAX_SINGLE_HOP_KM` the path reflects more than once and the obliquity belongs
to one hop, not to the whole distance.

**LOF, not LUF.** ITU-R P.533-13 §9 defines the lowest *usable* frequency with
a required signal-to-noise ratio and a monthly median: a property of a service
and of a month, and one sounding has neither. What an oblique sounder scales is
the lowest *observed* frequency. It tracks D-region absorption, so it follows
the sun rather than the F2 layer — which is why it is worth having on the same
axis: a MUF that moves while the LOF under it does not is a MUF worth doubting.
A sounding with a LOF and no MUF is a point on this chart, which it was not
before; on 2026-08-10 that is 120 soundings of Juliusruh→DOB with one MUF
between them, and the page used to show none of them.

**The model is evaluated at the sounding instants, one call per day.** At the
instants because that is what makes a residual subtractable, per day because
`iri.predict` reads its solar driver from the *first* timestamp it is given —
one F10.7 stretched across a window holding February and August would be wrong
with nothing on the page to show it. Beyond `MAX_MODEL_DAYS` (31) the page
declines and says so rather than spending a minute with the request open.
`model=off` in the query string, or `SERIES_MODEL=0` in the environment, skips
it entirely: this is the one part of the page that can reach the network.

On cyprus1→yoshkar-ola for 2026-02-04, `kmeans` against IRI is **r = +0.985**
over 101 pairs with a **+1.63 MHz** median bias and 13 lower bounds held out —
and the residual panel shows the shape that number hides: IRI runs low through
the morning rise and high after it, which is a diurnal disagreement rather than
a scale one.

### Is this box still keeping up?

`tools/benchmark.py` measures where the time goes and, more usefully, whether
that has changed. Take a run when things are healthy, keep the JSON, and pass
it back as `--baseline` when something feels slow:

```
python tools/benchmark.py --archive /data/lfs --json healthy.json
python tools/benchmark.py --archive /data/lfs --baseline healthy.json
```

It samples the archive with a fixed seed — `--files`, forty by default — runs
that sample serially and then across workers, and reports the split between
reading a sounding and processing it. Add `--url http://127.0.0.1:8000` to time
the served pages as well, and `--token` if that instance wants one.

Absolute timings do not travel between machines, so nothing is compared against
a figure from somewhere else. The verdicts come from ratios that mean the same
thing everywhere, and from this same box on an earlier day:

| what it watches                      | why it matters                                                                                                                             |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| parallel speed-up                    | below ~3x on four or more cores, the workers are fighting each other for threads rather than sharing the machine — check `MUF_PIN_THREADS` |
| share of a sounding spent reading it | about 3 % on a local disk; far above that is the archive mount, and it moves long before a page looks slow                                 |
| peak RSS across the sample           | a worker should settle after its first file                                                                                                |
| soundings that raised                | a correctness problem wearing a performance costume — see `tools/diagnose_reception.py`                                                    |
| the picks themselves                 | with `--picks`, a later run proves a speed-up did not move a measurement                                                                   |

It exits non-zero when it finds something serious, so it can be run from cron
and left to stay quiet.

### Can this host still reach its upstream?

Everything above is about the station. The **upstream** panel is about the
server itself: whether it can still fetch the solar indices the reference
models run on. That dependency is otherwise invisible in the output — with no
route out, `muf.reference.indices` falls back to its cache and keeps answering,
and a driver from six months ago renders exactly like a fresh one.

| pill          | means                                                |
|---------------|------------------------------------------------------|
| `INTERNET OK` | every index host answered                            |
| `PARTIAL`     | some answered; the panel names the ones that did not |
| `NO INTERNET` | none answered — models run on cache, or not at all   |
| `INTERNET?`   | no reading current enough to answer with             |

**It probes those hosts, not the internet.** A ping to a public resolver
answers a question nobody asked: it stays green behind a proxy that blocks
HTTPS to `sidc.be`, and it goes red on a host reaching every source it needs
through a mirror. The list is `indices.SOURCES`, so a source added there is
probed without anyone remembering to.

**Reachability and freshness are separate columns**, because they fail
independently and mean opposite things: unreachable with a fresh cache is a
model still answering correctly, and reachable with `never` is a model that has
never had a driver. Lightweight is a constraint rather than an aspiration — one
`HEAD` per *host*, three of them, concurrently, on a 4 s timeout, in a daemon
thread every 600 s. No request path ever waits on one: `/net` and `/ui` both
serve the last reading in about 1.5 ms, and the pill goes grey rather than
green if the checker itself stops running. `NET_CHECK=0` switches it off on a
deliberately isolated host; `NET_CHECK_INTERVAL_S` and `NET_TIMEOUT_S` tune it.

Six files on three hosts, and the redundancy is the point — any one of them can
be down or firewalled without a model losing its driver, because
`solar_indices` raises only when *nothing* answered and nothing is cached:

| source       | file                                | carries                                            |
|--------------|-------------------------------------|----------------------------------------------------|
| SILSO        | `SN_d_tot_V2.0.csv`                 | daily international sunspot number                 |
| SILSO        | `EISN_current.csv`                  | estimated SSN for the current month                |
| SILSO        | `SN_ms_tot_V2.0.csv`                | 13-month smoothed R12                              |
| NOAA SWPC    | `observed-solar-cycle-indices.json` | monthly SSN and F10.7                              |
| NOAA SWPC    | `f107_cm_flux.json`                 | daily 10.7 cm flux, last 42 days                   |
| irimodel.org | `apf107.dat`                        | daily F10.7, its 81-day mean, and `ap`, since 1958 |

The last three are what IRI actually wanted. Before them the only flux
available was SWPC's **monthly** figure; `apf107.dat` is the driver file IRI
itself reads, and SWPC's rolling 42 days closes the fortnight it runs behind —
on 2026-08-13 `apf107.dat` ended at 2026-07-28 and `SN_d_tot` at 2026-07-31,
which is why the estimated series is there too. Two traps in that file: `ap`
reaches 400 in a severe storm and the fields are three columns wide, so a split
on whitespace merges them (2003-10-29 reads `400300207236...`), and
irimodel.org runs mod_security, which refuses urllib's default `User-Agent`
with a 406 — and `curl`'s as well.

`SolarIndices.f107` is the observed daily flux and `f107_driver` **is what a**. They are separate fields on purpose: the CCIR and URSI maps
IRI interpolates were fitted against a smoothed index, so `f107_driver` prefers
the 81-day mean and falls back through monthly to daily. Feeding a map the
day's flux would swing foF2 across a solar rotation in a way the climatology
never claimed to predict.

Roughly 5 MB lands in the cache, at `$HOME/.cache/muf/indices` by default. In a
container `$HOME` is an image layer, so `MUF_INDEX_CACHE` points it at
`/data/indices` — the volume the database already uses — and a `docker compose pull` stops throwing it away.

Two things the `/ui/sources` page cannot do for you, both settled in the
identify form. Rows
are grouped by chirp rate and arrival phase, so several transmitters that all
start near the second boundary merge into one row — identify it as one of them
and tick only its slots. And `rep` is filled in as the assumed cycle, so an
emitter whose slots step every 30 s needs it corrected there, or you will hear
it a tenth as often.

**How many slots you can afford is set by the ringbuffer, not the CPU.** At
100 kHz/s a sweep across a 25 MHz recorded band takes ~250 s of a 300 s cycle,
so N scheduled slots means N sweeps in flight at once, and each must begin
processing while its samples are still in `/dev/shm`. `find_timings.py` prints
that margin per sounding:

```bash
grep -oE '\-?[0-9.]+ s left' ~/chirpsounder2/logs/find_timings.log \
  | awk '{n++; if($1<=0) z++} END {printf "n=%d lost=%d (%.2f%%)\n", n, z+0, 100*z/n}'
```

**Match the minus sign.** `grep -o '[0-9.]* s left'` drops it, so `-5.2 s left`
is counted as a comfortable 5.2 and every failure reads as a pass. And read the
whole log, not a `tail` — a 20-sounding sample on DOB gave a 90 s minimum and
"none under 60", which looked healthy. Measured properly over 1333 soundings it
was **57 negative, 4.28% lost**, matching the `missing data - skipping` lines
in `ionograms.log` one for one. A margin under 30 s is a near miss; only `<= 0`
is a loss, and the loss is silent.

**Storage under the consumer matters as much as the buffer's size.** That 4.28%
was measured while products went to a 5400 rpm laptop disk over ntfs-3g — FUSE,
userspace, seek-bound — with the archive mirror reading the same spindle every
five minutes. `ionice` does not reach the ntfs-3g daemon doing that I/O. A
network share is not automatically the worse choice; measure both.

**The ring buffer belongs in RAM, not on an SSD.** 25 MS/s of complex int16 is
100 MB/s — **8.64 TB written per day**. A 1 TB consumer SSD rated ~600 TBW is
exhausted in 69 days, and a write-intensive enterprise drive in under two
years; tmpfs has no endurance to spend. Buy RAM, and check first that a bigger
window would even help — another 120 s saves a sounding that missed by 5 s and
does nothing for one that missed by 200:

```bash
grep -oE '\-?[0-9.]+ s left' ~/chirpsounder2/logs/find_timings.log | awk '$1<=0' \
  | awk '{n++; if($1>-120) s++} END {printf "%d of %d saved by +120s\n", s+0, n}'
```

Add slots only once the loss rate is zero, and re-measure after each step.
Storage scales independently: at 0.6 MB per ionogram, 37 slots is 10,656
soundings and 6.4 GB a day.

---

## Changes in this merge

The clustering work previously lived in a separate `MUF_clustering` folder and
ran on rendered PNG ionograms, recovering frequency from pixel position. It now
runs on the numeric array, and both halves live here.

**Corrections**

- **The virtual-range axis was inverted.** The echo in `cyprus1_20260204_000010.lfs` sits at fftshifted bin 3909. Under the ascending axis used at `MUF.py:116` that is **-2732 km**, which no echo can occupy; under `R - idx*step` it is **+2739 km**, correct for a 2,588 km path. The plot looked right only because it was reversed with `[::-1]` at draw time, and `MUF.py:297`'s `if vrng < 0: vrng = R + vrng` was patching over the sign error. The axis is now defined once, in `calibrate.py`.
- `rx_longitude` **was reading the latitude.** `lfs_header.py:108` seeks offset 150, which is `rx_latitude`'s; the longitude is at 154. Yoshkar-Ola now reads 47.53E rather than 56.38.
- **Three defects in the algorithmic estimator**, documented in `extractors/algorithmic.py`: an uninitialised buffer, an `np.append` whose result was discarded, and an `np.amax(axis=0)` that assembled its returned `(frequency, range, row)` triple from *different* detections — so the reported range was not the range at the reported MUF. `--legacy-algo` reproduces the old decision where that is possible.
- **K-means invented MUFs out of noise.** On a recording containing no echo it still reported a value at the top of the band, because the selection rule fell back to "keep the brightest cluster" when nothing passed the threshold. It now returns no pick.
- **Pixel-position calibration is gone.** The scripts assumed 1,220 columns and a 3500-2500 km height span; the renderer actually produced 2500-4000 km. Axes now come from the file header. (`ion_col_num = 1220` was right by accident — it is `len(iq) // 8192` for this instrument.)
- **Truncated recordings are no longer stretched.** A recording cut short still declares the full sweep in its header — 10 files in `2026.02.05` hold 347 windows instead of 1,220 while claiming `dur=250`. Mapping the axis onto the nominal 32.5 MHz endpoint would place their last bin there when the transmitter had only reached 14.6 MHz, inflating MUF by up to 2.2x. The frequency axis is now derived from the chirp rate and elapsed time, which is what physically sets it; `sweep_complete` and `sweep_fraction` record the shortfall.

**Additions**

- Range gating from the header, applied inside the FFT loop — 40x less data, and the biggest single accuracy gain.
- The algorithmic estimator vectorised: a triple-nested Python loop over ~110M cells becomes a handful of array operations.
- One shared, tunable MUF decision rule with a continuity requirement, replacing three partial ad-hoc versions.
- Band-limited and truncated soundings detected and excluded from statistics.
- **Temporal tracking** (`muf track`) — Kalman filter with RTS smoothing, which fills gaps, rejects outliers and attaches an uncertainty to every point.
- **Trace fitting** (`muf/fit.py`) — an outlier detector that agrees 100% with `track`'s independent rejections, and repairs the values it flags.
- **Mode segmentation and reconstruction** (`muf/trace.py`) — splits a trace at propagation-mode boundaries and fits a weighted smoothing spline to one mode, turning a sparse scattered point set into a continuous `h(f)`. Revealed that 87% of soundings carry more than one propagation mode.
- **External references** (`muf/reference/`) — IRI, GIRO and a transparent solar-zenith model, with solar indices fetched and cached from SILSO and NOAA. This is what revealed the out-of-band problem above.
- **SAO.XML 5.0 export** (`muf export`) — the URSI/INAG interchange format GIRO publishes, so the segmented trace becomes an archivable product rather than a plot overlay. The MUF is emitted as a `<Custom>` characteristic, not URSI `MUF(3000)`, because UAG-23A §1.50 states those are different quantities; the band-limited flag becomes UAG-23A's qualifying letter `D`. `muf plot-sao` reads a record back and draws it — over its own ionogram when the `.lfs` is to hand, and from the XML alone when it is not, which is what distinguishes a published format from a private file with angle brackets. `--iri` adds the reference model's MUF, foF2 and hmF2 as `<Modeled>` characteristics, so the panel shows measured and modelled side by side without ever conflating them.
- **The low-frequency end** (`muf/lof.py`, `muf lof`) — LOF at the estimator's own trace and at a ladder of detection thresholds, with the threshold carried in every result. Correlates with the cosine of the solar zenith angle at r = +0.86, which is the D-region absorption signature. Found that the transmitter's real band floor is 8.0 MHz against a declared sweep start of 7.5, so 102 of 266 LOF values are upper bounds and earn URSI's `E`.
- **Several days at once**, written one file per day; `daily`, `track` and `compare` accept multiple tables or a directory of them.
- `--jobs` parallelism and a gated-array cache.
- `pyproject.toml` (installable, gives a `muf` command), `requirements.txt`, and `data/`/`*.lfs` added to `.gitignore` — the recordings were untracked but *not* ignored, so a single `git add .` would have committed tens of gigabytes.

See `BACKLOG.md` for what is deliberately not done: the archive-format analysis,
the machine-learning options and the literature behind them, and the historical
database values that need re-deriving.

**Note on** `--zero-periods`

Zero-padding subdivides range bins without resolving anything further: the true
resolution is set by the bandwidth swept during one window (14.65 km here) and
equals the *unpadded* bin spacing. The old `-z 10` therefore cost 11x memory and
FFT time for finer sampling of the same information. The default is now 0, and
sub-bin precision comes from parabolic interpolation of the peak instead.