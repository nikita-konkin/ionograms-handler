# Backlog

Decisions deferred, with the measurements behind them, so they can be picked up
without re-deriving anything. Ordered roughly by value.

---

## 1. Archive format: sparse Parquet instead of raw IQ

**Problem.** Raw `.lfs` is 80 MB per sounding, 23 GB per day, **8.4 TB per year**.
That does not fit anywhere for long. But the MUF work needs only a small part of
what is in those files.

**Measured** on `cyprus1_20260204_030010.lfs`, whole +/-60,000 km range axis
retained (1220 x 8055 cells):

| Format | Size | vs raw IQ |
|---|---|---|
| raw `.lfs` IQ | 80 MB | - |
| `.npy` float32 dense | 39.3 MB | 2x |
| `.npz` float32 deflate | 33.7 MB | 2.4x |
| Parquet dense `list<f32>` zstd | 34.2 MB | 2.3x |
| `.npz` float16 | 15.4 MB | 5x |
| `.npz` uint8 (0.31 dB steps) | 7.6 MB | 11x |
| Parquet dense `list<u8>` zstd | 7.5 MB | 11x |
| **Parquet sparse >= 30 dB** | **2.85 MB** | **28x** |
| Parquet sparse >= 35 dB | 0.07 MB | 1100x |

**Findings**

- Parquet *dense* buys nothing over npz (34.2 vs 33.7 MB). Its strengths --
  columnar layout, dictionary encoding, predicate pushdown -- do not apply to a
  dense numeric matrix; it is just a container around the same compressor.
- Quantising float32 -> uint8 over 0-80 dB is a free 5x: 0.31 dB error, far
  below the noise spread.
- Sparse only wins below ~20% occupancy. At >= 25 dB, occupancy is 54% and
  sparse is *worse* than dense uint8 (8.9 vs 7.5 MB) -- the `(f, r, db)` triple
  costs 5 bytes per cell against 1.
- At >= 30 dB (14.67% occupancy) the extracted MUF is **identical**: 12.200 MHz
  from the sparse array and from the full one.

**Proposed layout** -- keeps the whole range axis, which is a requirement:

```
sparse cells >= 30 dB          ~2.8 MB   all real structure, full range
+ per-frequency-row stats       ~30 KB   median, p90, p99 per row
+ header & calibration           ~1 KB
                               ---------
                                ~2.9 MB per sounding
```

The row statistics preserve noise-level work (cf. the original
`median_power_level.py`, "получение уровня шума"), which a signal-only archive
would throw away.

### Window is locked at archive time

The archive stores `|FFT|^2`; phase is gone, so a stored ionogram cannot be
re-windowed in either direction. `window` and `zero_periods` are fixed when the
archive is written.

**This costs almost nothing for MUF.** Measured across four soundings:

| window | freq bins | df | d(range) | mean |deviation| from 8192 |
|---|---|---|---|---|
| 2048 | 4882 | 5.1 kHz | 58.6 km | 0.278 MHz |
| 4096 | 2441 | 10.2 kHz | 29.3 km | **0.020 MHz** |
| 8192 | 1220 | 20.5 kHz | 14.7 km | - |
| 16384 | 610 | 41.0 kHz | 7.3 km | 0.092 MHz |
| 32768 | 305 | 81.9 kHz | 3.7 km | 0.113 MHz |

Across a 16x span the MUF moves by <= 0.11 MHz -- under half the disagreement
between our own methods (0.23 MHz MAE). Only 2048 is meaningfully worse, and
only at night (-0.52 MHz at 03:00 and 18:00): 58.6 km range bins are too coarse
to hold the faint trace edge.

Storage barely depends on window either, because cell count is invariant
(`n_freq x n_range = (len(iq)/W) x W ~ len(iq)`) and occupancy stays ~14.65%
(median equalisation normalises each row):

| window | cells | >= 30 dB | sparse | TB/year |
|---|---|---|---|---|
| 4096 | 9.83M | 14.68% | 2.72 MB | 0.29 |
| 8192 | 9.83M | 14.67% | 2.85 MB | 0.30 |
| 16384 | 9.83M | 14.64% | 3.03 MB | 0.32 |
| 32768 | 9.83M | 14.63% | 3.28 MB | 0.34 |

**Variants**

| Variant | Windows | TB/year | vs raw |
|---|---|---|---|
| Minimal | 8192 | 0.30 | 28x |
| **Recommended** | **8192 + 32768** | **0.64** | **13x** |
| No-regrets | 4096 + 8192 + 16384 + 32768 | 1.25 | 6.7x |
| Dense uint8 (keeps noise field) | 8192 | 0.79 | 11x |

8192 + 32768 gives one frequency-oriented view for MUF and one range-oriented
view (3.7 km) for multi-hop separation. Adding 4096 is near-pointless -- it
differs from 8192 by 0.020 MHz. Processing is ~0.3 s per sounding per window.

**To build:** `muf archive <dir> --out <dir> --window 8192 --window 32768
--threshold-db 30`, recording window, zero_periods, threshold and full
calibration in each file's Parquet metadata, and refusing to mix archives made
at different settings. Plus a reader so `run`/`plot`/`compare` work off archives
instead of raw `.lfs`.

**Caveat:** 30 dB is tuned to this path and receiver. Check occupancy on a
quieter and a noisier day before committing; keep it under ~20%.

---

## 2. Re-derive historical database values

`stuffr.py:319`'s `np.amax(MUFs, axis=0)` takes the maximum of each column
independently, so the `(frequency, range, row)` triple it returned was assembled
from three *different* detections. Tracing into `Load_muf_data_to_muf_db`:

- **`muf`** -- probably sound. It is the maximum frequency over detections,
  which is what MUF means. But with no continuity requirement, a single
  interference spike sets it.
- **`vrange`** -- effectively meaningless. The largest range index over all
  detections, not the range at the MUF, then passed through the inverted axis
  and the `if vrng < 0` patch at `MUF.py:297`.
- **`muf_column` / `muf_row`** -- from the same incoherent triple, and
  `muf_load_to_db.py:158` uses them as the deduplication key. Duplicate or
  dropped inserts are plausible.

Anything correlating MUF against virtual height rests on the `vrange` column.
Re-run the affected days through the new pipeline before trusting it.

---

## 3. The sounder cannot see the midday MUF

Established by comparing against IRI (`muf compare --ref-model iri`):

| Subset | n | bias | MAE | corr |
|---|---|---|---|---|
| IRI below the 32.5 MHz sweep top | 204 | **+0.55 MHz** | 2.84 | 0.890 |
| IRI above the sweep top | 58 | **-5.25 MHz** | 5.25 | 0.415 |

IRI puts the MUF above the band in **82 of 288 soundings**, every one between
06:00 and 13:00 UTC. Where the instrument can see the MUF we agree with IRI to
half a megahertz; where it cannot, we read 5 MHz low. The `limited_` flag caught
15 of 82, because it only fires when a pick lands within 3 bins of the top.

**Consequences**

- Midday MUF values are lower bounds, not measurements. Any daily maximum,
  diurnal amplitude, or seasonal trend computed from them is biased low.
- The same applies to every historical value in the database, and to anything
  downstream (`shap_fo_MUF_pred`, `ionosphere-shap-analysis`).
- Consider whether the sweep can be extended above 32.5 MHz. `dur=250` at
  `rate=100 kHz/s` from `cf-sample_rate/2 = 7.5 MHz` sets the top; reaching
  38 MHz (IRI's peak prediction for this day) needs `dur=305`.

**Caveat.** This rests on IRI being right, and IRI can overestimate foF2 near
solar maximum. The evidence is circumstantial but strong: the discrepancy is
confined exactly to the hours where IRI exceeds the band, and is near zero
elsewhere. A GIRO comparison would settle it -- see section 6.

**DOB has the same problem 8 MHz lower, and the flag never fires
(2026-08-12/13).** The chirpsounder2 sweep runs 0.525--24.825 MHz, not 32.5, so
the ceiling is much nearer. Cyprus (NIC, 3436 km) picked on **216 of 216**
method-soundings over two days -- a perfect yield, and the cleanest diurnal
curve this station has produced:

| hour UTC | mean MUF | max | mean SNR |
|---|---:|---:|---:|
| 01 | 10.51 | 10.60 | 50.7 |
| 03 | 15.04 | 16.10 | 48.1 |
| 04 | 18.73 | 21.50 | 48.9 |
| **05** | **24.45** | 24.55 | 50.8 |
| **06** | **24.46** | 24.55 | 52.6 |
| 07 | 23.99 | 24.50 | 49.5 |
| 08 | 22.89 | 24.25 | 49.0 |

**A curve moving 3--5 MHz per hour does not sit at 24.45 and then 24.46.** The
pick distribution says the same thing: 40 of 216 picks fall in the two bins
24.45/24.50, and **nothing in the whole dataset exceeds 24.55**. That is a
ceiling, and the mid-morning Cyprus MUF is a lower bound.

**`limited` fired on 0 of 216.** `band_edge = freq_stop - 3 * freq_step` is
24.675 MHz here, and the observed ceiling is ~24.55 -- roughly six bins below
nominal, so every clipped pick is recorded as an ordinary measurement. Two
separate faults in that:

- **Three bins is not a portable margin.** It was chosen against a 32.5 MHz
  sweep, where it is 0.6% of the band; here it is 0.15 MHz. A fractional
  criterion would travel between instruments.
- **`freq_stop` is the wrong anchor.** The nominal top is 24.825 but NIC's
  trace stops at ~24.55 while other emitters in the same archive reach 24.80,
  so the *usable* top is emitter-dependent -- a function of where that
  transmitter's signal falls below the detection level, not of the header. A
  flag anchored to the header cannot see it.

Rendering `lfm_ionogram-NIC-DOB-ch0-002-1786601035.00.h5` (2026-08-13 06:03:55Z,
picks 24.40/24.45/24.45) shows why: the trace is a flat line at 2650 km running
from 11.5 MHz to its cutoff with no nose at all. On a 3436 km path the group
delay stays near-constant until close to the MUF, so a trace that ends flat has
ended for a reason other than the MUF -- and the plot has ~0.3 MHz of empty
band to the right of the pick.

**Consequences for DOB specifically.** Unlike the legacy station this is not a
midday-only bias -- 24.8 MHz is reachable on this path from 05:00, and midday
will be worse. Any Cyprus daily maximum is currently a lower bound. The remedy
in this section's last bullet applies unchanged: the top comes from `dur` at the
configured rate, and raising it costs sweep time.

### The anchor is fixed (2026-08-13)

`Options.band_ceiling_mhz` supplies the frequency the circuit actually returns
echoes at, `muf.lof.measure_band_ceiling` recovers it from a set of recordings,
and `pipeline.band_edge_mhz` is now the single definition shared by the
`limited_` columns and the SAO `D` letter -- which had a second, independent
copy of the arithmetic in `saoxml.qualifying_letter`. The run records which
ceiling it used in a `band_ceiling` column, because two runs over the same files
with different ceilings produce identical MUF values and different censoring,
and nothing else in the row distinguishes them.

Measured over the 72 NIC soundings of 2026-08-12/13:

    measured band ceiling  24.53 MHz   (DIFFERS from the declared 24.83)

    limited_ per method     header anchor      --band-ceiling 24.53
      algo                       0/72                  15/72
      kmeans                     0/72                  18/72
      contour                    0/72                  17/72

24.53 against the ~24.55 read off the pick distribution independently, and a
fifth to a quarter of every method's picks move from "measurement" to "lower
bound".

**The measurement needs the continuity rule, and this is the interesting part.**
The first implementation was a straight mirror of `measure_band_floor`: the
highest bin above the detection level, high quantile instead of low. On this
archive it returns **24.80 MHz** and declares the header correct. The floor can
use a bare threshold because the bottom of the band really is dead -- that
deadness is what makes it measurable -- whereas narrowband interferers sit above
43 dB right up to the sweep stop, so at the top a single-bin rule measures the
interference. Reading the end of the last qualifying *run* instead is the whole
difference between 24.80 and 24.53. (The earlier draft of this section proposed
"the highest frequency with any detected energy", which is precisely the version
that fails.)

**Still open: three bins is not a portable margin.** The anchor fix defuses most
of it -- the ceiling is now measured at the same resolution as the picks -- but
`BAND_EDGE_BINS = 3` is still an absolute count chosen against a 32.5 MHz sweep.
A fractional criterion would travel between instruments, and changing it moves
every result already published, so it wants its own pass.

**The ceiling is now stored per circuit.** `Station.band_ceiling_mhz` keys it by
*receiver* -- `NIC` carries `(("DOB", 24.53),)` -- and `pipeline.circuit_ceiling`
resolves flag, then registry, then the sweep stop. The 72 NIC soundings now pick
up 24.53 with no flag at all.

Keyed by receiver rather than held as one number per transmitter because a
ceiling is a property of the circuit: it is where *this path's* signal drops
below the detection level, which depends on the receiving antenna, its noise
environment, the path length and the sweep that receiver runs. The same Nicosia
site measures 24.53 into DOB and 32.48 into Yoshkar-Ola, where it matches the
declared stop. A JSON registry may supply the same mapping; a bare number is
rejected rather than applied to every receiver, since the flag it feeds is the
one deciding whether a MUF is published as a measurement.

**Still open: only NIC->DOB is measured.** SGO->DOB has no entry and falls back
to the sweep stop. It needs a day of SGO data through local noon, which
section 17 wants anyway.

## 4. Investigate the 71-second truncations

Ten files on 2026.02.05 stop at ~71 s of a 250 s sweep, at no particular time of
day, in three distinct lengths (347 / 342 / 338 windows). Clustered but not
identical reads more like the acquisition system dropping out than a copy
artifact. 17% of that day's soundings are affected. The pipeline handles them
correctly either way (`sweep_complete`, `sweep_fraction`).

---

## 5. Code consolidation

- **`MUF_clustering/` is not under version control.** Fifteen scripts, several
  broken (`ionogr_clustering_DBSCAN_0.01.py` raises `NameError` on any run;
  `svr_0.01.py` has unreachable code after `sys.exit()`). Its working methods
  now live in `muf/extractors/`. Either commit it under `legacy/` with pointers
  to the replacements, or retire it deliberately.
- **`data_handler/` is dormant.** `psycopg2` is not installed and the host is a
  fixed LAN address. Decide whether PostgreSQL returns or the file outputs are
  now the record.
- **`data_handler/database.ini` is tracked with live credentials.**
  Gitignoring does not untrack it; `git rm --cached` would.
- **The old `README`** (no extension) documents a `median_power_level.py` that
  is not in the repo. Superseded by `README.md`.

---

## 6. Reference models -- status

- **GIRO / DIDBase.** Viable: the path's control point is 45.99N 39.09E and
  station **RV149 ROSTOV** sits **146 km** away, well within the ionospheric
  correlation scale. The `lgdc.uml.edu/common/DIDBGetValues` endpoint was
  returning 404 / connection failures while this was written; the adapter is
  built against the documented URL pattern and degrades with a clear message.
  Retry when the service is up. `DMUF=<path km>` makes the server do the oblique
  conversion.
- **Solar indices.** Verified working: SILSO daily sunspot number
  (`SN_d_tot_V2.0.csv`, current through 2026-06-30) and NOAA SWPC monthly
  indices. **R12 smoothed sunspot number is `-1.0` for recent months** -- it
  needs +/-6 months of data, so Feb 2026 will not have one until ~Aug 2026.
  Models needing R12 must fall back to daily SSN or F10.7.
- **IRI control points.** Every circuit is modelled over its own reflection
  point, from that sounding's own header: NIC -> Yoshkar-Ola over 45.99N
  39.09E, NIC -> DOB over 49.22N 24.59E. Since 2026-08-14 a path longer than
  `geometry.MAX_SINGLE_HOP_KM` also uses **both** control points and is limited
  by the worse of them, converting at the per-hop distance. `control_points`
  had returned two since it was written and nothing consumed them, so a long
  path was modelled off one midpoint and converted at a whole-path obliquity
  no ray achieves -- `m_factor` peaks at 3840 km and falls away after, so an
  8000 km path came out a fifth low, which reads as the instrument
  over-picking. No circuit in the archive is over 4000 km today; DOB hearing
  something further away is what this is waiting for.
- **MINIMUF.** The authoritative coefficients (Rose & Martin, NOSC TD 201,
  DTIC ADA066256) could not be retrieved. Implementing it from memory would
  manufacture a plausible-but-wrong reference, which is the exact failure mode
  this project has been correcting. Either obtain the report, or use the
  transparent secant-law model instead and call it what it is.

---

## 7. Machine learning, and the rest of the detection ideas

Context from the literature. The closest published analogue is **OIASA**
(Ippolito et al., *J. Space Weather Space Clim.* 8, A10, 2018) -- oblique
ionograms, MUF, chirp sounder. Its accuracy against manual scaling is worth
keeping in view:

| vs manual | May 2015 | Sept 2016 | Oct 2015 |
|---|---|---|---|
| acceptable (<=1.5 MHz, URSI standard) | 70% | 88% | 82% |
| accurate (<=0.5 MHz) | 31% | 53% | 47% |

Plus ~4% false MUF. Our estimators agreeing to 0.22 MHz is internal
consistency, not accuracy; where we sit on that table is still unknown.

The published deep-learning work is all **vertical** and all label-hungry:
multi-scale attention U-Net (*Radio Science* 2023), multiscale Transformer
segmentation (*Remote Sensing* 2024, 2,523 hand-labelled ionograms), GAN
autoscaling (*Radio Science* 2025), ResNet50 regression on 13 years of
Sodankyla data (*Earth and Space Science* 2024), NOIRE-Net (*Frontiers* 2024).

**Done** -- see `muf/fit.py` and `muf/track.py`, and section 8 for what the
trace fit did and did not deliver.

**Not done, in the order I would do them:**

1. **Bootstrap labels from agreement.** Where all three estimators agree within
   0.2 MHz *and* `run > 20`, that is a high-confidence pseudo-label -- roughly
   400 from two days. Hand-verify a stratified sample of ~200 covering the hard
   cases (Es-contaminated, spread-F, multi-hop). This unlocks everything below;
   nothing supervised is possible without it.
2. **Classify, don't regress.** "Predict MUF" is already solved to 0.22 MHz on
   easy soundings. The open problem is *which soundings are trustworthy and what
   mode am I looking at*: a small CNN over {clean 1F2, multi-hop, Es, spread-F,
   no trace} would route soundings to the right estimator and flag the rest --
   a learned version of OIASA's contrast threshold. Much less label-hungry than
   segmentation.
3. **Self-supervised pretraining.** 288 soundings/day over years is millions of
   ionograms, far more than any published dataset. Masked autoencoding on the
   gated dB tiles, then fine-tune on the small labelled set from (1). The 2022
   autoencoder had the right architecture and the wrong objective -- denoising
   rather than representation learning.
4. **Sequence models -- the novel angle.** Every published method scales *one*
   ionogram. Feeding a network the last 6 soundings (30 min) of gated tiles and
   predicting MUF plus uncertainty uses temporal context that a single ionogram
   cannot supply, and nobody appears to have done it for oblique paths. Paired
   with band-limit extrapolation (section 8), that is a publishable contribution
   rather than a reimplementation.
5. **Physics-informed constraints.** With no labels, use IRI as a prior or
   regulariser rather than fitting pseudo-labels alone. Also add the O/X
   constraint OIASA exploits: the ordinary and extraordinary traces are
   separated by f_H/2 (~0.7 MHz here) and fitting them jointly is a strong
   constraint we do not currently use.

**A hypothesis that was tested and rejected.** The kmeans estimator reads
+0.57 MHz above `algo`, and f_H/2 at the control point is ~0.7 MHz, suggesting
kmeans might be picking the extraordinary trace. The difference distribution is
unimodal with a smooth right tail (45% within 0.15 MHz of zero, only 12% near
0.7), not bimodal. It is a permissive cluster boundary, not mode confusion.

---

## 8. Trace fitting: what it delivered

`muf/fit.py` fits a parabola to the nose of the trace, OIASA-style, and reads
MUF from the vertex. Measured on this instrument's data, one of the three
predicted uses worked and two did not.

**Superseded in part.** This section recorded the fit as a weak estimator
because "only the low-ray branch is visible, so the vertex is extrapolated from
curvature with nothing on the far side to anchor it". That was **wrong** -- the
high ray is plainly present, and `muf/trace.py` was separating it into its own
track and discarding it. Feeding the fit both branches
(`trace.nose_points`) roughly halves the error: bias -0.12 -> -0.05 MHz, MAE
0.74 -> 0.37, median residual 0.31 -> 0.18. The narrow 1.0 MHz fitting window
that this makes possible is now the default.

**Works: outlier detection.** `|fit_<m> - muf_<m>| > 3 MHz` flags soundings
whose pick is wrong, and every flag on 2026-02-04 was **independently rejected
by `muf track`** on temporal grounds -- an entirely different mechanism agreeing
on both diagnosis and correction. It now fires less often, because a narrower
window makes it decline on marginal soundings instead of guessing; `track`
remains the primary outlier mechanism.

**Does not work: quality filtering.** `fitres_` separates smooth traces from
ragged ones, but that did not predict pick correctness. Filtering on
`fitres < 0.3` made agreement with `thresh` *worse* (MAE 0.302 against 0.245
unfiltered), and left the worst disagreement (7.45 MHz) untouched at every
threshold. Wrong picks tend to have excellent residuals -- the trace fits a
clean parabola, the pick landed on the wrong part of it.

**Superseded.** That was measured against the pre-fix `thresh` mask. Against the
fixed `contour` mask the same filter helps rather than hurts: 0.251 MHz over 177
soundings against 0.410 unfiltered over 251. The reference moved, so neither run
settles the question -- see section 13.

**Does not work: band-limit recovery.** This was the main hope, and it failed.
Of the soundings flagged `limited_`, none produce a usable extrapolation: when
the trace runs into the top of the sweep the nose was never reached, so the fit
declines -- correctly, but uselessly for this purpose. The 24 soundings that do
extrapolate (up to 1.81 MHz) are ones where the nose *was* resolved, which is
exactly the case that did not need help.

So section 3's out-of-band problem remains open. What might still work:

- recover the high-ray branch -- it is what would anchor the vertex. Check
  whether it is present but below threshold, or falling outside the range gate;
- fit `df/dh` and extrapolate linearly to zero rather than inverting a quadratic
  vertex: same mathematics, better conditioned, and it degrades more gracefully
  when the nose is not reached;
- accept that the instrument cannot see above 32.5 MHz and extend the sweep
  instead (`dur=305` reaches IRI's 38 MHz peak for this day).

---

## 9. Trace reconstruction -- the literature, and what it implies

Papers bearing on `muf/trace.py`, found after it was built.

**Closest to what we do**

- *Trace Extraction and Repair of the F Layer from Pictorial Ionograms*,
  Atmosphere 15:769 (2024). Noise preprocessing, "coupling noise" processing,
  then **two automatic filling algorithms** to repair gaps in the F trace. The
  nearest published analogue to our segment-then-reconstruct step. "Coupling
  noise" is a category we do not handle at all -- worth reading.
- *Optimizing vertical ionogram reconstruction: low cost and high quality with
  a pipeline*, Adv. Space Res. (2025). Compares gap-filling by linear, spline,
  **makima**, **pchip** and low-rank matrix completion, and uses smooth
  polynomial/spline fitting to compensate systematic bias.

  **Actionable:** we use a smoothing spline, which can overshoot. `makima` and
  `pchip` are shape-preserving and would not, which matters most exactly where
  it matters most -- the steep turn into the nose. Worth a comparison.

**Multi-mode separation** (what `group_tracks` does)

- OIASA: Ippolito et al., *J. Space Weather Space Clim.* 8:A10 (2018) and
  *Adv. Space Res.* 55:1624 (2015); reliability improvements in
  *Radio Science* (2016). Maximum-contrast parabola pairs; explicitly notes
  that 1F2, 2F2, 1F1, 1E, 2E and 1Es arrive together.
- US patents 11,557,079 and 12,223,582, *Iterative ray-tracing for autoscaling
  of oblique ionograms* -- note that this approach is patented.

**O/X modes -- a caveat on our 87% figure**

- Harris et al., *Radio Science* (2017): O/X separation for vertical incidence
  (2017RS006279) and for oblique (2017RS006280). A sounder sweeping the HF band
  "will typically see two distinct ionospheric returns at each frequency"
  because the magnetised ionosphere is birefringent. Separation needs a
  **polarimetric antenna**, and on oblique paths the phase difference varies
  with group delay rather than being constant.
- *Statistical and simulation study on the separation in junction frequencies
  between O and X wave in oblique ionograms*, Earth Planets Space 74 (2022).
  Davies' relations: `fX - fO ~ fH cos I` and `fX - fO ~ fH^2 / 2 fO`.
  Separation on east-west paths is sensitive to ionospheric variability; on
  north-south paths it is not.

  **Implication:** some of the modes `group_tracks` separates are O/X pairs of
  the *same* hop, not different hops. `nseg_` should be read as "resolvable
  echoes", not "propagation modes". Our `.lfs` recordings are single-channel,
  so O/X cannot be separated from them at all -- that needs polarimetry.

  Re-tested the kmeans +0.57 MHz bias against the correct *oblique* formula
  (`fH cos I` gives 0.49-0.61 MHz here, against the 0.7 MHz vertical value used
  before): still no clean bimodality -- 45% of differences within 0.15 MHz of
  zero, 19% near 0.53, smooth decay between. The permissive-cluster-boundary
  explanation stands, but it cannot be closed without polarimetric data.

---

## 10. Electron-density inversion

`muf/trace.py` now produces what an inversion needs: a continuous, single-mode
`h(f)`. Turning that into an electron-density profile `N(h)` is the natural next
scientific product, and the one thing this instrument could yield that a MUF
number alone cannot.

The standard route is a quasi-parabolic layer model fitted so its synthesised
oblique trace matches the measured one (Song et al., *Radio Science* 51, 2016,
hybrid genetic algorithm; also *Remote Sensing* 14:1671, 2022). It needs a
ray-tracing forward model, which is a real commitment -- hence not started.

Prerequisites now in place: segmented single-mode traces, a spline `h(f)` at
native resolution, hop identification (weak on this path), and an independent
foF2 reference from IRI to check the result against.

**The output format is already settled.** SAO.XML 5.0's `<ProfileList>` takes
`<Profile Type="off-vertical">` — the enumeration includes it — with the profile
as `<Tabulated>` altitudes and plasma densities, or as `<QuasiParabolicList>`,
which is exactly the representation the Song et al. route fits. `muf export`
emits everything but this element today, so an inversion drops straight in.

---

## 12. SAO.XML: two gaps and an arrangement

`muf/export/saoxml.py` writes valid SAO.XML 5.0, with three loose ends.

**`URSICode` is empty.** It is a required `<SAORecord>` attribute assigned by
the station registry, and this path has none. Getting one is a conversation with
UMLCAR, not a code change. GIRO states that non-Digisonde instruments using
SAO.XML can be integrated, but an oblique circuit is not the case they had in
mind, so the answer is not predictable.

**`<FrequencyStepping>` and `<RangeStepping>` are not emitted.** These are
standard `<SystemInfo>` sub-elements whose contents live in the specification's
Appendix B, which was not included in the copy consulted (the PDF ends at the
references, page 12). Rather than guess a schema, the sweep parameters go in a
custom `<Sweep>` element. Worth fixing if Appendix B turns up.

**`Layer="F2"` is an assumption.** `<Trace Layer=>` is required, but this
instrument cannot resolve which layer formed an echo — there is no vertical
trace to compare against. F2 is the standard assumption for the MUF-carrying
mode over 2600 km. It is overridable, and it is the one attribute in the output
that asserts something unmeasured.

Not a gap but worth recording: **no `Polarization`**. The attribute appears on
both traces in the specification's own sample, so its absence will look odd to a
consumer. It is deliberate — this receiver has no polarimetry, and the O/X
hypothesis was tested twice and rejected (section 7). The custom `Branch`
attribute carries low-ray/high-ray, which is what we actually know.

---

## 11. Smaller items

- **The kmeans +0.57 MHz bias** is a permissive cluster boundary, not O/X mode
  confusion (tested -- see section 7). Tightening `margin_db` would reduce it.
- **No CNN trainer.** The estimator runs only against a model supplied via
  `model_path`. The bundled 2022 `.h5` will not load under Keras 3 and was
  trained on different image geometry. Worth questioning whether it earns its
  TensorFlow dependency at all -- `algo` and `contour` agree to 0.41 MHz, and
  `kmeans` and `contour` to 0.17 MHz.
- **One day at a time.** No multi-day batch, no incremental "process only new
  files" mode.


---

## 13. Re-test what the contour fix invalidated

Fixing `contour`'s mask (frequency-only opening, and intersecting the retained
contours back with the cells that were actually above threshold) changed what
that estimator reports, and two findings recorded against the old version do not
survive it. Both were comparisons *to* `contour`, so the reference moved
underneath them; neither the old nor the new number settles anything.

**`fitres_` as a reliability filter.** Old: filtering `fitres < 0.3` made
agreement worse, 0.302 against 0.245. New: it helps, 0.251 over 177 soundings
against 0.410 over 251. The clean test is against something that did not move --
`track`'s Kalman residual, or the IRI/GIRO references in `muf/reference/` --
rather than against another estimator that is being changed in the same commit.

**`nseg_` as a strength proxy.** Old: monotonic, one-mode soundings the worst
(0.412 MHz) and 5+ the best (0.031). New: non-monotonic, with **two** modes the
worst case at 0.633 MHz -- twice the one-mode figure -- and the trend only
holding from three upward. The counts also shifted hard toward the low end
(77 one-mode soundings against 34). The hypothesis worth testing is that two
segments is where the splitter is most often wrong, either cutting one mode in
half or merging two; that needs labelled soundings, or at minimum a check of
whether the two-segment cases have implausible hop geometry.

**Also worth redoing:** the SAO qualifying-letter counts in section 12 and in
the README were measured with the old mask. `contour` now reaches the band edge
on more soundings (band-limited 11 against 1 for `algo`), so the `D` count will
have moved.

---

## 14. LOF: the descriptive letter B, and an absorption index

Two things the LOF work deliberately stopped short of.

**URSI descriptive letter B** -- "measurement influenced by, or impossible
because of, absorption near fmin" -- is the standard notation for a sounding
whose propagation window has closed far enough to compromise the *other*
measurements. It is not emitted, because it needs a criterion and the data does
not offer one: the usable window (MUF minus LOF at 43 dB) over 2026-02-04 runs
1.78 to 18.35 MHz with a completely smooth distribution -- p5 2.25, p25 4.56,
median 7.43, p75 15.02 -- and no break to put a threshold at. Inventing one here
is the mistake `UNCERTAIN_SNR_DB` already had to be rescued from once. A
defensible criterion probably comes from outside: soundings where the MUF pick
disagrees with `track`'s prediction *and* the window is narrow.

**An absorption index** would be better than any LOF, because it would be
threshold-independent. The physics is in ITU-R P.533-13 eq (20): non-deviative
loss goes as `1/(f + f_L)^2`, so fitting `SNR(f) = K - A/(f + f_L)^2` to the
low-frequency half of the trace should recover `A` directly.

*Tried, does not work as-is.* Fitted over nine soundings across 2026-02-04, `A`
came out between -562 and +3303 with no diurnal order and negative --
unphysical -- values on three of them. The reason is structural: every detected
point is above the 43 dB detection threshold by construction, so the rolloff is
censored exactly where the information is, and what is left is dominated by the
trace's own focusing near the nose rather than by absorption. Recovering `A`
needs *uncensored* SNR below the trace's end, which means characterising the
noise floor separately -- the peak-in-gate at frequencies with no detection is
noise, and its level has to be modelled before the transition into it can be
fitted. Worth doing; not worth guessing.

---

## 15. An ITU-R P.533 absorption reference

`muf/reference/` has no LOF counterpart, and IRI cannot be one: it models
electron density, not absorption, and exposes no collision frequency. See the
README's "External references" for the detail, including why the D region is the
part of IRI least able to support this.

The reachable model is ITU-R P.533-13 equation (20):

    L_i = (1 + 0.0067 R12) sec(i) SUM_j [ AT_jnoon / (f + f_Lj)^2 ]
              * F(chi_j)/F(chi_jnoon) * phi_n(f_v / foE_j)

with `F(chi) = cos^p(0.881 chi)` or 0.02, whichever is greater (eq 21), and
`f_L = |f_H sin(I)|` at 100 km (eq 23). Inputs already available: R12 from
`reference/indices.py`, chi from `reference/chapman.py`, foE from IRI, the path
geometry from `geometry.py`. Missing: `f_L` needs IGRF at the penetration
points, and `AT_jnoon` and `phi_n` are published as Figures 1-3 of the
Recommendation rather than as formulas, so they have to be digitised.

That yields a predicted absorption, not a predicted LOF. Turning one into the
other still needs the link budget of eq (17)-(18) -- transmitter power and
antenna gain -- so the honest output would be a modelled *absorption* to compare
against the measured LOF ladder, not a modelled LOF to compare against a
measured one.

---

## 16. One serendipitous day, owed to DOB

DOB went to `serendipitous = false` on 2026-08-12 without first spending a clean
day in search mode. That was a deliberate call -- the emitter census was already
in hand and the schedule was the thing wanted -- but it left two measurements
unpaid, and **scheduled mode is what blocks them**: `dombas.sh` only starts
`find_timings.py` when `serendipitous` is true (patch 0003's branch), so neither
`find_timings.log` nor `par-*.h5` exists any more. Reopening both costs one day
back in search mode, and they should be reopened together.

A third item was listed here and is now **answered from data already on disk** --
see "`epoch_offset_s`, resolved" below. It is worth reading before paying for a
day in search mode for anything else: the archived `par-*.h5` outlived the mode
switch, and the question turned out not to need new acquisition at all.

**1. The sounding-loss figure, against a 4.28% baseline.** `find_timings.log`'s
`s left` margins are the only measurement of how many soundings the pipeline
misses. 4.28% was measured with `calc_ionograms.py` running as a single process,
and is what patch 0004 (`-np 2`) was meant to improve. It has never been re-read
on a healthy recorder:

```bash
grep -oE '\-?[0-9.]+ s left' logs/find_timings.log \
  | awk '{n++; if($1<=0) z++} END {printf "n=%d lost=%d (%.2f%%)\n", n, z+0, 100*z/n}'
```

The `-?` matters -- without it the sign is dropped and every failure counts as a
comfortable pass. A cumulative count over the whole log is also worthless here;
count only entries past marker 15 (2026-08-12 17:06), which is where the stream
became clean. Everything before that was measured on a recording missing 6% at
the socket and up to 45% at the device.

**2. Capacity numbers that rest on a damaged stream.** Ringbuffer sizing, cycle
counts, consumer throughput and schedule margins were all derived before
2026-08-12. Nothing about slot counts or `/dev/shm` sizing should be decided
from them.

**`epoch_offset_s`, resolved (2026-08-12) -- no serendipitous day needed.**
Solved independently on two archived days and agreeing to 0.16 ms:

| day | slots | samples | offset | scatter |
|-----|------:|--------:|-------:|--------:|
| 2026-08-09 | 2 | 152 | −0.00227 s | ±0.01 ms |
| 2026-08-10 | 3 | 252 | −0.00211 s | ±0.10 ms |

Both confirm the −2.2 ms already on file, which was the thing in doubt. The
reasoning that made this look blocked was wrong in a way worth naming: scheduled
mode stops `par-*.h5` from being *written*, but the archive of what was already
written is untouched, and two days of it was plenty. **Check the archive before
booking acquisition time.**

Getting the sign right is the whole difficulty, and it is easy to get backwards
-- it was, here. The offset is applied to the *measured* arrival phase to
recover true delay, so a −2.27 ms offset makes an observed 9.27 ms into 11.54 ms
(3459 km), not 7.0 ms. Read the wrong way it rules out the correct transmitter:
Cyprus was briefly dismissed on exactly this error, on the grounds that 9.27 ms
was too short for a 3436 km path. It is, and that is why the offset exists.

**Not blocked, and worth keeping:** the drop sampler is mode-independent, so it
runs regardless of what `serendipitous` is set to. It is the open evidence for
patch 0008 (core isolation), whose validating windows all fell at load 6-8 while
the fault was measured at 9.4 -- see `docs/2026-08-11-recorder-packet-loss.md`
sec. 5 and sec. 6.

This entry previously claimed `~/drop-watch.sh` was already running on the
station and sampling every 15 minutes. **It did not exist.** The real thing is
`tools/drop-watch.sh` plus `chirp-drop-watch.{service,timer}`, installed
2026-08-12 22:59Z, logging to `/var/lib/chirp-drop-watch/drop-watch.log`. Two
documents asserted a measurement that nothing was taking; the check that would
have caught it is `pgrep -af drop-watch`.

**Already paid, do not redo:** the emitter census. Three days of `par-*.h5`
through `muf detect` gave five rate/phase groups. The schedule that came out of
it is SGO (500 kHz/s, `rep=120, chirpt=54`) and NIC (100 kHz/s, `rep=600,
chirpt=235`). Findings worth carrying forward:

- The 100 kHz/s group at ±3.50 ms was **several emitters merged**, and
  `muf detect --tolerance-ms 1.0` splits it -- the default 5 ms tolerance was
  doing the merging. Cyprus fell out of it: slots 0/235/240/280, 3459 km
  against a 3436 km ground path, and SNR 61.9, the strongest emitter DOB hears.
  The largest group in a census deserves a second pass at a finer tolerance
  before being written off as unusable.
- **Phase scatter is not the selection criterion; range is.** The 125 kHz/s
  group (±0.96 ms) looked like the obvious second schedule entry on scatter
  alone and was recommended as one. It resolves to 14,232--14,348 km -- a
  many-hop path and a poor MUF circuit. Tight scatter means the group is one
  emitter, which says nothing about whether it is a useful one. Resolve the
  geometry before scheduling.
- Two emitters have arrival phases that **cannot be propagation delays**:
  415.28 ms is 124,000 km and 863.37 ms is 259,000 km. The receiver epoch is
  shared and the other groups give sane ranges (5.46 ms -> 1640 km), so these
  transmitters start off the integer second. Whether `chirpt` wants the integer
  slot or the true sweep start including that offset is unresolved, and getting
  it wrong displaces every echo without any process reporting a fault.
---

## 17. First results from the schedule (2026-08-12)

DOB switched to `serendipitous = false` at ~20:30 UTC on 2026-08-12 with two
transmitters. The first 78 minutes:

| tx | path | soundings | picks | methods agreeing |
|---|---|---:|---:|---|
| NIC (Cyprus) | 3436 km | 4 | **12** | 0.10 MHz at 21:23:55, 1.85 MHz at 21:43:55 |
| SGO (Sodankylä) | 1013 km | 41 | **0** | -- |

Both entries are working as configured -- 41 and 4 sweeps arrived on schedule,
all `sweep_complete`. The asymmetry is in what came back, not in what was
recorded.

**NIC is the strongest thing DOB hears and it behaves like it.** SNR 40.8--52.9
across all twelve picks, `vrange` 2641--2648 km on every one of them, i.e. a
group range stable to 7 km over half an hour. Three independent estimators
converging to 0.10 MHz on a first attempt is a better result than the schedule
was expected to give. See section 3 for the reason not to trust the top of that
range.

**SGO returned nothing at all on the first night** -- not a weak pick, no
detections across 41 sweeps and three methods. **Answered the next morning: the
entry is sound and the zeros were the ionosphere.** 2026-08-13, SGO by hour:

| hour UTC | soundings | picks | mean MUF | mean SNR | mean vrange |
|---|---:|---:|---:|---:|---:|
| 00--02 | 90 | **0** | -- | -- | -- |
| 03 | 30 | 8 | 9.74 | 45.6 | 1716 km |
| 05 | 30 | 8 | 12.47 | 42.9 | 1643 km |
| 07 | 30 | 14 | 13.58 | 42.8 | 1719 km |
| 08 | 30 | 14 | 13.75 | 43.5 | 1718 km |

Nothing until 03:00 UTC, then a clean sunrise rise from 9.7 to 14.6 MHz at a
group range of 1640--1720 km against a 1013 km ground path -- a one-hop F
echo, which is what this circuit should give. `chirpt` is right, the slot is
right, and the night zeros are a 1013 km path with no layer high enough to
return 24 MHz at that incidence. **Do not remove the entry.**

**But it is a marginal circuit, and that is the finding to carry.** 52 picks
from a possible 828 is a **6.3% yield**, peaking at 15.6% in the best hour --
against NIC's 100%. Mean SNR is 39.6--45.6 dB against a 43 dB detection level:
SGO is sitting on the threshold, so it picks when the path happens to be a
decibel or two up and not otherwise. Two consequences worth deciding on:

- A 6.3% yield still costs a full MPI rank and a 120 s slot every cycle. That
  is the same cost as NIC, which returns 16x more.
- If the threshold is what is gating it rather than the propagation, a lower
  detection level recovers most of those 776 soundings. Check whether it also
  recovers 776 noise picks before changing anything.

### Other items opened by this session

- **`calc_ionograms` takes 86.95 s per cycle and nothing explains it.** This
  was attributed to a ~15 s upload retry inside the rank's budget. **That was
  wrong**: `calc_ionograms.py` contains no upload code. The uploaders are
  `ionowebsync.py` (not running) and `station_monitor.py` (running), and what
  the latter posts is a status JSON, not products. The cycle time is unexplained
  again, and it matters because it sets how many ranks a schedule can afford.
- **`station_monitor.py` posts to `http://4.235.86.214/upload.php`** -- plain
  HTTP, bare IP, no configured destination of ours. It is a status document, not
  data, but it is an outbound connection from an acquisition host that nobody
  chose deliberately. Passing `--upload-url ""` on `dombas.sh:152` disables it;
  the guard at line 354 already treats empty as off. Left running: it is the
  station owner's call, not a bug to fix unilaterally.
- **The archive mirror stalled.** Nothing reached the laptop between
  2026-08-10 23:30 and a manual download on 08-12, across the whole week of
  this work. `chirp-archive-sync.timer` is enabled. Undiagnosed, and it makes
  every product count taken here a lower bound.
- **Digisonde products yield essentially nothing.** 306 files from 08-10 gave 3
  picks; 249 from 08-12 gave 0. They stopped at 16:20--16:28 on 08-12 when
  patch 0007 removed the receivers. Worth deciding whether they are ingested at
  all, rather than leaving several hundred no-pick soundings a day in the
  database.
- **`-np` and `len(sounder_timings)` are one number in two files.**
  `calc_ionograms.py:452` indexes `conf.sounder_timings[rank]` with no guard;
  too few ranks silently stops sounding a transmitter, too many kills one rank
  with `IndexError` while the rest look healthy. Patch 0009 derives the number
  on the station, and `services/agent/control.py` refuses a schedule change that
  would break the match. Neither helps a station that has not taken 0009 --
  check by hand there.

## 18. The schedule the UI composed could not have run (2026-08-13)

Found while building the acquisition panel, not by anything failing loudly.
`calc_ionograms.py:444-447` reads **five** keys off each `sounder_timings`
entry with a bare subscript and no default -- `chirp-rate`, `rep`, `chirpt`,
`id`, `transmit_name`. `/ui/sources` composed three. A short entry is a
`KeyError` on that rank at its first slot, while the other ranks carry on and
the log looks normal.

Neither missing key can come from a census: **a detection is anonymous.** So
the fix is not a default, it is a step -- the operator identifies an emitter
once, the way `cyprus1` was resolved to `NIC`, and that identification is
stored with the census row it was made on (`transmitter` table, keyed by
receiver). Schedules are then composed by name. Both the server and
`services/agent/control.py` now refuse an entry short of any of the five; the
agent's copy of the list is deliberate duplication, because it is the last
check before the station's `.ini` and must not depend on the server being
up to date.

Three further faults surfaced only by driving the page:

- **`NaN` is not JSON, and a census row is an ordinary place to find one.** A
  group whose detections carry no SNR field gets `snr_median = NaN`, which
  Python writes as a bare token; `JSON.parse` throws on it. `/sources` was
  returning a document that says it is JSON and is not, and in the page the
  whole row was unreadable -- one absent field in a column nobody was reading
  disabled the button next to it. Non-finite floats are now `null` at the
  source.
- **Rank oversubscription is invisible on the station.** Two slots of one rank
  whose sweeps overlap cannot both be recorded -- a rank is one process. It
  takes the nearer slot and skips the other, silently. The console flags it,
  but only where a sweep length is known, which needs an ingested product for
  that receiver: sweep length is *measured* (band span / chirp rate), never
  configured.
- **`config_epoch` was specified in §5.4 and written by nothing.** It is now
  written on acknowledgement -- not on enqueue, because a queued command has
  changed nothing -- and whenever the *write* succeeded even if the restart
  failed, because the file has already changed.

Not addressed: an identification is per receiver, so a second station means
identifying the same transmitter again. That is correct (the slot second
differs per circuit) but the evidence and the name are not shared, and they
could be.

### The sources page took minutes, and it was the file opens

Reported from the deployed server: `/ui/sources` "always takes a long time,
about a few minutes". The census reads one HDF5 file per detection and cached
nothing, so **every page load re-opened every file**. Three days of DOB is
~1850 opens: 0.6 s against a local SSD, and at 50--100 ms per open on a network
archive, 1.5--3 minutes. Nothing about the grouping arithmetic was slow.

Fixed by not re-reading immutable files. A chirpsounder2 detection product is
written once and its name carries the second it belongs to, so the path is a
sound cache key; the scan is fingerprinted on the file *names*, which the
directory listing already yields, so an unchanged archive answers without a
single `stat`. Measured on the real archive: 0.63 s cold, 0.045 s warm, and one
new file costs exactly one open.

Two things worth keeping in mind:

- **The cheap files are the wrong files.** `cdetections-*.h5` holds the same
  span in 96 files instead of 1500 and loads 7x faster, which looks like the
  obvious fix. It is not: those are the detector's raw candidates, not its
  conclusions. On 2026-08-09 they yield a 100 kHz/s "emitter" with 26,137
  detections spread across nearly every second of the cycle, which the
  occupancy filter then rejects -- reading them first would lose NIC. The
  preference order is about quality and the cost is paid by caching instead.
- **One file in the DOB archive will not parse** (1846 matched, 1845 read).
  Harmless -- a detector caught mid-write -- but the count is now on the page,
  and a *steady* count is not the same thing as a transient one. Worth a look
  if it does not go away.

### The console could not say whether it was recording anything

Also reported from the deployed server: the panel needed an indicator for
whether sounding is actually *running*. Everything on it was arithmetic on the
ini against a clock -- `SOUNDING NIC` is true the second the schedule says a
chirp is due, and it stays true with the recorder dead.

The indicator is `newest_product_age_s` first, unit states second. That order
is forced by DOB: it reports `units: []`, because `dombas.sh` supervises it
rather than systemd, so anything built on unit states would read unknown
forever on the one station being watched. A definitely-dead unit still wins
when there is one -- it is a fact about now, while a fresh product age can be
fifteen minutes old.

Four states, not three. `NO PRODUCTS` (nothing arriving, nothing reporting
itself dead) is separate from `NOT ACQUIRING` because it is what the real
outage looked like: every unit green for two days with `/dev/shm` at 100%. And
a stale report reads `ACQUIRING?`, never red -- silence is the alert, but it is
the absence of evidence, and giving it the failure colour teaches the operator
to discount the failure colour.

Only `chirp-rx` and `chirp-ionograms` are consulted. `chirp-sync` and
`chirp-archive-sync` can fail for a week while the station sounds perfectly;
reporting every listed unit is the "eleven false reds" the station config
already warns about. They still show FAIL in the metrics table.

When the station is not acquiring, the schedule pills say `SLOT DUE` rather
than `SOUNDING` and the table says `due` rather than `sounding`, with the
distinction spelled out in the panel. Same arithmetic, different claim.

### It was not deployed, and that took two four-minute page loads to establish

The census fix was on `develope` and not on the server. The work server runs
`docker-compose.hub.yml`, which pulls images and never builds, so a push moves
nothing on its own -- and `watchtower` is an opt-in profile, so on a host that
never enabled it nothing pulls either. The other silent path is CI: with
`DOCKERHUB_USERNAME` or `DOCKERHUB_TOKEN` unset the publish step is skipped and
the run still goes green, leaving only a notice in the run summary.

None of that was diagnosable from outside, because `/healthz` reported
`version: "0.1.0"` -- a hand-edited constant that has read the same thing
through every deploy. The only available evidence was whether the fix's effects
were visible, on a page that takes four minutes to answer. Two loads, ten
minutes, to learn that the code was not there.

Both images now carry `API_BUILD_SHA` from CI and `/healthz` serves it as
`build`, alongside `built_at`. It reads `source` from a checkout and `unknown`
from an unstamped build; neither pretends to be a commit.

The second half: the census cache is in-process, so every container start
handed the cold read -- 234 s on the work server -- to whoever opened the page
first. That is indistinguishable from a broken page, and was read as one. The
api now does that read at startup in a daemon thread, with the parameters the
page defaults to, and prints what it cost. `CENSUS_WARM=0` opts out.

Worth noting for later: the warm-up and the page must ask the *same* question
or the warm pass is wasted -- the short-circuit is keyed on `max_days` and
`min_count`, so both now take them from `DEFAULT_MAX_DAYS`/`DEFAULT_MIN_COUNT`
rather than from two literals that happened to agree.

---

## 19. Solar indices: three more sources, and something that says whether they can be reached (2026-08-13)

**Why now.** The plan for IRI values on the sounding page rests on a
dependency the page cannot show: with no route out,
`muf.reference.indices` falls back to its cache and keeps answering, and a
driver from six months ago renders exactly like a fresh one.

**What the driver actually was.** F10.7 came from SWPC's *monthly*
`observed-solar-cycle-indices.json` and nothing else. Three files were added,
all verified reachable and parsed against a known value:

| source | carries | size | lag |
|---|---|---|---|
| `irimodel.org/indices/apf107.dat` | daily F10.7, 81-day mean, `ap`, since 1958 | 1.4 MB | ~16 d |
| `services.swpc.noaa.gov/json/f107_cm_flux.json` | daily F10.7, rolling 42 days | 23 KB | same day |
| `sidc.be/SILSO/DATA/EISN/EISN_current.csv` | estimated SSN, current month | 600 B | same day |

The two daily series are merged, newest source winning on the overlap, so the
81-day mean can be computed for dates past `apf107.dat`'s end. On 2026-08-13
that mattered: `apf107.dat` ended 2026-07-28 and `SN_d_tot` 2026-07-31.

`SolarIndices.f107` is now the observed daily flux and `f107_driver` is what a
model is handed -- 81-day mean, then monthly, then daily. They are separate
because the CCIR and URSI maps IRI interpolates were fitted on a smoothed
index; the day's flux moves 50 SFU across a rotation and the maps cannot
represent that.

**Two traps, both pinned by a test.** `ap` reaches 400 in a severe storm and
`apf107.dat`'s fields are three columns wide, so a whitespace split merges them
-- 2003-10-29 reads `400300207236179132 94 67236`. And irimodel.org runs
mod_security: it refuses urllib's default `User-Agent` with a 406, and
`curl/8.7.1` as well. `ionograms-handler/0.1` passes and
`ionograms-handler/0.1 (+solar index fetch; python-urllib)` does not, so the
constant carries a warning not to make it more informative without re-testing.

**Not added, with reasons**, so nobody re-derives them:

- `irimodel.org/indices/ig_rz.dat` (IG12 and Rz12, with predictions to 2028).
  Its header says last updated 2025-08-19, a year stale. Worse, the parse could
  not be verified: split at IRI's own `3 - imst + (iyend-iyst)*12 + imend`
  = 853, the 1958 values read as sunspot v1.0 and the cycle-24 peak (116.4 at
  2014-04) reads as v2.0. And PyIRI's `IRI_density_1day` takes F10.7, not IG12
  or Rz12, so none of it would reach the model.
- GFZ `Kp_ap_Ap_SN_F107_since_1932.txt` -- 5.5 MB for an `ap` that
  `apf107.dat` already carries.
- DRAO `fluxtable.txt` -- 2.2 MB, redundant with the two F10.7 sources above.

**The indicator.** `services/api/net.py`, the **upstream** panel on `/ui`, and
`GET /net`. It probes the hosts in `indices.SOURCES`, not the internet: a ping
to a resolver stays green behind a proxy that blocks `sidc.be` and goes red on
a host using a mirror. One `HEAD` per *host* -- three, concurrently, 4 s
timeout, daemon thread every 600 s. `/net` and `/ui` both serve the last
reading in ~1.5 ms and neither ever probes; a reading older than two intervals
decays to `unknown`, because a dead daemon thread must not leave a green light.

Reachability and cache age are separate columns. Unreachable with a fresh cache
is a model still answering correctly; reachable with `never` is a model that
has never had a driver.

### Still open

- **Cost per sounding is 0.42 s unless batched.** `IRI_density_1day` takes an
  array of hours, so a 288-sounding day is one call at 0.16 s. `services/api/sao.py`
  calls it once per sounding and memoises the whole scaling, which makes a
  revisit free and a first visit slow. A cache keyed on circuit and hour, or a
  per-circuit-day batch warmed alongside the census, would fix the cold case.

Two entries here are now closed. PyIRI **is** installed -- in the local venv
and in `deploy/requirements-api.txt` -- and the note above that it is "pure
Python; one line" was wrong: it pulls netCDF4, cftime, fortranformat and
opt_einsum, about 30 MB. And `services/` does call `solar_indices` now, through
`sao.py`'s IRI panel, so the indicator is load-bearing rather than insurance.


## 20. The sounding page: one scaling, three views (2026-08-13)

`muf export` had written SAO.XML 5.0 since long before this, and nothing in
`services/` knew about it. `/ui/sounding/{id}` showed a matplotlib PNG and the
extractions table, and there was no way to get the XML out of the server at
all. Now `services/api/sao.py` scales a sounding once and three surfaces read
it: `GET /soundings/{id}/sao.xml`, the characteristics panel, and an
interactive plot.

**What is data and what is a picture.** 486 x 3999 cells is 1.94 M numbers,
~11 MB as JSON against 164 KB as a PNG; the trace is 165-620 points. So the
raster is served bare (`?bare=true`: axes off, `bbox_inches="tight"`, no
overlay) and placed in *data* coordinates under scatter traces. Its extent runs
half a cell past the first and last sample -- `pcolormesh(shading="nearest")`
centres a cell on each -- and getting that wrong offsets every circle by half a
bin, which nobody sees until they zoom.

**plotly-basic 3.0.1 is vendored** at `services/api/static/plotly.min.js`,
1,032,507 bytes, served by a `StaticFiles` mount outside `require_read`. Not a
CDN: DOB has been off the internet for a week at a time. Still no build step,
which was the condition in `web_routes.py`'s docstring.

**`pipeline.circuit_ceiling` read `Options.stations is None` as "no registry"
rather than "the default one"** -- fixed 2026-08-14. A bare `Options()` took
NIC -> DOB's *geometry* from the built-in table and then missed its 24.53 MHz
ceiling, so the same sounding got different `D` letters from `muf export` and
from the server, which carried a workaround (`stations=resolve_stations(None)`)
to paper over it. `circuit_ceiling` now resolves the same way every loader call
does; `{}` still means no registry, and a bare coordinate mapping yields no
ceiling rather than raising. The workaround is gone, and the served XML and the
CLI both anchor NIC -> DOB at 24.38 MHz (24.53 less the three-bin margin).

**The console's stop button did nothing at all** -- fixed 2026-08-15, reported
as "I paste the control token, press STOP, nothing changes". `send()` opened
with `if (name === 'stop' && !confirm(...)) return;`, and `window.confirm`
returns `false` both when the operator cancels and when the browser suppresses
the dialog outright -- so the function returned with no request, no row and no
message. This page's own `setTimeout(() => location.reload(), 15000)` also
cancels an open dialog mid-read, which turns a deliberate confirmation into a
race the operator loses by reading too slowly. Instrumented in the rig:
`["native confirm returned", false]` while the same click sent by hand queued
`f995888d`. Only `stop` was gated, which is why start and restart looked fine.

The question is now in the page: a per-station `say-{name}` line, a second
press to confirm, and the 15 s refresh suspended while a stop is armed (with a
30 s expiry, so an unanswered question cannot freeze a health console into
looking live). Every path writes a line -- no token, rejected token, transport
failure, and the queue receipt, which says `pending until the station's agent
collects it` because a queued command is not an executed one: the row waits for
a pull that never comes if no agent is running.

Smaller things settled along the way:

- `gate=full` is a word the page's own toggle sends. `load_ion` used to read it
  as a range pair and raise `ValueError` -- a 500 on a link the interface
  offers.
- `Branch` comes back empty for most traces on an oblique circuit, so the
  legend falls back to segment colours and the frequency span. `contour` labels
  two of nine on this circuit; without the span both read "low".
- IRI goes *into* the record as `<Modeled>`, not beside it, so the panel and
  the download cannot disagree.
- `SAO_MODEL=0` turns the model off. The unit suite sets it, for the same
  reason it disables the reachability checker: a solar driver on a cold cache
  is a network fetch.

### Still open

- **The page is server-rendered, so a cold sounding costs ~1.2 s to first
  byte** (1.19 s scaling + 0.42 s IRI, memoised after). Fine for stepping
  through a day, wrong for a first visit on a slow box. Fetching the frame
  asynchronously would show a correct-but-empty plot instead of a slow correct
  one; the batching item in section 19 is the better fix.
- **Nothing draws the reconstruction or the nose fit** on the interactive plot,
  though `render.plot_sao` does. The polyline and the parabola are the two
  things an operator would most want to toggle against the points.
- **No test drives the JavaScript.** The tests check that the frame JSON is
  embedded, that every trace has a distinct legend name and that the bare PNG
  really is bare; whether Plotly then places the image correctly was verified
  by eye and by reading back `layout.images` in the browser. The same gap cost
  a working stop button (section 20, 2026-08-14): the console's control flow is
  now guarded only by string assertions on the rendered page --- that a `say-`
  box exists, that no `confirm(` remains, that the refresh stands down while a
  stop is armed. A headless browser driving one start/stop round trip would
  cover all of it and nothing else does.
- **The operator pastes the control token into every new tab.** The token in
  `deploy/.env` is the server's copy --- the value incoming requests are
  compared against --- so the browser has to present its own, and it cannot be
  rendered into the page: `/ui` is read-scope and `READ_TOKEN` unset means
  reads are open, which would put a token that stops a radio in front of
  anyone who can reach the port. `sessionStorage` is the compromise, and it is
  still one paste per tab and a secret living in JavaScript's reach.

  The fix is a session cookie: `POST /control/session` with the token once,
  the server compares it and sets `HttpOnly; SameSite=Strict; Path=/` (plus
  `Secure` behind TLS), and the control endpoints accept either that cookie or
  the bearer header the agent already uses. The token then never enters the
  DOM, `sessionStorage`, or a screenshot, and the buttons carry nothing secret.
  `SameSite=Strict` is the load-bearing half: with a cookie, a form on another
  site could otherwise make the operator's browser queue a stop, which the
  header-only scheme is immune to by construction. Wants a `DELETE` for sign
  out, an expiry (an unattended console should not stay armed all week), and a
  line on the page saying which of the two it is using.

---

## 21. The chooser moved to the console, and the warm census stopped re-reading (2026-08-15)

Both from the same report: "*i still wait too much on the sources tab, and I
think the list of the sounding stations/transmitters needs to move to the
console page -- then the user can choose stations to run and start sounding at
the same page*".

### Choosing and starting were on two pages

Ticking transmitters lived on `/ui/sources` and the start button lived on
`/ui`, so composing a schedule and running it meant two pages and a page load
between them -- and the page holding the control was the slow one, behind an
archive census the decision does not need. The schedule chooser is now a
**sounding plan** panel on the console, directly above start/stop/restart. Its
list is `db.transmitters()`, the *identify* step's output, so the console
carries the chooser without inheriting the archive read; identifying stays on
`/ui/sources`, where reading the archive belongs. The composer was deleted from
`sources.html` rather than duplicated -- two pages that could each queue a
`set_config` was one too many.

Three things the move had to preserve or fix:

- **No recorded mode is not `search`.** The select's fallback was the first
  option, which put a mode this server never observed one click from a live
  receiver -- and search mode records whatever sweeps past, so the mistake only
  surfaces later, as products that do not match the schedule. The unrecorded
  case now selects a valueless `— not recorded —` sentinel, and both the
  preview and `applyPlan` refuse it.
- **A queued schedule is not a running one.** Ticks come from
  `acquisition.scheduled`, which reads the slots the station *acknowledged*, so
  a pending command does not appear as configuration.
- **The 15 s refresh stands down while a plan is half-composed**, exactly as it
  does for an armed stop, with a 120 s expiry. Otherwise a refresh mid-choice
  puts every tick back the way the server has it, silently.

Verified end to end on the local rig: apply queued `198cdc8a — 2 rank(s) for
NIC, SGO`, the row carried `{"mode": "scheduled", "sounder_timings": ...}`, and
after an agent-style ack the reloaded page showed both boxes ticked with mode
`scheduled`.

### The warm census was not warm

Section 18 made `/ui/sources` cache every file it opens; the warm load should
then open nothing, and two things kept spending the saving.

- **`Path.resolve()` in the finders' dedupe is a `realpath` per file.** Used
  only as a "have I seen this path" key, it cost 38 ms for 1368 files locally
  against 0.4 ms for `abspath` -- for a directory listing that cost 3 ms -- and
  on the network archive it is a round trip each, the exact cost section 18 set
  out to remove. Now `muf/paths.py:dedupe_paths`, shared by `find_lfs`,
  `find_h5`, `io_detect._find` and `find_soundings`. What is given up: two
  different paths reaching one file through a symlink no longer collapse, which
  no caller relies on. (Its own module because `io_chirp` -> `calibrate` ->
  `io_lfs` already form an import chain -- putting it in `io_chirp` gave a
  circular import.)
- **One unreadable file disabled the short-circuit entirely.** The test was
  "did the last census skip anything at all", and a live archive always has
  something: the detector is always writing. The one truncated file out of 1846
  noted in section 18 turned every later page load into a full re-read and
  re-group, for the life of the process -- the cache was off precisely where it
  was needed. Only the paths that actually failed are re-stat-ed now.

Warm census 43 ms → 9.1 ms, warm page 45 ms → 10 ms on the local checkout. The
work removed is per-file system calls, which is where the server's minutes go.

### Still open

- **No test drives this JavaScript either**, same gap as section 20: the plan
  panel is covered by string assertions on the rendered page (the sentinel
  option, the pre-ticked codes, that `applyPlan` refuses an empty mode) and by
  hand in the rig. The preview arithmetic and the refresh hold are not.
- **The `-np` the plan states is advice, not an action.** The page says how
  many rank groups a schedule needs; making the launcher match it is still a
  human editing a command line on the station, and the agent's refusal is the
  only thing standing between a mismatch and a silently short schedule.

## 22. The series page: four parameters and a reference (2026-08-15)

From "*upgrade the series tab — analyse not only MUF but also LUF, foF2, IRI
results, interactive, plotly*". `/ui/series` drew one number as a hand-rolled
SVG scatter: MUF, one circuit, hover text and a click-through, nothing else.
It now draws MUF, LOF, an equivalent foF2 and IRI over the same axis, with a
residual panel under it and a summary table under that, on the plotly already
vendored for the sounding page. New module `services/api/series.py` builds the
frame; `web_routes.series` only queries and hands it over.

**LOF, not LUF, and the page says why.** P.533-13 §9's lowest *usable*
frequency carries a required S/N and a monthly median — a property of a service
and of a month, and one sounding has neither. `muf/lof.py` had settled this
argument long before; the page now inherits it rather than re-opening it in the
UI vocabulary.

**A sounding with a LOF and no MUF is a point now.** The query was
`WHERE e.muf IS NOT NULL`, so a trace that faded out before it reached the
ceiling was not on the chart at all — which reads as "nothing was recorded"
rather than "the top was never seen". Widened to `muf IS NOT NULL OR lof IS NOT
NULL`, and the same condition drives the circuit chooser and the day pills so
that a choice on offer always draws something. On this archive that is
Juliusruh → DOB for 2026-08-10: 120 soundings, one MUF between them, and the
page used to show none of it. LOF picks outnumber MUF picks roughly two to one
across every method here (895/601 `algo`, 1186/709 `contour`, 1198/710
`kmeans`), so this is most of what was invisible.

**Bounds are drawn at both ends and held out of the statistics.** Hollow MUF =
pinned to the top of the sweep, hollow LOF = pinned to the band floor. Both are
plotted — dropping either bends the curve towards the middle of the band, which
is section 3's argument run in both directions — and both are excluded from the
bias and RMS, with the excluded count printed next to the used one. Scoring a
ceiling-limited pick as a residual reports the *recorder's* ceiling as a
modelling error.

**foF2 is inverted over one hop, and `saoxml` now follows.** The measured MUF
goes back through the secant law at `EQUIVALENT_HMF2_KM` = 300 km over
`path_km / hop_count(path_km)` — the same convention `iri.predict` converts by,
so the measured and modelled foF2 curves sit on one geometry. The SAO export
inverted over the whole path until this was noticed here; it would have agreed
with the page on every circuit in this archive and disagreed on the first one
over 4000 km, which is the worst way for two numbers to differ. Its
`ModelOptions` now says which distance it meant —
`hmF2=300km,hop=2919km,D=5837km,2 hops`, with the path and the count present
only when they add something — because `D=` alone cannot be read as a hop.

**The model is called once per day, not once per window.** `iri.predict` reads
its solar driver off `index[0]` alone; the day pills make multi-day windows
normal, and one F10.7 across February and August would be wrong with nothing on
the page to show it. Per-day calls cost nothing extra — PyIRI already evaluates
a whole day per call — and the reported source counts the distinct drivers
rather than printing thirty near-identical sentences. Past `MAX_MODEL_DAYS`
(31) the page declines in words. A day that fails does not take the others with
it. Memoised on (endpoints, instants) at `CACHE_SIZE` = 8.

**No NaN crosses into the template.** Every array goes through `|tojson`, and
Python writes a bare `NaN` that `JSON.parse` refuses — one absent pick would
blank the entire plot with the reason visible only in a console nobody has
open. `_finite()` maps everything unusable to `None`, and a test asserts
`json.dumps(frame, allow_nan=False)`.

Smaller things settled along the way:

- **The right-hand hmF2 axis is hidden while its trace is.** Auto-ranged over
  nothing it read 0–4 beside a frequency plot, labelled km, and invited someone
  to believe it. It now follows its trace through `plotly_restyle`, so a legend
  click works as well as the checkbox.
- **The family checkboxes are unchecked in the markup and set from the script.**
  Two places deciding the first view is two places to disagree, and the boxes
  would then lie about what is on the axes. With several circuits overlaid LOF
  starts off: five paths times two parameters is ten curves before the models.
- **`SERIES_MODEL=0`, and `model=off` in the query string.** Same knob shape as
  `SAO_MODEL` and the same reason. The unit suite turns it off in the `client`
  fixture, because seeding a sounding with real coordinates is the natural
  thing to do and would otherwise put a solar-index download inside a test.
- **A circuit with no coordinates says so.** `unkown -> DOB` is 647 soundings
  with no transmitter position; the model needs both ends for a control point,
  and an empty panel would read as "IRI agrees with nothing" rather than as
  "IRI was never asked". Its stored `path_km` of 20015 km — half a
  circumference, an artefact — is not printed either, because the geometry is
  read as a pair or not at all.

Verified on the local rig against `data/ionograms.sqlite3` (2298 soundings,
8 circuits, 9 days). cyprus1 → yoshkar-ola for 2026-02-04, `kmeans`:
**r = +0.985** over 101 pairs, median bias **+1.63 MHz**, RMS 2.41 MHz, 13
lower bounds held out — and the residual panel shows what the number hides,
IRI low through the morning rise and ~5 MHz high after it. That is a diurnal
disagreement, not a scale one, and it is the first thing this interface has
been able to say.

### Still open

- **Still no test drives this JavaScript**, the same gap as sections 20 and 21.
  The frame is asserted as parsed JSON out of the rendered page — which is
  better than a string match and covers the data — but the trace assembly, the
  family toggles and the axis sync were checked by hand in the browser.
- **The residual panel's y-range is driven by the worst circuit** when several
  are overlaid, so a path IRI models badly compresses the ones it models well.
  A per-circuit scale would fix it and would also stop the panel being readable
  as one comparison, which is the harder call.
- **`/series/muf` still returns MUF alone.** The page's extra columns are not
  reachable as JSON, so anything scripted against this archive re-derives the
  foF2 and the residual itself, from a different set of assumptions than the
  page states. A `/series/parameters` returning the frame would settle it.

---

## 23. Stopping DOB is a manual sequence, and its evidence is truncated (2026-08-16)

Two findings from stopping acquisition on DOB by hand on 2026-08-15/16, both
about the same thing: this station is supervised by a shell script, and the
tooling built around it assumes systemd.

### `thor.log` is truncated on every restart, so drops leave no trace

The supervisor in `patches/0003-local-dombas.sh.result:73` runs the recorder in
a `while true` loop, redirecting its output with a single `>`. **There is no
24-hour timer in that loop.** It restarts the recorder whenever
`rx_uhd_ext_gps` exits, for any reason, and prints `Restarting recording (every
24 hours).` every time -- so a clean 24 h rotation and a crash after ten
minutes are indistinguishable in `dombas-launch.log`, and the recorder's own
output that would tell them apart is overwritten by the next launch five
seconds later.

Observed 2026-08-15: the two supervisor shells had been up **2 d 10 h** with
exactly **two** restart lines logged, while the running recorder was **10
minutes old**. Two exits cannot span 58 hours and leave a ten-minute-old third
run, so at least one run was nothing like 24 hours -- and there is no way to
find out which, or why. `logrotate` runs in the same loop but leaves no
`thor.log.1`, so nothing was ever archived either.

This is the mechanism that would produce section 16's 4.28% sounding loss with
every process reporting healthy: the recorder exits, the loop revives it five
seconds later, and the only record of it is a line that also appears when
nothing is wrong. Changing the redirect to `>>` -- one character -- would make
the next restart leave evidence. Rotating on size, so it cannot grow without
bound, is the other half.

### The console's stop button cannot stop this station

`control.py` refuses with "no systemd target configured for this station"
because DOB's `target` is empty, which is correct and deliberate --
`systemctl stop chirp.target` cannot reach another supervisor's children. But
it means the only way to stop DOB is a by-hand sequence that exists nowhere in
the documentation, and it has a footgun in it:

1. List the processes. **Two** processes match `bash ./dombas.sh` -- the outer
   script and the `while true` subshell -- and both must die before the
   recorder, or the loop revives it in five seconds.
2. Kill those two **by explicit PID**. Never `pkill -f dombas.sh`: on both
   2026-08-13 and 2026-08-15 that pattern also matched PID 14016,
   `git diff examples/marieluise/dombas.sh`, a pager idle since Aug 9 -- and it
   sorted first.
3. Only then stop the recorder, with **SIGINT** (`pkill -INT -f
   rx_uhd_ext_gps`). TERM or KILL leaves the USRP transmitting UDP and needs a
   physical power cycle in Dombas.
4. Verify past the five-second revive window.

Nothing else in the tree touches the radio: `drf ringbuffer`, both `mpirun`
groups, `station_monitor.py`, `sync_iono_data.py`, `iono_housekeeping.py` and
`detections2metadata.py` survive as orphans and are safe to leave or to kill by
PID. The ringbuffer holds ~14 GB in `/dev/shm` until it is.

This belongs in `deploy/README.md` as a named procedure. The better fix is the
migration to `services/agent/systemd/`, after which the console's button would
work and none of the above would be needed.

**Written up 2026-08-16 as `deploy/migrate-dob-to-systemd.md`**, with the
sequence above as its section 2. Three things came out of writing it that were
not visible before:

- **The agent calls `systemctl` bare, as `ionouser`.** Nothing in this repo
  grants that user the privilege, so the migration as previously imagined ends
  with the console failing on `Interactive authentication required` instead of
  on an empty target -- a different sentence and the same dead button. Ubuntu
  16.04 ships polkit 0.105, which reads `.pkla`; the `.rules` format every
  current answer shows needs 0.106 and is silently ignored here. Same shape as
  the `+` prefix in `chirp-rx.service`.
- **`chirp-drop-watch` fails green.** `tools/drop-watch.sh` counts `D`s in what
  `logs/thor.log` grew by; under systemd that file stops being written and the
  script reports zero drops forever. There is no unit-file fix on 229 --
  `StandardOutput=file:` needs 236, `append:` needs 240 -- and wrapping
  `ExecStart` in a redirecting shell would send the `SIGINT` to the wrapper
  rather than to the recorder. It has to read the journal instead.
- **`launcher` must be repointed at `chirp-ionograms.service`.** `set_config`
  validates the schedule's rank-group count against the `-np` it text-scans out
  of the launcher; left pointing at `dombas.sh` it would guard a script that no
  longer starts anything.

### The migration re-enabled the digisonde receivers, undoing patch 0007

Found on the live station 2026-08-16, after the cutover, by the operator
noticing digisonde rows still arriving. Section 1 of the runbook said to enable
four `chirp-digisonde@` instances, copied out of `_units_when_migrated` in
`deploy/station-dob.json.example`. Neither file had been checked against
`patches/0007`, whose whole subject is that those receivers **are** the 45%
sample loss: five of them cost ~969 dropped events/s and ~65,000
`RcvbufErrors`/s, and removing them gave zero drops over a full hour.

Three things made it survivable only by luck:

- **Nothing on the station reports it.** `chirp-drop-watch` fails green (above),
  so the one metric that would have shown 969/s answers zero for ever. Every
  unit reads `active` and the console says HEALTHY.
- **The unit list and the enabled set are separately edited.** Stopping an
  instance still named in `agent.json` turns it red, so backing the change out
  is two edits, and doing only the first is a visible failure that invites
  putting the receivers back.
- **A config example is executable.** This file is copied to the station as
  `agent.json` and its lists are read as instructions. Prose in it saying
  "these are optional" would not have helped; the list is the interface.

Fixed in both files, with `test_the_station_example_asks_for_no_digisonde_receiver`
so the list cannot regrow silently. The `chirp-digisonde@.service` template
stays — a station with cores to spare can still express them, and its header
comment is now the only place that names an instance.

Two follow-ups this leaves open. The template's own comment still argues *for*
running them ("a free extra circuit with a named, registered transmitter") and
says the sounders are "a few hundred km away" — true from Dombås (864–1360 km),
not from Yoshkar-Ola, where the same four are 2014–3169 km out. And with the
site corrected, whether any digisonde circuit is worth a core is a question
nobody has re-asked; every digisonde row on the console currently picks 0/3.

## 24. DOB's archive is 93x the census design point, and nothing prunes it (2026-08-16)

`/ui/sources` on the work server stopped answering. Not slowly -- at all: the
startup warm-up began at 19:06:28Z, never printed its completion line, never
warned, and held `_CENSUS_LOCK` for the rest of the process's life, so every
request for the page queued behind a read that was never going to finish. The
container was healthy the whole time (`restarts=0 oomkilled=false`), which is
exactly the failure mode that gets diagnosed as "the page is broken".

Ruled out first, because each had a plausible story: the in-process cache
(never restarted), multiple uvicorn workers (one process), the whole-tree
fallback in `_day_directories` (twelve date-named directories present, all
parsing), and a stale build (the deployed commit already carried the caching
fix from section 21).

It is scale. The three newest days under `/archive`:

    2026-08-13    68,288 files
    2026-08-14    57,332
    2026-08-15    46,436
    ------------------------
                 172,056 files

and 2026-08-15 broken down by prefix:

    chirp-*.h5           45,602
    lfm_ionogram-*.h5       750
    cdetections-*.h5         84
    par-*.h5                  0

The census reads `chirp-*.h5` first by preference, and that preference is about
quality -- see the docstring, and section 18's 100 kHz/s phantom. So it faced
~168,000 HDF5 opens at 50-100 ms each on a network archive: **two to five
hours**. The number this was designed and measured against was **1,846 files
over three days, 234 s cold** -- 93x smaller.

**Done in this cut:** `DEFAULT_MAX_FILES = 2000` in `services/api/sources.py`.
The census spends the budget newest day first and reads the newest files of the
product it already chose, so it trims *time* and never falls back to the cheap
files; `cost` gained `found`, `capped` and `budget`; the page carries a
`capped` notice saying it read part of the archive and which part; the warm-up
prints a line before it starts and on failure, so "still reading" and "died in
the thread" are no longer the same silence. Five tests in `tests/test_api.py`.

That is a ceiling, not a fix. Three things behind it are the station's:

### Nothing is pruning the archive

45,602 detection files in one day is not a detector working, it is a detector
firing on noise and nothing clearing up after it. `iono_housekeeping.py` is
running -- as an *orphan* from a launch two days old, alongside the other
processes section 23 lists -- and the archive still grew by 46k files that day.
Whether it is running with the wrong retention, against the wrong directory, or
losing to the write rate is unknown. There is also an unresolved duplicate pair
here: `iono_housekeeping.py` (the script) versus `chirp-archive-prune.service`
(the unit), the same split as `sync_iono_data.py` versus
`chirp-archive-sync.service`.

### `epoch_offset_s` is dead on DOB

**Zero `par-*.h5`.** The timing solutions are the product that carries the
receiver's clock offset, and they are the only external check on it -- the
check that caught the 0.956 s error. `find_timings.py` is absent from the
launcher's process list, so nothing is writing them. Until it is back, a
recurrence of that fault is invisible: the files a mis-clocked receiver writes
are internally perfect.

### 750 ionograms against 45,602 detections

One product of the day's soundings, one for every 61 detection files. Worth
knowing whether the detector's threshold moved, or whether this is what an
unpruned ringbuffer looks like from the outside.

The design point should also stop being implicit. A census over a live station
should be bounded by the archive it will actually meet, and the honest long-term
answer is an index -- one pass that records what each file contributed, so the
page reads a summary and not the archive.

## 25. The archive mount lists at 6.3 ms per directory entry (2026-08-16)

Section 24's ceiling was deployed as `0b8c488` and **the warm-up still never
finished** -- twice, across two container starts, the new "reading up to 2000
detection file(s)" line printed and no completion line followed. Since only
2000 files can now be opened, the time was not in the reads.

Measured inside the container:

    python -c "os.scandir('/archive/2026-08-15')"    46,436 entries, 293.8 s

**6.3 ms per directory entry.** Not a disk -- that is a round trip each, on a
mount. Listing the three newest days costs ~15 minutes before the first HDF5
file is opened, and it is paid on every census, including the ones that would
have answered from cache, because the fingerprint is built from the listing.

Two changes followed, in `b5d087d` and this cut:

- **One directory pass per day, not one per product.** `find_timings`,
  `find_detections` and `find_cdetections` each walked the tree. On DOB, which
  writes no `par-*.h5`, the first walk visited all 45,602 `chirp-*.h5` in the
  day to return an empty list, and the second walked them again.
  `io_detect.find_products` does one `os.walk` and buckets on the prefix
  before the first `-`, so a `Path` is built only for a file that is wanted.
  3x less scanning -- which on this mount still leaves ~5 minutes.
- **The request path no longer touches the archive.** `block=False` serves the
  last completed census with an `age_s`, starts one background refresh past
  `DEFAULT_MAX_AGE_S`, and reports `building` when nothing has finished yet.

Also fixed here: the ceiling was keeping the wrong 2000 files. It sorted on the
whole filename, but a name is `chirp-<channel>-<rate>-<i0>-<unix>.h5` and `i0`
is a sample index of no fixed width -- `9000` sorts after `44664265260000000`
on the leading digit, so the order was over channel and sample index with time
as a tiebreak. `sources._file_time` sorts on the trailing field.

### What is still unresolved

The page now renders, but every refresh still costs ~5 minutes of listing, and
a 2000-file ceiling on a 45,602-file day covers about the newest 40 minutes.
Three ways out, cheapest first:

1. **Prune.** At the 1846-file design point a listing is ~12 s. This is the
   station's problem and section 24 has it: `iono_housekeeping.py` runs as an
   orphan and the archive grew by 46k files that day anyway.
2. **Find out what `/archive` actually is.** 6.3 ms per `readdir` entry is
   pathological even for NFS. If it is sshfs, rclone or an S3-backed FUSE, or
   if it is the live rsync target from DOB and contending with the write
   stream, the fix may be a mount option rather than any code here. Nobody has
   looked; `df -T /archive` and `mount | grep archive` would say in a second.
3. **Census at the station.** The files are on local disk there. The agent
   already posts health on a schedule; posting a census alongside it would
   make this page a database read like the console's list already is, and the
   archive mount would stop being on the path at all. This is the same shape
   as the index proposed at the end of section 24, and it subsumes it.

## 23. Two measured wins, and a benchmark to keep them (2026-08-16)

Profiled the whole read path and the extraction pipeline against the real
archive rather than guessing. Most of what looked slow was not, and the two
things that were are both one-line fixes.

**What the profile actually said.** Every warm page answers in single-digit
milliseconds and the database is 1.6 MB, so there is no query work worth
doing. Three cold numbers stood out -- the sounding page at 2417 ms, the
all-circuits series at 1357 ms, one day plus IRI at 374 ms -- and none of them
are computation. They are first use: importing scikit-learn costs 1.19 s, and
PyIRI's first read of its CCIR/URSI coefficient files 0.87 s. Warm, an IRI day
is 70 ms. Reading an 80 MB `.lfs` takes 6-12 ms, about 3 % of what that
sounding costs to process, so the archive I/O is not on anyone's critical path
either.

**Threads, not cores, were capping the pipeline.** Eight workers on ten cores
were only 2.9x faster than one. That looked like a memory wall and was not:
pool start-up accounts for 0.80 s of it, and the rest was every worker process
opening its own BLAS and OpenMP pools -- up to eighty threads over ten cores.
`pipeline.PIN_THREADS` now holds each worker to one math thread for the
duration of the pool. Over one day of the archive, 133 files and 10.6 GB:
13.5 s to 8.7 s, 9.8 to 15.3 files/s, 2.9x to 4.7x. Re-measured since over a
40-file sample: 4.94 s to 3.26 s. Every pick identical, which is the only
reason it is on by default; `MUF_PIN_THREADS=0` turns it off. Deliberately not
applied at `jobs=1`, where the threads are free parallelism and taking them
away cost 5 %.

Set through the environment rather than a pool `initializer` because these
libraries read their thread counts once, as they load, and a worker imports
NumPy on the way to unpickling its first task -- before any initializer of ours
could run. Counts an operator has already set are left alone, and the
environment is restored when the pool closes.

**Nothing was compressed.** Every response went out `identity`. `GZipMiddleware`
at `minimum_size=1000` takes the vendored plotly bundle from 1008 to 348 KB, a
500-sounding listing from 237 to 7.0 KB, the all-circuits series page from 122
to 21 KB, and the soundings table from 74 to 3.5 KB. Confirmed over real HTTP,
not just in a test client. This is the largest single saving in the project and
it is one line.

**`tools/benchmark.py`** keeps both honest. It samples the archive with a fixed
seed, runs it serially and across workers, times the served pages if given a
`--url`, and writes JSON to use as a later `--baseline`. Absolute timings from
one box say nothing about another, so the verdicts are drawn from ratios that
travel -- parallel speed-up, the share of a sounding spent reading it, RSS
growth across files -- plus a same-machine diff against the baseline. It also
records the picks, so a run can prove a speed-up did not move a measurement.
Verified by running it unpinned, where it correctly reports the 2.16x and says
where to look.

### Still open

- The comment in `muf/pipeline.py` used to claim the FFTs dominate. They do not:
  inside the 172 ms spectrogram the per-frequency `np.median` costs 80 ms
  against the transforms' 54 ms. Docstring corrected, but the median itself is
  untouched and is now the hottest line in the pipeline.
- Two obvious follow-ons were measured and rejected. Batching the per-frequency
  FFT loop into one 2-D transform gains 16 % and costs ~160 MB per worker, on
  top of the 303 MB one already holds. Subsampling the noise-floor median is
  6x faster and shifts the floor by up to 24.8 %, which moves real MUFs. Both
  should stay rejected unless the trade changes.
- The heavy imports are still paid by whoever loads the first page. The lifespan
  already warms the census in a daemon thread and could warm these on the same
  one; it would move ~2 s off the first visitor and change no median, so judge
  it on p99.
- The static mount sets an ETag but no `Cache-Control`, so every page load still
  spends a round trip revalidating a 1 MB bundle that never changes.

## 26. The station rename is done in the database and not on disk (2026-08-16)

`tools/relabel_station.py` moves a receiver's name across `sounding`,
`extraction.hops`, `config_epoch`, `transmitter`, `health_report` and
`command`, and — the part that matters — recomputes the `rx_lat`/`rx_lon`/
`path_km` the name decides. Written for `DOB` → `Yoshkar-Ola`, where the stored
distance was wrong by 848 km on the Nicosia circuit and every foF2 on it was
divided by an M-factor of 3.353 instead of 3.145.

Three things it deliberately leaves, each of which is a real open question:

**The old name is still on disk.** It is in the filename and in the h5's own
`station_name`, and `io_chirp.read_header` prefers the file's attribute over
everything else — correctly, since the file records what was actually written.
So re-ingesting any pre-rename product puts `DOB` back on that row, and the
archive has ~6977 of them. Nothing currently prevents that, and the ingest path
gives no warning when it happens. The options are a rename map applied at
ingest, rewriting the attribute in place across the archive, or accepting that
a re-ingest needs the tool run again afterwards. None has been chosen.

**`gate_lo`/`gate_hi` still describe the old geometry.** They record the window
the estimators actually searched, so they are honest as history and wrong as a
description of where the echo should have been. `calibrate.default_gate` would
now choose differently on every renamed row. The fix is re-extraction, which is
a different job with a different cost.

**The `reference` rows are stale by ~1000 km.** IRI is evaluated at the control
point, the control point is the path midpoint, and the midpoint of
Nicosia→Dombås is not the midpoint of Nicosia→Yoshkar-Ola — README sec. on
control points has the two positions, 49.22N 24.59E against 45.99N 39.09E. The
GIRO station chosen by proximity moves with it, from a European sounder to
RV149 Rostov. `--drop-reference` deletes them so they get recomputed; it is not
the default, because deleting a user's data as a side effect of a rename is not
a tool's call to make. Recomputing them is the batch-IRI-per-circuit-day job
that is already on this list.

And one that is not the tool's problem but surfaces here: `Station.band_ceiling_mhz`
is keyed by receiver code, and this receiver now has two eras under one key —
24.53 MHz measured on the v2 24.825 MHz sweep as `DOB`, and no entry for
`yoshkar-ola` because the `.lfs` era's 32.49 MHz sweep genuinely was its own
limit. After the rename both collapse onto `yoshkar-ola`, and `(("DOB", 24.53),)`
on the `NIC` entry stops matching anything. Keying a ceiling by receiver alone
stopped being sufficient the moment the rename landed.
