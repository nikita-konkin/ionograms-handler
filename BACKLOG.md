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

---

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

- **GIRO / DIDBase.** Viable: the path's control point is 45.88N 39.45E and
  station **RV149 ROSTOV** sits **148 km** away, well within the ionospheric
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

**Not blocked, and worth keeping:** `~/drop-watch.sh` on the station samples
both recorder loss counters against the load average every 15 minutes and is
mode-independent. It is the open evidence for patch 0008 (core isolation), whose
validating windows all fell at load 6-8 while the fault was measured at 9.4 --
see `docs/2026-08-11-recorder-packet-loss.md` sec. 5.

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