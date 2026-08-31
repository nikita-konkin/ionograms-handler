# `algo` was reading low, and it was the picker's fault

**2026-08-30.** Asked to improve ionogram segmentation quality, with a neural
network as the presumed answer. The network is the *third* thing to try, and
this document is why.

The measurable problem: on 3,421 soundings where both ran, `algo` and `contour`
differ by more than 1 MHz on **11.1%** — and on **every one of those 379**,
`algo` is the lower of the two. Not 90%, not 97%. All of them. A disagreement
that never changes sign is not scatter; it is one estimator systematically
truncating.

Two fixes went in today. They cut the disagreement rate in the affected hours
by nearly half without any training data, any labels, or any network. The rest
of the list is recorded here in the order it should be attempted.

## 1. There is no ground truth, and that shapes everything

Nobody has drawn the true trace on a single sounding in this archive. The
digisonde receptions are this station's own oblique pickups, not the far end's
ARTIST products, so they carry no scaled characteristics either. **This is the
single most important constraint on the whole problem** — it is what makes the
supervised approach expensive and what makes it last on the list.

What is available instead is three independent extractors reading the same
array. Where they agree the sounding is probably easy; where they disagree at
least one is wrong. Not ground truth, but enough to localise the failures — and,
as it turned out, enough to identify *which* extractor was wrong and *why*.

| pair | n | MAE | bias | >1 MHz apart |
|---|---:|---:|---:|---:|
| `algo` vs `contour` | 3,421 | 0.436 | −0.436 | 379 (11.1%) |
| `algo` vs `kmeans` | 3,410 | 0.414 | −0.351 | 354 (10.4%) |
| `contour` vs `kmeans` | 3,482 | 0.234 | +0.066 | 190 (5.5%) |

MHz, band-limited picks excluded. `algo`'s MAE and its bias **are the same
number to three decimals** — the disagreement is entirely one-directional.
`contour` and `kmeans` agree about twice as closely with each other as either
does with `algo`, which is what points at `algo` rather than at the pair.

## 2. It fails at the terminator, and short runs predict it

Disagreement peaks at **30.7% at 20 UTC** and collapses to **1.4% at 15 UTC** —
a twenty-fold swing, with maxima either side of the evening terminator where
trace geometry changes fastest. A second peak follows just after midnight
(20.5% at 00 UTC).

The strongest single predictor is run length, not brightness:

| soundings grouped by the shorter winning run | n | disagree >1 MHz |
|---|---:|---:|
| run ≤ 8 frequency bins | 959 | 31.7% |
| run > 8 frequency bins | 2,462 | 3.0% |

Median run length is 43 bins where the two agree and 11 where they don't. And
the median SNR difference between the agreeing and disagreeing populations is
**1.3 dB** (51.9 against 50.6) — nothing. **The problem is continuity, not
signal.** That one comparison is what rules out "denoise harder" as a direction.

## 3. A hypothesis worth recording because it was wrong

The obvious explanation is the MUF nose: as the trace approaches the critical
frequency it turns sharply upward, so a detector expecting a flat feature would
lose it exactly where the MUF is — and would always read low, which fits the
100% figure perfectly.

**It does not survive the data.** Where `contour` reaches further in frequency
than `algo`, it climbs a median of **−1 km** of virtual range doing so — a slope
of about 0 km/MHz. Only **1 of 379** disagreements exceeds the 150 km/MHz that
`pick.py` already treats as the steepest a real trace may run. `contour` is not
following the trace up a nose; it is following it *sideways*, along a flat leg
that `algo` has dropped.

This mattered: the nose hypothesis would have sent the work into slope handling
and parabolic fitting, which is the wrong half of the code entirely.

## 4. The truncation happens in the picker, not in detection

Running `algo`'s own `detect()` on the four worst soundings settles it. On the
21 Aug 19:29 sounding, detection reaches **28.45 MHz** — and the stored MUF for
that sounding is **18.75 MHz**. Nearly ten megahertz of detected trace is
discarded downstream, by `pick_muf`'s requirement of five consecutive frequency
bins with slope continuity.

The high-frequency detections exist. They are simply too fragmented to form a
run. `contour` dilates along the frequency axis, which closes those gaps before
its picker ever sees them; `algo` had no bridging step at all. The same power
array reaches both extractors, and that dilation is the only structural
difference between them.

A second, smaller contributor sat in `detect()` itself: it marked a cell only
when **both** of its range-neighbours were lit. An oblique trace is often one or
two range bins tall — on these four soundings, 40–57% of lit frequencies carry a
trace ≤2 bins thick. This is the same class of defect `contour.py` already
documents and fixed for itself on 4 Feb 2026, when its 3×3 opening was erasing
thin traces and the kernel became frequency-only. `algo` never got the
equivalent fix.

## 5. What was done (2026-08-30)

**Fix A — bridge gaps along frequency before the picker sees them.**
`find_runs()` in [muf/pick.py](muf/pick.py) gained a `bridge` parameter:
`DEFAULT_BRIDGE = 2` undetected bins a run may skip. Merging happens *before*
the min-run test, and `run_len` counts **detections rather than span width**, so
a bridged run cannot claim evidence for the bins it skipped.

The slope test still applies across the bridge: `split_on_range_jumps` carries
`last, since` and compares against `tolerance_km * since`, so a gap of two bins
allows twice the range movement rather than an unbounded jump. Bridging decides
only whether bins may be *considered* together — it never launders a range jump.

One consequence found by test rather than by inspection: a split landing *inside*
a bridge leaves a span whose last bin holds no detection, and `percentile=100`
takes the highest bin in the span. That would report a MUF at a frequency the
sounder saw nothing at, biased upward by up to `bridge` bins. `pick_muf` now
restricts its candidate set to detected bins. It never fired on the 400-sounding
sample, but it is a wrong answer rather than a worse one, which is the kind that
should not wait to be observed.

**Fix B — either neighbour, not both.**
`detect()` in [muf/extractors/algorithmic.py](muf/extractors/algorithmic.py)
counts lit range-neighbours and takes `>= neighbours`, with
`DEFAULT_NEIGHBOURS = 1`.

`legacy=True` restores the historical behaviour exactly (`min_run=1, bridge=0,
neighbours=2`), because every `.lfs` result to date was produced with it.

### Measured, paired, on 400 real soundings

Against `contour` as comparator — not as truth, see §7:

| variant | MAE | >1 MHz | mean MUF |
|---|---:|---:|---:|
| before — both-neighbours, no bridge | 0.205 | 3.0% | 21.24 |
| fix B alone — either-neighbour | 0.134 | 1.8% | 21.31 |
| fix A alone — bridge=2 | 0.154 | 2.8% | 21.29 |
| **after — both, the new default** | **0.119** | **1.8%** | 21.33 |

Paired change in |error|: **−0.086 MHz (95% CI −0.145 .. −0.041)**,
distinguishable from no change. **88 picks moved up, 0 moved down**, 312
unchanged, median move +0.15 MHz. The one-directional result is what it should
be if the diagnosis was right: the fixes only ever extend a truncated trace.

And it moved where it was supposed to:

| | n | before | after |
|---|---:|---:|---:|
| terminator hours 18–01 UTC | 119 | 9.2% | 5.0% |
| all other hours | 281 | 0.4% | 0.4% |

The gain is concentrated exactly where §2 said the failures were, and the quiet
hours did not move at all. A change that improved everything uniformly would
have been evidence of a different mechanism than the one diagnosed.

## 6. What to do next, in this order

**a. Extract one continuous trace by dynamic programming** — **built
2026-08-30, and it does not yet beat the fixed bridge.**
[muf/extractors/viterbi.py](muf/extractors/viterbi.py), registered as `dp`,
**not** in `DEFAULT_METHODS`.

Every other method decides cell-by-cell and hopes a run emerges. The physics
says the trace is a single connected curve whose range varies smoothly with
frequency, which is a shortest-path problem over the array with the existing
150 km/MHz limit as the transition constraint. It cannot fragment by
construction, so there is nothing for a bridge parameter to repair, and it
needs no training data. This is OIASA's maximum-contrast approach in spirit.

Two design corrections came out of building it, both worth keeping:

- **The off-trace state had to go.** Modelling "not on the trace" as a state
  the path enters and leaves reads as the more general model, and it is
  strictly worse: rejoining resets the range, so the slope limit does not apply
  across the gap. On a synthetic sounding, six bins of interference 500 km off
  the trace and 200 bins above its end were returned as the MUF. The trace now
  starts once and ends once — Kadane's rule carried along a path — and the same
  interferer is ignored.
- **The drift budget must come from the slope, not from
  `pick._range_tolerance`.** That helper floors its allowance at
  `RANGE_SLOPE_FLOOR_BINS`, which is correct for jitter between two independent
  detections and wrong as a per-step budget for a path of hundreds of steps. It
  allowed 10 km per step on a 20.5 kHz axis — **488 km/MHz against a stated
  limit of 150** — enough to walk 650 km through pure noise and rejoin a
  different feature.

### Measured on the same 400 soundings

```
  algo     vs contour   MAE 0.119  bias -0.119  >1MHz  1.8%
  algo     vs dp        MAE 0.122  bias -0.075  >1MHz  1.0%
  contour  vs dp        MAE 0.078  bias +0.043  >1MHz  2.2%
  dp       vs kmeans    MAE 0.197  bias +0.089  >1MHz  2.8%

  vs (contour+kmeans)/2 -- NOT truth, and contour is inside it:
    algo 0.107   contour 0.087   kmeans 0.087   dp 0.132
  >1MHz at the terminator: algo 5.9%   contour 5.0%   dp 7.6%
```

`dp` agrees with `contour` more closely than any other pair in the table
(0.078), which is the encouraging half. Against the only comparator that
excludes both, it is worse than `algo`-with-the-fixes — 0.132 against 0.107 —
and at the terminator, which is where the whole problem lives, it is the worst
of the four at 7.6%.

**So it is built and it is not adopted.** 10 ms per sounding, so cost is not
the objection. The objection is that nothing here can adjudicate: every
comparator available is made of the estimators being compared, and a
comparator containing `contour` and `kmeans` cannot fairly rank a method
against them. `dp` being simultaneously the closest to `contour` and the
furthest from the consensus is the signature of a circular measurement, not a
finding.

**This is now the strongest argument for (b).** The next step for `dp` is not
more tuning — it is a reference that is not one of the estimators. The
hypothesis to test when there is one: that the unbounded gap crossing (which
has no upper length, by design) reaches past the true trace end at the
terminator, where the trace is weakest and interference is worst. That would
show up as `dp` reading *high* against GIRO exactly where it reads high against
the consensus now.

**b. Score against GIRO instead of against each other** — **the endpoint is
fixed; the reference is still unavailable for this circuit, for a different
reason.** Nothing in §1–§5 can distinguish "the extractors agree" from "the
extractors are right", and this is the only thing in the project that can.

Two separate faults were found on 2026-08-31, and only one of them was ours.

**The endpoint had moved, and is now fixed.** `/common/DIDBGetValues` answers
an Apache Tomcat 404 for every query, including the examples GIRO's own
documentation still publishes, while `/common/DIDBFastStationList` on the same
host serves normally. It has been replaced by **FastChar**:
`https://lgdc.uml.edu/fastchar/getbest`, same query parameters.

The way to rediscover it if it moves again — worth writing down, because
guessing did not work and searching found only stale references: POST the
public form at `https://giro.uml.edu/didbase/scaled.php` with
`location=<URSI>&date_start=…&date_end=…&DMUF=…&chosenchars[]=…` and read the
`Location` header of the 302 it answers with. That is what named FastChar.
(Route found via [knamlx/IonoAutoML](https://github.com/knamlx/IonoAutoML),
which drives the same form.)

Two format changes came with it:

- **`MUFD` is now `MUF(D)`.** The old spelling is rejected as "Unknown
  characteristic name" — and the request still succeeds, returning the other
  columns, so getting this wrong loses the MUF column silently.
- **FastChar ignores `DMUF`.** Asked for 2611 km it replies "Distance D for
  MUF calculations: 3000 km" and returns the values for 3000. Those are
  plausible numbers in the right units for a circuit 390 km longer than the
  real one, so the server-side MUF column is now a **correctness trap**:
  nothing downstream could detect it. `predict()` now reads the distance the
  server states and refuses the column unless it matches the path. Our own
  foF2 × secant conversion was already preferred and is unaffected — it uses
  the measured height and the real path length.

`parse()` needed no change: FastChar adds a `CS` autoscaling-confidence column
and duplicate `QD` columns, and keying on the header line absorbed both.

**But the station this path needs is silent.** `MAX_STATION_DISTANCE_KM` is
500 km because the F2 layer decorrelates over a few hundred. For the
Cyprus → Yoshkar-Ola control point at 45.92N 38.95E:

| station | km from control point | data for 2026-08-26 |
|---|---:|---|
| RV149 Rostov-on-Don | 153 | **none, at any date tried** |
| MO155 Moscow | 1068 | none |
| NI135 Nicosia | 1306 | 40 rows |
| AT138 Athens | 1547 | 72 rows |
| PQ052 Pruhonice | 1860 | 73 rows |
| DB049 Dourbes | 2573 | 72 rows |

RV149 is the **only** station inside the limit, and it returns "No measurement
data could be found" for every date from 2024 to now. Every Russian station in
the table behaves the same way; the European ones all publish normally. So this
is a data-sharing boundary, not an outage, and no retry fixes it.

Forced onto AT138 at 1547 km the module now works end to end — 25 of 25
timestamps, 13.1–28.5 MHz, "foF2 × secant law". **That is a working pipeline
against an invalid reference.** Athens is 3× past the distance at which the
ionosphere stays correlated; scoring the terminator behaviour of a
Cyprus→Yoshkar-Ola path against it would produce numbers, and they would not
mean what the table said they meant.

**What this costs.** §6a's `dp` cannot be adjudicated, and neither can the
2026-08-30 fixes: the honest verdict on both stays "measured against a
comparator built from the estimators being compared". Options, none free:

1. **Find a live sounder near 46N 39E** outside GIRO. If one exists and
   publishes, everything above is already built and waiting.
2. **Score a different circuit.** The receiver hears digisondes obliquely
   (`muf.io_digisonde`), and DB049, TR169 and EA036 all publish. A circuit
   whose *control point* is near a live station could be scored properly, and
   what is learned about the extractors there transfers — they are the same
   code. The doc's own caveat that the rate is geometry-dependent (3.9% on
   `cyprus1` against 11.1% here) is the limit on how far that transfers.
3. **Accept the circularity and say so**, which is what §7 already does.

Option 2 is the cheapest real answer and is what I would do next.

The integration itself is pinned either way:
[tests/test_compare.py](tests/test_compare.py) drives the scoring path with
`fetch` stubbed — the scoring case, the outage case, the missing-geometry case
and the out-of-band case — and
[tests/test_reference.py](tests/test_reference.py) covers the FastChar layout,
the `DMUF` refusal, and the server's own `STATUS` line reaching the error
message. That last one matters: before it, a silent station and an unreachable
service produced the same message, which is exactly why the endpoint move hid
behind the station outage for a day.

**c. A U-Net trained on agreement rather than hand labels** — weeks,
label-bound. `cnn.py`'s docstring already describes the path: gated dB tile as
input, agreement of the other estimators as the target mask. That trainer was
never written, and the bundled autoencoder is out-of-distribution for this
geometry and stored in a Keras 2 format current TensorFlow will not load.

**Why this is last and not first.** A network trained on extractor agreement
inherits exactly the failure documented above — the extractors agree least at
20–21 UTC, so the labels would be worst precisely where the model is most
needed. Fixing continuity first is also what makes the labels good enough to
train on. The supervised alternative is NOIRE-Net's: sixteen thousand
hand-scaled ionograms, which this project does not have and cannot cheaply get.

## 7. What this does not establish

- **Agreement is not accuracy.** All three extractors could be wrong together,
  and on the flat low-ray leg they probably sometimes are. Only the GIRO
  comparison (§6b) can speak to that. Every number in §5 is measured against
  `contour`, which is a comparator and not a reference.
- **"`algo` reads low" does not mean `contour` is right.** It means they diverge
  one way. `contour` extending along a flat leg could equally be following a
  second hop or an interference line; nothing here rules that out, and it is
  exactly what an independent reference would settle.
- **One instrument, one archive.** The 11.1% figure is 3,421 soundings, mostly
  NIC→Yoshkar-Ola, August 2026. The `cyprus1` circuit disagrees on only 3.9%, so
  the rate is geometry-dependent and these defaults may not transfer.
- **`DEFAULT_BRIDGE = 2` is not tuned.** It was chosen as the smallest gap that
  closes the observed fragmentation, and 400 soundings is enough to show the sign
  of the effect, not to place the optimum.
