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

**a. Extract one continuous trace by dynamic programming** — ~1 week, no labels.
*The highest-value item if §5 has not closed the gap.* Every current method
decides cell-by-cell and hopes a run emerges. The physics says the trace is a
single connected curve whose range varies smoothly with frequency, which is a
shortest-path problem over the array with the existing 150 km/MHz slope limit as
the transition cost. It produces a trace that **cannot fragment by
construction**, it needs no training data, and it *replaces* the weakest stage
rather than adding another one. This is what OIASA's maximum-contrast approach
is doing in spirit; OIASA is the oblique/chirp-sounder state of the art and is
unsupervised.

**b. Score against GIRO instead of against each other** — ~2 days, no labels.
[muf/reference/giro.py](muf/reference/giro.py) already fetches real measurements
from DIDBase and converts foF2/hmF2 to an oblique MUF. Nothing in §1–§5 can
distinguish "the extractors agree" from "the extractors are right". This can,
and it is already written — it needs wiring into the comparison, not building.
Cheap enough that it could reasonably go first.

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
