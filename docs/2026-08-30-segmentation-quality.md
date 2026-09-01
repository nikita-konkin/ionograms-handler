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
2026-08-30; adjudicated by eye 2026-09-01 and beaten by `algo`, 32-17.**
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

### Adjudicated by eye, blinded — `algo` wins, and the §6a hypothesis holds (2026-09-01)

50 soundings, sampled from the terminator hours where the two estimators
disagreed by at least 0.5 MHz, rendered as anonymous **A**/**B** markers with
the assignment coin-flipped per sounding and held only in `manifest.csv`.
[tools/adjudicate.py](tools/adjudicate.py); 25 rows carried `algo` as **A**.

```
50 judged, 49 picked a marker, 1 said neither
  algo  preferred on  32 of 49  (65%)
  dp    preferred on  17 of 49  (35%)
  two-sided exact binomial p = 0.044
```

**This is the first non-circular measurement in the whole document.** Every
number in §5 and in the table above is one estimator scored against another.
The eye is not, and it says `algo`.

**But read where the win comes from before spending it.**

```
  gap 0.5-1.0 MHz   n=20   algo 16   dp  4   p = 0.012
  gap 1.0-2.0 MHz   n=11   algo  6   dp  5   p = 1.000
  gap > 2.0 MHz     n=18   algo 10   dp  8   p = 0.815
```

The whole result lives in the band where the two barely disagree. On the
soundings this exercise was built to settle — where the estimators differ by
more than 2 MHz, and a wrong MUF actually costs something — the verdict is a
coin flip. **The question §6a asked is still open.** What was answered is a
smaller one.

**The direction inverts between the bands, and that is the finding.** Below
1 MHz `dp` reads *higher* on 15 of 20 (median +0.55 MHz) and is rejected. Above
2 MHz it reads *lower* on 16 of 18, and there the eye cannot separate them.

So §6a's stated hypothesis — that the unbounded gap crossing reaches past the
true trace end — is **confirmed, by the eye rather than by GIRO**. It just
turns out to be a small over-reach rather than a dramatic one.

Two candidate mechanisms were checked against the 12 soundings showing the
signature, and the obvious one is wrong:

- **Not unlit bins, and it cannot be.** `dp` builds `present` from path
  occupancy, so it marks fade bins present and the `qualifying` filter added to
  `pick_muf` in §5 is inert for it — which looks like it should let `dp` report
  a MUF where nothing was detected. On all 12 it terminates on a bin above
  threshold, and that is forced rather than lucky: `best = gain + max(reachable,
  0)`, so a below-threshold cell has negative `gain` and scores *below* its own
  best predecessor. The global argmax can never land on one. The guard `dp`
  bypasses is one it does not need — but only while the scoring keeps that
  shape, so the invariant is now pinned by a test rather than left to be
  rediscovered.
- **Marginal trace, not absent trace.** The disputed tail is 10–19 frequency
  bins long and **70% lit against 88% in the body**, and half the terminating
  bins sit within 1.5 dB of the 43 dB threshold. `dp` follows the trace into
  where it is fading; `algo` needs `min_run` consecutive detections and stops
  earlier. Both are defensible readings of the same pixels, and the eye
  preferred the earlier stop 16 times in 20.

**Caveat that limits how far this generalises.** The reviewer sees two markers
on a plot, not the truth. A marker sitting in a visibly dimmer region *looks*
wrong whether or not it is, so this measures "which stop a scaler endorses",
which is the operational definition ionospheric scaling has always used — and
still not the same as the true MUF.

**Consequences.**

1. `dp` is **not** adopted, `DEFAULT_METHODS` is unchanged, and no stored
   result needs recomputing. This is the same conclusion as before, now
   resting on something that is not made of the estimators.
2. **The useful follow-up is 40 more soundings drawn only from the >2 MHz
   band**, where n=18 answered nothing. The harness already does this; it needs
   a minimum gap of 2.0 rather than 0.5. Below 1 MHz the argument is nearly
   moot operationally — half a megahertz rarely changes a frequency choice —
   so further sampling there would buy significance about something that does
   not matter.
3. Sounding `020` — the one where both markers were rejected — turns out to
   show a failure mode neither estimator has any defence against, and it is
   worth stating separately.

**A flat line looked like an artefact, and the archive says it is the trace.**
NIC3 at 2026-08-20 23:39, the sounding the eye rejected outright:

```
  freq       lit cells   range of those cells
  13-14 MHz      342     median +2772 km   iqr +2758..+2794
  14-19 MHz       67     median +2642 km   iqr +2642..+2644   <- flat to 2 km
  30-31 MHz      385     median -3172 km   iqr -3364..-2398
```

A line constant to within one 2 km range bin across five megahertz reads as a
fixed delay — a correlation reference or direct leakage — because a trace
approaching its junction frequency must curve. Three more soundings showed the
same feature, twice running from 13 MHz to 25 while the curved segments stopped
by 16, with both estimators reporting a MUF on the flat one. That would have
made a large share of the archive's MUFs measurements of the instrument.

**It is not. The survey refutes it**, and the refutation is worth more than the
hypothesis was. [tools/flat_tails.py](tools/flat_tails.py) over all 7,432
soundings, 5,388 with detections:

```
  a flat feature >= 1 MHz:                     2948 of 5388  (54.7%)
  median flat span 2.75 MHz, longest 12.70
  median range +2702 km, iqr +2654..+2720

  median range by hour (NIC0-3, absolute axis):
    night 21-01 UTC  +2646 km        midday 09-12 UTC  +2725 km
```

**A fixed delay does not breathe with the ionosphere.** The flat feature's range
moves ~80 km on a smooth diurnal cycle, rising through dawn, peaking at midday,
falling through dusk, and it does so identically for NIC0/1/2/3. No cable, no
oscillator, and no correlation reference does that; 80 km is 0.27 ms.

And on 300 sampled soundings it has a companion. Looking at *every* lit cell
across the flat feature's own frequency band — not the one-range-per-frequency
profile the segmenter uses — **77% carry a second ray, sitting +86 km above in
range (iqr +58..+132), above rather than below on 89%**. A flat branch at the
shorter group path with a second branch above it is the ordinary two-ray
structure of an oblique ionogram. These are low-ray legs.

So the estimators walking along them are following the trace, which is what they
should do, and they do not stop at the end of the flat part either: where
`algo`'s MUF sits on a flat feature, it reads a median **1.25 MHz above** that
feature's top, continuing up the nose. Only 1.2% stop more than 0.5 MHz short.

**What this changes.** Sounding `020` is not an artefact case. Its flat 2642 km
branch is the real low ray, running to 16.95 MHz, and both estimators reported
13.95 and 14.85 — so the eye's "neither" almost certainly means *both read low*,
which is §1–§5's finding again rather than a new failure mode. The zero-slope
gap in the 150 km/MHz limit is still real as a matter of code — nothing forbids
a horizontal path — but the archive gives no evidence it is being exploited, and
a curvature requirement built against this would have rejected half the traces
in it.

**One methodological note, because it nearly produced a finding.** The first
version of the coexistence test asked whether flat and curved segments overlap
in frequency and got 0% on 300 of 300 — which looked decisive and was an
artefact of the measurement: `range_profile` keeps one range per frequency, so
two segments cannot share a frequency by construction. The 77% above comes from
re-asking it against the raw threshold mask.


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

### Is there a digisonde near the Cyprus–Yoshkar-Ola midpoint? (2026-08-31)

**Yes, and it is silent.** Checked against the *full* GIRO list — all 129
stations from `DIDBFastStationList`, not the 14 `giro.STATIONS` carries.

RV149 Rostov-on-Don sits **155 km** from the control point at 45.92N 38.95E,
which is well inside the 500 km correlation limit and about as well placed as
could be asked for. It returns "No measurement data could be found" for every
date tried from 2024 to now. The next nearest station in all of GIRO is Moscow
at **1068 km** — seven times further, and also silent.

The pattern is geographic, not incidental:

| | km from control point | data |
|---|---:|---|
| RV149 Rostov | 155 | silent |
| MO155 / MA155 Moscow | 1068 / 1070 | silent |
| MO156 Elektrougli | 1095 | silent |
| NI135 Nicosia | 1306 | **40 rows** |
| MZ152 Warsaw | 1471 | **24 rows** |
| AT138 Athens | 1547 | **72 rows** |
| OL246 Olsztyn | 1579 | silent |
| KL154 Kaliningrad | 1629 | silent |
| LD160 St Petersburg | 1657 | silent |
| SO148 Sopron | 1698 | **68 rows** |

Every Russian station is silent; the European ones publish normally. **Yoshkar-
Ola sits deep inside the silent region**, so this is not a matter of picking a
different station — every circuit *terminating* there has its control point in
or beside that region. No retry and no station list fixes it.

### The circuits that would work, and why we cannot use them

Taking each circuit this receiver is capable of hearing and asking what sits at
*its* control point:

| circuit into Yoshkar-Ola | path | control point | nearest live station |
|---|---:|---|---|
| EA036 El Arenosillo | 4503 km | two-hop, see below | PQ052 Pruhonice, 224 km |
| DB049 Dourbes | 2890 km | 55.18N 24.41E | MZ152 Warsaw, 397 km |
| RL052 Chilton | 3130 km | 56.37N 21.97E | MZ152 Warsaw, 467 km |
| NI135 Nicosia *(the one we record)* | 2611 km | 45.92N 38.95E | none inside 500 km |
| SGO Sodankylä *(recorded 12–16 Aug)* | 1625 km | 62.26N 38.98E | none inside 500 km |
| TR169 Tromsø | 2015 km | 63.66N 36.65E | none inside 500 km |
| IR352 / NV355 (Siberia) | 3602 / 2230 km | — | none inside 500 km |

El Arenosillo, Dourbes and Chilton all have a live reference within the limit.
(These distances were first measured to the *midpoint*; for the two-hop El
Arenosillo path that is the wrong point, and the corrected figures are in the
full survey below — where a better candidate than any of them turns up.)

**And the archive contains none of them.** Every ionogram on disk is Nicosia
(`NIC*`, 4533 files), Sodankylä (`SGO`, 1676 files over 2026-08-12..16) or the
v2 `unkown` marker (1223 files). The digisonde receptions were switched off on
2026-08-12 — see [2026-08-11-recorder-packet-loss.md](2026-08-11-recorder-packet-loss.md),
patch 0007, "Stop receiving the digisondes", ~2.7 of four cores — to stop the
recorder losing 43% of its samples.

So the reference exists, the code to use it now works, and the one circuit that
would make it valid is the one we stopped recording to keep the radio alive.

### Every oblique circuit into Yoshkar-Ola, and what could reference it

**Correction first.** The 39 km figure above was measured to the *midpoint*, and
El Arenosillo at 4503 km is a **two-hop** path — past `MAX_SINGLE_HOP_KM`, so
the midpoint is a place the signal never touches. Its real control points sit
~1250 km either side of it, and Pruhonice is **224 km** from the nearer one.
Still a good reference, comfortably inside the 500 km limit; not the co-located
one claimed. `giro.predict` used `midpoint()` unconditionally and converted at
the whole path length, and both are fixed — see below.

Surveyed all 129 GIRO stations as oblique transmitters into Yoshkar-Ola, using
`geometry.control_points` and taking the nearest station that **actually
published** on 2026-08-26. **42 of 129 stations publish**; 38 circuits have a
live reference within 500 km of a control point.

**The practical shortlist** — European, single-hop or two, plausibly receivable:

| TX | transmitter | path | hops | reference | km | rows/day |
|---|---|---:|---:|---|---:|---:|
| **EB040** | **Roquetes, Spain** | **3772 km** | **1** | **MZ152 Warsaw** | **151** | 79 |
| EA036 | El Arenosillo | 4503 km | 2 | PQ052 Pruhonice | 224 | 277 |
| DB049 | Dourbes | 2890 km | 1 | MZ152 Warsaw | 397 | 79 |
| RL052 | Chilton | 3130 km | 1 | MZ152 Warsaw | 467 | 79 |
| RO041 / RM041 | Rome | 2966 / 2974 km | 1 | MZ152 Warsaw | 476 / 478 | 79 |
| FF051 | Fairford | 3173 km | 1 | MZ152 Warsaw | 486 | 79 |
| SMJ67 | Sondrestrom | 4752 km | 2 | TR169 Tromsø | 92 | 375 |
| NQJ61 | Narssarssuaq | 4922 km | 2 | TR169 Tromsø | 394 | 375 |
| THJ76 | Thule | 4572 km | 2 | TR169 Tromsø | 467 | 375 |

**EB040 Roquetes is the best candidate on the whole list**, and beats El
Arenosillo on both axes that matter: **single hop** — one control point, no
`D/n` division, the cleanest physics available — and the **closest reference at
151 km**. Warsaw's cadence is coarser than Pruhonice's (79 rows a day against
277, roughly 18-minute against 5-minute), which costs alignment precision but
not validity; `_to_times` already matches by nearest neighbour within 20
minutes.

The remaining 29 circuits are transatlantic — Alpena, Austin, Boulder, Belem,
Tucumán and so on, 7800–13900 km at three or four hops. Several have excellent
reference geometry on paper (Alpena's control point is 79 km from Tromsø) and
none is a serious proposition: at four hops the MUF is set by the worst of four
control points and we would be measuring one of them.

**Circuits with no usable reference at all**, including everything currently
recorded:

| circuit | path | nearest station to a control point | |
|---|---:|---|---|
| **NI135 Nicosia** *(4533 files on disk)* | 2611 km | RV149 Rostov, 155 km | silent |
| **SGO Sodankylä** *(1676 files, 12–16 Aug)* | 1625 km | LD160 St Petersburg, 510 km | silent |
| AT138 Athens | 2706 km | RV149 Rostov | silent |
| PQ052 Pruhonice | 2280 km | MA155 Moscow | silent |
| MZ152 Warsaw | 1765 km | MA155 Moscow | silent |
| IR352 Irkutsk | 3596 km | NV355 Novosibirsk | silent |
| NO369 Norilsk | 2432 km | SH266 Salekhard | silent |

The shape of it: **the closer a transmitter is to Yoshkar-Ola, the more surely
its control point lands in the silent region.** Every circuit under 2800 km
fails for that reason. The usable ones are usable precisely because they are
long enough to reflect over Poland, Czechia or northern Scandinavia instead.

### What to record

**EB040 Roquetes**, if the station can spare it. Single hop, 151 km reference,
and at 3772 km it is a comparable geometry to the 2611 km Nicosia circuit the
extractors were tuned on — so what is learned about the terminator there has
the best chance of transferring. Dourbes at 2890 km is the closest match in
path length and costs 397 km of reference distance.

Both are digisonde receptions of the kind switched off on 2026-08-12 by patch
0007 ([2026-08-11-recorder-packet-loss.md](2026-08-11-recorder-packet-loss.md))
to stop the recorder losing 43% of its samples, on a host with four cores and a
4.4-core pipeline. This is a scheduling decision for the station, not a code
change — and a few soundings an hour between 18 and 01 UTC would be enough,
since the disagreement is concentrated in those six hours.

### Can the receiver hear Roquetes today? No — and checking found a stale band

Read from the station's clone of `chirpsounder2` beside this repo
(`my_station.ini`, last modified 2026-08-23, at `cd67b313`). **This is a local
copy, not the live station** — confirm against DOB before acting on it.

**1. Roquetes is not configured.** `receive_digisonde.py:68` selects
`[digisonde-<name>]` by sounder name, so several transmitters are supported,
one process each. Configured today:

| section | transmitter | interval | configured sweep |
|---|---|---:|---|
| `[digisonde]` | Ramfjordmoen (Tromsø) | 450 s | 1–18 MHz |
| `[digisonde-Dourbes]` | DB049 | 900 s | 1–12 MHz |
| `[digisonde-Chilton]` | Chilton | 600 s | 1–15 MHz |
| `[digisonde-Juliusruh]` | Juliusruh | 300 s | 1.01–14 MHz, `sounding_hrs = [5.5,18]` |
| `[digisonde-Thule]` | THJ76 | 450 s | 2.025–20 MHz |

No Roquetes. Adding one is a config section, not code.

**2. The digisonde receivers are switched off.** None of the eight systemd
units is a digisonde receiver, and `required_processes` in `[config]` does not
list `receive_digisonde.py`. This is patch 0007.

**3. Turning them back on is the thing that caused the sample loss.** Patch
0007 measured it directly, restoring *only* the receivers: **872,081 dropped
samples in 900 s (~969/s), load 10.40** on a four-core host. That is the whole
of the 43% loss, and it is why this is a station scheduling decision rather
than a config edit.

**4. Found while checking: every digisonde section predates the band move.**
Patch 0012 moved the receive window, and patch 0015 made the recorder take its
LO from the ini. With `center_freq = 19.5e6` at 25 MS/s the recorded band is
**7.0–32.0 MHz**. Every section above starts at 1–2 MHz, so the bottom of each
configured sweep is now outside the recorded band — Dourbes' 1–12 MHz survives
only as 7–12 MHz. Those sections were written for the old 0–25 MHz band and
were never revisited. Nothing warns about it: the digisonde config never goes
through `chirp_config`, so nothing validates it (see
[chirpsounder2-config.md](chirpsounder2-config.md), "two configuration
surfaces").

This also caps the measurement at the top. Dourbes' `freq_stop = 12 MHz` on a
2890 km path will read as band-limited whenever the true MUF is above it, which
is most of the day — `freq_stop` wants raising toward 20 MHz, which the
7–32 MHz band now allows.

### So record Dourbes, not Roquetes

Reversing the recommendation above, on what the receiver actually has:

| | Roquetes EB040 | Dourbes DB049 |
|---|---|---|
| config section | **must be written** | **already exists** |
| `offset_us` | unknown, must be found empirically | 7300, already established |
| path | 3772 km, **1 hop** | 2890 km, 1 hop |
| reference | Warsaw, 151 km | Warsaw, 397 km |
| reference cadence | 79 rows/day | 79 rows/day |
| own GIRO feed | 273 rows/day | 230 rows/day |

Roquetes still has the better geometry — the reference sits 151 km from its
control point against Dourbes' 397 km, and both are single-hop. But `offset_us`
is per-transmitter (7300 for Dourbes, 3900 Chilton, 2000 Juliusruh, 8066 Thule)
and has to be found by trying, which is an on-station iteration nobody can do
from here. Dourbes is already aligned. **Take the 397 km reference that works
over the 151 km one that needs commissioning**, and keep Roquetes as the
follow-up if Dourbes proves the method.

The edit, if the station can spare the CPU:

    [digisonde-Dourbes]
    freq_start = 7000000        # was 1000000, below the recorded band since 0012
    freq_stop  = 20000000       # was 12000000, truncates the MUF most of the day
    sounding_hrs = [18, 1]      # terminator only -- where the disagreement is

`sounding_hrs` is already supported and already used by Juliusruh, so the
terminator-only run needs no code. One receiver for seven hours a day is a
fraction of the five-receiver load patch 0007 removed — but it is a fraction of
a load that was measured to break the recorder, so it wants watching with
`drop-watch` from the first hour.

### Oblique sounders: cheap to receive, hard to locate

First, a correction to the premise. **The band does not rule digisondes out for
MUF.** `center_freq = 19.5e6` at 25 MS/s records 7.0–32.0 MHz, so a digisonde
sweep loses its bottom 1–7 MHz — and that is the LOF end, not the MUF end. MUF
on every circuit here sits well above 7 MHz. What the band costs is the low-
frequency leg and the `limited_` bookkeeping; what actually stops digisondes is
the CPU (patch 0007) and the stale `freq_start` values. Worth being exact about,
because "we cannot use digisondes" and "we would lose the bottom of the trace"
lead to different decisions.

Oblique chirp sounders are the better route anyway, and for a structural
reason: **they cost no additional process.**

| | digisonde | oblique chirp |
|---|---|---|
| per transmitter | one `receive_digisonde.py` process | none — `detect_chirps` already scans the band |
| CPU of adding one | the thing patch 0007 removed | zero |
| timing setup | `offset_us` found by trying, on station | detector solves the chirp itself |
| band tracking | **stale**: all sections still start at 1–2 MHz | **correct**: `min_freq`/`max_freq` = 7/32 MHz |
| range window | — | ±3999 km *relative to expected arrival*, so path length is not a gate |

A real file confirms the chirp path tracked the band move that the digisonde
sections missed: `lfm_ionogram-NIC3-…` spans **7.05–31.80 MHz** over 496 bins,
with `rmin/rmax = ±3999` km.

So the problems are not reception. They are these:

**1. `serendipitous = false`.** Only the four transmitters named in
`sounder_timings` (NIC1–NIC4) are kept. Anything else crossing the band is
detected and dropped.

**2. Three chirp rates.** `chirp_rates = [100e3, 125e3, 500.0084e3]`. A sounder
running any other rate is invisible, and `max_simultaneous_detections = 5`
caps how many can be tracked at once.

**3. The real blocker — an anonymous chirp cannot be scored at all.**
Not "scored badly": scored *at all*. A transmitter name resolves to
coordinates, coordinates give `path_km`, and `path_km` gives the M-factor that
every GIRO foF2 must be divided by — the reasoning
[tools/relabel_station.py](../tools/relabel_station.py) opens with. Without a
transmitter position there is no path length, so there is no conversion from
foF2 to an oblique MUF, so there is nothing to compare against. The reference
route dies at the first step. This is also why the archive's unidentified files
carry implied ranges of 2638, 2823, 16504, 16728, 23214 and 124194 km: with an
unknown transmitter `t0` is unknown and the range zero is arbitrary, which the
loader says outright — *differences are correct; the zero is not*.

### The identification is tractable, and it needs no station time

`sounder_timings` identifies a transmitter by three numbers — chirp rate,
repetition period, and start offset within the period. Every stored detection
carries `rate` and `t0`, and **`t0` is in the filename**, so this is analysable
from a directory listing without opening a file.

Run over the 1223 files under the `unkown` marker (2026-08-04..12):

```
rep = 300 s, 5 s bins        commonest interval between detections
  chirpt   0-  5 s : 177       5 s  x252      (multi-channel, same sounding)
  chirpt 230-235 s :  85     300 s  x165      <- rep = 300
  chirpt 235-240 s : 144     295 s  x113
  chirpt 240-245 s : 294     245 s  x39
  chirpt 245-250 s : 198     240 s  x36
```

The 230–250 s block is **NIC1/NIC2/NIC3**, whose configured `chirpt` values are
235, 240 and 245 with `rep: 300`. So most of these are the Nicosia transmitters
recorded before their `sounder_timings` entries existed — the `unkown` files run
2026-08-04..12 and the `NIC` names begin on 08-12. Not new circuits, just
un-named old ones.

**The 177 detections at `chirpt ≈ 0–5 s` match none of NIC's four slots.** That
is either a fifth Nicosia transmitter or a different sounder, and it is the only
lead in the archive toward a circuit we do not already have. Worth an hour:
group those detections, check whether their implied ranges are mutually
consistent (they will not be against the wrong `t0`, but they should be
consistent *with each other*), and see whether a rate/rep/chirpt of
100 kHz/300 s/~0 s matches any catalogued sounder.

**What this route cannot do** is choose its geometry. A digisonde can be picked
from a list to put its control point where a GIRO station is; a serendipitous
chirp is wherever it happens to be, and the odds of one landing over Poland or
Czechia are not ours to set. So this is worth doing because it is free and might
turn something up — not as a substitute for [Dourbes](#so-record-dourbes-not-roquetes),
which is a known geometry with a known reference and only needs the CPU.

### What that leaves

1. **Record a circuit that has a reference** — **Dourbes**, on the analysis
   above: already configured, `offset_us` already established, Warsaw 397 km
   from its control point. Needs its `freq_start` raised into the recorded band,
   `freq_stop` toward 20 MHz, and `sounding_hrs = [18, 1]`. The cost is CPU on a
   host with none spare (four cores, pipeline needs 4.4), so this is a
   scheduling question for the station, not a code change.
   **Deferred 2026-09-01** at the user's direction, along with the whole GIRO
   route.
2. **Hand-scale a small sample.** The doc has been treating labels as
   all-or-nothing against NOIRE-Net's sixteen thousand, and for *training* that
   is right. For *adjudicating* it is not: the question is whether `dp` or
   `algo` reads the terminator correctly, the disagreement is one-directional,
   and fifty scaled ionograms between 18 and 01 UTC would settle it. That is an
   afternoon, not a project.
3. **Accept the circularity and say so**, which is what §7 does.

**(2) is the active next step as of 2026-09-01**, GIRO and the digisonde work
having been set aside. It is also the only one of the three that depends on
nothing outside this repository: no station time, no network, no upstream
service. What it produces is the thing every measurement so far has lacked —
a verdict on `dp` against something that is not another extractor.

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
