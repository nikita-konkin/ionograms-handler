# Prediction: from picks to a forecast that has been judged

Reference for `services/prediction/` — what each stage does, what it refuses to
do, and how an operator gets a saved model from the research project into a
curve on the console.

Values marked **[measured …]** were read off the local OrbStack rig on that
date: `2026-08-23` running four real artifacts from `N:\muf`, `2026-08-24` the
console upload path and the first model this service fitted itself, both against
the DOB / Yoshkar-Ola archive. Anything else is read from this repository.

The per-module docstrings are the authority on *why* each rule exists and are
worth reading before changing one; this document is the path through them.

---

## 1. Overview

```
  extraction.muf / .lof              picks, at sounding instants, irregular
      │                              e.limited / e.loflim mark band-edge bounds
      ▼
  dataset.observations()             truth. Picks or nothing — never tracked
      │
      ▼
  dataset.tracked()                  muf.track: constant-velocity Kalman + RTS
      │                              regular 300 s grid, sigma per point
      │                              censored picks passed as gaps, kept flagged
      ▼
  legacy_features.parse()            the recipe, recovered from the column names
      │                              MUF(3000)F2_rolling_48_std_lag_288
      │                                └ alias ┘ └window┘└stat┘   └lag┘
      ▼
  legacy_features.build()            the exact frame the model was fitted on
      │                              index = the instant each row predicts
      ▼
  artifacts.load_verified()          joblib/keras load + golden check
      │
      ▼
  infer.run_model()                  predict once. NOTHING calls fit
      │
      ▼
  forecast rows                      value, sigma, horizon_s, quality JSON
      │
      ├──▶ scoring.run_once()        model + 4 baselines, same code, same pairs
      │        │
      │        ▼
      │    score rows ──▶ /scores, /ui/forecast leaderboard
      │
      └──▶ /forecast, /ui/series overlay
```

Two ways in, and the same wall behind both:

```
  /ui/forecast  ──POST /models/upload (raw bytes)──▶  api ──▶ /uploads/<sha256>
  /ui/forecast  ──POST /models/train  (JSON spec) ──▶  api ──▶ train_job row
  /ui/forecast  ──POST /models/run    (a circuit) ──▶  api ──▶ infer_job row
                                                       │
                          the api hashes and sniffs four bytes. It never
                          unpickles, it never fits, and it never predicts.
                                                       │
        registrar (10 s poll, no port) ────────────────┤
        trainer   (60 s poll, no port) ────────────────┤
        infer     (10 s slices of its interval) ───────┘
                     │ loads / fits / predicts, golden-checks, registers
                     ▼
        /models/objects/<aa>/<sha256>  +  model_registry  +  forecast rows
```

Three queues, three workers, one rule: **the process that answers HTTP writes
a row and nothing else.** Every artifact in this service is opened by a
container with no listening socket.

Two products come out of this one path and should not be confused. A **nowcast**
extends the tracked series a little past the last sounding and is nearly free. A
**forecast** runs days ahead and is a much harder problem. They differ only in
the active model and the lag it carries, so they are separated by model name and
horizon rather than by pretending one is the other.

---

## 2. The contract: what may be run, and what it may become

Three origins reach the registry, differing only in what they can prove about
themselves:

| origin | comes from | `target_src` default |
|---|---|---|
| `legacy` | joblib/Keras files predating this service; contract recovered from the artifact | **`modelled`** |
| `trained` | `train.py`, which builds the contract rather than recovering it | `measured` |
| `imported` | uploaded from the console, or dropped on the volume by hand | `measured` |

`target_src` is the gate on promotion, and it is enforced by the schema, not by
a warning:

```sql
CHECK (active = 0 OR target_src = 'measured'),
CHECK (active = 0 OR (tx IS NOT NULL AND rx IS NOT NULL))
```

A model fitted against a **modelled** target may be scored and compared for
ever and never made the operational forecast — training on modelled values and
then validating against them is circular. An **unbound** model (no `tx`/`rx`)
has no forecast it could be. `registry.activate` refuses both by name, and the
API returns **409** with the reason. **[measured 2026-08-23]**

`capability` decides which image can load it: `slim` (sklearn, xgboost) runs in
`Dockerfile.infer`; `deep` (keras, torch) needs `Dockerfile.train`. Checked at
import, so the failure names the remedy rather than surfacing as an ImportError
three frames inside a loader.

### The golden check

`import_artifact` records one feature row and what the model predicted from it,
before writing any registry row. Every subsequent load re-runs it and compares
against `GOLDEN_TOLERANCE = 1e-6`. This catches a library upgrade that changes
behaviour silently, which a version-string comparison cannot.

The baseline is computed **in the environment inference runs in**, so a fitted-vs-runtime
version gap is recorded in `env` without tripping the check. The rig's artifacts
were pickled by sklearn 1.4.2 and load under 1.9.0; the check passes and
`quality.golden` reads `ok`. **[measured 2026-08-23]** `--allow-version-skew`
runs a model whose check fails, recording the skew on every row it produces.

---

## 3. The stages

### 3.1 `dataset` — the tracker is the resampler

Extraction produces values at sounding instants, irregularly, with gaps; a
lagged-feature model needs a regular grid. The obvious bridge is interpolation
and it is the wrong one: it invents points with no error bar and no notion of
how fast the ionosphere may actually move.

`muf.track` already runs a constant-velocity Kalman filter with an RTS smoother
over exactly this series. Handing it the union of the sounding instants and the
target grid makes resampling and gap-filling the same operation, and every
filled point carries its own sigma.

**Censored picks are excluded from the fit and kept in the output.** A pick at
the top of the sweep is a lower bound, not a measurement; letting it anchor the
state pulls the midday peak down. It is passed to the tracker as a gap and still
appears in the returned frame, flagged.

`MIN_SAMPLES = 2 × 288` — a window shorter than two days cannot produce features
and is refused rather than returned empty.

### 3.2 `legacy_features` — the alias is the hazard

A model artifact names its columns and nothing else, so the recipe is
recoverable by one regex over the names. What is *not* recoverable is whether
feeding **this** series to **that** model means anything.

These models want a column literally named `MUF(3000)F2_lag_288`. Renaming one
of ours to match is a two-character edit and is exactly how a model of somebody
else's ionosphere quietly becomes "the forecast". So `build()` takes the alias
as a **required argument** rather than inferring it, raises if it disagrees with
the recipe, the registry stores it, and the frame carries it in `attrs`.

The rig's artifacts happen to use the alias `muf`, so no renaming is involved
there. **[measured 2026-08-23]**

Two more rules the numbers depend on:

- **Lagging is what makes this a forecast rather than a fit.** Every predictor
  carries `_lag_N`, so the row predicting time *T* is built from the series at
  *T − N*. Transforms are computed on observed data and the index shifted
  *forward*.
- **The trend is a centred filter.** `build` refuses a lag not larger than half
  the decomposition period, because below that the features would carry values
  from *after* the instant being predicted.

The seasonal decomposition period is the one thing the names never carry. 288 —
one day at five-minute sampling — is the default, and the recipe records it as
`period_assumed`, an assumption rather than a reading.

### 3.3 `infer` — nothing calls `fit`

The code this service replaces did: `xgb_evaluate` and `xgb_test` in the
research project both `joblib.load` a saved model and immediately refit it on
the training window, so the artifact contributes hyperparameters and the numbers
come from a model trained moments earlier. In a notebook that is a defensible
shortcut. In a service it is indistinguishable from inference until someone asks
why the "forecast" tracks the training data so well.
`tests/test_prediction_infer.py` makes `fit` raise and runs a whole pass.

**Horizon is lead time, not wall-clock distance from the run.** A lagged model
predicts an instant from data one lag earlier, so its lead is the lag — the same
24 h whether it runs live or over a 2023 archive. Measuring `valid_at − issued_at`
would make every backtest report negative horizons and put the same prediction
in a different bucket depending on the day someone ran it.

**Sigma is the tracker's, and is labelled as such.** These models emit a point
estimate and no interval of their own, so the recorded sigma is the input
uncertainty at the instant each row was built from — not dressed up as the
model's. Across a long gap it grows without bound; the plot clamps the band at
zero because an interval whose lower edge is −20 MHz states an uncertainty
including frequencies that cannot exist.

A run that finds no active model logs and exits zero. A prediction service with
nothing trained yet is the normal state of a fresh deployment, not a fault.

#### Issuing one on demand

The unattended interval is six hours, which is right for a 24 h forecast and
quite wrong for a person who has just activated a model and is looking at an
empty **Last issue** column. Activating does not issue anything; `infer` does.

So the sleep is cut into `POLL_S = 10` second slices, and between them `infer`
drains `infer_job` — rows written by `POST /models/run` from the console. No
fourth container: this process already mounts the store, already loads models
and already writes the rows. What it lacked was a reason to wake up early.

`model_id` on the row is what separates the two uses. Null means *whatever is
active for this circuit*, which is the button on the Live panel. Naming one
runs it as a comparison, exactly as `infer --model` does from a shell, without
promoting anything.

**A requested pass that writes zero rows is `failed`, not `done`** — and that
is deliberately not how the unattended loop behaves. The loop is right to treat
a model it cannot load as one bad circuit among many and carry on. A person who
pressed a button on one circuit and got no forecast has had their request fail,
whatever the reason; the reason is in `detail` either way, and what changes is
whether the pill is green.

---

### 3.4 `scoring` — the claim is comparative

Not "the MAE is 0.94 MHz" but "0.94 where yesterday's value gives 1.02". Absolute
error alone cannot answer the only question promotion turns on: whether running
this thing beats not running it.

`BASELINES = ("persistence", "recurrence-27d", "iri", "harmonic")`, scored by the
same code, over the same pairs, into the same table. If the pairing or censoring
rule changes it changes for the model and its competitors together.

- **Truth is measured, never tracked** — `dataset.observations`, not the smoothed
  grid. Scoring against filled points would partly score the model against
  another model's smoothing.
- **Censored picks are scored one-sidedly** — `max(0, observed − predicted)` at a
  band ceiling, mirrored at the floor, reported in their own columns so a
  headline MAE is never diluted by a bound.
- **Persistence is offset by the lead**, not by a fixed day — always the stronger
  comparison.

`HORIZONS = (3600, 21600, 86400, 604800)`; a forecast is bucketed to the nearest
by log ratio.


### 3.5 `store` — the hash is the address

`artifacts.sha256` has said since it was written that *"a models volume is
shared and writable by the training job, so a file at a given path is not
necessarily the file that was registered there"* — and `model_registry.artifact`
was a mutable path anyway. Uploads make that worse, because every operator picks
a filename, and training worse again, because every run wants one.

So new artifacts are addressed by content:

```
/models/objects/<first two hex>/<the full 64-hex sha256>     mode 0444
```

That is **DVC's cache layout**, deliberately. DVC itself is *not* used: its unit
of work is a developer's git commit against a configured remote, and what
happens here is an operator uploading to, or training on, a running server. The
registry already does what DVC could not — input contracts, golden checks,
measured-versus-modelled provenance, and a promotion rule that is a schema
CHECK rather than a convention. Matching the layout costs nothing and means a
`dvc remote` could be pointed at this directory later without moving a byte.

`0444` because the two workers mount the volume read-write and have to add to
it; that is not a reason for them to be able to replace something already in
it. `store.put` writes a temporary file in the destination directory and
`os.replace`s it into position, so a reader never sees a half-copied artifact,
and placing the same bytes twice converges on one object rather than racing.

**Both shapes stay live.** `store.resolve` takes whatever a registry row
records: an object, or the path of a file somebody put on the volume by hand.
`importer --store` is opt-in, so the four legacy rows on the rig still resolve
by path. Rewriting `artifact` under a running `infer` would be a worse failure
than the inconsistency.

`GET /models/<id>/artifact` is the pull, at **read** scope, streaming the object
with the digest in `X-Artifact-SHA256` so a second deployment can confirm it got
the bytes the registry names. Read rather than control is a trade worth stating:
with `READ_TOKEN` unset — the documented default for a rig on `127.0.0.1` — this
serves every registered artifact to anything that can reach the port. The
alternative is worse, because a host syncing models would then have to hold the
token that can stop an acquisition in order to perform a read.

### 3.6 `train` — the only module that calls `fit`

Everything else in this service exists to keep `fit` out of the inference path;
`tests/test_prediction_infer.py::test_inference_never_fits` makes it raise, and
`test_prediction_upload.py::test_only_the_trainer_fits` reads the syntax tree of
`services/` to confirm exactly one module calls it. Training runs in a separate
process, in a separate container, and never writes a `forecast` row.

Three decisions carry it, and each is a way of not fooling yourself.

**Inputs are the tracked grid; the target is a measured pick.** Features have to
exist at regular instants, which is what `dataset.tracked` is for. Truth does
not. `y` comes from `scoring.truth` — the same picks the leaderboard judges
against — and a feature row with no real pick within half a step of the instant
it predicts is *dropped*, not filled. The tracker would happily supply a value
there; fitting to it teaches the model the Kalman filter.

**Band-edge picks are excluded from the fit and kept in the score.** A `limited`
MUF is a lower bound. Regressing onto it teaches the model the sweep ceiling,
hardest at midday. `scoring.summarise` already counts bounds one-sidedly and
apart, so the holdout number is directly comparable with what the leaderboard
reports later — it is the same function.

**The holdout is the last N days, never a shuffle.** Every feature is a lagged
function of the target, so a random split puts each row's own future in the
training set and the MAE that comes back measures leakage.

The recipe is **constructed rather than parsed**: `train.feature_names` emits
columns in a fixed order — raw lag, then components sorted, then rolling
windows sorted with stats in `STATS` order, then time predictors — and that
order is what `feature_names_in_` records, so it round-trips through
`legacy_features.parse` on import. The archive's models carry `set` iteration
order from the source project; ours do not. Because the period is *chosen* here
rather than guessed, `parse(..., assumed=False)` stops the registry claiming an
assumption that was not made.

Estimators are `huber` (default), `ridge`, `xgboost`, `voting` and `stacking`.
The linear ones are wrapped in a `StandardScaler` pipeline: `HuberRegressor`'s
epsilon is a threshold on a standardised residual and its `alpha` penalises
coefficients, and the columns are megahertz, rolling standard deviations and a
month number. Trees are left bare, so `artifacts._framework_of` still reports
`xgboost`.

#### 3.6.1 Committees

`voting` and `stacking` are ports of the `muf` project's
`voting_stacking_models`. Both take a `members` list — any two or more of
`huber`, `ridge`, `xgboost` — and both are registered, scored and promoted like
any single estimator; nothing downstream of the registry can tell the
difference. A `members` list on a single estimator is refused rather than
ignored, and the key is *absent* from a single estimator's stored spec rather
than empty, because `plan_from_job` vets that spec a second time in the worker
and an empty list would come back through the refusal.

`stacking` is `muf`'s unchanged: three base members, a 50-tree random forest as
the final estimator.

`voting` diverges in one place, deliberately — **how a voter earns its weight.**
`muf` weighted by the mass of the fitted parameters: the sum of
`feature_importances_` for a tree, the sum of `abs(coef_)` for a linear model.
Three things are wrong with that here, and the first is fatal:

* The linear members are `Pipeline` objects, because they are scaled. A pipeline
  exposes neither attribute, so the original `hasattr` chain falls through to
  its `importance = 1.0` default and every linear voter silently gets the same
  weight. The scheme would not fail — it would quietly stop being the scheme.
* `sum(feature_importances_)` is 1.0 for any fitted booster, by construction.
  The tree's weight is a constant carrying no information about the tree.
* Coefficient mass measures scale, not skill.

So the intent is kept and the measure is replaced: the last 20 % of the
*training* rows is held back, each member is fitted on what precedes it, and the
weights are inverse MAE, normalised — chronological, like every other split
here. The inner block never touches the holdout; weighting on the judge would
make the reported MAE describe a model that had already read the answer. Below
30 rows either side the weights fall back to equal and say why. Who voted, on
what evidence, and with how much say is recorded in `metrics.holdout.ensemble`.

One honest caveat, recorded in the metrics as `cv_chronological: false`.
`StackingRegressor` builds its meta-features with `cross_val_predict`, which
requires folds that *partition* the rows; forward-chaining folds never do — the
earliest block has no past to be trained on, so it can be a training set or
excluded, never a test fold. sklearn refuses a `TimeSeriesSplit` here outright.
So the meta-learner sees out-of-fold predictions from members fitted partly on
later data. The leakage is confined to the blend: the whole stack is fitted on
rows before the cut and scored after it, so the reported number stays honest. A
stack that wins by a suspiciously wide margin is the case to look at first.

#### 3.6.2 Where the error is, not just how big

A holdout MAE is one number averaged over every hour of the day, and a MUF
series is not one regime. The sunlit hours are high, slowly varying and easy;
the hours either side of the nightly minimum are steep and are where this
service's models keep going wrong. A model 0.4 MHz out by day and 2 MHz out at
night reports the same 0.8 MHz as one that is uniformly mediocre — and only one
of those has a cause worth chasing.

So every run also writes `metrics.diurnal`, which is `scoring.diurnal`: per hour
of UTC day, the pair count, the MAE and the **bias**. Bias earns its place at
night, where a large error with a bias to match is a model sitting *above* a
trough it cannot reach — the signature of a predictor with no term for where in
the diurnal cycle it is.

It is computed twice, on the holdout and on the rows the model was fitted on,
and **the pair is the diagnostic**:

| night error on the fit | night error on the holdout | reading |
|---|---|---|
| large | large | the columns cannot express the nightly minimum. More archive will not help — see §3.6.3 |
| small | large | it learned a night that has since moved, or the holdout is too short to contain one |
| small | small | the night is fine; look elsewhere |

`train.describe` prints the worst hour when it exceeds 1.5× the headline MAE, so
it lands in the job row on the console without anyone opening the metrics JSON,
and **`/ui/model/<id>` draws both** — reachable from the model name in the
models table and from `see the fit →` on a finished training job.

**The learning curve is a separate question and answers a smaller one.** Only
the booster has one: Ridge has a closed form and Huber converges to one, so
"learning loss" is a quantity that does not exist for them, and the metrics say
so by omitting the key rather than storing an empty curve. Where xgboost is
involved — alone or as a committee member — `metrics.learning` carries per-round
train and validation MAE over the same inner chronological split the voting
weights use, thinned to 50 points with the last round always kept, plus
`best_round`. Still falling together at the end means under-trained; a
validation curve that turns up while training keeps falling means memorising.

#### 3.6.2.1 What it found on the rig, first time out

**[measured 2026-08-27]**, OrbStack, `NIC3 → Yoshkar-Ola`, 1330 soundings over
2026-08-16 .. 08-26 ingested from `ionozond_data2`, `xgboost` at a 24 h lead
with the cyclical block, 881 training rows and a 2-day holdout. Headline: MAE
1.56 MHz, beating persistence at 1.75. The plots said considerably more than
that number did.

**Error by hour** has two peaks, not one: 3.80 MHz at 03 UTC and 3.35 at 21,
against 0.6–1.3 through the middle of the day. At 03 the bias is −3.80 and at
21 it is +3.35 — *equal in magnitude to the MAE at both*, which means the model
is wrong in the same direction every single time there. That is a systematic
offset, not scatter.

And the training-row line stays at 0.3–1.5 MHz across all 24 hours, including
those two. So by the table above this is the **second** row, not the first: the
columns can express those hours perfectly well on rows the model has seen, and
it does not generalise them. That is a different fault from the feature poverty
of §3.6.3 and it points somewhere else first.

**The learning curve says where.** Training loss falls to 0.58 MHz by round 400
while validation bottoms at 1.65 at **round 68** and then drifts back up to
1.70. Three hundred and thirty of the four hundred rounds are pure
memorisation. `_estimator`'s `n_estimators=400` is roughly six times what this
much archive supports, and early stopping on the inner split — which the probe
already computes the number for — is the obvious next change.

Neither of those conclusions is available from "MAE 1.56, beats persistence".

Two things to know about the curve. It comes from a **probe** booster fitted on
the inner split and thrown away — the shipped model has seen every training row,
because a model fitted on less than its training data is not the model the run
is supposed to deliver, and the cost is one extra fit. And it is **blind to the
night**: it is one number per round, averaged over every hour. A model that fits
the day and misses the trough has a perfectly healthy learning curve. That is
what `metrics.diurnal` is for.

#### 3.6.3 The feature table, and why the night is where it fails

Until 2026-08-27 a console-trained model was given **three columns**:
`muf_lag_288`, `muf_rolling_48_mean_lag_288`, `muf_rolling_48_std_lag_288`.
All three are the MUF 24 h ago, smoothed two ways.

The imports it shares a leaderboard with carry **eighteen**. Read straight off
a registered artifact:

```
muf_lag_288
muf_trend_lag_288  muf_seasonal_lag_288  muf_residual_lag_288
muf_rolling_{12,48,288}_{mean,std,min,max}_lag_288      (12 columns)
hour  minute
```

which is exactly what `muf` builds on its vertical path —
`create_rolling_features_fnc(df_total, 'muf', windows=[12, 48, 288],
stats=['mean', 'std', 'min', 'max'])` and
`create_residual_trend_seasonal_features_fnc(df_total, 'muf',
model='additive', period=288)` in `data_handler/muf_data_handler.py`, with
`use_rolling_features` and `use_residual_trend_seasonal_features` both true in
`config_enum.py`.

Nothing chose the three. They were the smallest recipe that ran, and they made
every model trained here a thinner thing than the import it is meant to
replace. `DEFAULT_WINDOWS`, `DEFAULT_STATS` and the new `DEFAULT_COMPONENTS`
now match `muf` column for column.

**And there was no time predictor at all.** `vet` defaults `time` to the empty
tuple and the console form had no control for it, so **the model had no way to
know what hour it was** — it could only reproduce the shape of 24 h ago. That
is a persistence forecast with smoothing, and it fails in a specific,
predictable place: the nightly minimum, where the day-to-day spread is largest
in proportion to the value. When tonight's trough is deeper than last night's,
the curve sits above it. That is the shape in every "it does not fit at night"
report against this service.

`TIME_PREDICTORS` does carry the source project's seven calendar columns, and
they are **not the fix**:

* `hour` and `minute` are integers with a cliff in them. 23:55 and 00:00 are
  five minutes apart and 23 units apart. A tree can split around it; the two
  linear members of every committee cannot, and a straight line in `hour`
  cannot describe a diurnal cycle at all.
* `dayofweek` is noise. The ionosphere has no weekly cycle, and seven arbitrary
  categories are seven chances to overfit.
* `month`, `quarter`, `weekofyear` and `dayofyear` are four collinear spellings
  of one seasonal term, and over any training window this instrument has yet
  produced three of them are **constant** — a column with no variance in
  training and a new value in production.

So the second block of `TIME_PREDICTORS` was added: `daily_sin`, `daily_cos`,
`daily_sin2`, `daily_cos2` over the *fraction of the day* (not the integer
hour — a 24-step staircase is still a staircase), and `yearly_sin`/`yearly_cos`
for the seasonal term. They are index-only, so `parse` recovers them and
`infer` rebuilds them with no further plumbing. The console form ticks them by
default; `vet` still defaults to none, so the API contract is unchanged for a
programmatic caller.

**Measured on a synthetic series whose trough depth wanders day to day** — 16
days, 3-day holdout, `huber`, which is the failure mode this is aimed at, *not*
a prediction of what the station will do:

| recipe | features | MAE | night MAE | night bias |
|---|---|---|---|---|
| no time columns | 3 | 0.730 | 0.964 | −0.00 |
| `hour` + `minute` | 5 | 0.730 | 0.964 | −0.00 |
| `daily_*` (4 terms) | 7 | 0.667 | 0.811 | −0.66 |
| full cyclical | 9 | 0.679 | 0.775 | −0.53 |
| `daily_*` + 288-sample window | 9 | **0.628** | **0.735** | −0.53 |

The second row is the point: adding `hour` and `minute` to a linear model
changes the answer by **nothing at all**, to three decimal places. The diurnal
block cuts the night error by about a fifth, and adding a 288-sample (24 h)
rolling window on top — which gives the model yesterday's overall level and
spread rather than only its last four hours — takes roughly a quarter off both
numbers. The night stops being the worst hour of the day in every case.

Rows three and four already said what the rig later proved: adding the two
seasonal columns to the four diurnal ones made the headline MAE **worse**
(0.667 to 0.679) over a fixture only sixteen days long. It read as noise at
the time. It was not.

**The seasonal pair was in the default block for one afternoon, and this table
is what caught it.** `yearly_sin`/`yearly_cos` move through about 1% of their
range over a week of training rows — day 229 to day 236 — so they are very
nearly constant, and a nearly-constant column is the most dangerous thing you
can hand a regression: it cannot help, and it will absorb weight as a second
intercept. Fitted over seven days on the rig they took **31%** of an xgboost
model's gain and **60%** of a Huber model's coefficient mass. In production
day-of-year keeps moving, those columns drift into values never seen in
training, and the weight parked on them goes with it.

So `DIURNAL` (the four `daily_*` terms) is what the console ticks, and
`SEASONAL` stays available to a caller with months of archive to justify it.
Removing the pair moved the worst hour from 3.80 to 3.27 MHz and the booster's
best round from 68 to 95, with the headline MAE unchanged at 1.57.
**[measured 2026-08-27]**

#### What the parity recipe measured on the rig

**[measured 2026-08-27]**, `NIC3 → Yoshkar-Ola`, `xgboost` at a 24 h lead, same
holdout both times. Reported as it came out, not as it was hoped:

| recipe | columns | train rows | MAE | night 02–04 | best round |
|---|---|---|---|---|---|
| 3 lagged + `daily_*` | 7 | 884 | **1.57** | 2.73 | 95/400 |
| muf parity + `daily_*` | 20 | 777 | 1.64 | **2.69** | 313/400 |

The headline is slightly *worse* and the night marginally better. Two things
are confounded in that and both are worth naming: the 288-sample window and the
decomposition each need a day of history before the first row can be built, so
the parity recipe trains on **107 fewer rows** on an archive only ten days
long; and twenty columns on 777 rows is a thinner per-column budget. Huber over
the same twenty columns is worse still — 2.15 MHz, losing to persistence, which
is what a linear model on 777 rows and 20 columns looks like.

What the features table says, though, is that the new columns are carrying the
model:

| column | share |
|---|---|
| `muf_rolling_288_max_lag_288` | 10.5% |
| `muf_rolling_288_min_lag_288` | 8.1% |
| `daily_cos` | 7.6% |
| `muf_rolling_288_mean_lag_288` | 5.3% |
| … | |
| `muf_lag_288` | 2.5% (last) |

The day-scale window columns take the largest gain shares and the raw lag is
*last*. **Both of those readings are wrong**, and the permutation column beside
them is what says so.

Gain is model-internal and these columns are near-duplicates of one series —
`muf_lag_288` correlates at **0.967** with `muf_rolling_12_mean_lag_288` and at
0.94–0.96 with the other two 1-hour columns — so credit is split among
interchangeable columns roughly arbitrarily. Shuffling asks the other question,
and the two disagree flatly:

| column | share (gain) | permuted Δ MAE |
|---|---|---|
| `muf_seasonal_lag_288` | 5.0% | **+0.695** |
| `daily_cos` | 7.6% | **+0.614** |
| `muf_rolling_12_max_lag_288` | 6.4% | +0.328 |
| `muf_lag_288` | 2.5% *(last by gain)* | **+0.308** |
| … | | |
| `muf_rolling_288_min_lag_288` | 8.1% | −0.004 |
| `muf_rolling_288_mean_lag_288` | 5.3% | −0.020 |
| `muf_rolling_288_max_lag_288` | **10.5%** *(first by gain)* | **−0.022** |

**[measured 2026-08-27]**, baseline 1.641 MHz, 5 shuffles of 287 holdout rows.

Read the two ends of that. The column the booster leans on hardest,
`muf_rolling_288_max_lag_288`, contributes *nothing* to holdout error —
shuffling it makes the fit very slightly better. The whole 288-window block is
at or below zero. Meanwhile the seasonal component of the decomposition, at 5%
of the gain, is the single most load-bearing column in the model, and the raw
lag — dead last by gain — is fourth by permutation, exactly as the collinearity
predicted.

So the earlier reading here, that the day-scale columns were carrying weight
the three-column recipe had nowhere to put, was **wrong**. They carry *gain*
and not *skill*. What the parity recipe actually bought on this archive is the
decomposition, and `muf`'s `use_residual_trend_seasonal_features` is the flag
that mattered — not `windows=[12, 48, 288]`.

That is an argument for trimming, not for reverting: the 288 block is dead
weight on ten days of archive and may not be on ninety, and nothing here is
worth acting on from one circuit and one holdout. It is written down so the
next run has something to disagree with.

A note on what the two columns are, since they are easy to conflate. **Share**,
for a booster, is `feature_importances_` — gain-based and **already summing to
1**, so the normalisation is a no-op and the share *is* the importance. For a
linear member it is `abs(coef_)` renormalised: a coefficient magnitude on
standardised inputs, not an importance in the sklearn sense, which is why the
basis is recorded per member rather than both being called the same thing.
**Permuted Δ MAE** is `train._permutation_importance`: shuffle one column on
the uncensored holdout, average over `PERMUTATION_REPEATS` shuffles, and report
how much worse the MAE got, in megahertz. It measures effect on error rather
than internal weight, it is the same quantity for every estimator so a
committee gets one comparable number instead of three incomparable ones, and it
can be negative — reported as it came out rather than clamped, because
"shuffling this helped" is information.

Neither is a SHAP value, and neither solves collinearity: with two
near-identical columns the model leans on whichever survives the shuffle, so
both can look unimportant. They sit side by side because where they disagree,
the disagreement is the finding.

Two things the cyclical block does not do. It does not remove the bias, it
flips its sign:
the corrected models sit slightly *below* the trough instead of above it. And
it is still a model of one series' own past. The physical variable that governs
the nightly minimum is the solar zenith angle at the path's control point, and
this repo already computes it — `muf.reference.chapman.solar_zenith_cos`, used
by `scoring.harmonic_design` for the `harmonic` baseline. Feeding it to a
*model* rather than to a baseline is the next step and a larger one: `build`
takes a series and nothing else, so the circuit's control point has to reach it
through the recipe and be stored in the artifact contract, or `infer` cannot
rebuild the column.

**A refusal that is really a stale worker says so.** `queue_training` vets
before it inserts, so a row in `train_job` was accepted by *some* api. If the
worker's own `vet` then refuses the same spec, the request was never the
problem — the two builds disagree, and `trainer.settle` separates that failure
from a genuine one and names the build that refused it. `api` and `watch` are
watchtower-labelled and update themselves; `trainer`, `registrar` and `infer`
are not and stay on whatever image created them, which makes "updated api
offers a thing, months-old worker rejects it" the normal failure mode of this
deployment. It happened on 2026-08-26 with `voting`/`stacking` and again on
2026-08-27 with the cyclical time columns — both times the message described
the running code perfectly and gave no hint that the running code was stale.

Refusals state the arithmetic: how many grid points the circuit has, how many a
lag-*N* model with a *W*-sample window needs before it can build one row, and
how many measured rows fall before the holdout cut. Those are data limits, not
faults, and the message says so.

**It never activates.** A trained model is `measured` and bound to its circuit,
so it satisfies both promotion CHECKs and is eligible — which is the whole
difference between it and every legacy import. Promotion stays a deliberate act
behind the control token, because it changes what every consumer of `/forecast`
receives with nothing in the logs having asked for it.

---

## 4. Runbook: a saved model to a curve on the console

Worked end to end on the OrbStack rig. **[measured 2026-08-23]** for the shell
path, **[measured 2026-08-24]** for the console one.

### 4.1 From the console

`/ui/forecast` has two panels at the foot of the page, below the evidence:
**Add a model** takes a `.sav` or `.keras` file, and **Train a model** queues a
fit on a circuit's own measured picks. Both need the control token pasted on the
console page — the same one the start/stop buttons use.

An upload settles in about ten seconds and the page reloads itself when it does.
What it does *not* do is open the file: the api hashes it, checks the first four
bytes are a pickle or a zip, writes it to a quarantine volume and records a
`pending` row. `registrar` — a container with no port, that nothing can reach —
is what loads it, runs the golden check and writes the registry row.

**Run now**, on each row of the Live panel, issues a forecast without waiting
for the next interval. It is the third queue and behaves like the other two:
the api writes a row, `infer` picks it up within about ten seconds, and the
page reloads showing rows written and whether the pass was a backtest. Asking
for a circuit with nothing live is refused at the door rather than queued —
*"no MUF model is live for NIC3 -> Yoshkar-Ola, so there is no forecast to
issue"* — because "queued, then done, wrote 0 rows" is a worse answer than a
sentence. **[measured 2026-08-24]**

Refusals arrive as sentences on the page, not status codes. Uploading a text
file: **[measured 2026-08-24]**

```
notes.txt does not begin like a model artifact (first bytes: b'not ').
Expected a joblib pickle (.sav, .joblib) or a Keras .keras file.
```

A refused upload keeps its quarantined bytes, because the usual fix is to
register the same file again with an explicit feature list. `Forget` deletes
them.

**Two deployment traps, both seen on the first run.** [measured 2026-08-24]

The `models` volume must be owned by uid 10001. A named volume takes its
ownership from whatever populated it first, and one seeded by a throwaway root
container — which is what §4.2 tells you to do — cannot then be written by the
workers. The refusal names the fix:

```
docker run --rm -v ionograms_models:/models alpine chown -R 10001:10001 /models
```

And the `api` needs `models:/models:ro` for `GET /models/<id>/artifact` to have
anything to serve; without it the route correctly answers **410** naming the
digest it could not find. Both compose files carry the mount now.

### 4.2 Or from a shell: stage the artifact on the models volume

Still the right answer for a scripted bulk import, and unchanged. The volume is
mounted `:ro` into both `infer` and `importer` — deliberately: a process that
runs code out of an artifact should not also be able to replace one. So staging
needs a third container that mounts it read-write.

```bash
docker run --rm -v ionograms_models:/models -v "$PWD/stage":/stage:ro alpine cp /stage/model.sav /models/
```

**Then fix the mode.** Files arrive owned by root; the containers run as uid
10001 and cannot read `0600`. This is the first thing that goes wrong.

```bash
docker run --rm -v ionograms_models:/models alpine chmod 0444 /models/model.sav
```

Give the file a name that carries the lag — the registry's `name` defaults to
the file stem, and four rows called `huber_mae-…` are unreadable in a list.

### 4.3 Register it

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env --profile import run --rm importer /models/huber_lag288.sav --param muf --tx NIC3 --rx Yoshkar-Ola --origin legacy
```

**There is no HTTP import route**, and that is deliberate: registering a model
means reading a file from a shared volume and running code out of it. Exposing
that would give the prediction service an inbound surface.

The importer proves the model runs before writing anything, and prints what it
recorded:

```
#1 huber_lag288 [muf] NIC3 -> Yoshkar-Ola · legacy · sklearn 1.4.2 · 18 features · slim · comparison
  registered for comparison only: a modelled target cannot be promoted to the operational forecast.
  decomposition period assumed to be 288 samples; the artifact does not record it.
```

### 4.4 Run it

An **active** model runs on the unattended loop. A **comparison** model is run by
id, and needs the circuit named — the model says what it expects, not which
circuit's data to feed it:

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env run --rm infer python -m services.prediction.infer --once --param muf --method contour --model 1 --tx NIC3 --rx Yoshkar-Ola
```

### 4.5 See it

Nothing is drawn until a pass has run. **Last issue —** on the Live panel means
exactly that, and **Run now** beside it is the shortest way to fix it; the
alternative is `infer --once` from a shell, or up to six hours.

On the rig, one press over NIC3 → Yoshkar-Ola wrote **2,052 rows in 9 s**,
labelled `backtest at +24.0 h lead, valid 2026-08-17 17:40 .. 2026-08-24 20:35`
— the archive ends before now, so a 24 h lead predicts instants that have
already happened. That label is the honest one and it reaches the page.
**[measured 2026-08-24]**

On `/ui/series`, pick the circuit **and the method the picks are in**: the
selector defaults to `algo`, and a model trained on `contour` drawn over an
empty `algo` series looks like a broken deployment rather than a wrong choice.

Both read surfaces apply the same rule — **active models only, unless one is
named** — so a comparison model is invisible until asked for:

| surface | how to name one |
|---|---|
| `/forecast` | `?model=<id>` |
| `/ui/series` | `?forecast=<id>` — chooser row on the page |
| `/ui/forecast` | leaderboard lists every scored subject; no naming needed |

On the plot a candidate is **dotted and named for the model**; the operational
forecast is **dashed and named for the parameter**. They are not the same claim,
so they are not the same line.

### 4.5.1 `/ui/model/<id>` — why this one is wrong

The model name in the models table is a link, and a finished training job
carries `see the fit →`. Both land on one model's own page: its contract (every
feature column by name and by weight, not a count — "three columns and none of
them knows what hour it is" and "two of these are constant and carry a third of
the weight" were both real, and a count says neither), its holdout numbers
beside persistence, the features table below, and the two plots from §3.6.2.

It is a separate page rather than a section of `/ui/forecast` because the two
answer different questions. That page is a table and answers "what is live,
across every circuit"; this is two plots and answers "why is this one wrong at
02 UTC". Folding the second into the first would make an operator scroll past a
diagnostic for a model they did not ask about.

#### The features table

Every column the model was fitted on, **in contract order** — the order it
resolves its inputs by, not an order of importance. Reordering by weight would
invite comparing two models' rows position by position when the positions mean
different things.

Each row decodes the name (`legacy_features.describe_feature`, the per-name
half of `parse`, which labels an unparseable column rather than raising — a
model with one odd column should still render its other seventeen), says what
it was built from, and gives the lead **both ways**: in hours and, in brackets,
as the count of grid samples the model actually counts. Those are the two
things most often confused here, and 288 samples is a day only on a five-minute
grid.

The last columns are **how much of the fitted model's weight sits on each**,
recorded by `train._influence` at fit time. It has to be recorded there: the
api serves this page and deliberately cannot deserialise an artifact — that is
what `capability` is for — so without it the console could only ever list
names. Two sources, both normalised to shares of one so they sit in the same
column: absolute `coef_` for a linear model, comparable across columns
*because* `_estimator` standardises the inputs, and `feature_importances_` for
a booster, already normalised.

This is **not** the question `_voting_weights` refuses to answer with
coefficient mass, and the distinction is worth keeping straight. Ranking
*columns within* one fitted model is exactly what a scaled coefficient is for.
Ranking *models against each other* by the size of their coefficients measures
scale rather than skill. Same numbers, different question, opposite verdict.

A committee gets one column per member rather than a blend: an average of a
booster's gain and a linear model's coefficient is a number with no units, and
two members disagreeing about which column matters is a fact worth seeing. On
the rig they disagree considerably — Huber puts 42.7% on `daily_cos` and 0.9%
on `muf_lag_288`; Ridge puts 23.9% on the rolling mean and 8.0% on the lag.
**[measured 2026-08-27]**

Bars are drawn against the largest share in their own column, not against
100%. Nine columns cannot each hold much of the weight, so an absolute scale
draws nine near-identical stubs. The number beside each bar stays absolute.

Read scope, like everything under `/ui`. Promotion stays on the forecast page
behind the control token.

A model with no recorded diagnostics — a legacy import, or anything fitted by
an estimator with no rounds — gets a sentence saying which of those it is,
never an empty axis. An empty axis reads as "the model learned nothing", and
for Ridge that is simply false.

### 4.6.1 The grey stretch: what the model was fitted on

A lagged model run over a finished archive predicts instants that have already
happened — `infer` calls that a backtest and labels it. Where those instants
fall inside the window the model was *fitted* on, the curve is not a prediction
at all: it is the model reciting rows it has already seen, and it will look
superb. Read as a forecast, that is the single most flattering mistake this
page can invite.

So the forecast curve is **drawn in two colours**: grey over the hours the model
was fitted on, the parameter's own hue past them. The split comes from
`model_registry.trained_from` / `trained_to`, which `train.run` records as the
first and last instant of the rows it actually fitted. **Judge a model only
where its curve is coloured.**

It was a shaded rectangle behind every trace until 2026-08-27, and that was
wrong three ways: it read as a leftover zoom selection, it dimmed the
measurements it had nothing to say about, and with two models on one axis it
could not show which of them the hours belonged to. Recital is a property of
*one curve*, so it is now drawn on that curve — which also means it withdraws
with its own trace and needs no separate bookkeeping to stay consistent.

The boundary point belongs to **both** segments. Without it the line breaks at
exactly the instant worth seeing continuously, where recital stops and
prediction starts.

A legacy import is **coloured throughout**. It was fitted somewhere else and the
window was never recorded — which is a different statement from "trained on
nothing", and is why the null check is on the stamps rather than on the model.

---

## 5. What the rig actually produced

Four `huber` artifacts (`sklearn.linear_model.HuberRegressor`, 18 features,
`capability = slim`) against **NIC3 → Yoshkar-Ola**, 777 `contour` MUF picks over
6.3 days. **[measured 2026-08-23]**

| model | lag | lead | rows | scored in | MAE (MHz) | pairs |
|---|---|---|---|---|---|---|
| `huber_lag288_mae-0.2456` | 288 | 24 h | 1535 | 24 h | 3.03 | 516 |
| `huber_lag576_mae-0.3217` | 576 | 48 h | 1535 | 24 h | 2.93 | 353 |
| `huber_lag864_mae-0.5290` | 864 | 72 h | 1535 | 7 d | 3.07 | 332 |
| `huber_lag1440_mae-0.5627` | 1440 | 120 h | 1535 | 7 d | 2.04 | 48 |
| **persistence** | — | 24 h | — | 24 h | **2.46** | 269 |

**Lead and scoring bucket are not the same column.** `bucket()` snaps a horizon
to the nearest of `HORIZONS` by log ratio, so 48 h lands in the 24 h bucket and
both 72 h and 120 h land in 7 d. Only the lag-288 row compares like with like
against persistence; the rest are read across a bucket boundary and the 2.93
should not be taken as beating the 3.03.

**Every model loses to persistence at 24 h.** That is the honest result and the
one the design anticipates: these artifacts were fitted on a different circuit
and a different ionosphere, and `target_src = modelled` is the schema refusing to
let "the service runs them correctly" be confused with "the numbers are good".
The lag-1440 figure rests on 48 pairs and should not be read as a win.

### The first model this service fitted itself

`huber-muf-24h`, queued from the console, on the same circuit. **[measured
2026-08-24]**

| | |
|---|---|
| Features | 3 — `muf_lag_288`, `muf_rolling_48_{mean,std}_lag_288` |
| Fitted on | 474 measured, uncensored rows, 2026-08-17T17:40Z → 2026-08-21T20:30Z |
| Holdout | the last 2 days, 285 pairs |
| **MAE** | **2.66 MHz** (RMSE 3.42, bias **+1.17**) |
| persistence, same window | **2.00 MHz** over 142 pairs |

**It loses to persistence, and that is the result.** Not a defect in the
pipeline — the pipeline did what it is for, which is to make the comparison
visible before anybody promotes anything. Three things are worth reading off it:

* **474 rows is not much.** The circuit holds 6.3 days; a 24 h lag plus a
  48-sample window consumes the first day and change before a single feature row
  exists, and a 2-day holdout takes another third of what is left.
* **The bias is +1.17 MHz**, so it runs high rather than noisy. On this little
  data a linear model on three lagged features mostly learns the diurnal mean.
* **The two `n` differ — 285 against 142 — so the comparison is indicative, not
  paired.** `scoring._shifted` drops any instant with no measured pick one lead
  time back rather than interpolating one, which is what makes persistence an
  honest "do nothing" baseline and also what makes it score on a subset.

The remedy is archive, not architecture: the same command against a longer
record is the experiment worth running, and it is now one form submission.

The sparse-circuit contrast is starker. The same model against **SGO → DOB** —
150 picks over 12.1 days, so most of the grid is tracker-filled — scored **7.99**
against persistence's **1.82**, and was correctly labelled a *backtest*, its
validity window ending before the run.

Three baselines could not score on this archive, each saying why: `recurrence-27d`
and `harmonic` need history older than the window, and `iri` needs
`reference` rows the pipeline populates with `--ref-model iri`.

---

## 6. Not covered

- **A trained model that is worth promoting.** `train.py` exists and runs; what
  it produced on 6.3 days of one circuit loses to persistence (§5). Retraining
  across the accumulated multi-station record is the open question, and it is
  now a form submission rather than a missing module.
- **Hyperparameter search.** The estimators take fixed, defensible settings.
  `Dockerfile.train` carries `hyperopt` and nothing uses it.
- **Automatic promotion.** Deliberate, and unlikely to change: `scoring.drift`
  already surfaces a live model that has been overtaken, and demoting it stays a
  human decision behind the control token.
- **`deep` artifacts.** One Keras file exists in the archive (`1_1_model.keras`,
  11.9 MB). It needs the training image and has never been imported.
- **LOF forecasting in practice.** The path supports `--param lof` throughout and
  no LOF model has been registered. `iri` is structurally unavailable as a LOF
  baseline: it predicts the F2 peak and says nothing about the absorption floor.
- **Postgres.** `deploy/requirements-infer.txt` pins psycopg3 for the cutover; on
  SQLite it is unused.
