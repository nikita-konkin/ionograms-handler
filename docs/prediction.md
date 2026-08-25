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
                                                       │
                          the api hashes and sniffs four bytes. It never
                          unpickles, and it never fits.
                                                       │
        registrar (10 s poll, no port) ────────────────┤
        trainer   (60 s poll, no port) ────────────────┘
                     │ loads / fits, golden-checks, registers
                     ▼
        /models/objects/<aa>/<sha256>  +  model_registry row
```

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

Estimators are `huber` (default), `ridge` and `xgboost`. The linear two are
wrapped in a `StandardScaler` pipeline: `HuberRegressor`'s epsilon is a
threshold on a standardised residual and its `alpha` penalises coefficients, and
the columns are megahertz, rolling standard deviations and a month number. Trees
are left bare, so `artifacts._framework_of` still reports `xgboost`.

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
