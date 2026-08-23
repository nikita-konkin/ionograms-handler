# Prediction: from picks to a forecast that has been judged

Reference for `services/prediction/` — what each stage does, what it refuses to
do, and how an operator gets a saved model from the research project into a
curve on the console.

Values marked **[measured 2026-08-23]** were read off the local OrbStack rig on
that date, running four real artifacts from `N:\muf` against the DOB / Yoshkar-Ola
archive. Anything else is read from this repository.

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
| `trained` | the training job, which records the contract as it saves | `measured` |
| `imported` | dropped on the models volume and registered by hand | `measured` |

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

---

## 4. Runbook: a saved model to a curve on the console

Worked end to end on the OrbStack rig. **[measured 2026-08-23]**

### 4.1 Stage the artifact on the models volume

The volume is mounted `:ro` into both `infer` and `importer` — deliberately: a
process that runs code out of an artifact should not also be able to replace
one. So staging needs a third container that mounts it read-write.

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

### 4.2 Register it

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

### 4.3 Run it

An **active** model runs on the unattended loop. A **comparison** model is run by
id, and needs the circuit named — the model says what it expects, not which
circuit's data to feed it:

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env run --rm infer python -m services.prediction.infer --once --param muf --method contour --model 1 --tx NIC3 --rx Yoshkar-Ola
```

### 4.4 See it

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

The sparse-circuit contrast is starker. The same model against **SGO → DOB** —
150 picks over 12.1 days, so most of the grid is tracker-filled — scored **7.99**
against persistence's **1.82**, and was correctly labelled a *backtest*, its
validity window ending before the run.

Three baselines could not score on this archive, each saying why: `recurrence-27d`
and `harmonic` need history older than the window, and `iri` needs
`reference` rows the pipeline populates with `--ref-model iri`.

---

## 6. Not covered

- **Retraining.** M6's remaining bullet. `Dockerfile.train` exists; no training
  job has been run against the accumulated multi-station record.
- **`deep` artifacts.** One Keras file exists in the archive (`1_1_model.keras`,
  11.9 MB). It needs the training image and has never been imported.
- **LOF forecasting in practice.** The path supports `--param lof` throughout and
  no LOF model has been registered. `iri` is structurally unavailable as a LOF
  baseline: it predicts the F2 peak and says nothing about the absorption floor.
- **Postgres.** `deploy/requirements-infer.txt` pins psycopg3 for the cutover; on
  SQLite it is unused.
