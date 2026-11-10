# Rolling MUF training

`services.prediction.rolling` fits the voting estimator on a sliding **13 days
of training targets**, followed by **one day of validation targets**, at each
new five-minute measurement step. The forecast lead remains 24 hours.

The upper boundary is the end of the newest measured five-minute bin.
Training uses `[end - 14 days, end - 1 day)` and validation uses
`[end - 1 day, end)`. Earlier measurements warm up the lagged features and are
never additional fitting targets. Missing and censored measurements reduce
the number of fitted rows; they do not shift these calendar boundaries.
The validation day is never refitted into the operational artifact.

The recipe uses the existing Huber/ridge/XGBoost voting estimator, lagged MUF,
rolling statistics and daily sine/cosine features. It omits the legacy seasonal
decomposition, which estimates seasonal means over the full supplied record
and would expose the held-out day through the feature construction.
Validation is a fixed chronological holdout, not a simulated sequence of
historical deployments; operational forecast scores remain separate.

## Deployment

Build/pull an `ionograms-train` image containing this module before running it.
Configure `ROLLING_TX` and `ROLLING_RX` locally for the desired circuit.

First run once without activation to benchmark on the deployment host:

```sh
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env \
  --profile rolling run --rm rolling \
  python -m services.prediction.rolling --tx NIC2 --rx yoshkar-ola --once
```

Then enable the operational worker:

```sh
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env \
  --profile rolling up -d rolling
```

Keep the existing inference worker running. The rolling worker also writes
the candidate's future forecasts before activating it; failed validation,
training or prediction leaves the incumbent active. At least 60 uncensored
validation pairs and a finite validation MAE are required for activation.
This is a data-validity gate, not a claim that each fit improves accuracy.

The loop checks every 300 seconds, skips unchanged data and never queues missed
steps. OS locks in the shared model store prevent concurrent fits of the same
recipe. State survives restarts. Data more than 30 minutes old pauses fitting:
enable direct completed-file ingestion to avoid the archive-copy delay.

Each successful fit registers a distinct artifact so prior forecasts retain
their model identity. Artifacts and forecast history are retained; monitor
model-store/database growth. The worker writes only future forecasts, rather
than duplicating a complete archive backtest every five minutes. The validation
MAE is recorded in the registry, not presented as a live score.

Because each fit is a new model, no single version holds a history. The public
chart and its score table therefore use the live forecast **as it was issued**
(`scoring.issued`): for each instant, the newest forecast made before it by any
model that has been live on the circuit. Every point is a forecast made in
advance; a backtest written after the fact never enters it. The scoring pass
stores this record under the subject `live:forecast`.

Learning-curve probes and permutation diagnostics are disabled for rolling
fits. Ordinary manually requested training keeps its existing diagnostics.
