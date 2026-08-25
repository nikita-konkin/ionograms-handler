-- Schema for the api service. architecture.md sec. 5.2, plus the three tables
-- the health and control path of sec. 5.4 needs.
--
-- Long, not wide: adding a fifth estimator must not be a migration, so
-- `extraction` is one row per (sounding, method) rather than a column per
-- method. This is the normalized form of the 68-column `muf run` CSV.
--
-- SQLite, deliberately. This deployment is temporary and single-node; the
-- schema below is portable to Postgres unchanged, and an ops surface for a
-- test rig is a cost with no return.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;


-- One row per sounding file: acquisition facts and derived calibration.
CREATE TABLE IF NOT EXISTS sounding (
    id              INTEGER PRIMARY KEY,
    file            TEXT NOT NULL UNIQUE,
    -- Relative to the configured archive root, not absolute: the same database
    -- is read from the host and from inside a container, which mount the
    -- archive at different paths. Needed by the on-demand renderer (sec. 4.2);
    -- not part of sec. 5.2's column list.
    path            TEXT NOT NULL,
    format          TEXT,                 -- lfs | chirp2
    window          INTEGER,              -- with `format`: is this re-derivable (sec. 3.4)
    datetime        TEXT NOT NULL,        -- ISO-8601 UTC
    tx              TEXT,
    rx              TEXT,
    path_type       TEXT,
    tx_lat          REAL, tx_lon REAL,
    rx_lat          REAL, rx_lon REAL,
    path_km         REAL,
    freq_start      REAL, freq_stop REAL,
    gate_lo         REAL, gate_hi REAL,
    sweep_complete  INTEGER,
    sweep_fraction  REAL,
    -- Which acquisition configuration produced it. NULL until a config_epoch
    -- covering this datetime exists, which is the honest answer for anything
    -- recorded before the agent was deployed.
    config_epoch_id INTEGER REFERENCES config_epoch(id),
    ingested_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sounding_datetime ON sounding(datetime);
CREATE INDEX IF NOT EXISTS sounding_path     ON sounding(tx, rx, datetime);


-- One row per (sounding, method) -- the long axis.
CREATE TABLE IF NOT EXISTS extraction (
    sounding_id  INTEGER NOT NULL REFERENCES sounding(id) ON DELETE CASCADE,
    method       TEXT NOT NULL,
    muf          REAL, lof REAL, vrange REAL, snr REAL,
    ndet         INTEGER, run INTEGER, nseg INTEGER, hops INTEGER,
    branch       INTEGER, scatter REAL,
    fit          REAL, fitres REAL, fitex INTEGER,
    -- Quality columns travel with the value, never separately (sec. 4.1).
    -- `limited` means the pick sat at the top of the sweep, so the MUF is a
    -- lower bound; `loflim` is the same at the bottom of the band.
    limited      INTEGER, loflim INTEGER,
    muf_smooth   REAL,                    -- from track/daily; NULL until it runs
    PRIMARY KEY (sounding_id, method)
);

CREATE INDEX IF NOT EXISTS extraction_method ON extraction(method);


-- Modelled and third-party values, structurally apart from measurements so a
-- prediction service can never train on IRI and call it validation.
CREATE TABLE IF NOT EXISTS reference (
    sounding_id  INTEGER NOT NULL REFERENCES sounding(id) ON DELETE CASCADE,
    source       TEXT NOT NULL,           -- iri | giro | chapman | minimuf
    param        TEXT NOT NULL,           -- muf | fof2 | hmf2 | ...
    value        REAL,
    PRIMARY KEY (sounding_id, source, param)
);


-- Acquisition configuration over time; written by the control endpoints.
CREATE TABLE IF NOT EXISTS config_epoch (
    id          INTEGER PRIMARY KEY,
    station     TEXT NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_to    TEXT,
    changes     TEXT,                     -- JSON: what moved, from and to
    changed_by  TEXT,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS config_epoch_station ON config_epoch(station, valid_from);


-- Transmitters an operator has identified, and the timings to sound them at.
--
-- **Keyed by the receiving station, not by the transmitter alone.** A slot
-- second here is a *reception* second: the transmit second plus the one-way
-- travel time plus this receiver's own epoch offset (see `sources.py`). The
-- same transmitter heard at two receivers is two different `chirpt` values,
-- and a table that stored one of them would schedule the other receiver to
-- listen at the wrong instant -- which records noise while reporting healthy.
-- Same reasoning as the per-receiver band ceiling in `muf/stations.py`.
--
-- `timings` is the JSON list of `sounder_timings` entries for this
-- transmitter, plural because a transmitter has as many slots as it has: NIC
-- is heard in four seconds of every cycle at DOB, and which of them to sound
-- is the operator's choice, made once and kept here.
--
-- `evidence` is the census row it was read off, verbatim. Nothing in a
-- detection identifies a transmitter -- the identification is a human
-- judgement, exactly as `cyprus1` was resolved to `NIC` -- so what the
-- judgement was made on is kept with it.
--
-- `code` is not a label. It is written into `sounder_timings` as
-- `transmit_name`, and `calc_ionograms.py:344` puts it in the product's
-- **file name** and in `ho["txname"]`, which this pipeline ingests as
-- `sounding.tx` and resolves against `muf/stations.py` for coordinates and
-- for the band ceiling. Naming an emitter here is what makes every later
-- product identified, so it is worth getting right and worth keeping stable.
--
-- `sounder_id` is chirpsounder2's own `id` field, `%03d` in the same file
-- name. Stored rather than derived from position so that re-ordering the
-- schedule does not renumber files that are already on disk.
CREATE TABLE IF NOT EXISTS transmitter (
    id           INTEGER PRIMARY KEY,
    station      TEXT NOT NULL,          -- the RECEIVER these timings are for
    code         TEXT NOT NULL,          -- the operator's identification
    name         TEXT,
    sounder_id   INTEGER NOT NULL,       -- chirpsounder2's `id`, per station
    timings      TEXT NOT NULL,          -- JSON: list of sounder_timings entries
    evidence     TEXT,                   -- JSON: the census row, as seen
    verified_at  TEXT NOT NULL,
    verified_by  TEXT,
    note         TEXT,
    UNIQUE (station, code),
    UNIQUE (station, sounder_id)
);

CREATE INDEX IF NOT EXISTS transmitter_station ON transmitter(station, code);


-- --------------------------------------------------------------------------
-- Health and control (sec. 5.4)
-- --------------------------------------------------------------------------

-- Every push is kept, not just the latest. Under sec. 5.4 silence is itself
-- the alert, so the interesting question is "when did reports stop", which a
-- latest-only table cannot answer.
CREATE TABLE IF NOT EXISTS health_report (
    id            INTEGER PRIMARY KEY,
    station       TEXT NOT NULL,
    received_at   TEXT NOT NULL,          -- server clock, not the station's
    reported_at   REAL,                   -- the station's own timestamp
    healthy       INTEGER,
    agent_version TEXT,
    document      TEXT NOT NULL           -- the raw JSON, kept verbatim
);

CREATE INDEX IF NOT EXISTS health_report_station ON health_report(station, received_at);


-- `ok` is TRI-STATE and the column must stay nullable end to end. NULL means
-- "could not measure", which is a different situation from False and needs a
-- different response -- collapsing them makes a missing `systemctl` page
-- someone at 03:00. See services/agent/health.py.
CREATE TABLE IF NOT EXISTS health_metric (
    report_id  INTEGER NOT NULL REFERENCES health_report(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    value      TEXT,
    ok         INTEGER,                   -- NULL = unknown, 0 = failing, 1 = ok
    detail     TEXT,
    PRIMARY KEY (report_id, name)
);


-- Commands are queued, delivered once, and acknowledged with their result.
-- An unacknowledged command must not be redelivered forever: "restart
-- acquisition" delivered forever is worse than the fault it was meant to fix,
-- so delivery is recorded separately from acknowledgement.
CREATE TABLE IF NOT EXISTS command (
    id            TEXT PRIMARY KEY,
    station       TEXT NOT NULL,
    name          TEXT NOT NULL,
    params        TEXT,                   -- JSON
    issued_at     TEXT NOT NULL,
    issued_by     TEXT,
    delivered_at  TEXT,
    acked_at      TEXT,
    ok            INTEGER,
    results       TEXT                    -- JSON, as returned by the agent
);

CREATE INDEX IF NOT EXISTS command_pending ON command(station, delivered_at);


-- The newest picture each station has of each transmitter, from the station's
-- own disk. `arrivals` measures the *archive*, which reaches this server only
-- on `chirp-archive-sync`'s timer, so nothing else here can show what the
-- acquisition laptop is seeing right now. See services/agent/preview.py.
--
-- One row per circuit, overwritten in place: this is a live view, not a
-- record, and the archive is where the record belongs. Bounded by the number
-- of transmitters rather than by time, so unlike `health_report` it does not
-- grow -- and a circuit that stops reporting is swept after
-- `PREVIEW_RETENTION_DAYS` rather than leaving a picture up forever.
--
-- The only BLOB in this schema. A base64 TEXT column would cost a third more
-- for a value nothing ever reads as text: it goes out of `GET /preview/...`
-- as image bytes and is never joined, searched or compared.
CREATE TABLE IF NOT EXISTS station_preview (
    station     TEXT NOT NULL,
    tx          TEXT NOT NULL,          -- transmitter, from the product's name
    t0          REAL,                   -- the sounding's start time: identity,
                                        -- not arrival, and the browser's cache key
    received_at TEXT NOT NULL,          -- server clock, as everywhere else here
    width       INTEGER,
    height      INTEGER,
    freq_lo_hz  REAL,
    freq_hi_hz  REAL,
    range_lo_m  REAL,
    range_hi_m  REAL,
    cropped     INTEGER,                -- 1 when the range axis was narrowed
    image       BLOB NOT NULL,          -- PNG, as the agent encoded it
    PRIMARY KEY (station, tx)
);


-- Folders that are meant to be indexed, so that "what is in this database"
-- stops being whatever someone last ran `services.api.ingest` over by hand.
--
-- `relpath` is **relative to ARCHIVE_ROOT and never absolute**, for the same
-- reason `sounding.path` is: one database is read from the host and from
-- inside a container, which mount the archive at different paths. Storing an
-- absolute path here would work on exactly one of them and fail silently on
-- the other, which is the failure mode the whole archive-root convention
-- exists to prevent.
--
-- `methods` is what gets computed for every sounding in the folder --
-- `muf.extractors.DEFAULT_METHODS` unless someone chose otherwise. Kept per
-- archive rather than read from a global default so the page can show it: the
-- answer to "which extractors produced these numbers" should be on screen and
-- not implied. Widening it needs no migration and no reload flag --
-- `watch.already_done` counts a sounding finished only when it holds a row
-- for every requested method, so an added method pulls the older soundings of
-- that archive back into scope on the next scan.
CREATE TABLE IF NOT EXISTS archive (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    relpath          TEXT NOT NULL UNIQUE,
    format           TEXT,              -- lfs | chirp2 | digisonde; NULL = any
    methods          TEXT NOT NULL,     -- comma separated
    enabled          INTEGER NOT NULL DEFAULT 1,
    added_at         TEXT NOT NULL,
    -- What the last scan did, kept as the sentence `watch.describe` builds
    -- rather than as parsed counts: it already says the things worth saying
    -- ("N too fresh", "FUTURE-DATED", "SKIPPED N") and re-deriving them here
    -- would be a second vocabulary for the same facts.
    last_scan_at     TEXT,
    last_scan_result TEXT,
    last_scan_ok     INTEGER
);


-- --------------------------------------------------------------------------
-- Forecasting (architecture.md sec. 5.2, extended for inference)
-- --------------------------------------------------------------------------

-- What produced a forecast, and on what. The artifact itself is a file on the
-- models volume; this row is what makes it identifiable six months later.
--
-- **A row is a binding of one artifact to one circuit**, not just a file.
-- The same artifact serving two circuits is two rows sharing a `sha256`, which
-- keeps "one active model per circuit and parameter" expressible as an index
-- rather than as application logic over a JSON list.
--
-- `tx`/`rx` NULL means *unbound*: registered and runnable for comparison, but
-- attached to no circuit. Every legacy import lands here, because a model
-- trained on the Brisbane MUF(3000)F2 series is not a model of any circuit
-- this instrument sounds.
--
-- Two CHECKs carry the promotion rule, and they are the reason it is a rule
-- rather than a habit:
--
--   * `target_src = 'measured'` -- a model fitted against modelled values (IRI,
--     or another model's output) may be scored and compared, and may never
--     become the operational forecast. Structural separation, exactly as
--     `reference` is a separate table from `extraction`.
--   * bound to a circuit -- an unbound model has no circuit whose forecast it
--     could be.
--
-- `features` is stored VERBATIM and in order. sklearn's `feature_names_in_`
-- order is whatever `set` iteration produced at fit time, not sorted, and
-- feeding the columns back in a different order silently produces wrong
-- numbers rather than an error.
CREATE TABLE IF NOT EXISTS model_registry (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    param          TEXT NOT NULL,          -- muf | lof
    tx             TEXT, rx TEXT,          -- NULL = unbound, comparison only
    origin         TEXT NOT NULL,          -- legacy | trained | imported
    framework      TEXT NOT NULL,          -- sklearn | xgboost | keras | torch
    loader         TEXT NOT NULL,          -- joblib | keras | torch
    -- Which image can load it: 'slim' needs only sklearn/xgboost, 'deep' needs
    -- the training image. Checked before the import is attempted so the failure
    -- names the remedy instead of surfacing as an ImportError three frames down.
    capability     TEXT NOT NULL DEFAULT 'slim',
    artifact       TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    features       TEXT NOT NULL,          -- JSON list, ordered, verbatim
    target_alias   TEXT,                   -- the column name the features name
    feature_recipe TEXT,                   -- JSON: lag, windows, stats, time cols
    env            TEXT,                   -- JSON: library versions at fit time
    -- One feature row and what the model predicted from it at import time.
    -- Re-run on every load: a library upgrade that changes behaviour silently
    -- is what this catches, and a version-string comparison cannot.
    golden_input   TEXT,                   -- JSON list, aligned with `features`
    golden_output  REAL,
    target_src     TEXT NOT NULL,          -- measured | modelled
    metrics        TEXT,                   -- JSON: MAE by horizon, vs baselines
    note           TEXT,
    imported_at    TEXT NOT NULL,
    trained_from   TEXT, trained_to TEXT,
    active         INTEGER NOT NULL DEFAULT 0,
    activated_at   TEXT,
    activated_by   TEXT,
    CHECK (active = 0 OR target_src = 'measured'),
    CHECK (active = 0 OR (tx IS NOT NULL AND rx IS NOT NULL))
);

-- Identity, and the reason it is an expression index rather than a UNIQUE on
-- the table: `tx` and `rx` are NULL for an unbound model, and NULL is not
-- equal to NULL in a unique constraint, so two identical unbound imports would
-- both be accepted. Re-importing a file is the first thing anyone does when an
-- import looks wrong, and it must update the row rather than accumulate rows.
CREATE UNIQUE INDEX IF NOT EXISTS model_registry_identity ON model_registry(
    name, param, COALESCE(tx, ''), COALESCE(rx, ''), sha256);

-- One active model per circuit and parameter. A partial unique index rather
-- than a trigger: the constraint is declarative, and an activation that would
-- create a second one fails at the write instead of leaving two live.
CREATE UNIQUE INDEX IF NOT EXISTS model_registry_one_active
    ON model_registry(param, tx, rx) WHERE active = 1;

CREATE INDEX IF NOT EXISTS model_registry_param ON model_registry(param, tx, rx);


-- One row per (model, issue, valid time, parameter).
--
-- Old issues are never deleted -- they are what horizon scoring is computed
-- from, and a forecast you cannot score is decoration. On Postgres this table
-- is PARTITION BY RANGE (issued_at) monthly; SQLite has no partitioning, so
-- here it is one table with the index that matters.
--
-- `sigma` is the model's own uncertainty and `lo`/`hi` a prediction interval
-- where the model produces one. Both nullable: a point estimate that claims a
-- confidence it does not have is worse than one that admits it has none.
CREATE TABLE IF NOT EXISTS forecast (
    model_id   INTEGER NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
    param      TEXT NOT NULL,
    tx         TEXT NOT NULL, rx TEXT NOT NULL,
    issued_at  TEXT NOT NULL,
    valid_at   TEXT NOT NULL,
    horizon_s  INTEGER NOT NULL,
    value      REAL, sigma REAL, lo REAL, hi REAL,
    -- JSON: the solar driver's age and which source answered, the version skew
    -- if the model was run across one, and whether the input window was short.
    -- Travels with the value, never separately -- same rule as `extraction`.
    quality    TEXT,
    PRIMARY KEY (model_id, param, tx, rx, issued_at, valid_at)
);

CREATE INDEX IF NOT EXISTS forecast_valid ON forecast(param, tx, rx, valid_at);
CREATE INDEX IF NOT EXISTS forecast_issue ON forecast(issued_at);

-- Scores: how a model, or a baseline, did against the measurements.
--
-- **Models and baselines share this table on purpose.** A leaderboard that can
-- only hold models invites promoting the best of a bad set; putting
-- persistence and 27-day recurrence in the same rows, in the same units and
-- over the same pairs, makes "none of these is worth activating" a visible
-- answer. `subject` is `model:<id>` or `baseline:<name>` -- deliberately not a
-- foreign key, because half the rows have nothing to point at.
--
-- One row per subject, circuit and horizon: the latest scoring run replaces
-- the previous one. No history is lost by that -- `forecast` keeps every issue,
-- so any past window can be rescored.
--
-- `mae`/`rmse`/`bias` cover uncensored pairs only, and the band-edge picks are
-- counted separately in `n_censored`/`mae_censored`. Mixing them would let a
-- model look accurate by agreeing with a bound that was never a measurement.
CREATE TABLE IF NOT EXISTS score (
    subject      TEXT NOT NULL,          -- model:<id> | baseline:<name>
    param        TEXT NOT NULL,
    tx           TEXT NOT NULL, rx TEXT NOT NULL,
    horizon_s    INTEGER NOT NULL,       -- the bucket, not the exact lead
    scored_at    TEXT NOT NULL,
    window_from  TEXT, window_to TEXT,
    n            INTEGER NOT NULL,
    mae          REAL, rmse REAL, bias REAL,
    n_censored   INTEGER NOT NULL DEFAULT 0,
    mae_censored REAL,
    detail       TEXT,                   -- JSON: why a baseline is missing, etc.
    PRIMARY KEY (subject, param, tx, rx, horizon_s)
);

CREATE INDEX IF NOT EXISTS score_circuit ON score(param, tx, rx, horizon_s);


-- --------------------------------------------------------------------------
-- Two work queues, and the reason they exist
-- --------------------------------------------------------------------------
--
-- Registering a model means running code out of a file; training one means
-- running code that fits. Neither belongs in the process that answers HTTP,
-- and `services/prediction/importer.py` refused an inbound route for exactly
-- that reason. These tables are how the refusal is kept while the console
-- still gets a button: the api writes a row and never opens the artifact, and
-- a worker with no listening socket does the part that executes.

-- One uploaded artifact awaiting registration.
--
-- `sha256` is the identity, here as everywhere: the quarantine file is named
-- by it, so two uploads of the same bytes converge on one blob and one object.
-- The binding columns carry what the operator asked for, verbatim, to be
-- handed to `importer.import_artifact` unchanged -- the console path must not
-- become a second implementation of the shell one.
--
-- `detail` holds the refusal *sentence*, not a code. It is rendered in full on
-- the forecast page, in the same discipline the baselines table follows: an
-- unavailable thing states why, because a blank reads as neglect.
CREATE TABLE IF NOT EXISTS model_upload (
    id          INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL,          -- as the operator named it, for display
    sha256      TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    name        TEXT,
    param       TEXT NOT NULL,
    tx          TEXT, rx TEXT,
    origin      TEXT NOT NULL DEFAULT 'imported',
    target_src  TEXT,                   -- NULL: let the importer's default rule apply
    period      INTEGER,
    note        TEXT,
    state       TEXT NOT NULL DEFAULT 'pending',
    detail      TEXT,
    model_id    INTEGER REFERENCES model_registry(id) ON DELETE SET NULL,
    uploaded_at TEXT NOT NULL,
    uploaded_by TEXT,
    settled_at  TEXT,
    CHECK (state IN ('pending', 'registered', 'refused'))
);

CREATE INDEX IF NOT EXISTS model_upload_state ON model_upload(state, uploaded_at);
CREATE INDEX IF NOT EXISTS model_upload_sha ON model_upload(sha256);


-- One requested training run.
--
-- The recipe and estimator knobs travel as one JSON `spec` rather than a
-- column each: they are a single argument set handed to one function, and a
-- column per knob would need a migration every time the trainer learns a new
-- one. `param`/`tx`/`rx`/`method` are promoted out of it because they are what
-- the console lists jobs by.
--
-- `running` is a state and not a flag so that a worker restarted mid-fit
-- leaves something visible. A job silently returned to `queued` would be
-- retried for ever by a fit that crashes the container.
CREATE TABLE IF NOT EXISTS train_job (
    id           INTEGER PRIMARY KEY,
    param        TEXT NOT NULL,
    tx           TEXT NOT NULL, rx TEXT NOT NULL,
    method       TEXT NOT NULL DEFAULT 'contour',
    spec         TEXT NOT NULL,         -- JSON: lag, windows, stats, estimator, holdout
    state        TEXT NOT NULL DEFAULT 'queued',
    detail       TEXT,
    model_id     INTEGER REFERENCES model_registry(id) ON DELETE SET NULL,
    requested_at TEXT NOT NULL,
    requested_by TEXT,
    started_at   TEXT,
    settled_at   TEXT,
    CHECK (state IN ('queued', 'running', 'done', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS train_job_state ON train_job(state, requested_at);


-- One requested inference pass.
--
-- The third queue, and the one that closes the loop: activating a model does
-- not produce a forecast, `infer` does, and until this existed the only way to
-- make that happen was a shell on the host. Every other step of a model's life
-- is a button, so this is one too.
--
-- Drained by `infer` itself rather than by a fourth container. That process
-- already mounts the artifact store, already loads models and already writes
-- the rows -- what it lacked was a reason to wake up before its interval
-- elapsed, which is a change to how it sleeps rather than a new service.
--
-- `model_id` NULL means "whatever is active for this circuit", which is the
-- button on the Live panel. Naming one runs it as a comparison, exactly as
-- `infer --model` does from a shell, without promoting anything.
CREATE TABLE IF NOT EXISTS infer_job (
    id           INTEGER PRIMARY KEY,
    param        TEXT NOT NULL,
    tx           TEXT NOT NULL, rx TEXT NOT NULL,
    method       TEXT NOT NULL DEFAULT 'contour',
    model_id     INTEGER REFERENCES model_registry(id) ON DELETE SET NULL,
    state        TEXT NOT NULL DEFAULT 'queued',
    detail       TEXT,
    -- Forecast rows the pass wrote. Nullable rather than 0 by default: "has
    -- not run" and "ran and wrote nothing" are different answers, and the
    -- second one is the interesting failure.
    written      INTEGER,
    backtest     INTEGER,
    requested_at TEXT NOT NULL,
    requested_by TEXT,
    started_at   TEXT,
    settled_at   TEXT,
    CHECK (state IN ('queued', 'running', 'done', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS infer_job_state ON infer_job(state, requested_at);
