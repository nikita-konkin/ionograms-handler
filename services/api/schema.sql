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
