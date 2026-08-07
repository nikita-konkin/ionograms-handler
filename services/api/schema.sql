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
