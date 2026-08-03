# System architecture

Target structure for the ionospheric sounding, extraction and forecasting
system. Companion to [`signal-chain.md`](signal-chain.md), which covers
everything from the antenna to the gated ionogram.

Status of each decision is marked: **[settled]**, **[proposed]** (my
recommendation, not yet confirmed), or **[open]**.

---

## 1. Overview

Three tiers, split at the point where physics ends and data processing begins.

```
┌─ TIER 1 · ACQUISITION ─────────────────── laptop + USRP, Ubuntu, no container ─┐
│                                                                                │
│   HF antenna → USRP → chirpsounder v2 fork   (target)                          │
│                       chirpsounder v1 fork   (current, until cutover)          │
│                                  │                                             │
│                          station agent  ──►  health JSON / narrow control       │
│                                                                                │
│   OUT:  lfm_ionogram-*.h5   multi-station, ~0.4–15 MB    (v2)                  │
│         cyprus1_*.lfs       single path,   80 MB          (v1)                 │
└────────────────────────────────────────────────────────────────────────────────┘
                    │                              ▲
                    │  BULK FILE TRANSFER          │  health (push)
                    │  (rsync / Syncthing)         │  control (pull, authed)
                    │  not REST — see §5.1         │  see §5.4
                    ▼                              │
┌─ TIER 2 · PROCESSING ──────────────────────── server, 16 TB, containerized ────┐
│                                                                                │
│   archive  ──►  extractor  (queue worker, muf package)                         │
│      │              │   .lfs → spectro → calibrate ─┐                          │
│      │              │   .h5  → io_chirp ────────────┴─► estimators → pick → fit│
│      │              ▼                                                          │
│      │         PostgreSQL   sounding / extraction / reference /                │
│      │              ▲       forecast / config_epoch                            │
│      └──► renderer ─┘   ionogram PNG + SAO.XML, lazily, on request             │
│                                                                                │
│         prediction  ────────────────────────►  writes forecast rows            │
│           (N:\muf models)                                                      │
│                                                                                │
│         api  ───────────────────────────────►  REST, the only public surface   │
└────────────────────────────────────────────────────────────────────────────────┘
                              │  REST
                              ▼
┌─ TIER 3 · CONSUMPTION ─────────────────────────────────────────────────────────┐
│   web client — ionograms, diurnal curves, forecasts, station health, control   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tier 1 — Acquisition

### 2.1 Migration to chirpsounder2 **[settled: direction]**

**Multi-station sounding is the requirement that decides this.**

| | v1 fork (current) | v2 fork (target) |
|---|---|---|
| Transmitters | Schedule-driven; each period, chirptime, rate, duration and LO known in advance | `detect_chirps.py` finds unknown sweeps; `find_timings.py` infers repetition |
| Adding a station | Requires updating the acquisition program | Automatic |
| Output | `.lfs` — raw IQ, 80 MB/sounding | `lfm_ionogram-*.h5` — gzipped float16 SNR |
| Upstream | Explicitly unmaintained | Active (commits through Jul 2026) |
| Operator GUI | Qt console — genuinely good | None |

Storage settles it. From `signal-chain.md` §6:

| | per day | per year | 16 TB lasts |
|---|---|---|---|
| six stations, `.lfs` | ~260 GB | ~95 TB | **~62 days** |
| v2 products, ~2400 ionograms/day | ~1–36 GB | 0.4–13 TB | 1.2–40 years |

Multi-station is unaffordable on `.lfs` and comfortable on v2. This also
resolves the archive-format question previously left open (§5.1).

### 2.2 Fork by subtraction **[settled: principle]**

**Only delete from the fork; never add.** Upstream does not conflict with
deletions, so `git pull` stays clean indefinitely. Adding a custom writer is
precisely what stranded v1.

| Action | Files |
|---|---|
| **Keep** | `detect_chirps.py`, `find_timings.py`, `calc_ionograms.py`, `detections2metadata.py`, Digital RF, `chirp_config.py` |
| **Delete** | `web/` (all PHP), `ionowebsync.py`, `sync_iono_data.py` |
| **Keep, wrap** | `station_monitor.py` — already emits atomic JSON status; the station agent extends it rather than starting from zero |
| **Add** | Nothing. `io_chirp.py` lives in the `muf` package, not in the fork |

The web layer is dropped entirely. Tier 2's `api` is the single web surface for
the whole project — acquisition, extraction and forecasting — which is what the
PHP dashboard could never have been.

### 2.3 Migration cost — budget these

1. **`muf/io_chirp.py`** — ~150 lines; see §3.
2. **Noise-normalization offset** — the 43 dB threshold is 13 dB SNR *given*
   this pipeline's `4·ln2·median` normalization and 30 dB floor
   (`signal-chain.md` §5.2). v2's `SNR` is referenced to its own `noise_floor`.
   **Measure the offset, do not assume it.**
3. **Station coordinate registry** — v2's h5 carries `txname`/`station_name`
   strings but no lat/lon; `geometry.py` needs coordinates from config.
4. **Loss of the Qt console** — v2 has no GUI equivalent. The web client must
   replace it, which moves Tier 3 from "nice to have" to "required before
   cutover."
5. **`io_lfs.py` is permanent** — the historical archive stays `.lfs`.

### 2.4 Cutover: parallel run **[proposed]**

Split the antenna to a second USRP and run v1 and v2 simultaneously on
`cyprus1`. Comparing extracted MUF sounding-by-sounding on a path whose answer
is already known validates `io_chirp.py`, the coordinate registry, and — most
importantly — **measures** the normalization offset in §2.3.2 rather than
guessing it.

Without a second receiver, only an A/B in time is possible (a week each), which
is substantially weaker: the ionosphere differs between the periods.

> The second USRP is the **long-pole procurement item**. It is not needed until
> the cutover, but if it is wanted it should be ordered early.

### 2.5 Station agent **[proposed]**

Small daemon beside the acquisition process. The only genuinely new Tier 1
component.

Remote *viewing* is already solved — the Qt console is reached over AnyDesk. It
shows session progress, CPU, disk, live ionograms, S/N and PDP. What it does
not give is **history, alerting, or unattended operation**. That is the gap,
and it is much narrower than "build a web control plane."

#### Health — read-only, ship first

Atomic JSON status pushed to `api`. Metric list adapted from v2's
`station_monitor.py`, plus items specific to this station:

| Metric | Why |
|---|---|
| Process liveness — receiver, transfer job | the basic question |
| Age of newest product file | soundings stopped ≠ process died |
| Disk free on the data volume | **the binding constraint** |
| Transfer backlog — files awaiting sync | leading indicator of disk exhaustion |
| Sample rate vs expected 25 MHz | silent misconfiguration |
| `sweep_fraction` of recent soundings | catches the truncations in `BACKLOG.md` §4 |
| USRP-vs-system clock lag | `signal-chain.md` §7.3 |
| Startup grace period | avoid alerting during boot |

Read-only, near-zero risk, roughly all of the value. The disk-full failure mode
is what this exists to prevent.

#### Control — narrow, authenticated, journaled

Deliberately smaller than the Qt console: start/stop acquisition, edit the
active schedule, trigger a transfer. Not a remote replica of every local
control.

> **Any change to acquisition parameters must be journaled to the database.**
> The schedule sets `rate`, `dur` and `cf`, which determine the sweep bounds,
> the range axis, the frequency step — every derived quantity in
> `signal-chain.md`. A `dur` edit made from a web form silently changes what
> all subsequent numbers mean. Control endpoints touching acquisition
> parameters write a `config_epoch` row so every sounding can be attributed to
> the configuration that produced it.

Authentication from day one: this is RF acquisition hardware.

---

## 3. Dual-format input **[settled: requirement]**

The `muf` package reads **both** `.lfs` and `.h5`. Not a transitional measure —
the historical archive is `.lfs` permanently, and both must remain queryable
through one pipeline.

### 3.1 Where the formats converge

`Ionogram` (gated power array + `Calibration` + header) is the common type.
Everything downstream of it — estimators, `pick`, `fit`, `track`, `render`,
`export` — is already format-agnostic and needs no change.

```
.lfs → io_lfs.read_header  → spectro.compute  → calibrate ─┐
                              (FFT + gate)                  ├─► Ionogram → …
.h5  → io_chirp.load ──────────────────────────────────────┘
        (already gated by calc_ionograms.py)
```

The `.h5` path **skips `spectro` and `calibrate` entirely** — v2 has already
done the FFT and the range gate. This is why the reader is small.

### 3.2 Dispatch

```python
def load(path, *, window, zero_periods, gate_km) -> Ionogram:
    if path.suffix == ".lfs":  return spectro.load(path, window, zero_periods, gate_km)
    if path.suffix == ".h5":   return io_chirp.load(path, gate_km)
    raise ValueError(...)
```

Selected by extension; `--format` on the CLI only to override. `window` and
`zero_periods` are **meaningless for `.h5`** — the window was fixed when v2
computed the product. Passing them should warn, not silently no-op.

### 3.3 What `io_chirp.load()` must handle

| Concern | Treatment |
|---|---|
| Normalization | Divide `SNR` by `noise_floor` into the median-equalized convention so **`to_db()` and the 43 dB threshold keep their meaning**. Offset measured per §2.4, not assumed. |
| Geometry | `txname`/`station_name` → coordinates via the station registry; `geometry.py` unchanged. |
| Axes | `Calibration` built directly from the `freqs` and `ranges` datasets rather than derived from header arithmetic. |
| Gating | v2 already gated (`range_gate_start_m`/`stop_m`, `range_offset_applied`). Apply the `muf` gate **only if narrower**; never double-gate. |
| Completeness | No direct `sweep_complete` equivalent — derive from `freqs` coverage against the nominal sweep. |
| Caching | Cache key needs a format discriminator; `w`/`z` are not meaningful for `.h5`. |

### 3.4 The consequence to record

**`.h5` soundings cannot be re-derived at a different FFT window.** This is
`BACKLOG.md` §1's "window is locked at archive time," now concrete. The
`sounding` table therefore carries `format` and `window` so it is always
visible which soundings can be reprocessed and which cannot.

Mitigation if it matters: v2 can optionally store raw `z` (downconverted
complex64) in the h5. Enable it on a rolling window (30–60 days), not
permanently.

---

## 4. Tier 2 — Processing

All containerized. Four services, each independently restartable.

### 4.1 extractor **[proposed]**

Queue worker. Watches the archive, runs the `muf` pipeline, writes rows.

- One job per sounding file; idempotent, keyed on `(file, method)`
- Runs every enabled estimator (`algo`, `kmeans`, `contour`, `cnn`) — one row each
- Emits quality columns alongside every value, never separately
- The expensive service; scale by worker count

The only component that reads sounding files. Everything downstream reads the
database.

### 4.2 renderer **[proposed]**

Ionogram PNGs and SAO.XML, generated **on request** rather than precomputed.
288 images/day precomputed is 105 k files/year that mostly nobody opens;
rendering from a cached gated product is ~0.2 s.

### 4.3 api **[proposed]**

The only network-facing surface.

```
GET /soundings?tx=&rx=&from=&to=
GET /series/muf?method=&from=&to=&smoothed=
GET /ionogram/{id}.png            → renderer
GET /sao/{id}.xml                 → renderer
GET /forecast?horizon=&from=

GET  /health/stations                  latest status per station
GET  /health/stations/{id}/history     for trends and alerting
POST /health/report                    ← station agent pushes here

GET  /stations/{id}/schedule           authed
POST /stations/{id}/schedule           authed, writes a config_epoch row
POST /stations/{id}/acquisition        authed, start | stop
POST /stations/{id}/transfer           authed, trigger sync
```

One surface across acquisition, extraction and forecasting — but **separate
auth scopes**. Public read of soundings and forecasts must not share a scope
with anything that can stop an acquisition.

### 4.4 prediction **[settled in principle]**

Reads the smoothed/tracked series, writes forecast rows. Carries the `N:\muf`
models.

Must read `muf_smooth` from `track`/`daily`, not raw picks — **and must still
see `limited`/`loflim`**. Otherwise it trains on midday lower bounds, which per
`BACKLOG.md` §3 are biased ~5 MHz low for 82 of 288 soundings.

**[blocked]** — see §8.

---

## 5. Data contracts

### 5.1 Laptop → server: bulk file transfer **[settled]**

Sounding files move to the server's HDD by file sync, **not** REST. At scale,
HTTP multipart gives no resumption, no backpressure, and timeouts on a laptop
uplink; `rsync`/Syncthing gives all three.

Automatic, with delete-after-confirmed-copy. The laptop disk is what keeps
acquisition alive, not a convenience.

**Archive format — now settled by §2.1:** v2 `.h5` products going forward,
`.lfs` retained permanently for the historical record. Optional rolling raw `z`
window per §3.4.

### 5.2 Database schema **[settled: long form]**

Long, not wide — adding a fifth estimator must not be a migration.

```sql
-- one row per sounding file: acquisition facts and derived calibration
sounding(
  id, file, format, window,       -- format/window: is this re-derivable? (§3.4)
  datetime, tx, rx, path_type,
  tx_lat, tx_lon, rx_lat, rx_lon, path_km,
  freq_start, freq_stop, gate_lo, gate_hi,
  sweep_complete, sweep_fraction,
  config_epoch_id                 -- which configuration produced it
)

-- one row per (sounding, method) — the long axis
extraction(
  sounding_id, method,
  muf, lof, vrange, snr,
  ndet, run, nseg, hops, branch, scatter,
  fit, fitres, fitex,
  limited, loflim,
  muf_smooth                      -- from track/daily; NULL until smoothing runs
)

-- modelled and third-party values, kept structurally apart
reference(
  sounding_id, source,            -- iri | giro | chapman | minimuf
  param, value                    -- muf | fof2 | hmf2 | ...
)

-- acquisition configuration over time; written by control endpoints (§2.5)
config_epoch(
  id, station, valid_from, valid_to,
  rate, dur, cf, rep, chirpt, sample_rate, dec,
  changed_by, note
)

-- prediction service output
forecast(
  issued_at, valid_at, horizon_days, model, param, value, quality
)
```

The normalized form of the 68-column run CSV.

**`reference` is a separate table by design.** IRI is validation only. If
modelled and measured values ever share a table, the prediction service can
train on IRI and produce circular validation. Structural separation, not
convention.

**[proposed]** `fof2` derived from oblique MUF via `geometry.py` belongs in
`reference`, not `extraction` — it is a secant-law inference resting on a
300 km `hmF2` fallback, not a measurement.

### 5.3 Server-internal: REST **[settled]**

`api` ↔ web client. Services do not push payloads to one another; `renderer`
output is fetched, not posted.

### 5.4 Station ↔ server: health and control **[proposed]**

Separate from the bulk path (§5.1), with opposite direction of trust.

- **Health: push.** The agent POSTs JSON on an interval. Silence is itself the
  alert — no polling, and it works behind NAT.
- **Control: pull.** The agent polls for pending commands and acknowledges
  them. The station is never a listening service, keeping acquisition hardware
  off any inbound path.

Commands are queued, acknowledged, and recorded with the identity that issued
them. Parameter changes additionally write `config_epoch`.

---

## 6. Release plan

Ordered so that **everything through M3 is independent of the v1/v2 decision**.
That is deliberate: those steps make the migration *measurable* instead of
speculative, and none of the work is wasted whichever way it goes.

### M0 — Stop the bleeding
- Automatic file transfer, laptop → server, with delete-after-confirm
- Station agent, **health only**, read-only JSON
- Fix `config_enum.py`'s `input()` (§8) — no dependencies, unblocks M6

*Exit:* the disk cannot fill silently, and you learn when acquisition stops.

### M1 — Database and backfill
- Schema per §5.2, plus loader (port `data_handler/muf_load_to_db.py`)
- Backfill every historical `.lfs` extraction

*Exit:* the dataset is queryable — immediately useful for dissertation work,
independent of everything else.

### M2 — Extractor as a service
- Queue worker wrapping the existing `muf` CLI
- New soundings reach the database without intervention

### M3 — API and web, read-only
- Health views, ionogram browse, MUF/LOF series
- Answers "is it running?" without AnyDesk

### M4 — v2 fork and parallel run
- Fork by subtraction (§2.2); `muf/io_chirp.py` (§3)
- **Measure** the normalization offset against v1 output (§2.4)
- Station coordinate registry

*Exit:* v2 MUF agrees with v1 MUF on `cyprus1` within a stated tolerance.

### M5 — Multi-station and cutover
- Enable opportunistic detection; retire v1 once the web client replaces the
  Qt console
- Control endpoints, auth, `config_epoch`

### M6 — Prediction service
- Retrain on the accumulated multi-station record

---

## 7. Repository map

| Repo | Tier | Role |
|---|---|---|
| `chirpsounder` v1 fork | 1 | Current acquisition; `.lfs` writer, Qt console. Frozen, retired at M5 |
| `chirpsounder2` fork | 1 | Target acquisition; fork by subtraction, tracks upstream |
| `N:\ionograms-handler` | 2 | `muf` package — extraction, dual-format IO, SAO export, references |
| `N:\muf` | 2 | Forecasting models, BER / channel availability |

Kept separate. `muf` is a clean installable with its own CLI and test suite;
folding it into a GNU Radio application would destroy that, and the acquisition
forks have a different lifecycle from everything else.

---

## 8. Blockers and known risks

**`N:\muf\config_enum.py` calls `input()` inside an `Enum` class body.** The
config prompts on import, so nothing in the forecasting project can run
unattended. Prerequisite zero, with no dependencies of its own — scheduled at
M0 for that reason.

**The 43 dB detection threshold is not portable.** It is 13 dB SNR *given* this
pipeline's noise normalization (`signal-chain.md` §5.2). The v2 migration must
measure the offset (§2.4), not assume it. This is the failure mode most likely
to produce plausible-looking wrong numbers.

**Midday MUF is a lower bound on the cyprus1 path.** `BACKLOG.md` §3 — the
32.5 MHz sweep top is below the real MUF for 82 of 288 soundings. Not fixable
from the receiver side; needs a different transmitter, which is what M5
delivers.

**Losing the Qt console at cutover.** v2 has no GUI equivalent, so M5 cannot
complete until the web client covers what operators actually use.

**The v1 fork has no upstream.** Until M5, losing it loses both the `.lfs`
writer and the console. Version-controlled and OS-image-pinned in the meantime.
