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

### 2.2 Clone, do not fork **[settled: principle]**

**Change nothing upstream maintains. Do not deploy what you do not want.**

Taken to its conclusion, this means **there is no fork**. chirpsounder2 takes
its config as a file path, so the config lives in `ionograms-handler` and
chirpsounder2 stays a plain clone pinned to a commit — with nothing of ours in
it, `git pull` has nothing to conflict with. Fork only if something must be
patched; if that happens the patch will be small, and since the project is MIT
with an active maintainer, offer it upstream rather than carrying it.

What changes for v2 is therefore **one `.ini` file and a set of systemd units**
(§2.5). All the code is on our side: `io_chirp.py`, the detection-file reader,
the coordinate registry, the agent.

An earlier draft of this section said "delete the PHP." That is backwards.
Git merges per file, so:

- **Deleting** a file upstream still maintains produces a delete/modify
  conflict on every pull that touches it. `web/` is *actively* worked on —
  "Fix W2NAF web uploads", "Publish W2NAF near-range ionograms" (Jun 2026) —
  so deleting it means fighting upstream indefinitely.
- **Adding** a new path essentially never conflicts, because upstream is not
  writing there.

So the PHP stays in the tree and is simply never deployed: `web/deploy_web.sh`
is not run, and `ionowebsync.py` only posts when a URL is configured, so
leaving both in place costs nothing and risks nothing. Tier 2's `api` is the
single web surface regardless.

| Action | Files |
|---|---|
| **Use** | `detect_chirps.py`, `find_timings.py`, `calc_ionograms.py`, `detections2metadata.py`, Digital RF, `chirp_config.py` |
| **Leave in place, never deploy** | `web/` (all PHP), `ionowebsync.py`, `sync_iono_data.py` |
| **Read, do not modify** | `station_monitor.py` — the station agent shells out to it or reuses its metric list; it is not edited in place |
| **Add** | Nothing. The agent and `io_chirp.py` live in `ionograms-handler` (§7) |

What *did* strand v1 was adding a custom `.lfs` writer into the acquisition
source. The rule that matters is: **our code never lands in the fork**, in
either direction. Configuration in the fork; code in our repo.

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
| **Host clock plausibility** | see below — the epoch is copied from it verbatim |
| **Epoch offset vs a reference transmitter** | the 0.956 s fault; `muf.io_detect.solve_epoch_offset` |

Read-only, near-zero risk, roughly all of the value. The disk-full failure mode
is what this exists to prevent.

##### Why the host clock is a metric of its own

`rx_uhd_ext_gps` sets the USRP epoch from the **host clock**, not from the
GPSDO:

```cpp
// rx_uhd_ext_gps.cpp:433
usrp->set_time_next_pps(uhd::time_spec_t(pc_secs + 1));
```

It selects `gpsdo` as clock and time source, waits for `gps_locked`, prints
the result — and then never reads the `gps_time` sensor. So the PPS *edge* is
disciplined to sub-microsecond while the *second number* is whatever `ntpd`
last left behind. `rx_uhd.cpp:312` does read `gps_time`; the `_ext_gps`
variant, despite the name, is the one that inherits every NTP error.

This is not hypothetical. It produced DOB's 0.956 s offset on 2026-08-05
(300 km of range error per millisecond, and the transmit *second* wrong on
top), and on 2026-08-06 a dead RTC made the same line announce
`PC time now: 1617339242` — 2021-04-02, five years of mis-stamped data.

The epoch-offset metric cannot cover this case: a clock that wrong means there
are no recent products to solve against, so it reports "no timing solutions"
and the operator learns nothing. `health.system_clock` answers from nothing —
a sanity floor, a comparison against files already on disk (which needs no
hardcoded date), and NTP's own synchronisation state.

Upstream would fix it in one line, but chirpsounder2 is a pinned clone with
nothing of ours in it (§7), so we detect rather than patch, and the systemd
unit orders acquisition `After=time-sync.target`.

#### What "start/stop sounding" actually controls

v2 is not one program. It is independent long-running processes sharing a
config and a ringbuffer, so supervision is external and chirpsounder2 is not
modified to support it.

| Process | Role | Needed? |
|---|---|---|
| `rx_uhd_ext_gps` | Records USRP → Digital RF | yes |
| `detect_chirps.py` | Finds LFM sweeps — **this is stations search** | yes |
| `detections2metadata.py` | Consolidates detections | yes |
| `calc_ionograms.py` | Sweeps → `lfm_ionogram-*.h5` | yes |
| `plot_ionograms.py`, `plot_rtf.py`, `plot_detectionfiles.py` | PNG products | **no** — Tier 2 renders |
| `station_monitor.py` | Health JSON | reuse, unmodified |
| `sync_iono_data.py`, `ionowebsync.py` | Publishing | no |

Wrap each in a systemd unit under one `chirp.target`, bound with
`Requires=` / `After=` / `PartOf=`. The recorder must come up before its
consumers and go down after them, so the agent starts and stops **one target**
rather than sequencing five. Control endpoints (§4.3) map to
`systemctl start|stop chirp.target`.

**Three unit details this station has already paid for.** They are not
polish; each one cost a day.

- **`KillSignal=SIGINT` and a real `TimeoutStopSec=` on the recorder.** A
  USRP killed mid-stream keeps transmitting UDP to a host that is gone and
  wedges: it stops answering ARP and discovery, and no software on the host
  can recover it — only removing power. UHD's handler sends the
  stop-streaming command on SIGINT. `SIGTERM`/`SIGKILL` do not.
- **`ExecStartPre=` for settings that do not survive a reboot.**
  `ethtool -G <iface> rx 4096` resets to 256 on every boot, and at 25 MS/s
  over 1 GbE the default ring drops packets continuously.
- **Real-time priority, granted per-binary.** `setcap cap_sys_nice+ep` on the
  recorder is cleared by any rebuild, and `limits.conf` does nothing if
  `pam_limits.so` is absent from `common-session` — which it was here. A
  systemd unit sidesteps both: set `LimitRTPRIO=` in the unit and neither PAM
  nor file capabilities are involved.
- **`CPUAffinity=0` on the recorder, with `CPUAffinity=1-7` on every
  consumer.** The exclusion is half of the setting and worth nothing without
  it; unpinned, the recorder lost 358,691 samples per 900 s at load 9.4 with
  CPU still idle, because the fault is scheduling latency and not throughput.
- **`OOMScoreAdjust=-1000` on the recorder.** `KillSignal=` protects the USRP
  from systemd; this protects it from the kernel, which is the same site visit
  by another route — the OOM killer sends SIGKILL directly and `KillSignal` has
  no say. The recorder is the largest RSS on a box holding 13 GB of
  non-reclaimable tmpfs, i.e. exactly what the OOM killer selects. It is a
  precondition for growing the ringbuffer, not an afterthought to it.

**Write these against the station's systemd version, not the laptop's.** DOB
runs **systemd 229**; an unknown directive there is only a warning, so the unit
loads and fails somewhere that looks unrelated, while an unknown *value* or
prefix stops the unit loading at all. Both had already happened in this
directory before anything was installed — `docs/2026-08-13-systemd-229.md` has
the two cases, and `tests/test_systemd_units.py` now pins the floor.

#### Two acquisition modes, and one flag that is independent of both

`[lfm] serendipitous` selects between them, and they are not "search" versus
"receiver" — the second is a *schedule*, not a set of receivers:

| | `serendipitous = true` | `serendipitous = false` (default) |
|---|---|---|
| what runs | `analyze_parfiles` (`calc_ionograms.py:615`) | `sounder_timings[rank]` (`:422`) |
| which transmitters | whatever `find_timings.py` discovers | exactly those listed in `[lfm] sounder_timings` |
| `t0` | **measured** from detections | **imposed** from `rep` and `chirpt` |
| `txname` in the product | `"unkown"` (upstream's spelling, `:189`) | `transmit_name` from the config (`:456`) |
| geometry | unavailable — no name to look up | resolves via the station registry |

One process per sounder: `mpirun -np N` with N ≥ the number of entries, since
`sounder_timings` is indexed by MPI rank.

**`save_raw_voltage` is independent of the mode.** It is a plain `[lfm]` option
and applies in both, so switching to a schedule does not give you waveforms and
saving waveforms does not require a schedule. See §3.4.

The practical reason to run a schedule is that `t0` stops being measured. A
receiver whose epoch is wrong corrupts every range in serendipitous mode
(§3.4's cousin, and the 2026-08-05 DOB fault) — but a schedule does not rescue
it either: the dechirp reference is still built on the receiver's own clock, so
a 45 ms epoch error still displaces the echo by 13,500 km. It puts the echo
*outside a narrow `manual_range_extent` window*, which is the most likely
explanation for the 2026-08-04 archive being 381 consecutive empty ionograms
on a path that was 1013 km long. Unproven — the data outside the window was
discarded at acquisition — but the detection files from that day put the epoch
phase at ~43 ms, so the echo would have sat near 13,900 km against a stored
window of 800–1698 km.

#### Stations search, and reading the answer back

Enabling the search is running `detect_chirps.py`; its sensitivity is one
`[detection]` config value. The results are files:

| File | From | Holds |
|---|---|---|
| `chirp-*.h5` | `detect_chirps.py` | Raw detections: time, start frequency, rate, SNR |
| `cdetections-*.h5` | `detections2metadata.py` | Time-binned summaries |
| `par-*.h5` | `find_timings.py` | **Inferred repeat schedules per transmitter** |

**A detection-file reader is new work**, in `ionograms-handler` beside
`io_chirp.py` (§3). "Which transmitters are we hearing, on what schedule" is a
question answered by `par-*.h5` and `cdetections-*.h5`, and it feeds the API
and the web view.

This is the whole point of the migration. v1's schedule table is
hand-maintained: a transmitter you do not know about is invisible, and one that
changes its schedule silently yields noise. v2 discovers transmitters and
writes what it found to a file. The search is not something to build — it is
what is being migrated *to*. The work is the reader and the view.

#### Control — narrow, authenticated, journaled

Deliberately smaller than the Qt console. The surface is:

| Command | Maps to |
|---|---|
| start / stop / restart sounding | `systemctl start\|stop\|restart chirp.target` |
| change acquisition mode | `[lfm] serendipitous`, plus `sounder_timings` when scheduled (§2.5) |
| change storage path | `[config] output_dir` |
| change the active schedule | `sounder_timings` entries |
| trigger a transfer | the sync job, out of band from the target |

Not a remote replica of every local control. Anything not on this list is a
config edit made on the station.

**Parameter changes are edits to one `.ini`, applied by restart.** v2 reads
its config at process start and never re-reads it, so "change a parameter"
means: write the file, journal it, restart `chirp.target`. The agent owns that
sequence so a half-applied change cannot exist — and it writes the file
atomically, because a truncated config is an acquisition outage.

**A mode change is not just one flag.** Switching to `serendipitous = false`
without populating `sounder_timings` yields a station that records nothing and
reports healthy. The agent validates the combination before writing, and
refuses rather than applying half of it.

> **Any change to acquisition parameters must be journaled to the database.**
> The schedule sets `rate`, `dur` and `cf`, which determine the sweep bounds,
> the range axis, the frequency step — every derived quantity in
> `signal-chain.md`. A `dur` edit made from a web form silently changes what
> all subsequent numbers mean. Control endpoints touching acquisition
> parameters write a `config_epoch` row so every sounding can be attributed to
> the configuration that produced it.

Authentication from day one: this is RF acquisition hardware.

#### Logs — the third thing the agent exists for

Health says *that* something is wrong; logs say *what*. This station's failures
were all diagnosed by reading process output — dropped-packet markers,
`Unable to set the thread priority`, `ref_locked: false`, `no DigitalRF data
bounds available` — none of which any metric would have carried.

The agent exposes them **read-only, over the same pull channel as commands**,
so the station stays a non-listening service:

| Source | Why |
|---|---|
| `journalctl -u <unit>` per process in `chirp.target` | systemd already collects them; do not reinvent a log file |
| Last N lines, or a time window, bounded and rate-limited | a log endpoint that can return a gigabyte is a denial of service against your own station |
| Severity filter | the recorder is chatty by design; `D` markers are normal in ones and pathological in thousands |

**Counts belong in health, text belongs in logs.** "Dropped packets in the last
five minutes" is a metric and should trend; the surrounding text is what you
fetch once the metric moves. Shipping every line to the server continuously
would make the log the bulk-transfer problem of §5.1 all over again.

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

### 3.4 The consequence to record **[settled: mechanism]**

**By default, `.h5` soundings cannot be re-derived at a different FFT window.**
This is `BACKLOG.md` §1's "window is locked at archive time," now concrete: v2
computes its spectrogram, stores the result, and discards the waveform. `SNR`
is `|FFT|²` — magnitude squared, phase gone — so nothing can invert it. The
`sounding` table therefore carries `format` and `window` so it is always
visible which soundings can be reprocessed and which cannot.

**One config flag removes the limitation.** `save_raw_voltage = true` in v2's
`[lfm]` section (`chirp_config.py:91`, default false) makes `calc_ionograms.py`
write the dechirped voltage as dataset `z` alongside `SNR`
(`calc_ionograms.py:372`). That is the *same kind* of signal an `.lfs` file
carries — stretch-processed, decimated, complex — which is what makes
re-derivation possible rather than merely desirable.

Our side is built:

| Piece | Where |
|---|---|
| `ChirpHeader.has_raw_voltage` | `io_chirp` — True when the product carries `z` |
| `io_chirp.read_raw_voltage(path)` | the waveform, or an error naming the flag |
| `io_chirp.v2_spectrogram(...)` | a faithful port of `calc_ionograms.spectrogram`: 13× oversampled, Hann, on the conjugate |
| `io_chirp.reprocess(path, window)` | a new `Ionogram` at any window |
| `loader.load(..., window=N)` | routes to `reprocess` when `z` is present, warns and ignores when it is not |

The port is reimplemented rather than imported, for the §2.2 reason — the clone
is pinned and disposable, and our ionograms must not be a property of whichever
commit is checked out. It is kept honest by re-deriving a product at its
*original* window and requiring the stored array back.

**Reprocessing recovers two things storage destroys**, beyond the window:

- **Sparsification.** v2 writes NaN below `storage_snr_threshold`, and those
  cells read back as the row median — a hard floor at 25.571 dB with no noise
  texture beneath it. Anything estimating a noise *level* rather than crossing
  a threshold wants the real distribution.
- **float16 clipping.** `SNR` is stored `float16`, whose maximum 65504 lands at
  exactly **73.734 dB**. A stronger echo is clipped on the way to disk, so that
  value in a stored product means "at least this", not a measurement. The DOB
  archive of 2026-08-05 has 51 such cells in its strongest sounding alone.

**The cost is why it stays off by default.** At DOB — 486 frequency bins,
20000-sample step, 40 kHz decimated rate — `z` is 9.72 M complex64:

```
stored product today        0.6 MB
with z                     ~78 MB      (130x)
318 soundings/day          ~25 GB/day
```

Against DATA3's 928 GB free that is about 37 days. So: **enable it on a rolling
window (30–60 days), not permanently**, and let `iono_housekeeping.py` trim.
Turn it on before a campaign whose data will want reprocessing, not as a
standing default.

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

One scaling serves three surfaces — the XML download, the numbers panel and
the interactive plot — built once in `services/api/sao.py` and memoised on
path plus mtime, which identifies a write-once detection product for good.

**The raster stays a PNG.** 486 × 3999 cells is 1.9 M numbers, ~11 MB as JSON
against 164 KB as an image; only the few hundred scaled points cross the wire
as data. The image is placed in *data* coordinates under the traces, and its
extent runs half a cell past the first and last sample because
`pcolormesh(shading="nearest")` draws a cell centred on each one. Half a bin
out and every circle sits beside its echo instead of on it.

### 4.3 api **[proposed]**

The only network-facing surface.

```
GET /soundings?tx=&rx=&from=&to=
GET /series/muf?method=&from=&to=&smoothed=
GET /ionogram/{id}.png            → renderer; bare=true drops the axes and
                                  every overlay, for the interactive plot
                                  to place behind its own
GET /soundings/{id}/sao.xml       → SAO.XML 5.0, one record per estimator
GET /forecast?horizon=&from=
GET /net                          index-host reachability, last background
                                  pass only -- this route never probes

GET  /stations                         latest status per station, with age
GET  /stations/{id}/health             latest report and recent commands
GET  /stations/{id}/health/history     for trends and alerting

POST /stations/health                  ← station agent pushes here
GET  /stations/{id}/commands           ← agent pulls pending work
POST /stations/{id}/commands/{cid}/ack ← agent reports what it did

POST /stations/{id}/commands           authed, queue start | stop | restart
GET  /stations/{id}/schedule           live acquisition state: mode, slots,
                                       which one is sounding now, arrivals,
                                       and whether it is acquiring at all
POST /stations/{id}/schedule           authed, compose by transmitter name and
                                       queue it; a config_epoch row opens when
                                       the station acknowledges
GET  /stations/{id}/transmitters       verified transmitters at this receiver
POST /stations/{id}/transmitters       authed, identify one, with its evidence
DEL  /stations/{id}/transmitters/{code} authed
POST /stations/{id}/transfer           authed, trigger sync
```

> **The three agent paths were corrected to match `services/agent/client.py`.**
> This section originally proposed `POST /health/report`. The agent that got
> built posts to `/stations/health` and pulls from `/stations/{id}/commands`,
> and it is the deployed half — it lives on an acquisition laptop reached over
> AnyDesk, where a redeploy is a manual errand. The server serves what the
> client speaks. `tests/test_api.py` drives the real client against the real
> routes so the two cannot silently desynchronise again.

> **`POST /schedule` names transmitters; it does not take a raw list.** A
> census row cannot be scheduled: `calc_ionograms.py:444` subscripts `id` and
> `transmit_name` with no default, and a detection is anonymous. The identity
> is supplied once, by an operator, through `/transmitters` — and the schedule
> is composed from those records, one MPI rank group per transmitter.

> **The schedule is not an observation, and `GET /schedule` says both.** Its
> `slots` are the journalled ini read against a clock: `in_progress` stays true
> with the recorder dead. The `running` object beside them is the measurement —
> `running`, `stopped`, `silent` or `unknown` — led by `newest_product_age_s`
> because DOB reports no unit states at all (`dombas.sh` supervises it, not
> systemd), and overridden by a dead `chirp-rx` or `chirp-ionograms` when there
> is one. `silent` is distinct from `stopped` because the 2026-08-05 fault was
> every unit green with nothing being produced; `unknown` is distinct from
> `stopped` because a stale report is the absence of evidence.

One surface across acquisition, extraction and forecasting — but **separate
auth scopes**. Public read of soundings and forecasts must not share a scope
with anything that can stop an acquisition.

### 4.4 prediction **[built 2026-08-23]**

Reads the tracked series, writes forecast rows, scores them against four
baselines. Carries the `N:\muf` models. **`docs/prediction.md` is the
reference**; this section is the design constraint it satisfies.

Must read the *tracked* series rather than raw picks — **and must still see
`limited`/`loflim`**. Otherwise it trains on midday lower bounds, which per
`BACKLOG.md` §3 are biased ~5 MHz low for 82 of 288 soundings. `dataset` passes
a censored pick to the tracker as a gap and returns it flagged, and `scoring`
charges it one-sidedly.

**No longer blocked.** §8's blocker is `N:\muf\config_enum.py` calling
`input()` at import, which stops the *research* project running unattended and
never gated this service: the artifacts are read as files, not imported as a
package. Registering and running one is `docs/prediction.md` §4.

**Models arrive from the console, and the api still cannot run one.** Adding a
model used to mean a shell on the host, because loading a `.sav` unpickles it
and fitting one is worse. Rather than open that surface, it is split. The api
hashes an upload, checks four magic bytes and writes it to a quarantine volume;
`POST /models/train` and `POST /models/run` write rows. Three workers with no
listening socket — `registrar` (10 s), `trainer` (60 s), and `infer` itself,
whose interval is now cut into 10 s slices so a requested pass is served
without waiting six hours for one — are what unpickle, what fit and what
predict. The invariant is mechanical, not documentary:
`tests/test_prediction_upload.py` reads the syntax tree of `services/api/` and
fails if anything there imports `joblib`, and of all of `services/` to confirm
exactly one module calls `.fit`.

Artifacts are addressed by content — `/models/objects/<aa>/<sha256>`, mode
0444 — because `artifacts.sha256` was always right that a path is not an
identity. That is DVC's cache layout on purpose; DVC is not used, and
`docs/prediction.md` §3.5 says why. `GET /models/<id>/artifact` is the pull, at
read scope. The store is read-write in the two workers and `:ro` everywhere
else, `api` included.

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

-- who a receiver has identified, and what the identification rested on.
-- Keyed by RECEIVER: a slot second is a reception second, so the same
-- transmitter heard at two receivers has two different `chirpt` values.
transmitter(
  id, station, code, name,        -- `code` reaches the product's file name
  sounder_id,                     -- chirpsounder2's `id`, unique per station
  timings,                        -- JSON: the sounder_timings entries
  evidence,                       -- JSON: the census row it was read off
  verified_at, verified_by, note
)

-- prediction service output
forecast(
  issued_at, valid_at, horizon_days, model, param, value, quality
)
```

The normalized form of the run CSV. Not a fixed column count: the per-method
block scales with `--methods`, and `interference_rows` and `error` appear only
on the soundings that earned them.

**`reference` is a separate table by design.** IRI is validation only. If
modelled and measured values ever share a table, the prediction service can
train on IRI and produce circular validation. Structural separation, not
convention.

**[proposed]** `fof2` derived from oblique MUF via `geometry.py` belongs in
`reference`, not `extraction` — it is a secant-law inference resting on a
300 km `hmF2` fallback, not a measurement.

### 5.2.1 Pruning: what may be removed, and what makes it stick **[built 2026-08-28]**

Two things accumulate that nothing was able to remove: registry rows from
training sessions, and circuits nobody configured.

**Models.** `retire` was the only verb, and deliberately so -- the forecasts a
model issued are what its scores were computed from, and re-activating it is
how a promotion is rolled back. But an afternoon of experiments leaves a dozen
rows that were never meant to outlive it (17 on the rig by 2026-08-28, 12 of
them from one day), and a registry nobody can prune is a registry nobody
reads. `DELETE /models/{id}` removes the row, its forecasts and its scores.

`score` keys on a `subject` *string* (`model:15`), not a foreign key, so it
cannot cascade off `model_registry` and is deleted explicitly. Leaving those
rows behind would put a leaderboard entry in the database whose model cannot
be looked up.

One refusal: the **active** model. Not because it is unrecoverable -- nothing
here is -- but because "the live forecast" is a role held by exactly one row
per circuit, and dropping it silently would leave the circuit with no forecast
at all. Retire first, and the decision has been made deliberately.

The artifact is reaped only when no other row shares its `sha256`: one `.sav`
serving two circuits is two rows, and deleting one must not pull the file out
from under the other.

**Circuits, and why deleting is not enough.** A circuit in this database is a
`(tx, rx)` that was *ingested*, not one that was configured. `unkown -> DOB`
held 981 soundings and 2,943 extractions on the rig, and `unkown` is chirp v2's
marker for a transmitter it could not identify (`muf/stations.py:UNIDENTIFIED`,
upstream's spelling). So it is not one circuit at all -- it is every
unidentified emitter in the archive sharing one string, on a range axis with no
absolute zero (§3.4).

The trap is that deleting the rows does nothing lasting. `find_new` treats a
file with no row as **new forever**, so the next scan reads all 981 back in.
That is the same trap `DELETE /archives/{id}/orphans` refuses to walk into --
it will not run on an archive that admits every format, because the format is
what keeps the orphans gone.

`muted_circuit` is the equivalent rule for a circuit. `ingest` declines to
write a muted `(tx, rx)`, so the deletion stays done, and
`DELETE /circuits/{tx}/{rx}` **refuses unless the circuit is muted first**.
Unmuting and re-scanning is the undo, and it works because nothing here touches
a file: the mount is read-only.

Two details that are easy to get wrong, and were:

* **Case is folded on both sides.** v2 writes `DOB`, `.lfs` writes
  `yoshkar-ola`, and a rule that missed `Unkown` because the file said `unkown`
  would be a rule that silently does nothing -- the worst behaviour available
  to a filter.
* **`rx = NULL` means *every* receiver, not "rx IS NULL".** The first
  implementation stored the wildcard as NULL in the rule and then matched it
  with `lower(rx) IS NULL` in the delete, so a mute that read as applied
  deleted nothing at all. One spelling, one meaning, in both places --
  `db._circuit_where` is the single point that guarantees it.

The rules are read **once per scan** (`db.muted_matcher`), not per sounding:
per-row it would be tens of thousands of queries for an answer that did not
change during the pass. Read-once also means a rule added mid-scan takes effect
on the next one, which is the behaviour worth having -- a filter that changes
under a running pass ingests half an archive under each ruleset.

### 5.3 Server-internal: REST **[settled]**

`api` ↔ web client. Services do not push payloads to one another; `renderer`
output is fetched, not posted.

**The two model queues are the one place a payload does travel inward**, and
they are shaped so that it still is not pushed anywhere: the api writes a
`model_upload` or `train_job` row and a worker polls for it. Nothing reaches
into the workers, and the workers expose nothing to reach.

| route | scope | what it does |
|---|---|---|
| `POST /models/upload` | control | quarantines raw bytes; never opens them |
| `POST /models/train` | control | vets a spec and queues a fit |
| `POST /models/run` | control | queues a forecast pass over one circuit |
| `GET /models/uploads`, `GET /models/jobs` | read | what the console polls |
| `GET /models/<id>/artifact` | read | the artifact itself, by digest |
| `GET /models/runs` | read | passes asked for, and what they wrote |
| `DELETE /models/uploads/<id>`, `/models/jobs/<id>`, `/models/runs/<id>` | control | forget, cancel |

### 5.4 Station ↔ server: health and control **[proposed]**

Separate from the bulk path (§5.1), with opposite direction of trust.

- **Health: push.** The agent POSTs JSON on an interval. Silence is itself the
  alert — no polling, and it works behind NAT.
- **Control: pull.** The agent polls for pending commands and acknowledges
  them. The station is never a listening service, keeping acquisition hardware
  off any inbound path.

Commands are queued, acknowledged, and recorded with the identity that issued
them. Parameter changes additionally write `config_epoch`.

**The epoch opens on acknowledgement, not on enqueue.** A queued command has
changed nothing; the station may pull it minutes later or never. It opens
whenever the *write* succeeded even if the restart that followed failed —
the file on the station has already changed, so every sounding after that
belongs to the new epoch whether or not the process took it up.

---

## 6. Release plan

Two things drive the order. **A live reception fault on `cyprus1`** comes first,
because an acquisition that is not working makes every downstream milestone
moot. **Multi-station** is the reason for v2 (§2.1) and follows once the fault
is understood.

M0 and M1 are independent and should run in parallel: M0 is investigation, M1
is unattended-operation plumbing that is needed whatever M0 concludes.

> Separate **immediate remediation** from **structural fix**. Whatever M0 finds,
> the fastest repair is almost certainly a change to the running v1 station, not
> a migration. v2 is the answer to *"why could nobody tell what went wrong?"* and
> to multi-station — not to *"reception is down today."*

### M0 — Diagnose the fault **[in progress]**

`tools/diagnose_reception.py` exists and is validated. Run it on the period
that is actually failing:

```bash
python tools/diagnose_reception.py <failing-dir> --stride 4 --expect-range 2710
```

Baseline for comparison, measured over 2026-02-04 → 02-09 (72 soundings):
median peak echo range **2710.0 km**, σ 34.8 km, **0.59 ms peak-to-peak timing
stability**. That archive is healthy — the fault is not in it.

| Verdict | Reading | Immediate remediation |
|---|---|---|
| **TIMING** | Echo on the axis, outside the gate | Re-extract with `--gate`; the recordings are salvageable. Fix clock discipline or the schedule entry |
| **NO SIGNAL** | No coherent trace anywhere | Local chain first (antenna, feed, USRP, RFI), then whether Cyprus is on air or changed its chirp rate. Nothing recoverable from these files |
| **TRUNCATION** | Recordings stop mid-sweep | Acquisition dropouts, `BACKLOG.md` §4. Was ~3% in February |

*Exit:* the fault has a name, and M2's urgency is settled.

### M1 — Stop the bleeding **[parallel with M0]**
- Automatic file transfer, laptop → server, with delete-after-confirm
- Station agent, **health only** (§2.5). Add the diagnostic's two live fault
  indicators to the metric set: peak echo range against the 2710 km baseline,
  and `sweep_fraction`
- Fix `config_enum.py`'s `input()` (§8) — no dependencies, unblocks M6

*Exit:* the disk cannot fill silently, and a recurrence announces itself
instead of being found weeks later.

### M2 — v2 fork and parallel run
The structural fix. Priority set by M0: if the fault was undiagnosable from v1's
output, this moves up; if it was a simple timing or schedule error, it reverts
to being the multi-station enabler and can wait.

- Fork by subtraction (§2.2); `muf/io_chirp.py` (§3)
- **Measure** the normalization offset against v1 output (§2.4)
- Station coordinate registry
- Second USRP is the long-pole procurement item — order at M0 if wanted (§2.4)

*Exit:* v2 MUF agrees with v1 MUF on `cyprus1` within a stated tolerance.

### M2.5 — Docker test rig **[done 2026-08-07]**

Not originally a milestone. It exists because M3, M4 and the control half of
M5 could not be evaluated separately: the station agent was built and tested,
and there was nothing for it to talk to, so the loop that matters — push,
pull, execute, acknowledge — had never run end to end.

Deliberately throwaway, and it takes the shortcuts a temporary thing should:
SQLite rather than Postgres, stdlib SQL rather than an ORM, Jinja rather than
a JavaScript build, one process rather than a queue.

- `services/api/` — the §5.2 schema on SQLite, the three agent endpoints, read
  endpoints, on-demand ionogram rendering, two auth scopes
- `deploy/` — compose file, an api image and a **simulated station** running
  the real agent
- `tests/test_api.py` — including one test that drives the real
  `services.agent.runner` against the real routes

What it settled, which is the point of building it:

- **The endpoint paths in §4.3 were wrong.** The agent posts to
  `/stations/health`, not `/health/report`. Found at integration, corrected in
  the doc, and now covered by a test.
- **`render.plot` could not write to a stream**, so §4.2's "render on request"
  needed a temporary file per request. It takes a file object now.
- **The tri-state metric survives SQL only if you make it.** `health_metric.ok`
  is nullable end to end; a boolean column would turn "could not measure" into
  "failing" somewhere between the station and the screen.

*Exit:* an operator can see station health and queue a restart in a browser,
and 318 of 319 real v2 soundings load into the schema. **Superseded by M3 and
M4, which should not inherit its shortcuts.**

### M3 — Database and extractor
- Schema per §5.2 — **now exercised**; `services/api/schema.sql` is the SQLite
  form and the ingest path from `pipeline` output is written and tested
- Postgres, migrations, and a real queue worker: none of which M2.5 has
- Extractor as a queue worker wrapping the existing `muf` CLI
- Backfill historical `.lfs` extractions

Backfill is no longer urgent — the dissertation dataset is already in hand — so
this is infrastructure for what comes next, not a deliverable in itself.

### M4 — API and web, read-only
- Health views, ionogram browse, MUF/LOF series — **prototyped in M2.5**
- Answers "is it running?" without AnyDesk
- What M2.5 does not have: TLS, sessions, pagination beyond a limit, any
  caching of rendered products

### M5 — Multi-station cutover and control
- Enable opportunistic detection across transmitters
- Control endpoints, auth, `config_epoch` (§2.5). M2.5 routes start/stop/
  restart, and — since the schedule work — `set_config` through
  `POST /stations/{id}/schedule`, which is the validated parameter edit
  `control.py` always had. It is still a rewrite of the station's `.ini` that
  needs a restart to take effect, so the page says so and the epoch it opens
  is dated from the acknowledgement
- Retire v1 **only once the web client covers what operators actually use** —
  watch usage over M4 rather than building for feature parity

### M6 — Prediction service **[built 2026-08-23; training built 2026-08-24, a model worth promoting outstanding]**

The thin service above `N:\muf`, not a home for models: it reads the database,
rebuilds the frame an artifact expects, runs it *without* refitting, and writes
`forecast` rows. See **`docs/prediction.md`**.

- `dataset` resamples with `muf.track` rather than interpolating, so every
  filled point carries a sigma and no point is invented
- `legacy_features` recovers a fitted model's input recipe from its column
  names, and takes the alias as an argument rather than inferring it — renaming
  a column to make a model accept it is how somebody else's ionosphere becomes
  "the forecast"
- `artifacts` records a golden input/output at import and re-checks it on every
  load, which catches a library upgrade that changes behaviour silently
- **Nothing on the inference path calls `fit`.** The code this replaces refits
  on load, so its "predictions" come from a model trained seconds earlier;
  `tests/test_prediction_infer.py` makes `fit` raise to keep it that way, and
  `train.py` is now the single, source-checked exception
- `scoring` runs the model and four baselines through the same code, over the
  same pairs, into the same table. Truth is measured picks, never the tracked
  grid
- Promotion is a schema constraint, not a warning: a model fitted against a
  modelled target, or bound to no circuit, cannot be made active

- **Models arrive from the console**, and the process that answers HTTP still
  cannot open one: it quarantines bytes, and a worker with no listening socket
  unpickles them (§4.4). Artifacts are stored under their own sha-256
- **`train.py` fits on measured picks**, never on the tracked grid it draws
  features from, excludes band-edge bounds from the fit while keeping them in
  the score, and holds out the tail rather than a random sample
- **The console covers the whole lifecycle**: upload or train, promote, and
  issue. Activating a model does not produce a forecast, so *Run now* queues a
  pass and `infer` serves it between the slices of its interval

*Exit:* a registered model runs unattended and its forecast is on the console
beside the measurement, with a leaderboard saying whether it beats persistence.
**Met.**

**Outstanding, and narrower than it was:** a trained model actually worth
promoting. The service now fits its own — `huber-muf-24h` on NIC3 →
Yoshkar-Ola, 474 measured rows, holdout MAE **2.66 MHz against persistence's
2.00** (`docs/prediction.md` §5). Six days of one circuit at a 24 h lead is
thin, and the leaderboard exists to make that visible rather than to be argued
around. What is left is archive, not architecture: the same run against the
accumulated multi-station record, which is now a form submission.

Two things since narrow the gap without closing it. **Committees** —
`voting` and `stacking`, ported from the `muf` project's
`voting_stacking_models` — let a run average decorrelated errors across
`huber`, `ridge` and `xgboost` instead of betting on one; the voting weights
are earned on a chronological inner split rather than on `muf`'s coefficient
mass, for reasons `docs/prediction.md` §3.6.1 sets out. And `/ui/series` now
**shades the window a model was fitted on**, because the flattering way to read
a backtest is to read it over the training data and not notice.

### M7 — Bilingual console **[done 2026-08-23]**

Not originally a milestone. The operators are at Yoshkar-Ola and the station
work is done in Russian, so a console only its authors could read was a
usability fault, not a nicety.

- `services/api/i18n.py` and `services/api/locale/{en,ru}.py` — plain dicts,
  no gettext and no compile step, for the same reason M2.5 has no JavaScript
  build (`BACKLOG.md` sec. 35)
- ~5 200 words across all eight templates, including the third of them that
  lives inside inline `<script>`
- EN/RU in the header, remembered in a cookie; `UI_LANG` sets what a browser
  with no cookie gets. No `Accept-Language`: one URL, one rendering
- English is the default and renders byte-for-byte what it did before

*Exit:* every word a page renders is in the chosen language, and the guards in
`tests/test_i18n.py` fail when a string is added to one catalog and not the
other. **Not covered:** strings authored in Python and rendered into these
pages (`net.detail`, `acquisition.detail`, `band.why_not`, scan results), and
the `HTTPException.detail` texts the JSON API returns.

---

## 7. Repository map

Four repositories, and **M1–M6 do not add any**. A service is not a reason for a
repo; a different *lifecycle* is.

| Repo | Tier | Role | Boundary forced by |
|---|---|---|---|
| `chirpsounder` v1 fork | 1 | Current acquisition; `.lfs` writer, Qt console. Frozen, retired at M5 | Separate upstream (abandoned) |
| `chirpsounder2` clone | 1 | Target acquisition. **Pinned clone, not a fork** (§2.2) — nothing of ours in it | **Tracks live upstream** |
| `ionograms-handler` | 1–3 | `muf` package **and every service** | — |
| `N:\muf` | 2 | Research: BER / channel availability, and the archive's models | Own heavy deps, research lifecycle |

`services/prediction/` gained the pieces that make a model a first-class object
rather than a file somebody staged: `store.py` (content-addressed artifacts),
`queues.py` (the two work queues, and the only module that writes their tables),
`registrar.py` and `trainer.py` (the workers that open what the console sends),
and `train.py` — the one module in the repository allowed to call `fit`.

### The one hard boundary

`chirpsounder2` must contain **nothing but upstream minus deletions** (§2.2).
Add a `services/` directory there and `git pull` conflicts forever — the exact
failure that stranded v1. Everything of ours stays out of it, including
`io_chirp.py`.

### Why the services share one repo

They share a **database schema**. Splitting extractor, api and prediction into
separate repos turns every schema change into a coordinated multi-repo release
with version pinning between them — real cost, for one developer and one
deployment target, buying nothing. The health JSON schema is likewise shared
between the station agent and `api`; one repo means one definition rather than
two that drift.

The `muf` library keeps its identity inside the monorepo: it stays the only
published package, and service dependencies (FastAPI, psycopg, uvicorn) go in
`[project.optional-dependencies]` alongside the existing `parquet` / `iri` /
`cnn` / `db` extras, so installing the library never drags in a web stack.

```
ionograms-handler/
  muf/              extraction library — the published package, unchanged
  services/
    agent/          station health + control      (deploys to the laptop)  M1, M5
    extractor/      queue worker over muf         M3
    api/            REST, the only public surface M4, M5
    renderer/       PNG + SAO.XML on request      M4
    prediction/     thin wrapper over N:\muf      M6
  web/              client                        M4
  db/migrations/    schema, shared by everything  M3
  tools/            diagnose_reception.py, transfer    M0, M1
  docs/  tests/
```

`N:\muf` stays separate and stays a *library*: the prediction **service** is the
thin container above, importing the models. That keeps TensorFlow, Keras,
XGBoost, hyperopt and pmdarima out of the service image, and keeps experiment
code free of service concerns. It also depends on that repo becoming importable
— which is blocker §8.1.

### When to split later

Splitting a monorepo later is mechanical (`git filter-repo`); merging repos
later is not. The asymmetry says start together and split on evidence:

- the web client grows real frontend tooling and its own release cadence
- someone else operates a station and needs only the agent
- a component needs independent deploy cadence or separate access control

None of these hold today.

---

## 8. Blockers and known risks

**Live: `cyprus1` reception is failing.** Cause not yet established — M0. The
February 2026 archive is healthy (§6), so the fault post-dates it or the
failing recordings are held elsewhere. Everything downstream of acquisition is
academic until this is understood, which is why it leads the plan.

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
