# Signal chain: from USRP to gated ionogram

Reference for every parameter between the antenna and the array the estimators
consume, and for the arithmetic that connects them.

Values marked **[code]** are read from this repository. Values marked **[GUI]**
come from the acquisition console on the sounding laptop. Values marked
**[measured]** were read back from a cached product. Anything else is derived,
and the derivation is shown.

§2 and §3–§8 describe the v1 `.lfs` chain. **§2A describes the chirpsounder2
station's USRP and UHD settings**, which have no console and are configured
entirely by flags, kernel tunables and systemd units.

---

## 1. Overview

```
  HF antenna
      │
      ▼
  USRP                        sample_rate = 25 MHz            [GUI]
      │                       hardware LO set so the sweep fits the band
      ▼
  digital downconversion      cf = per-transmitter, kHz       [GUI: "гетеродин"]
      │                       one thread per scheduled station
      ▼
  decimation                  dec = 625                       [GUI]
      │                       fd = sample_rate / dec = 40 kHz
      ▼
  .lfs writer                 512-byte header + complex64 IQ  [code: io_lfs.py]
      │                       80 MB per 250 s sounding
      ▼  ─────────────── laptop / server boundary ───────────────
      │
  muf.spectro.compute()       Hann window, |FFT|², row-median equalization
      │                       window = 8192, zero_periods = 0  [code]
      ▼
  muf.calibrate               range gate applied inside the FFT loop
      │                       1220 × 182 float32 ≈ 0.89 MB     [measured]
      ▼
  estimators → pick → fit → track → SAO / database
```

---

## 2. Acquisition parameters (Qt console)

### 2.1 "Параметры → Основное" — applies to the whole receiver

| Field (RU) | Meaning | Value | Effect |
|---|---|---|---|
| Программа зондирования | Sounding program path | `…/chirpsounder/chirp.py` | chirpsounder v1, GNU Radio |
| Конфигурация | Config module | `…/chirp_config.py` | |
| Каталог для ионограмм | Output directory | `/media/ionouser/DATA3/ionozond_data/` | where `.lfs` lands |
| Частота дискретизации (кГц) | ADC sample rate before decimation | 25 000 | sets sweep window width — see §4.2 |
| Коэффициент децимации | Decimation factor | 625 | `fd = 25 MHz / 625 = 40 kHz` |
| Количество точек БПФ | FFT size, **live display only** | 16 384 | does *not* affect `.lfs`; see note below |
| Фильтрация / полоса фильтра | Filtering, filter bandwidth | on, 8192 | pre-decimation filter |
| Количество отсчётов | Sample count per display block | 30 000 | live display |
| Источник точного времени | Time source | Internet / NTP (`ns1.volgatech.net`) | disabled in the screenshot — see §7.3 |
| Вертикальная шкала | Ionogram y-axis for display | `h, km` | display only |
| Вырезание прямого сигнала | Direct-signal excision | off | |

> **The console's FFT size (16 384) is not this repository's window (8192).**
> The console computes its own spectra for the live display. `muf` re-derives
> everything from the raw IQ in the `.lfs` payload with its own window. The two
> are independent and may differ without either being wrong.

### 2.2 "Параметры → Суточный ход" — diurnal display only

Averaging settings for the console's signal/noise and power-delay-profile
strip charts (72 h period, 5 min time averaging, 1000 kHz frequency averaging).
**No effect on `.lfs` contents.** Recorded here so it isn't mistaken for a
processing parameter.

### 2.3 "Расписание" — per-transmitter, and the thing that governs `.lfs`

One column per transmitter; the checkbox selects which are actively received.

| Field (RU) | Meaning | Unit | `.lfs` header field |
|---|---|---|---|
| Период повторения | Repetition period of the transmitter's sweep | s | `rep` |
| chirptime | Sweep start offset within the period | s | `chirpt` |
| Скорость перестройки ЛЧМ | Chirp rate — how fast the sweep climbs | kHz/s | `rate` (stored Hz/s) |
| Продолжительность сеанса | Session duration — how long we record | s | `dur` |
| Частота программного гетеродина | Digital downconversion centre | kHz | `cf` (stored Hz) |
| Широта (Ю−, С+) | Transmitter latitude | ° | `tx_latitude` |
| Долгота (З−, В+) | Transmitter longitude | ° | `tx_longitude` |
| Максимум для "сигнал/шум" | Display colour-scale ceiling | dB | — display only |
| Максимум для ПЗМ | Power-delay-profile scale ceiling | — | display only |

The receiving station is selected separately ("Принимающая станция") and
supplies `rx_name`, `rx_latitude`, `rx_longitude`.

**Currently operational: `cyprus1` only.** Other columns in the schedule are
configured but not enabled.

---

## 2A. USRP and UHD parameters — the chirpsounder2 station

Everything in §2 describes the v1 Qt console. The v2 station at DOB has no
console: acquisition is `rx_uhd_ext_gps` talking to a USRP N200 over 1 GbE,
and every setting below is either a command-line flag, a kernel tunable, or a
property of the radio. All of them were touched between 2026-08-04 and
2026-08-06, most of them because something was broken.

Marked **[flag]** for a recorder command-line option, **[host]** for a Linux
setting, **[radio]** for hardware state read back from UHD.

### 2A.1 Radio and transport

| Parameter | Value | What it does | Why it is set that way |
|---|---|---|---|
| `--usrp_args=addr0=` **[flag]** | `192.168.10.2` | USRP IP on the dedicated NIC | the N200 is on a private link, not the site LAN |
| `recv_buff_size` **[flag]** | `500000000` | UHD's userspace receive buffer, bytes | 500 MB absorbs scheduler jitter; **useless unless `net.core.rmem_max` is at least as large** — UHD silently gets less and never says so |
| `--rate` **[flag]** | `25e6` | ADC sample rate, S/s | matches v1's 25 MHz so the range scale is common to both formats (§4.2) |
| `--subdev` **[flag]** | `A:A` | daughterboard subdevice | single channel |
| `--channels` **[flag]** | `0` | one thread per channel | one antenna |
| `--outdir` **[flag]** | `/dev/shm/hf25` | Digital RF ringbuffer | tmpfs: 25 MS/s × 4 B = **100 MB/s**, which no spinning disk sustains |
| `--gps-lock-timeout` **[flag]** | `-1` | seconds to wait for GPSDO lock, −1 = forever | a station that starts unlocked records unusable time |
| Ethernet link **[host]** | 1000 Mb/s | | 25 MS/s × 4 B = 800 Mb/s — **80 % of line rate.** There is no headroom, which is why every setting below matters |
| Frame size **[radio]** | 1472 bytes | UDP payload UHD negotiates | **jumbo frames are not available.** The N2x0 probes at 1472 whatever the host MTU says; `ip link` showing `mtu 9000` is irrelevant. ~68,000 packets/second |

> **Do not diagnose the link with `ping`.** The N200 does not answer ICMP at
> all — a failed `ping -M do -s 8972` proves nothing about MTU, and a failed
> plain `ping` proves nothing about the radio being alive. Use
> `uhd_find_devices`, or `tcpdump -i <nic> host 192.168.10.2` to see whether
> it is streaming.

### 2A.2 Host tuning — all three reset on reboot

| Parameter | Value | Symptom when wrong | Where it belongs |
|---|---|---|---|
| `net.core.rmem_max` **[host]** | `500000000` | `got no data in recv 0`, thousands of lines | `/etc/sysctl.d/99-uhd.conf`, and `ExecStartPre=` in the unit |
| NIC RX ring **[host]** | 4096 (default **256**, max 4096) | same, and this was the larger of the two effects | `ethtool -G <nic> rx 4096`; `ExecStartPre=` in the unit |
| `rtprio` limit **[host]** | 99 | `Unable to set the thread priority`, then continuous packet loss | `LimitRTPRIO=99` in the unit |

The `rtprio` one is worth spelling out, because two plausible fixes both
silently did nothing here:

- `/etc/security/limits.d/uhd.conf` is read by **`pam_limits.so`**, which was
  **absent from `/etc/pam.d/common-session`** on this machine. The file was
  correct and inert. Symptom: `ulimit -r` returns 0 while the config says 99.
- `setcap cap_sys_nice+ep` attaches to the **inode**. It survives reboots and
  is cleared by any rebuild of the binary, and it must be applied before the
  process `exec`s — applying it to a running recorder changes nothing.

`LimitRTPRIO=99` in the systemd unit sidesteps both. That is why
`chirp-rx.service` sets it rather than relying on either mechanism.

Verify with `ulimit -r` (as the acquisition user, not root) and
`getcap ~/chirpsounder2/rx_uhd_ext_gps`.

### 2A.3 Timing — clock source, time source, and epoch

Three different things, routinely conflated:

| Concept | Set by | What it controls | Failure signature |
|---|---|---|---|
| **Clock source** | `set_clock_source("gpsdo")` | the 10 MHz reference the ADC samples on | `ref_locked: false` → sample rate and frequency drift |
| **Time source** | `set_time_source("gpsdo")` | which PPS edge the USRP counts | edge jitter |
| **Epoch** | `set_time_next_pps(N + 1)` | *which second number* that edge is called | **nothing** — see below |

The first two are checked and reported at startup. The third was not, and it
is the one that caused every timing problem this station has had.

```
 * mboard 0 gps_locked: true          ← GPS receiver has satellites
 * F5F86F: false                      ← ref_locked; the 10 MHz has not settled
```

A `ref_locked: false` immediately after `gps_locked: true` is usually not a
fault: the FireFly GPSDO needs tens of seconds after satellite lock to
discipline its oscillator, and the recorder checks a few lines later. Confirm
with `uhd_usrp_probe` a minute after start before treating it as real.

**The epoch is the dangerous one.** Stock `rx_uhd_ext_gps` sets it from the
host clock (`rx_uhd_ext_gps.cpp:433`), never from the GPSDO's `gps_time`
sensor. Since this is stretch processing, `range = c·δt`:

```
1 ms  =  300 km          1 s  =  300,000 km
```

and there is no internal evidence of the error — the products stay perfectly
self-consistent. Two observed failures, same line:

| Date | Host clock error | What it produced |
|---|---|---|
| 2026-08-05 | −0.9557 s | every echo 286,000 km out; diagnosed only by comparing against Twente's published cyprus1 schedule, two days later |
| 2026-08-06 | −5.3 years | a run stamped 2021-04-02 after the RTC lost time and NTP had not stepped it |

Three defences, in order of how much they help:

1. **`patches/0001`** — take the epoch from `gps_time` and verify it. Removes
   the failure rather than detecting it.
2. **`chirp-rx.service`: `After=time-sync.target`** — do not start before NTP
   has stepped the clock. Helps only if NTP converges at all.
3. **`services/agent`: `health.system_clock` and `health.epoch_offset`** — the
   first catches an implausible clock from nothing on disk, the second
   measures the residual against a transmitter of known position and published
   schedule. Both were written because neither the products nor the logs said
   anything.

Compare §7.3, which measured v1's timing at DOB as stable to 0.59 ms
peak-to-peak. That was the *stability*, not the *epoch* — v1 was stable and
could have been wrong by any constant amount without the measurement noticing.

### 2A.4 Ringbuffer and storage

| Parameter | Value | Meaning |
|---|---|---|
| `data_dir` | `/dev/shm/hf25` | Digital RF ringbuffer, tmpfs — sized by `/dev/shm`, half of RAM by default |
| `ringbuffer_max_age_min` | 2 | how much raw voltage is retained; ~12 GB at 4 min, so ~6 GB here |
| `ringbuffer_cleanup` | `true` | enables the pruner — which runs **inside `iono_housekeeping.py`**, not inside the recorder |
| `output_dir` | the archive volume | where `lfm_ionogram-*.h5` lands |

The coupling in row three is the trap. If `iono_housekeeping.py` dies, nothing
prunes, and 100 MB/s fills the tmpfs in about a minute per gigabyte. The
recorder then dies with `errno = 28` at sample index 0 —
`dataset_samples_written = 0` — and the log is a page of HDF5 stack trace
whose one useful line is `No space left on device`.

`dombas.sh` clears the ringbuffer once at script start (via
`stop_ringbuffer.sh`), but its 24-hour restart loop only re-execs the
recorder. A session that ended with the pruner dead therefore hands a full
tmpfs to the next one. `chirp-rx.service` reclaims it with `ExecStopPost=`
instead — on stop rather than start, so an in-place restart cannot delete data
the consumers are still reading.

### 2A.5 Shutdown — the setting that is a site visit

`KillSignal=SIGINT`, `TimeoutStopSec=30`.

UHD sends the stop-streaming command to the radio on `SIGINT` and on nothing
else. A USRP killed mid-stream keeps blasting UDP at a host that is gone: it
stops answering discovery, `uhd_find_devices` reports `No UHD Devices Found`,
and `tcpdump -i <nic> host 192.168.10.2` shows it still transmitting.

```bash
pkill -INT -f rx_uhd     # correct
pkill -f rx_uhd          # SIGTERM; wedges the radio
```

Observed 2026-08-05, and the reason `chirp-rx.service` sets `KillSignal`
explicitly rather than accepting systemd's `SIGTERM` default.

**If it is already wedged, try the software reset before a site visit.** The
clone ships one:

```bash
./reset_usrp.sh
# uhd_image_loader --args "type=usrp2,addr=192.168.10.2,reset" --no-fpga
```

`--no-fpga` means it sends the reset command without reflashing anything. It
is not guaranteed — a device that has stopped answering discovery may not
answer this either — but it is the only remote option and it costs nothing to
attempt. Power removal is the fallback, not the first move.

### 2A.6 Launching

`dombas.sh` must run **as the acquisition user, not under `sudo`**, from the
repository root:

```bash
cd ~/chirpsounder2 && source .venv38/bin/activate && ./examples/marieluise/dombas.sh
```

`sudo` runs the script through `dash`, which has no `source`, so the venv
never activates and every `python3` in it becomes the system interpreter —
Python 3.5 on this box, where chirpsounder2 needs 3.8. The consumers then die
on import while the recorder keeps running, which is exactly the state that
fills the ringbuffer. It also leaves root-owned files in `/dev/shm` and the
output directory that block the next non-root run.

Nothing in the chain needs root: `rtprio` comes from `limits.d` (or the unit),
and `ethtool`/`sysctl` are `ExecStartPre=+` lines that systemd runs privileged
on their own.

---

## 3. The `.lfs` file

512-byte header followed by interleaved `complex64` IQ at `fd`.
**[code: `muf/io_lfs.py`]**

### 3.1 Header layout

Little-endian; offsets in bytes from the start of file. `s` = fixed-width
NUL-padded ASCII.

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `format` | 4s | Format tag |
| 4 | `format_ver` | f | Format version |
| 8 | `header_id` | 4s | Header identifier |
| 12 | `header_size` | H | Declared header size (observed: 498, padded to 512) |
| 14 | `tx_name` | 64s | Transmitter name |
| 78 | `tx_latitude` | f | Transmitter latitude, ° |
| 82 | `tx_longitude` | f | Transmitter longitude, ° |
| 86 | `rx_name` | 64s | Receiver name |
| 150 | `rx_latitude` | f | Receiver latitude, ° |
| 154 | `rx_longitude` | f | Receiver longitude, ° |
| 158 | `start_year` | H | Sounding start, UTC |
| 160 | `start_daynumber` | H | Day of year |
| 162–170 | `start_month/day/hour/minute/second` | H | |
| 172 | `start_epoch` | I | Unix seconds |
| 176 | `chirpt` | I | Sweep start offset within the repetition period, s |
| 180 | `cf` | I | Downconversion centre frequency, Hz |
| 184 | `dur` | H | Sweep duration recorded, s |
| 186 | `rate` | I | Chirp rate, Hz/s |
| 190 | `rep` | I | Transmitter repetition period, s |
| 194 | `rmin` | i | Intended range gate, low edge, km |
| 198 | `rmax` | i | Intended range gate, high edge, km |
| 202 | `dec` | I | Decimation factor |
| 206 | `sample_rate` | I | Sample rate **before** decimation, Hz |
| 210 | `whiten` | H | Whitening enabled |
| 212 | `whiten_len` | I | Whitening filter length |
| 216 | `whiten_n` | I | Whitening block count |

> **`rx_longitude` is at offset 154, not 150.** The inherited `lfs_header.py`
> read it at 150 — `rx_latitude`'s offset — so `header['rx_longitude']` in any
> pre-`muf` code holds the *latitude*. `rx_name` occupies 86–149 and
> `rx_latitude` 150–153, which puts the longitude at 154. Anything derived from
> the old reader's receiver longitude is wrong.

### 3.2 Payload and file size

```
payload_bytes = dur × fd × 8          (complex64 = 8 bytes)
file_bytes    = 512 + payload_bytes
```

`cyprus1`: `250 s × 40 000 Hz × 8 = 80 000 000 B` = **80.0 MB**, matching the
measured figure in `BACKLOG.md` §1.

---

## 4. Derived quantities

All formulas below are from **[code: `muf/calibrate.py`, `muf/io_lfs.py`]**.
Note `C_KM_S = 3e8 / 1e3 = 300 000 km/s` — the code uses exactly `3e8`, not
299 792 458.

### 4.1 Decimated rate

```
fd = sample_rate / dec = 25 000 000 / 625 = 40 000 Hz
```

Confirmed by the console log: `fd = 40000.000000`, `dtime = 0.000025` (= 1/fd).

### 4.2 Sweep bounds — `calibrate.sweep_bounds()`

```
freq_start = (cf − sample_rate/2) / 1e6
freq_stop  = freq_start + (dur × rate) / 1e6
```

`cyprus1`: `(20 000 000 − 12 500 000)/1e6 = 7.5 MHz`,
`7.5 + (250 × 100 000)/1e6 = 32.5 MHz` → **7.5–32.5 MHz**.

The **pre-decimation** `sample_rate` sets the start frequency; the
**post-decimation** `fd` sets the range axis. Both come from the same header
field and are easy to confuse.

### 4.3 Round-trip divisor — `LfsHeader.div_coef`

| Path type | Condition | `div_coef` |
|---|---|---|
| Oblique | `tx_name != rx_name` | 2.0 |
| Vertical | `tx_name == rx_name` | 4.0 |

A vertical sounding's echo travels up and back over the same path, so the delay
maps to a different range scale.

### 4.4 Range axis — `calibrate.range_half_span()`

```
half_span = C_KM_S × (fd / div_coef) / rate
```

`cyprus1`: `300 000 × (40 000/2) / 100 000` = **±60 000 km**.

### 4.5 Range resolution — `calibrate.range_resolution_km()`

```
range_step = 2 × half_span / window
```

`cyprus1` at `window = 8192`: `120 000 / 8192` = **14.65 km per bin**.

Zero-padding (`zero_periods > 0`) subdivides bins without resolving anything
further; true resolution stays `range_step × (1 + zero_periods)`.

### 4.6 Frequency axis — `calibrate.sweep_step_mhz()`

```
freq_step = (window / fd) × rate / 1e6
n_freq    = floor(dur × fd / window)
```

`cyprus1`: one window spans `8192/40 000 = 0.2048 s`, during which the
transmitter climbs `0.2048 × 100 kHz/s` = **20.48 kHz**.
`n_freq = floor(250 × 40 000 / 8192) = floor(1220.7)` = **1220 rows**
**[measured: 1220]**.

Deriving the frequency axis from the *window*, not from the nominal endpoint,
is what keeps truncated recordings correct — see `sweep_complete` /
`sweep_fraction`, and `BACKLOG.md` §4.

---

## 5. Spectrogram and gate

### 5.1 Noise equalization — `muf/spectro.py`

Each spectrogram row is divided by `NOISE_COEF × median(row)`, where
`NOISE_COEF = 4·ln2`. This converts the median of an exponentially-distributed
power spectrum into its mean, putting the noise floor at ~1.0 in linear power.

The median is taken over the **full** spectrum *before* gating, so gating does
not bias it.

**Both formats land in this convention, by construction.** v2 stores
`SNR = (P − median)/median` per row, so `SNR + 1 = P/median`, and `io_chirp`
divides by the same `NOISE_COEF`. A median-noise cell reads 25.571 dB either
way. That is what lets one 43 dB threshold mean one thing across `.lfs` and
`.h5` — see `docs/architecture.md` §3.3.

#### Whitening — where it happens, and what it does to this assumption

Two different whitenings, in two different places, and only one touches the
data this pipeline reads.

**v1 (`.lfs`): the console whitens, and it is in the stored IQ.**
`whiten`, `whiten_len`, `whiten_n` (header offsets 210/212/216) record it.
`muf` reads the fields and never applies or reverses them — the samples in the
payload already are or are not whitened, and you cannot whiten twice.

Measured on the archive, as mean/median of `|FFT|²` over 8192-sample windows.
An exponentially distributed power spectrum — ideal Rayleigh noise — gives
`1/ln2 = 1.4427`:

| file | `whiten` | mean/median |
|---|---|---|
| `cyprus1_20191023_071510` | 0 | 2.0887 |
| `cyprus1_20191023_090010` | 0 | 2.1147 |
| `cyprus1_20260204_000010` | 1 | **1.4439** |
| `cyprus1_20260204_100010` | 1 | **1.4497** |
| `cyprus1_20260209_082510` | 1 | **1.4441** |
| `cyprus1_20260209_164510` | 1 | **1.4461** |

**Whitening makes the noise-floor assumption hold, rather than breaking it.**
The whitened files land within 0.5 % of the exponential value; the unwhitened
2019 ones sit 45 % above it, because coloured structure — carriers, band-edge
shaping — inflates the mean without moving the median. The console's filter
removes the deterministic colouring and leaves the per-bin Rayleigh
fluctuation, which is exactly what a median-based noise floor wants.

The consequence is for the **2019 files, not the 2026 ones**: on `whiten=0`
recordings the noise estimate is biased and dB values from them are not
directly comparable with the rest of the archive. The whole 2026 archive is
`whiten=1` and internally consistent.

**v2 (`.h5`): the ionogram path does not whiten.**
`calc_ionograms.spectrogram` is Hann-windowed `|FFT|²`, 13× oversampled, with
the sub-steps combined by inverse-variance weighting — `std_est` is the MAD of
each sub-step's power spectrum and each contributes `1/std_est²`. That weights
*time* substeps by their noise level; it does not flatten across frequency.

There **is** a `Z/|Z|` spectral whitening in chirpsounder2, at
`chirp_det.py:213`, but it lives in `ChirpDetector.seek()` — the detection
path. `calc_ionograms.py` imports `chirp_det` only for `unix2dirname` and
`unix2datestr`. So it never reaches `lfm_ionogram-*.h5`; it reaches the `snr`
column of `chirp-*.h5` and `cdetections-*.h5`, which is why that statistic
orders detections and is not comparable to the 43 dB threshold.

Both formats export the flag in SAO.XML's `<Acquisition>` so two records can
be compared on it.

> **Aside on `NOISE_COEF`.** The measurement above puts the median-to-mean
> factor for this noise at `1/ln2 = 1.443`, while `NOISE_COEF = 4·ln2 = 2.773`
> — so the constant is 1.92× larger than the rationale inherited from
> `stuffr.medians()` states. It is *not* a bug to fix: it is a fixed scale
> factor applied identically on both format paths, and the 43 dB threshold and
> the 25.571 dB noise reading are both calibrated against it. Changing it
> would move every threshold in the package. Recorded here so the docstring's
> stated derivation is not mistaken for an audited one.

### 5.2 The dB scale — the part that does not travel

`to_db()` divides by `1e-3` before the log, so the equalized noise floor lands
at **30 dB, not 0** (`NOISE_FLOOR_DB = 30.0`).

Therefore **the shared 43 dB detection threshold is a 13 dB signal-to-noise
ratio**, equivalent to the historical linear threshold of 20.

> This is the single most portability-sensitive number in the pipeline. Any
> data source with a different noise normalization — another sounder, or
> chirpsounder2's `SNR` array referenced to its own `noise_floor` — needs the
> threshold recalibrated before results are comparable. Anything supplying a
> *true* SNR must add `NOISE_FLOOR_DB` before comparing against `Ionogram.db`.

### 5.3 The gate — `calibrate.default_gate()`

```
hi = rmax  if rmax > 0  else 5000.0
lo = ground_range_km × GROUND_RANGE_MARGIN        (oblique, geometry usable)
   = max(rmin, FALLBACK_MIN_RANGE_KM)             (otherwise)
lo = max(lo, rmin)
```

with `GROUND_RANGE_MARGIN = 0.90` and `FALLBACK_MIN_RANGE_KM = 1500.0`.

`cyprus1` → `yoshkar-ola`, great-circle 2587.8 km:

```
lo = 2587.8 × 0.90 = 2329.04 km      [measured: 2329.0436]
hi = 5000.0 km                        [measured: 5000.0]
```

**The transmitter position is the registry's, not the file's.** `cyprus1` in
this header and `NIC` in `stations.py` are one Nicosia site recorded 59.9 km
apart, and the loader now resolves `.lfs` headers against the table for the
same reason v2 products go through it — one site cannot have two positions
depending on which receiver logged the sweep. Read verbatim the header gives
35.00N 34.00E, 2588.4 km and `lo = 2329.5877`; the gate it produces is
identical, 3755–3936, because 0.54 km is a thirtieth of a bin.

The range axis **descends** — bin 0 is the largest virtual range. Established
empirically, not assumed; see the `muf/calibrate.py` module docstring for the
evidence and for the sign error it corrects in `MUF.py`.

Gate width in bins:

```
i_lo = ceil((60 000 − 5000) / 14.65)     = 3755
i_hi = floor((60 000 − 2329.04) / 14.65) = 3936
bins = i_hi − i_lo + 1                    = 182
```

**[measured: `power.shape == (1220, 182)`]** — a **45× reduction** from 8192,
discarding 97.8% of the range axis before any estimator runs.

### 5.4 Product size and cache naming

```
1220 × 182 × 4 bytes (float32) = 888 KB per sounding
```

Cache files are named `{stem}_w{window}_z{zero_periods}_g{gate}.npz`, e.g.
`cyprus1_20260204_000010_w8192_z0_gauto.npz` — window 8192, no zero-padding,
automatic gate. Arrays stored: `power`, `source`, `n_freq`, `window`,
`zero_periods`, `gate_lo`, `gate_hi`.

---

## 6. Data volume

Per transmitter, from the schedule parameters:

```
MB per sounding   = dur × 40 000 × 8 / 1e6
soundings per day = 86 400 / rep
GB per day        = MB_per_sounding × soundings_per_day / 1000
```

| Station | `rep` | `rate` | `dur` | Sweep | MB/sounding | /day | GB/day | TB/year |
|---|---|---|---|---|---|---|---|---|
| **cyprus1** *(active)* | 300 | 100 | 250 | 25.0 MHz | 80.0 | 288 | **23.0** | **8.4** |
| cyprus2 | 300 | 100 | 250 | 25.0 MHz | 80.0 | 288 | 23.0 | 8.4 |
| longreach | 180 | 125 | 250 | 31.3 MHz | 80.0 | 480 | 38.4 | 14.0 |
| norilsk | 30 | 500 | 50 | 25.0 MHz | 16.0 | 2880 | 46.1 | 16.8 |
| pr | 720 | 100 | 250 | 25.0 MHz | 80.0 | 120 | 9.6 | 3.5 |
| yoshkar-ola | 60 | 50 | 260 | 13.0 MHz | 83.2 | 1440 | 119.8 | 43.7 |

Only `cyprus1` is operational; the rest are shown to make the parameter→volume
relationship explicit for planning.

**Gated products are ~90× smaller than the IQ that produced them**
(888 KB vs 80 MB): 288 × 888 KB = **256 MB/day**, **93 GB/year** for `cyprus1`.

`cyprus1` alone at raw IQ fills a 16 TB store in ~1.9 years. The acquisition
laptop's own disk (931 GB, 725 GB free as of the console screenshot) holds
**~31 days**, which is the operational constraint on the laptop→server
transfer interval.

---

## 7. Discrepancies and open points

### 7.1 `README.md` states 205 gated bins; the value is 182

The README's "How it works" section says *"205 bins instead of 8,192, a 40x
reduction"*. Both the arithmetic in §5.3 and the cached product give **182
bins, a 45× reduction**. The 97.5% figure quoted alongside it should be 97.8%.
Likely stale from an earlier gate margin or `rmax`. Worth reconciling, since
the number appears in the project's headline description of the gate.

### 7.2 `range_resolution_km()` docstring is correct only for oblique paths

The docstring gives `C_KM_S × (sr/window) / rate`, but the code returns
`2 × range_half_span / window`, which carries `div_coef`. These agree when
`div_coef = 2` (oblique) and differ by 2× when `div_coef = 4` (vertical). The
code is right; the docstring should carry the divisor.

### 7.3 Time source disabled in the console — measured, and not a problem

"Источник точного времени" and the NTP server field are greyed out in the
screenshot, and the console log shows `Lag: -0.2 Sec: 1571378709.77771 Usrp sec
1571378710.00000` — a 0.2 s discrepancy between system and USRP time.

Timing matters here more than it looks. Because this is stretch processing, a
timing error maps straight to apparent range:

```
range offset = c × δt          1 ms  =  300 km
```

so 0.2 s would be a 60,000 km offset — the entire half-axis — and roughly 9 ms
is enough to walk the echo out of the 2670 km gate entirely. The symptom would
be indistinguishable from "no signal".

**Measured, it is fine.** `tools/diagnose_reception.py` over 2026-02-04 →
2026-02-09 (72 soundings, every 24th):

| | |
|---|---|
| in-gate soundings | 70 of 72 |
| median peak echo range | 2710.0 km |
| standard deviation | 34.8 km (2.4 range bins) |
| peak-to-peak spread | 175.7 km |
| **implied timing stability** | **0.59 ms peak-to-peak** |

A 35 km standard deviation is around two range bins, which is the ionosphere's
own diurnal change in reflection height rather than clock error. Acquisition
timing over that period was stable to well under a millisecond.

Note also that the console screenshot's timestamp, 1571378709, is **October
2019**. It documents the interface, not the current system's clock behaviour.

### 7.4 `header_size` reads 498, parser assumes 512

The console log reports `fheader.header_size: 498` while `size of fheader` is
512 and `io_lfs.HEADER_SIZE = 512`. The declared field is smaller than the
padded block. The parser's fixed 512 offset to the payload is correct for
observed files, but a file declaring something other than 498 would go
unnoticed — the declared value is currently never checked.

### 7.5 `dur` exceeds `rep` for several scheduled transmitters

`longreach` (250 / 180), `norilsk` (50 / 30) and `yoshkar-ola` (260 / 60) all
record longer than the transmitter's repetition period. Under the reading in
§2.3 that should be impossible on a single receive chain. Either `dur` is a
recording cap rather than the sweep length, or overlapping sweeps are handled
in a way not evident from the schedule alone. This needs confirming against the
acquisition source before any of those stations is enabled — it also bears on
whether `BACKLOG.md` §3's proposed `dur = 305` for `cyprus1` is achievable at
`rep = 300`.

---

## 8. Worked example — `cyprus1_20260204_000010.lfs`

| Quantity | Value | Source |
|---|---|---|
| Path | cyprus1 → yoshkar-ola, oblique | header |
| Transmitter | 35.18557N 33.38228E | registry (`NIC`; header says 35.00/34.00) |
| Receiver | 56.38N 47.53E | registry (`yoshkar-ola`, taken from this header) |
| Great-circle distance | 2587.8 km | `ground_range_km()` |
| `sample_rate` / `dec` / `fd` | 25 MHz / 625 / 40 kHz | header |
| `cf` | 20 MHz | header |
| `rate` | 100 kHz/s | header |
| `dur` | 250 s | header |
| Sweep | 7.5 – 32.5 MHz | §4.2 |
| File size | 80.0 MB | §3.2 |
| `div_coef` | 2.0 (oblique) | §4.3 |
| Range half-span | ±60 000 km | §4.4 |
| `window` / `zero_periods` | 8192 / 0 | defaults |
| Range step | 14.65 km | §4.5 |
| Frequency step | 20.48 kHz | §4.6 |
| Ungated shape | 1220 × 8192 | §4.6 |
| Gate | 2329.04 – 5000.0 km | §5.3 |
| Gated shape | **1220 × 182** | measured |
| Product size | 888 KB (float32) | §5.4 |
| Noise floor | 30 dB | §5.2 |
| Detection threshold | 43 dB = 13 dB SNR | §5.2 |
