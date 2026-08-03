# Signal chain: from USRP to gated ionogram

Reference for every parameter between the antenna and the array the estimators
consume, and for the arithmetic that connects them.

Values marked **[code]** are read from this repository. Values marked **[GUI]**
come from the acquisition console on the sounding laptop. Values marked
**[measured]** were read back from a cached product. Anything else is derived,
and the derivation is shown.

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

`cyprus1` → `yoshkar-ola`, great-circle 2588.4 km:

```
lo = 2588.4 × 0.90 = 2329.59 km      [measured: 2329.5877]
hi = 5000.0 km                        [measured: 5000.0]
```

The range axis **descends** — bin 0 is the largest virtual range. Established
empirically, not assumed; see the `muf/calibrate.py` module docstring for the
evidence and for the sign error it corrects in `MUF.py`.

Gate width in bins:

```
i_lo = ceil((60 000 − 5000) / 14.65)     = 3755
i_hi = floor((60 000 − 2329.59) / 14.65) = 3936
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

### 7.3 Time source disabled in the console

"Источник точного времени" and the NTP server field are greyed out in the
screenshot. The console log shows `Lag: -0.2 Sec: 1571378709.77771 Usrp sec
1571378710.00000` — a 0.2 s discrepancy between system and USRP time. At
100 kHz/s a 0.2 s timing error is a **20 kHz frequency offset**, or one
frequency bin. Worth confirming what disciplines the clock, since range
calibration depends on sweep timing.

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
| Transmitter | 35.00N 34.00E | header |
| Receiver | 56.38N 47.53E | header |
| Great-circle distance | 2588.4 km | `ground_range_km()` |
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
| Gate | 2329.59 – 5000.0 km | §5.3 |
| Gated shape | **1220 × 182** | measured |
| Product size | 888 KB (float32) | §5.4 |
| Noise floor | 30 dB | §5.2 |
| Detection threshold | 43 dB = 13 dB SNR | §5.2 |
