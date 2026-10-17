# Echo window — format v1

The dechirped voltage of a chirp sounding, kept only around its echoes:
shifted to the centre of their beat band, low-passed and decimated. About
1.6 MB a sounding instead of 80, and lossless within the band.

This document is the **specification**: the single source of truth for the
format. `muf/echo_window.py` is the reference implementation. ionozond's
`h5_archive.py` will implement the same format from this page for `.lfs`, so
everything a second implementation needs is written here, not left in the
code.

## 1. Why

A power ionogram is `|FFT|²`. It records where the echoes are and how strong
they are. It discards everything that makes up an HF channel on the fast
scale: each mode's Doppler shift and spread, its fading, the phase that
carries group-delay dispersion, and o/x splitting. A channel simulator built
from oblique soundings needs exactly that part. Until now the only place it
survived was the raw capture: 80 MB a sounding, or 23 GB per station-day.

Almost none of those 80 MB is echo. After dechirping, an echo at virtual
range `r` is a tone at a beat frequency proportional to `r`. Every mode on a
circuit sits within a few hundred km of range, so the echoes occupy a beat
band of a few hundred Hz out of the 40 kHz stored. Shifting that band to zero
and decimating keeps all of it.

The power product still answers MUF and everything scaled from the
ionogram. The window adds the part the power product cannot hold. It is
**additive**: a group named `echo_window` at the file root. v2 writes every
dataset at the root, so a group of our own cannot collide with anything
upstream adds. No existing dataset changes, and `io_chirp.REQUIRED` is
untouched.

## 2. Where it lives

| Source | Window goes | The source |
|---|---|---|
| v2 `lfm_ionogram-*.h5` with `z` | the group is appended to the product | kept; `--drop-raw` rewrites the product without `z`, after verifying |
| `.lfs` capture | standalone `<stem>.echo.h5`, or with `--into` appended to a chirpsounder2-format product of the same capture | **never modified or deleted** |

A standalone `<stem>.echo.h5` is the `echo_window` group plus root
attributes that describe the capture without the `.lfs`:

| Root attribute | Type | Meaning |
|---|---|---|
| `source_format` | str | `"lfs"` |
| `source_name` | str | the capture's file name |
| `rate_hz_s` | float | chirp rate, Hz/s |
| `sample_rate_hz` | float | ADC rate before decimation, Hz |
| `dec` | int | decimation to the stored rate |
| `sr_hz` | float | stored rate of the capture, `sample_rate_hz / dec` |
| `t0_unix_s` | float | the header's `start_epoch`: the time of sample 0 |
| `sweep_start_hz` | float | `cf − sample_rate/2`: sweep frequency at sample 0 |
| `tx_name`, `rx_name` | str | as in the header |
| `tx_latitude` … `rx_longitude` | float | as in the header; omitted if any is unusable |

`muf.loader` reads `*.echo.h5` by name. It deliberately leaves it out of
`loader.FORMATS`. That tuple drives discovery, the archive scan and
`--input-format`, and a window is a derivative of a sounding already counted
under `lfs`, not a sounding of its own.

## 3. The group, v1

### 3.1 Datasets

| Path | Type | Present | Meaning |
|---|---|---|---|
| `echo_window/z` | complex64 `[n]` | always | the window samples |
| `echo_window/taps` | float64 `[N]` | always | the filter `h` the samples were cut with; §3.6 |
| `echo_window/row_noise` | float64 `[rows]`, attr `window` (int) | `.lfs` only | full-band noise floor per spectrogram row; §3.5 |
| `echo_window/lfs_header` | uint8 `[512]` | `.lfs` only | the capture's header, byte for byte |
| `echo_window/params` | — | **reserved** | derived fast-channel parameters, a later task; nothing writes it |

`z` is chunked in equal chunks of at most 65,536 samples. Equal chunks matter
because HDF5 allocates a partial last chunk in full: with fixed 64k chunks
that wasted 0.34 MB in every cyprus1 window. `z` is stored gzip-4 with byte
shuffle; §6.3 has the measurement behind that choice.

### 3.2 Attributes

| Attribute | Type | Definition |
|---|---|---|
| `format` | str | `"echo_window"` |
| `version` | int | `1` |
| `source` | str | `"v2_z"` or `"lfs"` |
| `source_name` | str | the file the samples were cut from |
| `source_samples` | int | `n_src`, length of the source voltage |
| `source_sr_hz` | float | `sr_src`, its sample rate |
| `sr_hz` | float | `sr_src / decimation` |
| `decimation` | int | `D` |
| `shift_hz` | float | the frequency, *on disk*, moved to zero; §3.3 |
| `range_lo_km`, `range_hi_km` | float | the range window kept, on the axis `loader.load` gives the source |
| `range_axis` | str | `"absolute"` or `"relative"` (a search-mode v2 product whose zero is unknown) |
| `t0_unix_s` | float | the source's `t0`: v2's `t0`, or the `.lfs` `start_epoch` |
| `t_first_s` | float | time of stored sample 0 after `t0`, s, filter delay compensated. Negative; §3.3 |
| `sweep_hz_at_t_first` | float | sweep frequency at that time, Hz |
| `rate_hz_s` | float | chirp rate |
| `km_per_hz` | float | range per Hz of on-disk beat frequency; §3.4 |
| `range_zero_km` | float | range at zero on-disk beat frequency; §3.4 |
| `source_window`, `source_step` | int | the source's own spectrogram window and step: v2's `fftlen` and `fft_step`; for `.lfs`, the `row_noise` window, with step equal to it |
| `filter` | str, JSON | §3.6 |
| `selection` | str, JSON | §3.7 |
| `edge_energy_db` | float `[2]` | brightest power-ionogram cell within one margin inside the low and the high edge, dB on the shared scale |
| `truncated_suspect` | bool | either edge value ≥ 43 dB, the detection threshold |
| `writer` | str | `"ionograms-handler <git sha>"` |

A reader checks `format` and `version`, and refuses a version it does not
know. Within v1 a reader ignores attributes it does not recognise. Any change
to the meaning of an existing attribute requires a new version.

### 3.3 Samples: what `z[k]` is

Let `y[m] = z_src[m] · exp(−j2π · shift_hz · m / sr_src)` for
`0 ≤ m < n_src`, and `y = 0` outside the record. `z_src` is the voltage
**exactly as stored on disk**: v2's `z` before any conjugation, and the
`.lfs` samples. Let `h` be the filter of §3.6, `N` taps with `N` odd. With
`L = −t_first_s · sr_hz`, an integer:

```
z[k] = Σ_j h[j] · y[(k − L)·D + (N−1)/2 − j]
```

So stored sample `k` is source sample `(k − L)·D`, filtered with zero phase,
and its time and sweep frequency are exact:

```
t(k)     = t_first_s + k / sr_hz                        (s after t0)
sweep(k) = sweep_hz_at_t_first + rate_hz_s · k / sr_hz  (Hz)
```

**Why `t_first_s` is negative.** The filter responds to the first source
sample half its length early and to the last one half its length late. The
window keeps both tails, so `L = ⌈((N−1)/2) / D⌉` samples precede source
sample 0 (40 on cyprus1, which is −45 ms) and as many follow the last. Drop
the tails and the round trip is still exact in the interior, but the first
and last 45 ms reconstruct about 1e-2 wrong. With them it is exact over the
whole record (§6.1).

`sweep(k)` is the frequency of the **dechirp reference** at that sample's
time. This is the convention the ionogram's frequency axis already uses. An
echo with group delay τ arriving then was transmitted `rate·τ` lower, about
1 kHz for a 10 ms path at 100 kHz/s.

For v2, `t0` is the extrapolated 0 Hz crossing, so `sweep_hz_at_t_first` is
`rate · t_first_s`, a few kHz below zero. For `.lfs` it is
`cf − sample_rate/2 + rate · t_first_s`. The `.lfs` header records no
acquisition delay, so `start_epoch` is taken as the time of sample 0, the
same assumption `spectro` makes.

### 3.4 Range

A component at on-disk frequency `f` sits at

```
r = range_zero_km + km_per_hz · f          (window frame: f = f_win + shift_hz)
```

| Source | `km_per_hz` | `range_zero_km` |
|---|---|---|
| v2 | `−c / rate / 1e3`, `c = 299 792 458 m/s` | the direct-path offset `t0` implies, `frac(t0)·c`; 0 when `range_axis` is relative |
| `.lfs` oblique | `−2c / (div · rate)`, `c = 3e5 km/s`, `div = 2` | 0 |
| `.lfs` vertical | same, `div = 4` | 0 |

**Both slopes are negative, for different reasons. This is the sign trap.**
v2 forms its spectrogram on `conj(z)` (`calc_ionograms.py`), so an echo that
*reads* at a positive relative range is stored at a negative beat. `.lfs`
uses the samples as they are, on a descending axis that maps a negative beat
to positive range. A writer that gets either one wrong keeps an empty band
and then decimates it perfectly. `shift_hz = (f(range_lo) + f(range_hi)) / 2`
in these terms.

The v2 zero is computed the way `io_chirp.reprocess` computes it. That is
also how `io_chirp.load` computes it, except for products v2 wrote with
`range_offset_applied = true` and a window-relative axis. That case comes
from upstream's 0.75 heuristic, and none of those products has `z`.

### 3.5 Noise normalisation

The 43 dB threshold is measured against the **full-band** row median. A
narrow window cannot recompute that median from itself, since most of the
band is gone, so the median travels with the window:

- **v2** already stores it: `noise_floor[i]`, the median of v2's own
  spectrogram row `i` at `source_window`. A row re-derived from the window is
  `(P − noise_floor)/noise_floor` and then `snr_to_power`, as for the stored
  `SNR`.
- **`.lfs`**: `row_noise[i] = 4 ln 2 · median |FFT(hann(W) · x_i)|²`. Here
  `x_i` is the `i`-th non-overlapping `W`-sample row, `W =
  row_noise.attrs["window"]` (8192), and the median is taken over the whole
  spectrum before any gate. That is exactly the floor `spectro.compute`
  divides by.

At a window `W'` other than the stored one, the floor is scaled by
`Σ hann(W')² / Σ hann(W)²`, which is exact for white noise. For `.lfs` it is
also interpolated between row centres.

### 3.6 Filter

A linear-phase lowpass FIR with a Kaiser window, odd length, and unit DC gain
(`Σ h = 1`). Unit DC gain means noise power per Hz in the passband is
preserved: the window's noise has the same density as the source's, and a
tone keeps its amplitude. The `filter` JSON:

| Key | Meaning |
|---|---|
| `numtaps`, `kaiser_beta`, `cutoff_hz` | how the taps were designed, at `source_sr_hz` |
| `passband_edge_hz` | half-width kept at full gain |
| `stopband_edge_hz` | where ≥ 80 dB attenuation begins |
| `stopband_attenuation_db`, `passband_ripple_db` | **measured** on the taps, not assumed |
| `gain`, `type`, `doppler_margin_hz` | descriptive |

**The taps themselves are stored** in `echo_window/taps`, and a reader uses
those, never a filter it designs again. They cost 29 KB at 3595 taps, under
2 % of a window. Storing them means a reader needs no FIR designer that
agrees with `scipy.signal.firwin` to the last bit. It also stops the JSON's
rounded `kaiser_beta` (6 decimals) from drifting away from the filter
actually applied. Decided 2026-10-16.

A writer designs them as `scipy.signal.firwin(numtaps, cutoff_hz,
window=("kaiser", kaiser_beta), fs=source_sr_hz, scale=True)`. Written out:
`h[j] = w_β[j] · sinc(2·cutoff/fs · (j − (N−1)/2))`, then `h /= Σ h`, where
`w_β` is the Kaiser window and `sinc(x) = sin(πx)/(πx)`. A v1 reader refuses
a group without `taps`, or with a length other than `numtaps`.

**How the band and rate are chosen:**

- `passband = |f(range_hi) − f(range_lo)| / 2 + 20 Hz`. The 20 Hz Doppler
  margin keeps a shifted mode in the band.
- Each transition is `max(10 Hz, 10 % of the kept band)`. The stopband
  starts at `passband + transition`.
- `D = ⌊sr_src / (1.25 · 2 · stopband)⌋`. If that is 1, the window is most
  of the band, nothing is decimated, and `numtaps = 1`.
- Taps come from `kaiserord` at 83 dB. The response is measured with
  `freqz`, and the filter is lengthened until it meets **≥ 80 dB stopband
  and ≤ 0.01 dB passband ripple**.

**The design is alias-free, not just image-free.** The stopband starts below
the output Nyquist `sr_hz/2`. On decimation, what folds into `±stopband`
comes from beyond `sr_hz − stopband`, which is stopband. On interpolation,
the images land there too. That is what makes the reconstruction a defined
quantity rather than an approximation (§3.8).

### 3.7 Selection

The `selection` JSON records how the range window was chosen:

```json
{"method": "auto_gate+geometry", "margin_km": 150.0, "max_hops": 3,
 "auto_gate_km": [2666.016, 2827.148], "geometry_km": [2635.687, 3961.924],
 "manual_km": null}
```

`method` is `auto_gate`, `geometry`, `auto_gate+geometry` (the union), or
`manual`. Unused candidates are `null`.

**The window follows the energy, not the geometry.** `architecture.md` records
a 45 ms epoch error that displaced an echo by 13,500 km. A window built from
the path alone would have kept an empty band and dropped the only echo, and
once `z` is gone that cannot be undone. The window is therefore the
**union** of two candidates:

- `calibrate.auto_gate` on the product's own power ionogram. For v2 that is
  the stored `SNR`; for `.lfs` it is `spectro.compute` over the whole axis,
  not the header gate.
- `calibrate.geometry_gate` for 1…`max_hops` hops, where the geometry is
  known. A relative axis has none, so there the window is `auto_gate` alone.

The writer adds `margin_km` to each side of that union and clips it to what
the samples can represent. A `manual` range is kept exactly as given, with no
margin. If neither candidate exists, nothing is written: a sounding of pure
noise with no known path has no window, and guessing one would be the
irreversible kind of wrong.

**Interference stays in the window.** A broadcast carrier crossing the beat
band is part of what the receiver heard, and a simulator of this channel has
to see it too.

### 3.8 Reading it back

To reconstruct the window at the source rate, interpolate with the stored
taps and shift back:

```
up[i]  = D · Σ_k z[k] · h[i − k·D + (N−1)/2]
x̂[m]   = up[m + L·D] · exp(+j2π · shift_hz · m / sr_src),   0 ≤ m < n_src
```

Because the design is alias-free, `x̂` is the source passed through `h`
twice, `h ∗ h ∗ y`, shifted back. It is that to a relative RMS of 5.6e-6
(§6.1), so it equals the source within the passband to the filter's 0.0006 dB
ripple. `muf.echo_window.reconstruct` implements exactly this.

An ionogram is re-derived by putting `x̂` through the source's own
spectrogram and dividing by §3.5's noise:

- **v2:** `io_chirp.v2_spectrogram` on `conj(x̂)`, 13× oversampled, at
  `source_window`/`source_step`.
- **`.lfs`:** `spectro`'s non-overlapping Hann rows.

The result covers the window's range and nothing else. Without an explicit
gate, an `.lfs` window is also held to the gate its capture loads at
(`calibrate.default_gate`: the ground path to the header's `rmax`), so a
window as wide as the band — ionozond's `--full`, ±60 000 km — gives the
capture's own axis rather than the direct signal and the mirror half. On
cyprus1_20260204_120010 that makes all four extractors match the `.lfs`;
without it, k-means found no MUF. A window entirely outside that gate is kept
whole. v2 has no such step, since `io_chirp.reprocess` gives its whole axis
too.

## 4. Using it

```bash
# what would be kept, and how big -- writes nothing
muf echo-window /data/2026.02.04/cyprus1_20260204_120010.lfs --dry-run

# standalone windows for a day of captures, somewhere other than beside them
muf echo-window /data/2026.02.04 --out /archive/echo/2026.02.04 --verify

# a v2 product: append, then (separately, deliberately) reclaim the space
muf echo-window lfm_ionogram-*.h5 --verify
muf echo-window lfm_ionogram-*.h5 --drop-raw

# info prints one line for a window
muf info /archive/echo/2026.02.04/cyprus1_20260204_120010.echo.h5
#   echo window 2486-4112 km (absolute), 888.9 Hz (/45), 1.78 MB
```

| Flag | Effect |
|---|---|
| `--margin-km KM` | range kept beyond the candidates on each side (150) |
| `--max-hops N` | hops the geometry candidate covers (3); §6.4 |
| `--range LO,HI` | keep exactly this range instead of selecting one |
| `--into H5` | `.lfs` only: append to this product rather than writing `<stem>.echo.h5` |
| `--out DIR` | `.lfs` only: where `<stem>.echo.h5` goes (default: beside the capture) |
| `--drop-raw` | v2 only: rewrite without `z`; always verified first |
| `--verify` | re-cut and compare; §5 |
| `--dry-run` | print the window, rate and size |

`muf.loader` reads a window wherever it makes sense:

- `loader.load(x.echo.h5)` re-derives an ionogram over the window's range,
  and `muf run` and `muf plot` accept the file.
- For a v2 product whose `z` was dropped, `loader.load(path)` is unchanged:
  it reads the stored `SNR`. `loader.load(path, window=N)` re-derives from
  the window instead and **warns that the range is restricted to it**.
- A product that still has `z` behaves exactly as before.

`ChirpHeader.has_echo_window` and `io_chirp.read_echo_window(path)` expose
the window directly.

## 5. Safety

**Nothing here is destructive by default:**

- An `.lfs` is opened read-only and never modified or deleted.
- A full `z` is dropped only with `--drop-raw`, and only after the rewritten
  file verifies.
- The writer is idempotent and works per file. A target that already holds a
  window is left alone, so a second run is a no-op that leaves the file
  byte-identical.
- Every write goes to a temporary sibling in the same directory and is
  renamed into place atomically. An interrupted or failed write leaves the
  original exactly as it was and removes the temporary file.

**Verification** (`--verify`, and always for `--drop-raw`) checks five
things:

1. The group reads back with every attribute.
2. The stored taps meet ≥ 80 dB stopband and ≤ 0.01 dB ripple, measured
   afresh. This check needs no source, so a product whose `z` is gone can
   still be checked this far.
3. Re-cutting the source with the stored taps gives the stored samples
   **bit for bit**.
4. Power re-derived from the window matches the source's own within the
   window: median |Δ| ≤ 0.5 dB over cells 10 dB above noise.
5. For `--drop-raw`, every other dataset and root attribute came across
   unchanged.

HDF5 does not give space back when a dataset is deleted. That is why
`--drop-raw` builds a new file rather than deleting `z` in place.

## 6. Measured

### 6.1 Synthetic: v2 and `.lfs` (2026-10-15)

`tests/test_echo_window.py` builds both sources. A v2 product is built by
`make_chirp_z_h5` from a chosen waveform, with its `SNR` derived by v2's own
spectrogram. An `.lfs` is synthesized at cyprus1's rates.

| Criterion | Required | Measured |
|---|---|---|
| Position: re-derived echo at its injected range | same bin | same bin, v2 and `.lfs`. The v2 mirror range a missing conjugation would give lies outside the window |
| Suppression outside the window | ≥ 80 dB | 95.4 dB at 5 km past the stopband edge on either side; 136.7 and 148.8 dB at −4000 and 9000 km |
| Doppler ν, σ from the window | \|Δν\| < 0.02 Hz, σ ±5 % | Δν < 1e-4 Hz, σ +0.00 % for (0.3, 0.5) and (−1.2, 0.2) Hz |
| GDD: slope of an echo whose delay is linear in frequency | same slope | 59.9414 km/MHz full and window (injected 60; Δ 2e-9 relative) |
| Sample bookkeeping | every sample's time, sweep frequency and content | exact. A one-sample slip would be a 0.05 rad error, and the test fails one |
| Round trip vs `h ∗ h ∗ source` | rel. RMS < 1e-3 | **5.6e-6**, whole record |
| Compatibility | loads identically; `REQUIRED` unchanged; dropped product loads with `has_raw_voltage=False`, `has_echo_window=True`; `--drop-raw` reclaims space; interrupted rewrite leaves the original | all pass. A dropped product's size falls by more than 0.9 × `z` |

**The v2 path is synthetic-only.** No v2 product with `z` exists locally:
all 11,354 `lfm_ionogram-*.h5` under `/Users/w/data/ionozond_data2`
(2026-08-04 to 2026-10-14) were opened on 2026-10-15, and none has `z`
(15 were unreadable). `save_raw_voltage` has never been on at DOB.

### 6.2 Real: cyprus1 → Yoshkar-Ola, 2026-02-04 (measured 2026-10-15; sizes re-measured 2026-10-16 with the taps stored)

Five 80 MB captures at 100 kHz/s and 40 kHz, `/Users/w/data/2026.02.04/`.
Windows were written to a scratch folder, never beside the captures.
`tools/echo_window_check.py` produces the whole table.

| Capture | Window, km | `auto_gate` / geometry, km | Rate (D) | Taps | Samples | File | vs `.lfs` |
|---|---|---|---|---|---|---|---|
| 03:00 | 2486–4112 | 2681–2886 / 2636–3962 | 888.9 Hz (45) | 3595 | 1.78 MB | 1.58 MB | 1/51 |
| 06:00 | 2486–4112 | 2666–2827 / same | 888.9 Hz (45) | 3595 | 1.78 MB | 1.57 MB | 1/51 |
| 09:00 | 2486–4112 | 2666–2725 / same | 888.9 Hz (45) | 3595 | 1.78 MB | 1.57 MB | 1/51 |
| 12:00 | 2486–4112 | 2666–2827 / same | 888.9 Hz (45) | 3595 | 1.78 MB | 1.57 MB | 1/51 |
| 18:00 | 2486–4112 | 2695–2769 / same | 888.9 Hz (45) | 3595 | 1.78 MB | 1.58 MB | 1/51 |

Every capture verified: the stored filter re-measured at 82.6 dB stopband
and 0.0006 dB ripple, the re-cut was identical, and the in-window power
agreed to 0.000 dB. No window was flagged: the edge energies were 36.6–38.1 dB, under the
43 dB threshold. Writing and verifying took 2.9–4.5 s a capture on this Mac.
The file is the gzip'd samples (§6.3) plus 28,760 bytes of taps.

| Capture | MUF contour, full / window | MUF algo, full / window | Power: median \|Δ\|, cells ≥ 10 dB above noise |
|---|---|---|---|
| 03:00 | 12.221 / 12.221 | 12.200 / 12.200 | 0.0001 dB (899) |
| 06:00 | 28.912 / 28.912 | 28.891 / 28.891 | 0.0002 dB (2573) |
| 09:00 | 32.086 / 32.086 | 32.066 / 32.066 | 0.0002 dB (2615) |
| 12:00 | 30.161 / 30.161 | 30.141 / 30.141 | 0.0001 dB (2704) |
| 18:00 | 12.671 / 12.671 | 12.651 / 12.651 | 0.0001 dB (925) |

MUF is **identical** in every case: 0 bins, where one bin is 0.02048 MHz.
The full value comes from the whole pipeline on the `.lfs` at its default
gate, and the window value from the same pipeline on the `.echo.h5`.

**Doppler**, 1F along 22.8–26.8 MHz at 12:00. The estimator is
`doppler_along_sweep.py`'s: trace, cubic trajectory, second dechirp,
100 Hz, Welch with 10 s Hann segments, RMS width within ±3 Hz.

| Run | RMS width | Residual shift | Segments traced |
|---|---|---|---|
| full `.lfs` | 0.3664 Hz | +0.033 Hz | 295 |
| window, traced independently | 0.3692 Hz (**+0.77 %**) | +0.029 Hz | 284 |
| window, on the capture's trajectory | 0.3684 Hz (+0.56 %) | +0.033 Hz | 295 |

All three runs are within the 1 % required, and consistent with the 0.37 Hz
measured earlier. The independent trace accepts 11 fewer segments because
its noise median can only come from inside the band. That accounts for
0.2 points of the 0.77 %. The remaining 0.56 % appears even on a shared
trajectory, so it is a property of the data the estimator sees, not of the
trace. The block-mean decimation by 400 is a poor anti-alias filter, so
out-of-band content in the full capture reaches the 10–40 Hz floor region,
and the window has none to contribute. That explanation is a hypothesis,
not a measurement; see §7.

### 6.3 Compression (2026-10-15)

On the five windows above, gzip-4 with byte shuffle saves **14.4–14.9 %**:
1.78 → 1.54–1.55 MB of samples. The saving comes from the exponent bytes; the mantissas
of complex noise are incompressible. gzip is in every HDF5 build and every
reader, so it is on.

### 6.4 What `--max-hops` costs (12:00, dry run, 2026-10-15)

| `--max-hops` | Window | Rate (D) | Samples | |
|---|---|---|---|---|
| 1 | 2486–2977 km | 307.7 Hz (130) | 0.62 MB | **truncation suspected**: 2F energy at the high edge |
| 2 | 2486–3421 km | 533.3 Hz (75) | 1.07 MB | |
| 3 (default) | 2486–4112 km | 888.9 Hz (45) | 1.78 MB | |

The task expected a rate of about 300–500 Hz on cyprus1. That is what one or
two hops would give. The geometry candidate for three hops is what widens the
band. With one hop the edge check does its job: the 2F mode sits inside the
upper margin and the window says so.

## 7. Decisions and what is left

Decided 2026-10-16:

- **Three hops.** The geometry candidate covers 1F–3F, at 1.78 MB of samples
  rather than the 1.07 MB two hops would give, because the simulator keeps
  the 3F family.
- **Taps stored**, not regenerated (§3.6).
- **The Doppler residual accepted.** The 0.56 % on a shared trajectory is
  inside the 1 % criterion. It is attributed in §6.2 to the block-mean
  decimation, but that attribution is unconfirmed: confirming it would mean
  running the estimator on the full capture after the same band-limiting.
- **The `.lfs` epoch assumption accepted.** `start_epoch` is the time of
  sample 0. The header has no field for an acquisition delay; if the
  recorder has one, `t_first_s` is off by it.

Still open:

- **v2 vertical products.** The window follows the product's own range axis,
  which for vertical v2 is twice `.lfs`'s (`io_chirp.load` warns about this).
  It is untested for want of a vertical v2 product with `z`.
