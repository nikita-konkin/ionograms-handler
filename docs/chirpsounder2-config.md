# The chirpsounder2 configuration surface

What every key in `my_station.ini` does, who reads it, and whether that reader
runs on this station. Built 2026-08-19 against the station's own clone at
`df97ab3d`, checked out beside this repo at `../chirpsounder2`.

This exists because on 2026-08-18 a day was spent moving the receive band, and
most of that day went to three discoveries that this table would have given
away in a minute: `minimum_analysis_frequency` was parsed and read by nothing
on our path, `min_freq`/`max_freq` were a second cap nobody had connected to
the symptom, and `center_freq` has to agree with a number compiled into the
recorder. See BACKLOG sec. 3.

## Read this before trusting the table

**The "readers" column is mechanical and it under-reports.** It is a word-match
over every `.py` and `.sh` in the tree for the ini key, its config attribute,
and known derived attributes. Four separate mechanisms in this codebase defeat
a naive search, and every one of them fails in the direction of *"looks dead,
is live"*:

| mechanism | example |
|---|---|
| the attribute is renamed | `receiver_station_name` is `conf.station_name` |
| the value is only used through a derived attribute | `minimum_frequency_spacing` is read only as `conf.mfsi` (`chirp_det.py:251`) |
| access is via `getattr` with a default | `ringbuffer_max_age_sec` (`iono_housekeeping.py:204`) |
| the reader is a shell script, not Python | `require_gps_lock` (`examples/marieluise/w2naf.sh:32`) |
| the value is only reachable through a config *method* | `serendipitous_allowed_range_starts_km`, via `conf.serendipitous_range_start_allowed()` |

So: an empty readers cell is a **question**, not a verdict. Three of them
survived a hand check; the rest did not.

The first generated version of the table below got two attributes wrong in the
other direction -- it read `plot_timings` as `conf.fname` and
`max_simultaneous_detections` as `conf.mfsi`, because the pattern that finds
`self.X = ... cf[sec][key]` matched across an unrelated intervening
assignment. A wrong attribute then poisons the readers column with every file
that happens to use a common word like `fname`. The generator now refuses a
match with another `self.` inside it. Mentioned because it is the failure mode
of this kind of table: it is confidently wrong, and it looks like the rest.

## There are two configuration surfaces, not one

`chirp_config.py` defines defaults for six sections -- `config`, `detection`,
`lfm`, `transfer`, `rtf`, `stations` -- and every key below comes from there.

**The live ini also has a `[digisonde]` section that `chirp_config` has never
heard of.** `receive_digisonde.py` and `plot_rtf.py` open the ini with a plain
`configparser` and read it directly, and `receive_digisonde.py:62` also reads a
`[system]` section that does not exist in any default table either. Nothing
validates these, nothing defaults them, and a typo in `[digisonde]` cannot be
caught by anything that goes through `chirp_config`.

A settings panel that renders "the chirpsounder2 config" from `chirp_config`'s
defaults would silently omit the entire digisonde receiver.

## The parser validates exactly one key

`downconversion_filter` must be one of `fir`, `fir_slow_oscillator`, `boxcar`,
`cic`, and raises if it is not (`chirp_config.py:230`). That is the whole of
the validation in a 68-key config. Everything else -- frequencies, rates,
decimations, ranges, thread counts -- is `json.loads`'d and used. A band wider
than the sample rate, a decimation that does not divide, an analysis span
longer than the repetition period: all accepted, none reported.

`copy_destination` goes further and is parsed inside a bare `try/except: pass`,
so a malformed value is not an error, it is a missing attribute.

## The four classes

1. **Live** -- read on the path this station runs. Safe to edit.
2. **Inert** -- parsed but read by nothing, or read only by a process this
   station does not run. Editing does nothing and says nothing.
3. **Externally coupled** -- must agree with something outside the ini.
4. **Mutually coupled** -- must agree with other keys.

### Class 2 instances currently in the live ini

* **`serendipitous` is set in `[lfm]` and read from `[config]`.** The live file
  has it in both. `chirp_config.py:216` reads `cf["config"]["serendipitous"]`
  only, so the `[lfm]` copy is decoration. Both happen to say `false`, so this
  has never bitten -- it is a trap armed and waiting.
* **`copy_destination = "shovel@4.235.86.214:/var/www/html/iono/"` is read by
  nothing.** See below; this is the most consequential finding in the census.
* `serendipitous_ionogram_workers`, and the serendipitous range keys, are live
  code but `serendipitous = false`.

### Class 3: the one that costs money

`center_freq` must equal the `set_rx_freq` compiled into `rx_uhd_ext_gps`.
Nothing checks it, and nothing can -- the LO is not in the Digital RF
metadata. A mismatch dechirps by the difference and produces empty products
with no error anywhere. This blinded the station twice on 2026-08-19.

`center_freq` is also read a second way that is easy to miss:
`chirp_config.py:313` builds `self.fvec`, the detection frequency vector, as
`fftfreq(...) + center_freq`. So it steers both the analysis mixer and the
detector's frequency axis.

BACKLOG sec. 30 closes this by passing `--center-freq` to the recorder, exactly
as `--gps-lock-timeout` is already passed.

### Class 4: what has to agree with what

| constraint | where it bites |
|---|---|
| `max_freq - min_freq` <= `sample_rate` | wider than the digitised band; silently truncated |
| analysed span inside `[center_freq ± sample_rate/2]` | the band the recorder actually captured |
| `min_freq`/`max_freq` inside the analysis bounds | `manual_freq_extent = true` gates the *stored* axis, `calc_ionograms.py:296` |
| `(max_analysis - min_analysis) / chirp-rate` < `rep` | the sweep must finish inside the cycle; at 100 kHz/s and `rep = 300`, that is 30 MHz |
| span <= `r * B / (1 - r)` | the ringbuffer budget, BACKLOG sec. 3 |
| `decimation` against `sample_rate` and `frequency_resolution` | sets the ionogram's own axis |

None of these are checked anywhere today.

## The upload is configured to one place and goes to another

This came out of the census and is worth stating on its own.

`copy_to_server = true` gates **seven** call sites of
`ionowebsync.post_to_server` -- in `calc_ionograms.py:438`,
`detections2metadata.py:98` and `:122`, `sync_iono_data.py:45`,
`plot_ionograms.py:368`, `plot_summary.py:120`, `receive_digisonde.py:555`.

`ionowebsync.py` posts to `os.environ.get("IONOWEBSYNC_URL", "https://juha.no/upload.php")`
with `timeout=60`. `IONOWEBSYNC_URL` is set in none of our systemd units.

`copy_destination` is read by nothing.

So the operator configured `shovel@4.235.86.214:/var/www/html/iono/`, and every
product is POSTed to `juha.no` instead -- unreachable from this network, so
each call blocks for the full 60 s. The one in `calc_ionograms.py:438` is
inside the process with the hard deadline, and is the cause of the 8.37%
sounding loss in BACKLOG sec. 3.

Three ways out, and the choice is a policy question, not a technical one:

* `copy_to_server = false` -- one ini key, kills all seven. Also stops
  `sync_iono_data.py` (it exits at line 18). Loses nothing the operator
  controls, because the destination they configured was never being used.
* `Environment=IONOWEBSYNC_URL=...` on `chirp-ionograms.service` -- fixes the
  deadline-critical caller only, leaves publishing intact elsewhere.
* Keep publishing to juha.no deliberately, and drop the timeout.

If publishing upstream is wanted, that is a fine reason to keep it -- but it
should be a decision, and right now it is an accident.

## The table

`station value` is what `my_station.ini` sets on Yoshkar-Ola today; an em-dash
means the key is absent and the default applies. An empty `attr` means the
config attribute has the same name as the ini key.

| sec | key | station value | attr | readers |
|---|---|---|---|---|
| config | `sample_rate` | 25000000 |  | benchmark_python_overhead.py, calc_ionograms.py, chirp_det.py, detect_chirps.py, detections2metadata.py, digisonde_search.py, dombas.sh, plot_digisonde_search.py, plot_downconversion_filter_responses.py, plot_rf_spec.py, power_spectra.py, receive_digisonde.py, serendipitous_ionogram_queue.py, station_monitor.py |
| config | `center_freq` | 20e6 |  | calc_ionograms.py, digisonde_search.py, plot_rf_spec.py, power_spectra.py, receive_digisonde.py |
| config | `data_dir` | "/dev/shm/hf25" |  | calc_ionograms.py, detect_chirps.py, detections2metadata.py, digisonde_search.py, find_timings.py, freq_slice.py, iono_housekeeping.py, manual_ionogram_scaler.py, plot_archive_quicklooks.py, plot_chirp_detections.py, plot_detectionfiles.py, plot_digisonde.py, plot_rf_spec.py, plot_rtf.py, power_spectra.py, receive_digisonde.py, serendipitous_ionogram_queue.py, station_monitor.py |
| config | `kill_path` | — |  | calc_ionograms.py, detect_chirps.py, digisonde_search.py, find_timings.py, plot_ionograms.py, serendipitous_ionogram_queue.py |
| config | `output_dir` | "/home/ionouser/ionozond_data2/" |  | calc_ionograms.py, chirp_det.py, crop_ionograms.py, debug_digisonde.py, detections2metadata.py, digisonde_search.py, find_timings.py, iono_housekeeping.py, plot_archive_quicklooks.py, plot_chirp_band_aoa.py, plot_detectionfiles.py, plot_downconversion_filter_responses.py, plot_ionograms.py, plot_rtf.py, plot_summary.py, power_spectra.py, receive_digisonde.py, serendipitous_ionogram_queue.py, station_monitor.py |
| config | `receiver_station_name` | "Yoshkar-Ola" | station_name | calc_ionograms.py, detections2metadata.py, digisonde_search.py, iono_housekeeping.py, plot_archive_quicklooks.py, plot_detectionfiles.py, plot_ionograms.py, plot_map.py, plot_rtf.py, plot_summary.py, power_spectra.py, propagation.py, receive_digisonde.py, station_monitor.py, sync_iono_data.py |
| config | `plot_timings` | — |  | find_timings.py |
| config | `realtime` | true |  | aeroauto.sh, calc_ionograms.py, debug_digisonde.py, detect_chirps.py, detections2metadata.py, digisonde_search.py, dombas.sh, find_timings.py, jens.sh, plot_detectionfiles.py, plot_ionograms.py, plot_rtf.py, receive_digisonde.py |
| config | `ringbuffer_max_age_min` | 3 |  | iono_housekeeping.py |
| config | `ringbuffer_max_age_sec` | — |  | iono_housekeeping.py |
| config | `ringbuffer_cleanup` | true |  | iono_housekeeping.py |
| config | `require_gps_lock` | — |  | w2naf.sh |
| config | `gps_lock_timeout_sec` | — |  | w2naf.sh |
| config | `serendipitous` | false |  | calc_ionograms.py, dombas.sh, plot_ionograms.py, serendipitous_ionogram_queue.py |
| config | `serendipitous_ionogram_workers` | — |  | serendipitous_ionogram_queue.py |
| config | `parameter_file_retention_sec` | — |  | iono_housekeeping.py |
| config | `parameter_file_lock_timeout_sec` | — |  | iono_housekeeping.py, serendipitous_ionogram_queue.py |
| config | `required_processes` | [ "recorder=rx_uhd_ext_gps/rx_uhd" |  | station_monitor.py |
| detection | `threshold_snr` | 13.0 |  | chirp_det.py, digisonde_search.py |
| detection | `max_simultaneous_detections` | 5 |  | chirp_det.py |
| detection | `min_detections` | 3 |  | find_timings.py, plot_archive_quicklooks.py, plot_chirp_band_aoa.py, plot_detectionfiles.py |
| detection | `step` | 12 |  | calc_ionograms.py, debug_digisonde.py, detect_chirps.py, digisonde_search.py, iono_housekeeping.py, receive_digisonde.py |
| detection | `n_samples_per_block` | 5000000 |  | chirp_det.py, detect_chirps.py |
| detection | `realtime_detection_lag_sec` | — |  | detect_chirps.py |
| detection | `minimum_frequency_spacing` | 0.2e6 |  | chirp_det.py |
| detection | `chirp_rates` | [100e3,125e3,500.0084e3] |  | chirp_det.py, detect_chirps.py, find_timings.py, plot_chirp_detections.py |
| detection | `debug_timings` | — |  | find_timings.py |
| detection | `propagation_range_bands` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py |
| detection | `propagation_range_transmitters` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py, propagation.py |
| detection | `propagation_range_factor` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py, propagation.py |
| detection | `propagation_band_fraction` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py, propagation.py |
| detection | `propagation_range_band_overrides` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py, propagation.py |
| detection | `detection_range_filter` | — |  | chirp_det.py, find_timings.py, plot_archive_quicklooks.py, plot_detectionfiles.py |
| detection | `detection_range_filter_min_km` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py, propagation.py |
| detection | `detection_range_filter_max_km` | — |  | plot_archive_quicklooks.py, plot_detectionfiles.py, propagation.py |
| lfm | `range_resolution` | 2e3 |  | calc_ionograms.py |
| lfm | `frequency_resolution` | 50e3 |  | calc_ionograms.py |
| lfm | `maximum_analysis_frequency` | 32.5e6 |  | calc_ionograms.py, find_timings.py, plot_ionograms.py |
| lfm | `minimum_analysis_frequency` | 7.5e6 |  | calc_ionograms.py, serendipitous_ionogram_queue.py |
| lfm | `max_range_extent` | 4000e3 |  | calc_ionograms.py, plot_ionograms.py |
| lfm | `serendipitous_range_quantization_km` | — |  | calc_ionograms.py, serendipitous_ionogram_queue.py |
| lfm | `serendipitous_range_extent_km` | — |  | calc_ionograms.py |
| lfm | `serendipitous_range_buffer_km` | — |  | calc_ionograms.py, serendipitous_ionogram_queue.py |
| lfm | `serendipitous_allowed_range_starts_km` | — |  | **none** |
| lfm | `serendipitous_publish_range_starts_km` | — |  | **none** |
| lfm | `max_ionogram_frequency_steps` | — |  | calc_ionograms.py |
| lfm | `min_range` | 800e3 |  | calc_ionograms.py, plot_ionograms.py |
| lfm | `storage_snr_threshold` | 2 |  | calc_ionograms.py |
| lfm | `max_range` | 1700e3 |  | calc_ionograms.py, plot_ionograms.py |
| lfm | `manual_range_extent` | false |  | calc_ionograms.py, plot_ionograms.py |
| lfm | `save_raw_voltage` | false |  | calc_ionograms.py |
| lfm | `fast_boxcar_filter` | false |  | calc_ionograms.py, chirp_lib.py |
| lfm | `downconversion_filter` | "fir" |  | calc_ionograms.py, chirp_lib.py |
| lfm | `cic_stages` | — |  | calc_ionograms.py, chirp_lib.py, plot_downconversion_filter_responses.py |
| lfm | `n_downconversion_threads` | 1 |  | calc_ionograms.py |
| lfm | `downconversion_block_samples` | 4000 |  | calc_ionograms.py |
| lfm | `min_freq` | 7.5e6 |  | calc_ionograms.py, plot_ionograms.py, serendipitous_ionogram_queue.py |
| lfm | `max_freq` | 32.5e6 |  | calc_ionograms.py, plot_ionograms.py |
| lfm | `manual_freq_extent` | true |  | calc_ionograms.py, plot_ionograms.py, serendipitous_ionogram_queue.py |
| lfm | `decimation` | 625 |  | calc_ionograms.py, chirp_lib.py, debug_digisonde.py, plot_downconversion_filter_responses.py, receive_digisonde.py, serendipitous_ionogram_queue.py, sgo.sh |
| lfm | `sounder_timings` | [[{"chirp-rate": 100000, "rep": 30 |  | calc_ionograms.py, dombas.sh |
| transfer | `copy_to_server` | true |  | calc_ionograms.py, plot_ionograms.py, plot_summary.py, receive_digisonde.py, sync_iono_data.py |
| transfer | `copy_destination` | "shovel@4.235.86.214:/var/www/html |  | **none** |
| rtf | `links` | [["SGO","Yoshkar-Ola"],["Ramfjordm | rtf_links | plot_archive_quicklooks.py, plot_map.py, plot_rtf.py, propagation.py |
| stations | `station_info` | — | **never parsed** | chirp_aoa_interactive.py, plot_archive_quicklooks.py, plot_chirp_band_aoa.py, plot_detectionfiles.py, plot_map.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lat` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `lon` | — | **never parsed** | bodnar_nmea.py, chirp_aoa_interactive.py, plot_chirp_band_aoa.py, plot_gcp.py, plot_map.py, plot_stations.py, propagation.py |
| stations | `links` | — | station_links | plot_map.py, plot_rtf.py, propagation.py |

`[digisonde]` and `[system]` are not in this table because `chirp_config` does
not define them. `receive_digisonde.py` reads `sounding_interval_sec`,
`transmitter_station_name`, `snr_threshold`, `filter_strategy`,
`use_c_downconvert`, `decimation`, `freq_start`, `freq_stop` straight from the
ini, and they are set on this station. Extending the census to cover them is
the obvious next pass.
