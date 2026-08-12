# Patches against the pinned chirpsounder2 clone

`chirpsounder2` is a **pinned clone with nothing of ours in it**
(`docs/architecture.md` §7). When a change genuinely has to happen inside the
clone, it lives here as a reviewable diff and is applied by hand on the
station — so the clone can still be re-pinned, diffed against upstream, or
updated without silently losing our work.

Applied against `0d27125`.

| Patch | What it changes | Why |
|---|---|---|
| `0001-rx_uhd_ext_gps-set-epoch-from-gpsdo.patch` | `rx_uhd_ext_gps.cpp` takes the USRP epoch from the GPSDO's `gps_time` sensor instead of the host clock | the fault behind both observed timing errors at DOB |
| `0002-calc_ionograms-bounded-digitalrf-bounds-wait.patch` | `calc_ionograms.py`'s `get_valid_bounds` gives up after 30 s instead of polling forever | an empty ringbuffer wedged the reader permanently; two days of soundings with no ionograms |
| `0003-dombas-start-ringbuffer-and-fix-launch-order.patch` | `examples/marieluise/dombas.sh` starts `drf ringbuffer`, starts the recorder first, runs `find_timings.py` in serendipitous mode, and unbuffers every log | the ram disk was never trimmed, so it sat at 100% and the recording developed holes |
| `0003-local-dombas-DOB-…patch` | the same change, against **DOB's edited copy** rather than upstream's | DOB's `dombas.sh` has diverged (`$HOME` paths, `.venv38`, `my_station.ini`, quoted variables), so the upstream-based 0003 does not apply there |
| `0004-dombas-run-calc_ionograms-under-mpi.patch` | `calc_ionograms.py` runs under `$MPIRUN -np 2`, as `detect_chirps.py` already does | it ran as one process and missed the ringbuffer window for **4.28% of soundings**, lost silently as "missing data - skipping". **Two ranks, not four** — at four the machine saturated and the recorder overflowed |
| `0005-dombas-set-usrp-recv-buff-size.patch` | the recorder asks UHD for a 500 MB receive socket buffer | the socket queue had discarded **1.86 billion datagrams, ~6% of every recording**, invisible to every counter except `netstat -su` |
| `0006-makefile-build-recorder-with-O2.patch` | the `rx_uhd_ext_gps` rule builds with `-O2` | the rule had no `-O` flag at all, so `make` produced a **debug build** of a program moving 100 MB/s against a deadline |
| `0007-dombas-move-digisonde-and-drop-plotters.patch` | `dombas.sh` no longer starts the five `receive_digisonde.py` instances or the three plotters | the receivers demodulate digisondes **off air from the ringbuffer** — they are not downloaders — and five of them cost the recorder **~969 dropped events/s**. DOB does not use those products: their range zero is a configured `offset_us`, not a measured delay. Removing them took the drop rate to **zero over an hour** — necessary, but not sufficient: see 0008 |
| `0008-dombas-give-the-recorder-its-own-core.patch` | `dombas.sh` pins itself to CPU 1–7 so every child inherits it, and launches the recorder with `taskset -c 0` | with the receivers gone the recorder *still* lost 358,691 samples per 900 s at load 9.4. The fault is **latency, not throughput** — it needs 0.8 of a core the instant a packet arrives, and a run queue of 6–12 does not give it that. Pinned: **zero drops, `RcvbufErrors` frozen for 74 minutes** |

**0003 has two forms.** The unsuffixed one is against `0d27125` and is the
canonical diff, per this directory's convention. DOB's working copy has local
edits — better ones than upstream's, so the local variant is rebased onto
*those* rather than reverting them. Its base blob is `a530530`; check with
`git hash-object examples/marieluise/dombas.sh` before applying, and if it
differs, the file has moved on again and the patch needs rebasing.
`0003-local-dombas.sh.result` is the finished file, for when that is simpler
than a rebase.

0001-0002 and 0003-0005 are independent — different files, different faults — and all three
are about the same thing: a station that keeps running while producing nothing.
0002 turns a permanent hang into a logged retry, 0003 removes the condition
that caused it, and 0001's host-clock fallback is only visible because of them.

**0004 applies on top of 0003**, whose context it assumes — it edits the same
line region of the same file. It is the only patch here that is a performance
change rather than a correctness one, and it is measured: re-run

```bash
grep -oE '\-?[0-9.]+ s left' logs/find_timings.log \
  | awk '{n++; if($1<=0) z++} END {printf "n=%d lost=%d (%.2f%%)\n", n, z+0, 100*z/n}'
```

after a day. Baseline to beat is 4.28%. Note the `-?` — without it the sign is
dropped and every failure counts as a comfortable pass.

**0005 through 0008 are all one fault seen from four angles**, and it took two
of them to fix it. The host could not keep up with its own radio, so the socket
queue overflowed (0005), the device overflowed, and both counters were blamed
on things that were merely suboptimal. A bigger socket buffer and a `-O2` build
are both real improvements and neither moved the loss: 0005 was followed by
1.17 billion more `RcvbufErrors` overnight, and 0006 took the drop rate from
52% to 42%.

Taking 2.7 cores of unrelated work off the machine (0007) took it to zero — for
an hour, on a quiet evening. At load 9.4 the next evening the same six
processes lost 358,691 samples in 900 s. **0008 is what actually fixed it**:
the failure is *latency*, not throughput. There was CPU to spare the whole
time; the recorder simply was not scheduled the moment a packet arrived, and
the USRP discards what it cannot hand over. Giving it a core nothing else may
touch took the drops to zero and froze `RcvbufErrors` for 74 minutes.

Two lessons, and the second is the expensive one. **Prefer removing load over
tuning for it** — when a counter is still moving, no amount of buffer is the
answer. And **an average is not a margin**: 0007 was written up as the fix on
the strength of one hour at load 7.5, and the conditions that break this
recorder are at load 9.4. Measure a fix under the load that produced the fault,
and print `uptime` next to every number.

**Apply 0003 first if you are applying several.** It fixes the environment the
other two run in; without a trimmed ringbuffer the rest is treating symptoms.

## Applying

```bash
cd ~/chirpsounder2
git apply --check /path/to/0001-rx_uhd_ext_gps-set-epoch-from-gpsdo.patch  # dry run
git apply         /path/to/0001-rx_uhd_ext_gps-set-epoch-from-gpsdo.patch
make rx_uhd_ext_gps        # or whatever the build rule is; it is one .cpp
```

**Check the build carries `-O2`** -- patch 0006 fixes the rule, but a clone
that has not taken it still builds unoptimised. On 2026-08-11 the rule expanded to

    g++ -std=c++11 `pkg-config --cflags ...` -o rx_uhd_ext_gps rx_uhd_ext_gps.cpp ...

with no optimisation flag at all, which means `-O0`. That was suspected of
causing a recorder that dropped half its samples; it turned out not to be the
cause (see `docs/2026-08-11-recorder-packet-loss.md`), but an unoptimised
25 MS/s receive loop is not something to ship on purpose. Build it explicitly
if the rule does not:

```bash
g++ -O2 -std=c++11 $(pkg-config --cflags uhd hdf5 digital_rf) \
    -o rx_uhd_ext_gps rx_uhd_ext_gps.cpp -pthread \
    -lboost_program_options -lboost_system -lboost_thread -lboost_date_time \
    -lboost_regex -lboost_serialization -ldigital_rf \
    $(pkg-config --libs uhd hdf5 digital_rf)
```

The binary needs `cap_sys_nice` re-applied after any rebuild — `setcap` is
attached to the inode, not the path, so a new binary has none:

```bash
sudo setcap cap_sys_nice+ep ~/chirpsounder2/rx_uhd_ext_gps
```

Under systemd this does not matter where the unit grants `LimitRTPRIO=99`
directly. `examples/marieluise/chirpsounder_dombas.service` does not, so on DOB
the `setcap` above is required after every rebuild.

0002 is Python — nothing to build, but the running process keeps the old code
until it is restarted:

```bash
cd ~/chirpsounder2
git apply --check /path/to/0002-calc_ionograms-bounded-digitalrf-bounds-wait.patch
git apply         /path/to/0002-calc_ionograms-bounded-digitalrf-bounds-wait.patch
pkill -f calc_ionograms.py     # the launcher restarts it; if not, see 0002 below
```

0003 is a shell script, so it takes effect on the next launch. It moves the
recorder into a background subshell and ends on `wait`, so the script must be
started the way the unit starts it, not sourced:

```bash
cd ~/chirpsounder2
git apply --check /path/to/0003-dombas-start-ringbuffer-and-fix-launch-order.patch
git apply         /path/to/0003-dombas-start-ringbuffer-and-fix-launch-order.patch
./stop_ringbuffer.sh              # stop everything, including any hand-started process
./examples/marieluise/dombas.sh   # or: systemctl --user restart chirpsounder_dombas
```

Check it took, about a minute in:

```bash
pgrep -af 'drf ringbuffer'; df -h /dev/shm     # expect a PID, and well under 100%
```

For a station on a different config, `CONF_FILE` is now overridable:

```bash
CONF_FILE=~/chirpsounder2/my_station.ini ./examples/marieluise/dombas.sh
```

## 0001 — epoch from the GPSDO

### The fault

`rx_uhd_ext_gps` selects `gpsdo` as both clock source and time source, waits
for `gps_locked`, prints the result — and then sets the USRP clock from the
**host**:

```cpp
usrp->set_time_next_pps(uhd::time_spec_t(pc_secs + 1));   // :433
```

So the PPS *edge* is disciplined to GPS at sub-microsecond, while the *second
number* is whatever `ntpd` last left in the system clock. The `gps_time`
sensor, which is exact by construction and needs nothing configured on the
host, is never read. `rx_uhd.cpp:312` does read it — the plain recorder is
strictly better for absolute time, and the `_ext_gps` variant, despite the
name, is the one that inherits every NTP error.

Two failures at DOB, one line of code:

| Date | Host clock error | Effect |
|---|---|---|
| 2026-08-05 | −0.956 s | every echo displaced 286,000 km; products stayed perfectly self-consistent and took two days to diagnose against an external schedule |
| 2026-08-06 | −5.3 years | a run stamped `PC time now: 1617339242` = 2021-04-02 |

Because this is stretch processing, `range = c·δt` — 1 ms is 300 km. There is
nothing inside a product that can reveal the error, which is what made the
first one expensive.

### The change

1. `gps_locked` is hoisted out of the `if (using_internal_gpsdo)` block, so
   the epoch decision can see whether the GPSDO actually locked rather than
   only whether it exists.
2. The epoch comes from `get_mboard_sensor("gps_time", 0)` when there is an
   internal GPSDO, it is locked, and the sensor is present.
3. It falls back to the host clock otherwise, with a warning naming which
   condition failed. **The external 10 MHz / PPS installations are unchanged**
   — they have no `gps_time` sensor to read, and for them the host clock is
   all there is.
4. The host-vs-GPS difference is printed. Free diagnostics: it is the number
   we spent two days measuring indirectly.
5. After the clock is set, it is **verified** against a fresh `gps_time` read
   sampled at the same instant as the USRP clock.

### The sensor read blocks, and it took a false alarm to learn it

`get_mboard_sensor("gps_time")` does not return a cached value. It blocks on
the GPSDO's serial link until a fresh NMEA sentence arrives — **measured at
1–2 s on this FireFly**. The value therefore describes the second in progress
when the call *returns*, and anything sampled before the call is stale by the
read duration.

Two consequences, and the second one bit:

- **`set_time_next_pps(gps_time + 1)` is correct, and self-correcting rather
  than lucky.** The read completes inside the second it reports, so the next
  PPS edge is exactly that second plus one. This is why the canonical UHD
  idiom works.
- **Anything compared against a value latched before the read measures the
  serial link, not the clock.** The first version of this patch did exactly
  that, twice: it printed `host clock is -1 s from GPS` and then
  `EPOCH CHECK FAILED: USRP epoch is -2 s from GPS` on a station whose epoch
  was good to 2 ms. Worse, it *re-set the clock* on that misreading — a retry
  driven by a bad comparison is precisely how this code would introduce the
  fault it exists to prevent.

Ground truth came from the products, not the log: cyprus1's arrival phase was
`235.0094 / 240.0093 / 245.0094` before the restart and `235.0096 / 240.0097`
after, agreeing to 0.3 ms across it. A real 2 s error would have moved those
to 237.01 and 242.01.

So the current version:

- re-reads the host clock **after** the sensor, and reports how far into the
  GPS second it lands (expect 0 to 1) rather than a bogus whole-second delta;
- prints the sensor read duration, so the next person sees the 1–2 s
  immediately instead of deriving it from a false alarm;
- samples `get_time_now()` immediately after the sensor read for the check;
- reports **inconclusive** rather than failed when sampled within 0.3 s of a
  PPS edge, where the sentence for the new second may not have arrived yet —
  a false "epoch is wrong" on a healthy station teaches operators to ignore
  the line, which costs more than the check is worth;
- **does not re-set the clock automatically.**

### What the log looks like afterwards

```
 * mboard 0 gps_locked: true
PC time now: 1786043266 + 0.166608 sec
GPSDO gps_time: 1786043267 (sensor read took 1.03 s; host clock is 0.2 s into
    that GPS second -- expect 0 to 1)
Setting USRP time to: 1786043268 at next PPS [source: GPSDO gps_time]
USRP time now 1786043269.0184 USRP last pps 1786043269.0000
Epoch check OK: USRP clock agrees with GPSDO gps_time (1786043269)
```

`services/agent/logs.py` matches `EPOCH CHECK FAILED` and the host-clock
fallback warning, so `python -m services.agent triage` names either without
anyone reading the log.

### What it does not fix

The GPSDO must be locked. With no satellites the fallback is still the host
clock, so `health.system_clock` and `chirp-rx.service`'s
`After=time-sync.target` stay necessary — this patch removes the common
failure, not the need to keep NTP honest.

---

## 0002 — bounded wait for DigitalRF bounds

### The fault

`calc_ionograms.py` polls the ringbuffer for data bounds before it can process
anything, and the wait has no exit:

```python
def get_valid_bounds(d, ch, poll_s=1.0):          # calc_ionograms.py:58
    while True:
        b = d.get_bounds(ch)
        if b is not None and len(b) >= 2 and b[0] is not None and b[1] is not None:
            return b
        print("no DigitalRF data bounds available for channel %s; waiting for data" % (ch))
        time.sleep(poll_s)
```

`DigitalRFReader` caches its channel list **when it is constructed**. A reader
built before the recorder created the channel directory can therefore never see
that channel, however long it polls. The only cure is a new reader — and the
code that would build one is unreachable:

```python
elif conf.realtime:
    while True:
        try:
            d = drf.DigitalRFReader(conf.data_dir)   # :632  the only fix
            analyze_realtime(conf, d)                # :633  -> get_valid_bounds at :432
        except: print("error ... trying to restart"); time.sleep(1)
```

`get_valid_bounds` neither returns nor raises, so the `except` never fires and
`d` is never rebuilt. The process stays alive and busy for ever.

### What triggers it

Anything that leaves the channel empty at the moment the reader is built: a
reboot (`/dev/shm` is cleared), a recorder restart between soundings, or a
recorder that never starts streaming at all. The consumer is also started
before the producer — in `examples/marieluise/dombas.sh`, `calc_ionograms.py`
is line 62 and `rx_uhd_ext_gps` line 74 — so a cold start is a race in the
first place.

Observed twice at DOB, and the two look identical from outside:

| | ringbuffer | what was really wrong |
|---|---|---|
| 2026-08-05 → 08-07 | **had data** — 638 and 643 detections a day, metadata and digisonde products all arriving | a transient empty window wedged the reader, which then never reopened. This bug, and 0002 fixes it |
| 2026-08-08 | **empty** | `rx_uhd_ext_gps` stuck in its GPSDO lock wait, never streamed a sample. Not this bug — but 0002 is what put it in the log |

An earlier draft of this section blamed 0001 for the first case, on the grounds
that its two blocking `gps_time` reads add 2–4 s to recorder startup and could
tip the race. That was never established, and the second case shows the wait
hangs just as permanently with no 0001 involved. Recorded here because the
patch is easier to trust when its rationale is not overstated.

### Why nobody noticed for two days

Three things had to line up, and did:

- the process stays **alive**, so `pgrep` finds it;
- `chirpsounder_dombas.service` supervises `dombas.sh`, whose main process is
  the `rx_uhd_ext_gps` loop — a wedged background child is invisible to it, and
  `Restart=always` never fires;
- every *other* consumer keeps working, so detections, metadata and digisonde
  products all keep arriving. Only the ionograms stop.

The single symptom is the log line above repeating once a second in
`logs/ionograms.log`. Nothing in the products can say it.

### The change

`get_valid_bounds` takes `max_wait_s=30.0` and raises `IOError` when it expires,
which sends control back to the caller's `DigitalRFReader(conf.data_dir)`. Both
long-running callers (`:621` par-files, `:632` realtime) already wrap that in
`while True: try:`, so the retry is a reopen rather than a crash. Batch mode
(`:641`) has no `try`, so there it surfaces as a failure instead of a hang —
still the better outcome.

Data that merely arrives *late* is unaffected: the poll loop is unchanged up to
the deadline, and 30 s is far longer than a recorder that is simply starting up.

### Checking it

```bash
tail -20 ~/chirpsounder2/logs/ionograms.log
```

Wedged: `no DigitalRF data bounds available for channel ch0` repeating for ever.
Healthy: `Rank 0 chirp id 1 name SGO analyzing chirp-rate 500.01 kHz/s`.

After the patch a lost race self-heals within 30 s, and leaves a trail:

```
no DigitalRF data bounds available for channel ch0; waiting for data
...
error in calc_ionograms.py. trying to restart
IOError: no DigitalRF bounds for channel ch0 after 30 s; reopening the reader
Rank 0 chirp id 1 name SGO analyzing chirp-rate 500.01 kHz/s chirpt 54.0000 rep 60.00
```

To recover a wedged station without restarting acquisition, restart just that
child — the ringbuffer already has data, so it picks up immediately:

```bash
pkill -f calc_ionograms.py
cd ~/chirpsounder2 && python3 calc_ionograms.py --config ~/chirpsounder2/my_station.ini \
    > logs/ionograms.log 2>&1 &
```

### What it does not fix

The race itself. The patch makes losing it survivable, not impossible; starting
`calc_ionograms.py` after `rx_uhd_ext_gps`, or having it wait for
`$RINGBUFFER_DIR/ch0` to exist, would remove it. Nor does it give anything a way
to *notice* a stalled child — that needs either a per-process unit or an
external check on product age.

### Note for anyone regenerating this diff

It is against the committed `0d27125`, which has LF line endings. The clone
synced to the Mac has picked up CRLF, so `git apply --check` fails there while
succeeding on the station's own checkout. `dos2unix calc_ionograms.py` before
diffing against that copy.

---

## 0003 — start the ringbuffer, and start things in the right order

### The fault

`dombas.sh` never starts `drf ringbuffer`. Nothing else deletes old data, so the
ram disk fills and stays full:

```
/dev/shm   16G   16G used   122M free   100%
```

122 MB at 25 MS/s is **1.2 seconds** of headroom. The DigitalRF writer can no
longer allocate, the recording develops holes, and `read_vector_1d` throws on
one — which becomes `missing data - skipping` in `calc_ionograms.py:243`.
Meanwhile the recorder, the detector, the metadata merger and every plot keep
running and keep producing, so the station looks entirely healthy.

Every reference launcher runs the trimmer — `examples/ringbuffer/ringbuffer.sh`,
`ringbuffer_nodet.sh`, `ringbuffer_eclipse.sh`, `ringbuffer_serendip.sh` — and
`stop_ringbuffer.sh` already lists `"python.*drf"` among the processes it
expects to kill. It was simply never started here.

Starting it by hand took `/dev/shm` from 100% to **72%** and ionograms resumed
within one 300 s cycle.

### Three more, from the same comparison

- **The order is inverted.** `ringbuffer_serendip.sh` goes recorder → trimmer →
  consumers, ten seconds apart. `dombas.sh` starts all thirteen consumers and
  only then the recorder, so every one of them opens a `DigitalRFReader`
  against a ringbuffer that does not exist yet. 0002 makes that survivable;
  this makes it unlikely.
- **`find_timings.py` is missing.** Correct for `dombas.ini`, which is realtime
  mode — but a serendipitous config has `calc_ionograms.py` waiting on
  `par-*.h5` that no process produces. It now runs when, and only when, the
  config asks for it. **Its config argument is positional**: it reads
  `sys.argv[1]`, not `--config`, and given the flag it prints `No config
  provided - Using defaults` and scans `/mnt/data/juha/hf25`.
- **Every log is block-buffered.** `python3` writing to a redirected file
  buffers, so a waiting process and a wedged one produce identical silence, and
  a process's last words before hanging never reach the log. `-u` throughout.

### Sizing the ringbuffer

`RINGBUFFER_SIZE` is now set rather than commented out, and it is the number
that decides which soundings are analysable:

```
seconds of history = RINGBUFFER_SIZE / (sample_rate * 4 bytes)
```

At 25 MS/s that is 100 MB/s, so `12000MB` holds 120 s. A sounding must begin
processing within that window of its `t0` — `find_timings.py:138` prints the
margin for each as `N s left`, and measured start latency at DOB is 65–117 s.
That fits, but not by much. On a machine with more RAM, or with the ringbuffer
on an SSD or raid as `dombas.sh:17` suggests, more is strictly better: it is
the only lever that makes start latency stop mattering.

The read loop advances at `speed x realtime` while the buffer tail advances at
1x, so a **250 s sweep does not need a 250 s buffer** — only enough to hold
`t0` when processing starts, plus throughput above 1x thereafter. Do not size
the ringbuffer to the sweep duration; size it to the start latency.

### What it does not fix

Nothing supervises the background children. `chirpsounder_dombas.service`
watches this script, and the script waits on the recorder loop, so a consumer
that dies or wedges leaves the unit `active (running)`. **Watch the age of the
newest product, not the unit state** — that is the only signal that would have
caught any of 0002 or 0003 on the day rather than two days later.

`--gps-lock-timeout` is deliberately left at its default of `-1`, wait for
ever. An unlocked GPSDO means 0001 falls back to the host clock, and recording
plausible-looking wrong ranges is worse than recording nothing. The cost is
that the recorder can sit silently in that wait — which is exactly what
happened here, and what watching product age would have surfaced.
