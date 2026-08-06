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

## Applying

```bash
cd ~/chirpsounder2
git apply --check /path/to/0001-rx_uhd_ext_gps-set-epoch-from-gpsdo.patch  # dry run
git apply         /path/to/0001-rx_uhd_ext_gps-set-epoch-from-gpsdo.patch
make rx_uhd_ext_gps        # or whatever the build rule is; it is one .cpp
```

The binary needs `cap_sys_nice` re-applied after any rebuild — `setcap` is
attached to the inode, not the path, so a new binary has none:

```bash
sudo setcap cap_sys_nice+ep ~/chirpsounder2/rx_uhd_ext_gps
```

Under systemd this does not matter: `chirp-rx.service` grants `LimitRTPRIO=99`
directly, which is why the unit does it that way.

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
