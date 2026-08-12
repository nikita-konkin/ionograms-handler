#!/bin/sh
# One sample of the recorder's two packet-loss counters, against the load
# average that produced them.
#
# This exists because patch 0008 (core isolation) was validated in a single
# evening, and every clean window fell at load 6-8 while the fault it fixes was
# measured at 9.4. Four readings taken at the wrong load are an anecdote. The
# claim needs a daytime peak, and nobody is going to be at the keyboard for it.
#
# **Both counters, because they count different losses.** `RcvbufErrors` in
# /proc/net/snmp is the kernel discarding datagrams the process did not read in
# time -- a socket-level loss, the one patch 0005 addressed. The `D` markers in
# thor.log are UHD reporting that the *USRP* discarded samples it could not hand
# over, which is what latency causes and what 0008 fixes. One can be zero while
# the other runs.
#
# Stateless per invocation, so it suits a timer rather than a sleep loop: the
# previous sample lives in a state file, and the first run after a reboot
# reports a baseline rather than a nonsense delta.
#
# Reads only. It never touches the recorder, and it must not: `pkill` on a name
# that matches the recorder is a site visit.
set -eu

LOG=${DROP_WATCH_LOG:-$HOME/drop-watch.log}
STATE=${DROP_WATCH_STATE:-$HOME/.local/state/drop-watch}
THOR=${DROP_WATCH_THOR:-$HOME/chirpsounder2/logs/thor.log}

mkdir -p "$(dirname "$STATE")" "$(dirname "$LOG")"

now=$(date -u +%s)
stamp=$(date -u -d "@$now" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)
load=$(awk '{print $1}' /proc/loadavg)

# RcvbufErrors is found by *name* in the header line. Its column has moved
# between kernels, and a hardcoded index silently reports InErrors instead.
rcvbuf=$(awk '/^Udp:/ {
    if (head == "") { for (i = 2; i <= NF; i++) col[$i] = i; head = 1; next }
    print $(col["RcvbufErrors"]); exit
}' /proc/net/snmp)

# Read only what thor.log grew by. It is a 100 MB/s process writing progress
# markers; re-counting the whole file every 15 minutes would itself become
# load. A file that shrank was rotated or truncated, so start over rather than
# report a negative delta.
size=0
[ -f "$THOR" ] && size=$(wc -c < "$THOR" | tr -d ' ')

prev_size=0; prev_rcvbuf=""; prev_time=""
if [ -f "$STATE" ]; then
    # shellcheck disable=SC1090
    . "$STATE"
fi
[ "$size" -lt "$prev_size" ] && prev_size=0

drops=0
if [ -f "$THOR" ] && [ "$size" -gt "$prev_size" ]; then
    drops=$(tail -c "+$((prev_size + 1))" "$THOR" | tr -cd 'D' | wc -c | tr -d ' ')
fi

if [ -n "$prev_rcvbuf" ] && [ -n "$prev_time" ]; then
    printf '%s load=%s window_s=%s drops=%s rcvbuf_delta=%s rcvbuf=%s\n' \
        "$stamp" "$load" "$((now - prev_time))" "$drops" \
        "$((rcvbuf - prev_rcvbuf))" "$rcvbuf" >> "$LOG"
else
    printf '%s load=%s window_s=0 drops=0 rcvbuf_delta=0 rcvbuf=%s baseline\n' \
        "$stamp" "$load" "$rcvbuf" >> "$LOG"
fi

printf 'prev_size=%s\nprev_rcvbuf=%s\nprev_time=%s\n' \
    "$size" "$rcvbuf" "$now" > "$STATE"
