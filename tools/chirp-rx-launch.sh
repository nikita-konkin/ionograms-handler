#!/bin/sh
# Start the recorder with the LO the ini asks for.
#
# `chirp-rx.service` used to exec the binary directly, which left the LO in two
# places -- the compiled `set_rx_freq` and `center_freq` in my_station.ini --
# with nothing keeping them equal. calc_ionograms builds its downconversion
# mixer from the ini key (f0=-cf) while the samples come from wherever the
# recorder tuned; nothing checks the two and nothing can, because the LO is not
# written into the Digital RF metadata. A mismatch dechirps by the difference
# and produces empty products with no error in any log. It blinded the station
# twice on 2026-08-19.
#
# This is the systemd equivalent of patch 0015, which does the same for
# `dombas.sh` (the rollback path). It needs patch 0014 in the recorder: without
# it the binary has no `--center-freq` and boost::program_options rejects the
# unknown option, so an un-rebuilt recorder fails loudly here instead of
# tuning to the wrong band quietly. That is the intended behaviour -- do not
# "fix" it by dropping the flag when it is not recognised.
set -eu

CHIRP_DIR=${CHIRP_DIR:-/home/ionouser/chirpsounder2}
CONF_FILE=${CONF_FILE:-$CHIRP_DIR/my_station.ini}
PYTHON=${PYTHON:-$CHIRP_DIR/.venv38/bin/python3}
RINGBUFFER_DIR=${RINGBUFFER_DIR:-/dev/shm/hf25}
USRP_ARGS=${USRP_ARGS:-addr0=192.168.10.2,recv_buff_size=500000000}

# `build_fvec=False` skips the detection frequency vector -- n_samples_per_block
# floats, allocated only to be thrown away. `verbose=False` keeps chirp_config's
# own prints off this script's stdout.
CENTER_FREQ=$(cd "$CHIRP_DIR" && "$PYTHON" -c 'import sys, chirp_config; print(chirp_config.chirp_config(sys.argv[1], verbose=False, build_fvec=False).center_freq)' "$CONF_FILE")

# An unreadable ini stops the start. The alternative is falling back to the
# recorder's built-in default, which is the silent 12.5-vs-20 MHz split that
# caused the damage in the first place. Restart=always will retry, and the
# journal will say why every ten seconds, which is the loudest failure
# available here.
if [ -z "$CENTER_FREQ" ]; then
    echo "FATAL: could not read center_freq from $CONF_FILE" >&2
    exit 1
fi
echo "LO $CENTER_FREQ Hz (from $CONF_FILE)"

# `exec`, so the recorder inherits this PID and stays systemd's MAINPID.
# Without it the shell is MAINPID and KillSignal=SIGINT goes to the shell --
# and a USRP that does not receive SIGINT keeps transmitting UDP to a host that
# is gone, recoverable only by removing power. That is a site visit, so this
# `exec` is load-bearing safety, not tidiness.
exec "$CHIRP_DIR/rx_uhd_ext_gps" \
    --outdir="$RINGBUFFER_DIR" \
    --usrp_args="$USRP_ARGS" \
    --center-freq="$CENTER_FREQ"
