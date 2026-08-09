#!/bin/sh
# Local (non-Docker) launcher for the api + web console on this Mac.
#
# The Docker rig in deploy/ is the documented path, but its images have never
# been built end to end (deploy/README.md) and the daemon is not running here.
# Running uvicorn directly is the path deploy/README.md calls verified.
#
# Not committed on purpose -- delete it when the Docker rig works.
set -e
cd "$(dirname "$0")"

VENV="$HOME/.venvs/muf"
ARCHIVE="/Users/w/Library/CloudStorage/Nextcloud-192.168.50.117-radio/RAID0 Storage/lfs"

# CONTROL_TOKEN unset disables control rather than opening it, so read it from
# the same file the compose rig uses instead of inventing a second secret.
CONTROL_TOKEN=$(sed -n 's/^CONTROL_TOKEN=//p' deploy/.env)
export CONTROL_TOKEN
export API_DB=data/ionograms.sqlite3
export ARCHIVE_ROOT="$ARCHIVE"

exec "$VENV/bin/uvicorn" services.api.main:app --host 127.0.0.1 --port 8000 "$@"
