#!/usr/bin/env bash
# Open an SSH tunnel to TensorBoard on the rented Vast.ai H100.
# Usage: scripts/tb_tunnel.sh [local_port]   (default 6008; remote TB is 6006)
# Then browse http://localhost:<local_port>
# Requires the `owars` ssh alias (run scripts/owars_host.sh once to create it).
set -euo pipefail

LOCAL_PORT="${1:-6008}"
REMOTE_PORT=6006

echo "Tunneling localhost:${LOCAL_PORT} -> owars:${REMOTE_PORT} (remote TensorBoard)"
echo "Open http://localhost:${LOCAL_PORT} in your browser. Ctrl-C to close."
exec ssh -N -o ExitOnForwardFailure=yes -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" owars
