#!/usr/bin/env bash
# Refresh the `owars` SSH alias in ~/.ssh/config from the live Vast.ai instance.
# After running this, use:  ssh owars 'cmd'   and   rsync ... owars:path
# Usage: scripts/owars_host.sh [instance_id]
set -euo pipefail

INSTANCE_ID="${1:-42016466}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VAST_API_KEY="$(grep -E '^VAST_API_KEY=' "$ROOT/.env" | cut -d= -f2)"
VASTAI="$ROOT/.venv/bin/vastai"
KEY="$HOME/.ssh/vast_owars"

URL="$("$VASTAI" ssh-url "$INSTANCE_ID")"     # ssh://root@HOST:PORT
HOSTPORT="${URL#ssh://root@}"
HOST="${HOSTPORT%%:*}"
PORT="${HOSTPORT##*:}"

CFG="$HOME/.ssh/config"
touch "$CFG"; chmod 600 "$CFG"
# Strip any previous managed block, then append a fresh one.
python3 - "$CFG" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
s = re.sub(r"\n?# >>> owars-vast >>>.*?# <<< owars-vast <<<\n?", "\n", s, flags=re.S)
open(p, "w").write(s.rstrip("\n") + "\n")
PY
cat >> "$CFG" <<EOF

# >>> owars-vast >>>
Host owars
    HostName $HOST
    Port $PORT
    User root
    IdentityFile $KEY
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 30
# <<< owars-vast <<<
EOF

echo "ssh alias 'owars' -> $HOST:$PORT (instance $INSTANCE_ID)"
ssh owars 'echo connected: $(hostname); nproc; nvidia-smi --query-gpu=name --format=csv,noheader'
