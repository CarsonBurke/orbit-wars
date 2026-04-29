#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

"$PYTHON" scripts/latest_replay.py "$@" | tee "$tmp"

replay="$(awk -F': ' '/^replay: / { path=$2 } END { print path }' "$tmp")"
if [[ -z "$replay" ]]; then
  echo "latest_replay.py did not print a replay path" >&2
  exit 1
fi

firefox "$replay" >/dev/null 2>&1 &
