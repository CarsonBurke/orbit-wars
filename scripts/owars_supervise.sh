#!/usr/bin/env bash
# 24/7 training supervisor for the rented H100.
#
# Runs scripts/train.py and restarts it from the latest checkpoint if it dies,
# for crash-resilient long-haul training. A clean exit (code 0 = total_updates
# reached) stops the loop. Each (re)launch creates its OWN runs/<name>/<ts>/
# subdir, so TensorBoard logs are never overwritten.
#
# Usage:
#   scripts/owars_supervise.sh <name> <config> [total_updates] [extra train.py args...]
# Example:
#   scripts/owars_supervise.sh pmpo_sniper_v18_sdpa configs/ppo_sniper_pmpo.yaml 1000000
#
# Meant to be launched detached on the box, e.g.:
#   nohup scripts/owars_supervise.sh ... >/root/orbit-wars/logs/<name>.sup 2>&1 &
set -u

NAME="${1:?need run name}"
CONFIG="${2:?need config path}"
TOTAL="${3:-1000000}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")

REPO=/root/orbit-wars
cd "$REPO" || exit 1
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# Thread caps tuned for the ~64 effective cores (cgroup-limited) on this box.
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-62}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$REPO/logs"
LOG="$REPO/logs/$NAME.log"
CKPT_LATEST="$REPO/checkpoints/$NAME/latest.pt"

attempt=0
while true; do
  attempt=$(( attempt + 1 ))
  LOADARG=()
  if [ -f "$CKPT_LATEST" ]; then
    LOADARG=(--load "$CKPT_LATEST")
    resume_note="resume from latest.pt"
  else
    resume_note="fresh start"
  fi

  echo "=== $(date -u +%FT%TZ) [attempt $attempt] launching '$NAME' ($resume_note) total=$TOTAL ===" >>"$LOG"
  python scripts/train.py \
    --config "$CONFIG" \
    --name "$NAME" \
    --total-updates "$TOTAL" \
    "${LOADARG[@]}" "${EXTRA[@]}" >>"$LOG" 2>&1
  code=$?
  echo "=== $(date -u +%FT%TZ) [attempt $attempt] train.py exited code=$code ===" >>"$LOG"

  if [ "$code" -eq 0 ]; then
    echo "=== $(date -u +%FT%TZ) clean completion; supervisor stopping ===" >>"$LOG"
    break
  fi

  # Crash/kill: back off briefly, then resume from the last checkpoint.
  echo "=== $(date -u +%FT%TZ) restarting in 15s ===" >>"$LOG"
  sleep 15
done
