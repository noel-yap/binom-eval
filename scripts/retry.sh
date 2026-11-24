#!/usr/bin/env bash
# Run a command, retrying on failure with exponential backoff and full jitter.
#
# Usage: scripts/retry.sh [-n ATTEMPTS] [-b BASE_SECONDS] [-c CAP_SECONDS] -- CMD [ARGS...]
#
# Before retry N (N = 1, 2, ...) it sleeps a random number of seconds drawn
# uniformly from [0, min(CAP, BASE * 2^(N-1))] ("full jitter"), which spreads
# out concurrent retries instead of having them hit the service in lockstep.
# Exits 0 on the first success, otherwise with the last command's exit status.
#
# Defaults: 5 attempts, base 2s, cap 60s. RETRY_SLEEP overrides `sleep` (tests).
set -euo pipefail

attempts=5
base=2
cap=60

while getopts "n:b:c:" opt; do
  case "${opt}" in
    n) attempts="${OPTARG}" ;;
    b) base="${OPTARG}" ;;
    c) cap="${OPTARG}" ;;
    *) echo "usage: $0 [-n N] [-b BASE] [-c CAP] -- CMD [ARGS...]" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "retry.sh: no command given" >&2; exit 2; }

sleeper="${RETRY_SLEEP:-sleep}"

attempt=1
while true; do
  status=0
  "$@" || status=$?
  [ "${status}" -eq 0 ] && exit 0
  if [ "${attempt}" -ge "${attempts}" ]; then
    echo "retry.sh: giving up after ${attempt} attempts (exit ${status}): $*" >&2
    exit "${status}"
  fi
  ceiling=$((base << (attempt - 1)))
  [ "${ceiling}" -gt "${cap}" ] && ceiling="${cap}"
  # $RANDOM is 0..32767; scale to 0..ceiling inclusive.
  delay=$((RANDOM * (ceiling + 1) / 32768))
  echo "retry.sh: attempt ${attempt}/${attempts} failed (exit ${status}); retrying in ${delay}s (ceiling ${ceiling}s)" >&2
  "${sleeper}" "${delay}"
  attempt=$((attempt + 1))
done
