#!/usr/bin/env bash
set -euo pipefail

read -r weekday local_time <<< "$(TZ=Africa/Cairo date '+%w %H%M')"
if [[ -n ${EGX_SCHEDULE:-} ]] && (( weekday >= 5 )); then
  exit 0
fi

case ${EGX_SCHEDULE:-} in
  '50 8 * * 0-4')
    if (( 10#$local_time >= 1000 )); then
      echo 'Morning run reached the runner after the forecast window.'
      exit 0
    fi
    exec timeout --signal=INT --kill-after=10s --preserve-status 70m python -u app.py start --until 10:00
    ;;
  '32 10 * * 0-4')
    if (( 10#$local_time >= 1025 && 10#$local_time < 1515 )); then
      exec timeout --signal=INT --kill-after=10s --preserve-status 3m python -u app.py start --once
    fi
    ;;
  '35 15 * * 0-4')
    if (( 10#$local_time >= 1515 )); then
      exec timeout --signal=INT --kill-after=10s --preserve-status 6m python -u app.py start --once
    fi
    ;;
  '')
    exec timeout --signal=INT --kill-after=10s --preserve-status 6m python -u app.py start --once
    ;;
  *)
    echo 'Unhandled EGX schedule.' >&2
    exit 2
    ;;
esac
