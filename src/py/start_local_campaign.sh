#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin="$repo_root/.venv/bin/python"
campaign_name="${1:-cold-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ $# -gt 0 ]]; then shift; fi
case "$campaign_name" in
    ''|.|..|*/*) printf 'Use a single directory name for the campaign\n' >&2; exit 1 ;;
esac
"$python_bin" -c 'import numpy, pandas, scipy, numba, pint, matplotlib, pyarrow, psutil'
mkdir -p "$repo_root/data/campaigns"
if [[ -e "$repo_root/data/campaigns/$campaign_name" || -e "$repo_root/data/campaigns/$campaign_name.launch.log" ]]; then
    printf 'Campaign already exists: %s\n' "$campaign_name" >&2
    exit 1
fi
nohup "$python_bin" -u "$repo_root/src/py/run_local_campaign.py" \
    --name "$campaign_name" "$@" \
    > "$repo_root/data/campaigns/$campaign_name.launch.log" 2>&1 < /dev/null &
campaign_pid=$!
printf '%s\n' "$campaign_pid" > "$repo_root/data/campaigns/$campaign_name.launch.pid"
printf 'PID: %s\nLog: %s\n' "$campaign_pid" "$repo_root/data/campaigns/$campaign_name.launch.log"
