#!/usr/bin/env bash
# Upload and run full DHLAB setup (you will type Gaspar password once or twice).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${DHLAB_SSH_HOST:-dhlab}"

if [[ "${1:-}" == "probe" ]]; then
  echo "→ Probing scratch paths on ${HOST} ..."
  ssh "${HOST}" 'bash -s' < "${ROOT}/scripts/dhlab_probe_scratch.sh"
  exit 0
fi

echo "→ Running dhlab_full_setup.sh on ${HOST} ..."
echo "  (optional: DHLAB_SCRATCH=/rcp-scratch/students/yijliu $0)"
ssh "${HOST}" 'bash -s' < "${ROOT}/scripts/dhlab_full_setup.sh"
