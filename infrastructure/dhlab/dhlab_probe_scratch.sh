#!/usr/bin/env bash
# Run ON dhlabsrv1: bash dhlab_probe_scratch.sh
# Finds a writable project/scratch path (no changes except a .write_test file).

set -u
USER_NAME="${USER:-$(whoami)}"

echo "=== user: ${USER_NAME} @ $(hostname) ==="
echo
echo "=== df -h (look for scratch / rcp) ==="
df -h 2>/dev/null | grep -E 'Filesystem|scratch|rcp|home|/$' || df -h

echo
echo "=== top-level listings ==="
for d in /scratch /scratch/students /rcp_scratch /rcp-scratch /rcp_scratch/students; do
  if [[ -e "$d" ]]; then
    ls -ld "$d" 2>/dev/null || true
    ls -la "$d" 2>/dev/null | head -8 || true
  else
    echo "(missing) $d"
  fi
done

echo
echo "=== writable candidates ==="
candidates=(
  "/rcp-scratch/students/${USER_NAME}"
  "/rcp_scratch/students/${USER_NAME}"
  "/scratch/students/${USER_NAME}"
  "/scratch/${USER_NAME}"
  "${HOME}/scratch"
)
for c in "${candidates[@]}"; do
  if [[ -d "$c" && -w "$c" ]]; then
    echo "OK  exists+writable  $c"
    continue
  fi
  parent="$(dirname "$c")"
  if [[ -d "$parent" && -w "$parent" ]]; then
    if mkdir -p "$c" 2>/dev/null && [[ -w "$c" ]]; then
      echo "OK  created         $c"
    else
      echo "NO  cannot create   $c (parent writable?)"
    fi
  elif [[ -d "$c" ]]; then
    echo "NO  not writable     $c"
  else
    echo "--  missing          $c"
  fi
done

echo
echo "If all NO: ask supervisor to create your folder (often /rcp_scratch/students/${USER_NAME}"
echo "or tell you the path). Teammate can run:  cd \\$(conda info --base 2>/dev/null); pwd; ls -la /rcp_scratch/students 2>/dev/null"
