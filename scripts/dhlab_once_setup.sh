#!/usr/bin/env bash
# Run ONCE on dhlabsrv1 after: ssh dhlab
# Creates scratch layout + conda dirs (data/env on scratch, not home).

set -euo pipefail

USER_NAME="${USER:-yijliu}"
BASE=""
for candidate in "/rcp-scratch/students/${USER_NAME}" "/rcp_scratch/students/${USER_NAME}" "/scratch/students/${USER_NAME}" "/scratch/${USER_NAME}"; do
  if [[ -n "${DHLAB_SCRATCH:-}" ]]; then
    BASE="${DHLAB_SCRATCH}"
    break
  fi
  if [[ -d "$candidate" && -w "$candidate" ]]; then
    BASE="$candidate"
    break
  fi
  parent="$(dirname "$candidate")"
  if [[ -d "$parent" && -w "$parent" ]] && mkdir -p "$candidate" 2>/dev/null && [[ -w "$candidate" ]]; then
    BASE="$candidate"
    break
  fi
done

if [[ -z "$BASE" ]]; then
  echo "Could not find or create scratch. Ask supervisor for the correct path."
  echo "Tried: /scratch/\$USER, /rcp_scratch/students/\$USER, /scratch/students/\$USER"
  exit 1
fi

mkdir -p "${BASE}/ml-project2" \
  "${BASE}/ml-project2/data" \
  "${BASE}/.conda/envs" \
  "${BASE}/.conda/pkgs" \
  "${BASE}/hf_cache"

if command -v conda >/dev/null 2>&1; then
  conda config --add envs_dirs "${BASE}/.conda/envs" 2>/dev/null || true
  conda config --add pkgs_dirs "${BASE}/.conda/pkgs" 2>/dev/null || true
  echo "Conda envs_dirs / pkgs_dirs pointed at ${BASE}/.conda/"
else
  echo "conda not in PATH — load module or install, then re-run this script."
fi

cat <<EOF

Scratch base: ${BASE}
Project dir:  ${BASE}/ml-project2

Next (GeoCLIP env example):
  conda create --prefix ${BASE}/.conda/envs/ml-p2 python=3.11 -y
  conda activate ${BASE}/.conda/envs/ml-p2
  pip install torch torchvision geoclip

GPU (max 2 of 0,1,2 per lab policy):
  nvidia-smi
  export CUDA_VISIBLE_DEVICES=0

HF / torch cache on scratch (add to ~/.bashrc):
  export HF_HOME=${BASE}/hf_cache
  export TORCH_HOME=${BASE}/hf_cache/torch

EOF
