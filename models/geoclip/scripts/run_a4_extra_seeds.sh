#!/usr/bin/env bash
set -euo pipefail
ROOT="/rcp-scratch/students/yijliu/ml-project2/ML_project_1"
export P2_DATA="/rcp-scratch/students/yijliu/ml-project2/data"
export HF_HOME="/rcp-scratch/students/yijliu/hf_cache"
export TORCH_HOME="/rcp-scratch/students/yijliu/hf_cache/torch"
export TMPDIR="/rcp-scratch/students/yijliu/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="/rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python"
cd "$ROOT"
for cfg in models/geoclip/configs/A4_tau5_s1.yaml models/geoclip/configs/A4_tau5_s2.yaml; do
  echo "=== $cfg ===" | tee -a "$P2_DATA/results/s10.log"
  "$PY" models/geoclip/scripts/train.py --config "$cfg" 2>&1 | tee -a "$P2_DATA/results/s10.log"
done
"$PY" models/geoclip/scripts/summarize_a4_multiseed.py | tee -a "$P2_DATA/results/s10.log"