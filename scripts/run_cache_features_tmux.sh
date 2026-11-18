#!/usr/bin/env bash
set -euo pipefail
ROOT="/rcp-scratch/students/yijliu/ml-project2/ML_project_1"
export P2_DATA="/rcp-scratch/students/yijliu/ml-project2/data"
export HF_HOME="/rcp-scratch/students/yijliu/hf_cache"
export TORCH_HOME="/rcp-scratch/students/yijliu/hf_cache/torch"
export TMPDIR="/rcp-scratch/students/yijliu/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="/rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python"
mkdir -p "$P2_DATA/features"
cd "$ROOT"
exec "$PY" scripts/cache_features.py --workers 0 --batch-size 256 --shard-size 4096 "$@"