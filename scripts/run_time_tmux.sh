#!/usr/bin/env bash
# Train T1–T3 sequentially on a free GPU (not GPU 3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export P2_DATA="${P2_DATA:-/rcp-scratch/students/yijliu/ml-project2/data}"
export HF_HOME="${HF_HOME:-/rcp-scratch/students/yijliu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/rcp-scratch/students/yijliu/hf_cache/torch}"
export TMPDIR="${TMPDIR:-/rcp-scratch/students/yijliu/tmp}"
PY="${PY:-/rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
LOG="${P2_DATA}/logs/run_time_train.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
{
  echo "=== $(date -Is) time feats ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -c "from lib.time_feats import ensure_time_feats; ensure_time_feats()"
  echo "=== T0 ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_knn_time.py
  for cfg in configs/T1.yaml configs/T2.yaml configs/T3.yaml; do
    echo "=== train $cfg ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train.py --config "$cfg"
  done
  echo "=== done $(date -Is) ==="
} >>"$LOG" 2>&1