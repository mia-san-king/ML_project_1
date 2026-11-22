#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/rcp-scratch/students/zhang7/ml-project2/hf_cache TORCH_HOME=/rcp-scratch/students/zhang7/ml-project2/hf_cache/torch TMPDIR=/rcp-scratch/students/zhang7/ml-project2/tmp
exec /rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python -u run_t2.py >> console.log 2>&1
