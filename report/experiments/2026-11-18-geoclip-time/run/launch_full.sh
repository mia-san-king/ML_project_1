#!/usr/bin/env bash
set -euo pipefail
BASE=/rcp-scratch/students/zhang7/ml-project2
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export P2_DATA="$BASE/data" HF_HOME="$BASE/hf_cache" TORCH_HOME="$BASE/hf_cache/torch" TMPDIR="$BASE/tmp"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$BASE/ML_project_1"
exec /rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python "$BASE/runs/2026-11-18-geoclip-time/run_pipeline.py" >> "$BASE/runs/2026-11-18-geoclip-time/full_console.log" 2>&1
