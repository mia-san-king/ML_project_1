#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run_dir=$PWD
project=/rcp-scratch/students/zhang7/ml-project2
py=/rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python
exec >> "$run_dir/console.log" 2>&1
trap 'printf "%s\n" "$?" > "$run_dir/exit_code"' EXIT
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$project/tmp" HF_HOME="$project/hf_cache" TORCH_HOME="$project/hf_cache/torch" P2_DATA="$project/data"
test -s "$run_dir/input_manifest.json"
cd "$run_dir/source"
sha256sum -c "$run_dir/source.sha256"
args=(--checkpoint "$project/data/ckpt/zhang7_T1_dual_time_available_s0_20261118.pt" --data-root "$P2_DATA" --split-csv "$project/ML_project_1/data/splits/main.csv.gz" --output-dir "$run_dir/results" --threads 8 --query-chunk 128 --reference-chunk 8192 --device cpu)
nice -n 10 "$py" -u -m models.geoclip_time.scripts.rerank_locations "${args[@]}"
