#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
run_dir=$PWD
project=/rcp-scratch/students/zhang7/ml-project2
py=/rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python
stage=$1
export CUDA_VISIBLE_DEVICES="${2:-}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="$project/hf_cache" TORCH_HOME="$project/hf_cache/torch" TMPDIR="$project/tmp"
export P2_DATA="$run_dir/augmented"
exec >> "$stage.log" 2>&1
trap 'printf "%s\n" "$?" > "$stage.exit"' EXIT
entry="$run_dir/source/scripts/augmentation/train_models.py"
paths=(--data-root "$P2_DATA" --output-root "$run_dir/models" --threads 2)
case "$stage" in
    T1)
        "$py" "$entry" train --model T1 --device cuda "${paths[@]}"
        "$py" "$entry" predict --model T1 --device cuda "${paths[@]}"
        ;;
    T3)
        "$py" "$entry" train --model T0 --device cuda "${paths[@]}"
        "$py" "$entry" t3 --device cpu --query-chunk 256 --data-root "$P2_DATA" --output-root "$run_dir/models" --threads 4
        ;;
    T4)
        nice -n 5 "$py" "$entry" t4 "${paths[@]}"
        ;;
    *) exit 2 ;;
esac
