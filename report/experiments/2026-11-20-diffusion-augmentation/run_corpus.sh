#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
run_dir=$PWD
project=/rcp-scratch/students/zhang7/ml-project2
py=/rcp-scratch/students/yijliu/.conda/envs/ml-p2/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME="$project/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TMPDIR="$project/tmp" CUDA_VISIBLE_DEVICES=1
exec >> corpus.log 2>&1
trap 'printf "%s\n" "$?" > corpus.exit' EXIT
echo "Continuing from the visually checked pilot"
"$run_dir/diffenv/bin/python" -u source/scripts/augmentation/generate.py --data-root /rcp-scratch/students/zhang7/ml-project2/data --split-csv /rcp-scratch/students/zhang7/ml-project2/ML_project_1/data/splits/main.csv.gz --images-root /rcp-scratch/students/yijliu/ml-project2/data/images --model-dir /rcp-scratch/students/zhang7/ml-project2/runs/2026-11-20-diffusion-augmentation/sdxl-fp16 --model-revision 462165984030d82259a11f4367a4eed129e94a7b --store /rcp-scratch/students/zhang7/ml-project2/runs/2026-11-20-diffusion-augmentation/synthetic_store --variant fp16 --seed 0 --strength 0.18 --steps 30 --guidance-scale 3 --max-side 768 --device cuda --count 512 --resume
"$py" -u audit_images.py --store /rcp-scratch/students/zhang7/ml-project2/runs/2026-11-20-diffusion-augmentation/synthetic_store --images-root /rcp-scratch/students/yijliu/ml-project2/data/images --data-root /rcp-scratch/students/zhang7/ml-project2/data --split-csv /rcp-scratch/students/zhang7/ml-project2/ML_project_1/data/splits/main.csv.gz --preview preview.jpg --report image-qa.json
"$py" -u source/scripts/augmentation/build_dataset.py --source-data-root /rcp-scratch/students/zhang7/ml-project2/data --source-split-csv /rcp-scratch/students/zhang7/ml-project2/ML_project_1/data/splits/main.csv.gz --store /rcp-scratch/students/zhang7/ml-project2/runs/2026-11-20-diffusion-augmentation/synthetic_store --output-data-root /rcp-scratch/students/zhang7/ml-project2/runs/2026-11-20-diffusion-augmentation/augmented --images-root /rcp-scratch/students/yijliu/ml-project2/data/images --clip-model /rcp-scratch/students/yijliu/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 --device cuda --batch-size 32
"$py" -u source/scripts/augmentation/train_models.py check --data-root "$run_dir/augmented" --output-root "$run_dir/models" --threads 2
