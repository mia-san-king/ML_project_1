#!/usr/bin/env bash
# Idempotent DHLAB bootstrap for CS-433 Project 2 (GeoCLIP / SMAPshot).
# Run on dhlabsrv1: bash dhlab_full_setup.sh
# Or from Mac: ssh yijliu@dhlabsrv1.epfl.ch 'bash -s' < scripts/dhlab_full_setup.sh

set -euo pipefail

USER_NAME="${USER:-yijliu}"
REPO_URL="${DHLAB_REPO_URL:-https://github.com/mia-san-king/ML_project_1.git}"
REPO_BRANCH="${DHLAB_REPO_BRANCH:-project_2_Yijun_Liu}"
ENV_NAME="ml-p2"
MARK_BEGIN="# >>> DHLAB ml-p2 >>>"
MARK_END="# <<< DHLAB ml-p2 <<<"

log() { printf '\n[dhlab-setup] %s\n' "$*"; }

try_scratch_path() {
  local c="$1"
  if [[ -d "$c" && -w "$c" ]]; then
    echo "$c"
    return 0
  fi
  local parent
  parent="$(dirname "$c")"
  # Only create a leaf folder when the parent already exists and we can write there.
  if [[ -d "$parent" && -w "$parent" ]]; then
    if mkdir -p "$c" 2>/dev/null && [[ -d "$c" && -w "$c" ]]; then
      echo "$c"
      return 0
    fi
  fi
  return 1
}

find_scratch() {
  if [[ -n "${DHLAB_SCRATCH:-}" ]]; then
    try_scratch_path "${DHLAB_SCRATCH}" && return 0
  fi
  local c
  # Course README uses rcp_scratch; How-To says /scratch/students (often admin-created).
  for c in \
    "/rcp-scratch/students/${USER_NAME}" \
    "/rcp_scratch/students/${USER_NAME}" \
    "/scratch/students/${USER_NAME}" \
    "/scratch/${USER_NAME}"; do
    try_scratch_path "$c" && return 0
  done
  return 1
}

SCRATCH="$(find_scratch)" || {
  log "ERROR: no writable scratch folder."
  log "On the server run:  bash -s < dhlab_probe_scratch.sh   (or ask teammate for their path)"
  log "Or re-run with:     DHLAB_SCRATCH=/rcp-scratch/students/yijliu bash ..."
  log "Supervisor may need to create /rcp-scratch/students/${USER_NAME}"
  exit 1
}
log "Scratch: ${SCRATCH}"

mkdir -p "${SCRATCH}/ml-project2/data" \
  "${SCRATCH}/.conda/envs" \
  "${SCRATCH}/.conda/pkgs" \
  "${SCRATCH}/hf_cache" \
  "${SCRATCH}/tmp"

export TMPDIR="${SCRATCH}/tmp"

ensure_conda() {
  local prefix="${SCRATCH}/miniconda3"
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f "${prefix}/bin/conda" ]]; then
    # shellcheck source=/dev/null
    source "${prefix}/etc/profile.d/conda.sh"
    return 0
  fi
  log "Installing Miniconda under ${prefix} (~2 min)..."
  INSTALLER="${SCRATCH}/tmp/Miniconda3-latest-Linux-x86_64.sh"
  if [[ ! -f "$INSTALLER" ]]; then
    if command -v wget >/dev/null 2>&1; then
      wget -q -O "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    else
      log "ERROR: need wget or curl to install Miniconda"
      exit 1
    fi
  fi
  if [[ -d "$prefix" ]]; then
    # Empty dir from a previous partial run, or update in place.
    bash "$INSTALLER" -u -b -p "$prefix"
  else
    bash "$INSTALLER" -b -p "$prefix"
  fi
  # shellcheck source=/dev/null
  source "${prefix}/etc/profile.d/conda.sh"
}

ensure_conda
conda config --add envs_dirs "${SCRATCH}/.conda/envs" 2>/dev/null || true
conda config --add pkgs_dirs "${SCRATCH}/.conda/pkgs" 2>/dev/null || true

# Non-interactive SSH installs need explicit ToS accept (Conda 25+).
if conda tos --help >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

ENV_PREFIX="${SCRATCH}/.conda/envs/${ENV_NAME}"
if [[ ! -d "${ENV_PREFIX}/bin" ]]; then
  log "Creating conda env ${ENV_NAME} (python 3.11)..."
  conda create --prefix "${ENV_PREFIX}" python=3.11 pip -y \
    || conda create --prefix "${ENV_PREFIX}" --override-channels -c conda-forge python=3.11 pip -y
else
  log "Conda env exists: ${ENV_PREFIX}"
fi

# shellcheck source=/dev/null
source "${SCRATCH}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${ENV_PREFIX}"

log "Installing PyTorch + GeoCLIP stack (may take several minutes)..."
pip install -q --upgrade pip
# CUDA wheels; RTX 8000 + driver 580 on dhlabsrv1
pip install -q torch torchvision
pip install -q geoclip jupyter ipykernel tqdm pillow huggingface_hub pandas numpy matplotlib

if [[ ! -d "${SCRATCH}/ml-project2/ML_project_1/.git" ]]; then
  log "Cloning project repo..."
  git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${SCRATCH}/ml-project2/ML_project_1" \
    || git clone --depth 1 "${REPO_URL}" "${SCRATCH}/ml-project2/ML_project_1"
else
  log "Repo already at ${SCRATCH}/ml-project2/ML_project_1"
fi

python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME} @ DHLAB)" 2>/dev/null || true

BASHRC="${HOME}/.bashrc"
if ! grep -qF "${MARK_BEGIN}" "$BASHRC" 2>/dev/null; then
  log "Appending shell helpers to ${BASHRC}"
  cat >>"$BASHRC" <<EOF

${MARK_BEGIN}
export DHLAB_SCRATCH="${SCRATCH}"
export DHLAB_PROJECT="${SCRATCH}/ml-project2/ML_project_1"
export HF_HOME="${SCRATCH}/hf_cache"
export TORCH_HOME="${SCRATCH}/hf_cache/torch"
export TMPDIR="${SCRATCH}/tmp"
# GPU 3 often shared with lab services; use at most 2 GPUs (policy).
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -f "${SCRATCH}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${SCRATCH}/miniconda3/etc/profile.d/conda.sh"
fi
ml-p2() {
  conda activate "${ENV_PREFIX}"
  cd "\${DHLAB_PROJECT}"
}
cdp2() { cd "\${DHLAB_PROJECT}"; }
${MARK_END}
EOF
fi

QUICK="${HOME}/DHLAB_QUICKSTART.txt"
cat >"$QUICK" <<EOF
DHLAB ready for Project 2 (${USER_NAME}@$(hostname -f))

Scratch:    ${SCRATCH}
Project:    ${SCRATCH}/ml-project2/ML_project_1
Conda env:  ${ENV_PREFIX}

After login:
  source ~/.bashrc    # or open a new SSH session
  ml-p2               # activate env + cd to project
  nvidia-smi
  python -c "import torch; from geoclip import GeoCLIP; print('cuda', torch.cuda.device_count())"

Put datasets under: ${SCRATCH}/ml-project2/data

Docs: ${SCRATCH}/ml-project2/ML_project_1/scripts/dhlab_connect.md (after sync)
EOF

log "Smoke test..."
python <<'PY'
import torch
from geoclip import GeoCLIP
n = torch.cuda.device_count()
print("torch", torch.__version__, "gpus", n, "cuda_ok", torch.cuda.is_available())
if n:
    print("device0", torch.cuda.get_device_name(0))
GeoCLIP()
print("GeoCLIP import OK")
PY

log "Done. Read: ${QUICK}"
log "New shell: ssh dhlab  →  ml-p2"
