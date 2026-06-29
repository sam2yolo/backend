#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${SAM_ENV_NAME:-sam}"
CONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
SAM3_DIR="${SAM3_PROJECT_DIR:-/root/sam3}"
CHECKPOINT_URL="${SAM3_CHECKPOINT_URL:-https://drive.google.com/file/d/1U_SBWxdyRFx-519v_UQZh48cm4y4qLVm/view?usp=drive_link}"
CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$SAM3_DIR/hf/sam3.1_multiplex_mapped.pt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] Conda was not found at $CONDA_DIR." >&2
    echo "Run the main run.sh setup first or set MINICONDA_DIR." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[SAM] Creating Conda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.12
else
    echo "[SAM] Reusing Conda environment: $ENV_NAME"
fi

echo "[SAM] Installing runtime dependencies"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install \
    torch==2.10.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128
conda run -n "$ENV_NAME" python -m pip install \
    "setuptools<81" \
    einops \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    iopath \
    ftfy==6.1.1 \
    huggingface_hub \
    timm \
    "numpy>=1.26,<2" \
    pillow \
    psutil \
    pycocotools \
    gdown

if [[ ! -d "$SAM3_DIR/.git" ]]; then
    echo "[SAM] Cloning SAM3 into $SAM3_DIR"
    rm -rf "$SAM3_DIR"
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
else
    echo "[SAM] Reusing SAM3 checkout: $SAM3_DIR"
fi

echo "[SAM] Installing SAM3"
conda run -n "$ENV_NAME" python -m pip install -e "$SAM3_DIR"

echo "[SAM] Verifying SAM3 server imports"
conda run -n "$ENV_NAME" python -c \
    "from sam3 import build_sam3_image_model; from sam3.eval.postprocessors import PostProcessImage; from sam3.train.data.collator import collate_fn_api"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[SAM] Downloading checkpoint"
    archive="$SAM3_DIR/sam3.1.zip"
    mkdir -p "$SAM3_DIR/hf"
    conda run -n "$ENV_NAME" gdown "$CHECKPOINT_URL" -O "$archive"
    unzip -o "$archive" -d "$SAM3_DIR/hf"

    source_checkpoint="$(
        find "$SAM3_DIR/hf" -type f \
            \( -name 'sam3.1_multiplex.pt' -o -name 'sam3.1_multiplex_remapped.pt' \) \
            -print -quit
    )"
    if [[ -z "$source_checkpoint" ]]; then
        echo "[ERROR] SAM3 checkpoint was not found after extraction." >&2
        exit 1
    fi

    echo "[SAM] Remapping checkpoint"
    conda run -n "$ENV_NAME" python \
        "$SCRIPT_DIR/remap.py" \
        "$source_checkpoint" \
        "$CHECKPOINT_PATH"
    rm -f "$archive"
else
    echo "[SAM] Reusing checkpoint: $CHECKPOINT_PATH"
fi

echo "[SAM] Environment is ready"
printf 'SAM_ENV_NAME=%s\n' "$ENV_NAME"
printf 'SAM3_PROJECT_DIR=%s\n' "$SAM3_DIR"
printf 'SAM3_CHECKPOINT_PATH=%s\n' "$CHECKPOINT_PATH"
