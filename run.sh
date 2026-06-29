#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: $0 <room_id> <room_secret> [name]" >&2
    echo "[ERROR] room_id and room_secret are required." >&2
    exit 1
fi

ROOM_ID="$1"
ROOM_SECRET="$2"
NAME="${3:-samtoyolo-backend}"

apt update

# install utilities

apt install tmux git
git clone https://github.com/sam2yolo/backend.git

cd backend

APP_PORT=8000

# initiate frpc

cp frpc /bin
chmod +x /bin

BROKER=http://163.61.236.112:7001                                            
API_TOKEN='VKl7VeOSKDBLPheW7hdKyZQ5iq//B+1SSBXCkQpZ9HQ='                   
FRPS_TOKEN='kEfip7iduT8PpQ4PDsT1QCEtB0wI+nSENhcUjh7oBUM='

mkdir /root/scripts
cp tunnel /root/scripts

tmux new -d -s s1 -n tunnel "./quicktunnel --broker $BROKER --room-id $ROOM_ID --room-secret $ROOM_SECRET --api-token $API_TOKEN --cleanup --frps-token $FRPS_TOKEN $NAME $APP_PORT"

ENV_NAME="samtoyolo_conda_environment"
MINICONDA_DIR="$HOME/miniconda3"

# Check if conda exists
if ! command -v conda >/dev/null 2>&1; then
    echo "[INFO] Conda not found. Installing Miniconda..."

    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$ARCH" == "aarch64" ]]; then
        INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
    else
        echo "[ERROR] Unsupported architecture: $ARCH"
        exit 1
    fi

    wget -q "https://repo.anaconda.com/miniconda/${INSTALLER}" -O /tmp/miniconda.sh

    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f /tmp/miniconda.sh

    export PATH="$MINICONDA_DIR/bin:$PATH"
fi


conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# Initialize conda for non-interactive shell
eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[INFO] Creating conda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.9
else
    echo "[INFO] Environment already exists: $ENV_NAME"
fi

# Activate environment
conda activate "$ENV_NAME"

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
if [[ -f requirements.txt ]]; then
    echo "[INFO] Installing requirements..."
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found."
fi

echo "[INFO] Setup complete."

echo "[INFO] Setting up SAM environment..."
bash sam/sam_env.sh

SAM_ENV_NAME="${SAM_ENV_NAME:-sam}"
SAM3_PROJECT_DIR="${SAM3_PROJECT_DIR:-/root/sam3}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$SAM3_PROJECT_DIR/hf/sam3.1_multiplex_mapped.pt}"

echo "[INFO] Starting SAM server on port 8001..."
tmux kill-session -t sam-server 2>/dev/null || true
tmux new-session -d -s sam-server \
    "cd '$PWD' && \
     SAM3_PROJECT_DIR='$SAM3_PROJECT_DIR' \
     SAM3_CHECKPOINT_PATH='$SAM3_CHECKPOINT_PATH' \
     '$MINICONDA_DIR/bin/conda' run --no-capture-output -n '$SAM_ENV_NAME' \
     python sam/sam_fastapi_server.py"

echo "[INFO] Waiting for SAM server workers..."
for _ in $(seq 1 120); do
    health="$(curl -fsS http://127.0.0.1:8001/health 2>/dev/null || true)"
    if [[ "$health" == *'"status":"ready"'* ]]; then
        echo "[INFO] SAM server is ready."
        break
    fi
    sleep 5
done

health="$(curl -fsS http://127.0.0.1:8001/health 2>/dev/null || true)"
if [[ "$health" != *'"status":"ready"'* ]]; then
    echo "[ERROR] SAM server did not become ready." >&2
    tmux capture-pane -pt sam-server >&2 || true
    exit 1
fi

export MPLBACKEND=Agg

echo $PWD

python app.py
