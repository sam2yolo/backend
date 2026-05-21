#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${SAMTOYOLO_SAM3_ENV_NAME:-samtoyolo-model-sam3}"
CONDA_PREFIX_ROOT="${SAMTOYOLO_CONDA_INSTALL_PREFIX:-${HOME}/.samtoyolo/miniforge3}"

if [[ "${SAMTOYOLO_SKIP_MODEL_SERVER_SETUP:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/setup.sh"
fi

CONDA_BIN="${CONDA_EXE:-${CONDA_PREFIX_ROOT}/bin/conda}"
if [[ ! -x "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(command -v conda)"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
  python -m samtoyolo_model_servers.sam3.server
