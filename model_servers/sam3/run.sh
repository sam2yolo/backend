#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${SAMTOYOLO_SAM3_ENV_NAME:-samtoyolo-model-sam3}"
CONDA_PREFIX_ROOT="${SAMTOYOLO_CONDA_INSTALL_PREFIX:-${HOME}/.samtoyolo/miniforge3}"

if [[ "${SAMTOYOLO_SKIP_MODEL_SERVER_SETUP:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/setup.sh"
fi

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  for candidate in \
    "${CONDA_PREFIX_ROOT}/bin/conda" \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
  python -m samtoyolo_model_servers.sam3.server
