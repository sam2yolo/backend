#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${SAMTOYOLO_SAM3_ENV_NAME:-samtoyolo-model-sam3}"
PYTHON_VERSION="${SAMTOYOLO_SAM3_PYTHON:-3.12}"
CONDA_PREFIX_ROOT="${SAMTOYOLO_CONDA_INSTALL_PREFIX:-${HOME}/.samtoyolo/miniforge3}"
TORCH_INDEX_URL="${SAMTOYOLO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONDA_CHANNEL="${SAMTOYOLO_CONDA_CHANNEL:-conda-forge}"

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

install_miniforge() {
  local installer_dir="${REPO_ROOT}/.samtoyolo/installers"
  local installer_path="${installer_dir}/Miniforge3-Linux-x86_64.sh"
  mkdir -p "${installer_dir}" "$(dirname "${CONDA_PREFIX_ROOT}")"
  if [[ ! -f "${installer_path}" ]]; then
    curl -L \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
      -o "${installer_path}"
  fi
  bash "${installer_path}" -b -p "${CONDA_PREFIX_ROOT}"
}

if ! CONDA_BIN="$(find_conda)"; then
  install_miniforge
  CONDA_BIN="$(find_conda)"
fi

if [[ "${SAMTOYOLO_CONDA_AUTO_ACCEPT_TOS:-1}" != "0" ]]; then
  "${CONDA_BIN}" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  "${CONDA_BIN}" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -y \
    --override-channels \
    -c "${CONDA_CHANNEL}" \
    --strict-channel-priority \
    -n "${ENV_NAME}" \
    "python=${PYTHON_VERSION}" \
    pip
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url "${TORCH_INDEX_URL}"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  -r "${SCRIPT_DIR}/requirements.txt"

cat <<EOF
SAM 3.1 model server environment is ready.
Environment: ${ENV_NAME}
Run: ${SCRIPT_DIR}/run.sh
EOF
