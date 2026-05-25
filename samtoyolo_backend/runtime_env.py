from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


BOOTSTRAP_ENV_VAR = "SAMTOYOLO_CONDA_BOOTSTRAP"
BOOTSTRAPPED_ENV_VAR = "SAMTOYOLO_CONDA_BOOTSTRAPPED"
DEFAULT_ENV_NAME = "samtoyolo-backend"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_CONDA_CHANNEL = "conda-forge"
ANACONDA_TOS_CHANNELS = (
    "https://repo.anaconda.com/pkgs/main",
    "https://repo.anaconda.com/pkgs/r",
)
MINIFORGE_URLS = {
    ("Linux", "x86_64"): (
        "https://github.com/conda-forge/miniforge/releases/latest/download/"
        "Miniforge3-Linux-x86_64.sh"
    ),
    ("Linux", "aarch64"): (
        "https://github.com/conda-forge/miniforge/releases/latest/download/"
        "Miniforge3-Linux-aarch64.sh"
    ),
}


def ensure_runtime_environment() -> None:
    """Create and enter the SAM-to-YOLO conda environment before app imports."""

    if not _bool_env(BOOTSTRAP_ENV_VAR, True):
        return

    repo_root = Path(__file__).resolve().parents[1]
    env_name = os.getenv("SAMTOYOLO_CONDA_ENV_NAME", DEFAULT_ENV_NAME)
    conda = _find_or_install_conda(repo_root)
    env_prefix = _ensure_conda_env(conda, env_name)
    _ensure_env_packages(conda, env_name, env_prefix, repo_root)

    current_prefix = Path(sys.prefix).resolve()
    if current_prefix == env_prefix.resolve():
        os.environ[BOOTSTRAPPED_ENV_VAR] = "1"
        return

    os.environ[BOOTSTRAPPED_ENV_VAR] = "1"
    python = _env_python(env_prefix)
    os.execv(str(python), _reexec_args(python))


def runtime_environment_status() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    env_name = os.getenv("SAMTOYOLO_CONDA_ENV_NAME", DEFAULT_ENV_NAME)
    conda = _find_conda()
    env_prefix = _conda_env_prefix(conda, env_name) if conda else None
    requirements = _requirements_path(repo_root)
    return {
        "bootstrap_enabled": _bool_env(BOOTSTRAP_ENV_VAR, True),
        "bootstrapped": os.getenv(BOOTSTRAPPED_ENV_VAR) == "1",
        "conda_executable": str(conda) if conda else None,
        "conda_env_name": env_name,
        "conda_env_prefix": str(env_prefix) if env_prefix else None,
        "running_prefix": sys.prefix,
        "in_conda_env": bool(env_prefix and Path(sys.prefix).resolve() == env_prefix),
        "requirements_file": str(requirements),
        "requirements_hash": _file_hash(requirements),
    }


def _find_or_install_conda(repo_root: Path) -> Path:
    conda = _find_conda()
    if conda:
        return conda

    install_prefix = Path(
        os.getenv("SAMTOYOLO_CONDA_INSTALL_PREFIX", "~/.samtoyolo/miniforge3")
    ).expanduser()
    conda = install_prefix / "bin" / "conda"
    if conda.exists():
        return conda

    system = platform.system()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    installer_url = os.getenv("SAMTOYOLO_CONDA_INSTALLER_URL") or MINIFORGE_URLS.get(
        (system, machine)
    )
    if not installer_url:
        raise RuntimeError(
            "conda is not installed and automatic Miniforge install is only "
            f"configured for Linux x86_64/aarch64, not {system} {machine}"
        )

    installer_dir = repo_root / ".samtoyolo" / "installers"
    installer_dir.mkdir(parents=True, exist_ok=True)
    installer_path = installer_dir / Path(installer_url).name
    if not installer_path.exists():
        print(f"[samtoyolo] downloading conda installer: {installer_url}", flush=True)
        urllib.request.urlretrieve(installer_url, installer_path)

    print(f"[samtoyolo] installing conda to {install_prefix}", flush=True)
    _run(["bash", str(installer_path), "-b", "-p", str(install_prefix)])
    return conda


def _ensure_conda_env(conda: Path, env_name: str) -> Path:
    prefix = _conda_env_prefix(conda, env_name)
    if prefix:
        return prefix

    python_version = os.getenv("SAMTOYOLO_CONDA_PYTHON", DEFAULT_PYTHON_VERSION)
    channel = os.getenv("SAMTOYOLO_CONDA_CHANNEL", DEFAULT_CONDA_CHANNEL)
    _accept_conda_tos_if_available(conda)
    print(
        (
            f"[samtoyolo] creating conda env {env_name} with "
            f"python={python_version} from {channel}"
        ),
        flush=True,
    )
    _run(
        [
            str(conda),
            "create",
            "-y",
            "--override-channels",
            "-c",
            channel,
            "--strict-channel-priority",
            "-n",
            env_name,
            f"python={python_version}",
            "pip",
        ]
    )
    prefix = _conda_env_prefix(conda, env_name)
    if not prefix:
        raise RuntimeError(f"conda env was created but not found: {env_name}")
    return prefix


def _ensure_env_packages(
    conda: Path, env_name: str, env_prefix: Path, repo_root: Path
) -> None:
    marker_path = env_prefix / ".samtoyolo_env.json"
    marker = _read_json(marker_path)
    wanted = {
        "requirements_file": str(_requirements_path(repo_root)),
        "requirements_hash": _file_hash(_requirements_path(repo_root)),
        "torch_index_url": os.getenv("SAMTOYOLO_TORCH_INDEX_URL", DEFAULT_TORCH_INDEX_URL),
        "install_torch": _bool_env("SAMTOYOLO_INSTALL_TORCH", False),
    }
    if all(marker.get(key) == value for key, value in wanted.items()):
        return

    print(f"[samtoyolo] installing backend dependencies into {env_name}", flush=True)
    _run_conda_python(conda, env_name, ["-m", "pip", "install", "--upgrade", "pip"])
    if wanted["install_torch"] and wanted["torch_index_url"]:
        _run_conda_python(
            conda,
            env_name,
            [
                "-m",
                "pip",
                "install",
                "torch==2.10.0",
                "torchvision==0.25.0",
                "--index-url",
                str(wanted["torch_index_url"]),
            ],
        )
    _run_conda_python(
        conda,
        env_name,
        ["-m", "pip", "install", "-r", str(_requirements_path(repo_root))],
    )
    marker_path.write_text(json.dumps(wanted, indent=2, sort_keys=True) + "\n")


def _find_conda() -> Path | None:
    candidates = [
        os.getenv("CONDA_EXE"),
        shutil.which("conda"),
        str(Path("~/miniconda3/bin/conda").expanduser()),
        str(Path("~/miniforge3/bin/conda").expanduser()),
        str(Path("~/.samtoyolo/miniforge3/bin/conda").expanduser()),
        "/opt/conda/bin/conda",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    return None


def _conda_env_prefix(conda: Path, env_name: str) -> Path | None:
    result = _run([str(conda), "env", "list", "--json"], capture=True)
    envs = json.loads(result.stdout or "{}").get("envs", [])
    for env in envs:
        prefix = Path(env).resolve()
        if prefix.name == env_name:
            return prefix
    return None


def _run_conda_python(conda: Path, env_name: str, args: list[str]) -> None:
    _run([str(conda), "run", "-n", env_name, "python", *args])


def _accept_conda_tos_if_available(conda: Path) -> None:
    if not _bool_env("SAMTOYOLO_CONDA_AUTO_ACCEPT_TOS", True):
        return
    for channel in ANACONDA_TOS_CHANNELS:
        result = subprocess.run(
            [
                str(conda),
                "tos",
                "accept",
                "--override-channels",
                "--channel",
                channel,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode == 0:
            print(f"[samtoyolo] accepted conda TOS for {channel}", flush=True)


def _env_python(env_prefix: Path) -> Path:
    name = "python.exe" if os.name == "nt" else "python"
    subdir = "" if os.name == "nt" else "bin"
    return env_prefix / subdir / name if subdir else env_prefix / name


def _reexec_args(python: Path) -> list[str]:
    original = getattr(sys, "orig_argv", None)
    if original and len(original) > 1 and original[1] in {"-c", "-m"}:
        return [str(python), *original[1:]]
    return [str(python), *sys.argv]


def _requirements_path(repo_root: Path) -> Path:
    value = os.getenv("SAMTOYOLO_REQUIREMENTS_FILE")
    return Path(value).expanduser() if value else repo_root / "requirements.txt"


def _file_hash(requirements: Path) -> str:
    digest = hashlib.sha256()
    digest.update(requirements.read_bytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {"check": True, "text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(args, **kwargs)
