from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .model_sources import (
    DEFAULT_SAM3_MODEL_FILENAME,
    DEFAULT_SAM3_MODEL_URL,
    google_drive_download_url,
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(slots=True)
class Settings:
    """Runtime configuration, intentionally environment driven for cloud use."""

    project_root: Path = Path(os.getenv("SAMTOYOLO_PROJECT_ROOT", "projects")).resolve()
    mode: str = os.getenv("SAMTOYOLO_MODE", "local")
    host: str = os.getenv("SAMTOYOLO_HOST", "0.0.0.0")
    port: int = _int_env("SAMTOYOLO_PORT", 8000)
    server_name: str = os.getenv("SAMTOYOLO_SERVER_NAME", socket.gethostname())
    gpu_workers: int = _int_env("SAMTOYOLO_GPU_WORKERS", 0)
    allow_stub_ml: bool = _bool_env("SAMTOYOLO_ALLOW_STUB_ML", True)
    sam3_model_url: str = os.getenv("SAMTOYOLO_SAM3_MODEL_URL", DEFAULT_SAM3_MODEL_URL)
    sam3_model_filename: str = os.getenv(
        "SAMTOYOLO_SAM3_MODEL_FILENAME", DEFAULT_SAM3_MODEL_FILENAME
    )
    sam3_model_server_url: str = os.getenv(
        "SAMTOYOLO_SAM3_SERVER_URL", "ws://127.0.0.1:8101/v1/ws"
    )
    instance_ttl_seconds: int = _int_env("SAMTOYOLO_INSTANCE_TTL_SECONDS", 42_600)
    expiry_notice_seconds: int = _int_env("SAMTOYOLO_EXPIRY_NOTICE_SECONDS", 900)
    public_base_url: str | None = os.getenv("SAMTOYOLO_PUBLIC_BASE_URL")

    tunnelbroker_url: str | None = os.getenv("TUNNELBROKER_URL")
    tunnelbroker_group: str | None = os.getenv("TUNNELBROKER_GROUP")
    tunnelbroker_group_token: str | None = os.getenv("TUNNELBROKER_GROUP_TOKEN")
    peer_secret: str | None = os.getenv("TUNNELBROKER_PEER_SECRET")
    cloudflared_path: str = os.getenv("CLOUDFLARED_PATH", "cloudflared")
    tunnel_heartbeat_seconds: int = _int_env("TUNNEL_HEARTBEAT_SECONDS", 300)

    @property
    def is_remote(self) -> bool:
        return self.mode.lower() == "remote"

    @property
    def http_bind_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def sam3_model_download_url(self) -> str:
        return google_drive_download_url(self.sam3_model_url)

    @property
    def model_cache_dir(self) -> Path:
        return Path(os.getenv("SAMTOYOLO_MODEL_CACHE_DIR", self.project_root / "_models"))

    def resolved_gpu_workers(self) -> int:
        if self.gpu_workers > 0:
            return self.gpu_workers
        try:
            import torch  # type: ignore

            count = torch.cuda.device_count()
            return max(1, int(count))
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            count = len([line for line in result.stdout.splitlines() if line.strip()])
            return max(1, count)
        except Exception:
            return 1
