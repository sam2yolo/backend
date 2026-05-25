from __future__ import annotations

import os
import platform
import shutil
import stat
import urllib.request
from pathlib import Path


def ensure_cloudflared_path(configured_path: str) -> str:
    existing = shutil.which(configured_path)
    if existing:
        return existing

    configured = Path(configured_path).expanduser()
    if configured.is_file():
        return str(configured)

    target = Path(os.getenv("SAMTOYOLO_BIN_DIR", "~/.samtoyolo/bin")).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    binary = target / "cloudflared"
    if binary.is_file():
        return str(binary)

    url = _download_url()
    tmp = binary.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp)
    mode = tmp.stat().st_mode
    tmp.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp.replace(binary)
    return str(binary)


def _download_url() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux":
        raise RuntimeError(
            "cloudflared auto-install currently supports Linux only; set "
            "CLOUDFLARED_PATH to an existing binary"
        )
    arch = "arm64" if machine in {"aarch64", "arm64"} else "amd64"
    return (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{arch}"
    )
