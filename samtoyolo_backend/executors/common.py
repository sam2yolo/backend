from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import events
from ..records import new_id, utc_now
from ..tasks import TaskContext


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".dav"}


async def checkpoint_client_fetch(ctx: TaskContext, path: Path) -> None:
    relpath = ctx.store.relative_to_project(ctx.task.project_id, path)
    request_id = new_id("client_request")
    ctx.store.mutate_session(
        ctx.task.project_id,
        lambda session: session.setdefault("client_requests", {}).__setitem__(
            request_id,
            {
                "request_id": request_id,
                "type": "sync_file",
                "path": relpath,
                "task_id": ctx.task.task_id,
                "created_at": utc_now(),
                "status": "pending",
            },
        ),
    )
    await events.notify_client_ask(
        ctx.connections,
        project_id=ctx.task.project_id,
        request_id=request_id,
        request_type="sync_file",
        data={"task_id": ctx.task.task_id, "path": relpath},
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def zip_directory(source_dir: Path, target_zip: Path) -> None:
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source_dir))


def normalise_google_drive_url(url: str) -> str:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return url
    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    query_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_id:
        return f"https://drive.google.com/uc?export=download&id={query_id}"
    return url


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "download.bin"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not create unique path for {path}")
