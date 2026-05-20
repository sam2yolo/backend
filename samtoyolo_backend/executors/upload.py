from __future__ import annotations

from typing import Any

from .. import events
from ..records import new_id, utc_now
from ..tasks import TaskContext
from .common import filename_from_url, normalise_google_drive_url, unique_path


async def execute_upload_file(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    await ctx.progress(20, "validating uploaded file")
    path = ctx.store.resolve_project_file(ctx.task.project_id, params["stored_path"])
    if not path.exists():
        raise FileNotFoundError(f"uploaded file not found: {path}")

    upload_id = params.get("upload_id") or new_id("upload")
    relpath = ctx.store.relative_to_project(ctx.task.project_id, path)
    payload = {
        "upload_id": upload_id,
        "kind": params.get("kind", "file"),
        "filename": params.get("filename", path.name),
        "path": relpath,
        "size_bytes": path.stat().st_size,
        "created_at": utc_now(),
    }
    ctx.store.register_upload(ctx.task.project_id, upload_id, payload)

    if params.get("kind") == "dataset":
        dataset_id = params.get("dataset_id") or new_id("dataset")
        dataset_payload = {
            "dataset_id": dataset_id,
            "format": params.get("format", "unknown"),
            "source": "upload",
            "zip_path": relpath,
            "created_at": utc_now(),
        }
        ctx.store.register_dataset(ctx.task.project_id, dataset_id, dataset_payload)
        payload["dataset_id"] = dataset_id

    await ctx.progress(100, "upload registered")
    await events.notify_upload_success(
        ctx.connections,
        project_id=ctx.task.project_id,
        task_id=ctx.task.task_id,
        filename=payload["filename"],
        path=relpath,
    )
    return payload


async def execute_remote_download(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    source = params.get("source", "url")
    url = params["url"]
    if source == "mega":
        raise RuntimeError(
            "Mega public-link downloading requires a Mega adapter/CLI. "
            "Use HTTP upload or install a Mega downloader integration."
        )

    download_url = normalise_google_drive_url(url) if source == "google_drive" else url
    await ctx.progress(5, f"starting {source} download")
    filename = params.get("filename") or filename_from_url(download_url)
    target_dir = ctx.store.project_path(ctx.task.project_id) / "uploads" / source
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = unique_path(target_dir / filename)

    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx is required for remote downloads") from exc

    bytes_written = 0
    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", download_url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or "0")
            with target_path.open("wb") as handle:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    if ctx.is_cancelled():
                        raise RuntimeError("download cancelled")
                    handle.write(chunk)
                    bytes_written += len(chunk)
                    if total:
                        progress = 5 + min(90, bytes_written / total * 90)
                        await ctx.progress(progress, f"downloaded {bytes_written} bytes")

    relpath = ctx.store.relative_to_project(ctx.task.project_id, target_path)
    upload_id = new_id("upload")
    payload = {
        "upload_id": upload_id,
        "kind": params.get("kind", "file"),
        "filename": target_path.name,
        "path": relpath,
        "size_bytes": bytes_written,
        "source": source,
        "source_url": url,
        "created_at": utc_now(),
    }
    ctx.store.register_upload(ctx.task.project_id, upload_id, payload)

    if params.get("kind") == "dataset":
        dataset_id = new_id("dataset")
        dataset_payload = {
            "dataset_id": dataset_id,
            "format": params.get("format", "unknown"),
            "source": source,
            "zip_path": relpath,
            "created_at": utc_now(),
        }
        ctx.store.register_dataset(ctx.task.project_id, dataset_id, dataset_payload)
        payload["dataset_id"] = dataset_id

    await events.notify_upload_success(
        ctx.connections,
        project_id=ctx.task.project_id,
        task_id=ctx.task.task_id,
        filename=target_path.name,
        path=relpath,
    )
    await ctx.progress(100, "download complete")
    return payload
