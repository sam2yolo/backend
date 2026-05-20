from __future__ import annotations

import asyncio
import math
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .. import events
from ..model_assets import PreparedModelAsset, prepare_zip_model_asset
from ..records import utc_now
from ..tasks import TaskContext
from .common import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    checkpoint_client_fetch,
    write_json,
    zip_directory,
)


async def execute_inference_sam3(ctx: TaskContext) -> dict[str, Any]:
    return await _execute_inference(ctx, model_name="sam3")


async def execute_inference_yolo(ctx: TaskContext) -> dict[str, Any]:
    return await _execute_inference(ctx, model_name="yolov8")


async def _execute_inference(ctx: TaskContext, *, model_name: str) -> dict[str, Any]:
    params = ctx.task.params
    if not params.get("allow_stub_ml", True):
        raise RuntimeError(
            f"{model_name} inference adapter is not installed and stub ML is disabled"
        )
    model_asset: PreparedModelAsset | None = None
    progress_base = 2.0
    progress_start = 5.0
    if model_name == "sam3" and params.get("prepare_model", True):
        await ctx.progress(1, "preparing SAM 3.1 model")

        async def model_progress(percent: float, message: str) -> None:
            await ctx.progress(1 + percent * 0.07, message)

        model_asset = await prepare_zip_model_asset(
            model_name="sam3",
            source_url=params["model_source_url"],
            download_url=params.get("model_download_url"),
            filename=params["model_filename"],
            cache_dir=params["model_cache_dir"],
            progress=model_progress,
        )
        await ctx.progress(8, "SAM 3.1 model ready")
        progress_base = 10.0
        progress_start = 12.0

    prompts = params.get("prompts") or []
    prompt_to_class = params.get("prompt_to_class") or {}
    batch_size = max(1, int(params.get("batch_size", 4)))
    media_path = ctx.store.resolve_project_file(ctx.task.project_id, params["media_path"])

    result_dir = (
        ctx.store.project_path(ctx.task.project_id)
        / "inference_results"
        / ctx.task.task_id
    )
    frames_dir = result_dir / "frames"
    checkpoint_dir = result_dir / "checkpoints"
    frames_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    await ctx.progress(progress_base, "preparing media")
    frames = await asyncio.to_thread(
        _prepare_frames,
        media_path,
        frames_dir,
        params.get("sample_interval_seconds")
        or params.get("temporal_downsample")
        or params.get("downsample")
        or 1,
    )

    if not frames:
        metadata = {
            "warning": "no frames were extracted; adapter will need a valid media source",
            "source_media": str(media_path),
        }
    else:
        metadata = {"source_media": str(media_path)}

    annotations: dict[str, Any] = {
        "version": "samtoyolo.inference.v1",
        "model_name": model_name,
        "frames": [],
    }
    total_batches = max(1, math.ceil(len(frames) / batch_size))
    for batch_index in range(total_batches):
        if ctx.is_cancelled():
            raise RuntimeError("inference cancelled")
        start = batch_index * batch_size
        end = start + batch_size
        batch_frames = frames[start:end] or []
        batch_rows = [
            {
                "file_name": frame.name,
                "width": None,
                "height": None,
                "objects": [],
            }
            for frame in batch_frames
        ]
        annotations["frames"].extend(batch_rows)
        checkpoint_path = checkpoint_dir / f"batch_{batch_index + 1:06d}.json"
        write_json(
            checkpoint_path,
            {
                "batch_index": batch_index,
                "frames": batch_rows,
                "completed_at": utc_now(),
            },
        )
        progress = progress_start + ((batch_index + 1) / total_batches) * (
            90 - progress_start
        )
        await ctx.progress(progress, f"processed batch {batch_index + 1}/{total_batches}")
        await checkpoint_client_fetch(ctx, checkpoint_path)

    metadata.update(
        {
            "task_id": ctx.task.task_id,
            "model_name": model_name,
            "prompts": prompts,
            "prompt_to_class": prompt_to_class,
            "batch_size": batch_size,
            "sample_interval_seconds": params.get("sample_interval_seconds"),
            "model_source_url": params.get("model_source_url"),
            "model_download_url": params.get("model_download_url"),
            "model_filename": params.get("model_filename"),
            "model_asset": model_asset.to_dict() if model_asset else None,
            "created_at": utc_now(),
            "stub_result": True,
        }
    )
    write_json(result_dir / "annotations.json", annotations)
    write_json(result_dir / "metadata.json", metadata)
    zip_path = result_dir.with_suffix(".zip")
    await asyncio.to_thread(zip_directory, result_dir, zip_path)

    rel_zip = ctx.store.relative_to_project(ctx.task.project_id, zip_path)
    result_payload = {
        "task_id": ctx.task.task_id,
        "format": "samtoyolo.inference.zip",
        "zip_path": rel_zip,
        "download_url": (
            f"/v1/projects/{quote(ctx.task.project_id)}/downloads/inference/"
            f"{quote(ctx.task.task_id)}"
        ),
        "model_source_url": params.get("model_source_url"),
        "model_asset": model_asset.to_dict() if model_asset else None,
        "created_at": utc_now(),
    }
    ctx.store.register_inference_result(ctx.task.project_id, ctx.task.task_id, result_payload)

    await events.notify_inference_result_ready(
        ctx.connections,
        project_id=ctx.task.project_id,
        task_id=ctx.task.task_id,
        file_path_or_url=result_payload["download_url"],
    )
    if params.get("save_to_mega"):
        await events.notify_mega_mount_success(
            ctx.connections, project_id=ctx.task.project_id, task_id=ctx.task.task_id
        )
        mega_path = f"mega/projects/{ctx.task.project_id}/inference_results/{zip_path.name}"
        await events.notify_mega_upload_success(
            ctx.connections,
            project_id=ctx.task.project_id,
            task_id=ctx.task.task_id,
            mega_path=mega_path,
        )
        result_payload["mega_path"] = mega_path

    await ctx.progress(100, "inference result packaged")
    return result_payload


def _prepare_frames(source: Path, frames_dir: Path, sample_value: Any) -> list[Path]:
    if source.is_dir():
        copied = []
        for image in sorted(source.iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                target = frames_dir / image.name
                shutil.copy2(image, target)
                copied.append(target)
        return copied
    if source.suffix.lower() in IMAGE_EXTENSIONS and source.exists():
        target = frames_dir / source.name
        shutil.copy2(source, target)
        return [target]
    if source.suffix.lower() in VIDEO_EXTENSIONS and source.exists():
        return _extract_video_frames(source, frames_dir, sample_value)
    return []


def _extract_video_frames(source: Path, frames_dir: Path, sample_value: Any) -> list[Path]:
    try:
        import cv2  # type: ignore
    except Exception:
        return []

    interval_seconds = _sample_interval_seconds(sample_value)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(round(fps * interval_seconds)))
    output: list[Path] = []
    index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_interval == 0:
            target = frames_dir / f"frame_{saved:08d}.jpg"
            cv2.imwrite(str(target), frame)
            output.append(target)
            saved += 1
        index += 1
    capture.release()
    return output


def _sample_interval_seconds(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 1.0
    if numeric <= 0:
        return 1.0
    if numeric < 1:
        return 1 / numeric
    return numeric
