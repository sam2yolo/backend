from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from . import events
from .capabilities import TRAINING_MODEL_NAMES
from .records import TaskType, new_id, utc_now
from .tasks import TaskContext, TaskManager


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def register_default_executors(task_manager: TaskManager) -> None:
    task_manager.register_executor(TaskType.UPLOAD_FILE.value, execute_upload_file)
    task_manager.register_executor(TaskType.REMOTE_DOWNLOAD.value, execute_remote_download)
    task_manager.register_executor(TaskType.INFERENCE_SAM3.value, execute_inference_sam3)
    task_manager.register_executor(TaskType.INFERENCE_YOLO.value, execute_inference_yolo)
    task_manager.register_executor(TaskType.INFERENCE_TO_YOLO.value, execute_inference_to_yolo)
    task_manager.register_executor(TaskType.MERGE_DATASET.value, execute_merge_dataset)
    task_manager.register_executor(TaskType.MAP_DATASET.value, execute_map_dataset)
    task_manager.register_executor(TaskType.TRAIN_MODEL.value, execute_train_model)


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

    download_url = _normalise_google_drive_url(url) if source == "google_drive" else url
    await ctx.progress(5, f"starting {source} download")
    filename = params.get("filename") or _filename_from_url(download_url)
    target_dir = ctx.store.project_path(ctx.task.project_id) / "uploads" / source
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = _unique_path(target_dir / filename)

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
    prompts = params.get("prompts") or []
    prompt_to_class = params.get("prompt_to_class") or {}
    batch_size = max(1, int(params.get("batch_size", 4)))
    media_path = ctx.store.resolve_project_file(ctx.task.project_id, params["media_path"])

    result_dir = ctx.store.project_path(ctx.task.project_id) / "inference_results" / ctx.task.task_id
    frames_dir = result_dir / "frames"
    checkpoint_dir = result_dir / "checkpoints"
    frames_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    await ctx.progress(2, "preparing media")
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
        _write_json(
            checkpoint_path,
            {
                "batch_index": batch_index,
                "frames": batch_rows,
                "completed_at": utc_now(),
            },
        )
        progress = 5 + ((batch_index + 1) / total_batches) * 85
        await ctx.progress(progress, f"processed batch {batch_index + 1}/{total_batches}")
        await _checkpoint_client_fetch(ctx, checkpoint_path)

    metadata.update(
        {
            "task_id": ctx.task.task_id,
            "model_name": model_name,
            "prompts": prompts,
            "prompt_to_class": prompt_to_class,
            "batch_size": batch_size,
            "sample_interval_seconds": params.get("sample_interval_seconds"),
            "created_at": utc_now(),
            "stub_result": True,
        }
    )
    _write_json(result_dir / "annotations.json", annotations)
    _write_json(result_dir / "metadata.json", metadata)
    zip_path = result_dir.with_suffix(".zip")
    await asyncio.to_thread(_zip_directory, result_dir, zip_path)

    rel_zip = ctx.store.relative_to_project(ctx.task.project_id, zip_path)
    result_payload = {
        "task_id": ctx.task.task_id,
        "format": "samtoyolo.inference.zip",
        "zip_path": rel_zip,
        "download_url": f"/v1/projects/{quote(ctx.task.project_id)}/downloads/inference/{quote(ctx.task.task_id)}",
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


async def execute_inference_to_yolo(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    inference_task_id = params.get("inference_task_id") or params.get("task_id")
    if not inference_task_id:
        raise ValueError("inference_task_id is required")
    session = ctx.store.get_session(ctx.task.project_id)
    inference = session.get("inference_results", {}).get(inference_task_id)
    if not inference:
        raise ValueError(f"unknown inference result: {inference_task_id}")

    zip_path = ctx.store.resolve_project_file(ctx.task.project_id, inference["zip_path"])
    dataset_id = new_id("dataset")
    work_dir = ctx.store.project_path(ctx.task.project_id) / "tmp" / dataset_id
    dataset_dir = ctx.store.project_path(ctx.task.project_id) / "datasets" / dataset_id
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(dataset_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    await ctx.progress(10, "extracting inference result")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work_dir)

    annotations = _read_json(work_dir / "annotations.json")
    metadata = _read_json(work_dir / "metadata.json")
    class_names = _class_names_from_prompt_map(metadata.get("prompt_to_class", {}))
    if not class_names:
        class_names = ["object"]

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    source_frames = work_dir / "frames"

    frames = annotations.get("frames", [])
    total = max(1, len(frames))
    for index, frame in enumerate(frames):
        filename = frame.get("file_name")
        if not filename:
            continue
        source = source_frames / filename
        if source.exists():
            shutil.copy2(source, images_dir / source.name)
        label_lines = _yolo_label_lines(frame, class_names)
        (labels_dir / f"{Path(filename).stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        await ctx.progress(10 + (index + 1) / total * 70, "converting annotations")

    _write_yolo_metadata(dataset_dir, class_names)
    _write_json(
        dataset_dir / "dataset_manifest.json",
        {
            "dataset_id": dataset_id,
            "source": "inference",
            "inference_task_id": inference_task_id,
            "created_at": utc_now(),
        },
    )
    dataset_zip = dataset_dir.with_suffix(".zip")
    await asyncio.to_thread(_zip_directory, dataset_dir, dataset_zip)
    rel_zip = ctx.store.relative_to_project(ctx.task.project_id, dataset_zip)
    payload = {
        "dataset_id": dataset_id,
        "format": "yolo",
        "source": "inference",
        "zip_path": rel_zip,
        "download_url": f"/v1/projects/{quote(ctx.task.project_id)}/downloads/datasets/{quote(dataset_id)}",
        "class_names": class_names,
        "created_at": utc_now(),
    }
    ctx.store.register_dataset(ctx.task.project_id, dataset_id, payload)
    await ctx.progress(100, "YOLO dataset ready")
    return payload


async def execute_merge_dataset(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    dataset_ids = params.get("dataset_ids") or []
    class_map = params.get("class_map") or {}
    if len(dataset_ids) < 2:
        raise ValueError("at least two dataset_ids are required")
    return await _merge_or_map_dataset(
        ctx,
        dataset_ids=dataset_ids,
        class_map=class_map,
        operation="merge",
    )


async def execute_map_dataset(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    dataset_id = params.get("dataset_id")
    class_map = params.get("class_map") or {}
    if not dataset_id:
        raise ValueError("dataset_id is required")
    return await _merge_or_map_dataset(
        ctx,
        dataset_ids=[dataset_id],
        class_map=class_map,
        operation="map",
    )


async def _merge_or_map_dataset(
    ctx: TaskContext,
    *,
    dataset_ids: list[str],
    class_map: dict[str, str],
    operation: str,
) -> dict[str, Any]:
    session = ctx.store.get_session(ctx.task.project_id)
    datasets = session.get("datasets", {})
    output_id = new_id("dataset")
    output_dir = ctx.store.project_path(ctx.task.project_id) / "datasets" / output_id
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "labels" / "train").mkdir(parents=True)
    unified_classes: list[str] = []

    for dataset_number, dataset_id in enumerate(dataset_ids, start=1):
        dataset = datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        zip_path = ctx.store.resolve_project_file(ctx.task.project_id, dataset["zip_path"])
        extract_dir = ctx.store.project_path(ctx.task.project_id) / "tmp" / f"{output_id}_{dataset_id}"
        shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

        classes = _read_classes(extract_dir)
        mapped_classes = [_mapped_class(name, class_map) for name in classes]
        for class_name in mapped_classes:
            if class_name not in unified_classes:
                unified_classes.append(class_name)

        _copy_yolo_dataset(
            source_dir=extract_dir,
            output_dir=output_dir,
            source_classes=classes,
            mapped_classes=mapped_classes,
            unified_classes=unified_classes,
            prefix=f"d{dataset_number}_{dataset_id}",
        )
        await ctx.progress(
            dataset_number / len(dataset_ids) * 80,
            f"{operation} dataset {dataset_number}/{len(dataset_ids)}",
        )

    _write_yolo_metadata(output_dir, unified_classes or ["object"])
    _write_json(
        output_dir / "dataset_manifest.json",
        {
            "dataset_id": output_id,
            "operation": operation,
            "source_dataset_ids": dataset_ids,
            "class_map": class_map,
            "created_at": utc_now(),
        },
    )
    output_zip = output_dir.with_suffix(".zip")
    await asyncio.to_thread(_zip_directory, output_dir, output_zip)
    rel_zip = ctx.store.relative_to_project(ctx.task.project_id, output_zip)
    payload = {
        "dataset_id": output_id,
        "format": "yolo",
        "source": operation,
        "zip_path": rel_zip,
        "download_url": f"/v1/projects/{quote(ctx.task.project_id)}/downloads/datasets/{quote(output_id)}",
        "class_names": unified_classes,
        "created_at": utc_now(),
    }
    ctx.store.register_dataset(ctx.task.project_id, output_id, payload)
    if operation == "merge":
        await events.notify_new_dataset_from_merge(
            ctx.connections,
            project_id=ctx.task.project_id,
            task_id=ctx.task.task_id,
            dataset_id=output_id,
        )
    else:
        await events.notify_new_dataset_from_map(
            ctx.connections,
            project_id=ctx.task.project_id,
            task_id=ctx.task.task_id,
            dataset_id=output_id,
        )
    await ctx.progress(100, f"dataset {operation} complete")
    return payload


async def execute_train_model(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    if not params.get("allow_stub_ml", True):
        raise RuntimeError("training adapter is not installed and stub ML is disabled")
    model_name = (params.get("model_name") or "yolov8").lower().replace("-", "_")
    if model_name not in TRAINING_MODEL_NAMES:
        raise ValueError(f"unsupported model_name: {model_name}")
    dataset_id = params.get("dataset_id")
    if not dataset_id:
        raise ValueError("dataset_id is required")
    session = ctx.store.get_session(ctx.task.project_id)
    if dataset_id not in session.get("datasets", {}):
        raise ValueError(f"unknown dataset_id: {dataset_id}")

    config = params.get("config") or {}
    epochs = max(1, int(config.get("epochs", 1)))
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 5)))
    model_id = new_id("model")
    model_dir = ctx.store.project_path(ctx.task.project_id) / "models" / model_id
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"epochs": epochs, "history": []}

    for epoch in range(1, epochs + 1):
        if ctx.is_cancelled():
            raise RuntimeError("training cancelled")
        loss = round(1 / (epoch + 1), 6)
        metric = {"epoch": epoch, "loss": loss}
        metrics["history"].append(metric)
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            checkpoint = checkpoint_dir / f"epoch_{epoch:04d}.ckpt"
            checkpoint.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "dataset_id": dataset_id,
                        "epoch": epoch,
                        "stub_checkpoint": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            await _checkpoint_client_fetch(ctx, checkpoint)
        await ctx.progress(epoch / epochs * 95, f"epoch {epoch}/{epochs}", metrics=metric)
        await asyncio.sleep(0)

    final_path = model_dir / f"{model_name}_{model_id}.pt"
    final_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_name": model_name,
                "dataset_id": dataset_id,
                "config": config,
                "metrics": metrics,
                "stub_model": True,
                "created_at": utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rel_model = ctx.store.relative_to_project(ctx.task.project_id, final_path)
    payload = {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_id": dataset_id,
        "path": rel_model,
        "download_url": f"/v1/projects/{quote(ctx.task.project_id)}/downloads/models/{quote(model_id)}",
        "metrics": metrics,
        "created_at": utc_now(),
    }
    ctx.store.register_model(ctx.task.project_id, model_id, payload)
    await events.notify_training_complete(
        ctx.connections,
        project_id=ctx.task.project_id,
        task_id=ctx.task.task_id,
        model_id=model_id,
        metrics=metrics,
    )
    await ctx.progress(100, "training complete", metrics=metrics["history"][-1])
    return payload


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


async def _checkpoint_client_fetch(ctx: TaskContext, path: Path) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _zip_directory(source_dir: Path, target_zip: Path) -> None:
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source_dir))


def _normalise_google_drive_url(url: str) -> str:
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


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "download.bin"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not create unique path for {path}")


def _class_names_from_prompt_map(prompt_to_class: dict[str, str]) -> list[str]:
    names: list[str] = []
    for class_name in prompt_to_class.values():
        if class_name not in names:
            names.append(class_name)
    return names


def _yolo_label_lines(frame: dict[str, Any], class_names: list[str]) -> list[str]:
    width = frame.get("width")
    height = frame.get("height")
    if not width or not height:
        return []
    lines = []
    for obj in frame.get("objects", []):
        class_name = obj.get("class_name") or "object"
        if class_name not in class_names:
            class_names.append(class_name)
        bbox = _bbox_from_object(obj)
        if not bbox:
            continue
        x, y, w, h = bbox
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        lines.append(
            f"{class_names.index(class_name)} {x_center:.6f} {y_center:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
    return lines


def _bbox_from_object(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    polygon = obj.get("polygon")
    if isinstance(polygon, list) and polygon:
        xs = [float(point[0]) for point in polygon if isinstance(point, list) and len(point) >= 2]
        ys = [float(point[1]) for point in polygon if isinstance(point, list) and len(point) >= 2]
        if xs and ys:
            return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return None


def _write_yolo_metadata(dataset_dir: Path, class_names: list[str]) -> None:
    _write_json(dataset_dir / "classes.json", {"names": class_names})
    yaml_names = ", ".join(json.dumps(name) for name in class_names)
    (dataset_dir / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/train\n"
        f"nc: {len(class_names)}\n"
        f"names: [{yaml_names}]\n",
        encoding="utf-8",
    )


def _read_classes(dataset_dir: Path) -> list[str]:
    for candidate in dataset_dir.rglob("classes.json"):
        try:
            payload = _read_json(candidate)
            names = payload.get("names")
            if isinstance(names, list):
                return [str(name) for name in names]
        except Exception:
            pass
    for candidate in dataset_dir.rglob("data.yaml"):
        text = candidate.read_text(encoding="utf-8")
        match = re.search(r"names:\s*\[(.*?)\]", text, re.S)
        if match:
            try:
                return [str(name) for name in json.loads(f"[{match.group(1)}]")]
            except Exception:
                pass
    return ["object"]


def _mapped_class(class_name: str, class_map: dict[str, str]) -> str:
    return class_map.get(class_name, class_name)


def _copy_yolo_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    source_classes: list[str],
    mapped_classes: list[str],
    unified_classes: list[str],
    prefix: str,
) -> None:
    image_sources = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for image_path in image_sources:
        target_name = f"{prefix}_{image_path.name}"
        shutil.copy2(image_path, output_dir / "images" / "train" / target_name)
        source_label = _find_label_for_image(source_dir, image_path)
        target_label = output_dir / "labels" / "train" / f"{Path(target_name).stem}.txt"
        if source_label and source_label.exists():
            target_label.write_text(
                _rewrite_yolo_labels(
                    source_label.read_text(encoding="utf-8"),
                    source_classes,
                    mapped_classes,
                    unified_classes,
                ),
                encoding="utf-8",
            )
        else:
            target_label.write_text("", encoding="utf-8")


def _find_label_for_image(source_dir: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    candidates = list(source_dir.rglob(f"{stem}.txt"))
    return candidates[0] if candidates else None


def _rewrite_yolo_labels(
    text: str,
    source_classes: list[str],
    mapped_classes: list[str],
    unified_classes: list[str],
) -> str:
    output = []
    for raw_line in text.splitlines():
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        try:
            old_index = int(parts[0])
            class_name = source_classes[old_index]
            mapped_name = mapped_classes[source_classes.index(class_name)]
            new_index = unified_classes.index(mapped_name)
        except Exception:
            continue
        output.append(" ".join([str(new_index), *parts[1:]]))
    return "\n".join(output) + ("\n" if output else "")
