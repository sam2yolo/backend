from __future__ import annotations

import asyncio
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .. import events
from ..records import new_id, utc_now
from ..tasks import TaskContext
from .common import IMAGE_EXTENSIONS, read_json, write_json, zip_directory


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

    annotations = read_json(work_dir / "annotations.json")
    metadata = read_json(work_dir / "metadata.json")
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
    write_json(
        dataset_dir / "dataset_manifest.json",
        {
            "dataset_id": dataset_id,
            "source": "inference",
            "inference_task_id": inference_task_id,
            "created_at": utc_now(),
        },
    )
    dataset_zip = dataset_dir.with_suffix(".zip")
    await asyncio.to_thread(zip_directory, dataset_dir, dataset_zip)
    rel_zip = ctx.store.relative_to_project(ctx.task.project_id, dataset_zip)
    payload = {
        "dataset_id": dataset_id,
        "format": "yolo",
        "source": "inference",
        "zip_path": rel_zip,
        "download_url": (
            f"/v1/projects/{quote(ctx.task.project_id)}/downloads/datasets/"
            f"{quote(dataset_id)}"
        ),
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
        extract_dir = (
            ctx.store.project_path(ctx.task.project_id) / "tmp" / f"{output_id}_{dataset_id}"
        )
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
    write_json(
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
    await asyncio.to_thread(zip_directory, output_dir, output_zip)
    rel_zip = ctx.store.relative_to_project(ctx.task.project_id, output_zip)
    payload = {
        "dataset_id": output_id,
        "format": "yolo",
        "source": operation,
        "zip_path": rel_zip,
        "download_url": (
            f"/v1/projects/{quote(ctx.task.project_id)}/downloads/datasets/"
            f"{quote(output_id)}"
        ),
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
        xs = [
            float(point[0])
            for point in polygon
            if isinstance(point, list) and len(point) >= 2
        ]
        ys = [
            float(point[1])
            for point in polygon
            if isinstance(point, list) and len(point) >= 2
        ]
        if xs and ys:
            return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return None


def _write_yolo_metadata(dataset_dir: Path, class_names: list[str]) -> None:
    write_json(dataset_dir / "classes.json", {"names": class_names})
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
            payload = read_json(candidate)
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
