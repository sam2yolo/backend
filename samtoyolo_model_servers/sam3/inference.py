from __future__ import annotations

import contextlib
import gc
import io
import inspect
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[float, str, dict[str, Any] | None], None]
CancelChecker = Callable[[], bool]


class Sam3AdapterError(RuntimeError):
    """Raised when the SAM 3.1 runtime cannot perform real inference."""


@dataclass(slots=True)
class Sam3InferenceResult:
    frames: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PredictorKey:
    checkpoint_path: str
    gpu_index: int
    output_prob_thresh: float
    max_num_objects: int
    multiplex_count: int
    use_fa3: bool
    compile_model: bool
    async_loading_frames: bool
    allow_partial_checkpoint: bool


_PREDICTOR_CACHE: dict[_PredictorKey, Any] = {}
_PREDICTOR_LOCKS: dict[_PredictorKey, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def run_sam3_video_text_inference(
    *,
    frames_dir: Path,
    frames: list[Path],
    prompts: list[str],
    prompt_to_class: dict[str, str],
    model_extract_dir: Path,
    gpu_index: int | None,
    output_prob_thresh: float = 0.5,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    compile_model: bool = False,
    warm_up: bool = False,
    async_loading_frames: bool = False,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
    cache_model: bool = True,
    allow_partial_checkpoint: bool = False,
    include_masks: bool = True,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelChecker | None = None,
    progress_start: float = 12.0,
    progress_end: float = 88.0,
) -> Sam3InferenceResult:
    """Run official SAM 3.1 video predictor on sampled frames with text prompts."""

    rows = _empty_annotation_rows(frames)
    if not frames or not prompts:
        return Sam3InferenceResult(
            frames=rows,
            metadata={
                "sam3_backend": "official_sam3_video_predictor",
                "sam3_skipped": "no frames or prompts",
            },
        )

    checkpoint_path = find_sam3_checkpoint(model_extract_dir)
    gpu_id = _select_cuda_device(gpu_index)
    _progress(
        progress,
        progress_start,
        f"loading SAM 3.1 on gpu-{gpu_id}",
        {"gpu_index": gpu_id, "checkpoint_path": str(checkpoint_path)},
    )
    predictor = _get_predictor(
        checkpoint_path=checkpoint_path,
        gpu_index=gpu_id,
        output_prob_thresh=output_prob_thresh,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        compile_model=compile_model,
        warm_up=warm_up,
        async_loading_frames=async_loading_frames,
        cache_model=cache_model,
        allow_partial_checkpoint=allow_partial_checkpoint,
    )

    session_id: str | None = None
    completed_stream_steps = 0
    total_stream_steps = max(1, len(prompts) * max(1, len(frames)))
    try:
        _raise_if_cancelled(is_cancelled)
        with _patch_init_state_kwargs(predictor, {"offload_state_to_cpu"}):
            response = predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": str(frames_dir),
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "offload_state_to_cpu": offload_state_to_cpu,
                }
            )
        session_id = str(response["session_id"])

        for prompt_index, prompt in enumerate(prompts):
            _raise_if_cancelled(is_cancelled)
            if prompt_index:
                predictor.handle_request(
                    request={"type": "reset_session", "session_id": session_id}
                )

            class_name = prompt_to_class.get(prompt, prompt)
            prompt_progress = _interpolate(
                progress_start,
                progress_end,
                completed_stream_steps,
                total_stream_steps,
            )
            _progress(
                progress,
                prompt_progress,
                f"running SAM prompt {prompt_index + 1}/{len(prompts)}: {prompt}",
                {"gpu_index": gpu_id, "prompt": prompt},
            )
            first_response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                    "output_prob_thresh": output_prob_thresh,
                }
            )
            _replace_prompt_objects(
                rows=rows,
                frame_index=int(first_response["frame_index"]),
                outputs=first_response.get("outputs") or {},
                prompt=prompt,
                prompt_index=prompt_index,
                class_name=class_name,
                include_masks=include_masks,
            )

            for stream_response in predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "output_prob_thresh": output_prob_thresh,
                }
            ):
                _raise_if_cancelled(is_cancelled)
                frame_index = int(stream_response["frame_index"])
                _replace_prompt_objects(
                    rows=rows,
                    frame_index=frame_index,
                    outputs=stream_response.get("outputs") or {},
                    prompt=prompt,
                    prompt_index=prompt_index,
                    class_name=class_name,
                    include_masks=include_masks,
                )
                completed_stream_steps += 1
                if completed_stream_steps % max(1, len(frames) // 8) == 0:
                    _progress(
                        progress,
                        _interpolate(
                            progress_start,
                            progress_end,
                            completed_stream_steps,
                            total_stream_steps,
                        ),
                        (
                            f"SAM prompt {prompt_index + 1}/{len(prompts)} "
                            f"frame {min(frame_index + 1, len(frames))}/{len(frames)}"
                        ),
                        {"gpu_index": gpu_id, "prompt": prompt, "frame_index": frame_index},
                    )
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
            except Exception:
                pass
        if not cache_model:
            _drop_cached_predictor(checkpoint_path, gpu_id)
        gc.collect()

    object_count = sum(len(row.get("objects", [])) for row in rows)
    _progress(
        progress,
        progress_end,
        f"SAM 3.1 inference produced {object_count} objects",
        {"gpu_index": gpu_id, "object_count": object_count},
    )
    return Sam3InferenceResult(
        frames=rows,
        metadata={
            "sam3_backend": "official_sam3_video_predictor",
            "sam3_checkpoint_path": str(checkpoint_path),
            "sam3_gpu_index": gpu_id,
            "sam3_output_prob_thresh": output_prob_thresh,
            "sam3_max_num_objects": max_num_objects,
            "sam3_multiplex_count": multiplex_count,
            "sam3_use_fa3": use_fa3,
            "sam3_compile": compile_model,
            "sam3_async_loading_frames": async_loading_frames,
            "sam3_model_cached": cache_model,
            "sam3_allow_partial_checkpoint": allow_partial_checkpoint,
            "sam3_object_count": object_count,
            "sam3_real_inference": True,
        },
    )


def find_sam3_checkpoint(model_extract_dir: Path) -> Path:
    """Find the SAM checkpoint inside the prepared Google Drive zip extraction."""

    root = Path(model_extract_dir)
    for name in ("sam3.1_multiplex.pt", "sam3.pt"):
        candidate = root / name
        if candidate.exists():
            return candidate
    checkpoints = sorted(root.rglob("*.pt"))
    if checkpoints:
        return checkpoints[0]
    raise Sam3AdapterError(f"no SAM checkpoint .pt file found under {root}")


def _get_predictor(
    *,
    checkpoint_path: Path,
    gpu_index: int,
    output_prob_thresh: float,
    max_num_objects: int,
    multiplex_count: int,
    use_fa3: bool,
    compile_model: bool,
    warm_up: bool,
    async_loading_frames: bool,
    allow_partial_checkpoint: bool,
    cache_model: bool,
) -> Any:
    key = _PredictorKey(
        checkpoint_path=str(checkpoint_path.resolve()),
        gpu_index=gpu_index,
        output_prob_thresh=float(output_prob_thresh),
        max_num_objects=int(max_num_objects),
        multiplex_count=int(multiplex_count),
        use_fa3=bool(use_fa3),
        compile_model=bool(compile_model),
        async_loading_frames=bool(async_loading_frames),
        allow_partial_checkpoint=bool(allow_partial_checkpoint),
    )
    if cache_model and key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]

    with _CACHE_LOCK:
        lock = _PREDICTOR_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if cache_model and key in _PREDICTOR_CACHE:
            return _PREDICTOR_CACHE[key]
        predictor = _build_predictor(
            checkpoint_path=checkpoint_path,
            gpu_index=gpu_index,
            output_prob_thresh=output_prob_thresh,
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            compile_model=compile_model,
            warm_up=warm_up,
            async_loading_frames=async_loading_frames,
            allow_partial_checkpoint=allow_partial_checkpoint,
        )
        if cache_model:
            _PREDICTOR_CACHE[key] = predictor
        return predictor


def _build_predictor(
    *,
    checkpoint_path: Path,
    gpu_index: int,
    output_prob_thresh: float,
    max_num_objects: int,
    multiplex_count: int,
    use_fa3: bool,
    compile_model: bool,
    warm_up: bool,
    async_loading_frames: bool,
    allow_partial_checkpoint: bool,
) -> Any:
    try:
        import torch
        from sam3.model_builder import build_sam3_predictor
    except Exception as exc:
        raise Sam3AdapterError(
            "SAM 3.1 inference requires the official sam3 package. Install it with "
            "`pip install git+https://github.com/facebookresearch/sam3.git` after "
            "installing a CUDA-enabled torch/torchvision build."
        ) from exc

    if not torch.cuda.is_available():
        raise Sam3AdapterError("SAM 3.1 video inference requires a CUDA GPU")
    torch.cuda.set_device(gpu_index)
    build_log = io.StringIO()
    try:
        with contextlib.redirect_stdout(build_log):
            predictor = build_sam3_predictor(
                checkpoint_path=str(checkpoint_path),
                version="sam3.1",
                max_num_objects=max_num_objects,
                multiplex_count=multiplex_count,
                use_fa3=use_fa3,
                compile=compile_model,
                warm_up=warm_up,
                async_loading_frames=async_loading_frames,
                default_output_prob_thresh=output_prob_thresh,
            )
    except Exception as exc:
        summary = _checkpoint_load_summary(build_log.getvalue())
        details = f" Checkpoint loader output: {summary}" if summary else ""
        raise Sam3AdapterError(
            "SAM 3.1 predictor failed to load the prepared checkpoint."
            f"{details} Original error: {exc}"
        ) from exc

    summary = _checkpoint_load_summary(build_log.getvalue())
    if (
        summary
        and not allow_partial_checkpoint
        and not _is_expected_sam31_multiplex_loader_warning(summary)
    ):
        raise Sam3AdapterError(
            "SAM 3.1 checkpoint did not load cleanly with the installed "
            "facebookresearch/sam3 package, so real inference was stopped instead "
            "of using partially random weights. Install a SAM3 code branch/checkpoint "
            "pair that loads without missing/unexpected keys, or set "
            "`sam3_allow_partial_checkpoint=true` only for debugging. "
            f"Checkpoint loader output: {summary}"
        )
    return predictor


@contextlib.contextmanager
def _patch_init_state_kwargs(predictor: Any, candidate_kwargs: set[str]):
    """Drop kwargs that SAM 3.1's model init_state does not accept."""

    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if model is None or init_state is None:
        yield
        return

    try:
        signature = inspect.signature(init_state)
    except (TypeError, ValueError):
        yield
        return

    accepts_extra_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_extra_kwargs:
        yield
        return

    unsupported = {name for name in candidate_kwargs if name not in signature.parameters}
    if not unsupported:
        yield
        return

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        for name in unsupported:
            kwargs.pop(name, None)
        return init_state(*args, **kwargs)

    setattr(model, "init_state", compatible_init_state)
    try:
        yield
    finally:
        setattr(model, "init_state", init_state)


def _drop_cached_predictor(checkpoint_path: Path, gpu_index: int) -> None:
    checkpoint = str(checkpoint_path.resolve())
    with _CACHE_LOCK:
        stale_keys = [
            key
            for key in _PREDICTOR_CACHE
            if key.checkpoint_path == checkpoint and key.gpu_index == gpu_index
        ]
        for key in stale_keys:
            predictor = _PREDICTOR_CACHE.pop(key)
            try:
                predictor.shutdown()
            except Exception:
                pass


def _checkpoint_load_summary(output: str) -> str | None:
    text = output.strip()
    if not text:
        return None
    markers = ("Missing keys", "Unexpected keys", "size mismatch")
    if not any(marker in text for marker in markers):
        return None

    parts: list[str] = []
    for marker in markers:
        index = text.find(marker)
        if index < 0:
            continue
        segment = text[index : index + 900]
        line = segment.splitlines()[0] if "\n" in segment else segment
        count_match = re.search(rf"{re.escape(marker)}\s*\((\d+)\)", line)
        if count_match:
            parts.append(f"{marker}: {count_match.group(1)}")
        else:
            parts.append(_shorten(line, 300))
    return "; ".join(parts)


def _is_expected_sam31_multiplex_loader_warning(summary: str) -> bool:
    """Meta's SAM3.1 builder logs a noisy tracker-only preload warning.

    The merged `sam3.1_multiplex.pt` checkpoint is first loaded into the
    tracker-only submodel and then loaded again into the assembled detector +
    tracker model. That first internal load reports unprefixed tracker keys as
    missing and `tracker.model.*`/`detector.*` keys as unexpected even when the
    final assembled model load succeeds.
    """

    return (
        "Missing keys" in summary
        and "Unexpected keys" in summary
        and "tracker.model." in summary
        and "detector." in summary
    )


def _shorten(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _select_cuda_device(gpu_index: int | None) -> int:
    try:
        import torch
    except Exception as exc:
        raise Sam3AdapterError("PyTorch is required for SAM 3.1 inference") from exc
    if not torch.cuda.is_available():
        raise Sam3AdapterError("SAM 3.1 video inference requires a CUDA GPU")
    count = torch.cuda.device_count()
    if count <= 0:
        raise Sam3AdapterError("no CUDA GPUs are visible to PyTorch")
    selected = int(gpu_index or 0)
    if selected < 0 or selected >= count:
        raise Sam3AdapterError(
            f"task was assigned gpu-{selected}, but only {count} CUDA device(s) exist"
        )
    torch.cuda.set_device(selected)
    return selected


def _empty_annotation_rows(frames: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        width, height = _image_size(frame)
        rows.append(
            {
                "file_name": frame.name,
                "width": width,
                "height": height,
                "objects": [],
            }
        )
    return rows


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def _replace_prompt_objects(
    *,
    rows: list[dict[str, Any]],
    frame_index: int,
    outputs: dict[str, Any],
    prompt: str,
    prompt_index: int,
    class_name: str,
    include_masks: bool,
) -> None:
    if frame_index < 0 or frame_index >= len(rows):
        return
    row = rows[frame_index]
    row["objects"] = [
        obj for obj in row.get("objects", []) if obj.get("prompt") != prompt
    ]
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if width <= 0 or height <= 0:
        return
    row["objects"].extend(
        _objects_from_outputs(
            outputs=outputs,
            prompt=prompt,
            prompt_index=prompt_index,
            class_name=class_name,
            width=width,
            height=height,
            include_masks=include_masks,
        )
    )


def _objects_from_outputs(
    *,
    outputs: dict[str, Any],
    prompt: str,
    prompt_index: int,
    class_name: str,
    width: int,
    height: int,
    include_masks: bool,
) -> list[dict[str, Any]]:
    obj_ids = _to_list(_first_present(outputs, "out_obj_ids", "obj_ids"))
    boxes = _to_list(_first_present(outputs, "out_boxes_xywh", "boxes"))
    scores = _to_list(_first_present(outputs, "out_probs", "scores"))
    masks = _normalise_masks(_first_present(outputs, "out_binary_masks", "masks"))
    count = max(len(obj_ids), len(boxes), len(scores), len(masks))
    objects: list[dict[str, Any]] = []
    for index in range(count):
        mask = masks[index] if index < len(masks) else None
        if mask is not None:
            mask = _resize_mask(mask, width=width, height=height)
            if not _mask_has_pixels(mask):
                continue
        bbox = _absolute_bbox(
            boxes[index] if index < len(boxes) else None,
            width=width,
            height=height,
            mask=mask,
        )
        if bbox is None:
            continue
        raw_obj_id = _scalar(obj_ids[index]) if index < len(obj_ids) else index
        score = _scalar(scores[index]) if index < len(scores) else None
        obj: dict[str, Any] = {
            "object_id": f"prompt{prompt_index}_{raw_obj_id}",
            "sam_object_id": raw_obj_id,
            "prompt": prompt,
            "class_name": class_name,
            "confidence": float(score) if score is not None else None,
            "bbox": [round(value, 3) for value in bbox],
            "bbox_format": "xywh_abs",
        }
        if include_masks and mask is not None:
            obj["segmentation"] = _rle_from_mask(mask)
            obj["area"] = int(_mask_area(mask))
        objects.append(obj)
    return objects


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [value.item()]
            return list(value)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalise_masks(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np

        array = np.asarray(value)
        if array.ndim == 2:
            array = array[None, :, :]
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0, :, :]
        return [array[index] for index in range(array.shape[0])]
    except Exception:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]


def _resize_mask(mask: Any, *, width: int, height: int) -> Any:
    try:
        import numpy as np

        array = np.asarray(mask)
        if array.shape == (height, width):
            return array > 0
        from PIL import Image

        image = Image.fromarray((array > 0).astype("uint8") * 255)
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        image = image.resize((width, height), resample=resampling)
        return np.asarray(image) > 0
    except Exception:
        return mask


def _mask_has_pixels(mask: Any) -> bool:
    try:
        import numpy as np

        return bool(np.asarray(mask).any())
    except Exception:
        return False


def _mask_area(mask: Any) -> int:
    try:
        import numpy as np

        return int(np.asarray(mask).astype(bool).sum())
    except Exception:
        return 0


def _absolute_bbox(
    box: Any,
    *,
    width: int,
    height: int,
    mask: Any | None,
) -> tuple[float, float, float, float] | None:
    if box is not None:
        values = [_as_float(value) for value in _to_list(box)]
        if len(values) >= 4 and all(value is not None for value in values[:4]):
            x, y, w, h = [float(value) for value in values[:4]]
            if max(abs(x), abs(y), abs(w), abs(h)) <= 2.0:
                x *= width
                w *= width
                y *= height
                h *= height
            return _clamp_bbox(x, y, w, h, width=width, height=height)
    if mask is None:
        return None
    return _bbox_from_mask(mask, width=width, height=height)


def _bbox_from_mask(
    mask: Any, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    try:
        import numpy as np

        array = np.asarray(mask).astype(bool)
        ys, xs = np.where(array)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0 = float(xs.min())
        y0 = float(ys.min())
        x1 = float(xs.max() + 1)
        y1 = float(ys.max() + 1)
        return _clamp_bbox(x0, y0, x1 - x0, y1 - y0, width=width, height=height)
    except Exception:
        return None


def _clamp_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _rle_from_mask(mask: Any) -> dict[str, Any]:
    try:
        import numpy as np

        array = np.asarray(mask).astype("uint8")
        height, width = array.shape[:2]
        flat = array.T.reshape(-1)
        counts: list[int] = []
        last = 0
        run_length = 0
        for value in flat:
            bit = 1 if value else 0
            if bit == last:
                run_length += 1
            else:
                counts.append(run_length)
                run_length = 1
                last = bit
        counts.append(run_length)
        return {
            "format": "rle",
            "encoding": "uncompressed_coco_rle",
            "size": [int(height), int(width)],
            "counts": counts,
        }
    except Exception as exc:
        raise Sam3AdapterError("failed to encode SAM mask as RLE") from exc


def _scalar(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return value.tolist()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 1:
                return value.reshape(-1)[0].item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _as_float(value: Any) -> float | None:
    value = _scalar(value)
    try:
        return float(value)
    except Exception:
        return None


def _raise_if_cancelled(is_cancelled: CancelChecker | None) -> None:
    if is_cancelled and is_cancelled():
        raise RuntimeError("inference cancelled")


def _progress(
    progress: ProgressCallback | None,
    percent: float,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    if progress:
        progress(percent, message, metrics)


def _interpolate(start: float, end: float, current: int, total: int) -> float:
    return start + (min(total, current) / max(1, total)) * (end - start)
