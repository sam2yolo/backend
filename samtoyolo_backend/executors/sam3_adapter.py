from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..model_server_client import call_model_server


ProgressCallback = Callable[[float, str, dict[str, Any] | None], None]
CancelChecker = Callable[[], bool]


class Sam3AdapterError(RuntimeError):
    """Raised when the SAM 3.1 model server cannot perform inference."""


@dataclass(slots=True)
class Sam3InferenceResult:
    frames: list[dict[str, Any]]
    metadata: dict[str, Any]


def run_sam3_video_text_inference(
    *,
    server_url: str,
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
    """Run SAM 3.1 through its isolated model server."""

    if is_cancelled and is_cancelled():
        raise RuntimeError("inference cancelled")

    def relay_progress(
        percent: float,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if is_cancelled and is_cancelled():
            raise RuntimeError("inference cancelled")
        if progress:
            progress(percent, message, metrics)

    result = call_model_server(
        url=server_url,
        method="sam3.infer_video_text",
        params={
            "frames_dir": str(frames_dir),
            "frames": [str(frame) for frame in frames],
            "prompts": prompts,
            "prompt_to_class": prompt_to_class,
            "model_extract_dir": str(model_extract_dir),
            "gpu_index": gpu_index,
            "output_prob_thresh": output_prob_thresh,
            "max_num_objects": max_num_objects,
            "multiplex_count": multiplex_count,
            "use_fa3": use_fa3,
            "compile_model": compile_model,
            "warm_up": warm_up,
            "async_loading_frames": async_loading_frames,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "cache_model": cache_model,
            "allow_partial_checkpoint": allow_partial_checkpoint,
            "include_masks": include_masks,
            "progress_start": progress_start,
            "progress_end": progress_end,
        },
        progress=relay_progress,
    )
    if not isinstance(result, dict):
        raise Sam3AdapterError("SAM 3.1 model server returned a non-object result")

    frames_result = result.get("frames")
    metadata = result.get("metadata")
    if not isinstance(frames_result, list) or not isinstance(metadata, dict):
        raise Sam3AdapterError(
            "SAM 3.1 model server result must contain frames[] and metadata{}"
        )
    metadata = {
        **metadata,
        "model_server_url": server_url,
        "model_server_method": "sam3.infer_video_text",
    }
    return Sam3InferenceResult(frames=frames_result, metadata=metadata)
