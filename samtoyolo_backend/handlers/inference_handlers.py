from __future__ import annotations

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..model_sources import google_drive_download_url
from ..records import TaskType
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, required_str


def _validate_prompt_mapping(prompts: list[object], prompt_to_class: dict[str, object]) -> None:
    missing = [prompt for prompt in prompts if prompt not in prompt_to_class]
    if missing:
        raise JsonRpcError(
            INVALID_PARAMS,
            "prompt_to_class must contain every prompt",
            {"missing_prompts": missing},
        )


@registry.method("inference_sam3", aliases=("inference.sam3",))
async def handle_inference_sam3(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    media_path = required_str(data, "media_path")
    prompts = data.get("prompts") or []
    prompt_to_class = data.get("prompt_to_class") or {}
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise JsonRpcError(INVALID_PARAMS, "prompts must be a list of strings")
    if not isinstance(prompt_to_class, dict):
        raise JsonRpcError(INVALID_PARAMS, "prompt_to_class must be an object")
    _validate_prompt_mapping(prompts, prompt_to_class)
    model_source_url = data.get("model_source_url") or ctx.settings.sam3_model_url
    if not isinstance(model_source_url, str) or not model_source_url:
        raise JsonRpcError(INVALID_PARAMS, "model_source_url must be a non-empty string")
    model_download_url = data.get("model_download_url") or google_drive_download_url(
        model_source_url
    )
    model_filename = data.get("model_filename") or ctx.settings.sam3_model_filename
    if not isinstance(model_filename, str) or not model_filename:
        raise JsonRpcError(INVALID_PARAMS, "model_filename must be a non-empty string")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.INFERENCE_SAM3.value,
        params={
            "media_path": media_path,
            "prompts": prompts,
            "prompt_to_class": prompt_to_class,
            "sample_interval_seconds": data.get("sample_interval_seconds"),
            "temporal_downsample": data.get("temporal_downsample"),
            "downsample": data.get("downsample"),
            "batch_size": data.get("batch_size", 4),
            "save_to_mega": bool(data.get("save_to_mega", False)),
            "allow_stub_ml": ctx.settings.allow_stub_ml,
            "prepare_model": bool(data.get("prepare_model", True)),
            "model_source_url": model_source_url,
            "model_download_url": model_download_url,
            "model_filename": model_filename,
            "model_cache_dir": str(ctx.settings.model_cache_dir),
        },
        description=data.get("description") or "Run SAM 3.1 text-prompt inference",
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("inference_yolo", aliases=("inference.yolo",))
async def handle_inference_yolo(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    media_path = required_str(data, "media_path")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.INFERENCE_YOLO.value,
        params={
            "media_path": media_path,
            "batch_size": data.get("batch_size", 4),
            "sample_interval_seconds": data.get("sample_interval_seconds"),
            "save_to_mega": bool(data.get("save_to_mega", False)),
            "allow_stub_ml": ctx.settings.allow_stub_ml,
        },
        description=data.get("description") or "Run YOLO detector inference",
    )
    return {"task_id": task.task_id, "status": task.status}
