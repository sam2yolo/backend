from __future__ import annotations

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..model_sources import google_drive_download_url
from ..registry import HandlerContext, registry
from .common import object_params, required_str


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
    if not all(isinstance(value, str) for value in prompt_to_class.values()):
        raise JsonRpcError(INVALID_PARAMS, "prompt_to_class values must be strings")
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
    server = await ctx.app.state.model_server_manager.ensure_running("sam3")
    return {
        "delegated": True,
        "message": (
            "SAM 3.1 inference runs on the SAM3 model server; call the "
            "returned WebSocket endpoint."
        ),
        "model": "sam3",
        "model_server": server,
        "rpc_method": "inference_sam3",
        "compat_rpc_method": "sam3.infer_video_text",
        "params": {
            "project_id": project_id,
            "media_path": media_path,
            "prompts": prompts,
            "prompt_to_class": prompt_to_class,
            "model_source_url": model_source_url,
            "model_download_url": model_download_url,
            "model_filename": model_filename,
        },
    }


@registry.method("inference_yolo", aliases=("inference.yolo",))
async def handle_inference_yolo(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    media_path = required_str(data, "media_path")
    return {
        "delegated": True,
        "message": (
            "YOLO inference must be called on a YOLO model server. No YOLO "
            "model server is registered yet."
        ),
        "model": "yolov8",
        "project_id": project_id,
        "media_path": media_path,
    }
