from __future__ import annotations

from ..registry import HandlerContext, registry
from .common import object_params, required_str


@registry.method("train_model", aliases=("train",))
async def handle_train_model(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    model_name = required_str(data, "model_name")
    dataset_id = required_str(data, "dataset_id")
    config = data.get("config") or {}
    server = None
    if model_name.lower().replace("-", "_") == "sam3":
        server = await ctx.app.state.model_server_manager.ensure_running("sam3")
    return {
        "delegated": True,
        "message": (
            "Training runs on the matching model server; call the returned "
            "model-server endpoint."
        ),
        "model_name": model_name,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "config": config,
        "model_server": server,
        "rpc_method": "train_model",
    }


@registry.method("train_yolo", aliases=("train-yolo",))
async def handle_train_yolo(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    data = {**data, "model_name": data.get("model_name") or "yolov8"}
    return await handle_train_model(ctx, data)
