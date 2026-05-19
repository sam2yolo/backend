from __future__ import annotations

from ..records import TaskType
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, required_str


@registry.method("train_model", aliases=("train",))
async def handle_train_model(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    model_name = required_str(data, "model_name")
    dataset_id = required_str(data, "dataset_id")
    config = data.get("config") or {}
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.TRAIN_MODEL.value,
        params={
            "model_name": model_name,
            "dataset_id": dataset_id,
            "config": config,
            "allow_stub_ml": ctx.settings.allow_stub_ml,
        },
        description=data.get("description") or f"Train {model_name}",
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("train_yolo", aliases=("train-yolo",))
async def handle_train_yolo(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    data = {**data, "model_name": data.get("model_name") or "yolov8"}
    return await handle_train_model(ctx, data)
