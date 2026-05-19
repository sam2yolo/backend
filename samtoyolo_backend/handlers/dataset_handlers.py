from __future__ import annotations

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..records import TaskType
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, required_str


@registry.method(
    "convert_inference_to_yolo",
    aliases=("inference_to_dataset", "inference_to_yolo", "inference-to-yolo"),
)
async def handle_convert_inference_to_yolo(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    inference_task_id = required_str(data, "inference_task_id")
    target_format = data.get("target_format", "yolo")
    if target_format != "yolo":
        raise JsonRpcError(INVALID_PARAMS, "only target_format='yolo' is supported in v1")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.INFERENCE_TO_YOLO.value,
        params={"inference_task_id": inference_task_id, "target_format": target_format},
        description=data.get("description") or "Convert inference result to YOLO dataset",
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("merge_dataset", aliases=("dataset.merge",))
async def handle_merge_dataset(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    dataset_ids = data.get("dataset_ids")
    class_map = data.get("class_map") or {}
    if not isinstance(dataset_ids, list) or not all(
        isinstance(dataset_id, str) for dataset_id in dataset_ids
    ):
        raise JsonRpcError(INVALID_PARAMS, "dataset_ids must be a list of strings")
    if not isinstance(class_map, dict):
        raise JsonRpcError(INVALID_PARAMS, "class_map must be an object")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.MERGE_DATASET.value,
        params={"dataset_ids": dataset_ids, "class_map": class_map},
        description=data.get("description") or "Merge datasets",
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("map_dataset", aliases=("dataset.map",))
async def handle_map_dataset(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    dataset_id = required_str(data, "dataset_id")
    class_map = data.get("class_map") or {}
    if not isinstance(class_map, dict):
        raise JsonRpcError(INVALID_PARAMS, "class_map must be an object")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.MAP_DATASET.value,
        params={"dataset_id": dataset_id, "class_map": class_map},
        description=data.get("description") or "Map dataset classes",
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("list_datasets", aliases=("dataset.list",))
async def handle_list_datasets(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    session = ctx.store.get_session(project_id)
    return {"datasets": list(session.get("datasets", {}).values())}
