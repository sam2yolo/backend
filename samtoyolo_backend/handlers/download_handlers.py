from __future__ import annotations

from urllib.parse import quote

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, required_str


@registry.method(
    "download_inference_result",
    aliases=("getInferenceResult", "get_inference_result"),
)
async def handle_download_inference_result(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    task_id = required_str(data, "task_id")
    project_id = ctx.store.get_task_project(task_id)
    if not project_id:
        raise JsonRpcError(INVALID_PARAMS, f"unknown task_id: {task_id}")
    await bind_project(ctx, project_id)
    session = ctx.store.get_session(project_id)
    result = session.get("inference_results", {}).get(task_id)
    if not result:
        raise JsonRpcError(INVALID_PARAMS, f"inference result is not ready: {task_id}")
    delete_after = bool(data.get("delete_after_download", False))
    suffix = "?delete_after_download=true" if delete_after else ""
    return {
        "project_id": project_id,
        "task_id": task_id,
        "download_url": (
            f"/v1/projects/{quote(project_id)}/downloads/inference/{quote(task_id)}{suffix}"
        ),
        "result": result,
    }


@registry.method("download_model", aliases=("getModel", "get_model"))
async def handle_download_model(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    model_id = required_str(data, "model_id")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    session = ctx.store.get_session(project_id)
    model = session.get("models", {}).get(model_id)
    if not model:
        raise JsonRpcError(INVALID_PARAMS, f"unknown model_id: {model_id}")
    return {
        "project_id": project_id,
        "model_id": model_id,
        "download_url": f"/v1/projects/{quote(project_id)}/downloads/models/{quote(model_id)}",
        "model": model,
    }


@registry.method("download_dataset", aliases=("getDataset", "get_dataset"))
async def handle_download_dataset(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    dataset_id = required_str(data, "dataset_id")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    session = ctx.store.get_session(project_id)
    dataset = session.get("datasets", {}).get(dataset_id)
    if not dataset:
        raise JsonRpcError(INVALID_PARAMS, f"unknown dataset_id: {dataset_id}")
    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "download_url": f"/v1/projects/{quote(project_id)}/downloads/datasets/{quote(dataset_id)}",
        "dataset": dataset,
    }
