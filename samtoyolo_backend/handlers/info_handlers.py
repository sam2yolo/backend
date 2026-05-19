from __future__ import annotations

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, optional_str, required_str


@registry.method("get_task_list", aliases=("task.list",))
async def handle_get_task_list(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = optional_str(data, "project_id")
    if project_id:
        ctx.store.ensure_project(project_id)
        await bind_project(ctx, project_id)
    return {"tasks": ctx.store.list_tasks(project_id)}


@registry.method("get_task_status", aliases=("task.status",))
async def handle_get_task_status(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    task_id = required_str(data, "task_id")
    task = ctx.store.get_task(task_id)
    if task is None:
        raise JsonRpcError(INVALID_PARAMS, f"unknown task_id: {task_id}")
    await bind_project(ctx, task.project_id)
    return task.to_dict()
