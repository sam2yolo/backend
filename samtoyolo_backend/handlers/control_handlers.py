from __future__ import annotations

from .. import events
from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, required_str


@registry.method("cancel_task", aliases=("task.cancel",))
async def handle_cancel_task(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    task_id = required_str(data, "task_id")
    try:
        task = await ctx.task_manager.cancel(task_id)
    except KeyError as exc:
        raise JsonRpcError(INVALID_PARAMS, f"unknown task_id: {task_id}") from exc
    await bind_project(ctx, task.project_id)
    return task.to_dict()


@registry.method("client_response", aliases=("client.response",))
async def handle_client_response(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    request_id = data.get("requestId") or data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise JsonRpcError(INVALID_PARAMS, "requestId is required")
    status = data.get("status", "complete")
    project_id = data.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        for project in ctx.store.list_projects():
            session = ctx.store.get_session(project["project_id"])
            if request_id in session.get("client_requests", {}):
                project_id = project["project_id"]
                break
    if not isinstance(project_id, str) or not project_id:
        return {"requestId": request_id, "recorded": False, "reason": "request not found"}
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)

    def update(session: dict[str, object]) -> None:
        requests = session.setdefault("client_requests", {})
        if isinstance(requests, dict) and request_id in requests:
            request = requests[request_id]
            if isinstance(request, dict):
                request["status"] = status
                request["response"] = data.get("data")

    ctx.store.mutate_session(project_id, update)
    return {"requestId": request_id, "recorded": True}


@registry.method("backend_init", aliases=("backend.init",))
async def handle_backend_init(ctx: HandlerContext, params: object) -> dict[str, object]:
    object_params(params)
    await events.notify_backend_init(ctx.connections)
    await events.notify_backend_init_progress(
        ctx.connections, percent=25, message="project storage ready"
    )
    await events.notify_backend_init_progress(
        ctx.connections, percent=60, message="task workers ready"
    )
    await events.notify_backend_init_progress(
        ctx.connections, percent=100, message="backend ready"
    )
    await events.notify_backend_ready(ctx.connections)
    return {
        "ready": True,
        "project_root": str(ctx.settings.project_root),
        "gpu_workers": ctx.task_manager.gpu_workers,
        "mode": ctx.settings.mode,
    }
