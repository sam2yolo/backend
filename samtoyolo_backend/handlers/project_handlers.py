from __future__ import annotations

from ..registry import HandlerContext, registry
from .common import bind_project, object_params, optional_str, required_str


@registry.method("create_project", aliases=("project.create",))
async def handle_create_project(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    display_name = optional_str(data, "display_name")
    info = ctx.store.ensure_project(project_id, display_name=display_name)
    await bind_project(ctx, project_id)
    return info


@registry.method("get_project_list", aliases=("project.list",))
async def handle_get_project_list(ctx: HandlerContext, params: object) -> dict[str, object]:
    object_params(params)
    return {"projects": ctx.store.list_projects()}


@registry.method("get_project_info", aliases=("project.info",))
async def handle_get_project_info(ctx: HandlerContext, params: object) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    session = ctx.store.get_session(project_id)
    info = ctx.store.project_info(project_id)
    info["uploads"] = list(session.get("uploads", {}).values())
    info["datasets"] = list(session.get("datasets", {}).values())
    info["models"] = list(session.get("models", {}).values())
    info["inference_results"] = list(session.get("inference_results", {}).values())
    return info
