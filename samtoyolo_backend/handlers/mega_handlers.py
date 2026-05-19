from __future__ import annotations

from .. import events
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, optional_str, required_str


@registry.method("check_mega_credentials", aliases=("mega.credentials.check",))
async def handle_check_mega_credentials(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    project_id = optional_str(data, "project_id")
    email = required_str(data, "email")
    password = required_str(data, "password")
    valid = bool(email and password)
    message = (
        "credentials are present; live Mega validation adapter is not configured"
        if valid
        else "email and password are required"
    )
    await events.notify_mega_credential_check_result(
        ctx.connections, project_id=project_id, valid=None if valid else False, message=message
    )
    return {"valid": None if valid else False, "message": message}


@registry.method(
    "register_mega_credentials",
    aliases=("register_mega_credential", "register_mega"),
)
async def handle_register_mega_credentials(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    email = required_str(data, "email")
    password = required_str(data, "password")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    if not hasattr(ctx.app.state, "mega_credentials"):
        ctx.app.state.mega_credentials = {}
    ctx.app.state.mega_credentials[project_id] = {"email": email, "password": password}
    ctx.store.set_mega_registered(project_id, True)
    return {
        "project_id": project_id,
        "registered": True,
        "stored": "memory_only",
    }
