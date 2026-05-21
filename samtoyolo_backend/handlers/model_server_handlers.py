from __future__ import annotations

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..registry import HandlerContext, registry
from .common import object_params


@registry.method("get_model_server_list", aliases=("model_servers.list",))
async def handle_get_model_server_list(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    object_params(params)
    return ctx.app.state.model_server_manager.status()


@registry.method("get_model_server_status", aliases=("model_servers.status",))
async def handle_get_model_server_status(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    name = data.get("name") or data.get("model")
    if not isinstance(name, str) or not name:
        raise JsonRpcError(INVALID_PARAMS, "name is required")
    try:
        return ctx.app.state.model_server_manager.status(name)
    except KeyError as exc:
        raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc


@registry.method("setup_model_server", aliases=("model_servers.setup",))
async def handle_setup_model_server(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    name = data.get("name") or data.get("model")
    if not isinstance(name, str) or not name:
        raise JsonRpcError(INVALID_PARAMS, "name is required")
    try:
        return await ctx.app.state.model_server_manager.ensure_running(name)
    except KeyError as exc:
        raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc


@registry.method("restart_model_server", aliases=("model_servers.restart",))
async def handle_restart_model_server(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    data = object_params(params)
    name = data.get("name") or data.get("model")
    if not isinstance(name, str) or not name:
        raise JsonRpcError(INVALID_PARAMS, "name is required")
    try:
        return await ctx.app.state.model_server_manager.restart(name)
    except KeyError as exc:
        raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
