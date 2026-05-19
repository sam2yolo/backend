from __future__ import annotations

from typing import Any

from ..jsonrpc import INVALID_PARAMS, JsonRpcError, require_object
from ..registry import HandlerContext


def object_params(params: Any) -> dict[str, Any]:
    return require_object(params)


def required_str(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise JsonRpcError(INVALID_PARAMS, f"{name} is required")
    return value


def optional_str(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise JsonRpcError(INVALID_PARAMS, f"{name} must be a non-empty string")
    return value


async def bind_project(ctx: HandlerContext, project_id: str) -> None:
    await ctx.connections.bind_project(ctx.websocket, project_id)
