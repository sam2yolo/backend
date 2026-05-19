from __future__ import annotations

from ..registry import HandlerContext, registry
from .common import object_params


@registry.method("get_tunnel_status", aliases=("tunnel.status",))
async def handle_get_tunnel_status(ctx: HandlerContext, params: object) -> dict[str, object]:
    object_params(params)
    return ctx.app.state.tunnel_manager.status()


@registry.method("restart_tunnel", aliases=("tunnel.restart",))
async def handle_restart_tunnel(ctx: HandlerContext, params: object) -> dict[str, object]:
    object_params(params)
    state = await ctx.app.state.tunnel_manager.restart()
    return ctx.app.state.tunnel_manager.status() | {"endpoint": state.endpoint}
