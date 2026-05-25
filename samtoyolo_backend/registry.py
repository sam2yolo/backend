from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket

from .public_urls import join_public_url


HandlerResult = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonRpcHandler = Callable[["HandlerContext", Any], Awaitable[HandlerResult]]


@dataclass(slots=True)
class HandlerContext:
    app: Any
    websocket: WebSocket | None
    request_id: str | int | None

    @property
    def settings(self) -> Any:
        return self.app.state.settings

    @property
    def store(self) -> Any:
        return self.app.state.store

    @property
    def task_manager(self) -> Any:
        return self.app.state.task_manager

    @property
    def connections(self) -> Any:
        return self.app.state.connections

    @property
    def public_base_url(self) -> str | None:
        settings_url = getattr(self.settings, "public_base_url", None)
        tunnel_manager = getattr(self.app.state, "tunnel_manager", None)
        tunnel_url = getattr(getattr(tunnel_manager, "state", None), "endpoint", None)
        return settings_url or tunnel_url

    def public_url(self, path: str) -> str:
        return join_public_url(self.public_base_url, path)


class MethodRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, JsonRpcHandler] = {}
        self._canonical: dict[str, str] = {}

    def method(
        self, name: str, *, aliases: tuple[str, ...] = ()
    ) -> Callable[[JsonRpcHandler], JsonRpcHandler]:
        def decorator(func: JsonRpcHandler) -> JsonRpcHandler:
            self._handlers[name] = func
            self._canonical[name] = name
            for alias in aliases:
                self._handlers[alias] = func
                self._canonical[alias] = name
            return func

        return decorator

    def get(self, name: str) -> JsonRpcHandler | None:
        return self._handlers.get(name)

    def canonical_name(self, name: str) -> str | None:
        return self._canonical.get(name)

    def describe(self) -> list[dict[str, str]]:
        rows = []
        seen: set[str] = set()
        for public_name, canonical in sorted(self._canonical.items()):
            if public_name == canonical:
                seen.add(canonical)
                rows.append({"name": canonical, "canonical": canonical})
        for public_name, canonical in sorted(self._canonical.items()):
            if public_name != canonical and canonical in seen:
                rows.append({"name": public_name, "canonical": canonical})
        return rows


registry = MethodRegistry()
