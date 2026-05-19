from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._all: set[WebSocket] = set()
        self._projects: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._all.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._all.discard(websocket)
            for clients in self._projects.values():
                clients.discard(websocket)

    async def bind_project(self, websocket: WebSocket | None, project_id: str) -> None:
        if websocket is None:
            return
        async with self._lock:
            self._all.add(websocket)
            self._projects[project_id].add(websocket)

    async def send_to_project(
        self, project_id: str | None, message: dict[str, Any]
    ) -> None:
        async with self._lock:
            clients = (
                set(self._projects.get(project_id, set()))
                if project_id
                else set(self._all)
            )
        await self._send_many(clients, message)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            clients = set(self._all)
        await self._send_many(clients, message)

    async def _send_many(
        self, clients: set[WebSocket], message: dict[str, Any]
    ) -> None:
        stale: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            await self.disconnect(websocket)
