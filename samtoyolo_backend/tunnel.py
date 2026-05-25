from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from . import events
from .cloudflared import ensure_cloudflared_path
from .config import Settings
from .connections import ConnectionManager


@dataclass(slots=True)
class TunnelState:
    mode: str
    endpoint: str | None = None
    registered: bool = False
    last_error: str | None = None
    cloudflared_running: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TunnelManager:
    def __init__(self, settings: Settings, connections: ConnectionManager) -> None:
        self.settings = settings
        self.connections = connections
        self.state = TunnelState(mode=settings.mode)
        self._process: asyncio.subprocess.Process | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self.settings.is_remote:
            return
        if self._runner is None or self._runner.done():
            self._stop.clear()
            self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._runner:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()

    async def restart(self) -> TunnelState:
        if not self.settings.is_remote:
            self.state.last_error = "tunnel is disabled in local mode"
            return self.state
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
        self.state.endpoint = None
        self.state.registered = False
        await self._ensure_tunnel()
        await self._register_peer()
        return self.state

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._ensure_tunnel()
                await self._register_peer()
            except Exception as exc:
                self.state.last_error = str(exc)
                await events.notify_server_notification(
                    self.connections,
                    level="warning",
                    message=f"tunnel heartbeat failed: {exc}",
                )
            await asyncio.sleep(self.settings.tunnel_heartbeat_seconds)

    async def _ensure_tunnel(self) -> None:
        if self._process and self._process.returncode is None and self.state.endpoint:
            self.state.cloudflared_running = True
            return

        self.state.cloudflared_running = False
        self.state.endpoint = None
        cloudflared_path = await asyncio.to_thread(
            ensure_cloudflared_path, self.settings.cloudflared_path
        )
        self._process = await asyncio.create_subprocess_exec(
            cloudflared_path,
            "tunnel",
            "--url",
            self.settings.http_bind_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        endpoint = await self._read_cloudflared_endpoint(timeout=30)
        if not endpoint:
            raise RuntimeError("cloudflared did not report a public endpoint")
        self.state.endpoint = endpoint
        self.state.cloudflared_running = True
        await events.notify_tunnel_ready(
            self.connections, endpoint=endpoint, server_name=self.settings.server_name
        )

    async def _read_cloudflared_endpoint(self, *, timeout: int) -> str | None:
        if self._process is None or self._process.stderr is None:
            return None

        async def read_loop() -> str | None:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    return None
                text = line.decode(errors="replace")
                match = re.search(r"https://[-a-zA-Z0-9.]+trycloudflare\.com", text)
                if match:
                    return match.group(0)

        try:
            return await asyncio.wait_for(read_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _register_peer(self) -> None:
        if not self.state.endpoint:
            return
        if not (
            self.settings.tunnelbroker_url
            and self.settings.tunnelbroker_group
            and self.settings.peer_secret
        ):
            self.state.last_error = "tunnelbroker settings are incomplete"
            return

        try:
            import httpx
        except Exception as exc:
            raise RuntimeError("httpx is required for tunnelbroker registration") from exc

        url = (
            f"{self.settings.tunnelbroker_url.rstrip('/')}/v1/peers"
            f"?group={self.settings.tunnelbroker_group}"
        )
        headers = {"content-type": "application/json"}
        if self.settings.tunnelbroker_group_token:
            headers["authorization"] = f"Bearer {self.settings.tunnelbroker_group_token}"
        payload = {
            "peer": self.settings.server_name,
            "secret": self.settings.peer_secret,
            "endpoint": self.state.endpoint,
            "metadata": {
                "mode": self.settings.mode,
                "app": "samtoyolo-backend",
            },
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        self.state.registered = True
        self.state.last_error = None

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.state.mode,
            "endpoint": self.state.endpoint,
            "registered": self.state.registered,
            "last_error": self.state.last_error,
            "cloudflared_running": self.state.cloudflared_running,
            "server_name": self.settings.server_name,
        }
