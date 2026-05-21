from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings


@dataclass(frozen=True, slots=True)
class ModelServerSpec:
    name: str
    display_name: str
    setup_script: Path
    run_script: Path
    local_http_url: str
    local_ws_url: str
    peer_name: str
    capabilities: tuple[str, ...]


@dataclass(slots=True)
class ModelServerState:
    name: str
    display_name: str
    local_http_url: str
    local_ws_url: str
    public_http_url: str | None = None
    public_ws_url: str | None = None
    tunnel_registered: bool = False
    setup_complete: bool = False
    server_running: bool = False
    tunnel_running: bool = False
    last_error: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "local_http_url": self.local_http_url,
            "local_ws_url": self.local_ws_url,
            "public_http_url": self.public_http_url,
            "public_ws_url": self.public_ws_url,
            "tunnel_registered": self.tunnel_registered,
            "setup_complete": self.setup_complete,
            "server_running": self.server_running,
            "tunnel_running": self.tunnel_running,
            "last_error": self.last_error,
            "capabilities": list(self.capabilities),
            "metadata": self.metadata,
        }


class ModelServerManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        repo_root = Path(__file__).resolve().parents[1]
        sam3_port = settings.sam3_model_server_port
        self._specs = {
            "sam3": ModelServerSpec(
                name="sam3",
                display_name="Meta SAM 3.1",
                setup_script=repo_root / "model_servers" / "sam3" / "setup.sh",
                run_script=repo_root / "model_servers" / "sam3" / "run.sh",
                local_http_url=f"http://127.0.0.1:{sam3_port}",
                local_ws_url=f"ws://127.0.0.1:{sam3_port}/v1/ws",
                peer_name=f"{settings.server_name}-sam3",
                capabilities=("inference", "training"),
            )
        }
        self._states = {
            name: ModelServerState(
                name=spec.name,
                display_name=spec.display_name,
                local_http_url=spec.local_http_url,
                local_ws_url=spec.local_ws_url,
                capabilities=spec.capabilities,
            )
            for name, spec in self._specs.items()
        }
        self._server_processes: dict[str, asyncio.subprocess.Process] = {}
        self._tunnel_processes: dict[str, asyncio.subprocess.Process] = {}

    async def start(self) -> None:
        if not self.settings.model_servers_auto_start:
            return
        for name in self._specs:
            await self.ensure_running(name)

    async def stop(self) -> None:
        for process in [*self._tunnel_processes.values(), *self._server_processes.values()]:
            if process.returncode is None:
                process.terminate()
        for process in [*self._tunnel_processes.values(), *self._server_processes.values()]:
            if process.returncode is None:
                await process.wait()

    async def ensure_running(self, name: str) -> dict[str, Any]:
        spec = self._require_spec(name)
        state = self._states[name]
        try:
            await self._setup(spec, state)
            await self._start_server(spec, state)
            if self.settings.is_remote or self.settings.model_servers_public_tunnel:
                await self._start_tunnel(spec, state)
                await self._register_tunnel(spec, state)
            state.last_error = None
        except Exception as exc:
            state.last_error = str(exc)
            raise
        return state.to_dict()

    async def restart(self, name: str) -> dict[str, Any]:
        for mapping in (self._tunnel_processes, self._server_processes):
            process = mapping.pop(name, None)
            if process and process.returncode is None:
                process.terminate()
                await process.wait()
        state = self._states[name]
        state.server_running = False
        state.tunnel_running = False
        state.public_http_url = None
        state.public_ws_url = None
        state.tunnel_registered = False
        return await self.ensure_running(name)

    def status(self, name: str | None = None) -> dict[str, Any]:
        if name:
            self._require_spec(name)
            return self._states[name].to_dict()
        return {"model_servers": [state.to_dict() for state in self._states.values()]}

    def endpoint_for(self, name: str) -> str:
        self._require_spec(name)
        state = self._states[name]
        return state.public_ws_url or state.local_ws_url

    def _require_spec(self, name: str) -> ModelServerSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown model server: {name}") from exc

    async def _setup(self, spec: ModelServerSpec, state: ModelServerState) -> None:
        if state.setup_complete:
            return
        process = await asyncio.create_subprocess_exec(
            str(spec.setup_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"{spec.name} setup failed: "
                f"{stderr.decode(errors='replace') or stdout.decode(errors='replace')}"
            )
        state.setup_complete = True

    async def _start_server(
        self, spec: ModelServerSpec, state: ModelServerState
    ) -> None:
        process = self._server_processes.get(spec.name)
        if process and process.returncode is None and await self._health(spec.local_http_url):
            state.server_running = True
            return
        process = await asyncio.create_subprocess_exec(
            str(spec.run_script),
            env={**os.environ, "SAMTOYOLO_SKIP_MODEL_SERVER_SETUP": "1"},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._server_processes[spec.name] = process
        for _ in range(30):
            if process.returncode is not None:
                stderr = await self._read_process_tail(process)
                raise RuntimeError(f"{spec.name} server exited early: {stderr}")
            if await self._health(spec.local_http_url):
                state.server_running = True
                return
            await asyncio.sleep(1)
        raise RuntimeError(f"{spec.name} server did not become healthy")

    async def _start_tunnel(
        self, spec: ModelServerSpec, state: ModelServerState
    ) -> None:
        process = self._tunnel_processes.get(spec.name)
        if process and process.returncode is None and state.public_http_url:
            state.tunnel_running = True
            return
        process = await asyncio.create_subprocess_exec(
            self.settings.cloudflared_path,
            "tunnel",
            "--url",
            spec.local_http_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._tunnel_processes[spec.name] = process
        endpoint = await self._read_cloudflared_endpoint(process, timeout=30)
        if not endpoint:
            raise RuntimeError(f"cloudflared did not report endpoint for {spec.name}")
        state.public_http_url = endpoint
        state.public_ws_url = endpoint.replace("https://", "wss://") + "/v1/ws"
        state.tunnel_running = True

    async def _register_tunnel(
        self, spec: ModelServerSpec, state: ModelServerState
    ) -> None:
        if not state.public_http_url:
            return
        if not (
            self.settings.tunnelbroker_url
            and self.settings.tunnelbroker_group
            and self.settings.peer_secret
        ):
            state.last_error = "tunnelbroker settings are incomplete"
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
            "peer": spec.peer_name,
            "secret": self.settings.peer_secret,
            "endpoint": state.public_http_url,
            "metadata": {
                "app": "samtoyolo-model-server",
                "model": spec.name,
                "display_name": spec.display_name,
                "capabilities": list(spec.capabilities),
                "ws_endpoint": state.public_ws_url,
                "main_server": self.settings.server_name,
            },
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        state.tunnel_registered = True

    async def _health(self, base_url: str) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{base_url.rstrip('/')}/health")
            return response.status_code == 200 and response.json().get("ok") is True
        except Exception:
            return False

    async def _read_cloudflared_endpoint(
        self, process: asyncio.subprocess.Process, *, timeout: int
    ) -> str | None:
        if process.stderr is None:
            return None

        async def read_loop() -> str | None:
            while True:
                line = await process.stderr.readline()
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

    async def _read_process_tail(self, process: asyncio.subprocess.Process) -> str:
        chunks = []
        for stream in (process.stdout, process.stderr):
            if stream:
                try:
                    chunks.append((await stream.read()).decode(errors="replace"))
                except Exception:
                    pass
        return "\n".join(chunks)[-2000:]
