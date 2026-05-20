from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .runtime_env import ensure_runtime_environment

ensure_runtime_environment()

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.websockets import WebSocketDisconnect

from . import events
from .config import Settings
from .connections import ConnectionManager
from .executors import register_default_executors
from .jsonrpc import PARSE_ERROR, dispatch_payload, error_response
from .records import TaskType, new_id
from .registry import HandlerContext, registry
from .storage import ProjectStore, StorageError
from .tasks import TaskManager
from .tunnel import TunnelManager

# Import side effect: registers JSON-RPC methods via decorators.
from . import handlers as handlers  # noqa: F401


UPLOAD_KINDS = {"video", "image", "dataset", "file"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    connections = ConnectionManager()
    store = ProjectStore(settings.project_root)
    task_manager = TaskManager(
        store=store,
        connections=connections,
        gpu_workers=settings.resolved_gpu_workers(),
    )
    register_default_executors(task_manager)
    tunnel_manager = TunnelManager(settings, connections)

    app.state.settings = settings
    app.state.connections = connections
    app.state.store = store
    app.state.task_manager = task_manager
    app.state.tunnel_manager = tunnel_manager
    app.state.mega_credentials = {}

    await task_manager.start()
    await tunnel_manager.start()
    expiry_monitor = asyncio.create_task(_session_expiry_monitor(settings, connections))
    app.state.expiry_monitor = expiry_monitor
    try:
        yield
    finally:
        expiry_monitor.cancel()
        await asyncio.gather(expiry_monitor, return_exceptions=True)
        await tunnel_manager.stop()
        await task_manager.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SAM-to-YOLO Transfer Learning Backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "service": "samtoyolo-backend"}

    @app.get("/v1/methods")
    async def list_methods() -> dict[str, Any]:
        return {"methods": registry.describe()}

    @app.websocket("/v1/ws")
    async def websocket_rpc(websocket: WebSocket) -> None:
        await app.state.connections.connect(websocket)
        try:
            while True:
                raw_message = await websocket.receive_text()
                try:
                    payload = json.loads(raw_message)
                except json.JSONDecodeError as exc:
                    await websocket.send_json(
                        error_response(None, PARSE_ERROR, "invalid JSON", str(exc))
                    )
                    continue
                context = HandlerContext(app=app, websocket=websocket, request_id=None)
                response = await dispatch_payload(
                    payload,
                    context=context,
                    registry=registry,
                )
                if response is not None:
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            await app.state.connections.disconnect(websocket)
        except Exception:
            await app.state.connections.disconnect(websocket)
            raise

    @app.post("/v1/projects/{project_id}/uploads/{kind}")
    async def upload_project_file(
        project_id: str,
        kind: str,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        if kind not in UPLOAD_KINDS:
            raise HTTPException(
                status_code=400,
                detail=f"kind must be one of: {', '.join(sorted(UPLOAD_KINDS))}",
            )
        try:
            app.state.store.ensure_project(project_id)
        except StorageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        filename = _safe_upload_name(file.filename or "upload.bin")
        target_dir = app.state.store.project_path(project_id) / "uploads" / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = _unique_path(target_dir / filename)
        size_bytes = 0
        with target_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                handle.write(chunk)

        relpath = app.state.store.relative_to_project(project_id, target_path)
        task = await app.state.task_manager.submit(
            project_id=project_id,
            task_type=TaskType.UPLOAD_FILE.value,
            params={
                "kind": kind,
                "stored_path": relpath,
                "filename": target_path.name,
                "size_bytes": size_bytes,
                "upload_id": new_id("upload"),
            },
            description=f"Register uploaded {kind}",
        )
        return {
            "task_id": task.task_id,
            "status": task.status,
            "filename": target_path.name,
            "path": relpath,
            "size_bytes": size_bytes,
        }

    @app.get("/v1/projects/{project_id}/downloads/inference/{task_id}")
    async def download_inference_result(
        project_id: str,
        task_id: str,
        delete_after_download: bool = Query(False),
    ) -> FileResponse:
        session = _session_or_404(app, project_id)
        result = session.get("inference_results", {}).get(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="inference result not found")
        path = app.state.store.resolve_project_file(project_id, result["zip_path"])
        return _file_response(
            path,
            delete_after_download=delete_after_download,
            cleanup=lambda: app.state.store.mutate_session(
                project_id,
                lambda session: session.get("inference_results", {}).pop(task_id, None),
            ),
        )

    @app.get("/v1/projects/{project_id}/downloads/datasets/{dataset_id}")
    async def download_dataset(project_id: str, dataset_id: str) -> FileResponse:
        session = _session_or_404(app, project_id)
        dataset = session.get("datasets", {}).get(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="dataset not found")
        path = app.state.store.resolve_project_file(project_id, dataset["zip_path"])
        return _file_response(path)

    @app.get("/v1/projects/{project_id}/downloads/models/{model_id}")
    async def download_model(project_id: str, model_id: str) -> FileResponse:
        session = _session_or_404(app, project_id)
        model = session.get("models", {}).get(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="model not found")
        path = app.state.store.resolve_project_file(project_id, model["path"])
        return _file_response(path)

    return app


def _session_or_404(app: FastAPI, project_id: str) -> dict[str, Any]:
    try:
        app.state.store.ensure_project(project_id)
        return app.state.store.get_session(project_id)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _file_response(
    path: Path,
    *,
    delete_after_download: bool = False,
    cleanup: Any = None,
) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")

    background = None
    if delete_after_download:
        background = BackgroundTask(_delete_file_and_cleanup, path, cleanup)
    return FileResponse(path, filename=path.name, background=background)


def _delete_file_and_cleanup(path: Path, cleanup: Any = None) -> None:
    try:
        path.unlink(missing_ok=True)
    finally:
        if cleanup:
            cleanup()


def _safe_upload_name(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "upload.bin"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=500, detail="could not create unique upload path")


async def _session_expiry_monitor(
    settings: Settings, connections: ConnectionManager
) -> None:
    if settings.instance_ttl_seconds <= 0:
        return
    started = time.monotonic()
    emitted = False
    while not emitted:
        elapsed = int(time.monotonic() - started)
        remaining = settings.instance_ttl_seconds - elapsed
        if remaining <= settings.expiry_notice_seconds:
            await events.notify_session_expiring(
                connections, remaining_seconds=max(0, remaining)
            )
            emitted = True
            break
        await asyncio.sleep(min(60, max(1, remaining - settings.expiry_notice_seconds)))


app = create_app()
