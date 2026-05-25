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

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.websockets import WebSocketDisconnect

from . import events
from .config import Settings
from .connections import ConnectionManager
from .executors import register_default_executors
from .jsonrpc import PARSE_ERROR, dispatch_payload, error_response
from .model_servers import ModelServerManager
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
    tunnel_manager = TunnelManager(settings, connections)
    task_manager = TaskManager(
        store=store,
        connections=connections,
        gpu_workers=settings.resolved_gpu_workers(),
        public_base_url_getter=lambda: settings.public_base_url
        or tunnel_manager.state.endpoint,
    )
    register_default_executors(task_manager)
    model_server_manager = ModelServerManager(settings)

    app.state.settings = settings
    app.state.connections = connections
    app.state.store = store
    app.state.task_manager = task_manager
    app.state.tunnel_manager = tunnel_manager
    app.state.model_server_manager = model_server_manager
    app.state.mega_credentials = {}

    await task_manager.start()
    await model_server_manager.start()
    await tunnel_manager.start()
    expiry_monitor = asyncio.create_task(_session_expiry_monitor(settings, connections))
    app.state.expiry_monitor = expiry_monitor
    try:
        yield
    finally:
        expiry_monitor.cancel()
        await asyncio.gather(expiry_monitor, return_exceptions=True)
        await tunnel_manager.stop()
        await model_server_manager.stop()
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

    @app.post("/v1/projects/{project_id}/uploads/{kind}/chunked/init")
    async def init_chunked_upload(
        project_id: str,
        kind: str,
        payload: dict[str, Any] = Body(default={}),
    ) -> dict[str, Any]:
        _validate_upload_kind(kind)
        try:
            app.state.store.ensure_project(project_id)
        except StorageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        filename = _safe_upload_name(str(payload.get("filename") or "upload.bin"))
        upload_id = new_id("upload")
        temp_dir = app.state.store.project_path(project_id) / "tmp" / "uploads" / upload_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "upload_id": upload_id,
            "kind": kind,
            "filename": filename,
            "size_bytes": int(payload.get("size_bytes") or 0),
            "created_at": time.time(),
            "chunks": [],
        }
        (temp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {
            "upload_id": upload_id,
            "kind": kind,
            "filename": filename,
            "chunk_url": (
                f"/v1/projects/{project_id}/uploads/{kind}/chunked/"
                f"{upload_id}/chunks/{{chunk_index}}"
            ),
            "complete_url": (
                f"/v1/projects/{project_id}/uploads/{kind}/chunked/{upload_id}/complete"
            ),
        }

    @app.post("/v1/projects/{project_id}/uploads/{kind}/chunked/{upload_id}/chunks/{chunk_index}")
    async def upload_project_file_chunk(
        project_id: str,
        kind: str,
        upload_id: str,
        chunk_index: int,
        chunk: UploadFile = File(...),
    ) -> dict[str, Any]:
        _validate_upload_kind(kind)
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail="chunk_index must be >= 0")
        temp_dir, metadata = _chunk_upload_or_404(app, project_id, kind, upload_id)
        chunk_path = temp_dir / f"{chunk_index:08d}.part"
        size_bytes = 0
        with chunk_path.open("wb") as handle:
            while data := await chunk.read(1024 * 1024):
                size_bytes += len(data)
                handle.write(data)
        chunks = set(int(index) for index in metadata.get("chunks", []))
        chunks.add(chunk_index)
        metadata["chunks"] = sorted(chunks)
        metadata["updated_at"] = time.time()
        (temp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {
            "upload_id": upload_id,
            "chunk_index": chunk_index,
            "size_bytes": size_bytes,
            "received_chunks": len(metadata["chunks"]),
        }

    @app.post("/v1/projects/{project_id}/uploads/{kind}/chunked/{upload_id}/complete")
    async def complete_chunked_upload(
        project_id: str,
        kind: str,
        upload_id: str,
        payload: dict[str, Any] = Body(default={}),
    ) -> dict[str, Any]:
        _validate_upload_kind(kind)
        temp_dir, metadata = _chunk_upload_or_404(app, project_id, kind, upload_id)
        chunk_count = int(payload.get("chunk_count") or len(metadata.get("chunks", [])))
        if chunk_count <= 0:
            raise HTTPException(status_code=400, detail="chunk_count must be > 0")
        missing = [
            index for index in range(chunk_count) if not (temp_dir / f"{index:08d}.part").exists()
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={"message": "missing chunks", "missing_chunks": missing[:100]},
            )

        filename = _safe_upload_name(
            str(payload.get("filename") or metadata.get("filename") or "upload.bin")
        )
        target_dir = app.state.store.project_path(project_id) / "uploads" / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = _unique_path(target_dir / filename)
        size_bytes = 0
        with target_path.open("wb") as output:
            for index in range(chunk_count):
                chunk_path = temp_dir / f"{index:08d}.part"
                with chunk_path.open("rb") as input_file:
                    while data := input_file.read(1024 * 1024):
                        size_bytes += len(data)
                        output.write(data)

        relpath = app.state.store.relative_to_project(project_id, target_path)
        task = await app.state.task_manager.submit(
            project_id=project_id,
            task_type=TaskType.UPLOAD_FILE.value,
            params={
                "kind": kind,
                "stored_path": relpath,
                "filename": target_path.name,
                "size_bytes": size_bytes,
                "upload_id": upload_id,
            },
            description=f"Register chunked {kind} upload",
        )
        _delete_tree(temp_dir)
        return {
            "task_id": task.task_id,
            "status": task.status,
            "upload_id": upload_id,
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


def _validate_upload_kind(kind: str) -> None:
    if kind not in UPLOAD_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of: {', '.join(sorted(UPLOAD_KINDS))}",
        )


def _chunk_upload_or_404(
    app: FastAPI, project_id: str, kind: str, upload_id: str
) -> tuple[Path, dict[str, Any]]:
    try:
        app.state.store.ensure_project(project_id)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not re.fullmatch(r"upload_[a-f0-9]{16}", upload_id):
        raise HTTPException(status_code=400, detail="invalid upload_id")
    temp_dir = app.state.store.project_path(project_id) / "tmp" / "uploads" / upload_id
    metadata_path = temp_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="chunked upload not found")
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="chunked upload metadata is invalid") from exc
    if metadata.get("kind") != kind:
        raise HTTPException(status_code=400, detail="upload kind mismatch")
    return temp_dir, metadata


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


def _delete_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


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
