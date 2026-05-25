from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import websockets
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
DOWNLOAD_DIR = APP_ROOT / "downloads"
DEFAULT_BROKER_URL = "https://tunnelbroker.sam2yolo.workers.dev"
FINAL_TASK_STATUSES = {"succeeded", "failed", "cancelled"}
DEFAULT_CHUNK_SIZE_BYTES = int(
    os.getenv("SAMTOYOLO_DEMO_CHUNK_SIZE_BYTES", str(8 * 1024 * 1024))
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.getenv("SAMTOYOLO_DEMO_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024 * 1024))
)


@dataclass(slots=True)
class DemoJob:
    job_id: str
    created_at: float = field(default_factory=time.time)
    events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)
    state: dict[str, Any] = field(default_factory=dict)
    done: bool = False

    def emit(self, event: str, **payload: Any) -> None:
        message = {"event": event, "at": time.time(), **payload}
        self.events.put(message)
        self.state.setdefault("events", []).append(message)
        self.state["last_event"] = message
        self.state["updated_at"] = message["at"]

    def set_progress(self, stage: str, percent: float, message: str) -> None:
        progress = self.state.setdefault("progress", {})
        progress[stage] = {
            "percent": max(0.0, min(100.0, float(percent))),
            "message": message,
        }
        self.emit("progress", stage=stage, percent=progress[stage]["percent"], message=message)

    def finish(self, ok: bool, message: str) -> None:
        self.state["ok"] = ok
        self.state["finished_at"] = time.time()
        self.done = True
        self.emit("finished", ok=ok, message=message)


JOBS: dict[str, DemoJob] = {}


class JsonRpcClient:
    def __init__(self, url: str, job: DemoJob, label: str) -> None:
        self.url = url
        self.job = job
        self.label = label
        self.websocket: Any = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.reader_task: asyncio.Task[None] | None = None
        self.counter = 0

    async def __aenter__(self) -> "JsonRpcClient":
        self.websocket = await websockets.connect(self.url, max_size=None, ping_interval=20)
        self.reader_task = asyncio.create_task(self._reader())
        self.job.emit("rpc_connected", label=self.label, url=self.url)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.reader_task:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        if self.websocket:
            await self.websocket.close()

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.counter += 1
        request_id = f"{self.label}-{self.counter}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.pending[request_id] = future
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        await self.websocket.send(json.dumps(payload))
        self.job.emit("rpc_request", label=self.label, method=method, request_id=request_id)
        response = await future
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    async def _reader(self) -> None:
        while True:
            raw = await self.websocket.recv()
            message = json.loads(raw)
            request_id = message.get("id")
            if request_id in self.pending:
                future = self.pending.pop(request_id)
                if not future.done():
                    future.set_result(message)
                continue
            self._handle_notification(message)

    def _handle_notification(self, message: dict[str, Any]) -> None:
        method = message.get("method") or "notification"
        params = message.get("params") or {}
        if method == "task_progress":
            self.job.set_progress(
                "inference",
                float(params.get("progress") or 0),
                str(params.get("message") or "task progress"),
            )
        elif method == "model.progress":
            self.job.set_progress(
                "model",
                float(params.get("progress") or 0),
                str(params.get("message") or "model progress"),
            )
        elif method == "upload_success":
            self.job.set_progress("upload", 100, "upload registered")
        elif method == "inference_result_ready":
            self.job.state["inference_result_ready"] = params
            self.job.set_progress("inference", 100, "inference result ready")
        self.job.emit("rpc_event", label=self.label, method=method, params=params)


class MultipartUploadStream(httpx.SyncByteStream):
    def __init__(
        self,
        file_path: Path,
        field_name: str,
        boundary: str,
        progress: Callable[[int, int], None],
    ) -> None:
        self.file_path = file_path
        self.field_name = field_name
        self.boundary = boundary
        self.progress = progress
        self.file_size = file_path.stat().st_size
        self.head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{file_path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        self.tail = f"\r\n--{boundary}--\r\n".encode()
        self.total = len(self.head) + self.file_size + len(self.tail)

    def __iter__(self):
        sent = 0
        sent += len(self.head)
        self.progress(sent, self.total)
        yield self.head
        with self.file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                sent += len(chunk)
                self.progress(sent, self.total)
                yield chunk
        sent += len(self.tail)
        self.progress(sent, self.total)
        yield self.tail


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        default_broker_url=DEFAULT_BROKER_URL,
        default_project_id=f"demo-{int(time.time())}",
        existing_uploads=_existing_uploads(),
    )


@app.post("/jobs")
def create_job() -> Response:
    job_id = uuid.uuid4().hex[:12]
    job = DemoJob(job_id=job_id)
    JOBS[job_id] = job

    uploaded_file = request.files.get("video_file")
    existing_upload = request.form.get("existing_upload") or ""
    upload_path: str | None = None
    if existing_upload:
        existing_path = (UPLOAD_DIR / Path(existing_upload).name).resolve()
        if existing_path.parent != UPLOAD_DIR.resolve() or not existing_path.exists():
            return _job_create_error("selected existing upload was not found")
        upload_path = str(existing_path)
    elif uploaded_file and uploaded_file.filename:
        target = UPLOAD_DIR / f"{job_id}-{Path(uploaded_file.filename).name}"
        uploaded_file.save(target)
        upload_path = str(target)

    form = request.form.to_dict(flat=True)
    form["upload_path"] = upload_path or ""
    job.state["config"] = _safe_config(form)
    thread = threading.Thread(target=_run_job_thread, args=(job, form), daemon=True)
    thread.start()
    if _wants_json():
        return jsonify({"job_id": job_id, "url": url_for("job_page", job_id=job_id)})
    return redirect(url_for("job_page", job_id=job_id))


@app.get("/jobs/<job_id>")
def job_page(job_id: str) -> str:
    job = _require_job(job_id)
    return render_template("job.html", job_id=job_id, state=job.state)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str) -> Response:
    job = _require_job(job_id)
    return jsonify(job.state | {"done": job.done, "job_id": job_id})


@app.get("/events/<job_id>")
def job_events(job_id: str) -> Response:
    job = _require_job(job_id)

    def stream():
        yield _sse("snapshot", job.state | {"done": job.done, "job_id": job_id})
        while True:
            try:
                event = job.events.get(timeout=5)
            except queue.Empty:
                yield _sse("heartbeat", {"done": job.done})
                if job.done:
                    break
                continue
            yield _sse(event["event"], event)
            if job.done and job.events.empty():
                break

    return Response(stream(), mimetype="text/event-stream")


@app.get("/downloads/<job_id>/<path:filename>")
def download_file(job_id: str, filename: str) -> Response:
    job = _require_job(job_id)
    downloads = job.state.get("downloads") or []
    for item in downloads:
        if item.get("filename") == filename:
            return send_file(item["path"], as_attachment=True, download_name=filename)
    return Response("download not found", status=404)


@app.get("/api/uploads")
def list_uploads() -> Response:
    return jsonify({"uploads": _existing_uploads()})


def _run_job_thread(job: DemoJob, form: dict[str, str]) -> None:
    try:
        asyncio.run(_run_job(job, form))
    except Exception as exc:
        job.emit("error", message=str(exc))
        job.finish(False, str(exc))


async def _run_job(job: DemoJob, form: dict[str, str]) -> None:
    broker_url = _clean_url(form.get("broker_url")) or DEFAULT_BROKER_URL
    group = _required(form, "group")
    group_token = form.get("group_token") or ""
    backend_peer = form.get("backend_peer") or ""
    backend_http_override = _clean_url(form.get("backend_http"))
    backend_ws_override = _clean_url(form.get("backend_ws"))
    project_id = _required(form, "project_id")

    job.set_progress("discover", 0, "discovering peers")
    endpoints = await _discover_endpoints(
        broker_url=broker_url,
        group=group,
        group_token=group_token,
        backend_peer=backend_peer,
        backend_http_override=backend_http_override,
        backend_ws_override=backend_ws_override,
        job=job,
    )
    job.state["endpoints"] = endpoints
    job.set_progress("discover", 100, "endpoints ready")

    async with JsonRpcClient(endpoints["backend_ws"], job, "backend") as backend:
        job.set_progress("project", 5, "creating project")
        project = await backend.call(
            "create_project",
            {"project_id": project_id, "display_name": form.get("display_name") or project_id},
        )
        job.state["project"] = project
        job.set_progress("project", 100, "project ready")

        media_path = await _prepare_media(job, form, backend, endpoints["backend_http"], project_id)
        job.state["media_path"] = media_path

        if _bool(form.get("setup_model_server"), True):
            job.set_progress("model_setup", 5, "setting up model server")
            setup = await backend.call(
                "setup_model_server",
                {"name": form.get("model_server_name") or "sam3"},
            )
            job.state["model_server"] = setup
            if setup.get("public_ws_url"):
                endpoints["sam3_ws"] = setup["public_ws_url"]
            job.set_progress("model_setup", 100, "model server ready")

        inference_params = _build_inference_params(form, project_id, media_path)
        job.state["inference_params"] = inference_params
        job.set_progress("inference", 1, "starting inference request")
        inference = await backend.call(form.get("inference_method") or "inference_sam3", inference_params)
        job.state["inference_response"] = inference

        task_id = _extract_task_id(inference)
        if task_id:
            job.state["inference_task_id"] = task_id
            await _wait_for_task(backend, job, task_id, "inference")
            await _download_inference_result(backend, job, task_id)
            job.finish(True, "full inference test completed")
            return

        model_server = inference.get("model_server") if isinstance(inference, dict) else None
        if isinstance(model_server, dict):
            job.state["delegated_model_server"] = model_server
            job.emit(
                "delegated",
                message=(
                    "Backend returned a model-server delegation response instead of "
                    "a packaged inference task. The demo verified discovery, upload, "
                    "project creation, and model setup; packaged download requires an "
                    "inference task_id from the backend."
                ),
                model_server=model_server,
            )
            job.set_progress("inference", 100, "delegated to model server")
            job.finish(True, "delegated inference flow verified")
            return

        raise RuntimeError("inference response did not include task_id or model_server")


async def _discover_endpoints(
    *,
    broker_url: str,
    group: str,
    group_token: str,
    backend_peer: str,
    backend_http_override: str | None,
    backend_ws_override: str | None,
    job: DemoJob,
) -> dict[str, str]:
    if backend_http_override or backend_ws_override:
        backend_http = backend_http_override or backend_ws_override.replace("wss://", "https://").removesuffix("/v1/ws")
        backend_ws = backend_ws_override or backend_http.replace("https://", "wss://").rstrip("/") + "/v1/ws"
        return {"backend_http": backend_http.rstrip("/"), "backend_ws": backend_ws}

    headers = {}
    if group_token:
        headers["Authorization"] = f"Bearer {group_token}"
    url = f"{broker_url.rstrip('/')}/v1/groups/{group}/peers"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    peers = response.json().get("peers") or []
    job.state["discovered_peers"] = peers
    backend = _select_backend_peer(peers, backend_peer)
    backend_http = _peer_endpoint(backend)
    backend_ws = backend_http.replace("https://", "wss://").rstrip("/") + "/v1/ws"
    sam3 = _select_sam3_peer(peers, backend.get("peer"))
    endpoints = {"backend_http": backend_http.rstrip("/"), "backend_ws": backend_ws}
    if sam3:
        metadata = sam3.get("metadata") or {}
        if metadata.get("ws_endpoint"):
            endpoints["sam3_ws"] = metadata["ws_endpoint"]
    return endpoints


async def _prepare_media(
    job: DemoJob,
    form: dict[str, str],
    backend: JsonRpcClient,
    backend_http: str,
    project_id: str,
) -> str:
    source_mode = form.get("source_mode") or "gdrive"
    if source_mode == "local":
        upload_path = form.get("upload_path")
        if not upload_path:
            raise RuntimeError("local upload mode selected but no file was uploaded")
        path = Path(upload_path)
        job.set_progress("upload", 0, f"uploading {path.name}")
        result = await asyncio.to_thread(
            _upload_file_sync,
            backend_http,
            project_id,
            path,
            lambda sent, total: job.set_progress(
                "upload",
                sent / max(1, total) * 100,
                f"uploaded {sent}/{total} bytes",
            ),
        )
        job.state["upload_result"] = result
        job.set_progress("upload", 100, "upload complete")
        return str(result["path"])

    url = _required(form, "gdrive_url")
    filename = form.get("remote_filename") or "video.dav"
    job.set_progress("upload", 5, "submitting Google Drive import")
    result = await backend.call(
        "upload_from_google_drive",
        {
            "project_id": project_id,
            "url": url,
            "kind": "video",
            "filename": filename,
        },
    )
    job.state["upload_result"] = result
    task_id = result.get("task_id")
    if task_id:
        task = await _wait_for_task(backend, job, task_id, "upload")
        task_result = task.get("result") or {}
        if task_result.get("path"):
            return str(task_result["path"])
    return f"uploads/video/{filename}"


def _upload_file_sync(
    backend_http: str,
    project_id: str,
    path: Path,
    progress: Callable[[int, int], None],
) -> dict[str, Any]:
    try:
        return _upload_file_chunked_sync(backend_http, project_id, path, progress)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        return _upload_file_single_request_sync(backend_http, project_id, path, progress)


def _upload_file_single_request_sync(
    backend_http: str,
    project_id: str,
    path: Path,
    progress: Callable[[int, int], None],
) -> dict[str, Any]:
    boundary = f"----samtoyolo-demo-{uuid.uuid4().hex}"
    stream = MultipartUploadStream(path, "file", boundary, progress)
    headers = {
        "content-type": f"multipart/form-data; boundary={boundary}",
        "content-length": str(stream.total),
    }
    url = f"{backend_http.rstrip('/')}/v1/projects/{project_id}/uploads/video"
    with httpx.Client(timeout=None) as client:
        response = client.post(url, content=stream, headers=headers)
        response.raise_for_status()
        return response.json()


def _upload_file_chunked_sync(
    backend_http: str,
    project_id: str,
    path: Path,
    progress: Callable[[int, int], None],
) -> dict[str, Any]:
    base = backend_http.rstrip("/")
    file_size = path.stat().st_size
    chunk_size = max(1024 * 1024, DEFAULT_CHUNK_SIZE_BYTES)
    chunk_count = max(1, (file_size + chunk_size - 1) // chunk_size)
    with httpx.Client(timeout=None) as client:
        init = client.post(
            f"{base}/v1/projects/{project_id}/uploads/video/chunked/init",
            json={"filename": path.name, "size_bytes": file_size},
        )
        init.raise_for_status()
        upload_id = init.json()["upload_id"]

        sent = 0
        with path.open("rb") as handle:
            for chunk_index in range(chunk_count):
                data = handle.read(chunk_size)
                response = client.post(
                    (
                        f"{base}/v1/projects/{project_id}/uploads/video/chunked/"
                        f"{upload_id}/chunks/{chunk_index}"
                    ),
                    files={"chunk": (f"{chunk_index:08d}.part", data, "application/octet-stream")},
                )
                response.raise_for_status()
                sent += len(data)
                progress(sent, file_size)

        complete = client.post(
            f"{base}/v1/projects/{project_id}/uploads/video/chunked/{upload_id}/complete",
            json={"filename": path.name, "chunk_count": chunk_count, "size_bytes": file_size},
        )
        complete.raise_for_status()
        progress(file_size, file_size)
        return complete.json()


async def _wait_for_task(
    backend: JsonRpcClient,
    job: DemoJob,
    task_id: str,
    stage: str,
) -> dict[str, Any]:
    timeout_seconds = int(job.state["config"].get("task_timeout_seconds") or 3600)
    deadline = time.time() + timeout_seconds
    last_status = ""
    while time.time() < deadline:
        task = await backend.call("get_task_status", {"task_id": task_id})
        status = str(task.get("status") or "")
        progress = float(task.get("progress") or 0)
        message = str(task.get("message") or status or "running")
        if status != last_status:
            job.emit("task_status", stage=stage, task_id=task_id, status=status)
            last_status = status
        job.set_progress(stage, progress, message)
        if status in FINAL_TASK_STATUSES:
            if status != "succeeded":
                raise RuntimeError(f"{stage} task {task_id} ended with {status}: {task.get('error')}")
            return task
        await asyncio.sleep(float(job.state["config"].get("poll_interval_seconds") or 2))
    raise RuntimeError(f"timed out waiting for {stage} task {task_id}")


async def _download_inference_result(
    backend: JsonRpcClient,
    job: DemoJob,
    task_id: str,
) -> None:
    result = await backend.call(
        "download_inference_result",
        {
            "task_id": task_id,
            "delete_after_download": _bool(job.state["config"].get("delete_after_download"), False),
        },
    )
    download_url = result.get("download_url")
    if not download_url:
        raise RuntimeError("download_inference_result did not return download_url")
    job.state["download_result"] = result
    target = DOWNLOAD_DIR / f"{job.job_id}-{task_id}.zip"
    job.set_progress("download", 0, "starting download")
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        async with client.stream("GET", download_url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            received = 0
            with target.open("wb") as handle:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    received += len(chunk)
                    handle.write(chunk)
                    if total:
                        percent = received / total * 100
                    else:
                        percent = min(99.0, received / (1024 * 1024))
                    job.set_progress(
                        "download",
                        percent,
                        f"downloaded {received}/{total or '?'} bytes",
                    )
    item = {
        "filename": target.name,
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "url": f"/downloads/{job.job_id}/{target.name}",
    }
    job.state.setdefault("downloads", []).append(item)
    job.set_progress("download", 100, "download complete")


def _build_inference_params(
    form: dict[str, str],
    project_id: str,
    media_path: str,
) -> dict[str, Any]:
    prompts, prompt_to_class = _parse_prompts(form.get("prompts") or "")
    return {
        "project_id": project_id,
        "media_path": media_path,
        "prompts": prompts,
        "prompt_to_class": prompt_to_class,
        "sample_strategy": form.get("sample_strategy") or "random",
        "max_frames": _int(form.get("max_frames"), 20),
        "random_seed": _optional_int(form.get("random_seed")),
        "max_frame_width": _optional_int(form.get("max_frame_width")),
        "batch_size": _int(form.get("batch_size"), 4),
        "inference_backend": form.get("inference_backend") or "image",
        "output_mode": form.get("output_mode") or "both",
        "include_masks": _bool(form.get("include_masks"), True),
        "visualize": _bool(form.get("visualize"), True),
        "confidence_threshold": _float(form.get("confidence_threshold"), 0.35),
        "sam3_max_num_objects": _int(form.get("sam3_max_num_objects"), 16),
        "sam3_multiplex_count": _int(form.get("sam3_multiplex_count"), 16),
        "sam3_use_fa3": _bool(form.get("sam3_use_fa3"), False),
        "sam3_compile": _bool(form.get("sam3_compile"), False),
        "sam3_cache_model": _bool(form.get("sam3_cache_model"), True),
        "sam3_allow_partial_checkpoint": _bool(
            form.get("sam3_allow_partial_checkpoint"), False
        ),
        "prepare_model": _bool(form.get("prepare_model"), True),
    }


def _parse_prompts(text: str) -> tuple[list[str], dict[str, str]]:
    prompts: list[str] = []
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            prompt, class_name = [part.strip() for part in line.split("=", 1)]
        elif ":" in line:
            prompt, class_name = [part.strip() for part in line.split(":", 1)]
        else:
            prompt = class_name = line
        if prompt:
            prompts.append(prompt)
            mapping[prompt] = class_name or prompt
    if not prompts:
        raise RuntimeError("at least one prompt is required")
    return prompts, mapping


def _select_backend_peer(peers: list[dict[str, Any]], backend_peer: str) -> dict[str, Any]:
    if backend_peer:
        for peer in peers:
            if peer.get("peer") == backend_peer:
                return peer
        raise RuntimeError(f"backend peer not found: {backend_peer}")
    for peer in peers:
        metadata = peer.get("metadata") or {}
        if metadata.get("app") == "samtoyolo-backend":
            return peer
    raise RuntimeError("no samtoyolo-backend peer found in Tunnelbroker group")


def _select_sam3_peer(
    peers: list[dict[str, Any]], backend_peer: str | None
) -> dict[str, Any] | None:
    expected = f"{backend_peer}-sam3" if backend_peer else None
    for peer in peers:
        metadata = peer.get("metadata") or {}
        if expected and peer.get("peer") == expected:
            return peer
        if metadata.get("app") == "samtoyolo-model-server" and metadata.get("model") == "sam3":
            return peer
    return None


def _peer_endpoint(peer: dict[str, Any]) -> str:
    contacts = peer.get("contacts") or []
    if contacts:
        return str(contacts[0]["endpoint"])
    if peer.get("endpoint"):
        return str(peer["endpoint"])
    raise RuntimeError(f"peer has no endpoint: {peer.get('peer')}")


def _extract_task_id(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("task_id"), str):
        return result["task_id"]
    nested = result.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("task_id"), str):
        return nested["task_id"]
    return None


def _safe_config(form: dict[str, str]) -> dict[str, str]:
    hidden = {"group_token"}
    return {key: ("<redacted>" if key in hidden and value else value) for key, value in form.items()}


def _existing_uploads() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(UPLOAD_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "updated_at": path.stat().st_mtime,
            }
        )
    return rows[:50]


def _wants_json() -> bool:
    return (
        request.headers.get("x-requested-with") == "XMLHttpRequest"
        or "application/json" in request.headers.get("accept", "")
    )


def _job_create_error(message: str) -> Response:
    if _wants_json():
        return jsonify({"error": message}), 400
    return Response(message, status=400)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _require_job(job_id: str) -> DemoJob:
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return job


def _required(form: dict[str, str], name: str) -> str:
    value = form.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _clean_url(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "checked"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value or default)
    except Exception:
        return default


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return _int(value, 0)


def _float(value: str | None, default: float) -> float:
    try:
        return float(value or default)
    except Exception:
        return default


if __name__ == "__main__":
    app.run(
        host=os.getenv("SAMTOYOLO_DEMO_HOST", "127.0.0.1"),
        port=int(os.getenv("SAMTOYOLO_DEMO_PORT", "5055")),
        debug=os.getenv("SAMTOYOLO_DEMO_DEBUG", "").lower() in {"1", "true", "yes"},
        threaded=True,
    )
