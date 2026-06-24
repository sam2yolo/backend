import json
import pickle
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import websocket


class BackendClient:
    def __init__(self, base_url: str, timeout: int = 3600):
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = f"{ws_scheme}://{parsed.netloc}/ws"
        self.timeout = timeout
        self.ws = None

    def __enter__(self):
        self.ws = websocket.create_connection(
            self.ws_url,
            timeout=self.timeout,
        )
        return self

    def __exit__(self, *_):
        if self.ws is not None:
            self.ws.close()

    def send(self, action: str, payload=None):
        self.ws.send(
            json.dumps({"action": action, "payload": payload or {}})
        )

    def receive(self):
        raw = self.ws.recv()
        if not raw:
            raise ConnectionError("The backend WebSocket closed")
        return json.loads(raw)

    def wait_for(self, actions, task_id=None):
        if isinstance(actions, str):
            actions = {actions}
        while True:
            message = self.receive()
            action = message.get("action")
            payload = message.get("payload", {})
            candidate = payload.get("task_id") or payload.get("id")
            print(action, payload)
            if action in {"task_failed", "inference_task_error"}:
                raise RuntimeError(message)
            if action in actions and (
                task_id is None or candidate == task_id
            ):
                return message

    def upload(self, path):
        path = Path(path)
        with path.open("rb") as stream:
            response = requests.post(
                f"{self.base_url}/upload",
                files={"file": (path.name, stream)},
                timeout=300,
            )
        response.raise_for_status()
        return response.json()["file_id"]

    def upload_many(self, paths):
        return [self.upload(path) for path in paths]

    def init_yolo(self, variant="yolo11n"):
        self.send(
            "init_model",
            {"model_name": "yolo", "variant_name": variant},
        )
        result = self.wait_for(
            {"model_init_completed", "model_init_error"}
        )
        if result["action"] != "model_init_completed":
            raise RuntimeError(result)

    def init_sam(self, sam_url="http://127.0.0.1:8001"):
        self.send(
            "init_model",
            {"model_name": "sam", "base_url": sam_url},
        )
        result = self.wait_for(
            {"model_init_completed", "model_init_error"}
        )
        if result["action"] != "model_init_completed":
            raise RuntimeError(result)

    def run_inference(self, payload):
        self.send("create_inference_task", payload)
        added = self.wait_for(
            {"task_added", "create_inference_task_error"}
        )
        if added["action"] != "task_added":
            raise RuntimeError(added)

        task_id = added["payload"]["id"]
        self.send("start_inference_from_queue", {})
        chunks = []

        while True:
            message = self.receive()
            action = message.get("action")
            data = message.get("payload", {})
            candidate = data.get("task_id") or data.get("id")
            print(action, data)

            if candidate not in (None, task_id):
                continue
            if action == "inference_task_chunk_result":
                chunks.append(data["chunk_id"])
            elif action in {"inference_task_error", "task_failed"}:
                raise RuntimeError(message)
            elif action == "inference_completed":
                return task_id, chunks

    def download_chunk(self, chunk_id, destination):
        response = requests.get(
            f"{self.base_url}/inference_result",
            params={"id": chunk_id},
            timeout=600,
        )
        response.raise_for_status()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination

    def download_and_load(self, chunk_id, destination):
        path = self.download_chunk(chunk_id, destination)
        with path.open("rb") as stream:
            return pickle.load(stream)

