from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from samtoyolo_model_servers.common.jsonrpc import (
    JsonRpcServer,
    PARSE_ERROR,
    error_response,
    require_object,
)
from samtoyolo_model_servers.sam3.inference import run_sam3_video_text_inference


def create_app() -> FastAPI:
    rpc = _create_rpc()
    app = FastAPI(title="SAM-to-YOLO SAM 3.1 Model Server", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "samtoyolo-model-server",
            "model": "sam3",
            "methods": rpc.describe(),
        }

    @app.websocket("/v1/ws")
    async def websocket_rpc(websocket: WebSocket) -> None:
        await websocket.accept()
        connection_rpc = _create_rpc(websocket)
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

                response = await connection_rpc.dispatch(payload)
                if response is not None:
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    return app


def _create_rpc(websocket: WebSocket | None = None) -> JsonRpcServer:
    rpc = JsonRpcServer()

    @rpc.method("health")
    async def rpc_health(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "model": "sam3",
            "pid": os.getpid(),
            "methods": rpc.describe(),
        }

    @rpc.method("sam3.infer_video_text")
    async def infer_video_text(params: dict[str, Any]) -> dict[str, Any]:
        if websocket is None:
            raise RuntimeError("sam3.infer_video_text requires a WebSocket connection")
        return await _infer_video_text(websocket, params)

    return rpc


async def _infer_video_text(websocket: WebSocket, params: dict[str, Any]) -> dict[str, Any]:
    data = require_object(params)
    loop = asyncio.get_running_loop()

    def progress(
        percent: float,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "model.progress",
                    "params": {
                        "progress": percent,
                        "message": message,
                        "metrics": metrics or {},
                    },
                }
            ),
            loop,
        )
        future.result()

    result = await asyncio.to_thread(
        run_sam3_video_text_inference,
        frames_dir=Path(str(data["frames_dir"])),
        frames=[Path(str(frame)) for frame in data.get("frames", [])],
        prompts=[str(prompt) for prompt in data.get("prompts", [])],
        prompt_to_class={
            str(prompt): str(class_name)
            for prompt, class_name in (data.get("prompt_to_class") or {}).items()
        },
        model_extract_dir=Path(str(data["model_extract_dir"])),
        gpu_index=data.get("gpu_index"),
        output_prob_thresh=float(data.get("output_prob_thresh", 0.5)),
        max_num_objects=max(1, int(data.get("max_num_objects", 16))),
        multiplex_count=max(1, int(data.get("multiplex_count", 16))),
        use_fa3=bool(data.get("use_fa3", False)),
        compile_model=bool(data.get("compile_model", False)),
        warm_up=bool(data.get("warm_up", False)),
        async_loading_frames=bool(data.get("async_loading_frames", False)),
        offload_video_to_cpu=bool(data.get("offload_video_to_cpu", True)),
        offload_state_to_cpu=bool(data.get("offload_state_to_cpu", False)),
        cache_model=bool(data.get("cache_model", True)),
        allow_partial_checkpoint=bool(data.get("allow_partial_checkpoint", False)),
        include_masks=bool(data.get("include_masks", True)),
        progress=progress,
        progress_start=float(data.get("progress_start", 0.0)),
        progress_end=float(data.get("progress_end", 100.0)),
    )
    return {"frames": result.frames, "metadata": result.metadata}


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("SAMTOYOLO_MODEL_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SAMTOYOLO_SAM3_SERVER_PORT", "8101"))
    uvicorn.run(
        "samtoyolo_model_servers.sam3.server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
