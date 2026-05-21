from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any


ProgressCallback = Callable[[float, str, dict[str, Any] | None], None]


class ModelServerError(RuntimeError):
    """Raised when a per-model server rejects or cannot complete a request."""


_REQUEST_IDS = itertools.count(1)


def call_model_server(
    *,
    url: str,
    method: str,
    params: dict[str, Any],
    progress: ProgressCallback | None = None,
    open_timeout: float = 30.0,
    close_timeout: float = 10.0,
) -> Any:
    """Call a model server JSON-RPC method over WebSocket.

    Model servers can emit JSON-RPC notifications while a request is running.
    The backend currently listens for `model.progress` notifications and maps
    them onto the parent task's progress events.
    """

    try:
        from websockets.sync.client import connect
    except Exception as exc:
        raise ModelServerError(
            "websockets is required for model-server adapters; install backend "
            "requirements or add websockets>=12"
        ) from exc

    request_id = f"backend-{next(_REQUEST_IDS)}"
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    try:
        with connect(
            url,
            max_size=None,
            open_timeout=open_timeout,
            close_timeout=close_timeout,
        ) as websocket:
            websocket.send(json.dumps(request))
            while True:
                raw_message = websocket.recv()
                payload = json.loads(raw_message)
                if isinstance(payload, list):
                    for item in payload:
                        result = _handle_payload(item, request_id, progress)
                        if result is not _NO_RESULT:
                            return result
                    continue

                result = _handle_payload(payload, request_id, progress)
                if result is not _NO_RESULT:
                    return result
    except ModelServerError:
        raise
    except Exception as exc:
        raise ModelServerError(
            f"could not call model server {url!r} method {method!r}: {exc}"
        ) from exc


_NO_RESULT = object()


def _handle_payload(
    payload: Any,
    request_id: str,
    progress: ProgressCallback | None,
) -> Any:
    if not isinstance(payload, dict):
        return _NO_RESULT

    method = payload.get("method")
    if method == "model.progress":
        params = payload.get("params") or {}
        if progress and isinstance(params, dict):
            progress(
                float(params.get("progress", 0.0)),
                str(params.get("message", "")),
                params.get("metrics")
                if isinstance(params.get("metrics"), dict)
                else None,
            )
        return _NO_RESULT

    if payload.get("id") != request_id:
        return _NO_RESULT

    if "error" in payload:
        error = payload.get("error") or {}
        if isinstance(error, dict):
            message = str(error.get("message", "model server request failed"))
            data = error.get("data")
            if data is not None:
                message = f"{message}: {data}"
            raise ModelServerError(message)
        raise ModelServerError(str(error))

    return payload.get("result")
