from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


JsonRpcHandler = Callable[[dict[str, Any]], Awaitable[Any]]


PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcServer:
    def __init__(self) -> None:
        self._handlers: dict[str, JsonRpcHandler] = {}

    def method(self, name: str) -> Callable[[JsonRpcHandler], JsonRpcHandler]:
        def decorator(handler: JsonRpcHandler) -> JsonRpcHandler:
            self._handlers[name] = handler
            return handler

        return decorator

    async def dispatch(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return error_response(None, INVALID_REQUEST, "request must be an object")

        request_id = payload.get("id")
        if payload.get("jsonrpc") != "2.0":
            return error_response(request_id, INVALID_REQUEST, "jsonrpc must be '2.0'")

        method = payload.get("method")
        if not isinstance(method, str) or not method:
            return error_response(request_id, INVALID_REQUEST, "method is required")

        handler = self._handlers.get(method)
        if handler is None:
            return error_response(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")

        try:
            result = await handler(payload.get("params") or {})
        except ValueError as exc:
            return error_response(request_id, INVALID_PARAMS, str(exc))
        except Exception as exc:
            return error_response(request_id, INTERNAL_ERROR, str(exc))

        if "id" not in payload:
            return None
        return success_response(request_id, result)

    def describe(self) -> list[str]:
        return sorted(self._handlers)


def success_response(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def require_object(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    return params
