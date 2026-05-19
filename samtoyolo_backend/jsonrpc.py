from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .registry import HandlerContext, MethodRegistry


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def success_response(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(
    request_id: str | int | None, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def require_object(params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise JsonRpcError(INVALID_PARAMS, "params must be an object")
    return params


async def dispatch_payload(
    payload: Any,
    *,
    context: HandlerContext,
    registry: MethodRegistry,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    if isinstance(payload, list):
        if not payload:
            return error_response(None, INVALID_REQUEST, "batch must not be empty")
        responses: list[dict[str, Any]] = []
        for item in payload:
            response = await dispatch_payload(
                item, context=context, registry=registry
            )
            if isinstance(response, list):
                responses.extend(response)
            elif response is not None:
                responses.append(response)
        return responses or None

    if not isinstance(payload, dict):
        return error_response(None, INVALID_REQUEST, "request must be an object")

    request_id = payload.get("id")
    context.request_id = request_id
    if payload.get("jsonrpc") != "2.0":
        return error_response(request_id, INVALID_REQUEST, "jsonrpc must be '2.0'")

    method_name = payload.get("method")
    if not isinstance(method_name, str) or not method_name:
        return error_response(request_id, INVALID_REQUEST, "method is required")

    handler = registry.get(method_name)
    if handler is None:
        return error_response(
            request_id, METHOD_NOT_FOUND, f"unknown method: {method_name}"
        )

    try:
        result = await handler(
            context, payload["params"] if "params" in payload else {}
        )
    except JsonRpcError as exc:
        if "id" not in payload:
            return None
        return error_response(request_id, exc.code, exc.message, exc.data)
    except ValidationError as exc:
        if "id" not in payload:
            return None
        return error_response(
            request_id,
            INVALID_PARAMS,
            "validation failed",
            exc.errors(include_url=False),
        )
    except ValueError as exc:
        if "id" not in payload:
            return None
        return error_response(request_id, INVALID_PARAMS, str(exc))
    except Exception as exc:
        if "id" not in payload:
            return None
        return error_response(request_id, INTERNAL_ERROR, str(exc))

    if "id" not in payload:
        return None
    return success_response(request_id, result)
