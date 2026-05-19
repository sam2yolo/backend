from __future__ import annotations

from typing import Any

from .connections import ConnectionManager


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params or {}}


async def emit(
    connections: ConnectionManager,
    event_name: str,
    params: dict[str, Any] | None = None,
    *,
    project_id: str | None = None,
) -> None:
    await connections.send_to_project(project_id, notification(event_name, params))


async def notify_backend_init(connections: ConnectionManager) -> None:
    await emit(connections, "backend_init")


async def notify_backend_init_progress(
    connections: ConnectionManager, *, percent: float, message: str
) -> None:
    await emit(
        connections,
        "backend_init_progress",
        {"percent": percent, "message": message},
    )


async def notify_backend_ready(connections: ConnectionManager) -> None:
    await emit(connections, "backend_ready")


async def notify_task_started(
    connections: ConnectionManager, *, project_id: str, task_id: str, description: str
) -> None:
    await emit(
        connections,
        "task_started",
        {"task_id": task_id, "description": description},
        project_id=project_id,
    )


async def notify_task_progress(
    connections: ConnectionManager,
    *,
    project_id: str,
    task_id: str,
    progress: float,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "progress": progress,
        "message": message,
    }
    if metrics:
        payload["metrics"] = metrics
    await emit(connections, "task_progress", payload, project_id=project_id)


async def notify_task_complete(
    connections: ConnectionManager,
    *,
    project_id: str,
    task_id: str,
    result: dict[str, Any],
) -> None:
    await emit(
        connections,
        "task_complete",
        {"task_id": task_id, "result": result},
        project_id=project_id,
    )


async def notify_task_failed(
    connections: ConnectionManager, *, project_id: str, task_id: str, error: str
) -> None:
    await emit(
        connections,
        "task_failed",
        {"task_id": task_id, "error": error},
        project_id=project_id,
    )


async def notify_task_cancelled(
    connections: ConnectionManager, *, project_id: str, task_id: str
) -> None:
    await emit(connections, "task_cancelled", {"task_id": task_id}, project_id=project_id)


async def notify_inference_result_ready(
    connections: ConnectionManager,
    *,
    project_id: str,
    task_id: str,
    file_path_or_url: str,
) -> None:
    await emit(
        connections,
        "inference_result_ready",
        {"task_id": task_id, "file_path_or_url": file_path_or_url},
        project_id=project_id,
    )


async def notify_training_complete(
    connections: ConnectionManager,
    *,
    project_id: str,
    task_id: str,
    model_id: str,
    metrics: dict[str, Any],
) -> None:
    await emit(
        connections,
        "training_complete",
        {"task_id": task_id, "model_id": model_id, "metrics": metrics},
        project_id=project_id,
    )


async def notify_new_dataset_from_merge(
    connections: ConnectionManager, *, project_id: str, task_id: str, dataset_id: str
) -> None:
    await emit(
        connections,
        "new_dataset_from_merge",
        {"task_id": task_id, "dataset_id": dataset_id},
        project_id=project_id,
    )


async def notify_new_dataset_from_map(
    connections: ConnectionManager, *, project_id: str, task_id: str, dataset_id: str
) -> None:
    await emit(
        connections,
        "new_dataset_from_map",
        {"task_id": task_id, "dataset_id": dataset_id},
        project_id=project_id,
    )


async def notify_mega_credential_check_result(
    connections: ConnectionManager,
    *,
    project_id: str | None,
    valid: bool | None,
    message: str,
) -> None:
    await emit(
        connections,
        "mega_credential_check_result",
        {"valid": valid, "message": message},
        project_id=project_id,
    )


async def notify_mega_mount_success(
    connections: ConnectionManager, *, project_id: str, task_id: str
) -> None:
    await emit(connections, "mega_mount_success", {"task_id": task_id}, project_id=project_id)


async def notify_mega_upload_success(
    connections: ConnectionManager, *, project_id: str, task_id: str, mega_path: str
) -> None:
    await emit(
        connections,
        "mega_upload_success",
        {"task_id": task_id, "mega_path": mega_path},
        project_id=project_id,
    )


async def notify_upload_success(
    connections: ConnectionManager,
    *,
    project_id: str,
    task_id: str,
    filename: str,
    path: str,
) -> None:
    await emit(
        connections,
        "upload_success",
        {"task_id": task_id, "filename": filename, "path": path},
        project_id=project_id,
    )


async def notify_session_expiring(
    connections: ConnectionManager, *, remaining_seconds: int
) -> None:
    await emit(connections, "session_expiring", {"remaining_seconds": remaining_seconds})


async def notify_client_ask(
    connections: ConnectionManager,
    *,
    project_id: str,
    request_id: str,
    request_type: str,
    data: dict[str, Any],
) -> None:
    await emit(
        connections,
        "client.ask",
        {"requestId": request_id, "type": request_type, "data": data},
        project_id=project_id,
    )


async def notify_server_notification(
    connections: ConnectionManager,
    *,
    level: str,
    message: str,
    project_id: str | None = None,
) -> None:
    await emit(
        connections,
        "server.notification",
        {"level": level, "message": message},
        project_id=project_id,
    )


async def notify_tunnel_ready(
    connections: ConnectionManager, *, endpoint: str, server_name: str
) -> None:
    await emit(
        connections,
        "tunnel_ready",
        {"endpoint": endpoint, "server_name": server_name},
    )
