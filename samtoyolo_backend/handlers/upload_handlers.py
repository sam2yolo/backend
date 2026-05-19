from __future__ import annotations

from ..jsonrpc import INVALID_PARAMS, JsonRpcError
from ..records import TaskType
from ..registry import HandlerContext, registry
from .common import bind_project, object_params, optional_str, required_str


def _upload_endpoint(project_id: str, kind: str) -> str:
    return f"/v1/projects/{project_id}/uploads/{kind}"


async def _handle_upload_request(
    ctx: HandlerContext,
    params: object,
    *,
    kind: str,
    default_description: str,
) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)

    source_path = optional_str(data, "source_path")
    if not source_path:
        return {
            "transport": "http",
            "method": "POST",
            "upload_endpoint": _upload_endpoint(project_id, kind),
            "field": "file",
            "note": "Send multipart/form-data to create an upload task.",
        }

    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.UPLOAD_FILE.value,
        params={
            "kind": kind,
            "stored_path": source_path,
            "filename": data.get("filename"),
            "format": data.get("format"),
        },
        description=data.get("description") or default_description,
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("upload_video", aliases=("video_upload",))
async def handle_upload_video(ctx: HandlerContext, params: object) -> dict[str, object]:
    return await _handle_upload_request(
        ctx, params, kind="video", default_description="Register uploaded video"
    )


@registry.method("upload_image", aliases=("image_upload",))
async def handle_upload_image(ctx: HandlerContext, params: object) -> dict[str, object]:
    return await _handle_upload_request(
        ctx, params, kind="image", default_description="Register uploaded image"
    )


@registry.method("upload_dataset", aliases=("dataset_upload",))
async def handle_upload_dataset(ctx: HandlerContext, params: object) -> dict[str, object]:
    return await _handle_upload_request(
        ctx, params, kind="dataset", default_description="Register uploaded dataset"
    )


async def _submit_remote_download(
    ctx: HandlerContext,
    params: object,
    *,
    source: str,
    default_kind: str = "file",
) -> dict[str, object]:
    data = object_params(params)
    project_id = required_str(data, "project_id")
    url = required_str(data, "url")
    kind = data.get("kind") or default_kind
    if kind not in {"file", "video", "image", "dataset"}:
        raise JsonRpcError(INVALID_PARAMS, "kind must be file, video, image, or dataset")
    ctx.store.ensure_project(project_id)
    await bind_project(ctx, project_id)
    task = await ctx.task_manager.submit(
        project_id=project_id,
        task_type=TaskType.REMOTE_DOWNLOAD.value,
        params={
            "source": source,
            "url": url,
            "kind": kind,
            "filename": data.get("filename"),
            "format": data.get("format"),
        },
        description=data.get("description") or f"Download from {source}",
    )
    return {"task_id": task.task_id, "status": task.status}


@registry.method("upload_from_google_drive", aliases=("gdrive_upload", "download_gdrive"))
async def handle_upload_from_google_drive(
    ctx: HandlerContext, params: object
) -> dict[str, object]:
    return await _submit_remote_download(ctx, params, source="google_drive")


@registry.method("upload_from_mega", aliases=("mega_upload", "download_mega"))
async def handle_upload_from_mega(ctx: HandlerContext, params: object) -> dict[str, object]:
    return await _submit_remote_download(ctx, params, source="mega")


@registry.method("upload_from_url", aliases=("direct_link_upload", "download_url"))
async def handle_upload_from_url(ctx: HandlerContext, params: object) -> dict[str, object]:
    return await _submit_remote_download(ctx, params, source="url")
