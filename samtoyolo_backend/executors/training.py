from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import quote

from .. import events
from ..capabilities import TRAINING_MODEL_NAMES
from ..records import new_id, utc_now
from ..tasks import TaskContext
from .common import checkpoint_client_fetch


async def execute_train_model(ctx: TaskContext) -> dict[str, Any]:
    params = ctx.task.params
    if not params.get("allow_stub_ml", True):
        raise RuntimeError("training adapter is not installed and stub ML is disabled")
    model_name = (params.get("model_name") or "yolov8").lower().replace("-", "_")
    if model_name not in TRAINING_MODEL_NAMES:
        raise ValueError(f"unsupported model_name: {model_name}")
    dataset_id = params.get("dataset_id")
    if not dataset_id:
        raise ValueError("dataset_id is required")
    session = ctx.store.get_session(ctx.task.project_id)
    if dataset_id not in session.get("datasets", {}):
        raise ValueError(f"unknown dataset_id: {dataset_id}")

    config = params.get("config") or {}
    epochs = max(1, int(config.get("epochs", 1)))
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 5)))
    model_id = new_id("model")
    model_dir = ctx.store.project_path(ctx.task.project_id) / "models" / model_id
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"epochs": epochs, "history": []}

    for epoch in range(1, epochs + 1):
        if ctx.is_cancelled():
            raise RuntimeError("training cancelled")
        loss = round(1 / (epoch + 1), 6)
        metric = {"epoch": epoch, "loss": loss}
        metrics["history"].append(metric)
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            checkpoint = checkpoint_dir / f"epoch_{epoch:04d}.ckpt"
            checkpoint.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "dataset_id": dataset_id,
                        "epoch": epoch,
                        "stub_checkpoint": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            await checkpoint_client_fetch(ctx, checkpoint)
        await ctx.progress(epoch / epochs * 95, f"epoch {epoch}/{epochs}", metrics=metric)
        await asyncio.sleep(0)

    final_path = model_dir / f"{model_name}_{model_id}.pt"
    final_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_name": model_name,
                "dataset_id": dataset_id,
                "config": config,
                "metrics": metrics,
                "stub_model": True,
                "created_at": utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rel_model = ctx.store.relative_to_project(ctx.task.project_id, final_path)
    payload = {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_id": dataset_id,
        "path": rel_model,
        "download_url": (
            f"/v1/projects/{quote(ctx.task.project_id)}/downloads/models/{quote(model_id)}"
        ),
        "metrics": metrics,
        "created_at": utc_now(),
    }
    ctx.store.register_model(ctx.task.project_id, model_id, payload)
    await events.notify_training_complete(
        ctx.connections,
        project_id=ctx.task.project_id,
        task_id=ctx.task.task_id,
        model_id=model_id,
        metrics=metrics,
    )
    await ctx.progress(100, "training complete", metrics=metrics["history"][-1])
    return payload
