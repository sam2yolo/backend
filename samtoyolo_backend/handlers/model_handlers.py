from __future__ import annotations

from copy import deepcopy

from ..capabilities import INFERENCE_MODELS, TRAINING_MODELS
from ..model_sources import google_drive_file_id
from ..registry import HandlerContext, registry
from .common import object_params


@registry.method("models", aliases=("list_models",))
async def handle_models(ctx: HandlerContext, params: object) -> dict[str, object]:
    object_params(params)
    inference_models = deepcopy(INFERENCE_MODELS)
    for model in inference_models:
        if model.get("name") == "sam3":
            model["model_source_url"] = ctx.settings.sam3_model_url
            model["model_download_url"] = ctx.settings.sam3_model_download_url
            model["model_file_id"] = google_drive_file_id(ctx.settings.sam3_model_url)
            model["model_filename"] = ctx.settings.sam3_model_filename
            model["model_server_url"] = ctx.app.state.model_server_manager.endpoint_for(
                "sam3"
            )
            model["model_server"] = ctx.app.state.model_server_manager.status("sam3")
    return {
        "inference_models": inference_models,
        "training_models": TRAINING_MODELS,
        "stub_ml_enabled": ctx.settings.allow_stub_ml,
        "model_cache_dir": str(ctx.settings.model_cache_dir),
    }
