from __future__ import annotations

from ..capabilities import INFERENCE_MODELS, TRAINING_MODELS
from ..registry import HandlerContext, registry
from .common import object_params


@registry.method("models", aliases=("list_models",))
async def handle_models(ctx: HandlerContext, params: object) -> dict[str, object]:
    object_params(params)
    return {
        "inference_models": INFERENCE_MODELS,
        "training_models": TRAINING_MODELS,
        "stub_ml_enabled": ctx.settings.allow_stub_ml,
    }
