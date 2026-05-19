"""Importing this package registers every JSON-RPC handler."""

from . import control_handlers as control_handlers
from . import dataset_handlers as dataset_handlers
from . import download_handlers as download_handlers
from . import inference_handlers as inference_handlers
from . import info_handlers as info_handlers
from . import mega_handlers as mega_handlers
from . import model_handlers as model_handlers
from . import project_handlers as project_handlers
from . import training_handlers as training_handlers
from . import tunnel_handlers as tunnel_handlers
from . import upload_handlers as upload_handlers

__all__ = [
    "control_handlers",
    "dataset_handlers",
    "download_handlers",
    "inference_handlers",
    "info_handlers",
    "mega_handlers",
    "model_handlers",
    "project_handlers",
    "training_handlers",
    "tunnel_handlers",
    "upload_handlers",
]
