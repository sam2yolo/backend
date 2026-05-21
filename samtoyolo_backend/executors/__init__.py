from __future__ import annotations

from ..records import TaskType
from ..tasks import TaskManager
from .dataset import execute_inference_to_yolo, execute_map_dataset, execute_merge_dataset
from .upload import execute_remote_download, execute_upload_file


def register_default_executors(task_manager: TaskManager) -> None:
    task_manager.register_executor(TaskType.UPLOAD_FILE.value, execute_upload_file)
    task_manager.register_executor(TaskType.REMOTE_DOWNLOAD.value, execute_remote_download)
    task_manager.register_executor(TaskType.INFERENCE_TO_YOLO.value, execute_inference_to_yolo)
    task_manager.register_executor(TaskType.MERGE_DATASET.value, execute_merge_dataset)
    task_manager.register_executor(TaskType.MAP_DATASET.value, execute_map_dataset)


__all__ = [
    "execute_inference_to_yolo",
    "execute_map_dataset",
    "execute_merge_dataset",
    "execute_remote_download",
    "execute_upload_file",
    "register_default_executors",
]
