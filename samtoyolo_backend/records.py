from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(StrEnum):
    UPLOAD_FILE = "upload_file"
    REMOTE_DOWNLOAD = "remote_download"
    INFERENCE_SAM3 = "inference_sam3"
    INFERENCE_YOLO = "inference_yolo"
    INFERENCE_TO_YOLO = "inference_to_yolo"
    MERGE_DATASET = "merge_dataset"
    MAP_DATASET = "map_dataset"
    TRAIN_MODEL = "train_model"


FINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED.value,
    TaskStatus.FAILED.value,
    TaskStatus.CANCELLED.value,
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    project_id: str
    task_type: str
    status: str
    params: dict[str, Any]
    description: str
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    worker_id: str | None = None
    gpu_index: int | None = None

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        task_type: str,
        params: dict[str, Any],
        description: str,
    ) -> "TaskRecord":
        return cls(
            task_id=new_id("task"),
            project_id=project_id,
            task_type=task_type,
            status=TaskStatus.QUEUED.value,
            params=params,
            description=description,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRecord":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def touch(self) -> None:
        self.updated_at = utc_now()
