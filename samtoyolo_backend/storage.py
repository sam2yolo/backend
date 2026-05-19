from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .records import FINAL_TASK_STATUSES, TaskRecord, utc_now


PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


class StorageError(ValueError):
    pass


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def validate_project_id(self, project_id: str) -> None:
        if not PROJECT_ID_RE.fullmatch(project_id):
            raise StorageError(
                "project_id must contain only letters, numbers, dot, underscore, "
                "or dash and be at most 96 characters"
            )

    def project_path(self, project_id: str) -> Path:
        self.validate_project_id(project_id)
        return self.root / project_id

    def session_path(self, project_id: str) -> Path:
        return self.project_path(project_id) / "session.json"

    @property
    def task_index_path(self) -> Path:
        return self.root / "_task_index.json"

    def ensure_project(self, project_id: str, display_name: str | None = None) -> dict[str, Any]:
        with self._lock:
            path = self.project_path(project_id)
            path.mkdir(parents=True, exist_ok=True)
            for name in (
                "uploads",
                "frames",
                "inference_results",
                "datasets",
                "models",
                "checkpoints",
                "tmp",
            ):
                (path / name).mkdir(exist_ok=True)
            session = self._load_session(project_id)
            if display_name and session.get("display_name") != display_name:
                session["display_name"] = display_name
                self._save_session(project_id, session)
            return self.project_info(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            projects = []
            for child in sorted(self.root.iterdir()):
                if child.is_dir() and (child / "session.json").exists():
                    projects.append(self.project_info(child.name))
            return projects

    def project_info(self, project_id: str) -> dict[str, Any]:
        session = self._load_session(project_id)
        path = self.project_path(project_id)
        return {
            "project_id": project_id,
            "display_name": session.get("display_name"),
            "path": str(path),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "task_count": len(session.get("tasks", {})),
            "dataset_count": len(session.get("datasets", {})),
            "model_count": len(session.get("models", {})),
            "inference_result_count": len(session.get("inference_results", {})),
        }

    def get_session(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return self._load_session(project_id)

    def mutate_session(
        self, project_id: str, mutator: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        with self._lock:
            session = self._load_session(project_id)
            mutator(session)
            session["updated_at"] = utc_now()
            self._save_session(project_id, session)
            return session

    def upsert_task(self, task: TaskRecord) -> None:
        with self._lock:
            self.ensure_project(task.project_id)
            task.touch()

            def update(session: dict[str, Any]) -> None:
                session.setdefault("tasks", {})[task.task_id] = task.to_dict()

            self.mutate_session(task.project_id, update)
            index = self._load_task_index()
            index[task.task_id] = task.project_id
            self._atomic_write_json(self.task_index_path, index)

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            project_id = self.get_task_project(task_id)
            if project_id is None:
                return None
            session = self._load_session(project_id)
            payload = session.get("tasks", {}).get(task_id)
            return TaskRecord.from_dict(payload) if payload else None

    def get_task_project(self, task_id: str) -> str | None:
        index = self._load_task_index()
        project_id = index.get(task_id)
        if project_id:
            return project_id
        for project in self.list_projects():
            session = self._load_session(project["project_id"])
            if task_id in session.get("tasks", {}):
                index[task_id] = project["project_id"]
                self._atomic_write_json(self.task_index_path, index)
                return project["project_id"]
        return None

    def list_tasks(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if project_id:
                session = self._load_session(project_id)
                return sorted(
                    session.get("tasks", {}).values(),
                    key=lambda row: row.get("created_at", ""),
                    reverse=True,
                )
            tasks: list[dict[str, Any]] = []
            for project in self.list_projects():
                tasks.extend(self.list_tasks(project["project_id"]))
            return tasks

    def incomplete_tasks(self) -> list[TaskRecord]:
        tasks = []
        for payload in self.list_tasks():
            if payload.get("status") not in FINAL_TASK_STATUSES:
                task = TaskRecord.from_dict(payload)
                task.status = "queued"
                task.worker_id = None
                task.gpu_index = None
                tasks.append(task)
        return tasks

    def register_upload(self, project_id: str, upload_id: str, payload: dict[str, Any]) -> None:
        self.mutate_session(
            project_id,
            lambda session: session.setdefault("uploads", {}).__setitem__(upload_id, payload),
        )

    def register_inference_result(
        self, project_id: str, task_id: str, payload: dict[str, Any]
    ) -> None:
        self.mutate_session(
            project_id,
            lambda session: session.setdefault("inference_results", {}).__setitem__(
                task_id, payload
            ),
        )

    def register_dataset(self, project_id: str, dataset_id: str, payload: dict[str, Any]) -> None:
        self.mutate_session(
            project_id,
            lambda session: session.setdefault("datasets", {}).__setitem__(dataset_id, payload),
        )

    def register_model(self, project_id: str, model_id: str, payload: dict[str, Any]) -> None:
        self.mutate_session(
            project_id,
            lambda session: session.setdefault("models", {}).__setitem__(model_id, payload),
        )

    def set_mega_registered(self, project_id: str, registered: bool) -> None:
        self.mutate_session(
            project_id,
            lambda session: session.setdefault("secrets", {}).__setitem__(
                "mega_credentials_registered", registered
            ),
        )

    def resolve_project_file(self, project_id: str, path_value: str | Path) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = self.project_path(project_id) / path
        resolved = path.resolve()
        project_root = self.project_path(project_id).resolve()
        if resolved != project_root and project_root not in resolved.parents:
            raise StorageError("path is outside the project directory")
        return resolved

    def relative_to_project(self, project_id: str, path: Path) -> str:
        return str(path.resolve().relative_to(self.project_path(project_id).resolve()))

    def _load_session(self, project_id: str) -> dict[str, Any]:
        path = self.session_path(project_id)
        if not path.exists():
            session = {
                "project_id": project_id,
                "display_name": None,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "tasks": {},
                "uploads": {},
                "inference_results": {},
                "datasets": {},
                "models": {},
                "client_requests": {},
                "secrets": {},
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(path, session)
            return session
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_session(self, project_id: str, session: dict[str, Any]) -> None:
        self._atomic_write_json(self.session_path(project_id), session)

    def _load_task_index(self) -> dict[str, str]:
        if not self.task_index_path.exists():
            return {}
        with self.task_index_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
