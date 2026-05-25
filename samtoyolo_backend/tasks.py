from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from . import events
from .connections import ConnectionManager
from .records import TaskRecord, TaskStatus, utc_now
from .public_urls import join_public_url
from .storage import ProjectStore


@dataclass(slots=True)
class TaskContext:
    manager: "TaskManager"
    store: ProjectStore
    connections: ConnectionManager
    task: TaskRecord

    @property
    def public_base_url(self) -> str | None:
        return self.manager.public_base_url()

    def public_url(self, path: str) -> str:
        return join_public_url(self.public_base_url, path)

    async def progress(
        self,
        progress: float,
        message: str,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        await self.manager.update_progress(
            self.task.task_id, progress, message, metrics=metrics
        )

    def is_cancelled(self) -> bool:
        latest = self.store.get_task(self.task.task_id)
        return latest is not None and latest.status == TaskStatus.CANCELLED.value


TaskExecutor = Callable[[TaskContext], Awaitable[dict[str, Any]]]
PublicBaseUrlGetter = Callable[[], str | None]


class TaskManager:
    def __init__(
        self,
        *,
        store: ProjectStore,
        connections: ConnectionManager,
        gpu_workers: int,
        public_base_url_getter: PublicBaseUrlGetter | None = None,
    ) -> None:
        self.store = store
        self.connections = connections
        self.gpu_workers = max(1, gpu_workers)
        self._public_base_url_getter = public_base_url_getter or (lambda: None)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._executors: dict[str, TaskExecutor] = {}
        self._started = False

    def register_executor(self, task_type: str, executor: TaskExecutor) -> None:
        self._executors[task_type] = executor

    def public_base_url(self) -> str | None:
        return self._public_base_url_getter()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for gpu_index in range(self.gpu_workers):
            worker = asyncio.create_task(self._worker_loop(gpu_index))
            self._workers.append(worker)
        for task in self.store.incomplete_tasks():
            self.store.upsert_task(task)
            await self._queue.put(task.task_id)

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def submit(
        self,
        *,
        project_id: str,
        task_type: str,
        params: dict[str, Any],
        description: str,
    ) -> TaskRecord:
        task = TaskRecord.create(
            project_id=project_id,
            task_type=task_type,
            params=params,
            description=description,
        )
        self.store.upsert_task(task)
        await self._queue.put(task.task_id)
        return task

    async def cancel(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.status in {
            TaskStatus.SUCCEEDED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        }:
            return task
        task.status = TaskStatus.CANCELLED.value
        task.completed_at = utc_now()
        task.message = "cancelled by client"
        self.store.upsert_task(task)
        await events.notify_task_cancelled(
            self.connections, project_id=task.project_id, task_id=task.task_id
        )
        return task

    async def update_progress(
        self,
        task_id: str,
        progress: float,
        message: str,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        task = self.store.get_task(task_id)
        if task is None:
            return
        task.progress = max(0.0, min(100.0, float(progress)))
        task.message = message
        self.store.upsert_task(task)
        await events.notify_task_progress(
            self.connections,
            project_id=task.project_id,
            task_id=task.task_id,
            progress=task.progress,
            message=message,
            metrics=metrics,
        )

    async def _worker_loop(self, gpu_index: int) -> None:
        worker_id = f"gpu-{gpu_index}"
        while True:
            task_id = await self._queue.get()
            try:
                await self._run_task(task_id, worker_id, gpu_index)
            finally:
                self._queue.task_done()

    async def _run_task(self, task_id: str, worker_id: str, gpu_index: int) -> None:
        task = self.store.get_task(task_id)
        if task is None or task.status == TaskStatus.CANCELLED.value:
            return

        executor = self._executors.get(task.task_type)
        if executor is None:
            task.status = TaskStatus.FAILED.value
            task.error = f"no executor registered for task type {task.task_type}"
            task.completed_at = utc_now()
            self.store.upsert_task(task)
            await events.notify_task_failed(
                self.connections,
                project_id=task.project_id,
                task_id=task.task_id,
                error=task.error,
            )
            return

        task.status = TaskStatus.RUNNING.value
        task.started_at = task.started_at or utc_now()
        task.worker_id = worker_id
        task.gpu_index = gpu_index
        self.store.upsert_task(task)
        await events.notify_task_started(
            self.connections,
            project_id=task.project_id,
            task_id=task.task_id,
            description=task.description,
        )

        try:
            result = await executor(
                TaskContext(
                    manager=self,
                    store=self.store,
                    connections=self.connections,
                    task=task,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            latest = self.store.get_task(task.task_id) or task
            latest.status = TaskStatus.FAILED.value
            latest.error = str(exc)
            latest.completed_at = utc_now()
            latest.progress = min(latest.progress, 99.0)
            self.store.upsert_task(latest)
            await events.notify_task_failed(
                self.connections,
                project_id=latest.project_id,
                task_id=latest.task_id,
                error=latest.error or "task failed",
            )
            return

        latest = self.store.get_task(task.task_id) or task
        if latest.status == TaskStatus.CANCELLED.value:
            return
        latest.status = TaskStatus.SUCCEEDED.value
        latest.result = result
        latest.progress = 100.0
        latest.completed_at = utc_now()
        latest.message = "complete"
        self.store.upsert_task(latest)
        await events.notify_task_complete(
            self.connections,
            project_id=latest.project_id,
            task_id=latest.task_id,
            result=result,
        )
