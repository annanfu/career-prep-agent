"""Background task manager for long-running pipeline operations."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """Tracks the lifecycle of a single background task."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: str = ""


class TaskManager:
    """Thread-pool based manager for background tasks."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskState] = {}

    def submit(
        self, fn: Callable[..., dict], *args: Any, **kwargs: Any,
    ) -> str:
        """Submit a function to run in a background thread.

        The function may accept ``_task_state`` as a keyword
        argument to report progress.

        Returns:
            A unique task_id for polling.
        """
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = TaskState(
            task_id=task_id, status=TaskStatus.RUNNING,
        )
        self._executor.submit(
            self._run, task_id, fn, *args, **kwargs,
        )
        return task_id

    def _run(
        self,
        task_id: str,
        fn: Callable[..., dict],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute the function and update task state."""
        task_state = self._tasks[task_id]
        try:
            result = fn(*args, _task_state=task_state, **kwargs)
            task_state.result = result
            task_state.status = TaskStatus.COMPLETED
        except Exception as e:
            task_state.error = str(e)
            task_state.status = TaskStatus.FAILED

    def get(self, task_id: str) -> Optional[TaskState]:
        """Return the TaskState for a given id, or None."""
        return self._tasks.get(task_id)

    def cleanup(self, task_id: str) -> None:
        """Remove a finished task from memory."""
        self._tasks.pop(task_id, None)
