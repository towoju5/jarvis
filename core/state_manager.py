"""Tracks in-flight tasks and gates publishing behind a Telegram approval.

A task moves GENERATING -> PENDING_APPROVAL -> (PUBLISHING | REJECTED).
The suspend-and-wait the blueprint describes is exactly
TelegramApprovalBridge.request_approval's awaited Future under the hood;
this class just tracks task state around that call so callers (the main
loop, social_poster) have somewhere to check status.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from communication.notify_bridge import TelegramApprovalBridge

logger = logging.getLogger(__name__)


class TaskState(Enum):
    IDLE = auto()
    GENERATING = auto()
    PENDING_APPROVAL = auto()
    PUBLISHING = auto()
    DONE = auto()
    REJECTED = auto()
    FAILED = auto()


@dataclass
class Task:
    id: str
    description: str
    state: TaskState = TaskState.IDLE
    result: Any = None
    error: str | None = None


class StateManager:
    def __init__(self, approval_bridge: TelegramApprovalBridge | None = None, approval_timeout: float | None = 1800) -> None:
        self._approval_bridge = approval_bridge
        self._approval_timeout = approval_timeout
        self.tasks: dict[str, Task] = {}

    def create_task(self, description: str) -> Task:
        task = Task(id=uuid.uuid4().hex[:8], description=description, state=TaskState.GENERATING)
        self.tasks[task.id] = task
        logger.info("task %s created: %s", task.id, description)
        return task

    async def request_publish_approval(self, task: Task, summary: str) -> bool:
        """Suspends until the user taps Approve/Reject in Telegram (or it times out).

        Returns True iff approved; the task's state reflects the outcome
        either way so callers can branch on `task.state` afterward too.
        """
        task.state = TaskState.PENDING_APPROVAL
        logger.info("task %s pending approval", task.id)

        if self._approval_bridge is None or not self._approval_bridge.is_running:
            logger.warning("no Telegram approval bridge configured; task %s cannot be approved", task.id)
            task.state = TaskState.REJECTED
            task.error = "no approval channel configured"
            return False

        approved = await self._approval_bridge.request_approval(summary, timeout=self._approval_timeout)
        task.state = TaskState.PUBLISHING if approved else TaskState.REJECTED
        logger.info("task %s %s", task.id, "approved" if approved else "rejected/timed out")
        return approved

    def mark_done(self, task: Task, result: Any = None) -> None:
        task.state = TaskState.DONE
        task.result = result

    def mark_failed(self, task: Task, error: str) -> None:
        task.state = TaskState.FAILED
        task.error = error
        logger.error("task %s failed: %s", task.id, error)
