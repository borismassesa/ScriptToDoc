"""Simple in-memory job registry for tracking progress of transcript processing."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"
    progress: float = 0.0
    current_step: str = "queued"
    detail: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(self) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record:
                record.updated_at = time.time()
        return record

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = progress
            if current_step is not None:
                record.current_step = current_step
            if detail is not None:
                record.detail = detail
            if metadata:
                record.metadata.update(metadata)
            record.updated_at = time.time()

    def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = "completed"
            record.progress = 1.0
            record.current_step = "completed"
            record.result = result
            record.updated_at = time.time()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = "failed"
            record.error = error
            record.current_step = "failed"
            record.updated_at = time.time()


job_manager = JobManager()
