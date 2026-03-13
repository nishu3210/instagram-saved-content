"""Thread-safe status management for background jobs."""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JobStatus:
    """Status of a background job."""

    running: bool = False
    progress: int = 0
    message: str = ""
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class ThreadSafeStatusManager:
    """Thread-safe manager for job status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = JobStatus()

    def get_status(self) -> JobStatus:
        """Get current status (thread-safe)."""
        with self._lock:
            return JobStatus(
                running=self._status.running,
                progress=self._status.progress,
                message=self._status.message,
                error=self._status.error,
                results=self._status.results.copy() if self._status.results else None,
            )

    def start(self, message: str = "Initializing...") -> None:
        """Mark job as started."""
        with self._lock:
            self._status.running = True
            self._status.progress = 0
            self._status.message = message
            self._status.error = None
            self._status.results = None

    def update_progress(self, progress: int, message: str) -> None:
        """Update progress."""
        with self._lock:
            self._status.progress = min(100, max(0, progress))
            self._status.message = message

    def complete(
        self, results: Optional[Dict[str, Any]] = None, message: str = "Complete!"
    ) -> None:
        """Mark job as complete."""
        with self._lock:
            self._status.running = False
            self._status.progress = 100
            self._status.message = message
            self._status.results = results

    def fail(self, error: str) -> None:
        """Mark job as failed."""
        with self._lock:
            self._status.running = False
            self._status.error = error

    def reset(self) -> None:
        """Reset status to initial state."""
        with self._lock:
            self._status = JobStatus()


# Global status managers
analysis_status = ThreadSafeStatusManager()
task_job_status = ThreadSafeStatusManager()
