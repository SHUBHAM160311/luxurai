"""
LuxurAI — models/job.py
─────────────────────────────────────────────────────────────────
Jobs table definition + dataclass.
No logic here — only structure.
─────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class UserTier(str, Enum):
    FREE = "free"
    PAID = "paid"
    API  = "api"


# ─────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────
JOBS_TABLE = """
    CREATE TABLE IF NOT EXISTS jobs (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id),
        prompt      TEXT NOT NULL,
        resolution  TEXT NOT NULL,
        addons      TEXT NOT NULL DEFAULT '{}',
        cost_lc     REAL NOT NULL,
        status      TEXT NOT NULL DEFAULT 'queued',
        priority    INTEGER NOT NULL DEFAULT 3,
        attempt     INTEGER NOT NULL DEFAULT 0,
        image_url   TEXT,
        fail_reason TEXT,
        created_at  TEXT NOT NULL,
        started_at  TEXT,
        finished_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_jobs_user
        ON jobs(user_id, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_jobs_queue
        ON jobs(status, priority, created_at)
        WHERE status = 'queued';
"""

# Queue priority mapping (lower = higher priority)
QUEUE_PRIORITY = {
    UserTier.API:  1,
    UserTier.PAID: 2,
    UserTier.FREE: 3,
}


# ─────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────
@dataclass
class Job:
    id:          str
    user_id:     str
    prompt:      str
    resolution:  str
    addons:      dict        # {"fast": True, "no_watermark": False}
    cost_lc:     float
    status:      JobStatus
    priority:    int
    attempt:     int
    image_url:   Optional[str]
    fail_reason: Optional[str]
    created_at:  str
    started_at:  Optional[str]
    finished_at: Optional[str]

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.FAILED)

    @property
    def is_queued(self) -> bool:
        return self.status == JobStatus.QUEUED
