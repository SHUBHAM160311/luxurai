"""
LuxurAI — jobs.py
─────────────────────────────────────────────────────────────────
Job / Generation System
- Job creation with idempotency
- Status tracking (queued → running → done / failed)
- Queue position awareness ("You are #4")
- Auto-refund on failure (via wallet.py GenerationBilling)
- Retry logic (max 3 attempts)
- Full job history per user

Integrates with:
    wallet.py  → debit before generate, refund on fail
    auth.py    → user_id validation

Usage:
    job_service = JobService(db_path)
    job = await job_service.create_job(user_id, prompt, resolution, addons)
    await job_service.start_job(job.id)
    await job_service.complete_job(job.id, image_url)
    # or
    await job_service.fail_job(job.id, reason)  # auto-refund triggered
─────────────────────────────────────────────────────────────────
"""

import os
import secrets
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import aiosqlite
from wallet import WalletService, GenerationBilling, calculate_generation_cost


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH    = os.getenv("DB_PATH", "luxurai.db")
MAX_RETRY  = 3          # Max auto-retry attempts before marking FAILED
QUEUE_PRIO = {          # Lower number = higher priority
    "api":  1,
    "paid": 2,
    "free": 3,
}

logger = logging.getLogger("luxurai.jobs")


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
# Exceptions
# ─────────────────────────────────────────────
class JobError(Exception):
    """Base job exception."""

class JobNotFoundError(JobError):
    """Job ID does not exist."""

class JobAlreadyRunningError(JobError):
    """Tried to start a job that isn't queued."""

class JobAlreadyDoneError(JobError):
    """Tried to modify a completed/failed job."""

class InsufficientBalanceError(JobError):
    """User doesn't have enough LC to create job."""


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class Job:
    id:          str
    user_id:     str
    prompt:      str
    resolution:  str
    addons:      Dict[str, bool]   # {"fast": True, "no_watermark": False, ...}
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


# ─────────────────────────────────────────────
# DB Init (call once on startup)
# ─────────────────────────────────────────────
async def init_job_tables(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                prompt       TEXT NOT NULL,
                resolution   TEXT NOT NULL,
                addons       TEXT NOT NULL DEFAULT '{}',   -- JSON string
                cost_lc      REAL NOT NULL,
                status       TEXT NOT NULL DEFAULT 'queued',
                priority     INTEGER NOT NULL DEFAULT 3,
                attempt      INTEGER NOT NULL DEFAULT 0,
                image_url    TEXT,
                fail_reason  TEXT,
                created_at   TEXT NOT NULL,
                started_at   TEXT,
                finished_at  TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_user
                ON jobs(user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_jobs_queue
                ON jobs(status, priority, created_at)
                WHERE status = 'queued';

            CREATE INDEX IF NOT EXISTS idx_jobs_running
                ON jobs(status)
                WHERE status = 'running';
        """)
        await db.commit()
    logger.info("Job tables initialized.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_job_id() -> str:
    return "job_" + secrets.token_urlsafe(12)

def _addons_from_dict(d: Dict[str, bool]) -> str:
    import json
    return json.dumps(d)

def _addons_to_dict(s: str) -> Dict[str, bool]:
    import json
    return json.loads(s) if s else {}

def _row_to_job(row) -> Job:
    return Job(
        id          = row["id"],
        user_id     = row["user_id"],
        prompt      = row["prompt"],
        resolution  = row["resolution"],
        addons      = _addons_to_dict(row["addons"]),
        cost_lc     = float(row["cost_lc"]),
        status      = JobStatus(row["status"]),
        priority    = int(row["priority"]),
        attempt     = int(row["attempt"]),
        image_url   = row["image_url"],
        fail_reason = row["fail_reason"],
        created_at  = row["created_at"],
        started_at  = row["started_at"],
        finished_at = row["finished_at"],
    )


# ─────────────────────────────────────────────
# JobService
# ─────────────────────────────────────────────
class JobService:
    """
    Core job lifecycle management for LuxurAI.
    Handles creation, queueing, execution tracking, and refunds.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.wallet  = WalletService(db_path)
        self.billing = GenerationBilling(self.wallet)

    # ─── Create ────────────────────────────────

    async def create_job(
        self,
        user_id:    str,
        prompt:     str,
        resolution: str,
        addons:     Optional[Dict[str, bool]] = None,
        user_tier:  UserTier = UserTier.FREE,
    ) -> Job:
        """
        Create a new generation job.
        - Calculates cost from pricing engine (wallet.py)
        - Checks balance BEFORE queuing
        - Debits LC immediately (refund on failure)
        - Returns Job object with queue position info
        """
        addons = addons or {}

        # 1. Calculate cost (single source of truth via wallet.py)
        cost_lc = calculate_generation_cost(
            resolution   = resolution,
            fast         = addons.get("fast", False),
            no_watermark = addons.get("no_watermark", False),
            is_api       = (user_tier == UserTier.API),
            bulk         = addons.get("bulk", False),
            bulk_count   = addons.get("bulk_count", 1),
        )

        # 2. Check balance before doing anything
        balance = await self.wallet.get_balance(user_id)
        if balance < cost_lc:
            raise InsufficientBalanceError(
                f"Need {cost_lc} LC, wallet has {balance:.2f} LC. "
                f"Please top up your wallet."
            )

        # 3. Build job record
        job_id   = _new_job_id()
        priority = QUEUE_PRIO.get(user_tier.value, 3)
        now      = _now()

        # 4. Debit LC (wallet.py handles idempotency + ledger)
        await self.wallet.debit(
            user_id   = user_id,
            amount_lc = cost_lc,
            reason    = "image_generation",
            ref_id    = job_id,
        )

        # 5. Insert job row
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """INSERT INTO jobs
                   (id, user_id, prompt, resolution, addons, cost_lc,
                    status, priority, attempt, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id, user_id, prompt, resolution,
                    _addons_from_dict(addons), cost_lc,
                    JobStatus.QUEUED.value, priority, 0, now,
                )
            )
            await db.commit()

            # Fetch back to return full object
            async with db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()

        logger.info(f"Job created: {job_id} | user={user_id} | cost={cost_lc} LC")
        return _row_to_job(row)

    # ─── Queue position ────────────────────────

    async def get_queue_position(self, job_id: str) -> int:
        """
        Returns how many jobs are ahead in queue.
        0 = next to run. -1 = job not queued.
        Useful for UX: 'You are #4 in queue'
        """
        job = await self.get_job(job_id)
        if job.status != JobStatus.QUEUED:
            return -1

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT COUNT(*) FROM jobs
                   WHERE status = 'queued'
                   AND (
                       priority < ?
                       OR (priority = ? AND created_at < ?)
                   )""",
                (job.priority, job.priority, job.created_at)
            ) as cur:
                row = await cur.fetchone()
        return int(row[0])

    # ─── Lifecycle ─────────────────────────────

    async def start_job(self, job_id: str) -> Job:
        """
        Worker calls this when it picks up the job.
        Transitions: QUEUED → RUNNING
        """
        job = await self.get_job(job_id)

        if job.status == JobStatus.RUNNING:
            raise JobAlreadyRunningError(f"Job {job_id} is already running.")
        if job.is_terminal:
            raise JobAlreadyDoneError(f"Job {job_id} is already {job.status.value}.")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """UPDATE jobs
                   SET status = ?, attempt = attempt + 1, started_at = ?
                   WHERE id = ?""",
                (JobStatus.RUNNING.value, _now(), job_id)
            )
            await db.commit()

            async with db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()

        logger.info(f"Job started: {job_id} (attempt {row['attempt']})")
        return _row_to_job(row)

    async def complete_job(self, job_id: str, image_url: str) -> Job:
        """
        Worker calls this when generation succeeds.
        Transitions: RUNNING → DONE
        LC is NOT refunded (already consumed).
        """
        job = await self.get_job(job_id)

        if job.is_terminal:
            raise JobAlreadyDoneError(f"Job {job_id} already {job.status.value}.")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """UPDATE jobs
                   SET status = ?, image_url = ?, finished_at = ?
                   WHERE id = ?""",
                (JobStatus.DONE.value, image_url, _now(), job_id)
            )
            await db.commit()

            async with db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()

        logger.info(f"Job completed: {job_id} → {image_url}")
        completed_job = _row_to_job(row)

        # Fire loyalty rewards (weekly + milestone) — non-blocking
        try:
            from loyalty import check_and_award_loyalty
            await check_and_award_loyalty(completed_job.user_id)
        except Exception as e:
            logger.warning(f"Loyalty check failed for {completed_job.user_id}: {e}")

        return completed_job

    async def fail_job(
        self,
        job_id:      str,
        reason:      str,
        retry:       bool = True,
    ) -> Job:
        """
        Mark job as failed.
        - If retry=True and attempts < MAX_RETRY → re-queues automatically
        - If terminal failure → AUTO-REFUNDS LC via wallet.py
        """
        job = await self.get_job(job_id)

        if job.is_terminal:
            raise JobAlreadyDoneError(f"Job {job_id} already {job.status.value}.")

        should_retry = retry and (job.attempt < MAX_RETRY)

        if should_retry:
            # Re-queue for another attempt
            new_status = JobStatus.QUEUED.value
            logger.warning(
                f"Job {job_id} failed (attempt {job.attempt}/{MAX_RETRY}), "
                f"re-queuing. Reason: {reason}"
            )
        else:
            # Terminal failure → refund LC
            new_status = JobStatus.FAILED.value
            try:
                await self.wallet.refund(
                    user_id        = job.user_id,
                    amount_lc      = job.cost_lc,
                    original_ref_id = job_id,
                )
                logger.info(
                    f"Auto-refunded {job.cost_lc} LC to user {job.user_id} "
                    f"for failed job {job_id}"
                )
            except Exception as e:
                # Refund failed — log loudly, don't crash
                logger.error(
                    f"CRITICAL: Refund failed for job {job_id} "
                    f"(user={job.user_id}, LC={job.cost_lc}). Error: {e}"
                )

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """UPDATE jobs
                   SET status = ?, fail_reason = ?,
                       finished_at = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                   WHERE id = ?""",
                (new_status, reason, new_status, _now(), job_id)
            )
            await db.commit()

            async with db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()

        return _row_to_job(row)

    # ─── Read ──────────────────────────────────

    async def get_job(self, job_id: str) -> Job:
        """Fetch single job by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
        if not row:
            raise JobNotFoundError(f"Job '{job_id}' not found.")
        return _row_to_job(row)

    async def get_user_jobs(
        self,
        user_id: str,
        limit:   int = 20,
        offset:  int = 0,
        status:  Optional[JobStatus] = None,
    ) -> List[Job]:
        """Paginated job history for a user, optionally filtered by status."""
        query  = "SELECT * FROM jobs WHERE user_id = ?"
        params: List[Any] = [user_id]

        if status:
            query  += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
        return [_row_to_job(r) for r in rows]

    async def get_next_queued(self) -> Optional[Job]:
        """
        Worker calls this to get the next job to run.
        Respects priority (API > Paid > Free) then FIFO.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM jobs
                   WHERE status = 'queued'
                   ORDER BY priority ASC, created_at ASC
                   LIMIT 1"""
            ) as cur:
                row = await cur.fetchone()
        return _row_to_job(row) if row else None

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Admin / monitoring: queue snapshot."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
            ) as cur:
                status_rows = await cur.fetchall()

            async with db.execute(
                """SELECT COUNT(*) as count FROM jobs
                   WHERE status = 'queued' AND priority = 1"""
            ) as cur:
                api_q = (await cur.fetchone())["count"]

            async with db.execute(
                """SELECT COUNT(*) as count FROM jobs
                   WHERE status = 'queued' AND priority = 2"""
            ) as cur:
                paid_q = (await cur.fetchone())["count"]

            async with db.execute(
                """SELECT COUNT(*) as count FROM jobs
                   WHERE status = 'queued' AND priority = 3"""
            ) as cur:
                free_q = (await cur.fetchone())["count"]

        stats = {r["status"]: r["count"] for r in status_rows}
        stats["queue_by_tier"] = {
            "api":  api_q,
            "paid": paid_q,
            "free": free_q,
        }
        return stats


# ─────────────────────────────────────────────
# Worker Loop (lightweight — swap for Redis/Celery later)
# ─────────────────────────────────────────────
import asyncio

async def worker_loop(
    job_service: JobService,
    generate_fn,            # async fn(job: Job) -> str  (returns image_url)
    poll_interval: float = 2.0,
):
    """
    Simple in-process worker loop.
    Polls the DB for queued jobs and runs them.

    In production: replace with Redis queue + Celery/ARQ workers.

    Args:
        job_service:   JobService instance
        generate_fn:   Your actual image generation coroutine.
                       Must accept a Job and return an image URL string.
        poll_interval: Seconds between polls when queue is empty.

    ─────────────────────────────────────────────────────────────────
    ⚠️  IMAGE GENERATION API — PLUG IN HERE
    ─────────────────────────────────────────────────────────────────
    Yahan apni image generation API ka async function pass karo.
    generate_fn ek async function hona chahiye jo Job object leta hai
    aur generated image ka URL (string) return karta hai.

    Recommended APIs (koi bhi ek choose karo):
      • Replicate      → https://replicate.com/docs          (SDXL, Flux, etc.)
      • fal.ai         → https://fal.ai/docs                 (Flux, fast inference)
      • Together AI    → https://docs.together.ai            (SDXL, etc.)
      • Stability AI   → https://platform.stability.ai/docs  (Stable Diffusion)
      • RunPod         → https://docs.runpod.io              (self-hosted GPU)

    Example (Replicate ke saath):
        import replicate

        async def my_generator(job: Job) -> str:
            # ▼▼▼ APNI API KEY .env mein daalo: REPLICATE_API_TOKEN=r8_xxx ▼▼▼
            output = await replicate.async_run(
                "black-forest-labs/flux-schnell",   # ← model naam
                input={"prompt": job.prompt, "width": 1024, "height": 1024}
            )
            return output[0]   # image URL

        asyncio.create_task(worker_loop(job_service, my_generator))

    Example (fal.ai ke saath):
        import fal_client

        async def my_generator(job: Job) -> str:
            # ▼▼▼ APNI API KEY .env mein daalo: FAL_KEY=xxx ▼▼▼
            result = await fal_client.run_async(
                "fal-ai/flux/schnell",
                arguments={"prompt": job.prompt}
            )
            return result["images"][0]["url"]

        asyncio.create_task(worker_loop(job_service, my_generator))
    ─────────────────────────────────────────────────────────────────
    """
    logger.info("Worker loop started.")
    while True:
        job = await job_service.get_next_queued()

        if not job:
            await asyncio.sleep(poll_interval)
            continue

        try:
            job = await job_service.start_job(job.id)
            logger.info(f"Worker picked up job {job.id}")

            image_url = await generate_fn(job)
            await job_service.complete_job(job.id, image_url)

        except asyncio.CancelledError:
            # Graceful shutdown — re-queue if job was running
            if job and job.status == JobStatus.RUNNING:
                await job_service.fail_job(
                    job.id, reason="worker_shutdown", retry=True
                )
            raise

        except Exception as e:
            logger.error(f"Job {job.id} failed in worker: {e}", exc_info=True)
            await job_service.fail_job(
                job.id, reason=str(e), retry=True
            )

        # Tight loop when there's work
        await asyncio.sleep(0.1)


# ─────────────────────────────────────────────
# FastAPI Router (plug into your main app)
# ─────────────────────────────────────────────
try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request
    from pydantic import BaseModel

    router = APIRouter(prefix="/jobs", tags=["jobs"])

    # ── Request / Response models ──────────────
    class CreateJobRequest(BaseModel):
        prompt:     str
        resolution: str = "1024x1024"
        fast:       bool = False
        no_watermark: bool = False

    class JobResponse(BaseModel):
        job_id:         str
        status:         str
        cost_lc:        float
        queue_position: Optional[int] = None
        image_url:      Optional[str] = None
        fail_reason:    Optional[str] = None
        created_at:     str

    # ── Real auth dependency ──────────────────────
    async def get_current_user_id(request: Request) -> str:
        import os, jwt as pyjwt
        from fastapi import Cookie
        secret = os.getenv("JWT_SECRET", os.getenv("SECRET_KEY", "dev-secret"))
        token  = request.cookies.get("session_token") or request.headers.get("Authorization","").replace("Bearer ","")
        if not token:
            raise HTTPException(401, "Not authenticated")
        try:
            payload = pyjwt.decode(token, secret, algorithms=["HS256"])
            return payload["sub"]
        except Exception:
            raise HTTPException(401, "Invalid or expired session")

    def get_job_service() -> JobService:
        return JobService(DB_PATH)

    # ── Endpoints ──────────────────────────────

    @router.post("/", response_model=JobResponse, status_code=201)
    async def create_job(
        req:        CreateJobRequest,
        user_id:    str        = Depends(get_current_user_id),
        jobs:       JobService = Depends(get_job_service),
    ):
        """Submit a new image generation job."""
        try:
            job = await jobs.create_job(
                user_id    = user_id,
                prompt     = req.prompt,
                resolution = req.resolution,
                addons     = {
                    "fast":         req.fast,
                    "no_watermark": req.no_watermark,
                },
            )
            position = await jobs.get_queue_position(job.id)
            return JobResponse(
                job_id         = job.id,
                status         = job.status.value,
                cost_lc        = job.cost_lc,
                queue_position = position,
                image_url      = job.image_url,
                created_at     = job.created_at,
            )

        except InsufficientBalanceError as e:
            raise HTTPException(status_code=402, detail=str(e))
        except Exception as e:
            logger.error(f"create_job error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Job creation failed.")

    @router.get("/{job_id}", response_model=JobResponse)
    async def get_job_status(
        job_id:  str,
        user_id: str        = Depends(get_current_user_id),
        jobs:    JobService = Depends(get_job_service),
    ):
        """Poll job status. Frontend uses this for live updates."""
        try:
            job = await jobs.get_job(job_id)
            if job.user_id != user_id:
                raise HTTPException(status_code=403, detail="Not your job.")

            position = await jobs.get_queue_position(job_id)
            return JobResponse(
                job_id         = job.id,
                status         = job.status.value,
                cost_lc        = job.cost_lc,
                queue_position = position if position >= 0 else None,
                image_url      = job.image_url,
                fail_reason    = job.fail_reason,
                created_at     = job.created_at,
            )

        except JobNotFoundError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    @router.get("/", response_model=List[JobResponse])
    async def list_my_jobs(
        limit:   int        = Query(20, le=100),
        offset:  int        = Query(0),
        user_id: str        = Depends(get_current_user_id),
        jobs:    JobService = Depends(get_job_service),
    ):
        """Get paginated job history for current user."""
        job_list = await jobs.get_user_jobs(user_id, limit=limit, offset=offset)
        return [
            JobResponse(
                job_id      = j.id,
                status      = j.status.value,
                cost_lc     = j.cost_lc,
                image_url   = j.image_url,
                fail_reason = j.fail_reason,
                created_at  = j.created_at,
            )
            for j in job_list
        ]

    @router.get("/admin/stats")
    async def queue_stats(jobs: JobService = Depends(get_job_service)):
        """Admin endpoint: queue health snapshot."""
        return await jobs.get_queue_stats()

except ImportError:
    # FastAPI not installed — skip router
    router = None
    logger.info("FastAPI not found — router not registered.")
