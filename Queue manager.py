"""
LuxurAI Queue Manager
6-batch system:
  Batch 0 → Spammer (FASTEST - burn LC! 😎)
  Batch 1-2 → Normal users
  Batch 3-4 → Fast Gen users
  Batch 5 → Buffer/overflow

Artificial delays to make it feel premium.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Callable
import random

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
TOTAL_BATCHES = 6

# Artificial wait times shown to user (seconds)
SHOWN_WAIT = {
    "normal": (25, 35),       # shown: 25-35s | real: ~3s
    "fast_gen": (8, 12),      # shown: 8-12s  | real: ~2s
    "spammer": (3, 6),        # shown: 3-6s   | FASTEST real (burn LC!)
}

# Batch assignments
BATCH_SPAMMER = 0
BATCH_NORMAL = [1, 2]
BATCH_FAST_GEN = [3, 4]
BATCH_BUFFER = 5


# ==================== DATA CLASSES ====================
class UserType(Enum):
    NORMAL = "normal"
    FAST_GEN = "fast_gen"
    SPAMMER = "spammer"


@dataclass
class GenerationJob:
    job_id: str
    user_id: int
    prompt: str
    width: int
    height: int
    lc_cost: float
    fast_gen: bool
    no_watermark: bool
    user_type: UserType
    priority: int                   # lower = higher priority
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future = None

    # What user SEES
    shown_wait_seconds: int = 30

    # Batch assignment
    batch: int = 1


# ==================== QUEUE MANAGER ====================
class QueueManager:
    def __init__(self):
        # 6 separate queues for 6 batches
        self.queues: Dict[int, deque] = {i: deque() for i in range(TOTAL_BATCHES)}
        self.active_jobs: Dict[str, GenerationJob] = {}
        self.batch_busy: Dict[int, bool] = {i: False for i in range(TOTAL_BATCHES)}

        # Callbacks
        self.generation_callback: Optional[Callable] = None

        self._running = False
        self._tasks = []

        logger.info("✅ QueueManager initialized (6 batches)")

    def set_generation_callback(self, callback: Callable):
        """Set the actual JAX model generation function"""
        self.generation_callback = callback

    # ==================== CLASSIFY USER ====================
    def classify_user(self, user_id: int, fast_gen: bool, requests_last_minute: int) -> UserType:
        """
        Spammer detection:
        - More than 8 requests in last minute = spammer
        - Gets fastest batch (burns LC faster 😎)
        """
        if requests_last_minute >= 8:
            return UserType.SPAMMER
        if fast_gen:
            return UserType.FAST_GEN
        return UserType.NORMAL

    def assign_batch(self, user_type: UserType) -> int:
        """Assign job to least busy batch of appropriate type"""
        if user_type == UserType.SPAMMER:
            return BATCH_SPAMMER

        if user_type == UserType.FAST_GEN:
            # Pick least loaded fast gen batch
            return min(BATCH_FAST_GEN, key=lambda b: len(self.queues[b]))

        # Normal user - pick least loaded normal batch or buffer
        candidates = BATCH_NORMAL + [BATCH_BUFFER]
        return min(candidates, key=lambda b: len(self.queues[b]))

    def get_shown_wait(self, user_type: UserType, queue_len: int) -> int:
        """Calculate artificial wait time shown to user"""
        base_min, base_max = SHOWN_WAIT[user_type.value]

        # Add extra time per person ahead in queue
        extra = queue_len * random.randint(3, 7)

        total = random.randint(base_min, base_max) + extra
        return min(total, 180)  # cap at 3 minutes

    def get_queue_position(self, user_type: UserType) -> int:
        """Return queue position for user type"""
        if user_type == UserType.SPAMMER:
            return len(self.queues[BATCH_SPAMMER]) + 1
        if user_type == UserType.FAST_GEN:
            return min(len(self.queues[b]) for b in BATCH_FAST_GEN) + 1
        return min(len(self.queues[b]) for b in BATCH_NORMAL) + 1

    # ==================== ENQUEUE ====================
    async def enqueue(
        self,
        user_id: int,
        prompt: str,
        width: int,
        height: int,
        lc_cost: float,
        fast_gen: bool,
        no_watermark: bool,
        requests_last_minute: int = 0
    ) -> GenerationJob:
        """Add job to appropriate queue"""

        user_type = self.classify_user(user_id, fast_gen, requests_last_minute)
        batch = self.assign_batch(user_type)
        queue_pos = self.get_queue_position(user_type)
        shown_wait = self.get_shown_wait(user_type, queue_pos - 1)

        job = GenerationJob(
            job_id=str(uuid.uuid4()),
            user_id=user_id,
            prompt=prompt,
            width=width,
            height=height,
            lc_cost=lc_cost,
            fast_gen=fast_gen,
            no_watermark=no_watermark,
            user_type=user_type,
            priority=0 if user_type == UserType.SPAMMER else (1 if user_type == UserType.FAST_GEN else 2),
            shown_wait_seconds=shown_wait,
            batch=batch,
            future=asyncio.get_event_loop().create_future()
        )

        self.queues[batch].append(job)
        self.active_jobs[job.job_id] = job

        logger.info(
            f"📥 Queued job {job.job_id[:8]} | "
            f"User:{user_id} | Type:{user_type.value} | "
            f"Batch:{batch} | Pos:{queue_pos} | "
            f"Shown wait:{shown_wait}s"
        )

        return job

    # ==================== WORKERS ====================
    async def start(self):
        """Start all 6 batch workers"""
        self._running = True
        for batch_id in range(TOTAL_BATCHES):
            task = asyncio.create_task(self._batch_worker(batch_id))
            self._tasks.append(task)
        logger.info(f"🚀 Started {TOTAL_BATCHES} batch workers")

    async def stop(self):
        """Stop all workers"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("🛑 Queue workers stopped")

    async def _batch_worker(self, batch_id: int):
        """Worker for a single batch"""
        batch_name = {
            0: "SPAMMER🔥",
            1: "Normal-A",
            2: "Normal-B",
            3: "FastGen-A",
            4: "FastGen-B",
            5: "Buffer"
        }.get(batch_id, f"Batch-{batch_id}")

        logger.info(f"  Worker [{batch_name}] started")

        while self._running:
            if not self.queues[batch_id]:
                await asyncio.sleep(0.1)
                continue

            job = self.queues[batch_id].popleft()
            self.batch_busy[batch_id] = True

            try:
                await self._process_job(job, batch_name)
            except Exception as e:
                logger.error(f"❌ Job {job.job_id[:8]} failed in {batch_name}: {e}")
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                self.batch_busy[batch_id] = False
                self.active_jobs.pop(job.job_id, None)

    async def _process_job(self, job: GenerationJob, batch_name: str):
        """Process a single generation job"""
        start_time = time.time()

        logger.info(f"⚙️  Processing [{batch_name}] job {job.job_id[:8]} | {job.width}x{job.height}")

        # Update status to processing
        from database import update_generation_status
        update_generation_status(job.job_id, "processing")

        try:
            if self.generation_callback:
                # Call actual JAX model
                image_path = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.generation_callback,
                    job.prompt,
                    job.width,
                    job.height,
                    job.no_watermark,
                    job.job_id
                )
            else:
                # Placeholder (model not loaded yet)
                await asyncio.sleep(0.5)
                image_path = f"outputs/placeholder_{job.job_id}.png"

            real_time_ms = int((time.time() - start_time) * 1000)

            # Mark done
            update_generation_status(job.job_id, "done", image_path, real_time_ms)

            logger.info(
                f"✅ Done [{batch_name}] job {job.job_id[:8]} | "
                f"Real time: {real_time_ms}ms"
            )

            if not job.future.done():
                job.future.set_result({
                    "job_id": job.job_id,
                    "image_path": image_path,
                    "real_time_ms": real_time_ms
                })

        except Exception as e:
            update_generation_status(job.job_id, "failed")
            raise e

    # ==================== STATUS ====================
    def get_job_status(self, job_id: str) -> dict:
        """Get current status of a job"""
        from database import get_generation
        gen = get_generation(job_id)
        if not gen:
            return {"error": "Job not found"}

        result = {
            "job_id": job_id,
            "status": gen["status"],
            "shown_wait": gen["wait_time_shown"],
            "queue_position": gen["queue_position"],
        }

        if gen["status"] == "done":
            result["image_url"] = f"/api/generate/image/{job_id}"

        return result

    def total_queued(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def active_batches(self) -> int:
        return sum(1 for busy in self.batch_busy.values() if busy)

    def queue_stats(self) -> dict:
        return {
            f"batch_{i}": {
                "queued": len(self.queues[i]),
                "busy": self.batch_busy[i]
            }
            for i in range(TOTAL_BATCHES)
        }