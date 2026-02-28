"""
LuxurAI — worker/generator.py
─────────────────────────────────────────────────────────────────
Image Generation Worker using RunPod Serverless API

Setup:
  1. Add to Railway Variables (or .env locally):
       RUNPOD_API_KEY=your_runpod_api_key_here
       RUNPOD_ENDPOINT_ID=your_endpoint_id_here   (from RunPod dashboard)

  2. Recommended RunPod model: FLUX.1-schnell or SDXL
     Deploy a serverless endpoint on RunPod, copy the endpoint ID.

Run worker standalone:
    python worker/generator.py

Or launched automatically by main.py on startup.
─────────────────────────────────────────────────────────────────
"""

import os
import asyncio
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs import JobService, worker_loop, Job
from storage.s3 import upload_from_url, check_s3_connection

logger = logging.getLogger("luxurai.generator")
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DB_PATH          = os.getenv("DB_PATH", "luxurai.db")
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT  = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE      = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}"

# Timeout settings
SUBMIT_TIMEOUT   = 30    # seconds to submit job to RunPod
POLL_INTERVAL    = 2     # seconds between status polls
MAX_WAIT         = 300   # max seconds to wait for RunPod result (5 min)


# ─────────────────────────────────────────────
# Resolution Parser
# ─────────────────────────────────────────────
def _parse_resolution(resolution: str) -> tuple[int, int]:
    """'1024×1024' or '1024x1024' → (1024, 1024)"""
    try:
        w, h = resolution.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1024


# ─────────────────────────────────────────────
# RunPod API — submit + poll
# ─────────────────────────────────────────────
async def _submit_runpod_job(job: Job) -> str:
    """
    Submit image generation job to RunPod serverless endpoint.
    Returns the RunPod job_id for polling.
    """
    width, height = _parse_resolution(job.resolution)

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }

    # Payload format for RunPod FLUX / SDXL workers
    # Adjust 'input' fields based on your RunPod worker template
    payload = {
        "input": {
            "prompt":      job.prompt,
            "width":       width,
            "height":      height,
            "num_outputs": 1,
            "output_format": "png",
        }
    }

    async with httpx.AsyncClient(timeout=SUBMIT_TIMEOUT) as client:
        resp = await client.post(
            f"{RUNPOD_BASE}/run",
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"RunPod submit failed [{resp.status_code}]: {resp.text}")

    data = resp.json()
    runpod_job_id = data.get("id")
    if not runpod_job_id:
        raise RuntimeError(f"RunPod returned no job ID: {data}")

    logger.info(f"RunPod job submitted: {runpod_job_id} for LuxurAI job {job.id}")
    return runpod_job_id


async def _poll_runpod_job(runpod_job_id: str) -> str:
    """
    Poll RunPod until job completes. Returns image URL.
    Raises on failure or timeout.
    """
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    elapsed = 0
    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < MAX_WAIT:
            resp = await client.get(
                f"{RUNPOD_BASE}/status/{runpod_job_id}",
                headers=headers,
            )

            if resp.status_code != 200:
                raise RuntimeError(f"RunPod status check failed [{resp.status_code}]: {resp.text}")

            data   = resp.json()
            status = data.get("status", "")

            if status == "COMPLETED":
                output = data.get("output")
                if not output:
                    raise RuntimeError(f"RunPod completed but no output: {data}")
                # Output is list of image URLs or a dict with 'images'
                if isinstance(output, list):
                    return output[0]
                if isinstance(output, dict):
                    images = output.get("images") or output.get("image_url") or output.get("url")
                    if isinstance(images, list):
                        return images[0]
                    if isinstance(images, str):
                        return images
                raise RuntimeError(f"Unexpected RunPod output format: {output}")

            elif status == "FAILED":
                error = data.get("error", "Unknown RunPod error")
                raise RuntimeError(f"RunPod job failed: {error}")

            elif status in ("IN_QUEUE", "IN_PROGRESS", "EXECUTING"):
                logger.debug(f"RunPod job {runpod_job_id} status: {status}, waiting...")
                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

            else:
                logger.warning(f"Unknown RunPod status: {status}, continuing to wait...")
                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

    raise RuntimeError(f"RunPod job {runpod_job_id} timed out after {MAX_WAIT}s")


# ─────────────────────────────────────────────
# Main Generator Function
# ─────────────────────────────────────────────
async def generate(job: Job) -> str:
    """
    Called by worker_loop() for each job.

    Steps:
      1. Submit to RunPod → get RunPod job ID
      2. Poll until COMPLETED → get temp image URL
      3. Upload to S3 → get permanent URL
      4. Return permanent URL → saved to DB
    """
    logger.info(f"Starting generation | job={job.id} | prompt={job.prompt[:60]}...")

    # Step 1: Submit to RunPod
    runpod_job_id = await _submit_runpod_job(job)

    # Step 2: Poll for result
    temp_url = await _poll_runpod_job(runpod_job_id)
    logger.info(f"RunPod result received for job {job.id}: {temp_url[:80]}")

    # Step 3: Upload to S3 for permanent storage
    permanent_url = await upload_from_url(
        image_url = temp_url,
        job_id    = job.id,
        extension = "png",
    )

    logger.info(f"Job {job.id} complete → {permanent_url}")
    return permanent_url


# ─────────────────────────────────────────────
# Worker Entry Point
# ─────────────────────────────────────────────
async def start_worker():
    """Start the generation worker. Called from main.py on startup."""

    # Validate RunPod config
    if not RUNPOD_API_KEY:
        logger.error("❌ RUNPOD_API_KEY not set in environment. Worker cannot start.")
        logger.error("   Add RUNPOD_API_KEY to Railway Variables or .env file.")
        return

    if not RUNPOD_ENDPOINT:
        logger.error("❌ RUNPOD_ENDPOINT_ID not set in environment. Worker cannot start.")
        logger.error("   Create a serverless endpoint on RunPod dashboard, copy the ID.")
        return

    # S3 connection check
    s3_status = check_s3_connection()
    if not s3_status["ok"]:
        logger.error(f"❌ S3 not ready: {s3_status['error']}")
        logger.error("   Add S3 credentials to Railway Variables or .env file.")
        return

    logger.info(f"✓ RunPod endpoint: {RUNPOD_ENDPOINT}")
    logger.info(f"✓ S3 bucket: {s3_status['bucket']}")
    logger.info("🚀 LuxurAI worker started — waiting for jobs...")

    job_service = JobService(DB_PATH)

    await worker_loop(
        job_service   = job_service,
        generate_fn   = generate,
        poll_interval = float(os.getenv("POLL_INTERVAL", "2.0")),
    )


# ─────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(start_worker())
