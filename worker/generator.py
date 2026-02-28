"""
LuxurAI — worker/generator.py
─────────────────────────────────────────────────────────────────
Image Generation Worker

Ye file:
  1. Jobs queue se job uthata hai (jobs.py via worker_loop)
  2. Image generation API call karta hai   ← APNI API YAHAN DAALO
  3. Generated image ko S3 pe upload karta hai (storage/s3.py)
  4. Permanent S3 URL DB mein save karta hai

Run karo:
  python worker/generator.py

Ya main.py ke saath background task mein:
  asyncio.create_task(start_worker())

─────────────────────────────────────────────────────────────────
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

# Project imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs import JobService, worker_loop, Job
from storage.s3 import upload_from_url, check_s3_connection

logger = logging.getLogger("luxurai.generator")
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DB_PATH = os.getenv("DB_PATH", "luxurai.db")


# ─────────────────────────────────────────────
# ⚠️  IMAGE GENERATION API — YAHAN DAALO
# ─────────────────────────────────────────────
# Niche do examples hain — Replicate aur fal.ai
# Ek uncomment karo aur .env mein API key daalo
# ─────────────────────────────────────────────

# ── OPTION 1: Replicate ──────────────────────
# pip install replicate
# .env mein: REPLICATE_API_TOKEN=r8_xxxxxxxxxxxx
#
# import replicate
#
# async def _call_generation_api(job: Job) -> str:
#     """Replicate se image generate karo."""
#     width, height = _parse_resolution(job.resolution)
#
#     output = await replicate.async_run(
#         "black-forest-labs/flux-schnell",   # ← Model naam badal sakte ho
#         # Other good models:
#         # "black-forest-labs/flux-dev"
#         # "stability-ai/sdxl"
#         input={
#             "prompt":          job.prompt,
#             "width":           width,
#             "height":          height,
#             "num_outputs":     1,
#             "output_format":   "png",
#         }
#     )
#     return output[0]   # Replicate list return karta hai, pehla URL lo


# ── OPTION 2: fal.ai ─────────────────────────
# pip install fal-client
# .env mein: FAL_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#
# import fal_client
#
# async def _call_generation_api(job: Job) -> str:
#     """fal.ai se image generate karo."""
#     width, height = _parse_resolution(job.resolution)
#
#     result = await fal_client.run_async(
#         "fal-ai/flux/schnell",   # ← Model naam badal sakte ho
#         # Other good models:
#         # "fal-ai/flux/dev"
#         # "fal-ai/stable-diffusion-xl"
#         arguments={
#             "prompt":       job.prompt,
#             "image_size": {
#                 "width":  width,
#                 "height": height,
#             },
#             "num_images": 1,
#         }
#     )
#     return result["images"][0]["url"]


# ── OPTION 3: Together AI ─────────────────────
# pip install together
# .env mein: TOGETHER_API_KEY=xxxxxxxxxxxxxxxx
#
# from together import AsyncTogether
#
# async def _call_generation_api(job: Job) -> str:
#     """Together AI se image generate karo."""
#     client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
#     width, height = _parse_resolution(job.resolution)
#
#     response = await client.images.generate(
#         prompt = job.prompt,
#         model  = "black-forest-labs/FLUX.1-schnell-Free",
#         width  = width,
#         height = height,
#         n      = 1,
#     )
#     return response.data[0].url


# ─────────────────────────────────────────────
# Resolution Parser
# ─────────────────────────────────────────────
def _parse_resolution(resolution: str) -> tuple[int, int]:
    """
    "1024x1024" → (1024, 1024)
    "768x1344"  → (768, 1344)
    """
    try:
        w, h = resolution.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1024   # Default fallback


# ─────────────────────────────────────────────
# Main Generator Function
# ─────────────────────────────────────────────
async def generate(job: Job) -> str:
    """
    worker_loop() is function ko call karta hai.

    Steps:
      1. Generation API call karo → temporary image URL milti hai
      2. S3 pe permanently upload karo → permanent URL milta hai
      3. Permanent URL return karo → DB mein save hoga

    Args:
        job: Job object (prompt, resolution, addons sab isme hai)

    Returns:
        Permanent S3 image URL
    """
    logger.info(f"Generating image for job {job.id} | prompt: {job.prompt[:50]}...")

    # ── Step 1: Image generate karo ──────────
    # ⚠️  _call_generation_api ko uncomment karo (upar se ek option choose karo)
    temp_url = await _call_generation_api(job)
    logger.info(f"Job {job.id} → temp URL received: {temp_url[:60]}...")

    # ── Step 2: S3 pe upload karo ────────────
    permanent_url = await upload_from_url(
        image_url = temp_url,
        job_id    = job.id,
        extension = "png",
    )

    logger.info(f"Job {job.id} → stored at: {permanent_url}")
    return permanent_url


# ─────────────────────────────────────────────
# Worker Entry Point
# ─────────────────────────────────────────────
async def start_worker():
    """
    Worker start karo.
    main.py mein background task ki tarah run karo:

        from worker.generator import start_worker
        asyncio.create_task(start_worker())
    """
    # S3 connection check
    s3_status = check_s3_connection()
    if not s3_status["ok"]:
        logger.error(f"❌ S3 not ready: {s3_status['error']}")
        logger.error("Worker will not start. Fix S3 config first.")
        return
    logger.info(f"✓ S3 ready → bucket: {s3_status['bucket']}")

    # Job service
    job_service = JobService(DB_PATH)

    logger.info("🚀 LuxurAI worker started — waiting for jobs...")

    # Start the loop — yahan generate() pass ho raha hai
    await worker_loop(
        job_service  = job_service,
        generate_fn  = generate,
        poll_interval = float(os.getenv("POLL_INTERVAL", "2.0")),
    )


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(start_worker())
