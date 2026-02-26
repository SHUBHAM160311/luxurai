"""
LuxurAI Generation Routes
Core image generation endpoint with LC cost calculation,
queue management, and image delivery
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from typing import Optional
import asyncio
import logging
import os
from pathlib import Path

import database as db
from routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


# ==================== LC PRICING ====================
def calculate_lc_cost(width: int, height: int, fast_gen: bool, no_watermark: bool) -> float:
    """Calculate total LC cost for a generation"""

    # Base image cost by resolution
    pixels = width * height

    if width <= 100 and height <= 100:
        base = 0.5
    elif width <= 200 and height <= 200:
        base = 1.0
    elif width == height and width <= 500:
        base = 4.0
    elif width <= 500 and height <= 500:
        # One side bigger
        if max(width, height) >= 500:
            base = 2.5
        else:
            base = 1.25
    elif width <= 500 or height <= 500:
        base = 2.5
    elif width == height and width <= 1024:
        if width == 1024:
            base = 10.0
        else:
            base = 6.0
    elif max(width, height) <= 1024 and min(width, height) <= 500:
        base = 7.0
    elif max(width, height) > 1024 and min(width, height) > 1024:
        # Both bigger than 1024
        base = 10.0
    elif max(width, height) > 1024:
        base = 9.0
    else:
        base = 8.0

    total = base
    if fast_gen:
        total += 3.0
    if no_watermark:
        total += 7.0

    return round(total, 2)


# ==================== SCHEMAS ====================
class GenerateRequest(BaseModel):
    prompt: str
    width: int = 200
    height: int = 200
    fast_gen: bool = False
    no_watermark: bool = False

    @validator('prompt')
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        if len(v) > 500:
            raise ValueError("Prompt too long (max 500 chars)")
        return v.strip()

    @validator('width', 'height')
    def valid_dimensions(cls, v):
        if v < 64:
            raise ValueError("Minimum size is 64px")
        if v > 1024:
            raise ValueError("Maximum size is 1024px")
        return v


class GenerateResponse(BaseModel):
    job_id: str
    lc_cost: float
    lc_balance_remaining: float
    shown_wait_seconds: int
    queue_position: int
    user_type: str


# ==================== ROUTES ====================
@router.post("/", response_model=GenerateResponse)
async def generate_image(
    body: GenerateRequest,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit image generation request
    - Calculates LC cost
    - Checks balance
    - Deducts LC
    - Adds to queue
    - Returns job_id for polling
    """

    user_id = current_user["id"]

    # Calculate cost
    lc_cost = calculate_lc_cost(
        body.width, body.height,
        body.fast_gen, body.no_watermark
    )

    # Check balance
    balance = db.get_lc_balance(user_id)
    if balance < lc_cost:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "Insufficient LC balance",
                "required": lc_cost,
                "available": balance,
                "needed": round(lc_cost - balance, 2)
            }
        )

    # Count recent requests (spammer detection)
    conn = db.get_conn()
    recent_count = conn.execute("""
        SELECT COUNT(*) as cnt FROM generations
        WHERE user_id = ?
        AND created_at >= datetime('now', '-1 minute')
    """, (user_id,)).fetchone()["cnt"]
    conn.close()

    # Deduct LC
    deducted = db.deduct_lc(
        user_id, lc_cost,
        f"Generate {body.width}x{body.height}",
    )
    if not deducted:
        raise HTTPException(status_code=402, detail="Failed to deduct LC")

    # Update loyalty counters
    if body.fast_gen:
        db.increment_fast_gen_count(user_id)
    if body.no_watermark:
        db.increment_no_wm_count(user_id)

    # Get queue manager from app state
    queue_manager = request.app.state.queue_manager if hasattr(request.app.state, 'queue_manager') else None

    # Import global queue manager
    from main import queue_manager as qm

    # Add to queue
    job = await qm.enqueue(
        user_id=user_id,
        prompt=body.prompt,
        width=body.width,
        height=body.height,
        lc_cost=lc_cost,
        fast_gen=body.fast_gen,
        no_watermark=body.no_watermark,
        requests_last_minute=recent_count
    )

    # Save to DB
    db.create_generation(
        gen_id=job.job_id,
        user_id=user_id,
        prompt=body.prompt,
        width=body.width,
        height=body.height,
        lc_cost=lc_cost,
        fast_gen=body.fast_gen,
        no_watermark=body.no_watermark,
        queue_position=qm.get_queue_position(job.user_type),
        wait_time_shown=job.shown_wait_seconds
    )

    # Check loyalty rewards (non-blocking)
    rewards = db.check_and_award_loyalty(user_id)
    if rewards:
        logger.info(f"🎁 Loyalty rewards for user {user_id}: {rewards}")

    new_balance = db.get_lc_balance(user_id)

    return GenerateResponse(
        job_id=job.job_id,
        lc_cost=lc_cost,
        lc_balance_remaining=new_balance,
        shown_wait_seconds=job.shown_wait_seconds,
        queue_position=qm.get_queue_position(job.user_type),
        user_type=job.user_type.value
    )


@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Poll generation status"""
    from main import queue_manager as qm

    gen = db.get_generation(job_id)
    if not gen:
        raise HTTPException(status_code=404, detail="Job not found")

    # Security: only owner can check
    if gen["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not your job")

    response = {
        "job_id": job_id,
        "status": gen["status"],
        "shown_wait": gen["wait_time_shown"],
        "queue_position": gen["queue_position"],
        "prompt": gen["prompt"],
        "dimensions": f"{gen['width']}x{gen['height']}",
    }

    if gen["status"] == "done":
        response["image_url"] = f"/api/generate/image/{job_id}"
        response["real_time_ms"] = gen["real_time_ms"]

    return response


@router.get("/image/{job_id}")
async def get_image(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download generated image"""
    gen = db.get_generation(job_id)

    if not gen:
        raise HTTPException(status_code=404, detail="Not found")
    if gen["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not your image")
    if gen["status"] != "done":
        raise HTTPException(status_code=425, detail="Image not ready yet")

    image_path = Path(gen["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")

    return FileResponse(
        str(image_path),
        media_type="image/png",
        filename=f"luxurai_{job_id[:8]}.png"
    )


@router.get("/pricing")
async def get_pricing():
    """Get current LC pricing for all resolutions"""
    return {
        "resolutions": [
            {"label": "100×100", "lc": 0.5},
            {"label": "200×200", "lc": 1.0},
            {"label": "One side 200+", "lc": 1.25},
            {"label": "Both 200+", "lc": 2.0},
            {"label": "One side 500+", "lc": 2.5},
            {"label": "Both 500", "lc": 4.0},
            {"label": "Both 500+", "lc": 6.0},
            {"label": "One side 1024", "lc": 7.0},
            {"label": "Both 1024", "lc": 10.0},
            {"label": "One side 1024+", "lc": 9.0},
        ],
        "addons": {
            "fast_gen": 3.0,
            "no_watermark": 7.0,
        },
        "lc_rate_inr": 0.25
    }
