"""
LuxurAI Wallet Routes
LC balance, transaction history, loyalty rewards
"""

from fastapi import APIRouter, Depends, HTTPException
import database as db
from routes.auth import get_current_user

router = APIRouter()


# ==================== ROUTES ====================
@router.get("/balance")
async def get_balance(current_user: dict = Depends(get_current_user)):
    """Get current LC balance"""
    return {
        "lc_balance": current_user["lc_balance"],
        "total_generated": current_user["total_generated"]
    }


@router.get("/transactions")
async def get_transactions(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get LC transaction history"""
    txns = db.get_transaction_history(current_user["id"], limit)
    return {"transactions": txns}


@router.get("/history")
async def get_generation_history(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get image generation history"""
    gens = db.get_user_generations(current_user["id"], limit)
    return {"generations": gens}


@router.get("/loyalty")
async def get_loyalty_stats(current_user: dict = Depends(get_current_user)):
    """Get loyalty stats and upcoming rewards"""
    conn = db.get_conn()
    loyalty = conn.execute(
        "SELECT * FROM loyalty WHERE user_id = ?", (current_user["id"],)
    ).fetchone()
    conn.close()

    if not loyalty:
        return {"error": "Loyalty record not found"}

    loyalty = dict(loyalty)
    total_gen = current_user["total_generated"]

    next_milestone = ((total_gen // 500) + 1) * 500
    images_to_milestone = next_milestone - total_gen

    fast_gen_to_free = 10 - (loyalty["fast_gen_count"] % 10)
    no_wm_to_free = 10 - (loyalty["no_wm_count"] % 10)

    return {
        "fast_gen_count": loyalty["fast_gen_count"],
        "no_wm_count": loyalty["no_wm_count"],
        "weekly_streak": loyalty["weekly_streak"],
        "total_milestones_hit": loyalty["total_images_500"],
        "next_milestone_at": next_milestone,
        "images_to_next_milestone": images_to_milestone,
        "fast_gen_to_free": fast_gen_to_free,
        "no_wm_to_free": no_wm_to_free
    }