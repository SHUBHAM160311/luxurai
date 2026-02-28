"""
LuxurAI — loyalty.py
─────────────────────────────────────────────────────────────────
Loyalty Reward System

TWO REWARD TYPES:

1. WEEKLY ACTIVE REWARD — 1 LC/week
   - User generates at least 1 image in a week
   - Gets 1 LC back automatically
   - Resets every Monday (ISO week)

2. GENERATION MILESTONE — 1 LC per 500 generations
   - Every 500th completed generation → 1 LC bonus
   - Cumulative lifetime counter (500, 1000, 1500, ...)
   - e.g. user does 500 total gens → +1 LC, 1000 total → +1 LC

HOW IT RUNS:
   - Call check_and_award_loyalty(user_id) after every completed job
   - It checks both conditions and rewards if due
   - Idempotent — safe to call multiple times

ENDPOINTS:
  GET  /api/loyalty/status     → User's loyalty stats
  GET  /api/loyalty/history    → User's reward history

INTERNAL:
  check_and_award_loyalty(user_id)  → called from jobs.py on job complete
─────────────────────────────────────────────────────────────────
"""

import os
import secrets
import logging
import aiosqlite
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from jose import JWTError, jwt
from dotenv import load_dotenv

from wallet import WalletService, REASON_LOYALTY

load_dotenv()

logger = logging.getLogger("luxurai.loyalty")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH    = os.getenv("DB_PATH", "luxurai.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod!")
ALGORITHM  = "HS256"

wallet = WalletService(DB_PATH)

# Reward values
WEEKLY_REWARD_LC      = 1.0    # 1 LC for being active in a week
MILESTONE_REWARD_LC   = 1.0    # 1 LC per 500 generations
MILESTONE_EVERY_N     = 500    # every 500 completed gens


# ─────────────────────────────────────────────
# DB Init
# ─────────────────────────────────────────────
async def init_loyalty_tables(db_path: str = DB_PATH):
    """Create loyalty tracking tables. Call on startup."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            -- Tracks weekly reward per user
            -- One row per user per ISO week (e.g. "2026-W08")
            CREATE TABLE IF NOT EXISTS loyalty_weekly (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                iso_week    TEXT NOT NULL,   -- "YYYY-Www" e.g. "2026-W08"
                rewarded    INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                UNIQUE(user_id, iso_week)
            );

            -- Tracks lifetime generation count for milestone rewards
            CREATE TABLE IF NOT EXISTS loyalty_gen_tracker (
                user_id             TEXT PRIMARY KEY,
                total_completed     INTEGER NOT NULL DEFAULT 0,
                last_milestone      INTEGER NOT NULL DEFAULT 0,  -- last milestone rewarded (500, 1000, ...)
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        await db.commit()
    logger.info("Loyalty tables initialized.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return secrets.token_urlsafe(12)

def _current_iso_week() -> str:
    """Returns current ISO week string e.g. '2026-W08'"""
    now = datetime.now(timezone.utc)
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


async def get_current_user(request: Request) -> str:
    token = request.cookies.get("luxurai_session")
    if not token:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid session")


# ─────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────
async def check_and_award_loyalty(user_id: str) -> dict:
    """
    Main loyalty engine — call this after every completed generation.

    Checks:
      1. Weekly reward — if user generated this week and not yet rewarded
      2. Milestone reward — if user crossed a new 500-gen milestone

    Returns dict of rewards given (empty if nothing awarded).
    """
    rewards = []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # ── 1. Weekly Reward ──────────────────
        iso_week = _current_iso_week()

        # Check if already rewarded this week
        async with db.execute(
            "SELECT rewarded FROM loyalty_weekly WHERE user_id = ? AND iso_week = ?",
            (user_id, iso_week)
        ) as cur:
            weekly_row = await cur.fetchone()

        if weekly_row is None:
            # First activity this week — create record
            await db.execute(
                "INSERT OR IGNORE INTO loyalty_weekly (id, user_id, iso_week, rewarded, created_at) VALUES (?,?,?,0,?)",
                (_new_id(), user_id, iso_week, _now())
            )
            await db.commit()
            weekly_row = {"rewarded": 0}

        if not weekly_row["rewarded"]:
            # Award weekly LC
            try:
                await wallet.credit(
                    user_id,
                    WEEKLY_REWARD_LC,
                    reason=f"loyalty_weekly:{iso_week}",
                    ref_id=f"weekly_{user_id}_{iso_week}"
                )
                # Mark as rewarded
                await db.execute(
                    "UPDATE loyalty_weekly SET rewarded = 1 WHERE user_id = ? AND iso_week = ?",
                    (user_id, iso_week)
                )
                await db.commit()
                rewards.append({
                    "type":   "weekly_active",
                    "lc":     WEEKLY_REWARD_LC,
                    "reason": f"Active this week ({iso_week}) — 1 LC reward!",
                })
                logger.info(f"Weekly reward: {WEEKLY_REWARD_LC} LC → user {user_id[:8]} ({iso_week})")
            except Exception as e:
                logger.error(f"Weekly reward failed for {user_id}: {e}")

        # ── 2. Generation Milestone Reward ────
        # Get or create tracker
        async with db.execute(
            "SELECT total_completed, last_milestone FROM loyalty_gen_tracker WHERE user_id = ?",
            (user_id,)
        ) as cur:
            tracker = await cur.fetchone()

        if tracker is None:
            await db.execute(
                "INSERT INTO loyalty_gen_tracker (user_id, total_completed, last_milestone) VALUES (?,1,0)",
                (user_id,)
            )
            await db.commit()
            total_completed = 1
            last_milestone  = 0
        else:
            # Increment counter
            total_completed = tracker["total_completed"] + 1
            last_milestone  = tracker["last_milestone"]
            await db.execute(
                "UPDATE loyalty_gen_tracker SET total_completed = ? WHERE user_id = ?",
                (total_completed, user_id)
            )
            await db.commit()

        # Check if crossed a new milestone
        current_milestone = (total_completed // MILESTONE_EVERY_N) * MILESTONE_EVERY_N

        if current_milestone > last_milestone and current_milestone > 0:
            # Crossed one or more milestones
            milestones_crossed = (current_milestone - last_milestone) // MILESTONE_EVERY_N
            reward_lc = milestones_crossed * MILESTONE_REWARD_LC

            try:
                await wallet.credit(
                    user_id,
                    reward_lc,
                    reason=f"loyalty_milestone:{current_milestone}_gens",
                    ref_id=f"milestone_{user_id}_{current_milestone}"
                )
                await db.execute(
                    "UPDATE loyalty_gen_tracker SET last_milestone = ? WHERE user_id = ?",
                    (current_milestone, user_id)
                )
                await db.commit()
                rewards.append({
                    "type":      "generation_milestone",
                    "lc":        reward_lc,
                    "milestone": current_milestone,
                    "reason":    f"{current_milestone} generations milestone — {reward_lc} LC reward!",
                })
                logger.info(
                    f"Milestone reward: {reward_lc} LC → user {user_id[:8]} "
                    f"(milestone: {current_milestone} gens)"
                )
            except Exception as e:
                logger.error(f"Milestone reward failed for {user_id}: {e}")

    return {
        "rewards_given": rewards,
        "total_lc_awarded": sum(r["lc"] for r in rewards),
    }


async def get_loyalty_status(user_id: str) -> dict:
    """
    Returns user's current loyalty standing:
    - This week rewarded or not
    - Total gens, next milestone, how far
    """
    iso_week = _current_iso_week()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Weekly status
        async with db.execute(
            "SELECT rewarded FROM loyalty_weekly WHERE user_id = ? AND iso_week = ?",
            (user_id, iso_week)
        ) as cur:
            weekly = await cur.fetchone()

        weekly_rewarded = weekly["rewarded"] if weekly else False

        # Milestone tracker
        async with db.execute(
            "SELECT total_completed, last_milestone FROM loyalty_gen_tracker WHERE user_id = ?",
            (user_id,)
        ) as cur:
            tracker = await cur.fetchone()

        total_completed = tracker["total_completed"] if tracker else 0
        last_milestone  = tracker["last_milestone"] if tracker else 0
        next_milestone  = last_milestone + MILESTONE_EVERY_N
        gens_to_next    = next_milestone - total_completed

        # Total LC earned from loyalty
        async with db.execute(
            "SELECT COALESCE(SUM(delta_lc), 0) as total FROM wallet_ledger WHERE user_id = ? AND reason LIKE 'loyalty%'",
            (user_id,)
        ) as cur:
            row = await cur.fetchone()
        total_loyalty_lc = row["total"] if row else 0

    return {
        "weekly": {
            "current_week":    iso_week,
            "rewarded":        bool(weekly_rewarded),
            "reward_lc":       WEEKLY_REWARD_LC,
            "message":         "Already earned this week!" if weekly_rewarded else "Generate an image to earn 1 LC this week!",
        },
        "milestones": {
            "total_completed":  total_completed,
            "last_milestone":   last_milestone,
            "next_milestone":   next_milestone,
            "gens_to_next":     max(0, gens_to_next),
            "reward_per_milestone": MILESTONE_REWARD_LC,
            "milestone_every_n":    MILESTONE_EVERY_N,
        },
        "total_loyalty_lc_earned": round(total_loyalty_lc, 2),
    }


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/loyalty", tags=["loyalty"])


@router.get("/status")
async def loyalty_status(request: Request):
    """
    User's loyalty dashboard:
    - Weekly reward status
    - Generation milestone progress
    - Total LC earned from loyalty
    """
    user_id = await get_current_user(request)
    return await get_loyalty_status(user_id)


@router.get("/history")
async def loyalty_history(request: Request, limit: int = 20):
    """All loyalty rewards received by this user."""
    user_id = await get_current_user(request)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT delta_lc, reason, created_at FROM wallet_ledger
               WHERE user_id = ? AND reason LIKE 'loyalty%'
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit)
        ) as cur:
            rows = await cur.fetchall()

    return {
        "history": [dict(r) for r in rows],
    }
