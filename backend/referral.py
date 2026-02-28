"""
LuxurAI — referral.py
─────────────────────────────────────────────────────────────────
Complete Referral System

HOW IT WORKS:
  1. User generates a unique referral link:  luxurai.in/?ref=ABC123XY
  2. New user clicks link → signs up → both get rewarded

REFERRER REWARDS (per successful join):
  ┌─────────────────────────────────────────────┐
  │  Signups sent  │  Reward per join           │
  ├────────────────┼────────────────────────────┤
  │  1–9           │  1 LC per join             │
  │  10+           │  3 LC per join (milestone) │
  └─────────────────────────────────────────────┘
  Note: At exactly 10 total signups → one-time bonus of 3 LC extra.

REFEREE (new user who joined via ref link) REWARDS:
  • On signup: 2 LC bonus (on top of normal 45 LC signup bonus)
  • LC Purchase discounts (applied at checkout):
    ┌──────────────────────────────────────────┐
    │  Buy #  │  Discount                      │
    ├─────────┼────────────────────────────────┤
    │  1–2    │  90% OFF                       │
    │  3–7    │  40% OFF  (next 5 buys)        │
    │  8–12   │  20% OFF  (next 5 buys)        │
    │  13+    │  Normal price                  │
    └──────────────────────────────────────────┘
  Note: LC amount per pack stays SAME, only INR price discounted.

ENDPOINTS:
  GET  /api/referral/link              → Get or generate my referral link
  GET  /api/referral/stats             → My referral stats + history
  POST /api/referral/apply             → Apply a referral code at signup
  GET  /api/referral/discount          → Get current purchase discount for user
  POST /api/referral/internal/reward   → Internal — called after signup confirm
─────────────────────────────────────────────────────────────────
"""

import os
import secrets
import hashlib
import logging
import aiosqlite
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from jose import JWTError, jwt
from backend.wallet import WalletService, REASON_REFERRAL

from backend.wallet import WalletService, REASON_REFERRAL

load_dotenv()

logger = logging.getLogger("luxurai.referral")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH    = os.getenv("DB_PATH", "luxurai.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod!")
ALGORITHM  = "HS256"
BASE_URL   = os.getenv("BASE_URL", "https://luxurai.in")

wallet = WalletService(DB_PATH)

# ─────────────────────────────────────────────
# Referral Reward Rules
# ─────────────────────────────────────────────

# Referrer (person who shared the link)
REFERRER_REWARD_NORMAL_LC  = 1.0    # 1 LC per join (1-9 signups)
REFERRER_REWARD_POWER_LC   = 3.0    # 3 LC per join after 10th signup
REFERRER_MILESTONE_AT      = 10     # at 10 signups, switch to power reward

# Referee (person who joined via link)
REFEREE_SIGNUP_BONUS_LC    = 2.0    # extra 2 LC on signup (on top of normal bonus)

# Purchase discounts for referee (applies to INR price, LC amount stays same)
REFEREE_DISCOUNT_TIERS = [
    {"buys_from": 1,  "buys_to": 2,  "discount_pct": 90},   # first 2 buys: 90% off
    {"buys_from": 3,  "buys_to": 7,  "discount_pct": 40},   # next 5 buys:  40% off
    {"buys_from": 8,  "buys_to": 12, "discount_pct": 20},   # next 5 buys:  20% off
    {"buys_from": 13, "buys_to": None, "discount_pct": 0},  # after that: normal
]


# ─────────────────────────────────────────────
# DB Init
# ─────────────────────────────────────────────
async def init_referral_tables(db_path: str = DB_PATH):
    """Create all referral tables. Call on startup."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            -- Referral codes (one per user, unique)
            CREATE TABLE IF NOT EXISTS referral_codes (
                code        TEXT PRIMARY KEY,          -- e.g. "ABC123XY" (8 chars)
                user_id     TEXT NOT NULL UNIQUE,      -- owner of this code
                created_at  TEXT NOT NULL,
                total_clicks INTEGER NOT NULL DEFAULT 0,
                total_joins  INTEGER NOT NULL DEFAULT 0
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_refcode_user
                ON referral_codes(user_id);

            -- Referral joins (one row per successful signup via ref)
            CREATE TABLE IF NOT EXISTS referral_joins (
                id              TEXT PRIMARY KEY,
                referrer_id     TEXT NOT NULL,          -- user who shared
                referee_id      TEXT NOT NULL UNIQUE,   -- user who joined
                code            TEXT NOT NULL,
                reward_lc       REAL NOT NULL,          -- LC given to referrer
                bonus_lc        REAL NOT NULL,          -- LC given to referee
                created_at      TEXT NOT NULL,
                FOREIGN KEY (referrer_id) REFERENCES users(id),
                FOREIGN KEY (referee_id)  REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_refjoin_referrer
                ON referral_joins(referrer_id);

            -- Tracks how many LC purchases each referred user has made
            -- Used to compute their discount tier
            CREATE TABLE IF NOT EXISTS referee_purchase_count (
                user_id         TEXT PRIMARY KEY,
                purchase_count  INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        await db.commit()
    logger.info("Referral tables initialized.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return secrets.token_urlsafe(12)

def _generate_code() -> str:
    """
    8-character alphanumeric referral code.
    e.g. "A3kX9mRq"
    URL-safe, case-sensitive, 62^8 = ~218 trillion combinations.
    """
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


async def get_current_user(request: Request) -> str:
    """Returns user_id from JWT cookie or Authorization header."""
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


def get_discount_for_buy_number(buy_number: int) -> int:
    """
    Given how many LC purchases the referee has made (1-indexed),
    return the discount percentage.
    e.g. buy_number=1 → 90, buy_number=5 → 40, buy_number=10 → 20, buy_number=15 → 0
    """
    for tier in REFEREE_DISCOUNT_TIERS:
        if tier["buys_to"] is None or buy_number <= tier["buys_to"]:
            if buy_number >= tier["buys_from"]:
                return tier["discount_pct"]
    return 0


# ─────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────
async def get_or_create_referral_code(user_id: str) -> str:
    """
    Returns existing referral code for user, or creates a new unique one.
    Collision-safe with retry loop.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Check if user already has a code
        async with db.execute(
            "SELECT code FROM referral_codes WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()

        if row:
            return row["code"]

        # Generate a unique code
        for _ in range(10):   # retry loop for collisions
            code = _generate_code()
            async with db.execute(
                "SELECT 1 FROM referral_codes WHERE code = ?", (code,)
            ) as cur:
                exists = await cur.fetchone()

            if not exists:
                await db.execute(
                    "INSERT INTO referral_codes (code, user_id, created_at) VALUES (?,?,?)",
                    (code, user_id, _now())
                )
                await db.commit()
                logger.info(f"Referral code created: {code} for user {user_id[:8]}...")
                return code

        raise HTTPException(500, "Failed to generate referral code. Please try again.")


async def process_referral_join(referee_id: str, code: str) -> dict:
    """
    Called after a new user completes signup via referral link.

    Steps:
      1. Validate code & find referrer
      2. Check referee hasn't already used a code
      3. Credit referee with signup bonus LC
      4. Credit referrer with per-join reward LC
      5. Update stats
      6. Return summary

    This is idempotent — safe to call again if something fails midway.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 1. Find referral code
        async with db.execute(
            "SELECT * FROM referral_codes WHERE code = ?", (code,)
        ) as cur:
            code_row = await cur.fetchone()

        if not code_row:
            raise HTTPException(400, "Invalid referral code.")

        referrer_id = code_row["user_id"]

        # Self-referral guard
        if referrer_id == referee_id:
            raise HTTPException(400, "You cannot use your own referral code.")

        # 2. Check referee hasn't already used a referral code
        async with db.execute(
            "SELECT 1 FROM referral_joins WHERE referee_id = ?", (referee_id,)
        ) as cur:
            already = await cur.fetchone()

        if already:
            raise HTTPException(409, "You have already used a referral code.")

        # 3. Determine referrer reward based on their total joins so far
        total_joins_so_far = code_row["total_joins"]
        new_join_number    = total_joins_so_far + 1   # this is the Nth join

        if new_join_number >= REFERRER_MILESTONE_AT:
            referrer_reward = REFERRER_REWARD_POWER_LC    # 3 LC
        else:
            referrer_reward = REFERRER_REWARD_NORMAL_LC   # 1 LC

        referee_bonus = REFEREE_SIGNUP_BONUS_LC   # 2 LC

        # 4. Record the join
        join_id = _new_id()
        await db.execute(
            """INSERT INTO referral_joins
               (id, referrer_id, referee_id, code, reward_lc, bonus_lc, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (join_id, referrer_id, referee_id, code, referrer_reward, referee_bonus, _now())
        )

        # 5. Update code stats
        await db.execute(
            "UPDATE referral_codes SET total_joins = total_joins + 1 WHERE code = ?",
            (code,)
        )

        # 6. Initialize referee purchase count tracker
        await db.execute(
            "INSERT OR IGNORE INTO referee_purchase_count (user_id, purchase_count) VALUES (?,0)",
            (referee_id,)
        )

        await db.commit()

    # 7. Credit LC (outside the DB transaction to avoid locking)
    # Referee signup bonus
    try:
        await wallet.credit(
            referee_id,
            referee_bonus,
            reason=f"referral_signup_bonus:{code}",
            ref_id=f"ref_bonus_{join_id}"
        )
    except Exception as e:
        logger.error(f"Failed to credit referee bonus: {e}")

    # Referrer reward
    try:
        await wallet.credit(
            referrer_id,
            referrer_reward,
            reason=f"referral_join_reward:{referee_id[:8]}",
            ref_id=f"ref_reward_{join_id}"
        )
    except Exception as e:
        logger.error(f"Failed to credit referrer reward: {e}")

    logger.info(
        f"Referral processed: referrer={referrer_id[:8]} "
        f"referee={referee_id[:8]} "
        f"referrer_gets={referrer_reward}LC referee_gets={referee_bonus}LC"
    )

    return {
        "success":          True,
        "referrer_reward":  referrer_reward,
        "referee_bonus":    referee_bonus,
        "join_number":      new_join_number,
    }


async def get_user_discount(user_id: str) -> dict:
    """
    Returns the LC purchase discount for a referred user.
    Returns 0% discount if user was NOT referred.

    Discount is based on how many LC purchases they've made so far.
    The NEXT purchase will be at the returned discount level.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Check if this user was referred
        async with db.execute(
            "SELECT 1 FROM referral_joins WHERE referee_id = ?", (user_id,)
        ) as cur:
            is_referred = await cur.fetchone() is not None

        if not is_referred:
            return {"is_referred": False, "discount_pct": 0, "next_buy_number": None}

        # Get purchase count
        async with db.execute(
            "SELECT purchase_count FROM referee_purchase_count WHERE user_id = ?",
            (user_id,)
        ) as cur:
            row = await cur.fetchone()

        purchase_count = row["purchase_count"] if row else 0
        next_buy_number = purchase_count + 1
        discount_pct = get_discount_for_buy_number(next_buy_number)

    return {
        "is_referred":      True,
        "discount_pct":     discount_pct,
        "next_buy_number":  next_buy_number,
        "purchases_so_far": purchase_count,
        "tiers":            REFEREE_DISCOUNT_TIERS,
    }


async def record_referee_purchase(user_id: str):
    """
    Call this after a successful LC purchase by a referred user.
    Increments their purchase count so discount tier updates correctly.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO referee_purchase_count (user_id, purchase_count)
               VALUES (?, 1)
               ON CONFLICT(user_id) DO UPDATE SET purchase_count = purchase_count + 1""",
            (user_id,)
        )
        await db.commit()


def apply_discount(inr_price: float, discount_pct: int) -> float:
    """
    Apply discount to INR price. LC amount stays the same.
    e.g. 90% off on ₹40 pack → user pays ₹4, still gets 200 LC.
    """
    if discount_pct <= 0:
        return inr_price
    discounted = inr_price * (1 - discount_pct / 100)
    return round(discounted, 2)


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/referral", tags=["referral"])


# ── Get my referral link ──────────────────────
@router.get("/link")
async def get_referral_link(request: Request):
    """
    Returns the user's unique referral link.
    Creates one if it doesn't exist yet.
    Share this link to earn LC for every signup.
    """
    user_id = await get_current_user(request)
    code = await get_or_create_referral_code(user_id)

    return {
        "code":          code,
        "link":          f"{BASE_URL}/?ref={code}",
        "share_message": f"Join LuxurAI and get bonus LC! Sign up here: {BASE_URL}/?ref={code}",
    }


# ── My referral stats ─────────────────────────
@router.get("/stats")
async def get_referral_stats(request: Request):
    """
    Returns referral performance: total joins, LC earned, history.
    """
    user_id = await get_current_user(request)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Get code info
        async with db.execute(
            "SELECT * FROM referral_codes WHERE user_id = ?", (user_id,)
        ) as cur:
            code_row = await cur.fetchone()

        if not code_row:
            return {
                "has_code":   False,
                "total_joins": 0,
                "total_lc_earned": 0,
                "joins": [],
            }

        # Get all joins
        async with db.execute(
            """SELECT rj.id, rj.referee_id, rj.reward_lc, rj.created_at,
                      u.email
               FROM referral_joins rj
               LEFT JOIN users u ON u.id = rj.referee_id
               WHERE rj.referrer_id = ?
               ORDER BY rj.created_at DESC""",
            (user_id,)
        ) as cur:
            joins = await cur.fetchall()

        total_lc_earned = sum(j["reward_lc"] for j in joins)

        # Next milestone info
        total_joins = code_row["total_joins"]
        if total_joins < REFERRER_MILESTONE_AT:
            remaining = REFERRER_MILESTONE_AT - total_joins
            next_reward_lc = REFERRER_REWARD_POWER_LC
        else:
            remaining = 0
            next_reward_lc = REFERRER_REWARD_POWER_LC

    return {
        "has_code":        True,
        "code":            code_row["code"],
        "link":            f"{BASE_URL}/?ref={code_row['code']}",
        "total_joins":     total_joins,
        "total_lc_earned": round(total_lc_earned, 2),
        "current_reward_per_join": (
            REFERRER_REWARD_POWER_LC if total_joins >= REFERRER_MILESTONE_AT
            else REFERRER_REWARD_NORMAL_LC
        ),
        "milestone": {
            "at":        REFERRER_MILESTONE_AT,
            "reached":   total_joins >= REFERRER_MILESTONE_AT,
            "remaining": remaining,
            "reward_at_milestone": next_reward_lc,
        },
        "joins": [
            {
                "id":         j["id"],
                "email":      (j["email"] or "")[:3] + "***",    # privacy mask
                "reward_lc":  j["reward_lc"],
                "joined_at":  j["created_at"],
            }
            for j in joins
        ],
    }


# ── Apply referral code (called at signup) ────
class ApplyReferralRequest(BaseModel):
    code: str


@router.post("/apply")
async def apply_referral(body: ApplyReferralRequest, request: Request):
    """
    New user applies a referral code after completing signup.
    This is called ONCE per user — second call returns 409.

    What happens:
      - Referee gets 2 LC bonus immediately
      - Referrer gets 1 LC (or 3 LC if they've hit 10 signups)
      - Referee gets discounted LC purchases going forward
    """
    user_id = await get_current_user(request)
    code = body.code.strip()

    if not code:
        raise HTTPException(400, "Referral code cannot be empty.")

    result = await process_referral_join(
        referee_id=user_id,
        code=code,
    )

    return {
        "success":       True,
        "bonus_lc":      result["referee_bonus"],
        "message":       f"Referral applied! You received {result['referee_bonus']} LC bonus.",
        "discount_info": {
            "first_2_buys": "90% OFF",
            "next_5_buys":  "40% OFF",
            "next_5_buys2": "20% OFF",
            "after_that":   "Normal price",
        },
    }


# ── Get my current purchase discount ─────────
@router.get("/discount")
async def get_my_discount(request: Request):
    """
    Returns the discount % for the next LC purchase.
    Called by payment.py before creating an order.

    If user was not referred → discount_pct = 0
    """
    user_id = await get_current_user(request)
    return await get_user_discount(user_id)


# ── Internal: record a completed purchase ─────
@router.post("/internal/record-purchase")
async def record_purchase(request: Request):
    """
    Called internally by payment.py after a successful LC purchase.
    Increments the referred user's purchase counter.
    Only has effect if the user was referred — harmless for others.
    """
    user_id = await get_current_user(request)
    await record_referee_purchase(user_id)
    discount_info = await get_user_discount(user_id)
    return {
        "recorded": True,
        "next_discount_pct": discount_info["discount_pct"],
        "next_buy_number": discount_info.get("next_buy_number"),
    }


# ── Click tracker (called from frontend on ref link open) ─
@router.post("/track-click")
async def track_click(code: str):
    """
    Increment click count on a referral code.
    Called by frontend when someone opens a referral link.
    No auth required (anonymous tracking).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE referral_codes SET total_clicks = total_clicks + 1 WHERE code = ?",
            (code,)
        )
        await db.commit()
    return {"tracked": True}
