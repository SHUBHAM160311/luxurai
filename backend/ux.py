"""
LuxurAI — ux.py
─────────────────────────────────────────────────────────────────
Dashboard Backend — Powers LuxurAI_UX.html

Endpoints this file provides:
  GET  /api/ux/dashboard          → User stats, balance, recent jobs
  GET  /api/ux/history            → Generation history (paginated)
  GET  /api/ux/history/{job_id}   → Single job detail
  POST /api/ux/generate           → Submit generation job
  GET  /api/ux/job/{job_id}       → Poll job status + queue position
  GET  /api/ux/wallet             → LC balance + ledger
  POST /api/ux/apikey/create      → Generate new API key
  POST /api/ux/apikey/rotate      → Rotate existing API key
  GET  /api/ux/apikey/list        → List user's API keys
  DELETE /api/ux/apikey/{key_id}  → Revoke API key
  GET  /api/ux/burn-chart         → LC usage last 7 days (chart data)

Connects to:
  wallet.py  → LC balance, ledger
  jobs.py    → Job creation, status, history
  auth.py    → JWT session (get_current_user)
─────────────────────────────────────────────────────────────────
"""

import os
import hmac
import hashlib
import secrets
import logging
import aiosqlite
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel
from jose import JWTError, jwt
from dotenv import load_dotenv

from backend.wallet import WalletService, REASON_API_CALL
from backend.jobs import JobService, JobStatus, UserTier, InsufficientBalanceError
load_dotenv()

logger = logging.getLogger("luxurai.ux")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH    = os.getenv("DB_PATH", "luxurai.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod!")
ALGORITHM  = "HS256"

wallet = WalletService(DB_PATH)
jobs   = JobService(DB_PATH)


# ─────────────────────────────────────────────
# Auth helper
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# DB Init
# ─────────────────────────────────────────────
async def init_ux_tables(db_path: str = DB_PATH):
    """Create API keys table. Call once on startup."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                key_hash    TEXT NOT NULL UNIQUE,   -- SHA256 of actual key
                key_prefix  TEXT NOT NULL,           -- lxr_xxxx (shown to user)
                label       TEXT,
                is_active   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL,
                last_used   TEXT,
                total_calls INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_apikeys_user
                ON api_keys(user_id, is_active);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_apikeys_hash
                ON api_keys(key_hash);
        """)
        await db.commit()
    logger.info("UX tables initialized.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return secrets.token_urlsafe(12)

def _generate_api_key() -> tuple[str, str, str]:
    """
    Cryptographically strong, globally unique API key generator.

    Format:  lxr_v1_<8-char-prefix>_<64-char-random>
    Example: lxr_v1_aB3xKm9Z_xQr7tY2pNvW8mK4jH6cL...

    Why this format:
      - lxr_     → LuxurAI brand prefix (instantly identifiable in logs)
      - v1_      → Version tag (future-proof, can rotate scheme later)
      - 8-char   → Display prefix shown in dashboard (unique per key visually)
      - 64-char  → 288 bits of entropy → brute force impossible (2^288 combos)
      - SHA256   → Only hash stored in DB, raw key NEVER saved

    Collision probability: astronomically low (~1 in 10^86).
    Even at 1 million keys/sec it would take longer than age of universe.

    Returns:
        full_key   → shown to user ONCE, never stored in DB
        key_prefix → lxr_v1_XXXXXXXX (shown in dashboard with •••• mask)
        key_hash   → SHA256(full_key), stored in DB for validation
    """
    # 48 bytes → 64 url-safe base64 chars → 288 bits of randomness
    random_part    = secrets.token_urlsafe(48)
    # 6 bytes → 8 chars for the display prefix
    display_prefix = secrets.token_urlsafe(6)[:8]

    full_key   = f"lxr_v1_{display_prefix}_{random_part}"
    key_prefix = f"lxr_v1_{display_prefix}"
    key_hash   = hashlib.sha256(full_key.encode()).hexdigest()

    return full_key, key_prefix, key_hash


def _validate_key_format(key: str) -> bool:
    """
    Format sanity check BEFORE hitting the DB.
    Instantly rejects obviously fake, random, or malformed keys.

    Valid:   lxr_v1_aB3xKm9Z_xQr7tY2pNvW8mK4jH6cL...  (80+ chars)
    Invalid: anything else — random strings, old format, wrong prefix etc.
    """
    # Must start with lxr_v1_
    if not key.startswith("lxr_v1_"):
        return False
    # Split: lxr | v1 | display_prefix | random_part
    parts = key.split("_", 3)
    if len(parts) != 4:
        return False
    if len(parts[2]) < 8:     # display prefix must be 8 chars
        return False
    if len(parts[3]) < 60:    # random part must be long enough
        return False
    return True


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/ux", tags=["dashboard"])


# ─────────────────────────────────────────────
# Dashboard Summary
# ─────────────────────────────────────────────
@router.get("/dashboard")
async def get_dashboard(request: Request):
    """
    Single call to load the full dashboard.
    Returns: balance, recent jobs, usage stats.
    """
    user_id = await get_current_user(request)

    # Balance
    balance = await wallet.get_balance(user_id)

    # Recent 5 jobs
    recent_jobs = await jobs.get_user_jobs(user_id, limit=5)

    # Stats: total generated, total LC spent
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            """SELECT
                COUNT(*) as total_jobs,
                COALESCE(SUM(CASE WHEN status='done' THEN 1 ELSE 0 END), 0) as completed,
                COALESCE(SUM(CASE WHEN status='done' THEN cost_lc ELSE 0 END), 0) as lc_spent
               FROM jobs WHERE user_id = ?""",
            (user_id,)
        ) as cur:
            stats = dict(await cur.fetchone())

    return {
        "balance_lc":      balance,
        "total_jobs":      stats["total_jobs"],
        "completed":       stats["completed"],
        "total_lc_spent":  round(stats["lc_spent"], 2),
        "recent_jobs": [
            {
                "job_id":    j.id,
                "prompt":    j.prompt[:80] + "..." if len(j.prompt) > 80 else j.prompt,
                "resolution": j.resolution,
                "cost_lc":   j.cost_lc,
                "status":    j.status.value,
                "image_url": j.image_url,
                "created_at": j.created_at,
            }
            for j in recent_jobs
        ]
    }


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt:       str
    resolution:   str  = "1024x1024"
    fast:         bool = False
    no_watermark: bool = False
    bulk:         bool = False
    bulk_count:   int  = 1          # 1-10, only used when bulk=True
    api_key:      Optional[str] = None


@router.post("/generate")
async def submit_generation(body: GenerateRequest, request: Request):
    """
    Frontend Generate button -> this endpoint.
    Creates job, deducts LC, returns job_id + queue position.

    Pricing rules:
      - Normal users : full price, NO bulk access
      - API users    : 1 LC discount on resolutions >200px (both sides)
                       fast_gen  -> 2 LC (was 3)
                       no_wm     -> 6 LC (was 7)
                       bulk      -> flat 5 LC for up to 10 images
    """
    from backend.wallet import BULK_MAX_IMGS  
    user_id = await get_current_user(request)

    # Determine tier
    user_tier = UserTier.FREE
    if body.api_key:
        valid = await _validate_api_key(body.api_key, user_id)
        if valid:
            user_tier = UserTier.API
            await _record_api_key_usage(body.api_key)
        else:
            raise HTTPException(403, "Invalid or inactive API key")

    is_api = (user_tier == UserTier.API)

    # Bulk guard (API only)
    if body.bulk and not is_api:
        raise HTTPException(403, "Bulk generation is only available for API key users.")

    bulk_count = min(max(1, body.bulk_count), BULK_MAX_IMGS) if body.bulk else 1

    try:
        job = await jobs.create_job(
            user_id    = user_id,
            prompt     = body.prompt,
            resolution = body.resolution,
            addons     = {
                "fast":         body.fast,
                "no_watermark": body.no_watermark,
                "bulk":         body.bulk,
                "bulk_count":   bulk_count,
                "api":          is_api,
            },
            user_tier = user_tier,
        )
    except InsufficientBalanceError as e:
        raise HTTPException(402, str(e))

    position = await jobs.get_queue_position(job.id)

    return {
        "job_id":         job.id,
        "status":         job.status.value,
        "cost_lc":        job.cost_lc,
        "bulk_count":     bulk_count if body.bulk else 1,
        "queue_position": position,
        "eta_seconds":    (position + 1) * 8,
    }


# ─────────────────────────────────────────────
# Job Status (frontend polls this)
# ─────────────────────────────────────────────
@router.get("/job/{job_id}")
async def poll_job(job_id: str, request: Request):
    """
    Frontend polls this every 2s for live queue updates.
    Returns position, ETA, status, image_url when done.
    """
    user_id = await get_current_user(request)

    try:
        job = await jobs.get_job(job_id)
    except Exception:
        raise HTTPException(404, "Job not found")

    if job.user_id != user_id:
        raise HTTPException(403, "Not your job")

    position = await jobs.get_queue_position(job_id)

    return {
        "job_id":         job.id,
        "status":         job.status.value,
        "queue_position": position if position >= 0 else None,
        "eta_seconds":    max(0, (position + 1) * 8) if position >= 0 else 0,
        "image_url":      job.image_url,
        "fail_reason":    job.fail_reason,
        "cost_lc":        job.cost_lc,
    }


# ─────────────────────────────────────────────
# Generation History
# ─────────────────────────────────────────────
@router.get("/history")
async def get_history(
    request: Request,
    limit:   int = Query(20, le=100),
    offset:  int = Query(0),
):
    """Paginated generation history — powers History panel in dashboard."""
    user_id = await get_current_user(request)
    job_list = await jobs.get_user_jobs(user_id, limit=limit, offset=offset)

    return {
        "jobs": [
            {
                "job_id":     j.id,
                "prompt":     j.prompt,
                "resolution": j.resolution,
                "cost_lc":    j.cost_lc,
                "status":     j.status.value,
                "image_url":  j.image_url,
                "created_at": j.created_at,
                "addons":     j.addons,
            }
            for j in job_list
        ],
        "limit":  limit,
        "offset": offset,
    }


# ─────────────────────────────────────────────
# Wallet
# ─────────────────────────────────────────────
@router.get("/wallet")
async def get_wallet(request: Request, ledger_limit: int = 10):
    """LC balance + recent ledger — powers Wallet section."""
    user_id = await get_current_user(request)
    balance = await wallet.get_balance(user_id)
    ledger  = await wallet.get_ledger(user_id, limit=ledger_limit)

    return {
        "balance_lc": balance,
        "ledger": [
            {
                "id":           e.id,
                "delta_lc":     e.delta_lc,
                "reason":       e.reason,
                "balance_after": e.balance_after,
                "created_at":   e.created_at,
            }
            for e in ledger
        ]
    }


# ─────────────────────────────────────────────
# LC Burn Chart (last 7 days)
# ─────────────────────────────────────────────
@router.get("/burn-chart")
async def get_burn_chart(request: Request):
    """
    Returns LC usage per day for last 7 days.
    Powers the burn chart in API dashboard panel.
    """
    user_id = await get_current_user(request)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT
                DATE(created_at) as day,
                SUM(cost_lc) as lc_used
               FROM jobs
               WHERE user_id = ?
                 AND status = 'done'
                 AND created_at >= DATE('now', '-7 days')
               GROUP BY DATE(created_at)
               ORDER BY day ASC""",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()

    # Fill missing days with 0
    today    = datetime.now(timezone.utc).date()
    day_map  = {str(today - timedelta(days=i)): 0 for i in range(6, -1, -1)}
    for r in rows:
        day_map[r["day"]] = round(r["lc_used"], 2)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    result = []
    for i, (date_str, lc) in enumerate(day_map.items()):
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        label = "Today" if date_str == str(today) else dt.strftime("%a")
        result.append({"day": date_str, "lc_used": lc, "is_today": date_str == str(today)})

    return {"days": result}



@router.post("/apikey/subscribe")
async def subscribe_api_key(request: Request):
    """
    Charge 2 LC/month subscription for API key access.
    Deducts 2 LC from wallet and records subscription in DB.
    Call this once per month per user (frontend handles renewal reminder).
    """
    from wallet import get_api_subscription_cost, REASON_API_CALL
    user_id = await get_current_user(request)

    sub_cost = get_api_subscription_cost()   # 2 LC

    # Check balance first
    balance = await wallet.get_balance(user_id)
    if balance < sub_cost:
        raise HTTPException(402, f"Need {sub_cost} LC for API subscription. Current balance: {balance:.2f} LC")

    # Deduct subscription LC
    ref_id = f"api_sub_{user_id}_{_now()[:7]}"   # unique per user per month (YYYY-MM)
    try:
        new_balance = await wallet.debit(
            user_id,
            sub_cost,
            reason="api_key_subscription",
            ref_id=ref_id,
        )
    except Exception as e:
        # DuplicateTransactionError = already paid this month
        if "already" in str(e).lower():
            raise HTTPException(409, "API subscription already active for this month.")
        raise HTTPException(500, str(e))

    # Record subscription in DB
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS api_subscriptions (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                period      TEXT NOT NULL,   -- YYYY-MM
                cost_lc     REAL NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        await db.execute(
            "INSERT OR IGNORE INTO api_subscriptions (id, user_id, period, cost_lc, created_at) VALUES (?,?,?,?,?)",
            (_new_id(), user_id, _now()[:7], sub_cost, _now())
        )
        await db.commit()

    return {
        "success":     True,
        "cost_lc":     sub_cost,
        "new_balance": new_balance,
        "period":      _now()[:7],
        "message":     f"API access activated for {_now()[:7]}. {sub_cost} LC deducted.",
    }


@router.get("/apikey/subscription-status")
async def get_subscription_status(request: Request):
    """Check if user has active API subscription for current month."""
    user_id = await get_current_user(request)
    current_period = _now()[:7]   # YYYY-MM

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            async with db.execute(
                "SELECT * FROM api_subscriptions WHERE user_id=? AND period=?",
                (user_id, current_period)
            ) as cur:
                row = await cur.fetchone()
        except Exception:
            row = None

    return {
        "active":  row is not None,
        "period":  current_period,
        "cost_lc": 2.0,
    }

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────
class CreateKeyRequest(BaseModel):
    label: Optional[str] = "My API Key"


@router.post("/apikey/create")
async def create_api_key(body: CreateKeyRequest, request: Request):
    """
    Generate a new unique API key for the user.

    Security guarantees:
      - 288 bits of entropy → brute force / guessing impossible
      - SHA256 hash stored in DB → raw key never persisted
      - Collision retry loop → guaranteed unique even at massive scale
      - Format validation on every use → fake keys rejected instantly
      - Full key shown ONCE only → if lost, must rotate
    """
    user_id = await get_current_user(request)

    # Max 5 active keys per user
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM api_keys WHERE user_id=? AND is_active=1",
            (user_id,)
        ) as cur:
            count = (await cur.fetchone())[0]

    if count >= 5:
        raise HTTPException(400, "Max 5 active API keys allowed. Revoke one first.")

    # ── Collision-safe key generation ──────────
    # Retry loop: generate until we get a hash not already in DB
    # In practice this NEVER loops — just a safety net
    max_attempts = 5
    full_key = key_prefix = key_hash = key_id = None

    for attempt in range(max_attempts):
        full_key, key_prefix, key_hash = _generate_api_key()
        key_id = _new_id()

        async with aiosqlite.connect(DB_PATH) as db:
            # Check if this hash already exists (collision check)
            async with db.execute(
                "SELECT 1 FROM api_keys WHERE key_hash = ?", (key_hash,)
            ) as cur:
                exists = await cur.fetchone()

            if not exists:
                # Unique! Insert it
                try:
                    await db.execute(
                        """INSERT INTO api_keys
                           (id, user_id, key_hash, key_prefix, label, is_active, created_at)
                           VALUES (?,?,?,?,?,1,?)""",
                        (key_id, user_id, key_hash, key_prefix, body.label, _now())
                    )
                    await db.commit()
                    break   # ✓ Success
                except Exception:
                    # Rare: race condition between check and insert
                    # Just retry
                    continue
        else:
            logger.warning(f"Key hash collision on attempt {attempt + 1}, retrying...")
    else:
        # Extremely unlikely — would require 5 consecutive collisions
        raise HTTPException(500, "Key generation failed. Please try again.")

    logger.info(f"API key created: {key_prefix}•••• for user {user_id[:8]}...")

    return {
        "key_id":   key_id,
        "full_key": full_key,     # ⚠️ Shown ONCE — never stored raw, never shown again
        "prefix":   key_prefix,   # Safe to display in dashboard
        "label":    body.label,
        "format":   "lxr_v1_<prefix>_<random>",
        "warning":  "Save this key immediately — it will NOT be shown again.",
    }


@router.post("/apikey/rotate")
async def rotate_api_key(key_id: str, request: Request):
    """
    Rotate an existing API key.

    - Old key is IMMEDIATELY invalidated (atomic swap in DB)
    - New key has same format: lxr_v1_<prefix>_<random>
    - New key shown ONCE — save it immediately
    - Collision retry ensures new key is globally unique
    """
    user_id = await get_current_user(request)

    # Verify key belongs to this user
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM api_keys WHERE id=? AND user_id=? AND is_active=1",
            (key_id, user_id)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        raise HTTPException(404, "API key not found or already revoked")

    # ── Collision-safe new key generation ─────
    max_attempts = 5
    full_key = key_prefix = key_hash = None

    for attempt in range(max_attempts):
        full_key, key_prefix, key_hash = _generate_api_key()

        async with aiosqlite.connect(DB_PATH) as db:
            # Check collision (excluding current key being rotated)
            async with db.execute(
                "SELECT 1 FROM api_keys WHERE key_hash=? AND id != ?",
                (key_hash, key_id)
            ) as cur:
                exists = await cur.fetchone()

            if not exists:
                # Atomic swap: old key gone, new key in
                await db.execute(
                    """UPDATE api_keys
                       SET key_hash=?, key_prefix=?, created_at=?
                       WHERE id=? AND user_id=?""",
                    (key_hash, key_prefix, _now(), key_id, user_id)
                )
                await db.commit()
                break
        else:
            logger.warning(f"Rotation collision on attempt {attempt + 1}, retrying...")
    else:
        raise HTTPException(500, "Key rotation failed. Please try again.")

    logger.info(f"API key rotated: {key_prefix}•••• for user {user_id[:8]}...")

    return {
        "key_id":   key_id,
        "full_key": full_key,     # ⚠️ Shown ONCE — save immediately
        "prefix":   key_prefix,
        "warning":  "Old key is now INVALID. Save this new key — it will NOT be shown again.",
    }


@router.get("/apikey/list")
async def list_api_keys(request: Request):
    """List user's API keys (prefix only, never full key)."""
    user_id = await get_current_user(request)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, key_prefix, label, is_active, created_at, last_used, total_calls
               FROM api_keys WHERE user_id=? ORDER BY created_at DESC""",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()

    return {
        "keys": [
            {
                "key_id":      r["id"],
                # Show full prefix (lxr_v1_XXXXXXXX) + mask rest
                "display":     r["key_prefix"] + "_••••••••••••••••••••••••",
                "label":       r["label"],
                "is_active":   bool(r["is_active"]),
                "created_at":  r["created_at"],
                "last_used":   r["last_used"],
                "total_calls": r["total_calls"],
            }
            for r in rows
        ]
    }


@router.delete("/apikey/{key_id}")
async def revoke_api_key(key_id: str, request: Request):
    """Revoke (deactivate) an API key."""
    user_id = await get_current_user(request)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE api_keys SET is_active=0 WHERE id=? AND user_id=?",
            (key_id, user_id)
        )
        await db.commit()

    return {"status": "revoked", "key_id": key_id}


# ─────────────────────────────────────────────
# Internal API key helpers
# ─────────────────────────────────────────────
async def _validate_api_key(full_key: str, user_id: str) -> bool:
    """
    Validate an API key with 2-layer security:

    Layer 1 — Format check (no DB hit):
        Instantly rejects any key that doesn't match lxr_v1_<prefix>_<random> format.
        Stops brute-force attempts, random strings, old format keys at zero DB cost.

    Layer 2 — DB hash check:
        SHA256(full_key) must exist in api_keys table AND be active.
        Raw key is never stored — only the hash is compared.
    """
    # ── Layer 1: Format validation ────────────
    # Fast reject — no DB query needed for obviously fake keys
    if not _validate_key_format(full_key):
        logger.warning(f"Invalid API key format attempt from user {user_id[:8]}...")
        return False

    # ── Layer 2: DB hash lookup ───────────────
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT user_id FROM api_keys
               WHERE key_hash=? AND is_active=1""",
            (key_hash,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        logger.warning(f"API key not found or inactive for user {user_id[:8]}...")
        return False

    # ── Layer 3: Ownership check ──────────────
    # Key must belong to the authenticated user
    if row["user_id"] != user_id:
        logger.warning(f"API key ownership mismatch — key belongs to different user!")
        return False

    return True


async def _record_api_key_usage(full_key: str):
    """Update last_used and total_calls on the key."""
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE api_keys
               SET last_used=?, total_calls=total_calls+1
               WHERE key_hash=?""",
            (_now(), key_hash)
        )
        await db.commit()
