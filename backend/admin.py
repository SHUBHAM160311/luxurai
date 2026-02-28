"""
LuxurAI — admin.py
─────────────────────────────────────────────────────────────────
Admin Panel Backend

SECURITY:
  - Login requires BOTH: ADMIN_EMAIL + ADMIN_SECRET_KEY (from .env)
  - Sensitive actions (promo create, block, credit) require
    a second password confirmation (same password, re-entered)
  - All actions are logged with timestamp

.env required:
  ADMIN_EMAIL=achyut@luxurai.in
  ADMIN_SECRET_KEY=your_strong_password

ENDPOINTS:
  POST /api/admin/login                     → Email + password login
  POST /api/admin/verify-action             → Confirm password for sensitive actions
  GET  /api/admin/overview                  → Platform stats
  GET  /api/admin/users                     → All users (paginated, searchable)
  GET  /api/admin/users/{user_id}           → Single user detail
  POST /api/admin/users/{user_id}/block     → Block user (needs confirm)
  POST /api/admin/users/{user_id}/unblock   → Unblock user (needs confirm)
  POST /api/admin/users/{user_id}/credit    → Manual LC credit (needs confirm)
  GET  /api/admin/payments                  → All payments
  GET  /api/admin/jobs                      → All jobs
  GET  /api/admin/referrals                 → Referral stats
  GET  /api/admin/loyalty                   → Loyalty stats
  POST /api/admin/promo/create              → Create promo code (needs confirm)
  GET  /api/admin/promo/list                → All promo codes
  POST /api/admin/promo/toggle/{code}       → Enable/disable a promo code
─────────────────────────────────────────────────────────────────
"""

import os
import secrets
import logging
import aiosqlite
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from jose import JWTError, jwt
from dotenv import load_dotenv

from backend.wallet import WalletService, REASON_ADMIN

load_dotenv()
load_dotenv("CashFree.env")

logger = logging.getLogger("luxurai.admin")

# ─────────────────────────────────────────────
# Config — SET THESE IN .env
# ─────────────────────────────────────────────
DB_PATH          = os.getenv("DB_PATH", "luxurai.db")
ADMIN_EMAIL      = os.getenv("ADMIN_EMAIL", "")           # e.g. achyut@luxurai.in
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "")      # your strong password
JWT_SECRET       = os.getenv("SECRET_KEY", "dev-secret")
ALGORITHM        = "HS256"
ADMIN_TOKEN_HOURS = 8   # admin session expires in 8 hours

wallet = WalletService(DB_PATH)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return secrets.token_urlsafe(12)

def _make_admin_jwt() -> str:
    payload = {
        "sub":  "admin",
        "role": "admin",
        "exp":  datetime.now(timezone.utc) + timedelta(hours=ADMIN_TOKEN_HOURS)
    }
    return jwt.encode(payload, ADMIN_SECRET_KEY, algorithm=ALGORITHM)

def _verify_admin_password(password: str) -> bool:
    """Constant-time compare to prevent timing attacks."""
    import hmac
    return hmac.compare_digest(password, ADMIN_SECRET_KEY)

async def require_admin(request: Request):
    """Verify admin JWT from Authorization header."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(401, "Admin token required.")
    try:
        payload = jwt.decode(token, ADMIN_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(403, "Not an admin token.")
    except JWTError:
        raise HTTPException(401, "Invalid or expired admin token. Please login again.")


# ─────────────────────────────────────────────
# DB Init
# ─────────────────────────────────────────────
async def init_admin_tables(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS promo_codes (
                code            TEXT PRIMARY KEY,
                discount_type   TEXT NOT NULL,   -- 'percent' or 'flat_lc'
                discount_value  REAL NOT NULL,   -- e.g. 50 (for 50% or 50 LC)
                min_pack_lc     REAL NOT NULL DEFAULT 0,   -- minimum pack size to apply
                max_uses        INTEGER,         -- NULL = unlimited
                uses_so_far     INTEGER NOT NULL DEFAULT 0,
                valid_from      TEXT NOT NULL,
                valid_until     TEXT,            -- NULL = no expiry
                is_active       INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL,
                note            TEXT             -- admin note
            );

            CREATE TABLE IF NOT EXISTS promo_uses (
                id          TEXT PRIMARY KEY,
                code        TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                order_id    TEXT NOT NULL,
                discount_lc REAL NOT NULL,
                used_at     TEXT NOT NULL,
                UNIQUE(code, user_id)            -- one use per user per code
            );

            CREATE TABLE IF NOT EXISTS admin_action_log (
                id          TEXT PRIMARY KEY,
                action      TEXT NOT NULL,
                detail      TEXT,
                performed_at TEXT NOT NULL
            );
        """)
        await db.commit()
    logger.info("Admin + promo tables initialized.")


async def _log_action(action: str, detail: str = ""):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO admin_action_log (id, action, detail, performed_at) VALUES (?,?,?,?)",
            (_new_id(), action, detail, _now())
        )
        await db.commit()


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── LOGIN ─────────────────────────────────────
class AdminLoginRequest(BaseModel):
    email:    str
    password: str

@router.post("/login")
async def admin_login(body: AdminLoginRequest):
    """
    Login with email + password from .env
    Returns admin JWT valid for 8 hours.
    """
    if not ADMIN_EMAIL or not ADMIN_SECRET_KEY:
        raise HTTPException(503, "Admin credentials not configured in .env")

    # Check email first
    if body.email.lower().strip() != ADMIN_EMAIL.lower().strip():
        logger.warning(f"Failed admin login — wrong email: {body.email}")
        raise HTTPException(403, "Invalid credentials.")

    # Check password (constant-time)
    if not _verify_admin_password(body.password):
        logger.warning("Failed admin login — wrong password.")
        raise HTTPException(403, "Invalid credentials.")

    token = _make_admin_jwt()
    await _log_action("admin_login", f"email={body.email}")
    logger.info(f"Admin logged in: {body.email}")

    return {
        "token":      token,
        "expires_in": f"{ADMIN_TOKEN_HOURS} hours",
        "admin":      body.email,
    }


# ── CONFIRM PASSWORD (for sensitive actions) ──
class ConfirmRequest(BaseModel):
    password: str
    action:   str   # human readable — "I am blocking user X" etc.

@router.post("/verify-action")
async def verify_action(body: ConfirmRequest, request: Request):
    """
    Second password check before sensitive actions.
    Frontend shows a confirm popup, user re-enters password.
    Returns a short-lived action_token valid for 2 minutes.
    """
    await require_admin(request)

    if not _verify_admin_password(body.password):
        logger.warning(f"Admin action confirm failed — wrong password. Action: {body.action}")
        raise HTTPException(403, "Wrong password. Action cancelled.")

    # Create a short-lived confirm token
    confirm_payload = {
        "sub":    "admin_confirm",
        "action": body.action,
        "exp":    datetime.now(timezone.utc) + timedelta(minutes=2)
    }
    confirm_token = jwt.encode(confirm_payload, ADMIN_SECRET_KEY, algorithm=ALGORITHM)

    await _log_action("action_confirmed", body.action)
    logger.info(f"Admin confirmed action: {body.action}")

    return {
        "confirmed":     True,
        "action":        body.action,
        "confirm_token": confirm_token,
        "valid_for":     "2 minutes",
    }


def _verify_confirm_token(token: str, expected_action_prefix: str = "") -> bool:
    """Verify the short-lived confirm token."""
    try:
        payload = jwt.decode(token, ADMIN_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") != "admin_confirm":
            return False
        if expected_action_prefix:
            action = payload.get("action", "")
            if expected_action_prefix.lower() not in action.lower():
                return False
        return True
    except JWTError:
        return False


# ── OVERVIEW ──────────────────────────────────
@router.get("/overview")
async def get_overview(request: Request):
    """Main admin dashboard — all platform stats."""
    await require_admin(request)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT COUNT(*) as c FROM users") as cur:
            total_users = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(*) as c FROM users WHERE created_at >= DATE('now', '-1 day')") as cur:
            new_today = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(*) as c FROM users WHERE created_at >= DATE('now', '-7 days')") as cur:
            new_this_week = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(*) as c FROM users WHERE is_blocked = 1") as cur:
            blocked_users = (await cur.fetchone())["c"]

        async with db.execute(
            "SELECT COUNT(*) as c, COALESCE(SUM(lc_amount),0) as lc, COALESCE(SUM(amount_inr),0) as inr FROM payment_orders WHERE status='fulfilled'"
        ) as cur:
            row = await cur.fetchone()
            total_purchases, total_lc_sold, total_revenue_inr = row["c"], row["lc"], row["inr"]

        async with db.execute(
            "SELECT COALESCE(SUM(amount_inr),0) as inr FROM payment_orders WHERE status='fulfilled' AND created_at >= DATE('now','-1 day')"
        ) as cur:
            revenue_today = (await cur.fetchone())["inr"]

        async with db.execute(
            "SELECT COALESCE(SUM(amount_inr),0) as inr FROM payment_orders WHERE status='fulfilled' AND created_at >= DATE('now','-7 days')"
        ) as cur:
            revenue_week = (await cur.fetchone())["inr"]

        async with db.execute("SELECT COUNT(*) as c FROM jobs") as cur:
            total_jobs = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(*) as c FROM jobs WHERE status='done'") as cur:
            total_done = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(*) as c FROM jobs WHERE created_at >= DATE('now','-1 day')") as cur:
            jobs_today = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(*) as c FROM jobs WHERE status='failed'") as cur:
            total_failed = (await cur.fetchone())["c"]
        async with db.execute("SELECT COUNT(DISTINCT user_id) as c FROM jobs WHERE created_at >= DATE('now','-7 days')") as cur:
            active_users_week = (await cur.fetchone())["c"]
        async with db.execute("SELECT COALESCE(SUM(balance_lc),0) as total FROM wallets") as cur:
            lc_in_circulation = (await cur.fetchone())["total"]

        # Top 5 spenders
        async with db.execute("""
            SELECT u.email, u.id,
                   COALESCE(SUM(CASE WHEN wl.delta_lc < 0 THEN ABS(wl.delta_lc) ELSE 0 END),0) as lc_spent,
                   COALESCE(w.balance_lc,0) as balance
            FROM users u
            LEFT JOIN wallet_ledger wl ON wl.user_id = u.id
            LEFT JOIN wallets w ON w.user_id = u.id
            GROUP BY u.id ORDER BY lc_spent DESC LIMIT 5
        """) as cur:
            top_spenders = [dict(r) for r in await cur.fetchall()]

        # Recent signups
        async with db.execute("SELECT id, email, provider, created_at FROM users ORDER BY created_at DESC LIMIT 10") as cur:
            recent_signups = [dict(r) for r in await cur.fetchall()]

        # Promo stats
        try:
            async with db.execute("SELECT COUNT(*) as c FROM promo_codes WHERE is_active=1") as cur:
                active_promos = (await cur.fetchone())["c"]
            async with db.execute("SELECT COALESCE(SUM(uses_so_far),0) as c FROM promo_codes") as cur:
                total_promo_uses = (await cur.fetchone())["c"]
        except Exception:
            active_promos = 0
            total_promo_uses = 0

        # Referral stats
        try:
            async with db.execute("SELECT COUNT(*) as c FROM referral_joins") as cur:
                total_referral_joins = (await cur.fetchone())["c"]
        except Exception:
            total_referral_joins = 0

        # Loyalty stats
        try:
            async with db.execute(
                "SELECT COUNT(*) as c, COALESCE(SUM(delta_lc),0) as lc FROM wallet_ledger WHERE reason LIKE 'loyalty%'"
            ) as cur:
                row = await cur.fetchone()
                loyalty_events, loyalty_lc = row["c"], row["lc"]
        except Exception:
            loyalty_events = loyalty_lc = 0

    return {
        "users": {
            "total":         total_users,
            "new_today":     new_today,
            "new_this_week": new_this_week,
            "blocked":       blocked_users,
            "active_7d":     active_users_week,
        },
        "revenue": {
            "total_inr":      round(total_revenue_inr, 2),
            "today_inr":      round(revenue_today, 2),
            "this_week_inr":  round(revenue_week, 2),
            "total_purchases": total_purchases,
            "total_lc_sold":  round(total_lc_sold, 2),
            "lc_in_wallets":  round(lc_in_circulation, 2),
        },
        "generations": {
            "total":        total_jobs,
            "completed":    total_done,
            "failed":       total_failed,
            "today":        jobs_today,
            "success_rate": round((total_done / total_jobs * 100) if total_jobs else 0, 1),
        },
        "promos": {
            "active_codes": active_promos,
            "total_uses":   total_promo_uses,
        },
        "referrals":  {"total_joins": total_referral_joins},
        "loyalty":    {"events": loyalty_events, "lc_given": round(loyalty_lc, 2)},
        "top_spenders": [
            {
                "email":    r["email"][:3] + "***@" + r["email"].split("@")[-1] if "@" in r["email"] else "***",
                "user_id":  r["id"],
                "lc_spent": round(r["lc_spent"], 2),
                "balance":  round(r["balance"], 2),
            }
            for r in top_spenders
        ],
        "recent_signups": recent_signups,
        "generated_at":   _now(),
    }


# ── USERS ─────────────────────────────────────
@router.get("/users")
async def list_users(request: Request, limit: int = 50, offset: int = 0, search: str = ""):
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if search:
            q = """SELECT u.id, u.email, u.provider, u.is_blocked, u.created_at, u.last_login,
                          COALESCE(w.balance_lc,0) as balance_lc,
                          COUNT(DISTINCT j.id) as total_jobs,
                          COALESCE(SUM(CASE WHEN po.status='fulfilled' THEN po.amount_inr ELSE 0 END),0) as total_spent_inr
                   FROM users u
                   LEFT JOIN wallets w ON w.user_id=u.id
                   LEFT JOIN jobs j ON j.user_id=u.id
                   LEFT JOIN payment_orders po ON po.user_id=u.id
                   WHERE u.email LIKE ? GROUP BY u.id ORDER BY u.created_at DESC LIMIT ? OFFSET ?"""
            args = (f"%{search}%", limit, offset)
        else:
            q = """SELECT u.id, u.email, u.provider, u.is_blocked, u.created_at, u.last_login,
                          COALESCE(w.balance_lc,0) as balance_lc,
                          COUNT(DISTINCT j.id) as total_jobs,
                          COALESCE(SUM(CASE WHEN po.status='fulfilled' THEN po.amount_inr ELSE 0 END),0) as total_spent_inr
                   FROM users u
                   LEFT JOIN wallets w ON w.user_id=u.id
                   LEFT JOIN jobs j ON j.user_id=u.id
                   LEFT JOIN payment_orders po ON po.user_id=u.id
                   GROUP BY u.id ORDER BY u.created_at DESC LIMIT ? OFFSET ?"""
            args = (limit, offset)
        async with db.execute(q, args) as cur:
            rows = await cur.fetchall()
        count_q = "SELECT COUNT(*) as c FROM users WHERE email LIKE ?" if search else "SELECT COUNT(*) as c FROM users"
        count_args = (f"%{search}%",) if search else ()
        async with db.execute(count_q, count_args) as cur:
            total = (await cur.fetchone())["c"]
    return {
        "total": total, "limit": limit, "offset": offset,
        "users": [
            {"id": r["id"], "email": r["email"], "provider": r["provider"],
             "is_blocked": bool(r["is_blocked"]), "balance_lc": round(r["balance_lc"], 2),
             "total_jobs": r["total_jobs"], "total_spent_inr": round(r["total_spent_inr"], 2),
             "created_at": r["created_at"], "last_login": r["last_login"]}
            for r in rows
        ],
    }


@router.get("/users/{user_id}")
async def get_user_detail(user_id: str, request: Request):
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE id=?", (user_id,)) as cur:
            user = await cur.fetchone()
        if not user:
            raise HTTPException(404, "User not found.")
        async with db.execute("SELECT balance_lc FROM wallets WHERE user_id=?", (user_id,)) as cur:
            w = await cur.fetchone()
        balance = w["balance_lc"] if w else 0.0
        async with db.execute("SELECT * FROM wallet_ledger WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (user_id,)) as cur:
            ledger = [dict(r) for r in await cur.fetchall()]
        async with db.execute("SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (user_id,)) as cur:
            jobs_list = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) as done, COALESCE(SUM(CASE WHEN status='done' THEN cost_lc ELSE 0 END),0) as lc_spent FROM jobs WHERE user_id=?",
            (user_id,)
        ) as cur:
            jstats = dict(await cur.fetchone())
        async with db.execute("SELECT * FROM payment_orders WHERE user_id=? ORDER BY created_at DESC", (user_id,)) as cur:
            payments = [dict(r) for r in await cur.fetchall()]
        total_paid = sum(p["amount_inr"] for p in payments if p["status"] == "fulfilled")
        try:
            async with db.execute("SELECT * FROM referral_joins WHERE referee_id=?", (user_id,)) as cur:
                ref_used = await cur.fetchone()
            async with db.execute("SELECT * FROM referral_codes WHERE user_id=?", (user_id,)) as cur:
                ref_code = await cur.fetchone()
        except Exception:
            ref_used = ref_code = None
        # Promo uses
        try:
            async with db.execute("SELECT * FROM promo_uses WHERE user_id=? ORDER BY used_at DESC", (user_id,)) as cur:
                promo_uses = [dict(r) for r in await cur.fetchall()]
        except Exception:
            promo_uses = []
    return {
        "profile":     {"id": user["id"], "email": user["email"], "provider": user["provider"],
                        "is_blocked": bool(user["is_blocked"]), "created_at": user["created_at"], "last_login": user["last_login"]},
        "wallet":      {"balance_lc": round(balance, 2), "total_lc_spent": round(jstats["lc_spent"], 2),
                        "total_paid_inr": round(total_paid, 2), "recent_ledger": ledger},
        "generations": {"total": jstats["total"], "completed": jstats["done"], "recent": jobs_list},
        "payments":    payments,
        "promo_uses":  promo_uses,
        "referral":    {"was_referred": ref_used is not None, "referred_by": dict(ref_used) if ref_used else None,
                        "has_ref_code": ref_code is not None, "ref_code_info": dict(ref_code) if ref_code else None},
    }


# ── BLOCK / UNBLOCK ───────────────────────────
class ConfirmedActionRequest(BaseModel):
    confirm_token: str

@router.post("/users/{user_id}/block")
async def block_user(user_id: str, body: ConfirmedActionRequest, request: Request):
    await require_admin(request)
    if not _verify_confirm_token(body.confirm_token, "block"):
        raise HTTPException(403, "Invalid or expired confirm token. Re-confirm the action.")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE users SET is_blocked=1 WHERE id=?", (user_id,))
        await db.commit()
    await _log_action("block_user", f"user_id={user_id}")
    logger.warning(f"ADMIN: User {user_id} BLOCKED.")
    return {"success": True, "user_id": user_id, "action": "blocked"}

@router.post("/users/{user_id}/unblock")
async def unblock_user(user_id: str, body: ConfirmedActionRequest, request: Request):
    await require_admin(request)
    if not _verify_confirm_token(body.confirm_token, "unblock"):
        raise HTTPException(403, "Invalid or expired confirm token. Re-confirm the action.")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE users SET is_blocked=0 WHERE id=?", (user_id,))
        await db.commit()
    await _log_action("unblock_user", f"user_id={user_id}")
    return {"success": True, "user_id": user_id, "action": "unblocked"}


# ── MANUAL LC CREDIT ──────────────────────────
class AdminCreditRequest(BaseModel):
    amount_lc:     float
    note:          str = "Admin manual credit"
    confirm_token: str

@router.post("/users/{user_id}/credit")
async def admin_credit_user(user_id: str, body: AdminCreditRequest, request: Request):
    await require_admin(request)
    if not _verify_confirm_token(body.confirm_token, "credit"):
        raise HTTPException(403, "Invalid or expired confirm token. Re-confirm the action.")
    if body.amount_lc <= 0:
        raise HTTPException(400, "Amount must be positive.")
    new_balance = await wallet.admin_credit(user_id=user_id, amount_lc=body.amount_lc, note=body.note, admin_id="admin")
    await _log_action("manual_credit", f"user={user_id} amount={body.amount_lc} note={body.note}")
    return {"success": True, "user_id": user_id, "credited_lc": body.amount_lc, "new_balance": new_balance}


# ── PROMO CODES ───────────────────────────────

class CreatePromoRequest(BaseModel):
    code:           str            # e.g. "SAVE50" — you choose it
    discount_type:  str            # "percent" or "flat_lc"
    discount_value: float          # e.g. 50 → 50% off OR 50 LC free
    min_pack_lc:    float = 0      # minimum pack size to apply promo
    max_uses:       Optional[int] = None   # None = unlimited
    valid_until:    Optional[str] = None   # ISO date string, None = no expiry
    note:           str = ""
    confirm_token:  str            # must re-confirm password before creating

@router.post("/promo/create")
async def create_promo_code(body: CreatePromoRequest, request: Request):
    """
    Create a new promo code.
    Requires confirm_token from /verify-action (password re-entry).

    discount_type = "percent"  → e.g. 50% off the INR price
    discount_type = "flat_lc"  → e.g. get 100 LC extra free
    """
    await require_admin(request)

    if not _verify_confirm_token(body.confirm_token, "promo"):
        raise HTTPException(403, "Invalid or expired confirm token. Re-confirm with your password.")

    if body.discount_type not in ("percent", "flat_lc"):
        raise HTTPException(400, "discount_type must be 'percent' or 'flat_lc'")
    if body.discount_value <= 0:
        raise HTTPException(400, "discount_value must be positive")
    if body.discount_type == "percent" and body.discount_value > 100:
        raise HTTPException(400, "Percent discount cannot exceed 100%")

    code = body.code.upper().strip()
    if not code:
        raise HTTPException(400, "Promo code cannot be empty.")

    async with aiosqlite.connect(DB_PATH) as db:
        # Check if code already exists
        async with db.execute("SELECT 1 FROM promo_codes WHERE code=?", (code,)) as cur:
            if await cur.fetchone():
                raise HTTPException(409, f"Promo code '{code}' already exists.")

        await db.execute(
            """INSERT INTO promo_codes
               (code, discount_type, discount_value, min_pack_lc, max_uses,
                uses_so_far, valid_from, valid_until, is_active, created_at, note)
               VALUES (?,?,?,?,?,0,?,?,1,?,?)""",
            (code, body.discount_type, body.discount_value, body.min_pack_lc,
             body.max_uses, _now(), body.valid_until, _now(), body.note)
        )
        await db.commit()

    await _log_action(
        "promo_created",
        f"code={code} type={body.discount_type} value={body.discount_value} max_uses={body.max_uses}"
    )
    logger.info(f"ADMIN: Promo code created: {code} ({body.discount_type} {body.discount_value})")

    return {
        "success":        True,
        "code":           code,
        "discount_type":  body.discount_type,
        "discount_value": body.discount_value,
        "min_pack_lc":    body.min_pack_lc,
        "max_uses":       body.max_uses,
        "valid_until":    body.valid_until,
        "note":           body.note,
    }


@router.get("/promo/list")
async def list_promo_codes(request: Request):
    """List all promo codes with usage stats."""
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM promo_codes ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    return {"promos": [dict(r) for r in rows]}


class TogglePromoRequest(BaseModel):
    confirm_token: str

@router.post("/promo/toggle/{code}")
async def toggle_promo(code: str, body: TogglePromoRequest, request: Request):
    """Enable or disable a promo code. Requires password confirm."""
    await require_admin(request)
    if not _verify_confirm_token(body.confirm_token, "promo"):
        raise HTTPException(403, "Invalid or expired confirm token.")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT is_active FROM promo_codes WHERE code=?", (code.upper(),)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(404, f"Promo code '{code}' not found.")
        new_status = 0 if row["is_active"] else 1
        await db.execute("UPDATE promo_codes SET is_active=? WHERE code=?", (new_status, code.upper()))
        await db.commit()

    status_str = "enabled" if new_status else "disabled"
    await _log_action("promo_toggled", f"code={code.upper()} status={status_str}")
    return {"success": True, "code": code.upper(), "is_active": bool(new_status), "status": status_str}


# ── PAYMENTS / JOBS / REFERRALS / LOYALTY ─────
@router.get("/payments")
async def list_all_payments(request: Request, limit: int = 50, offset: int = 0, status: str = ""):
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if status:
            async with db.execute(
                "SELECT po.*, u.email FROM payment_orders po LEFT JOIN users u ON u.id=po.user_id WHERE po.status=? ORDER BY po.created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT po.*, u.email FROM payment_orders po LEFT JOIN users u ON u.id=po.user_id ORDER BY po.created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ) as cur:
                rows = await cur.fetchall()
    return {"payments": [dict(r) for r in rows], "limit": limit, "offset": offset}


@router.get("/jobs")
async def list_all_jobs(request: Request, limit: int = 50, offset: int = 0, status: str = ""):
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if status:
            async with db.execute(
                "SELECT j.*, u.email FROM jobs j LEFT JOIN users u ON u.id=j.user_id WHERE j.status=? ORDER BY j.created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT j.*, u.email FROM jobs j LEFT JOIN users u ON u.id=j.user_id ORDER BY j.created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ) as cur:
                rows = await cur.fetchall()
    return {"jobs": [dict(r) for r in rows], "limit": limit, "offset": offset}


@router.get("/referrals")
async def admin_referral_stats(request: Request):
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            async with db.execute("""
                SELECT rc.user_id, u.email, rc.code, rc.total_clicks, rc.total_joins,
                       COALESCE(SUM(rj.reward_lc),0) as total_lc_earned
                FROM referral_codes rc
                LEFT JOIN users u ON u.id=rc.user_id
                LEFT JOIN referral_joins rj ON rj.referrer_id=rc.user_id
                GROUP BY rc.user_id ORDER BY rc.total_joins DESC LIMIT 20
            """) as cur:
                top_referrers = [dict(r) for r in await cur.fetchall()]
            async with db.execute("SELECT COUNT(*) as c FROM referral_joins") as cur:
                total_joins = (await cur.fetchone())["c"]
            async with db.execute("SELECT COALESCE(SUM(reward_lc+bonus_lc),0) as lc FROM referral_joins") as cur:
                total_lc = (await cur.fetchone())["lc"]
        except Exception:
            top_referrers, total_joins, total_lc = [], 0, 0
    return {"total_joins": total_joins, "total_lc_given": round(total_lc, 2), "top_referrers": top_referrers}


@router.get("/loyalty")
async def admin_loyalty_stats(request: Request):
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT wl.user_id, u.email, wl.delta_lc, wl.reason, wl.created_at
            FROM wallet_ledger wl LEFT JOIN users u ON u.id=wl.user_id
            WHERE wl.reason LIKE 'loyalty%' ORDER BY wl.created_at DESC LIMIT 100
        """) as cur:
            rows = await cur.fetchall()
        async with db.execute(
            "SELECT COUNT(*) as c, COALESCE(SUM(delta_lc),0) as lc FROM wallet_ledger WHERE reason LIKE 'loyalty%'"
        ) as cur:
            stats = await cur.fetchone()
    return {"total_events": stats["c"], "total_lc_given": round(stats["lc"], 2), "recent": [dict(r) for r in rows]}


@router.get("/action-log")
async def get_action_log(request: Request, limit: int = 50):
    """Full log of all admin actions."""
    await require_admin(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM admin_action_log ORDER BY performed_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return {"log": [dict(r) for r in rows]}
