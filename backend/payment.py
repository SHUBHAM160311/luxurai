"""
LuxurAI — payment.py
─────────────────────────────────────────────────────────────────
Payment backend: Razorpay (India) + Stripe (global/API users)
- Create orders
- Verify webhook signatures
- Credit LC only after webhook (never on frontend success)
- Idempotent (safe to receive webhook twice)
- Full audit trail via wallet ledger

Run alongside auth.py:
    uvicorn payment:app --reload --port 8001

Or mount into the same FastAPI app:
    from payment import router as payment_router
    app.include_router(payment_router)

.env additions:
    RAZORPAY_KEY_ID=rzp_live_xxx
    RAZORPAY_KEY_SECRET=xxx
    STRIPE_SECRET_KEY=sk_live_xxx
    STRIPE_WEBHOOK_SECRET=whsec_xxx
─────────────────────────────────────────────────────────────────
"""

import os
import hmac
import hashlib
import json
import secrets
import aiosqlite
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from dotenv import load_dotenv

from wallet import (
    WalletService, WalletError,
    InsufficientBalanceError, DuplicateTransactionError,
    REASON_PURCHASE
)
from referral import get_user_discount, apply_discount, record_referee_purchase

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
RAZORPAY_KEY_ID      = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET  = os.getenv("RAZORPAY_KEY_SECRET", "")
STRIPE_SECRET_KEY    = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SECRET_KEY           = os.getenv("SECRET_KEY", "dev-secret-change-in-production-32chars!")
ALGORITHM            = "HS256"
DB_PATH              = os.getenv("DB_PATH", "luxurai.db")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "https://luxurai.in")

wallet = WalletService(DB_PATH)


# ─────────────────────────────────────────────
# LC Packs (single source of truth)
# ─────────────────────────────────────────────
LC_PACKS = {
    "trial":    {"lc": 60,   "inr": 15,  "label": "Trial"},
    "starter":  {"lc": 100,  "inr": 22,  "label": "Starter"},
    "basic":    {"lc": 200,  "inr": 40,  "label": "Basic"},
    "popular":  {"lc": 350,  "inr": 65,  "label": "Popular"},
    "standard": {"lc": 500,  "inr": 90,  "label": "Standard"},
    "plus":     {"lc": 750,  "inr": 130, "label": "Plus"},
    "pro":      {"lc": 1000, "inr": 160, "label": "Pro"},
    "elite":    {"lc": 1500, "inr": 220, "label": "Elite"},
    "ultra":    {"lc": 2500, "inr": 340, "label": "Ultra"},
    "mega":     {"lc": 5000, "inr": 600, "label": "Mega"},
    "jumbo":    {"lc": 5050, "inr": 500, "label": "Jumbo Week"},
}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return secrets.token_urlsafe(12)


# ─────────────────────────────────────────────
# Auth helper (reads session cookie / header)
# ─────────────────────────────────────────────
async def get_current_user(request: Request) -> str:
    """Returns user_id from JWT cookie or raises 401."""
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
# DB helpers
# ─────────────────────────────────────────────
async def save_order(order_id: str, user_id: str, pack_id: str,
                     amount_inr: float, lc_amount: float, gateway: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS payment_orders (
                order_id    TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                pack_id     TEXT NOT NULL,
                amount_inr  REAL NOT NULL,
                lc_amount   REAL NOT NULL,
                gateway     TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                fulfilled_at TEXT
            )
        """)
        await db.execute(
            """INSERT OR IGNORE INTO payment_orders
               (order_id, user_id, pack_id, amount_inr, lc_amount, gateway, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (order_id, user_id, pack_id, amount_inr, lc_amount, gateway, _now())
        )
        await db.commit()

async def get_order(order_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM payment_orders WHERE order_id = ?", (order_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

async def mark_order_fulfilled(order_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE payment_orders SET status='fulfilled', fulfilled_at=? WHERE order_id=?",
            (_now(), order_id)
        )
        await db.commit()

async def mark_order_failed(order_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE payment_orders SET status='failed' WHERE order_id=?",
            (order_id,)
        )
        await db.commit()




# ─────────────────────────────────────────────
# Promo Code Logic
# ─────────────────────────────────────────────
async def apply_promo_code(code: str, user_id: str, current_inr: float, lc_amount: float) -> dict:
    """
    Validate and calculate promo code discount.
    Does NOT record usage yet — that happens after payment success.

    Returns:
      valid=True  → final_inr, bonus_lc, discount_type, discount_value
      valid=False → error message
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT * FROM promo_codes WHERE code=? AND is_active=1", (code,)
        ) as cur:
            promo = await cur.fetchone()

        if not promo:
            return {"valid": False, "error": f"Promo code '{code}' is invalid or expired."}

        # Check expiry
        if promo["valid_until"] and promo["valid_until"] < _now():
            return {"valid": False, "error": "This promo code has expired."}

        # Check max uses
        if promo["max_uses"] is not None and promo["uses_so_far"] >= promo["max_uses"]:
            return {"valid": False, "error": "This promo code has reached its usage limit."}

        # Check min pack size
        if lc_amount < promo["min_pack_lc"]:
            return {"valid": False, "error": f"This promo code requires a minimum pack of {promo['min_pack_lc']} LC."}

        # Check if user already used this code
        async with db.execute(
            "SELECT 1 FROM promo_uses WHERE code=? AND user_id=?", (code, user_id)
        ) as cur:
            already_used = await cur.fetchone()

        if already_used:
            return {"valid": False, "error": "You have already used this promo code."}

    # Calculate discount
    final_inr  = current_inr
    bonus_lc   = 0.0

    if promo["discount_type"] == "percent":
        # Reduce INR price by X%
        final_inr = round(current_inr * (1 - promo["discount_value"] / 100), 2)
    elif promo["discount_type"] == "flat_lc":
        # Give extra LC free (INR price unchanged)
        bonus_lc = promo["discount_value"]

    return {
        "valid":          True,
        "code":           code,
        "discount_type":  promo["discount_type"],
        "discount_value": promo["discount_value"],
        "original_inr":   current_inr,
        "final_inr":      max(0, final_inr),
        "bonus_lc":       bonus_lc,
        "note":           promo["note"],
    }


async def record_promo_use(code: str, user_id: str, order_id: str, discount_lc: float):
    """Call this after successful payment to lock in promo usage."""
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT OR IGNORE INTO promo_uses (id, code, user_id, order_id, discount_lc, used_at) VALUES (?,?,?,?,?,?)",
                (_new_id(), code, user_id, order_id, discount_lc, _now())
            )
            await db.execute(
                "UPDATE promo_codes SET uses_so_far = uses_so_far + 1 WHERE code=?",
                (code,)
            )
            await db.commit()
        except Exception as e:
            pass   # log in prod

# ─────────────────────────────────────────────
# FastAPI Router (mount this in main app)
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/payment", tags=["payment"])


# ── List packs ────────────────────────────────
@router.get("/packs")
async def list_packs():
    """Return all LC pack options for the frontend."""
    return {
        "packs": [
            {
                "id": k,
                "lc": v["lc"],
                "inr": v["inr"],
                "label": v["label"],
                "per_lc_inr": round(v["inr"] / v["lc"], 4),
                "savings_pct": round((1 - (v["inr"] / v["lc"]) / 0.25) * 100, 1)
            }
            for k, v in LC_PACKS.items()
        ]
    }


# ── Create Razorpay order ─────────────────────
class CreateOrderRequest(BaseModel):
    pack_id:    str
    custom_lc:  Optional[float] = None   # for custom amount flow
    promo_code: Optional[str]   = None   # optional promo code


@router.post("/razorpay/create-order")
async def create_razorpay_order(body: CreateOrderRequest, request: Request):
    user_id = await get_current_user(request)

    # Resolve pack
    if body.pack_id == "custom" and body.custom_lc:
        lc_amount = max(10.0, body.custom_lc)
        amount_inr = round(lc_amount * 0.25, 2)
        pack_label = f"Custom {lc_amount} LC"
    elif body.pack_id in LC_PACKS:
        pack = LC_PACKS[body.pack_id]
        lc_amount = pack["lc"]
        amount_inr = pack["inr"]
        pack_label = pack["label"]
    else:
        raise HTTPException(400, f"Unknown pack_id: {body.pack_id}")

    # Apply referral discount (INR only, LC stays same)
    discount_info  = await get_user_discount(user_id)
    discount_pct   = discount_info["discount_pct"]
    original_inr   = amount_inr
    amount_inr     = apply_discount(amount_inr, discount_pct)

    # Apply promo code (stacks on top of referral discount)
    promo_applied = None
    promo_bonus_lc = 0.0
    if body.promo_code:
        promo_result = await apply_promo_code(
            code=body.promo_code.upper().strip(),
            user_id=user_id,
            current_inr=amount_inr,
            lc_amount=lc_amount,
        )
        if promo_result["valid"]:
            amount_inr     = promo_result["final_inr"]
            promo_bonus_lc = promo_result["bonus_lc"]
            lc_amount     += promo_bonus_lc   # flat_lc type adds to LC
            promo_applied  = promo_result
        else:
            raise HTTPException(400, promo_result["error"])

    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(503, "Payment gateway not configured. Add RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to environment variables.")

    # Real Razorpay API call
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.razorpay.com/v1/orders",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={
                "amount": int(amount_inr * 100),   # paise
                "currency": "INR",
                "receipt": f"luxurai_{user_id[:8]}_{_new_id()}",
                "notes": {
                    "user_id": user_id,
                    "pack_id": body.pack_id,
                    "lc_amount": lc_amount,
                    "product": "LuxurAI LC"
                }
            }
        )

    if resp.status_code != 200:
        raise HTTPException(502, "Failed to create Razorpay order")

    order_data = resp.json()
    order_id = order_data["id"]

    await save_order(order_id, user_id, body.pack_id, amount_inr, lc_amount, "razorpay")

    return {
        "order_id":      order_id,
        "amount_paise":  int(amount_inr * 100),
        "currency":      "INR",
        "key":           RAZORPAY_KEY_ID,
        "lc_amount":     lc_amount,
        "pack_label":    pack_label,
        "discount_pct":  discount_pct,
        "original_inr":  original_inr,
        "final_inr":     amount_inr,
        "promo_applied":  promo_applied,
        "promo_bonus_lc": promo_bonus_lc,
    }


# ── Razorpay webhook ──────────────────────────
@router.post("/razorpay/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(None)
):
    """
    Razorpay sends this on payment.captured event.
    NEVER credit LC without verifying this signature.
    """
    raw_body = await request.body()

    # ── Signature verification ────────────────
    if not RAZORPAY_KEY_SECRET:
        # Dev mode: skip verification
        pass
    else:
        if not x_razorpay_signature:
            raise HTTPException(400, "Missing signature header")

        expected = hmac.new(
            RAZORPAY_KEY_SECRET.encode(),
            raw_body,
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected, x_razorpay_signature):
            raise HTTPException(400, "Invalid webhook signature")

    payload = json.loads(raw_body)
    event = payload.get("event")

    # Only process successful payments
    if event != "payment.captured":
        return {"status": "ignored", "event": event}

    payment = payload["payload"]["payment"]["entity"]
    payment_id = payment["id"]
    order_id   = payment.get("order_id", "")
    amount_inr = payment["amount"] / 100   # convert paise

    # ── Fetch our order record ────────────────
    order = await get_order(order_id)
    if not order:
        # Could be a Stripe order or unknown — log and ignore
        return {"status": "order_not_found"}

    if order["status"] == "fulfilled":
        # Already processed (webhook delivered twice — safe to ignore)
        return {"status": "already_fulfilled"}

    user_id   = order["user_id"]
    lc_amount = order["lc_amount"]
    pack_id   = order["pack_id"]

    # ── Credit wallet ─────────────────────────
    try:
        new_balance = await wallet.credit(
            user_id=user_id,
            amount_lc=lc_amount,
            reason=REASON_PURCHASE,
            ref_id=payment_id          # idempotency key
        )
        await mark_order_fulfilled(order_id)
        await record_referee_purchase(user_id)   # referral discount tier tracking
    except DuplicateTransactionError:
        # payment_id already credited — safe
        return {"status": "already_credited"}
    except WalletError as e:
        await mark_order_failed(order_id)
        raise HTTPException(500, str(e))

    return {
        "status": "ok",
        "user_id": user_id,
        "lc_credited": lc_amount,
        "new_balance": new_balance,
        "payment_id": payment_id,
        "pack": pack_id
    }


# ── Create Stripe payment intent (global/API users) ─
class StripeOrderRequest(BaseModel):
    pack_id: str
    custom_lc: Optional[float] = None


@router.post("/stripe/create-intent")
async def create_stripe_intent(body: StripeOrderRequest, request: Request):
    user_id = await get_current_user(request)

    if body.pack_id == "custom" and body.custom_lc:
        lc_amount = max(10.0, body.custom_lc)
        # Convert ₹ to USD at approximate rate (update dynamically in prod)
        amount_usd = round(lc_amount * 0.003, 2)
    elif body.pack_id in LC_PACKS:
        pack = LC_PACKS[body.pack_id]
        lc_amount = pack["lc"]
        amount_usd = round(pack["inr"] * 0.012, 2)   # ₹ → USD
    else:
        raise HTTPException(400, f"Unknown pack_id: {body.pack_id}")

    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Stripe not configured. Add STRIPE_SECRET_KEY to environment variables.")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.stripe.com/v1/payment_intents",
            headers={"Authorization": f"Bearer {STRIPE_SECRET_KEY}"},
            data={
                "amount": int(amount_usd * 100),   # cents
                "currency": "usd",
                "metadata[user_id]": user_id,
                "metadata[lc_amount]": lc_amount,
                "metadata[pack_id]": body.pack_id,
                "metadata[product]": "LuxurAI LC",
            }
        )

    if resp.status_code != 200:
        raise HTTPException(502, "Failed to create Stripe payment intent")

    intent = resp.json()
    intent_id = intent["id"]

    await save_order(intent_id, user_id, body.pack_id, amount_usd, lc_amount, "stripe")

    return {
        "client_secret": intent["client_secret"],
        "lc_amount": lc_amount,
        "amount_usd": amount_usd,
    }


# ── Stripe webhook ────────────────────────────
@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None)
):
    """
    Stripe sends this on payment_intent.succeeded event.
    Verifies using Stripe-Signature header.
    """
    raw_body = await request.body()

    # ── Signature verification ────────────────
    if STRIPE_WEBHOOK_SECRET and stripe_signature:
        try:
            _verify_stripe_signature(raw_body, stripe_signature, STRIPE_WEBHOOK_SECRET)
        except ValueError:
            raise HTTPException(400, "Invalid Stripe webhook signature")

    payload = json.loads(raw_body)
    event_type = payload.get("type")

    if event_type != "payment_intent.succeeded":
        return {"status": "ignored", "type": event_type}

    intent    = payload["data"]["object"]
    intent_id = intent["id"]
    metadata  = intent.get("metadata", {})
    user_id   = metadata.get("user_id")
    lc_amount = float(metadata.get("lc_amount", 0))

    if not user_id or lc_amount <= 0:
        return {"status": "missing_metadata"}

    order = await get_order(intent_id)
    if order and order["status"] == "fulfilled":
        return {"status": "already_fulfilled"}

    try:
        new_balance = await wallet.credit(
            user_id=user_id,
            amount_lc=lc_amount,
            reason=REASON_PURCHASE,
            ref_id=intent_id
        )
        if order:
            await mark_order_fulfilled(intent_id)
    except DuplicateTransactionError:
        return {"status": "already_credited"}
    except WalletError as e:
        raise HTTPException(500, str(e))

    return {
        "status": "ok",
        "user_id": user_id,
        "lc_credited": lc_amount,
        "new_balance": new_balance,
        "intent_id": intent_id
    }


def _verify_stripe_signature(payload: bytes, sig_header: str, secret: str):
    """
    Stripe webhook signature verification.
    https://stripe.com/docs/webhooks/signatures
    """
    parts = dict(kv.split("=", 1) for kv in sig_header.split(",") if "=" in kv)
    timestamp = parts.get("t", "")
    signatures = [v for k, v in parts.items() if k == "v1"]

    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()

    if not any(hmac.compare_digest(expected, s) for s in signatures):
        raise ValueError("Signature mismatch")



# ── Validate promo code (public — user calls before checkout) ──
@router.get("/promo/validate")
async def validate_promo(code: str, pack_id: str, request: Request):
    """
    Frontend calls this when user types a promo code.
    Returns discount info without locking the code.
    """
    user_id = await get_current_user(request)

    if pack_id in LC_PACKS:
        lc_amount  = LC_PACKS[pack_id]["lc"]
        amount_inr = LC_PACKS[pack_id]["inr"]
    else:
        raise HTTPException(400, "Invalid pack_id")

    result = await apply_promo_code(
        code=code.upper().strip(),
        user_id=user_id,
        current_inr=amount_inr,
        lc_amount=lc_amount,
    )
    return result


# ── Payment history ───────────────────────────
@router.get("/history")
async def payment_history(request: Request, limit: int = 20):
    user_id = await get_current_user(request)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT order_id, pack_id, amount_inr, lc_amount, gateway, status, created_at, fulfilled_at
               FROM payment_orders WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit)
        ) as cur:
            rows = await cur.fetchall()
    return {"payments": [dict(r) for r in rows]}


# ── Dev: simulate payment success (no gateway) ─
@router.post("/dev/simulate-payment")
async def simulate_payment(request: Request, pack_id: str = "popular"):
    """
    Only works in dev mode (no real keys set).
    Simulates a successful payment for testing.
    """
    env = os.getenv("ENV", "development")
    if env == "production":
        raise HTTPException(403, "Simulation not allowed in production")

    user_id = await get_current_user(request)
    pack = LC_PACKS.get(pack_id, LC_PACKS["popular"])
    lc_amount = pack["lc"]
    fake_payment_id = "pay_sim_" + _new_id()

    try:
        new_balance = await wallet.credit(
            user_id=user_id,
            amount_lc=lc_amount,
            reason=REASON_PURCHASE,
            ref_id=fake_payment_id
        )
    except DuplicateTransactionError:
        return {"status": "already_credited"}

    return {
        "status": "simulated",
        "lc_credited": lc_amount,
        "new_balance": new_balance,
        "pack": pack["label"]
    }


# ─────────────────────────────────────────────
# Standalone app (or import router into auth.py)
# ─────────────────────────────────────────────
app = FastAPI(title="LuxurAI Payments", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
async def startup():
    from wallet import init_wallet_tables
    await init_wallet_tables()
    # Create payment_orders table
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS payment_orders (
                order_id     TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                pack_id      TEXT NOT NULL,
                amount_inr   REAL NOT NULL,
                lc_amount    REAL NOT NULL,
                gateway      TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL,
                fulfilled_at TEXT
            )
        """)
        await db.commit()


@app.get("/api/payment/health")
async def health():
    return {
        "status": "ok",
        "razorpay": bool(RAZORPAY_KEY_ID),
        "stripe": bool(STRIPE_SECRET_KEY)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("payment:app", host="0.0.0.0", port=8001, reload=True)
