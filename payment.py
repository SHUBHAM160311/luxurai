"""
LuxurAI Payment Routes
Cashfree Payment Gateway integration for LC pack purchases
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import httpx
import hmac
import hashlib
import logging
import os
import json
import time
import base64

import database as db
from routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# ==================== CONFIG ====================
CASHFREE_APP_ID     = os.getenv("CASHFREE_APP_ID", "YOUR_CASHFREE_APP_ID")
CASHFREE_SECRET_KEY = os.getenv("CASHFREE_SECRET_KEY", "YOUR_CASHFREE_SECRET_KEY")

# Production endpoint
CASHFREE_BASE_URL = "https://api.cashfree.com/pg"

CASHFREE_HEADERS = {
    "x-api-version": "2023-08-01",
    "x-client-id": CASHFREE_APP_ID,
    "x-client-secret": CASHFREE_SECRET_KEY,
    "Content-Type": "application/json"
}


# ==================== LC PACKS ====================
LC_PACKS = {
    "trial":        {"name": "Trial",        "lc": 60,    "inr": 15},
    "starter":      {"name": "Starter",      "lc": 100,   "inr": 22},
    "basic":        {"name": "Basic",        "lc": 200,   "inr": 40},
    "popular":      {"name": "Popular",      "lc": 350,   "inr": 65},
    "standard":     {"name": "Standard",     "lc": 500,   "inr": 90},
    "plus":         {"name": "Plus",         "lc": 750,   "inr": 130},
    "pro":          {"name": "Pro",          "lc": 1000,  "inr": 160},
    "elite":        {"name": "Elite",        "lc": 1500,  "inr": 220},
    "ultra":        {"name": "Ultra",        "lc": 2500,  "inr": 340},
    "mega":         {"name": "Mega 🐋",      "lc": 5000,  "inr": 600},
    "jumbo_week":   {"name": "Jumbo Week 🔥","lc": 5050,  "inr": 500},
    "custom":       {"name": "Custom",       "lc": None,  "inr": None},
}


# ==================== SCHEMAS ====================
class CreateOrderRequest(BaseModel):
    pack_id: str
    custom_lc: int = None


class VerifyPaymentRequest(BaseModel):
    order_id: str
    payment_id: str
    payment_signature: str


# ==================== ROUTES ====================
@router.get("/packs")
async def get_packs():
    """Get all available LC packs"""
    return {"packs": LC_PACKS}


@router.post("/create-order")
async def create_order(
    body: CreateOrderRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create Cashfree order for LC purchase"""

    if body.pack_id not in LC_PACKS:
        raise HTTPException(status_code=400, detail="Invalid pack")

    pack = LC_PACKS[body.pack_id]

    if body.pack_id == "custom":
        if not body.custom_lc or body.custom_lc < 10:
            raise HTTPException(status_code=400, detail="Minimum 10 LC for custom")
        lc_amount  = body.custom_lc
        inr_amount = round(lc_amount * 0.25, 2)
        pack_name  = f"Custom {lc_amount} LC"
    else:
        lc_amount  = pack["lc"]
        inr_amount = pack["inr"]
        pack_name  = pack["name"]

    cf_order_id = f"LUXUR_{current_user['id']}_{int(time.time())}"

    payload = {
        "order_id":       cf_order_id,
        "order_amount":   float(inr_amount),
        "order_currency": "INR",
        "order_note":     f"{pack_name} - LuxurAI LC Purchase",
        "customer_details": {
            "customer_id":    str(current_user["id"]),
            "customer_name":  current_user["name"],
            "customer_email": current_user["gmail"],
            "customer_phone": "9999999999"
        },
        "order_meta": {
            "return_url": f"https://luxarai.in/payment-success?order_id={{order_id}}&order_token={{order_token}}"
        },
        "order_tags": {
            "user_id":   str(current_user["id"]),
            "pack_id":   body.pack_id,
            "lc_amount": str(lc_amount)
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{CASHFREE_BASE_URL}/orders",
                headers=CASHFREE_HEADERS,
                json=payload,
                timeout=10.0
            )
            resp.raise_for_status()
            cf_order = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Cashfree order creation failed: {e.response.text}")
        raise HTTPException(status_code=500, detail="Payment gateway error")
    except Exception as e:
        logger.error(f"Cashfree error: {e}")
        raise HTTPException(status_code=500, detail="Payment gateway error")

    db.create_payment(
        user_id=current_user["id"],
        order_id=cf_order_id,
        pack_name=pack_name,
        lc_amount=lc_amount,
        inr_amount=inr_amount
    )

    logger.info(f"📦 Order created: {cf_order_id} | Rs.{inr_amount} | {pack_name}")

    return {
        "order_id":            cf_order_id,
        "payment_session_id":  cf_order.get("payment_session_id"),
        "amount":              inr_amount,
        "currency":            "INR",
        "pack_name":           pack_name,
        "lc_amount":           lc_amount,
        "user_name":           current_user["name"],
        "user_email":          current_user["gmail"]
    }


@router.post("/verify")
async def verify_payment(
    body: VerifyPaymentRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Verify Cashfree payment signature
    Credits LC to user on success
    """

    # Cashfree signature verification
    raw_signature = f"{body.order_id}{body.payment_id}"
    expected_signature = hmac.new(
        CASHFREE_SECRET_KEY.encode("utf-8"),
        raw_signature.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, body.payment_signature):
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    # Double-check with Cashfree API
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{CASHFREE_BASE_URL}/orders/{body.order_id}/payments/{body.payment_id}",
                headers=CASHFREE_HEADERS,
                timeout=10.0
            )
            resp.raise_for_status()
            payment_data = resp.json()
    except Exception as e:
        logger.error(f"Cashfree payment verification API error: {e}")
        raise HTTPException(status_code=500, detail="Could not verify payment with gateway")

    cf_status = payment_data.get("payment_status", "")
    if cf_status != "SUCCESS":
        raise HTTPException(status_code=400, detail=f"Payment not successful. Status: {cf_status}")

    payment = db.complete_payment(body.order_id, body.payment_id)
    if not payment:
        raise HTTPException(status_code=400, detail="Payment not found or already processed")

    if payment["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Payment mismatch")

    db.credit_lc(
        user_id=current_user["id"],
        amount=payment["lc_amount"],
        type="purchase",
        description=f"Purchased {payment['pack_name']}",
        ref_id=body.payment_id
    )

    new_balance = db.get_lc_balance(current_user["id"])

    logger.info(
        f"💰 Payment success: User {current_user['id']} | "
        f"+{payment['lc_amount']} LC | Rs.{payment['inr_amount']} | Order: {body.order_id}"
    )

    return {
        "success":     True,
        "lc_credited": payment["lc_amount"],
        "new_balance": new_balance,
        "pack_name":   payment["pack_name"]
    }


@router.post("/webhook")
async def cashfree_webhook(request: Request):
    """
    Cashfree Webhook - Auto credit LC on successful payment
    Set this URL in Cashfree Dashboard > Webhooks:
    https://luxarai.in/api/payment/webhook
    """
    body_bytes = await request.body()

    cf_signature = request.headers.get("x-webhook-signature")
    cf_timestamp = request.headers.get("x-webhook-timestamp")

    if not cf_signature or not cf_timestamp:
        raise HTTPException(status_code=400, detail="Missing webhook headers")

    raw = cf_timestamp + body_bytes.decode("utf-8")
    expected_hex = hmac.new(
        CASHFREE_SECRET_KEY.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    expected_b64 = base64.b64encode(bytes.fromhex(expected_hex)).decode("utf-8")

    if not hmac.compare_digest(expected_b64, cf_signature):
        logger.warning("Webhook signature mismatch")
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event      = json.loads(body_bytes)
    event_type = event.get("type", "")

    if event_type == "PAYMENT_SUCCESS_WEBHOOK":
        data       = event["data"]
        order_id   = data["order"]["order_id"]
        payment_id = data["payment"]["cf_payment_id"]
        cf_status  = data["payment"]["payment_status"]

        if cf_status == "SUCCESS":
            payment = db.complete_payment(order_id, str(payment_id))
            if payment:
                db.credit_lc(
                    user_id=payment["user_id"],
                    amount=payment["lc_amount"],
                    type="purchase",
                    description=f"Purchased {payment['pack_name']} (webhook)",
                    ref_id=str(payment_id)
                )
                logger.info(f"Webhook credited: Order {order_id} | +{payment['lc_amount']} LC")

    return {"status": "ok"}


@router.get("/history")
async def payment_history(current_user: dict = Depends(get_current_user)):
    """Get payment history for current user"""
    conn = db.get_conn()
    rows = conn.execute("""
        SELECT * FROM payments
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 20
    """, (current_user["id"],)).fetchall()
    conn.close()

    return {"payments": [dict(r) for r in rows]}
