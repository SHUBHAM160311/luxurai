"""
LuxurAI Auth Routes
Gmail OAuth via Google Identity
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx
import jwt
import logging
import os
from datetime import datetime, timedelta

import database as db

logger = logging.getLogger(__name__)
router = APIRouter()

# ==================== CONFIG ====================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")
JWT_SECRET = os.getenv("JWT_SECRET", "luxurai-secret-change-in-production")
JWT_EXPIRE_DAYS = 30


# ==================== SCHEMAS ====================
class GoogleTokenRequest(BaseModel):
    credential: str  # Google ID token from frontend


class AuthResponse(BaseModel):
    token: str
    user: dict
    is_new_user: bool


# ==================== HELPERS ====================
def create_jwt(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_jwt(token: str) -> int:
    """Returns user_id or raises HTTPException"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def verify_google_token(credential: str) -> dict:
    """Verify Google ID token and return user info"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}"
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Google token")

        data = resp.json()

        if data.get("aud") != GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Token audience mismatch")

        return {
            "google_id": data["sub"],
            "gmail": data["email"],
            "name": data.get("name", ""),
            "picture": data.get("picture", "")
        }


# ==================== DEPENDENCY ====================
async def get_current_user(request: Request) -> dict:
    """FastAPI dependency - extracts and validates JWT"""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.removeprefix("Bearer ")
    user_id = verify_jwt(token)

    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user["is_banned"]:
        raise HTTPException(status_code=403, detail="Account suspended")

    db.update_last_seen(user_id)
    return user


# ==================== ROUTES ====================
@router.post("/google", response_model=AuthResponse)
async def google_auth(body: GoogleTokenRequest):
    """
    Verify Google credential from frontend
    Creates account if new user (with 45 LC bonus!)
    Returns JWT token
    """
    google_info = await verify_google_token(body.credential)

    existing = db.get_user_by_google_id(google_info["google_id"])
    is_new = existing is None

    if is_new:
        user = db.create_user(
            google_id=google_info["google_id"],
            gmail=google_info["gmail"],
            name=google_info["name"],
            picture=google_info["picture"]
        )
        logger.info(f"🎉 New user registered: {google_info['gmail']}")
    else:
        user = existing
        db.update_last_seen(user["id"])
        logger.info(f"👋 User login: {google_info['gmail']}")

    token = create_jwt(user["id"])

    return AuthResponse(
        token=token,
        user={
            "id": user["id"],
            "name": user["name"],
            "gmail": user["gmail"],
            "picture": user["picture"],
            "lc_balance": user["lc_balance"],
        },
        is_new_user=is_new
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "gmail": current_user["gmail"],
        "picture": current_user["picture"],
        "lc_balance": current_user["lc_balance"],
        "total_generated": current_user["total_generated"],
        "member_since": current_user["created_at"]
    }


@router.post("/logout")
async def logout():
    """Logout (client should delete token)"""
    return {"message": "Logged out successfully"}