"""
LuxurAI — core/security.py
─────────────────────────────────────────────────────────────────
All JWT and auth helpers in one place.

Previously scattered across auth.py, payment.py, ux.py, jobs.py.
Now every file imports from here.

Usage:
    from core.security import make_jwt, decode_jwt, get_current_user

    # In a route:
    user_id = await get_current_user(request)

    # Create a session token:
    token = make_jwt({"sub": user_id}, days=30)
─────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Request, HTTPException
from jose import JWTError, jwt

from core.config import cfg

logger = logging.getLogger("luxurai.security")


# ─────────────────────────────────────────────
# JWT
# ─────────────────────────────────────────────
def make_jwt(payload: dict, days: int = None, minutes: int = None) -> str:
    """
    Create a signed JWT.

    Examples:
        make_jwt({"sub": user_id}, days=30)          # session token
        make_jwt({"sub": user_id, "type": "magic"}, minutes=10)  # magic link
    """
    if days:
        expires = datetime.now(timezone.utc) + timedelta(days=days)
    elif minutes:
        expires = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    else:
        expires = datetime.now(timezone.utc) + timedelta(days=cfg.SESSION_DAYS)

    data = {**payload, "exp": expires}
    return jwt.encode(data, cfg.JWT_SECRET, algorithm=cfg.ALGORITHM)


def decode_jwt(token: str) -> dict:
    """
    Decode and verify a JWT.
    Raises JWTError if invalid or expired.
    """
    return jwt.decode(token, cfg.JWT_SECRET, algorithms=[cfg.ALGORITHM])


# ─────────────────────────────────────────────
# Request helpers
# ─────────────────────────────────────────────
def get_token_from_request(request: Request) -> Optional[str]:
    """
    Extract JWT from:
    1. Cookie: luxurai_session
    2. Header: Authorization: Bearer <token>
    Returns None if not found.
    """
    token = request.cookies.get("luxurai_session")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
    return token or None


async def get_current_user(request: Request) -> str:
    """
    FastAPI dependency — returns user_id from JWT.

    Usage in any route:
        user_id = await get_current_user(request)

    Raises 401 if:
        - No token found
        - Token is invalid or expired
        - Token has no 'sub' field
    """
    token = get_token_from_request(request)

    if not token:
        raise HTTPException(401, "Not authenticated. Please log in.")

    try:
        payload = decode_jwt(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token payload.")
        return user_id

    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        raise HTTPException(401, "Session expired. Please log in again.")


async def get_current_user_optional(request: Request) -> Optional[str]:
    """
    Same as get_current_user but returns None instead of raising 401.
    Use for routes that work for both logged-in and anonymous users.
    """
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


# ─────────────────────────────────────────────
# Cookie helpers
# ─────────────────────────────────────────────
def set_session_cookie(response, token: str):
    """Set the JWT as an HTTP-only secure cookie."""
    response.set_cookie(
        key      = "luxurai_session",
        value    = token,
        httponly = True,
        secure   = cfg.is_production,   # HTTPS only in prod
        samesite = "lax",
        max_age  = cfg.SESSION_DAYS * 86400,
    )


def clear_session_cookie(response):
    """Delete the session cookie (logout)."""
    response.delete_cookie("luxurai_session")
