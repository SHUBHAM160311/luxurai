"""
LuxurAI — auth.py
─────────────────────────────────────────────────────────────────
Authentication backend: Google OAuth + Passwordless Magic Link
─────────────────────────────────────────────────────────────────
"""

import os, secrets, hashlib, asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SECRET_KEY           = os.getenv("SECRET_KEY", "dev-secret-change-in-production-32chars!")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
RESEND_API_KEY       = os.getenv("RESEND_API_KEY", "")
EMAIL_FROM           = os.getenv("EMAIL_FROM", "noreply@luxurai.in")
BASE_URL             = os.getenv("BASE_URL", "https://luxurai.in")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "https://luxurai.in")

ALGORITHM          = "HS256"
SESSION_DAYS       = 30
MAGIC_LINK_MINUTES = 10
SIGNUP_BONUS_LC    = 45
RATE_LIMIT_HOURLY  = 3
RATE_LIMIT_DAILY   = 10

DB_PATH = os.getenv("DB_PATH", "luxurai.db")


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE NOT NULL,
                provider    TEXT NOT NULL DEFAULT 'email',
                google_id   TEXT,
                is_blocked  INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                last_login  TEXT
            );

            CREATE TABLE IF NOT EXISTS wallets (
                user_id     TEXT PRIMARY KEY REFERENCES users(id),
                balance_lc  REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS wallet_ledger (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                delta_lc    REAL NOT NULL,
                reason      TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash  TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                expires_at  TEXT NOT NULL,
                used        INTEGER NOT NULL DEFAULT 0,
                ip          TEXT,
                user_agent  TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS magic_link_rate (
                email       TEXT NOT NULL,
                sent_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                ip          TEXT,
                user_agent  TEXT,
                created_at  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                revoked     INTEGER NOT NULL DEFAULT 0
            );
        """)
        await db.commit()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def uuid4() -> str:
    return secrets.token_urlsafe(16)

def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()

def make_jwt(payload: dict, expires_delta: timedelta) -> str:
    data = {**payload, "exp": datetime.now(timezone.utc) + expires_delta}
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def decode_jwt(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

def get_session_token(request: Request) -> Optional[str]:
    return request.cookies.get("luxurai_session")


# ─────────────────────────────────────────────
# User helpers
# ─────────────────────────────────────────────
async def get_user_by_email(db, email: str) -> Optional[dict]:
    async with db.execute("SELECT * FROM users WHERE email = ?", (email,)) as cur:
        row = await cur.fetchone()
        return dict(row) if row else None

async def get_user_by_id(db, user_id: str) -> Optional[dict]:
    async with db.execute("SELECT * FROM users WHERE id = ?", (user_id,)) as cur:
        row = await cur.fetchone()
        return dict(row) if row else None

async def get_wallet(db, user_id: str) -> float:
    async with db.execute("SELECT balance_lc FROM wallets WHERE user_id = ?", (user_id,)) as cur:
        row = await cur.fetchone()
        return row["balance_lc"] if row else 0.0

async def create_user(db, email: str, provider: str = "email", google_id: str = None) -> dict:
    user_id = uuid4()
    ts = now_utc()
    await db.execute(
        "INSERT INTO users (id, email, provider, google_id, created_at) VALUES (?,?,?,?,?)",
        (user_id, email, provider, google_id, ts)
    )
    await db.execute(
        "INSERT INTO wallets (user_id, balance_lc) VALUES (?,?)",
        (user_id, SIGNUP_BONUS_LC)
    )
    await db.execute(
        "INSERT INTO wallet_ledger (id, user_id, delta_lc, reason, created_at) VALUES (?,?,?,?,?)",
        (uuid4(), user_id, SIGNUP_BONUS_LC, "signup_bonus", ts)
    )
    await db.commit()
    return {"id": user_id, "email": email, "lc_balance": SIGNUP_BONUS_LC}

async def update_last_login(db, user_id: str):
    await db.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_utc(), user_id))
    await db.commit()

async def create_session(db, user_id: str, request: Request) -> str:
    session_id = uuid4()
    ts = now_utc()
    await db.execute(
        "INSERT INTO sessions (session_id, user_id, ip, user_agent, created_at, last_seen) VALUES (?,?,?,?,?,?)",
        (session_id, user_id, request.client.host, request.headers.get("user-agent", ""), ts, ts)
    )
    await db.commit()
    return session_id


# ─────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────
async def check_magic_link_rate(db, email: str) -> bool:
    now = datetime.now(timezone.utc)
    one_hour_ago = (now - timedelta(hours=1)).isoformat()
    one_day_ago  = (now - timedelta(days=1)).isoformat()

    async with db.execute(
        "SELECT COUNT(*) as c FROM magic_link_rate WHERE email = ? AND sent_at > ?",
        (email, one_hour_ago)
    ) as cur:
        row = await cur.fetchone()
        if row["c"] >= RATE_LIMIT_HOURLY:
            return False

    async with db.execute(
        "SELECT COUNT(*) as c FROM magic_link_rate WHERE email = ? AND sent_at > ?",
        (email, one_day_ago)
    ) as cur:
        row = await cur.fetchone()
        if row["c"] >= RATE_LIMIT_DAILY:
            return False

    return True

async def record_magic_link_send(db, email: str):
    await db.execute(
        "INSERT INTO magic_link_rate (email, sent_at) VALUES (?,?)",
        (email, now_utc())
    )
    await db.commit()


# ─────────────────────────────────────────────
# Email via Resend
# ─────────────────────────────────────────────
async def send_magic_email(email: str, link: str):
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ margin:0; padding:0; background:#080808; font-family:'Helvetica Neue',Arial,sans-serif; }}
  .wrap {{ max-width:480px; margin:60px auto; padding:0 24px; }}
  .logo {{ font-size:13px; letter-spacing:0.3em; color:#C9A84C; text-transform:uppercase; margin-bottom:48px; }}
  .logo span {{ color:#888070; font-weight:300; }}
  .line {{ width:40px; height:1px; background:#C9A84C; opacity:0.4; margin-bottom:32px; }}
  h1 {{ font-size:28px; font-weight:300; color:#F8F4ED; line-height:1.3; margin:0 0 20px; letter-spacing:-0.01em; }}
  p {{ font-size:13px; color:#888070; line-height:1.8; margin:0 0 36px; letter-spacing:0.02em; }}
  .btn {{ display:inline-block; background:#C9A84C; color:#080808; text-decoration:none; padding:14px 40px; font-size:11px; font-weight:500; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:40px; }}
  .note {{ font-size:11px; color:#555; letter-spacing:0.08em; line-height:1.7; border-top:1px solid #1E1E1E; padding-top:24px; }}
  .footer {{ font-size:10px; color:#333; letter-spacing:0.15em; margin-top:48px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="logo">Luxur<span>AI</span></div>
  <div class="line"></div>
  <h1>Here's your<br>secure sign‑in link.</h1>
  <p>Click below to continue. No password required.</p>
  <a href="{link}" class="btn">Access LuxurAI</a>
  <div class="note">
    This link expires in {MAGIC_LINK_MINUTES} minutes and can only be used once.<br>
    If you didn't request this, you can safely ignore this email.
  </div>
  <div class="footer">— LuxurAI &nbsp;·&nbsp; Private AI Generation &nbsp;·&nbsp; luxurai.in</div>
</div>
</body>
</html>
"""
    if not RESEND_API_KEY:
        print(f"\n[DEV] Magic link for {email}:\n  {link}\n")
        return

    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": f"LuxurAI <{EMAIL_FROM}>",
                "to": [email],
                "subject": "Your LuxurAI sign‑in link",
                "html": html_body,
            }
        )


# ─────────────────────────────────────────────
# FastAPI app
# NOTE: Routes are WITHOUT /api/auth prefix
# because main.py mounts this app at /api/auth
# So /health here becomes /api/auth/health
# ─────────────────────────────────────────────
app = FastAPI(title="LuxurAI Auth", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await init_db()


class MagicLinkRequest(BaseModel):
    email: EmailStr
    resend: bool = False


# ── Health ────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "LuxurAI Auth",
        "google":  bool(GOOGLE_CLIENT_ID),
        "email":   bool(RESEND_API_KEY),
    }


# ── /me ───────────────────────────────────────
@app.get("/me")
async def get_me(request: Request):
    token = get_session_token(request)
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = decode_jwt(token)
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid session")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        user = await get_user_by_id(db, user_id)
        if not user or user["is_blocked"]:
            raise HTTPException(401, "User not found or blocked")
        lc = await get_wallet(db, user_id)
        await db.execute(
            "UPDATE sessions SET last_seen = ? WHERE user_id = ? AND revoked = 0",
            (now_utc(), user_id)
        )
        await db.commit()

    return {
        "id":         user["id"],
        "email":      user["email"],
        "provider":   user["provider"],
        "balance_lc": lc,
        "is_new":     False,
    }


# ── Session ───────────────────────────────────
@app.get("/session")
async def get_session(request: Request):
    token = get_session_token(request)
    if not token:
        raise HTTPException(401, "No session")
    try:
        payload = decode_jwt(token)
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid session")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        user = await get_user_by_id(db, user_id)
        if not user or user["is_blocked"]:
            raise HTTPException(401, "User not found or blocked")
        lc = await get_wallet(db, user_id)
        await db.execute(
            "UPDATE sessions SET last_seen = ? WHERE user_id = ? AND revoked = 0",
            (now_utc(), user_id)
        )
        await db.commit()

    return {
        "user": {
            "id":         user["id"],
            "email":      user["email"],
            "provider":   user["provider"],
            "lc_balance": lc,
        }
    }


# ── Magic Link: send ──────────────────────────
@app.post("/magic-link")
async def send_magic_link(payload: MagicLinkRequest, request: Request):
    email = payload.email.lower().strip()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        allowed = await check_magic_link_rate(db, email)
        if not allowed:
            return JSONResponse({"error": "rate_limit"}, status_code=429)

        await record_magic_link_send(db, email)

        raw_token  = secrets.token_urlsafe(48)
        token_hash = hash_token(raw_token)
        expires    = (datetime.now(timezone.utc) + timedelta(minutes=MAGIC_LINK_MINUTES)).isoformat()

        user = await get_user_by_email(db, email)
        if not user:
            user_id = uuid4()
            ts = now_utc()
            await db.execute(
                "INSERT INTO users (id, email, provider, created_at) VALUES (?,?,?,?)",
                (user_id, email, "email", ts)
            )
            await db.commit()
        else:
            user_id = user["id"]

        await db.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, expires_at, ip, user_agent, created_at) VALUES (?,?,?,?,?,?)",
            (token_hash, user_id, expires, request.client.host,
             request.headers.get("user-agent", ""), now_utc())
        )
        await db.commit()

    link = f"{BASE_URL}/api/auth/verify?token={raw_token}"
    await send_magic_email(email, link)
    return {"success": True}


# ── Magic Link: verify ────────────────────────
@app.get("/verify")
async def verify_magic_link(token: str, request: Request, response: Response):
    token_hash = hash_token(token)
    now = now_utc()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT * FROM auth_tokens WHERE token_hash = ?", (token_hash,)
        ) as cur:
            row = await cur.fetchone()

        if not row:
            raise HTTPException(400, "Invalid or expired link")
        if row["used"]:
            raise HTTPException(400, "This link has already been used")
        if row["expires_at"] < now:
            raise HTTPException(400, "Link expired · Request a new one")

        await db.execute(
            "UPDATE auth_tokens SET used = 1 WHERE token_hash = ?", (token_hash,)
        )

        user_id = row["user_id"]
        user    = await get_user_by_id(db, user_id)

        async with db.execute(
            "SELECT balance_lc FROM wallets WHERE user_id = ?", (user_id,)
        ) as cur:
            wallet_row = await cur.fetchone()

        is_new = wallet_row is None
        if is_new:
            await db.execute(
                "INSERT INTO wallets (user_id, balance_lc) VALUES (?,?)",
                (user_id, SIGNUP_BONUS_LC)
            )
            await db.execute(
                "INSERT INTO wallet_ledger (id, user_id, delta_lc, reason, created_at) VALUES (?,?,?,?,?)",
                (uuid4(), user_id, SIGNUP_BONUS_LC, "signup_bonus", now)
            )
            lc_balance = SIGNUP_BONUS_LC
        else:
            lc_balance = wallet_row["balance_lc"]

        await update_last_login(db, user_id)
        session_id = await create_session(db, user_id, request)
        await db.commit()

    jwt_token = make_jwt({"sub": user_id}, timedelta(days=SESSION_DAYS))
    redirect_url = (
        f"{FRONTEND_URL}/pages/dashboard.html?welcome=1&new=1"
        if is_new
        else f"{FRONTEND_URL}/pages/dashboard.html"
    )
    resp = RedirectResponse(url=redirect_url, status_code=302)
    resp.set_cookie(
        key      = "luxurai_session",
        value    = jwt_token,
        httponly = True,
        secure   = BASE_URL.startswith("https"),
        samesite = "lax",
        max_age  = SESSION_DAYS * 86400
    )
    return resp


# ── Google OAuth: initiate ────────────────────
@app.get("/google")
async def google_auth_init(request: Request):
    if not GOOGLE_CLIENT_ID:
        return RedirectResponse(url=f"{FRONTEND_URL}?auth_error=google_not_configured")

    redirect_uri = f"{BASE_URL}/api/auth/google/callback"
    state        = secrets.token_urlsafe(16)

    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
        f"&state={state}"
        "&access_type=offline"
        "&prompt=select_account"
    )
    return RedirectResponse(url=url)


# ── Google OAuth: callback ────────────────────
@app.get("/google/callback")
async def google_auth_callback(code: str, state: str, request: Request):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(503, "Google OAuth not configured")

    redirect_uri = f"{BASE_URL}/api/auth/google/callback"

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            }
        )
        token_data = token_res.json()

        if "error" in token_data:
            raise HTTPException(400, f"Google error: {token_data.get('error_description', 'OAuth failed')}")

        access_token = token_data["access_token"]

        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        userinfo = userinfo_res.json()

    email     = userinfo.get("email", "").lower()
    google_id = userinfo.get("id", "")

    if not email:
        raise HTTPException(400, "Could not retrieve email from Google")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        user   = await get_user_by_email(db, email)
        is_new = user is None

        if is_new:
            user    = await create_user(db, email, provider="google", google_id=google_id)
            user_id = user["id"]
        else:
            user_id = user["id"]
            if not user.get("google_id"):
                await db.execute(
                    "UPDATE users SET google_id = ?, provider = 'google' WHERE id = ?",
                    (google_id, user_id)
                )
                await db.commit()

        await update_last_login(db, user_id)
        await create_session(db, user_id, request)
        await db.commit()

    jwt_token = make_jwt({"sub": user_id}, timedelta(days=SESSION_DAYS))
    redirect_url = (
        f"{FRONTEND_URL}/pages/dashboard.html?welcome=1&new=1"
        if is_new
        else f"{FRONTEND_URL}/pages/dashboard.html"
    )
    resp = RedirectResponse(url=redirect_url, status_code=302)
    resp.set_cookie(
        key      = "luxurai_session",
        value    = jwt_token,
        httponly = True,
        secure   = BASE_URL.startswith("https"),
        samesite = "lax",
        max_age  = SESSION_DAYS * 86400
    )
    return resp


# ── Sign out ──────────────────────────────────
@app.post("/signout")
async def signout(request: Request, response: Response):
    token = get_session_token(request)
    if token:
        try:
            payload = decode_jwt(token)
            user_id = payload.get("sub")
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE sessions SET revoked = 1 WHERE user_id = ?", (user_id,)
                )
                await db.commit()
        except JWTError:
            pass
    response.delete_cookie("luxurai_session")
    return {"success": True}


# ── Sign out all ──────────────────────────────
@app.post("/signout-all")
async def signout_all(request: Request, response: Response):
    token = get_session_token(request)
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = decode_jwt(token)
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid session")

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET revoked = 1 WHERE user_id = ?", (user_id,)
        )
        await db.commit()
    response.delete_cookie("luxurai_session")
    return {"success": True}


# ── Sessions list ─────────────────────────────
@app.get("/sessions")
async def list_sessions(request: Request):
    token = get_session_token(request)
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = decode_jwt(token)
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid session")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT session_id, ip, user_agent, created_at, last_seen
               FROM sessions WHERE user_id = ? AND revoked = 0
               ORDER BY last_seen DESC""",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()

    return {"sessions": [dict(r) for r in rows]}


# ── Wallet ────────────────────────────────────
@app.get("/wallet")
async def get_wallet_balance(request: Request):
    token = get_session_token(request)
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = decode_jwt(token)
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(401, "Invalid session")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        lc = await get_wallet(db, user_id)
        async with db.execute(
            """SELECT delta_lc, reason, created_at
               FROM wallet_ledger WHERE user_id = ?
               ORDER BY created_at DESC LIMIT 20""",
            (user_id,)
        ) as cur:
            ledger = [dict(r) for r in await cur.fetchall()]

    return {"balance_lc": lc, "ledger": ledger}


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("auth:app", host="0.0.0.0", port=8000, reload=True)
