"""
LuxurAI — main.py
─────────────────────────────────────────────────────────────────
Central entry point. All routers mount here.

Start server:
    uvicorn main:app --reload --port 8000

File map:
    auth.py     → /api/auth/*       (login, google, magic link)
    payment.py  → /api/payment/*    (cashfree, stripe, packs)
    ux.py       → /api/ux/*         (dashboard, generate, history)
    wallet.py   → service only      (no direct routes)
    jobs.py     → service only      (no direct routes)
─────────────────────────────────────────────────────────────────
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ── Load env files ────────────────────────────
load_dotenv("CashFree.env")
load_dotenv()

# ── Config ────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://luxurai.in")
DB_PATH      = os.getenv("DB_PATH", "luxurai.db")
ENV          = os.getenv("ENV", "development")   # "production" in prod

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("luxurai.main")


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup → initializes all DB tables.
    Runs on shutdown → cleanup if needed.
    """
    logger.info(f"🚀 LuxurAI starting [{ENV}]")

    # Init all tables in correct order
    # (users first — others have foreign keys to it)
    from backend.auth     import init_db           as init_auth_tables
    from backend.wallet   import init_wallet_tables
    from backend.jobs     import init_job_tables
    from backend.ux       import init_ux_tables
    from backend.referral import init_referral_tables
    from backend.loyalty  import init_loyalty_tables
    from backend.admin    import init_admin_tables

    await init_auth_tables()
    logger.info("✓ Auth tables ready")

    await init_wallet_tables()
    logger.info("✓ Wallet tables ready")

    await init_job_tables()
    logger.info("✓ Job tables ready")

    await init_ux_tables()
    logger.info("✓ UX tables ready")

    await init_referral_tables()
    logger.info("✓ Referral tables ready")

    await init_loyalty_tables()
    logger.info("✓ Loyalty tables ready")

    await init_admin_tables()
    logger.info("✓ Admin + Promo tables ready")

    # Payment orders table
    import aiosqlite
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
    logger.info("✓ Payment tables ready")

    logger.info("✅ All tables initialized. LuxurAI is live.")

    # Start image generation worker in background
    import asyncio
    if os.getenv("RUNPOD_API_KEY") and os.getenv("RUNPOD_ENDPOINT_ID"):
        try:
            from backend.worker.generator import start_worker
            worker_task = asyncio.create_task(start_worker())
            logger.info("✓ RunPod worker started")
        except Exception as e:
            logger.warning(f"Worker could not start: {e}")
    else:
        logger.warning("⚠️  RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set — worker not started")

    yield  # App runs here

    logger.info("LuxurAI shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "LuxurAI API",
    description = "Backend for LuxurAI — AI image generation platform",
    version     = "1.0.0",
    docs_url    = "/docs"   if ENV != "production" else None,  # hide in prod
    redoc_url   = "/redoc"  if ENV != "production" else None,
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins      = [FRONTEND_URL, "https://luxurai.in", "https://www.luxurai.in"],
    allow_credentials  = True,
    allow_methods      = ["*"],
    allow_headers      = ["*"],
)


# ─────────────────────────────────────────────
# Mount Routers
# ─────────────────────────────────────────────

# Auth routes → /api/auth/*
from backend.auth import app as auth_app
app.mount("/api/auth", auth_app)

# Payment routes → /api/payment/*
from backend.payment import router as payment_router
app.include_router(payment_router)

# Dashboard / UX routes → /api/ux/*
from backend.ux import router as ux_router
app.include_router(ux_router)

# Referral routes → /api/referral/*
from backend.referral import router as referral_router
app.include_router(referral_router)

# Loyalty routes → /api/loyalty/*
from backend.loyalty import router as loyalty_router
app.include_router(loyalty_router)

# Jobs routes → /api/jobs/*
try:
    from backend.jobs import router as jobs_router
    if jobs_router:
        app.include_router(jobs_router, prefix="/api")
except Exception as e:
    logger.warning(f"Jobs router not loaded: {e}")

# Admin routes → /api/admin/*
from backend.admin import router as admin_router
app.include_router(admin_router)


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    """Quick ping — load balancer / uptime monitor uses this."""
    return {
        "status":  "ok",
        "app":     "LuxurAI",
        "version": "1.0.0",
        "env":     ENV,
    }


@app.get("/", tags=["system"])
async def root():
    return {"message": "LuxurAI API is running. Docs at /docs"}


# ─────────────────────────────────────────────
# Global Error Handler
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error. Our team has been notified."}
    )


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = ENV != "production",
        workers = 1 if ENV != "production" else 4,
    )
