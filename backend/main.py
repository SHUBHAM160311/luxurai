"""
LuxurAI — main.py
─────────────────────────────────────────────────────────────────
Central entry point. All routers mount here.
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────
load_dotenv("CashFree.env")
load_dotenv()

# ── Config ────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://luxurai.in")
DB_PATH      = os.getenv("DB_PATH", "luxurai.db")
ENV          = os.getenv("ENV", "development")

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("luxurai.main")

# ── Paths ─────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent   # backend/
ROOT_DIR     = BASE_DIR.parent                   # project root/
FRONTEND_DIR = ROOT_DIR / "frontend"
PAGES_DIR    = FRONTEND_DIR / "pages"

# Ensure backend/ modules importable
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 LuxurAI starting [{ENV}]")

    from auth     import init_db            as init_auth_tables
    from wallet   import init_wallet_tables
    from jobs     import init_job_tables
    from ux       import init_ux_tables
    from referral import init_referral_tables
    from loyalty  import init_loyalty_tables
    from admin    import init_admin_tables

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

    # Start RunPod worker if keys set
    import asyncio
    if os.getenv("RUNPOD_API_KEY") and os.getenv("RUNPOD_ENDPOINT_ID"):
        try:
            sys.path.insert(0, str(ROOT_DIR))
            from worker.generator import start_worker
            asyncio.create_task(start_worker())
            logger.info("✓ RunPod worker started")
        except Exception as e:
            logger.warning(f"Worker could not start: {e}")
    else:
        logger.warning("⚠️  RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set — worker not started")

    yield

    logger.info("LuxurAI shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "LuxurAI API",
    description = "Backend for LuxurAI — AI image generation platform",
    version     = "1.0.0",
    docs_url    = "/docs"  if ENV != "production" else None,
    redoc_url   = "/redoc" if ENV != "production" else None,
    lifespan    = lifespan,
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [FRONTEND_URL, "https://luxurai.in", "https://www.luxurai.in"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────
# Mount Routers
# ─────────────────────────────────────────────
from auth     import app as auth_app
app.mount("/api/auth", auth_app)

from payment  import router as payment_router
app.include_router(payment_router)

from ux       import router as ux_router
app.include_router(ux_router)

from referral import router as referral_router
app.include_router(referral_router)

from loyalty  import router as loyalty_router
app.include_router(loyalty_router)

from admin    import router as admin_router
app.include_router(admin_router)

try:
    from jobs import router as jobs_router
    if jobs_router:
        app.include_router(jobs_router, prefix="/api")
except Exception as e:
    logger.warning(f"Jobs router not loaded: {e}")

# ─────────────────────────────────────────────
# Static Files
# ─────────────────────────────────────────────
_public = FRONTEND_DIR / "public"
_styles = FRONTEND_DIR / "styles"

if _public.is_dir():
    app.mount("/public", StaticFiles(directory=str(_public)), name="public")

if _styles.is_dir():
    app.mount("/styles", StaticFiles(directory=str(_styles)), name="styles")

# ─────────────────────────────────────────────
# Frontend HTML Routes
# ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(PAGES_DIR / "index.html"))

@app.get("/pages/{page_name}", include_in_schema=False)
async def serve_page(page_name: str):
    filepath = PAGES_DIR / page_name
    if filepath.is_file():
        return FileResponse(str(filepath))
    return FileResponse(str(PAGES_DIR / "index.html"))

# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "app": "LuxurAI", "version": "1.0.0", "env": ENV}

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
        port    = int(os.getenv("PORT", 8080)),
        reload  = ENV != "production",
        workers = 1,
    )
