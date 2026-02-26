"""
LuxurAI Backend - Main FastAPI Server
Phase 2: Auth + LC Wallet + Queue System
"""

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
from pathlib import Path

from Database import init_db
from Auth import router as auth_router
from Wallet import router as wallet_router
from Generate import router as generate_router
from Payment import router as payment_router
from queue_manager import QueueManager

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== APP ====================
app = FastAPI(
    title="LuxurAI API",
    description="LuxurAI Image Generation Backend",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== QUEUE MANAGER (GLOBAL) ====================
queue_manager = QueueManager()

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup():
    logger.info("LuxurAI Server Starting...")
    init_db()
    await queue_manager.start()
    logger.info("LuxurAI Server Ready!")

@app.on_event("shutdown")
async def shutdown():
    await queue_manager.stop()
    logger.info("LuxurAI Server Stopped")

# ==================== ROUTES ====================
app.include_router(auth_router,     prefix="/api/auth",     tags=["Auth"])
app.include_router(wallet_router,   prefix="/api/wallet",   tags=["Wallet"])
app.include_router(generate_router, prefix="/api/generate", tags=["Generate"])
app.include_router(payment_router,  prefix="/api/payment",  tags=["Payment"])

# ==================== HEALTH ====================
@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "queue_size": queue_manager.total_queued(),
        "batches_active": queue_manager.active_batches()
    }

# ==================== RUN ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=1
    )
