"""
LuxurAI — core/database.py
─────────────────────────────────────────────────────────────────
Single place for:
  - DB connection helper
  - ALL table CREATE statements
  - One init_all_tables() call on startup

Every file used to have its own init — now it's all here.

Usage:
    from core.database import get_db, init_all_tables

    # In main.py startup:
    await init_all_tables()

    # In any route:
    async with get_db() as db:
        await db.execute(...)
─────────────────────────────────────────────────────────────────
"""

import logging
import aiosqlite
from contextlib import asynccontextmanager
from core.config import cfg

logger = logging.getLogger("luxurai.database")


# ─────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────
@asynccontextmanager
async def get_db():
    """
    Use instead of aiosqlite.connect() everywhere.

    async with get_db() as db:
        await db.execute(...)
    """
    async with aiosqlite.connect(cfg.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


# ─────────────────────────────────────────────
# Table Definitions
# ─────────────────────────────────────────────

# ── Users & Auth ──────────────────────────────
USERS_SQL = """
    CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        email       TEXT UNIQUE NOT NULL,
        provider    TEXT NOT NULL DEFAULT 'email',
        google_id   TEXT,
        is_blocked  INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL,
        last_login  TEXT
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
        email   TEXT NOT NULL,
        sent_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES users(id),
        ip         TEXT,
        user_agent TEXT,
        created_at TEXT NOT NULL,
        last_seen  TEXT NOT NULL,
        revoked    INTEGER NOT NULL DEFAULT 0
    );
"""

# ── Wallet ────────────────────────────────────
WALLET_SQL = """
    CREATE TABLE IF NOT EXISTS wallets (
        user_id    TEXT PRIMARY KEY REFERENCES users(id),
        balance_lc REAL NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS wallet_ledger (
        id            TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL REFERENCES users(id),
        delta_lc      REAL NOT NULL,
        reason        TEXT NOT NULL,
        ref_id        TEXT,
        balance_after REAL NOT NULL,
        created_at    TEXT NOT NULL
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_ref_id
        ON wallet_ledger(ref_id)
        WHERE ref_id IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_ledger_user
        ON wallet_ledger(user_id, created_at DESC);
"""

# ── Jobs ──────────────────────────────────────
JOBS_SQL = """
    CREATE TABLE IF NOT EXISTS jobs (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id),
        prompt      TEXT NOT NULL,
        resolution  TEXT NOT NULL,
        addons      TEXT NOT NULL DEFAULT '{}',
        cost_lc     REAL NOT NULL,
        status      TEXT NOT NULL DEFAULT 'queued',
        priority    INTEGER NOT NULL DEFAULT 3,
        attempt     INTEGER NOT NULL DEFAULT 0,
        image_url   TEXT,
        fail_reason TEXT,
        created_at  TEXT NOT NULL,
        started_at  TEXT,
        finished_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_jobs_user
        ON jobs(user_id, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_jobs_queue
        ON jobs(status, priority, created_at)
        WHERE status = 'queued';
"""

# ── Payments ──────────────────────────────────
PAYMENTS_SQL = """
    CREATE TABLE IF NOT EXISTS payment_orders (
        order_id     TEXT PRIMARY KEY,
        user_id      TEXT NOT NULL REFERENCES users(id),
        pack_id      TEXT NOT NULL,
        amount_inr   REAL NOT NULL,
        lc_amount    REAL NOT NULL,
        gateway      TEXT NOT NULL,
        status       TEXT NOT NULL DEFAULT 'pending',
        created_at   TEXT NOT NULL,
        fulfilled_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_orders_user
        ON payment_orders(user_id, created_at DESC);
"""

# ── API Keys ──────────────────────────────────
API_KEYS_SQL = """
    CREATE TABLE IF NOT EXISTS api_keys (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id),
        key_hash    TEXT NOT NULL UNIQUE,
        key_prefix  TEXT NOT NULL,
        label       TEXT,
        is_active   INTEGER NOT NULL DEFAULT 1,
        created_at  TEXT NOT NULL,
        last_used   TEXT,
        total_calls INTEGER NOT NULL DEFAULT 0
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_apikeys_hash
        ON api_keys(key_hash);

    CREATE INDEX IF NOT EXISTS idx_apikeys_user
        ON api_keys(user_id, is_active);
"""


# ─────────────────────────────────────────────
# Init — call once on startup
# ─────────────────────────────────────────────
async def init_all_tables():
    """
    Creates all tables in correct order.
    Safe to call multiple times (IF NOT EXISTS).

    Order matters — foreign keys require parent tables first:
    users → wallets → jobs → payments → api_keys
    """
    async with aiosqlite.connect(cfg.DB_PATH) as db:
        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON")

        await db.executescript(USERS_SQL)
        logger.info("✓ Users & Auth tables")

        await db.executescript(WALLET_SQL)
        logger.info("✓ Wallet tables")

        await db.executescript(JOBS_SQL)
        logger.info("✓ Jobs tables")

        await db.executescript(PAYMENTS_SQL)
        logger.info("✓ Payment tables")

        await db.executescript(API_KEYS_SQL)
        logger.info("✓ API Keys tables")

        await db.commit()

    logger.info(f"✅ Database ready → {cfg.DB_PATH}")
