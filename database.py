"""
LuxurAI Database
SQLite with all tables for users, LC wallet, transactions, queue history
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = Path("luxurai.db")

# ==================== INIT ====================
def init_db():
    """Create all tables if they don't exist"""
    conn = get_conn()
    c = conn.cursor()

    # USERS
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            gmail           TEXT UNIQUE NOT NULL,
            name            TEXT,
            picture         TEXT,
            google_id       TEXT UNIQUE NOT NULL,
            lc_balance      REAL DEFAULT 45.0,
            total_generated INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now')),
            last_seen       TEXT DEFAULT (datetime('now')),
            is_banned       INTEGER DEFAULT 0,
            is_spammer      INTEGER DEFAULT 0
        )
    """)

    # LC TRANSACTIONS
    c.execute("""
        CREATE TABLE IF NOT EXISTS lc_transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            type        TEXT NOT NULL,  -- 'signup_bonus','purchase','spend','reward'
            amount      REAL NOT NULL,  -- positive=credit, negative=debit
            description TEXT,
            ref_id      TEXT,           -- payment ref or generation id
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # GENERATIONS
    c.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id              TEXT PRIMARY KEY,  -- UUID
            user_id         INTEGER NOT NULL,
            prompt          TEXT NOT NULL,
            width           INTEGER NOT NULL,
            height          INTEGER NOT NULL,
            lc_cost         REAL NOT NULL,
            fast_gen        INTEGER DEFAULT 0,
            no_watermark    INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'queued',  -- queued/processing/done/failed
            image_path      TEXT,
            queue_position  INTEGER,
            wait_time_shown INTEGER,  -- artificial delay shown to user (seconds)
            real_time_ms    INTEGER,  -- actual generation time
            created_at      TEXT DEFAULT (datetime('now')),
            completed_at    TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # PAYMENTS
    c.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            razorpay_id     TEXT UNIQUE,
            order_id        TEXT,
            pack_name       TEXT NOT NULL,
            lc_amount       REAL NOT NULL,
            inr_amount      REAL NOT NULL,
            status          TEXT DEFAULT 'pending',  -- pending/success/failed
            created_at      TEXT DEFAULT (datetime('now')),
            completed_at    TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # LOYALTY TRACKING
    c.execute("""
        CREATE TABLE IF NOT EXISTS loyalty (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER UNIQUE NOT NULL,
            fast_gen_count      INTEGER DEFAULT 0,
            no_wm_count         INTEGER DEFAULT 0,
            weekly_streak       INTEGER DEFAULT 0,
            last_active_week    TEXT,
            total_images_500    INTEGER DEFAULT 0,  -- milestones hit
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("✅ Database initialized")


# ==================== CONNECTION ====================
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ==================== USER OPERATIONS ====================
def get_user_by_google_id(google_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE google_id = ?", (google_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(google_id: str, gmail: str, name: str, picture: str) -> dict:
    """Create new user with 45 LC signup bonus"""
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT INTO users (google_id, gmail, name, picture, lc_balance)
        VALUES (?, ?, ?, ?, 45.0)
    """, (google_id, gmail, name, picture))

    user_id = c.lastrowid

    # Log signup bonus transaction
    c.execute("""
        INSERT INTO lc_transactions (user_id, type, amount, description)
        VALUES (?, 'signup_bonus', 45.0, 'Welcome bonus - 45 LC free!')
    """, (user_id,))

    # Create loyalty record
    c.execute("""
        INSERT INTO loyalty (user_id) VALUES (?)
    """, (user_id,))

    conn.commit()

    user = dict(conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone())

    conn.close()
    logger.info(f"✅ New user: {gmail} (45 LC bonus credited)")
    return user


def update_last_seen(user_id: int):
    conn = get_conn()
    conn.execute(
        "UPDATE users SET last_seen = datetime('now') WHERE id = ?", (user_id,)
    )
    conn.commit()
    conn.close()


def get_lc_balance(user_id: int) -> float:
    conn = get_conn()
    row = conn.execute(
        "SELECT lc_balance FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return row["lc_balance"] if row else 0.0


def deduct_lc(user_id: int, amount: float, description: str, ref_id: str = None) -> bool:
    """Deduct LC from user balance. Returns False if insufficient."""
    conn = get_conn()
    c = conn.cursor()

    row = c.execute(
        "SELECT lc_balance FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    if not row or row["lc_balance"] < amount:
        conn.close()
        return False

    c.execute(
        "UPDATE users SET lc_balance = lc_balance - ? WHERE id = ?",
        (amount, user_id)
    )
    c.execute("""
        INSERT INTO lc_transactions (user_id, type, amount, description, ref_id)
        VALUES (?, 'spend', ?, ?, ?)
    """, (user_id, -amount, description, ref_id))

    conn.commit()
    conn.close()
    return True


def credit_lc(user_id: int, amount: float, type: str, description: str, ref_id: str = None):
    """Credit LC to user balance"""
    conn = get_conn()
    conn.execute(
        "UPDATE users SET lc_balance = lc_balance + ? WHERE id = ?",
        (amount, user_id)
    )
    conn.execute("""
        INSERT INTO lc_transactions (user_id, type, amount, description, ref_id)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, type, amount, description, ref_id))
    conn.commit()
    conn.close()


def get_transaction_history(user_id: int, limit: int = 20) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM lc_transactions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==================== GENERATION OPERATIONS ====================
def create_generation(
    gen_id: str, user_id: int, prompt: str,
    width: int, height: int, lc_cost: float,
    fast_gen: bool, no_watermark: bool,
    queue_position: int, wait_time_shown: int
) -> dict:
    conn = get_conn()
    conn.execute("""
        INSERT INTO generations (
            id, user_id, prompt, width, height, lc_cost,
            fast_gen, no_watermark, status,
            queue_position, wait_time_shown
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
    """, (
        gen_id, user_id, prompt, width, height, lc_cost,
        int(fast_gen), int(no_watermark),
        queue_position, wait_time_shown
    ))
    conn.commit()
    row = dict(conn.execute(
        "SELECT * FROM generations WHERE id = ?", (gen_id,)
    ).fetchone())
    conn.close()
    return row


def update_generation_status(gen_id: str, status: str, image_path: str = None, real_time_ms: int = None):
    conn = get_conn()
    if status == "done":
        conn.execute("""
            UPDATE generations
            SET status = ?, image_path = ?, real_time_ms = ?,
                completed_at = datetime('now')
            WHERE id = ?
        """, (status, image_path, real_time_ms, gen_id))
        # Increment total_generated for user
        conn.execute("""
            UPDATE users SET total_generated = total_generated + 1
            WHERE id = (SELECT user_id FROM generations WHERE id = ?)
        """, (gen_id,))
    else:
        conn.execute(
            "UPDATE generations SET status = ? WHERE id = ?",
            (status, gen_id)
        )
    conn.commit()
    conn.close()


def get_generation(gen_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM generations WHERE id = ?", (gen_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_generations(user_id: int, limit: int = 20) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM generations
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==================== PAYMENT OPERATIONS ====================
def create_payment(user_id: int, order_id: str, pack_name: str, lc_amount: float, inr_amount: float) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO payments (user_id, order_id, pack_name, lc_amount, inr_amount)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, order_id, pack_name, lc_amount, inr_amount))
    payment_id = c.lastrowid
    conn.commit()
    conn.close()
    return payment_id


def complete_payment(order_id: str, razorpay_id: str) -> dict | None:
    conn = get_conn()
    c = conn.cursor()

    row = c.execute(
        "SELECT * FROM payments WHERE order_id = ? AND status = 'pending'",
        (order_id,)
    ).fetchone()

    if not row:
        conn.close()
        return None

    payment = dict(row)

    c.execute("""
        UPDATE payments
        SET status = 'success', razorpay_id = ?, completed_at = datetime('now')
        WHERE order_id = ?
    """, (razorpay_id, order_id))

    conn.commit()
    conn.close()
    return payment


# ==================== LOYALTY OPERATIONS ====================
def check_and_award_loyalty(user_id: int):
    """Check loyalty milestones and award LC if earned"""
    conn = get_conn()
    c = conn.cursor()

    user = dict(c.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone())

    loyalty = dict(c.execute(
        "SELECT * FROM loyalty WHERE user_id = ?", (user_id,)
    ).fetchone())

    rewards = []

    # 500 images milestone
    milestones_hit = user["total_generated"] // 500
    if milestones_hit > loyalty["total_images_500"]:
        new_milestones = milestones_hit - loyalty["total_images_500"]
        lc_reward = new_milestones * 0.5
        c.execute(
            "UPDATE loyalty SET total_images_500 = ? WHERE user_id = ?",
            (milestones_hit, user_id)
        )
        c.execute(
            "UPDATE users SET lc_balance = lc_balance + ? WHERE id = ?",
            (lc_reward, user_id)
        )
        c.execute("""
            INSERT INTO lc_transactions (user_id, type, amount, description)
            VALUES (?, 'reward', ?, '🎨 500 images milestone reward!')
        """, (user_id, lc_reward))
        rewards.append(f"+{lc_reward} LC for 500 images milestone!")

    # Fast gen every 10th free
    fast_gen_count = loyalty["fast_gen_count"]
    if fast_gen_count > 0 and fast_gen_count % 10 == 0:
        c.execute(
            "UPDATE users SET lc_balance = lc_balance + 3.0 WHERE id = ?",
            (user_id,)
        )
        c.execute("""
            INSERT INTO lc_transactions (user_id, type, amount, description)
            VALUES (?, 'reward', 3.0, '⚡ 10th Fast Gen free reward!')
        """, (user_id,))
        rewards.append("+3 LC for 10th Fast Gen!")

    # No WM every 10th free
    no_wm_count = loyalty["no_wm_count"]
    if no_wm_count > 0 and no_wm_count % 10 == 0:
        c.execute(
            "UPDATE users SET lc_balance = lc_balance + 7.0 WHERE id = ?",
            (user_id,)
        )
        c.execute("""
            INSERT INTO lc_transactions (user_id, type, amount, description)
            VALUES (?, 'reward', 7.0, '🚫 10th No WM free reward!')
        """, (user_id,))
        rewards.append("+7 LC for 10th No Watermark!")

    conn.commit()
    conn.close()
    return rewards


def increment_fast_gen_count(user_id: int):
    conn = get_conn()
    conn.execute(
        "UPDATE loyalty SET fast_gen_count = fast_gen_count + 1 WHERE user_id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()


def increment_no_wm_count(user_id: int):
    conn = get_conn()
    conn.execute(
        "UPDATE loyalty SET no_wm_count = no_wm_count + 1 WHERE user_id = ?",
        (user_id,)
    )
    conn.commit()

    conn.close()
