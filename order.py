"""
LuxurAI — models/order.py
─────────────────────────────────────────────────────────────────
Payment orders + wallet table definitions + dataclasses.
No logic here — only structure.
─────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class OrderStatus(str, Enum):
    PENDING   = "pending"
    FULFILLED = "fulfilled"
    FAILED    = "failed"


class Gateway(str, Enum):
    CASHFREE     = "cashfree"
    CASHFREE_DEV = "cashfree_dev"
    STRIPE       = "stripe"
    SIMULATED    = "pay_sim"


# ─────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────
PAYMENT_ORDERS_TABLE = """
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

WALLETS_TABLE = """
    CREATE TABLE IF NOT EXISTS wallets (
        user_id    TEXT PRIMARY KEY REFERENCES users(id),
        balance_lc REAL NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );
"""

WALLET_LEDGER_TABLE = """
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

API_KEYS_TABLE = """
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

# LC Packs — single source of truth
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


# ─────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────
@dataclass
class PaymentOrder:
    order_id:     str
    user_id:      str
    pack_id:      str
    amount_inr:   float
    lc_amount:    float
    gateway:      str
    status:       OrderStatus
    created_at:   str
    fulfilled_at: Optional[str]

    @property
    def is_fulfilled(self) -> bool:
        return self.status == OrderStatus.FULFILLED


@dataclass
class WalletLedgerEntry:
    id:           str
    user_id:      str
    delta_lc:     float
    reason:       str
    ref_id:       Optional[str]
    balance_after: float
    created_at:   str

    @property
    def is_credit(self) -> bool:
        return self.delta_lc > 0

    @property
    def is_debit(self) -> bool:
        return self.delta_lc < 0
