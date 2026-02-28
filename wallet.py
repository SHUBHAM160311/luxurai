"""
LuxurAI — wallet.py
─────────────────────────────────────────────────────────────────
Bank-style LC Wallet Service
- Ledger-first (source of truth)
- Atomic debit / credit
- Full audit trail
- Refund-safe
- Rebuild balance from ledger anytime

Usage:
    wallet = WalletService(db_connection)
    await wallet.credit(user_id, 200, "lc_purchase", payment_id)
    await wallet.debit(user_id, 9, "image_generation", job_id)
─────────────────────────────────────────────────────────────────
"""

import os
import secrets
import aiosqlite
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "luxurai.db")

# LC pricing (₹ per LC = 0.25 base)
LC_RATE_INR = 0.25

# Debit reasons
REASON_IMAGE_GEN      = "image_generation"
REASON_API_CALL       = "api_call"
REASON_FAST_GEN       = "fast_gen_addon"
REASON_NO_WATERMARK   = "no_watermark_addon"
REASON_BULK_GEN       = "bulk_generation"

# Credit reasons
REASON_PURCHASE       = "lc_purchase"
REASON_SIGNUP_BONUS   = "signup_bonus"
REASON_REFUND         = "refund_generation_failed"
REASON_LOYALTY        = "loyalty_reward"
REASON_ADMIN          = "admin_credit"
REASON_REFERRAL       = "referral_bonus"


# ─────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────
class WalletError(Exception):
    """Base wallet exception."""

class InsufficientBalanceError(WalletError):
    """User doesn't have enough LC."""

class InvalidAmountError(WalletError):
    """Amount is zero or negative."""

class DuplicateTransactionError(WalletError):
    """Same ref_id already processed (idempotency guard)."""

class WalletNotFoundError(WalletError):
    """Wallet doesn't exist for this user."""


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class LedgerEntry:
    id: str
    user_id: str
    delta_lc: float
    reason: str
    ref_id: Optional[str]
    balance_after: float
    created_at: str

@dataclass
class WalletState:
    user_id: str
    balance_lc: float
    updated_at: str


# ─────────────────────────────────────────────
# DB init (call once on startup)
# ─────────────────────────────────────────────
async def init_wallet_tables(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                user_id     TEXT PRIMARY KEY,
                balance_lc  REAL NOT NULL DEFAULT 0,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wallet_ledger (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                delta_lc      REAL NOT NULL,
                reason        TEXT NOT NULL,
                ref_id        TEXT,
                balance_after REAL NOT NULL,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            -- Unique index on ref_id prevents duplicate credits/debits
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_ref_id
                ON wallet_ledger(ref_id)
                WHERE ref_id IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_ledger_user
                ON wallet_ledger(user_id, created_at DESC);
        """)
        await db.commit()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return secrets.token_urlsafe(12)


# ─────────────────────────────────────────────
# WalletService
# ─────────────────────────────────────────────
class WalletService:
    """
    All wallet operations.
    Pass db_path or use default from env.
    Every method opens its own connection for safety.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ─── Read ─────────────────────────────────

    async def get_balance(self, user_id: str) -> float:
        """Return current LC balance. Returns 0.0 if wallet doesn't exist yet."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT balance_lc FROM wallets WHERE user_id = ?", (user_id,)
            ) as cur:
                row = await cur.fetchone()
            return float(row["balance_lc"]) if row else 0.0

    async def get_wallet(self, user_id: str) -> Optional[WalletState]:
        """Full wallet object or None."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM wallets WHERE user_id = ?", (user_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return None
            return WalletState(
                user_id=row["user_id"],
                balance_lc=float(row["balance_lc"]),
                updated_at=row["updated_at"]
            )

    async def get_ledger(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[LedgerEntry]:
        """Paginated ledger history for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM wallet_ledger
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (user_id, limit, offset)
            ) as cur:
                rows = await cur.fetchall()
            return [
                LedgerEntry(
                    id=r["id"], user_id=r["user_id"],
                    delta_lc=float(r["delta_lc"]), reason=r["reason"],
                    ref_id=r["ref_id"], balance_after=float(r["balance_after"]),
                    created_at=r["created_at"]
                )
                for r in rows
            ]

    async def rebuild_balance(self, user_id: str) -> float:
        """
        Recalculate balance from ledger sum.
        Use for auditing or corruption recovery.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT COALESCE(SUM(delta_lc), 0) AS total FROM wallet_ledger WHERE user_id = ?",
                (user_id,)
            ) as cur:
                row = await cur.fetchone()
            total = float(row["total"])
            # Sync wallet table
            await db.execute(
                "UPDATE wallets SET balance_lc = ?, updated_at = ? WHERE user_id = ?",
                (total, _now(), user_id)
            )
            await db.commit()
        return total

    # ─── Write ────────────────────────────────

    async def _ensure_wallet(self, db, user_id: str) -> float:
        """Create wallet row if missing. Returns current balance."""
        async with db.execute(
            "SELECT balance_lc FROM wallets WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            await db.execute(
                "INSERT INTO wallets (user_id, balance_lc, updated_at) VALUES (?,?,?)",
                (user_id, 0.0, _now())
            )
            return 0.0
        return float(row[0])

    async def _check_duplicate(self, db, ref_id: str) -> bool:
        """Returns True if ref_id already exists in ledger."""
        if not ref_id:
            return False
        async with db.execute(
            "SELECT 1 FROM wallet_ledger WHERE ref_id = ?", (ref_id,)
        ) as cur:
            return await cur.fetchone() is not None

    async def _write_ledger(self, db, user_id: str, delta_lc: float,
                             reason: str, ref_id: Optional[str], balance_after: float):
        await db.execute(
            """INSERT INTO wallet_ledger
               (id, user_id, delta_lc, reason, ref_id, balance_after, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (_new_id(), user_id, delta_lc, reason, ref_id, balance_after, _now())
        )

    async def credit(
        self,
        user_id: str,
        amount_lc: float,
        reason: str,
        ref_id: Optional[str] = None
    ) -> float:
        """
        Add LC to wallet.
        Returns new balance.
        Raises DuplicateTransactionError if ref_id already processed.
        """
        if amount_lc <= 0:
            raise InvalidAmountError(f"Credit amount must be positive, got {amount_lc}")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Idempotency guard
            if ref_id and await self._check_duplicate(db, ref_id):
                raise DuplicateTransactionError(f"ref_id '{ref_id}' already credited")

            current = await self._ensure_wallet(db, user_id)
            new_balance = round(current + amount_lc, 4)

            await db.execute(
                "UPDATE wallets SET balance_lc = ?, updated_at = ? WHERE user_id = ?",
                (new_balance, _now(), user_id)
            )
            await self._write_ledger(db, user_id, +amount_lc, reason, ref_id, new_balance)
            await db.commit()

        return new_balance

    async def debit(
        self,
        user_id: str,
        amount_lc: float,
        reason: str,
        ref_id: Optional[str] = None
    ) -> float:
        """
        Deduct LC from wallet.
        Returns new balance.
        Raises InsufficientBalanceError if not enough LC.
        Raises DuplicateTransactionError if ref_id already processed.
        """
        if amount_lc <= 0:
            raise InvalidAmountError(f"Debit amount must be positive, got {amount_lc}")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Idempotency guard
            if ref_id and await self._check_duplicate(db, ref_id):
                raise DuplicateTransactionError(f"ref_id '{ref_id}' already debited")

            current = await self._ensure_wallet(db, user_id)

            if current < amount_lc:
                raise InsufficientBalanceError(
                    f"Need {amount_lc} LC, wallet has {current} LC"
                )

            new_balance = round(current - amount_lc, 4)

            await db.execute(
                "UPDATE wallets SET balance_lc = ?, updated_at = ? WHERE user_id = ?",
                (new_balance, _now(), user_id)
            )
            await self._write_ledger(db, user_id, -amount_lc, reason, ref_id, new_balance)
            await db.commit()

        return new_balance

    async def refund(
        self,
        user_id: str,
        amount_lc: float,
        original_ref_id: str,
        reason: str = REASON_REFUND
    ) -> float:
        """
        Refund LC back to wallet after a failed generation.
        ref_id is prefixed with 'refund_' to avoid collision.
        """
        refund_ref_id = f"refund_{original_ref_id}"
        return await self.credit(user_id, amount_lc, reason, refund_ref_id)

    async def admin_credit(
        self,
        user_id: str,
        amount_lc: float,
        note: str,
        admin_id: str
    ) -> float:
        """
        Manual credit by admin (logged with admin_id).
        Always has a unique ref_id so it's auditable.
        """
        ref_id = f"admin_{admin_id}_{_new_id()}"
        return await self.credit(
            user_id, amount_lc,
            reason=f"{REASON_ADMIN}:{note}",
            ref_id=ref_id
        )


# ─────────────────────────────────────────────
# Generation lifecycle helper
# ─────────────────────────────────────────────
class GenerationBilling:
    """
    Handles the debit-before-generate, refund-on-fail pattern.

    Usage:
        billing = GenerationBilling(wallet_service)
        async with billing.charge(user_id, cost_lc, job_id):
            # run image generation here
            # auto-refunds if exception is raised
    """

    def __init__(self, wallet: WalletService):
        self.wallet = wallet

    class _Charge:
        def __init__(self, wallet, user_id, amount_lc, job_id):
            self.wallet = wallet
            self.user_id = user_id
            self.amount_lc = amount_lc
            self.job_id = job_id
            self.committed = False

        async def __aenter__(self):
            await self.wallet.debit(
                self.user_id, self.amount_lc,
                reason=REASON_IMAGE_GEN,
                ref_id=self.job_id
            )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None and not self.committed:
                # Generation failed → refund
                try:
                    await self.wallet.refund(
                        self.user_id, self.amount_lc,
                        original_ref_id=self.job_id
                    )
                except Exception:
                    pass  # Log in production
            return False  # Don't suppress exceptions

        def commit(self):
            """Call after successful generation."""
            self.committed = True

    def charge(self, user_id: str, amount_lc: float, job_id: str) -> "_Charge":
        return self._Charge(self.wallet, user_id, amount_lc, job_id)


# ─────────────────────────────────────────────
# Image generation cost calculator
# ─────────────────────────────────────────────

# Base resolution costs (same for all users)
BASE_COST = {
    "100x100":   0.5,
    "200x200":   1.0,
    "500x500":   3.0,
    "512x512":   3.0,
    "512x1024":  6.0,
    "1024x512":  6.0,
    "1024x1024": 9.0,
    "2048x2048": 15.0,
}

# Addon costs (normal user prices)
ADDON_COST = {
    "fast_gen":     3.0,   # API user pays 2.0 (1 LC discount)
    "no_watermark": 7.0,   # API user pays 6.0 (1 LC discount)
}

# API Key subscription cost (per month, one-time deduct)
API_KEY_SUBSCRIPTION_LC = 2.0

# Bulk: only for API users, flat 5 LC for up to 10 images
BULK_COST_LC   = 5.0
BULK_MAX_IMGS  = 10

# Resolutions where API discount does NOT apply (≤200 on both sides)
NO_DISCOUNT_RESOLUTIONS = {"100x100", "200x200"}

# 1 LC discount for API users on all resolutions above 200
API_DISCOUNT_LC = 1.0


def _is_above_200(resolution: str) -> bool:
    """
    Returns True if BOTH dimensions are above 200px.
    e.g. 500x500 → True, 200x200 → False, 1024x1024 → True
    """
    res = resolution.lower().replace(" ", "")
    if "x" not in res:
        return False
    try:
        w, h = res.split("x")
        return int(w) > 200 and int(h) > 200
    except ValueError:
        return False


def calculate_generation_cost(
    resolution: str,
    fast: bool = False,
    no_watermark: bool = False,
    is_api: bool = False,
    bulk: bool = False,
    bulk_count: int = 1,
) -> float:
    """
    Single source of truth for generation pricing.

    Rules:
      - Normal users: base cost + addons at full price. No bulk.
      - API users:
          * All resolutions >200 (both sides) → base cost - 1 LC
          * fast_gen addon  → 2 LC (instead of 3)
          * no_watermark    → 6 LC (instead of 7)
          * bulk (max 10)   → flat 5 LC total (ignores per-image cost)

    Returns total LC cost.
    """
    # ── Bulk (API only, flat rate) ────────────
    if bulk:
        if not is_api:
            raise PermissionError("Bulk generation is only available for API key users.")
        count = min(max(1, bulk_count), BULK_MAX_IMGS)
        return BULK_COST_LC   # 5 LC flat for up to 10 images

    # ── Base resolution cost ──────────────────
    res_key = resolution.lower().replace(" ", "")
    cost = BASE_COST.get(res_key, 1.0)

    # API discount on resolution (only if >200 both sides)
    if is_api and _is_above_200(resolution):
        cost = max(0, cost - API_DISCOUNT_LC)   # never go below 0

    # ── Addons ────────────────────────────────
    # fast_gen: only when bulk is NOT used (bulk already covers it)
    if fast:
        cost += (ADDON_COST["fast_gen"] - API_DISCOUNT_LC) if is_api else ADDON_COST["fast_gen"]

    # no_watermark
    if no_watermark:
        cost += (ADDON_COST["no_watermark"] - API_DISCOUNT_LC) if is_api else ADDON_COST["no_watermark"]

    return round(cost, 2)


def get_api_subscription_cost() -> float:
    """Returns the monthly API key subscription cost in LC."""
    return API_KEY_SUBSCRIPTION_LC
