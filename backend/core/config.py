"""
LuxurAI — core/config.py
─────────────────────────────────────────────────────────────────
Single source of truth for ALL environment variables.

Every other file imports from here — no more os.getenv() scattered
across auth.py, payment.py, wallet.py etc.

Usage:
    from core.config import cfg

    print(cfg.DB_PATH)
    print(cfg.CASHFREE_APP_ID)
─────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv

# Load both env files
load_dotenv("CashFree.env")
load_dotenv()   # .env for everything else


class Config:
    # ── App ───────────────────────────────────
    ENV:          str = os.getenv("ENV", "development")   # "production" in prod
    DB_PATH:      str = os.getenv("DB_PATH", "luxurai.db")
    BASE_URL:     str = os.getenv("BASE_URL", "https://luxurai.in")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://luxurai.in")

    # ── Security ──────────────────────────────
    SECRET_KEY:   str = os.getenv("SECRET_KEY", "dev-secret-change-in-prod!")
    JWT_SECRET:   str = os.getenv("JWT_SECRET",  os.getenv("SECRET_KEY", "dev-secret"))
    ALGORITHM:    str = "HS256"
    SESSION_DAYS: int = 30

    # ── Auth ──────────────────────────────────
    GOOGLE_CLIENT_ID:     str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    RESEND_API_KEY:       str = os.getenv("RESEND_API_KEY", "")
    EMAIL_FROM:           str = os.getenv("EMAIL_FROM", "noreply@luxurai.in")

    MAGIC_LINK_MINUTES: int = 10
    SIGNUP_BONUS_LC:    int = 45
    RATE_LIMIT_HOURLY:  int = 3
    RATE_LIMIT_DAILY:   int = 10

    # ── Cashfree ──────────────────────────────
    CASHFREE_APP_ID:     str = os.getenv("CASHFREE_APP_ID", "")
    CASHFREE_SECRET_KEY: str = os.getenv("CASHFREE_SECRET_KEY", "")
    CASHFREE_ENV:        str = os.getenv("CASHFREE_ENV", "TEST")   # PROD or TEST

    @property
    def CASHFREE_BASE_URL(self) -> str:
        return (
            "https://api.cashfree.com/pg"
            if self.CASHFREE_ENV == "PROD"
            else "https://sandbox.cashfree.com/pg"
        )

    # ── Stripe ────────────────────────────────
    STRIPE_SECRET_KEY:     str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # ── Wallet / LC ───────────────────────────
    LC_RATE_INR: float = 0.25   # ₹ per LC

    # ── Jobs / Queue ──────────────────────────
    MAX_RETRY:     int = 3
    POLL_INTERVAL: float = 2.0   # seconds between worker polls

    # ── Shortcuts ─────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def is_dev(self) -> bool:
        return self.ENV == "development"

    @property
    def cashfree_ready(self) -> bool:
        return bool(self.CASHFREE_APP_ID and self.CASHFREE_SECRET_KEY)

    @property
    def stripe_ready(self) -> bool:
        return bool(self.STRIPE_SECRET_KEY)

    @property
    def google_ready(self) -> bool:
        return bool(self.GOOGLE_CLIENT_ID and self.GOOGLE_CLIENT_SECRET)

    def __repr__(self):
        return (
            f"<Config env={self.ENV} "
            f"cashfree={'✓' if self.cashfree_ready else '✗'} "
            f"stripe={'✓' if self.stripe_ready else '✗'} "
            f"google={'✓' if self.google_ready else '✗'}>"
        )


# Single global instance — import this everywhere
cfg = Config()
