"""
LuxurAI — models/user.py
─────────────────────────────────────────────────────────────────
User & Auth table definitions + dataclasses.
No logic here — only structure.
─────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────
USERS_TABLE = """
    CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        email       TEXT UNIQUE NOT NULL,
        provider    TEXT NOT NULL DEFAULT 'email',
        google_id   TEXT,
        is_blocked  INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL,
        last_login  TEXT
    );
"""

SESSIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id  TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id),
        ip          TEXT,
        user_agent  TEXT,
        created_at  TEXT NOT NULL,
        last_seen   TEXT NOT NULL,
        revoked     INTEGER NOT NULL DEFAULT 0
    );
"""

AUTH_TOKENS_TABLE = """
    CREATE TABLE IF NOT EXISTS auth_tokens (
        token_hash  TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id),
        expires_at  TEXT NOT NULL,
        used        INTEGER NOT NULL DEFAULT 0,
        ip          TEXT,
        user_agent  TEXT,
        created_at  TEXT NOT NULL
    );
"""

MAGIC_LINK_RATE_TABLE = """
    CREATE TABLE IF NOT EXISTS magic_link_rate (
        email   TEXT NOT NULL,
        sent_at TEXT NOT NULL
    );
"""


# ─────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────
@dataclass
class User:
    id:         str
    email:      str
    provider:   str        # "email" | "google"
    google_id:  Optional[str]
    is_blocked: bool
    created_at: str
    last_login: Optional[str]


@dataclass
class Session:
    session_id: str
    user_id:    str
    ip:         Optional[str]
    user_agent: Optional[str]
    created_at: str
    last_seen:  str
    revoked:    bool
