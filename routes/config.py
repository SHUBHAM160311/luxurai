"""
LuxurAI Configuration
Centralized environment variable management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ==================== GOOGLE OAUTH ====================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# ==================== JWT SECRET ====================
JWT_SECRET = os.getenv("JWT_SECRET", "luxurai-default-secret-CHANGE-ME")
JWT_EXPIRE_DAYS = 30

# ==================== RAZORPAY ====================
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

# ==================== DATABASE ====================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./luxurai.db")

# ==================== SERVER ====================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "luxurai.db"

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ==================== VALIDATION ====================
def validate_config():
    """Check if all required env vars are set"""
    warnings = []
    
    if not GOOGLE_CLIENT_ID:
        warnings.append("⚠️  GOOGLE_CLIENT_ID not set")
    
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        warnings.append("⚠️  Razorpay credentials not set")
    
    if JWT_SECRET == "luxurai-default-secret-CHANGE-ME":
        warnings.append("⚠️  Using default JWT_SECRET - INSECURE!")
    
    return warnings
