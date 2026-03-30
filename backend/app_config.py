"""Application configuration via environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR / 'floorplan.db'}")

# OpenRouter API - Preferred planner/model gateway
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "black-forest-labs/flux.2-klein-4b")
OPENROUTER_TEXT_MODEL = os.getenv("OPENROUTER_TEXT_MODEL", "deepseek/deepseek-r1").strip()
OPENROUTER_PLANNER_MODEL = os.getenv("OPENROUTER_PLANNER_MODEL", "deepseek/deepseek-chat").strip()
OPENROUTER_IMAGE_MODEL = os.getenv("OPENROUTER_IMAGE_MODEL", OPENROUTER_MODEL).strip()
OPENROUTER_TEXT_TEMPERATURE = _env_float("OPENROUTER_TEXT_TEMPERATURE", 0.3)
OPENROUTER_TEXT_MAX_TOKENS = _env_int("OPENROUTER_TEXT_MAX_TOKENS", 2400)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_ENABLED = os.getenv("OPENROUTER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
OPENROUTER_VERIFY_SSL = os.getenv("OPENROUTER_VERIFY_SSL", "true").strip().lower() not in {"0", "false", "no", "off"}

# File Storage
UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

if APP_ENV == "production":
    if SECRET_KEY == "dev-secret-key-change-in-production" or len(SECRET_KEY) < 16:
        raise RuntimeError("Invalid SECRET_KEY for production. Set a strong SECRET_KEY (>=16 chars).")

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")