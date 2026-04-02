"""
Configuration — loads environment variables for OpenRouter API.
"""
import os
from dotenv import load_dotenv

# Load from backend/.env first, fallback to root .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free")
OPENROUTER_PLAN_MODEL = os.getenv(
    "OPENROUTER_PLAN_MODEL", "qwen/qwen3-coder:free"
)
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)
