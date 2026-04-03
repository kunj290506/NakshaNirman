"""
Configuration — loads environment variables for OpenRouter API.
"""
import os
from dotenv import load_dotenv

# Load from backend/.env first, fallback to root .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEY_SECONDARY = os.getenv("OPENROUTER_API_KEY_SECONDARY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free")
# If plan model is not explicitly set, inherit the main OpenRouter model.
# This avoids accidentally forcing weaker defaults for architectural planning.
OPENROUTER_PLAN_MODEL = os.getenv("OPENROUTER_PLAN_MODEL", OPENROUTER_MODEL)
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# Public no-key fallback provider (OpenAI-compatible) used when OpenRouter
# is unavailable or quota-restricted.
PUBLIC_LLM_FALLBACK_ENABLED = os.getenv(
    "PUBLIC_LLM_FALLBACK_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}
PUBLIC_LLM_FALLBACK_URL = os.getenv(
    "PUBLIC_LLM_FALLBACK_URL", "https://text.pollinations.ai/openai"
)
PUBLIC_LLM_FALLBACK_MODEL = os.getenv(
    "PUBLIC_LLM_FALLBACK_MODEL", "openai"
)

# Fast-fallback mode trades extra LLM retry depth for lower latency and
# deterministic completion when external providers are slow.
FAST_FALLBACK_MODE = os.getenv(
    "FAST_FALLBACK_MODE", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# Force deterministic local planning (skip external LLM variability).
FORCE_LOCAL_PLANNER = os.getenv(
    "FORCE_LOCAL_PLANNER", "false"
).strip().lower() in {"1", "true", "yes", "on"}

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)
