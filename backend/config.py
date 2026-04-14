"""
NakshaNirman Config — Local-only mode.
Runs on GTX 1650 + 24GB RAM. Zero external API dependency.
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# ── Local Ollama Model Settings ──────────────────────────────────────
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1").strip().rstrip("/")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M").strip()
LOCAL_LLM_PLAN_MODEL = os.getenv("LOCAL_LLM_PLAN_MODEL", LOCAL_LLM_MODEL).strip()
LOCAL_LLM_BACKUP_MODEL = os.getenv("LOCAL_LLM_BACKUP_MODEL", LOCAL_LLM_MODEL).strip()
LOCAL_LLM_ADVISORY_MODEL = os.getenv("LOCAL_LLM_ADVISORY_MODEL", LOCAL_LLM_MODEL).strip()

# ── External APIs — all empty, all disabled ──────────────────────────
OPENROUTER_API_KEY = ""
OPENROUTER_API_KEY_SECONDARY = ""
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = ""
OPENROUTER_PLAN_MODEL = ""

PUBLIC_LLM_FALLBACK_ENABLED = False
PUBLIC_LLM_FALLBACK_URL = ""
PUBLIC_LLM_FALLBACK_MODEL = ""

ANTHROPIC_API_KEY = ""
CLAUDE_MODEL = ""

# ── Feature Flags ────────────────────────────────────────────────────
FORCE_LOCAL_PLANNER = False
FAST_FALLBACK_MODE = False
ARCHITECT_REASONING_ENABLED = os.getenv(
    "ARCHITECT_REASONING_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}

# ── Security ─────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "naksha-local-dev-key")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

# ── Paths ────────────────────────────────────────────────────────────
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)
