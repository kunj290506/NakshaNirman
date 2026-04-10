"""
Configuration — loads environment variables for local model execution.
"""
import os
from dotenv import load_dotenv

# Load from backend/.env first, then root .env.
# override=True ensures stale shell env vars do not pin old runtime behavior.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

LOCAL_LLM_ENABLED = os.getenv(
    "LOCAL_LLM_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_LLM_BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1"
).strip().rstrip("/")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
LOCAL_LLM_PLAN_MODEL = os.getenv("LOCAL_LLM_PLAN_MODEL", LOCAL_LLM_MODEL)
LOCAL_LLM_BACKUP_MODEL = os.getenv("LOCAL_LLM_BACKUP_MODEL", "llama3.1:8b-instruct")
LOCAL_LLM_ADVISORY_MODEL = os.getenv("LOCAL_LLM_ADVISORY_MODEL", LOCAL_LLM_MODEL)

# Fast-fallback mode trades extra LLM retry depth for lower latency and
# deterministic completion when external providers are slow.
FAST_FALLBACK_MODE = os.getenv(
    "FAST_FALLBACK_MODE", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# Enables a pre-plot architect reasoning stage (deterministic + optional LLM advisory).
ARCHITECT_REASONING_ENABLED = os.getenv(
    "ARCHITECT_REASONING_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}

# Force deterministic local planning (skip external LLM variability).
FORCE_LOCAL_PLANNER = os.getenv(
    "FORCE_LOCAL_PLANNER", "false"
).strip().lower() in {"1", "true", "yes", "on"}

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)
