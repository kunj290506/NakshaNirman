"""
NakshaNirman — FastAPI backend.
Uses LLM-first layout engine with BSP fallback.
"""
from __future__ import annotations
import hashlib
import hmac
import logging
import os
import re
import secrets
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import EXPORTS_DIR
from models import PlanRequest, PlanResponse
from layout_engine import generate_architect_reasoning, generate_plan, generate_plan_emergency_local
from dxf_export import plan_to_dxf
from plan_validator import validate_llm_plan

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "nakshanirman")
MONGO_USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION", "users")
_mongo_client: MongoClient | None = None


class LoginRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class SignupRequest(LoginRequest):
    email: str = Field(default="", max_length=254)
    full_name: str = Field(default="", max_length=120)


# Backward-compatible alias for old route payload.
AuthRequest = LoginRequest


def _get_users_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=2500,
            connectTimeoutMS=2500,
        )
    _mongo_client.admin.command("ping")
    return _mongo_client[MONGO_DB_NAME][MONGO_USERS_COLLECTION]


def _clean_credential(value: str) -> str:
    return str(value or "").strip()


def _clean_user_id(value: str) -> str:
    return _clean_credential(value)


def _clean_full_name(value: str) -> str:
    cleaned = _clean_credential(value)
    return re.sub(r"\s+", " ", cleaned)


def _clean_email(value: str) -> str:
    return _clean_credential(value).lower()


def _is_valid_email(value: str) -> bool:
    # Lightweight validation for API-level checks without extra dependencies.
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$", value))


def _hash_password(raw_password: str) -> str:
    iterations = 240_000
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def _verify_password(raw_password: str, encoded_hash: str) -> bool:
    try:
        algo, iters_s, salt, expected_hex = str(encoded_hash or "").split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            raw_password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iters_s),
        )
        return hmac.compare_digest(digest.hex(), expected_hex)
    except Exception:
        return False


def _ensure_valid_credentials(user_id: str, password: str):
    if not user_id or not password:
        raise HTTPException(status_code=400, detail="User ID and password are required")
    if not re.match(r"^[a-zA-Z0-9_.@-]{3,64}$", user_id):
        raise HTTPException(
            status_code=400,
            detail="User ID must be 3-64 chars and may contain letters, numbers, ., _, @, -",
        )


def _ensure_valid_signup(user_id: str, password: str, email: str):
    _ensure_valid_credentials(user_id, password)
    if email and not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address")


def _auth_success_payload(user_id: str, full_name: str, email: str, *, new_user: bool) -> dict[str, Any]:
    return {
        "ok": True,
        "user_id": user_id,
        "full_name": full_name,
        "email": email,
        "saved_in_db": True,
        "new_user": new_user,
    }


def _create_user(
    users,
    user_id: str,
    password: str,
    full_name: str,
    now: str,
    *,
    email: str = "",
) -> dict[str, Any]:
    clean_email = _clean_email(email)
    user_doc: dict[str, Any] = {
        "user_id": user_id,
        "full_name": full_name,
        "password_hash": _hash_password(password),
        "created_at": now,
        "updated_at": now,
        "last_login_at": now,
    }

    if clean_email:
        user_doc["email"] = clean_email
        user_doc["email_lower"] = clean_email

    users.insert_one(user_doc)
    return _auth_success_payload(user_id, full_name, clean_email, new_user=True)


def _login_existing_user(users, existing: dict[str, Any], password: str, now: str) -> dict[str, Any]:
    user_id = str(existing.get("user_id", ""))
    full_name = _clean_full_name(existing.get("full_name", ""))
    email = _clean_email(existing.get("email", ""))
    password_hash = str(existing.get("password_hash", ""))
    legacy_password = str(existing.get("password", ""))

    is_valid = False
    if password_hash:
        is_valid = _verify_password(password, password_hash)
    elif legacy_password:
        # Backward compatibility for older plain-text records before hash rollout.
        is_valid = hmac.compare_digest(legacy_password, password)

    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid user ID or password")

    update_doc: dict[str, Any] = {
        "$set": {
            "updated_at": now,
            "last_login_at": now,
        }
    }
    if legacy_password and not password_hash:
        update_doc["$set"]["password_hash"] = _hash_password(password)
        update_doc["$set"]["password_migrated_at"] = now
        update_doc["$unset"] = {"password": ""}

    users.update_one({"_id": existing["_id"]}, update_doc)
    return _auth_success_payload(user_id, full_name, email, new_user=False)

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="NakshaNirman API",
    version="3.0.0",
    description="Indian residential floor plan generator with deterministic BSP layout",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Security headers middleware ──────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ── Simple rate limiter (per IP) ─────────────────────────────
_rate_store: Dict[str, List[float]] = defaultdict(list)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)).strip()))
    except Exception:
        return default


RATE_LIMIT = _env_int("RATE_LIMIT_GENERATE_PER_MINUTE", 120)
RATE_WINDOW = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
RATE_LIMIT_BYPASS_LOCAL = os.getenv(
    "RATE_LIMIT_BYPASS_LOCAL", "true"
).strip().lower() in {"1", "true", "yes", "on"}


def _is_local_client(ip: str) -> bool:
    ip_norm = str(ip or "").strip().lower()
    return ip_norm in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


@app.middleware("http")
async def rate_limit_generate(request: Request, call_next):
    if request.url.path == "/api/generate" and request.method == "POST":
        ip = request.client.host if request.client else "unknown"

        if RATE_LIMIT_BYPASS_LOCAL and _is_local_client(ip):
            return await call_next(request)

        now = time.time()
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
        if len(_rate_store[ip]) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(RATE_WINDOW)},
                content={"detail": "Too many requests. Please wait and retry."},
            )
        _rate_store[ip].append(now)

        # Opportunistic cleanup to prevent unbounded key growth over time.
        if len(_rate_store) > 5000:
            stale_ips = [k for k, times in _rate_store.items() if not times or now - times[-1] > RATE_WINDOW * 2]
            for stale_ip in stale_ips:
                _rate_store.pop(stale_ip, None)

    return await call_next(request)


# ── Health ───────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "NakshaNirman", "version": "3.0.0"}


# ── Auth ─────────────────────────────────────────────────────
@app.post("/api/auth/signup")
async def signup(payload: SignupRequest):
    user_id = _clean_user_id(payload.user_id)
    password = _clean_credential(payload.password)
    email = _clean_email(payload.email)
    full_name = _clean_full_name(payload.full_name)

    _ensure_valid_signup(user_id, password, email)

    now = datetime.now(timezone.utc).isoformat()

    try:
        users = _get_users_collection()
        if email:
            existing = users.find_one(
                {
                    "$or": [
                        {"user_id": user_id},
                        {"email_lower": email},
                    ]
                }
            )
        else:
            existing = users.find_one({"user_id": user_id})
        if existing:
            raise HTTPException(status_code=409, detail="User already exists. Please login.")

        return _create_user(users, user_id, password, full_name, now, email=email)

    except HTTPException:
        raise
    except PyMongoError as e:
        log.error("MongoDB signup error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not connect to MongoDB at mongodb://localhost:27017/",
        )


@app.post("/api/auth/login")
async def login(payload: LoginRequest):
    user_id = _clean_user_id(payload.user_id)
    password = _clean_credential(payload.password)

    _ensure_valid_credentials(user_id, password)

    now = datetime.now(timezone.utc).isoformat()

    try:
        users = _get_users_collection()
        existing = users.find_one({"user_id": user_id})
        if not existing:
            raise HTTPException(status_code=401, detail="Invalid user ID or password")

        return _login_existing_user(users, existing, password, now)

    except HTTPException:
        raise
    except PyMongoError as e:
        log.error("MongoDB login error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not connect to MongoDB at mongodb://localhost:27017/",
        )


@app.post("/api/auth/save-and-login")
async def save_and_login(payload: AuthRequest):
    user_id = _clean_user_id(payload.user_id)
    password = _clean_credential(payload.password)

    _ensure_valid_credentials(user_id, password)

    now = datetime.now(timezone.utc).isoformat()

    try:
        users = _get_users_collection()
        existing = users.find_one({"user_id": user_id})

        if existing:
            return _login_existing_user(users, existing, password, now)

        return _create_user(users, user_id, password, "", now)

    except HTTPException:
        raise
    except PyMongoError as e:
        log.error("MongoDB auth error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not connect to MongoDB at mongodb://localhost:27017/",
        )


# ── Generate ─────────────────────────────────────────────────
@app.post("/api/architect/reason")
async def architect_reason(req: PlanRequest | dict[str, Any]):
    """Run architect reasoning stage only (no layout plotting)."""
    if isinstance(req, dict):
        payload = dict(req)
        if str(payload.get("family_type", "")).strip().lower() == "working_couple":
            payload["family_type"] = "couple"
        req = PlanRequest(**payload)

    try:
        reasoning = await generate_architect_reasoning(req)
        return {
            "ok": True,
            "reasoning": reasoning,
        }
    except Exception as e:
        log.exception("Architect reasoning failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", response_model=PlanResponse)
async def generate(req: PlanRequest | dict[str, Any]):
    """Generate a floor plan using LLM-first engine with BSP fallback."""
    # Support a known frontend/client alias without changing PlanResponse shape.
    if isinstance(req, dict):
        payload = dict(req)
        if str(payload.get("family_type", "")).strip().lower() == "working_couple":
            payload["family_type"] = "couple"
        req = PlanRequest(**payload)

    log.info(
        "Generate request: %sx%s %dBHK %s extras=%s",
        req.plot_width, req.plot_length, req.bedrooms, req.facing, req.extras,
    )
    try:
        plan = await generate_plan(req)
    except Exception:
        log.exception("Plan generation failed in primary pipeline; switching to emergency local mode")
        try:
            plan = await generate_plan_emergency_local(req)
        except Exception as e:
            log.exception("Emergency local generation also failed")
            raise HTTPException(status_code=500, detail=str(e))

    # Generate DXF
    try:
        filename = f"plan_{uuid.uuid4().hex[:8]}.dxf"
        filepath = os.path.join(EXPORTS_DIR, filename)
        plan_to_dxf(plan, filepath)
        plan.dxf_url = f"/api/download/{filename}"
        log.info("DXF saved: %s", filename)
    except Exception as e:
        log.warning("DXF generation failed: %s", e)
        plan.dxf_url = None

    return plan


# ── Download DXF ─────────────────────────────────────────────
@app.get("/api/download/{filename}")
async def download(filename: str):
    """Download a generated DXF file."""
    if not re.match(r"^plan_[a-f0-9]{8}\.dxf$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename format")
    filepath = os.path.join(EXPORTS_DIR, filename)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(os.path.realpath(EXPORTS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        real_path,
        media_type="application/dxf",
        filename=filename,
    )


# ── Validate ─────────────────────────────────────────────────
@app.post("/api/validate")
async def validate(plan: PlanResponse):
    """Validate an existing plan using the same rules as the generation pipeline."""
    # Earlier this endpoint used stricter ad-hoc checks than plan generation,
    # which created contradictory pass/fail results. Reuse the shared validator.
    plan_dict = {
        "rooms": [
            {
                "id": room.id,
                "type": room.type,
                "label": room.label,
                "x": room.x,
                "y": room.y,
                "width": room.width,
                "height": room.height,
                "area": room.area,
                "polygon": [{"x": p.x, "y": p.y} for p in (room.polygon or [])],
            }
            for room in plan.rooms
        ]
    }

    is_valid, issues = validate_llm_plan(
        plan_dict,
        plan.plot.usable_width,
        plan.plot.usable_length,
        None,
    )

    return {
        "valid": is_valid,
        "issues": issues,
        "room_count": len(plan.rooms),
    }


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
