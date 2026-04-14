"""
NakshaNirman Backend — FastAPI application.
Local Ollama mode. GTX 1650 + 24GB RAM.

Endpoints:
  POST /api/generate       — Generate floor plan (LLM + BSP fallback)
  GET  /api/health         — Health check (verifies Ollama connectivity)
  POST /api/auth/login     — Simple in-memory auth
  POST /api/auth/signup    — Simple in-memory auth
  POST /api/auth/save-and-login — Legacy auth compat
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CORS_ORIGINS, LOCAL_LLM_BASE_URL, ARCHITECT_REASONING_ENABLED
from prompt_builder import build_user_prompt, build_system_prompt
from llm import call_openrouter_plan, call_openrouter, NAKSHA_SYSTEM_PROMPT
from layout_engine import generate_bsp_layout
from validators import validate_plan, fix_overlaps
from quality_engine import evaluate_real_life_fit, build_real_life_architect_note

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-14s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="NakshaNirman",
    description="AI Floor Plan Generator — Local Ollama Mode",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory user store (no MongoDB needed for local mode) ──────────
_users: dict[str, dict] = {}


# ── Request / Response Models ────────────────────────────────────────
class GenerateRequest(BaseModel):
    plot_width: float = Field(30, ge=15, le=200)
    plot_length: float = Field(40, ge=15, le=200)
    bedrooms: int = Field(2, ge=1, le=4)
    facing: str = Field("east")
    extras: list[str] = Field(default_factory=list)
    family_type: str = Field("nuclear")
    city: str = Field("")
    state: str = Field("")
    vastu: bool = Field(True)
    elder_friendly: bool = Field(False)
    work_from_home: bool = Field(False)
    notes: str = Field("")
    bathrooms_target: int = Field(0, ge=0, le=8)
    floors: int = Field(1, ge=1, le=4)
    design_style: str = Field("modern")
    kitchen_preference: str = Field("semi_open")
    parking_slots: int = Field(0, ge=0, le=4)
    vastu_priority: int = Field(3, ge=1, le=5)
    natural_light_priority: int = Field(3, ge=1, le=5)
    privacy_priority: int = Field(3, ge=1, le=5)
    storage_priority: int = Field(3, ge=1, le=5)
    strict_real_life: bool = Field(False)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    engine_mode: str = Field("gnn_advanced")
    total_area: float = Field(0)
    bathrooms: int = Field(0, ge=0, le=8)


class AuthRequest(BaseModel):
    user_id: str
    password: str
    email: str = ""
    full_name: str = ""


ROOM_TOKEN_ALIASES: dict[str, str] = {
    "puja": "pooja",
    "mandir": "pooja",
    "office": "study",
    "home_office": "study",
    "guest_room": "bedroom",
    "guest_bedroom": "bedroom",
    "common_bath": "bathroom",
    "common_bathroom": "bathroom",
    "wc": "bathroom",
    "stairs": "staircase",
}


def _normalize_room_token(raw: str) -> str:
    token = str(raw or "").strip().lower().replace(" ", "_")
    token = token.replace("-", "_")
    return ROOM_TOKEN_ALIASES.get(token, token)


def _normalize_tag_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").split(",")
    out = []
    seen = set()
    for item in items:
        token = _normalize_room_token(str(item or ""))
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _merge_practical_program(data: dict[str, Any]) -> None:
    extras = {_normalize_room_token(x) for x in data.get("extras", []) if str(x or "").strip()}
    must_have = set(_normalize_tag_list(data.get("must_have", [])))
    avoid = set(_normalize_tag_list(data.get("avoid", [])))

    known_extras = {
        "pooja", "study", "store", "balcony", "garage", "utility", "foyer", "staircase"
    }

    for token in must_have:
        if token in known_extras:
            extras.add(token)

    for token in avoid:
        if token in extras:
            extras.remove(token)

    data["must_have"] = sorted(must_have)
    data["avoid"] = sorted(avoid)
    data["extras"] = sorted(extras)


def _remove_corridor_rooms(plan: dict[str, Any]) -> int:
    rooms = plan.get("rooms")
    if not isinstance(rooms, list):
        return 0

    kept: list[Any] = []
    removed_ids: set[str] = set()
    removed = 0

    for room in rooms:
        if isinstance(room, dict) and str(room.get("type", "")).strip().lower() == "corridor":
            removed += 1
            rid = str(room.get("id", "")).strip()
            if rid:
                removed_ids.add(rid)
            continue
        kept.append(room)

    if removed == 0:
        return 0

    plan["rooms"] = kept

    # Drop door/window entries tied to removed corridor ids.
    for key in ("doors", "windows"):
        values = plan.get(key)
        if not isinstance(values, list):
            continue
        filtered: list[Any] = []
        for item in values:
            if not isinstance(item, dict):
                filtered.append(item)
                continue
            room_id = str(item.get("room_id", "")).strip()
            if room_id and room_id in removed_ids:
                continue
            filtered.append(item)
        plan[key] = filtered

    return removed


# ── Health ───────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Check if backend and Ollama are running."""
    ollama_ok = False
    ollama_msg = "not checked"

    try:
        # Check Ollama is reachable
        base = LOCAL_LLM_BASE_URL.replace("/v1", "")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(base)
        if resp.status_code == 200:
            ollama_ok = True
            ollama_msg = "running"
        else:
            ollama_msg = f"responded with {resp.status_code}"
    except httpx.ConnectError:
        ollama_msg = "not reachable — is Ollama running?"
    except Exception as e:
        ollama_msg = f"error: {str(e)[:80]}"

    return {
        "status": "ok" if ollama_ok else "degraded",
        "backend": "running",
        "ollama": ollama_msg,
        "ollama_url": LOCAL_LLM_BASE_URL,
    }


# ── Generate Floor Plan ─────────────────────────────────────────────
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """
    Main generation endpoint.

    Flow:
    1. Build user prompt from form data
    2. (Optional) Call advisory LLM for architect reasoning
    3. Call LLM for floor plan JSON
    4. Validate the plan
    5. If invalid, try to fix overlaps
    6. If still invalid, fall back to BSP engine
    7. Return final plan
    """
    start_time = time.time()
    data = req.model_dump()
    _merge_practical_program(data)
    strict_mode = bool(data.get("strict_real_life", False))
    engine_mode = str(data.get("engine_mode", "gnn_advanced") or "gnn_advanced").strip().lower()
    force_bsp = engine_mode in {"bsp", "deterministic", "rule_based", "local", "practical_strict", "strict"}
    reasoning_trace: list[str] = []

    log.info(
        "Generate: %sx%s %dBHK %s-facing",
        data["plot_width"], data["plot_length"],
        data["bedrooms"], data["facing"],
    )

    # Step 1: Build prompts
    user_prompt = build_user_prompt(data)
    system_prompt = build_system_prompt(data)
    reasoning_trace.append(
        f"Input: {data['plot_width']:.0f}x{data['plot_length']:.0f}, "
        f"{data['bedrooms']}BHK, {data['facing']}-facing"
    )

    # Step 2: Advisory reasoning (optional)
    advisory = {}

    # Step 3: Try LLM generation (unless deterministic mode is requested)
    plan = None
    generation_method = "bsp"

    if force_bsp:
        reasoning_trace.append(
            f"Engine mode '{engine_mode}' selected — skipping LLM and using deterministic BSP"
        )
    else:
        if ARCHITECT_REASONING_ENABLED:
            try:
                reasoning_trace.append("Running architect reasoning analysis...")
                advisory = await call_openrouter(system_prompt, user_prompt)
                if advisory.get("design_strategy"):
                    reasoning_trace.append(f"Strategy: {advisory['design_strategy'][:120]}")
            except Exception as e:
                log.warning("Advisory failed (non-fatal): %s", str(e)[:80])
                reasoning_trace.append("Advisory reasoning skipped (non-fatal)")

        try:
            reasoning_trace.append("Calling local LLM for floor plan generation...")
            # Keep backward compatibility for monkeypatched tests using old signature.
            try:
                plan = await call_openrouter_plan(
                    system_prompt,
                    user_prompt,
                    advisory=advisory,
                    request_data=data,
                )
            except TypeError:
                plan = await call_openrouter_plan(system_prompt, user_prompt)
            generation_method = "llm"
            reasoning_trace.append(
                f"LLM returned {len(plan.get('rooms', []))} rooms"
            )
        except Exception as e:
            log.warning("LLM generation failed: %s", str(e)[:150])
            reasoning_trace.append(f"LLM failed: {str(e)[:100]}")

    # Step 4: Validate
    if plan and isinstance(plan.get("rooms"), list) and len(plan["rooms"]) >= 2:
        validation = validate_plan(plan, data["plot_width"], data["plot_length"])

        if not validation["valid"]:
            reasoning_trace.append(
                f"Validation issues: {len(validation['issues'])} "
                f"(overlaps={len(validation['overlap_pairs'])}, "
                f"boundary={len(validation['boundary_violations'])}, "
                f"size={len(validation['size_violations'])})"
            )

            # Step 5: Try to fix overlaps
            if not validation["law1_ok"]:
                plan = fix_overlaps(plan, data["plot_width"], data["plot_length"])
                validation = validate_plan(plan, data["plot_width"], data["plot_length"])

                if validation["law1_ok"]:
                    reasoning_trace.append("Overlaps fixed successfully")
                else:
                    reasoning_trace.append("Could not fix all overlaps — using BSP fallback")
                    plan = None

            # If boundary or size issues, still use the plan but note issues
            if plan and (not validation["law2_ok"] or not validation["law3_ok"]):
                if generation_method == "llm":
                    reasoning_trace.append(
                        "LLM layout has boundary/size violations — switching to BSP fallback"
                    )
                    plan = None
                else:
                    reasoning_trace.append(
                        "Minor validation issues (boundary/size) — plan usable"
                    )
        else:
            reasoning_trace.append("Validation passed — all 3 laws satisfied")
    else:
        if plan:
            reasoning_trace.append("LLM plan has too few rooms — using BSP fallback")
        plan = None

    # Step 6: BSP fallback
    if plan is None:
        reasoning_trace.append("Generating plan with BSP layout engine...")
        plan = generate_bsp_layout(
            plot_width=data["plot_width"],
            plot_length=data["plot_length"],
            bedrooms=data["bedrooms"],
            facing=data["facing"],
            extras=data["extras"],
            family_type=data["family_type"],
            bathrooms_target=data.get("bathrooms_target", 0),
            work_from_home=data.get("work_from_home", False),
            elder_friendly=data.get("elder_friendly", False),
            floors=data.get("floors", 1),
            parking_slots=data.get("parking_slots", 0),
            kitchen_preference=data.get("kitchen_preference", "semi_open"),
            must_have=data.get("must_have", []),
            avoid=data.get("avoid", []),
        )
        generation_method = "bsp"
        reasoning_trace.append(
            f"BSP generated {len(plan.get('rooms', []))} rooms"
        )

    removed_corridors = _remove_corridor_rooms(plan)
    if removed_corridors:
        reasoning_trace.append(
            f"Removed {removed_corridors} corridor room(s) to maximize practical usable space"
        )

    # Step 6.1: Practical quality gate (real-life use case readiness)
    quality_report = evaluate_real_life_fit(plan, data)
    reasoning_trace.append(
        f"Real-life fit score: {quality_report.get('score', 0)}/100 "
        f"(grade {quality_report.get('grade', 'C')})"
    )

    def _strict_coverage_failed(report: dict[str, Any]) -> bool:
        if not strict_mode:
            return False
        cov = report.get("coverage", {}) if isinstance(report, dict) else {}
        required_full = ("core", "bedroom", "bathroom", "must_have", "avoid")
        return any(float(cov.get(k, 1.0) or 0.0) < 1.0 for k in required_full)

    quality_gate_threshold = 75 if strict_mode else 62
    strict_coverage_fail = _strict_coverage_failed(quality_report)

    if generation_method == "llm" and (
        quality_report.get("score", 0) < quality_gate_threshold or strict_coverage_fail
    ):
        if strict_coverage_fail:
            reasoning_trace.append(
                "Strict practical mode: coverage incomplete in LLM output "
                "(core/bed/bath/must-have/avoid)"
            )
        reasoning_trace.append(
            "LLM layout practical score is low — switching to enhanced BSP fallback"
        )
        plan = generate_bsp_layout(
            plot_width=data["plot_width"],
            plot_length=data["plot_length"],
            bedrooms=data["bedrooms"],
            facing=data["facing"],
            extras=data["extras"],
            family_type=data["family_type"],
            bathrooms_target=data.get("bathrooms_target", 0),
            work_from_home=data.get("work_from_home", False),
            elder_friendly=data.get("elder_friendly", False),
            floors=data.get("floors", 1),
            parking_slots=data.get("parking_slots", 0),
            kitchen_preference=data.get("kitchen_preference", "semi_open"),
            must_have=data.get("must_have", []),
            avoid=data.get("avoid", []),
        )
        removed_corridors = _remove_corridor_rooms(plan)
        if removed_corridors:
            reasoning_trace.append(
                f"Removed {removed_corridors} corridor room(s) from enhanced BSP layout"
            )
        generation_method = "bsp"
        quality_report = evaluate_real_life_fit(plan, data)
        strict_coverage_fail = _strict_coverage_failed(quality_report)
        reasoning_trace.append(
            f"Enhanced BSP score: {quality_report.get('score', 0)}/100"
        )
        if strict_coverage_fail:
            reasoning_trace.append(
                "Strict practical mode: some required constraints remain unmet for this plot/program"
            )

    for suggestion in quality_report.get("opportunities", [])[:2]:
        reasoning_trace.append(f"Refinement: {suggestion}")

    if strict_mode and (
        quality_report.get("score", 0) < quality_gate_threshold or strict_coverage_fail
    ):
        reasoning_trace.append(
            "Strict practical mode warning: constraints are difficult for current plot and program"
        )

    # Final geometry safety pass after all planner choices.
    final_validation = validate_plan(plan, data["plot_width"], data["plot_length"])
    if not final_validation["law1_ok"]:
        plan = fix_overlaps(plan, data["plot_width"], data["plot_length"])
        final_validation = validate_plan(plan, data["plot_width"], data["plot_length"])

    if not final_validation["valid"] and generation_method == "llm":
        reasoning_trace.append(
            "Final LLM geometry still invalid — forcing deterministic BSP fallback"
        )
        plan = generate_bsp_layout(
            plot_width=data["plot_width"],
            plot_length=data["plot_length"],
            bedrooms=data["bedrooms"],
            facing=data["facing"],
            extras=data["extras"],
            family_type=data["family_type"],
            bathrooms_target=data.get("bathrooms_target", 0),
            work_from_home=data.get("work_from_home", False),
            elder_friendly=data.get("elder_friendly", False),
            floors=data.get("floors", 1),
            parking_slots=data.get("parking_slots", 0),
            kitchen_preference=data.get("kitchen_preference", "semi_open"),
            must_have=data.get("must_have", []),
            avoid=data.get("avoid", []),
        )
        removed_corridors = _remove_corridor_rooms(plan)
        if removed_corridors:
            reasoning_trace.append(
                f"Removed {removed_corridors} corridor room(s) in final fallback"
            )
        generation_method = "bsp"
        quality_report = evaluate_real_life_fit(plan, data)
        strict_coverage_fail = _strict_coverage_failed(quality_report)
        final_validation = validate_plan(plan, data["plot_width"], data["plot_length"])
        reasoning_trace.append(
            f"Final fallback BSP score: {quality_report.get('score', 0)}/100"
        )

    if not final_validation["valid"]:
        reasoning_trace.append(
            "Final layout has remaining geometry warnings "
            f"(boundary={len(final_validation['boundary_violations'])}, "
            f"size={len(final_validation['size_violations'])})"
        )
    else:
        reasoning_trace.append("Final geometry check passed")

    # Step 7: Enrich plan with metadata
    elapsed = round(time.time() - start_time, 1)
    meta = plan.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        plan["metadata"] = meta

    plan["generation_method"] = generation_method
    plan.setdefault("vastu_score", meta.get("vastu_score", 65))
    plan.setdefault("adjacency_score", meta.get("adjacency_score", 70))

    base_note = str(plan.get("architect_note") or meta.get("architect_note") or "").strip()
    advisory_strategy = str(advisory.get("design_strategy") or "").strip()
    architect_note = build_real_life_architect_note(
        request_data=data,
        quality_report=quality_report,
        base_note=base_note,
        advisory_strategy=advisory_strategy,
    )

    plan["architect_note"] = architect_note
    plan["real_life_score"] = quality_report.get("score", 0)
    plan["quality_report"] = quality_report
    plan["model_alignment"] = quality_report.get("model_alignment", {})
    meta["architect_note"] = architect_note
    meta["quality_report"] = quality_report

    # Add plot info for frontend area calculations
    uw = data["plot_width"] - 7.0
    ul = data["plot_length"] - 11.5
    plan.setdefault("plot", {
        "width": data["plot_width"],
        "length": data["plot_length"],
        "usable_width": uw,
        "usable_length": ul,
        "boundary": plan.get("plot_boundary", [
            {"x": 0, "y": 0}, {"x": uw, "y": 0},
            {"x": uw, "y": ul}, {"x": 0, "y": ul},
        ]),
    })

    reasoning_trace.append(
        f"Done in {elapsed}s — {generation_method.upper()} engine, "
        f"{len(plan.get('rooms', []))} rooms"
    )
    plan["reasoning_trace"] = reasoning_trace

    log.info(
        "Generated %dBHK via %s in %.1fs (%d rooms)",
        data["bedrooms"], generation_method, elapsed, len(plan.get("rooms", [])),
    )

    return plan


# ── Auth Endpoints (simple in-memory) ────────────────────────────────
@app.post("/api/auth/signup")
async def signup(req: AuthRequest):
    """Register a new user (in-memory, no database)."""
    uid = req.user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id is required")
    if uid in _users:
        raise HTTPException(409, "User already exists")

    _users[uid] = {
        "user_id": uid,
        "password": req.password,
        "email": req.email,
        "full_name": req.full_name,
    }
    return {
        "status": "ok",
        "userId": uid,
        "fullName": req.full_name,
        "email": req.email,
    }


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    """Login (in-memory, no database). Auto-creates user if not found."""
    uid = req.user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id is required")

    user = _users.get(uid)
    if user and user["password"] != req.password:
        raise HTTPException(401, "Invalid password")

    # Auto-register if not found (convenience for local mode)
    if not user:
        _users[uid] = {
            "user_id": uid,
            "password": req.password,
            "email": req.email,
            "full_name": req.full_name or uid,
        }
        user = _users[uid]

    return {
        "status": "ok",
        "userId": uid,
        "fullName": user.get("full_name", uid),
        "email": user.get("email", ""),
    }


@app.post("/api/auth/save-and-login")
async def save_and_login(req: AuthRequest):
    """Legacy endpoint — same as login with auto-create."""
    return await login(req)


# ── Root redirect ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "app": "NakshaNirman", "mode": "local-ollama"}
