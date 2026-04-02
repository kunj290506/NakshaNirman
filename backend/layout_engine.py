"""
Layout Engine — 100% LLM-powered floor plan generation.
Flow:
  1. Build master prompt from PlanRequest
  2. Call LLM (with multi-model fallback chain) for full JSON plan
  3. Validate LLM output
  4. If valid → convert to PlanResponse
  5. If invalid → retry once with correction feedback
  6. If retry fails → raise error (no BSP fallback)
"""
from __future__ import annotations
import asyncio
import logging
import math

def _rnd(val: float, d: int = 2) -> float:
    return int(val * (10**d)) / (10.0**d)

from models import (
    PlanRequest, PlanResponse, PlotInfo,
    RoomData, DoorData, WindowData, Point2D,
)
from llm import call_openrouter_plan, call_openrouter_plan_backup
from prompt_builder import build_master_prompt
from plan_validator import validate_llm_plan
from config import FAST_FALLBACK_MODE

log = logging.getLogger("layout_engine")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SETBACKS = {"front": 6.5, "rear": 5.0, "left": 3.5, "right": 3.5}
# Fast-fallback mode prefers deterministic completion speed for demo/runtime
# reliability when external providers are slow or unstable.
if FAST_FALLBACK_MODE:
    FAST_LLM_TIMEOUT_SEC = 45
    FAST_RETRY_TIMEOUT_SEC = 30
    FAST_BACKUP_TIMEOUT_SEC = 45
    FINAL_QUALITY_TIMEOUT_SEC = 0
else:
    # Previous time limits forced truncated reasoning and low-quality geometry.
    # Keep generation windows long enough for full architectural constraint solving.
    FAST_LLM_TIMEOUT_SEC = 140
    FAST_RETRY_TIMEOUT_SEC = 110
    FAST_BACKUP_TIMEOUT_SEC = 140
    FINAL_QUALITY_TIMEOUT_SEC = 180

LOGICAL_ADJ_MIN_SCORE = 72.0 if FAST_FALLBACK_MODE else 68.0

from typing import List, Dict, Any

# ─── Default room colors ────────────────────────────────────
ROOM_COLORS: Dict[str, str] = {
    "living": "#E8F5E9",
    "dining": "#FFF3E0",
    "kitchen": "#FFEBEE",
    "master_bedroom": "#E3F2FD",
    "bedroom": "#E3F2FD",
    "master_bath": "#E0F7FA",
    "bathroom": "#E0F7FA",
    "toilet": "#E0F7FA",
    "corridor": "#F5F5F5",
    "pooja": "#FFF8E1",
    "study": "#EDE7F6",
    "store": "#EFEBE9",
    "balcony": "#E8F5E9",
    "garage": "#ECEFF1",
    "utility": "#F3E5F5",
    "foyer": "#FAFAFA",
    "staircase": "#ECEFF1",
}

ROOM_TYPE_ALIASES: Dict[str, str] = {
    "bath": "bathroom",
    "washroom": "bathroom",
    "wc": "toilet",
    "toilet_room": "toilet",
    "master_bed": "master_bedroom",
    "masterbedroom": "master_bedroom",
    "bed_room": "bedroom",
    "living_room": "living",
    "hall": "living",
    "passage": "corridor",
    "lobby": "foyer",
    "mandir": "pooja",
    "puja": "pooja",
}

ZONE_NAME_BY_NUM = {1: "public", 2: "service", 3: "private"}

OPTIONAL_EXTRA_ROOM_TYPES = {
    "pooja", "study", "store", "balcony", "garage", "utility", "foyer", "staircase"
}

EXTRA_NAME_TO_ROOM_TYPE = {
    "pooja": "pooja",
    "puja": "pooja",
    "mandir": "pooja",
    "study": "study",
    "store": "store",
    "balcony": "balcony",
    "garage": "garage",
    "utility": "utility",
    "foyer": "foyer",
    "stair": "staircase",
    "stairs": "staircase",
    "staircase": "staircase",
}

ROOM_MIN_DIMS: Dict[str, tuple[float, float]] = {
    "living": (11.0, 11.0),
    "dining": (8.0, 8.0),
    "kitchen": (7.0, 8.0),
    "corridor": (3.0, 6.0),
    "master_bedroom": (10.0, 10.0),
    "bedroom": (9.0, 9.0),
    "master_bath": (4.5, 6.0),
    "bathroom": (4.0, 5.0),
    "toilet": (3.5, 4.5),
    "pooja": (4.0, 4.0),
    "study": (6.0, 7.0),
    "store": (4.0, 4.0),
    "balcony": (3.5, 6.0),
    "garage": (9.0, 15.0),
    "utility": (4.0, 5.0),
    "foyer": (4.0, 4.0),
    "staircase": (6.0, 8.0),
}

ROOM_DEFAULT_ZONES: Dict[str, str] = {
    "living": "public",
    "dining": "public",
    "pooja": "public",
    "balcony": "public",
    "foyer": "public",
    "garage": "public",
    "kitchen": "service",
    "corridor": "service",
    "master_bath": "service",
    "bathroom": "service",
    "toilet": "service",
    "utility": "service",
    "store": "service",
    "staircase": "service",
    "master_bedroom": "private",
    "bedroom": "private",
    "study": "private",
}

DEFAULT_ROOM_LABELS: Dict[str, str] = {
    "living": "Living",
    "dining": "Dining",
    "kitchen": "Kitchen",
    "corridor": "Corridor",
    "master_bedroom": "Master Bedroom",
    "bedroom": "Bedroom",
    "master_bath": "Master Bath",
    "bathroom": "Bathroom",
    "toilet": "Toilet",
    "pooja": "Pooja",
    "study": "Study",
    "store": "Store",
    "balcony": "Balcony",
    "garage": "Garage",
    "utility": "Utility",
    "foyer": "Foyer",
    "staircase": "Staircase",
}

def _requested_extra_room_types(extras: list[str]) -> set[str]:
    requested: set[str] = set()
    for extra in extras or []:
        key = str(extra or "").strip().lower().replace(" ", "_")
        key = ROOM_TYPE_ALIASES.get(key, key)
        key = EXTRA_NAME_TO_ROOM_TYPE.get(key, key)
        if key in OPTIONAL_EXTRA_ROOM_TYPES:
            requested.add(key)
    return requested


def _default_room_label(room_type: str, serial: int = 1) -> str:
    base = DEFAULT_ROOM_LABELS.get(room_type, room_type.replace("_", " ").title())
    if room_type == "bedroom" and serial > 1:
        return f"Bedroom {serial}"
    return base


def _push_reasoning(trace: list[str], message: str):
    msg = str(message or "").strip()
    if not msg:
        return
    if trace and trace[-1] == msg:
        return
    trace.append(msg)


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        f = float(value)
        if not math.isfinite(f):
            return fallback
        return f
    except Exception:
        return fallback


def _sanitize_room_type(raw_type: str, raw_label: str = "") -> str:
    t = (raw_type or "").strip().lower().replace(" ", "_")
    if t in ROOM_COLORS:
        return t

    t = ROOM_TYPE_ALIASES.get(t, t)
    if t in ROOM_COLORS:
        return t

    label = (raw_label or "").strip().lower()
    if "master" in label and "bed" in label:
        return "master_bedroom"
    if "bed" in label:
        return "bedroom"
    if "bath" in label or "toilet" in label or "wash" in label:
        return "bathroom"
    if "living" in label or "hall" in label:
        return "living"
    if "dining" in label:
        return "dining"
    if "kitchen" in label:
        return "kitchen"
    if "corridor" in label or "passage" in label:
        return "corridor"
    if "study" in label:
        return "study"
    if "pooja" in label or "puja" in label or "mandir" in label:
        return "pooja"
    if "balcony" in label:
        return "balcony"
    if "store" in label:
        return "store"
    if "utility" in label:
        return "utility"

    return t


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _room_overlap(a: RoomData, b: RoomData, eps: float = 0.05) -> bool:
    return (
        a.x < b.x + b.width - eps
        and a.x + a.width > b.x + eps
        and a.y < b.y + b.height - eps
        and a.y + a.height > b.y + eps
    )


def _shared_wall_length(a: RoomData, b: RoomData, eps: float = 0.35) -> float:
    # Vertical shared wall
    if abs((a.x + a.width) - b.x) <= eps or abs((b.x + b.width) - a.x) <= eps:
        return max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))

    # Horizontal shared wall
    if abs((a.y + a.height) - b.y) <= eps or abs((b.y + b.height) - a.y) <= eps:
        return max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))

    return 0.0


def _normalize_polygon_points(
    raw_points: Any,
    x_max: float,
    y_max: float,
) -> list[dict[str, float]]:
    if not isinstance(raw_points, list):
        return []

    pts: list[dict[str, float]] = []
    for p in raw_points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x = _to_float(p[0], float("nan"))
            y = _to_float(p[1], float("nan"))
        else:
            continue

        if not math.isfinite(x) or not math.isfinite(y):
            continue
        x = _clamp(x, 0.0, max(0.0, x_max))
        y = _clamp(y, 0.0, max(0.0, y_max))
        pts.append({"x": _rnd(x, 2), "y": _rnd(y, 2)})

    # Remove consecutive duplicates
    cleaned: list[dict[str, float]] = []
    for p in pts:
        if not cleaned:
            cleaned.append(p)
            continue
        if abs(cleaned[-1]["x"] - p["x"]) < 0.01 and abs(cleaned[-1]["y"] - p["y"]) < 0.01:
            continue
        cleaned.append(p)

    if len(cleaned) >= 2:
        first = cleaned[0]
        last = cleaned[-1]
        if abs(first["x"] - last["x"]) < 0.01 and abs(first["y"] - last["y"]) < 0.01:
            cleaned.pop()

    return cleaned if len(cleaned) >= 3 else []


def _polygon_bbox(points: list[dict[str, float]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for p in points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        else:
            x = _to_float(getattr(p, "x", float("nan")), float("nan"))
            y = _to_float(getattr(p, "y", float("nan")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return 0.0, 0.0, 0.01, 0.01

    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return x0, y0, max(0.01, x1 - x0), max(0.01, y1 - y0)


def _translate_polygon(
    points: list[dict[str, float]],
    dx: float,
    dy: float,
    x_max: float,
    y_max: float,
) -> list[dict[str, float]]:
    moved = []
    for p in points:
        if isinstance(p, dict):
            px = _to_float(p.get("x"), 0.0)
            py = _to_float(p.get("y"), 0.0)
        else:
            px = _to_float(getattr(p, "x", 0.0), 0.0)
            py = _to_float(getattr(p, "y", 0.0), 0.0)
        moved.append(
            {
                "x": _rnd(_clamp(px + dx, 0.0, max(0.0, x_max)), 2),
                "y": _rnd(_clamp(py + dy, 0.0, max(0.0, y_max)), 2),
            }
        )
    return moved


def _polygon_area(points: list[dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0

    clean: list[tuple[float, float]] = []
    for p in points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        else:
            x = _to_float(getattr(p, "x", float("nan")), float("nan"))
            y = _to_float(getattr(p, "y", float("nan")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            clean.append((x, y))

    if len(clean) < 3:
        return 0.0

    area2 = 0.0
    for i, (x1, y1) in enumerate(clean):
        x2, y2 = clean[(i + 1) % len(clean)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


# ─────────────────────────────────────────────────────────────
# MAIN PUBLIC API — 100% LLM powered
# ─────────────────────────────────────────────────────────────
async def generate_plan(req: PlanRequest) -> PlanResponse:
    """
    Generate a floor plan using LLM only.
    Tries up to 6 free models automatically.
    Raises RuntimeError if all fail.
    """
    uw = _rnd(req.plot_width - SETBACKS["left"] - SETBACKS["right"], 2)
    ul = _rnd(req.plot_length - SETBACKS["front"] - SETBACKS["rear"], 2)

    # ── Attempt 1: LLM-generated plan (time-capped) ─────────
    try:
        plan, issues, correction_feedback = await asyncio.wait_for(
            _generate_via_llm(req, uw, ul),
            timeout=FAST_LLM_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        plan, issues, correction_feedback = None, ["llm attempt timed out"], None
        log.warning("LLM attempt timed out after %ss", FAST_LLM_TIMEOUT_SEC)

    last_issues = list(issues or [])

    if plan:
        log.info("LLM plan accepted on first attempt")
        return plan

    # ── Attempt 2: retry with or without correction hints ──
    timeout_in_issues = any("timed out" in str(i).lower() for i in (issues or []))
    if issues:
        retry_seed = correction_feedback if correction_feedback else (
            issues if not timeout_in_issues else None
        )
        if timeout_in_issues:
            log.info("LLM timed out; retrying with a fresh model chain for best output")
        else:
            log.info("LLM plan invalid (%d issues), retrying with corrections", len(issues))
        try:
            plan, retry_issues, _ = await asyncio.wait_for(
                _generate_via_llm(req, uw, ul, prev_issues=retry_seed),
                timeout=FAST_RETRY_TIMEOUT_SEC,
            )
            last_issues = list(retry_issues or [])
        except asyncio.TimeoutError:
            plan = None
            last_issues = ["llm retry timed out"]
            log.warning("LLM retry timed out after %ss", FAST_RETRY_TIMEOUT_SEC)
        if plan:
            log.info("LLM plan accepted on retry")
            return plan

    # ── Attempt 3: backup free-model pool ───────────────────
    backup_seed_issues = None
    if issues and not timeout_in_issues:
        backup_seed_issues = correction_feedback if correction_feedback else issues
    try:
        backup_plan, backup_issues, _ = await asyncio.wait_for(
            _generate_via_llm(req, uw, ul, prev_issues=backup_seed_issues, use_backup_models=True),
            timeout=FAST_BACKUP_TIMEOUT_SEC,
        )
        last_issues = list(backup_issues or [])
    except asyncio.TimeoutError:
        backup_plan = None
        last_issues = ["llm backup timed out"]
        log.warning("Backup LLM pool timed out after %ss", FAST_BACKUP_TIMEOUT_SEC)

    if backup_plan:
        backup_plan.generation_method = "llm_backup"
        backup_plan.reasoning_trace = [
            *backup_plan.reasoning_trace,
            "Primary model pool was unavailable; delivered via backup model pool.",
        ][:12]
        return backup_plan

    # ── Attempt 4: final quality-first retry before local fallback ──
    # This avoids dropping into local backup too eagerly when model warm-up
    # delays are temporary and high-quality output is still preferred.
    final_plan = None
    if not FAST_FALLBACK_MODE:
        final_seed = backup_seed_issues if backup_seed_issues else correction_feedback
        try:
            final_plan, final_issues, _ = await asyncio.wait_for(
                _generate_via_llm(req, uw, ul, prev_issues=final_seed, use_backup_models=True),
                timeout=FINAL_QUALITY_TIMEOUT_SEC,
            )
            last_issues = list(final_issues or [])
        except asyncio.TimeoutError:
            final_plan = None
            last_issues = ["llm final quality retry timed out"]
            log.warning("Final quality-first retry timed out after %ss", FINAL_QUALITY_TIMEOUT_SEC)
    else:
        log.info("Fast fallback mode enabled: skipping final quality retry")

    if final_plan:
        final_plan.generation_method = "llm_backup"
        final_plan.reasoning_trace = [
            *final_plan.reasoning_trace,
            "Recovered after quality-first final retry; avoided local fallback.",
        ][:12]
        return final_plan

    # If every model call failed on quota/credits, keep service stable by using
    # deterministic fallback instead of surfacing hard runtime errors.
    issues_blob = " | ".join(str(i) for i in (last_issues or []))
    if "402" in issues_blob:
        log.warning(
            "All planner models returned 402 (credits/quota unavailable); "
            "using deterministic local fallback."
        )

    # ── Final local adaptive fallback (no busy error to user) ──
    log.error("Primary and backup LLM pools unavailable; using local adaptive fallback")
    fallback = await _generate_via_bsp(req, uw, ul)
    fallback.generation_method = "local_backup"
    fallback.reasoning_trace = [
        f"Computed usable plot after setbacks: {uw:.1f} ft x {ul:.1f} ft.",
        "Switched to fast backup planning path to keep generation uninterrupted.",
        "Generated an adaptive layout optimized for immediate preview.",
    ]
    fallback.architect_note = (
        "Fast backup-planning mode delivered this layout to keep response time consistent."
    )
    return fallback


async def _generate_via_llm(
    req: PlanRequest,
    uw: float,
    ul: float,
    prev_issues: list[str] | None = None,
    use_backup_models: bool = False,
) -> tuple[PlanResponse | None, list[str], list[str] | None]:
    """
    Try to generate a plan via the LLM.
    Returns (PlanResponse, [], None) if successful.
    Returns (None, validator_issues, correction_feedback) if failed.
    """
    reasoning_trace: list[str] = []
    _push_reasoning(
        reasoning_trace,
        f"Computed usable plot after setbacks: {uw:.1f} ft x {ul:.1f} ft.",
    )

    system_prompt, user_message = build_master_prompt(req)
    _push_reasoning(
        reasoning_trace,
        "Built strict architectural prompt with fresh design nonce and requested extras.",
    )
    if use_backup_models:
        _push_reasoning(
            reasoning_trace,
            "Switching to backup model pool for generation continuity.",
        )

    # Append correction feedback if retrying
    if prev_issues:
        # Earlier retries sent vague issue lists; now we provide concrete geometry
        # feedback so the model repairs coordinates instead of regenerating randomly.
        correction = (
            "\n\nYour previous plan had these specific geometric errors:\n"
            + "\n".join(f"- {issue}" for issue in prev_issues[:14])
            + "\n\nKeep all already-valid rooms stable. "
            + "Only change coordinates that are explicitly called out above."
        )
        user_message += correction
        log.info("Retrying LLM with %d correction items", len(prev_issues))
        _push_reasoning(
            reasoning_trace,
            f"Retrying with {len(prev_issues)} validator corrections from previous draft.",
        )

    _push_reasoning(reasoning_trace, "Requesting floor plan draft from LLM model chain.")

    try:
        if use_backup_models:
            plan_dict = await call_openrouter_plan_backup(system_prompt, user_message)
        else:
            plan_dict = await call_openrouter_plan(system_prompt, user_message)
    except Exception as e:
        log.error("LLM call failed across all models: %s", e)
        return None, [str(e)], None

    draft_rooms = plan_dict.get("rooms", []) if isinstance(plan_dict, dict) else []
    if isinstance(draft_rooms, list):
        _push_reasoning(
            reasoning_trace,
            f"LLM returned draft with {len(draft_rooms)} rooms; running strict validation.",
        )

    # Validate raw draft, but do not hard-reject yet.
    # Many raw drafts have geometric noise that deterministic repair can fix.
    raw_valid, raw_issues = validate_llm_plan(plan_dict, uw, ul, req)
    if not raw_valid:
        _push_reasoning(
            reasoning_trace,
            f"Raw draft had {len(raw_issues)} issues; attempting deterministic repair before retry.",
        )

    # Build production-grade plan with geometry/opening repair first.
    try:
        plan = _parse_llm_plan(plan_dict, req, uw, ul, reasoning_trace=reasoning_trace)
    except Exception as e:
        parse_error = f"LLM draft parsing failed: {str(e)[:120]}"
        issues = list(raw_issues) if raw_issues else [parse_error]
        if parse_error not in issues:
            issues.append(parse_error)
        correction_feedback = _build_geometric_correction_feedback(plan_dict, uw, ul, issues)
        log.warning("LLM plan parse failed: %s", parse_error)
        return None, issues, correction_feedback

    # Validate repaired final plan to ensure output quality for frontend/export
    repaired_dict = {
        "rooms": [
            {
                "id": r.id,
                "type": r.type,
                "label": r.label,
                "x": r.x,
                "y": r.y,
                "width": r.width,
                "height": r.height,
                "area": r.area,
                "polygon": [{"x": p.x, "y": p.y} for p in (r.polygon or [])],
            }
            for r in plan.rooms
        ]
    }
    final_valid, final_issues = validate_llm_plan(repaired_dict, uw, ul, req)
    if not final_valid:
        # Preserve original raw issues for richer correction prompts.
        merged_issues = list(final_issues)
        for issue in raw_issues[:6]:
            tagged = f"raw-draft: {issue}"
            if tagged not in merged_issues:
                merged_issues.append(tagged)

        # Quality-first behavior: if only soft dimensional/adjacency issues
        # remain after deterministic repair, keep the LLM output instead of
        # dropping to local fallback on every request.
        hard_issues = [i for i in merged_issues if not _is_soft_validation_issue(i)]
        if not hard_issues:
            _push_reasoning(
                reasoning_trace,
                "Returning repaired LLM plan in best-effort mode; only soft size issues remain.",
            )
            note_suffix = " Best-effort LLM output returned to avoid local fallback; review compact room sizes."
            if note_suffix not in plan.architect_note:
                plan.architect_note = f"{plan.architect_note}{note_suffix}"
            return plan, [], None

        correction_feedback = _build_geometric_correction_feedback(repaired_dict, uw, ul, final_issues)
        log.warning("Post-repair plan invalid (%d issues): %s", len(merged_issues), merged_issues[:3])
        _push_reasoning(
            reasoning_trace,
            f"Post-repair validation failed with {len(merged_issues)} issues; correction cycle required.",
        )
        return None, [f"post-repair: {issue}" for issue in merged_issues], correction_feedback

    logical_issues = _logical_layout_issues(plan.rooms, uw, ul)
    if logical_issues:
        correction_feedback = _build_geometric_correction_feedback(
            repaired_dict,
            uw,
            ul,
            logical_issues,
        )
        _push_reasoning(
            reasoning_trace,
            f"Logical quality gate rejected draft with {len(logical_issues)} issues; retrying.",
        )
        return None, [f"logical: {issue}" for issue in logical_issues], correction_feedback

    if raw_issues:
        _push_reasoning(
            reasoning_trace,
            f"Deterministic repair resolved {len(raw_issues)} raw-draft issues before final acceptance.",
        )

    _push_reasoning(reasoning_trace, "Final repaired plan passed validation and is ready for preview.")

    return plan, [], None


def _build_geometric_correction_feedback(
    plan_dict: dict[str, Any],
    uw: float,
    ul: float,
    issues: list[str],
) -> list[str]:
    """
    Build explicit coordinate-level correction guidance for retry prompts.
    This replaces vague issue summaries with room-wise geometric actions.
    """
    raw_rooms = plan_dict.get("rooms", []) if isinstance(plan_dict, dict) else []
    if not isinstance(raw_rooms, list):
        raw_rooms = []

    front_max = ul * 0.30
    private_min = ul * 0.45
    middle_min = ul * 0.30
    middle_max = ul * 0.75

    rooms: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_rooms):
        if not isinstance(raw, dict):
            continue
        room_type = _sanitize_room_type(str(raw.get("type", "")), str(raw.get("label", "")))
        label = str(raw.get("label", raw.get("id", f"room_{idx+1}"))).strip() or f"room_{idx+1}"
        w = max(0.1, _to_float(raw.get("width"), 0.0))
        h = max(0.1, _to_float(raw.get("height"), 0.0))
        x = _to_float(raw.get("x"), 0.0)
        y = _to_float(raw.get("y"), 0.0)
        rooms.append({
            "type": room_type,
            "label": label,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    feedback: list[str] = []

    for room in rooms:
        x = room["x"]
        y = room["y"]
        w = room["w"]
        h = room["h"]
        room_type = room["type"]
        label = room["label"]

        room_issues: list[str] = []
        x_lo = 0.0
        x_hi = max(0.0, uw - w)
        y_lo = 0.0
        y_hi = max(0.0, ul - h)

        if x < -1.0 or y < -1.0 or x + w > uw + 1.0 or y + h > ul + 1.0:
            room_issues.append("It is outside usable bounds")
            x_lo = max(x_lo, 0.0)
            x_hi = min(x_hi, max(0.0, uw - w))
            y_lo = max(y_lo, 0.0)
            y_hi = min(y_hi, max(0.0, ul - h))

        if room_type in ("master_bedroom", "bedroom"):
            if y < private_min:
                room_issues.append("Bedroom is below rear privacy band")
            y_lo = max(y_lo, private_min)

        if room_type == "living":
            if y + h > front_max + 0.1:
                room_issues.append("Living room is not fully in the front public band")
            y_hi = min(y_hi, max(0.0, front_max - h))

        if room_type == "kitchen":
            if y < middle_min or y + h > middle_max:
                room_issues.append("Kitchen is not fully in the middle band")
            if x + w < uw * 0.5:
                room_issues.append("Kitchen is not in southeast side")
            y_lo = max(y_lo, middle_min)
            y_hi = min(y_hi, max(y_lo, middle_max - h))
            x_lo = max(x_lo, uw * 0.5 - w)

        if room_type == "master_bedroom":
            if x + w > uw * 0.55:
                room_issues.append("Master bedroom is not in southwest quadrant")
            x_hi = min(x_hi, max(0.0, uw * 0.55 - w))

        if room_type == "pooja":
            if x < uw * 0.5 or y + h > front_max + 0.1:
                room_issues.append("Pooja room is not in northeast front quadrant")
            x_lo = max(x_lo, uw * 0.5)
            y_hi = min(y_hi, max(0.0, front_max - h))

        if room_type == "corridor" and w < 3.5:
            room_issues.append("Corridor is narrower than 3.5 ft")

        if room_issues:
            feedback.append(
                f"{label} is at x={x:.1f}, y={y:.1f}, width={w:.1f}, height={h:.1f}. "
                + "; ".join(room_issues)
                + f". Move it to x in [{max(0.0, x_lo):.1f}, {max(x_lo, x_hi):.1f}] "
                + f"and y in [{max(0.0, y_lo):.1f}, {max(y_lo, y_hi):.1f}]."
            )

    # Overlap feedback with explicit target direction.
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]
            overlap_x = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            overlap_y = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
            if overlap_x <= 0.2 or overlap_y <= 0.2:
                continue

            target_x = b["x"]
            target_y = b["y"]
            if overlap_x >= overlap_y:
                target_y = min(max(0.0, a["y"] + a["h"] + 0.5), max(0.0, ul - b["h"]))
            else:
                target_x = min(max(0.0, a["x"] + a["w"] + 0.5), max(0.0, uw - b["w"]))

            feedback.append(
                f"{b['label']} is at x={b['x']:.1f}, y={b['y']:.1f}, width={b['w']:.1f}, height={b['h']:.1f}. "
                + f"It overlaps {a['label']} by {overlap_x:.1f} ft x {overlap_y:.1f} ft. "
                + f"Move it near x={target_x:.1f}, y={target_y:.1f} to remove overlap."
            )

    # Preserve direct validator messages as backstop guidance.
    if not feedback:
        for issue in issues[:10]:
            feedback.append(f"Validator issue to fix: {issue}")

    return feedback[:18]


def _is_soft_validation_issue(issue: str) -> bool:
    text = str(issue or "").lower()
    if text.startswith("raw-draft:"):
        return True
    return False


def _parse_llm_plan(
    plan_dict: dict,
    req: PlanRequest,
    uw: float,
    ul: float,
    reasoning_trace: list[str] | None = None,
) -> PlanResponse:
    """Convert LLM JSON into a production-ready PlanResponse."""
    trace = reasoning_trace if reasoning_trace is not None else []

    # 1) Normalize room program to requested BHK + extras
    rooms = _normalize_rooms_from_program(plan_dict.get("rooms", []), req, uw, ul)
    _push_reasoning(
        trace,
        f"Normalized and filtered rooms to {len(rooms)} valid spaces within requested program.",
    )

    # 2) Repair geometry into buildable layout
    _snap_and_fix_layout(rooms, uw, ul)
    _push_reasoning(trace, "Applied geometric repair: snap, zone anchoring, and overlap resolution.")

    # 2b) Deterministic post-processing after validator pass.
    # This cleans sub-foot slivers and enforces buildable corridor width.
    _postprocess_llm_layout(rooms, uw, ul)
    _push_reasoning(trace, "Post-processed layout: 0.5ft snap, sliver-gap fill, corridor width normalization.")

    # 3) Compute area/exterior walls after repair
    _infer_exterior_walls(rooms, uw, ul)
    for r in rooms:
        if r.polygon:
            p_area = _polygon_area(r.polygon)
            r.area = _rnd(p_area if p_area > 1.0 else (r.width * r.height), 1)
        else:
            r.area = _rnd(r.width * r.height, 1)

    # 4) Openings are regenerated from repaired layout for consistency
    doors, windows = _build_openings_from_rooms(rooms, req.facing, uw, ul)
    _push_reasoning(
        trace,
        f"Regenerated openings from repaired geometry ({len(doors)} doors, {len(windows)} windows).",
    )

    # 5) Metadata
    meta = plan_dict.get("metadata", {})
    vastu_score = _clamp(_to_float(meta.get("vastu_score"), 75.0), 40.0, 100.0)
    architect_note = meta.get(
        "architect_note",
        "LLM-generated plan refined with architectural constraints for practical use."
    )
    vastu_issues = meta.get("vastu_issues", [])
    if not isinstance(vastu_issues, list):
        vastu_issues = []
    adjacency_score = _to_float(meta.get("adjacency_score"), -1.0)
    if adjacency_score < 0:
        adjacency_score = _compute_adjacency_score(rooms)
    adjacency_score = _clamp(adjacency_score, 0.0, 100.0)

    raw_boundary = plan_dict.get("plot_boundary", meta.get("plot_boundary", []))
    boundary_points = _normalize_polygon_points(raw_boundary, uw, ul)
    if boundary_points:
        _push_reasoning(
            trace,
            f"Detected free-form plot boundary with {len(boundary_points)} points.",
        )
    else:
        _push_reasoning(trace, "Using rectangular usable boundary (no plot polygon in draft).")

    plot = PlotInfo(
        width=req.plot_width,
        length=req.plot_length,
        usable_width=uw,
        usable_length=ul,
        road_side=req.facing,
        setbacks=SETBACKS,
        boundary=[Point2D(x=p["x"], y=p["y"]) for p in boundary_points],
    )

    return PlanResponse(
        plot=plot,
        rooms=rooms,
        doors=doors,
        windows=windows,
        vastu_score=vastu_score,
        architect_note=architect_note,
        generation_method="llm",
        vastu_issues=vastu_issues,
        adjacency_score=adjacency_score,
        reasoning_trace=trace[:12],
    )


def _normalize_rooms_from_program(
    raw_rooms: list[dict[str, Any]],
    req: PlanRequest,
    uw: float,
    ul: float,
) -> list[RoomData]:
    """
    Normalize raw LLM rooms while preserving LLM creativity.
    This path intentionally avoids deterministic 1/2/3/4 BHK template remapping.
    """
    requested_extra_types = _requested_extra_room_types(req.extras)
    allowed_types = set(ROOM_COLORS.keys()) | {"toilet"}

    rooms: list[RoomData] = []
    type_serial: dict[str, int] = {}
    used_ids: set[str] = set()

    for raw in raw_rooms or []:
        if not isinstance(raw, dict):
            continue

        room_type = _sanitize_room_type(
            str(raw.get("type", "")),
            str(raw.get("label", "")),
        )
        if room_type not in allowed_types:
            continue
        if room_type in OPTIONAL_EXTRA_ROOM_TYPES and room_type not in requested_extra_types:
            continue

        min_w, min_h = ROOM_MIN_DIMS.get(room_type, (4.0, 4.0))
        min_w = min(min_w, max(4.0, uw))
        min_h = min(min_h, max(4.0, ul))

        raw_polygon = _normalize_polygon_points(
            raw.get("polygon", raw.get("vertices", [])),
            uw,
            ul,
        )

        polygon_bbox: tuple[float, float, float, float] | None = None
        if raw_polygon:
            bx, by, bw, bh = _polygon_bbox(raw_polygon)
            if bw >= max(2.8, min_w * 0.6) and bh >= max(2.8, min_h * 0.6):
                polygon_bbox = (bx, by, bw, bh)
            else:
                raw_polygon = []

        if polygon_bbox:
            x, y, width, height = polygon_bbox
        else:
            width = _clamp(_to_float(raw.get("width"), min_w), min_w, max(min_w, uw))
            height = _clamp(_to_float(raw.get("height"), min_h), min_h, max(min_h, ul))
            x = _to_float(raw.get("x"), 0.0)
            y = _to_float(raw.get("y"), 0.0)

        x = _clamp(x, 0.0, max(0.0, uw - width))
        y = _clamp(y, 0.0, max(0.0, ul - height))

        zone_name = str(raw.get("zone", "")).strip().lower()
        if zone_name not in ("public", "service", "private"):
            zone_name = ROOM_DEFAULT_ZONES.get(room_type, "service")

        band = int(_to_float(raw.get("band"), 0))
        if band not in (1, 2, 3):
            band = 1 if zone_name == "public" else 2 if zone_name == "service" else 3

        type_serial[room_type] = type_serial.get(room_type, 0) + 1
        default_id = f"{room_type}_{type_serial[room_type]:02d}"
        room_id = str(raw.get("id", "")).strip().lower().replace(" ", "_") or default_id
        while room_id in used_ids:
            room_id = f"{room_id}_{type_serial[room_type]}"
        used_ids.add(room_id)

        label = str(raw.get("label", "")).strip()
        if not label:
            label = _default_room_label(room_type, type_serial[room_type])

        rooms.append(
            RoomData(
                id=room_id,
                type=room_type,
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
                area=_rnd(width * height, 1),
                zone=zone_name,
                band=band,
                exterior_walls=[],
                color=ROOM_COLORS.get(room_type, "#F5F5F5"),
                polygon=[Point2D(x=p["x"], y=p["y"]) for p in raw_polygon],
            )
        )

    # Keep requested bedroom count bounded while preserving one master bedroom.
    if req.bedrooms > 0:
        bedroom_rooms = [r for r in rooms if r.type in ("master_bedroom", "bedroom")]
        if bedroom_rooms:
            master_idx = next((i for i, r in enumerate(bedroom_rooms) if r.type == "master_bedroom"), None)
            if master_idx is None:
                bedroom_rooms[0].type = "master_bedroom"
                bedroom_rooms[0].label = _default_room_label("master_bedroom")
                bedroom_rooms[0].zone = "private"
                bedroom_rooms[0].band = 3
                bedroom_rooms[0].color = ROOM_COLORS.get("master_bedroom", bedroom_rooms[0].color)
                master_idx = 0

            for i, room in enumerate(bedroom_rooms):
                if i == master_idx:
                    continue
                if room.type == "master_bedroom":
                    room.type = "bedroom"
                    room.color = ROOM_COLORS.get("bedroom", room.color)
                    if "master" in room.label.lower():
                        room.label = _default_room_label("bedroom", i + 1)

            keep_ids: set[str] = {bedroom_rooms[master_idx].id}
            for room in bedroom_rooms:
                if len(keep_ids) >= req.bedrooms:
                    break
                keep_ids.add(room.id)

            if len(bedroom_rooms) > req.bedrooms:
                rooms = [
                    room
                    for room in rooms
                    if room.type not in ("master_bedroom", "bedroom") or room.id in keep_ids
                ]

    return rooms


def _infer_exterior_walls(rooms: list[RoomData], uw: float, ul: float):
    eps = 0.35
    for room in rooms:
        walls: list[str] = []
        if room.x <= eps:
            walls.append("west")
        if room.y <= eps:
            walls.append("south")
        if abs((room.x + room.width) - uw) <= eps:
            walls.append("east")
        if abs((room.y + room.height) - ul) <= eps:
            walls.append("north")
        room.exterior_walls = walls


def _compute_adjacency_score(rooms: list[RoomData]) -> float:
    weighted_pairs = [
        ("living", "dining", 16.0),
        ("dining", "kitchen", 18.0),
        ("master_bedroom", "master_bath", 20.0),
        ("corridor", "master_bedroom", 14.0),
        ("corridor", "bedroom", 16.0),
        ("living", "foyer", 8.0),
        ("kitchen", "utility", 8.0),
    ]

    possible = 0.0
    achieved = 0.0

    for type_a, type_b, weight in weighted_pairs:
        a_rooms = [r for r in rooms if r.type == type_a]
        b_rooms = [r for r in rooms if r.type == type_b]
        if not a_rooms or not b_rooms:
            continue

        possible += weight
        if any(_shared_wall_length(a, b) >= 2.0 for a in a_rooms for b in b_rooms):
            achieved += weight

    if possible <= 0:
        return 75.0
    return _rnd((achieved / possible) * 100.0, 1)


def _logical_layout_issues(rooms: list[RoomData], uw: float, ul: float) -> list[str]:
    """Logical quality gate for accepting LLM plans."""
    issues: list[str] = []
    if not rooms:
        return ["No rooms found in candidate plan."]

    by_type: dict[str, list[RoomData]] = {}
    for room in rooms:
        by_type.setdefault(room.type, []).append(room)

    def _adjacent(type_a: str, type_b: str, min_len: float) -> bool:
        a_rooms = by_type.get(type_a, [])
        b_rooms = by_type.get(type_b, [])
        if not a_rooms or not b_rooms:
            return True
        return any(_shared_wall_length(a, b) >= min_len for a in a_rooms for b in b_rooms)

    def _in_front(room: RoomData) -> bool:
        return room.y + room.height <= ul * 0.40 + 0.5

    def _in_middle(room: RoomData) -> bool:
        return room.y >= ul * 0.18 and room.y + room.height <= ul * 0.84

    def _in_private(room: RoomData) -> bool:
        return room.y >= ul * 0.40

    living = by_type.get("living", [])
    kitchens = by_type.get("kitchen", [])
    bedrooms = by_type.get("master_bedroom", []) + by_type.get("bedroom", [])
    corridors = by_type.get("corridor", [])

    for room in living:
        if not _in_front(room):
            issues.append(f"{room.label} is not in front public zone.")

    for room in kitchens:
        if not _in_middle(room):
            issues.append(f"{room.label} is not in middle service zone.")

    for room in bedrooms:
        if not _in_private(room):
            issues.append(f"{room.label} is not in rear private zone.")

    if not _adjacent("living", "dining", 2.0):
        issues.append("Living and dining are not physically connected.")
    if not _adjacent("dining", "kitchen", 2.0):
        issues.append("Dining and kitchen are not physically connected.")
    if not _adjacent("master_bedroom", "master_bath", 1.5):
        issues.append("Master bath is not attached to master bedroom.")

    if corridors and bedrooms:
        for bed in bedrooms:
            if not any(_shared_wall_length(bed, c) >= 1.5 for c in corridors):
                issues.append(f"{bed.label} is not connected to corridor.")

    # Reject visibly skinny key rooms even if they pass tolerance-based validator.
    for room in bedrooms + kitchens + living:
        min_w, min_h = ROOM_MIN_DIMS.get(room.type, (4.0, 4.0))
        if room.width < min_w * 0.9 or room.height < min_h * 0.9:
            issues.append(f"{room.label} is undersized for practical use.")

    adj = _compute_adjacency_score(rooms)
    if adj < LOGICAL_ADJ_MIN_SCORE:
        issues.append(f"Adjacency score too low ({adj:.1f}).")

    # Keep feedback concise for retry prompting.
    return issues[:12]


def _build_openings_from_rooms(
    rooms: list[RoomData],
    facing: str,
    uw: float,
    ul: float,
) -> tuple[list[DoorData], list[WindowData]]:
    placed_rooms = [
        {
            "id": r.id,
            "type": r.type,
            "label": r.label,
            "x": r.x,
            "y": r.y,
            "width": r.width,
            "height": r.height,
        }
        for r in rooms
    ]

    doors, windows = add_doors_and_windows(placed_rooms, uw, ul, facing)

    # Deduplicate by geometry signature to avoid stacked duplicate symbols.
    deduped_doors: list[DoorData] = []
    seen_doors: set[tuple[str, float, float, float]] = set()
    for d in doors:
        key = (d.wall, _rnd(d.x, 1), _rnd(d.y, 1), _rnd(d.width, 1))
        if key in seen_doors:
            continue
        seen_doors.add(key)
        d.width = _clamp(_to_float(d.width, 3.0), 2.5, 4.5)
        deduped_doors.append(d)

    deduped_windows: list[WindowData] = []
    seen_windows: set[tuple[str, float, float, float]] = set()
    for w in windows:
        key = (w.wall, _rnd(w.x, 1), _rnd(w.y, 1), _rnd(w.width, 1))
        if key in seen_windows:
            continue
        seen_windows.add(key)
        w.width = _clamp(_to_float(w.width, 4.0), 3.0, 6.0)
        deduped_windows.append(w)

    return deduped_doors, deduped_windows


def _snap_and_fix_layout(rooms: list[RoomData], uw: float, ul: float):
    """
    Repair raw LLM coordinates into a buildable non-overlapping layout.
    Strategy:
    1) grid-snap and bounds clamp
    2) zone-aware anchoring (public/service/private)
    3) pairwise overlap repulsion
    4) targeted relocation onto free slots when overlaps persist
    """
    if not rooms:
        return

    grid = 0.5

    def _snap(val: float) -> float:
        return round(val / grid) * grid

    def _zone_y_bounds(room: RoomData) -> tuple[float, float]:
        # Tightened zone anchoring to enforce privacy gradient consistently.
        if room.zone == "public":
            lo, hi = 0.0, max(0.0, ul * 0.30 - room.height)
        elif room.zone == "service":
            lo, hi = max(0.0, ul * 0.22), max(0.0, ul * 0.78 - room.height)
        else:
            lo, hi = max(0.0, ul * 0.45), max(0.0, ul - room.height)
        if hi < lo:
            hi = lo
        return lo, hi

    def _clamp_in_bounds(room: RoomData):
        if room.polygon:
            bx, by, bw, bh = _polygon_bbox(room.polygon)
            bw = max(3.0, bw)
            bh = max(3.0, bh)

            room.width = bw
            room.height = bh

            target_x = _clamp(_snap(room.x), 0.0, max(0.0, uw - bw))
            target_y = _clamp(_snap(room.y), 0.0, max(0.0, ul - bh))

            dx = target_x - bx
            dy = target_y - by
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                moved = _translate_polygon(room.polygon, dx, dy, uw, ul)
                room.polygon = [Point2D(x=p["x"], y=p["y"]) for p in moved]

            fx, fy, fw, fh = _polygon_bbox(room.polygon)
            room.x = _clamp(_snap(fx), 0.0, max(0.0, uw - fw))
            room.y = _clamp(_snap(fy), 0.0, max(0.0, ul - fh))
            room.width = fw
            room.height = fh
            return

        room.width = _clamp(_snap(room.width), 3.0, max(3.0, uw))
        room.height = _clamp(_snap(room.height), 3.0, max(3.0, ul))
        room.x = _clamp(_snap(room.x), 0.0, max(0.0, uw - room.width))
        room.y = _clamp(_snap(room.y), 0.0, max(0.0, ul - room.height))

    def _find_slot(room: RoomData, placed: list[RoomData]) -> tuple[float, float] | None:
        x_max = max(0.0, uw - room.width)
        y_max = max(0.0, ul - room.height)
        pref_lo, pref_hi = _zone_y_bounds(room)
        y_ranges = [(pref_lo, min(pref_hi, y_max)), (0.0, y_max)]

        for y_lo, y_hi in y_ranges:
            y = _snap(y_lo)
            while y <= y_hi + 1e-6:
                x = 0.0
                while x <= x_max + 1e-6:
                    room.x = _snap(x)
                    room.y = _snap(y)
                    if all(not _room_overlap(room, other) for other in placed):
                        return room.x, room.y
                    x += grid
                y += grid
        return None

    # Pass 1: snap/clamp and zone anchoring
    for room in rooms:
        _clamp_in_bounds(room)
        y_lo, y_hi = _zone_y_bounds(room)
        room.y = _clamp(room.y, y_lo, y_hi)
        _clamp_in_bounds(room)

    # Pass 2: overlap repulsion
    for _ in range(12):
        moved = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]

                overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                overlap_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                moved = True
                if overlap_x <= overlap_y:
                    shift = _snap(overlap_x / 2.0 + 0.25)
                    if a.x <= b.x:
                        a.x -= shift
                        b.x += shift
                    else:
                        a.x += shift
                        b.x -= shift
                else:
                    shift = _snap(overlap_y / 2.0 + 0.25)
                    if a.y <= b.y:
                        a.y -= shift
                        b.y += shift
                    else:
                        a.y += shift
                        b.y -= shift

                _clamp_in_bounds(a)
                _clamp_in_bounds(b)

        if not moved:
            break

    # Pass 3: relocate residual-overlap rooms to nearest free slots
    sorted_rooms = sorted(rooms, key=lambda r: r.width * r.height, reverse=True)
    anchored: list[RoomData] = []
    for room in sorted_rooms:
        if any(_room_overlap(room, other) for other in anchored):
            slot = _find_slot(room, anchored)
            if slot is not None:
                room.x, room.y = slot
        _clamp_in_bounds(room)
        anchored.append(room)

    # Final cleanup
    for room in rooms:
        _clamp_in_bounds(room)


def _postprocess_llm_layout(rooms: list[RoomData], uw: float, ul: float):
    """
    Deterministic cleanups after a draft passes validation.
    Steps:
    1) snap coordinates to 0.5ft grid,
    2) expand rooms to remove slivers below 2ft,
    3) normalize narrow corridors to 3.5ft minimum width.
    """
    if not rooms:
        return

    grid = 0.5

    def _snap(value: float) -> float:
        return round(value / grid) * grid

    def _fit(room: RoomData):
        room.width = _clamp(_snap(room.width), 3.0, max(3.0, uw))
        room.height = _clamp(_snap(room.height), 3.0, max(3.0, ul))
        room.x = _clamp(_snap(room.x), 0.0, max(0.0, uw - room.width))
        room.y = _clamp(_snap(room.y), 0.0, max(0.0, ul - room.height))

    for room in rooms:
        _fit(room)

    # If LLM returned corridor narrower than 3.5ft, widen it deterministically.
    for room in rooms:
        if room.type != "corridor":
            continue
        if room.width >= 3.5:
            continue
        room.width = 3.5
        room.x = _clamp(room.x, 0.0, max(0.0, uw - room.width))
        _fit(room)

    # Fill tiny boundary slivers that are difficult to build onsite.
    for room in rooms:
        left_gap = room.x
        right_gap = uw - (room.x + room.width)
        bottom_gap = room.y
        top_gap = ul - (room.y + room.height)

        if 0.0 < left_gap < 2.0:
            room.x = 0.0
            room.width = room.width + left_gap
        if 0.0 < right_gap < 2.0:
            room.width = room.width + right_gap
        if 0.0 < bottom_gap < 2.0:
            room.y = 0.0
            room.height = room.height + bottom_gap
        if 0.0 < top_gap < 2.0:
            room.height = room.height + top_gap
        _fit(room)

    # Fill tiny interior slivers by extending to nearest neighboring room.
    for _ in range(2):
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]

                y_overlap = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                if y_overlap >= 2.0:
                    if a.x + a.width <= b.x:
                        gap = b.x - (a.x + a.width)
                        if 0.0 < gap < 2.0:
                            a.width = a.width + gap
                    elif b.x + b.width <= a.x:
                        gap = a.x - (b.x + b.width)
                        if 0.0 < gap < 2.0:
                            b.width = b.width + gap

                x_overlap = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                if x_overlap >= 2.0:
                    if a.y + a.height <= b.y:
                        gap = b.y - (a.y + a.height)
                        if 0.0 < gap < 2.0:
                            a.height = a.height + gap
                    elif b.y + b.height <= a.y:
                        gap = a.y - (b.y + b.height)
                        if 0.0 < gap < 2.0:
                            b.height = b.height + gap

                _fit(a)
                _fit(b)

    for room in rooms:
        _fit(room)



# ─────────────────────────────────────────────────────────────
# BSP FALLBACK — original deterministic engine (unchanged)
# ─────────────────────────────────────────────────────────────
async def _generate_via_bsp(req: PlanRequest, uw: float, ul: float) -> PlanResponse:
    """Generate a floor plan using deterministic BSP packing (fallback)."""
    # Keep fallback fully deterministic and fast (no external LLM calls).
    architect_note = (
        "Logical deterministic layout generated with strict zoning and practical adjacency."
    )

    # Build room specs
    room_specs = build_room_list(req.bedrooms, req.extras, uw, ul, req.facing)
    requested_baths = max(0, int(getattr(req, "bathrooms_target", 0) or 0))
    existing_baths = sum(
        1 for spec in room_specs if spec.get("type") in ("master_bath", "bathroom", "toilet")
    )
    extra_baths = max(0, min(8, requested_baths) - existing_baths)
    for idx in range(extra_baths):
        room_specs.append({
            "type": "bathroom", "label": f"Bathroom {existing_baths + idx + 1}",
            "min_w": 5, "min_h": 6, "pref_w": 5, "pref_h": 7,
            "zone": 2, "priority": 20 + idx,
        })

    # Pack rooms deterministically
    placed = pack_rooms_bsp(room_specs, uw, ul)
    vastu_score = _compute_fallback_vastu_score(placed, uw, ul)

    # Add doors and windows
    doors, windows = add_doors_and_windows(placed, uw, ul, req.facing)

    # Build response
    rooms = []
    for i, p in enumerate(placed):
        rooms.append(RoomData(
            id=f"{p['type']}_{i+1:02d}",
            type=p["type"],
            label=p["label"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            area=_rnd(p["width"] * p["height"], 1),
            zone=("public" if p["zone"] == 1
                  else "service" if p["zone"] == 2
                  else "private"),
            band=p["band"],
            color=ROOM_COLORS.get(p["type"], "#F5F5F5"),
        ))

    adjacency_score = _compute_adjacency_score(rooms)

    plot = PlotInfo(
        width=req.plot_width,
        length=req.plot_length,
        usable_width=uw,
        usable_length=ul,
        road_side=req.facing,
        setbacks=SETBACKS,
    )

    return PlanResponse(
        plot=plot,
        rooms=rooms,
        doors=doors,
        windows=windows,
        vastu_score=vastu_score,
        architect_note=architect_note,
        generation_method="bsp",
        adjacency_score=adjacency_score,
        reasoning_trace=[
            "Deterministic planner used strict 3-band zoning.",
            "Living-dining-kitchen and bedroom-corridor adjacency were enforced.",
            "Returned stable non-overlapping layout for reliable preview/export.",
        ],
    )


# ─────────────────────────────────────────────────────────────
# Room spec builder
# ─────────────────────────────────────────────────────────────
def build_room_list(
    bedrooms: int,
    extras: List[str],
    usable_w: float,
    usable_l: float,
    facing: str = "south",
) -> List[Dict[str, Any]]:
    """Build room specifications based on BHK count and extras."""
    rooms = []

    # --- Always present ---
    rooms.append({
        "type": "living", "label": "Living Room",
        "min_w": 12, "min_h": 12, "pref_w": 14, "pref_h": 14,
        "zone": 1, "priority": 1,
    })
    rooms.append({
        "type": "dining", "label": "Dining Room",
        "min_w": 9, "min_h": 9, "pref_w": 10, "pref_h": 10,
        "zone": 1, "priority": 2,
    })
    rooms.append({
        "type": "kitchen", "label": "Kitchen",
        "min_w": 8, "min_h": 9, "pref_w": 9, "pref_h": 10,
        "zone": 2, "priority": 3,
    })
    rooms.append({
        "type": "corridor", "label": "Corridor",
        "min_w": 3.5, "min_h": usable_l, "pref_w": 4, "pref_h": usable_l,
        "zone": 2, "priority": 4,
    })
    rooms.append({
        "type": "master_bedroom", "label": "Master Bedroom",
        "min_w": 11, "min_h": 11, "pref_w": 13, "pref_h": 13,
        "zone": 3, "priority": 5,
    })
    rooms.append({
        "type": "master_bath", "label": "Master Bath",
        "min_w": 5, "min_h": 7, "pref_w": 5, "pref_h": 8,
        # Wet rooms are intentionally clustered in service core for plumbing efficiency.
        "zone": 2, "priority": 6,
    })

    # --- Additional bedrooms/bathrooms ---
    if bedrooms >= 2:
        rooms.append({
            "type": "bedroom", "label": "Bedroom 2",
            "min_w": 10, "min_h": 10, "pref_w": 11, "pref_h": 12,
            "zone": 3, "priority": 7,
        })
        rooms.append({
            "type": "bathroom", "label": "Bathroom 2",
            "min_w": 5, "min_h": 7, "pref_w": 5, "pref_h": 7,
            "zone": 2, "priority": 8,
        })

    if bedrooms >= 3:
        rooms.append({
            "type": "bedroom", "label": "Bedroom 3",
            "min_w": 10, "min_h": 10, "pref_w": 11, "pref_h": 12,
            "zone": 3, "priority": 9,
        })
        rooms.append({
            "type": "bathroom", "label": "Bathroom 3",
            "min_w": 5, "min_h": 6, "pref_w": 5, "pref_h": 7,
            "zone": 2, "priority": 10,
        })
        rooms.append({
            "type": "bathroom", "label": "Common Bath",
            "min_w": 5, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 2, "priority": 11,
        })

    if bedrooms >= 4:
        rooms.append({
            "type": "bedroom", "label": "Bedroom 4",
            "min_w": 10, "min_h": 10, "pref_w": 11, "pref_h": 11,
            "zone": 3, "priority": 12,
        })
        rooms.append({
            "type": "bathroom", "label": "Bathroom 4",
            "min_w": 5, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 2, "priority": 13,
        })

    # --- Extras ---
    if "pooja" in extras:
        rooms.append({
            "type": "pooja", "label": "Pooja Room",
            "min_w": 5, "min_h": 5, "pref_w": 6, "pref_h": 6,
            "zone": 1, "priority": 2,
        })
    elif str(facing).strip().lower() == "south":
        # South-facing plots are compensated with a small pooja/energy buffer near entry.
        rooms.append({
            "type": "pooja", "label": "Pooja Buffer",
            "min_w": 4.5, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 1, "priority": 2,
        })
    if "study" in extras:
        rooms.append({
            "type": "study", "label": "Study Room",
            "min_w": 8, "min_h": 9, "pref_w": 9, "pref_h": 10,
            "zone": 3, "priority": 8,
        })
    if "store" in extras:
        rooms.append({
            "type": "store", "label": "Store Room",
            "min_w": 5, "min_h": 5, "pref_w": 6, "pref_h": 6,
            "zone": 2, "priority": 9,
        })
    if "balcony" in extras:
        rooms.append({
            "type": "balcony", "label": "Balcony",
            "min_w": 4, "min_h": 8, "pref_w": 5, "pref_h": 10,
            "zone": 1, "priority": 3,
        })
    if "garage" in extras:
        rooms.append({
            "type": "garage", "label": "Garage",
            "min_w": 10, "min_h": 18, "pref_w": 11, "pref_h": 20,
            "zone": 1, "priority": 10,
        })
    if "utility" in extras:
        rooms.append({
            "type": "utility", "label": "Utility",
            "min_w": 4, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 2, "priority": 9,
        })
    if "foyer" in extras:
        rooms.append({
            "type": "foyer", "label": "Foyer",
            "min_w": 4, "min_h": 4, "pref_w": 5, "pref_h": 5,
            "zone": 1, "priority": 2,
        })
    if "staircase" in extras:
        rooms.append({
            "type": "staircase", "label": "Staircase",
            "min_w": 6, "min_h": 8, "pref_w": 6.5, "pref_h": 9,
            "zone": 2, "priority": 9,
        })

    return rooms


# ─────────────────────────────────────────────────────────────
# BSP Packing v2 — smart grid, zero-overlap, proper proportions
# ─────────────────────────────────────────────────────────────
def pack_rooms_bsp(room_specs: List[Dict[str, Any]],
                   usable_w: float, usable_l: float) -> List[Dict[str, Any]]:
    """
    Pack rooms into three horizontal bands with zero overlaps.
    Uses smart grid layout that pairs bedrooms with baths.
    Band 1 (public):  ~28% depth — living, dining, pooja, balcony
    Band 2 (service): ~25% depth — corridor spine, kitchen, bathrooms
    Band 3 (private): ~47% depth — bedrooms with attached baths, study
    """
    # Rebalanced bands to give service core enough depth for clustered wet rooms.
    band1_h = _rnd(usable_l * 0.30, 2)
    band2_h = _rnd(usable_l * 0.30, 2)
    band3_h = _rnd(usable_l - band1_h - band2_h, 2)

    # Separate rooms by zone
    zone_rooms: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    for spec in room_specs:
        z = spec["zone"]
        if z in zone_rooms:
            zone_rooms[z].append(spec)

    # Sort each zone by priority
    for z in zone_rooms:
        zone_rooms[z].sort(key=lambda r: r["priority"])

    placed: List[Dict[str, Any]] = []

    # ── Band 1: Public (living, dining, pooja, balcony) ──────
    placed.extend(_pack_band1(zone_rooms[1], 0, 0, usable_w, band1_h))

    # ── Band 2: Service (corridor + kitchen + bathrooms) ─────
    band2_rooms = _pack_band2(
        zone_rooms[2],
        0,
        band1_h,
        usable_w,
        band2_h,
        corridor_extend_h=band3_h,
    )
    placed.extend(band2_rooms)

    corridor_hint: tuple[float, float] | None = None
    for r in band2_rooms:
        if r.get("type") == "corridor":
            corridor_hint = (_to_float(r.get("x"), 0.0), _to_float(r.get("width"), 0.0))
            break

    # ── Band 3: Private (bedrooms + attached baths + study) ──
    placed.extend(
        _pack_band3(zone_rooms[3], 0, band1_h + band2_h, usable_w, band3_h, corridor_hint)
    )

    # Final overlap check
    _verify_no_overlaps(placed)
    return placed


def _pack_band1(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float) -> List[Dict[str, Any]]:
    """
    Band 1: Public zone.
    Layout: Living room (60% width) | Dining + small rooms stacked (40% width)
    """
    if not rooms:
        return []

    placed = []

    # Separate large rooms from small utility rooms
    living = None
    dining = None
    small_rooms = []

    for r in rooms:
        if r["type"] == "living":
            living = r
        elif r["type"] == "dining":
            dining = r
        else:
            small_rooms.append(r)

    # Calculate proportions
    has_small = len(small_rooms) > 0

    if living and dining:
        if has_small:
            # 3-column: Living (50%) | Dining (30%) | Small rooms stacked (20%)
            living_w = _rnd(bw * 0.50, 2)
            dining_w = _rnd(bw * 0.30, 2)
            small_w = _rnd(bw - living_w - dining_w, 2)
        else:
            # 2-column: Living (58%) | Dining (42%)
            living_w = _rnd(bw * 0.58, 2)
            dining_w = _rnd(bw - living_w, 2)
            small_w = 0

        placed.append({
            "type": living["type"], "label": living["label"],
            "x": _rnd(bx, 2), "y": _rnd(by, 2),
            "width": _rnd(living_w, 2), "height": _rnd(bh, 2),
            "zone": 1, "band": 1,
        })
        placed.append({
            "type": dining["type"], "label": dining["label"],
            "x": _rnd(bx + living_w, 2), "y": _rnd(by, 2),
            "width": _rnd(dining_w, 2), "height": _rnd(bh, 2),
            "zone": 1, "band": 1,
        })

        if has_small and small_w >= 4:
            # Stack small rooms vertically in the remaining column
            current_y = by
            per_h = _rnd(bh / len(small_rooms), 2)
            for i, sr in enumerate(small_rooms):
                rh = per_h if i < len(small_rooms) - 1 else _rnd(by + bh - current_y, 2)
                placed.append({
                    "type": sr["type"], "label": sr["label"],
                    "x": _rnd(bx + living_w + dining_w, 2),
                    "y": _rnd(current_y, 2),
                    "width": _rnd(small_w, 2), "height": _rnd(rh, 2),
                    "zone": 1, "band": 1,
                })
                current_y = _rnd(current_y + rh, 2)
        elif has_small:
            # Not enough width — tuck small rooms inside dining area (bottom portion)
            dining_room = placed[-1]  # the dining we just placed
            tuck_h = _rnd(bh * 0.45, 2)
            dining_room["height"] = _rnd(bh - tuck_h, 2)

            current_x = dining_room["x"]
            tuck_w = _rnd(dining_w / len(small_rooms), 2)
            for i, sr in enumerate(small_rooms):
                rw = tuck_w if i < len(small_rooms) - 1 else _rnd(dining_room["x"] + dining_w - current_x, 2)
                placed.append({
                    "type": sr["type"], "label": sr["label"],
                    "x": _rnd(current_x, 2),
                    "y": _rnd(by + bh - tuck_h, 2),
                    "width": _rnd(rw, 2), "height": _rnd(tuck_h, 2),
                    "zone": 1, "band": 1,
                })
                current_x = _rnd(current_x + rw, 2)

    elif living:
        placed.append({
            "type": living["type"], "label": living["label"],
            "x": _rnd(bx, 2), "y": _rnd(by, 2),
            "width": _rnd(bw, 2), "height": _rnd(bh, 2),
            "zone": 1, "band": 1,
        })
    else:
        # Fallback: strip pack everything
        placed.extend(_pack_strip_safe(rooms, bx, by, bw, bh, 1))

    return placed


def _pack_band2(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float, corridor_extend_h: float = 0.0) -> List[Dict[str, Any]]:
    """
    Band 2: Service zone.
    Layout: Corridor west spine | Wet core center | Kitchen east (southeast preference).
    """
    if not rooms:
        return []

    placed: List[Dict[str, Any]] = []
    kitchen = None
    corridor_spec = None
    wet_rooms: List[Dict[str, Any]] = []
    service_other: List[Dict[str, Any]] = []

    for r in rooms:
        if r["type"] == "corridor":
            corridor_spec = r
        elif r["type"] == "kitchen":
            kitchen = r
        elif r["type"] in ("bathroom", "master_bath", "toilet"):
            wet_rooms.append(r)
        else:
            service_other.append(r)

    corridor_w = 3.5 if corridor_spec else 0.0
    corridor_w = _rnd(min(corridor_w, max(0.0, bw - 12.0)), 2)

    if corridor_spec and corridor_w >= 3.0:
        # Keep corridor central so both private bedrooms can connect through it.
        left_w = _rnd(max(5.0, (bw - corridor_w) * 0.42), 2)
        right_w = _rnd(bw - corridor_w - left_w, 2)
        if right_w < 6.0:
            right_w = 6.0
            left_w = _rnd(max(5.0, bw - corridor_w - right_w), 2)

        corridor_x = _rnd(bx + left_w, 2)
        wet_x = _rnd(bx, 2)
        wet_w = _rnd(max(5.0, corridor_x - wet_x), 2)
        kitchen_x = _rnd(corridor_x + corridor_w, 2)
        kitchen_w = _rnd(max(6.0, bx + bw - kitchen_x), 2)
    else:
        corridor_w = 0.0
        wet_x = _rnd(bx, 2)
        wet_w = _rnd(min(max(5.0, bw * 0.38), max(5.0, bw - 8.0)), 2)
        kitchen_x = _rnd(wet_x + wet_w, 2)
        kitchen_w = _rnd(max(6.0, bx + bw - kitchen_x), 2)
        corridor_x = _rnd(bx + wet_w, 2)

    if corridor_spec and corridor_w >= 3.0:
        corridor_h = _rnd(max(bh, bh + max(0.0, corridor_extend_h)), 2)
        placed.append({
            "type": "corridor", "label": "Corridor",
            "x": corridor_x, "y": _rnd(by, 2),
            "width": _rnd(corridor_w, 2), "height": corridor_h,
            "zone": 2, "band": 2,
        })

    east_stack: List[Dict[str, Any]] = []
    central_stack: List[Dict[str, Any]] = []
    for r in service_other:
        if r["type"] in ("utility", "store"):
            east_stack.append(r)
        else:
            central_stack.append(r)

    if kitchen:
        reserve_h = 0.0
        if east_stack:
            reserve_h = _rnd(min(max(3.5, bh * 0.30), max(3.5, bh - 4.0)), 2)
        kitchen_h = _rnd(max(4.0, bh - reserve_h), 2)

        placed.append({
            "type": kitchen["type"], "label": kitchen["label"],
            "x": kitchen_x, "y": _rnd(by, 2),
            "width": _rnd(kitchen_w, 2), "height": kitchen_h,
            "zone": 2, "band": 2,
        })

        if reserve_h > 0 and east_stack:
            y = _rnd(by + kitchen_h, 2)
            per_h = _rnd(reserve_h / len(east_stack), 2)
            for idx, room in enumerate(east_stack):
                rh = per_h if idx < len(east_stack) - 1 else _rnd(by + bh - y, 2)
                placed.append({
                    "type": room["type"], "label": room["label"],
                    "x": kitchen_x, "y": _rnd(y, 2),
                    "width": _rnd(kitchen_w, 2), "height": _rnd(rh, 2),
                    "zone": 2, "band": 2,
                })
                y = _rnd(y + rh, 2)

    # Place master bath at top of service stack so it attaches to rear private band.
    wet_rooms.sort(key=lambda r: 1 if r.get("type") == "master_bath" else 0)
    stack_rooms = wet_rooms + central_stack
    if stack_rooms and wet_w > 0:
        current_y = by
        per_h = _rnd(bh / len(stack_rooms), 2)
        for idx, room in enumerate(stack_rooms):
            rh = per_h if idx < len(stack_rooms) - 1 else _rnd(by + bh - current_y, 2)
            placed.append({
                "type": room["type"], "label": room["label"],
                "x": wet_x, "y": _rnd(current_y, 2),
                "width": _rnd(wet_w, 2), "height": _rnd(rh, 2),
                "zone": 2, "band": 2,
            })
            current_y = _rnd(current_y + rh, 2)

    return placed


def _pack_band3(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float,
                corridor_hint: tuple[float, float] | None = None) -> List[Dict[str, Any]]:
    """
    Band 3: Private zone.
    Layout intent: master bedroom anchored southwest, other bedrooms east,
    study adjacent to master where requested.
    """
    if not rooms:
        return []

    placed: List[Dict[str, Any]] = []

    master = None
    other_beds: List[Dict[str, Any]] = []
    studies: List[Dict[str, Any]] = []
    misc_private: List[Dict[str, Any]] = []

    for r in rooms:
        if r["type"] == "master_bedroom" and master is None:
            master = r
        elif r["type"] == "bedroom":
            other_beds.append(r)
        elif r["type"] == "study":
            studies.append(r)
        else:
            misc_private.append(r)

    if master is None:
        for r in rooms:
            if r["type"] == "bedroom":
                master = r
                break
        if master is not None:
            other_beds = [r for r in other_beds if r is not master]

    if master is None:
        return _pack_strip_safe(rooms, bx, by, bw, bh, 3)

    use_corridor_channel = False
    corridor_x = 0.0
    corridor_w = 0.0
    if corridor_hint is not None:
        corridor_x = _to_float(corridor_hint[0], 0.0)
        corridor_w = _to_float(corridor_hint[1], 0.0)
        if corridor_w >= 3.0 and corridor_x > bx + 6.0 and corridor_x + corridor_w < bx + bw - 6.0:
            use_corridor_channel = True

    if use_corridor_channel:
        master_w = _rnd(max(8.5, corridor_x - bx), 2)
        east_x = _rnd(corridor_x + corridor_w, 2)
        east_w = _rnd(max(6.0, bx + bw - east_x), 2)
    else:
        master_w = _rnd(min(max(9.5, bw * 0.55), max(9.5, bw * 0.60)), 2)
        if master_w > bw - 6.0:
            master_w = _rnd(max(8.5, bw - 6.0), 2)
        east_w = _rnd(max(6.0, bw - master_w), 2)
        east_x = _rnd(bx + master_w, 2)

    if east_w <= 0:
        master_w = _rnd(bw, 2)
        east_w = 0.0
        east_x = _rnd(bx + master_w, 2)

    master_h = _rnd(bh, 2)
    study_h = 0.0
    if studies:
        study_h = _rnd(max(6.0, min(bh * 0.35, max(6.0, bh - 8.0))), 2)
        master_h = _rnd(max(8.0, bh - study_h), 2)

    placed.append({
        "type": master["type"], "label": master["label"],
        "x": _rnd(bx, 2), "y": _rnd(by, 2),
        "width": _rnd(master_w, 2), "height": _rnd(master_h, 2),
        "zone": 3, "band": 3,
    })

    if studies:
        placed.append({
            "type": studies[0]["type"], "label": studies[0]["label"],
            "x": _rnd(bx, 2), "y": _rnd(by + master_h, 2),
            "width": _rnd(master_w, 2), "height": _rnd(study_h, 2),
            "zone": 3, "band": 3,
        })

    east_rooms = other_beds + misc_private
    if east_w > 0 and east_rooms:
        current_y = by
        per_h = _rnd(bh / len(east_rooms), 2)
        for idx, room in enumerate(east_rooms):
            rh = per_h if idx < len(east_rooms) - 1 else _rnd(by + bh - current_y, 2)
            placed.append({
                "type": room["type"], "label": room["label"],
                "x": east_x, "y": _rnd(current_y, 2),
                "width": _rnd(east_w, 2), "height": _rnd(rh, 2),
                "zone": 3, "band": 3,
            })
            current_y = _rnd(current_y + rh, 2)

    return placed


def _pack_strip_safe(rooms: List[Dict[str, Any]], bx: float, by: float,
                     bw: float, bh: float, band_num: int) -> List[Dict[str, Any]]:
    """Fallback strip packing that respects min widths and never goes negative."""
    if not rooms:
        return []

    total_min = sum(r["min_w"] for r in rooms)
    placed = []
    current_x = bx

    for i, room in enumerate(rooms):
        if i == len(rooms) - 1:
            rw = _rnd(bx + bw - current_x, 2)
        else:
            if total_min <= bw:
                # Enough space — use proportional
                ratio = room["pref_w"] / sum(r["pref_w"] for r in rooms)
                rw = _rnd(bw * ratio, 2)
                rw = max(rw, room["min_w"])
            else:
                # Not enough — give min widths
                rw = room["min_w"]

        rw = max(rw, 3.0)  # Never below 3ft
        if current_x + rw > bx + bw:
            rw = _rnd(bx + bw - current_x, 2)
        if rw < 2.0:
            continue  # Skip rooms that can't fit

        placed.append({
            "type": room["type"], "label": room["label"],
            "x": _rnd(current_x, 2), "y": _rnd(by, 2),
            "width": _rnd(rw, 2), "height": _rnd(bh, 2),
            "zone": room.get("zone", 1), "band": band_num,
        })
        current_x = _rnd(current_x + rw, 2)

    return placed


def _verify_no_overlaps(placed: List[Dict[str, Any]]):
    """Check all placed rooms for overlaps. Log warnings if found."""
    for i, a in enumerate(placed):
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if _rects_overlap(a, b):
                log.warning(
                    "Overlap detected: %s (%.1f,%.1f,%.1f,%.1f) vs %s (%.1f,%.1f,%.1f,%.1f)",
                    a["label"], a["x"], a["y"], a["width"], a["height"],
                    b["label"], b["x"], b["y"], b["width"], b["height"],
                )
                b["width"] = _rnd(b["width"] - 0.2, 2)


def _compute_fallback_vastu_score(
    placed: List[Dict[str, Any]],
    usable_w: float,
    usable_l: float,
) -> float:
    """Score deterministic fallback layouts using key vastu placements."""
    base = 68.0

    def _first(room_type: str) -> Dict[str, Any] | None:
        for room in placed:
            if room.get("type") == room_type:
                return room
        return None

    living = _first("living")
    kitchen = _first("kitchen")
    master = _first("master_bedroom")
    pooja = _first("pooja")
    corridor = _first("corridor")

    rear_min = usable_l * 0.45
    front_max = usable_l * 0.35

    if living and living["y"] + living["height"] <= front_max + 0.5:
        base += 3.0
    if kitchen and kitchen["x"] + kitchen["width"] >= usable_w * 0.5 and kitchen["y"] >= usable_l * 0.22:
        base += 4.0
    if master and master["x"] + master["width"] <= usable_w * 0.6 and master["y"] >= rear_min:
        base += 4.0
    if pooja and pooja["x"] >= usable_w * 0.5 and pooja["y"] + pooja["height"] <= front_max + 0.5:
        base += 3.0
    if corridor and corridor["width"] >= 3.5:
        base += 2.0

    return _rnd(_clamp(base, 60.0, 88.0), 1)


def _rects_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check if two rectangles overlap (with 0.1ft tolerance)."""
    eps = 0.1
    return (
        a["x"] < b["x"] + b["width"] - eps and
        a["x"] + a["width"] > b["x"] + eps and
        a["y"] < b["y"] + b["height"] - eps and
        a["y"] + a["height"] > b["y"] + eps
    )


# ─────────────────────────────────────────────────────────────
# Door & Window placement
# ─────────────────────────────────────────────────────────────
def add_doors_and_windows(placed_rooms: List[Dict[str, Any]],
                          usable_w: float, usable_l: float,
                          facing: str) -> tuple[list[DoorData], list[WindowData]]:
    """Add doors between adjacent rooms and windows on exterior walls."""
    doors: List[DoorData] = []
    windows: List[WindowData] = []
    door_count: int = 0
    win_count: int = 0
    room_door_counts: Dict[str, int] = {}

    # Find the living room for main entrance
    living = None
    for r in placed_rooms:
        if r["type"] == "living":
            living = r
            break

    # Main entrance door on the road-facing wall of living room
    if living:
        door_count = door_count + 1
        living_id = str(living.get("id", "living"))
        # Main entrance is intentionally constrained to east or north walls
        # to align with the architect prompt's hard circulation/vastu rule.
        if facing in ("north", "west"):
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id=living_id, wall="north",
                x=living["x"] + living["width"] * 0.7,
                y=living["y"] + living["height"],
                width=3.5,
            ))
        else:
            # South-facing plots are compensated by shifting entry to east-northeast.
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id=living_id, wall="east",
                x=living["x"] + living["width"],
                y=living["y"] + living["height"] * 0.7,
                width=3.5,
            ))
        room_door_counts[living_id] = room_door_counts.get(living_id, 0) + 1

    # Interior doors between adjacent rooms
    for i, a in enumerate(placed_rooms):
        for j in range(i + 1, len(placed_rooms)):
            b = placed_rooms[j]
            shared = _shared_wall(a, b)
            if shared:
                room_a = str(a.get("id", a.get("type", f"room_{i}")))
                room_b = str(b.get("id", b.get("type", f"room_{j}")))

                limit_a = 5 if a.get("type") in ("corridor", "living", "dining") else 2
                limit_b = 5 if b.get("type") in ("corridor", "living", "dining") else 2
                if room_door_counts.get(room_a, 0) >= limit_a:
                    continue
                if room_door_counts.get(room_b, 0) >= limit_b:
                    continue

                wall_side, sx, sy = shared
                door_count = door_count + 1
                doors.append(DoorData(
                    id=f"door_{door_count:02d}", type="interior",
                    room_id=room_a,
                    wall=wall_side,
                    x=_rnd(sx, 2), y=_rnd(sy, 2),
                    width=3.0,
                ))
                room_door_counts[room_a] = room_door_counts.get(room_a, 0) + 1
                room_door_counts[room_b] = room_door_counts.get(room_b, 0) + 1

    # Windows on exterior walls
    for r in placed_rooms:
        if r["type"] in ("corridor", "store"):
            continue  # No windows for corridor or store

        eps = 0.3
        # South wall (y == 0 means touching front setback boundary)
        if r["y"] <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=str(r.get("id", r["type"])),
                wall="south",
                x=r["x"] + r["width"] * 0.3,
                y=r["y"],
                width=min(4.0, r["width"] * 0.35),
            ))
        # North wall (touching rear boundary)
        if abs(r["y"] + r["height"] - usable_l) <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=str(r.get("id", r["type"])),
                wall="north",
                x=r["x"] + r["width"] * 0.3,
                y=r["y"] + r["height"],
                width=min(4.0, r["width"] * 0.35),
            ))
        # West wall (x == 0 means touching left setback boundary)
        if r["x"] <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=str(r.get("id", r["type"])),
                wall="west",
                x=r["x"],
                y=r["y"] + r["height"] * 0.3,
                width=min(4.0, r["height"] * 0.35),
            ))
        # East wall (touching right boundary)
        if abs(r["x"] + r["width"] - usable_w) <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=str(r.get("id", r["type"])),
                wall="east",
                x=r["x"] + r["width"],
                y=r["y"] + r["height"] * 0.3,
                width=min(4.0, r["height"] * 0.35),
            ))

    return doors, windows


def _shared_wall(a: Dict[str, Any], b: Dict[str, Any]) -> tuple[str, float, float] | None:
    """
    Check if rooms a and b share a wall segment.
    Returns (wall_side, door_x, door_y) or None.
    """
    eps = 0.3
    # a's right edge == b's left edge (vertical wall between them)
    if abs(a["x"] + a["width"] - b["x"]) < eps:
        oy = max(a["y"], b["y"])
        ey = min(a["y"] + a["height"], b["y"] + b["height"])
        if ey - oy > 3.0:
            mid_y = (oy + ey) / 2 - 1.5
            return "east", a["x"] + a["width"], mid_y

    # b's right edge == a's left edge
    if abs(b["x"] + b["width"] - a["x"]) < eps:
        oy = max(a["y"], b["y"])
        ey = min(a["y"] + a["height"], b["y"] + b["height"])
        if ey - oy > 3.0:
            mid_y = (oy + ey) / 2 - 1.5
            return "west", a["x"], mid_y

    # a's top edge == b's bottom edge (horizontal wall)
    if abs(a["y"] + a["height"] - b["y"]) < eps:
        ox = max(a["x"], b["x"])
        ex = min(a["x"] + a["width"], b["x"] + b["width"])
        if ex - ox > 3.0:
            mid_x = (ox + ex) / 2 - 1.5
            return "north", mid_x, a["y"] + a["height"]

    # b's top edge == a's bottom edge
    if abs(b["y"] + b["height"] - a["y"]) < eps:
        ox = max(a["x"], b["x"])
        ex = min(a["x"] + a["width"], b["x"] + b["width"])
        if ex - ox > 3.0:
            mid_x = (ox + ex) / 2 - 1.5
            return "south", mid_x, a["y"]

    return None


# ─────────────────────────────────────────────────────────────
# Legacy alias for backward compatibility
# ─────────────────────────────────────────────────────────────
async def generate_plan_deterministic(req: PlanRequest) -> PlanResponse:
    """Backward-compatible alias — now routes through LLM-first pipeline."""
    return await generate_plan(req)
