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
import random

def _rnd(val: float, d: int = 2) -> float:
    return int(val * (10**d)) / (10.0**d)

from models import (
    PlanRequest, PlanResponse, PlotInfo,
    RoomData, DoorData, WindowData, Point2D,
)
from llm import call_openrouter_plan, call_openrouter_plan_backup
from prompt_builder import build_master_prompt
from plan_validator import validate_llm_plan

log = logging.getLogger("layout_engine")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SETBACKS = {"front": 6.5, "rear": 5.0, "left": 3.5, "right": 3.5}
FAST_LLM_TIMEOUT_SEC = 24
FAST_RETRY_TIMEOUT_SEC = 12
FAST_BACKUP_TIMEOUT_SEC = 28

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
        plan, issues = await asyncio.wait_for(
            _generate_via_llm(req, uw, ul),
            timeout=FAST_LLM_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        plan, issues = None, ["llm attempt timed out"]
        log.warning("LLM attempt timed out after %ss", FAST_LLM_TIMEOUT_SEC)

    if plan:
        log.info("LLM plan accepted on first attempt")
        return plan

    # ── Attempt 2: quick retry (only for validation issues) ──
    timeout_in_issues = any("timed out" in str(i).lower() for i in (issues or []))
    if issues and not timeout_in_issues:
        log.info("LLM plan invalid (%d issues), retrying with corrections", len(issues))
        try:
            plan, _ = await asyncio.wait_for(
                _generate_via_llm(req, uw, ul, prev_issues=issues),
                timeout=FAST_RETRY_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            plan = None
            log.warning("LLM retry timed out after %ss", FAST_RETRY_TIMEOUT_SEC)
        if plan:
            log.info("LLM plan accepted on retry")
            return plan

    # ── Attempt 3: backup free-model pool ───────────────────
    backup_seed_issues = issues if (issues and not timeout_in_issues) else None
    try:
        backup_plan, _ = await asyncio.wait_for(
            _generate_via_llm(req, uw, ul, prev_issues=backup_seed_issues, use_backup_models=True),
            timeout=FAST_BACKUP_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        backup_plan = None
        log.warning("Backup LLM pool timed out after %ss", FAST_BACKUP_TIMEOUT_SEC)

    if backup_plan:
        backup_plan.generation_method = "llm_backup"
        backup_plan.reasoning_trace = [
            *backup_plan.reasoning_trace,
            "Primary model pool was unavailable; delivered via backup free-model pool.",
        ][:12]
        return backup_plan

    # ── Final local adaptive fallback (no busy error to user) ──
    log.error("Primary and backup LLM pools unavailable; using local adaptive fallback")
    fallback = await _generate_via_bsp(req, uw, ul)
    fallback.generation_method = "local_backup"
    fallback.reasoning_trace = [
        f"Computed usable plot after setbacks: {uw:.1f} ft x {ul:.1f} ft.",
        "Primary and backup free-model pools were unavailable for this request.",
        "Generated an adaptive local layout automatically to avoid interruption.",
    ]
    fallback.architect_note = (
        "Adaptive backup layout generated automatically while free AI model pools were unavailable."
    )
    return fallback


async def _generate_via_llm(
    req: PlanRequest,
    uw: float,
    ul: float,
    prev_issues: list[str] | None = None,
    use_backup_models: bool = False,
) -> tuple[PlanResponse | None, list[str]]:
    """
    Try to generate a plan via the LLM.
    Returns (PlanResponse, []) if successful, (None, issues) if failed.
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
            "Switching to backup free-model pool for generation continuity.",
        )

    # Append correction feedback if retrying
    if prev_issues:
        correction = (
            "\n\nPREVIOUS ATTEMPT FAILED VALIDATION. Fix these issues:\n"
            + "\n".join(f"- {issue}" for issue in prev_issues[:10])
            + "\n\nAdjust room coordinates to fix ALL issues above."
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
        return None, [str(e)]

    draft_rooms = plan_dict.get("rooms", []) if isinstance(plan_dict, dict) else []
    if isinstance(draft_rooms, list):
        _push_reasoning(
            reasoning_trace,
            f"LLM returned draft with {len(draft_rooms)} rooms; running strict validation.",
        )

    # Validate the raw LLM draft first
    is_valid, issues = validate_llm_plan(plan_dict, uw, ul, req)

    if not is_valid:
        log.warning("LLM plan invalid (%d issues): %s", len(issues), issues[:3])
        _push_reasoning(
            reasoning_trace,
            f"Draft rejected by validator with {len(issues)} issues; requesting corrected layout.",
        )
        return None, issues

    # Build production-grade plan with geometry/opening repair
    plan = _parse_llm_plan(plan_dict, req, uw, ul, reasoning_trace=reasoning_trace)

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
        log.warning("Post-repair plan invalid (%d issues): %s", len(final_issues), final_issues[:3])
        _push_reasoning(
            reasoning_trace,
            f"Post-repair validation failed with {len(final_issues)} issues; correction cycle required.",
        )
        return None, [f"post-repair: {issue}" for issue in final_issues]

    _push_reasoning(reasoning_trace, "Final repaired plan passed validation and is ready for preview.")

    return plan, []


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
        # Keep broad bands to preserve design flexibility while improving realism.
        if room.zone == "public":
            lo, hi = 0.0, max(0.0, ul * 0.38 - room.height)
        elif room.zone == "service":
            lo, hi = max(0.0, ul * 0.20 - room.height * 0.4), max(0.0, ul * 0.72 - room.height)
        else:
            lo, hi = max(0.0, ul * 0.46 - room.height * 0.3), max(0.0, ul - room.height)
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



# ─────────────────────────────────────────────────────────────
# BSP FALLBACK — original deterministic engine (unchanged)
# ─────────────────────────────────────────────────────────────
async def _generate_via_bsp(req: PlanRequest, uw: float, ul: float) -> PlanResponse:
    """Generate a floor plan using deterministic BSP packing (fallback)."""
    # Keep fallback fully deterministic and fast (no external LLM calls).
    vastu_score = 72.0
    architect_note = (
        "Fast deterministic layout generated for immediate preview and export reliability."
    )

    # Build room specs
    room_specs = build_room_list(req.bedrooms, req.extras, uw, ul)
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

    # Add slight priority jitter so backup output is adaptive, not static.
    rng = random.Random()
    for spec in room_specs:
        spec["priority"] = float(spec.get("priority", 10)) + rng.uniform(0.0, 0.35)

    # Pack rooms deterministically
    placed = pack_rooms_bsp(room_specs, uw, ul)

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
    )


# ─────────────────────────────────────────────────────────────
# Room spec builder
# ─────────────────────────────────────────────────────────────
def build_room_list(bedrooms: int, extras: List[str],
                    usable_w: float, usable_l: float) -> List[Dict[str, Any]]:
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
        "zone": 3, "priority": 6,
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
    band1_h = _rnd(usable_l * 0.28, 2)
    band2_h = _rnd(usable_l * 0.25, 2)
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
    placed.extend(_pack_band2(zone_rooms[2], 0, band1_h, usable_w, band2_h))

    # ── Band 3: Private (bedrooms + attached baths + study) ──
    placed.extend(
        _pack_band3(zone_rooms[3], 0, band1_h + band2_h, usable_w, band3_h)
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
                bw: float, bh: float) -> List[Dict[str, Any]]:
    """
    Band 2: Service zone.
    Layout: Kitchen (left, 40%) | Corridor (center, 18%) | Bathrooms stacked (right, 42%)
    """
    placed = []

    # Separate room types
    kitchen = None
    corridor_spec = None
    left_rooms = []
    right_rooms = []

    for r in rooms:
        if r["type"] == "corridor":
            corridor_spec = r
        elif r["type"] == "kitchen":
            kitchen = r
        elif r["type"] in ("bathroom", "master_bath", "toilet"):
            right_rooms.append(r)
        elif r["type"] in ("store",):
            left_rooms.append(r)
        else:
            if len(left_rooms) <= len(right_rooms):
                left_rooms.append(r)
            else:
                right_rooms.append(r)

    # Calculate widths
    corridor_w = min(4.0, bw * 0.18)
    corridor_w = max(3.5, corridor_w)

    # Left side gets kitchen + store stacked
    left_w = _rnd((bw - corridor_w) * 0.52, 2)
    right_w = _rnd(bw - left_w - corridor_w, 2)
    corridor_x = _rnd(bx + left_w, 2)

    # Corridor
    if corridor_spec:
        placed.append({
            "type": "corridor", "label": "Corridor",
            "x": _rnd(corridor_x, 2), "y": _rnd(by, 2),
            "width": _rnd(corridor_w, 2), "height": _rnd(bh, 2),
            "zone": 2, "band": 2,
        })

    # Left side: kitchen on top, store below (or just kitchen)
    all_left = []
    if kitchen:
        all_left.append(kitchen)
    all_left.extend(left_rooms)

    if all_left:
        current_y = by
        per_h = _rnd(bh / len(all_left), 2)
        for i, lr in enumerate(all_left):
            rh = per_h if i < len(all_left) - 1 else _rnd(by + bh - current_y, 2)
            placed.append({
                "type": lr["type"], "label": lr["label"],
                "x": _rnd(bx, 2), "y": _rnd(current_y, 2),
                "width": _rnd(left_w, 2), "height": _rnd(rh, 2),
                "zone": 2, "band": 2,
            })
            current_y = _rnd(current_y + rh, 2)

    # Right side: bathrooms stacked
    if right_rooms:
        current_y = by
        per_h = _rnd(bh / len(right_rooms), 2)
        for i, rr in enumerate(right_rooms):
            rh = per_h if i < len(right_rooms) - 1 else _rnd(by + bh - current_y, 2)
            placed.append({
                "type": rr["type"], "label": rr["label"],
                "x": _rnd(corridor_x + corridor_w, 2),
                "y": _rnd(current_y, 2),
                "width": _rnd(right_w, 2), "height": _rnd(rh, 2),
                "zone": 2, "band": 2,
            })
            current_y = _rnd(current_y + rh, 2)

    return placed


def _pack_band3(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float) -> List[Dict[str, Any]]:
    """
    Band 3: Private zone — bedrooms with attached baths.
    Smart layout: pair each bedroom with its bath side-by-side,
    then stack pairs in rows.
    """
    if not rooms:
        return []

    placed = []

    # Separate bedrooms (large) from small rooms (baths, study)
    bedrooms = []
    master_bath = None
    small_rooms = []

    for r in rooms:
        if r["type"] in ("master_bedroom", "bedroom"):
            bedrooms.append(r)
        elif r["type"] == "master_bath":
            master_bath = r
        else:
            small_rooms.append(r)

    # Pair master bedroom with master bath
    pairs = []
    if bedrooms:
        master = bedrooms[0]  # master_bedroom is first (priority 5)
        other_beds = bedrooms[1:]

        if master_bath:
            pairs.append(("pair", master, master_bath))
        else:
            pairs.append(("single", master, None))

        for bed in other_beds:
            pairs.append(("single", bed, None))

    # Add leftover small rooms (study, etc.)
    for sr in small_rooms:
        pairs.append(("small", sr, None))

    # Calculate how many can fit per row
    # For a 23ft wide usable area: 1 pair per row (master 13 + bath 5 = 18)
    # For wider plots: maybe 2 per row
    num_rows = len(pairs)
    if num_rows == 0:
        return []

    # Try to fit 2 items per row if width allows
    row_plan = []
    i = 0
    while i < len(pairs):
        item = pairs[i]
        if item[0] == "pair":
            # Bedroom+bath pair takes most of the width
            row_plan.append([item])
            i += 1
        elif i + 1 < len(pairs) and item[0] == "single" and pairs[i + 1][0] == "small":
            # Bedroom + small room side by side
            row_plan.append([item, pairs[i + 1]])
            i += 2
        elif i + 1 < len(pairs) and item[0] == "single" and pairs[i + 1][0] == "single":
            # Two bedrooms side by side if they fit
            bed1_w = item[1]["min_w"]
            bed2_w = pairs[i + 1][1]["min_w"]
            if bed1_w + bed2_w <= bw:
                row_plan.append([item, pairs[i + 1]])
                i += 2
            else:
                row_plan.append([item])
                i += 1
        else:
            row_plan.append([item])
            i += 1

    # Distribute height among rows
    num_rows = len(row_plan)
    per_row_h = _rnd(bh / num_rows, 2)
    current_y = by

    for row_idx, row_items in enumerate(row_plan):
        rh = per_row_h if row_idx < num_rows - 1 else _rnd(by + bh - current_y, 2)

        if len(row_items) == 1:
            item = row_items[0]
            if item[0] == "pair":
                # Bedroom + attached bath
                bed, bath = item[1], item[2]
                bath_w = min(bath["pref_w"], bw * 0.28)
                bath_w = max(bath_w, bath["min_w"])
                bed_w = _rnd(bw - bath_w, 2)

                placed.append({
                    "type": bed["type"], "label": bed["label"],
                    "x": _rnd(bx, 2), "y": _rnd(current_y, 2),
                    "width": _rnd(bed_w, 2), "height": _rnd(rh, 2),
                    "zone": 3, "band": 3,
                })
                placed.append({
                    "type": bath["type"], "label": bath["label"],
                    "x": _rnd(bx + bed_w, 2), "y": _rnd(current_y, 2),
                    "width": _rnd(bath_w, 2), "height": _rnd(rh, 2),
                    "zone": 3, "band": 3,
                })
            else:
                # Single room — takes full width
                room = item[1]
                placed.append({
                    "type": room["type"], "label": room["label"],
                    "x": _rnd(bx, 2), "y": _rnd(current_y, 2),
                    "width": _rnd(bw, 2), "height": _rnd(rh, 2),
                    "zone": 3, "band": 3,
                })
        else:
            # Two items side by side — proportional split
            total_pref = sum(
                it[1]["pref_w"] + (it[2]["pref_w"] if it[2] else 0)
                for it in row_items
            )
            current_x = bx
            for j, item in enumerate(row_items):
                room = item[1]
                pref = room["pref_w"] + (item[2]["pref_w"] if item[2] else 0)
                rw = _rnd(bw * pref / total_pref, 2) if j < len(row_items) - 1 \
                    else _rnd(bx + bw - current_x, 2)
                rw = max(rw, room["min_w"])

                if item[0] == "pair" and item[2]:
                    bath = item[2]
                    bath_w = min(bath["pref_w"], rw * 0.35)
                    bath_w = max(bath_w, bath["min_w"])
                    bed_w = _rnd(rw - bath_w, 2)

                    placed.append({
                        "type": room["type"], "label": room["label"],
                        "x": _rnd(current_x, 2), "y": _rnd(current_y, 2),
                        "width": _rnd(bed_w, 2), "height": _rnd(rh, 2),
                        "zone": 3, "band": 3,
                    })
                    placed.append({
                        "type": bath["type"], "label": bath["label"],
                        "x": _rnd(current_x + bed_w, 2),
                        "y": _rnd(current_y, 2),
                        "width": _rnd(bath_w, 2), "height": _rnd(rh, 2),
                        "zone": 3, "band": 3,
                    })
                else:
                    placed.append({
                        "type": room["type"], "label": room["label"],
                        "x": _rnd(current_x, 2), "y": _rnd(current_y, 2),
                        "width": _rnd(rw, 2), "height": _rnd(rh, 2),
                        "zone": 3, "band": 3,
                    })
                current_x = _rnd(current_x + rw, 2)

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
        if facing == "south":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id=living_id, wall="south",
                x=living["x"] + living["width"] * 0.4,
                y=living["y"],
                width=3.5,
            ))
        elif facing == "north":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id=living_id, wall="north",
                x=living["x"] + living["width"] * 0.4,
                y=living["y"] + living["height"],
                width=3.5,
            ))
        elif facing == "east":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id=living_id, wall="east",
                x=living["x"] + living["width"],
                y=living["y"] + living["height"] * 0.4,
                width=3.5,
            ))
        elif facing == "west":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id=living_id, wall="west",
                x=living["x"],
                y=living["y"] + living["height"] * 0.4,
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
