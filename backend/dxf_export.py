"""
DXF export — converts a PlanResponse into an AutoCAD DXF file using ezdxf.
"""
from __future__ import annotations
import math
import os
import ezdxf
from ezdxf.enums import TextEntityAlignment
from models import PlanResponse

# ── Conversion ───────────────────────────────────────────────
FT_TO_MM = 304.8   # 1 foot = 304.8 mm
WALL_OFFSET_MM = 115  # 4.5 inches inner wall offset

# ── Room colors (by AutoCAD color index) ─────────────────────
ROOM_COLORS = {
    "living": 150,       # light blue
    "dining": 190,       # light purple
    "kitchen": 80,       # green
    "master_bedroom": 40,  # yellow
    "bedroom": 30,       # orange
    "bathroom": 140,     # teal
    "toilet": 140,
    "pooja": 50,         # gold
    "study": 200,        # lavender
    "store": 8,          # gray
    "corridor": 9,       # light gray
    "balcony": 150,
    "garage": 8,
}


def plan_to_dxf(plan: PlanResponse, filepath: str) -> str:
    """Generate a DXF file and return the file path."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # ── Create layers ────────────────────────────────────────
    _create_layers(doc)

    pw = plan.plot.width * FT_TO_MM
    pl = plan.plot.length * FT_TO_MM
    uw = plan.plot.usable_width * FT_TO_MM
    ul = plan.plot.usable_length * FT_TO_MM

    sb = plan.plot.setbacks
    offset_x = sb.get("left", 3.5) * FT_TO_MM
    offset_y = sb.get("front", 6.5) * FT_TO_MM

    # ── Plot boundary ────────────────────────────────────────
    msp.add_lwpolyline(
        [(0, 0), (pw, 0), (pw, pl), (0, pl), (0, 0)],
        dxfattribs={"layer": "BOUNDARY", "lineweight": 50},
    )

    # ── Usable area (dashed) ─────────────────────────────────
    msp.add_lwpolyline(
        [
            (offset_x, offset_y),
            (offset_x + uw, offset_y),
            (offset_x + uw, offset_y + ul),
            (offset_x, offset_y + ul),
            (offset_x, offset_y),
        ],
        dxfattribs={"layer": "BOUNDARY", "linetype": "DASHED"},
    )

    # ── Rooms ────────────────────────────────────────────────
    for room in plan.rooms:
        rx = offset_x + room.x * FT_TO_MM
        ry = offset_y + room.y * FT_TO_MM
        rw = room.width * FT_TO_MM
        rh = room.height * FT_TO_MM

        # Outer wall
        msp.add_lwpolyline(
            [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh), (rx, ry)],
            dxfattribs={"layer": "WALLS", "lineweight": 35},
        )

        # Inner wall (double-wall effect)
        wo = WALL_OFFSET_MM
        if rw > wo * 3 and rh > wo * 3:
            msp.add_lwpolyline(
                [
                    (rx + wo, ry + wo),
                    (rx + rw - wo, ry + wo),
                    (rx + rw - wo, ry + rh - wo),
                    (rx + wo, ry + rh - wo),
                    (rx + wo, ry + wo),
                ],
                dxfattribs={"layer": "WALL_INNER"},
            )

        # Room label
        cx = rx + rw / 2
        cy = ry + rh / 2
        font_h = rw * 0.04
        font_h = max(font_h, 120)
        font_h = min(font_h, 350)

        msp.add_text(
            room.label,
            height=font_h,
            dxfattribs={"layer": "LABELS"},
        ).set_placement(
            (cx, cy + font_h * 0.3),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

        # Area text below label
        area_text = f"{room.area:.0f} sq.ft"
        msp.add_text(
            area_text,
            height=font_h * 0.65,
            dxfattribs={"layer": "LABELS"},
        ).set_placement(
            (cx, cy - font_h * 0.8),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

    # ── Doors ────────────────────────────────────────────────
    for door in plan.doors:
        dx = offset_x + door.x * FT_TO_MM
        dy = offset_y + door.y * FT_TO_MM
        dw = door.width * FT_TO_MM

        # Door line
        if door.wall in ("south", "north"):
            msp.add_line((dx, dy), (dx + dw, dy), dxfattribs={"layer": "DOORS"})
            # Arc swing
            msp.add_arc(
                center=(dx, dy),
                radius=dw,
                start_angle=0 if door.wall == "south" else 180,
                end_angle=90 if door.wall == "south" else 270,
                dxfattribs={"layer": "DOORS"},
            )
        else:
            msp.add_line((dx, dy), (dx, dy + dw), dxfattribs={"layer": "DOORS"})
            msp.add_arc(
                center=(dx, dy),
                radius=dw,
                start_angle=90 if door.wall == "east" else 270,
                end_angle=180 if door.wall == "east" else 360,
                dxfattribs={"layer": "DOORS"},
            )

    # ── Windows ──────────────────────────────────────────────
    for win in plan.windows:
        wx = offset_x + win.x * FT_TO_MM
        wy = offset_y + win.y * FT_TO_MM
        ww = win.width * FT_TO_MM
        gap = 40  # mm spacing between the three lines

        if win.wall in ("south", "north"):
            for offset in (-gap, 0, gap):
                msp.add_line(
                    (wx, wy + offset),
                    (wx + ww, wy + offset),
                    dxfattribs={"layer": "WINDOWS"},
                )
        else:
            for offset in (-gap, 0, gap):
                msp.add_line(
                    (wx + offset, wy),
                    (wx + offset, wy + ww),
                    dxfattribs={"layer": "WINDOWS"},
                )

    # ── North arrow ──────────────────────────────────────────
    _draw_north_arrow(msp, pw, pl, plan.plot.road_side)

    # ── Title block ──────────────────────────────────────────
    _draw_title_block(msp, plan, pw)

    # ── Dimension lines on plot boundary ─────────────────────
    _draw_dimensions(msp, pw, pl, plan)

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc.saveas(filepath)
    return filepath


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _create_layers(doc):
    layers = {
        "BOUNDARY": {"color": 8},
        "WALLS": {"color": 7, "lineweight": 35},
        "WALL_INNER": {"color": 8},
        "DOORS": {"color": 3},      # green
        "WINDOWS": {"color": 5},     # blue
        "LABELS": {"color": 1},      # red
        "DIMENSIONS": {"color": 6},  # magenta
        "NORTH_ARROW": {"color": 7},
        "TITLE": {"color": 7},
    }
    for name, attrs in layers.items():
        doc.layers.add(name, color=attrs.get("color", 7))


def _draw_north_arrow(msp, pw, pl, road_side):
    """Draw a north arrow in the top-right corner."""
    # Arrow position
    ax = pw - 600
    ay = pl - 600

    # Determine north direction based on road side
    # Road side is south → north is up (default)
    rotation = {
        "south": 0,
        "west": 90,
        "north": 180,
        "east": 270,
    }.get(road_side, 0)

    size = 400
    rad = math.radians(rotation)

    # Triangle points (pointing up by default)
    tip_x = ax + size * 0.5 * math.sin(rad + math.pi)
    tip_y = ay + size * 0.5 * math.cos(rad + math.pi)
    left_x = ax + size * 0.3 * math.sin(rad + math.pi * 0.7)
    left_y = ay + size * 0.3 * math.cos(rad + math.pi * 0.7)
    right_x = ax + size * 0.3 * math.sin(rad + math.pi * 1.3)
    right_y = ay + size * 0.3 * math.cos(rad + math.pi * 1.3)

    # Simple upward triangle
    cx, cy = ax, ay
    pts = [
        (cx, cy + size * 0.5),         # tip (north)
        (cx - size * 0.2, cy - size * 0.3),  # bottom-left
        (cx + size * 0.2, cy - size * 0.3),  # bottom-right
        (cx, cy + size * 0.5),         # close
    ]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "NORTH_ARROW"})

    # "N" label
    msp.add_text(
        "N",
        height=200,
        dxfattribs={"layer": "NORTH_ARROW"},
    ).set_placement(
        (cx, cy + size * 0.7),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )


def _draw_title_block(msp, plan: PlanResponse, plot_width_mm):
    """Title block below the plan."""
    tx = plot_width_mm / 2
    ty = -800  # below the plot

    msp.add_text(
        "RESIDENTIAL FLOOR PLAN",
        height=250,
        dxfattribs={"layer": "TITLE"},
    ).set_placement((tx, ty), align=TextEntityAlignment.MIDDLE_CENTER)

    bhk = sum(1 for r in plan.rooms if r.type in ("master_bedroom", "bedroom"))
    msp.add_text(
        f"Plot: {plan.plot.width:.0f}x{plan.plot.length:.0f} ft | {bhk} BHK | Vastu Compliant",
        height=160,
        dxfattribs={"layer": "TITLE"},
    ).set_placement((tx, ty - 400), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_text(
        "Scale 1:100 | All dimensions in feet",
        height=130,
        dxfattribs={"layer": "TITLE"},
    ).set_placement((tx, ty - 700), align=TextEntityAlignment.MIDDLE_CENTER)


def _draw_dimensions(msp, pw, pl, plan: PlanResponse):
    """Simple dimension text on plot edges."""
    # Bottom edge (width)
    msp.add_text(
        f"{plan.plot.width:.0f} ft",
        height=150,
        dxfattribs={"layer": "DIMENSIONS"},
    ).set_placement(
        (pw / 2, -300),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )
    msp.add_line((0, -150), (pw, -150), dxfattribs={"layer": "DIMENSIONS"})

    # Left edge (length)
    msp.add_text(
        f"{plan.plot.length:.0f} ft",
        height=150,
        rotation=90,
        dxfattribs={"layer": "DIMENSIONS"},
    ).set_placement(
        (-300, pl / 2),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )
    msp.add_line((-150, 0), (-150, pl), dxfattribs={"layer": "DIMENSIONS"})
