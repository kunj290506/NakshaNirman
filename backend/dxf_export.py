"""
DXF export — converts a PlanResponse into a professional AutoCAD-style DXF file.
Includes furniture symbols, hatch patterns, dimension lines, north arrow, and title block.
"""
from __future__ import annotations
import os
import ezdxf
from ezdxf.enums import TextEntityAlignment
from models import PlanResponse

# ── Conversion ───────────────────────────────────────────────
FT_TO_MM = 304.8
WALL_OFFSET_MM = 115  # 4.5 inches inner wall offset


def plan_to_dxf(plan: PlanResponse, filepath: str) -> str:
    """Generate a professional DXF file and return the file path."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # ── Create layers ────────────────────────────────────────
    _create_layers(doc)

    # ── Create text styles ───────────────────────────────────
    doc.styles.add("STANDARD", font="Arial")
    doc.styles.add("TITLE", font="Arial")

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
            dxfattribs={"layer": "WALLS", "lineweight": 50},
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
                dxfattribs={"layer": "WALL_INNER", "lineweight": 13},
            )

        # Room label
        cx = rx + rw / 2
        cy = ry + rh / 2
        font_h = min(max(rw * 0.04, 120), 350)

        msp.add_text(
            room.label.upper(),
            height=font_h,
            dxfattribs={"layer": "LABELS", "style": "STANDARD"},
        ).set_placement(
            (cx, cy + font_h * 0.4),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

        # Area text
        area_text = f"{room.area:.0f} sq.ft"
        msp.add_text(
            area_text,
            height=font_h * 0.6,
            dxfattribs={"layer": "LABELS", "style": "STANDARD"},
        ).set_placement(
            (cx, cy - font_h * 0.6),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

        # Dimensions text
        dim_text = f"{room.width:.1f}' × {room.height:.1f}'"
        msp.add_text(
            dim_text,
            height=font_h * 0.5,
            dxfattribs={"layer": "DIMENSIONS", "style": "STANDARD"},
        ).set_placement(
            (cx, cy - font_h * 1.3),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

        # Hatch patterns
        _add_hatch(msp, room.type, rx, ry, rw, rh)

        # Furniture symbols
        _draw_furniture(msp, room.type, rx + wo, ry + wo,
                        rw - wo * 2, rh - wo * 2)

    # ── Doors ────────────────────────────────────────────────
    for door in plan.doors:
        dx = offset_x + door.x * FT_TO_MM
        dy = offset_y + door.y * FT_TO_MM
        dw = door.width * FT_TO_MM

        # White gap to break wall
        _draw_door_gap(msp, dx, dy, dw, door.wall)

        # Door swing arc
        if door.wall in ("south", "north"):
            msp.add_line(
                (dx, dy), (dx + dw, dy),
                dxfattribs={"layer": "DOORS"},
            )
            msp.add_arc(
                center=(dx, dy),
                radius=dw,
                start_angle=0 if door.wall == "south" else 180,
                end_angle=90 if door.wall == "south" else 270,
                dxfattribs={"layer": "DOORS"},
            )
        else:
            msp.add_line(
                (dx, dy), (dx, dy + dw),
                dxfattribs={"layer": "DOORS"},
            )
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
        gap = 40  # mm spacing

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

    # ── Dimension lines ──────────────────────────────────────
    _draw_dimensions(msp, pw, pl, plan)

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc.saveas(filepath)
    return filepath


# ─────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────
def _create_layers(doc):
    layers = {
        "BOUNDARY": {"color": 8},
        "WALLS": {"color": 7, "lineweight": 50},
        "WALL_INNER": {"color": 8, "lineweight": 13},
        "DOORS": {"color": 3},
        "WINDOWS": {"color": 5},
        "LABELS": {"color": 7},
        "DIMENSIONS": {"color": 6},
        "FURNITURE": {"color": 8},
        "NORTH_ARROW": {"color": 7},
        "TITLE": {"color": 7},
        "HATCH": {"color": 9},
    }
    for name, attrs in layers.items():
        doc.layers.add(name, color=attrs.get("color", 7))


# ─────────────────────────────────────────────────────────────
# Hatch patterns
# ─────────────────────────────────────────────────────────────
def _add_hatch(msp, room_type: str, rx, ry, rw, rh):
    """Add hatch patterns for specific room types."""
    if room_type in ("bathroom", "master_bath", "toilet"):
        # Diagonal lines for wet areas
        spacing = 150
        x1, y1, y2 = rx, ry, ry + rh
        d = 0
        while d < rw + rh:
            lx1 = x1 + min(d, rw)
            ly1 = y1 + max(0, d - rw)
            lx2 = x1 + max(0, d - rh)
            ly2 = y1 + min(d, rh)
            if lx1 >= x1 and lx2 >= x1 and ly1 <= y2 and ly2 <= y2:
                msp.add_line(
                    (lx1, ly1), (lx2, ly2),
                    dxfattribs={"layer": "HATCH"},
                )
            d += spacing
    elif room_type == "kitchen":
        # Cross hatch for kitchen
        spacing = 200
        for y in range(int(ry), int(ry + rh), spacing):
            msp.add_line(
                (rx, y), (rx + rw, y),
                dxfattribs={"layer": "HATCH"},
            )
        for x in range(int(rx), int(rx + rw), spacing):
            msp.add_line(
                (x, ry), (x, ry + rh),
                dxfattribs={"layer": "HATCH"},
            )


# ─────────────────────────────────────────────────────────────
# Door gap helper
# ─────────────────────────────────────────────────────────────
def _draw_door_gap(msp, dx, dy, dw, wall):
    """Draw a white gap in the wall at door position."""
    gap_h = 200  # mm
    if wall in ("south", "north"):
        pts = [
            (dx - 20, dy - gap_h / 2),
            (dx + dw + 20, dy - gap_h / 2),
            (dx + dw + 20, dy + gap_h / 2),
            (dx - 20, dy + gap_h / 2),
            (dx - 20, dy - gap_h / 2),
        ]
    else:
        pts = [
            (dx - gap_h / 2, dy - 20),
            (dx + gap_h / 2, dy - 20),
            (dx + gap_h / 2, dy + dw + 20),
            (dx - gap_h / 2, dy + dw + 20),
            (dx - gap_h / 2, dy - 20),
        ]
    # White fill polyline (color 7 = white in dark background, but in plotting it's fine)
    msp.add_lwpolyline(pts, dxfattribs={"layer": "DOORS"})


# ─────────────────────────────────────────────────────────────
# Furniture drawing
# ─────────────────────────────────────────────────────────────
def _draw_furniture(msp, room_type: str, rx, ry, rw, rh):
    """Draw furniture symbols inside a room on the FURNITURE layer."""
    layer = "FURNITURE"

    if room_type in ("master_bedroom", "bedroom"):
        _draw_bed(msp, rx, ry, rw, rh, room_type == "master_bedroom", layer)
    elif room_type in ("bathroom", "master_bath", "toilet"):
        _draw_bathroom_fixtures(msp, rx, ry, rw, rh,
                                room_type == "master_bath", layer)
    elif room_type == "living":
        _draw_sofa(msp, rx, ry, rw, rh, layer)
    elif room_type == "dining":
        _draw_dining_table(msp, rx, ry, rw, rh, layer)
    elif room_type == "kitchen":
        _draw_kitchen_fixtures(msp, rx, ry, rw, rh, layer)


def _draw_bed(msp, rx, ry, rw, rh, is_master, layer):
    """Draw bed with pillows and nightstand."""
    bed_w = rw * 0.65
    bed_h = rh * 0.50
    bx = rx + (rw - bed_w) / 2
    by = ry + rh - bed_h - rh * 0.1

    # Bed frame
    msp.add_lwpolyline(
        [(bx, by), (bx + bed_w, by), (bx + bed_w, by + bed_h),
         (bx, by + bed_h), (bx, by)],
        dxfattribs={"layer": layer},
    )

    # Headboard
    msp.add_line(
        (bx, by + bed_h), (bx + bed_w, by + bed_h),
        dxfattribs={"layer": layer, "lineweight": 25},
    )

    # Pillows
    pillow_h = bed_h * 0.15
    if is_master:
        # Two pillows
        pw = bed_w * 0.45
        msp.add_lwpolyline(
            _rounded_rect(bx + bed_w * 0.025, by + bed_h - pillow_h - 20,
                          pw, pillow_h),
            dxfattribs={"layer": layer},
        )
        msp.add_lwpolyline(
            _rounded_rect(bx + bed_w * 0.525, by + bed_h - pillow_h - 20,
                          pw, pillow_h),
            dxfattribs={"layer": layer},
        )
        # Nightstand
        ns = min(200, rw * 0.08)
        msp.add_lwpolyline(
            [(bx - ns - 30, by + bed_h - ns),
             (bx - 30, by + bed_h - ns),
             (bx - 30, by + bed_h),
             (bx - ns - 30, by + bed_h),
             (bx - ns - 30, by + bed_h - ns)],
            dxfattribs={"layer": layer},
        )
    else:
        # One pillow
        msp.add_lwpolyline(
            _rounded_rect(bx + bed_w * 0.1, by + bed_h - pillow_h - 20,
                          bed_w * 0.8, pillow_h),
            dxfattribs={"layer": layer},
        )


def _draw_bathroom_fixtures(msp, rx, ry, rw, rh, is_master, layer):
    """Draw toilet, sink, and shower."""
    # Toilet (oval + tank)
    tc_x = rx + rw - rw * 0.25
    tc_y = ry + rh * 0.15
    bowl_rx = min(150, rw * 0.12)
    bowl_ry = min(200, rh * 0.15)
    if bowl_ry <= bowl_rx:
        msp.add_ellipse(
            center=(tc_x, tc_y + bowl_ry),
            major_axis=(bowl_rx, 0, 0),
            ratio=bowl_ry / max(bowl_rx, 1),
            dxfattribs={"layer": layer},
        )
    else:
        msp.add_ellipse(
            center=(tc_x, tc_y + bowl_ry),
            major_axis=(0, bowl_ry, 0),
            ratio=bowl_rx / max(bowl_ry, 1),
            dxfattribs={"layer": layer},
        )
    # Tank
    tank_w = bowl_rx * 1.6
    tank_h = bowl_ry * 0.6
    msp.add_lwpolyline(
        [(tc_x - tank_w / 2, tc_y - tank_h / 2),
         (tc_x + tank_w / 2, tc_y - tank_h / 2),
         (tc_x + tank_w / 2, tc_y + tank_h / 2),
         (tc_x - tank_w / 2, tc_y + tank_h / 2),
         (tc_x - tank_w / 2, tc_y - tank_h / 2)],
        dxfattribs={"layer": layer},
    )

    # Sink (circle)
    sink_r = min(100, rw * 0.06)
    msp.add_circle(
        center=(rx + rw * 0.2, ry + rh - rh * 0.15),
        radius=sink_r,
        dxfattribs={"layer": layer},
    )

    # Shower area (corner dashed lines)
    sh_w = rw * 0.4
    sh_h = rh * 0.4
    msp.add_line(
        (rx, ry + sh_h), (rx + sh_w, ry + sh_h),
        dxfattribs={"layer": layer, "linetype": "DASHED"},
    )
    msp.add_line(
        (rx + sh_w, ry), (rx + sh_w, ry + sh_h),
        dxfattribs={"layer": layer, "linetype": "DASHED"},
    )

    # Bathtub for master
    if is_master:
        bt_w = rw * 0.6
        bt_h = rh * 0.2
        btx = rx + (rw - bt_w) / 2
        bty = ry + rh * 0.5
        msp.add_lwpolyline(
            [(btx, bty), (btx + bt_w, bty), (btx + bt_w, bty + bt_h),
             (btx, bty + bt_h), (btx, bty)],
            dxfattribs={"layer": layer},
        )
        rx_val = bt_w * 0.4
        ry_val = bt_h * 0.35
        if ry_val <= rx_val:
            msp.add_ellipse(
                center=(btx + bt_w / 2, bty + bt_h / 2),
                major_axis=(rx_val, 0, 0),
                ratio=ry_val / max(rx_val, 1),
                dxfattribs={"layer": layer},
            )
        else:
            msp.add_ellipse(
                center=(btx + bt_w / 2, bty + bt_h / 2),
                major_axis=(0, ry_val, 0),
                ratio=rx_val / max(ry_val, 1),
                dxfattribs={"layer": layer},
            )


def _draw_sofa(msp, rx, ry, rw, rh, layer):
    """Draw L-shaped sofa and coffee table."""
    # Sofa
    sw = rw * 0.6
    sh = rh * 0.18
    sx = rx + rw * 0.2
    sy = ry + rh - sh - rh * 0.1

    # Main seat
    msp.add_lwpolyline(
        [(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh), (sx, sy)],
        dxfattribs={"layer": layer},
    )
    # Back
    back_h = sh * 0.25
    msp.add_lwpolyline(
        [(sx, sy + sh), (sx + sw, sy + sh),
         (sx + sw, sy + sh + back_h), (sx, sy + sh + back_h),
         (sx, sy + sh)],
        dxfattribs={"layer": layer},
    )
    # Armrests
    arm_w = sh * 0.3
    msp.add_lwpolyline(
        [(sx - arm_w, sy), (sx, sy), (sx, sy + sh), (sx - arm_w, sy + sh),
         (sx - arm_w, sy)],
        dxfattribs={"layer": layer},
    )
    msp.add_lwpolyline(
        [(sx + sw, sy), (sx + sw + arm_w, sy),
         (sx + sw + arm_w, sy + sh), (sx + sw, sy + sh),
         (sx + sw, sy)],
        dxfattribs={"layer": layer},
    )

    # Coffee table
    ct_w = sw * 0.45
    ct_h = sh * 0.6
    ctx = sx + (sw - ct_w) / 2
    cty = sy - ct_h - rh * 0.05
    msp.add_lwpolyline(
        [(ctx, cty), (ctx + ct_w, cty), (ctx + ct_w, cty + ct_h),
         (ctx, cty + ct_h), (ctx, cty)],
        dxfattribs={"layer": layer},
    )


def _draw_dining_table(msp, rx, ry, rw, rh, layer):
    """Draw dining table with chairs."""
    tw = rw * 0.5
    th = rh * 0.4
    tx = rx + (rw - tw) / 2
    ty = ry + (rh - th) / 2

    # Table
    msp.add_lwpolyline(
        [(tx, ty), (tx + tw, ty), (tx + tw, ty + th), (tx, ty + th), (tx, ty)],
        dxfattribs={"layer": layer},
    )

    # Chairs (top and bottom)
    ch_w = tw * 0.2
    ch_h = th * 0.15
    for i in range(3):
        cx = tx + tw * (i + 1) / 4 - ch_w / 2
        # Top chairs
        msp.add_lwpolyline(
            [(cx, ty - ch_h - 20), (cx + ch_w, ty - ch_h - 20),
             (cx + ch_w, ty - 20), (cx, ty - 20),
             (cx, ty - ch_h - 20)],
            dxfattribs={"layer": layer},
        )
        # Bottom chairs
        msp.add_lwpolyline(
            [(cx, ty + th + 20), (cx + ch_w, ty + th + 20),
             (cx + ch_w, ty + th + ch_h + 20), (cx, ty + th + ch_h + 20),
             (cx, ty + th + 20)],
            dxfattribs={"layer": layer},
        )

    # End chairs
    ecy = ty + (th - ch_w) / 2
    msp.add_lwpolyline(
        [(tx - ch_h - 20, ecy), (tx - 20, ecy),
         (tx - 20, ecy + ch_w), (tx - ch_h - 20, ecy + ch_w),
         (tx - ch_h - 20, ecy)],
        dxfattribs={"layer": layer},
    )
    msp.add_lwpolyline(
        [(tx + tw + 20, ecy), (tx + tw + ch_h + 20, ecy),
         (tx + tw + ch_h + 20, ecy + ch_w), (tx + tw + 20, ecy + ch_w),
         (tx + tw + 20, ecy)],
        dxfattribs={"layer": layer},
    )


def _draw_kitchen_fixtures(msp, rx, ry, rw, rh, layer):
    """Draw L-shaped counter, sink, stove, fridge."""
    # Counter along top wall
    cw = rw * 0.85
    ch = min(rh * 0.2, 500)
    cx = rx + (rw - cw) / 2
    cy = ry + rh - ch

    msp.add_lwpolyline(
        [(cx, cy), (cx + cw, cy), (cx + cw, cy + ch), (cx, cy + ch), (cx, cy)],
        dxfattribs={"layer": layer},
    )

    # Side counter (L-shape)
    sc_w = min(rw * 0.15, 350)
    sc_h = rh * 0.5
    msp.add_lwpolyline(
        [(cx, cy - sc_h), (cx + sc_w, cy - sc_h),
         (cx + sc_w, cy), (cx, cy),
         (cx, cy - sc_h)],
        dxfattribs={"layer": layer},
    )

    # Sink (rectangle with X)
    sink_w = cw * 0.15
    sink_h = ch * 0.6
    sx = cx + cw * 0.3
    sy = cy + (ch - sink_h) / 2
    msp.add_lwpolyline(
        [(sx, sy), (sx + sink_w, sy), (sx + sink_w, sy + sink_h),
         (sx, sy + sink_h), (sx, sy)],
        dxfattribs={"layer": layer},
    )
    msp.add_line((sx, sy), (sx + sink_w, sy + sink_h), dxfattribs={"layer": layer})
    msp.add_line((sx + sink_w, sy), (sx, sy + sink_h), dxfattribs={"layer": layer})

    # Stove burners (4 circles)
    burner_r = min(60, ch * 0.15)
    bx = cx + cw * 0.65
    by = cy + ch / 2
    for dx, dy in [(-burner_r * 1.3, -burner_r * 1.1),
                   (burner_r * 1.3, -burner_r * 1.1),
                   (-burner_r * 1.3, burner_r * 1.1),
                   (burner_r * 1.3, burner_r * 1.1)]:
        msp.add_circle(
            center=(bx + dx, by + dy), radius=burner_r,
            dxfattribs={"layer": layer},
        )

    # Fridge
    fr_w = min(250, rw * 0.1)
    fr_h = min(350, rh * 0.15)
    msp.add_lwpolyline(
        [(rx + rw - fr_w, ry), (rx + rw, ry),
         (rx + rw, ry + fr_h), (rx + rw - fr_w, ry + fr_h),
         (rx + rw - fr_w, ry)],
        dxfattribs={"layer": layer},
    )
    msp.add_text(
        "REF", height=80,
        dxfattribs={"layer": layer},
    ).set_placement(
        (rx + rw - fr_w / 2, ry + fr_h / 2),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )


def _rounded_rect(x, y, w, h):
    """Return points for a simple rectangle (rounded effect in actual CAD)."""
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]


# ─────────────────────────────────────────────────────────────
# North arrow
# ─────────────────────────────────────────────────────────────
def _draw_north_arrow(msp, pw, pl, road_side):
    """Draw a north arrow in the top-right corner."""
    cx = pw - 600
    cy = pl - 600
    size = 400

    # Triangle pointing up (north)
    pts = [
        (cx, cy + size * 0.5),
        (cx - size * 0.2, cy - size * 0.3),
        (cx + size * 0.2, cy - size * 0.3),
        (cx, cy + size * 0.5),
    ]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "NORTH_ARROW"})

    # Fill half for 3D effect
    msp.add_lwpolyline(
        [(cx, cy + size * 0.5), (cx + size * 0.2, cy - size * 0.3),
         (cx, cy - size * 0.1), (cx, cy + size * 0.5)],
        dxfattribs={"layer": "NORTH_ARROW"},
    )

    # "N" label
    msp.add_text(
        "N", height=200,
        dxfattribs={"layer": "NORTH_ARROW"},
    ).set_placement(
        (cx, cy + size * 0.75),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )


# ─────────────────────────────────────────────────────────────
# Title block
# ─────────────────────────────────────────────────────────────
def _draw_title_block(msp, plan: PlanResponse, plot_width_mm):
    """Title block below the plan."""
    tx = plot_width_mm / 2
    ty = -800

    msp.add_text(
        "RESIDENTIAL FLOOR PLAN",
        height=250,
        dxfattribs={"layer": "TITLE", "style": "TITLE"},
    ).set_placement((tx, ty), align=TextEntityAlignment.MIDDLE_CENTER)

    bhk = sum(1 for r in plan.rooms if r.type in ("master_bedroom", "bedroom"))
    msp.add_text(
        f"Plot: {plan.plot.width:.0f}' x {plan.plot.length:.0f}' | "
        f"{bhk} BHK | Vastu Score: {plan.vastu_score:.0f}/100",
        height=160,
        dxfattribs={"layer": "TITLE"},
    ).set_placement((tx, ty - 400), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_text(
        f"Road Facing: {plan.plot.road_side.capitalize()} | Scale 1:100 | All dimensions in feet",
        height=130,
        dxfattribs={"layer": "TITLE"},
    ).set_placement((tx, ty - 700), align=TextEntityAlignment.MIDDLE_CENTER)


# ─────────────────────────────────────────────────────────────
# Dimension lines
# ─────────────────────────────────────────────────────────────
def _draw_dimensions(msp, pw, pl, plan: PlanResponse):
    """Draw dimension lines on plot edges with tick marks."""
    # ── Bottom edge (width) ──
    dim_y = -200
    msp.add_line((0, dim_y), (pw, dim_y), dxfattribs={"layer": "DIMENSIONS"})
    # Tick marks
    msp.add_line((0, dim_y - 80), (0, dim_y + 80),
                 dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((pw, dim_y - 80), (pw, dim_y + 80),
                 dxfattribs={"layer": "DIMENSIONS"})
    # Extension lines
    msp.add_line((0, 0), (0, dim_y + 80), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((pw, 0), (pw, dim_y + 80), dxfattribs={"layer": "DIMENSIONS"})
    # Text
    msp.add_text(
        f"{plan.plot.width:.0f}'",
        height=150,
        dxfattribs={"layer": "DIMENSIONS"},
    ).set_placement(
        (pw / 2, dim_y - 200),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )

    # ── Left edge (length) ──
    dim_x = -200
    msp.add_line((dim_x, 0), (dim_x, pl), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((dim_x - 80, 0), (dim_x + 80, 0),
                 dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((dim_x - 80, pl), (dim_x + 80, pl),
                 dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((0, 0), (dim_x + 80, 0), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_line((0, pl), (dim_x + 80, pl), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text(
        f"{plan.plot.length:.0f}'",
        height=150,
        rotation=90,
        dxfattribs={"layer": "DIMENSIONS"},
    ).set_placement(
        (dim_x - 200, pl / 2),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )
