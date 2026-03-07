import { useMemo } from 'react'

/* ══════════════════════════════════════════════════════════════
   Architectural Floor Plan Renderer
   Professional CAD-style black & white drawing
   ══════════════════════════════════════════════════════════════ */

const ROOM_LABELS = {
    living:         'DRAWING ROOM',
    dining:         'DINING',
    kitchen:        'KITCHEN',
    master_bedroom: 'BED ROOM',
    bedroom:        'BED ROOM',
    bathroom:       'BATH',
    toilet:         'TOILET',
    study:          'STUDY',
    pooja:          'PUJA GHR',
    store:          'STORE ROOM',
    balcony:        'BALCONY',
    garage:         'CAR PARK',
    hallway:        'HALL',
    utility:        'UTILITY',
    foyer:          'SIT OUT',
    passage:        'PASSAGE',
    entrance:       'ENTRANCE',
    staircase:      'STAIRCASE',
    hallway:        'HALL',
    corridor:       'CORRIDOR',
    lobby:          'LOBBY',
}

/* ── Feet-inch formatter ── */
function fmtDim(ft) {
    const feet = Math.floor(ft)
    const inches = Math.round((ft - feet) * 12)
    if (inches === 0) return `${feet}'-0"`
    if (inches === 12) return `${feet + 1}'-0"`
    return `${feet}'-${inches}"`
}

function fmtDimPair(w, h) {
    return `${fmtDim(w)}X${fmtDim(h)}`
}

/* ──────────────────────────────────────────────
   Furniture Symbols (CAD line-drawing style)
   ────────────────────────────────────────────── */
function furnitureBed(cx, cy, w, h) {
    const bw = Math.min(w * 0.65, h * 0.75)
    const bh = bw * 0.6
    return (
        <g>
            <rect x={cx - bw/2} y={cy - bh/2} width={bw} height={bh}
                fill="none" stroke="#000" strokeWidth={0.15} />
            <rect x={cx - bw/2} y={cy - bh/2} width={bw} height={bh * 0.1}
                fill="#000" />
            <rect x={cx - bw/2 + bw*0.05} y={cy - bh/2 + bh*0.14}
                width={bw*0.42} height={bh*0.18}
                fill="none" stroke="#000" strokeWidth={0.1} rx={0.1} />
            <rect x={cx + bw/2 - bw*0.47} y={cy - bh/2 + bh*0.14}
                width={bw*0.42} height={bh*0.18}
                fill="none" stroke="#000" strokeWidth={0.1} rx={0.1} />
        </g>
    )
}

function furnitureSofa(cx, cy, w, h) {
    const sw = Math.min(w * 0.6, h * 0.65)
    const sh = sw * 0.4
    return (
        <g>
            <rect x={cx - sw/2} y={cy - sh/2} width={sw} height={sh}
                fill="none" stroke="#000" strokeWidth={0.15} />
            <rect x={cx - sw/2} y={cy - sh/2} width={sw} height={sh * 0.2}
                fill="#444" />
            <rect x={cx - sw/2} y={cy - sh/2} width={sw * 0.07} height={sh}
                fill="#444" />
            <rect x={cx + sw/2 - sw*0.07} y={cy - sh/2} width={sw * 0.07} height={sh}
                fill="#444" />
            {/* Coffee table */}
            <ellipse cx={cx} cy={cy + sh*0.9} rx={sw*0.2} ry={sw*0.1}
                fill="none" stroke="#000" strokeWidth={0.08} />
        </g>
    )
}

function furnitureKitchen(cx, cy, w, h) {
    const kw = Math.min(w * 0.8, 8)
    const kh = Math.min(h * 0.2, 2)
    const ky = cy - h * 0.4
    return (
        <g>
            <rect x={cx - kw/2} y={ky} width={kw} height={kh}
                fill="none" stroke="#000" strokeWidth={0.15} />
            <rect x={cx - kw*0.15} y={ky + kh*0.15} width={kw*0.2} height={kh*0.7}
                fill="none" stroke="#000" strokeWidth={0.08} rx={0.08} />
            <circle cx={cx + kw*0.25} cy={ky + kh*0.5} r={kh*0.2}
                fill="none" stroke="#000" strokeWidth={0.08} />
            <circle cx={cx + kw*0.35} cy={ky + kh*0.5} r={kh*0.15}
                fill="none" stroke="#000" strokeWidth={0.08} />
        </g>
    )
}

function furnitureBathroom(cx, cy, w, h) {
    const s = Math.min(w, h) * 0.2
    return (
        <g>
            {/* WC */}
            <ellipse cx={cx + w*0.2} cy={cy + h*0.15} rx={s*0.5} ry={s*0.65}
                fill="none" stroke="#000" strokeWidth={0.12} />
            <rect x={cx + w*0.2 - s*0.4} y={cy + h*0.15 - s*0.8} width={s*0.8} height={s*0.3}
                fill="none" stroke="#000" strokeWidth={0.1} rx={0.05} />
            {/* Wash basin */}
            <ellipse cx={cx - w*0.2} cy={cy - h*0.2} rx={s*0.45} ry={s*0.35}
                fill="none" stroke="#000" strokeWidth={0.12} />
        </g>
    )
}

function furnitureDining(cx, cy, w, h) {
    const ts = Math.min(w * 0.4, h * 0.5)
    return (
        <g>
            <rect x={cx - ts/2} y={cy - ts*0.35} width={ts} height={ts*0.7}
                fill="none" stroke="#000" strokeWidth={0.12} />
            {[-1, 1].map(dx => [-1, 1].map(dy => (
                <rect key={`${dx}${dy}`}
                    x={cx + dx*(ts*0.6) - ts*0.06}
                    y={cy + dy*(ts*0.2) - ts*0.06}
                    width={ts*0.12} height={ts*0.12}
                    fill="none" stroke="#000" strokeWidth={0.08} />
            )))}
        </g>
    )
}

function furnitureStudy(cx, cy, w, h) {
    const ds = Math.min(w * 0.5, h * 0.5)
    return (
        <g>
            <rect x={cx - ds/2} y={cy - ds*0.2} width={ds} height={ds*0.35}
                fill="none" stroke="#000" strokeWidth={0.12} />
            <rect x={cx - ds*0.08} y={cy + ds*0.25} width={ds*0.16} height={ds*0.16}
                fill="none" stroke="#000" strokeWidth={0.08} />
        </g>
    )
}

function furniturePooja(cx, cy, w, h) {
    const ps = Math.min(w, h) * 0.3
    return (
        <g>
            <rect x={cx - ps/2} y={cy - ps*0.4} width={ps} height={ps*0.8}
                fill="none" stroke="#000" strokeWidth={0.12} />
            <circle cx={cx} cy={cy} r={ps*0.1}
                fill="none" stroke="#000" strokeWidth={0.08} />
        </g>
    )
}

function furnitureStore(cx, cy, w, h) {
    const ss = Math.min(w, h) * 0.35
    return (
        <g>
            <rect x={cx - ss/2} y={cy - ss*0.4} width={ss} height={ss*0.8}
                fill="none" stroke="#000" strokeWidth={0.12} />
            <line x1={cx - ss/2} y1={cy} x2={cx + ss/2} y2={cy}
                stroke="#000" strokeWidth={0.08} />
            <line x1={cx - ss/2} y1={cy - ss*0.2} x2={cx + ss/2} y2={cy - ss*0.2}
                stroke="#000" strokeWidth={0.08} />
        </g>
    )
}

function furnitureStaircase(cx, cy, w, h) {
    const steps = Math.max(6, Math.floor(h / 0.8))
    const stepH = h * 0.85 / steps
    const sx = cx - w * 0.4
    const sy = cy - h * 0.42
    const sw = w * 0.8
    const lines = []
    for (let i = 0; i <= steps; i++) {
        lines.push(
            <line key={i} x1={sx} y1={sy + i * stepH}
                x2={sx + sw} y2={sy + i * stepH}
                stroke="#000" strokeWidth={0.1} />
        )
    }
    return (
        <g>
            <rect x={sx} y={sy} width={sw} height={steps * stepH}
                fill="none" stroke="#000" strokeWidth={0.15} />
            {lines}
            {/* UP arrow */}
            <line x1={cx} y1={sy + steps*stepH - stepH} x2={cx} y2={sy + stepH}
                stroke="#000" strokeWidth={0.12} />
            <polygon points={`${cx},${sy + stepH} ${cx-0.4},${sy + stepH*2} ${cx+0.4},${sy + stepH*2}`}
                fill="#000" />
            <text x={cx + w*0.05} y={cy + h*0.1}
                fontSize={Math.min(w, h) * 0.12} fill="#000"
                fontFamily="'Times New Roman', serif"
                transform={`rotate(-90, ${cx + w*0.05}, ${cy + h*0.1})`}
                textAnchor="middle">
                UP
            </text>
        </g>
    )
}

const FURNITURE_MAP = {
    master_bedroom: furnitureBed,
    bedroom: furnitureBed,
    living: furnitureSofa,
    kitchen: furnitureKitchen,
    bathroom: furnitureBathroom,
    toilet: furnitureBathroom,
    dining: furnitureDining,
    study: furnitureStudy,
    pooja: furniturePooja,
    store: furnitureStore,
    staircase: furnitureStaircase,
}

/* ──────────────── Helper functions ──────────────── */
function toPathD(coords) {
    if (!coords || coords.length < 2) return ''
    return coords.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ') + ' Z'
}

function getRoomDims(polygon) {
    if (!polygon || polygon.length < 3) return null
    const rxs = polygon.map(p => p[0])
    const rys = polygon.map(p => p[1])
    return {
        w: Math.max(...rxs) - Math.min(...rxs),
        h: Math.max(...rys) - Math.min(...rys),
        x1: Math.min(...rxs), y1: Math.min(...rys),
        x2: Math.max(...rxs), y2: Math.max(...rys),
    }
}

function getRoomLabel(room) {
    return (ROOM_LABELS[room.room_type] || room.room_type || 'ROOM').toUpperCase()
}

/* ──────────────── Main Component ──────────────── */

export default function PlanPreview({ plan }) {
    const layout = useMemo(() => {
        if (!plan || !plan.boundary || plan.boundary.length < 3) return null

        const allPoints = [
            ...plan.boundary,
            ...plan.rooms.flatMap(r => r.polygon || []),
        ]
        const xs = allPoints.map(p => p[0])
        const ys = allPoints.map(p => p[1])
        const minX = Math.min(...xs)
        const minY = Math.min(...ys)
        const maxX = Math.max(...xs)
        const maxY = Math.max(...ys)
        const plotW = maxX - minX
        const plotH = maxY - minY
        const scale = Math.max(plotW, plotH)
        const pad = scale * 0.22

        const viewBox = `${minX - pad} ${minY - pad * 1.2} ${plotW + pad * 2} ${plotH + pad * 2.8}`

        return { viewBox, minX, minY, maxX, maxY, plotW, plotH, scale, pad }
    }, [plan])

    if (!layout) return (
        <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            height: '100%', color: '#666', fontFamily: 'serif',
        }}>
            <p>No plan data</p>
        </div>
    )

    const { viewBox, minX, minY, maxX, maxY, plotW, plotH, scale } = layout

    // Architectural constants
    const EXT_WALL = 0.75          // 9" external wall thickness in ft
    const INT_WALL = 0.375         // 4.5" internal wall thickness in ft
    const LINE_W = 0.08            // thin line width
    const COL_SIZE = 0.4           // column/pillar size

    const totalArea = plan.total_area || plan.area_summary?.plot_area ||
        plan.rooms.reduce((sum, r) => sum + (r.actual_area || r.area || 0), 0)

    // Collect all unique x and y coordinates from rooms for dimension chains
    const roomXCoords = useMemo(() => {
        const xs = new Set([minX, maxX])
        plan.rooms.forEach(r => {
            const d = getRoomDims(r.polygon)
            if (d) { xs.add(Math.round(d.x1 * 10) / 10); xs.add(Math.round(d.x2 * 10) / 10) }
        })
        return [...xs].sort((a, b) => a - b)
    }, [plan, minX, maxX])

    const roomYCoords = useMemo(() => {
        const ys = new Set([minY, maxY])
        plan.rooms.forEach(r => {
            const d = getRoomDims(r.polygon)
            if (d) { ys.add(Math.round(d.y1 * 10) / 10); ys.add(Math.round(d.y2 * 10) / 10) }
        })
        return [...ys].sort((a, b) => a - b)
    }, [plan, minY, maxY])

    // Collect column positions (boundary corners + room corners on boundary)
    const columns = useMemo(() => {
        if (plan.columns) return plan.columns
        const cols = new Map()
        plan.boundary.slice(0, -1).forEach(p => {
            cols.set(`${p[0].toFixed(1)},${p[1].toFixed(1)}`, p)
        })
        plan.rooms.forEach(r => {
            const d = getRoomDims(r.polygon)
            if (!d) return
            const corners = [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]]
            corners.forEach(([cx, cy]) => {
                const onBnd = Math.abs(cx - minX) < 1.5 || Math.abs(cx - maxX) < 1.5 ||
                    Math.abs(cy - minY) < 1.5 || Math.abs(cy - maxY) < 1.5
                if (onBnd) {
                    const key = `${cx.toFixed(1)},${cy.toFixed(1)}`
                    if (!cols.has(key)) cols.set(key, [cx, cy])
                }
            })
        })
        return [...cols.values()]
    }, [plan, minX, minY, maxX, maxY])

    return (
        <div className="plan-svg" style={{
            padding: '1rem',
            background: '#fff',
        }}>
            <svg viewBox={viewBox} xmlns="http://www.w3.org/2000/svg"
                style={{ background: '#fff', width: '100%' }}>

                {/* ══════ DEFS ══════ */}
                <defs>
                    {plan.rooms.map((room, i) => {
                        if (!['bathroom', 'toilet'].includes(room.room_type)) return null
                        if (!room.polygon || room.polygon.length < 3) return null
                        return (
                            <clipPath key={`cp-${i}`} id={`clip-wet-${i}`}>
                                <path d={toPathD(room.polygon)} />
                            </clipPath>
                        )
                    })}
                </defs>

                {/* ══════ WHITE BACKGROUND ══════ */}
                <rect x={minX - layout.pad} y={minY - layout.pad * 1.5}
                    width={plotW + layout.pad * 2} height={plotH + layout.pad * 3}
                    fill="#fff" />

                {/* ══════ ROOM OUTLINES — clean stroke rectangles ══════ */}
                {plan.rooms.map((room, i) => {
                    const d = getRoomDims(room.polygon)
                    if (!d) return null
                    return (
                        <rect key={`room-${i}`}
                            x={d.x1} y={d.y1} width={d.w} height={d.h}
                            fill="#fff" stroke="#222" strokeWidth={INT_WALL}
                            strokeLinejoin="miter" />
                    )
                })}

                {/* ══════ EXTERIOR BOUNDARY — thicker outline on top ══════ */}
                <rect x={minX} y={minY} width={plotW} height={plotH}
                    fill="none" stroke="#000" strokeWidth={EXT_WALL}
                    strokeLinejoin="miter" />

                {/* ══════ Cross-hatching for wet areas ══════ */}
                {plan.rooms.map((room, i) => {
                    if (!['bathroom', 'toilet'].includes(room.room_type)) return null
                    const d = getRoomDims(room.polygon)
                    if (!d) return null
                    const spacing = 0.6
                    const diag = d.w + d.h
                    const lines = []
                    for (let dd = 0; dd < diag; dd += spacing) {
                        lines.push(
                            <line key={dd}
                                x1={d.x1 + dd} y1={d.y1}
                                x2={d.x1} y2={d.y1 + dd}
                                stroke="#000" strokeWidth={0.04} />
                        )
                    }
                    return (
                        <g key={`hatch-${i}`} clipPath={`url(#clip-wet-${i})`} opacity="0.35">
                            {lines}
                        </g>
                    )
                })}

                {/* ══════ Staircase — step lines if room type ══════ */}
                {plan.rooms.map((room, i) => {
                    if (room.room_type !== 'staircase') return null
                    const d = getRoomDims(room.polygon)
                    if (!d) return null
                    return (
                        <g key={`stair-${i}`}>
                            {furnitureStaircase(
                                d.x1 + d.w / 2, d.y1 + d.h / 2, d.w, d.h
                            )}
                        </g>
                    )
                })}

                {/* ══════ Furniture Icons ══════ */}
                {plan.rooms.map((room, i) => {
                    if (room.room_type === 'staircase') return null
                    const fn = FURNITURE_MAP[room.room_type]
                    if (!fn) return null
                    const d = getRoomDims(room.polygon)
                    if (!d) return null
                    return (
                        <g key={`furn-${i}`} opacity="0.7">
                            {fn(d.x1 + d.w / 2, d.y1 + d.h / 2, d.w, d.h)}
                        </g>
                    )
                })}

                {/* ══════ DOOR OPENINGS ══════ */}
                {plan.doors?.map((door, i) => {
                    if (!door.hinge) {
                        const px = door.position?.[0] || 0, py = door.position?.[1] || 0
                        return (
                            <g key={`door-${i}`}>
                                <line x1={px - 1.2} y1={py} x2={px + 1.2} y2={py}
                                    stroke="#fff" strokeWidth={INT_WALL * 3} />
                            </g>
                        )
                    }
                    const dw = door.width
                    const hx = door.hinge[0], hy = door.hinge[1]
                    const sx = door.swing_dir[0], sy = door.swing_dir[1]
                    const dex = door.door_end[0], dey = door.door_end[1]

                    const wallDx = dex - hx, wallDy = dey - hy
                    const arcEndX = hx + sx * dw
                    const arcEndY = hy + sy * dw
                    const cross = wallDx * sy - wallDy * sx
                    const sweepFlag = cross > 0 ? 0 : 1

                    // Determine if on external or internal wall
                    const onExt = Math.abs(hy - minY) < 1.2 || Math.abs(hy - maxY) < 1.2 ||
                        Math.abs(hx - minX) < 1.2 || Math.abs(hx - maxX) < 1.2 ||
                        Math.abs(dey - minY) < 1.2 || Math.abs(dey - maxY) < 1.2 ||
                        Math.abs(dex - minX) < 1.2 || Math.abs(dex - maxX) < 1.2
                    const clearW = onExt ? EXT_WALL * 1.6 : INT_WALL * 1.8

                    return (
                        <g key={`door-${i}`}>
                            {/* Clear wall gap */}
                            <line x1={hx} y1={hy} x2={dex} y2={dey}
                                stroke="#fff" strokeWidth={clearW} />
                            {/* Door leaf line */}
                            <line x1={hx} y1={hy} x2={arcEndX} y2={arcEndY}
                                stroke="#000" strokeWidth={0.08} />
                            {/* Quarter-circle swing arc */}
                            <path d={`M ${arcEndX},${arcEndY} A ${dw},${dw} 0 0 ${sweepFlag} ${dex},${dey}`}
                                fill="none" stroke="#000" strokeWidth={0.06} />
                            {/* Hinge dot */}
                            <circle cx={hx} cy={hy} r={0.12} fill="#000" />
                        </g>
                    )
                })}

                {/* ══════ WINDOW SYMBOLS ══════ */}
                {plan.windows?.map((win, i) => {
                    if (!win.start || !win.end) return null
                    const wsx = win.start[0], wsy = win.start[1]
                    const wex = win.end[0], wey = win.end[1]
                    const wdx = wex - wsx, wdy = wey - wsy
                    const wlen = Math.sqrt(wdx * wdx + wdy * wdy)
                    if (wlen < 0.5) return null
                    const wnx = -wdy / wlen, wny = wdx / wlen
                    const gap = EXT_WALL * 0.45

                    return (
                        <g key={`win-${i}`}>
                            {/* Clear wall */}
                            <line x1={wsx} y1={wsy} x2={wex} y2={wey}
                                stroke="#fff" strokeWidth={EXT_WALL * 2} />
                            {/* Double glass lines */}
                            <line x1={wsx + wnx*gap} y1={wsy + wny*gap}
                                x2={wex + wnx*gap} y2={wey + wny*gap}
                                stroke="#000" strokeWidth={0.06} />
                            <line x1={wsx - wnx*gap} y1={wsy - wny*gap}
                                x2={wex - wnx*gap} y2={wey - wny*gap}
                                stroke="#000" strokeWidth={0.06} />
                            {/* Center glass line */}
                            <line x1={wsx} y1={wsy} x2={wex} y2={wey}
                                stroke="#000" strokeWidth={0.03} />
                            {/* Frame ends */}
                            <line x1={wsx + wnx*gap} y1={wsy + wny*gap}
                                x2={wsx - wnx*gap} y2={wsy - wny*gap}
                                stroke="#000" strokeWidth={0.06} />
                            <line x1={wex + wnx*gap} y1={wey + wny*gap}
                                x2={wex - wnx*gap} y2={wey - wny*gap}
                                stroke="#000" strokeWidth={0.06} />
                        </g>
                    )
                })}

                {/* ══════ COLUMN / PILLAR MARKS ══════ */}
                {columns.map((pt, i) => (
                    <rect key={`col-${i}`}
                        x={pt[0] - COL_SIZE/2} y={pt[1] - COL_SIZE/2}
                        width={COL_SIZE} height={COL_SIZE}
                        fill="#000" stroke="#000" strokeWidth={0.04} />
                ))}

                {/* ══════ ROOM LABELS ══════ */}
                {plan.rooms.map((room, i) => {
                    const d = getRoomDims(room.polygon)
                    if (!d) return null
                    const cx = d.x1 + d.w / 2
                    const cy = d.y1 + d.h / 2
                    const label = getRoomLabel(room)
                    const dimText = fmtDimPair(d.w, d.h)
                    const nameSize = Math.min(d.w, d.h) * 0.13
                    const dimSize = nameSize * 0.75

                    return (
                        <g key={`label-${i}`}>
                            <text x={cx} y={cy - nameSize * 0.3}
                                textAnchor="middle" fill="#000"
                                fontSize={Math.max(nameSize, 0.8)} fontWeight="bold"
                                fontFamily="'Times New Roman', serif">
                                {label}
                            </text>
                            <text x={cx} y={cy + nameSize * 1.0}
                                textAnchor="middle" fill="#000"
                                fontSize={Math.max(dimSize, 0.6)}
                                fontFamily="'Times New Roman', serif">
                                {dimText}
                            </text>
                        </g>
                    )
                })}

                {/* ══════ DIMENSION CHAINS — Top (X-axis) ══════ */}
                {(() => {
                    const yBase = minY
                    const dimY1 = yBase - 2.5  // inner chain
                    const dimY2 = yBase - 4.5  // outer overall

                    const elements = []

                    // Inner chain — per-room segments
                    for (let k = 0; k < roomXCoords.length - 1; k++) {
                        const x1 = roomXCoords[k], x2 = roomXCoords[k + 1]
                        const seg = x2 - x1
                        if (seg < 0.5) continue
                        const mx = (x1 + x2) / 2

                        elements.push(
                            <g key={`dimx-${k}`}>
                                {/* Extension lines */}
                                <line x1={x1} y1={yBase} x2={x1} y2={dimY1 - 0.3}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={x2} y1={yBase} x2={x2} y2={dimY1 - 0.3}
                                    stroke="#000" strokeWidth={0.04} />
                                {/* Dimension line with arrows */}
                                <line x1={x1} y1={dimY1} x2={x2} y2={dimY1}
                                    stroke="#000" strokeWidth={0.04} />
                                {/* Ticks */}
                                <line x1={x1} y1={dimY1 - 0.3} x2={x1} y2={dimY1 + 0.3}
                                    stroke="#000" strokeWidth={0.06} />
                                <line x1={x2} y1={dimY1 - 0.3} x2={x2} y2={dimY1 + 0.3}
                                    stroke="#000" strokeWidth={0.06} />
                                {/* Label */}
                                <rect x={mx - 1.5} y={dimY1 - 0.55} width={3} height={0.8}
                                    fill="#fff" />
                                <text x={mx} y={dimY1 + 0.15}
                                    textAnchor="middle" fill="#000"
                                    fontSize={0.7} fontFamily="'Times New Roman', serif">
                                    {fmtDim(seg)}
                                </text>
                            </g>
                        )
                    }

                    // Outer overall dimension
                    const totalLen = maxX - minX
                    elements.push(
                        <g key="dimx-total">
                            <line x1={minX} y1={yBase} x2={minX} y2={dimY2 - 0.3}
                                stroke="#000" strokeWidth={0.04} />
                            <line x1={maxX} y1={yBase} x2={maxX} y2={dimY2 - 0.3}
                                stroke="#000" strokeWidth={0.04} />
                            <line x1={minX} y1={dimY2} x2={maxX} y2={dimY2}
                                stroke="#000" strokeWidth={0.05} />
                            <line x1={minX} y1={dimY2 - 0.3} x2={minX} y2={dimY2 + 0.3}
                                stroke="#000" strokeWidth={0.06} />
                            <line x1={maxX} y1={dimY2 - 0.3} x2={maxX} y2={dimY2 + 0.3}
                                stroke="#000" strokeWidth={0.06} />
                            <rect x={(minX + maxX)/2 - 1.5} y={dimY2 - 0.55}
                                width={3} height={0.8} fill="#fff" />
                            <text x={(minX + maxX) / 2} y={dimY2 + 0.15}
                                textAnchor="middle" fill="#000"
                                fontSize={0.75} fontWeight="bold"
                                fontFamily="'Times New Roman', serif">
                                {fmtDim(totalLen)}
                            </text>
                        </g>
                    )

                    return elements
                })()}

                {/* ══════ DIMENSION CHAINS — Right side (Y-axis) ══════ */}
                {(() => {
                    const xBase = maxX
                    const dimX1 = xBase + 2.5
                    const dimX2 = xBase + 4.5

                    const elements = []

                    for (let k = 0; k < roomYCoords.length - 1; k++) {
                        const y1 = roomYCoords[k], y2 = roomYCoords[k + 1]
                        const seg = y2 - y1
                        if (seg < 0.5) continue
                        const my = (y1 + y2) / 2

                        elements.push(
                            <g key={`dimy-${k}`}>
                                <line x1={xBase} y1={y1} x2={dimX1 + 0.3} y2={y1}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={xBase} y1={y2} x2={dimX1 + 0.3} y2={y2}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={dimX1} y1={y1} x2={dimX1} y2={y2}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={dimX1 - 0.3} y1={y1} x2={dimX1 + 0.3} y2={y1}
                                    stroke="#000" strokeWidth={0.06} />
                                <line x1={dimX1 - 0.3} y1={y2} x2={dimX1 + 0.3} y2={y2}
                                    stroke="#000" strokeWidth={0.06} />
                                <rect x={dimX1 - 0.4} y={my - 1.2} width={1.1} height={2.4}
                                    fill="#fff" />
                                <text x={dimX1 + 0.15} y={my}
                                    textAnchor="middle" fill="#000"
                                    fontSize={0.7}
                                    fontFamily="'Times New Roman', serif"
                                    transform={`rotate(90, ${dimX1 + 0.15}, ${my})`}>
                                    {fmtDim(seg)}
                                </text>
                            </g>
                        )
                    }

                    // Overall
                    const totalH = maxY - minY
                    elements.push(
                        <g key="dimy-total">
                            <line x1={xBase} y1={minY} x2={dimX2 + 0.3} y2={minY}
                                stroke="#000" strokeWidth={0.04} />
                            <line x1={xBase} y1={maxY} x2={dimX2 + 0.3} y2={maxY}
                                stroke="#000" strokeWidth={0.04} />
                            <line x1={dimX2} y1={minY} x2={dimX2} y2={maxY}
                                stroke="#000" strokeWidth={0.05} />
                            <line x1={dimX2 - 0.3} y1={minY} x2={dimX2 + 0.3} y2={minY}
                                stroke="#000" strokeWidth={0.06} />
                            <line x1={dimX2 - 0.3} y1={maxY} x2={dimX2 + 0.3} y2={maxY}
                                stroke="#000" strokeWidth={0.06} />
                            <rect x={dimX2 - 0.4} y={(minY+maxY)/2 - 1.2}
                                width={1.1} height={2.4} fill="#fff" />
                            <text x={dimX2 + 0.15} y={(minY + maxY) / 2}
                                textAnchor="middle" fill="#000"
                                fontSize={0.75} fontWeight="bold"
                                fontFamily="'Times New Roman', serif"
                                transform={`rotate(90, ${dimX2 + 0.15}, ${(minY + maxY) / 2})`}>
                                {fmtDim(totalH)}
                            </text>
                        </g>
                    )

                    return elements
                })()}

                {/* ══════ DIMENSION CHAINS — Bottom (X repeat) ══════ */}
                {(() => {
                    const yBase = maxY
                    const dimY1 = yBase + 2.5
                    const elements = []

                    for (let k = 0; k < roomXCoords.length - 1; k++) {
                        const x1 = roomXCoords[k], x2 = roomXCoords[k + 1]
                        const seg = x2 - x1
                        if (seg < 0.5) continue
                        const mx = (x1 + x2) / 2
                        elements.push(
                            <g key={`dimxb-${k}`}>
                                <line x1={x1} y1={yBase} x2={x1} y2={dimY1 + 0.3}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={x2} y1={yBase} x2={x2} y2={dimY1 + 0.3}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={x1} y1={dimY1} x2={x2} y2={dimY1}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={x1} y1={dimY1 - 0.3} x2={x1} y2={dimY1 + 0.3}
                                    stroke="#000" strokeWidth={0.06} />
                                <line x1={x2} y1={dimY1 - 0.3} x2={x2} y2={dimY1 + 0.3}
                                    stroke="#000" strokeWidth={0.06} />
                                <rect x={mx - 1.5} y={dimY1 - 0.55} width={3} height={0.8}
                                    fill="#fff" />
                                <text x={mx} y={dimY1 + 0.15}
                                    textAnchor="middle" fill="#000"
                                    fontSize={0.7} fontFamily="'Times New Roman', serif">
                                    {fmtDim(seg)}
                                </text>
                            </g>
                        )
                    }
                    return elements
                })()}

                {/* ══════ DIMENSION CHAINS — Left side (Y repeat) ══════ */}
                {(() => {
                    const xBase = minX
                    const dimX1 = xBase - 2.5
                    const elements = []

                    for (let k = 0; k < roomYCoords.length - 1; k++) {
                        const y1 = roomYCoords[k], y2 = roomYCoords[k + 1]
                        const seg = y2 - y1
                        if (seg < 0.5) continue
                        const my = (y1 + y2) / 2
                        elements.push(
                            <g key={`dimyl-${k}`}>
                                <line x1={xBase} y1={y1} x2={dimX1 - 0.3} y2={y1}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={xBase} y1={y2} x2={dimX1 - 0.3} y2={y2}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={dimX1} y1={y1} x2={dimX1} y2={y2}
                                    stroke="#000" strokeWidth={0.04} />
                                <line x1={dimX1 - 0.3} y1={y1} x2={dimX1 + 0.3} y2={y1}
                                    stroke="#000" strokeWidth={0.06} />
                                <line x1={dimX1 - 0.3} y1={y2} x2={dimX1 + 0.3} y2={y2}
                                    stroke="#000" strokeWidth={0.06} />
                                <rect x={dimX1 - 0.55} y={my - 1.2} width={1.1} height={2.4}
                                    fill="#fff" />
                                <text x={dimX1} y={my}
                                    textAnchor="middle" fill="#000"
                                    fontSize={0.7}
                                    fontFamily="'Times New Roman', serif"
                                    transform={`rotate(-90, ${dimX1}, ${my})`}>
                                    {fmtDim(seg)}
                                </text>
                            </g>
                        )
                    }
                    return elements
                })()}

                {/* ══════ ROAD label — Bottom ══════ */}
                <g>
                    <line x1={minX - 1} y1={maxY + layout.pad * 0.45}
                        x2={maxX + 1} y2={maxY + layout.pad * 0.45}
                        stroke="#000" strokeWidth={0.08} strokeDasharray="1 0.5" />
                    <text x={(minX + maxX) / 2} y={maxY + layout.pad * 0.62}
                        textAnchor="middle" fill="#000"
                        fontSize={1.4} fontWeight="bold"
                        fontFamily="'Times New Roman', serif"
                        letterSpacing={0.8}>
                        ROAD
                    </text>
                </g>

                {/* ══════ Title block (centered bottom) ══════ */}
                <g>
                    <text x={(minX + maxX) / 2} y={maxY + layout.pad * 1.3}
                        fill="#000" fontSize={2.0} fontWeight="bold"
                        fontFamily="'Times New Roman', serif"
                        textAnchor="middle"
                        letterSpacing={1.5}>
                        GROUND FLOOR PLAN
                    </text>
                    <text x={(minX + maxX) / 2} y={maxY + layout.pad * 1.6}
                        fill="#333" fontSize={0.9}
                        fontFamily="'Times New Roman', serif"
                        textAnchor="middle">
                        Plot: {fmtDim(plan.plot?.width || plotW)} x {fmtDim(plan.plot?.length || plotH)}
                        {' | '}{totalArea > 0 ? `${totalArea.toFixed(0)} sq.ft` : ''}
                        {' | '}{plan.rooms.length} rooms
                    </text>
                </g>

                {/* ══════ North Arrow (top-right corner) ══════ */}
                {(() => {
                    const nx = maxX + layout.pad * 0.3
                    const ny = minY - layout.pad * 0.5
                    const arrH = 1.6
                    const arrW = 0.5
                    return (
                        <g>
                            <polygon points={`${nx},${ny - arrH} ${nx - arrW},${ny} ${nx + arrW},${ny}`}
                                fill="#000" />
                            <line x1={nx} y1={ny} x2={nx} y2={ny + arrH * 0.4}
                                stroke="#000" strokeWidth={0.06} />
                            <text x={nx} y={ny - arrH - 0.3}
                                textAnchor="middle" fill="#000"
                                fontSize={0.9} fontWeight="bold"
                                fontFamily="'Times New Roman', serif">
                                N
                            </text>
                        </g>
                    )
                })()}

                {/* ══════ Drawing border frame ══════ */}
                <rect
                    x={minX - layout.pad * 0.85}
                    y={minY - layout.pad * 1.05}
                    width={plotW + layout.pad * 1.7}
                    height={plotH + layout.pad * 2.5}
                    fill="none" stroke="#000" strokeWidth={0.12} />
            </svg>
        </div>
    )
}
