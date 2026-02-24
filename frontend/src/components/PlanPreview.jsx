import { useMemo } from 'react'

/* ──────────────────────────────────────────────
   Zone-based color palette (soft pastels)
   ────────────────────────────────────────────── */
const ZONE_COLORS = {
    public:       { fill: '#DBEAFE', stroke: '#3B82F6', text: '#1E40AF' },
    semi_private: { fill: '#D1FAE5', stroke: '#10B981', text: '#065F46' },
    private:      { fill: '#EDE9FE', stroke: '#8B5CF6', text: '#5B21B6' },
    service:      { fill: '#FEF3C7', stroke: '#F59E0B', text: '#92400E' },
    utility:      { fill: '#F3F4F6', stroke: '#6B7280', text: '#374151' },
}

const ROOM_TYPE_COLORS = {
    living:         ZONE_COLORS.public,
    dining:         { fill: '#CFFAFE', stroke: '#06B6D4', text: '#155E75' },
    kitchen:        { fill: '#FFEDD5', stroke: '#F97316', text: '#9A3412' },
    master_bedroom: { fill: '#EDE9FE', stroke: '#8B5CF6', text: '#5B21B6' },
    bedroom:        { fill: '#E8DEF8', stroke: '#7C3AED', text: '#4C1D95' },
    bathroom:       { fill: '#CCFBF1', stroke: '#14B8A6', text: '#134E4A' },
    toilet:         { fill: '#CCFBF1', stroke: '#14B8A6', text: '#134E4A' },
    study:          { fill: '#FEF9C3', stroke: '#EAB308', text: '#854D0E' },
    pooja:          { fill: '#FCE7F3', stroke: '#EC4899', text: '#9D174D' },
    store:          { fill: '#E7E5E4', stroke: '#78716C', text: '#44403C' },
    balcony:        { fill: '#DCFCE7', stroke: '#22C55E', text: '#166534' },
    garage:         { fill: '#E5E7EB', stroke: '#6B7280', text: '#374151' },
    hallway:        { fill: '#F9FAFB', stroke: '#9CA3AF', text: '#4B5563' },
    utility:        { fill: '#F3F4F6', stroke: '#6B7280', text: '#374151' },
    foyer:          { fill: '#FEF3C7', stroke: '#F59E0B', text: '#92400E' },
    passage:        { fill: '#F9FAFB', stroke: '#9CA3AF', text: '#4B5563' },
}

function getRoomColor(room) {
    return ROOM_TYPE_COLORS[room.room_type] ||
           ZONE_COLORS[room.zone] ||
           { fill: '#F9FAFB', stroke: '#9CA3AF', text: '#4B5563' }
}

/* ──────────────────────────────────────────────
   Furniture SVG helpers  (all return JSX groups)
   Drawn relative to room centroid and scaled
   ────────────────────────────────────────────── */
function furnitureBed(cx, cy, w, h, scale) {
    const bw = Math.min(w * 0.5, h * 0.65) * scale
    const bh = bw * 0.6
    return (
        <g opacity="0.45">
            <rect x={cx - bw / 2} y={cy - bh / 2} width={bw} height={bh}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} rx={scale * 0.2} />
            <rect x={cx - bw / 2 + bw * 0.08} y={cy - bh / 2 + bh * 0.08}
                width={bw * 0.35} height={bh * 0.25}
                fill="none" stroke="#777" strokeWidth={scale * 0.2} rx={scale * 0.15} />
            <rect x={cx + bw / 2 - bw * 0.43} y={cy - bh / 2 + bh * 0.08}
                width={bw * 0.35} height={bh * 0.25}
                fill="none" stroke="#777" strokeWidth={scale * 0.2} rx={scale * 0.15} />
        </g>
    )
}

function furnitureSofa(cx, cy, w, h, scale) {
    const sw = Math.min(w * 0.45, h * 0.55) * scale
    const sh = sw * 0.4
    return (
        <g opacity="0.45">
            <rect x={cx - sw / 2} y={cy - sh / 2} width={sw} height={sh}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} rx={scale * 0.4} />
            <rect x={cx - sw / 2} y={cy - sh / 2} width={sw} height={sh * 0.3}
                fill="none" stroke="#777" strokeWidth={scale * 0.25} rx={scale * 0.3} />
            <ellipse cx={cx} cy={cy + sh * 0.7} rx={sw * 0.2} ry={sw * 0.1}
                fill="none" stroke="#999" strokeWidth={scale * 0.2} />
        </g>
    )
}

function furnitureKitchen(cx, cy, w, h, scale) {
    const kw = Math.min(w * 0.4, h * 0.6) * scale
    const kh = kw * 0.35
    return (
        <g opacity="0.45">
            <rect x={cx - kw / 2} y={cy - kh / 2} width={kw} height={kh}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} />
            <circle cx={cx - kw * 0.15} cy={cy} r={kw * 0.07}
                fill="none" stroke="#777" strokeWidth={scale * 0.2} />
            <circle cx={cx + kw * 0.12} cy={cy - kh * 0.12} r={kw * 0.05}
                fill="none" stroke="#777" strokeWidth={scale * 0.2} />
            <circle cx={cx + kw * 0.27} cy={cy - kh * 0.12} r={kw * 0.05}
                fill="none" stroke="#777" strokeWidth={scale * 0.2} />
        </g>
    )
}

function furnitureBathroom(cx, cy, w, h, scale) {
    const bs = Math.min(w, h) * 0.3 * scale
    return (
        <g opacity="0.45">
            <ellipse cx={cx} cy={cy + bs * 0.25} rx={bs * 0.25} ry={bs * 0.35}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} />
            <rect x={cx - bs * 0.2} y={cy - bs * 0.35} width={bs * 0.4} height={bs * 0.3}
                fill="none" stroke="#777" strokeWidth={scale * 0.2} rx={scale * 0.2} />
        </g>
    )
}

function furnitureDining(cx, cy, w, h, scale) {
    const ts = Math.min(w * 0.35, h * 0.45) * scale
    return (
        <g opacity="0.45">
            <rect x={cx - ts / 2} y={cy - ts * 0.35} width={ts} height={ts * 0.7}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} rx={scale * 0.2} />
            {[-1, 1].map(dx => [-1, 1].map(dy => (
                <rect key={`${dx}${dy}`}
                    x={cx + dx * (ts * 0.6) - ts * 0.08}
                    y={cy + dy * (ts * 0.15) - ts * 0.08}
                    width={ts * 0.16} height={ts * 0.16}
                    fill="none" stroke="#999" strokeWidth={scale * 0.15} rx={scale * 0.1} />
            )))}
        </g>
    )
}

function furnitureStudy(cx, cy, w, h, scale) {
    const ds = Math.min(w * 0.4, h * 0.45) * scale
    return (
        <g opacity="0.45">
            <rect x={cx - ds / 2} y={cy - ds * 0.2} width={ds} height={ds * 0.4}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} />
            <circle cx={cx} cy={cy + ds * 0.4} r={ds * 0.12}
                fill="none" stroke="#999" strokeWidth={scale * 0.2} />
        </g>
    )
}

function furniturePooja(cx, cy, w, h, scale) {
    const ps = Math.min(w, h) * 0.25 * scale
    return (
        <g opacity="0.45">
            <rect x={cx - ps / 2} y={cy - ps * 0.4} width={ps} height={ps * 0.8}
                fill="none" stroke="#EC4899" strokeWidth={scale * 0.3} rx={scale * 0.1} />
            <circle cx={cx} cy={cy - ps * 0.15} r={ps * 0.12}
                fill="none" stroke="#EC4899" strokeWidth={scale * 0.2} />
        </g>
    )
}

function furnitureStore(cx, cy, w, h, scale) {
    const ss = Math.min(w, h) * 0.3 * scale
    return (
        <g opacity="0.45">
            <rect x={cx - ss / 2} y={cy - ss * 0.35} width={ss} height={ss * 0.7}
                fill="none" stroke="#78716C" strokeWidth={scale * 0.3} />
            <line x1={cx - ss / 2} y1={cy} x2={cx + ss / 2} y2={cy}
                stroke="#78716C" strokeWidth={scale * 0.2} />
        </g>
    )
}

function furnitureBalcony(cx, cy, w, h, scale) {
    const bs = Math.min(w, h) * 0.25 * scale
    return (
        <g opacity="0.4">
            <circle cx={cx - bs * 0.3} cy={cy} r={bs * 0.2}
                fill="none" stroke="#22C55E" strokeWidth={scale * 0.25} />
            <circle cx={cx + bs * 0.3} cy={cy} r={bs * 0.15}
                fill="none" stroke="#22C55E" strokeWidth={scale * 0.25} />
        </g>
    )
}

function furnitureGarage(cx, cy, w, h, scale) {
    const gs = Math.min(w * 0.4, h * 0.5) * scale
    return (
        <g opacity="0.4">
            <rect x={cx - gs / 2} y={cy - gs * 0.3} width={gs} height={gs * 0.6}
                fill="none" stroke="#6B7280" strokeWidth={scale * 0.3} rx={scale * 0.3} />
            <circle cx={cx - gs * 0.2} cy={cy + gs * 0.25} r={gs * 0.08}
                fill="none" stroke="#6B7280" strokeWidth={scale * 0.2} />
            <circle cx={cx + gs * 0.2} cy={cy + gs * 0.25} r={gs * 0.08}
                fill="none" stroke="#6B7280" strokeWidth={scale * 0.2} />
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
    balcony: furnitureBalcony,
    garage: furnitureGarage,
}

/* ──────────────── Compass Rose ──────────────── */
function CompassRose({ x, y, size }) {
    const arm = size * 0.4
    const triW = size * 0.1
    return (
        <g>
            <circle cx={x} cy={y} r={size * 0.5} fill="white" stroke="#CBD5E1" strokeWidth={size * 0.015} />
            <polygon points={`${x},${y - arm} ${x - triW},${y} ${x + triW},${y}`} fill="#1E293B" />
            <polygon points={`${x},${y + arm} ${x - triW},${y} ${x + triW},${y}`} fill="none" stroke="#94A3B8" strokeWidth={size * 0.01} />
            <line x1={x - arm * 0.6} y1={y} x2={x + arm * 0.6} y2={y} stroke="#94A3B8" strokeWidth={size * 0.012} />
            <text x={x} y={y - arm - size * 0.08} textAnchor="middle" fill="#1E293B" fontSize={size * 0.2} fontWeight="700" fontFamily="Inter, system-ui, sans-serif">N</text>
        </g>
    )
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
        const w = maxX - minX
        const h = maxY - minY
        const pad = Math.max(w, h) * 0.18
        const viewBox = `${minX - pad} ${minY - pad * 1.8} ${w + pad * 2} ${h + pad * 3.0}`
        const scale = Math.max(w, h)

        return { viewBox, minX, minY, maxX, maxY, w, h, scale, pad }
    }, [plan])

    if (!layout) return (
        <div className="preview-empty" style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            height: '100%', color: '#94A3B8', fontFamily: 'Inter, system-ui, sans-serif',
        }}>
            <p>No plan data</p>
        </div>
    )

    const { viewBox, minX, minY, maxX, maxY, w, h, scale } = layout

    const toPathD = (coords) => {
        if (!coords || coords.length < 2) return ''
        return coords.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ') + ' Z'
    }

    const wallThick = scale * 0.007
    const innerWall = scale * 0.003
    const fontSize = {
        roomName: scale * 0.016,
        roomArea: scale * 0.012,
        roomDims: scale * 0.010,
        dimLabel: scale * 0.011,
        title: scale * 0.026,
        subtitle: scale * 0.013,
    }

    const getRoomDims = (polygon) => {
        if (!polygon || polygon.length < 3) return null
        const rxs = polygon.map(p => p[0])
        const rys = polygon.map(p => p[1])
        const rw = Math.max(...rxs) - Math.min(...rxs)
        const rh = Math.max(...rys) - Math.min(...rys)
        return { w: rw, h: rh }
    }

    const totalArea = plan.total_area || plan.area_summary?.plot_area ||
        plan.rooms.reduce((sum, r) => sum + (r.actual_area || r.area || 0), 0)

    // Legend entries (unique room types)
    const zoneSet = new Map()
    plan.rooms.forEach(r => {
        const c = getRoomColor(r)
        const key = r.room_type
        if (!zoneSet.has(key)) {
            zoneSet.set(key, { label: (r.label || r.name || r.room_type).replace(/\s*\d+$/, ''), color: c })
        }
    })

    return (
        <div className="plan-svg" style={{
            padding: '1rem',
            background: 'linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%)',
            borderRadius: '12px',
            border: '1px solid #E2E8F0',
        }}>
            <svg viewBox={viewBox} xmlns="http://www.w3.org/2000/svg" style={{ background: 'transparent' }}>
                <defs>
                    <filter id="wall-shadow" x="-8%" y="-8%" width="116%" height="116%">
                        <feDropShadow dx="0" dy={scale * 0.001} stdDeviation={scale * 0.003} floodColor="#000" floodOpacity="0.12" />
                    </filter>
                    <filter id="room-glow" x="-2%" y="-2%" width="104%" height="104%">
                        <feDropShadow dx="0" dy="0" stdDeviation={scale * 0.001} floodColor="#000" floodOpacity="0.04" />
                    </filter>
                    <pattern id="grid-pattern" width={scale * 0.05} height={scale * 0.05} patternUnits="userSpaceOnUse">
                        <path d={`M ${scale * 0.05} 0 L 0 0 0 ${scale * 0.05}`}
                            fill="none" stroke="#E2E8F0" strokeWidth={scale * 0.0005} />
                    </pattern>
                </defs>

                {/* Background */}
                <rect
                    x={minX - layout.pad}
                    y={minY - layout.pad * 1.8}
                    width={w + layout.pad * 2}
                    height={h + layout.pad * 3.0}
                    fill="#FAFBFD"
                />
                {/* Subtle grid behind the plan */}
                <rect
                    x={minX - layout.pad * 0.3}
                    y={minY - layout.pad * 0.3}
                    width={w + layout.pad * 0.6}
                    height={h + layout.pad * 0.6}
                    fill="url(#grid-pattern)"
                    opacity="0.5"
                />

                {/* Outer boundary — thick dark walls with shadow */}
                <path
                    d={toPathD(plan.boundary)}
                    fill="#FFFFFF"
                    stroke="#1E293B"
                    strokeWidth={wallThick}
                    strokeLinejoin="miter"
                    filter="url(#wall-shadow)"
                />

                {/* Room polygons — zone-colored fills, thin interior walls */}
                {plan.rooms.map((room, i) => {
                    const colors = getRoomColor(room)
                    return (
                        <path
                            key={`room-${i}`}
                            d={toPathD(room.polygon)}
                            fill={colors.fill}
                            stroke="#1E293B"
                            strokeWidth={innerWall}
                            strokeLinejoin="miter"
                            filter="url(#room-glow)"
                        />
                    )
                })}

                {/* Diagonal hatching for wet areas (bathroom, kitchen) */}
                {plan.rooms.map((room, i) => {
                    if (!['bathroom', 'toilet', 'kitchen'].includes(room.room_type)) return null
                    if (!room.polygon || room.polygon.length < 3) return null
                    const rxs = room.polygon.map(p => p[0])
                    const rys = room.polygon.map(p => p[1])
                    const rx1 = Math.min(...rxs), ry1 = Math.min(...rys)
                    const rx2 = Math.max(...rxs), ry2 = Math.max(...rys)
                    const spacing = scale * 0.015
                    const lines = []
                    const diag = (rx2 - rx1) + (ry2 - ry1)
                    for (let d = 0; d < diag; d += spacing) {
                        const x1c = Math.min(rx2, rx1 + d)
                        const y1c = Math.max(ry1, ry1 + d - (rx2 - rx1))
                        const x2c = Math.max(rx1, rx1 + d - (ry2 - ry1))
                        const y2c = Math.min(ry2, ry1 + d)
                        lines.push(
                            <line key={`h-${i}-${d}`}
                                x1={x1c} y1={y1c} x2={x2c} y2={y2c}
                                stroke={room.room_type === 'kitchen' ? '#F97316' : '#14B8A6'}
                                strokeWidth={scale * 0.0004}
                                opacity="0.18"
                            />
                        )
                    }
                    return <g key={`hatchg-${i}`}>{lines}</g>
                })}

                {/* Furniture icons */}
                {plan.rooms.map((room, i) => {
                    const fn = FURNITURE_MAP[room.room_type]
                    if (!fn || !room.centroid || !room.polygon) return null
                    const dims = getRoomDims(room.polygon)
                    if (!dims) return null
                    return <g key={`furn-${i}`}>{fn(room.centroid[0], room.centroid[1], dims.w, dims.h, 1)}</g>
                })}

                {/* Door arcs */}
                {plan.doors?.map((door, i) => {
                    if (!door.hinge) {
                        return (
                            <circle key={`door-${i}`}
                                cx={door.position[0]} cy={door.position[1]}
                                r={scale * 0.005} fill="none" stroke="#475569" strokeWidth={scale * 0.0015}
                            />
                        )
                    }
                    const r = door.width
                    const hx = door.hinge[0], hy = door.hinge[1]
                    const sx = door.swing_dir[0], sy = door.swing_dir[1]
                    const arcEndX = hx + sx * r
                    const arcEndY = hy + sy * r
                    const doorEndX = door.door_end[0]
                    const doorEndY = door.door_end[1]

                    return (
                        <g key={`door-${i}`}>
                            <line x1={hx} y1={hy} x2={doorEndX} y2={doorEndY}
                                stroke="#FFFFFF" strokeWidth={wallThick * 1.3} />
                            <line x1={hx} y1={hy} x2={arcEndX} y2={arcEndY}
                                stroke="#475569" strokeWidth={scale * 0.0015} />
                            <path
                                d={`M ${arcEndX},${arcEndY} A ${r},${r} 0 0 ${sy > 0 || sx < 0 ? 1 : 0} ${doorEndX},${doorEndY}`}
                                fill="none" stroke="#475569"
                                strokeWidth={scale * 0.0012}
                                strokeDasharray={`${scale * 0.003} ${scale * 0.003}`}
                            />
                        </g>
                    )
                })}

                {/* Windows — three parallel lines */}
                {plan.windows?.map((win, i) => {
                    if (!win.start || !win.end) {
                        return (
                            <rect key={`win-${i}`}
                                x={win.position[0] - scale * 0.01}
                                y={win.position[1] - scale * 0.002}
                                width={scale * 0.02} height={scale * 0.004}
                                fill="#fff" stroke="#475569" strokeWidth={scale * 0.001}
                            />
                        )
                    }
                    const wsx = win.start[0], wsy = win.start[1]
                    const wex = win.end[0], wey = win.end[1]
                    const wdx = wex - wsx, wdy = wey - wsy
                    const wlen = Math.sqrt(wdx * wdx + wdy * wdy)
                    if (wlen < 0.1) return null
                    const wnx = -wdy / wlen, wny = wdx / wlen
                    const offset = scale * 0.003
                    return (
                        <g key={`win-${i}`}>
                            <line x1={wsx} y1={wsy} x2={wex} y2={wey}
                                stroke="#fff" strokeWidth={wallThick * 1.3} />
                            {[-1, 0, 1].map(m => (
                                <line key={m}
                                    x1={wsx + wnx * offset * m} y1={wsy + wny * offset * m}
                                    x2={wex + wnx * offset * m} y2={wey + wny * offset * m}
                                    stroke="#1E293B" strokeWidth={scale * 0.001}
                                />
                            ))}
                        </g>
                    )
                })}

                {/* Room labels — name, area, dimensions */}
                {plan.rooms.map((room, i) => {
                    if (!room.centroid) return null
                    const cx = room.centroid[0]
                    const cy = room.centroid[1]
                    const dims = getRoomDims(room.polygon)
                    const colors = getRoomColor(room)
                    const furn = FURNITURE_MAP[room.room_type]
                    const labelOffset = furn && dims ? Math.min(dims.w, dims.h) * 0.22 : 0
                    const area = room.actual_area || room.area || 0
                    const roomW = room.width || (dims?.w || 0)
                    const roomH = room.length || (dims?.h || 0)

                    return (
                        <g key={`label-${i}`}>
                            {/* Room name */}
                            <text
                                x={cx} y={cy - scale * 0.012 + labelOffset}
                                textAnchor="middle"
                                fill={colors.text}
                                fontSize={fontSize.roomName}
                                fontWeight="700"
                                fontFamily="Inter, system-ui, sans-serif"
                                letterSpacing={scale * 0.001}
                            >
                                {room.label?.toUpperCase()}
                            </text>
                            {/* Area */}
                            <text
                                x={cx} y={cy + scale * 0.008 + labelOffset}
                                textAnchor="middle"
                                fill={colors.text}
                                fontSize={fontSize.roomArea}
                                fontWeight="500"
                                fontFamily="Inter, system-ui, sans-serif"
                                opacity="0.85"
                            >
                                {area > 0 ? `${area.toFixed(0)} sq ft` : ''}
                            </text>
                            {/* Dimensions (W' x L') */}
                            {roomW > 0 && roomH > 0 && (
                                <text
                                    x={cx} y={cy + scale * 0.024 + labelOffset}
                                    textAnchor="middle"
                                    fill={colors.text}
                                    fontSize={fontSize.roomDims}
                                    fontFamily="Inter, system-ui, sans-serif"
                                    opacity="0.6"
                                >
                                    {roomW.toFixed(1)}' × {roomH.toFixed(1)}'
                                </text>
                            )}
                        </g>
                    )
                })}

                {/* Dimension lines along boundary edges */}
                {plan.boundary && plan.boundary.length > 2 && (() => {
                    const pts = plan.boundary
                    const dimLines = []
                    for (let i = 0; i < pts.length - 1; i++) {
                        const [x1, y1] = pts[i]
                        const [x2, y2] = pts[i + 1]
                        const dx = x2 - x1, dy = y2 - y1
                        const segLen = Math.sqrt(dx * dx + dy * dy)
                        if (segLen < 3) continue
                        const mx = (x1 + x2) / 2
                        const my = (y1 + y2) / 2
                        const nx = -dy / segLen, ny = dx / segLen
                        const dimOffset = scale * 0.04

                        const ox1 = x1 + nx * dimOffset
                        const oy1 = y1 + ny * dimOffset
                        const ox2 = x2 + nx * dimOffset
                        const oy2 = y2 + ny * dimOffset
                        const omx = mx + nx * dimOffset
                        const omy = my + ny * dimOffset

                        let angle = Math.atan2(dy, dx) * 180 / Math.PI
                        if (angle > 90 || angle < -90) angle += 180

                        dimLines.push(
                            <g key={`dim-${i}`}>
                                <line x1={x1} y1={y1}
                                    x2={x1 + nx * (dimOffset + scale * 0.006)}
                                    y2={y1 + ny * (dimOffset + scale * 0.006)}
                                    stroke="#94A3B8" strokeWidth={scale * 0.0008} />
                                <line x1={x2} y1={y2}
                                    x2={x2 + nx * (dimOffset + scale * 0.006)}
                                    y2={y2 + ny * (dimOffset + scale * 0.006)}
                                    stroke="#94A3B8" strokeWidth={scale * 0.0008} />
                                <line x1={ox1} y1={oy1} x2={ox2} y2={oy2}
                                    stroke="#64748B" strokeWidth={scale * 0.001} />
                                <circle cx={ox1} cy={oy1} r={scale * 0.0018} fill="#64748B" />
                                <circle cx={ox2} cy={oy2} r={scale * 0.0018} fill="#64748B" />
                                {/* Label background */}
                                <rect
                                    x={omx - scale * 0.025}
                                    y={omy - scale * 0.012}
                                    width={scale * 0.05}
                                    height={scale * 0.015}
                                    fill="#FAFBFD"
                                    rx={scale * 0.002}
                                    transform={`rotate(${angle}, ${omx}, ${omy - scale * 0.005})`}
                                />
                                <text
                                    x={omx} y={omy - scale * 0.003}
                                    textAnchor="middle"
                                    fill="#475569"
                                    fontSize={fontSize.dimLabel}
                                    fontWeight="600"
                                    fontFamily="Inter, system-ui, sans-serif"
                                    transform={`rotate(${angle}, ${omx}, ${omy - scale * 0.003})`}
                                >
                                    {segLen.toFixed(1)}'
                                </text>
                            </g>
                        )
                    }
                    return dimLines
                })()}

                {/* Compass Rose (top-right) */}
                <CompassRose
                    x={maxX + layout.pad * 0.55}
                    y={minY - layout.pad * 0.3}
                    size={scale * 0.06}
                />

                {/* Title block (top) */}
                <g>
                    {/* Decorative accent line */}
                    <line
                        x1={minX} y1={minY - h * 0.1}
                        x2={minX + scale * 0.06} y2={minY - h * 0.1}
                        stroke="#3B82F6" strokeWidth={scale * 0.003} strokeLinecap="round"
                    />
                    <text
                        x={minX}
                        y={minY - h * 0.065}
                        fill="#0F172A"
                        fontSize={fontSize.title}
                        fontWeight="800"
                        fontFamily="Inter, system-ui, sans-serif"
                        letterSpacing={scale * 0.003}
                    >
                        FLOOR PLAN
                    </text>
                    <text
                        x={minX}
                        y={minY - h * 0.025}
                        fill="#64748B"
                        fontSize={fontSize.subtitle}
                        fontWeight="500"
                        fontFamily="Inter, system-ui, sans-serif"
                    >
                        {totalArea > 0 ? `${totalArea.toFixed(0)} sq ft` : '\u2014'}  ·  {plan.rooms.length} rooms  ·  {plan.plot?.width || w.toFixed(0)}' × {plan.plot?.length || h.toFixed(0)}'
                        {plan.engine ? `  ·  ${plan.engine.toUpperCase()} engine` : ''}
                    </text>
                </g>

                {/* Legend (bottom) */}
                {(() => {
                    const entries = Array.from(zoneSet.entries())
                    const legendY = maxY + layout.pad * 0.55
                    const boxSize = scale * 0.012
                    const cols = Math.min(entries.length, 5)
                    const colWidth = scale * 0.14
                    const startX = minX

                    return (
                        <g>
                            <text
                                x={startX} y={legendY - scale * 0.008}
                                fill="#64748B" fontSize={scale * 0.01} fontWeight="600"
                                fontFamily="Inter, system-ui, sans-serif"
                                letterSpacing={scale * 0.001}
                            >
                                ROOM LEGEND
                            </text>
                            {entries.map(([type, { label, color }], idx) => {
                                const col = idx % cols
                                const row = Math.floor(idx / cols)
                                const lx = startX + col * colWidth
                                const ly = legendY + row * (boxSize * 2.2)
                                return (
                                    <g key={`legend-${idx}`}>
                                        <rect
                                            x={lx} y={ly}
                                            width={boxSize} height={boxSize}
                                            fill={color.fill} stroke={color.stroke}
                                            strokeWidth={scale * 0.0008} rx={scale * 0.001}
                                        />
                                        <text
                                            x={lx + boxSize * 1.5} y={ly + boxSize * 0.8}
                                            fill="#475569" fontSize={scale * 0.009}
                                            fontWeight="500"
                                            fontFamily="Inter, system-ui, sans-serif"
                                        >
                                            {label}
                                        </text>
                                    </g>
                                )
                            })}
                        </g>
                    )
                })()}

            </svg>
        </div>
    )
}
