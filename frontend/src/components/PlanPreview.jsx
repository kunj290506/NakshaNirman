import { useMemo } from 'react'

/* ═══════════════════════════════════════════════════════════════════
   NAKSHA NIRMAN — Fully Dynamic CAD Floor Plan Renderer
   Every value derived from actual room data. Nothing hardcoded.
   ═══════════════════════════════════════════════════════════════════ */

const ROOM_LABELS = {
  living: 'DRAWING ROOM', drawing: 'DRAWING ROOM', hall: 'HALL',
  dining: 'DINING', kitchen: 'KITCHEN',
  master_bedroom: 'MASTER BED', bedroom: 'BED ROOM',
  bathroom: 'BATH', toilet: 'TOILET', wc: 'W.C.',
  study: 'STUDY', pooja: 'PUJA', store: 'STORE',
  balcony: 'BALCONY', porch: 'PORCH', terrace: 'TERRACE',
  garage: 'GARAGE', parking: 'PARKING',
  hallway: 'HALL', utility: 'UTILITY', wash: 'WASH',
  foyer: 'SIT OUT', passage: 'PASSAGE', entrance: 'ENTRANCE',
  staircase: 'STAIRCASE', corridor: 'CORRIDOR', lobby: 'LOBBY',
  garden: 'GARDEN',
}

const WET = new Set(['bathroom', 'toilet', 'wc', 'wash', 'utility'])
const OUTDOOR = new Set(['balcony', 'porch', 'terrace', 'garden', 'foyer'])

// ── helpers ──
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

function fmtDim(ft) {
  const f = Math.floor(ft)
  const i = Math.round((ft - f) * 12)
  if (i === 0) return `${f}'-0"`
  if (i === 12) return `${f + 1}'-0"`
  return `${f}'-${i}"`
}

// ── normalizeRoom: any format → {x,y,w,h,room_type} ──
function normalizeRoom(room) {
  let x, y, w, h
  if (room.polygon && room.polygon.length >= 3) {
    const xs = room.polygon.map(p => p[0])
    const ys = room.polygon.map(p => p[1])
    x = Math.min(...xs); y = Math.min(...ys)
    w = Math.max(...xs) - x; h = Math.max(...ys) - y
  } else if (room.x != null && room.y != null && (room.width || room.w)) {
    x = room.x; y = room.y; w = room.width || room.w; h = room.height || room.h
  } else if (room.position && room.dimensions) {
    const pos = Array.isArray(room.position) ? room.position : [room.position.x, room.position.y]
    const dim = Array.isArray(room.dimensions) ? room.dimensions : [room.dimensions[0], room.dimensions[1]]
    x = pos[0]; y = pos[1]; w = dim[0]; h = dim[1]
  } else if (room.position && (room.width || room.length)) {
    const pos = Array.isArray(room.position) ? room.position : [room.position.x, room.position.y]
    x = pos[0]; y = pos[1]; w = room.width; h = room.length || room.height
  } else if (room.bbox) {
    x = room.bbox.x1; y = room.bbox.y1
    w = room.bbox.x2 - room.bbox.x1; h = room.bbox.y2 - room.bbox.y1
  } else {
    return null
  }
  if (w < 0.5 || h < 0.5) return null
  const rt = (room.room_type || room.type || 'room').toLowerCase().replace(/\s+/g, '_')
  return { x, y, w, h, room_type: rt, label: room.label || room.name }
}

// ═══════════════════════════════════════════════════════════════
// FURNITURE RENDERERS  — each takes {x,y,w,h} of its room
// ═══════════════════════════════════════════════════════════════

function BedFurniture({ x, y, w, h, isMaster, thin }) {
  const m = thin * 8  // wall inset
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const maxW = isMaster ? 6.5 : 5.0
  const maxH = isMaster ? 7.0 : 6.5
  const bw = Math.min(iw * 0.60, maxW)
  const bh = Math.min(ih * 0.55, maxH)
  const bx = ix + (iw - bw) / 2
  const by = iy + ih * 0.05
  const headH = bh * 0.13
  const pillowH = bh * 0.14
  const pillowY = by + headH + bh * 0.03
  const wardW = iw * 0.50
  const wardH = ih * 0.10
  const sw = thin * 1.2

  return (
    <g>
      <rect x={bx} y={by} width={bw} height={bh} fill="none" stroke="#333" strokeWidth={sw} />
      <rect x={bx} y={by} width={bw} height={headH} fill="#d4d4d4" stroke="#333" strokeWidth={sw} />
      {isMaster ? (
        <>
          <rect x={bx + bw * 0.06} y={pillowY} width={bw * 0.39} height={pillowH}
            rx={pillowH * 0.3} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
          <rect x={bx + bw * 0.55} y={pillowY} width={bw * 0.39} height={pillowH}
            rx={pillowH * 0.3} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
        </>
      ) : (
        <rect x={bx + bw * 0.16} y={pillowY} width={bw * 0.68} height={pillowH}
          rx={pillowH * 0.3} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
      )}
      <rect x={ix + (iw - wardW) / 2} y={iy + ih - wardH - ih * 0.03}
        width={wardW} height={wardH} fill="#e8e8e8" stroke="#555" strokeWidth={sw} />
    </g>
  )
}

function SofaFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const sofaW = Math.min(iw * 0.58, Math.max(iw * 0.42, 7.0))
  const sofaH = Math.min(ih * 0.20, Math.max(ih * 0.14, 2.4))
  const sx = ix + (iw - sofaW) / 2
  const sy = iy + ih * 0.06
  const backH = sofaH * 0.28
  const armW = sofaW * 0.09

  const tableW = sofaW * 0.40
  const tableH = sofaH * 0.42
  const tx = ix + (iw - tableW) / 2
  const ty = sy + sofaH + ih * 0.05

  return (
    <g>
      <rect x={sx} y={sy} width={sofaW} height={sofaH} fill="none" stroke="#333" strokeWidth={sw} />
      <rect x={sx} y={sy} width={sofaW} height={backH} fill="#d9d9d9" stroke="#333" strokeWidth={sw} />
      <rect x={sx} y={sy + backH} width={armW} height={sofaH - backH} fill="#e0e0e0" stroke="#333" strokeWidth={sw * 0.8} />
      <rect x={sx + sofaW - armW} y={sy + backH} width={armW} height={sofaH - backH} fill="#e0e0e0" stroke="#333" strokeWidth={sw * 0.8} />
      <line x1={sx + sofaW * 0.36} y1={sy + backH} x2={sx + sofaW * 0.36} y2={sy + sofaH} stroke="#999" strokeWidth={sw * 0.6} />
      <line x1={sx + sofaW * 0.64} y1={sy + backH} x2={sx + sofaW * 0.64} y2={sy + sofaH} stroke="#999" strokeWidth={sw * 0.6} />
      <rect x={tx} y={ty} width={tableW} height={tableH}
        rx={tableH * 0.2} fill="none" stroke="#555" strokeWidth={sw} />
      {iw > 14 && (
        <>
          <rect x={ix + iw * 0.04} y={sy + sofaH * 0.3} width={iw * 0.10} height={sofaH * 0.7}
            fill="none" stroke="#555" strokeWidth={sw * 0.8} />
          <rect x={ix + iw - iw * 0.14} y={sy + sofaH * 0.3} width={iw * 0.10} height={sofaH * 0.7}
            fill="none" stroke="#555" strokeWidth={sw * 0.8} />
        </>
      )}
    </g>
  )
}

function DiningFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const rx = Math.min(iw * 0.24, 2.5)
  const ry = Math.min(ih * 0.22, 1.4)
  const cx = ix + iw / 2
  const cy = iy + ih / 2
  const cw = rx * 0.32
  const ch = ry * 0.44

  const sideCount = iw > 13 ? 2 : 1

  return (
    <g>
      {/* Oval table */}
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke="#333" strokeWidth={sw} />
      {/* Top chair */}
      <rect x={cx - cw / 2} y={cy - ry - ch - ry * 0.15} width={cw} height={ch}
        rx={cw * 0.18} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
      {/* Bottom chair */}
      <rect x={cx - cw / 2} y={cy + ry + ry * 0.15} width={cw} height={ch}
        rx={cw * 0.18} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
      {/* Side chairs */}
      {Array.from({ length: sideCount }).map((_, i) => {
        const offY = sideCount === 1 ? 0 : (i - 0.5) * ry * 0.9
        return (
          <g key={`side-${i}`}>
            <rect x={cx - rx - ch - rx * 0.15} y={cy + offY - cw / 2} width={ch} height={cw}
              rx={cw * 0.18} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
            <rect x={cx + rx + rx * 0.15} y={cy + offY - cw / 2} width={ch} height={cw}
              rx={cw * 0.18} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
          </g>
        )
      })}
    </g>
  )
}

function KitchenFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const depth = Math.min(iw * 0.20, ih * 0.22, 1.8)

  const topW = iw - depth
  const rightH = ih - depth

  const sinkCx = ix + topW * 0.58
  const sinkCy = iy + depth * 0.5
  const basinRx = depth * 0.22
  const basinRy = depth * 0.28

  const hobCx = ix + topW * 0.3
  const hobCy = iy + depth * 0.5
  const burnerR = depth * 0.12
  const burnerGap = burnerR * 1.6

  const fridgeW = depth * 0.75
  const fridgeH = rightH * 0.22
  const fridgeX = ix + iw - depth + (depth - fridgeW) / 2
  const fridgeY = iy + ih - fridgeH - ih * 0.04

  return (
    <g>
      {/* Top counter */}
      <rect x={ix} y={iy} width={topW} height={depth} fill="#eee" stroke="#333" strokeWidth={sw} />
      {/* Right counter */}
      <rect x={ix + iw - depth} y={iy} width={depth} height={ih} fill="#eee" stroke="#333" strokeWidth={sw} />
      {/* Sink basins */}
      <ellipse cx={sinkCx - basinRx * 1.1} cy={sinkCy} rx={basinRx} ry={basinRy} fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      <ellipse cx={sinkCx + basinRx * 1.1} cy={sinkCy} rx={basinRx} ry={basinRy} fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      {/* Gas hob */}
      <circle cx={hobCx - burnerGap / 2} cy={hobCy - burnerGap / 2} r={burnerR} fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      <circle cx={hobCx + burnerGap / 2} cy={hobCy - burnerGap / 2} r={burnerR} fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      <circle cx={hobCx - burnerGap / 2} cy={hobCy + burnerGap / 2} r={burnerR} fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      <circle cx={hobCx + burnerGap / 2} cy={hobCy + burnerGap / 2} r={burnerR} fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      {/* Fridge */}
      <rect x={fridgeX} y={fridgeY} width={fridgeW} height={fridgeH} fill="#e0e0e0" stroke="#555" strokeWidth={sw} />
    </g>
  )
}

function BathroomFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const isNarrow = ih > iw * 1.3
  const unit = Math.min(iw, ih) * 0.28

  if (isNarrow) {
    const wcCx = ix + iw * 0.5
    const wcCy = iy + ih * 0.72
    const bowlRx = unit * 0.55
    const bowlRy = unit * 0.7
    const cisternW = bowlRx * 1.6
    const cisternH = unit * 0.35

    const basinX = ix + iw * 0.2
    const basinY = iy + ih * 0.05
    const basinW = iw * 0.6
    const basinH = ih * 0.18

    return (
      <g>
        {/* WC bowl */}
        <ellipse cx={wcCx} cy={wcCy} rx={bowlRx} ry={bowlRy} fill="none" stroke="#555" strokeWidth={sw} />
        {/* WC cistern */}
        <rect x={wcCx - cisternW / 2} y={wcCy + bowlRy} width={cisternW} height={cisternH}
          fill="#e8e8e8" stroke="#555" strokeWidth={sw} />
        {/* Basin */}
        <rect x={basinX} y={basinY} width={basinW} height={basinH}
          rx={basinH * 0.3} fill="none" stroke="#555" strokeWidth={sw} />
        <ellipse cx={basinX + basinW * 0.5} cy={basinY + basinH * 0.45}
          rx={basinW * 0.2} ry={basinH * 0.22} fill="none" stroke="#888" strokeWidth={sw * 0.6} />
        <circle cx={basinX + basinW * 0.5} cy={basinY + basinH * 0.2} r={unit * 0.06} fill="#888" />
      </g>
    )
  }
  // Wide: WC at right, basin at left
  const wcCx = ix + iw * 0.72
  const wcCy = iy + ih * 0.5
  const bowlRx = unit * 0.7
  const bowlRy = unit * 0.55
  const cisternW = unit * 0.35
  const cisternH = bowlRy * 1.6

  const basinX = ix + iw * 0.05
  const basinY = iy + ih * 0.25
  const basinW = unit * 0.6
  const basinH = ih * 0.5

  return (
    <g>
      <ellipse cx={wcCx} cy={wcCy} rx={bowlRx} ry={bowlRy} fill="none" stroke="#555" strokeWidth={sw} />
      <rect x={wcCx + bowlRx} y={wcCy - cisternH / 2} width={cisternW} height={cisternH}
        fill="#e8e8e8" stroke="#555" strokeWidth={sw} />
      <rect x={basinX} y={basinY} width={basinW} height={basinH}
        rx={basinW * 0.3} fill="none" stroke="#555" strokeWidth={sw} />
      <ellipse cx={basinX + basinW * 0.5} cy={basinY + basinH * 0.45}
        rx={basinW * 0.22} ry={basinH * 0.2} fill="none" stroke="#888" strokeWidth={sw * 0.6} />
      <circle cx={basinX + basinW * 0.5} cy={basinY + basinH * 0.2} r={unit * 0.06} fill="#888" />
    </g>
  )
}

function PoojaFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const altW = iw * 0.7
  const altH = ih * 0.35
  const ax = ix + (iw - altW) / 2
  const ay = iy + ih * 0.08
  const diyaR = Math.min(iw, ih) * 0.07

  return (
    <g>
      {/* Altar cabinet */}
      <rect x={ax} y={ay} width={altW} height={altH} fill="#f0ead6" stroke="#555" strokeWidth={sw} />
      {/* Cross dividers */}
      <line x1={ax} y1={ay + altH / 2} x2={ax + altW} y2={ay + altH / 2} stroke="#999" strokeWidth={sw * 0.6} />
      <line x1={ax + altW / 2} y1={ay} x2={ax + altW / 2} y2={ay + altH} stroke="#999" strokeWidth={sw * 0.6} />
      {/* Diya */}
      <circle cx={ix + iw / 2} cy={ay + altH + ih * 0.12} r={diyaR} fill="none" stroke="#b8860b" strokeWidth={sw} />
    </g>
  )
}

function StaircaseFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const steps = Math.max(5, Math.min(14, Math.round(ih / 0.75)))
  const stepH = ih / steps
  const fontSize = Math.min(iw * 0.14, 0.9)
  const arrowX = ix + iw * 0.5

  return (
    <g>
      {Array.from({ length: steps }).map((_, i) => (
        <line key={i} x1={ix} y1={iy + stepH * i} x2={ix + iw} y2={iy + stepH * i}
          stroke="#999" strokeWidth={sw * 0.7} />
      ))}
      <line x1={arrowX} y1={iy + ih * 0.75} x2={arrowX} y2={iy + ih * 0.2}
        stroke="#333" strokeWidth={sw * 1.2} />
      <polygon points={`${arrowX},${iy + ih * 0.15} ${arrowX - iw * 0.08},${iy + ih * 0.23} ${arrowX + iw * 0.08},${iy + ih * 0.23}`}
        fill="#333" />
      <text x={arrowX + iw * 0.2} y={iy + ih * 0.55}
        fontSize={fontSize} fontFamily="Times New Roman"
        fill="#333" textAnchor="middle" transform={`rotate(-90,${arrowX + iw * 0.2},${iy + ih * 0.55})`}>
        UP
      </text>
    </g>
  )
}

function StudyFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const deskW = iw * 0.72
  const deskH = ih * 0.22
  const dx = ix + (iw - deskW) / 2
  const dy = iy + ih * 0.08
  const monW = deskW * 0.4
  const monH = deskH * 0.35
  const chairW = deskW * 0.3
  const chairH = ih * 0.14

  return (
    <g>
      {/* Desk */}
      <rect x={dx} y={dy} width={deskW} height={deskH} fill="none" stroke="#333" strokeWidth={sw} />
      {/* Monitor */}
      <rect x={dx + (deskW - monW) / 2} y={dy + deskH * 0.15} width={monW} height={monH}
        fill="none" stroke="#555" strokeWidth={sw * 0.7} />
      {/* Chair */}
      <rect x={dx + (deskW - chairW) / 2} y={dy + deskH + ih * 0.06} width={chairW} height={chairH}
        rx={chairW * 0.15} fill="none" stroke="#555" strokeWidth={sw * 0.8} />
    </g>
  )
}

function StoreFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const shelfH = Math.max(ih * 0.05, 0.18)
  return (
    <g>
      {[0.25, 0.5, 0.75].map(pct => (
        <rect key={pct} x={ix + iw * 0.05} y={iy + ih * pct - shelfH / 2}
          width={iw * 0.9} height={shelfH} fill="#e0e0e0" stroke="#888" strokeWidth={sw * 0.7} />
      ))}
    </g>
  )
}

function BalconyFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 0.8
  const spacing = Math.max(1.8, Math.min(iw, ih) * 0.28)
  const lines = []
  for (let d = spacing; d < iw + ih; d += spacing) {
    lines.push({
      x1: ix + Math.max(0, d - ih), y1: iy + Math.min(d, ih),
      x2: ix + Math.min(d, iw),     y2: iy + Math.max(0, d - iw),
    })
  }
  return (
    <g>
      {lines.map((l, i) => (
        <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}
          stroke="#bbb" strokeWidth={sw} />
      ))}
    </g>
  )
}

function GarageFurniture({ x, y, w, h, thin }) {
  const m = thin * 8
  const ix = x + m, iy = y + m, iw = w - m * 2, ih = h - m * 2
  const sw = thin * 1.2
  const carW = iw * 0.7
  const carH = ih * 0.55
  const cx = ix + (iw - carW) / 2
  const cy = iy + (ih - carH) / 2
  const wheelR = ih * 0.09
  const roofInset = carW * 0.1

  return (
    <g>
      {/* Car body */}
      <rect x={cx} y={cy} width={carW} height={carH} rx={carH * 0.08}
        fill="none" stroke="#333" strokeWidth={sw} />
      {/* Roof shape */}
      <line x1={cx + roofInset} y1={cy + carH * 0.3} x2={cx + carW - roofInset} y2={cy + carH * 0.3}
        stroke="#555" strokeWidth={sw * 0.8} />
      <line x1={cx + roofInset} y1={cy + carH * 0.3} x2={cx} y2={cy + carH * 0.5}
        stroke="#555" strokeWidth={sw * 0.8} />
      <line x1={cx + carW - roofInset} y1={cy + carH * 0.3} x2={cx + carW} y2={cy + carH * 0.5}
        stroke="#555" strokeWidth={sw * 0.8} />
      {/* Wheels */}
      <circle cx={cx + carW * 0.22} cy={cy} r={wheelR} fill="none" stroke="#333" strokeWidth={sw} />
      <circle cx={cx + carW * 0.78} cy={cy} r={wheelR} fill="none" stroke="#333" strokeWidth={sw} />
      <circle cx={cx + carW * 0.22} cy={cy + carH} r={wheelR} fill="none" stroke="#333" strokeWidth={sw} />
      <circle cx={cx + carW * 0.78} cy={cy + carH} r={wheelR} fill="none" stroke="#333" strokeWidth={sw} />
    </g>
  )
}

// ── dispatcher ──
function getFurniture(room, thin) {
  const t = room.room_type
  if (t === 'master_bedroom') return <BedFurniture {...room} isMaster thin={thin} />
  if (t === 'bedroom') return <BedFurniture {...room} isMaster={false} thin={thin} />
  if (t === 'living' || t === 'drawing' || t === 'hall' || t === 'hallway') return <SofaFurniture {...room} thin={thin} />
  if (t === 'dining') return <DiningFurniture {...room} thin={thin} />
  if (t === 'kitchen') return <KitchenFurniture {...room} thin={thin} />
  if (t === 'bathroom' || t === 'toilet' || t === 'wc') return <BathroomFurniture {...room} thin={thin} />
  if (t === 'pooja') return <PoojaFurniture {...room} thin={thin} />
  if (t === 'staircase') return <StaircaseFurniture {...room} thin={thin} />
  if (t === 'study') return <StudyFurniture {...room} thin={thin} />
  if (t === 'store') return <StoreFurniture {...room} thin={thin} />
  if (t === 'balcony' || t === 'porch' || t === 'terrace' || t === 'garden') return <BalconyFurniture {...room} thin={thin} />
  if (t === 'garage' || t === 'parking') return <GarageFurniture {...room} thin={thin} />
  return null // passage, corridor, lobby, entrance, foyer — empty
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function PlanPreview({ plan, selectedRoomId, showDimensions = true, showLabels = true, showFurniture = true }) {

  // ── 1. Normalise rooms ──
  const rooms = useMemo(() => {
    if (!plan) return []
    const src = plan.rooms || plan.layout?.rooms || []
    return src.map(normalizeRoom).filter(Boolean)
  }, [plan])

  // ── 2. Layout metrics (bounding box, walls, padding, viewBox) ──
  const layout = useMemo(() => {
    if (rooms.length === 0) return null

    // bounding box from rooms
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const r of rooms) {
      if (r.x < minX) minX = r.x
      if (r.y < minY) minY = r.y
      if (r.x + r.w > maxX) maxX = r.x + r.w
      if (r.y + r.h > maxY) maxY = r.y + r.h
    }

    const plotW = maxX - minX
    const plotH = maxY - minY

    const EXT_W = clamp(plotW * 0.022, 0.5, 0.85)
    const INT_W = EXT_W * 0.5

    const pad = Math.max(plotW, plotH) * 0.20
    const thinLine = Math.max(plotW, plotH) * 0.002

    // viewBox with padding for dimensions, road, title
    const vx = minX - pad * 1.1
    const vy = minY - pad * 0.9
    const vw = plotW + pad * 2.2
    const vh = plotH + pad * 2.6

    const totalArea = plan?.total_area || rooms.reduce((s, r) => s + r.w * r.h, 0)

    return { minX, minY, maxX, maxY, plotW, plotH, EXT_W, INT_W, pad, thinLine, vx, vy, vw, vh, totalArea }
  }, [rooms, plan])

  // ── 3. Compute walls from room adjacency ──
  const walls = useMemo(() => {
    if (!layout) return { internal: [], external: [] }
    const { minX, minY, maxX, maxY, EXT_W, INT_W } = layout
    const internal = []

    for (let i = 0; i < rooms.length; i++) {
      for (let j = i + 1; j < rooms.length; j++) {
        const A = rooms[i], B = rooms[j]

        // vertical shared edge: A's right touches B's left
        if (Math.abs((A.x + A.w) - B.x) < 0.15) {
          const oy1 = Math.max(A.y, B.y)
          const oy2 = Math.min(A.y + A.h, B.y + B.h)
          if (oy2 - oy1 > 0.1) {
            internal.push({ x: A.x + A.w - INT_W / 2, y: oy1, w: INT_W, h: oy2 - oy1 })
          }
        }
        // vertical: B's right touches A's left
        if (Math.abs((B.x + B.w) - A.x) < 0.15) {
          const oy1 = Math.max(A.y, B.y)
          const oy2 = Math.min(A.y + A.h, B.y + B.h)
          if (oy2 - oy1 > 0.1) {
            internal.push({ x: B.x + B.w - INT_W / 2, y: oy1, w: INT_W, h: oy2 - oy1 })
          }
        }
        // horizontal: A's bottom touches B's top
        if (Math.abs((A.y + A.h) - B.y) < 0.15) {
          const ox1 = Math.max(A.x, B.x)
          const ox2 = Math.min(A.x + A.w, B.x + B.w)
          if (ox2 - ox1 > 0.1) {
            internal.push({ x: ox1, y: A.y + A.h - INT_W / 2, w: ox2 - ox1, h: INT_W })
          }
        }
        // horizontal: B's bottom touches A's top
        if (Math.abs((B.y + B.h) - A.y) < 0.15) {
          const ox1 = Math.max(A.x, B.x)
          const ox2 = Math.min(A.x + A.w, B.x + B.w)
          if (ox2 - ox1 > 0.1) {
            internal.push({ x: ox1, y: B.y + B.h - INT_W / 2, w: ox2 - ox1, h: INT_W })
          }
        }
      }
    }

    // External boundary walls — 4 rects
    const external = [
      { x: minX - EXT_W, y: minY - EXT_W, w: (maxX - minX) + EXT_W * 2, h: EXT_W }, // top
      { x: minX - EXT_W, y: maxY, w: (maxX - minX) + EXT_W * 2, h: EXT_W }, // bottom
      { x: minX - EXT_W, y: minY, w: EXT_W, h: maxY - minY }, // left
      { x: maxX, y: minY, w: EXT_W, h: maxY - minY }, // right
    ]

    return { internal, external }
  }, [rooms, layout])

  // ── guard ──
  if (!plan || rooms.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: '#999', fontFamily: 'Times New Roman', fontSize: '1.1rem' }}>
        No floor plan data
      </div>
    )
  }

  const { minX, minY, maxX, maxY, plotW, plotH, EXT_W, INT_W, pad, thinLine, vx, vy, vw, vh, totalArea } = layout

  // room fill colour
  const fillOf = (rt) => WET.has(rt) ? '#e6f0f8' : OUTDOOR.has(rt) ? '#ebf2e6' : '#fafafa'

  // column mark size
  const colSz = EXT_W * 0.55
  const colHalf = colSz / 2

  // collect only the 4 boundary corners for column marks (not internal junctions)
  const columnMarks = [
    [minX, minY], [maxX, minY], [minX, maxY], [maxX, maxY]
  ]

  // doors & windows from plan — deduplicate doors that overlap (two rooms adding a door on same shared wall)
  const rawDoors = plan.doors || []
  const doors = rawDoors.filter((d, i) => {
    if (!d.hinge) return false
    for (let j = 0; j < i; j++) {
      if (!rawDoors[j].hinge) continue
      const dx = Math.abs(d.hinge[0] - rawDoors[j].hinge[0])
      const dy = Math.abs(d.hinge[1] - rawDoors[j].hinge[1])
      if (dx + dy < 2.0) return false // skip duplicate
    }
    return true
  })
  const windows = plan.windows || []

  // dimension label font
  const dimFont = Math.max(0.6, pad * 0.10)
  const tickLen = pad * 0.04
  const dimOffset = pad * 0.22

  // north arrow
  const naR = pad * 0.12
  const naCx = maxX + pad * 0.55
  const naCy = minY - pad * 0.25

  // road
  const roadY = maxY + EXT_W + pad * 0.32
  const roadFont = Math.max(0.9, pad * 0.16)

  // title
  const titleY = roadY + pad * 0.35
  const titleFont = Math.max(1.2, pad * 0.22)
  const infoFont = Math.max(0.6, pad * 0.10)

  return (
    <div style={{ width: '100%', height: '100%', background: '#fff' }}>
      <svg
        viewBox={`${vx} ${vy} ${vw} ${vh}`}
        width="100%" height="100%"
        style={{ display: 'block' }}
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* 1 — white background */}
        <rect x={vx} y={vy} width={vw} height={vh} fill="#fff" />

        {/* 2 — room fills */}
        {rooms.map((r, i) => {
          const rid = r.id || `room-${i}`
          const isSelected = rid === selectedRoomId
          return (
            <g key={`fill-${i}`} data-room-id={rid} style={{ cursor: 'pointer' }}>
              <rect x={r.x} y={r.y} width={r.w} height={r.h}
                fill={fillOf(r.room_type)}
                stroke={isSelected ? '#111' : 'none'}
                strokeWidth={isSelected ? thinLine * 4 : 0}
                strokeDasharray={isSelected ? `${thinLine * 3} ${thinLine * 2}` : 'none'}
              />
            </g>
          )
        })}

        {/* 3 — furniture */}
        {showFurniture && <g opacity={0.88}>
          {rooms.map((r, i) => (
            <g key={`furn-${i}`}>{getFurniture(r, thinLine)}</g>
          ))}
        </g>}

        {/* 4 — column marks (only at plot boundary corners, not internal junctions) */}
        {columnMarks.map(([cx, cy], i) => (
          <rect key={`col-${i}`} x={cx - colHalf} y={cy - colHalf} width={colSz} height={colSz}
            fill="#222" stroke="#111" strokeWidth={thinLine * 0.5} />
        ))}

        {/* 5 — internal walls */}
        {walls.internal.map((w, i) => (
          <rect key={`iw-${i}`} x={w.x} y={w.y} width={w.w} height={w.h} fill="#2d2d2d" />
        ))}

        {/* 6 — external walls */}
        {walls.external.map((w, i) => (
          <rect key={`ew-${i}`} x={w.x} y={w.y} width={w.w} height={w.h} fill="#111" />
        ))}

        {/* 7 — doors */}
        {doors.map((d, i) => {
          if (!d.hinge || !d.door_end || !d.swing_dir || !d.width) return null
          const [hx, hy] = d.hinge
          const [ex, ey] = d.door_end
          const [sx, sy] = d.swing_dir
          // Door leaf length = distance from hinge to door_end
          const leafLen = Math.sqrt((ex - hx) ** 2 + (ey - hy) ** 2)
          // Cap visual arc radius so it doesn't overwhelm small rooms
          const arcR = Math.min(leafLen, 2.8)
          const scale = arcR / (leafLen || 1)
          // Scaled end points
          const doorEndScaled = [hx + (ex - hx) * scale, hy + (ey - hy) * scale]
          const swingEndScaled = [hx + sx * arcR, hy + sy * arcR]
          const gapSw = INT_W * 2.2
          return (
            <g key={`door-${i}`}>
              {/* White gap through wall */}
              <line x1={hx} y1={hy} x2={ex} y2={ey} stroke="#fff" strokeWidth={gapSw} />
              {/* Door leaf line */}
              <line x1={hx} y1={hy} x2={swingEndScaled[0]} y2={swingEndScaled[1]}
                stroke="#333" strokeWidth={thinLine * 1.2} />
              {/* Swing arc */}
              <path
                d={`M${swingEndScaled[0]},${swingEndScaled[1]} A${arcR},${arcR} 0 0 ${sx * sy >= 0 ? 0 : 1} ${doorEndScaled[0]},${doorEndScaled[1]}`}
                fill="none" stroke="#666" strokeWidth={thinLine * 0.7}
              />
              {/* Hinge dot */}
              <circle cx={hx} cy={hy} r={EXT_W * 0.12} fill="#333" />
            </g>
          )
        })}

        {/* 8 — windows */}
        {windows.map((win, i) => {
          if (!win.start || !win.end) return null
          const [sx, sy] = win.start
          const [ex, ey] = win.end
          const isVert = Math.abs(ex - sx) < Math.abs(ey - sy)
          const off = 0.22
          const sw = thinLine * 1.4
          return (
            <g key={`win-${i}`}>
              <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="#fff" strokeWidth={0.55} />
              <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="#333" strokeWidth={sw} />
              {isVert ? (
                <>
                  <line x1={sx - off} y1={sy} x2={ex - off} y2={ey} stroke="#333" strokeWidth={sw * 0.7} />
                  <line x1={sx + off} y1={sy} x2={ex + off} y2={ey} stroke="#333" strokeWidth={sw * 0.7} />
                  <line x1={sx - off} y1={sy} x2={sx + off} y2={sy} stroke="#333" strokeWidth={sw * 0.7} />
                  <line x1={ex - off} y1={ey} x2={ex + off} y2={ey} stroke="#333" strokeWidth={sw * 0.7} />
                </>
              ) : (
                <>
                  <line x1={sx} y1={sy - off} x2={ex} y2={ey - off} stroke="#333" strokeWidth={sw * 0.7} />
                  <line x1={sx} y1={sy + off} x2={ex} y2={ey + off} stroke="#333" strokeWidth={sw * 0.7} />
                  <line x1={sx} y1={sy - off} x2={sx} y2={sy + off} stroke="#333" strokeWidth={sw * 0.7} />
                  <line x1={ex} y1={ey - off} x2={ex} y2={ey + off} stroke="#333" strokeWidth={sw * 0.7} />
                </>
              )}
            </g>
          )
        })}

        {/* 9 — room labels — positioned in lower portion to avoid furniture overlap */}
        {showLabels && rooms.map((r, i) => {
          if (Math.min(r.w, r.h) < 1.8) return null
          const fontSize = clamp(Math.min(r.w, r.h) * 0.13, 0.45, 1.5)
          const cx = r.x + r.w / 2
          // Place label in the lower 60% of room to avoid furniture (beds, sofas, counters at top)
          const hasFurnitureTop = ['master_bedroom', 'bedroom', 'living', 'drawing', 'kitchen', 'dining'].includes(r.room_type)
          const cy = hasFurnitureTop ? r.y + r.h * 0.62 : r.y + r.h / 2
          const lbl = (ROOM_LABELS[r.room_type] || r.label || r.room_type || 'ROOM').toUpperCase()
          // White background behind label for readability
          const lblW = lbl.length * fontSize * 0.52
          const lblH = fontSize * 2.0
          return (
            <g key={`lbl-${i}`}>
              <rect x={cx - lblW / 2} y={cy - lblH / 2} width={lblW} height={lblH}
                fill="#fff" fillOpacity={0.85} rx={0.15} />
              <text x={cx} y={cy - fontSize * 0.2}
                fontSize={fontSize} fontFamily="Times New Roman" fontWeight="bold"
                fill="#111" textAnchor="middle" dominantBaseline="central">
                {lbl}
              </text>
              <text x={cx} y={cy + fontSize * 0.8}
                fontSize={fontSize * 0.68} fontFamily="Times New Roman"
                fill="#555" textAnchor="middle" dominantBaseline="central">
                {fmtDim(r.w)} × {fmtDim(r.h)}
              </text>
            </g>
          )
        })}

        {/* 10 — only 2 dimension lines: top (width) + left (depth) */}
        {showDimensions && <>
        {/* Top — overall width */}
        <g>
          <line x1={minX} y1={minY - dimOffset} x2={maxX} y2={minY - dimOffset}
            stroke="#333" strokeWidth={thinLine} />
          {/* ticks */}
          <line x1={minX} y1={minY - dimOffset - tickLen} x2={minX} y2={minY - dimOffset + tickLen}
            stroke="#333" strokeWidth={thinLine * 1.5} />
          <line x1={maxX} y1={minY - dimOffset - tickLen} x2={maxX} y2={minY - dimOffset + tickLen}
            stroke="#333" strokeWidth={thinLine * 1.5} />
          <text x={(minX + maxX) / 2} y={minY - dimOffset - tickLen - dimFont * 0.4}
            fontSize={dimFont} fontFamily="Times New Roman" fill="#333" textAnchor="middle">
            {fmtDim(plotW)}
          </text>
        </g>
        {/* Left — overall depth */}
        <g>
          <line x1={minX - dimOffset} y1={minY} x2={minX - dimOffset} y2={maxY}
            stroke="#333" strokeWidth={thinLine} />
          <line x1={minX - dimOffset - tickLen} y1={minY} x2={minX - dimOffset + tickLen} y2={minY}
            stroke="#333" strokeWidth={thinLine * 1.5} />
          <line x1={minX - dimOffset - tickLen} y1={maxY} x2={minX - dimOffset + tickLen} y2={maxY}
            stroke="#333" strokeWidth={thinLine * 1.5} />
          <text x={minX - dimOffset - tickLen - dimFont * 0.4} y={(minY + maxY) / 2}
            fontSize={dimFont} fontFamily="Times New Roman" fill="#333" textAnchor="middle"
            transform={`rotate(-90,${minX - dimOffset - tickLen - dimFont * 0.4},${(minY + maxY) / 2})`}>
            {fmtDim(plotH)}
          </text>
        </g>
        </>}

        {/* 11 — road label */}
        <line x1={minX - EXT_W} y1={roadY} x2={maxX + EXT_W} y2={roadY}
          stroke="#333" strokeWidth={thinLine} strokeDasharray={`${thinLine * 6} ${thinLine * 3}`} />
        <text x={(minX + maxX) / 2} y={roadY + roadFont * 1.1}
          fontSize={roadFont} fontFamily="Times New Roman" fontWeight="bold"
          fill="#333" textAnchor="middle" letterSpacing={pad * 0.03}>
          R O A D
        </text>

        {/* 12 — north arrow */}
        <circle cx={naCx} cy={naCy} r={naR} fill="none" stroke="#333" strokeWidth={thinLine * 1.5} />
        <polygon
          points={`${naCx},${naCy - naR * 1.1} ${naCx - naR * 0.35},${naCy - naR * 0.2} ${naCx + naR * 0.35},${naCy - naR * 0.2}`}
          fill="#333"
        />
        <text x={naCx} y={naCy + naR * 0.45}
          fontSize={Math.max(0.7, naR * 0.85)} fontFamily="Times New Roman" fontWeight="bold"
          fill="#333" textAnchor="middle" dominantBaseline="central">
          N
        </text>

        {/* 13 — title block */}
        <line x1={minX - EXT_W} y1={titleY} x2={maxX + EXT_W} y2={titleY}
          stroke="#333" strokeWidth={thinLine * 2} />
        <text x={(minX + maxX) / 2} y={titleY + titleFont * 1.2}
          fontSize={titleFont} fontFamily="Times New Roman" fontWeight="bold"
          fill="#111" textAnchor="middle">
          GROUND FLOOR PLAN
        </text>
        <text x={(minX + maxX) / 2} y={titleY + titleFont * 1.2 + infoFont * 1.6}
          fontSize={infoFont} fontFamily="Times New Roman"
          fill="#555" textAnchor="middle">
          Plot: {fmtDim(plotW)} × {fmtDim(plotH)} | Area: {Math.round(totalArea)} sq.ft
        </text>

        {/* 14 — drawing border */}
        <rect x={vx + pad * 0.08} y={vy + pad * 0.08}
          width={vw - pad * 0.16} height={vh - pad * 0.16}
          fill="none" stroke="#111" strokeWidth={thinLine * 2} />
      </svg>
    </div>
  )
}
