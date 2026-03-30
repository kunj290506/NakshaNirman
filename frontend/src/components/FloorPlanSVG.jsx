/**
 * FloorPlanSVG — Professional CAD-style architectural floor plan renderer.
 * Pure black-and-white. Furniture symbols, door arcs, dimension lines,
 * hatch patterns, north arrow, scale bar, title block.
 * Zero colors — only grayscale values.
 */

const SCALE = 12  // pixels per foot
const MARGIN = 50 // px margin for dimension lines

// ─────────────────────────────────────────────────────────────
// Furniture drawing functions
// ─────────────────────────────────────────────────────────────
function drawBedFurniture(x, y, w, h, isMain) {
  const rw = w * SCALE
  const rh = h * SCALE
  const bedW = Math.min(rw * 0.65, isMain ? 78 : 66)
  const bedH = Math.min(rh * 0.50, isMain ? 80 : 68)
  const bx = x + (rw - bedW) / 2
  const by = y + (rh - bedH) / 2

  const pillowH = bedH * 0.14
  const headH = 3

  return (
    <g>
      {/* Bed frame */}
      <rect x={bx} y={by} width={bedW} height={bedH}
        fill="none" stroke="#333" strokeWidth="0.8" />
      {/* Headboard */}
      <rect x={bx} y={by} width={bedW} height={headH}
        fill="#333" stroke="none" />
      {/* Mattress inner line */}
      <rect x={bx + 3} y={by + pillowH + 6} width={bedW - 6} height={bedH - pillowH - 9}
        fill="none" stroke="#888" strokeWidth="0.4" />
      {/* Pillows */}
      {isMain ? (
        <>
          <rect x={bx + 4} y={by + headH + 3} width={bedW / 2 - 6} height={pillowH}
            fill="none" stroke="#555" strokeWidth="0.5" rx="2" />
          <rect x={bx + bedW / 2 + 2} y={by + headH + 3} width={bedW / 2 - 6} height={pillowH}
            fill="none" stroke="#555" strokeWidth="0.5" rx="2" />
        </>
      ) : (
        <rect x={bx + bedW * 0.1} y={by + headH + 3} width={bedW * 0.8} height={pillowH}
          fill="none" stroke="#555" strokeWidth="0.5" rx="2" />
      )}
      {/* Nightstand (master only) */}
      {isMain && (
        <rect x={bx - 14} y={by + bedH - 16} width={12} height={14}
          fill="none" stroke="#888" strokeWidth="0.5" />
      )}
    </g>
  )
}

function drawBathroomFixtures(x, y, w, h, isMaster) {
  const rw = w * SCALE
  const rh = h * SCALE

  const wcW = Math.min(16, rw * 0.28)
  const wcH = Math.min(22, rh * 0.30)
  const sinkR = Math.min(7, rw * 0.10)

  return (
    <g>
      {/* Toilet bowl (oval) */}
      <ellipse cx={x + rw - wcW / 2 - 8} cy={y + rh - wcH / 2 - 8}
        rx={wcW / 2} ry={wcH / 2}
        fill="none" stroke="#444" strokeWidth="0.7" />
      {/* Toilet tank */}
      <rect x={x + rw - wcW - 8} y={y + rh - wcH - 8}
        width={wcW} height={wcH * 0.35}
        fill="none" stroke="#444" strokeWidth="0.7" rx="1" />
      {/* Sink */}
      <circle cx={x + sinkR + 10} cy={y + sinkR + 10}
        r={sinkR} fill="none" stroke="#444" strokeWidth="0.7" />
      {/* Shower area (dashed corner) */}
      <line x1={x + 4} y1={y + rh * 0.42}
        x2={x + rw * 0.38} y2={y + rh * 0.42}
        stroke="#888" strokeWidth="0.5" strokeDasharray="3,2" />
      <line x1={x + rw * 0.38} y1={y + rh * 0.42}
        x2={x + rw * 0.38} y2={y + rh - 4}
        stroke="#888" strokeWidth="0.5" strokeDasharray="3,2" />
      {/* Bathtub for master */}
      {isMaster && rw > 50 && (
        <g>
          <rect x={x + rw * 0.05} y={y + rh * 0.55}
            width={rw * 0.35} height={rh * 0.3}
            fill="none" stroke="#555" strokeWidth="0.6" rx="2" />
          <ellipse cx={x + rw * 0.225} cy={y + rh * 0.7}
            rx={rw * 0.14} ry={rh * 0.1}
            fill="none" stroke="#888" strokeWidth="0.4" />
        </g>
      )}
    </g>
  )
}

function drawKitchenFixtures(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  const counterW = rw * 0.85
  const counterH = Math.min(18, rh * 0.2)

  return (
    <g>
      {/* Counter along top wall */}
      <rect x={x + (rw - counterW) / 2} y={y + 4}
        width={counterW} height={counterH}
        fill="none" stroke="#444" strokeWidth="0.7" />
      {/* Side counter (L-shape) */}
      <rect x={x + 4} y={y + 4 + counterH}
        width={Math.min(rw * 0.15, 16)} height={rh * 0.35}
        fill="none" stroke="#444" strokeWidth="0.6" />
      {/* Sink */}
      <rect x={x + rw * 0.28} y={y + 6}
        width={rw * 0.14} height={counterH - 4}
        fill="none" stroke="#666" strokeWidth="0.5" rx="1" />
      {/* Sink X mark */}
      <line x1={x + rw * 0.28} y1={y + 6}
        x2={x + rw * 0.42} y2={y + counterH + 2}
        stroke="#888" strokeWidth="0.3" />
      <line x1={x + rw * 0.42} y1={y + 6}
        x2={x + rw * 0.28} y2={y + counterH + 2}
        stroke="#888" strokeWidth="0.3" />
      {/* Stove burners */}
      {[0, 1, 2, 3].map(i => {
        const col = i % 2
        const row = Math.floor(i / 2)
        const bcx = x + rw * 0.62 + col * rw * 0.1
        const bcy = y + 4 + counterH / 2 + (row - 0.5) * (counterH * 0.35)
        return (
          <circle key={`burner-${i}`} cx={bcx} cy={bcy}
            r={Math.min(3.5, counterH * 0.15)}
            fill="none" stroke="#555" strokeWidth="0.5" />
        )
      })}
      {/* Fridge */}
      <rect x={x + rw - 16} y={y + counterH + 12}
        width={12} height={20}
        fill="none" stroke="#444" strokeWidth="0.6" />
      <text x={x + rw - 10} y={y + counterH + 24}
        textAnchor="middle" fill="#888" fontSize="5"
        fontFamily="Inter, sans-serif">REF</text>
    </g>
  )
}

function drawLivingFurniture(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  const sofaW = Math.min(rw * 0.55, 84)
  const sofaH = Math.min(rh * 0.16, 20)
  const sx = x + (rw - sofaW) / 2
  const sy = y + rh - sofaH - 12

  return (
    <g>
      {/* Sofa body */}
      <rect x={sx} y={sy} width={sofaW} height={sofaH}
        fill="none" stroke="#333" strokeWidth="0.8" rx="1" />
      {/* Sofa back */}
      <rect x={sx} y={sy + sofaH} width={sofaW} height={sofaH * 0.22}
        fill="none" stroke="#444" strokeWidth="0.5" />
      {/* Armrests */}
      <rect x={sx - sofaH * 0.3} y={sy} width={sofaH * 0.3} height={sofaH}
        fill="none" stroke="#555" strokeWidth="0.5" rx="1" />
      <rect x={sx + sofaW} y={sy} width={sofaH * 0.3} height={sofaH}
        fill="none" stroke="#555" strokeWidth="0.5" rx="1" />
      {/* Cushion divisions */}
      {[1, 2].map(i => (
        <line key={`cushion-${i}`}
          x1={sx + sofaW * i / 3} y1={sy + 2}
          x2={sx + sofaW * i / 3} y2={sy + sofaH - 2}
          stroke="#CCCCCC" strokeWidth="0.4" />
      ))}
      {/* Coffee table */}
      <rect x={sx + sofaW * 0.2} y={sy - 20}
        width={sofaW * 0.6} height={14}
        fill="none" stroke="#555" strokeWidth="0.5" />
    </g>
  )
}

function drawDiningFurniture(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  const tableW = Math.min(rw * 0.48, 54)
  const tableH = Math.min(rh * 0.36, 40)
  const tx = x + (rw - tableW) / 2
  const ty = y + (rh - tableH) / 2

  return (
    <g>
      {/* Table */}
      <rect x={tx} y={ty} width={tableW} height={tableH}
        fill="none" stroke="#333" strokeWidth="0.8" />
      {/* Chairs (top & bottom, 2 each) */}
      {[0, 1].map(i => {
        const cx = tx + tableW * (i + 1) / 3 - 5
        return (
          <g key={`chair-pair-${i}`}>
            <rect x={cx} y={ty - 8} width={10} height={6}
              fill="none" stroke="#888" strokeWidth="0.5" rx="1" />
            <rect x={cx} y={ty + tableH + 2} width={10} height={6}
              fill="none" stroke="#888" strokeWidth="0.5" rx="1" />
          </g>
        )
      })}
      {/* End chairs */}
      <rect x={tx - 8} y={ty + tableH / 2 - 5} width={6} height={10}
        fill="none" stroke="#888" strokeWidth="0.5" rx="1" />
      <rect x={tx + tableW + 2} y={ty + tableH / 2 - 5} width={6} height={10}
        fill="none" stroke="#888" strokeWidth="0.5" rx="1" />
    </g>
  )
}

function drawPoojaFixtures(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  // Simple shelf/mandir outline
  const shelfW = rw * 0.5
  const shelfH = rh * 0.3
  const sx = x + (rw - shelfW) / 2
  const sy = y + rh * 0.25
  return (
    <g>
      <rect x={sx} y={sy} width={shelfW} height={shelfH}
        fill="none" stroke="#555" strokeWidth="0.6" />
      {/* Small triangle top (temple shape) */}
      <polygon points={`${sx + shelfW / 2},${sy - shelfH * 0.3} ${sx},${sy} ${sx + shelfW},${sy}`}
        fill="none" stroke="#555" strokeWidth="0.5" />
    </g>
  )
}

function getFurniture(room, rx, ry) {
  const type = room.type
  if (type === 'master_bedroom') return drawBedFurniture(rx, ry, room.width, room.height, true)
  if (type === 'bedroom') return drawBedFurniture(rx, ry, room.width, room.height, false)
  if (type === 'bathroom' || type === 'toilet') return drawBathroomFixtures(rx, ry, room.width, room.height, false)
  if (type === 'master_bath') return drawBathroomFixtures(rx, ry, room.width, room.height, true)
  if (type === 'kitchen') return drawKitchenFixtures(rx, ry, room.width, room.height)
  if (type === 'living') return drawLivingFurniture(rx, ry, room.width, room.height)
  if (type === 'dining') return drawDiningFurniture(rx, ry, room.width, room.height)
  if (type === 'pooja') return drawPoojaFixtures(rx, ry, room.width, room.height)
  return null
}

// ─────────────────────────────────────────────────────────────
// Hatch patterns
// ─────────────────────────────────────────────────────────────
function getRoomHatch(room, rx, ry, rw, rh, clipId) {
  const type = room.type
  const lines = []

  if (type === 'bathroom' || type === 'master_bath' || type === 'toilet') {
    // Diagonal hatch lines (45 degrees)
    const spacing = 8
    const maxD = rw + rh
    for (let d = 0; d < maxD; d += spacing) {
      const x1 = rx + Math.min(d, rw)
      const y1 = ry + Math.max(0, d - rw)
      const x2 = rx + Math.max(0, d - rh)
      const y2 = ry + Math.min(d, rh)
      lines.push(
        <line key={`hatch-${d}`}
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="#CCCCCC" strokeWidth="0.4" />
      )
    }
  } else if (type === 'kitchen') {
    // Cross-hatch
    const spacing = 12
    for (let oy = 0; oy < rh; oy += spacing) {
      lines.push(
        <line key={`hh-${oy}`}
          x1={rx} y1={ry + oy} x2={rx + rw} y2={ry + oy}
          stroke="#DDDDDD" strokeWidth="0.3" />
      )
    }
    for (let ox = 0; ox < rw; ox += spacing) {
      lines.push(
        <line key={`hv-${ox}`}
          x1={rx + ox} y1={ry} x2={rx + ox} y2={ry + rh}
          stroke="#DDDDDD" strokeWidth="0.3" />
      )
    }
  } else if (type === 'corridor') {
    // Light horizontal lines
    const spacing = 14
    for (let oy = 0; oy < rh; oy += spacing) {
      lines.push(
        <line key={`ch-${oy}`}
          x1={rx} y1={ry + oy} x2={rx + rw} y2={ry + oy}
          stroke="#EEEEEE" strokeWidth="0.3" />
      )
    }
  }

  if (lines.length === 0) return null

  return (
    <g clipPath={`url(#${clipId})`}>
      {lines}
    </g>
  )
}


// ─────────────────────────────────────────────────────────────
// Main SVG component
// ─────────────────────────────────────────────────────────────
export default function FloorPlanSVG({ plan }) {
  if (!plan || !plan.rooms) return null

  const { plot, rooms, doors = [], windows = [] } = plan
  const uw = plot.usable_width
  const ul = plot.usable_length
  const pw = plot.width
  const pl = plot.length

  const sb = plot.setbacks || { front: 6.5, rear: 5, left: 3.5, right: 3.5 }

  // SVG dimensions with margin for dimension labels
  const svgW = (pw + 8) * SCALE + MARGIN * 2
  const svgH = (pl + 10) * SCALE + MARGIN * 2  // extra for road label + title

  // Coordinate transforms
  // floor plan y=0 at bottom (road), SVG y=0 at top
  function tx(roomX) { return MARGIN + (sb.left + roomX) * SCALE }
  function ty(roomY) { return MARGIN + (pl - sb.front - roomY) * SCALE }

  const plotX = MARGIN
  const plotY = MARGIN
  const plotW = pw * SCALE
  const plotH = pl * SCALE

  const bhkCount = rooms.filter(r => r.type === 'master_bedroom' || r.type === 'bedroom').length

  return (
    <svg
      width={svgW}
      height={svgH}
      viewBox={`0 0 ${svgW} ${svgH}`}
      xmlns="http://www.w3.org/2000/svg"
      style={{ background: '#FFFFFF' }}
    >
      {/* ── Clip path definitions for hatch ─────────── */}
      <defs>
        {rooms.map((room, idx) => {
          const rx = tx(room.x)
          const ry = ty(room.y + room.height)
          const rw = room.width * SCALE
          const rh = room.height * SCALE
          return (
            <clipPath key={`clip-${idx}`} id={`clip-room-${idx}`}>
              <rect x={rx} y={ry} width={rw} height={rh} />
            </clipPath>
          )
        })}
      </defs>

      {/* ── Plot boundary (thick outer line) ─────────── */}
      <rect
        x={plotX} y={plotY}
        width={plotW} height={plotH}
        fill="none" stroke="#000" strokeWidth="3"
      />

      {/* ── Usable area (thin dashed) ────────────────── */}
      <rect
        x={tx(0)} y={ty(ul)}
        width={uw * SCALE} height={ul * SCALE}
        fill="none" stroke="#AAAAAA" strokeWidth="0.5"
        strokeDasharray="6,4"
      />

      {/* ── Rooms ────────────────────────────────────── */}
      {rooms.map((room, idx) => {
        const rx = tx(room.x)
        const ry = ty(room.y + room.height)
        const rw = room.width * SCALE
        const rh = room.height * SCALE

        const fontSize = Math.max(8, Math.min(14, Math.min(rw, rh) * 0.09))
        const clipId = `clip-room-${idx}`

        // Abbreviations for small rooms
        let displayLabel = room.label.toUpperCase()
        if (rw < 60 || rh < 60) {
          const abbrevMap = {
            'MASTER BEDROOM': 'M.BED',
            'MASTER BATH': 'M.BATH',
            'BEDROOM 2': 'BED 2',
            'BEDROOM 3': 'BED 3',
            'BEDROOM 4': 'BED 4',
            'BATHROOM 2': 'BATH 2',
            'BATHROOM 3': 'BATH 3',
            'BATHROOM 4': 'BATH 4',
            'COMMON BATH': 'C.BATH',
            'LIVING ROOM': 'LIVING',
            'DINING ROOM': 'DINING',
            'POOJA ROOM': 'POOJA',
            'STUDY ROOM': 'STUDY',
            'STORE ROOM': 'STORE',
          }
          displayLabel = abbrevMap[displayLabel] || displayLabel
        }

        return (
          <g key={room.id || idx}>
            {/* Room fill — pure white */}
            <rect
              x={rx} y={ry} width={rw} height={rh}
              fill="#FFFFFF" stroke="#000" strokeWidth="1.5"
            />

            {/* Hatch patterns (clipped to room bounds) */}
            {getRoomHatch(room, rx, ry, rw, rh, clipId)}

            {/* Furniture symbols */}
            {getFurniture(room, rx, ry)}

            {/* Room label */}
            <text
              x={rx + rw / 2}
              y={ry + rh / 2 - fontSize * 0.2}
              textAnchor="middle"
              dominantBaseline="auto"
              fill="#111"
              fontSize={fontSize}
              fontWeight="700"
              fontFamily="Inter, sans-serif"
              letterSpacing="0.5"
            >
              {displayLabel}
            </text>
            {/* Area text */}
            <text
              x={rx + rw / 2}
              y={ry + rh / 2 + fontSize * 0.9}
              textAnchor="middle"
              dominantBaseline="auto"
              fill="#555"
              fontSize={fontSize * 0.65}
              fontWeight="400"
              fontFamily="Inter, sans-serif"
            >
              {room.area?.toFixed(0) || (room.width * room.height).toFixed(0)} sq.ft
            </text>
            {/* Dimensions */}
            <text
              x={rx + rw / 2}
              y={ry + rh / 2 + fontSize * 1.7}
              textAnchor="middle"
              dominantBaseline="auto"
              fill="#888"
              fontSize={fontSize * 0.55}
              fontWeight="400"
              fontFamily="Inter, sans-serif"
            >
              ({room.width.toFixed(1)}' × {room.height.toFixed(1)}')
            </text>
          </g>
        )
      })}

      {/* ── Doors ────────────────────────────────────── */}
      {doors.map((door) => {
        const dx = tx(door.x)
        const dy = ty(door.y)
        const dw = door.width * SCALE
        const isMain = door.type === 'main'

        // Door opening (white gap in wall)
        let gapProps = {}
        if (door.wall === 'south' || door.wall === 'north') {
          gapProps = { x: dx - 1, y: dy - 3, width: dw + 2, height: 6 }
        } else {
          gapProps = { x: dx - 3, y: dy - dw - 1, width: 6, height: dw + 2 }
        }

        // Arc sweep path
        let arcPath = ''
        if (door.wall === 'south' || door.wall === 'north') {
          const dir = door.wall === 'south' ? -1 : 1
          arcPath = `M ${dx} ${dy} L ${dx + dw} ${dy} M ${dx} ${dy} A ${dw} ${dw} 0 0 ${dir === -1 ? 1 : 0} ${dx + dw} ${dy + dir * dw}`
        } else {
          const dir = door.wall === 'east' ? 1 : -1
          arcPath = `M ${dx} ${dy} L ${dx} ${dy - dw} M ${dx} ${dy} A ${dw} ${dw} 0 0 ${dir === 1 ? 0 : 1} ${dx + dir * dw} ${dy - dw}`
        }

        return (
          <g key={door.id}>
            <rect {...gapProps} fill="#FFFFFF" stroke="none" />
            <path d={arcPath} fill="none"
              stroke={isMain ? '#000' : '#555'}
              strokeWidth={isMain ? 1.2 : 0.7}
              strokeDasharray={isMain ? 'none' : '3,2'}
            />
          </g>
        )
      })}

      {/* ── Windows ──────────────────────────────────── */}
      {windows.map((win) => {
        const wx = tx(win.x)
        const wy = ty(win.y)
        const ww = win.width * SCALE

        if (win.wall === 'south' || win.wall === 'north') {
          return (
            <g key={win.id}>
              <rect x={wx - 1} y={wy - 3} width={ww + 2} height={6} fill="#FFFFFF" stroke="none" />
              {[-2, 0, 2].map((offset, i) => (
                <line key={i}
                  x1={wx} y1={wy + offset}
                  x2={wx + ww} y2={wy + offset}
                  stroke="#333" strokeWidth={i === 1 ? 1 : 0.5}
                />
              ))}
            </g>
          )
        } else {
          return (
            <g key={win.id}>
              <rect x={wx - 3} y={wy - ww - 1} width={6} height={ww + 2} fill="#FFFFFF" stroke="none" />
              {[-2, 0, 2].map((offset, i) => (
                <line key={i}
                  x1={wx + offset} y1={wy}
                  x2={wx + offset} y2={wy - ww}
                  stroke="#333" strokeWidth={i === 1 ? 1 : 0.5}
                />
              ))}
            </g>
          )
        }
      })}

      {/* ── Exterior Dimension Lines ─────────────────── */}
      {/* Bottom (width) */}
      <g>
        <line x1={plotX} y1={plotY + plotH + 18}
          x2={plotX + plotW} y2={plotY + plotH + 18}
          stroke="#333" strokeWidth="0.8" />
        <line x1={plotX} y1={plotY + plotH + 12}
          x2={plotX} y2={plotY + plotH + 24}
          stroke="#333" strokeWidth="0.8" />
        <line x1={plotX + plotW} y1={plotY + plotH + 12}
          x2={plotX + plotW} y2={plotY + plotH + 24}
          stroke="#333" strokeWidth="0.8" />
        <text x={plotX + plotW / 2} y={plotY + plotH + 34}
          textAnchor="middle" fill="#000" fontSize="11"
          fontWeight="600" fontFamily="Inter, sans-serif">
          {pw.toFixed(0)} ft
        </text>
      </g>

      {/* Right (length) */}
      <g>
        <line x1={plotX + plotW + 18} y1={plotY}
          x2={plotX + plotW + 18} y2={plotY + plotH}
          stroke="#333" strokeWidth="0.8" />
        <line x1={plotX + plotW + 12} y1={plotY}
          x2={plotX + plotW + 24} y2={plotY}
          stroke="#333" strokeWidth="0.8" />
        <line x1={plotX + plotW + 12} y1={plotY + plotH}
          x2={plotX + plotW + 24} y2={plotY + plotH}
          stroke="#333" strokeWidth="0.8" />
        <text x={plotX + plotW + 30} y={plotY + plotH / 2}
          textAnchor="start" dominantBaseline="middle" fill="#000" fontSize="11"
          fontWeight="600" fontFamily="Inter, sans-serif"
          transform={`rotate(90, ${plotX + plotW + 30}, ${plotY + plotH / 2})`}>
          {pl.toFixed(0)} ft
        </text>
      </g>

      {/* ── North Arrow ──────────────────────────────── */}
      <g transform={`translate(${svgW - 48}, 30)`}>
        <polygon points="0,-20 -6,6 6,6" fill="#000" stroke="#000" strokeWidth="0.5" />
        <polygon points="0,-20 0,6 6,6" fill="#555" stroke="none" />
        <text x="0" y="-26" textAnchor="middle" fill="#000"
          fontSize="12" fontWeight="700" fontFamily="Inter, sans-serif">N</text>
      </g>

      {/* ── Scale Bar ────────────────────────────────── */}
      <g transform={`translate(${plotX + 8}, ${svgH - 36})`}>
        <line x1="0" y1="0" x2={10 * SCALE} y2="0" stroke="#000" strokeWidth="1.5" />
        <line x1="0" y1="-5" x2="0" y2="5" stroke="#000" strokeWidth="1.5" />
        <line x1={5 * SCALE} y1="-3" x2={5 * SCALE} y2="3" stroke="#000" strokeWidth="1" />
        <line x1={10 * SCALE} y1="-5" x2={10 * SCALE} y2="5" stroke="#000" strokeWidth="1.5" />
        {/* Alternating fill blocks */}
        <rect x="0" y="-2" width={5 * SCALE} height="4" fill="#000" stroke="none" />
        <rect x={5 * SCALE} y="-2" width={5 * SCALE} height="4" fill="none" stroke="#000" strokeWidth="0.5" />
        <text x="0" y="14" textAnchor="start" fill="#000"
          fontSize="8" fontFamily="Inter, sans-serif">0</text>
        <text x={5 * SCALE} y="14" textAnchor="middle" fill="#000"
          fontSize="8" fontFamily="Inter, sans-serif">5</text>
        <text x={10 * SCALE} y="14" textAnchor="end" fill="#000"
          fontSize="8" fontFamily="Inter, sans-serif">10 ft</text>
      </g>

      {/* ── Road label ───────────────────────────────── */}
      <text
        x={plotX + plotW / 2} y={svgH - 8}
        textAnchor="middle" fill="#555"
        fontSize="10" fontWeight="600"
        fontFamily="Inter, sans-serif"
        letterSpacing="3"
      >
        ── ROAD ({plot.road_side?.toUpperCase() || 'SOUTH'}) ──
      </text>

      {/* ── Title block ──────────────────────────────── */}
      <text x={plotX} y={plotY - 24}
        fill="#000" fontSize="11" fontWeight="700"
        fontFamily="Inter, sans-serif" letterSpacing="1.5">
        RESIDENTIAL FLOOR PLAN
      </text>
      <text x={plotX} y={plotY - 10}
        fill="#555" fontSize="8" fontWeight="500"
        fontFamily="Inter, sans-serif" letterSpacing="0.5">
        {pw.toFixed(0)}' × {pl.toFixed(0)}' | {bhkCount} BHK | Vastu Score: {plan.vastu_score?.toFixed(0) || '—'}/100 | Road: {plot.road_side?.toUpperCase()} | Scale 1:100
      </text>
    </svg>
  )
}
