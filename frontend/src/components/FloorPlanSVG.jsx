/**
 * FloorPlanSVG — Professional CAD-style architectural floor plan renderer.
 * Matches clean black-and-white architectural drawing style with
 * furniture symbols, door arcs, dimension lines, and room labels.
 */

const SCALE = 12 // pixels per foot

// Furniture drawing functions return SVG elements
function drawBedFurniture(x, y, w, h, isMain) {
  const bedW = Math.min(w * 0.6, isMain ? 6 : 5) * SCALE
  const bedH = Math.min(h * 0.5, isMain ? 6.5 : 5.5) * SCALE
  const bx = x + (w * SCALE - bedW) / 2
  const by = y + (h * SCALE - bedH) / 2

  const pillowH = bedH * 0.15
  return (
    <g>
      {/* Bed frame */}
      <rect x={bx} y={by} width={bedW} height={bedH}
        fill="none" stroke="#555" strokeWidth="0.8" />
      {/* Mattress */}
      <rect x={bx + 3} y={by + pillowH + 3} width={bedW - 6} height={bedH - pillowH - 6}
        fill="none" stroke="#888" strokeWidth="0.5" />
      {/* Pillow(s) */}
      {isMain ? (
        <>
          <rect x={bx + 4} y={by + 3} width={bedW / 2 - 6} height={pillowH - 2}
            fill="none" stroke="#888" strokeWidth="0.5" rx="2" />
          <rect x={bx + bedW / 2 + 2} y={by + 3} width={bedW / 2 - 6} height={pillowH - 2}
            fill="none" stroke="#888" strokeWidth="0.5" rx="2" />
        </>
      ) : (
        <rect x={bx + 4} y={by + 3} width={bedW - 8} height={pillowH - 2}
          fill="none" stroke="#888" strokeWidth="0.5" rx="2" />
      )}
      {/* Nightstand (master only) */}
      {isMain && (
        <rect x={bx - 14} y={by + bedH - 18} width={12} height={16}
          fill="none" stroke="#888" strokeWidth="0.5" />
      )}
    </g>
  )
}

function drawBathroomFixtures(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  // WC
  const wcW = Math.min(16, rw * 0.3)
  const wcH = Math.min(20, rh * 0.35)
  // Sink
  const sinkR = Math.min(8, rw * 0.12)

  return (
    <g>
      {/* WC - toilet bowl */}
      <ellipse cx={x + rw - wcW / 2 - 6} cy={y + rh - wcH / 2 - 6}
        rx={wcW / 2} ry={wcH / 2}
        fill="none" stroke="#666" strokeWidth="0.7" />
      <rect x={x + rw - wcW - 6} y={y + rh - wcH - 6}
        width={wcW} height={wcH * 0.4}
        fill="none" stroke="#666" strokeWidth="0.7" rx="2" />
      {/* Sink */}
      <circle cx={x + sinkR + 8} cy={y + sinkR + 8}
        r={sinkR} fill="none" stroke="#666" strokeWidth="0.7" />
      {/* Shower area (dashed corner) */}
      <line x1={x + 4} y1={y + rh * 0.45}
        x2={x + rw * 0.4} y2={y + rh * 0.45}
        stroke="#888" strokeWidth="0.5" strokeDasharray="3,2" />
      <line x1={x + rw * 0.4} y1={y + rh * 0.45}
        x2={x + rw * 0.4} y2={y + rh - 4}
        stroke="#888" strokeWidth="0.5" strokeDasharray="3,2" />
    </g>
  )
}

function drawKitchenFixtures(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  const counterW = rw * 0.85
  const counterH = Math.min(16, rh * 0.2)

  return (
    <g>
      {/* Counter along top wall */}
      <rect x={x + (rw - counterW) / 2} y={y + 4}
        width={counterW} height={counterH}
        fill="none" stroke="#666" strokeWidth="0.7" />
      {/* Sink in counter */}
      <rect x={x + rw * 0.3} y={y + 6}
        width={rw * 0.15} height={counterH - 4}
        fill="none" stroke="#888" strokeWidth="0.5" rx="2" />
      {/* Stove burners */}
      <circle cx={x + rw * 0.65} cy={y + 4 + counterH / 2}
        r={4} fill="none" stroke="#888" strokeWidth="0.5" />
      <circle cx={x + rw * 0.75} cy={y + 4 + counterH / 2}
        r={4} fill="none" stroke="#888" strokeWidth="0.5" />
      {/* Fridge */}
      <rect x={x + rw - 18} y={y + counterH + 10}
        width={14} height={20}
        fill="none" stroke="#666" strokeWidth="0.7" />
    </g>
  )
}

function drawLivingFurniture(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  const sofaW = Math.min(rw * 0.6, 80)
  const sofaH = Math.min(rh * 0.2, 20)
  const sx = x + (rw - sofaW) / 2
  const sy = y + rh - sofaH - 10

  return (
    <g>
      {/* Sofa */}
      <rect x={sx} y={sy} width={sofaW} height={sofaH}
        fill="none" stroke="#555" strokeWidth="0.8" rx="2" />
      {/* Sofa back */}
      <rect x={sx} y={sy + sofaH - 4} width={sofaW} height={4}
        fill="none" stroke="#555" strokeWidth="0.5" rx="1" />
      {/* Armrests */}
      <rect x={sx} y={sy} width={sofaH * 0.4} height={sofaH}
        fill="none" stroke="#777" strokeWidth="0.5" rx="1" />
      <rect x={sx + sofaW - sofaH * 0.4} y={sy} width={sofaH * 0.4} height={sofaH}
        fill="none" stroke="#777" strokeWidth="0.5" rx="1" />
      {/* Coffee table */}
      <rect x={sx + sofaW * 0.25} y={sy - 18}
        width={sofaW * 0.5} height={12}
        fill="none" stroke="#888" strokeWidth="0.5" />
    </g>
  )
}

function drawDiningFurniture(x, y, w, h) {
  const rw = w * SCALE
  const rh = h * SCALE
  const tableW = Math.min(rw * 0.5, 50)
  const tableH = Math.min(rh * 0.4, 36)
  const tx = x + (rw - tableW) / 2
  const ty = y + (rh - tableH) / 2

  return (
    <g>
      {/* Dining table */}
      <rect x={tx} y={ty} width={tableW} height={tableH}
        fill="none" stroke="#555" strokeWidth="0.8" />
      {/* Chairs (top & bottom) */}
      {[0, 1, 2].map(i => {
        const cx = tx + tableW * (i + 1) / 4 - 5
        return (
          <g key={`chair-${i}`}>
            <rect x={cx} y={ty - 9} width={10} height={7}
              fill="none" stroke="#888" strokeWidth="0.5" rx="1" />
            <rect x={cx} y={ty + tableH + 2} width={10} height={7}
              fill="none" stroke="#888" strokeWidth="0.5" rx="1" />
          </g>
        )
      })}
    </g>
  )
}

function getFurniture(room, rx, ry) {
  const type = room.type
  if (type === 'master_bedroom') return drawBedFurniture(rx, ry, room.width, room.height, true)
  if (type === 'bedroom') return drawBedFurniture(rx, ry, room.width, room.height, false)
  if (type === 'bathroom' || type === 'toilet') return drawBathroomFixtures(rx, ry, room.width, room.height)
  if (type === 'kitchen') return drawKitchenFixtures(rx, ry, room.width, room.height)
  if (type === 'living') return drawLivingFurniture(rx, ry, room.width, room.height)
  if (type === 'dining') return drawDiningFurniture(rx, ry, room.width, room.height)
  return null
}

export default function FloorPlanSVG({ plan }) {
  if (!plan || !plan.rooms) return null

  const { plot, rooms, doors = [], windows = [] } = plan
  const uw = plot.usable_width
  const ul = plot.usable_length
  const pw = plot.width
  const pl = plot.length

  const sb = plot.setbacks || { front: 6.5, rear: 5, left: 3.5, right: 3.5 }

  // SVG dimensions with padding for dimension labels
  const pad = 5
  const svgW = (pw + pad * 2) * SCALE
  const svgH = (pl + pad * 2 + 3) * SCALE  // extra space for road label

  // Coordinate transforms — floor plan y=0 at bottom, SVG y=0 at top.
  function tx(x) { return (pad + sb.left + x) * SCALE }
  function ty(y) { return (pad + pl - sb.front - y) * SCALE }

  const plotX = pad * SCALE
  const plotY = pad * SCALE
  const plotW = pw * SCALE
  const plotH = pl * SCALE

  return (
    <svg
      width={svgW}
      height={svgH}
      viewBox={`0 0 ${svgW} ${svgH}`}
      xmlns="http://www.w3.org/2000/svg"
      style={{ background: '#FFFFFF' }}
    >
      {/* ── Plot boundary (thick outer line) ─────────────── */}
      <rect
        x={plotX} y={plotY}
        width={plotW} height={plotH}
        fill="none" stroke="#222" strokeWidth="2.5"
      />

      {/* ── Usable area (thin dashed) ────────────────────── */}
      <rect
        x={tx(0)} y={ty(ul)}
        width={uw * SCALE} height={ul * SCALE}
        fill="#FCFCFC" stroke="#bbb" strokeWidth="0.5"
        strokeDasharray="6,4"
      />

      {/* ── Rooms ────────────────────────────────────────── */}
      {rooms.map((room) => {
        const rx = tx(room.x)
        const ry = ty(room.y + room.height)
        const rw = room.width * SCALE
        const rh = room.height * SCALE

        // Light fill for open/corridor, very subtle for others
        const isOpenType = room.type === 'corridor' || room.type === 'balcony'
        const fill = isOpenType ? '#F5F5F5' : '#FAFAFA'

        const fontSize = Math.max(9, Math.min(13, Math.min(rw, rh) * 0.09))

        return (
          <g key={room.id}>
            {/* Room outline */}
            <rect
              x={rx} y={ry} width={rw} height={rh}
              fill={fill} stroke="#333" strokeWidth="1.5"
            />

            {/* Furniture symbols */}
            {getFurniture(room, rx, ry)}

            {/* Room label */}
            <text
              x={rx + rw / 2}
              y={ry + rh / 2 - 3}
              textAnchor="middle"
              dominantBaseline="auto"
              fill="#222"
              fontSize={fontSize}
              fontWeight="600"
              fontFamily="Inter, sans-serif"
              letterSpacing="0.5"
            >
              {room.label.toUpperCase()}
            </text>
            {/* Area */}
            <text
              x={rx + rw / 2}
              y={ry + rh / 2 + fontSize + 2}
              textAnchor="middle"
              dominantBaseline="auto"
              fill="#666"
              fontSize={fontSize * 0.7}
              fontWeight="400"
              fontFamily="Inter, sans-serif"
            >
              {room.area?.toFixed(0) || (room.width * room.height).toFixed(0)} sq.ft
            </text>
          </g>
        )
      })}

      {/* ── Doors ────────────────────────────────────────── */}
      {doors.map((door) => {
        const dx = tx(door.x)
        const dy = ty(door.y)
        const dw = door.width * SCALE
        const isMain = door.type === 'main'

        // Door opening (white gap in wall)
        const gapLen = dw
        let gapProps = {}
        if (door.wall === 'south' || door.wall === 'north') {
          gapProps = { x: dx - 1, y: dy - 2, width: gapLen + 2, height: 4 }
        } else {
          gapProps = { x: dx - 2, y: dy - gapLen - 1, width: 4, height: gapLen + 2 }
        }

        // Arc sweep
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
            {/* White gap to "erase" wall line */}
            <rect {...gapProps} fill="white" stroke="none" />
            {/* Door arc */}
            <path d={arcPath} fill="none"
              stroke={isMain ? '#222' : '#555'}
              strokeWidth={isMain ? 1.2 : 0.8}
              strokeDasharray={isMain ? 'none' : '3,2'}
            />
          </g>
        )
      })}

      {/* ── Windows ──────────────────────────────────────── */}
      {windows.map((win) => {
        const wx = tx(win.x)
        const wy = ty(win.y)
        const ww = win.width * SCALE

        // Three parallel lines (architectural window symbol)
        if (win.wall === 'south' || win.wall === 'north') {
          return (
            <g key={win.id}>
              <rect x={wx - 1} y={wy - 3} width={ww + 2} height={6} fill="white" stroke="none" />
              {[-2, 0, 2].map((offset, i) => (
                <line key={i}
                  x1={wx} y1={wy + offset}
                  x2={wx + ww} y2={wy + offset}
                  stroke="#444" strokeWidth={i === 1 ? 1 : 0.6}
                />
              ))}
            </g>
          )
        } else {
          return (
            <g key={win.id}>
              <rect x={wx - 3} y={wy - ww - 1} width={6} height={ww + 2} fill="white" stroke="none" />
              {[-2, 0, 2].map((offset, i) => (
                <line key={i}
                  x1={wx + offset} y1={wy}
                  x2={wx + offset} y2={wy - ww}
                  stroke="#444" strokeWidth={i === 1 ? 1 : 0.6}
                />
              ))}
            </g>
          )
        }
      })}

      {/* ── Exterior Dimension Lines ─────────────────────── */}
      {/* Bottom (width) */}
      <g>
        <line x1={plotX} y1={plotY + plotH + 16}
          x2={plotX + plotW} y2={plotY + plotH + 16}
          stroke="#444" strokeWidth="0.8" />
        {/* Ticks */}
        <line x1={plotX} y1={plotY + plotH + 10}
          x2={plotX} y2={plotY + plotH + 22}
          stroke="#444" strokeWidth="0.8" />
        <line x1={plotX + plotW} y1={plotY + plotH + 10}
          x2={plotX + plotW} y2={plotY + plotH + 22}
          stroke="#444" strokeWidth="0.8" />
        <text x={plotX + plotW / 2} y={plotY + plotH + 30}
          textAnchor="middle" fill="#333" fontSize="11"
          fontWeight="500" fontFamily="Inter, sans-serif">
          {pw.toFixed(0)}'
        </text>
      </g>

      {/* Right (length) */}
      <g>
        <line x1={plotX + plotW + 16} y1={plotY}
          x2={plotX + plotW + 16} y2={plotY + plotH}
          stroke="#444" strokeWidth="0.8" />
        <line x1={plotX + plotW + 10} y1={plotY}
          x2={plotX + plotW + 22} y2={plotY}
          stroke="#444" strokeWidth="0.8" />
        <line x1={plotX + plotW + 10} y1={plotY + plotH}
          x2={plotX + plotW + 22} y2={plotY + plotH}
          stroke="#444" strokeWidth="0.8" />
        <text x={plotX + plotW + 28} y={plotY + plotH / 2}
          textAnchor="start" dominantBaseline="middle" fill="#333" fontSize="11"
          fontWeight="500" fontFamily="Inter, sans-serif"
          transform={`rotate(90, ${plotX + plotW + 28}, ${plotY + plotH / 2})`}>
          {pl.toFixed(0)}'
        </text>
      </g>

      {/* ── North Arrow ──────────────────────────────────── */}
      <g transform={`translate(${svgW - 55}, 40)`}>
        <polygon points="0,-22 -7,6 7,6" fill="#333" stroke="#333" strokeWidth="0.5" />
        <polygon points="0,-22 0,6 7,6" fill="#666" stroke="none" />
        <text x="0" y="-28" textAnchor="middle" fill="#333"
          fontSize="13" fontWeight="700" fontFamily="Inter, sans-serif">N</text>
      </g>

      {/* ── Scale bar ────────────────────────────────────── */}
      <g transform={`translate(${plotX + 10}, ${svgH - 24})`}>
        <line x1="0" y1="0" x2={10 * SCALE} y2="0" stroke="#333" strokeWidth="1.5" />
        <line x1="0" y1="-5" x2="0" y2="5" stroke="#333" strokeWidth="1.5" />
        <line x1={10 * SCALE} y1="-5" x2={10 * SCALE} y2="5" stroke="#333" strokeWidth="1.5" />
        <line x1={5 * SCALE} y1="-3" x2={5 * SCALE} y2="3" stroke="#333" strokeWidth="1" />
        <text x={5 * SCALE} y="-10" textAnchor="middle" fill="#333"
          fontSize="9" fontFamily="Inter, sans-serif">10 ft</text>
      </g>

      {/* ── Road label ───────────────────────────────────── */}
      <text
        x={svgW / 2} y={svgH - 6}
        textAnchor="middle" fill="#777"
        fontSize="10" fontWeight="500"
        fontFamily="Inter, sans-serif"
        letterSpacing="3"
      >
        ROAD ({plot.road_side?.toUpperCase() || 'SOUTH'})
      </text>

      {/* ── Title block ──────────────────────────────────── */}
      <text x={plotX} y={plotY - 16}
        fill="#333" fontSize="9" fontWeight="600"
        fontFamily="Inter, sans-serif" letterSpacing="1">
        RESIDENTIAL FLOOR PLAN — {pw.toFixed(0)}' × {pl.toFixed(0)}' |{' '}
        {rooms.filter(r => r.type === 'master_bedroom' || r.type === 'bedroom').length} BHK | VASTU COMPLIANT
      </text>
    </svg>
  )
}
