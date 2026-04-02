/**
 * FloorPlanSVG — clean architectural floor plan renderer.
 * Uses presentation-friendly room tones, selective symbols, readable labels,
 * door/window detailing, north arrow, scale bar, and title block.
 */

const SCALE = 12  // pixels per foot
const MARGIN = 50 // px margin for dimension lines
const FONT_FAMILY = 'Inter, "Segoe UI", sans-serif'

const ROOM_FILLS = {
  living: '#E8EFE6',
  dining: '#F3E9D8',
  kitchen: '#F4E4E4',
  master_bedroom: '#DCE8F1',
  bedroom: '#DCE8F1',
  study: '#E8E2EF',
  bathroom: '#E6F1F2',
  toilet: '#E6F1F2',
  master_bath: '#E6F1F2',
  corridor: '#EFEFEF',
  pooja: '#F7F0DA',
  store: '#ECE8E2',
  balcony: '#E6EEE4',
  utility: '#ECEAEC',
  foyer: '#F1F1F1',
  garage: '#E7EBEE',
}

const COMPACT_LABELS = {
  'MASTER BEDROOM': 'MASTER BED',
  'MASTER BATH': 'M.BATH',
  'MASTER TOILET': 'M.TOILET',
  'BEDROOM 2': 'BED 2',
  'BEDROOM 3': 'BED 3',
  'BEDROOM 4': 'BED 4',
  'COMMON BATH': 'C.BATH',
  'BATHROOM': 'BATH',
  'BATHROOM 2': 'BATH 2',
  'BATHROOM 3': 'BATH 3',
  'BATHROOM 4': 'BATH 4',
  'LIVING ROOM': 'LIVING',
  'DINING ROOM': 'DINING',
  'POOJA ROOM': 'POOJA',
  'STUDY ROOM': 'STUDY',
  'STORE ROOM': 'STORE',
  'UTILITY ROOM': 'UTILITY',
}

function getRoomFill(roomType) {
  return ROOM_FILLS[roomType] || '#EFF2F5'
}

function getCompactLabel(label) {
  return COMPACT_LABELS[label] || label
}

function shouldRenderFurniture(room) {
  const area = Number(room.area || room.width * room.height)
  if (room.width < 7 || room.height < 6 || area < 55) return false
  return !['corridor', 'foyer', 'balcony', 'store', 'utility'].includes(room.type)
}

function getRoomTextConfig(room, rw, rh) {
  const rawLabel = String(room.label || room.type || 'Room').toUpperCase().trim()
  const isTiny = rw < 56 || rh < 40
  const isCompact = rw < 84 || rh < 60

  const preferredLabel = isCompact ? getCompactLabel(rawLabel) : rawLabel
  const words = preferredLabel.split(' ').filter(Boolean)
  let labelLines = [preferredLabel]

  if (!isTiny && words.length >= 3) {
    const splitAt = Math.ceil(words.length / 2)
    labelLines = [
      words.slice(0, splitAt).join(' '),
      words.slice(splitAt).join(' '),
    ]
  }

  const labelSize = Math.max(7.5, Math.min(13.5, Math.min(rw, rh) * (isTiny ? 0.16 : 0.1)))
  const areaFt = Number(room.area || room.width * room.height)
  const showArea = !isTiny && rw >= 84 && rh >= 58 && areaFt >= 30
  const showDims = !isCompact && rw >= 98 && rh >= 74 && room.width >= 7 && room.height >= 7

  const lines = [
    ...labelLines.map((text) => ({
      text,
      size: labelSize,
      weight: 700,
      color: '#0F172A',
      tracking: 0.35,
    })),
  ]

  if (showArea) {
    lines.push({
      text: `${Math.round(areaFt)} sq.ft`,
      size: labelSize * 0.64,
      weight: 500,
      color: '#475569',
      tracking: 0,
    })
  }

  if (showDims) {
    lines.push({
      text: `(${room.width.toFixed(1)}' x ${room.height.toFixed(1)}')`,
      size: labelSize * 0.56,
      weight: 500,
      color: '#64748B',
      tracking: 0,
    })
  }

  const totalHeight = lines.reduce((sum, line) => sum + line.size * 0.95, 0) + Math.max(0, lines.length - 1) * 2
  const maxTextWidth = lines.reduce((max, line) => Math.max(max, line.text.length * line.size * 0.56), 0)

  return {
    lines,
    totalHeight,
    showBackground: lines.length > 1 && rw > 62 && rh > 46,
    bgWidth: Math.max(24, Math.min(rw - 8, maxTextWidth + 12)),
    bgHeight: Math.max(16, Math.min(rh - 8, totalHeight + 8)),
  }
}

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
        fontFamily={FONT_FAMILY}>REF</text>
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
  if (!shouldRenderFurniture(room)) return null

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
    const spacing = 11
    const maxD = rw + rh
    for (let d = 0; d < maxD; d += spacing) {
      const x1 = rx + Math.min(d, rw)
      const y1 = ry + Math.max(0, d - rw)
      const x2 = rx + Math.max(0, d - rh)
      const y2 = ry + Math.min(d, rh)
      lines.push(
        <line key={`hatch-${d}`}
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="#D1D9E0" strokeWidth="0.35" />
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
  function py(pointY) { return MARGIN + (pl - sb.front - pointY) * SCALE }

  function getSvgPolygon(rawPolygon) {
    if (!Array.isArray(rawPolygon) || rawPolygon.length < 3) return null

    const points = rawPolygon
      .map((p) => {
        if (!p || typeof p !== 'object') return null
        const x = Number(p.x)
        const y = Number(p.y)
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null
        return { x: tx(x), y: py(y) }
      })
      .filter(Boolean)

    return points.length >= 3 ? points : null
  }

  function polygonBounds(points) {
    const xs = points.map((p) => p.x)
    const ys = points.map((p) => p.y)
    const minX = Math.min(...xs)
    const minY = Math.min(...ys)
    const maxX = Math.max(...xs)
    const maxY = Math.max(...ys)
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
  }

  function pointsToString(points) {
    return points.map((p) => `${p.x},${p.y}`).join(' ')
  }

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
      shapeRendering="geometricPrecision"
      textRendering="geometricPrecision"
      style={{ background: '#FFFFFF' }}
    >
      {/* ── Clip path definitions for hatch ─────────── */}
      <defs>
        {rooms.map((room, idx) => {
          const roomPoly = getSvgPolygon(room.polygon)
          const isPoly = roomPoly && roomPoly.length >= 3
          const rx = tx(room.x)
          const ry = ty(room.y + room.height)
          const rw = room.width * SCALE
          const rh = room.height * SCALE
          return (
            <clipPath key={`clip-${idx}`} id={`clip-room-${idx}`}>
              {isPoly ? (
                <polygon points={pointsToString(roomPoly)} />
              ) : (
                <rect x={rx} y={ry} width={rw} height={rh} />
              )}
            </clipPath>
          )
        })}
      </defs>

      {/* ── Plot boundary (thick outer line) ─────────── */}
      {(() => {
        const plotPoly = getSvgPolygon(plot.boundary)
        if (plotPoly && plotPoly.length >= 3) {
          return (
            <polygon
              points={pointsToString(plotPoly)}
              fill="none"
              stroke="#000"
              strokeWidth="3"
            />
          )
        }

        return (
          <rect
            x={plotX} y={plotY}
            width={plotW} height={plotH}
            fill="none" stroke="#000" strokeWidth="3"
          />
        )
      })()}

      {/* ── Usable area (thin dashed) ────────────────── */}
      <rect
        x={tx(0)} y={ty(ul)}
        width={uw * SCALE} height={ul * SCALE}
        fill="none" stroke="#AAAAAA" strokeWidth="0.5"
        strokeDasharray="6,4"
      />

      {/* ── Rooms ────────────────────────────────────── */}
      {rooms.map((room, idx) => {
        const roomPoly = getSvgPolygon(room.polygon)
        const isPoly = roomPoly && roomPoly.length >= 3

        let rx = tx(room.x)
        let ry = ty(room.y + room.height)
        let rw = room.width * SCALE
        let rh = room.height * SCALE

        if (isPoly) {
          const b = polygonBounds(roomPoly)
          rx = b.x
          ry = b.y
          rw = b.w
          rh = b.h
        }

        const textConfig = getRoomTextConfig(room, rw, rh)
        const clipId = `clip-room-${idx}`

        let lineCursorY = ry + rh / 2 - textConfig.totalHeight / 2
        const renderedTextLines = textConfig.lines.map((line, lineIdx) => {
          const baseline = lineCursorY + line.size * 0.78
          const node = (
            <text
              key={`line-${lineIdx}`}
              x={rx + rw / 2}
              y={baseline}
              textAnchor="middle"
              dominantBaseline="auto"
              fill={line.color}
              fontSize={line.size}
              fontWeight={line.weight}
              fontFamily={FONT_FAMILY}
              letterSpacing={line.tracking}
            >
              {line.text}
            </text>
          )
          lineCursorY += line.size * 0.95 + 2
          return node
        })

        return (
          <g key={room.id || idx}>
            {isPoly ? (
              <polygon
                points={pointsToString(roomPoly)}
                fill={getRoomFill(room.type)}
                stroke="#1F2937"
                strokeWidth="1.35"
              />
            ) : (
              <rect
                x={rx} y={ry} width={rw} height={rh}
                fill={getRoomFill(room.type)} stroke="#1F2937" strokeWidth="1.35"
              />
            )}

            {/* Hatch patterns (clipped to room bounds) */}
            {getRoomHatch(room, rx, ry, rw, rh, clipId)}

            {/* Furniture symbols */}
            <g clipPath={`url(#${clipId})`}>
              {getFurniture(room, rx, ry)}
            </g>

            {textConfig.showBackground && (
              <rect
                x={rx + rw / 2 - textConfig.bgWidth / 2}
                y={ry + rh / 2 - textConfig.bgHeight / 2}
                width={textConfig.bgWidth}
                height={textConfig.bgHeight}
                fill="#FFFFFF"
                fillOpacity="0.78"
                rx="3"
                stroke="none"
              />
            )}

            {renderedTextLines}
          </g>
        )
      })}

      {/* ── Doors ────────────────────────────────────── */}
      {doors.map((door, idx) => {
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
          <g key={door.id || `door-${idx}`}>
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
      {windows.map((win, idx) => {
        const wx = tx(win.x)
        const wy = ty(win.y)
        const ww = win.width * SCALE

        if (win.wall === 'south' || win.wall === 'north') {
          return (
            <g key={win.id || `win-${idx}`}>
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
            <g key={win.id || `win-${idx}`}>
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
          fontWeight="600" fontFamily={FONT_FAMILY}>
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
          fontWeight="600" fontFamily={FONT_FAMILY}
          transform={`rotate(90, ${plotX + plotW + 30}, ${plotY + plotH / 2})`}>
          {pl.toFixed(0)} ft
        </text>
      </g>

      {/* ── North Arrow ──────────────────────────────── */}
      <g transform={`translate(${svgW - 48}, 30)`}>
        <polygon points="0,-20 -6,6 6,6" fill="#000" stroke="#000" strokeWidth="0.5" />
        <polygon points="0,-20 0,6 6,6" fill="#555" stroke="none" />
        <text x="0" y="-26" textAnchor="middle" fill="#000"
          fontSize="12" fontWeight="700" fontFamily={FONT_FAMILY}>N</text>
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
          fontSize="8" fontFamily={FONT_FAMILY}>0</text>
        <text x={5 * SCALE} y="14" textAnchor="middle" fill="#000"
          fontSize="8" fontFamily={FONT_FAMILY}>5</text>
        <text x={10 * SCALE} y="14" textAnchor="end" fill="#000"
          fontSize="8" fontFamily={FONT_FAMILY}>10 ft</text>
      </g>

      {/* ── Road label ───────────────────────────────── */}
      <text
        x={plotX + plotW / 2} y={svgH - 8}
        textAnchor="middle" fill="#555"
        fontSize="10" fontWeight="600"
        fontFamily={FONT_FAMILY}
        letterSpacing="1.2"
      >
        ROAD ({plot.road_side?.toUpperCase() || 'SOUTH'})
      </text>

      {/* ── Title block ──────────────────────────────── */}
      <text x={plotX} y={plotY - 24}
        fill="#000" fontSize="11" fontWeight="700"
        fontFamily={FONT_FAMILY} letterSpacing="1.5">
        RESIDENTIAL FLOOR PLAN
      </text>
      <text x={plotX} y={plotY - 10}
        fill="#555" fontSize="8" fontWeight="500"
        fontFamily={FONT_FAMILY} letterSpacing="0.5">
        {pw.toFixed(0)}' × {pl.toFixed(0)}' | {bhkCount} BHK | Vastu Score: {plan.vastu_score?.toFixed(0) || '—'}/100 | Road: {plot.road_side?.toUpperCase()} | Scale 1:100
      </text>
    </svg>
  )
}
