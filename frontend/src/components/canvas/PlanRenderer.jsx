import { useMemo } from 'react'

const ROOM_COLORS = {
  living: '#EFF6FF',
  dining: '#F5F3FF',
  kitchen: '#ECFDF5',
  master_bedroom: '#FFFBEB',
  bedroom: '#FFF7ED',
  bathroom: '#F0FDFA',
  master_bath: '#F0FDFA',
  toilet: '#F0FDFA',
  open_area: '#F8FAFC',
  garage: '#F1F5F9',
  pooja: '#FEFCE8',
  study: '#FDF4FF',
  corridor: '#F8FAFC',
  staircase: '#F8FAFC',
}

const WALL_THICK_FT = 0.28
const WALL_STROKE_FT = 0.07

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function round1(value) {
  return Math.round(Number(value || 0) * 10) / 10
}

function ellipsize(text, maxChars) {
  const raw = String(text || '').trim()
  if (raw.length <= maxChars) return raw
  if (maxChars <= 3) return raw.slice(0, maxChars)
  return `${raw.slice(0, Math.max(1, maxChars - 3)).trimEnd()}...`
}

function wrapLabelLines(text, maxChars, maxLines = 2) {
  const words = String(text || '').trim().split(/\s+/).filter(Boolean)
  if (!words.length) return ['ROOM']

  const lines = []
  let current = ''

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word
    if (!current || candidate.length <= maxChars) {
      current = candidate
      continue
    }
    lines.push(current)
    current = word
  }
  if (current) lines.push(current)

  if (lines.length > maxLines) {
    const compact = lines.slice(0, maxLines)
    compact[maxLines - 1] = ellipsize(lines.slice(maxLines - 1).join(' '), maxChars)
    return compact
  }
  return lines.map((line) => ellipsize(line, maxChars))
}

function shouldShowFurniture(room, innerW, innerH) {
  if (['corridor', 'staircase', 'foyer', 'open_area', 'balcony', 'store', 'utility'].includes(room.type)) {
    return false
  }
  return innerW >= 3.1 && innerH >= 2.7
}

function getRoomLabelLayout(room, innerW, innerH, reserveBottom = 0) {
  const labelText = String(room.label || room.type || 'Room').toUpperCase().replace(/\s+/g, ' ').trim()
  const maxChars = Math.max(5, Math.floor(innerW / 0.34))
  const lines = wrapLabelLines(labelText, maxChars, innerH < 2.2 ? 1 : 2)
  const longest = lines.reduce((max, line) => Math.max(max, line.length), 1)

  const freeHeight = Math.max(0.9, innerH - reserveBottom)
  const labelSize = clamp(
    Math.min(innerW / Math.max(2.6, longest * 0.56), freeHeight / (lines.length + 1.0)),
    0.32,
    0.76,
  )

  const showArea = innerW >= 2.0 && freeHeight >= 1.55
  const areaSize = clamp(labelSize * 0.72, 0.26, 0.5)
  const blockHeight = lines.length * labelSize * 1.04 + (showArea ? areaSize * 1.08 + 0.12 : 0)
  const textWidth = longest * labelSize * 0.56

  return {
    lines,
    labelSize,
    areaSize,
    showArea,
    blockHeight,
    textWidth,
  }
}

function toSvgY(usableLength, y, h = 0) {
  return usableLength - y - h
}

function normalizeRooms(plan) {
  return (plan?.rooms || [])
    .map((room, idx) => ({
      id: room.id || `room_${idx + 1}`,
      type: room.type || 'room',
      label: room.label || 'Room',
      x: Number(room.x || 0),
      y: Number(room.y || 0),
      width: Number(room.width || 0),
      height: Number(room.height || 0),
      area: Number(room.area || Number(room.width || 0) * Number(room.height || 0)),
      color: room.color,
    }))
    .filter((room) => room.width > 0.1 && room.height > 0.1)
}

function roomTopY(room, usableLength) {
  return toSvgY(usableLength, room.y, room.height)
}

function roomCenterOnWall(room, wall, usableLength, hint) {
  if (!room) return null

  const xHint = Number(hint?.x)
  const yHint = Number(hint?.y)

  const tHorizontal = Number.isFinite(xHint)
    ? clamp((xHint - room.x) / Math.max(0.01, room.width), 0.18, 0.82)
    : 0.5
  const tVertical = Number.isFinite(yHint)
    ? clamp((yHint - room.y) / Math.max(0.01, room.height), 0.18, 0.82)
    : 0.5

  if (wall === 'north') {
    return {
      x: room.x + room.width * tHorizontal,
      y: toSvgY(usableLength, room.y + room.height),
    }
  }
  if (wall === 'south') {
    return {
      x: room.x + room.width * tHorizontal,
      y: toSvgY(usableLength, room.y),
    }
  }
  if (wall === 'east') {
    return {
      x: room.x + room.width,
      y: toSvgY(usableLength, room.y + room.height * tVertical),
    }
  }
  return {
    x: room.x,
    y: toSvgY(usableLength, room.y + room.height * tVertical),
  }
}

function doorGeometry(door, room, usableLength) {
  const wall = String(door.wall || 'south').toLowerCase()
  const width = clamp(Number(door.width || 3), 2.4, 4.5)
  const anchor = roomCenterOnWall(room, wall, usableLength, door)
  if (!anchor) return null

  const x = anchor.x
  const y = anchor.y

  if (wall === 'south') {
    return {
      gap: [x - width / 2, y, x + width / 2, y],
      arc: `M ${x - width / 2} ${y} A ${width} ${width} 0 0 1 ${x} ${y - width}`,
      label: { x, y: y - 1.1 },
    }
  }
  if (wall === 'north') {
    return {
      gap: [x - width / 2, y, x + width / 2, y],
      arc: `M ${x + width / 2} ${y} A ${width} ${width} 0 0 0 ${x} ${y + width}`,
      label: { x, y: y + 1.2 },
    }
  }
  if (wall === 'east') {
    return {
      gap: [x, y - width / 2, x, y + width / 2],
      arc: `M ${x} ${y - width / 2} A ${width} ${width} 0 0 0 ${x - width} ${y}`,
      label: { x: x - 1.6, y },
    }
  }

  return {
    gap: [x, y - width / 2, x, y + width / 2],
    arc: `M ${x} ${y + width / 2} A ${width} ${width} 0 0 1 ${x + width} ${y}`,
    label: { x: x + 1.6, y },
  }
}

function windowGeometry(win, room, usableLength) {
  const wall = String(win.wall || 'south').toLowerCase()
  const width = clamp(Number(win.width || 3.5), 2.4, 6.0)
  const anchor = roomCenterOnWall(room, wall, usableLength, win)
  if (!anchor) return []

  const x = anchor.x
  const y = anchor.y
  const off = 0.12

  if (wall === 'south' || wall === 'north') {
    return [
      [x - width / 2, y - off, x + width / 2, y - off],
      [x - width / 2, y + off, x + width / 2, y + off],
      [x - width / 2, y - 0.28, x - width / 2, y + 0.28],
      [x + width / 2, y - 0.28, x + width / 2, y + 0.28],
    ]
  }

  return [
    [x - off, y - width / 2, x - off, y + width / 2],
    [x + off, y - width / 2, x + off, y + width / 2],
    [x - 0.28, y - width / 2, x + 0.28, y - width / 2],
    [x - 0.28, y + width / 2, x + 0.28, y + width / 2],
  ]
}

function renderFurniture(room, usableLength) {
  const x = room.x + WALL_THICK_FT + 0.3
  const y = roomTopY(room, usableLength) + WALL_THICK_FT + 0.3
  const w = Math.max(0, room.width - WALL_THICK_FT * 2 - 0.6)
  const h = Math.max(0, room.height - WALL_THICK_FT * 2 - 0.6)

  if (w < 2.5 || h < 2.5) return null

  if (room.type === 'master_bedroom' || room.type === 'bedroom') {
    const bedW = Math.min(w * 0.72, w - 0.2)
    const bedH = Math.min(h * 0.62, h - 0.2)
    const bx = x + (w - bedW) / 2
    const by = y + (h - bedH) / 2
    return (
      <g>
        <rect x={bx} y={by} width={bedW} height={bedH} fill='none' stroke='#1F2937' strokeWidth={0.08} />
        <rect x={bx + 0.2} y={by + 0.2} width={bedW - 0.4} height={Math.max(0.4, bedH * 0.18)} fill='none' stroke='#475569' strokeWidth={0.06} />
      </g>
    )
  }

  if (room.type === 'bathroom' || room.type === 'master_bath' || room.type === 'toilet') {
    return (
      <g>
        <ellipse cx={x + w * 0.7} cy={y + h * 0.72} rx={Math.max(0.35, w * 0.13)} ry={Math.max(0.5, h * 0.16)} fill='none' stroke='#1F2937' strokeWidth={0.08} />
        <circle cx={x + w * 0.25} cy={y + h * 0.22} r={Math.max(0.22, Math.min(w, h) * 0.1)} fill='none' stroke='#1F2937' strokeWidth={0.08} />
      </g>
    )
  }

  if (room.type === 'kitchen') {
    const leg = Math.min(w * 0.62, h * 0.62)
    return (
      <g>
        <rect x={x} y={y} width={leg} height={0.45} fill='none' stroke='#1F2937' strokeWidth={0.08} />
        <rect x={x} y={y} width={0.45} height={leg} fill='none' stroke='#1F2937' strokeWidth={0.08} />
      </g>
    )
  }

  if (room.type === 'living') {
    const sofaW = Math.min(w * 0.7, w - 0.3)
    const sofaH = Math.max(0.55, h * 0.22)
    const sx = x + (w - sofaW) / 2
    const sy = y + h * 0.62
    return (
      <g>
        <rect x={sx} y={sy} width={sofaW} height={sofaH} fill='none' stroke='#1F2937' strokeWidth={0.08} />
        <rect x={x + w * 0.34} y={y + h * 0.34} width={Math.max(0.8, w * 0.32)} height={Math.max(0.45, h * 0.2)} fill='none' stroke='#475569' strokeWidth={0.07} />
      </g>
    )
  }

  if (room.type === 'dining') {
    const tableW = Math.max(0.9, w * 0.45)
    const tableH = Math.max(0.7, h * 0.3)
    const tx = x + (w - tableW) / 2
    const ty = y + (h - tableH) / 2
    return (
      <g>
        <rect x={tx} y={ty} width={tableW} height={tableH} fill='none' stroke='#1F2937' strokeWidth={0.08} />
        <rect x={tx - 0.28} y={ty + tableH * 0.2} width={0.2} height={0.35} fill='none' stroke='#475569' strokeWidth={0.06} />
        <rect x={tx - 0.28} y={ty + tableH * 0.65} width={0.2} height={0.35} fill='none' stroke='#475569' strokeWidth={0.06} />
        <rect x={tx + tableW + 0.08} y={ty + tableH * 0.2} width={0.2} height={0.35} fill='none' stroke='#475569' strokeWidth={0.06} />
        <rect x={tx + tableW + 0.08} y={ty + tableH * 0.65} width={0.2} height={0.35} fill='none' stroke='#475569' strokeWidth={0.06} />
      </g>
    )
  }

  return null
}

function getRoadLabelPlacement(facing, uw, ul) {
  const side = String(facing || 'south').toLowerCase()
  if (side === 'east') {
    return { x: uw + 3.2, y: ul / 2, transform: `rotate(90 ${uw + 3.2} ${ul / 2})` }
  }
  if (side === 'west') {
    return { x: -3.2, y: ul / 2, transform: `rotate(-90 ${-3.2} ${ul / 2})` }
  }
  if (side === 'north') {
    return { x: uw / 2, y: -3.2, transform: undefined }
  }
  return { x: uw / 2, y: ul + 4.4, transform: undefined }
}

function topDimensionSegments(rooms) {
  if (!rooms.length) return []
  const minY = Math.min(...rooms.map((room) => room.y))
  const row = rooms
    .filter((room) => Math.abs(room.y - minY) <= 0.6)
    .sort((a, b) => a.x - b.x)

  return row.map((room) => ({
    x1: room.x,
    x2: room.x + room.width,
    label: `${round1(room.width)} ft`,
  }))
}

function rightDimensionSegments(rooms, usableLength) {
  if (!rooms.length) return []
  const maxEast = Math.max(...rooms.map((room) => room.x + room.width))
  const col = rooms
    .filter((room) => Math.abs((room.x + room.width) - maxEast) <= 0.6)
    .sort((a, b) => a.y - b.y)

  return col.map((room) => ({
    y1: toSvgY(usableLength, room.y),
    y2: toSvgY(usableLength, room.y + room.height),
    label: `${round1(room.height)} ft`,
  }))
}

export default function PlanRenderer({
  plan,
  selectedRoomId,
  showDimensions = true,
  showLabels = true,
  showFurniture = true,
}) {
  const rooms = useMemo(() => normalizeRooms(plan), [plan])

  if (!plan || rooms.length === 0) {
    return <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: '#64748B' }}>No floor plan data</div>
  }

  const plot = plan.plot || {}
  const plotW = Number(plot.width || 0)
  const plotL = Number(plot.length || 0)
  const uw = Number(plot.usable_width || Math.max(...rooms.map((room) => room.x + room.width)))
  const ul = Number(plot.usable_length || Math.max(...rooms.map((room) => room.y + room.height)))
  const setbacks = plot.setbacks || { left: 0, right: 0, front: 0, rear: 0 }

  const margin = 8
  const vbX = -setbacks.left - margin
  const vbY = -setbacks.rear - margin
  const vbW = plotW + margin * 2
  const vbH = plotL + margin * 2

  const roomById = new Map(rooms.map((room) => [String(room.id), room]))
  const doors = plan.doors || []
  const windows = plan.windows || []
  const facing = String(plot.facing || plot.road_side || 'south').toLowerCase()

  const plotBoundary = Array.isArray(plot.boundary)
    ? plot.boundary
        .map((pt) => {
          if (!pt || typeof pt !== 'object') return null
          const x = Number(pt.x)
          const y = Number(pt.y)
          if (!Number.isFinite(x) || !Number.isFinite(y)) return null
          return [x, toSvgY(ul, y)]
        })
        .filter(Boolean)
    : []

  const scaleBarLength = Math.max(4, Math.min(10, Math.floor(Math.max(0, plotW) / 4) || 4))
  const roadLabel = getRoadLabelPlacement(facing, uw, ul)
  const topSegments = topDimensionSegments(rooms)
  const rightSegments = rightDimensionSegments(rooms, ul)

  return (
    <div style={{ width: '100%', height: '100%', background: '#fff' }}>
      <svg viewBox={`${vbX} ${vbY} ${vbW} ${vbH}`} width='100%' height='100%'>
        <defs>
          <pattern id='openHatch' width='0.8' height='0.8' patternUnits='userSpaceOnUse' patternTransform='rotate(35)'>
            <line x1='0' y1='0' x2='0' y2='0.8' stroke='#CBD5E1' strokeWidth='0.08' />
          </pattern>
        </defs>

        <rect x={vbX} y={vbY} width={vbW} height={vbH} fill='#fff' />

        {plotBoundary.length >= 3 ? (
          <polygon
            points={plotBoundary.map((pt) => `${pt[0]},${pt[1]}`).join(' ')}
            fill='#F8FAFC'
            stroke='#111827'
            strokeWidth='0.22'
          />
        ) : (
          <rect
            x={-setbacks.left}
            y={-setbacks.rear}
            width={plotW}
            height={plotL}
            fill='#F8FAFC'
            stroke='#111827'
            strokeWidth='0.22'
          />
        )}

        <rect
          x={0}
          y={0}
          width={uw}
          height={ul}
          fill='none'
          stroke='#94A3B8'
          strokeDasharray='0.8 0.6'
          strokeWidth='0.1'
        />

        {rooms.map((room) => {
          const topY = roomTopY(room, ul)
          const innerX = room.x + WALL_THICK_FT
          const innerY = topY + WALL_THICK_FT
          const innerW = room.width - WALL_THICK_FT * 2
          const innerH = room.height - WALL_THICK_FT * 2
          const isOpen = room.type === 'open_area'
          const clipId = `clip-${room.id}`
          const midX = room.x + room.width / 2
          const midY = topY + room.height / 2
          const fill = room.color || ROOM_COLORS[room.type] || '#F8FAFC'
          const canShowRoomDimensions = showDimensions && !isOpen && innerW >= 3.4 && innerH >= 2.8
          const reserveBottom = canShowRoomDimensions ? 0.62 : 0
          const furnitureVisible = showFurniture && shouldShowFurniture(room, innerW, innerH) && !isOpen
          const labelLayout = getRoomLabelLayout(room, innerW, innerH, reserveBottom)
          const preferredTop = furnitureVisible
            ? innerY + Math.max(0.12, innerH * 0.1)
            : innerY + (innerH - labelLayout.blockHeight) * 0.5
          const labelTop = clamp(
            preferredTop,
            innerY + 0.1,
            innerY + Math.max(0.12, innerH - labelLayout.blockHeight - reserveBottom - 0.08),
          )
          const labelBgW = Math.min(Math.max(0.4, innerW - 0.12), labelLayout.textWidth + 0.3)
          const labelBgH = Math.min(Math.max(0.28, innerH - reserveBottom - 0.08), labelLayout.blockHeight + 0.16)

          return (
            <g key={room.id} data-room-id={room.id}>
              {isOpen ? (
                <rect
                  x={room.x}
                  y={topY}
                  width={room.width}
                  height={room.height}
                  fill='url(#openHatch)'
                  stroke='#64748B'
                  strokeWidth='0.1'
                />
              ) : innerW > 0.2 && innerH > 0.2 ? (
                <>
                  <path
                    d={`M ${room.x} ${topY} H ${room.x + room.width} V ${topY + room.height} H ${room.x} Z M ${innerX} ${innerY} H ${innerX + innerW} V ${innerY + innerH} H ${innerX} Z`}
                    fill={room.id === selectedRoomId ? '#374151' : '#4B5563'}
                    fillRule='evenodd'
                  />
                  <rect
                    x={innerX}
                    y={innerY}
                    width={innerW}
                    height={innerH}
                    fill={fill}
                  />
                  <rect
                    x={room.x}
                    y={topY}
                    width={room.width}
                    height={room.height}
                    fill='none'
                    stroke='#111827'
                    strokeWidth={WALL_STROKE_FT}
                  />
                  <rect
                    x={innerX}
                    y={innerY}
                    width={innerW}
                    height={innerH}
                    fill='none'
                    stroke='#111827'
                    strokeWidth={WALL_STROKE_FT}
                  />
                  <defs>
                    <clipPath id={clipId}>
                      <rect x={innerX} y={innerY} width={innerW} height={innerH} />
                    </clipPath>
                  </defs>
                </>
              ) : (
                <rect
                  x={room.x}
                  y={topY}
                  width={room.width}
                  height={room.height}
                  fill={fill}
                  stroke='#111827'
                  strokeWidth={WALL_STROKE_FT}
                />
              )}

              <g clipPath={innerW > 0.2 && innerH > 0.2 && !isOpen ? `url(#${clipId})` : undefined}>
                {furnitureVisible && renderFurniture(room, ul)}

                {showLabels && innerW >= 1.4 && innerH >= 1.1 && (
                  <>
                    <rect
                      x={midX - labelBgW / 2}
                      y={labelTop - 0.08}
                      width={labelBgW}
                      height={labelBgH}
                      fill='#FFFFFF'
                      fillOpacity='0.85'
                      rx='0.06'
                      stroke='none'
                    />
                    {labelLayout.lines.map((line, lineIdx) => (
                      <text
                        key={`${room.id}-label-${lineIdx}`}
                        x={midX}
                        y={labelTop + labelLayout.labelSize * (0.82 + lineIdx * 1.04)}
                        textAnchor='middle'
                        fill='#0F172A'
                        fontSize={labelLayout.labelSize}
                        fontWeight='700'
                        style={{ pointerEvents: 'none' }}
                      >
                        {line}
                      </text>
                    ))}
                    {labelLayout.showArea && (
                      <text
                        x={midX}
                        y={labelTop + labelLayout.lines.length * labelLayout.labelSize * 1.04 + labelLayout.areaSize * 0.92 + 0.04}
                        textAnchor='middle'
                        fill='#334155'
                        fontSize={labelLayout.areaSize}
                        style={{ pointerEvents: 'none' }}
                      >
                        {Math.round(room.area)} sq ft
                      </text>
                    )}
                  </>
                )}

                {canShowRoomDimensions && (
                  <>
                    <text
                      x={midX}
                      y={innerY + innerH - 0.12}
                      textAnchor='middle'
                      fill='#475569'
                      fontSize='0.44'
                      fontWeight='600'
                    >
                      {round1(room.width)} ft
                    </text>
                    <text
                      x={innerX + 0.16}
                      y={midY}
                      textAnchor='middle'
                      fill='#64748B'
                      fontSize='0.42'
                      fontWeight='600'
                      transform={`rotate(-90 ${innerX + 0.16} ${midY})`}
                    >
                      {round1(room.height)} ft
                    </text>
                  </>
                )}
              </g>
            </g>
          )
        })}

        {windows.map((win, idx) => {
          const key = win.id || `win-${idx}`
          const room = roomById.get(String(win.room_id || ''))
          if (!room) return null
          const lines = windowGeometry(win, room, ul)
          if (!lines.length) return null
          return (
            <g key={key}>
              {lines.map((line, lineIdx) => (
                <line
                  key={`${key}-line-${lineIdx}`}
                  x1={line[0]}
                  y1={line[1]}
                  x2={line[2]}
                  y2={line[3]}
                  stroke='#2563EB'
                  strokeWidth='0.1'
                />
              ))}
            </g>
          )
        })}

        {doors.map((door, idx) => {
          const key = door.id || `door-${idx}`
          const room = roomById.get(String(door.room_id || ''))
          if (!room) return null
          const geom = doorGeometry(door, room, ul)
          if (!geom) return null

          const stroke = door.type === 'main' ? '#DC2626' : '#334155'
          return (
            <g key={key}>
              <line
                x1={geom.gap[0]}
                y1={geom.gap[1]}
                x2={geom.gap[2]}
                y2={geom.gap[3]}
                stroke='#FFFFFF'
                strokeWidth='0.22'
              />
              <path
                d={geom.arc}
                fill='none'
                stroke={stroke}
                strokeWidth='0.08'
                strokeDasharray={door.type === 'internal' ? '0.45 0.25' : 'none'}
              />
              {door.type === 'main' && (
                <text
                  x={geom.label.x}
                  y={geom.label.y}
                  textAnchor='middle'
                  fontSize='0.6'
                  fill='#DC2626'
                  fontWeight='700'
                >
                  MAIN DOOR
                </text>
              )}
            </g>
          )
        })}

        {showDimensions && topSegments.length > 0 && (
          <g>
            {topSegments.map((seg, idx) => {
              const y = -1.5
              const mid = (seg.x1 + seg.x2) / 2
              return (
                <g key={`top-dim-${idx}`}>
                  <line x1={seg.x1} y1={y} x2={seg.x2} y2={y} stroke='#0F172A' strokeWidth='0.08' />
                  <line x1={seg.x1} y1={y - 0.28} x2={seg.x1 + 0.22} y2={y + 0.28} stroke='#0F172A' strokeWidth='0.08' />
                  <line x1={seg.x2 - 0.22} y1={y - 0.28} x2={seg.x2} y2={y + 0.28} stroke='#0F172A' strokeWidth='0.08' />
                  <text x={mid} y={y - 0.34} textAnchor='middle' fill='#334155' fontSize='0.56'>
                    {seg.label}
                  </text>
                </g>
              )
            })}
          </g>
        )}

        {showDimensions && rightSegments.length > 0 && (
          <g>
            {rightSegments.map((seg, idx) => {
              const x = uw + 1.6
              const mid = (seg.y1 + seg.y2) / 2
              return (
                <g key={`right-dim-${idx}`}>
                  <line x1={x} y1={seg.y1} x2={x} y2={seg.y2} stroke='#0F172A' strokeWidth='0.08' />
                  <line x1={x - 0.28} y1={seg.y1} x2={x + 0.28} y2={seg.y1 + 0.22} stroke='#0F172A' strokeWidth='0.08' />
                  <line x1={x - 0.28} y1={seg.y2 - 0.22} x2={x + 0.28} y2={seg.y2} stroke='#0F172A' strokeWidth='0.08' />
                  <text x={x + 0.4} y={mid} fill='#334155' fontSize='0.56' dominantBaseline='middle'>
                    {seg.label}
                  </text>
                </g>
              )
            })}
          </g>
        )}

        <g transform={`translate(${uw - 2.4}, 1.2)`}>
          <polygon points='0,-0.9 0.75,0.8 0,0.32 -0.75,0.8' fill='#0F172A' />
          <text x='0' y='1.9' textAnchor='middle' fontSize='1.0' fontWeight='700' fill='#0F172A'>N</text>
        </g>

        <g transform={`translate(1.2, ${ul + 2.4})`}>
          <line x1='0' y1='0' x2={scaleBarLength} y2='0' stroke='#0F172A' strokeWidth='0.12' />
          <line x1='0' y1='-0.36' x2='0' y2='0.36' stroke='#0F172A' strokeWidth='0.12' />
          <line x1={scaleBarLength / 2} y1='-0.36' x2={scaleBarLength / 2} y2='0.36' stroke='#0F172A' strokeWidth='0.12' />
          <line x1={scaleBarLength} y1='-0.36' x2={scaleBarLength} y2='0.36' stroke='#0F172A' strokeWidth='0.12' />
          <text x={scaleBarLength / 2} y='1.15' textAnchor='middle' fill='#334155' fontSize='0.9'>
            Scale: {scaleBarLength} ft
          </text>
        </g>

        <text
          x={roadLabel.x}
          y={roadLabel.y}
          transform={roadLabel.transform}
          textAnchor='middle'
          fontSize='1.0'
          fill='#334155'
          fontWeight='700'
        >
          ROAD ({facing.toUpperCase()})
        </text>
      </svg>
    </div>
  )
}
