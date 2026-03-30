import { useMemo } from 'react'

const ROOM_COLORS = {
  living: '#EFF6FF',
  dining: '#F5F3FF',
  kitchen: '#F0FDF4',
  master_bedroom: '#FFFBEB',
  bedroom: '#FFF7ED',
  bathroom: '#F0FDFA',
  toilet: '#F0FDFA',
  open_area: '#F8FAFC',
  garage: '#F1F5F9',
  pooja: '#FEFCE8',
  study: '#FDF4FF',
}

function normalizeRooms(plan) {
  return (plan?.rooms || [])
    .map((r, idx) => ({
      id: r.id || `room_${idx + 1}`,
      type: r.type || 'room',
      label: r.label || 'Room',
      x: Number(r.x || 0),
      y: Number(r.y || 0),
      width: Number(r.width || 0),
      height: Number(r.height || 0),
      area: Number(r.area || Number(r.width || 0) * Number(r.height || 0)),
      furniture: Array.isArray(r.furniture) ? r.furniture : [],
    }))
    .filter((r) => r.width > 0.1 && r.height > 0.1)
}

function northArrowTransform(facing) {
  const side = String(facing || 'east').toLowerCase()
  if (side === 'north') return 'rotate(180)'
  if (side === 'south') return 'rotate(0)'
  if (side === 'west') return 'rotate(-90)'
  return 'rotate(90)'
}

function toSvgY(usableLength, y, h = 0) {
  return usableLength - y - h
}

function renderFurniture(room, usableLength) {
  return room.furniture.map((f, idx) => {
    const x = room.x + Number(f.x || 0)
    const y = toSvgY(usableLength, room.y + Number(f.y || 0), Number(f.height || 0))
    const w = Number(f.width || 0.5)
    const h = Number(f.height || 0.5)
    const key = `${room.id}-f-${idx}`

    if (f.type === 'pillow') {
      return (
        <ellipse
          key={key}
          cx={x + w / 2}
          cy={y + h / 2}
          rx={w / 2}
          ry={h / 2}
          fill='#E2E8F0'
          stroke='#94A3B8'
          vectorEffect='non-scaling-stroke'
        />
      )
    }

    if (f.type === 'shower') {
      return (
        <g key={key}>
          <rect x={x} y={y} width={w} height={h} fill='#E2E8F0' stroke='#94A3B8' vectorEffect='non-scaling-stroke' />
          <line x1={x} y1={y} x2={x + w} y2={y + h} stroke='#94A3B8' vectorEffect='non-scaling-stroke' />
          <line x1={x + w} y1={y} x2={x} y2={y + h} stroke='#94A3B8' vectorEffect='non-scaling-stroke' />
        </g>
      )
    }

    return (
      <rect
        key={key}
        x={x}
        y={y}
        width={w}
        height={h}
        rx={f.type === 'wheel' ? 0.2 : 0}
        fill='#E2E8F0'
        stroke='#94A3B8'
        vectorEffect='non-scaling-stroke'
      />
    )
  })
}

function doorPath(door, usableLength) {
  const x = Number(door.x || 0)
  const y = toSvgY(usableLength, Number(door.y || 0))
  const w = Number(door.width || 3)

  if (door.wall === 'south') {
    return {
      gap: [x - w / 2, y, x + w / 2, y],
      arc: `M ${x - w / 2} ${y} A ${w} ${w} 0 0 0 ${x} ${y - w}`,
    }
  }
  if (door.wall === 'north') {
    return {
      gap: [x - w / 2, y, x + w / 2, y],
      arc: `M ${x + w / 2} ${y} A ${w} ${w} 0 0 0 ${x} ${y + w}`,
    }
  }
  if (door.wall === 'east') {
    return {
      gap: [x, y - w / 2, x, y + w / 2],
      arc: `M ${x} ${y - w / 2} A ${w} ${w} 0 0 1 ${x - w} ${y}`,
    }
  }
  return {
    gap: [x, y - w / 2, x, y + w / 2],
    arc: `M ${x} ${y + w / 2} A ${w} ${w} 0 0 1 ${x + w} ${y}`,
  }
}

function windowLines(win, usableLength) {
  const x = Number(win.x || 0)
  const y = toSvgY(usableLength, Number(win.y || 0))
  const w = Number(win.width || 3.5)
  const off = 0.12

  if (win.wall === 'south' || win.wall === 'north') {
    return [
      [x - w / 2, y - off, x + w / 2, y - off],
      [x - w / 2, y + off, x + w / 2, y + off],
      [x - w / 2, y - 0.25, x - w / 2, y + 0.25],
      [x + w / 2, y - 0.25, x + w / 2, y + 0.25],
    ]
  }

  return [
    [x - off, y - w / 2, x - off, y + w / 2],
    [x + off, y - w / 2, x + off, y + w / 2],
    [x - 0.25, y - w / 2, x + 0.25, y - w / 2],
    [x - 0.25, y + w / 2, x + 0.25, y + w / 2],
  ]
}

export default function PlanRenderer({ plan, selectedRoomId, showDimensions = true, showLabels = true, showFurniture = true }) {
  const rooms = useMemo(() => normalizeRooms(plan), [plan])

  if (!plan || rooms.length === 0) {
    return <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: '#64748B' }}>No floor plan data</div>
  }

  const plot = plan.plot || {}
  const plotW = Number(plot.width || 0)
  const plotL = Number(plot.length || 0)
  const uw = Number(plot.usable_width || Math.max(...rooms.map((r) => r.x + r.width)))
  const ul = Number(plot.usable_length || Math.max(...rooms.map((r) => r.y + r.height)))
  const setbacks = plot.setbacks || { left: 0, right: 0, front: 0, rear: 0 }
  const margin = 6
  const vbX = -setbacks.left - margin
  const vbY = -setbacks.rear - margin
  const vbW = plotW + margin * 2
  const vbH = plotL + margin * 2

  const doors = plan.doors || []
  const windows = plan.windows || []
  const facing = String(plot.facing || plot.road_side || 'east').toLowerCase()

  return (
    <div style={{ width: '100%', height: '100%', background: '#fff' }}>
      <svg viewBox={`${vbX} ${vbY} ${vbW} ${vbH}`} width='100%' height='100%'>
        <defs>
          <pattern id='openHatch' width='0.8' height='0.8' patternUnits='userSpaceOnUse' patternTransform='rotate(35)'>
            <line x1='0' y1='0' x2='0' y2='0.8' stroke='#CBD5E1' strokeWidth='0.08' />
          </pattern>

        </defs>

        <rect x={vbX} y={vbY} width={vbW} height={vbH} fill='#fff' />

        <rect
          x={-setbacks.left}
          y={-setbacks.rear}
          width={plotW}
          height={plotL}
          fill='#F8FAFC'
          stroke='#111827'
          strokeWidth='3'
          vectorEffect='non-scaling-stroke'
        />

        <rect
          x={0}
          y={0}
          width={uw}
          height={ul}
          fill='none'
          stroke='#94A3B8'
          strokeDasharray='1.2 0.8'
          strokeWidth='1'
          vectorEffect='non-scaling-stroke'
        />

        {rooms.map((room) => {
          const y = toSvgY(ul, room.y, room.height)
          const fill = room.type === 'open_area' ? 'url(#openHatch)' : (room.color || ROOM_COLORS[room.type] || '#F8FAFC')
          const midX = room.x + room.width / 2
          const midY = y + room.height / 2
          return (
            <g key={room.id} data-room-id={room.id}>
              <rect
                x={room.x}
                y={y}
                width={room.width}
                height={room.height}
                fill={fill}
                stroke={room.id === selectedRoomId ? '#0F172A' : '#334155'}
                strokeWidth='1.5'
                vectorEffect='non-scaling-stroke'
              />
              {showFurniture && renderFurniture(room, ul)}
              {showLabels && (() => {
                const minDim = Math.min(room.width, room.height)
                const baseSize = Math.max(0.6, Math.min(1.4, minDim * 0.15))
                const yOffset = baseSize * 0.45
                
                return (
                  <>
                    <text
                      x={room.x + room.width / 2}
                      y={y + room.height / 2 - yOffset}
                      textAnchor='middle'
                      fill='#0F172A'
                      fontSize={baseSize}
                      fontWeight='700'
                      style={{ pointerEvents: 'none' }}
                    >
                      {room.label}
                    </text>
                    <text
                      x={room.x + room.width / 2}
                      y={y + room.height / 2 + yOffset * 1.5}
                      textAnchor='middle'
                      fill='#334155'
                      fontSize={baseSize * 0.8}
                      style={{ pointerEvents: 'none' }}
                    >
                      {Math.round(room.area)} sq ft
                    </text>
                  </>
                )
              })()}
              {showDimensions && (
                <>
                  <text
                    x={midX}
                    y={Math.max(y + 0.8, y + room.height * 0.2)}
                    textAnchor='middle'
                    fill='#475569'
                    fontSize='0.8'
                    fontWeight='600'
                    style={{ pointerEvents: 'none' }}
                  >
                    {Math.round(room.width * 10) / 10} ft
                  </text>
                  <text
                    x={Math.max(room.x + 0.7, room.x + room.width * 0.2)}
                    y={midY}
                    textAnchor='middle'
                    fill='#64748B'
                    fontSize='0.75'
                    fontWeight='600'
                    transform={`rotate(-90 ${Math.max(room.x + 0.7, room.x + room.width * 0.2)} ${midY})`}
                    style={{ pointerEvents: 'none' }}
                  >
                    {Math.round(room.height * 10) / 10} ft
                  </text>
                </>
              )}
            </g>
          )
        })}

        {windows.map((win, idx) => {
          const key = win.id || `win-${idx}`
          return (
          <g key={key}>
            {windowLines(win, ul).map((ln, i) => (
              <line
                key={`${key}-${i}`}
                x1={ln[0]}
                y1={ln[1]}
                x2={ln[2]}
                y2={ln[3]}
                stroke='#3B82F6'
                strokeWidth='1.2'
                vectorEffect='non-scaling-stroke'
              />
            ))}
          </g>
          )
        })}

        {doors.map((door, idx) => {
          const key = door.id || `door-${idx}`
          const g = doorPath(door, ul)
          const stroke = door.type === 'main' ? '#DC2626' : '#475569'
          return (
            <g key={key}>
              <line x1={g.gap[0]} y1={g.gap[1]} x2={g.gap[2]} y2={g.gap[3]} stroke='#fff' strokeWidth='2.2' vectorEffect='non-scaling-stroke' />
              <path
                d={g.arc}
                fill='none'
                stroke={stroke}
                strokeWidth='1.2'
                strokeDasharray={door.type === 'internal' ? '1.6 1.1' : 'none'}
                vectorEffect='non-scaling-stroke'
              />
              {door.type === 'main' && (
                <text x={Number(door.x || 0)} y={toSvgY(ul, Number(door.y || 0)) - 1.0} textAnchor='middle' fontSize='1.0' fill='#DC2626'>
                  MAIN DOOR
                </text>
              )}
            </g>
          )
        })}



        <g transform={`translate(${uw - 2.5}, 1.2)`}>
          <g transform={northArrowTransform(facing)}>
            <polygon points='0,-0.8 0.7,0.8 0,0.3 -0.7,0.8' fill='#0F172A' />
          </g>
          <text x='0' y='1.8' textAnchor='middle' fontSize='1.1' fontWeight='700' fill='#0F172A'>N</text>
        </g>

        <g transform={`translate(1.2, ${ul + 2.4})`}>
          <line x1='0' y1='0' x2='10' y2='0' stroke='#0F172A' strokeWidth='1.2' vectorEffect='non-scaling-stroke' />
          <line x1='0' y1='-0.4' x2='0' y2='0.4' stroke='#0F172A' strokeWidth='1.2' vectorEffect='non-scaling-stroke' />
          <line x1='5' y1='-0.4' x2='5' y2='0.4' stroke='#0F172A' strokeWidth='1.2' vectorEffect='non-scaling-stroke' />
          <line x1='10' y1='-0.4' x2='10' y2='0.4' stroke='#0F172A' strokeWidth='1.2' vectorEffect='non-scaling-stroke' />
          <text x='5' y='1.2' textAnchor='middle' fill='#334155' fontSize='1.0'>Scale: 10 ft</text>
        </g>

        <text x={uw / 2} y={ul + 4.4} textAnchor='middle' fontSize='1.1' fill='#334155' fontWeight='700'>ROAD / FRONT SIDE ({facing.toUpperCase()})</text>
      </svg>
    </div>
  )
}
