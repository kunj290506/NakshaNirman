import { useMemo } from 'react'

const COLORS = {
  living: '#E8F4FD',
  master_bedroom: '#FFF3E0',
  bedroom: '#FFF3E0',
  kitchen: '#E8F5E9',
  dining: '#F3E5F5',
  bathroom: '#E0F2F1',
  corridor: '#F5F5F5',
  pooja: '#FFF9C4',
  study: '#FFF9C4',
  store: '#FFF9C4',
  balcony: '#FFF9C4',
  garage: '#FFF9C4',
}

function normalizeRooms(plan) {
  return (plan?.rooms || []).map((r, idx) => {
    let x = Number(r.x ?? r.position?.x ?? 0)
    let y = Number(r.y ?? r.position?.y ?? 0)
    let width = Number(r.width ?? r.w ?? 0)
    let height = Number(r.height ?? r.h ?? r.length ?? 0)

    if ((!width || !height) && r.dimensions) {
      if (Array.isArray(r.dimensions)) {
        width = Number(r.dimensions[0] || 0)
        height = Number(r.dimensions[1] || 0)
      }
    }

    if ((!width || !height) && Array.isArray(r.polygon) && r.polygon.length >= 3) {
      const xs = r.polygon.map((p) => Number(p[0]))
      const ys = r.polygon.map((p) => Number(p[1]))
      x = Math.min(...xs)
      y = Math.min(...ys)
      width = Math.max(...xs) - x
      height = Math.max(...ys) - y
    }

    return {
      id: r.id || r.room_id || `room_${idx + 1}`,
      type: r.type || r.room_type || 'room',
      label: r.label || r.name || r.type || 'Room',
      x,
      y,
      width,
      height,
      area: Number(r.area || (width || 0) * (height || 0)),
    }
  }).filter((r) => r.width > 0 && r.height > 0)
}

function doorGeometry(room, door) {
  const pos = Number(door.position || 0)
  const w = Number(door.width || 3)

  if (door.wall === 'south') {
    return {
      x1: room.x + pos - w / 2,
      y1: room.y,
      x2: room.x + pos + w / 2,
      y2: room.y,
      arc: `M ${room.x + pos - w / 2} ${room.y} A ${w} ${w} 0 0 1 ${room.x + pos} ${room.y + w}`,
    }
  }
  if (door.wall === 'north') {
    return {
      x1: room.x + pos - w / 2,
      y1: room.y + room.height,
      x2: room.x + pos + w / 2,
      y2: room.y + room.height,
      arc: `M ${room.x + pos + w / 2} ${room.y + room.height} A ${w} ${w} 0 0 1 ${room.x + pos} ${room.y + room.height - w}`,
    }
  }
  if (door.wall === 'east') {
    return {
      x1: room.x + room.width,
      y1: room.y + pos - w / 2,
      x2: room.x + room.width,
      y2: room.y + pos + w / 2,
      arc: `M ${room.x + room.width} ${room.y + pos - w / 2} A ${w} ${w} 0 0 1 ${room.x + room.width - w} ${room.y + pos}`,
    }
  }
  return {
    x1: room.x,
    y1: room.y + pos - w / 2,
    x2: room.x,
    y2: room.y + pos + w / 2,
    arc: `M ${room.x} ${room.y + pos + w / 2} A ${w} ${w} 0 0 1 ${room.x + w} ${room.y + pos}`,
  }
}

function windowGeometry(room, win) {
  const pos = Number(win.position || 0)
  const w = Number(win.width || 3)
  const off = 0.18

  if (win.wall === 'south' || win.wall === 'north') {
    const y = win.wall === 'south' ? room.y : room.y + room.height
    return {
      x1: room.x + pos - w / 2,
      y1: y - off,
      x2: room.x + pos + w / 2,
      y2: y - off,
      x3: room.x + pos - w / 2,
      y3: y + off,
      x4: room.x + pos + w / 2,
      y4: y + off,
    }
  }

  const x = win.wall === 'west' ? room.x : room.x + room.width
  return {
    x1: x - off,
    y1: room.y + pos - w / 2,
    x2: x - off,
    y2: room.y + pos + w / 2,
    x3: x + off,
    y3: room.y + pos - w / 2,
    x4: x + off,
    y4: room.y + pos + w / 2,
  }
}

export default function PlanRenderer({
  plan,
  selectedRoomId,
  showDimensions = true,
  showLabels = true,
}) {
  const rooms = useMemo(() => normalizeRooms(plan), [plan])

  if (!plan || rooms.length === 0) {
    return (
      <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: '#666' }}>
        No floor plan data
      </div>
    )
  }

  const plot = plan.plot || {}
  const usableWidth = Number(plot.width || Math.max(...rooms.map((r) => r.x + r.width)))
  const usableLength = Number(plot.length || Math.max(...rooms.map((r) => r.y + r.height)))
  const originalWidth = Number(plot.original_width || usableWidth)
  const originalLength = Number(plot.original_length || usableLength)
  const setbacks = plot.setbacks || { left: 0, right: 0, front: 0, rear: 0 }

  const vbPad = Math.max(8, Math.max(originalWidth, originalLength) * 0.25)
  const viewBox = [
    -setbacks.left - 2,
    -setbacks.front - 2,
    originalWidth + 4,
    originalLength + 4,
  ]

  const doors = plan.doors || []
  const windows = plan.windows || []

  return (
    <div style={{ width: '100%', height: '100%', background: '#fff' }}>
      <svg viewBox={`${viewBox[0]} ${viewBox[1]} ${viewBox[2]} ${viewBox[3]}`} width='100%' height='100%'>
        <rect
          x={0 - setbacks.left}
          y={0 - setbacks.front}
          width={originalWidth}
          height={originalLength}
          fill='none'
          stroke='#111'
          strokeWidth={0.35}
        />

        <rect
          x={0}
          y={0}
          width={usableWidth}
          height={usableLength}
          fill='none'
          stroke='#bdbdbd'
          strokeWidth={0.2}
          strokeDasharray='0.9 0.55'
        />

        {rooms.map((room) => {
          const selected = room.id === selectedRoomId
          const fill = COLORS[room.type] || '#F8FAFC'
          const fs = Math.max(0.6, Math.min(room.width, room.height) * 0.13)

          return (
            <g key={room.id} data-room-id={room.id}>
              <rect
                x={room.x}
                y={room.y}
                width={room.width}
                height={room.height}
                fill={fill}
                stroke={selected ? '#111' : '#666'}
                strokeWidth={selected ? 0.28 : 0.12}
              />

              {showLabels && room.width > 2.5 && room.height > 2.2 && (
                <>
                  <text
                    x={room.x + room.width / 2}
                    y={room.y + room.height / 2 - fs * 0.2}
                    fontSize={fs}
                    fontFamily='Georgia, serif'
                    textAnchor='middle'
                    fill='#202020'
                  >
                    {room.label}
                  </text>
                  <text
                    x={room.x + room.width / 2}
                    y={room.y + room.height / 2 + fs * 0.9}
                    fontSize={fs * 0.8}
                    fontFamily='Georgia, serif'
                    textAnchor='middle'
                    fill='#4a4a4a'
                  >
                    ({Math.round(room.area)} sq ft)
                  </text>
                </>
              )}

              {showDimensions && (
                <>
                  <line x1={room.x} y1={room.y - 0.35} x2={room.x + room.width} y2={room.y - 0.35} stroke='#555' strokeWidth={0.05} />
                  <line x1={room.x} y1={room.y - 0.5} x2={room.x} y2={room.y - 0.2} stroke='#555' strokeWidth={0.05} />
                  <line x1={room.x + room.width} y1={room.y - 0.5} x2={room.x + room.width} y2={room.y - 0.2} stroke='#555' strokeWidth={0.05} />
                  <text x={room.x + room.width / 2} y={room.y - 0.55} fontSize={0.5} textAnchor='middle' fill='#555'>
                    {room.width.toFixed(1)} ft
                  </text>

                  <line x1={room.x - 0.35} y1={room.y} x2={room.x - 0.35} y2={room.y + room.height} stroke='#555' strokeWidth={0.05} />
                  <line x1={room.x - 0.5} y1={room.y} x2={room.x - 0.2} y2={room.y} stroke='#555' strokeWidth={0.05} />
                  <line x1={room.x - 0.5} y1={room.y + room.height} x2={room.x - 0.2} y2={room.y + room.height} stroke='#555' strokeWidth={0.05} />
                  <text
                    x={room.x - 0.6}
                    y={room.y + room.height / 2}
                    fontSize={0.5}
                    textAnchor='middle'
                    fill='#555'
                    transform={`rotate(-90, ${room.x - 0.6}, ${room.y + room.height / 2})`}
                  >
                    {room.height.toFixed(1)} ft
                  </text>
                </>
              )}
            </g>
          )
        })}

        {doors.map((door) => {
          const room = rooms.find((r) => r.id === door.room_id)
          if (!room) return null
          const g = doorGeometry(room, door)
          return (
            <g key={door.id}>
              <line x1={g.x1} y1={g.y1} x2={g.x2} y2={g.y2} stroke='#fff' strokeWidth={0.24} />
              <path d={g.arc} fill='none' stroke='#3d3d3d' strokeWidth={0.1} />
            </g>
          )
        })}

        {windows.map((win) => {
          const room = rooms.find((r) => r.id === win.room_id)
          if (!room) return null
          const g = windowGeometry(room, win)
          return (
            <g key={win.id}>
              <line x1={g.x1} y1={g.y1} x2={g.x2} y2={g.y2} stroke='#3d3d3d' strokeWidth={0.07} />
              <line x1={g.x3} y1={g.y3} x2={g.x4} y2={g.y4} stroke='#3d3d3d' strokeWidth={0.07} />
            </g>
          )
        })}

        <g transform={`translate(${originalWidth - 3.5}, ${0.8})`}>
          <polygon points='0,0 0.7,1.8 0,1.25 -0.7,1.8' fill='#111' />
          <text x='0' y='2.65' textAnchor='middle' fontSize='0.75' fill='#111'>N</text>
        </g>

        <g transform={`translate(${(originalWidth - 6) / 2}, ${originalLength + 1.4})`}>
          <line x1='0' y1='0' x2='6' y2='0' stroke='#111' strokeWidth={0.1} />
          <line x1='0' y1='-0.3' x2='0' y2='0.3' stroke='#111' strokeWidth={0.1} />
          <line x1='3' y1='-0.3' x2='3' y2='0.3' stroke='#111' strokeWidth={0.1} />
          <line x1='6' y1='-0.3' x2='6' y2='0.3' stroke='#111' strokeWidth={0.1} />
          <text x='3' y='0.8' textAnchor='middle' fontSize='0.5' fill='#222'>Scale bar: 6 ft</text>
        </g>

        <rect
          x={viewBox[0] - vbPad * 0.02}
          y={viewBox[1] - vbPad * 0.02}
          width={viewBox[2] + vbPad * 0.04}
          height={viewBox[3] + vbPad * 0.04}
          fill='none'
          stroke='#111'
          strokeWidth={0.08}
        />
      </svg>
    </div>
  )
}
