import { createContext, useContext, useMemo, useReducer } from 'react'

const GRID_STEP = 0.5

const initialState = {
  zoom: 1,
  panX: 0,
  panY: 0,
  layout: null,
  rooms: [],
  selectedRoomId: null,
  showGrid: true,
  showDimensions: true,
  showLabels: true,
  showFurniture: true,
  snapToGrid: true,
  previewMode: '2d',
  boundaryData: null,
  loading: false,
  loadingMessage: '',
  error: null,
  isDirty: false,
  zones: [],
  designScore: null,
  architectNarrative: '',
}

const LayoutStateContext = createContext(null)
const LayoutActionsContext = createContext(null)

function roundToGrid(value) {
  return Math.round(Number(value || 0) / GRID_STEP) * GRID_STEP
}

function toNum(value, fallback = 0) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function normalizeRoom(rawRoom, index) {
  const id = String(rawRoom?.id || `room_${index + 1}`)
  const roomType = String(rawRoom?.type || 'room')
  const label = String(rawRoom?.label || roomType.replace(/_/g, ' '))
  const width = Math.max(0.5, toNum(rawRoom?.width, 0))
  const height = Math.max(0.5, toNum(rawRoom?.height, 0))

  return {
    id,
    name: label,
    room_type: roomType,
    width,
    length: height,
    position: {
      x: toNum(rawRoom?.x, 0),
      y: toNum(rawRoom?.y, 0),
    },
    zone: String(rawRoom?.zone || 'service'),
    band: toNum(rawRoom?.band, 2),
    color: rawRoom?.color,
  }
}

function toRenderableRooms(rooms) {
  return rooms.map((room) => ({
    id: room.id,
    type: room.room_type,
    label: room.name,
    x: room.position.x,
    y: room.position.y,
    width: room.width,
    height: room.length,
    zone: room.zone,
    band: room.band,
    color: room.color,
  }))
}

function buildZones(rooms) {
  const grouped = new Map()
  for (const room of rooms) {
    const zoneName = room.zone || 'service'
    if (!grouped.has(zoneName)) {
      grouped.set(zoneName, {
        name: zoneName,
        band: room.band || 2,
        rooms: [],
      })
    }
    grouped.get(zoneName).rooms.push(room.name)
  }
  return Array.from(grouped.values()).sort((a, b) => a.band - b.band)
}

function deriveDesignScore(plan) {
  const vastu = Math.max(0, Math.min(100, Math.round(toNum(plan?.vastu_score, 0))))
  const adjacency = Math.max(0, Math.min(100, Math.round(toNum(plan?.adjacency_score, 0))))
  const composite = Math.round(vastu * 0.55 + adjacency * 0.45)
  const grade = composite >= 80 ? 'A' : composite >= 65 ? 'B' : composite >= 50 ? 'C' : 'D'

  return {
    composite,
    grade,
    breakdown: {
      vastu_alignment: { score: vastu },
      adjacency_quality: { score: adjacency },
    },
    issues: [],
    vastu_bonuses: [],
  }
}

function deriveAreaSummary(plan, renderableRooms) {
  const plotArea = toNum(plan?.plot?.usable_width, 0) * toNum(plan?.plot?.usable_length, 0)
  const builtArea = renderableRooms.reduce(
    (sum, room) => sum + Math.max(0, toNum(room.width, 0) * toNum(room.height, 0)),
    0,
  )
  const circulationArea = renderableRooms
    .filter((room) => room.type === 'corridor')
    .reduce((sum, room) => sum + Math.max(0, toNum(room.width, 0) * toNum(room.height, 0)), 0)

  if (plotArea <= 0) {
    return {
      plot_area: 0,
      built_area: Math.round(builtArea * 10) / 10,
      utilization_percentage: '0%',
      circulation_percentage: '0%',
    }
  }

  return {
    plot_area: Math.round(plotArea * 10) / 10,
    built_area: Math.round(builtArea * 10) / 10,
    utilization_percentage: `${Math.round((builtArea / plotArea) * 100)}%`,
    circulation_percentage: `${Math.round((circulationArea / plotArea) * 100)}%`,
  }
}

function normalizeLayoutPayload(plan) {
  const safePlan = plan || {}
  const normalizedRooms = (safePlan.rooms || []).map((rawRoom, idx) => normalizeRoom(rawRoom, idx))
  const renderableRooms = toRenderableRooms(normalizedRooms)
  const zones = buildZones(normalizedRooms)
  const areaSummary = deriveAreaSummary(safePlan, renderableRooms)
  const bedrooms = renderableRooms.filter((room) => room.type === 'master_bedroom' || room.type === 'bedroom').length

  const layout = {
    ...safePlan,
    rooms: renderableRooms,
    bhk: bedrooms,
    layout_signature:
      safePlan.layout_signature ||
      `plan-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    area_summary: safePlan.area_summary || areaSummary,
  }

  return {
    layout,
    rooms: normalizedRooms,
    zones,
    designScore: deriveDesignScore(safePlan),
    architectNarrative: String(safePlan.architect_note || ''),
    boundaryData: safePlan.plot?.boundary || null,
  }
}

function updateLayoutRooms(layout, rooms) {
  if (!layout) return layout
  return {
    ...layout,
    rooms: toRenderableRooms(rooms),
  }
}

function reducer(state, action) {
  switch (action.type) {
    case 'SET_ZOOM': {
      const zoom = Math.max(0.25, Math.min(4, toNum(action.payload, 1)))
      return { ...state, zoom }
    }

    case 'SET_PAN':
      return {
        ...state,
        panX: toNum(action.payload?.x, state.panX),
        panY: toNum(action.payload?.y, state.panY),
      }

    case 'SET_LAYOUT': {
      if (!action.payload) {
        return {
          ...state,
          layout: null,
          rooms: [],
          selectedRoomId: null,
          zones: [],
          designScore: null,
          architectNarrative: '',
          boundaryData: null,
          isDirty: false,
        }
      }

      const normalized = normalizeLayoutPayload(action.payload)
      return {
        ...state,
        ...normalized,
        selectedRoomId: null,
        isDirty: false,
        error: null,
      }
    }

    case 'SELECT_ROOM':
      return { ...state, selectedRoomId: action.payload || null }

    case 'MOVE_ROOM': {
      const roomId = action.payload?.roomId
      const dx = toNum(action.payload?.dx, 0)
      const dy = toNum(action.payload?.dy, 0)
      if (!roomId || (dx === 0 && dy === 0)) return state

      const updatedRooms = state.rooms.map((room) => {
        if (room.id !== roomId) return room
        let nextX = room.position.x + dx
        let nextY = room.position.y + dy
        if (state.snapToGrid) {
          nextX = roundToGrid(nextX)
          nextY = roundToGrid(nextY)
        }
        return {
          ...room,
          position: { x: nextX, y: nextY },
        }
      })

      return {
        ...state,
        rooms: updatedRooms,
        layout: updateLayoutRooms(state.layout, updatedRooms),
        zones: buildZones(updatedRooms),
        isDirty: true,
      }
    }

    case 'RESIZE_ROOM': {
      const roomId = action.payload?.roomId
      const width = Math.max(3, toNum(action.payload?.width, 0))
      const length = Math.max(3, toNum(action.payload?.length, 0))
      if (!roomId) return state

      const updatedRooms = state.rooms.map((room) =>
        room.id === roomId
          ? {
              ...room,
              width: state.snapToGrid ? roundToGrid(width) : width,
              length: state.snapToGrid ? roundToGrid(length) : length,
            }
          : room,
      )

      return {
        ...state,
        rooms: updatedRooms,
        layout: updateLayoutRooms(state.layout, updatedRooms),
        zones: buildZones(updatedRooms),
        isDirty: true,
      }
    }

    case 'TOGGLE_GRID':
      return { ...state, showGrid: !state.showGrid }

    case 'TOGGLE_SNAP':
      return { ...state, snapToGrid: !state.snapToGrid }

    case 'TOGGLE_DIMENSIONS':
      return { ...state, showDimensions: !state.showDimensions }

    case 'TOGGLE_LABELS':
      return { ...state, showLabels: !state.showLabels }

    case 'TOGGLE_FURNITURE':
      return { ...state, showFurniture: !state.showFurniture }

    case 'SET_LOADING':
      return {
        ...state,
        loading: Boolean(action.payload?.loading),
        loadingMessage: String(action.payload?.message || ''),
      }

    case 'SET_ERROR':
      return { ...state, error: action.payload || null }

    case 'SET_BOUNDARY':
      return { ...state, boundaryData: action.payload || null }

    case 'SET_PREVIEW_MODE':
      return {
        ...state,
        previewMode: action.payload === 'boundary' ? 'boundary' : '2d',
      }

    default:
      return state
  }
}

export function LayoutProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  const actions = useMemo(
    () => ({
      setZoom: (zoom) => dispatch({ type: 'SET_ZOOM', payload: zoom }),
      setPan: (x, y) => dispatch({ type: 'SET_PAN', payload: { x, y } }),
      setLayout: (layout) => dispatch({ type: 'SET_LAYOUT', payload: layout }),
      selectRoom: (roomId) => dispatch({ type: 'SELECT_ROOM', payload: roomId }),
      moveRoom: (roomId, dx, dy) => dispatch({ type: 'MOVE_ROOM', payload: { roomId, dx, dy } }),
      resizeRoom: (roomId, width, length) => dispatch({ type: 'RESIZE_ROOM', payload: { roomId, width, length } }),
      toggleGrid: () => dispatch({ type: 'TOGGLE_GRID' }),
      toggleSnap: () => dispatch({ type: 'TOGGLE_SNAP' }),
      toggleDimensions: () => dispatch({ type: 'TOGGLE_DIMENSIONS' }),
      toggleLabels: () => dispatch({ type: 'TOGGLE_LABELS' }),
      toggleFurniture: () => dispatch({ type: 'TOGGLE_FURNITURE' }),
      setLoading: (loading, message = '') =>
        dispatch({ type: 'SET_LOADING', payload: { loading, message } }),
      setError: (error) => dispatch({ type: 'SET_ERROR', payload: error }),
      setBoundary: (boundaryData) => dispatch({ type: 'SET_BOUNDARY', payload: boundaryData }),
      setPreviewMode: (mode) => dispatch({ type: 'SET_PREVIEW_MODE', payload: mode }),
    }),
    [],
  )

  return (
    <LayoutStateContext.Provider value={{ state }}>
      <LayoutActionsContext.Provider value={actions}>
        {children}
      </LayoutActionsContext.Provider>
    </LayoutStateContext.Provider>
  )
}

export function useLayout() {
  const context = useContext(LayoutStateContext)
  if (!context) {
    throw new Error('useLayout must be used within LayoutProvider')
  }
  return context
}

export function useLayoutActions() {
  const actions = useContext(LayoutActionsContext)
  if (!actions) {
    throw new Error('useLayoutActions must be used within LayoutProvider')
  }
  return actions
}
