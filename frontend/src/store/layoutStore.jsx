/**
 * Centralized Layout State — Single source of truth for the entire plan.
 *
 * All UI components read from this state.
 * Mutations go through dispatch actions only.
 */

import { createContext, useContext, useReducer, useCallback } from 'react'

// ─── Initial State ───
const initialState = {
    // Project
    projectId: null,
    sessionId: 'sess_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36),

    // Plot
    plot: null,        // { width, length, unit }
    boundary: null,    // polygon points
    boundaryData: null,

    // Layout (from engine)
    layout: null,      // full layout JSON from backend
    rooms: [],         // normalized rooms array
    doors: [],
    windows: [],
    walls: { internal: [], external: [] },
    zones: [],
    adjacencyGraph: [],
    routingGraph: [],
    constraints: {},
    circulation: null,

    // Design Intelligence
    designScore: null,         // { composite, grade, breakdown, issues, vastu_bonuses, ... }
    architectNarrative: null,  // professional architect text

    // Editing
    selectedRoomId: null,
    editHistory: [],   // undo stack
    isDirty: false,

    // UI
    loading: false,
    loadingMessage: '',
    error: null,
    activePanel: 'form',     // 'chat' | 'form'
    previewMode: '2d',       // '2d' | 'boundary' | '3d'
    backendHealthy: true,

    // Canvas
    zoom: 1,
    panX: 0,
    panY: 0,
    showGrid: true,
    showDimensions: true,
    showLabels: true,
    showFurniture: true,
    snapToGrid: true,
    gridSize: 1,  // 1 foot
}

// ─── Action Types ───
export const Actions = {
    SET_PROJECT: 'SET_PROJECT',
    SET_PLOT: 'SET_PLOT',
    SET_BOUNDARY: 'SET_BOUNDARY',
    SET_LAYOUT: 'SET_LAYOUT',
    CLEAR_LAYOUT: 'CLEAR_LAYOUT',
    NEW_PROJECT: 'NEW_PROJECT',

    SELECT_ROOM: 'SELECT_ROOM',
    UPDATE_ROOM: 'UPDATE_ROOM',
    MOVE_ROOM: 'MOVE_ROOM',
    RESIZE_ROOM: 'RESIZE_ROOM',

    SET_LOADING: 'SET_LOADING',
    SET_ERROR: 'SET_ERROR',
    SET_PANEL: 'SET_PANEL',
    SET_PREVIEW: 'SET_PREVIEW',
    SET_BACKEND_HEALTH: 'SET_BACKEND_HEALTH',

    SET_ZOOM: 'SET_ZOOM',
    SET_PAN: 'SET_PAN',
    TOGGLE_GRID: 'TOGGLE_GRID',
    TOGGLE_DIMENSIONS: 'TOGGLE_DIMENSIONS',
    TOGGLE_LABELS: 'TOGGLE_LABELS',
    TOGGLE_FURNITURE: 'TOGGLE_FURNITURE',
    TOGGLE_SNAP: 'TOGGLE_SNAP',
}

// ─── Reducer ───
function layoutReducer(state, action) {
    switch (action.type) {
        case Actions.SET_PROJECT:
            return { ...state, projectId: action.projectId }

        case Actions.SET_PLOT:
            return { ...state, plot: action.plot }

        case Actions.SET_BOUNDARY:
            return {
                ...state,
                boundary: action.boundary,
                boundaryData: action.boundaryData || state.boundaryData,
                previewMode: 'boundary',
            }

        case Actions.SET_LAYOUT: {
            const layout = action.layout
            return {
                ...state,
                layout,
                rooms: (layout.rooms || []).map((r, i) => ({ ...r, id: r.id || `room-${i}` })),
                doors: layout.doors || [],
                windows: layout.windows || [],
                zones: layout.zones || [],
                adjacencyGraph: layout.adjacency_graph || [],
                routingGraph: layout.routing_graph || [],
                constraints: layout.constraints || {},
                circulation: layout.circulation || null,
                plot: layout.plot || state.plot,
                designScore: action.designScore || null,
                architectNarrative: action.architectNarrative || null,
                selectedRoomId: null,
                isDirty: false,
                previewMode: '2d',
                error: null,
            }
        }

        case Actions.CLEAR_LAYOUT:
            return {
                ...state,
                layout: null,
                rooms: [],
                doors: [],
                windows: [],
                zones: [],
                adjacencyGraph: [],
                routingGraph: [],
                selectedRoomId: null,
                isDirty: false,
            }

        case Actions.NEW_PROJECT:
            return {
                ...initialState,
                sessionId: 'sess_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36),
                backendHealthy: state.backendHealthy,
            }

        case Actions.SELECT_ROOM:
            return { ...state, selectedRoomId: action.roomId }

        case Actions.UPDATE_ROOM: {
            const rooms = state.rooms.map(r =>
                r.id === action.roomId ? { ...r, ...action.updates } : r
            )
            return { ...state, rooms, isDirty: true }
        }

        case Actions.MOVE_ROOM: {
            const { roomId, dx, dy } = action
            const rooms = state.rooms.map(r => {
                if (r.id !== roomId) return r
                const newX = state.snapToGrid
                    ? Math.round((r.x + dx) / state.gridSize) * state.gridSize
                    : r.x + dx
                const newY = state.snapToGrid
                    ? Math.round((r.y + dy) / state.gridSize) * state.gridSize
                    : r.y + dy
                return { ...r, x: newX, y: newY }
            })
            return { ...state, rooms, isDirty: true }
        }

        case Actions.RESIZE_ROOM: {
            const { roomId, newWidth, newLength } = action
            const rooms = state.rooms.map(r => {
                if (r.id !== roomId) return r
                const w = state.snapToGrid
                    ? Math.round(newWidth / state.gridSize) * state.gridSize
                    : newWidth
                const l = state.snapToGrid
                    ? Math.round(newLength / state.gridSize) * state.gridSize
                    : newLength
                return {
                    ...r,
                    width: Math.max(w, 4),
                    length: Math.max(l, 4),
                    area: Math.round(w * l * 10) / 10,
                }
            })
            return { ...state, rooms, isDirty: true }
        }

        case Actions.SET_LOADING:
            return { ...state, loading: action.loading, loadingMessage: action.message || '' }

        case Actions.SET_ERROR:
            return { ...state, error: action.error, loading: false }

        case Actions.SET_PANEL:
            return { ...state, activePanel: action.panel }

        case Actions.SET_PREVIEW:
            return { ...state, previewMode: action.mode }

        case Actions.SET_BACKEND_HEALTH:
            return { ...state, backendHealthy: action.healthy }

        case Actions.SET_ZOOM:
            return { ...state, zoom: Math.max(0.25, Math.min(4, action.zoom)) }

        case Actions.SET_PAN:
            return { ...state, panX: action.x, panY: action.y }

        case Actions.TOGGLE_GRID:
            return { ...state, showGrid: !state.showGrid }

        case Actions.TOGGLE_DIMENSIONS:
            return { ...state, showDimensions: !state.showDimensions }

        case Actions.TOGGLE_LABELS:
            return { ...state, showLabels: !state.showLabels }

        case Actions.TOGGLE_FURNITURE:
            return { ...state, showFurniture: !state.showFurniture }

        case Actions.TOGGLE_SNAP:
            return { ...state, snapToGrid: !state.snapToGrid }

        default:
            return state
    }
}

// ─── Context ───
const LayoutContext = createContext(null)

export function LayoutProvider({ children }) {
    const [state, dispatch] = useReducer(layoutReducer, initialState)

    return (
        <LayoutContext.Provider value={{ state, dispatch }}>
            {children}
        </LayoutContext.Provider>
    )
}

export function useLayout() {
    const ctx = useContext(LayoutContext)
    if (!ctx) throw new Error('useLayout must be used within LayoutProvider')
    return ctx
}

export function useLayoutActions() {
    const { dispatch } = useLayout()

    return {
        setProject: useCallback((id) => dispatch({ type: Actions.SET_PROJECT, projectId: id }), [dispatch]),
        setPlot: useCallback((plot) => dispatch({ type: Actions.SET_PLOT, plot }), [dispatch]),
        setBoundary: useCallback((boundary, data) => dispatch({ type: Actions.SET_BOUNDARY, boundary, boundaryData: data }), [dispatch]),
        setLayout: useCallback((layout, designScore, architectNarrative) => dispatch({ type: Actions.SET_LAYOUT, layout, designScore, architectNarrative }), [dispatch]),
        clearLayout: useCallback(() => dispatch({ type: Actions.CLEAR_LAYOUT }), [dispatch]),
        newProject: useCallback(() => dispatch({ type: Actions.NEW_PROJECT }), [dispatch]),
        selectRoom: useCallback((id) => dispatch({ type: Actions.SELECT_ROOM, roomId: id }), [dispatch]),
        updateRoom: useCallback((id, updates) => dispatch({ type: Actions.UPDATE_ROOM, roomId: id, updates }), [dispatch]),
        moveRoom: useCallback((id, dx, dy) => dispatch({ type: Actions.MOVE_ROOM, roomId: id, dx, dy }), [dispatch]),
        resizeRoom: useCallback((id, w, l) => dispatch({ type: Actions.RESIZE_ROOM, roomId: id, newWidth: w, newLength: l }), [dispatch]),
        setLoading: useCallback((loading, msg) => dispatch({ type: Actions.SET_LOADING, loading, message: msg }), [dispatch]),
        setError: useCallback((err) => dispatch({ type: Actions.SET_ERROR, error: err }), [dispatch]),
        setPanel: useCallback((panel) => dispatch({ type: Actions.SET_PANEL, panel }), [dispatch]),
        setPreview: useCallback((mode) => dispatch({ type: Actions.SET_PREVIEW, mode }), [dispatch]),
        setBackendHealth: useCallback((healthy) => dispatch({ type: Actions.SET_BACKEND_HEALTH, healthy }), [dispatch]),
        setZoom: useCallback((z) => dispatch({ type: Actions.SET_ZOOM, zoom: z }), [dispatch]),
        setPan: useCallback((x, y) => dispatch({ type: Actions.SET_PAN, x, y }), [dispatch]),
        toggleGrid: useCallback(() => dispatch({ type: Actions.TOGGLE_GRID }), [dispatch]),
        toggleDimensions: useCallback(() => dispatch({ type: Actions.TOGGLE_DIMENSIONS }), [dispatch]),
        toggleLabels: useCallback(() => dispatch({ type: Actions.TOGGLE_LABELS }), [dispatch]),
        toggleFurniture: useCallback(() => dispatch({ type: Actions.TOGGLE_FURNITURE }), [dispatch]),
        toggleSnap: useCallback(() => dispatch({ type: Actions.TOGGLE_SNAP }), [dispatch]),
    }
}
