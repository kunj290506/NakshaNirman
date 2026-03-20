/**
 * CadCanvas — Interactive CAD-style canvas wrapper.
 *
 * Provides:
 * - Grid background
 * - Zoom/pan with mouse wheel + drag
 * - Room selection on click
 * - Room dragging
 * - Renders PlanPreview SVG inside a zoomable container
 */

import { useRef, useState, useCallback, useEffect } from 'react'
import { useLayout, useLayoutActions } from '../../store/layoutStore'
import PlanPreview from '../PlanPreview'
import BoundaryPreview from '../BoundaryPreview'

export default function CadCanvas() {
    const { state } = useLayout()
    const actions = useLayoutActions()
    const containerRef = useRef(null)
    const [isPanning, setIsPanning] = useState(false)
    const [panStart, setPanStart] = useState({ x: 0, y: 0 })
    const [dragRoom, setDragRoom] = useState(null)
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

    const { zoom, panX, panY, layout, rooms, previewMode, boundaryData, loading, error } = state

    // Wheel zoom
    const handleWheel = useCallback((e) => {
        e.preventDefault()
        const delta = e.deltaY > 0 ? 0.9 : 1.1
        actions.setZoom(zoom * delta)
    }, [zoom, actions])

    useEffect(() => {
        const el = containerRef.current
        if (!el) return
        el.addEventListener('wheel', handleWheel, { passive: false })
        return () => el.removeEventListener('wheel', handleWheel)
    }, [handleWheel])

    // Pan with middle mouse or Ctrl+drag
    const handleMouseDown = useCallback((e) => {
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            e.preventDefault()
            setIsPanning(true)
            setPanStart({ x: e.clientX - panX, y: e.clientY - panY })
        }
    }, [panX, panY])

    const handleMouseMove = useCallback((e) => {
        if (isPanning) {
            actions.setPan(e.clientX - panStart.x, e.clientY - panStart.y)
        }
        if (dragRoom) {
            const dx = (e.clientX - dragStart.x) / zoom / 10  // 10px per foot approx
            const dy = -(e.clientY - dragStart.y) / zoom / 10
            if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
                actions.moveRoom(dragRoom, dx, dy)
                setDragStart({ x: e.clientX, y: e.clientY })
            }
        }
    }, [isPanning, panStart, dragRoom, dragStart, zoom, actions])

    const handleMouseUp = useCallback(() => {
        setIsPanning(false)
        setDragRoom(null)
    }, [])

    // Room click handler — attached to SVG via event delegation
    const handleCanvasClick = useCallback((e) => {
        const roomEl = e.target.closest('[data-room-id]')
        if (roomEl) {
            const roomId = roomEl.getAttribute('data-room-id')
            actions.selectRoom(roomId)
        } else if (!e.target.closest('.canvas-toolbar')) {
            actions.selectRoom(null)
        }
    }, [actions])

    // Room drag start
    const handleRoomMouseDown = useCallback((e) => {
        const roomEl = e.target.closest('[data-room-id]')
        if (roomEl && e.button === 0 && !e.altKey) {
            const roomId = roomEl.getAttribute('data-room-id')
            actions.selectRoom(roomId)
            setDragRoom(roomId)
            setDragStart({ x: e.clientX, y: e.clientY })
            e.stopPropagation()
        }
    }, [actions])

    const showEmpty = !layout && !boundaryData && !loading

    return (
        <div
            className="cad-canvas"
            ref={containerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleCanvasClick}
            style={{ cursor: isPanning ? 'grabbing' : dragRoom ? 'move' : 'default' }}
        >
            {/* Grid background */}
            {state.showGrid && (
                <div className="cad-grid" style={{
                    backgroundSize: `${20 * zoom}px ${20 * zoom}px`,
                    backgroundPosition: `${panX}px ${panY}px`,
                }} />
            )}

            {/* Content */}
            <div
                className="cad-canvas-content"
                style={{
                    transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
                    transformOrigin: 'center center',
                }}
            >
                {previewMode === '2d' && layout ? (
                    <div onMouseDown={handleRoomMouseDown}>
                        <PlanPreview
                            plan={{ ...layout, rooms }}
                            selectedRoomId={state.selectedRoomId}
                            showGrid={state.showGrid}
                            showDimensions={state.showDimensions}
                            showLabels={state.showLabels}
                            showFurniture={state.showFurniture}
                        />
                    </div>
                ) : previewMode === 'boundary' && boundaryData ? (
                    <BoundaryPreview boundaryData={boundaryData} />
                ) : null}
            </div>

            {/* Loading overlay */}
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner" />
                    <span className="loading-text">{state.loadingMessage || 'Processing...'}</span>
                </div>
            )}

            {/* Error */}
            {error && !loading && (
                <div className="cad-error">{error}</div>
            )}

            {/* Empty state */}
            {showEmpty && (
                <div className="preview-empty">
                    <div className="preview-empty-icon">
                        <svg width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                        </svg>
                    </div>
                    <h3>No Floor Plan Yet</h3>
                    <p>Use the form or chat on the left to generate your first floor plan</p>
                </div>
            )}

            {/* North arrow */}
            {layout && (
                <div className="cad-north-arrow" title="North">
                    <svg width="28" height="28" viewBox="0 0 28 28">
                        <polygon points="14,2 18,12 14,9 10,12" fill="#111" />
                        <text x="14" y="24" textAnchor="middle" fontSize="9" fontWeight="600" fill="#111">N</text>
                    </svg>
                </div>
            )}

            {layout && (
                <div
                    style={{
                        position: 'absolute',
                        top: 12,
                        left: 12,
                        background: 'rgba(255,255,255,0.92)',
                        border: '1px solid #d1d5db',
                        borderRadius: 10,
                        padding: '6px 10px',
                        fontSize: '12px',
                        color: '#111827',
                        zIndex: 8,
                        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                    }}
                >
                    <div style={{ fontWeight: 700 }}>{(layout.bhk || '?')} BHK</div>
                    <div style={{ fontSize: 11, color: '#475569' }}>{layout.layout_signature || 'no-signature'}</div>
                </div>
            )}
        </div>
    )
}
