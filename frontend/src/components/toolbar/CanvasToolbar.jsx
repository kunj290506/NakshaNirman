/**
 * CanvasToolbar — Zoom, pan, grid, snap, and layer toggles
 * for the CAD canvas.
 */

import { useLayout, useLayoutActions } from '../../store/layoutStore'

const IconBtn = ({ onClick, active, title, children }) => (
    <button
        className={`canvas-toolbar-btn${active ? ' active' : ''}`}
        onClick={onClick}
        title={title}
    >
        {children}
    </button>
)

export default function CanvasToolbar() {
    const { state } = useLayout()
    const actions = useLayoutActions()

    const zoomIn = () => actions.setZoom(state.zoom * 1.25)
    const zoomOut = () => actions.setZoom(state.zoom / 1.25)
    const zoomFit = () => { actions.setZoom(1); actions.setPan(0, 0) }

    return (
        <div className="canvas-toolbar">
            {/* Zoom controls */}
            <div className="toolbar-group">
                <IconBtn onClick={zoomOut} title="Zoom Out">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35M8 11h6" />
                    </svg>
                </IconBtn>
                <span className="toolbar-zoom-label">{Math.round(state.zoom * 100)}%</span>
                <IconBtn onClick={zoomIn} title="Zoom In">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35M8 11h6M11 8v6" />
                    </svg>
                </IconBtn>
                <IconBtn onClick={zoomFit} title="Fit to View">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                    </svg>
                </IconBtn>
            </div>

            <div className="toolbar-divider" />

            {/* Layer toggles */}
            <div className="toolbar-group">
                <IconBtn onClick={actions.toggleGrid} active={state.showGrid} title="Grid">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                        <path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18" />
                    </svg>
                </IconBtn>
                <IconBtn onClick={actions.toggleSnap} active={state.snapToGrid} title="Snap to Grid">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path d="M4 4h4v4H4zM16 4h4v4h-4zM4 16h4v4H4zM16 16h4v4h-4z" />
                    </svg>
                </IconBtn>
                <IconBtn onClick={actions.toggleDimensions} active={state.showDimensions} title="Dimensions">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path d="M4 12h16M4 8v8M20 8v8" />
                    </svg>
                </IconBtn>
                <IconBtn onClick={actions.toggleLabels} active={state.showLabels} title="Labels">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path d="M4 7V4h16v3M9 20h6M12 4v16" />
                    </svg>
                </IconBtn>
                <IconBtn onClick={actions.toggleFurniture} active={state.showFurniture} title="Furniture">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                        <rect x="3" y="8" width="18" height="8" rx="2" /><path d="M5 8V6a2 2 0 012-2h10a2 2 0 012 2v2M5 16v2M19 16v2" />
                    </svg>
                </IconBtn>
            </div>
        </div>
    )
}
