/**
 * Workspace — Main 3-panel application layout.
 *
 * LEFT:   Design Controls (Chat / Form tabs)
 * CENTER: CAD Canvas with toolbar
 * RIGHT:  Property Panel
 */

import { useState, useEffect, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useLayout, useLayoutActions } from '../store/layoutStore'
import * as api from '../services/api'

import AIDesignChat from '../components/AIDesignChat'
import FormInterface from '../components/FormInterface'
import CadCanvas from '../components/canvas/CadCanvas'
import CanvasToolbar from '../components/toolbar/CanvasToolbar'
import PropertyPanel from '../components/panels/PropertyPanel'
import ExportPanel from '../components/ExportPanel'
import Viewer3D from '../components/Viewer3D'

export default function Workspace() {
    const navigate = useNavigate()
    const { state } = useLayout()
    const actions = useLayoutActions()
    const [rightCollapsed, setRightCollapsed] = useState(false)
    const [lastStrategy, setLastStrategy] = useState(null)

    // Health check
    useEffect(() => {
        let mounted = true
        const check = async () => {
            try {
                const ok = await api.healthCheck()
                if (mounted) actions.setBackendHealth(ok)
            } catch {
                if (mounted) actions.setBackendHealth(false)
            }
        }
        check()
        const t = setInterval(check, 15000)
        return () => { mounted = false; clearInterval(t) }
    }, [actions])

    // Ensure project exists
    const ensureProject = useCallback(async (totalArea) => {
        if (state.projectId) return state.projectId
        const data = await api.createProject(state.sessionId, totalArea)
        const pid = data.project_id
        actions.setProject(pid)
        return pid
    }, [state.projectId, state.sessionId, actions])

    // Unified generate handler — used by both Chat and Form
    const handleGenerate = useCallback(async (rooms, totalArea, requirements = null) => {
        actions.setLoading(true, 'Creating project...')
        try {
            const pid = await ensureProject(totalArea)

            if (requirements) {
                try { await api.storeRequirements({ ...requirements, project_id: pid }) }
                catch { /* non-critical */ }
            }

            actions.setLoading(true, 'Generating floor plan...')

            const roomExtras = rooms
                .filter(r => !['master_bedroom', 'bedroom', 'bathroom', 'kitchen', 'living'].includes(r.room_type))
                .map(r => r.room_type)
            const reqExtras = requirements?.extras || []
            const allExtras = [...new Set([...roomExtras, ...reqExtras])]

            const payload = {
                project_id: pid,
                total_area: totalArea,
                plot_width: requirements?.plot_width || null,
                plot_length: requirements?.plot_length || null,
                engine_mode: requirements?.engine_mode || 'gnn_advanced',
                rooms,
                bedrooms: Number(requirements?.bedrooms) || rooms.filter(r => ['bedroom', 'master_bedroom'].includes(r.room_type)).reduce((s, r) => s + (r.quantity || 1), 0) || 2,
                bathrooms: Number(requirements?.bathrooms) || rooms.filter(r => r.room_type === 'bathroom').reduce((s, r) => s + (r.quantity || 1), 0) || 1,
                kitchens: rooms.filter(r => r.room_type === 'kitchen').reduce((s, r) => s + (r.quantity || 1), 0) || 1,
                floors: requirements?.floors || 1,
                extras: allExtras,
                boundary_polygon: state.boundary || null,
                previous_strategy: lastStrategy,
            }

            const json = await api.generateDesign(payload)

            if (json.layout) {
                actions.setLayout(json.layout, json.design_score, json.architect_narrative)
                if (json.project_id) actions.setProject(json.project_id)
                if (json.zoning_strategy) setLastStrategy(json.zoning_strategy)
            } else if (json.error) {
                actions.setError(json.error)
            }
        } catch (err) {
            actions.setError(err.message || 'Generation failed. Please try again.')
        } finally {
            actions.setLoading(false)
        }
    }, [ensureProject, state.boundary, actions])

    // Boundary upload handler
    const handleBoundaryUpload = useCallback(async (file) => {
        actions.setLoading(true, 'Uploading boundary...')
        try {
            let pid = state.projectId
            if (!pid) {
                try { pid = await ensureProject(1200) } catch { pid = null }
            }

            const uploadData = await api.uploadBoundary(file, pid)
            actions.setLoading(true, 'Extracting boundary...')
            const extractData = await api.extractBoundary(uploadData.file_id)
            actions.setLoading(true, 'Computing buildable area...')
            const footprintData = await api.buildableFootprint(uploadData.file_id)

            const resultData = {
                boundary: extractData.boundary_polygon,
                usable_polygon: footprintData.usable_polygon,
                area: footprintData.boundary_area,
                usable_area: footprintData.usable_area,
                setback: footprintData.setback_applied,
                coverage_ratio: footprintData.coverage_ratio,
                num_vertices: extractData.num_vertices,
                file_id: uploadData.file_id,
            }

            actions.setBoundary(footprintData.usable_polygon, resultData)
            actions.setLoading(false)
            return resultData
        } catch (err) {
            actions.setError(err.message || 'Boundary processing failed.')
            actions.setLoading(false)
            return null
        }
    }, [state.projectId, ensureProject, actions])

    const handleNewProject = () => actions.newProject()

    return (
        <div className={`workspace workspace-3panel${rightCollapsed ? ' right-collapsed' : ''}`}>
            {/* ── Top Navigation ── */}
            <nav className="workspace-nav">
                <Link to="/" className="logo">
                    <span className="logo-icon">
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                    </span>
                    NakshaNirman
                </Link>

                <div className="workspace-nav-center">
                    {state.previewMode === '2d' && state.layout && <CanvasToolbar />}
                </div>

                <div className="workspace-nav-right">
                    <div className="status-indicator">
                        <span className={`status-dot ${state.backendHealthy ? 'online' : 'offline'}`} />
                        <span className="status-text">{state.backendHealthy ? 'Online' : 'Offline'}</span>
                    </div>

                    {/* Preview mode tabs */}
                    <div className="preview-mode-tabs">
                        <button className={`mode-tab${state.previewMode === '2d' ? ' active' : ''}`} onClick={() => actions.setPreview('2d')}>
                            2D
                        </button>
                        {state.boundaryData && (
                            <button className={`mode-tab${state.previewMode === 'boundary' ? ' active' : ''}`} onClick={() => actions.setPreview('boundary')}>
                                Boundary
                            </button>
                        )}
                        <button className={`mode-tab${state.previewMode === '3d' ? ' active' : ''}`} onClick={() => actions.setPreview('3d')}>
                            3D
                        </button>
                    </div>

                    {state.projectId && (
                        <ExportPanel projectId={state.projectId} onGenerate3D={() => actions.setPreview('3d')} />
                    )}

                    <button className="btn btn-secondary btn-sm" onClick={handleNewProject} title="New Project">
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                        </svg>
                        New
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={() => navigate('/')} title="Home">
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z" />
                        </svg>
                    </button>
                </div>
            </nav>

            {/* ── Left Panel: Design Controls ── */}
            <div className="sidebar">
                <div className="sidebar-header">
                    <span className="sidebar-title">
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                        </svg>
                        Design Controls
                    </span>
                </div>
                <div className="tab-switcher">
                    <button className={`tab-btn ${state.activePanel === 'chat' ? 'active' : ''}`} onClick={() => actions.setPanel('chat')}>
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        Chat
                    </button>
                    <button className={`tab-btn ${state.activePanel === 'form' ? 'active' : ''}`} onClick={() => actions.setPanel('form')}>
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        Form
                    </button>
                </div>
                <div className="sidebar-content">
                    {state.activePanel === 'chat' ? (
                        <AIDesignChat
                            onGenerate={handleGenerate}
                            onBoundaryUpload={handleBoundaryUpload}
                            loading={state.loading}
                            projectId={state.projectId}
                            plan={state.layout}
                        />
                    ) : (
                        <FormInterface
                            onGenerate={handleGenerate}
                            onBoundaryUpload={handleBoundaryUpload}
                            boundary={state.boundary}
                            boundaryData={state.boundaryData}
                            loading={state.loading}
                            backendHealthy={state.backendHealthy}
                            onCheckBackend={async () => {
                                try {
                                    const ok = await api.healthCheck()
                                    actions.setBackendHealth(ok)
                                    return ok
                                } catch {
                                    actions.setBackendHealth(false)
                                    return false
                                }
                            }}
                        />
                    )}
                </div>
            </div>

            {/* ── Center: Canvas ── */}
            <main className="canvas-panel">
                {state.previewMode === '3d' ? (
                    (state.projectId && state.layout)
                        ? <Viewer3D projectId={state.projectId} />
                        : <div className="preview-empty"><h3>Generate a plan first</h3><p>Switch to 2D mode and create a floor plan</p></div>
                ) : (
                    <CadCanvas />
                )}
            </main>

            {/* ── Right: Properties ── */}
            <aside className={`right-panel${rightCollapsed ? ' collapsed' : ''}`}>
                <button
                    className="panel-toggle"
                    onClick={() => setRightCollapsed(!rightCollapsed)}
                    title={rightCollapsed ? 'Show Properties' : 'Hide Properties'}
                >
                    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d={rightCollapsed ? 'M15 19l-7-7 7-7' : 'M9 5l7 7-7 7'} />
                    </svg>
                </button>
                {!rightCollapsed && <PropertyPanel />}
            </aside>
        </div>
    )
}
