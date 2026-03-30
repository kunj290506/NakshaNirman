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

import FormInterface from '../components/FormInterface'
import CadCanvas from '../components/canvas/CadCanvas'
import ExportPanel from '../components/ExportPanel'
import Viewer3D from '../components/Viewer3D'

export default function Workspace() {
    const navigate = useNavigate()
    const { state } = useLayout()
    const actions = useLayoutActions()
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
                plan_mode: 'perfcat',
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
        <div className="workspace-simple">
            <header className="workspace-simple-header">
                <div className="workspace-simple-brand">
                    <Link to="/" className="logo workspace-brand">
                        <span className="logo-icon">
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                            </svg>
                        </span>
                        NakshaNirman
                    </Link>
                    <p className="workspace-simple-subtitle">Simple home planning, step by step.</p>
                </div>

                <div className="workspace-simple-header-actions">
                    <span className={`workspace-health ${state.backendHealthy ? 'ok' : 'bad'}`}>
                        {state.backendHealthy ? 'Backend Online' : 'Backend Offline'}
                    </span>
                    <button className="btn btn-secondary btn-sm" onClick={handleNewProject} title="New Project">
                        New
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={() => navigate('/')} title="Home">
                        Home
                    </button>
                </div>
            </header>

            <main className="workspace-simple-main">
                <section className="workspace-simple-card workspace-simple-input">
                    <h2>Enter Requirements</h2>
                    <p>Fill this form and click generate.</p>
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
                </section>

                <section className="workspace-simple-card workspace-simple-output">
                    <div className="workspace-simple-output-head">
                        <h2>Preview</h2>
                        <div className="workspace-simple-output-actions">
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
                        </div>
                    </div>

                    <div className="workspace-simple-canvas">
                        {state.previewMode === '3d' ? (
                            state.projectId && state.layout ? (
                                <Viewer3D projectId={state.projectId} />
                            ) : (
                                <div className="preview-empty">
                                    <h3>Generate a plan first</h3>
                                    <p>Use the form to create your layout.</p>
                                </div>
                            )
                        ) : (
                            <CadCanvas />
                        )}
                    </div>
                </section>
            </main>
        </div>
    )
}
