import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import ChatInterface from '../components/ChatInterface'
import FormInterface from '../components/FormInterface'
import PlanPreview from '../components/PlanPreview'
import Viewer3D from '../components/Viewer3D'
import ExportPanel from '../components/ExportPanel'

function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36)
}

export default function Workspace() {
    const navigate = useNavigate()
    const [activeTab, setActiveTab] = useState('form')
    const [previewMode, setPreviewMode] = useState('2d')
    const [plan, setPlan] = useState(null)
    const [loading, setLoading] = useState(false)
    const [loadingMessage, setLoadingMessage] = useState('')
    const [projectId, setProjectId] = useState(null)
    const [boundary, setBoundary] = useState(null)
    const [sessionId] = useState(() => generateSessionId())
    const [error, setError] = useState(null)

    const handleNewProject = () => {
        setPlan(null)
        setProjectId(null)
        setBoundary(null)
        setPreviewMode('2d')
        setLoading(false)
        setError(null)
    }

    // Step 1: Create a project if we don't have one yet
    const ensureProject = async (totalArea) => {
        if (projectId) return projectId

        const res = await fetch('/api/project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, total_area: totalArea }),
        })

        if (!res.ok) throw new Error('Failed to create project')
        const data = await res.json()
        const newId = data.project_id
        setProjectId(newId)
        return newId
    }

    // Unified generate function shared by both Chat and Form
    const handleGenerate = async (rooms, totalArea) => {
        setLoading(true)
        setLoadingMessage('Creating project...')
        setError(null)
        try {
            // Ensure project exists
            const pid = await ensureProject(totalArea)

            setLoadingMessage('Generating your floor plan...')

            // Call generate-floorplan with the correct schema
            const res = await fetch('/api/generate-floorplan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: pid,
                    rooms: rooms,
                    total_area: totalArea,
                    boundary_polygon: boundary || null,
                }),
            })

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}))
                throw new Error(errData.detail || `Server error ${res.status}`)
            }

            const data = await res.json()
            if (data.plan) {
                setPlan(data.plan)
                setProjectId(data.project_id || pid)
                setPreviewMode('2d')
            }
        } catch (err) {
            console.error('Generation failed:', err)
            setError(err.message || 'Generation failed. Please try again.')
        } finally {
            setLoading(false)
            setLoadingMessage('')
        }
    }

    // Boundary upload handler — Phase 1 pipeline
    const handleBoundaryUpload = async (file) => {
        setLoadingMessage('Uploading boundary file...')
        setLoading(true)
        setError(null)
        try {
            // Optionally link to project
            let pid = projectId
            if (!pid) {
                try { pid = await ensureProject(1200) } catch { pid = null }
            }

            // Step 1: Upload file → get file_id
            const uploadForm = new FormData()
            uploadForm.append('file', file)
            if (pid) uploadForm.append('project_id', pid)
            uploadForm.append('scale', '1.0')

            const uploadRes = await fetch('/api/upload-boundary', {
                method: 'POST',
                body: uploadForm,
            })

            if (!uploadRes.ok) {
                const errData = await uploadRes.json().catch(() => ({}))
                throw new Error(errData.detail || 'File upload failed')
            }

            const uploadData = await uploadRes.json()
            const fileId = uploadData.file_id

            // Step 2: Extract boundary polygon
            setLoadingMessage('Extracting boundary...')
            const extractRes = await fetch(`/api/extract-boundary/${fileId}?scale=1.0`)

            if (!extractRes.ok) {
                const errData = await extractRes.json().catch(() => ({}))
                throw new Error(errData.detail || 'Boundary extraction failed')
            }

            const extractData = await extractRes.json()

            // Step 3: Compute buildable footprint (India MVP setback)
            setLoadingMessage('Computing buildable area...')
            const footprintRes = await fetch(`/api/buildable-footprint/${fileId}?region=india_mvp`, {
                method: 'POST',
            })

            if (!footprintRes.ok) {
                const errData = await footprintRes.json().catch(() => ({}))
                throw new Error(errData.detail || 'Buildable footprint computation failed')
            }

            const footprintData = await footprintRes.json()

            setBoundary(footprintData.usable_polygon)
            setLoading(false)
            return {
                boundary: extractData.boundary_polygon,
                usable_polygon: footprintData.usable_polygon,
                area: footprintData.boundary_area,
                usable_area: footprintData.usable_area,
                setback: footprintData.setback_applied,
                coverage_ratio: footprintData.coverage_ratio,
                preview_url: footprintData.preview_url,
                num_vertices: extractData.num_vertices,
            }
        } catch (err) {
            console.error('Boundary processing failed:', err)
            setError(err.message || 'Boundary processing failed.')
        }
        setLoading(false)
        return null
    }

    return (
        <div className="workspace" style={{ gridTemplateRows: 'auto 1fr' }}>
            {/* Top Navigation */}
            <nav className="workspace-nav">
                <Link to="/" className="logo">
                    <span className="logo-icon">
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                    </span>
                    NakshaNirman
                </Link>
                <div className="workspace-nav-right">
                    <span className="project-label">Workspace</span>
                    <button className="btn btn-secondary btn-sm" onClick={handleNewProject}>
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                        </svg>
                        New Project
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={() => navigate('/')}>
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z" />
                        </svg>
                        Home
                    </button>
                </div>
            </nav>

            {/* Sidebar - Input Panel */}
            <div className="sidebar">
                <div className="sidebar-header">
                    <span className="sidebar-title">Input</span>
                </div>
                <div className="tab-switcher">
                    <button
                        className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
                        onClick={() => setActiveTab('chat')}
                    >
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        Chat
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'form' ? 'active' : ''}`}
                        onClick={() => setActiveTab('form')}
                    >
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        Form
                    </button>
                </div>
                <div className="sidebar-content">
                    {activeTab === 'chat' ? (
                        <ChatInterface
                            onGenerate={handleGenerate}
                            onBoundaryUpload={handleBoundaryUpload}
                            loading={loading}
                            projectId={projectId}
                        />
                    ) : (
                        <FormInterface
                            onGenerate={handleGenerate}
                            onBoundaryUpload={handleBoundaryUpload}
                            boundary={boundary}
                            loading={loading}
                        />
                    )}
                </div>
            </div>

            {/* Preview Panel */}
            <div className="preview-panel">
                <div className="preview-header">
                    <div className="preview-tabs">
                        <button
                            className={`preview-tab ${previewMode === '2d' ? 'active' : ''}`}
                            onClick={() => setPreviewMode('2d')}
                        >
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
                            </svg>
                            2D Plan
                        </button>
                        <button
                            className={`preview-tab ${previewMode === '3d' ? 'active' : ''}`}
                            onClick={() => setPreviewMode('3d')}
                        >
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                            </svg>
                            3D View
                        </button>
                    </div>
                    {projectId && <ExportPanel projectId={projectId} onGenerate3D={() => setPreviewMode('3d')} />}
                </div>
                <div className="preview-content">
                    {loading && (
                        <div className="loading-overlay">
                            <div className="spinner"></div>
                            <span className="loading-text">{loadingMessage || 'Processing...'}</span>
                        </div>
                    )}
                    {error && !loading && (
                        <div style={{
                            position: 'absolute', bottom: '1rem', left: '50%', transform: 'translateX(-50%)',
                            padding: '0.65rem 1.25rem', background: 'var(--error-bg)', border: '1px solid #fecaca',
                            borderRadius: 'var(--radius-full)', fontSize: '0.82rem', color: '#991b1b',
                            zIndex: 10, maxWidth: '90%', textAlign: 'center',
                        }}>
                            {error}
                        </div>
                    )}
                    {previewMode === '2d' ? (
                        plan ? <PlanPreview plan={plan} /> : (
                            !loading && (
                                <div className="preview-empty">
                                    <div className="preview-empty-icon">
                                        <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                                        </svg>
                                    </div>
                                    <h3>No Floor Plan Yet</h3>
                                    <p>Use the chat or form on the left to generate your first floor plan</p>
                                </div>
                            )
                        )
                    ) : (
                        projectId ? <Viewer3D projectId={projectId} /> : (
                            !loading && (
                                <div className="preview-empty">
                                    <div className="preview-empty-icon">
                                        <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                                        </svg>
                                    </div>
                                    <h3>3D View</h3>
                                    <p>Generate a floor plan first, then switch here to view it in 3D</p>
                                </div>
                            )
                        )
                    )}
                </div>
            </div>
        </div>
    )
}
