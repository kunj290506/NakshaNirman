import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import ChatInterface from '../components/ChatInterface'
import FormInterface from '../components/FormInterface'
import PlanPreview from '../components/PlanPreview'
import BoundaryPreview from '../components/BoundaryPreview'
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
    const [boundaryData, setBoundaryData] = useState(null)  // full Phase 1 data
    const [sessionId] = useState(() => generateSessionId())
    const [error, setError] = useState(null)

    const [backendHealthy, setBackendHealthy] = useState(true)
    const backendUrl = '/api'

    useEffect(() => {
        let mounted = true
        const check = async () => {
            try {
                const res = await fetch('/api/health', { cache: 'no-store' })
                if (!mounted) return
                setBackendHealthy(res.ok)
            } catch (e) {
                if (!mounted) return
                setBackendHealthy(false)
            }
        }
        check()
        const t = setInterval(check, 10000)
        return () => { mounted = false; clearInterval(t) }
    }, [])

    const handleNewProject = () => {
        setPlan(null)
        setProjectId(null)
        setBoundary(null)
        setBoundaryData(null)
        setPreviewMode('2d')
        setLoading(false)
        setError(null)
    }

    const checkBackend = async () => {
        try {
            const r = await fetch('/api/health')
            setBackendHealthy(r.ok)
            return r.ok
        } catch (e) {
            setBackendHealthy(false)
            return false
        }
    }

    // Small helper that rejects if fetch doesn't complete within `ms` milliseconds
    const fetchWithTimeout = (url, opts = {}, ms = 15000) => {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Request timed out')), ms)
            fetch(url, opts).then(res => {
                clearTimeout(timer)
                resolve(res)
            }).catch(err => {
                clearTimeout(timer)
                reject(err)
            })
        })
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
    const handleGenerate = async (rooms, totalArea, requirements = null) => {
        setLoading(true)
        setLoadingMessage('Creating project...')
        setError(null)
        try {
            // Ensure project exists
            const pid = await ensureProject(totalArea)

            // If requirements provided, store them linked to project
            if (requirements) {
                try {
                    await fetch('/api/requirements', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ...requirements, project_id: pid }),
                    })
                } catch (err) {
                    console.warn('Storing requirements failed:', err)
                }
            }

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

            // Guard: ensure backend reachable before uploading
            if (!backendHealthy) {
                try {
                    const ping = await fetch('/api/health')
                    if (!ping.ok) throw new Error('unhealthy')
                    setBackendHealthy(true)
                } catch (err) {
                    setError('Backend is unreachable. Please start the backend server.')
                    setLoading(false)
                    return null
                }
            }

            // Step 1: Upload file → get file_id
            const uploadForm = new FormData()
            uploadForm.append('file', file)
            if (pid) uploadForm.append('project_id', pid)
            uploadForm.append('scale', '1.0')

            const uploadRes = await fetchWithTimeout('/api/upload-boundary', {
                method: 'POST',
                body: uploadForm,
            }, 20000)

            if (!uploadRes.ok) {
                const errData = await uploadRes.json().catch(() => ({}))
                throw new Error(errData.detail || 'File upload failed')
            }

            const uploadData = await uploadRes.json()
            const fileId = uploadData.file_id

            // Step 2: Extract boundary polygon
            setLoadingMessage('Extracting boundary...')
            const extractRes = await fetchWithTimeout(`/api/extract-boundary/${fileId}?scale=1.0`, {}, 15000)

            if (!extractRes.ok) {
                const errData = await extractRes.json().catch(() => ({}))
                throw new Error(errData.detail || 'Boundary extraction failed')
            }

            const extractData = await extractRes.json()

            // Step 3: Compute buildable footprint (India MVP setback)
            setLoadingMessage('Computing buildable area...')
            const footprintRes = await fetchWithTimeout(`/api/buildable-footprint/${fileId}?region=india_mvp`, {
                method: 'POST',
            }, 20000)

            if (!footprintRes.ok) {
                const errData = await footprintRes.json().catch(() => ({}))
                throw new Error(errData.detail || 'Buildable footprint computation failed')
            }

            const footprintData = await footprintRes.json()

            const resultData = {
                boundary: extractData.boundary_polygon,
                usable_polygon: footprintData.usable_polygon,
                area: footprintData.boundary_area,
                usable_area: footprintData.usable_area,
                setback: footprintData.setback_applied,
                coverage_ratio: footprintData.coverage_ratio,
                preview_url: footprintData.preview_url,
                num_vertices: extractData.num_vertices,
                file_id: fileId,
            }

            setBoundary(footprintData.usable_polygon)
            setBoundaryData(resultData)
            setPreviewMode('boundary')
            setLoading(false)
            return resultData
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
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '0.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.82rem' }}>
                            <span style={{ width: 10, height: 10, borderRadius: 99, background: backendHealthy ? '#10b981' : '#ef4444', display: 'inline-block' }} />
                            <span style={{ color: backendHealthy ? '#065f46' : '#7f1d1d', fontWeight: 600 }}>{backendHealthy ? 'Backend OK' : 'Backend Down'}</span>
                        </div>
                        {!backendHealthy && (
                            <button className="btn btn-secondary btn-sm" onClick={() => checkBackend()}>
                                Retry
                            </button>
                        )}
                    </div>
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
                            boundaryData={boundaryData}
                            loading={loading}
                            backendHealthy={backendHealthy}
                            onCheckBackend={async () => {
                                try {
                                    const r = await fetch('/api/health')
                                    setBackendHealthy(r.ok)
                                    return r.ok
                                } catch (e) {
                                    setBackendHealthy(false)
                                    return false
                                }
                            }}
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
                        {boundaryData && (
                            <button
                                className={`preview-tab ${previewMode === 'boundary' ? 'active' : ''}`}
                                onClick={() => setPreviewMode('boundary')}
                            >
                                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -2 }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm0 8a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z" />
                                </svg>
                                Boundary
                            </button>
                        )}
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
                            boundaryData ? (
                                <BoundaryPreview boundaryData={boundaryData} />
                            ) : (
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
                        )
                    ) : previewMode === 'boundary' ? (
                        boundaryData ? (
                            <BoundaryPreview boundaryData={boundaryData} />
                        ) : (
                            !loading && (
                                <div className="preview-empty">
                                    <div className="preview-empty-icon">
                                        <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm0 8a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z" />
                                        </svg>
                                    </div>
                                    <h3>No Boundary Uploaded</h3>
                                    <p>Upload a DXF file to see the plot boundary with setback overlay</p>
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
