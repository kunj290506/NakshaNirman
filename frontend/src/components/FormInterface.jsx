import { useState, useRef } from 'react'
import RequirementsForm from './RequirementsForm'

const ROOM_TYPES = [
    { key: 'master_bedroom', label: 'Master Bedroom', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z' },
    { key: 'bedroom', label: 'Bedroom', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z' },
    { key: 'bathroom', label: 'Bathroom', icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10' },
    { key: 'kitchen', label: 'Kitchen', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
    { key: 'living', label: 'Living Room', icon: 'M4 6h16M4 10h16M4 14h16M4 18h16' },
    { key: 'dining', label: 'Dining Room', icon: 'M12 8v13m0-13V6a4 4 0 00-4-4H6.5a2.5 2.5 0 000 5H8m4-3v3m0 0h4.5a2.5 2.5 0 000-5H14a4 4 0 00-4 4' },
    { key: 'study', label: 'Study', icon: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253' },
    { key: 'pooja', label: 'Pooja Room', icon: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z' },
    { key: 'store', label: 'Store Room', icon: 'M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4' },
    { key: 'garage', label: 'Garage', icon: 'M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4' },
    { key: 'balcony', label: 'Balcony', icon: 'M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z' },
    { key: 'hallway', label: 'Hallway', icon: 'M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7' },
]

export default function FormInterface({ onGenerate, onBoundaryUpload, boundary, boundaryData, loading, backendHealthy = true, onCheckBackend }) {
    const [step, setStep] = useState(0)
    const [inputMode, setInputMode] = useState('plot') // 'plot' or 'boundary'
    const [totalArea, setTotalArea] = useState(1200)
    const [selectedRooms, setSelectedRooms] = useState({
        master_bedroom: { selected: true, qty: 1 },
        bedroom: { selected: true, qty: 1 },
        bathroom: { selected: true, qty: 1 },
        kitchen: { selected: true, qty: 1 },
        living: { selected: true, qty: 1 },
        dining: { selected: true, qty: 1 },
    })
    const fileInputRef = useRef(null)
    const [requirements, setRequirements] = useState(null)

    const toggleRoom = (key) => {
        setSelectedRooms(prev => ({
            ...prev,
            [key]: {
                ...prev[key],
                selected: !prev[key]?.selected,
                qty: prev[key]?.qty || 1,
            },
        }))
    }

    const setQty = (key, qty) => {
        setSelectedRooms(prev => ({
            ...prev,
            [key]: { ...prev[key], qty: Math.max(1, parseInt(qty) || 1) },
        }))
    }

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0]
        if (file) await onBoundaryUpload(file)
    }

    const handleGenerate = () => {
        if (loading) return
        const rooms = Object.entries(selectedRooms)
            .filter(([_, v]) => v.selected)
            .map(([key, v]) => ({
                room_type: key,
                quantity: v.qty || 1,
            }))
        onGenerate(rooms, totalArea, requirements)
    }

    const selectedCount = Object.values(selectedRooms).filter(v => v.selected).length
    const totalRooms = Object.entries(selectedRooms)
        .filter(([_, v]) => v.selected)
        .reduce((sum, [_, v]) => sum + (v.qty || 1), 0)

    return (
        <div style={{ paddingTop: '0.5rem' }}>
            {/* Step indicator */}
            <div className="step-indicator">
                {[0, 1, 2].map(s => (
                    <div
                        key={s}
                        className={`step-dot ${s === step ? 'active' : s < step ? 'done' : ''}`}
                    />
                ))}
            </div>

            {/* Step labels */}
            <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: '0.65rem', color: 'var(--text-muted)',
                marginBottom: '1.25rem', marginTop: '0.3rem',
                fontWeight: 500,
            }}>
                <span style={{ color: step === 0 ? 'var(--accent)' : step > 0 ? 'var(--success)' : undefined }}>Plot / Boundary</span>
                <span style={{ color: step === 1 ? 'var(--accent)' : step > 1 ? 'var(--success)' : undefined }}>Rooms</span>
                <span style={{ color: step === 2 ? 'var(--accent)' : undefined }}>Generate</span>
            </div>

            {/* Step 0: Plot OR Boundary (toggle) */}
            {step === 0 && (
                <div className="form-section">
                    {/* Toggle: Plot vs Boundary */}
                    <div style={{
                        display: 'flex', gap: '0', marginBottom: '1.25rem',
                        border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', overflow: 'hidden',
                    }}>
                        <button
                            onClick={() => setInputMode('plot')}
                            style={{
                                flex: 1, padding: '0.6rem 0.75rem',
                                border: 'none', cursor: 'pointer',
                                fontWeight: 600, fontSize: '0.82rem',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
                                background: inputMode === 'plot' ? 'var(--accent)' : 'var(--bg-primary)',
                                color: inputMode === 'plot' ? 'white' : 'var(--text-secondary)',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z" />
                            </svg>
                            Enter Area
                        </button>
                        <button
                            onClick={() => setInputMode('boundary')}
                            style={{
                                flex: 1, padding: '0.6rem 0.75rem',
                                border: 'none', cursor: 'pointer',
                                fontWeight: 600, fontSize: '0.82rem',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
                                background: inputMode === 'boundary' ? 'var(--accent)' : 'var(--bg-primary)',
                                color: inputMode === 'boundary' ? 'white' : 'var(--text-secondary)',
                                borderLeft: '1px solid var(--border)',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            Upload DXF
                        </button>
                    </div>

                    <div style={{ marginBottom: '0.8rem' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.45rem', fontWeight: 600 }}>Requirements</div>
                        <RequirementsForm value={requirements} onChange={(v) => setRequirements(v)} />
                    </div>

                    {/* Plot mode: manual area entry */}
                    {inputMode === 'plot' && (
                        <>
                            <h3>
                                <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -3 }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z" />
                                </svg>
                                Plot Details
                            </h3>
                            <div className="form-group">
                                <label>Total Area (sq ft)</label>
                                <input
                                    className="form-input"
                                    type="number"
                                    value={totalArea}
                                    onChange={(e) => setTotalArea(parseInt(e.target.value) || 0)}
                                    min={100}
                                    max={50000}
                                    placeholder="Enter total plot area"
                                />
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                                    Typical: 600 (1BHK) / 1200 (2BHK) / 1800 (3BHK) / 2500+ (4BHK)
                                </div>
                            </div>
                            <button className="btn btn-primary" onClick={() => setStep(1)} style={{ width: '100%', marginTop: '1rem' }}
                                disabled={!totalArea || totalArea < 100}
                            >
                                Next
                                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        </>
                    )}

                    {/* Boundary mode: DXF upload */}
                    {inputMode === 'boundary' && (
                        <>
                            <h3>
                                <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -3 }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                Upload Plot Boundary
                            </h3>
                            <div
                                className="file-upload"
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    onChange={handleFileUpload}
                                    accept=".dxf,.png,.jpg,.jpeg"
                                    style={{ display: 'none' }}
                                />
                                <div className="file-upload-icon">
                                    <svg width="28" height="28" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                    </svg>
                                </div>
                                <p><strong>Click to upload</strong> your plot boundary</p>
                                <p style={{ fontSize: '0.72rem' }}>DXF file recommended. PNG/JPEG also accepted.</p>
                            </div>

                            {/* Show backend unreachable warning inside upload panel */}
                            {!backendHealthy && (
                                <div style={{ marginTop: '0.5rem', padding: '0.6rem', borderRadius: '8px', background: '#fee2e2', border: '1px solid #fecaca', color: '#7f1d1d', fontSize: '0.85rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div>Backend unreachable — unable to upload. Start the backend and retry.</div>
                                    <div>
                                        <button className="btn btn-secondary btn-sm" onClick={async (e) => { e.stopPropagation(); onCheckBackend && await onCheckBackend(); }}>
                                            Retry
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Boundary data summary after upload */}
                            {boundaryData && (
                                <div style={{
                                    marginTop: '0.75rem',
                                    padding: '0.85rem',
                                    background: 'var(--success-bg)',
                                    border: '1px solid #a7f3d0',
                                    borderRadius: 'var(--radius-sm)',
                                    fontSize: '0.8rem',
                                    color: '#065f46',
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontWeight: 600 }}>
                                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                        </svg>
                                        Boundary Extracted
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem 1rem', fontSize: '0.75rem' }}>
                                        <span>Plot area: <strong>{boundaryData.area?.toFixed(1)} sq.m</strong></span>
                                        <span>Vertices: <strong>{boundaryData.num_vertices}</strong></span>
                                        <span>Usable area: <strong>{boundaryData.usable_area?.toFixed(1)} sq.m</strong></span>
                                        <span>Setback: <strong>{boundaryData.setback}m</strong></span>
                                    </div>
                                </div>
                            )}

                            {loading && (
                                <div style={{
                                    marginTop: '0.75rem', padding: '0.65rem 0.85rem',
                                    textAlign: 'center', fontSize: '0.82rem', color: 'var(--text-muted)',
                                }}>
                                    <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2, marginBottom: '0.4rem' }}></div>
                                    Processing boundary...
                                </div>
                            )}

                            <button className="btn btn-primary" onClick={() => setStep(1)} style={{ width: '100%', marginTop: '1rem' }}
                                disabled={!boundaryData || loading}
                            >
                                Next
                                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        </>
                    )}
                </div>
            )}

            {/* Step 1: Rooms */}
            {step === 1 && (
                <div className="form-section">
                    <h3>
                        <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -3 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                        </svg>
                        Rooms and Amenities
                    </h3>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                        Select rooms and set quantities. Tap to toggle, use the number field to adjust count.
                    </div>
                    <div className="amenity-grid">
                        {ROOM_TYPES.map(r => {
                            const state = selectedRooms[r.key]
                            return (
                                <div
                                    key={r.key}
                                    className={`amenity-item ${state?.selected ? 'selected' : ''}`}
                                    onClick={() => toggleRoom(r.key)}
                                >
                                    <input
                                        type="checkbox"
                                        checked={!!state?.selected}
                                        onChange={() => toggleRoom(r.key)}
                                        onClick={(e) => e.stopPropagation()}
                                    />
                                    <span>{r.label}</span>
                                    {state?.selected && (
                                        <input
                                            className="form-input amenity-qty"
                                            type="number"
                                            value={state.qty || 1}
                                            onClick={(e) => e.stopPropagation()}
                                            onChange={(e) => setQty(r.key, e.target.value)}
                                            min={1}
                                            max={10}
                                        />
                                    )}
                                </div>
                            )
                        })}
                    </div>

                    <div style={{
                        marginTop: '1rem', padding: '0.5rem 0.75rem',
                        background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)',
                        fontSize: '0.78rem', color: 'var(--text-secondary)',
                    }}>
                        {selectedCount} room types selected / {totalRooms} total rooms
                    </div>

                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                        <button className="btn btn-secondary" onClick={() => setStep(0)} style={{ flex: 1 }}>
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                            </svg>
                            Back
                        </button>
                        <button className="btn btn-primary" onClick={() => setStep(2)} style={{ flex: 1 }}
                            disabled={selectedCount === 0}
                        >
                            Next
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                            </svg>
                        </button>
                    </div>
                </div>
            )}

            {/* Step 2: Review & Generate */}
            {step === 2 && (
                <div className="form-section">
                    <h3>
                        <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: -3 }}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Review and Generate
                    </h3>

                    <div style={{
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-md)',
                        padding: '1rem',
                        marginBottom: '1.25rem',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.85rem', paddingBottom: '0.85rem', borderBottom: '1px solid var(--border)' }}>
                            <div>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                                    {inputMode === 'boundary' ? 'Plot Area' : 'Total Area'}
                                </div>
                                <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--accent)' }}>
                                    {inputMode === 'boundary' && boundaryData
                                        ? `${boundaryData.area?.toFixed(1)} sq.m`
                                        : `${totalArea} sq ft`}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Input Mode</div>
                                <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                                    {inputMode === 'boundary'
                                        ? `DXF (${boundaryData?.num_vertices || '?'} vertices)`
                                        : 'Manual Area'}
                                </div>
                            </div>
                        </div>

                        {inputMode === 'boundary' && boundaryData && (
                            <div style={{ marginBottom: '0.85rem', paddingBottom: '0.85rem', borderBottom: '1px solid var(--border)' }}>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.3rem' }}>Buildable Footprint</div>
                                <div style={{ fontSize: '0.85rem' }}>
                                    Usable area: <strong>{boundaryData.usable_area?.toFixed(1)} sq.m</strong>
                                    <span style={{ color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                                        (setback: {boundaryData.setback}m, coverage: {(boundaryData.coverage_ratio * 100).toFixed(1)}%)
                                    </span>
                                </div>
                            </div>
                        )}

                        <div>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.4rem' }}>
                                Rooms ({totalRooms})
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                                {Object.entries(selectedRooms)
                                    .filter(([_, v]) => v.selected)
                                    .map(([key, v]) => {
                                        const room = ROOM_TYPES.find(r => r.key === key)
                                        return (
                                            <span key={key} style={{
                                                padding: '0.25rem 0.6rem',
                                                background: 'var(--bg-primary)',
                                                border: '1px solid var(--border)',
                                                borderRadius: 'var(--radius-full)',
                                                fontSize: '0.75rem',
                                                fontWeight: 500,
                                            }}>
                                                {room?.label} x{v.qty}
                                            </span>
                                        )
                                    })}
                            </div>
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button className="btn btn-secondary" onClick={() => setStep(1)} style={{ flex: 1 }} disabled={loading}>
                            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                            </svg>
                            Back
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleGenerate}
                            style={{ flex: 2, position: 'relative' }}
                            disabled={loading}
                        >
                            {loading ? (
                                <>
                                    <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2, marginBottom: 0, borderTopColor: 'white', borderColor: 'rgba(255,255,255,0.3)' }}></div>
                                    Generating...
                                </>
                            ) : (
                                <>
                                    Generate Floor Plan
                                    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                                    </svg>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
