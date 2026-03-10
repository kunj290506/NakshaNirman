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
]

export default function FormInterface({ onGenerate, onBoundaryUpload, boundary, boundaryData, loading, backendHealthy = true, onCheckBackend }) {
    const [step, setStep] = useState(0)
    const [inputMode, setInputMode] = useState('plot') // 'plot' or 'boundary'
    const [totalArea, setTotalArea] = useState(1200)
    const [plotWidth, setPlotWidth] = useState(30)
    const [plotLength, setPlotLength] = useState(40)
    const [areaInputMode, setAreaInputMode] = useState('dimensions') // 'dimensions' or 'sqft'
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
    const [roomsManuallyEdited, setRoomsManuallyEdited] = useState(false)

    // Sync RequirementsForm values → room card selections whenever requirements change
    const handleRequirementsChange = (req) => {
        setRequirements(req)
        if (!req) return

        // Only sync RequirementsForm → room cards if user hasn't manually edited rooms
        if (roomsManuallyEdited) return

        setSelectedRooms(prev => {
            const next = { ...prev }

            // Sync bedrooms: 1 master + rest as regular bedrooms
            const totalBed = parseInt(req.bedrooms) || 0
            if (totalBed >= 1) {
                next.master_bedroom = { selected: true, qty: 1 }
                if (totalBed > 1) {
                    next.bedroom = { selected: true, qty: totalBed - 1 }
                } else {
                    next.bedroom = { selected: false, qty: 1 }
                }
            } else {
                next.master_bedroom = { selected: false, qty: 1 }
                next.bedroom = { selected: false, qty: 1 }
            }

            // Sync bathrooms — only update if user explicitly set bathrooms > 0
            const totalBath = parseInt(req.bathrooms) || 0
            if (totalBath > 0) {
                next.bathroom = { selected: true, qty: totalBath }
            }

            // Sync kitchen
            const totalKitchen = parseInt(req.kitchen) || 0
            if (totalKitchen >= 1) {
                next.kitchen = { selected: true, qty: totalKitchen }
            } else {
                next.kitchen = { selected: false, qty: 1 }
            }

            // Sync extras from checkboxes
            next.balcony = { selected: !!req.balcony, qty: prev.balcony?.qty || 1 }
            next.pooja = { selected: !!req.pooja_room, qty: prev.pooja?.qty || 1 }
            next.garage = { selected: !!req.parking, qty: prev.garage?.qty || 1 }

            return next
        })

        // Sync max_area to totalArea if user entered it
        if (req.max_area && req.max_area > 0) {
            setTotalArea(Math.round(req.max_area * 10.764))  // sq.m → sq.ft
        }
    }

    const toggleRoom = (key) => {
        setRoomsManuallyEdited(true)
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
        setRoomsManuallyEdited(true)
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
        // Merge requirements data so floors + extras always reach the backend
        const mergedRequirements = {
            ...(requirements || {}),
            floors: requirements?.floors || 1,
            bedrooms: rooms.filter(r => r.room_type === 'bedroom' || r.room_type === 'master_bedroom').reduce((s, r) => s + (r.quantity || 1), 0),
            bathrooms: rooms.filter(r => r.room_type === 'bathroom').reduce((s, r) => s + (r.quantity || 1), 0),
            plot_width: plotWidth || null,
            plot_length: plotLength || null,
        }
        onGenerate(rooms, totalArea, mergedRequirements)
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
                fontSize: '0.7rem', color: '#bbb',
                marginBottom: '1rem', marginTop: '0.25rem',
                fontWeight: 500,
            }}>
                <span style={{ color: step >= 0 ? '#111' : undefined }}>Plot</span>
                <span style={{ color: step >= 1 ? '#111' : undefined }}>Rooms</span>
                <span style={{ color: step >= 2 ? '#111' : undefined }}>Generate</span>
            </div>

            {/* Step 0: Plot OR Boundary (toggle) */}
            {step === 0 && (
                <div className="form-section">
                    {/* Toggle: Plot vs Boundary */}
                    <div style={{
                        display: 'flex', gap: '0', marginBottom: '1.25rem',
                        border: '1px solid #e2e8f0', borderRadius: '10px', overflow: 'hidden',
                    }}>
                        <button
                            onClick={() => setInputMode('plot')}
                            style={{
                                flex: 1, padding: '0.6rem 0.75rem',
                                border: 'none', cursor: 'pointer',
                                fontWeight: 600, fontSize: '0.82rem',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
                                background: inputMode === 'plot' ? '#111' : '#fff',
                                color: inputMode === 'plot' ? 'white' : '#64748b',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            Enter Area
                        </button>
                        <button
                            onClick={() => setInputMode('boundary')}
                            style={{
                                flex: 1, padding: '0.6rem 0.75rem',
                                border: 'none', cursor: 'pointer',
                                fontWeight: 600, fontSize: '0.82rem',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
                                background: inputMode === 'boundary' ? '#111' : '#fff',
                                color: inputMode === 'boundary' ? 'white' : '#64748b',
                                borderLeft: '1px solid #e2e8f0',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            Upload DXF
                        </button>
                    </div>

                    <div style={{ marginBottom: '0.8rem' }}>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '0.45rem', fontWeight: 600 }}>Requirements</div>
                        <RequirementsForm value={requirements} onChange={handleRequirementsChange} />
                    </div>

                    {/* Plot mode: manual area entry */}
                    {inputMode === 'plot' && (
                        <>
                            <h3>Plot Details</h3>
                            <div className="form-group">
                                {/* Area input mode toggle */}
                                <div style={{ display: 'flex', gap: 0, marginBottom: '0.75rem',
                                    border: '1px solid #e2e8f0', borderRadius: 8, overflow: 'hidden' }}>
                                    <button
                                        onClick={() => setAreaInputMode('dimensions')}
                                        style={{
                                            flex: 1, padding: '0.5rem', border: 'none', cursor: 'pointer',
                                            fontSize: '0.78rem', fontWeight: 600,
                                            background: areaInputMode === 'dimensions' ? '#111' : '#fff',
                                            color: areaInputMode === 'dimensions' ? '#fff' : '#64748b',
                                        }}
                                    >Width × Length</button>
                                    <button
                                        onClick={() => setAreaInputMode('sqft')}
                                        style={{
                                            flex: 1, padding: '0.5rem', border: 'none', cursor: 'pointer',
                                            fontSize: '0.78rem', fontWeight: 600, borderLeft: '1px solid #e2e8f0',
                                            background: areaInputMode === 'sqft' ? '#111' : '#fff',
                                            color: areaInputMode === 'sqft' ? '#fff' : '#64748b',
                                        }}
                                    >Total Sq Ft</button>
                                </div>

                                {areaInputMode === 'dimensions' ? (
                                    <div>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '0.5rem',
                                            alignItems: 'center', marginBottom: '0.4rem' }}>
                                            <div>
                                                <label style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 600 }}>Width (ft)</label>
                                                <input
                                                    className="form-input"
                                                    type="number"
                                                    value={plotWidth}
                                                    onChange={(e) => {
                                                        const w = parseInt(e.target.value) || 0
                                                        setPlotWidth(w)
                                                        setTotalArea(w * plotLength)
                                                    }}
                                                    min={10} max={200} placeholder="30"
                                                />
                                            </div>
                                            <span style={{ textAlign: 'center', fontWeight: 700, color: '#94a3b8',
                                                paddingTop: '1.2rem' }}>×</span>
                                            <div>
                                                <label style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 600 }}>Length (ft)</label>
                                                <input
                                                    className="form-input"
                                                    type="number"
                                                    value={plotLength}
                                                    onChange={(e) => {
                                                        const l = parseInt(e.target.value) || 0
                                                        setPlotLength(l)
                                                        setTotalArea(plotWidth * l)
                                                    }}
                                                    min={10} max={200} placeholder="40"
                                                />
                                            </div>
                                        </div>
                                        <div style={{ fontSize: '0.72rem', color: '#10b981', fontWeight: 600, textAlign: 'center' }}>
                                            = {plotWidth * plotLength} sq ft total area
                                        </div>
                                    </div>
                                ) : (
                                    <div>
                                        <label style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 600 }}>Total Area (sq ft)</label>
                                        <input
                                            className="form-input"
                                            type="number"
                                            value={totalArea}
                                            onChange={(e) => {
                                                setTotalArea(parseInt(e.target.value) || 0)
                                                setPlotWidth(null)
                                                setPlotLength(null)
                                            }}
                                            min={100} max={50000} placeholder="1200"
                                        />
                                        <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.3rem' }}>
                                            2BHK: ~1200 sqft | 3BHK: ~1800 sqft | 4BHK: ~2500 sqft
                                        </div>
                                    </div>
                                )}
                            </div>
                            <button className="btn btn-primary" onClick={() => setStep(1)} style={{ width: '100%', marginTop: '1rem' }}
                                disabled={areaInputMode === 'dimensions'
                                    ? !plotWidth || !plotLength || plotWidth < 10 || plotLength < 10
                                    : !totalArea || totalArea < 100
                                }
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
                            <h3>Upload Plot Boundary</h3>
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
                                <div style={{ marginTop: '0.5rem', padding: '0.6rem', borderRadius: '10px', background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', fontSize: '0.82rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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
                                    background: '#ecfdf5',
                                    border: '1px solid #a7f3d0',
                                    borderRadius: '10px',
                                    fontSize: '0.8rem',
                                    color: '#065f46',
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontWeight: 600, color: '#065f46' }}>
                                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                        </svg>
                                        Boundary Extracted
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem 1rem', fontSize: '0.75rem', color: '#047857' }}>
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
                    <h3>Rooms</h3>
                    <div style={{ fontSize: '0.72rem', color: '#999', marginBottom: '0.65rem' }}>
                        Select rooms and adjust quantities.
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
                        marginTop: '1rem', padding: '0.55rem 0.8rem',
                        background: '#f8fafc', borderRadius: '8px',
                        fontSize: '0.78rem', color: '#475569',
                        border: '1px solid #e2e8f0',
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
                    <h3>Review</h3>

                    <div style={{
                        background: '#f8fafc',
                        border: '1px solid #e2e8f0',
                        borderRadius: '10px',
                        padding: '1rem',
                        marginBottom: '1.25rem',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.85rem', paddingBottom: '0.85rem', borderBottom: '1px solid #e2e8f0' }}>
                            <div>
                                <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600 }}>
                                    {inputMode === 'boundary' ? 'Plot Area' : 'Total Area'}
                                </div>
                                <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#111' }}>
                                    {inputMode === 'boundary' && boundaryData
                                        ? `${boundaryData.area?.toFixed(1)} sq.m`
                                        : `${totalArea} sq ft`}
                                </div>
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600 }}>Floors</div>
                                <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                                    {requirements?.floors || 1}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600 }}>Input Mode</div>
                                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#0f172a' }}>
                                    {inputMode === 'boundary'
                                        ? `DXF (${boundaryData?.num_vertices || '?'} vertices)`
                                        : 'Manual Area'}
                                </div>
                            </div>
                        </div>

                        {inputMode === 'boundary' && boundaryData && (
                            <div style={{ marginBottom: '0.85rem', paddingBottom: '0.85rem', borderBottom: '1px solid #e2e8f0' }}>
                                <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, marginBottom: '0.3rem' }}>Buildable Footprint</div>
                                <div style={{ fontSize: '0.85rem', color: '#0f172a' }}>
                                    Usable area: <strong>{boundaryData.usable_area?.toFixed(1)} sq.m</strong>
                                    <span style={{ color: '#94a3b8', marginLeft: '0.5rem' }}>
                                        (setback: {boundaryData.setback}m, coverage: {(boundaryData.coverage_ratio * 100).toFixed(1)}%)
                                    </span>
                                </div>
                            </div>
                        )}

                        <div>
                            <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, marginBottom: '0.4rem' }}>
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
                                                background: '#fff',
                                                border: '1px solid #e2e8f0',
                                                borderRadius: '99px',
                                                fontSize: '0.75rem',
                                                fontWeight: 500,
                                                color: '#475569',
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
