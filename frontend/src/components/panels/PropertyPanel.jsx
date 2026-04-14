/**
 * PropertyPanel — Right panel for room properties and editing.
 *
 * Shows:
 * - Selected room properties (name, type, dimensions, area)
 * - Room list with zone grouping
 * - Edit controls for position, size
 * - Layout summary stats
 */

import { useLayout, useLayoutActions } from '../../store/layoutStore'

const ZONE_COLORS = {
    public: '#e8f5e9',
    semi_private: '#fff8e1',
    private: '#e3f2fd',
    service: '#fce4ec',
    circulation: '#f3e5f5',
}

const ZONE_LABELS = {
    public: 'Public Zone',
    semi_private: 'Semi-Private',
    private: 'Private Zone',
    service: 'Service',
    circulation: 'Circulation',
}

export default function PropertyPanel() {
    const { state } = useLayout()
    const actions = useLayoutActions()
    const { rooms, selectedRoomId, layout, zones, designScore, architectNarrative } = state

    const selectedRoom = rooms.find(r => r.id === selectedRoomId)

    if (!layout) {
        return (
            <div className="property-panel">
                <div className="property-panel-header">
                    <h3>Properties</h3>
                </div>
                <div className="property-panel-empty">
                    <p>Generate a floor plan to see room properties here</p>
                </div>
            </div>
        )
    }

    return (
        <div className="property-panel">
            <div className="property-panel-header">
                <h3>Properties</h3>
                {state.isDirty && <span className="dirty-badge">Modified</span>}
            </div>

            {/* Generation Proof */}
            {(layout.layout_signature || layout.connectivity_checks) && (
                <div className="property-section">
                    <div className="property-section-title">Generation Proof</div>
                    {layout.layout_signature && (
                        <div style={{ fontSize: '0.78rem', color: '#334155', marginBottom: '0.45rem' }}>
                            Signature: <strong>{layout.layout_signature}</strong>
                        </div>
                    )}
                    {layout.connectivity_checks && (
                        <div style={{ display: 'grid', gap: '0.3rem' }}>
                            {Object.entries(layout.connectivity_checks).map(([name, passed]) => (
                                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                                    <span style={{ color: '#475569' }}>{name.replace(/_/g, ' ')}</span>
                                    <span style={{ color: passed ? '#16a34a' : '#dc2626', fontWeight: 700 }}>
                                        {passed ? 'PASS' : 'FAIL'}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Selected Room Details */}
            {selectedRoom ? (
                <div className="property-section">
                    <div className="property-section-title">
                        <span className="room-type-dot" style={{ background: ZONE_COLORS[selectedRoom.zone] || '#eee' }} />
                        {selectedRoom.name}
                    </div>
                    <div className="property-grid">
                        <label>Type</label>
                        <span>{selectedRoom.room_type}</span>
                        <label>Zone</label>
                        <span>{ZONE_LABELS[selectedRoom.zone] || selectedRoom.zone}</span>
                        <label>Width</label>
                        <div className="property-input-row">
                            <input
                                type="number"
                                value={selectedRoom.width}
                                step="0.5"
                                min="4"
                                onChange={e => actions.resizeRoom(selectedRoom.id, +e.target.value, selectedRoom.length)}
                            />
                            <span className="unit">ft</span>
                        </div>
                        <label>Length</label>
                        <div className="property-input-row">
                            <input
                                type="number"
                                value={selectedRoom.length}
                                step="0.5"
                                min="4"
                                onChange={e => actions.resizeRoom(selectedRoom.id, selectedRoom.width, +e.target.value)}
                            />
                            <span className="unit">ft</span>
                        </div>
                        <label>Area</label>
                        <span>{Math.round(selectedRoom.width * selectedRoom.length * 10) / 10} sq ft</span>
                        <label>Position</label>
                        <span>({selectedRoom.position?.x ?? '–'}, {selectedRoom.position?.y ?? '–'})</span>
                    </div>
                    <button
                        className="btn btn-secondary btn-sm property-deselect"
                        onClick={() => actions.selectRoom(null)}
                    >
                        Deselect
                    </button>
                </div>
            ) : (
                <div className="property-hint">
                    Click a room on the canvas to edit its properties
                </div>
            )}

            {/* Room List by Zone */}
            <div className="property-section">
                <div className="property-section-title">Rooms ({rooms.length})</div>
                <div className="room-list">
                    {zones.map(zone => (
                        <div key={zone.name} className="room-zone-group">
                            <div className="zone-header" style={{ borderLeft: `3px solid ${ZONE_COLORS[zone.name] || '#ccc'}` }}>
                                {ZONE_LABELS[zone.name] || zone.name}
                                <span className="zone-band">{zone.band}</span>
                            </div>
                            {(zone.rooms || []).map(roomName => {
                                const room = rooms.find(r => r.name === roomName)
                                if (!room) return null
                                return (
                                    <button
                                        key={room.id}
                                        className={`room-list-item${room.id === selectedRoomId ? ' selected' : ''}`}
                                        onClick={() => actions.selectRoom(room.id)}
                                    >
                                        <span className="room-list-name">{room.name}</span>
                                        <span className="room-list-area">{Math.round(room.width * room.length)} ft²</span>
                                    </button>
                                )
                            })}
                        </div>
                    ))}
                </div>
            </div>

            {/* Layout Summary */}
            {layout.area_summary && (
                <div className="property-section">
                    <div className="property-section-title">Layout Summary</div>
                    <div className="property-grid summary">
                        <label>Plot Area</label>
                        <span>{layout.area_summary?.plot_area ?? '–'} ft²</span>
                        <label>Built Area</label>
                        <span>{layout.area_summary?.built_area ?? '–'} ft²</span>
                        <label>Utilization</label>
                        <span>{layout.area_summary.utilization_percentage}</span>
                        <label>Circulation</label>
                        <span>{layout.area_summary.circulation_percentage}</span>
                    </div>
                </div>
            )}

            {/* Practical Fit Report */}
            {layout.quality_report && (
                <div className="property-section">
                    <div className="property-section-title">Practical Fit</div>
                    <div className="property-grid summary">
                        <label>Real-Life Score</label>
                        <span>{layout.real_life_score ?? layout.quality_report.score ?? '–'} / 100</span>
                        <label>Grade</label>
                        <span>{layout.quality_report.grade || '–'}</span>
                        <label>Gemini-Like</label>
                        <span>{layout.model_alignment?.gemini_like ?? '–'}</span>
                        <label>ChatGPT-Like</label>
                        <span>{layout.model_alignment?.chatgpt_like ?? '–'}</span>
                    </div>

                    {Array.isArray(layout.quality_report.findings) && layout.quality_report.findings.length > 0 && (
                        <div style={{ marginTop: '0.55rem' }}>
                            <div className="property-section-title" style={{ fontSize: '0.8rem' }}>Satisfied</div>
                            <div style={{ display: 'grid', gap: '0.25rem' }}>
                                {layout.quality_report.findings.slice(0, 4).map((item, idx) => (
                                    <div key={`${item}-${idx}`} className="score-issue-item bonus">✓ {item}</div>
                                ))}
                            </div>
                        </div>
                    )}

                    {Array.isArray(layout.quality_report.opportunities) && layout.quality_report.opportunities.length > 0 && (
                        <div style={{ marginTop: '0.55rem' }}>
                            <div className="property-section-title" style={{ fontSize: '0.8rem' }}>Need Improvement</div>
                            <div style={{ display: 'grid', gap: '0.25rem' }}>
                                {layout.quality_report.opportunities.slice(0, 4).map((item, idx) => (
                                    <div key={`${item}-${idx}`} className="score-issue-item warning">○ {item}</div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Design Score Card */}
            {designScore && (
                <div className="property-section score-card">
                    <div className="property-section-title">Design Quality</div>

                    {/* Score Ring */}
                    <div className="score-ring-container">
                        <svg width="100" height="100" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="42" fill="none" stroke="#e5e7eb" strokeWidth="8" />
                            <circle cx="50" cy="50" r="42" fill="none" stroke={
                                designScore.composite >= 80 ? '#22c55e' :
                                designScore.composite >= 60 ? '#eab308' : '#ef4444'
                            } strokeWidth="8" strokeDasharray={`${designScore.composite * 2.64} 264`}
                              strokeLinecap="round" transform="rotate(-90 50 50)"
                              style={{ transition: 'stroke-dasharray 0.8s ease' }} />
                            <text x="50" y="46" textAnchor="middle" fontSize="22" fontWeight="700" fill="#1f2937">
                                {designScore.composite}
                            </text>
                            <text x="50" y="62" textAnchor="middle" fontSize="12" fill="#6b7280">
                                {designScore.grade}
                            </text>
                        </svg>
                    </div>

                    {/* Breakdown Bars */}
                    {designScore.breakdown && (
                        <div className="score-breakdown">
                            {Object.entries(designScore.breakdown).map(([dim, info]) => (
                                <div key={dim} className="score-bar-row">
                                    <span className="score-bar-label">
                                        {dim.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                    </span>
                                    <div className="score-bar-track">
                                        <div
                                            className="score-bar-fill"
                                            style={{
                                                width: `${info.score}%`,
                                                background: info.score >= 80 ? '#22c55e' : info.score >= 60 ? '#eab308' : '#ef4444',
                                                transition: 'width 0.6s ease',
                                            }}
                                        />
                                    </div>
                                    <span className="score-bar-value">{info.score}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Climate + Family badges */}
                    {(designScore.climate_zone || designScore.family_type) && (
                        <div className="score-badges">
                            {designScore.climate_zone && (
                                <span className="badge badge-climate">{designScore.climate_zone}</span>
                            )}
                            {designScore.family_type && (
                                <span className="badge badge-family">
                                    {designScore.family_type.replace(/_/g, ' ')}
                                </span>
                            )}
                        </div>
                    )}

                    {/* Vastu Bonuses */}
                    {designScore.vastu_bonuses?.length > 0 && (
                        <div className="score-issues">
                            <div className="score-issues-title vastu-bonus">Vastu Highlights</div>
                            {designScore.vastu_bonuses.slice(0, 5).map((b, i) => (
                                <div key={i} className="score-issue-item bonus">✓ {b.message}</div>
                            ))}
                        </div>
                    )}

                    {/* Issues */}
                    {designScore.issues?.length > 0 && (
                        <div className="score-issues">
                            <div className="score-issues-title">Issues ({designScore.issues.length})</div>
                            {designScore.issues.slice(0, 6).map((issue, i) => (
                                <div key={i} className={`score-issue-item ${issue.severity || 'warning'}`}>
                                    {issue.severity === 'critical' ? '✗' : '○'} {issue.message}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Architect Narrative */}
            {architectNarrative && (
                <div className="property-section">
                    <div className="property-section-title">Architect's Note</div>
                    <div className="architect-narrative">
                        {architectNarrative.split('\n').filter(Boolean).map((line, i) => (
                            <p key={i}>{line}</p>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
