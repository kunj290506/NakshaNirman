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
    const { rooms, selectedRoomId, layout, zones } = state

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
        </div>
    )
}
