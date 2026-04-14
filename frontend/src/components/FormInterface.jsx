import { useMemo, useState } from 'react'

const EXTRA_OPTIONS = [
  { key: 'pooja', label: 'Pooja Room' },
  { key: 'study', label: 'Study' },
  { key: 'store', label: 'Store Room' },
  { key: 'balcony', label: 'Balcony' },
  { key: 'garage', label: 'Garage' },
  { key: 'utility', label: 'Utility' },
  { key: 'foyer', label: 'Foyer' },
  { key: 'staircase', label: 'Staircase' },
]

function toRooms(payload) {
  const rooms = [
    { room_type: 'master_bedroom', quantity: 1 },
    { room_type: 'living', quantity: 1 },
    { room_type: 'kitchen', quantity: 1 },
    { room_type: 'dining', quantity: 1 },
  ]

  if (payload.bedrooms > 1) {
    rooms.push({ room_type: 'bedroom', quantity: payload.bedrooms - 1 })
  }
  rooms.push({ room_type: 'bathroom', quantity: payload.bathrooms })

  payload.extras.forEach((extra) => rooms.push({ room_type: extra, quantity: 1 }))
  return rooms
}

export default function FormInterface({ onGenerate, loading }) {
  const [plotWidth, setPlotWidth] = useState(30)
  const [plotLength, setPlotLength] = useState(40)

  const [bedrooms, setBedrooms] = useState(2)
  const [bathrooms, setBathrooms] = useState(2)
  const [facing, setFacing] = useState('east')
  const [vastu, setVastu] = useState(true)
  const [extras, setExtras] = useState([])

  const area = useMemo(() => Number(plotWidth || 0) * Number(plotLength || 0), [plotWidth, plotLength])

  const dimensions = useMemo(
    () => ({
      width: Number(plotWidth || 0),
      length: Number(plotLength || 0),
    }),
    [plotWidth, plotLength],
  )

  const canProceed = dimensions.width > 15 && dimensions.length > 15 && bedrooms >= 1 && bedrooms <= 4

  const toggleExtra = (key) => {
    setExtras((prev) => (prev.includes(key) ? prev.filter((x) => x !== key) : [...prev, key]))
  }

  const submit = () => {
    if (loading || !canProceed) return

    const payload = {
      plot_width: dimensions.width,
      plot_length: dimensions.length,
      total_area: Math.round(dimensions.width * dimensions.length * 10) / 10,
      bedrooms: Number(bedrooms),
      bathrooms: Number(bathrooms || bedrooms),
      bathrooms_target: Number(bathrooms || bedrooms),
      engine_mode: 'practical_strict',
      facing,
      vastu,
      extras,
    }

    const rooms = toRooms(payload)
    onGenerate(rooms, payload.total_area, payload)
  }

  return (
    <div className="form-compact-shell">
      <div className="form-section form-section-dense">
        <h3>Basic Inputs</h3>

        <div className="form-compact-grid-2">
          <div className="form-group">
            <label className="form-label">Width (ft)</label>
            <input className="form-input" type="number" min={16} value={plotWidth} onChange={(e) => setPlotWidth(Number(e.target.value || 0))} />
          </div>
          <div className="form-group">
            <label className="form-label">Length (ft)</label>
            <input className="form-input" type="number" min={16} value={plotLength} onChange={(e) => setPlotLength(Number(e.target.value || 0))} />
          </div>
        </div>

        <div className="form-group form-summary-chip">
          <label className="form-label">Summary</label>
          <div>{dimensions.width}x{dimensions.length} • {Math.round(area)} sqft</div>
        </div>

        <div className="form-compact-grid-4">
          <div className="form-group">
            <label className="form-label">BHK</label>
            <select
              className="form-input"
              value={bedrooms}
              onChange={(e) => {
                const n = Number(e.target.value)
                setBedrooms(n)
                setBathrooms((prev) => (prev < n ? n : prev))
              }}
            >
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>{n}BHK</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Bath</label>
            <input className="form-input" type="number" min={1} max={6} value={bathrooms} onChange={(e) => setBathrooms(Number(e.target.value || bedrooms))} />
          </div>
          <div className="form-group">
            <label className="form-label">Facing</label>
            <select className="form-input" value={facing} onChange={(e) => setFacing(e.target.value)}>
              {['east', 'north', 'west', 'south'].map((dir) => (
                <option key={dir} value={dir}>{dir.charAt(0).toUpperCase() + dir.slice(1)}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Vastu</label>
            <select className="form-input" value={vastu.toString()} onChange={(e) => setVastu(e.target.value === 'true')}>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Extra Rooms</label>
          <div className="amenity-grid amenity-grid-dense">
            {EXTRA_OPTIONS.map((opt) => {
              const selected = extras.includes(opt.key)
              return (
                <button
                  key={opt.key}
                  type="button"
                  className={`amenity-item ${selected ? 'selected' : ''}`}
                  onClick={() => toggleExtra(opt.key)}
                >
                  <input type="checkbox" readOnly checked={selected} />
                  <span>{opt.label}</span>
                </button>
              )
            })}
          </div>
        </div>

        <button
          type='button'
          className="btn btn-primary"
          disabled={!canProceed || loading}
          onClick={submit}
          style={{ width: '100%' }}
        >
          {loading ? 'Generating...' : 'Generate Floor Plan'}
        </button>
      </div>
    </div>
  )
}
