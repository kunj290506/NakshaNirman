import { useMemo, useState } from 'react'

const EXTRA_OPTIONS = [
  { key: 'pooja', label: 'Pooja Room' },
  { key: 'study', label: 'Study' },
  { key: 'store', label: 'Store Room' },
  { key: 'balcony', label: 'Balcony' },
  { key: 'garage', label: 'Garage' },
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
  const [step, setStep] = useState(1)
  const [mode, setMode] = useState('dimensions')

  const [plotWidth, setPlotWidth] = useState(30)
  const [plotLength, setPlotLength] = useState(40)
  const [totalSqft, setTotalSqft] = useState(1200)

  const [bedrooms, setBedrooms] = useState(2)
  const [bathrooms, setBathrooms] = useState(2)
  const [engineMode, setEngineMode] = useState('gnn_advanced')
  const [facing, setFacing] = useState('east')
  const [vastu, setVastu] = useState(true)
  const [extras, setExtras] = useState([])

  const area = useMemo(() => {
    if (mode === 'dimensions') return Number(plotWidth || 0) * Number(plotLength || 0)
    return Number(totalSqft || 0)
  }, [mode, plotWidth, plotLength, totalSqft])

  const dimensions = useMemo(() => {
    if (mode === 'dimensions') {
      return {
        width: Number(plotWidth || 0),
        length: Number(plotLength || 0),
      }
    }
    const w = Math.sqrt(area * 0.75)
    const l = area / Math.max(w, 1)
    return { width: Math.round(w * 10) / 10, length: Math.round(l * 10) / 10 }
  }, [mode, plotWidth, plotLength, area])

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
      engine_mode: engineMode,
      facing,
      vastu,
      extras,
    }

    const rooms = toRooms(payload)
    onGenerate(rooms, payload.total_area, payload)
  }

  return (
    <div style={{ display: 'grid', gap: '0.9rem' }}>
      <div className="form-section">
        <h3>Step 1: Plot + BHK</h3>

        <div className="form-group">
          <label className="form-label">Plot Input</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            <button className={`btn btn-secondary ${mode === 'dimensions' ? 'active' : ''}`} type='button' onClick={() => setMode('dimensions')}>
              Width x Length
            </button>
            <button className={`btn btn-secondary ${mode === 'sqft' ? 'active' : ''}`} type='button' onClick={() => setMode('sqft')}>
              Total Sqft
            </button>
          </div>
        </div>

        {mode === 'dimensions' ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem' }}>
            <div className="form-group">
              <label className="form-label">Plot Width (ft)</label>
              <input className="form-input" type="number" min={16} value={plotWidth} onChange={(e) => setPlotWidth(Number(e.target.value || 0))} />
            </div>
            <div className="form-group">
              <label className="form-label">Plot Length (ft)</label>
              <input className="form-input" type="number" min={16} value={plotLength} onChange={(e) => setPlotLength(Number(e.target.value || 0))} />
            </div>
          </div>
        ) : (
          <div className="form-group">
            <label className="form-label">Total Area (sqft)</label>
            <input className="form-input" type="number" min={300} value={totalSqft} onChange={(e) => setTotalSqft(Number(e.target.value || 0))} />
          </div>
        )}

        <div className="form-group">
          <label className="form-label">BHK Type</label>
          <select 
            className="form-input" 
            value={bedrooms} 
            onChange={(e) => {
              const n = Number(e.target.value);
              setBedrooms(n);
              setBathrooms((prev) => (prev < n ? n : prev));
            }}
          >
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>{n}BHK</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Bathrooms</label>
          <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 40px', gap: '0.45rem', alignItems: 'center' }}>
            <button
              type='button'
              className='btn btn-secondary'
              onClick={() => setBathrooms((v) => Math.max(1, v - 1))}
            >
              -
            </button>
            <input className="form-input" type="number" min={1} max={6} value={bathrooms} onChange={(e) => setBathrooms(Number(e.target.value || bedrooms))} />
            <button
              type='button'
              className='btn btn-secondary'
              onClick={() => setBathrooms((v) => Math.min(6, v + 1))}
            >
              +
            </button>
          </div>
          <div style={{ fontSize: '0.78rem', color: '#64748B' }}>(1 per bedroom recommended)</div>
        </div>

        <div style={{ fontSize: '0.8rem', color: '#666' }}>
          Approx area: <strong>{Math.round(area)}</strong> sqft
        </div>

        <button type='button' className="btn btn-primary" disabled={!canProceed} onClick={() => setStep(2)}>
          Next →
        </button>
      </div>

      {step >= 2 && (
        <div className="form-section">
          <h3>Step 2: Preferences</h3>

          <div style={{ border: '1px solid #e2e8f0', borderRadius: '10px', padding: '0.65rem 0.75rem', background: '#f8fafc' }}>
            <div style={{ fontSize: '0.82rem', color: '#475569' }}>Summary</div>
            <div style={{ fontWeight: 700, color: '#0f172a' }}>
              {dimensions.width} ft x {dimensions.length} ft • {bedrooms}BHK • {bathrooms} Bath
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Layout Engine</label>
            <select className="form-input" value={engineMode} onChange={(e) => setEngineMode(e.target.value)}>
              <option value="gnn_advanced">GNN Advanced (Recommended)</option>
              <option value="standard">Standard Stable</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Facing</label>
            <select className="form-input" value={facing} onChange={(e) => setFacing(e.target.value)}>
              {['east', 'north', 'west', 'south'].map((dir) => (
                <option key={dir} value={dir}>
                  {dir === 'east' ? 'East (Recommended)' : dir.charAt(0).toUpperCase() + dir.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Vastu Compliance</label>
            <select className="form-input" value={vastu.toString()} onChange={(e) => setVastu(e.target.value === 'true')}>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Extra Rooms (Multi-select)</label>
            <select 
              className="form-input" 
              multiple 
              value={extras}
              onChange={(e) => {
                const values = Array.from(e.target.selectedOptions, option => option.value);
                setExtras(values);
              }}
              style={{ height: '100px', padding: '0.5rem' }}
            >
              {EXTRA_OPTIONS.map((opt) => (
                <option key={opt.key} value={opt.key}>
                  {opt.label}
                </option>
              ))}
            </select>
            <div style={{ fontSize: '0.78rem', color: '#64748B', marginTop: '4px' }}>Hold Ctrl/Cmd to select multiple</div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.55rem' }}>
            <button type='button' className='btn btn-secondary' onClick={() => setStep(1)}>← Back</button>
            <button type='button' className="btn btn-primary" disabled={!canProceed || loading} onClick={submit}>
              {loading ? 'Generating...' : 'Generate Floor Plan'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
