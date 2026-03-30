import { useState } from 'react'

const EXTRAS = [
  { id: 'pooja', label: 'Pooja Room' },
  { id: 'study', label: 'Study Room' },
  { id: 'garage', label: 'Garage' },
  { id: 'balcony', label: 'Balcony' },
  { id: 'store', label: 'Store Room' },
]

export default function FloorPlanForm({ onGenerate, loading }) {
  const [formData, setFormData] = useState({
    plot_width: 30,
    plot_length: 40,
    bedrooms: 2,
    facing: 'south',
    extras: [],
  })

  function handleChange(e) {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'bedrooms' || name === 'plot_width' || name === 'plot_length'
        ? Number(value)
        : value,
    }))
  }

  function toggleExtra(extraId) {
    setFormData(prev => ({
      ...prev,
      extras: prev.extras.includes(extraId)
        ? prev.extras.filter(e => e !== extraId)
        : [...prev.extras, extraId],
    }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onGenerate(formData)
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* Plot Dimensions */}
      <div className="form-section">
        <div className="form-section-title">Plot Dimensions</div>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="plot_width">Width (ft)</label>
            <input
              id="plot_width"
              type="number"
              name="plot_width"
              min="20"
              max="200"
              value={formData.plot_width}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="plot_length">Length (ft)</label>
            <input
              id="plot_length"
              type="number"
              name="plot_length"
              min="20"
              max="200"
              value={formData.plot_length}
              onChange={handleChange}
            />
          </div>
        </div>
      </div>

      {/* Configuration */}
      <div className="form-section" style={{ marginTop: 14 }}>
        <div className="form-section-title">Configuration</div>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="bedrooms">Bedrooms (BHK)</label>
            <select
              id="bedrooms"
              name="bedrooms"
              value={formData.bedrooms}
              onChange={handleChange}
            >
              <option value={1}>1 BHK</option>
              <option value={2}>2 BHK</option>
              <option value={3}>3 BHK</option>
              <option value={4}>4 BHK</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="facing">Road Facing</label>
            <select
              id="facing"
              name="facing"
              value={formData.facing}
              onChange={handleChange}
            >
              <option value="south">South</option>
              <option value="north">North</option>
              <option value="east">East</option>
              <option value="west">West</option>
            </select>
          </div>
        </div>
      </div>

      {/* Extras */}
      <div className="form-section" style={{ marginTop: 14 }}>
        <div className="form-section-title">Optional Rooms</div>
        <div className="extras-grid">
          {EXTRAS.map(extra => (
            <label
              key={extra.id}
              className={`extras-chip ${formData.extras.includes(extra.id) ? 'active' : ''}`}
            >
              <input
                type="checkbox"
                checked={formData.extras.includes(extra.id)}
                onChange={() => toggleExtra(extra.id)}
              />
              <span className="extras-chip-check" />
              <span className="extras-chip-label">{extra.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        className={`btn-generate ${loading ? 'loading' : ''}`}
        disabled={loading}
        style={{ marginTop: 18 }}
      >
        {loading ? 'Generating Plan...' : 'Generate Floor Plan'}
      </button>
    </form>
  )
}
