import { useState } from 'react'

const EXTRAS = [
  { id: 'pooja', label: 'Pooja Room' },
  { id: 'study', label: 'Study / Home Office' },
  { id: 'garage', label: 'Garage' },
  { id: 'balcony', label: 'Balcony' },
  { id: 'store', label: 'Store Room' },
  { id: 'utility', label: 'Utility / Laundry' },
  { id: 'foyer', label: 'Foyer' },
  { id: 'staircase', label: 'Staircase' },
]

const STYLE_OPTIONS = [
  { id: 'modern', label: 'Modern' },
  { id: 'contemporary', label: 'Contemporary' },
  { id: 'traditional', label: 'Traditional' },
  { id: 'minimal', label: 'Minimal' },
]

const KITCHEN_OPTIONS = [
  { id: 'open', label: 'Open Kitchen' },
  { id: 'semi_open', label: 'Semi-Open Kitchen' },
  { id: 'closed', label: 'Closed Kitchen' },
]

const PRIORITY_LABEL = {
  1: 'Low',
  2: 'Medium-Low',
  3: 'Balanced',
  4: 'High',
  5: 'Very High',
}

export default function FloorPlanForm({ onGenerate, loading }) {
  const [formData, setFormData] = useState({
    plot_width: 30,
    plot_length: 50,
    bedrooms: 3,
    bathrooms_target: 0,
    floors: 1,
    facing: '',
    design_style: '',
    kitchen_preference: '',
    parking_slots: 0,
    vastu_priority: 4,
    natural_light_priority: 4,
    privacy_priority: 3,
    storage_priority: 3,
    elder_friendly: false,
    work_from_home: false,
    extras: [],
    notes: '',
  })

  const numericFields = new Set([
    'plot_width',
    'plot_length',
    'bedrooms',
    'bathrooms_target',
    'floors',
    'parking_slots',
    'vastu_priority',
    'natural_light_priority',
    'privacy_priority',
    'storage_priority',
  ])

  function handleChange(e) {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: numericFields.has(name) ? Number(value) : value,
    }))
  }

  function toggleSwitch(fieldName) {
    setFormData(prev => ({
      ...prev,
      [fieldName]: !prev[fieldName],
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
    onGenerate({
      ...formData,
      facing: formData.facing || 'east',
      design_style: formData.design_style || 'modern',
      kitchen_preference: formData.kitchen_preference || 'semi_open',
      bathrooms_target: Number(formData.bathrooms_target) || 0,
    })
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
              <option value="">Select Facing</option>
              <option value="south">South</option>
              <option value="north">North</option>
              <option value="east">East</option>
              <option value="west">West</option>
            </select>
          </div>
        </div>
      </div>

      {/* Detailed Requirements */}
      <div className="form-section" style={{ marginTop: 14 }}>
        <div className="form-section-title">Detailed Requirements</div>
        <div className="form-row form-row-4">
          <div className="form-group">
            <label htmlFor="bathrooms_target">Bathrooms</label>
            <input
              id="bathrooms_target"
              type="number"
              name="bathrooms_target"
              min="0"
              max="8"
              value={formData.bathrooms_target}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="floors">Floors</label>
            <select
              id="floors"
              name="floors"
              value={formData.floors}
              onChange={handleChange}
            >
              <option value={1}>1 Floor</option>
              <option value={2}>2 Floors</option>
              <option value={3}>3 Floors</option>
              <option value={4}>4 Floors</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="parking_slots">Parking Slots</label>
            <input
              id="parking_slots"
              type="number"
              name="parking_slots"
              min="0"
              max="4"
              value={formData.parking_slots}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="design_style">Design Style</label>
            <select
              id="design_style"
              name="design_style"
              value={formData.design_style}
              onChange={handleChange}
            >
              <option value="">Select Style</option>
              {STYLE_OPTIONS.map(style => (
                <option key={style.id} value={style.id}>{style.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-group" style={{ marginTop: 8 }}>
          <label htmlFor="kitchen_preference">Kitchen Preference</label>
          <select
            id="kitchen_preference"
            name="kitchen_preference"
            value={formData.kitchen_preference}
            onChange={handleChange}
          >
            <option value="">Select Kitchen Type</option>
            {KITCHEN_OPTIONS.map(option => (
              <option key={option.id} value={option.id}>{option.label}</option>
            ))}
          </select>
        </div>

        <div className="toggle-grid">
          <label className={`toggle-pill ${formData.elder_friendly ? 'active' : ''}`}>
            <input
              type="checkbox"
              checked={formData.elder_friendly}
              onChange={() => toggleSwitch('elder_friendly')}
            />
            Elder Friendly Layout
          </label>
          <label className={`toggle-pill ${formData.work_from_home ? 'active' : ''}`}>
            <input
              type="checkbox"
              checked={formData.work_from_home}
              onChange={() => toggleSwitch('work_from_home')}
            />
            Work-From-Home Friendly
          </label>
        </div>
      </div>

      {/* Priority Sliders */}
      <div className="form-section" style={{ marginTop: 14 }}>
        <div className="form-section-title">Planning Priorities</div>

        <div className="priority-item">
          <div className="priority-head">
            <span>Vastu Strictness</span>
            <strong>{PRIORITY_LABEL[formData.vastu_priority]}</strong>
          </div>
          <input
            type="range"
            min="1"
            max="5"
            name="vastu_priority"
            value={formData.vastu_priority}
            onChange={handleChange}
          />
        </div>

        <div className="priority-item">
          <div className="priority-head">
            <span>Natural Light</span>
            <strong>{PRIORITY_LABEL[formData.natural_light_priority]}</strong>
          </div>
          <input
            type="range"
            min="1"
            max="5"
            name="natural_light_priority"
            value={formData.natural_light_priority}
            onChange={handleChange}
          />
        </div>

        <div className="priority-item">
          <div className="priority-head">
            <span>Privacy</span>
            <strong>{PRIORITY_LABEL[formData.privacy_priority]}</strong>
          </div>
          <input
            type="range"
            min="1"
            max="5"
            name="privacy_priority"
            value={formData.privacy_priority}
            onChange={handleChange}
          />
        </div>

        <div className="priority-item">
          <div className="priority-head">
            <span>Storage Utility</span>
            <strong>{PRIORITY_LABEL[formData.storage_priority]}</strong>
          </div>
          <input
            type="range"
            min="1"
            max="5"
            name="storage_priority"
            value={formData.storage_priority}
            onChange={handleChange}
          />
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

      {/* Custom Notes */}
      <div className="form-section" style={{ marginTop: 14 }}>
        <div className="form-section-title">Custom Notes</div>
        <div className="form-group">
          <label htmlFor="notes">Design Note for AI</label>
          <textarea
            id="notes"
            name="notes"
            rows={3}
            maxLength={240}
            placeholder="Example: keep living + dining open, master bedroom in a quiet corner, wider passage near staircase."
            value={formData.notes}
            onChange={handleChange}
          />
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        className={`btn-generate ${loading ? 'loading' : ''}`}
        disabled={loading}
        style={{ marginTop: 18 }}
      >
        {loading ? 'Generating Detailed Plan...' : 'Generate Detailed Floor Plan'}
      </button>
    </form>
  )
}
