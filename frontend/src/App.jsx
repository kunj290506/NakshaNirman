import { useState } from 'react'
import FloorPlanForm from './components/FloorPlanForm'
import FloorPlanSVG from './components/FloorPlanSVG'
import { generatePlan, getDownloadUrl } from './services/api'

export default function App() {
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleGenerate(formData) {
    setLoading(true)
    setError(null)
    setPlan(null)
    try {
      const result = await generatePlan(formData)
      setPlan(result)
    } catch (err) {
      setError(err.message || 'Failed to generate plan')
    } finally {
      setLoading(false)
    }
  }

  function handleDownload() {
    if (!plan?.dxf_url) return
    const url = getDownloadUrl(plan.dxf_url)
    window.open(url, '_blank')
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">N</div>
          <h1>Naksha<span>Nirman</span></h1>
          <span className="app-subtitle">AI Floor Plan Generator</span>
        </div>
        <div className="app-badge">CAD ENGINE v3</div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Sidebar — Form */}
        <aside className="sidebar">
          <FloorPlanForm
            onGenerate={handleGenerate}
            loading={loading}
          />

          {error && (
            <div className="error-banner">{error}</div>
          )}

          {plan && (
            <>
              {/* Plan Info */}
              <div className="plan-info">
                <div className="plan-info-row">
                  <span className="label">Usable Area</span>
                  <span className="value">
                    {plan.plot.usable_width.toFixed(1)} × {plan.plot.usable_length.toFixed(1)} ft
                  </span>
                </div>
                <div className="plan-info-row">
                  <span className="label">Total Rooms</span>
                  <span className="value">{plan.rooms.length}</span>
                </div>
                <div className="plan-info-row">
                  <span className="label">Vastu Score</span>
                  <span className="value">{plan.vastu_score}/100</span>
                </div>
                <div className="plan-info-row">
                  <span className="label">Road Facing</span>
                  <span className="value" style={{ textTransform: 'capitalize' }}>
                    {plan.plot.road_side}
                  </span>
                </div>
              </div>

              {plan.architect_note && (
                <div className="architect-note">{plan.architect_note}</div>
              )}

              {plan.dxf_url && (
                <button className="btn-download" onClick={handleDownload}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download DXF (AutoCAD)
                </button>
              )}
            </>
          )}
        </aside>

        {/* Canvas — SVG Preview */}
        <section className="canvas">
          {plan ? (
            <div className="svg-container">
              <FloorPlanSVG plan={plan} />
            </div>
          ) : (
            <div className="canvas-empty">
              <div className="canvas-empty-border">
                <div className="canvas-empty-icon">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#AAAAAA" strokeWidth="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="1"/>
                    <line x1="3" y1="9" x2="21" y2="9"/>
                    <line x1="9" y1="21" x2="9" y2="9"/>
                  </svg>
                </div>
              </div>
              <p>
                {loading
                  ? 'Generating your floor plan...'
                  : 'Enter plot dimensions and click Generate'}
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
