import { useState } from 'react'
import FloorPlanForm from './components/FloorPlanForm'
import FloorPlanSVG from './components/FloorPlanSVG'
import { generatePlan, getDownloadUrl } from './services/api'

const LIVE_REASONING_STEPS = [
  'Analyzing plot dimensions and road-facing constraints.',
  'Preparing fresh room-program prompt with selected optional rooms only.',
  'Requesting layout draft from the AI model chain.',
  'Validating room overlaps, bounds, and bedroom requirements.',
  'Repairing geometry and regenerating doors/windows for buildability.',
  'Final quality check before rendering and DXF export.',
]

export default function App() {
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [reasoningFeed, setReasoningFeed] = useState([])

  async function handleGenerate(formData) {
    setLoading(true)
    setError(null)
    setPlan(null)

    const firstReasoning =
      `Received ${formData.plot_width} x ${formData.plot_length} ft, ${formData.bedrooms}BHK brief. Starting AI reasoning.`
    setReasoningFeed([firstReasoning, LIVE_REASONING_STEPS[0]])

    let stepCursor = 0
    const reasoningTimer = window.setInterval(() => {
      stepCursor += 1
      if (stepCursor >= LIVE_REASONING_STEPS.length) {
        window.clearInterval(reasoningTimer)
        return
      }

      const nextStep = LIVE_REASONING_STEPS[stepCursor]
      setReasoningFeed(prev => (prev.includes(nextStep) ? prev : [...prev, nextStep]))
    }, 1800)

    try {
      const result = await generatePlan(formData)
      setPlan(result)
      if (Array.isArray(result.reasoning_trace) && result.reasoning_trace.length > 0) {
        setReasoningFeed(result.reasoning_trace)
      } else {
        setReasoningFeed(prev => (
          prev.includes('Plan generation completed and rendered.')
            ? prev
            : [...prev, 'Plan generation completed and rendered.']
        ))
      }
    } catch (err) {
      setError(err.message || 'Failed to generate plan')
      setReasoningFeed(prev => (
        prev.includes('Generation stopped due to an error response from server.')
          ? prev
          : [...prev, 'Generation stopped due to an error response from server.']
      ))
    } finally {
      window.clearInterval(reasoningTimer)
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
        <div className="app-badge">CAD ENGINE v4</div>
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

          {(loading || reasoningFeed.length > 0) && (
            <div className={`reasoning-panel ${loading ? 'is-live' : ''}`}>
              <div className="reasoning-panel-title">
                {loading ? 'Model Reasoning (Live)' : 'Model Reasoning'}
              </div>
              <ul className="reasoning-list">
                {reasoningFeed.map((step, idx) => (
                  <li
                    key={`${idx}-${step}`}
                    className={loading && idx === reasoningFeed.length - 1 ? 'active' : ''}
                  >
                    <span className="reasoning-dot" />
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
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
                <div className="plan-info-row">
                  <span className="label">Engine</span>
                  <span className="value">
                    <span
                      className={`engine-badge ${
                        plan.generation_method === 'llm' || plan.generation_method === 'llm_backup'
                          ? 'llm'
                          : plan.generation_method === 'local_backup' || plan.generation_method === 'bsp_fallback'
                            ? 'fallback'
                            : 'bsp'
                      }`}
                    >
                      {plan.generation_method === 'llm'
                        ? 'AI Generated'
                        : plan.generation_method === 'llm_backup'
                          ? 'AI Backup Model'
                          : plan.generation_method === 'local_backup' || plan.generation_method === 'bsp_fallback'
                            ? 'Adaptive Backup'
                          : 'BSP Layout'}
                    </span>
                  </span>
                </div>
                {plan.adjacency_score > 0 && (
                  <div className="plan-info-row">
                    <span className="label">Adjacency Score</span>
                    <span className="value">{plan.adjacency_score}/100</span>
                  </div>
                )}
              </div>

              {plan.architect_note && (
                <div className="architect-note">{plan.architect_note}</div>
              )}

              {/* Vastu Issues */}
              {plan.vastu_issues && plan.vastu_issues.length > 0 && (
                <div className="vastu-issues">
                  <div className="vastu-issues-title">Vastu Notes</div>
                  <ul>
                    {plan.vastu_issues.map((issue, i) => (
                      <li key={i}>{issue}</li>
                    ))}
                  </ul>
                </div>
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
                  ? (reasoningFeed[reasoningFeed.length - 1] || 'Generating your floor plan...')
                  : 'Enter plot dimensions and click Generate'}
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
