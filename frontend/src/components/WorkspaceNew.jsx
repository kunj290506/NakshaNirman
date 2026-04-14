import { useMemo, useState } from 'react'
import FormInterface from './FormInterface'
import CadCanvas from './canvas/CadCanvas'
import CanvasToolbar from './toolbar/CanvasToolbar'
import PropertyPanel from './panels/PropertyPanel'
import { useLayout, useLayoutActions } from '../store/layoutStore'
import { generatePlan, getDownloadUrl } from '../services/api'

function normalizeFamilyType(raw) {
  const key = String(raw || 'nuclear').trim().toLowerCase().replace(/[-\s]+/g, '_')
  if (key === 'joint' || key === 'joint_family') return 'joint'
  if (key === 'couple' || key === 'working_couple') return 'couple'
  return 'nuclear'
}

function normalizeExtras(rawExtras) {
  if (!Array.isArray(rawExtras)) return []
  const valid = new Set(['pooja', 'study', 'store', 'balcony', 'garage', 'utility', 'foyer', 'staircase'])
  return Array.from(
    new Set(
      rawExtras
        .map((extra) => String(extra || '').trim().toLowerCase().replace(/\s+/g, '_'))
        .filter((extra) => valid.has(extra)),
    ),
  )
}

function normalizeTagList(raw) {
  const arr = Array.isArray(raw)
    ? raw
    : String(raw || '')
        .split(',')
        .map((item) => item.trim())
  return Array.from(
    new Set(
      arr
        .map((item) => String(item || '').trim().toLowerCase().replace(/\s+/g, '_'))
        .filter(Boolean),
    ),
  )
}

function buildRequestPayload(rawPayload) {
  const payload = rawPayload && typeof rawPayload === 'object' ? rawPayload : {}
  const plotWidth = Math.max(20, Number(payload.plot_width || 30))
  const plotLength = Math.max(20, Number(payload.plot_length || 40))
  const bedrooms = Math.min(4, Math.max(1, Number(payload.bedrooms || 2)))
  const facing = String(payload.facing || payload.road_side || 'east').toLowerCase()

  return {
    plot_width: plotWidth,
    plot_length: plotLength,
    bedrooms,
    facing: ['north', 'south', 'east', 'west'].includes(facing) ? facing : 'east',
    extras: normalizeExtras(payload.extras),
    bathrooms_target: Math.max(0, Number(payload.bathrooms_target ?? payload.bathrooms ?? 0)),
    floors: Math.min(4, Math.max(1, Number(payload.floors || 1))),
    design_style: String(payload.design_style || 'modern'),
    kitchen_preference: String(payload.kitchen_preference || 'semi_open'),
    parking_slots: Math.max(0, Number(payload.parking_slots || 0)),
    vastu_priority: Math.min(5, Math.max(1, Number(payload.vastu_priority || 3))),
    natural_light_priority: Math.min(5, Math.max(1, Number(payload.natural_light_priority || 3))),
    privacy_priority: Math.min(5, Math.max(1, Number(payload.privacy_priority || 3))),
    storage_priority: Math.min(5, Math.max(1, Number(payload.storage_priority || 3))),
    strict_real_life: Boolean(payload.strict_real_life),
    must_have: normalizeTagList(payload.must_have),
    avoid: normalizeTagList(payload.avoid),
    elder_friendly: Boolean(payload.elder_friendly),
    work_from_home: Boolean(payload.work_from_home),
    city: String(payload.city || ''),
    state: String(payload.state || ''),
    family_type: normalizeFamilyType(payload.family_type),
    family_notes: String(payload.family_notes || ''),
    notes: String(payload.notes || ''),
  }
}

export default function WorkspaceNew({ onLogout = null, userId = '' }) {
  const { state } = useLayout()
  const actions = useLayoutActions()
  const [reasoningFeed, setReasoningFeed] = useState([])
  const [exportingPng, setExportingPng] = useState(false)

  const statusLine = useMemo(() => {
    if (!state.layout) return 'No plan loaded'
    const method = String(state.layout.generation_method || 'bsp').toUpperCase()
    return `${method} • Vastu ${Math.round(state.layout.vastu_score || 0)} • Adj ${Math.round(state.layout.adjacency_score || 0)}`
  }, [state.layout])

  async function handleGenerate(roomsArg, _totalArea, payloadArg) {
    const fallbackPayload =
      roomsArg && !Array.isArray(roomsArg) && typeof roomsArg === 'object' ? roomsArg : {}
    const payload = buildRequestPayload(payloadArg || fallbackPayload)

    actions.setError(null)
    actions.setLoading(true, 'Generating layout with architectural constraints...')
    setReasoningFeed([
      `Input accepted: ${payload.plot_width} x ${payload.plot_length} ft, ${payload.bedrooms}BHK, ${payload.facing} facing.`,
      'Calling planner and validating room geometry...',
    ])

    try {
      const plan = await generatePlan(payload)
      actions.setLayout(plan)
      setReasoningFeed(
        Array.isArray(plan.reasoning_trace) && plan.reasoning_trace.length > 0
          ? plan.reasoning_trace
          : ['Plan generated and synchronized to CAD workspace.'],
      )
    } catch (error) {
      const message = error?.message || 'Plan generation failed'
      actions.setError(message)
      setReasoningFeed((prev) => [...prev, `Generation failed: ${message}`])
    } finally {
      actions.setLoading(false)
    }
  }

  function handleDownload() {
    if (!state.layout?.dxf_url) return
    window.open(getDownloadUrl(state.layout.dxf_url), '_blank')
  }

  async function handleDownloadPng() {
    if (!state.layout) {
      actions.setError('Generate a plan first, then export PNG.')
      return
    }

    const svg = document.querySelector('.cad-canvas-content svg')
    if (!svg) {
      actions.setError('PNG export failed: plan preview is not ready yet.')
      return
    }

    setExportingPng(true)
    try {
      const viewBox = svg.viewBox?.baseVal
      const unitWidth = Number(viewBox?.width || 80)
      const unitHeight = Number(viewBox?.height || 60)

      let pngWidth = Math.max(900, Math.round(unitWidth * 30))
      let pngHeight = Math.max(700, Math.round(unitHeight * 30))
      const fitScale = Math.min(2400 / pngWidth, 1800 / pngHeight, 1)
      pngWidth = Math.max(900, Math.round(pngWidth * fitScale))
      pngHeight = Math.max(700, Math.round(pngHeight * fitScale))

      const clonedSvg = svg.cloneNode(true)
      clonedSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      clonedSvg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
      clonedSvg.setAttribute('width', String(pngWidth))
      clonedSvg.setAttribute('height', String(pngHeight))
      clonedSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet')

      const serialized = new XMLSerializer().serializeToString(clonedSvg)
      const svgBlob = new Blob([serialized], { type: 'image/svg+xml;charset=utf-8' })
      const svgUrl = URL.createObjectURL(svgBlob)

      await new Promise((resolve, reject) => {
        const img = new Image()
        img.onload = () => {
          try {
            const canvas = document.createElement('canvas')
            canvas.width = pngWidth
            canvas.height = pngHeight
            const ctx = canvas.getContext('2d')
            if (!ctx) {
              URL.revokeObjectURL(svgUrl)
              reject(new Error('Could not prepare image canvas.'))
              return
            }

            ctx.fillStyle = '#ffffff'
            ctx.fillRect(0, 0, pngWidth, pngHeight)
            ctx.drawImage(img, 0, 0, pngWidth, pngHeight)

            canvas.toBlob(
              (pngBlob) => {
                URL.revokeObjectURL(svgUrl)
                if (!pngBlob) {
                  reject(new Error('PNG conversion failed.'))
                  return
                }

                const fileUrl = URL.createObjectURL(pngBlob)
                const fileBase = String(state.layout?.layout_signature || `plan-${Date.now()}`)
                  .replace(/[^a-zA-Z0-9-_]+/g, '-')
                  .toLowerCase()
                const anchor = document.createElement('a')
                anchor.href = fileUrl
                anchor.download = `${fileBase}.png`
                document.body.appendChild(anchor)
                anchor.click()
                anchor.remove()
                URL.revokeObjectURL(fileUrl)
                resolve(true)
              },
              'image/png',
              1,
            )
          } catch (error) {
            URL.revokeObjectURL(svgUrl)
            reject(error)
          }
        }
        img.onerror = () => {
          URL.revokeObjectURL(svgUrl)
          reject(new Error('Failed to render SVG for PNG export.'))
        }
        img.src = svgUrl
      })

      actions.setError(null)
      setReasoningFeed((prev) => [...prev, 'Final clean PNG exported successfully.'])
    } catch (error) {
      const message = error?.message || 'PNG export failed'
      actions.setError(message)
      setReasoningFeed((prev) => [...prev, `PNG export failed: ${message}`])
    } finally {
      setExportingPng(false)
    }
  }

  return (
    <div className='workspace-shell'>
      <header className='workspace-topbar'>
        <div>
          <h1 className='workspace-title'>NakshaNirman Workspace</h1>
          <p className='workspace-status'>{statusLine}</p>
        </div>
        <div className='workspace-mode-toggle'>
          <button
            type='button'
            className={`btn btn-secondary${state.previewMode === '2d' ? ' active' : ''}`}
            onClick={() => actions.setPreviewMode('2d')}
          >
            2D Plan
          </button>
          <button
            type='button'
            className={`btn btn-secondary${state.previewMode === 'boundary' ? ' active' : ''}`}
            onClick={() => actions.setPreviewMode('boundary')}
          >
            Boundary
          </button>
          {userId ? <span className='workspace-user-pill'>{userId}</span> : null}
          {onLogout ? (
            <button type='button' className='btn btn-secondary' onClick={onLogout}>
              Logout
            </button>
          ) : null}
        </div>
      </header>

      <div className='workspace-grid'>
        <aside className='workspace-left-pane'>
          <div className='workspace-status'>Fill the form and generate your plan in one step.</div>

          <div className='workspace-input-panel'>
            <FormInterface onGenerate={handleGenerate} loading={state.loading} />
          </div>

          {reasoningFeed.length > 0 && (
            <div className='reasoning-panel'>
              <div className='reasoning-panel-title'>Planner Trace</div>
              <ul className='reasoning-list'>
                {reasoningFeed.map((item, idx) => (
                  <li key={`${item}-${idx}`}>
                    <span className='reasoning-dot' />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {state.layout && (
            <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
              {state.layout?.dxf_url ? (
                <button type='button' className='btn btn-primary' onClick={handleDownload}>
                  Download DXF
                </button>
              ) : null}
              <button type='button' className='btn btn-secondary' onClick={handleDownloadPng} disabled={exportingPng}>
                {exportingPng ? 'Exporting PNG...' : 'Download Clean PNG'}
              </button>
            </div>
          )}
        </aside>

        <section className='workspace-center-pane'>
          <CanvasToolbar />
          <div className='workspace-canvas-wrap'>
            <CadCanvas />
          </div>
        </section>

        <aside className='workspace-right-pane'>
          <PropertyPanel />
        </aside>
      </div>
    </div>
  )
}
