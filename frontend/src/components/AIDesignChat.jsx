import { useState, useRef, useEffect, useCallback } from 'react'

const PIPELINE_STAGES = [
    { key: 'chat', label: 'Chat', icon: '1', desc: 'Collecting requirements' },
    { key: 'extraction', label: 'Extract', icon: '2', desc: 'Structuring data' },
    { key: 'design', label: 'Design', icon: '3', desc: 'Generating layout' },
    { key: 'validation', label: 'Validate', icon: '4', desc: 'Checking compliance' },
    { key: 'generation', label: 'Generate', icon: '5', desc: 'Creating DXF' },
]

const STAGE_ORDER = ['chat', 'extraction', 'design', 'validation', 'generation', 'complete']

export default function AIDesignChat({ onGenerate, onBoundaryUpload, loading, projectId, onReviewPlan, plan }) {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: "Welcome to NakshaNirman AI Architect.\n\nI'll help you design your dream home step by step.\n\nStart by telling me about your plot — for example:\n\"I have a 30x40 feet plot and need 3 bedrooms, 2 bathrooms.\"\n\nOr just tell me your plot size and I'll guide you through the rest.",
            provider: 'system',
            stage: 'chat',
        }
    ])
    const [input, setInput] = useState('')
    const [isTyping, setIsTyping] = useState(false)
    const [ws, setWs] = useState(null)
    const [wsReady, setWsReady] = useState(false)
    const [currentStage, setCurrentStage] = useState('chat')
    const [extractedData, setExtractedData] = useState({ rooms: [], total_area: null })
    const [layoutJson, setLayoutJson] = useState(null)
    const [validationReport, setValidationReport] = useState(null)
    const [requirementsJson, setRequirementsJson] = useState(null)
    const [reviewData, setReviewData] = useState(null)
    const [isReviewing, setIsReviewing] = useState(false)
    const [expandedReasoning, setExpandedReasoning] = useState({})
    const messagesEndRef = useRef(null)
    const fileInputRef = useRef(null)
    const extractedRef = useRef({ rooms: [], total_area: null })

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isTyping])

    useEffect(() => {
        extractedRef.current = extractedData
    }, [extractedData])

    // WebSocket connection — tries engine endpoint first, falls back to ai-design
    useEffect(() => {
        let socket = null
        let reconnectTimeout = null

        const connect = () => {
            try {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
                // Try the new engine endpoint first, fallback to ai-design
                const wsUrl = `${proto}//${window.location.host}/api/engine/chat`
                socket = new WebSocket(wsUrl)

                socket.onopen = () => {
                    setWs(socket)
                    setWsReady(true)
                }

                socket.onmessage = (event) => {
                    const data = JSON.parse(event.data)
                    setIsTyping(false)

                    // Update stage — support both 'stage' (ai-design) and 'mode' (engine) fields
                    const stageOrMode = data.stage || data.mode
                    if (stageOrMode) {
                        setCurrentStage(stageOrMode)
                    }

                    // Build message object
                    const msgObj = {
                        role: 'assistant',
                        content: data.reply,
                        provider: data.provider || (data.mode ? 'engine' : 'unknown'),
                        stage: stageOrMode,
                        stageTransition: data.stage_transition || false,
                        extractedData: data.extracted_data,
                        requirementsJson: data.requirements_json,
                        layoutJson: data.layout_json || data.layout,
                        validationReport: data.validation_report || data.validation,
                    }
                    setMessages(prev => [...prev, msgObj])

                    // Store extracted data
                    if (data.extracted_data) {
                        setExtractedData(prev => {
                            const updated = { ...prev }
                            if (data.extracted_data.total_area) updated.total_area = data.extracted_data.total_area
                            if (data.extracted_data.rooms) {
                                updated.rooms = data.extracted_data.rooms
                            }
                            return updated
                        })
                    }

                    if (data.requirements_json) {
                        setRequirementsJson(data.requirements_json)
                    }

                    // Support both 'layout_json' (ai-design) and 'layout' (engine)
                    if (data.layout_json || data.layout) {
                        setLayoutJson(data.layout_json || data.layout)
                    }

                    // Support both 'validation_report' (ai-design) and 'validation' (engine)
                    if (data.validation_report || data.validation) {
                        setValidationReport(data.validation_report || data.validation)
                    }

                    // Handle engine 'collected' data (parsed requirements from chat)
                    if (data.collected) {
                        setExtractedData(prev => {
                            const updated = { ...prev }
                            if (data.collected.total_area) updated.total_area = data.collected.total_area
                            if (data.collected.bedrooms) {
                                // Convert collected data to rooms format
                                const rooms = []
                                if (data.collected.bedrooms) {
                                    rooms.push({ room_type: 'master_bedroom', quantity: 1 })
                                    if (data.collected.bedrooms > 1) {
                                        rooms.push({ room_type: 'bedroom', quantity: data.collected.bedrooms - 1 })
                                    }
                                }
                                if (data.collected.bathrooms) {
                                    rooms.push({ room_type: 'bathroom', quantity: data.collected.bathrooms })
                                }
                                rooms.push({ room_type: 'living', quantity: 1 })
                                rooms.push({ room_type: 'kitchen', quantity: 1 })
                                if (data.collected.extras) {
                                    data.collected.extras.forEach(e => {
                                        rooms.push({ room_type: e, quantity: 1 })
                                    })
                                }
                                updated.rooms = rooms
                            }
                            return updated
                        })
                    }

                    // Auto-generate when pipeline completes
                    if (data.should_generate && data.extracted_data) {
                        const rooms = data.extracted_data.rooms || extractedRef.current.rooms
                        const totalArea = data.extracted_data.total_area || extractedRef.current.total_area
                        if (totalArea && rooms?.length > 0) {
                            const chatRequirements = {
                                floors: data.extracted_data.floors || data.collected?.floors || 1,
                                bedrooms: data.extracted_data.bedrooms || data.collected?.bedrooms || rooms.filter(r => r.room_type === 'bedroom' || r.room_type === 'master_bedroom').reduce((s, r) => s + (r.quantity || 1), 0),
                                bathrooms: data.extracted_data.bathrooms || data.collected?.bathrooms || rooms.filter(r => r.room_type === 'bathroom').reduce((s, r) => s + (r.quantity || 1), 0),
                                extras: data.extracted_data.extras || data.collected?.extras || [],
                            }
                            onGenerate(rooms, totalArea, chatRequirements)
                        }
                    }
                }

                socket.onerror = () => {
                    setWsReady(false)
                }

                socket.onclose = () => {
                    setWs(null)
                    setWsReady(false)
                    reconnectTimeout = setTimeout(connect, 5000)
                }
            } catch {
                setWsReady(false)
            }
        }

        connect()

        return () => {
            if (reconnectTimeout) clearTimeout(reconnectTimeout)
            if (socket) socket.close()
        }
    }, [])

    const sendMessage = useCallback(() => {
        if (!input.trim() || loading) return

        const userMsg = { role: 'user', content: input.trim() }
        setMessages(prev => [...prev, userMsg])
        setIsTyping(true)

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ message: input.trim(), project_id: projectId }))
        } else {
            // Fallback: Use REST API
            fetch('/api/ai-design/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input.trim(), project_id: projectId }),
            })
                .then(res => res.json())
                .then(data => {
                    setIsTyping(false)
                    const msgObj = {
                        role: 'assistant',
                        content: data.reasoning || data.reply || 'Analysis complete.',
                        provider: data.provider || 'unknown',
                        extractedData: data.extracted_data,
                        vastuRecommendations: data.vastu_recommendations,
                        complianceNotes: data.compliance_notes,
                        designScore: data.design_score,
                    }
                    setMessages(prev => [...prev, msgObj])

                    if (data.rooms?.length > 0) {
                        setExtractedData(prev => ({
                            ...prev,
                            rooms: data.rooms,
                            total_area: data.extracted_data?.total_area || prev.total_area,
                        }))
                    }

                    if (data.ready_to_generate) {
                        const totalArea = data.extracted_data?.total_area || extractedRef.current.total_area
                        const rooms = data.rooms?.length > 0 ? data.rooms : extractedRef.current.rooms
                        if (totalArea && rooms?.length > 0) {
                            const chatRequirements = {
                                floors: data.extracted_data?.floors || 1,
                                bedrooms: data.extracted_data?.bedrooms || rooms.filter(r => r.room_type === 'bedroom' || r.room_type === 'master_bedroom').reduce((s, r) => s + (r.quantity || 1), 0),
                                bathrooms: data.extracted_data?.bathrooms || rooms.filter(r => r.room_type === 'bathroom').reduce((s, r) => s + (r.quantity || 1), 0),
                                extras: data.extracted_data?.extras || [],
                            }
                            onGenerate(rooms, totalArea, chatRequirements)
                        }
                    }
                })
                .catch(err => {
                    setIsTyping(false)
                    setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: `Connection issue: ${err.message}. Please check the backend.`,
                        provider: 'error',
                    }])
                })
        }

        setInput('')
    }, [input, ws, loading, projectId])

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0]
        if (!file) return

        setMessages(prev => [...prev, {
            role: 'user', content: `Uploaded: ${file.name}`
        }])

        setIsTyping(true)
        const result = await onBoundaryUpload(file)
        setIsTyping(false)

        if (result) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Boundary extracted! ${result.num_vertices || 'Several'} vertices, ${result.area ? result.area.toFixed(0) : 'calculated'} sq units.\n\nUsable area after setback: ${result.usable_area ? result.usable_area.toFixed(0) : 'N/A'} sq units.\n\nNow describe your rooms — I'll design the layout for you.`,
                provider: 'system',
                stage: 'chat',
            }])
        } else {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Could not extract boundary. Try a clearer image or DXF file.',
                provider: 'system',
            }])
        }
    }

    const handleReviewPlan = async () => {
        if (!plan) return
        setIsReviewing(true)

        setMessages(prev => [...prev, {
            role: 'user', content: 'Review this floor plan for compliance'
        }])

        try {
            const res = await fetch('/api/ai-design/review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ floor_plan: plan, project_id: projectId }),
            })

            const data = await res.json()
            setReviewData(data.scores)

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.review_text || 'Review complete.',
                provider: data.provider || 'unknown',
                reviewScores: data.scores,
            }])
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Review failed: ${err.message}`,
                provider: 'error',
            }])
        }
        setIsReviewing(false)
    }

    const handleGenerateFromAI = () => {
        const current = extractedRef.current
        if (current.total_area && current.rooms?.length > 0) {
            onGenerate(current.rooms, current.total_area)
        }
    }

    const toggleReasoning = (idx) => {
        setExpandedReasoning(prev => ({ ...prev, [idx]: !prev[idx] }))
    }

    const getProviderBadge = (provider) => {
        const badges = {
            grok: { label: 'AI', color: '#000', bg: '#f0f0f0' },
            groq: { label: 'AI', color: '#000', bg: '#f0f0f0' },
            fallback: { label: 'Offline', color: '#666', bg: '#f0f0f0' },
            system: { label: 'System', color: '#333', bg: '#f0f0f0' },
            error: { label: 'Error', color: '#333', bg: '#eee' },
        }
        return badges[provider] || badges.fallback
    }

    const renderScoreBadge = (score, label) => {
        return (
            <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
                padding: '0.15rem 0.5rem', borderRadius: '99px',
                fontSize: '0.72rem', fontWeight: 700, color: '#000', background: '#f0f0f0',
                border: '1px solid #ddd',
            }}>
                {label}: {score}/10
            </span>
        )
    }

    // Get stage index for progress bar
    const currentStageIdx = STAGE_ORDER.indexOf(currentStage)

    // Stage-specific typing message
    const getTypingMessage = () => {
        switch (currentStage) {
            case 'chat': return 'AI Architect is thinking...'
            case 'extraction': return 'Extracting requirements...'
            case 'design': return 'Generating layout...'
            case 'validation': return 'Validating design...'
            case 'generation': return 'Creating floor plan...'
            default: return 'Processing...'
        }
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Pipeline Stage Progress Indicator */}
            <div style={{
                padding: '0.6rem 0.75rem',
                background: '#f7f7f7',
                border: '1px solid #e0e0e0',
                borderRadius: 'var(--radius-sm)',
                marginBottom: '0.5rem',
            }}>
                {/* Header row */}
                <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    marginBottom: '0.5rem',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#000' }}>
                            AI Design Pipeline
                        </span>
                        <span style={{
                            width: 8, height: 8, borderRadius: 99,
                            background: wsReady ? '#333' : '#999',
                            display: 'inline-block',
                        }} />
                    </div>
                    <div style={{ display: 'flex', gap: '0.3rem' }}>
                        {plan && (
                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={handleReviewPlan}
                                disabled={isReviewing}
                                style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
                            >
                                {isReviewing ? '...' : 'AI Review'}
                            </button>
                        )}
                        {extractedData.rooms?.length > 0 && extractedData.total_area && (
                            <button
                                className="btn btn-primary btn-sm"
                                onClick={handleGenerateFromAI}
                                disabled={loading}
                                style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
                            >
                                Generate
                            </button>
                        )}
                    </div>
                </div>

                {/* Stage progress bar */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '0.15rem',
                }}>
                    {PIPELINE_STAGES.map((stage, idx) => {
                        const isActive = stage.key === currentStage
                        const isComplete = currentStageIdx > idx || currentStage === 'complete'
                        const isPending = currentStageIdx < idx && currentStage !== 'complete'

                        return (
                            <div key={stage.key} style={{
                                display: 'flex', alignItems: 'center', flex: 1,
                            }}>
                                <div style={{
                                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                                    flex: 1, position: 'relative',
                                }}>
                                    {/* Stage dot/icon */}
                                    <div style={{
                                        width: isActive ? 28 : 22,
                                        height: isActive ? 28 : 22,
                                        borderRadius: '50%',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        fontSize: isActive ? '0.8rem' : '0.65rem',
                                        fontWeight: 700,
                                        background: isComplete ? '#000'
                                            : isActive ? '#333'
                                                : '#e0e0e0',
                                        color: (isComplete || isActive) ? '#fff' : '#999',
                                        transition: 'all 0.3s ease',
                                        boxShadow: isActive ? '0 0 0 3px rgba(0,0,0,0.1)' : 'none',
                                    }}>
                                        {isComplete ? '✓' : stage.icon}
                                    </div>
                                    {/* Stage label */}
                                    <span style={{
                                        fontSize: '0.6rem',
                                        fontWeight: isActive ? 700 : 500,
                                        color: isActive ? '#000' : isComplete ? '#333' : '#999',
                                        marginTop: '0.2rem',
                                        whiteSpace: 'nowrap',
                                    }}>
                                        {stage.label}
                                    </span>
                                </div>
                                {/* Connector line */}
                                {idx < PIPELINE_STAGES.length - 1 && (
                                    <div style={{
                                        flex: 0.5,
                                        height: 2,
                                        background: currentStageIdx > idx || currentStage === 'complete'
                                            ? '#000' : '#e0e0e0',
                                        marginBottom: '1rem',
                                        transition: 'background 0.3s ease',
                                    }} />
                                )}
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Messages */}
            <div className="chat-messages" style={{ flex: 1, overflowY: 'auto', padding: '0.5rem 0' }}>
                {messages.map((msg, i) => {
                    const badge = msg.provider ? getProviderBadge(msg.provider) : null
                    const isLongMessage = msg.content && msg.content.length > 300
                    const isTransition = msg.stageTransition

                    // Stage transition messages get special styling
                    if (isTransition) {
                        return (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'center', gap: '0.5rem',
                                padding: '0.4rem 0.8rem',
                                margin: '0.3rem 0',
                                background: 'linear-gradient(135deg, #ede9fe, #ddd6fe)',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '0.78rem',
                                fontWeight: 600,
                                color: '#5b21b6',
                                border: '1px solid #c4b5fd40',
                            }}>
                                <span style={{
                                    animation: 'spin 1s linear infinite',
                                    display: 'inline-block',
                                }}>⚙️</span>
                                {msg.content}
                            </div>
                        )
                    }

                    return (
                        <div key={i} className={`chat-bubble ${msg.role}`} style={{
                            position: 'relative',
                        }}>
                            {/* Provider badge */}
                            {msg.role === 'assistant' && badge && (
                                <div style={{
                                    display: 'flex', alignItems: 'center', gap: '0.4rem',
                                    marginBottom: '0.4rem',
                                }}>
                                    <span style={{
                                        fontSize: '0.68rem', fontWeight: 700,
                                        color: badge.color, background: badge.bg,
                                        padding: '0.1rem 0.4rem', borderRadius: '99px',
                                        border: `1px solid ${badge.color}20`,
                                    }}>
                                        {badge.label}
                                    </span>
                                    {msg.designScore > 0 && renderScoreBadge(msg.designScore, 'Design')}
                                    {msg.stage && msg.stage !== 'chat' && (
                                        <span style={{
                                            fontSize: '0.65rem', fontWeight: 600,
                                            color: '#7c3aed', background: '#f5f3ff',
                                            padding: '0.1rem 0.35rem', borderRadius: '99px',
                                            border: '1px solid #7c3aed20',
                                        }}>
                                            Stage: {msg.stage}
                                        </span>
                                    )}
                                </div>
                            )}

                            {/* Message content */}
                            {isLongMessage && msg.role === 'assistant' ? (
                                <div>
                                    <div style={{
                                        maxHeight: expandedReasoning[i] ? 'none' : '150px',
                                        overflow: 'hidden',
                                        position: 'relative',
                                    }}>
                                        {msg.content.split('\n').map((line, j) => (
                                            <span key={j}>
                                                {line.startsWith('## ') ? (
                                                    <strong style={{ display: 'block', marginTop: '0.5rem', color: '#5b21b6' }}>
                                                        {line.replace('## ', '')}
                                                    </strong>
                                                ) : line.startsWith('**') ? (
                                                    <strong>{line.replace(/\*\*/g, '')}</strong>
                                                ) : (
                                                    line
                                                )}
                                                {j < msg.content.split('\n').length - 1 && <br />}
                                            </span>
                                        ))}
                                        {!expandedReasoning[i] && (
                                            <div style={{
                                                position: 'absolute', bottom: 0, left: 0, right: 0,
                                                height: '40px',
                                                background: msg.role === 'assistant'
                                                    ? 'linear-gradient(transparent, var(--gray-50, #f9fafb))'
                                                    : 'linear-gradient(transparent, var(--accent-light, #ede9fe))',
                                            }} />
                                        )}
                                    </div>
                                    <button
                                        onClick={() => toggleReasoning(i)}
                                        style={{
                                            background: 'none', border: 'none', cursor: 'pointer',
                                            color: '#7c3aed', fontSize: '0.75rem', fontWeight: 600,
                                            padding: '0.3rem 0', display: 'block',
                                        }}
                                    >
                                        {expandedReasoning[i] ? 'Show less' : 'Show full reasoning'}
                                    </button>
                                </div>
                            ) : (
                                msg.content.split('\n').map((line, j) => (
                                    <span key={j}>
                                        {line.startsWith('## ') ? (
                                            <strong style={{ display: 'block', marginTop: '0.5rem', color: '#5b21b6' }}>
                                                {line.replace('## ', '')}
                                            </strong>
                                        ) : line.startsWith('**') ? (
                                            <strong>{line.replace(/\*\*/g, '')}</strong>
                                        ) : (
                                            line
                                        )}
                                        {j < msg.content.split('\n').length - 1 && <br />}
                                    </span>
                                ))
                            )}

                            {/* Requirements JSON display */}
                            {msg.requirementsJson && (
                                <div style={{
                                    marginTop: '0.5rem', padding: '0.5rem 0.6rem',
                                    background: '#f5f5f5', borderRadius: '6px',
                                    border: '1px solid #ddd',
                                    fontSize: '0.72rem', fontFamily: 'monospace',
                                    maxHeight: '200px', overflowY: 'auto',
                                }}>
                                    <strong style={{ color: '#000', display: 'block', marginBottom: '0.3rem' }}>
                                        Extracted Requirements:
                                    </strong>
                                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', color: '#333' }}>
                                        {JSON.stringify(msg.requirementsJson, null, 2)}
                                    </pre>
                                </div>
                            )}

                            {/* Validation report display */}
                            {msg.validationReport && (
                                <div style={{
                                    marginTop: '0.5rem', padding: '0.5rem 0.6rem',
                                    background: '#f5f5f5',
                                    borderRadius: '6px',
                                    border: '1px solid #ddd',
                                    fontSize: '0.72rem',
                                }}>
                                    <strong style={{
                                        color: '#000',
                                        display: 'block', marginBottom: '0.3rem',
                                    }}>
                                        {msg.validationReport.compliant ? 'Design Compliant' : 'Issues Found'}
                                    </strong>
                                    {msg.validationReport.issues?.length > 0 && (
                                        <div>
                                            {msg.validationReport.issues.map((issue, ii) => (
                                                <div key={ii} style={{ color: '#333', marginTop: '0.1rem' }}>
                                                    - {issue}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {msg.validationReport.suggestions?.length > 0 && (
                                        <div style={{ marginTop: '0.3rem' }}>
                                            {msg.validationReport.suggestions.map((s, si) => (
                                                <div key={si} style={{ color: '#555', marginTop: '0.1rem' }}>
                                                    {s}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Vastu recommendations */}
                            {msg.vastuRecommendations?.length > 0 && (
                                <div style={{
                                    marginTop: '0.5rem', padding: '0.4rem 0.6rem',
                                    background: '#f5f5f5', borderRadius: '6px',
                                    border: '1px solid #ddd',
                                    fontSize: '0.75rem',
                                }}>
                                    <strong style={{ color: '#000' }}>Vastu:</strong>
                                    {msg.vastuRecommendations.map((v, vi) => (
                                        <div key={vi} style={{ marginTop: '0.2rem', color: '#333' }}>
                                            - {v.room}: {v.recommended_direction} -- {v.reason}
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Review scores */}
                            {msg.reviewScores && (
                                <div style={{
                                    marginTop: '0.5rem', display: 'flex', flexWrap: 'wrap', gap: '0.3rem',
                                }}>
                                    {msg.reviewScores.overall_score && renderScoreBadge(msg.reviewScores.overall_score, 'Overall')}
                                    {msg.reviewScores.vastu_compliance?.score && renderScoreBadge(msg.reviewScores.vastu_compliance.score, 'Vastu')}
                                    {msg.reviewScores.building_code?.score && renderScoreBadge(msg.reviewScores.building_code.score, 'Code')}
                                    {msg.reviewScores.ventilation?.score && renderScoreBadge(msg.reviewScores.ventilation.score, 'Ventilation')}
                                    {msg.reviewScores.circulation?.score && renderScoreBadge(msg.reviewScores.circulation.score, 'Flow')}
                                </div>
                            )}

                            {/* Compliance notes */}
                            {msg.complianceNotes?.length > 0 && (
                                <div style={{
                                    marginTop: '0.5rem', padding: '0.4rem 0.6rem',
                                    background: '#ecfdf5', borderRadius: '6px',
                                    border: '1px solid #6ee7b7',
                                    fontSize: '0.75rem',
                                }}>
                                    <strong style={{ color: '#065f46' }}>Compliance:</strong>
                                    {msg.complianceNotes.map((n, ni) => (
                                        <div key={ni} style={{ marginTop: '0.2rem', color: '#047857' }}>- {n}</div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )
                })}

                {isTyping && (
                    <div className="typing-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <span style={{ fontSize: '0.72rem', color: '#7c3aed', marginLeft: '0.3rem', fontWeight: 600 }}>
                            {getTypingMessage()}
                        </span>
                    </div>
                )}

                {loading && (
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '0.6rem',
                        padding: '0.75rem 1rem',
                        background: 'linear-gradient(135deg, #ede9fe, #ddd6fe)',
                        border: '1px solid #c4b5fd',
                        borderRadius: 'var(--radius-md)',
                        alignSelf: 'flex-start', maxWidth: '88%',
                    }}>
                        <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2, marginBottom: 0 }}></div>
                        <span style={{ fontSize: '0.82rem', color: '#5b21b6', fontWeight: 600 }}>
                            Generating your floor plan...
                        </span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Extracted data summary bar */}
            {(extractedData.rooms?.length > 0 || extractedData.total_area) && (
                <div style={{
                    padding: '0.4rem 0.6rem',
                    background: '#f0fdf4',
                    border: '1px solid #bbf7d0',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.72rem',
                    color: '#166534',
                    marginBottom: '0.4rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: '0.3rem',
                }}>
                    <span>
                        {extractedData.total_area ? `${extractedData.total_area} sqft` : ''}
                        {extractedData.rooms?.length > 0 ? ` | ${extractedData.rooms.length} room types` : ''}
                    </span>
                    {extractedData.rooms?.length > 0 && extractedData.total_area && (
                        <button
                            className="btn btn-primary btn-sm"
                            onClick={handleGenerateFromAI}
                            disabled={loading}
                            style={{ fontSize: '0.68rem', padding: '0.15rem 0.4rem' }}
                        >
                            Generate Plan
                        </button>
                    )}
                </div>
            )}

            {/* Input area */}
            <div className="chat-input-area" style={{ position: 'sticky', bottom: 0 }}>
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept=".png,.jpg,.jpeg,.dxf"
                    style={{ display: 'none' }}
                />
                <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => fileInputRef.current?.click()}
                    title="Upload boundary image"
                    disabled={loading}
                >
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                    </svg>
                </button>
                <input
                    className="chat-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                        currentStage === 'chat'
                            ? 'Describe your dream home...'
                            : loading
                                ? 'Processing...'
                                : 'Type a message...'
                    }
                    disabled={loading || (currentStage !== 'chat' && currentStage !== 'complete')}
                />
                <button
                    className="btn btn-primary btn-sm"
                    onClick={sendMessage}
                    disabled={loading || !input.trim()}
                    style={{
                        opacity: (loading || !input.trim()) ? 0.5 : 1,
                        background: 'linear-gradient(135deg, #7c3aed, #6d28d9)',
                    }}
                >
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                </button>
            </div>
        </div>
    )
}
