import { useState, useRef, useEffect, useCallback } from 'react'

export default function AIDesignChat({ onGenerate, onBoundaryUpload, loading, projectId, onReviewPlan, plan }) {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: "Welcome to AI Design Advisor powered by Grok!\n\nI'm your senior architect. Tell me about your dream home -- for example:\n\n\"I want a 1200 sqft 3BHK house with Vastu compliance, 2 bathrooms, a pooja room, and parking.\"\n\nI'll analyze your requirements, check Vastu, structural feasibility, and Indian Building Code compliance before generating the plan.",
            provider: 'system',
        }
    ])
    const [input, setInput] = useState('')
    const [isTyping, setIsTyping] = useState(false)
    const [ws, setWs] = useState(null)
    const [wsReady, setWsReady] = useState(false)
    const [extractedData, setExtractedData] = useState({ rooms: [], total_area: null })
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

    // WebSocket connection to AI Design endpoint
    useEffect(() => {
        let socket = null
        let reconnectTimeout = null

        const connect = () => {
            try {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
                const wsUrl = `${proto}//${window.location.host}/api/ai-design/chat`
                socket = new WebSocket(wsUrl)

                socket.onopen = () => {
                    setWs(socket)
                    setWsReady(true)
                }

                socket.onmessage = (event) => {
                    const data = JSON.parse(event.data)
                    setIsTyping(false)

                    const msgObj = {
                        role: 'assistant',
                        content: data.reply,
                        provider: data.provider || 'unknown',
                        extractedData: data.extracted_data,
                    }
                    setMessages(prev => [...prev, msgObj])

                    if (data.extracted_data) {
                        setExtractedData(prev => {
                            const updated = { ...prev }
                            if (data.extracted_data.total_area) updated.total_area = data.extracted_data.total_area
                            if (data.extracted_data.rooms) {
                                updated.rooms = [...(updated.rooms || []), ...data.extracted_data.rooms]
                            }
                            return updated
                        })
                    }

                    if (data.should_generate) {
                        const current = extractedRef.current
                        if (current.total_area && current.rooms?.length > 0) {
                            onGenerate(current.rooms, current.total_area)
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
                            onGenerate(rooms, totalArea)
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
                content: `Boundary extracted successfully! ${result.num_vertices || 'Several'} vertices, ${result.area ? result.area.toFixed(0) : 'calculated'} sq units area.\n\nUsable area after setback: ${result.usable_area ? result.usable_area.toFixed(0) : 'N/A'} sq units.\n\nNow describe what rooms you'd like. I'll analyze them for Vastu compliance and structural feasibility.`,
                provider: 'system',
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
            grok: { label: 'Grok', color: '#8b5cf6', bg: '#ede9fe' },
            groq: { label: 'Groq', color: '#f97316', bg: '#fff7ed' },
            fallback: { label: 'Offline', color: '#6b7280', bg: '#f3f4f6' },
            system: { label: 'System', color: '#059669', bg: '#ecfdf5' },
            error: { label: 'Error', color: '#dc2626', bg: '#fef2f2' },
        }
        return badges[provider] || badges.fallback
    }

    const renderScoreBadge = (score, label) => {
        const color = score >= 8 ? '#059669' : score >= 5 ? '#d97706' : '#dc2626'
        const bg = score >= 8 ? '#ecfdf5' : score >= 5 ? '#fffbeb' : '#fef2f2'
        return (
            <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
                padding: '0.15rem 0.5rem', borderRadius: '99px',
                fontSize: '0.72rem', fontWeight: 700, color, background: bg,
                border: `1px solid ${color}20`,
            }}>
                {label}: {score}/10
            </span>
        )
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Header with AI provider status */}
            <div style={{
                padding: '0.5rem 0.75rem',
                background: 'linear-gradient(135deg, #7c3aed08 0%, #6d28d908 100%)',
                border: '1px solid #8b5cf620',
                borderRadius: 'var(--radius-sm)',
                marginBottom: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>AI</span>
                    <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#5b21b6' }}>
                        AI Design Advisor
                    </span>
                    <span style={{
                        width: 8, height: 8, borderRadius: 99,
                        background: wsReady ? '#10b981' : '#f59e0b',
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
                            {isReviewing ? '...' : ''} AI Review
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

            {/* Messages */}
            <div className="chat-messages" style={{ flex: 1, overflowY: 'auto', padding: '0.5rem 0' }}>
                {messages.map((msg, i) => {
                    const badge = msg.provider ? getProviderBadge(msg.provider) : null
                    const isLongMessage = msg.content && msg.content.length > 300

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

                            {/* Vastu recommendations */}
                            {msg.vastuRecommendations?.length > 0 && (
                                <div style={{
                                    marginTop: '0.5rem', padding: '0.4rem 0.6rem',
                                    background: '#fef3c7', borderRadius: '6px',
                                    border: '1px solid #fbbf24',
                                    fontSize: '0.75rem',
                                }}>
                                    <strong style={{ color: '#92400e' }}>Vastu:</strong>
                                    {msg.vastuRecommendations.map((v, vi) => (
                                        <div key={vi} style={{ marginTop: '0.2rem', color: '#78350f' }}>
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
                            Grok is thinking...
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
                    placeholder={loading ? 'Generating plan...' : 'Describe your dream home to the AI architect...'}
                    disabled={loading}
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
