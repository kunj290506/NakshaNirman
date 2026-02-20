import { useState, useRef, useEffect, useCallback } from 'react'

export default function ChatInterface({ onGenerate, onBoundaryUpload, loading }) {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: "Hi! I'm ready to help you design a floor plan. Describe what you need -- for example:\n\n\"I want a 1500 sq ft 2BHK with a living room, kitchen, dining, master bedroom with attached bath, one guest bedroom, and a balcony.\"" }
    ])
    const [input, setInput] = useState('')
    const [isTyping, setIsTyping] = useState(false)
    const [ws, setWs] = useState(null)
    const [wsReady, setWsReady] = useState(false)
    const [extractedData, setExtractedData] = useState({ rooms: [], total_area: null })
    const messagesEndRef = useRef(null)
    const fileInputRef = useRef(null)
    const extractedRef = useRef({ rooms: [], total_area: null })

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isTyping])

    // Keep extractedRef in sync
    useEffect(() => {
        extractedRef.current = extractedData
    }, [extractedData])

    // WebSocket connection
    useEffect(() => {
        let socket = null
        let reconnectTimeout = null

        const connect = () => {
            try {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
                const wsUrl = `${proto}//${window.location.host}/api/chat`
                socket = new WebSocket(wsUrl)

                socket.onopen = () => {
                    setWs(socket)
                    setWsReady(true)
                }

                socket.onmessage = (event) => {
                    const data = JSON.parse(event.data)
                    setIsTyping(false)

                    setMessages(prev => [...prev, { role: 'assistant', content: data.reply }])

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

    const sendMessage = useCallback(async () => {
        if (!input.trim() || loading) return

        const userMsg = { role: 'user', content: input.trim() }
        setMessages(prev => [...prev, userMsg])
        setIsTyping(true)

        try {
            if (ws && ws.readyState === WebSocket.OPEN) {
                // Use WebSocket if available
                ws.send(JSON.stringify({ message: input.trim() }))
            } else {
                // Fallback to HTTP POST
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: input.trim(),
                        project_id: null,
                        history: messages.map(m => ({
                            role: m.role,
                            content: m.content
                        }))
                    })
                })

                if (!response.ok) {
                    throw new Error('Chat service unavailable')
                }

                const data = await response.json()
                setIsTyping(false)
                setMessages(prev => [...prev, { role: 'assistant', content: data.reply }])

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
        } catch (err) {
            setIsTyping(false)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "I encountered a connection issue. Please try again or use the Form tab which works offline.",
            }])
        }

        setInput('')
    }, [input, ws, loading, messages, onGenerate])

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
                content: `Boundary extracted successfully! It has ${result.num_vertices || 'several'} vertices and an area of ${result.area ? result.area.toFixed(0) : 'calculated'} sq units. Now describe what rooms you would like inside this boundary.`,
            }])
        } else {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Sorry, I could not extract a boundary from that file. Please try a clearer image or a DXF file.',
            }])
        }
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {!wsReady && (
                <div style={{
                    padding: '0.5rem 0.8rem',
                    background: 'var(--warning-bg)',
                    border: '1px solid #fde68a',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.78rem',
                    color: '#92400e',
                    marginBottom: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                }}>
                    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    Chat connecting... Use the Form tab in the meantime.
                </div>
            )}
            <div className="chat-messages" style={{ flex: 1, overflowY: 'auto', padding: '1rem 0' }}>
                {messages.map((msg, i) => (
                    <div key={i} className={`chat-bubble ${msg.role}`}>
                        {msg.content.split('\n').map((line, j) => (
                            <span key={j}>{line}{j < msg.content.split('\n').length - 1 && <br />}</span>
                        ))}
                    </div>
                ))}
                {isTyping && (
                    <div className="typing-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                    </div>
                )}
                {loading && (
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.6rem',
                        padding: '0.75rem 1rem',
                        background: 'var(--accent-light)',
                        border: '1px solid var(--accent-soft)',
                        borderRadius: 'var(--radius-md)',
                        alignSelf: 'flex-start',
                        maxWidth: '88%',
                    }}>
                        <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2, marginBottom: 0 }}></div>
                        <span style={{ fontSize: '0.82rem', color: 'var(--accent)', fontWeight: 600 }}>
                            Generating your floor plan...
                        </span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

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
                    placeholder={loading ? 'Generating plan...' : 'Describe your dream home...'}
                    disabled={loading}
                />
                <button
                    className="btn btn-primary btn-sm"
                    onClick={sendMessage}
                    disabled={loading || !input.trim()}
                    style={{ opacity: (loading || !input.trim()) ? 0.5 : 1 }}
                >
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                </button>
            </div>
        </div>
    )
}
