import { useState, useRef, useEffect, useCallback } from 'react'

// ── Text cleaning ──

function cleanInput(raw) {
  return raw
    .trim()
    .replace(/[*_~`#>|]/g, '')
    .replace(/\s{2,}/g, ' ')
    .replace(/^[-=_*]{2,}/gm, '')
    .replace(/[-=_*]{2,}$/gm, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
    .trim()
}

function inlineFormat(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} style={{ fontWeight: 600 }}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

function formatReply(text) {
  if (!text) return null

  let clean = text
    .replace(/[═─━┃┄┅┆┇┈┉┊┋│┼╭╮╯╰╱╲╳]/g, '')
    .replace(/[▸▹►▻◂◃◄◅▴▵▶▷▼▽◆◇○●◉]/g, '')
    .replace(/^[═─━=*#]{3,}.*$/gm, '')
    .replace(/\*{3,}/g, '')
    .replace(/_{3,}/g, '')

  const lines = clean.split('\n').filter(l => l.trim())
  const elements = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i].trim()
    if (!line) { i++; continue }

    if (/^\*\*[^*]+\*\*$/.test(line) || /^#{1,3}\s/.test(line)) {
      const txt = line.replace(/^\*\*/, '').replace(/\*\*$/, '').replace(/^#{1,3}\s/, '')
      elements.push(
        <p key={i} style={{ fontWeight: 700, color: '#1a1a2e', marginBottom: 4, marginTop: i > 0 ? 10 : 0, fontSize: 13 }}>
          {txt}
        </p>
      )
      i++; continue
    }

    if (/^[-•*]\s/.test(line) || /^\d+\.\s/.test(line)) {
      const bullets = []
      while (i < lines.length && (/^[-•*]\s/.test(lines[i]?.trim()) || /^\d+\.\s/.test(lines[i]?.trim()))) {
        const bt = lines[i].trim().replace(/^[-•*]\s/, '').replace(/^\d+\.\s/, '')
        bullets.push(inlineFormat(bt))
        i++
      }
      elements.push(
        <ul key={`ul-${i}`} style={{ paddingLeft: 16, margin: '4px 0' }}>
          {bullets.map((b, j) => (
            <li key={j} style={{ marginBottom: 2, fontSize: 13, color: '#2a2a3e', lineHeight: 1.5 }}>{b}</li>
          ))}
        </ul>
      )
      continue
    }

    elements.push(
      <p key={i} style={{ margin: '3px 0', fontSize: 13, color: '#2a2a3e', lineHeight: 1.55 }}>
        {inlineFormat(line)}
      </p>
    )
    i++
  }

  return <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>{elements}</div>
}

// ── Error messages ──

const ERROR_MESSAGES = {
  PARSE_ERROR: 'Received an unexpected response. Please try again.',
  SEND_FAILED: 'Could not send your message. Please check your connection.',
  UPLOAD_FAILED: 'Could not read the uploaded file. Try a different format.',
  BACKEND_DOWN: 'The design server is not running. Please start the backend on port 8000.',
}

// ── Sub-components ──

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'

  const timeStr = msg.timestamp
    ? new Date(msg.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true })
    : ''

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: isUser ? 'flex-end' : 'flex-start',
      maxWidth: '88%',
      alignSelf: isUser ? 'flex-end' : 'flex-start',
    }}>
      <div style={{
        background: isUser ? '#1a3a6b' : '#ffffff',
        color: isUser ? '#ffffff' : '#1a1a2e',
        padding: '9px 13px',
        borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
        maxWidth: '100%',
        wordBreak: 'break-word',
      }}>
        {isUser
          ? <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5 }}>{msg.content}</p>
          : formatReply(msg.content)
        }
      </div>
      <span style={{ fontSize: 10, color: '#999', marginTop: 3, paddingLeft: 4, paddingRight: 4 }}>
        {timeStr}
      </span>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 4,
      padding: '8px 13px',
      background: '#fff',
      borderRadius: '16px 16px 16px 4px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      width: 'fit-content',
    }}>
      {[0, 1, 2].map(i => (
        <span
          key={i}
          style={{
            width: 7,
            height: 7,
            borderRadius: '50%',
            background: '#94a3b8',
            display: 'inline-block',
            animation: 'typingBounce 1.2s ease-in-out infinite',
            animationDelay: `${i * 0.2}s`,
          }}
        />
      ))}
      <style>{`
        @keyframes typingBounce {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-5px); opacity: 1; }
        }
      `}</style>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function AIDesignChat({
  onGenerate,
  onBoundaryUpload,
  loading,
  projectId,
  onReviewPlan,
  plan,
}) {
  const [messages, setMessages] = useState([{
    id: 'welcome',
    role: 'assistant',
    content: 'Welcome! I am your AI architect.\n\nTell me about your plot and I will design your floor plan.\n\nExample: I have a 30 by 40 feet plot and need 3 bedrooms with a pooja room.',
    timestamp: new Date(),
    status: 'sent',
  }])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [ws, setWs] = useState(null)
  const [wsReady, setWsReady] = useState(false)
  const [currentStage, setCurrentStage] = useState('chat')
  const [extractedData, setExtractedData] = useState({ rooms: [], total_area: null })
  const [layoutJson, setLayoutJson] = useState(null)
  const [validationReport, setValidationReport] = useState(null)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const fileInputRef = useRef(null)
  const extractedRef = useRef({ rooms: [], total_area: null })

  // Keep ref in sync with state
  useEffect(() => { extractedRef.current = extractedData }, [extractedData])

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // Auto-focus input
  useEffect(() => { inputRef.current?.focus() }, [])

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 100) + 'px'
    }
  }, [input])

  // ── addAssistantMessage ──
  const addAssistantMessage = useCallback((content) => {
    const text = (content || '').toString()
    if (!text) return
    setMessages(prev => {
      // Deduplicate by content within last 2 messages
      const last = prev[prev.length - 1]
      if (last && last.role === 'assistant' && last.content === text) return prev
      return [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: text,
        timestamp: new Date(),
        status: 'sent',
      }]
    })
  }, [])

  // ── handleDataFromResponse ──
  const handleDataFromResponse = useCallback((data) => {
    if (data.stage || data.mode) setCurrentStage(data.stage || data.mode)

    if (data.extracted_data) {
      const ed = data.extracted_data
      setExtractedData(prev => ({
        ...prev,
        total_area: ed.total_area || prev.total_area,
        rooms: ed.rooms?.length > 0 ? ed.rooms : prev.rooms,
      }))
    }

    // Handle engine 'collected' data
    if (data.collected) {
      setExtractedData(prev => {
        const updated = { ...prev }
        if (data.collected.total_area) updated.total_area = data.collected.total_area
        if (data.collected.bedrooms) {
          const rooms = []
          rooms.push({ room_type: 'master_bedroom', quantity: 1 })
          if (data.collected.bedrooms > 1) {
            rooms.push({ room_type: 'bedroom', quantity: data.collected.bedrooms - 1 })
          }
          if (data.collected.bathrooms) {
            rooms.push({ room_type: 'bathroom', quantity: data.collected.bathrooms })
          }
          rooms.push({ room_type: 'living', quantity: 1 })
          rooms.push({ room_type: 'kitchen', quantity: 1 })
          if (data.collected.extras) {
            data.collected.extras.forEach(e => rooms.push({ room_type: e, quantity: 1 }))
          }
          updated.rooms = rooms
        }
        return updated
      })
    }

    if (data.layout_json || data.layout) setLayoutJson(data.layout_json || data.layout)
    if (data.validation_report || data.validation) setValidationReport(data.validation_report || data.validation)

    // Auto-generate
    if (data.should_generate && data.extracted_data) {
      const rooms = data.extracted_data.rooms || extractedRef.current.rooms
      const totalArea = data.extracted_data.total_area || extractedRef.current.total_area
      if (totalArea && rooms?.length > 0) {
        onGenerate(rooms, totalArea, {
          floors: data.extracted_data.floors || data.collected?.floors || 1,
          bedrooms: data.extracted_data.bedrooms || data.collected?.bedrooms || rooms.filter(r => r.room_type === 'bedroom' || r.room_type === 'master_bedroom').reduce((s, r) => s + (r.quantity || 1), 0),
          bathrooms: data.extracted_data.bathrooms || data.collected?.bathrooms || rooms.filter(r => r.room_type === 'bathroom').reduce((s, r) => s + (r.quantity || 1), 0),
          extras: data.extracted_data.extras || data.collected?.extras || [],
        })
      }
    }

    // Also handle REST 'ready_to_generate'
    if (data.ready_to_generate) {
      const totalArea = data.extracted_data?.total_area || extractedRef.current.total_area
      const rooms = data.rooms?.length > 0 ? data.rooms : (data.extracted_data?.rooms || extractedRef.current.rooms)
      if (totalArea && rooms?.length > 0) {
        onGenerate(rooms, totalArea, {
          floors: data.extracted_data?.floors || 1,
          bedrooms: data.extracted_data?.bedrooms || rooms.filter(r => r.room_type === 'bedroom' || r.room_type === 'master_bedroom').reduce((s, r) => s + (r.quantity || 1), 0),
          bathrooms: data.extracted_data?.bathrooms || rooms.filter(r => r.room_type === 'bathroom').reduce((s, r) => s + (r.quantity || 1), 0),
          extras: data.extracted_data?.extras || [],
        })
      }
    }
  }, [onGenerate])

  // ── WebSocket ──
  useEffect(() => {
    let socket = null
    let retryTimer = null
    let retries = 0

    const connect = () => {
      try {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        socket = new WebSocket(`${proto}//${window.location.host}/api/architect/ws`)

        socket.onopen = () => {
          setWs(socket)
          setWsReady(true)
          retries = 0
        }

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            setIsTyping(false)

            const reply = data.reply || data.message || data.text || data.content || ''
            if (reply) addAssistantMessage(reply)

            handleDataFromResponse(data)
          } catch {
            setIsTyping(false)
            addAssistantMessage(ERROR_MESSAGES.PARSE_ERROR)
          }
        }

        socket.onerror = () => {
          setWsReady(false)
        }

        socket.onclose = () => {
          setWs(null)
          setWsReady(false)
          const delay = Math.min(3000 * Math.pow(2, retries), 12000)
          retries++
          retryTimer = setTimeout(connect, delay)
        }
      } catch {
        setWsReady(false)
      }
    }

    connect()

    return () => {
      clearTimeout(retryTimer)
      socket?.close()
    }
  }, [addAssistantMessage, handleDataFromResponse])

  // ── sendMessage ──
  const sendMessage = useCallback(() => {
    const text = cleanInput(input)
    if (!text || loading || isTyping) return

    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date(),
      status: 'sent',
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsTyping(true)

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ message: text, project_id: projectId }))
    } else {
      fetch('/api/architect/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: [], project_id: projectId }),
      })
        .then(res => res.json())
        .then(data => {
          setIsTyping(false)
          const reply = data.reply || data.message || 'Understood. Please continue.'
          addAssistantMessage(reply)
          handleDataFromResponse(data)
        })
        .catch(() => {
          setIsTyping(false)
          addAssistantMessage(ERROR_MESSAGES.SEND_FAILED)
        })
    }
  }, [input, ws, loading, isTyping, projectId, addAssistantMessage, handleDataFromResponse])

  // ── keyboard ──
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // ── file upload ──
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file || !onBoundaryUpload) return

    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: 'user',
      content: `Uploaded: ${file.name}`,
      timestamp: new Date(),
      status: 'sent',
    }])

    setIsTyping(true)
    try {
      const result = await onBoundaryUpload(file)
      setIsTyping(false)
      if (result) {
        addAssistantMessage(
          `Boundary extracted successfully.\n\n` +
          `- Vertices: ${result.num_vertices || 'Several'}\n` +
          `- Area: ${result.area ? Math.round(result.area) : 'Calculated'} sq units\n` +
          `- Usable area after setback: ${result.usable_area ? Math.round(result.usable_area) : 'N/A'} sq units\n\n` +
          `Now describe your rooms and I will design the layout.`
        )
      } else {
        addAssistantMessage(ERROR_MESSAGES.UPLOAD_FAILED)
      }
    } catch {
      setIsTyping(false)
      addAssistantMessage(ERROR_MESSAGES.UPLOAD_FAILED)
    }
  }

  // ── manual generate ──
  const handleGenerateFromAI = () => {
    const current = extractedRef.current
    if (current.total_area && current.rooms?.length > 0) {
      onGenerate(current.rooms, current.total_area)
    }
  }

  // ── disabled state for input ──
  const inputDisabled = loading || isTyping
  const sendDisabled = !input.trim() || loading || isTyping

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fff' }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #eee',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        background: '#fff',
        flexShrink: 0,
      }}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%',
          background: wsReady ? '#22c55e' : '#f59e0b',
        }} />
        <span style={{ fontSize: 13, fontWeight: 600, color: '#1a1a2e' }}>
          AI Architect
        </span>
        <span style={{ fontSize: 11, color: '#888', marginLeft: 'auto' }}>
          {wsReady ? 'Connected' : 'Connecting...'}
        </span>
        {extractedData.rooms?.length > 0 && extractedData.total_area && (
          <button
            onClick={handleGenerateFromAI}
            disabled={loading}
            style={{
              fontSize: 11,
              fontWeight: 600,
              padding: '4px 10px',
              background: loading ? '#e2e8f0' : '#1a3a6b',
              color: loading ? '#94a3b8' : '#fff',
              border: 'none',
              borderRadius: 6,
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            Generate Plan
          </button>
        )}
      </div>

      {/* Messages area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          background: '#f8f9fb',
        }}
      >
        {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
        {isTyping && <TypingIndicator />}
        {loading && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '9px 13px',
            background: '#fff',
            borderRadius: '16px 16px 16px 4px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
            width: 'fit-content',
            fontSize: 13,
            color: '#1a1a2e',
            fontWeight: 500,
          }}>
            Generating your floor plan...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div style={{
        borderTop: '1px solid #eee',
        padding: '10px 14px 12px',
        background: '#fff',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        flexShrink: 0,
      }}>
        {/* Input row */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={inputDisabled ? 'AI is typing...' : 'Describe your plot and requirements...'}
            rows={1}
            style={{
              width: '100%',
              resize: 'none',
              border: '1px solid #dde1e7',
              borderRadius: 12,
              padding: '10px 14px',
              fontSize: 13,
              fontFamily: 'inherit',
              outline: 'none',
              lineHeight: 1.5,
              maxHeight: 100,
              overflow: 'auto',
              background: '#fff',
              color: '#1a1a2e',
              boxSizing: 'border-box',
            }}
            disabled={inputDisabled}
          />
          <button
            onClick={sendMessage}
            disabled={sendDisabled}
            style={{
              padding: '10px 16px',
              background: sendDisabled ? '#e2e8f0' : '#1a3a6b',
              color: sendDisabled ? '#94a3b8' : '#fff',
              border: 'none',
              borderRadius: 10,
              cursor: sendDisabled ? 'not-allowed' : 'pointer',
              fontSize: 13,
              fontWeight: 600,
              minWidth: 64,
              transition: 'background 0.15s',
              flexShrink: 0,
            }}
          >
            {isTyping ? '...' : 'Send'}
          </button>
        </div>

        {/* Upload row */}
        {onBoundaryUpload && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                fontSize: 11,
                color: '#64748b',
                background: 'none',
                border: '1px dashed #cbd5e1',
                borderRadius: 6,
                padding: '4px 10px',
                cursor: 'pointer',
              }}
            >
              Upload boundary
            </button>
            <span style={{ fontSize: 10, color: '#aaa' }}>DXF or image</span>
            <input ref={fileInputRef} type="file" accept=".dxf,.png,.jpg,.jpeg"
              style={{ display: 'none' }} onChange={handleFileUpload} />
          </div>
        )}

        {/* Hint */}
        <span style={{ fontSize: 10, color: '#bbb' }}>
          Press Enter to send, Shift+Enter for new line
        </span>
      </div>
    </div>
  )
}
