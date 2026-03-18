import { useMemo, useState } from 'react'

function parseGenerateToken(reply) {
  if (!reply || !reply.includes('GENERATE_PLAN:')) return null
  const token = reply.split('GENERATE_PLAN:')[1]?.trim()
  if (!token) return null
  try {
    return JSON.parse(token)
  } catch {
    return null
  }
}

function payloadToRooms(payload) {
  const rooms = [
    { room_type: 'master_bedroom', quantity: 1 },
    { room_type: 'living', quantity: 1 },
    { room_type: 'kitchen', quantity: 1 },
    { room_type: 'dining', quantity: 1 },
  ]

  const bedrooms = Number(payload.bedrooms || 2)
  const bathrooms = Number(payload.bathrooms || bedrooms)

  if (bedrooms > 1) {
    rooms.push({ room_type: 'bedroom', quantity: bedrooms - 1 })
  }
  rooms.push({ room_type: 'bathroom', quantity: bathrooms })

  ;(payload.extras || []).forEach((extra) => rooms.push({ room_type: extra, quantity: 1 }))
  return rooms
}

export default function AIDesignChat({ onGenerate, loading }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Namaste. I can design your ground-floor plan. Share plot size and BHK, for example: 30x40, 2BHK.',
      ts: Date.now(),
    },
  ])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)

  const busy = loading || sending

  const history = useMemo(
    () => messages.map((m) => ({ role: m.role, content: m.content })),
    [messages]
  )

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return

    const nextMessages = [...messages, { role: 'user', content: text, ts: Date.now() }]
    setMessages(nextMessages)
    setInput('')
    setSending(true)

    try {
      const res = await fetch('/api/architect/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history }),
      })
      const data = await res.json().catch(() => ({}))
      const reply = data.reply || 'Please share your plot size and BHK requirement.'

      setMessages((prev) => [...prev, { role: 'assistant', content: reply, ts: Date.now() }])

      const payload = data.generate_payload || parseGenerateToken(reply)
      if (payload) {
        const rooms = payloadToRooms(payload)
        const totalArea = Number(payload.total_area || (payload.plot_width * payload.plot_length))
        onGenerate(rooms, totalArea, payload)
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'I could not reach the chat service. Please try again.',
          ts: Date.now(),
        },
      ])
    } finally {
      setSending(false)
    }
  }

  return (
    <div style={{ display: 'grid', gridTemplateRows: '1fr auto', height: '100%', gap: '0.75rem' }}>
      <div style={{ overflowY: 'auto', display: 'grid', gap: '0.55rem', paddingRight: '0.25rem' }}>
        {messages.map((m, idx) => (
          <div
            key={`${m.ts}-${idx}`}
            style={{
              justifySelf: m.role === 'user' ? 'end' : 'start',
              background: m.role === 'user' ? '#1f3a5f' : '#ffffff',
              color: m.role === 'user' ? '#fff' : '#1d1d1d',
              borderRadius: '12px',
              border: m.role === 'assistant' ? '1px solid #e5e7eb' : 'none',
              padding: '0.55rem 0.7rem',
              fontSize: '0.86rem',
              maxWidth: '92%',
              whiteSpace: 'pre-wrap',
            }}
          >
            {m.content}
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '0.5rem' }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder='Type requirements, e.g. 30x40 3BHK with pooja and study'
          rows={2}
          style={{ resize: 'none' }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
        />
        <button className='btn btn-primary' onClick={send} disabled={busy}>
          {busy ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
