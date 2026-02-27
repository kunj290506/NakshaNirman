import { useState } from 'react'

export default function RequirementsForm({ value, onChange }) {
    const [state, setState] = useState(value || {
        floors: 1,
        bedrooms: 2,
        bathrooms: 0,
        kitchen: 1,
        max_area: 0,
        balcony: false,
        parking: false,
        pooja_room: false,
    })

    const update = (k, v) => {
        const next = { ...state, [k]: v }
        setState(next)
        onChange && onChange(next)
    }

    const numberField = (label, key, min, max) => (
        <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0.5rem 0.65rem', background: 'var(--bg-input)',
            borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
        }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>{label}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <button
                    onClick={() => update(key, Math.max(min, (state[key] || min) - 1))}
                    style={{
                        width: 26, height: 26, border: '1px solid var(--border)',
                        borderRadius: '50%', background: 'var(--bg-primary)',
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)',
                    }}
                >-</button>
                <span style={{
                    width: 28, textAlign: 'center', fontSize: '0.9rem',
                    fontWeight: 700, color: 'var(--accent)',
                }}>{state[key] || min}</span>
                <button
                    onClick={() => update(key, Math.min(max, (state[key] || min) + 1))}
                    style={{
                        width: 26, height: 26, border: '1px solid var(--border)',
                        borderRadius: '50%', background: 'var(--bg-primary)',
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)',
                    }}
                >+</button>
            </div>
        </div>
    )

    const toggleField = (label, key, icon) => (
        <button
            onClick={() => update(key, !state[key])}
            style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                padding: '0.45rem 0.75rem',
                border: state[key] ? '1.5px solid var(--accent)' : '1px solid var(--border)',
                borderRadius: 'var(--radius-full)',
                background: state[key] ? 'var(--accent-light)' : 'var(--bg-input)',
                color: state[key] ? 'var(--accent)' : 'var(--text-muted)',
                fontSize: '0.78rem', fontWeight: state[key] ? 700 : 500,
                cursor: 'pointer', transition: 'all 0.15s',
            }}
        >
            {icon}
            {label}
        </button>
    )

    return (
        <div style={{
            display: 'flex', flexDirection: 'column', gap: '0.4rem',
            padding: '0.65rem', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', background: 'var(--bg-primary)',
        }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem' }}>
                {numberField('Floors', 'floors', 1, 4)}
                {numberField('Master Beds', 'bedrooms', 1, 10)}
                {numberField('Extra Bath', 'bathrooms', 0, 6)}
                {numberField('Kitchen', 'kitchen', 1, 3)}
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', padding: '0 0.3rem', lineHeight: 1.3 }}>
                Each Master Bed auto-gets an attached bathroom. Extra Bath = additional common bathrooms.
            </div>

            <div style={{
                display: 'flex', gap: '0.4rem', marginTop: '0.25rem', flexWrap: 'wrap',
            }}>
                {toggleField('Balcony', 'balcony',
                    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
                    </svg>
                )}
                {toggleField('Parking', 'parking',
                    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                )}
                {toggleField('Pooja Room', 'pooja_room',
                    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                    </svg>
                )}
            </div>
        </div>
    )
}
