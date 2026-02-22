import { useState } from 'react'

export default function RequirementsForm({ value, onChange }) {
    const [state, setState] = useState(value || {
        floors: 1,
        bedrooms: 1,
        bathrooms: 1,
        kitchen: 1,
        max_area: 100.0,
        balcony: false,
        parking: false,
        pooja_room: false,
    })

    const update = (k, v) => {
        const next = { ...state, [k]: v }
        setState(next)
        onChange && onChange(next)
    }

    return (
        <div style={{ padding: '0.5rem', border: '1px solid var(--border)', borderRadius: 8 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                <label>
                    Floors
                    <input type="number" className="form-input" value={state.floors} min={1} onChange={e => update('floors', parseInt(e.target.value) || 1)} />
                </label>
                <label>
                    Bedrooms
                    <input type="number" className="form-input" value={state.bedrooms} min={0} onChange={e => update('bedrooms', parseInt(e.target.value) || 0)} />
                </label>
                <label>
                    Bathrooms
                    <input type="number" className="form-input" value={state.bathrooms} min={0} onChange={e => update('bathrooms', parseInt(e.target.value) || 0)} />
                </label>
                <label>
                    Kitchen
                    <input type="number" className="form-input" value={state.kitchen} min={0} onChange={e => update('kitchen', parseInt(e.target.value) || 0)} />
                </label>
                <label style={{ gridColumn: '1 / -1' }}>
                    Max Area (sq.m)
                    <input type="number" className="form-input" value={state.max_area} min={1} onChange={e => update('max_area', parseFloat(e.target.value) || 1)} />
                </label>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.6rem' }}>
                <label style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                    <input type="checkbox" checked={state.balcony} onChange={e => update('balcony', e.target.checked)} /> Balcony
                </label>
                <label style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                    <input type="checkbox" checked={state.parking} onChange={e => update('parking', e.target.checked)} /> Parking
                </label>
                <label style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                    <input type="checkbox" checked={state.pooja_room} onChange={e => update('pooja_room', e.target.checked)} /> Pooja Room
                </label>
            </div>
        </div>
    )
}
