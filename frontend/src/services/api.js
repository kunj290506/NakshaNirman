/**
 * API Service — Centralized backend communication.
 * All fetch calls go through here. No direct fetch() in components.
 */

const BASE = '/api'

async function fetchJSON(url, opts = {}) {
    const res = await fetch(`${BASE}${url}`, {
        headers: { 'Content-Type': 'application/json', ...opts.headers },
        ...opts,
    })
    if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || err.error || `Request failed (${res.status})`)
    }
    return res.json()
}

export async function healthCheck() {
    const res = await fetch(`${BASE}/health`, { cache: 'no-store' })
    return res.ok
}

export async function createProject(sessionId, totalArea) {
    return fetchJSON('/project', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, total_area: totalArea }),
    })
}

export async function generateDesign(payload) {
    return fetchJSON('/architect/design', {
        method: 'POST',
        body: JSON.stringify(payload),
    })
}

export async function storeRequirements(requirements) {
    return fetchJSON('/requirements', {
        method: 'POST',
        body: JSON.stringify(requirements),
    })
}

export async function uploadBoundary(file, projectId) {
    const form = new FormData()
    form.append('file', file)
    if (projectId) form.append('project_id', projectId)
    form.append('scale', '1.0')

    const res = await fetch(`${BASE}/upload-boundary`, { method: 'POST', body: form })
    if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Upload failed')
    }
    return res.json()
}

export async function extractBoundary(fileId) {
    return fetchJSON(`/extract-boundary/${encodeURIComponent(fileId)}?scale=1.0`)
}

export async function buildableFootprint(fileId, region = 'india_mvp') {
    return fetchJSON(`/buildable-footprint/${encodeURIComponent(fileId)}?region=${encodeURIComponent(region)}`, {
        method: 'POST',
    })
}

export async function generate3DModel(projectId) {
    return fetchJSON(`/generate-3d/${encodeURIComponent(projectId)}`, {
        method: 'POST',
    })
}

export function createWebSocket(path = '/api/architect/ws') {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return new WebSocket(`${proto}//${window.location.host}${path}`)
}
