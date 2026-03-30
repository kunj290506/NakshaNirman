const API_BASE = 'http://localhost:8000';

export async function generatePlan(data) {
  const resp = await fetch(`${API_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${resp.status}`);
  }
  return resp.json();
}

export function getDownloadUrl(dxfUrl) {
  if (!dxfUrl) return null;
  return `${API_BASE}${dxfUrl}`;
}
