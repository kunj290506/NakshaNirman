const API_BASE = '';  // Use Vite proxy — requests go through /api
const REQUEST_TIMEOUT_MS = 90000; // 90 seconds
const MAX_ATTEMPTS = 1;
const RETRYABLE_HTTP = new Set([408, 425, 429, 500, 502, 503, 504]);

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function generatePlanAttempt(data, attempt, maxAttempts) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const resp = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      cache: 'no-store',
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const detail = err.detail || `Server error: ${resp.status}`;
      const retriable = RETRYABLE_HTTP.has(resp.status) && attempt < maxAttempts;
      if (retriable) {
        await delay(1200 * attempt);
        return generatePlanAttempt(data, attempt + 1, maxAttempts);
      }
      throw new Error(detail);
    }

    return resp.json();
  } catch (e) {
    clearTimeout(timeout);

    const isTimeout = e?.name === 'AbortError';
    if ((isTimeout || e instanceof TypeError) && attempt < maxAttempts) {
      await delay(1200 * attempt);
      return generatePlanAttempt(data, attempt + 1, maxAttempts);
    }

    if (isTimeout) {
      throw new Error(
        'Generation is taking too long due to model load. Please try again; fast fallback mode is enabled.'
      );
    }
    throw e;
  }
}

export async function generatePlan(data) {
  return generatePlanAttempt(data, 1, MAX_ATTEMPTS);
}

export function getDownloadUrl(dxfUrl) {
  if (!dxfUrl) return null;
  // Use proxy path (no need for absolute URL)
  return `${API_BASE}${dxfUrl}`;
}
