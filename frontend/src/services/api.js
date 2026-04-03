const API_BASE = '';  // Use Vite proxy — requests go through /api
const REQUEST_TIMEOUT_MS = 150000; // Base timeout for first attempt
const RETRY_TIMEOUT_INCREMENT_MS = 60000; // Give retries substantial extra warm-up time
const MAX_ATTEMPTS = 2;
const RETRYABLE_HTTP = new Set([408, 425, 429, 500, 502, 503, 504]);

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function generatePlanAttempt(data, attempt, maxAttempts) {
  const controller = new AbortController();
  const timeoutMs = REQUEST_TIMEOUT_MS + ((attempt - 1) * RETRY_TIMEOUT_INCREMENT_MS);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Client-Attempt': String(attempt),
      },
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
        'Generation timed out after automatic retries. Please try again; fast fallback mode remains enabled.'
      );
    }
    throw e;
  }
}

export async function generatePlan(data) {
  return generatePlanAttempt(data, 1, MAX_ATTEMPTS);
}

async function readAuthResponse(resp, fallbackMessage) {
  const body = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    throw new Error(body.detail || fallbackMessage)
  }
  return body
}

export async function loginUser(userId, password) {
  const resp = await fetch(`${API_BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, password }),
    cache: 'no-store',
  })

  return readAuthResponse(resp, 'Login failed')
}

export async function signupUser({ email = '', userId, password, fullName = '' }) {
  const resp = await fetch(`${API_BASE}/api/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, password, email, full_name: fullName }),
    cache: 'no-store',
  })

  return readAuthResponse(resp, 'Signup failed')
}

// Backward compatibility for older callers.
export async function saveAndLoginUser(userId, password) {
  const resp = await fetch(`${API_BASE}/api/auth/save-and-login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, password }),
    cache: 'no-store',
  })

  return readAuthResponse(resp, 'Login failed')
}

export function getDownloadUrl(dxfUrl) {
  if (!dxfUrl) return null;
  // Use proxy path (no need for absolute URL)
  return `${API_BASE}${dxfUrl}`;
}
