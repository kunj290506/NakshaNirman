const AUTH_STORAGE_KEY = 'nakshanirman_auth_user'

export function saveAuthInWeb(sessionOrUserId, legacyPassword = '') {
  // Supports old saveAuthInWeb(userId, password) calls while preferring object payload.
  const payload =
    typeof sessionOrUserId === 'object' && sessionOrUserId !== null
      ? {
          userId: String(sessionOrUserId.userId || '').trim(),
          fullName: String(sessionOrUserId.fullName || '').trim(),
          email: String(sessionOrUserId.email || '').trim(),
          authMode: String(sessionOrUserId.authMode || 'login'),
          loggedInAt: new Date().toISOString(),
        }
      : {
          userId: String(sessionOrUserId || '').trim(),
          fullName: '',
          email: '',
          authMode: legacyPassword ? 'legacy' : 'login',
          loggedInAt: new Date().toISOString(),
        }

  if (!payload.userId) {
    return null
  }

  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(payload))
  return payload
}

export function getAuthFromWeb() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed?.userId) return null
    return parsed
  } catch {
    return null
  }
}

export function clearAuthFromWeb() {
  localStorage.removeItem(AUTH_STORAGE_KEY)
}

export function isAuthenticated() {
  return Boolean(getAuthFromWeb())
}

export { AUTH_STORAGE_KEY }
