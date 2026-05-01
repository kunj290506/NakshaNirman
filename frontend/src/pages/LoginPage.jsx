import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { loginUser, signupUser } from '../services/api'
import { isAuthenticated, saveAuthInWeb } from '../services/auth'

function normalizeMode(rawMode) {
  return rawMode === 'signup' ? 'signup' : 'login'
}

export default function LoginPage({ initialMode = 'login' }) {
  const navigate = useNavigate()
  const [mode, setMode] = useState(normalizeMode(initialMode))
  const [email, setEmail] = useState('')
  const [userId, setUserId] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setMode(normalizeMode(initialMode))
    setError('')
  }, [initialMode])

  function switchMode(nextMode) {
    const normalized = normalizeMode(nextMode)
    setMode(normalized)
    setError('')
    if (normalized === 'signup') {
      navigate('/signup', { replace: true })
    } else {
      navigate('/login', { replace: true })
    }
  }

  useEffect(() => {
    if (isAuthenticated()) {
      navigate('/dashboard', { replace: true })
    }
  }, [navigate])

  async function handleSubmit(e) {
    e.preventDefault()
    if (loading) return

    const cleanEmail = String(email || '').trim().toLowerCase()
    const cleanUserId = String(userId || '').trim()
    const cleanPassword = String(password || '')
    if (!cleanUserId || !cleanPassword) {
      setError('Username and password are required.')
      return
    }

    if (mode === 'signup') {
      if (!cleanEmail || !/^\S+@\S+\.\S+$/.test(cleanEmail)) {
        setError('Please enter a valid email address.')
        return
      }
      if (cleanPassword.length < 6) {
        setError('Password should be at least 6 characters long.')
        return
      }
    }

    setLoading(true)
    setError('')
    try {
      let response = null
      if (mode === 'signup') {
        response = await signupUser({
          email: cleanEmail,
          userId: cleanUserId,
          password: cleanPassword,
        })
      } else {
        response = await loginUser(cleanUserId, cleanPassword)
      }

      saveAuthInWeb({
        userId: response?.user_id || cleanUserId,
        fullName: response?.full_name || '',
        email: response?.email || cleanEmail,
        authMode: mode,
      })
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError(err?.message || 'Authentication failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className='auth-shell'>
      <form className='auth-card' onSubmit={handleSubmit}>
        <div>
          <h1>{mode === 'signup' ? 'Create an account' : 'Welcome back'}</h1>
          <p className='auth-subtitle'>
            {mode === 'signup'
              ? 'Start designing precise floor plans today.'
              : 'Enter your credentials to access the workspace.'}
          </p>
        </div>

        {mode === 'signup' && (
          <div>
            <label className='auth-label'>Email</label>
            <input
              className='auth-input'
              autoComplete='email'
              type='email'
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder='you@example.com'
            />
          </div>
        )}

        <div>
          <label className='auth-label'>Username</label>
          <input
            className='auth-input'
            autoComplete='username'
            type='text'
            required
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder='architect_01'
          />
        </div>

        <div>
          <label className='auth-label'>Password</label>
          <input
            className='auth-input'
            autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
            type='password'
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder='••••••••'
          />
        </div>

        {error && <p className='auth-error'>{error}</p>}

        <div className='auth-actions'>
          <button className='fw-button fw-button-solid pulse-on-hover glow-on-hover' type='submit' disabled={loading} style={{ width: '100%', padding: '12px 24px', fontSize: '0.95rem' }}>
            {loading ? (mode === 'signup' ? 'Creating...' : 'Signing in...') : 'Continue'}
          </button>
        </div>

        <div style={{ textAlign: 'center', marginTop: '16px', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--saas-muted)' }}>
            {mode === 'signup' ? 'Already have an account? ' : "Don't have an account? "}
          </span>
          <button
            type='button'
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--saas-primary)',
              fontWeight: 600,
              cursor: 'pointer',
              textDecoration: 'underline'
            }}
            onClick={() => switchMode(mode === 'signup' ? 'login' : 'signup')}
            disabled={loading}
          >
            {mode === 'signup' ? 'Log in' : 'Sign up'}
          </button>
        </div>

        <div style={{ textAlign: 'center', marginTop: '8px', fontSize: '0.8rem' }}>
          <Link to='/' style={{ color: 'var(--saas-muted)', textDecoration: 'none' }}>
            &larr; Back to Landing Page
          </Link>
        </div>
      </form>
    </div>
  )
}
