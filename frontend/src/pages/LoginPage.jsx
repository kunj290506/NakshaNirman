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
    <div className='auth-uiverse-shell'>
      <form className='auth-uiverse-card' onSubmit={handleSubmit}>
        <p className='auth-uiverse-title'>
          {mode === 'signup' ? 'Sign Up' : 'Login'}
        </p>

        {mode === 'signup' ? (
          <div className='auth-uiverse-input auth-uiverse-input-email'>
            <input
              autoComplete='email'
              type='email'
              required={mode === 'signup'}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <span>Email</span>
          </div>
        ) : null}

        <div className='auth-uiverse-input'>
          <input
            autoComplete='username'
            type='text'
            required
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
          <span>Username</span>
        </div>

        <div className='auth-uiverse-input'>
          <input
            autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
            type='password'
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <span>Password</span>
        </div>

        {error ? <p className='auth-uiverse-error'>{error}</p> : null}

        <button className='auth-uiverse-enter' type='submit' disabled={loading}>
          {loading ? (mode === 'signup' ? 'Creating...' : 'Entering...') : 'Enter'}
        </button>

        <button
          className='auth-uiverse-switch'
          type='button'
          onClick={() => switchMode(mode === 'signup' ? 'login' : 'signup')}
          disabled={loading}
        >
          {mode === 'signup' ? 'Use Login Instead' : 'Create New Account'}
        </button>

        <Link className='auth-uiverse-back' to='/'>
          Back to Landing
        </Link>
      </form>
    </div>
  )
}
