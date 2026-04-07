import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import { isAuthenticated } from './services/auth'

function ProtectedRoute({ children }) {
  if (!isAuthenticated()) {
    return <Navigate to='/login' replace />
  }
  return children
}

function LoginRoute() {
  if (isAuthenticated()) {
    return <Navigate to='/dashboard' replace />
  }
  return <LoginPage initialMode='login' />
}

function SignupRoute() {
  if (isAuthenticated()) {
    return <Navigate to='/dashboard' replace />
  }
  return <LoginPage initialMode='signup' />
}

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path='/' element={<LandingPage />} />
        <Route path='/login' element={<LoginRoute />} />
        <Route path='/signup' element={<SignupRoute />} />
        <Route
          path='/dashboard'
          element={(
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          )}
        />
        <Route path='*' element={<Navigate to='/' replace />} />
      </Routes>
    </BrowserRouter>
  )
}
