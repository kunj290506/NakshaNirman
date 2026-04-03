import { useNavigate } from 'react-router-dom'
import WorkspaceNew from '../components/WorkspaceNew'
import { LayoutProvider } from '../store/layoutStore'
import { clearAuthFromWeb, getAuthFromWeb } from '../services/auth'

export default function DashboardPage() {
  const navigate = useNavigate()
  const auth = getAuthFromWeb()
  const userId = auth?.fullName || auth?.userId || 'user'

  function handleLogout() {
    clearAuthFromWeb()
    navigate('/', { replace: true })
  }

  return (
    <LayoutProvider>
      <WorkspaceNew onLogout={handleLogout} userId={userId} />
    </LayoutProvider>
  )
}
