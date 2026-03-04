import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { LayoutProvider } from './store/layoutStore'
import LandingPage from './pages/LandingPage'
import WorkspaceNew from './pages/WorkspaceNew'

function App() {
    return (
        <LayoutProvider>
            <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                <Routes>
                    <Route path="/" element={<LandingPage />} />
                    <Route path="/workspace" element={<WorkspaceNew />} />
                </Routes>
            </BrowserRouter>
        </LayoutProvider>
    )
}

export default App
