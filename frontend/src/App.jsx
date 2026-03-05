import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import Workspace from './pages/Workspace'

function App() {
    return (
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/workspace" element={<Workspace />} />
            </Routes>
        </BrowserRouter>
    )
}

export default App
