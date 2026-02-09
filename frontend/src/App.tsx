import { Routes, Route, useLocation } from 'react-router-dom'
import { Box } from '@mui/material'
import Navbar from './components/Layout/Navbar'
import Landing from './pages/Landing'
import Upload from './pages/Upload'
import Processing from './pages/Processing'
import Results from './pages/Results'

function App() {
    const location = useLocation()
    const isLandingPage = location.pathname === '/'

    return (
        <Box sx={{ minHeight: '100vh', bgcolor: '#f8fafc' }}>
            {!isLandingPage && <Navbar />}
            <Routes>
                <Route path="/" element={<Landing />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/processing/:jobId" element={<Processing />} />
                <Route path="/results/:jobId" element={<Results />} />
            </Routes>
        </Box>
    )
}

export default App
