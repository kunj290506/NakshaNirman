import { AppBar, Toolbar, Typography, Button, Box, Container } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import ArchitectureIcon from '@mui/icons-material/Architecture'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import HomeIcon from '@mui/icons-material/Home'

export default function Navbar() {
    const navigate = useNavigate()

    return (
        <AppBar
            position="sticky"
            elevation={0}
            sx={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(10px)',
                borderBottom: '1px solid #e2e8f0'
            }}
        >
            <Container maxWidth="xl">
                <Toolbar sx={{ justifyContent: 'space-between' }}>
                    <Box
                        sx={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                        onClick={() => navigate('/')}
                    >
                        <Box
                            sx={{
                                width: 40,
                                height: 40,
                                borderRadius: '10px',
                                background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                mr: 2,
                                color: 'white'
                            }}
                        >
                            <ArchitectureIcon />
                        </Box>
                        <Typography
                            variant="h6"
                            sx={{
                                fontWeight: 700,
                                color: '#1e293b'
                            }}
                        >
                            AutoArchitect AI
                        </Typography>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button
                            startIcon={<HomeIcon />}
                            onClick={() => navigate('/')}
                            sx={{ color: '#64748b', '&:hover': { color: '#1e293b' } }}
                        >
                            Home
                        </Button>
                        <Button
                            variant="contained"
                            startIcon={<CloudUploadIcon />}
                            onClick={() => navigate('/upload')}
                            sx={{
                                background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                '&:hover': {
                                    background: 'linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%)'
                                }
                            }}
                        >
                            Start Design
                        </Button>
                    </Box>
                </Toolbar>
            </Container>
        </AppBar>
    )
}
