import { Box, Container, Typography, Button, Grid, Card } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ArchitectureIcon from '@mui/icons-material/Architecture'
import ViewInArIcon from '@mui/icons-material/ViewInAr'
import SpeedIcon from '@mui/icons-material/Speed'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import BrushIcon from '@mui/icons-material/Brush'

export default function Landing() {
    const navigate = useNavigate()

    const features = [
        {
            icon: <AutoAwesomeIcon sx={{ fontSize: 32 }} />,
            title: 'AI-Powered Design',
            description: 'Smart algorithms optimize room layouts for any plot shape',
            color: '#2563eb'
        },
        {
            icon: <ArchitectureIcon sx={{ fontSize: 32 }} />,
            title: 'CAD Export',
            description: 'Download professional DXF files ready for any CAD software',
            color: '#7c3aed'
        },
        {
            icon: <ViewInArIcon sx={{ fontSize: 32 }} />,
            title: '3D Visualization',
            description: 'Interactive 3D model with walkthrough animation',
            color: '#059669'
        },
        {
            icon: <SpeedIcon sx={{ fontSize: 32 }} />,
            title: 'Instant Results',
            description: 'Get complete floor plans in seconds, not hours',
            color: '#ea580c'
        }
    ]

    const steps = [
        { icon: <CloudUploadIcon />, title: 'Upload Plot', desc: 'Upload your plot boundary image' },
        { icon: <BrushIcon />, title: 'Set Requirements', desc: 'Specify rooms and preferences' },
        { icon: <AutoAwesomeIcon />, title: 'AI Designs', desc: 'Our AI creates optimal layouts' },
        { icon: <ArchitectureIcon />, title: 'Download', desc: 'Get JPG, PNG, and DXF files' }
    ]

    return (
        <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh' }}>
            {/* Hero Section */}
            <Box
                sx={{
                    position: 'relative',
                    overflow: 'hidden',
                    background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%)',
                    pt: 12,
                    pb: 16
                }}
            >
                {/* Decorative elements */}
                <Box
                    sx={{
                        position: 'absolute',
                        top: '10%',
                        left: '5%',
                        width: 300,
                        height: 300,
                        background: 'radial-gradient(circle, rgba(37,99,235,0.1) 0%, transparent 70%)',
                        borderRadius: '50%',
                        filter: 'blur(40px)'
                    }}
                />
                <Box
                    sx={{
                        position: 'absolute',
                        bottom: '10%',
                        right: '10%',
                        width: 400,
                        height: 400,
                        background: 'radial-gradient(circle, rgba(124,58,237,0.08) 0%, transparent 70%)',
                        borderRadius: '50%',
                        filter: 'blur(60px)'
                    }}
                />

                <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
                    <Box sx={{ textAlign: 'center', maxWidth: 800, mx: 'auto' }}>
                        <Typography
                            variant="overline"
                            sx={{
                                display: 'inline-block',
                                px: 2,
                                py: 0.5,
                                bgcolor: 'rgba(37,99,235,0.1)',
                                color: '#2563eb',
                                borderRadius: 2,
                                fontWeight: 600,
                                letterSpacing: 1.5,
                                mb: 3
                            }}
                        >
                            AI-Powered Architecture
                        </Typography>

                        <Typography
                            variant="h1"
                            sx={{
                                fontSize: { xs: '2.5rem', md: '4rem' },
                                fontWeight: 800,
                                color: '#1e293b',
                                lineHeight: 1.1,
                                mb: 3
                            }}
                        >
                            Transform Your Plot Into
                            <Box
                                component="span"
                                sx={{
                                    display: 'block',
                                    background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                    backgroundClip: 'text',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent'
                                }}
                            >
                                Perfect Floor Plans
                            </Box>
                        </Typography>

                        <Typography
                            variant="h6"
                            sx={{
                                color: '#64748b',
                                fontWeight: 400,
                                mb: 5,
                                lineHeight: 1.6
                            }}
                        >
                            Upload any plot boundary and let our AI design optimized floor plans
                            with smart room placement. Export to CAD-ready DXF files instantly.
                        </Typography>

                        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
                            <Button
                                variant="contained"
                                size="large"
                                onClick={() => navigate('/upload')}
                                endIcon={<ArrowForwardIcon />}
                                sx={{
                                    px: 4,
                                    py: 1.5,
                                    fontSize: '1rem',
                                    fontWeight: 600,
                                    background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                    borderRadius: 2,
                                    boxShadow: '0 4px 20px rgba(37,99,235,0.3)',
                                    '&:hover': {
                                        background: 'linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%)',
                                        boxShadow: '0 6px 30px rgba(37,99,235,0.4)'
                                    }
                                }}
                            >
                                Start Designing
                            </Button>
                            <Button
                                variant="outlined"
                                size="large"
                                sx={{
                                    px: 4,
                                    py: 1.5,
                                    fontSize: '1rem',
                                    fontWeight: 600,
                                    borderColor: '#e2e8f0',
                                    color: '#475569',
                                    borderRadius: 2,
                                    '&:hover': {
                                        borderColor: '#2563eb',
                                        bgcolor: 'rgba(37,99,235,0.05)'
                                    }
                                }}
                            >
                                View Examples
                            </Button>
                        </Box>

                        {/* Stats */}
                        <Grid container spacing={4} sx={{ mt: 8 }}>
                            {[
                                { value: '10K+', label: 'Designs Created' },
                                { value: '150+', label: 'Countries' },
                                { value: '99%', label: 'Satisfaction' },
                                { value: '<30s', label: 'Generation Time' }
                            ].map((stat) => (
                                <Grid item xs={6} md={3} key={stat.label}>
                                    <Typography variant="h3" sx={{ fontWeight: 700, color: '#1e293b' }}>
                                        {stat.value}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: '#64748b' }}>
                                        {stat.label}
                                    </Typography>
                                </Grid>
                            ))}
                        </Grid>
                    </Box>
                </Container>
            </Box>

            {/* How It Works */}
            <Container maxWidth="lg" sx={{ py: 12 }}>
                <Typography
                    variant="h3"
                    sx={{ textAlign: 'center', fontWeight: 700, color: '#1e293b', mb: 2 }}
                >
                    How It Works
                </Typography>
                <Typography
                    variant="body1"
                    sx={{ textAlign: 'center', color: '#64748b', mb: 8, maxWidth: 600, mx: 'auto' }}
                >
                    Four simple steps to go from plot boundary to professional floor plan
                </Typography>

                <Grid container spacing={3}>
                    {steps.map((step, index) => (
                        <Grid item xs={12} sm={6} md={3} key={step.title}>
                            <Card
                                sx={{
                                    p: 4,
                                    textAlign: 'center',
                                    height: '100%',
                                    bgcolor: '#fff',
                                    border: '1px solid #e2e8f0',
                                    borderRadius: 3,
                                    transition: 'all 0.3s ease',
                                    '&:hover': {
                                        transform: 'translateY(-4px)',
                                        boxShadow: '0 12px 40px rgba(0,0,0,0.08)'
                                    }
                                }}
                            >
                                <Box
                                    sx={{
                                        width: 56,
                                        height: 56,
                                        borderRadius: '50%',
                                        background: 'linear-gradient(135deg, rgba(37,99,235,0.1) 0%, rgba(124,58,237,0.1) 100%)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        mx: 'auto',
                                        mb: 2,
                                        color: '#2563eb'
                                    }}
                                >
                                    {step.icon}
                                </Box>
                                <Typography
                                    variant="caption"
                                    sx={{
                                        color: '#94a3b8',
                                        fontWeight: 600,
                                        letterSpacing: 1
                                    }}
                                >
                                    STEP {index + 1}
                                </Typography>
                                <Typography variant="h6" sx={{ fontWeight: 600, color: '#1e293b', mt: 1 }}>
                                    {step.title}
                                </Typography>
                                <Typography variant="body2" sx={{ color: '#64748b', mt: 1 }}>
                                    {step.desc}
                                </Typography>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            </Container>

            {/* Features */}
            <Box sx={{ bgcolor: '#fff', py: 12 }}>
                <Container maxWidth="lg">
                    <Typography
                        variant="h3"
                        sx={{ textAlign: 'center', fontWeight: 700, color: '#1e293b', mb: 2 }}
                    >
                        Powerful Features
                    </Typography>
                    <Typography
                        variant="body1"
                        sx={{ textAlign: 'center', color: '#64748b', mb: 8, maxWidth: 600, mx: 'auto' }}
                    >
                        Everything you need to create professional architectural floor plans
                    </Typography>

                    <Grid container spacing={4}>
                        {features.map((feature) => (
                            <Grid item xs={12} sm={6} md={3} key={feature.title}>
                                <Card
                                    sx={{
                                        p: 3,
                                        height: '100%',
                                        bgcolor: '#f8fafc',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: 3,
                                        transition: 'all 0.3s ease',
                                        '&:hover': {
                                            borderColor: feature.color,
                                            boxShadow: `0 8px 30px ${feature.color}15`
                                        }
                                    }}
                                >
                                    <Box sx={{ color: feature.color, mb: 2 }}>
                                        {feature.icon}
                                    </Box>
                                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#1e293b', mb: 1 }}>
                                        {feature.title}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: '#64748b' }}>
                                        {feature.description}
                                    </Typography>
                                </Card>
                            </Grid>
                        ))}
                    </Grid>
                </Container>
            </Box>

            {/* CTA */}
            <Container maxWidth="md" sx={{ py: 12 }}>
                <Card
                    sx={{
                        p: 6,
                        textAlign: 'center',
                        background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                        borderRadius: 4,
                        boxShadow: '0 20px 60px rgba(37,99,235,0.3)'
                    }}
                >
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', mb: 2 }}>
                        Ready to Design Your Floor Plan?
                    </Typography>
                    <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.85)', mb: 4 }}>
                        Start creating professional architectural designs in seconds
                    </Typography>
                    <Button
                        variant="contained"
                        size="large"
                        onClick={() => navigate('/upload')}
                        endIcon={<ArrowForwardIcon />}
                        sx={{
                            px: 5,
                            py: 1.5,
                            fontSize: '1.1rem',
                            fontWeight: 600,
                            bgcolor: '#fff',
                            color: '#2563eb',
                            borderRadius: 2,
                            '&:hover': {
                                bgcolor: '#f8fafc'
                            }
                        }}
                    >
                        Get Started Free
                    </Button>
                </Card>
            </Container>

            {/* Footer */}
            <Box sx={{ bgcolor: '#1e293b', py: 4 }}>
                <Container maxWidth="lg">
                    <Typography variant="body2" sx={{ textAlign: 'center', color: '#94a3b8' }}>
                        © 2026 AutoArchitect AI. All rights reserved.
                    </Typography>
                </Container>
            </Box>
        </Box>
    )
}
