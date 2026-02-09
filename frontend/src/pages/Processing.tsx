import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { Box, Container, Typography, Grid, Stepper, Step, StepLabel, Alert, Button } from '@mui/material'
import DescriptionIcon from '@mui/icons-material/Description'
import SearchIcon from '@mui/icons-material/Search'
import ViewInArIcon from '@mui/icons-material/ViewInAr'
import RefreshIcon from '@mui/icons-material/Refresh'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ArchitectureIcon from '@mui/icons-material/Architecture'
import { RootState } from '../store'
import { setProgress, setStatus } from '../store/projectSlice'

const stages = [
    { label: 'File Processing', key: 'processing_file', icon: DescriptionIcon },
    { label: 'Boundary Analysis', key: 'extracting_boundary', icon: SearchIcon },
    { label: 'AI Design', key: 'generating_design', icon: AutoAwesomeIcon },
    { label: 'CAD Generation', key: 'creating_cad', icon: ArchitectureIcon },
    { label: '3D & Animation', key: 'rendering_3d', icon: ViewInArIcon }
]

export default function Processing() {
    const { jobId } = useParams<{ jobId: string }>()
    const navigate = useNavigate()
    const dispatch = useDispatch()
    const { progress, message } = useSelector((state: RootState) => state.project)
    const [activeStep, setActiveStep] = useState(0)
    const [error, setError] = useState<string | null>(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const wsRef = useRef<WebSocket | null>(null)
    const pollIntervalRef = useRef<number | null>(null)
    const timerRef = useRef<number | null>(null)
    const startTimeRef = useRef<number>(Date.now())
    const mountedRef = useRef(true)

    // Format elapsed time as MM:SS
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }

    useEffect(() => {
        mountedRef.current = true
        startTimeRef.current = Date.now()

        // Timer for elapsed time
        timerRef.current = window.setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000)
            setElapsedTime(elapsed)
        }, 1000)

        if (!jobId) {
            setError('No job ID provided. Please upload a file first.')
            return
        }

        const pollStatus = async () => {
            if (!mountedRef.current) return

            try {
                const response = await fetch(`/api/job/${jobId}/status`)
                if (!response.ok) return

                const data = await response.json()
                if (!mountedRef.current) return

                dispatch(setProgress({
                    progress: data.progress || 0,
                    stage: data.stage || 'processing',
                    message: data.message || 'Processing...'
                }))

                const stageIndex = stages.findIndex(s => s.key === data.stage)
                if (stageIndex >= 0) setActiveStep(stageIndex)

                if (data.status === 'completed') {
                    dispatch(setStatus('completed'))
                    navigate(`/results/${jobId}`)
                } else if (data.status === 'failed') {
                    setError(data.message || 'Processing failed')
                }
            } catch (err) {
                // Silently ignore polling errors
            }
        }

        pollStatus()
        pollIntervalRef.current = window.setInterval(pollStatus, 3000)

        return () => {
            mountedRef.current = false
            if (wsRef.current) {
                wsRef.current.close()
                wsRef.current = null
            }
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current)
                pollIntervalRef.current = null
            }
            if (timerRef.current) {
                clearInterval(timerRef.current)
                timerRef.current = null
            }
        }
    }, [jobId, dispatch, navigate])

    const handleGoBack = () => {
        navigate('/upload')
    }

    return (
        <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh' }}>
            <Container maxWidth="md" sx={{ py: 10 }}>
                <Box textAlign="center" sx={{ mb: 6 }}>
                    {/* Animated Loader */}
                    <div className="uiverse-loader" style={{ marginBottom: 32 }}>
                        <div className="uiverse-loader-ring"></div>
                        <div className="uiverse-loader-ring"></div>
                        <div className="uiverse-loader-ring"></div>
                        <div className="uiverse-loader-ring"></div>
                    </div>

                    <Typography variant="h3" sx={{ mb: 2, fontWeight: 700, color: '#1e293b' }}>
                        Generating Your Design
                    </Typography>
                    <Typography variant="body1" sx={{ color: '#64748b' }}>
                        Our AI is crafting your perfect floor plan
                    </Typography>
                </Box>

                {error && (
                    <Alert
                        severity="warning"
                        sx={{ mb: 4 }}
                        action={
                            <Button color="inherit" size="small" onClick={handleGoBack} startIcon={<RefreshIcon />}>
                                Start Over
                            </Button>
                        }
                    >
                        {error}
                    </Alert>
                )}

                {/* Progress Card */}
                <div className="uiverse-card" style={{ marginBottom: 24 }}>
                    <Box sx={{ mb: 4, p: 2 }}>
                        {/* Timer Display */}
                        <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'center', 
                            alignItems: 'center',
                            mb: 3,
                            gap: 2
                        }}>
                            <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1,
                                px: 3,
                                py: 1.5,
                                bgcolor: 'rgba(37,99,235,0.1)',
                                borderRadius: 3
                            }}>
                                <Typography variant="caption" sx={{ color: '#64748b', fontWeight: 500 }}>
                                    Elapsed Time
                                </Typography>
                                <Typography 
                                    variant="h5" 
                                    sx={{ 
                                        fontWeight: 700, 
                                        fontFamily: 'monospace',
                                        background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                        backgroundClip: 'text',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent'
                                    }}
                                >
                                    {formatTime(elapsedTime)}
                                </Typography>
                            </Box>
                        </Box>

                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                {message || 'Processing...'}
                            </Typography>
                            <span className="uiverse-gradient-text" style={{ fontWeight: 600 }}>
                                {progress}%
                            </span>
                        </Box>
                        <div className="uiverse-progress">
                            <div className="uiverse-progress-bar" style={{ width: `${progress}%` }}></div>
                        </div>
                    </Box>

                    <Stepper activeStep={activeStep} alternativeLabel>
                        {stages.map((s, index) => (
                            <Step key={s.key} completed={index < activeStep}>
                                <StepLabel>{s.label}</StepLabel>
                            </Step>
                        ))}
                    </Stepper>
                </div>

                {/* Info Cards */}
                <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                        <div className="uiverse-card" style={{ textAlign: 'center' }}>
                            <div className="uiverse-card-icon" style={{
                                width: 60, height: 60, borderRadius: '50%',
                                background: 'rgba(37,99,235,0.1)', margin: '0 auto 16px',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                            }}>
                                <DescriptionIcon sx={{ fontSize: 30, color: '#2563eb' }} />
                            </div>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Optimizing room layout
                            </Typography>
                        </div>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <div className="uiverse-card" style={{ textAlign: 'center' }}>
                            <div className="uiverse-card-icon" style={{
                                width: 60, height: 60, borderRadius: '50%',
                                background: 'rgba(124,58,237,0.1)', margin: '0 auto 16px',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                            }}>
                                <SearchIcon sx={{ fontSize: 30, color: '#7c3aed' }} />
                            </div>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Calculating dimensions
                            </Typography>
                        </div>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <div className="uiverse-card" style={{ textAlign: 'center' }}>
                            <div className="uiverse-card-icon" style={{
                                width: 60, height: 60, borderRadius: '50%',
                                background: 'rgba(34,197,94,0.1)', margin: '0 auto 16px',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                            }}>
                                <ViewInArIcon sx={{ fontSize: 30, color: '#22c55e' }} />
                            </div>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Preparing 3D view
                            </Typography>
                        </div>
                    </Grid>
                </Grid>

                {/* Back button */}
                <Box sx={{ mt: 6, textAlign: 'center' }}>
                    <button
                        className="uiverse-btn uiverse-btn-secondary"
                        onClick={handleGoBack}
                    >
                        Cancel and Go Back
                    </button>
                </Box>
            </Container>
        </Box>
    )
}
