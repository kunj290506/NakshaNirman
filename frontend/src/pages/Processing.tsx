import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { Box, Container, Typography, LinearProgress, Card, Grid, Stepper, Step, StepLabel, Button, Alert } from '@mui/material'
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
    const wsRef = useRef<WebSocket | null>(null)
    const pollIntervalRef = useRef<number | null>(null)
    const mountedRef = useRef(true)

    useEffect(() => {
        mountedRef.current = true

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
        }
    }, [jobId, dispatch, navigate])

    const handleGoBack = () => {
        navigate('/upload')
    }

    return (
        <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh' }}>
            <Container maxWidth="md" sx={{ py: 10 }}>
                <Box textAlign="center" sx={{ mb: 6 }}>
                    <Box
                        sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 100,
                            height: 100,
                            borderRadius: '50%',
                            background: 'linear-gradient(135deg, rgba(37,99,235,0.1) 0%, rgba(124,58,237,0.1) 100%)',
                            mb: 4
                        }}
                    >
                        <AutoAwesomeIcon sx={{ fontSize: 50, color: '#7c3aed' }} />
                    </Box>

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
                <Card sx={{ p: 4, mb: 4, border: '1px solid #e2e8f0', borderRadius: 3 }}>
                    <Box sx={{ mb: 4 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                {message || 'Processing...'}
                            </Typography>
                            <Typography
                                variant="body2"
                                sx={{
                                    fontWeight: 600,
                                    background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                    backgroundClip: 'text',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent'
                                }}
                            >
                                {progress}%
                            </Typography>
                        </Box>
                        <LinearProgress
                            variant="determinate"
                            value={progress}
                            sx={{
                                height: 10,
                                borderRadius: 5,
                                bgcolor: '#e2e8f0',
                                '& .MuiLinearProgress-bar': {
                                    background: 'linear-gradient(90deg, #2563eb 0%, #7c3aed 100%)',
                                    borderRadius: 5
                                }
                            }}
                        />
                    </Box>

                    <Stepper activeStep={activeStep} alternativeLabel>
                        {stages.map((s, index) => (
                            <Step key={s.key} completed={index < activeStep}>
                                <StepLabel>{s.label}</StepLabel>
                            </Step>
                        ))}
                    </Stepper>
                </Card>

                {/* Info Cards */}
                <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                        <Card sx={{ p: 3, textAlign: 'center', border: '1px solid #e2e8f0', borderRadius: 3 }}>
                            <Box
                                sx={{
                                    width: 60,
                                    height: 60,
                                    borderRadius: '50%',
                                    background: 'rgba(37,99,235,0.1)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mx: 'auto',
                                    mb: 2
                                }}
                            >
                                <DescriptionIcon sx={{ fontSize: 30, color: '#2563eb' }} />
                            </Box>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Optimizing room layout
                            </Typography>
                        </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Card sx={{ p: 3, textAlign: 'center', border: '1px solid #e2e8f0', borderRadius: 3 }}>
                            <Box
                                sx={{
                                    width: 60,
                                    height: 60,
                                    borderRadius: '50%',
                                    background: 'rgba(124,58,237,0.1)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mx: 'auto',
                                    mb: 2
                                }}
                            >
                                <SearchIcon sx={{ fontSize: 30, color: '#7c3aed' }} />
                            </Box>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Calculating dimensions
                            </Typography>
                        </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Card sx={{ p: 3, textAlign: 'center', border: '1px solid #e2e8f0', borderRadius: 3 }}>
                            <Box
                                sx={{
                                    width: 60,
                                    height: 60,
                                    borderRadius: '50%',
                                    background: 'rgba(34,197,94,0.1)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mx: 'auto',
                                    mb: 2
                                }}
                            >
                                <ViewInArIcon sx={{ fontSize: 30, color: '#22c55e' }} />
                            </Box>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Preparing 3D view
                            </Typography>
                        </Card>
                    </Grid>
                </Grid>

                {/* Back button */}
                <Box sx={{ mt: 6, textAlign: 'center' }}>
                    <Button
                        variant="outlined"
                        onClick={handleGoBack}
                        sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                    >
                        Cancel and Go Back
                    </Button>
                </Box>
            </Container>
        </Box>
    )
}
