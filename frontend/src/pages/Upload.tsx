import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDispatch } from 'react-redux'
import { useDropzone } from 'react-dropzone'
import {
    Box,
    Container,
    Typography,
    Card,
    Grid,
    TextField,
    MenuItem,
    Stepper,
    Step,
    StepLabel,
    Alert
} from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ImageIcon from '@mui/icons-material/Image'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { setJobId, setProgress } from '../store/projectSlice'

const steps = ['Upload Plot', 'Set Requirements', 'Generate']

export default function Upload() {
    const navigate = useNavigate()
    const dispatch = useDispatch()
    const [activeStep, setActiveStep] = useState(0)
    const [file, setFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const [requirements, setRequirements] = useState({
        bedrooms: 3,
        bathrooms: 2,
        style: 'modern',
        floors: 1
    })

    const onDrop = useCallback((acceptedFiles: File[]) => {
        const f = acceptedFiles[0]
        if (f) {
            setFile(f)
            const reader = new FileReader()
            reader.onload = () => setPreview(reader.result as string)
            reader.readAsDataURL(f)
            setError(null)
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
        maxFiles: 1
    })

    const handleSubmit = async () => {
        if (!file) {
            setError('Please upload a plot boundary image')
            return
        }

        setLoading(true)
        setError(null)

        const formData = new FormData()
        formData.append('file', file)
        formData.append('requirements', JSON.stringify(requirements))

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            })

            if (!response.ok) throw new Error('Upload failed')

            const data = await response.json()
            dispatch(setJobId(data.job_id))
            dispatch(setProgress({ progress: 0, stage: 'processing', message: 'Starting...' }))
            navigate(`/processing/${data.job_id}`)
        } catch (err) {
            setError('Failed to upload. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh', py: 6 }}>
            <Container maxWidth="md">
                {/* Header */}
                <Box sx={{ textAlign: 'center', mb: 6 }}>
                    <Typography variant="h3" sx={{ fontWeight: 700, color: '#1e293b', mb: 2 }}>
                        Create Your Floor Plan
                    </Typography>
                    <Typography variant="body1" sx={{ color: '#64748b' }}>
                        Upload your plot boundary and let AI design the perfect layout
                    </Typography>
                </Box>

                {/* Stepper */}
                <Stepper activeStep={activeStep} sx={{ mb: 6 }}>
                    {steps.map((label) => (
                        <Step key={label}>
                            <StepLabel>{label}</StepLabel>
                        </Step>
                    ))}
                </Stepper>

                {error && (
                    <Alert severity="error" sx={{ mb: 4 }}>
                        {error}
                    </Alert>
                )}

                {/* Step 0: Upload */}
                {activeStep === 0 && (
                    <Card sx={{ p: 4, borderRadius: 3, border: '1px solid #e2e8f0' }}>
                        <div
                            {...getRootProps()}
                            className={`uiverse-dropzone ${isDragActive ? 'active' : ''}`}
                            style={{
                                borderColor: file ? '#22c55e' : undefined,
                                background: file ? 'rgba(34,197,94,0.05)' : undefined
                            }}
                        >
                            <input {...getInputProps()} />

                            {preview ? (
                                <Box sx={{ textAlign: 'center' }}>
                                    <Box
                                        component="img"
                                        src={preview}
                                        alt="Preview"
                                        sx={{
                                            maxWidth: '100%',
                                            maxHeight: 300,
                                            borderRadius: 2,
                                            mb: 2,
                                            boxShadow: '0 8px 30px rgba(0,0,0,0.12)'
                                        }}
                                    />
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                                        <CheckCircleIcon sx={{ color: '#22c55e' }} />
                                        <Typography sx={{ color: '#22c55e', fontWeight: 600 }}>
                                            {file?.name}
                                        </Typography>
                                    </Box>
                                    <Typography variant="body2" sx={{ color: '#64748b', mt: 1 }}>
                                        Click or drop to replace
                                    </Typography>
                                </Box>
                            ) : (
                                <Box sx={{ textAlign: 'center' }}>
                                    <div className="uiverse-dropzone-icon">
                                        {isDragActive ? (
                                            <ImageIcon sx={{ fontSize: 36, color: 'white' }} />
                                        ) : (
                                            <CloudUploadIcon sx={{ fontSize: 36, color: 'white' }} />
                                        )}
                                    </div>
                                    <Typography variant="h5" sx={{ color: '#1e293b', fontWeight: 600, mb: 1 }}>
                                        {isDragActive ? 'Drop your image here' : 'Upload Plot Boundary'}
                                    </Typography>
                                    <Typography variant="body1" sx={{ color: '#64748b', mb: 2 }}>
                                        Drag and drop or click to browse
                                    </Typography>
                                    <span className="uiverse-badge uiverse-badge-processing">
                                        PNG, JPG, WEBP up to 10MB
                                    </span>
                                </Box>
                            )}
                        </div>

                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 4 }}>
                            <button
                                className="uiverse-btn"
                                onClick={() => setActiveStep(1)}
                                disabled={!file}
                                style={{ opacity: file ? 1 : 0.5 }}
                            >
                                Continue
                                <ArrowForwardIcon className="uiverse-btn-icon" sx={{ fontSize: 18 }} />
                            </button>
                        </Box>
                    </Card>
                )}

                {/* Step 1: Requirements */}
                {activeStep === 1 && (
                    <Card sx={{ p: 4, borderRadius: 3, border: '1px solid #e2e8f0' }}>
                        <Typography variant="h6" sx={{ mb: 4, color: '#1e293b' }}>
                            Design Requirements
                        </Typography>

                        <Grid container spacing={3}>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    select
                                    label="Bedrooms"
                                    value={requirements.bedrooms}
                                    onChange={(e) => setRequirements({ ...requirements, bedrooms: +e.target.value })}
                                >
                                    {[1, 2, 3, 4, 5].map((n) => (
                                        <MenuItem key={n} value={n}>{n} Bedroom{n > 1 ? 's' : ''}</MenuItem>
                                    ))}
                                </TextField>
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    select
                                    label="Bathrooms"
                                    value={requirements.bathrooms}
                                    onChange={(e) => setRequirements({ ...requirements, bathrooms: +e.target.value })}
                                >
                                    {[1, 2, 3, 4].map((n) => (
                                        <MenuItem key={n} value={n}>{n} Bathroom{n > 1 ? 's' : ''}</MenuItem>
                                    ))}
                                </TextField>
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    select
                                    label="Style"
                                    value={requirements.style}
                                    onChange={(e) => setRequirements({ ...requirements, style: e.target.value })}
                                >
                                    <MenuItem value="modern">Modern</MenuItem>
                                    <MenuItem value="traditional">Traditional</MenuItem>
                                    <MenuItem value="minimalist">Minimalist</MenuItem>
                                    <MenuItem value="contemporary">Contemporary</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    select
                                    label="Floors"
                                    value={requirements.floors}
                                    onChange={(e) => setRequirements({ ...requirements, floors: +e.target.value })}
                                >
                                    {[1, 2, 3].map((n) => (
                                        <MenuItem key={n} value={n}>{n} Floor{n > 1 ? 's' : ''}</MenuItem>
                                    ))}
                                </TextField>
                            </Grid>
                        </Grid>

                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                            <button
                                className="uiverse-btn uiverse-btn-secondary"
                                onClick={() => setActiveStep(0)}
                            >
                                <ArrowBackIcon sx={{ fontSize: 18 }} />
                                Back
                            </button>
                            <button
                                className="uiverse-btn"
                                onClick={() => setActiveStep(2)}
                            >
                                Continue
                                <ArrowForwardIcon className="uiverse-btn-icon" sx={{ fontSize: 18 }} />
                            </button>
                        </Box>
                    </Card>
                )}

                {/* Step 2: Confirm & Generate */}
                {activeStep === 2 && (
                    <Card sx={{ p: 4, borderRadius: 3, border: '1px solid #e2e8f0' }}>
                        <Typography variant="h6" sx={{ mb: 4, color: '#1e293b' }}>
                            Review & Generate
                        </Typography>

                        <Grid container spacing={4}>
                            <Grid item xs={12} md={6}>
                                {preview && (
                                    <Box
                                        component="img"
                                        src={preview}
                                        alt="Plot"
                                        sx={{
                                            width: '100%',
                                            borderRadius: 2,
                                            border: '1px solid #e2e8f0'
                                        }}
                                    />
                                )}
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <Typography variant="subtitle2" sx={{ color: '#64748b', mb: 1 }}>
                                    Summary
                                </Typography>
                                <Box sx={{ bgcolor: '#f8fafc', p: 3, borderRadius: 2, border: '1px solid #e2e8f0' }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                        <Typography sx={{ color: '#64748b' }}>Bedrooms</Typography>
                                        <Typography sx={{ fontWeight: 600, color: '#1e293b' }}>{requirements.bedrooms}</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                        <Typography sx={{ color: '#64748b' }}>Bathrooms</Typography>
                                        <Typography sx={{ fontWeight: 600, color: '#1e293b' }}>{requirements.bathrooms}</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                        <Typography sx={{ color: '#64748b' }}>Style</Typography>
                                        <Typography sx={{ fontWeight: 600, color: '#1e293b', textTransform: 'capitalize' }}>{requirements.style}</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography sx={{ color: '#64748b' }}>Floors</Typography>
                                        <Typography sx={{ fontWeight: 600, color: '#1e293b' }}>{requirements.floors}</Typography>
                                    </Box>
                                </Box>
                            </Grid>
                        </Grid>

                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                            <button
                                className="uiverse-btn uiverse-btn-secondary"
                                onClick={() => setActiveStep(1)}
                            >
                                <ArrowBackIcon sx={{ fontSize: 18 }} />
                                Back
                            </button>
                            <button
                                className="uiverse-btn"
                                onClick={handleSubmit}
                                disabled={loading}
                                style={{ minWidth: 200 }}
                            >
                                {loading ? (
                                    <div className="uiverse-dots">
                                        <span className="uiverse-dot"></span>
                                        <span className="uiverse-dot"></span>
                                        <span className="uiverse-dot"></span>
                                    </div>
                                ) : (
                                    <>Generate Floor Plan<ArrowForwardIcon className="uiverse-btn-icon" sx={{ fontSize: 18 }} /></>
                                )}
                            </button>
                        </Box>
                    </Card>
                )}
            </Container>
        </Box>
    )
}
