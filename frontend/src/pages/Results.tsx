import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
    Box,
    Container,
    Typography,
    Card,
    Tabs,
    Tab,
    Grid,
    Button,
    IconButton,
    CircularProgress,
    Tooltip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Alert
} from '@mui/material'
import DownloadIcon from '@mui/icons-material/Download'
import ShareIcon from '@mui/icons-material/Share'
import ViewInArIcon from '@mui/icons-material/ViewInAr'
import HomeIcon from '@mui/icons-material/Home'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PauseIcon from '@mui/icons-material/Pause'
import FullscreenIcon from '@mui/icons-material/Fullscreen'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import * as THREE from 'three'
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

interface TabPanelProps {
    children?: React.ReactNode
    index: number
    value: number
}

function TabPanel({ children, value, index }: TabPanelProps) {
    return (
        <Box role="tabpanel" hidden={value !== index} sx={{ p: 3 }}>
            {value === index && children}
        </Box>
    )
}

interface RoomData {
    name: string
    room_type: string
    x: number
    y: number
    width: number
    height: number
    area_sqm: number
}

interface DesignData {
    rooms: RoomData[]
    total_area: number
    style: string
}

interface ResultFiles {
    png_url?: string
    jpg_url?: string
    dxf_url?: string
    json_url?: string
}

interface JobResults {
    job_id: string
    status: string
    files?: ResultFiles
}

const ROOM_COLORS: { [key: string]: number } = {
    living: 0x3b82f6,
    kitchen: 0xf97316,
    bedroom: 0x22c55e,
    bathroom: 0xa855f7,
    dining: 0xeab308
}

export default function Results() {
    const { jobId } = useParams<{ jobId: string }>()
    const navigate = useNavigate()
    const [tabValue, setTabValue] = useState(0)
    const [results, setResults] = useState<JobResults | null>(null)
    const [design, setDesign] = useState<DesignData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [isAnimating, setIsAnimating] = useState(false)
    const [imageLoaded, setImageLoaded] = useState(false)

    const threeContainerRef = useRef<HTMLDivElement>(null)
    const sceneRef = useRef<THREE.Scene | null>(null)
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<OrbitControls | null>(null)
    const animationRef = useRef<number | null>(null)

    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await fetch(`/api/results/${jobId}`)
                if (!response.ok) throw new Error('Failed to fetch results')
                const data = await response.json()
                setResults(data)

                const designResponse = await fetch(`/outputs/${jobId}/design.json`)
                if (designResponse.ok) {
                    const designData = await designResponse.json()
                    setDesign(designData)
                }
            } catch (err) {
                setError('Failed to load results. The design may still be processing.')
            } finally {
                setLoading(false)
            }
        }

        if (jobId) fetchResults()
    }, [jobId])

    // Initialize 3D scene
    useEffect(() => {
        if (tabValue !== 1 || !threeContainerRef.current || !design) return

        const container = threeContainerRef.current

        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0xf8fafc)
        sceneRef.current = scene

        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000)
        camera.position.set(20, 15, 20)
        cameraRef.current = camera

        const renderer = new THREE.WebGLRenderer({ antialias: true })
        renderer.setSize(container.clientWidth, container.clientHeight)
        renderer.setPixelRatio(window.devicePixelRatio)
        renderer.shadowMap.enabled = true
        container.appendChild(renderer.domElement)
        rendererRef.current = renderer

        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.05
        controls.target.set(7.5, 0, 5)
        controlsRef.current = controls

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
        scene.add(ambientLight)

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
        directionalLight.position.set(20, 30, 10)
        directionalLight.castShadow = true
        scene.add(directionalLight)

        const floorGeometry = new THREE.PlaneGeometry(20, 15)
        const floorMaterial = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, side: THREE.DoubleSide })
        const floor = new THREE.Mesh(floorGeometry, floorMaterial)
        floor.rotation.x = -Math.PI / 2
        floor.position.set(10, 0, 7.5)
        floor.receiveShadow = true
        scene.add(floor)

        design.rooms.forEach((room) => {
            const color = ROOM_COLORS[room.room_type] || 0x6366f1

            const roomFloorGeo = new THREE.PlaneGeometry(room.width, room.height)
            const roomFloorMat = new THREE.MeshStandardMaterial({ color, side: THREE.DoubleSide })
            const roomFloor = new THREE.Mesh(roomFloorGeo, roomFloorMat)
            roomFloor.rotation.x = -Math.PI / 2
            roomFloor.position.set(room.x + room.width / 2, 0.01, room.y + room.height / 2)
            scene.add(roomFloor)

            const wallHeight = 3
            const wallThickness = 0.1
            const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff })

            const northWall = new THREE.Mesh(new THREE.BoxGeometry(room.width, wallHeight, wallThickness), wallMaterial)
            northWall.position.set(room.x + room.width / 2, wallHeight / 2, room.y + room.height)
            northWall.castShadow = true
            scene.add(northWall)

            const southWall = new THREE.Mesh(new THREE.BoxGeometry(room.width, wallHeight, wallThickness), wallMaterial)
            southWall.position.set(room.x + room.width / 2, wallHeight / 2, room.y)
            southWall.castShadow = true
            scene.add(southWall)

            const eastWall = new THREE.Mesh(new THREE.BoxGeometry(wallThickness, wallHeight, room.height), wallMaterial)
            eastWall.position.set(room.x + room.width, wallHeight / 2, room.y + room.height / 2)
            eastWall.castShadow = true
            scene.add(eastWall)

            const westWall = new THREE.Mesh(new THREE.BoxGeometry(wallThickness, wallHeight, room.height), wallMaterial)
            westWall.position.set(room.x, wallHeight / 2, room.y + room.height / 2)
            westWall.castShadow = true
            scene.add(westWall)
        })

        const gridHelper = new THREE.GridHelper(20, 20, 0x94a3b8, 0xcbd5e1)
        gridHelper.position.set(10, 0.02, 7.5)
        scene.add(gridHelper)

        const animate = () => {
            animationRef.current = requestAnimationFrame(animate)
            controls.update()
            renderer.render(scene, camera)
        }
        animate()

        const handleResize = () => {
            const w = container.clientWidth
            const h = container.clientHeight
            camera.aspect = w / h
            camera.updateProjectionMatrix()
            renderer.setSize(w, h)
        }
        window.addEventListener('resize', handleResize)

        return () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current)
            window.removeEventListener('resize', handleResize)
            renderer.dispose()
        }
    }, [tabValue, design])

    const startAnimation = () => {
        if (!cameraRef.current || !controlsRef.current) return
        setIsAnimating(true)

        let angle = 0
        const radius = 20
        const centerX = 10
        const centerZ = 7.5

        const animatePath = () => {
            if (!cameraRef.current || !controlsRef.current) return

            angle += 0.005
            cameraRef.current.position.x = centerX + Math.cos(angle) * radius
            cameraRef.current.position.z = centerZ + Math.sin(angle) * radius
            cameraRef.current.position.y = 15 + Math.sin(angle * 2) * 3
            cameraRef.current.lookAt(centerX, 2, centerZ)

            if (angle < Math.PI * 2 && isAnimating) {
                requestAnimationFrame(animatePath)
            } else {
                setIsAnimating(false)
            }
        }
        animatePath()
    }

    const handleDownload = async (type: string) => {
        if (!results?.files) return

        let url = ''
        let filename = ''

        switch (type) {
            case 'png':
                url = results.files.png_url || ''
                filename = `floorplan_${jobId}.png`
                break
            case 'jpg':
                url = results.files.jpg_url || ''
                filename = `floorplan_${jobId}.jpg`
                break
            case 'dxf':
                url = results.files.dxf_url || ''
                filename = `floorplan_${jobId}.dxf`
                break
            case 'json':
                url = `/outputs/${jobId}/design.json`
                filename = `design_${jobId}.json`
                break
        }

        if (url) {
            const link = document.createElement('a')
            link.href = url
            link.download = filename
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
        }
    }

    if (loading) {
        return (
            <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Box sx={{ textAlign: 'center' }}>
                    <CircularProgress sx={{ color: '#2563eb', mb: 2 }} />
                    <Typography sx={{ color: '#64748b' }}>Loading your design...</Typography>
                </Box>
            </Box>
        )
    }

    if (error) {
        return (
            <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh' }}>
                <Container maxWidth="md" sx={{ py: 10 }}>
                    <Alert severity="error">{error}</Alert>
                    <Box sx={{ mt: 4, textAlign: 'center' }}>
                        <Button
                            variant="outlined"
                            startIcon={<ArrowBackIcon />}
                            onClick={() => navigate('/upload')}
                            sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                        >
                            Back to Upload
                        </Button>
                    </Box>
                </Container>
            </Box>
        )
    }

    return (
        <Box sx={{ bgcolor: '#f8fafc', minHeight: '100vh' }}>
            <Container maxWidth="xl" sx={{ py: 4 }}>
                {/* Header */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box
                            sx={{
                                width: 60,
                                height: 60,
                                borderRadius: '50%',
                                background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                mr: 3
                            }}
                        >
                            <CheckCircleIcon sx={{ fontSize: 32, color: '#fff' }} />
                        </Box>
                        <Box>
                            <Typography variant="h4" sx={{ fontWeight: 700, color: '#1e293b' }}>
                                Your Design is Ready!
                            </Typography>
                            <Typography sx={{ color: '#64748b' }}>
                                Job ID: {jobId?.slice(0, 8)}...
                            </Typography>
                        </Box>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button
                            variant="outlined"
                            startIcon={<ShareIcon />}
                            sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                        >
                            Share
                        </Button>
                        <Button
                            variant="contained"
                            startIcon={<DownloadIcon />}
                            onClick={() => handleDownload('png')}
                            sx={{
                                background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                '&:hover': { background: 'linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%)' }
                            }}
                        >
                            Download All
                        </Button>
                    </Box>
                </Box>

                {/* Tabs */}
                <Card sx={{ mb: 4, border: '1px solid #e2e8f0', borderRadius: 3 }}>
                    <Tabs
                        value={tabValue}
                        onChange={(_, v) => setTabValue(v)}
                        sx={{ borderBottom: '1px solid #e2e8f0' }}
                    >
                        <Tab icon={<HomeIcon />} iconPosition="start" label="2D Floor Plan" />
                        <Tab icon={<ViewInArIcon />} iconPosition="start" label="3D Model" />
                        <Tab icon={<PlayArrowIcon />} iconPosition="start" label="Animation" />
                        <Tab label="Specifications" />
                    </Tabs>

                    {/* 2D Floor Plan */}
                    <TabPanel value={tabValue} index={0}>
                        <Grid container spacing={3}>
                            <Grid item xs={12} md={8}>
                                <Card
                                    sx={{
                                        p: 2,
                                        bgcolor: '#fff',
                                        minHeight: 400,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: 3
                                    }}
                                >
                                    {results?.files?.png_url ? (
                                        <>
                                            {!imageLoaded && <CircularProgress sx={{ color: '#2563eb' }} />}
                                            <Box
                                                component="img"
                                                src={results.files.png_url}
                                                alt="Floor Plan"
                                                onLoad={() => setImageLoaded(true)}
                                                onError={() => setImageLoaded(false)}
                                                sx={{
                                                    maxWidth: '100%',
                                                    maxHeight: 500,
                                                    display: imageLoaded ? 'block' : 'none',
                                                    borderRadius: 2
                                                }}
                                            />
                                        </>
                                    ) : (
                                        <Typography sx={{ color: '#64748b' }}>
                                            Floor plan image not available
                                        </Typography>
                                    )}
                                </Card>
                            </Grid>
                            <Grid item xs={12} md={4}>
                                <Card sx={{ p: 3, border: '1px solid #e2e8f0', borderRadius: 3 }}>
                                    <Typography variant="h6" sx={{ mb: 2, color: '#1e293b' }}>Download Options</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                        <Button
                                            fullWidth
                                            variant="outlined"
                                            startIcon={<DownloadIcon />}
                                            onClick={() => handleDownload('png')}
                                            disabled={!results?.files?.png_url}
                                            sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                                        >
                                            PNG Image
                                        </Button>
                                        <Button
                                            fullWidth
                                            variant="outlined"
                                            startIcon={<DownloadIcon />}
                                            onClick={() => handleDownload('jpg')}
                                            disabled={!results?.files?.jpg_url}
                                            sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                                        >
                                            JPG Image
                                        </Button>
                                        <Button
                                            fullWidth
                                            variant="outlined"
                                            startIcon={<DownloadIcon />}
                                            onClick={() => handleDownload('dxf')}
                                            disabled={!results?.files?.dxf_url}
                                            sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                                        >
                                            DXF (CAD)
                                        </Button>
                                        <Button
                                            fullWidth
                                            variant="outlined"
                                            startIcon={<DownloadIcon />}
                                            onClick={() => handleDownload('json')}
                                            sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                                        >
                                            Design Data (JSON)
                                        </Button>
                                    </Box>
                                </Card>

                                {design && (
                                    <Card sx={{ p: 3, mt: 2, border: '1px solid #e2e8f0', borderRadius: 3 }}>
                                        <Typography variant="h6" sx={{ mb: 2, color: '#1e293b' }}>Quick Stats</Typography>
                                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography sx={{ color: '#64748b' }}>Total Area</Typography>
                                                <Typography fontWeight={600} sx={{ color: '#1e293b' }}>{design.total_area?.toFixed(0) || '--'} sqm</Typography>
                                            </Box>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography sx={{ color: '#64748b' }}>Rooms</Typography>
                                                <Typography fontWeight={600} sx={{ color: '#1e293b' }}>{design.rooms?.length || 0}</Typography>
                                            </Box>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography sx={{ color: '#64748b' }}>Style</Typography>
                                                <Typography fontWeight={600} sx={{ color: '#1e293b', textTransform: 'capitalize' }}>{design.style}</Typography>
                                            </Box>
                                        </Box>
                                    </Card>
                                )}
                            </Grid>
                        </Grid>
                    </TabPanel>

                    {/* 3D Model */}
                    <TabPanel value={tabValue} index={1}>
                        <Card
                            sx={{
                                bgcolor: '#f8fafc',
                                height: 500,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                position: 'relative',
                                overflow: 'hidden',
                                borderRadius: 3,
                                border: '1px solid #e2e8f0'
                            }}
                        >
                            <Box ref={threeContainerRef} sx={{ width: '100%', height: '100%' }} />
                            <Box sx={{ position: 'absolute', bottom: 16, right: 16, display: 'flex', gap: 1 }}>
                                <Tooltip title="Start Walkthrough">
                                    <IconButton
                                        onClick={startAnimation}
                                        disabled={isAnimating}
                                        sx={{ bgcolor: '#fff', boxShadow: 1, '&:hover': { bgcolor: '#f1f5f9' } }}
                                    >
                                        {isAnimating ? <PauseIcon /> : <PlayArrowIcon />}
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title="Fullscreen">
                                    <IconButton sx={{ bgcolor: '#fff', boxShadow: 1, '&:hover': { bgcolor: '#f1f5f9' } }}>
                                        <FullscreenIcon />
                                    </IconButton>
                                </Tooltip>
                            </Box>
                        </Card>
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="body2" sx={{ color: '#64748b' }}>
                                Use mouse to orbit, scroll to zoom, right-click to pan
                            </Typography>
                        </Box>
                    </TabPanel>

                    {/* Animation */}
                    <TabPanel value={tabValue} index={2}>
                        <Card
                            sx={{
                                p: 4,
                                bgcolor: '#fff',
                                border: '1px solid #e2e8f0',
                                borderRadius: 3,
                                height: 400,
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                textAlign: 'center'
                            }}
                        >
                            <Box
                                sx={{
                                    width: 80,
                                    height: 80,
                                    borderRadius: '50%',
                                    background: 'linear-gradient(135deg, rgba(37,99,235,0.1) 0%, rgba(124,58,237,0.1) 100%)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mb: 3
                                }}
                            >
                                <ViewInArIcon sx={{ fontSize: 40, color: '#7c3aed' }} />
                            </Box>
                            <Typography variant="h6" sx={{ color: '#1e293b', mb: 1 }}>
                                Animated Walkthrough
                            </Typography>
                            <Typography sx={{ color: '#64748b', mb: 3, maxWidth: 400 }}>
                                Switch to the 3D Model tab and click the play button to start an animated camera walkthrough of your design.
                            </Typography>
                            <Button
                                variant="contained"
                                onClick={() => setTabValue(1)}
                                startIcon={<ViewInArIcon />}
                                sx={{
                                    background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                                    '&:hover': { background: 'linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%)' }
                                }}
                            >
                                View 3D Model
                            </Button>
                        </Card>
                    </TabPanel>

                    {/* Specifications */}
                    <TabPanel value={tabValue} index={3}>
                        <Grid container spacing={3}>
                            <Grid item xs={12} md={6}>
                                <Card sx={{ p: 3, border: '1px solid #e2e8f0', borderRadius: 3 }}>
                                    <Typography variant="h6" sx={{ mb: 2, color: '#1e293b' }}>Room Breakdown</Typography>
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell sx={{ fontWeight: 600, color: '#1e293b' }}>Room</TableCell>
                                                    <TableCell align="right" sx={{ fontWeight: 600, color: '#1e293b' }}>Area (sqm)</TableCell>
                                                    <TableCell align="right" sx={{ fontWeight: 600, color: '#1e293b' }}>Dimensions</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {(design?.rooms || []).map((room, index) => (
                                                    <TableRow key={index}>
                                                        <TableCell sx={{ color: '#475569' }}>{room.name}</TableCell>
                                                        <TableCell align="right" sx={{ color: '#475569' }}>{room.area_sqm?.toFixed(1)}</TableCell>
                                                        <TableCell align="right" sx={{ color: '#475569' }}>{room.width?.toFixed(1)}m x {room.height?.toFixed(1)}m</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Card>
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <Card sx={{ p: 3, border: '1px solid #e2e8f0', borderRadius: 3 }}>
                                    <Typography variant="h6" sx={{ mb: 2, color: '#1e293b' }}>Summary</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography sx={{ color: '#64748b' }}>Total Area</Typography>
                                            <Typography fontWeight={600} sx={{ color: '#1e293b' }}>
                                                {design?.total_area?.toFixed(0) || '--'} sqm
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography sx={{ color: '#64748b' }}>Style</Typography>
                                            <Typography fontWeight={600} sx={{ color: '#1e293b', textTransform: 'capitalize' }}>
                                                {design?.style || 'Modern'}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography sx={{ color: '#64748b' }}>Bedrooms</Typography>
                                            <Typography fontWeight={600} sx={{ color: '#1e293b' }}>
                                                {design?.rooms?.filter(r => r.room_type === 'bedroom').length || 0}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography sx={{ color: '#64748b' }}>Bathrooms</Typography>
                                            <Typography fontWeight={600} sx={{ color: '#1e293b' }}>
                                                {design?.rooms?.filter(r => r.room_type === 'bathroom').length || 0}
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Card>
                            </Grid>
                        </Grid>
                    </TabPanel>
                </Card>

                {/* Back button */}
                <Box sx={{ mt: 4, textAlign: 'center' }}>
                    <Button
                        variant="outlined"
                        startIcon={<ArrowBackIcon />}
                        onClick={() => navigate('/upload')}
                        sx={{ borderColor: '#e2e8f0', color: '#475569' }}
                    >
                        Create Another Design
                    </Button>
                </Box>
            </Container>
        </Box>
    )
}
