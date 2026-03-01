import { Suspense, useState, useEffect, useRef, Component } from 'react'
import { Canvas, useLoader, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, Center, Environment, ContactShadows, Sky, Html } from '@react-three/drei'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import * as THREE from 'three'

// Error boundary to catch Three.js / GLTFLoader crashes
class CanvasErrorBoundary extends Component {
    constructor(props) {
        super(props)
        this.state = { hasError: false, error: null }
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error }
    }
    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexDirection: 'column', gap: '0.75rem', padding: '2rem',
                    width: '100%', height: '100%'
                }}>
                    <svg width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.2" style={{ color: '#ef4444' }}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>3D Render Error</span>
                    <span style={{ fontSize: '0.82rem', color: '#666', textAlign: 'center', maxWidth: 320 }}>
                        {this.state.error?.message || 'Failed to render 3D model'}
                    </span>
                    <button
                        onClick={() => this.setState({ hasError: false, error: null })}
                        style={{
                            padding: '6px 16px', borderRadius: 6, border: '1px solid #ddd',
                            background: '#f5f5f5', cursor: 'pointer', fontSize: '0.82rem', marginTop: 4
                        }}
                    >
                        Retry
                    </button>
                </div>
            )
        }
        return this.props.children
    }
}

function Model({ url }) {
    const gltf = useLoader(GLTFLoader, url)
    const ref = useRef()

    useEffect(() => {
        if (gltf.scene) {
            gltf.scene.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true
                    child.receiveShadow = true
                    // Enhance material quality
                    if (child.material) {
                        child.material.side = THREE.DoubleSide
                        if (child.material.transparent) {
                            child.material.opacity = Math.max(child.material.opacity, 0.3)
                        }
                    }
                }
            })
        }
    }, [gltf])

    return <primitive ref={ref} object={gltf.scene} />
}

function SceneLights() {
    const dirLightRef = useRef()

    return (
        <>
            {/* Ambient fill light — soft overall illumination */}
            <ambientLight intensity={0.4} color="#f5f0eb" />

            {/* Main directional sun light — warm tone, casts shadows */}
            <directionalLight
                ref={dirLightRef}
                position={[40, 60, 30]}
                intensity={1.2}
                color="#fff8f0"
                castShadow
                shadow-mapSize-width={2048}
                shadow-mapSize-height={2048}
                shadow-camera-left={-60}
                shadow-camera-right={60}
                shadow-camera-top={60}
                shadow-camera-bottom={-60}
                shadow-camera-near={0.5}
                shadow-camera-far={200}
                shadow-bias={-0.0005}
            />

            {/* Secondary fill light from opposite side */}
            <directionalLight
                position={[-30, 40, -20]}
                intensity={0.4}
                color="#e8eeff"
            />

            {/* Warm accent light from below — simulates ground bounce */}
            <hemisphereLight
                skyColor="#b1e1ff"
                groundColor="#b97a20"
                intensity={0.3}
            />

            {/* Point light inside the building for interior warmth */}
            <pointLight
                position={[0, 8, 0]}
                intensity={0.3}
                color="#ffcc88"
                distance={80}
                decay={2}
            />
        </>
    )
}

function CameraSetup() {
    const { camera } = useThree()

    useEffect(() => {
        camera.near = 0.1
        camera.far = 500
        camera.updateProjectionMatrix()
    }, [camera])

    return null
}

export default function Viewer3D({ projectId }) {
    const [modelUrl, setModelUrl] = useState(null)
    const [generating, setGenerating] = useState(false)
    const [error, setError] = useState(null)

    useEffect(() => {
        if (!projectId) return

        let cancelled = false

        const generateAndLoad = async () => {
            setGenerating(true)
            setError(null)
            setModelUrl(null)

            try {
                // Step 1: Generate the 3D model
                const res = await fetch(`/api/generate-3d/${projectId}`, { method: 'POST' })

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}))
                    throw new Error(errData.detail || `Failed to generate 3D model (${res.status})`)
                }

                const data = await res.json()
                if (!cancelled) {
                    // Step 2: Set the model URL so the Canvas loads it
                    setModelUrl(data.model_url || `/api/3d-model/${projectId}`)
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err.message || 'Failed to generate 3D model')
                }
            } finally {
                if (!cancelled) {
                    setGenerating(false)
                }
            }
        }

        generateAndLoad()

        return () => { cancelled = true }
    }, [projectId])

    if (generating) {
        return (
            <div className="viewer-3d" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.75rem' }}>
                <div className="spinner"></div>
                <span className="loading-text">Generating 3D model...</span>
            </div>
        )
    }

    if (error) {
        return (
            <div className="viewer-3d" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.75rem', padding: '2rem' }}>
                <svg width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.2" style={{ color: 'var(--error)' }}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>3D Generation Failed</span>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', textAlign: 'center', maxWidth: 320 }}>{error}</span>
                <button className="btn btn-primary btn-sm" onClick={() => {
                    setGenerating(true)
                    setError(null)
                    fetch(`/api/generate-3d/${projectId}`, { method: 'POST' })
                        .then(r => {
                            if (!r.ok) return r.json().then(e => { throw new Error(e.detail || 'Generation failed') })
                            return r.json()
                        })
                        .then(data => {
                            setModelUrl(data.model_url || `/api/3d-model/${projectId}`)
                            setGenerating(false)
                        })
                        .catch(err => {
                            setError(err.message || 'Retry failed')
                            setGenerating(false)
                        })
                }} style={{ marginTop: '0.5rem' }}>
                    Retry
                </button>
            </div>
        )
    }

    if (!modelUrl) {
        return (
            <div className="viewer-3d" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className="loading-text">Waiting for 3D model...</span>
            </div>
        )
    }

    return (
        <div className="viewer-3d">
            <CanvasErrorBoundary>
            <Canvas
                camera={{ position: [60, 50, 60], fov: 45, near: 0.1, far: 500 }}
                shadows
                gl={{
                    antialias: true,
                    toneMapping: THREE.ACESFilmicToneMapping,
                    toneMappingExposure: 1.1,
                    outputColorSpace: THREE.SRGBColorSpace,
                }}
                style={{ background: 'linear-gradient(180deg, #87ceeb 0%, #e0e8f0 60%, #d4dbe2 100%)' }}
            >
                <CameraSetup />
                <SceneLights />

                {/* Sky dome for outdoor feel */}
                <Sky
                    distance={450000}
                    sunPosition={[40, 60, 30]}
                    inclination={0.52}
                    azimuth={0.25}
                    turbidity={8}
                    rayleigh={0.5}
                />

                <Suspense fallback={
                    <Html center>
                        <div style={{
                            background: 'rgba(255,255,255,0.9)',
                            padding: '12px 24px',
                            borderRadius: 8,
                            fontSize: '0.85rem',
                            color: '#555',
                            fontFamily: 'system-ui'
                        }}>
                            Loading 3D model...
                        </div>
                    </Html>
                }>
                    <Center>
                        <Model url={modelUrl} />
                    </Center>

                    {/* Contact shadows on ground plane */}
                    <ContactShadows
                        position={[0, -0.5, 0]}
                        opacity={0.4}
                        scale={150}
                        blur={2.5}
                        far={50}
                        color="#3a3520"
                    />
                </Suspense>

                <OrbitControls
                    enableDamping
                    dampingFactor={0.08}
                    minDistance={15}
                    maxDistance={250}
                    maxPolarAngle={Math.PI / 2.1}
                    minPolarAngle={Math.PI / 8}
                    target={[0, 3, 0]}
                    autoRotate={false}
                    enablePan
                    panSpeed={0.8}
                    rotateSpeed={0.6}
                    zoomSpeed={1.0}
                />

                {/* Ground grid for spatial reference */}
                <gridHelper
                    args={[200, 40, '#bbb', '#ddd']}
                    position={[0, -0.55, 0]}
                    rotation={[0, 0, 0]}
                />
            </Canvas>
            </CanvasErrorBoundary>

            {/* View controls overlay */}
            <div style={{
                position: 'absolute',
                bottom: 12,
                left: 12,
                display: 'flex',
                gap: 6,
                zIndex: 10
            }}>
                <div style={{
                    background: 'rgba(255,255,255,0.85)',
                    padding: '4px 10px',
                    borderRadius: 6,
                    fontSize: '0.72rem',
                    color: '#666',
                    backdropFilter: 'blur(8px)',
                    fontFamily: 'system-ui'
                }}>
                    Drag to rotate &bull; Scroll to zoom &bull; Right-click to pan
                </div>
            </div>
        </div>
    )
}
