import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Canvas } from '@react-three/fiber'
import HouseScrollScene from '../components/canvas/HouseScrollScene'

const FEATURES = [
    {
        icon: (
            <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
        ),
        title: 'AI-Powered Chat',
        desc: 'Describe your dream home in natural language. Our AI understands your requirements and converts them into professional layouts.',
    },
    {
        icon: (
            <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
        ),
        title: 'GNN Layout Engine',
        desc: 'Graph Neural Network-inspired algorithms with smart room adjacency, Vastu compliance, and adaptive layout strategies for any plot shape.',
    },
    {
        icon: (
            <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
        ),
        title: 'CAD Export',
        desc: 'Download professional DXF files compatible with AutoCAD, LibreCAD, and other industry-standard CAD software.',
    },
    {
        icon: (
            <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
        ),
        title: '3D Visualization',
        desc: 'Instantly view your floor plan in interactive 3D. Rotate, zoom, and explore every detail of your future home.',
    },
    {
        icon: (
            <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
        ),
        title: 'Boundary Upload',
        desc: 'Upload a sketch of your plot boundary. Our engine extracts the shape and fits rooms inside automatically.',
    },
    {
        icon: (
            <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
        ),
        title: 'Instant Generation',
        desc: 'Get your complete floor plan in under 10 seconds. No manual drafting or CAD expertise required.',
    },
]

const CHAPTERS = [
    { id: 'site', label: 'Site + Grid', from: 0.0, to: 0.22 },
    { id: 'zoning', label: 'Zoning', from: 0.22, to: 0.44 },
    { id: 'adjacency', label: 'Adjacency', from: 0.44, to: 0.64 },
    { id: 'stacking', label: 'Room Stacking', from: 0.64, to: 0.84 },
    { id: 'final', label: 'Final Plan', from: 0.84, to: 1.01 },
]



const ArrowIcon = () => (
    <span className="btn-icon">
        <svg width="10" height="10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
    </span>
)

export default function LandingPage() {
    const navigate = useNavigate()
    const [scrollProgress, setScrollProgress] = useState(0)

    const activeChapter = CHAPTERS.find((chapter) => scrollProgress >= chapter.from && scrollProgress < chapter.to) || CHAPTERS[CHAPTERS.length - 1]

    useEffect(() => {
        const handleScroll = () => {
            const totalScrollable = document.documentElement.scrollHeight - window.innerHeight
            const progress = totalScrollable > 0 ? window.scrollY / totalScrollable : 0
            setScrollProgress(Math.max(0, Math.min(1, progress)))
        }

        window.addEventListener('scroll', handleScroll, { passive: true })
        handleScroll()

        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    return (
        <div className="landing landing-fresh">
            {/* NAV */}
            <nav className="landing-nav">
                <a href="/" className="logo">
                    <span className="logo-icon">
                        <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                    </span>
                    NakshaNirman
                </a>
                <div className="nav-links">
                    <a href="#cinema">3D Tour</a>
                    <a href="#features">Features</a>
                    <a href="#project">Project</a>
                    <button className="btn btn-primary btn-sm" onClick={() => navigate('/workspace')}>
                        Get Started <ArrowIcon />
                    </button>
                </div>
            </nav>

            {/* HERO */}
            <section className="hero" id="project">
                <div className="glass-read hero-glass">
                <div className="hero-badge fade-in fade-in-1">
                    <span className="badge-dot"></span>
                    GNN-Powered Architecture Engine
                </div>
                <h1 className="fade-in fade-in-2">
                    AI-Powered <span className="accent-text">Floor Plan</span><br />
                    Architecture Generator
                </h1>
                <p className="hero-subtitle fade-in fade-in-3">
                    Generate professional 2D floor plans, 3D models, and CAD exports using our
                    GNN-inspired layout engine. Upload your plot boundary, configure rooms, and
                    get Vastu-compliant designs instantly.
                </p>
                <p className="hero-scroll-note fade-in fade-in-4">Scroll to direct the 3D camera and reveal each CAD planning stage.</p>
                <div className="hero-buttons fade-in fade-in-4">
                    <button className="btn btn-primary btn-lg" onClick={() => navigate('/workspace')}>
                        Start Designing <ArrowIcon />
                    </button>
                </div>
                <div className="hero-meta-row fade-in fade-in-4">
                    <span className="hero-meta-pill">Plot-aware layouts</span>
                    <span className="hero-meta-pill">CAD-first exports</span>
                    <span className="hero-meta-pill">Interactive 3D story</span>
                </div>
                </div>

                {/* 3D Background Canvas */}
                <div className="canvas-bg fade-in">
                    <Canvas shadows camera={{ position: [0, 8, 8], fov: 45 }} dpr={[1, 1.6]}>
                        <HouseScrollScene progress={scrollProgress} />
                    </Canvas>
                </div>
            </section>

            <section className="cad-cinema" id="cinema">
                <div className="section">
                    <div className="section-header glass-read section-glass">
                        <span className="section-badge">CAD Film</span>
                        <h2 className="section-title">Scroll-Controlled 3D House Plan Tour</h2>
                        <p className="section-subtitle">
                            Built using scroll-storyboarding patterns from modern WebGL landing pages:
                            guided camera spline, phase-based reveal, and scanline pass to mimic a CAD design film.
                        </p>
                    </div>
                    <div className="cad-video-shell">
                        <div className="cad-video-topbar">
                            <span className="rec-dot"></span>
                            <span>REC</span>
                            <span className="cad-scene-label">SCENE: {activeChapter.label}</span>
                            <span className="cad-time">{`${String(Math.floor(scrollProgress * 2)).padStart(2, '0')}:${String(Math.floor((scrollProgress * 60) % 60)).padStart(2, '0')}`}</span>
                        </div>
                        <div className="cad-video-canvas">
                            <Canvas camera={{ position: [0, 7, 7], fov: 42 }} dpr={[1, 1.5]}>
                                <HouseScrollScene progress={scrollProgress} cinematic />
                            </Canvas>
                            <div className="scanline-overlay" />
                        </div>
                        <div className="cad-video-progress">
                            <div className="cad-video-progress-bar" style={{ width: `${scrollProgress * 100}%` }} />
                        </div>
                        <div className="cad-chapters">
                            {CHAPTERS.map((chapter) => (
                                <span key={chapter.id} className={`cad-chapter-chip ${activeChapter.id === chapter.id ? 'active' : ''}`}>
                                    {chapter.label}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </section>

            {/* FEATURES */}
            <section className="features-section" id="features">
                <div className="section">
                    <div className="section-header glass-read section-glass">
                        <span className="section-badge">Features</span>
                        <h2 className="section-title">Everything You Need to Create Floor Plans</h2>
                        <p className="section-subtitle">
                            From AI-powered chat to professional CAD exports, NakshaNirman gives you
                            a complete toolkit for residential design.
                        </p>
                    </div>
                    <div className="features-grid">
                        {FEATURES.map((f, i) => (
                            <div className="feature-card" key={i}>
                                <div className="feature-icon">{f.icon}</div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="cta-section">
                <div className="section">
                    <div className="cta-box glass-read section-glass">
                        <span className="section-badge">Project Workspace</span>
                        <h2 className="section-title">Open the CAD Project Workspace</h2>
                        <p className="section-subtitle">
                            Start a new project, set requirements, upload boundary, and generate CAD-ready floor plans.
                        </p>
                        <button className="btn btn-primary btn-lg" onClick={() => navigate('/workspace')}>
                            Start Designing Now <ArrowIcon />
                        </button>
                    </div>
                </div>
            </section>

            {/* FOOTER */}
            <footer className="footer">
                <div className="footer-grid">
                    <div className="footer-brand">
                        <a href="/" className="logo">
                            <span className="logo-icon">
                                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                                </svg>
                            </span>
                            NakshaNirman
                        </a>
                        <p>
                            Generate professional 2D floor plans and 3D models from simple descriptions.
                            Built with FastAPI, React, and AI.
                        </p>
                    </div>
                    <div className="footer-col">
                        <h4>Project</h4>
                        <a href="#features">Features</a>
                        <a href="#cinema">3D Tour</a>
                        <a href="#project">Overview</a>
                    </div>
                    <div className="footer-col">
                        <h4>Deliverables</h4>
                        <a href="#features">DXF (AutoCAD)</a>
                        <a href="#features">GLB (3D Model)</a>
                        <a href="#features">SVG Preview</a>
                    </div>
                    <div className="footer-col">
                        <h4>Technology</h4>
                        <a href="#features">GNN Engine</a>
                        <a href="#features">React + Vite</a>
                        <a href="#features">FastAPI</a>
                        <a href="#features">Three.js</a>
                    </div>
                </div>
                <div className="footer-bottom">
                    <span>Copyright 2026 NakshaNirman. All Rights Reserved.</span>
                    <div style={{ display: 'flex', gap: '1.5rem' }}>
                        <a href="#" style={{ color: 'var(--text-muted)' }}>Terms of Service</a>
                        <a href="#" style={{ color: 'var(--text-muted)' }}>Privacy Policy</a>
                    </div>
                </div>
            </footer>
        </div>
    )
}
