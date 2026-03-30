import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

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

const ArrowIcon = () => (
    <span className="btn-icon">
        <svg width="10" height="10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
    </span>
)

export default function LandingPage() {
    const navigate = useNavigate()
    const [showAllFeatures, setShowAllFeatures] = useState(false)
    const visibleFeatures = showAllFeatures ? FEATURES : FEATURES.slice(0, 4)

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
                    Simple Home Planning Dashboard
                </div>
                <h1 className="fade-in fade-in-2">
                    Plan Your Home In <span className="accent-text">Three Simple Steps</span>
                </h1>
                <p className="hero-subtitle fade-in fade-in-3">
                    Enter your plot details, choose room requirements, and generate a practical plan.
                    Everything is designed for quick decisions and easy edits.
                </p>
                <div className="hero-buttons fade-in fade-in-4">
                    <button className="btn btn-primary btn-lg" onClick={() => navigate('/workspace')}>
                        Start Designing <ArrowIcon />
                    </button>
                </div>
                <div className="hero-meta-row fade-in fade-in-4">
                    <span className="hero-meta-pill">Step 1: Plot</span>
                    <span className="hero-meta-pill">Step 2: Rooms</span>
                    <span className="hero-meta-pill">Step 3: Preview & Export</span>
                </div>
                </div>
            </section>

            {/* FEATURES */}
            <section className="features-section" id="features">
                <div className="section">
                    <div className="section-header glass-read section-glass">
                        <span className="section-badge">Features</span>
                        <h2 className="section-title">Everything You Need In One Place</h2>
                        <p className="section-subtitle">
                            A focused toolset that keeps planning fast and straightforward.
                        </p>
                    </div>
                    <div className="features-grid">
                        {visibleFeatures.map((f, i) => (
                            <div className="feature-card" key={i}>
                                <div className="feature-icon">{f.icon}</div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                    <div style={{ marginTop: '0.9rem', textAlign: 'center' }}>
                        <button className="btn btn-secondary" onClick={() => setShowAllFeatures((prev) => !prev)}>
                            {showAllFeatures ? 'Show Less' : 'Show All Features'}
                        </button>
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
                            Open a new project, fill one simple form, and generate your plan.
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
