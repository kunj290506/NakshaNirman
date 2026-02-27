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

const BENEFITS = [
    { title: 'Save hours of manual drafting', desc: 'Generate production-ready plans in seconds instead of spending hours in CAD software.' },
    { title: 'No architectural expertise needed', desc: 'Our AI handles placement, proportions, and spatial relationships automatically.' },
    { title: 'Industry-standard DXF output', desc: 'Export files that work directly in AutoCAD, Revit, and other professional tools.' },
    { title: 'Interactive 3D walkthroughs', desc: 'Give clients an immersive preview of the space before any construction begins.' },
]

const TESTIMONIALS = [
    { name: 'Arjun Patel', role: 'Architect', text: 'This tool has cut my initial drafting time by 80%. I use it to quickly explore layout options before refining in AutoCAD.' },
    { name: 'Sarah Chen', role: 'Interior Designer', text: 'The 3D visualization feature is incredible. My clients can see exactly what their space will look like before we start any work.' },
    { name: 'Rahul Sharma', role: 'Civil Engineer', text: 'The boundary upload feature is a game changer. I photograph the plot, upload it, and get a fitted floor plan instantly.' },
    { name: 'Emily Brooks', role: 'Real Estate Developer', text: 'We use NakshaNirman to generate quick layouts for client presentations. The professional DXF exports save us countless hours.' },
    { name: 'Vikram Mehta', role: 'Homeowner', text: 'I had no idea how to design my house plan. This tool made it so simple - I just described what I wanted and got a perfect layout.' },
    { name: 'Lisa Wang', role: 'Studio Lead', text: 'The AI chat understands exactly what you want. I described a modern open-plan layout and it delivered something better than expected.' },
]

const FAQS = [
    { q: 'What is NakshaNirman?', a: 'NakshaNirman is an intelligent floor plan generator that uses AI to convert natural language descriptions into professional 2D floor plans and 3D models. It supports custom plot boundaries, room configuration, and exports to industry-standard DXF format.' },
    { q: 'Do I need architectural experience?', a: 'Not at all. NakshaNirman is designed for everyone from homeowners to professional architects. Simply describe your requirements or use the form interface to configure rooms, and the AI handles all placement, proportions, and spatial relationships.' },
    { q: 'What file formats can I export?', a: 'You can export your floor plans as DXF files (compatible with AutoCAD, LibreCAD, and other CAD software) and 3D models as GLB files (compatible with most 3D viewers and game engines).' },
    { q: 'Can I upload my own plot boundary?', a: 'Yes. You can upload an image (PNG, JPEG) of your plot sketch or a DXF file. Our computer vision engine extracts the boundary polygon automatically and generates a floor plan that fits perfectly within it.' },
    { q: 'Is it free to use?', a: 'NakshaNirman is currently free to use during the beta period. Generate unlimited floor plans, 3D models, and DXF exports at no cost.' },
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
    const [openFaq, setOpenFaq] = useState(null)

    return (
        <div className="landing">
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
                    <a href="#benefits">Benefits</a>
                    <a href="#testimonials">Testimonials</a>
                    <a href="#faq">FAQ</a>
                    <button className="btn btn-primary btn-sm" onClick={() => navigate('/workspace')}>
                        Get Started <ArrowIcon />
                    </button>
                </div>
            </nav>

            {/* HERO */}
            <section className="hero">
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
                <div className="hero-buttons fade-in fade-in-4">
                    <button className="btn btn-primary btn-lg" onClick={() => navigate('/workspace')}>
                        Start Designing <ArrowIcon />
                    </button>
                </div>

                {/* Trust Bar */}
                <div className="trust-bar fade-in fade-in-4">
                    <span className="trust-label">Compatible with</span>
                    <div className="trust-logos">
                        <span className="trust-logo">AutoCAD</span>
                        <span className="trust-logo">Revit</span>
                        <span className="trust-logo">SketchUp</span>
                        <span className="trust-logo">LibreCAD</span>
                    </div>
                </div>

                {/* Dashboard Mockup */}
                <div className="hero-dashboard fade-in">
                    <div className="dashboard-inner">
                        <div className="dashboard-toolbar">
                            <span className="toolbar-dot red"></span>
                            <span className="toolbar-dot yellow"></span>
                            <span className="toolbar-dot green"></span>
                            <span className="toolbar-title">NakshaNirman - Workspace</span>
                        </div>
                        <div className="dashboard-body">
                            <div className="dashboard-sidebar">
                                <div className="dash-nav-item active">Design Form</div>
                                <div className="dash-nav-item">AI Chat</div>
                                <div className="dash-nav-item">Upload Boundary</div>
                                <div className="dash-nav-item">GNN Engine</div>
                                <div className="dash-nav-item">Export</div>
                            </div>
                            <div className="dashboard-canvas">
                                <div className="room-block room-living">Living Room</div>
                                <div className="room-block room-kitchen">Kitchen</div>
                                <div className="room-block room-master">Master Bedroom</div>
                                <div className="room-block room-bed">Bedroom</div>
                                <div className="room-block room-bath">Bath</div>
                                <div className="room-block room-dining">Dining Room</div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* FEATURES */}
            <section className="features-section" id="features">
                <div className="section">
                    <div className="section-header">
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

            {/* BENEFITS */}
            <section id="benefits">
                <div className="section">
                    <div className="section-header">
                        <span className="section-badge">Benefits</span>
                        <h2 className="section-title">Why Teams Love Using NakshaNirman</h2>
                        <p className="section-subtitle">
                            Whether you are a solo architect or a large firm, our platform
                            streamlines your entire floor plan workflow.
                        </p>
                    </div>
                    <div className="benefits-grid">
                        {BENEFITS.map((b, i) => (
                            <div className="benefit-item" key={i}>
                                <div className="benefit-check">
                                    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                    </svg>
                                </div>
                                <div>
                                    <h4>{b.title}</h4>
                                    <p>{b.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* TESTIMONIALS */}
            <section className="testimonials-section" id="testimonials">
                <div className="section">
                    <div className="section-header">
                        <span className="section-badge">Testimonials</span>
                        <h2 className="section-title">What Our Users Are Saying</h2>
                        <p className="section-subtitle">
                            Architects, designers, and homeowners trust NakshaNirman to streamline their design process.
                        </p>
                    </div>
                </div>
                <div className="testimonials-track">
                    {[...TESTIMONIALS, ...TESTIMONIALS].map((t, i) => (
                        <div className="testimonial-card" key={i}>
                            <p className="testimonial-text">"{t.text}"</p>
                            <div className="testimonial-author">
                                <div className="testimonial-avatar">
                                    {t.name.split(' ').map(n => n[0]).join('')}
                                </div>
                                <div>
                                    <div className="testimonial-name">{t.name}</div>
                                    <div className="testimonial-role">{t.role}</div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* FAQ */}
            <section id="faq">
                <div className="section">
                    <div className="section-header">
                        <span className="section-badge">FAQ</span>
                        <h2 className="section-title">Frequently Asked Questions</h2>
                        <p className="section-subtitle">
                            Everything you need to know about NakshaNirman and how it can help
                            streamline your architectural workflow.
                        </p>
                    </div>
                    <div className="faq-list">
                        {FAQS.map((faq, i) => (
                            <div className={`faq-item ${openFaq === i ? 'open' : ''}`} key={i}>
                                <button className="faq-question" onClick={() => setOpenFaq(openFaq === i ? null : i)}>
                                    <span>{faq.q}</span>
                                    <span className="faq-toggle">
                                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
                                        </svg>
                                    </span>
                                </button>
                                <div className="faq-answer">
                                    <div className="faq-answer-inner">{faq.a}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="cta-section">
                <div className="section">
                    <div className="cta-box">
                        <span className="section-badge">Get Started</span>
                        <h2 className="section-title">Ready to Design Your Floor Plan?</h2>
                        <p className="section-subtitle">
                            Start generating professional floor plans, 3D models, and CAD exports today.
                            No sign-up required during beta.
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
                        <h4>Product</h4>
                        <a href="#features">Features</a>
                        <a href="#benefits">Benefits</a>
                        <a href="#faq">FAQ</a>
                    </div>
                    <div className="footer-col">
                        <h4>Export Formats</h4>
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
