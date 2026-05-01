import { Link } from 'react-router-dom'
import { useEffect, useRef } from 'react'
import { getAuthFromWeb, isAuthenticated } from '../services/auth'

export default function LandingPage() {
  const hasSession = isAuthenticated()
  const auth = getAuthFromWeb()
  const displayName = auth?.fullName || auth?.userId || 'User'
  const observerRef = useRef(null)

  useEffect(() => {
    observerRef.current = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible')
        }
      })
    }, { threshold: 0.1 })

    document.querySelectorAll('.reveal-on-scroll').forEach((el) => {
      observerRef.current.observe(el)
    })

    return () => observerRef.current?.disconnect()
  }, [])

  // Generate a few 'active' cells for the 3D grid effect
  const cells = Array.from({ length: 16 }).map((_, i) => (
    <div key={i} className={`lp-3d-cell ${[0, 5, 10, 15].includes(i) ? 'active' : ''}`} />
  ))

  return (
    <div className='lp-shell'>
      <header className='lp-topbar'>
        <div className='lp-brand'>
          <span className='lp-brand-mark'>N</span>
          <div className='lp-brand-copy'>
            <h1>NakshaNirman</h1>
          </div>
        </div>

        <nav className='lp-nav'>
          <a href='#how-it-works'>How it works</a>
          <a href='#features'>Features</a>
          <a href='#export'>Export</a>
          <Link className='fw-button fw-button-outline glow-on-hover' style={{ padding: '8px 16px' }} to='/login'>
            Log in
          </Link>
          <Link className='fw-button fw-button-solid pulse-on-hover' style={{ padding: '8px 16px' }} to='/signup'>
            Sign up
          </Link>
        </nav>
      </header>

      <main className='lp-main'>
        {/* Hero Section */}
        <section className='lp-hero reveal-on-scroll'>
          <div className='lp-hero-content'>
            <span className='lp-kicker'>Instant Floor Plans</span>
            <h2>Design your dream home with a single click.</h2>
            <p className='lp-subtitle'>
              NakshaNirman takes your ideas—like the number of rooms, bathrooms, and the size of your land—and automatically draws a complete floor plan for you. You don't need any technical skills, architectural background, or expensive software. Just tell us what you want, and watch it appear.
            </p>

            <div className='lp-hero-actions'>
              {!hasSession ? (
                <>
                  <Link className='fw-button fw-button-solid pulse-on-hover' to='/signup'>
                    Start Designing Free
                  </Link>
                  <Link className='fw-button fw-button-outline glow-on-hover' to='/login'>
                    View Dashboard
                  </Link>
                </>
              ) : (
                <Link className='fw-button fw-button-solid pulse-on-hover' to='/dashboard'>
                  Continue as {displayName}
                </Link>
              )}
            </div>
          </div>

          <div className='lp-3d-scene interactive-tilt'>
            <div className='lp-3d-grid'>
              <div className='lp-3d-plane'>
                {cells}
              </div>
            </div>
          </div>
        </section>

        {/* How it Works */}
        <section id='how-it-works' className='lp-steps-section reveal-on-scroll'>
          <div className='lp-steps-header'>
            <h2>How it works</h2>
            <p>From idea to complete blueprint in three simple steps.</p>
          </div>
          <div className='lp-step-grid'>
            <div className='lp-step-card interactive-lift'>
              <div className='step-number'>1</div>
              <h4>Tell us what you need</h4>
              <p>Just type in your plot size and how many bedrooms or bathrooms you want.</p>
            </div>
            <div className='lp-step-card interactive-lift'>
              <div className='step-number'>2</div>
              <h4>Watch the AI draw</h4>
              <p>Our smart system automatically figures out the best place for every room, ensuring doors don't block each other and sunlight flows naturally.</p>
            </div>
            <div className='lp-step-card interactive-lift'>
              <div className='step-number'>3</div>
              <h4>Download and share</h4>
              <p>Save a beautiful picture to show your family, or download a professional blueprint file to hand straight to your builder to start construction.</p>
            </div>
          </div>
        </section>

        {/* Who is this for? */}
        <section className='lp-steps-section reveal-on-scroll' style={{ background: 'var(--lp-surface)' }}>
          <div className='lp-steps-header'>
            <h2>Who is this for?</h2>
            <p>Designed to be incredibly simple for beginners, but powerful enough for professionals.</p>
          </div>
          <div className='lp-step-grid'>
            <div className='lp-step-card interactive-lift' style={{ background: '#fff' }}>
              <h4 style={{ color: 'var(--lp-primary)' }}>Homeowners</h4>
              <p>Planning to build or renovate? Visualize your dream house layout for free before you spend money hiring an expensive architect.</p>
            </div>
            <div className='lp-step-card interactive-lift' style={{ background: '#fff' }}>
              <h4 style={{ color: 'var(--lp-primary)' }}>Real Estate Agents</h4>
              <p>Help buyers see the potential of an empty plot of land by quickly generating beautiful floor plans to show them what they could build.</p>
            </div>
            <div className='lp-step-card interactive-lift' style={{ background: '#fff' }}>
              <h4 style={{ color: 'var(--lp-primary)' }}>Builders & Contractors</h4>
              <p>Need a fast, accurate sketch to give a client an estimate? Generate perfectly measured blueprints in seconds to speed up your workflow.</p>
            </div>
          </div>
        </section>

        {/* Feature 1: Smart Automatic Layouts */}
        <section id='features' className='lp-split-section reveal-on-scroll'>
          <div className='lp-split-text'>
            <h3>Smart Automatic Layouts</h3>
            <p>Take the guesswork out of planning. Our system automatically figures out the best design for your specific needs.</p>
            <div className='lp-feature-list'>
              <div className='lp-feature-item'>
                <h4>AI Room Placement</h4>
                <p>We figure out exactly where the kitchen and bedrooms should go so you don't have to.</p>
              </div>
              <div className='lp-feature-item'>
                <h4>Traditional Design Checks</h4>
                <p>Automatically checks if the design follows traditional home building rules (Vastu).</p>
              </div>
              <div className='lp-feature-item'>
                <h4>Perfect Measurements</h4>
                <p>Ensures every room fits perfectly without overlapping walls.</p>
              </div>
            </div>
          </div>
          <div className='lp-split-visual interactive-tilt'>
            <div className='gan-animation-container'>
              {/* This div will be animated via CSS to look like a Neural Network forming a house */}
              <div className='gan-node node-1'></div>
              <div className='gan-node node-2'></div>
              <div className='gan-node node-3'></div>
              <div className='gan-node node-4'></div>
              <div className='gan-node node-5'></div>
              <div className='gan-edge edge-1'></div>
              <div className='gan-edge edge-2'></div>
              <div className='gan-edge edge-3'></div>
              <div className='gan-edge edge-4'></div>
              <div className='gan-house-outline'></div>
            </div>
          </div>
        </section>

        {/* Feature 3: Export to Architect Tools */}
        <section id='export' className='lp-steps-section reveal-on-scroll' style={{ textAlign: 'center' }}>
          <div className='lp-steps-header' style={{ marginBottom: '40px' }}>
            <h2>Export to Architect Tools</h2>
            <p>Your data deserves the best. Move instantly from ideation to professional building software.</p>
          </div>
          <div className='lp-step-grid'>
            <div className='lp-step-card interactive-lift'>
              <h4 style={{ color: 'var(--lp-primary)' }}>Ready for Builders</h4>
              <p>Export your final design directly to professional software like AutoCAD.</p>
            </div>
            <div className='lp-step-card interactive-lift'>
              <h4 style={{ color: 'var(--lp-primary)' }}>High Quality Images</h4>
              <p>Download clear, sharp pictures of your floor plan to share with friends and family.</p>
            </div>
            <div className='lp-step-card interactive-lift'>
              <h4 style={{ color: 'var(--lp-primary)' }}>Easy to Edit</h4>
              <p>Walls and labels are separated so your builder can easily make changes later.</p>
            </div>
          </div>
        </section>

        {/* Local / Free CTA */}
        <section className='lp-cta-section reveal-on-scroll'>
          <h2>100% Free to Use.</h2>
          <p>
            Your architectural data never leaves your machine unless you want it to. 
            Enjoy fast and secure generation without expensive subscriptions.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', flexWrap: 'wrap' }}>
            <Link className='fw-button fw-button-outline pulse-on-hover' style={{ borderColor: 'rgba(255,255,255,0.5)', color: '#fff' }} to='/signup'>
              Launch Workspace
            </Link>
          </div>
        </section>
      </main>

      <footer className='lp-footer'>
        <div>
          <strong>NakshaNirman</strong>
          <p style={{ marginTop: '8px', opacity: 0.7 }}>Empowering everyone to design their own home.</p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <p>Created by:</p>
          <p style={{ marginTop: '8px', opacity: 0.7 }}>
            Path Patel (23aiml055@charusat.edu.in)<br />
            Chauhan Kunj (d24aiml082@charusat.edu.in)
          </p>
        </div>
      </footer>
    </div>
  )
}
