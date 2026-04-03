import { Link } from 'react-router-dom'
import { getAuthFromWeb, isAuthenticated } from '../services/auth'

export default function LandingPage() {
  const hasSession = isAuthenticated()
  const auth = getAuthFromWeb()
  const displayName = auth?.fullName || auth?.userId || 'User'

  return (
    <div className='lp-shell'>
      <header className='lp-topbar'>
        <div className='lp-brand'>
          <span className='lp-brand-mark'>N</span>
          <div className='lp-brand-copy'>
            <h1>NakshaNirman</h1>
            <p>AI-assisted planning for practical residential floor layouts.</p>
          </div>
        </div>

        <nav className='lp-nav'>
          <a href='#advantages'>Advantages</a>
          <a href='#workflow'>Workflow</a>
          <a href='#features'>Features</a>
          <Link className='lp-link-button' to='/login'>
            Login
          </Link>
          <Link className='lp-link-button lp-link-button-solid' to='/signup'>
            Sign Up
          </Link>
        </nav>
      </header>

      <main className='lp-main'>
        <section className='lp-hero'>
          <p className='lp-kicker'>Production-ready Architecture Workflow</p>
          <h2>Plan smarter layouts in minutes, not days.</h2>
          <p className='lp-subtitle'>
            NakshaNirman helps teams move from requirements to CAD-ready drawings with
            faster iteration, consistent constraints, and cleaner planning decisions.
          </p>

          <div className='lp-hero-actions'>
            <Link className='lp-link-button lp-link-button-solid' to='/signup'>
              Start Project
            </Link>
            <Link className='lp-link-button' to='/login'>
              Login
            </Link>
            {hasSession ? (
              <Link className='lp-link-button' to='/dashboard'>
                Continue as {displayName}
              </Link>
            ) : null}
          </div>
        </section>

        <section id='advantages' className='lp-section'>
          <h3>Application Advantages</h3>
          <div className='lp-card-grid'>
            <article className='lp-card'>
              <h4>Faster Design Decisions</h4>
              <p>Generate multiple practical options quickly and compare before execution.</p>
            </article>
            <article className='lp-card'>
              <h4>Better Planning Confidence</h4>
              <p>Constraint-aware room placement reduces layout conflicts in later phases.</p>
            </article>
            <article className='lp-card'>
              <h4>Delivery Ready Outputs</h4>
              <p>Export DXF-ready plans aligned to downstream CAD and engineering workflows.</p>
            </article>
          </div>
        </section>

        <section id='workflow' className='lp-section'>
          <h3>User Workflow</h3>
          <div className='lp-step-grid'>
            <article className='lp-step'>
              <span>01</span>
              <h4>Input Requirements</h4>
              <p>Enter plot details, room counts, orientation, and custom project needs.</p>
            </article>
            <article className='lp-step'>
              <span>02</span>
              <h4>Generate + Review</h4>
              <p>Create layouts, inspect quality metrics, and evaluate circulation logic.</p>
            </article>
            <article className='lp-step'>
              <span>03</span>
              <h4>Export + Deliver</h4>
              <p>Download production-ready DXF files and continue in CAD execution pipelines.</p>
            </article>
          </div>
        </section>

        <section id='features' className='lp-section'>
          <h3>Core Features</h3>
          <div className='lp-feature-grid'>
            <article className='lp-feature'>
              <h4>Layout Validation</h4>
              <p>Checks room structure and plan quality before export.</p>
            </article>
            <article className='lp-feature'>
              <h4>Interactive Dashboard</h4>
              <p>Central place to generate, inspect, and download project plans.</p>
            </article>
            <article className='lp-feature'>
              <h4>Secure Account Access</h4>
              <p>Simple login/signup flow with persistent session support.</p>
            </article>
            <article className='lp-feature'>
              <h4>Export Pipeline</h4>
              <p>Direct downloadable DXF output for production handoff.</p>
            </article>
          </div>
        </section>
      </main>

      <footer className='lp-footer'>
        <span>NakshaNirman Platform</span>
        <span>White / Light / Black UI</span>
      </footer>
    </div>
  )
}
