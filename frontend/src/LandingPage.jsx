import { Link } from 'react-router-dom'
import { SEO, StructuredData, organizationSchema, websiteSchema, softwareApplicationSchema, faqSchema, howToDebateSchema } from './SEO'

function LandingPage() {
  return (
    <>
      <SEO
        title="DebateLab - AI-Powered Debate Practice Platform"
        description="Practice debating with an AI opponent. Get instant feedback, improve your argumentation skills, and master debate techniques with personalized coaching. Built by Yale Debate Association members."
        keywords="debate, debate practice, AI debate, argumentation, debate training, debate skills, public speaking, debate coaching, Yale debate, competitive debate"
        url="https://debatelab.ai/"
      />
      <StructuredData data={organizationSchema} />
      <StructuredData data={websiteSchema} />
      <StructuredData data={softwareApplicationSchema} />
      <StructuredData data={faqSchema} />
      <StructuredData data={howToDebateSchema} />
      <div className="app landing-mode">
      <main className="landing-container">
        <div className="landing-content">
          <header className="landing-hero">
            <h1 className="landing-title">
              <div className="title-with-logo">
                <span className="title-main">DebateLab</span>
                <img 
                  src="/favicon.png" 
                  alt="DebateLab Logo" 
                  className="landing-logo"
                />
              </div>
              <span className="title-subtitle">AI-Powered Debate Practice</span>
            </h1>
            <p className="landing-description">
              Hone your debating skills with an intelligent AI opponent powered by a proprietary model 
              trained on hundreds of hours of debates and speeches from the world's best.
            </p>
            <nav className="landing-actions" aria-label="Main navigation">
              <Link 
                to="/new-debate"
                className="cta-button"
              >
                Start Debate
                <span className="cta-arrow" aria-hidden="true">→</span>
              </Link>
              <Link 
                to="/debate-drills"
                className="cta-button cta-button-secondary"
              >
                Practice Drills
              </Link>
            </nav>
            
            <aside className="landing-community" aria-label="Community links">
              <p className="community-label">Join our community</p>
              <div className="community-links">
                <a 
                  href="https://discord.gg/CrdxpqdnR7" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="community-link"
                  aria-label="Join our Discord server"
                >
                  <svg className="community-logo" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                  Discord
                </a>
                <Link 
                  to="/mailing-list"
                  className="community-link"
                  aria-label="Join our mailing list"
                >
                  <svg className="community-logo" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  Mailing List
                </Link>
              </div>
            </aside>
          </header>

          <section className="landing-features" aria-label="Features">
            <article className="feature-card">
              <div className="feature-icon" aria-hidden="true">🎯</div>
              <h2>Targeted Practice</h2>
              <p>Focus on specific skills with weakness-based drills for rebuttal, structure, weighing, evidence, and strategy.</p>
            </article>
            <article className="feature-card">
              <div className="feature-icon" aria-hidden="true">🎓</div>
              <h2>Yale Debate</h2>
              <p>Built by members of the <strong>Yale Debate Association</strong> and former national team members. Trained on hundreds of hours of pro-level debate content.</p>
            </article>
            <article className="feature-card">
              <div className="feature-icon" aria-hidden="true">⚡</div>
              <h2>Instant Feedback</h2>
              <p>Receive real-time scoring and detailed feedback to improve your debate skills with every practice session.</p>
            </article>
          </section>
        </div>
      </main>
    </div>
    </>
  )
}

export default LandingPage

