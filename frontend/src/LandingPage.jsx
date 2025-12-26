import { Link } from 'react-router-dom'

function LandingPage() {
  return (
    <div className="app landing-mode">
      <div className="landing-container">
        <div className="landing-content">
          <div className="landing-hero">
            <h1 className="landing-title">
              <span className="title-main">DebateLab</span>
              <span className="title-subtitle">AI-Powered Debate Practice</span>
            </h1>
            <p className="landing-description">
              Hone your debating skills with an intelligent AI opponent powered by a proprietary model 
              trained on hundreds of hours of debates and speeches from the world's best.
            </p>
            <div className="landing-actions">
              <Link 
                to="/debate"
                className="cta-button"
              >
                Start Debate
                <span className="cta-arrow">→</span>
              </Link>
              <Link 
                to="/drills"
                className="cta-button cta-button-secondary"
              >
                Practice Drills
              </Link>
            </div>
          </div>

          <div className="landing-features">
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>Targeted Practice</h3>
              <p>Focus on specific skills with weakness-based drills for rebuttal, structure, weighing, evidence, and strategy.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎓</div>
              <h3>Trained on the best</h3>
              <p>Trained by members of the <strong>Yale Debate Association</strong> and former national team members. Trained on hundreds of hours of pro-level debate content.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Instant Feedback</h3>
              <p>Receive real-time scoring and detailed feedback to improve your debate skills with every practice session.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LandingPage

