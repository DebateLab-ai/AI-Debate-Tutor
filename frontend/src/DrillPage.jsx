import { useState, useEffect, useRef } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useToast, ToastContainer } from './Toast'
import { SEO } from './SEO'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Render markdown bold (**text**) as JSX
const renderMarkdown = (text) => {
  if (!text) return null

  const parts = []
  let lastIndex = 0
  const regex = /\*\*(.+?)\*\*/g
  let match

  while ((match = regex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }
    // Add bold text
    parts.push(<strong key={match.index}>{match[1]}</strong>)
    lastIndex = regex.lastIndex
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts.length > 0 ? parts : text
}

// Fetch with timeout utility
const fetchWithTimeout = async (url, options = {}, timeoutMs = 60000) => {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. Please try again.')
    }
    throw error
  }
}

function DrillPage() {
  const toast = useToast()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const motion = searchParams.get('motion') || 'This House, as a college student, would pursue their passions over selling out'
  const position = searchParams.get('position') || 'for'
  const weaknessType = searchParams.get('weakness') || null
  const timerParam = searchParams.get('timer')
  const timerEnabled = timerParam !== null
  const timerMinutes = timerParam ? parseInt(timerParam) : 3

  const [currentClaim, setCurrentClaim] = useState(null)
  const [claimPosition, setClaimPosition] = useState(null)
  const [rebuttal, setRebuttal] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [lastScore, setLastScore] = useState(null)
  const [attemptCount, setAttemptCount] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(null)
  const timerIntervalRef = useRef(null)
  
  // Offline detection
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)
    
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Start drill - get first claim
  useEffect(() => {
    startDrill()
  }, [])

  // Timer countdown effect
  useEffect(() => {
    if (timerEnabled && currentClaim && !submitting) {
      // Start or reset timer when new claim appears
      if (timeRemaining === null) {
        setTimeRemaining(timerMinutes * 60)
      }

      timerIntervalRef.current = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            clearInterval(timerIntervalRef.current)
            return 0
          }
          return prev - 1
        })
      }, 1000)

      return () => {
        if (timerIntervalRef.current) {
          clearInterval(timerIntervalRef.current)
        }
      }
    } else if (!currentClaim || submitting) {
      // Clear timer when waiting for claim or submitting
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current)
      }
    }
  }, [timerEnabled, currentClaim, submitting, timeRemaining === null])

  // Auto-submit when timer expires
  useEffect(() => {
    if (timeRemaining === 0 && currentClaim && !submitting) {
      submitRebuttal(true)
    }
  }, [timeRemaining])

  const startDrill = async () => {
    if (!isOnline) {
      toast.error('You are offline. Please check your connection.')
      return
    }
    setLoading(true)
    try {
      const requestBody = {
        motion: motion,
        user_position: position,
      }
      if (weaknessType) {
        requestBody.weakness_type = weaknessType
      }
      
      const response = await fetchWithTimeout(`${API_BASE}/v1/drills/rebuttal/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }, 30000) // 30 second timeout

      if (!response.ok) throw new Error('Failed to start drill')

      const data = await response.json()
      setCurrentClaim(data.claim)
      setClaimPosition(data.claim_position)
    } catch (error) {
      console.error('Error starting drill:', error)
      if (error.message && (error.message.includes('fetch') || error.message.includes('network') || error.message.includes('Failed to fetch'))) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else if (error.message && error.message.includes('timed out')) {
        toast.error('Request took too long. Please try again.')
      } else {
        toast.error('Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const submitRebuttal = async (autoSubmit = false) => {
    if (!rebuttal.trim() && !autoSubmit) {
      toast.error('Please enter your rebuttal')
      return
    }

    if (!isOnline) {
      toast.error('You are offline. Please check your connection and try again.')
      return
    }

    setSubmitting(true)
    try {
      const requestBody = {
        motion: motion,
        claim: currentClaim,
        claim_position: claimPosition,
        rebuttal: rebuttal.trim(),
      }
      if (weaknessType) {
        requestBody.weakness_type = weaknessType
      }
      
      const response = await fetchWithTimeout(`${API_BASE}/v1/drills/rebuttal/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }, 60000) // 60 second timeout for scoring

      if (!response.ok) {
        let errorMessage = 'Something went wrong. Please try again.'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.error || errorMessage
        } catch {
          const errorText = await response.text()
          if (errorText) {
            try {
              const parsed = JSON.parse(errorText)
              errorMessage = parsed.detail || parsed.error || errorMessage
            } catch {
              // If it's not JSON, use generic message
            }
          }
        }
        toast.error(errorMessage)
        return
      }

      const data = await response.json()
      
      // Update state first to avoid losing updates during navigation
      setLastScore(data)
      setCurrentClaim(data.next_claim)
      setClaimPosition(data.next_claim_position)
      setRebuttal('')
      setAttemptCount(prev => prev + 1)
      // Reset timer for next claim
      setTimeRemaining(null)
      
      // Track drill rebuttal submission in Vercel Analytics after state updates
      // TrackingPage will automatically redirect back to the drill URL
      const currentPath = window.location.pathname
      const currentSearch = window.location.search
      const returnUrl = `${currentPath}${currentSearch}`
      // Use setTimeout to defer tracking so state updates complete first
      setTimeout(() => {
        navigate(`/track/drill-rebuttal-submit?return=${encodeURIComponent(returnUrl)}`, { replace: false })
      }, 100)

      // Scroll to score feedback
      setTimeout(() => {
        document.querySelector('.drill-score')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (error) {
      console.error('Error submitting rebuttal:', error)
      // NEVER expose error.message - could contain prompts during network failures
      if (error.message && (error.message.includes('fetch') || error.message.includes('network') || error.message.includes('Failed to fetch'))) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else if (error.message && error.message.includes('timed out')) {
        toast.error('Request took too long. Please try again.')
      } else {
        toast.error('Something went wrong. Please try again.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  const getScoreColor = (score) => {
    if (score >= 8) return '#22c55e'
    if (score >= 6) return '#eab308'
    if (score >= 4) return '#f97316'
    return '#ef4444'
  }

  if (loading && !currentClaim) {
    return (
      <>
        <SEO
          title="Loading Debate Drill - DebateLab"
          description="Loading debate drill practice session..."
          keywords="debate drill, debate practice, debate training"
          url="https://debatelab.ai/debate-drill-rebuttal"
        />
        <div className="app drill-mode">
          <div className="drill-container">
            <p>Loading drill...</p>
          </div>
        </div>
      </>
    )
  }

  const drillTitle = weaknessType 
    ? `${weaknessType.charAt(0).toUpperCase() + weaknessType.slice(1)} Debate Drill - DebateLab`
    : 'Rebuttal Debate Drill - DebateLab'
  const drillDescription = weaknessType
    ? `Practice ${weaknessType} skills with targeted debate drills. Get instant feedback on your debate argumentation and improve your debate performance.`
    : 'Practice rebuttal skills with targeted debate drills. Get instant feedback on your debate argumentation and improve your debate performance.'

  return (
    <>
      <SEO
        title={drillTitle}
        description={drillDescription}
        keywords={`${weaknessType || 'rebuttal'} debate drill, debate practice, debate training, debate argumentation skills, debate exercises, debate coaching`}
        url={`https://debatelab.ai/debate-drill-rebuttal?motion=${encodeURIComponent(motion)}&position=${position}&weakness=${weaknessType || 'rebuttal'}`}
      />
      <div className="app drill-mode">
      <ToastContainer toasts={toast.toasts} removeToast={toast.removeToast} />
      <button
        className="return-to-landing"
        onClick={() => navigate('/')}
        title="Return to home"
      >
        ← Home
      </button>

      <div className="drill-container">
        {!isOnline && (
          <div style={{
            background: 'rgba(255, 107, 107, 0.2)',
            border: '1px solid rgba(255, 107, 107, 0.5)',
            color: '#ff6b6b',
            padding: '8px 16px',
            borderRadius: '8px',
            marginBottom: '12px',
            fontSize: '14px',
            textAlign: 'center'
          }}>
            ⚠️ You are offline. Please check your connection.
          </div>
        )}
        <div className="drill-header">
          <h1>{weaknessType ? `${weaknessType.charAt(0).toUpperCase() + weaknessType.slice(1)} Debate Drill` : 'Rebuttal Debate Drill'}</h1>
          <p className="drill-subtitle">Practice {weaknessType ? weaknessType : 'refuting claims'} on: <strong>{motion}</strong></p>
          <p className="drill-info">
            You argued <strong>{position.toUpperCase()}</strong> •
            {weaknessType ? `Focus: ${weaknessType}` : `Rebut claims from the ${claimPosition?.toUpperCase()} side`} •
            Attempts: {attemptCount}
          </p>
        </div>

        {lastScore && (
          <div className="drill-score">
            <h3>Last Attempt Score</h3>
            <div className="score-display-mini">
              <div
                className="score-number-mini"
                style={{ color: getScoreColor(lastScore.overall_score) }}
              >
                {lastScore.overall_score.toFixed(1)}/10
              </div>
              <div className="score-metrics-mini">
                <div>
                  <span>Refutation</span>
                  <strong>{lastScore.metrics.refutation_quality.toFixed(1)}</strong>
                </div>
                <div>
                  <span>Evidence</span>
                  <strong>{lastScore.metrics.evidence_examples.toFixed(1)}</strong>
                </div>
                <div>
                  <span>Impact</span>
                  <strong>{lastScore.metrics.impact_comparison.toFixed(1)}</strong>
                </div>
              </div>
            </div>
            <p className="drill-feedback">{renderMarkdown(lastScore.feedback)}</p>
          </div>
        )}

        <div className="drill-claim-box">
          <h3>Claim to Respond To</h3>
          <div className="claim-content">
            <span className="claim-position-tag">{claimPosition?.toUpperCase()}</span>
            <p>{renderMarkdown(currentClaim)}</p>
          </div>
        </div>

        <div className="drill-tips">
          <h4>Tips for {weaknessType ? `Strong ${weaknessType.charAt(0).toUpperCase() + weaknessType.slice(1)}` : 'Strong Rebuttals'}</h4>
          <ul>
            {weaknessType === 'rebuttal' ? (
              <>
                <li><strong>Negate when possible:</strong> Think of anything that can reduce the impact of their claim</li>
                <li><strong>Identify logical fallacies:</strong> Find gaps in logic that they haven't proven. Don't always assume your opponent's claims are well made!</li>
                <li><strong>Weigh if you can't negate:</strong> If you cannot think of a believable response, weigh your claim against theirs. Walk the judge through why they should care more about your claim in the context of the debate. Consider: Is your impact larger? Does your impact affect more vulnerable actors? What about timeframe—short-term vs long-term?</li>
              </>
            ) : weaknessType === 'structure' ? (
              <>
                <li><strong>Signpost with numbers:</strong> If you have three responses, say so upfront, then label each one ("First...", "Second...", "Third..."). This helps the judge keep track of the many things you said</li>
                <li><strong>Indicate case vs. refutation:</strong> Tell the judge when you're talking about your case versus their case. If refuting, say "Firstly, some refutations..." There are many moving parts of a debate—this helps the judge keep track</li>
                <li><strong>Finish ideas before moving on:</strong> Don't start with an underdeveloped idea and then come back to it 3 minutes later when it's no longer fresh in the judge's head. Complete each point before starting the next</li>
              </>
            ) : weaknessType === 'weighing' ? (
              <>
                <li><strong>Make comparisons explicit:</strong> In close rounds, explicit weighing breaks the tie. Common metrics: <em>Probability</em> - is your claim more likely to happen? <em>Scale of impact</em> - even if less probable, is your impact more significant? <em>Timeframe</em> - should we care more about short-term urgency or long-term consequences?</li>
                <li><strong>Claim logical priority ("upstream"):</strong> Sometimes you can argue your case is "logically upstream"—meaning their case requires yours to happen first. Example: "Our mental health argument is upstream of their connectivity claim—people need to be mentally healthy before they can form meaningful connections, so our harms must be addressed first"</li>
              </>
            ) : weaknessType === 'evidence' ? (
              <>
                <li><strong>Use commonly known examples:</strong> Think of headlines that would or have appeared on the New York Times front page. A reasonable judge would know these examples, and they are good ways to make your case more concrete in a judge's head</li>
                <li><strong>Thought experiments as backup:</strong> Sometimes, if you really don't know any examples, you can think of a thought experiment. This will get the judge's intuition going. Though beware of spending too much time thinking of this! Maybe prepare them during prep time</li>
              </>
            ) : weaknessType === 'strategy' ? (
              <>
                <li><strong>Manage time—don't tunnel vision:</strong> Don't spend too much time on one point! Even if you think it's important and you completely take it out, a judge could call you out for not responding to other things</li>
                <li><strong>Frontload the round-winners:</strong> Prioritize and address the most important, potentially round-winning points first. This takes practice and requires intuition—our drills are perfect for developing this skill!</li>
              </>
            ) : (
              <>
                <li><strong>Negate when possible:</strong> Think of anything that can reduce the impact of their claim</li>
                <li><strong>Identify logical fallacies:</strong> Find gaps in logic that they haven't proven. Don't always assume your opponent's claims are well made!</li>
                <li><strong>Weigh if you can't negate:</strong> If you cannot think of a believable response, weigh your claim against theirs. Walk the judge through why they should care more about your claim. Consider impact size, vulnerability of affected actors, and timeframe</li>
              </>
            )}
          </ul>
        </div>

        <div className="drill-input-section">
          <h3>Your Response</h3>
          {timerEnabled && timeRemaining !== null && (
            <div className={`timer-display ${timeRemaining < 60 ? 'timer-warning' : ''}`}>
              ⏱️ Time Remaining: {Math.floor(timeRemaining / 60)}:{String(timeRemaining % 60).padStart(2, '0')}
            </div>
          )}
          <textarea
            className="drill-textarea"
            value={rebuttal}
            onChange={(e) => setRebuttal(e.target.value)}
            placeholder={
              weaknessType === 'rebuttal' 
                ? "Write your rebuttal here... Focus on: (1) Negating/mitigating the claim, (2) Identifying flaws in logic, (3) Challenging assumptions"
                : weaknessType === 'structure'
                ? "Write your response here... Focus on: (1) Clear signposting and organization, (2) Logical flow, (3) Explicit links between ideas"
                : weaknessType === 'weighing'
                ? "Write your response here... Focus on: (1) Comparing probability, magnitude, and timeframe, (2) Making clear comparative statements, (3) Explaining why your point matters more"
                : weaknessType === 'evidence'
                ? "Write your response here... Focus on: (1) Using concrete, specific examples, (2) Referencing real-world scenarios, (3) Providing substantial evidence"
                : weaknessType === 'strategy'
                ? "Write your response here... Focus on: (1) Prioritizing the most important points, (2) Allocating space appropriately, (3) Making clear strategic decisions"
                : "Write your rebuttal here... Focus on: (1) Negating/mitigating the claim, (2) Using evidence/examples, (3) Comparing impacts"
            }
            rows={8}
            maxLength={10000}
            disabled={submitting}
          />
          <button
            className="btn-primary btn-large"
            onClick={submitRebuttal}
            disabled={submitting || !rebuttal.trim()}
          >
            {submitting ? 'Scoring...' : 'Submit Rebuttal'}
          </button>
        </div>
      </div>
    </div>
    </>
  )
}

export default DrillPage
