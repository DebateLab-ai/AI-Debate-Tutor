import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function DrillPage() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const motion = searchParams.get('motion') || 'Social media does more harm than good'
  const position = searchParams.get('position') || 'for'
  const weaknessType = searchParams.get('weakness') || null

  const [currentClaim, setCurrentClaim] = useState(null)
  const [claimPosition, setClaimPosition] = useState(null)
  const [rebuttal, setRebuttal] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [lastScore, setLastScore] = useState(null)
  const [attemptCount, setAttemptCount] = useState(0)

  // Start drill - get first claim
  useEffect(() => {
    startDrill()
  }, [])

  const startDrill = async () => {
    setLoading(true)
    try {
      const requestBody = {
        motion: motion,
        user_position: position,
      }
      if (weaknessType) {
        requestBody.weakness_type = weaknessType
      }
      
      const response = await fetch(`${API_BASE}/v1/drills/rebuttal/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) throw new Error('Failed to start drill')

      const data = await response.json()
      setCurrentClaim(data.claim)
      setClaimPosition(data.claim_position)
    } catch (error) {
      console.error('Error starting drill:', error)
      alert('Failed to start drill. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const submitRebuttal = async () => {
    if (!rebuttal.trim()) {
      alert('Please enter your rebuttal')
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
      
      const response = await fetch(`${API_BASE}/v1/drills/rebuttal/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) throw new Error('Failed to submit rebuttal')

      const data = await response.json()
      setLastScore(data)
      setCurrentClaim(data.next_claim)
      setClaimPosition(data.next_claim_position)
      setRebuttal('')
      setAttemptCount(prev => prev + 1)

      // Scroll to score feedback
      setTimeout(() => {
        document.querySelector('.drill-score')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (error) {
      console.error('Error submitting rebuttal:', error)
      // NEVER expose error.message - could contain prompts during network failures
      alert('Failed to submit rebuttal. Please try again.')
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
      <div className="app drill-mode">
        <div className="drill-container">
          <p>Loading drill...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app drill-mode">
      <button
        className="return-to-landing"
        onClick={() => navigate('/')}
        title="Return to home"
      >
        ← Home
      </button>

      <div className="drill-container">
        <div className="drill-header">
          <h1>{weaknessType ? `${weaknessType.charAt(0).toUpperCase() + weaknessType.slice(1)} Drill` : 'Rebuttal Drill'}</h1>
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
            <p className="drill-feedback">{lastScore.feedback}</p>
          </div>
        )}

        <div className="drill-claim-box">
          <h3>Claim to Respond To</h3>
          <div className="claim-content">
            <span className="claim-position-tag">{claimPosition?.toUpperCase()}</span>
            <p>{currentClaim}</p>
          </div>
        </div>

        <div className="drill-input-section">
          <h3>Your Response</h3>
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

        <div className="drill-tips">
          <h4>Tips for {weaknessType ? `Strong ${weaknessType.charAt(0).toUpperCase() + weaknessType.slice(1)}` : 'Strong Rebuttals'}</h4>
          <ul>
            {weaknessType === 'rebuttal' ? (
              <>
                <li><strong>Negate first:</strong> Show why the claim isn't true or doesn't work</li>
                <li><strong>Identify flaws:</strong> Point out gaps in logic or assumptions</li>
                <li><strong>Challenge directly:</strong> Address the core mechanism of the claim</li>
              </>
            ) : weaknessType === 'structure' ? (
              <>
                <li><strong>Signpost clearly:</strong> Use labels like "First...", "Second...", "In conclusion..."</li>
                <li><strong>Logical flow:</strong> Make explicit connections between ideas</li>
                <li><strong>Organize thoughts:</strong> Group related points together</li>
              </>
            ) : weaknessType === 'weighing' ? (
              <>
                <li><strong>Compare explicitly:</strong> Use phrases like "This outweighs because..."</li>
                <li><strong>Address probability, magnitude, timeframe:</strong> Cover all three dimensions</li>
                <li><strong>Make it comparative:</strong> Show why your point matters MORE</li>
              </>
            ) : weaknessType === 'evidence' ? (
              <>
                <li><strong>Be specific:</strong> Use concrete examples, not vague references</li>
                <li><strong>Real-world scenarios:</strong> Reference things people recognize</li>
                <li><strong>Substantial support:</strong> Provide enough evidence to make your point</li>
              </>
            ) : weaknessType === 'strategy' ? (
              <>
                <li><strong>Prioritize:</strong> Focus on the most important responses first</li>
                <li><strong>Allocate wisely:</strong> Spend more time on stronger points</li>
                <li><strong>Make choices:</strong> Decide what to emphasize and what to skip</li>
              </>
            ) : (
              <>
                <li><strong>Negate first:</strong> Show why the claim isn't true or doesn't work</li>
                <li><strong>Use examples:</strong> Counter with specific real-world evidence</li>
                <li><strong>Weigh impacts:</strong> Explain why your refutation matters more</li>
              </>
            )}
          </ul>
        </div>
      </div>
    </div>
  )
}

export default DrillPage
