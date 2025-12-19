import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
  const [debateId, setDebateId] = useState(null)
  const [debate, setDebate] = useState(null)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [score, setScore] = useState(null)
  const [scoring, setScoring] = useState(false)
  const [scoreError, setScoreError] = useState(null)
  
  // Landing/Setup state
  const [showLanding, setShowLanding] = useState(true)
  const [topic, setTopic] = useState('')
  const [position, setPosition] = useState('for') // 'for' or 'against'
  const [numRounds, setNumRounds] = useState(1)
  const [setupComplete, setSetupComplete] = useState(false)
  
  // Input state
  const [argument, setArgument] = useState('')
  const [audioFile, setAudioFile] = useState(null)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState(null)
  const inputRef = useRef(null)

  const fetchScore = async (targetId = debateId, compute = false) => {
    if (!targetId) return null
    // Prevent multiple simultaneous score fetches
    if (scoring) {
      console.log('Score fetch already in progress, skipping...')
      return null
    }
    setScoring(true)
    try {
      const method = compute ? 'POST' : 'GET'
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}/score`, {
        method,
        headers: { 'Content-Type': 'application/json' },
      })

      if (response.ok) {
        const data = await response.json()
        setScore(data)
        setScoreError(null)
        return data
      }

      if (response.status === 404 && !compute) {
        // Score not yet generated; compute it
        return await fetchScore(targetId, true)
      }

      const errorText = await response.text()
      throw new Error(errorText || 'Failed to fetch score')
    } catch (error) {
      console.error('Error fetching score:', error)
      setScoreError(error.message)
      return null
    } finally {
      setScoring(false)
    }
  }
  useEffect(() => {
    if (debateId) {
      fetchDebate()
      // Only poll if debate is active, stop polling once completed
      const interval = setInterval(async () => {
        const data = await fetchDebate()
        if (data?.status === 'completed') {
          clearInterval(interval)
        }
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [debateId])

  const fetchDebate = async (targetId = debateId) => {
    if (!targetId) return null
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}`)
      if (!response.ok) throw new Error('Failed to fetch debate')
      const data = await response.json()
      setDebate(data)
      setMessages(data.messages || [])

      if (
        data.status === 'completed' &&
        targetId === debateId &&
        (!score || score.debate_id !== targetId) &&
        !scoring &&
        !scoreError
      ) {
        // Only fetch score once when debate completes
        fetchScore(targetId, false)
      }
      return data
    } catch (error) {
      console.error('Error fetching debate:', error)
      return null
    }
  }

  const startDebate = async () => {
    if (!topic.trim()) {
      alert('Please enter a debate topic')
      return
    }

    setLoading(true)
    try {
      // Determine starter based on position
      // If user is "for", user starts; if "against", assistant starts (arguing for)
      const starter = position === 'for' ? 'user' : 'assistant'
      
      const response = await fetch(`${API_BASE}/v1/debates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_rounds: numRounds,
          starter,
          title: `${topic} (User: ${position}, You take the opposite position)`,
        }),
      })

      if (!response.ok) throw new Error('Failed to create debate')
      const data = await response.json()
      setDebateId(data.id)
      setDebate(data)
      setSetupComplete(true)
      setScore(null)
      setScoreError(null)

      // If assistant starts, generate their first turn
      if (starter === 'assistant') {
        setTimeout(() => generateAITurn(data.id), 500)
      }
    } catch (error) {
      alert('Failed to start debate: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const transcribeAudio = async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetch(`${API_BASE}/v1/transcribe`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error('Transcription failed')
      const data = await response.json()
      return data.text
    } catch (error) {
      alert('Transcription failed: ' + error.message)
      return null
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const audioChunks = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data)
        }
      }

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
        
        setTranscribing(true)
        const transcribed = await transcribeAudio(audioFile)
        if (transcribed) {
          setArgument(prev => prev ? `${prev} ${transcribed}` : transcribed)
        }
        setTranscribing(false)
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop())
      }

      recorder.start()
      setMediaRecorder(recorder)
      setRecording(true)
    } catch (error) {
      alert('Failed to start recording: ' + error.message)
      console.error('Recording error:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorder && recording) {
      mediaRecorder.stop()
      setRecording(false)
      setMediaRecorder(null)
    }
  }

  const submitArgument = async () => {
    if (!debateId || !argument.trim()) {
      alert('Please enter your argument')
      return
    }

    setSubmitting(true)
    try {
      const finalContent = argument.trim()

      const response = await fetch(`${API_BASE}/v1/debates/${debateId}/turns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speaker: 'user',
          content: finalContent,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText)
      }

      const turnData = await response.json()
      console.log('Turn submitted, response:', turnData)
      
      setArgument('')
      setAudioFile(null)
      
      // Refresh debate state to get latest messages
      const updatedDebate = await fetchDebate()
      console.log('Updated debate after turn:', updatedDebate)
      
      // Auto-generate AI response if it's the assistant's turn
      if (turnData.status === 'active' && turnData.next_speaker === 'assistant') {
        console.log('Triggering AI turn generation...')
        // Small delay to ensure UI updates
        setTimeout(() => {
          generateAITurn(debateId)
        }, 800)
      } else if (turnData.status === 'completed') {
        await fetchScore(debateId, false)
      }
    } catch (error) {
      alert('Failed to submit argument: ' + error.message)
    } finally {
      setSubmitting(false)
    }
  }

  const generateAITurn = async (targetId = debateId) => {
    if (!targetId) return

    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}/auto-turn`, {
        method: 'POST',
      })
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText)
      }
      
      // Get the AI response data - this contains the message content
      const aiTurnData = await response.json()
      console.log('AI turn generated:', aiTurnData)
      
      // Immediately refresh to show the new message
      const updated = await fetchDebate(targetId)
      if (updated?.status === 'completed') {
        await fetchScore(targetId, false)
      }
    } catch (error) {
      console.error('Error generating AI turn:', error)
      alert('Failed to generate AI response: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const finishDebate = async () => {
    if (!debateId) return
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${debateId}/finish`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Failed to finish debate')
      await fetchDebate()
      await fetchScore(debateId, true)
    } catch (error) {
      alert('Failed to finish debate: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const resetDebate = () => {
    // Stop recording if active
    if (recording && mediaRecorder) {
      stopRecording()
    }
    
    setDebateId(null)
    setDebate(null)
    setMessages([])
    setSetupComplete(false)
    setShowLanding(false)
    setArgument('')
    setAudioFile(null)
    setScore(null)
    setScoreError(null)
    setRecording(false)
    setTranscribing(false)
    setMediaRecorder(null)
  }

  const getScoreGrade = (score) => {
    // Score is on a 0-10 scale
    if (score >= 9) return 'Excellent'
    if (score >= 8) return 'Great'
    if (score >= 7) return 'Good'
    if (score >= 6) return 'Fair'
    return 'Needs Improvement'
  }

  // Landing page
  if (showLanding) {
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
                trained on thousands of hours of debates and speeches from the world's best.
              </p>
              <button 
                className="cta-button"
                onClick={() => setShowLanding(false)}
              >
                Try it out
                <span className="cta-arrow">→</span>
              </button>
            </div>

            <div className="landing-features">
              <div className="feature-card">
                <div className="feature-icon">🤖</div>
                <h3>AI Opponent</h3>
                <p>Debate against an advanced AI trained on thousands of hours of world-class debates and speeches.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🎓</div>
                <h3>Yale Debate Association</h3>
                <p>Developed by international debaters from the Yale Debate Association</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📊</div>
                <h3>Detailed Scoring</h3>
                <p>Get comprehensive feedback on clarity, structure, engagement, and strategic balance after each debate.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Setup screen
  if (!setupComplete) {
    return (
      <div className="app setup-mode">
        <button className="return-to-landing" onClick={() => setShowLanding(true)} title="Return to home">
          ← Home
        </button>
        <div className="setup-container">
          <div className="setup-card">
            <h1>DebateLab</h1>
            <p className="subtitle">Practice your debating skills with an AI opponent</p>
            
            <div className="form-group">
              <label>Debate Topic</label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Social media does more harm than good"
                className="input-large"
                onKeyPress={(e) => e.key === 'Enter' && startDebate()}
              />
            </div>

            <div className="form-group">
              <label>Your Position</label>
              <div className="position-buttons">
                <button
                  className={position === 'for' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setPosition('for')}
                >
                  For
                </button>
                <button
                  className={position === 'against' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setPosition('against')}
                >
                  Against
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Number of Rounds</label>
              <select
                value={numRounds}
                onChange={(e) => setNumRounds(parseInt(e.target.value))}
                className="input-large"
              >
                <option value={1}>1 Round</option>
                <option value={2}>2 Rounds</option>
                <option value={3}>3 Rounds</option>
              </select>
            </div>

            <button
              className="btn-primary btn-large"
              onClick={startDebate}
              disabled={loading || !topic.trim()}
            >
              {loading ? 'Starting...' : 'Start Debate'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Debate screen
  return (
    <div className="app debate-mode">
      <button className="return-to-landing" onClick={() => { setShowLanding(true); resetDebate(); }} title="Return to home">
        ← Home
      </button>
      <header className="debate-header">
        <div className="header-content">
          <div>
            <h2>{topic}</h2>
            <p className="header-subtitle">
              Your position: <strong>{position === 'for' ? 'FOR' : 'AGAINST'}</strong>
              {' • '}
              Round {debate?.current_round || 1} of {debate?.num_rounds || 3}
              {' • '}
              <span className={`status ${debate?.status || 'active'}`}>
                {debate?.status === 'active' ? 'Active' : 'Completed'}
              </span>
            </p>
          </div>
          <button className="btn-secondary" onClick={resetDebate}>
            New Debate
          </button>
        </div>
      </header>

      <main className="debate-main">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>The debate is starting. {debate?.next_speaker === 'assistant' ? 'Waiting for AI...' : 'Make your first argument!'}</p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div
              key={message.id}
              className={`message ${message.speaker === 'user' ? 'message-user' : 'message-ai'}`}
            >
              <div className="message-header">
                <span className="message-speaker">
                  {message.speaker === 'user' 
                    ? `You (${position === 'for' ? 'FOR' : 'AGAINST'})`
                    : `AI (${position === 'for' ? 'AGAINST' : 'FOR'})`
                  }
                </span>
                <span className="message-round">Round {message.round_no}</span>
              </div>
              <div className="message-content">{message.content}</div>
            </div>
          ))}
          
          {loading && debate?.status === 'active' && (
            <div className="message message-ai">
              <div className="message-header">
                <span className="message-speaker">
                  AI ({position === 'for' ? 'AGAINST' : 'FOR'})
                </span>
              </div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        {debate?.status === 'active' && debate?.next_speaker === 'user' && (
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                ref={inputRef}
                value={argument}
                onChange={(e) => setArgument(e.target.value)}
                placeholder="Type your argument here..."
                className="argument-input"
                rows={4}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    submitArgument()
                  }
                }}
                disabled={submitting}
              />
              <div className="input-actions">
                <button
                  className={`speech-btn ${recording ? 'recording' : ''}`}
                  onClick={recording ? stopRecording : startRecording}
                  disabled={transcribing || submitting}
                  title={recording ? 'Stop recording' : 'Start voice recording'}
                >
                  {transcribing ? '⏳ Transcribing...' : recording ? '🔴 Stop' : '🎤 Voice'}
                </button>
                <button
                  className="btn-primary"
                  onClick={submitArgument}
                  disabled={submitting || !argument.trim()}
                >
                  {submitting ? 'Submitting...' : 'Send'}
                </button>
              </div>
              <p className="input-hint">Press Cmd/Ctrl + Enter to send</p>
            </div>
          </div>
        )}

        {debate?.status === 'completed' && (
          <div className="debate-ended">
            <p>Debate completed! Review your score or start a new debate to continue practicing.</p>
            {score ? (
              <div className="score-card">
                <h3>Your Debate Performance</h3>
                <div 
                  className="score-display"
                  data-grade={getScoreGrade(score.overall).toLowerCase().replace(/\s+/g, '-')}
                >
                  <div className="score-number">{Math.round(score.overall)}</div>
                  <div className="score-label">out of 10</div>
                  <div className="score-bar-container">
                    <div 
                      className="score-bar-fill" 
                      style={{ width: `${(score.overall / 10) * 100}%` }}
                    ></div>
                  </div>
                  <div className="score-grade">{getScoreGrade(score.overall)}</div>
                </div>
                <div className="score-metrics">
                  <div>
                    <span>Content & Structure</span>
                    <strong>{score.metrics?.content_structure != null ? score.metrics.content_structure.toFixed(1) : '–'}</strong>
                    <small>{score.content_structure_feedback || 'No feedback available.'}</small>
                  </div>
                  <div>
                    <span>Engagement & Clash</span>
                    <strong>{score.metrics?.engagement != null ? score.metrics.engagement.toFixed(1) : '–'}</strong>
                    <small>{score.engagement_feedback || 'No feedback available.'}</small>
                  </div>
                  <div>
                    <span>Strategy & Execution</span>
                    <strong>{score.metrics?.strategy != null ? score.metrics.strategy.toFixed(1) : '–'}</strong>
                    <small>{score.strategy_feedback || 'No feedback available.'}</small>
                  </div>
                </div>
                <div className="score-feedback">
                  <h4>Judge Feedback</h4>
                  <p>{score.feedback || 'No overall feedback available.'}</p>
                </div>
              </div>
            ) : (
              <div className="score-card">
                <p className="muted">
                  {scoring
                    ? 'Calculating score...'
                    : scoreError
                      ? `Unable to load score: ${scoreError}`
                      : 'Score not available yet.'}
                </p>
                <button
                  className="btn-secondary"
                  onClick={() => fetchScore(debateId, true)}
                  disabled={scoring}
                >
                  {scoring ? 'Scoring...' : 'Calculate Score'}
                </button>
              </div>
            )}
            <button className="btn-primary" onClick={resetDebate}>
              New Debate
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
