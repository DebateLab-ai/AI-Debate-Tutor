import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const DRILL_TYPES = [
  { value: 'rebuttal', label: 'Rebuttal', description: 'Practice directly refuting opponent arguments' },
  { value: 'structure', label: 'Structure', description: 'Improve organization and clarity of arguments' },
  { value: 'weighing', label: 'Weighing', description: 'Master impact comparison and prioritization' },
  { value: 'evidence', label: 'Evidence', description: 'Strengthen use of examples and data' },
  { value: 'strategy', label: 'Strategy', description: 'Develop strategic thinking and time allocation' },
]

const SAMPLE_TOPICS = [
  'This House, as a college student, would pursue their passions over selling out',
  'This House believes the US dollar\'s global dominance is harmful',
  'This House believes success is primarily determined by luck rather than effort',
  'This House would restrict free speech to combat right-wing populism',
  'This House opposes leftist bias in universities',
  'This House prefers economic sanctions to military action',
  'This House supports the development of AI in creative industries',
  'This House believes democracy is a human right',
  'This House prefers flawed democracies to technocratic governance',
  'This House would nationalize all land ownership',
  'This House believes that a nationalized pharmaceutical industry is preferable to a private one',
  'This House supports Marxist economic principles',
  'This House believes that government regulation stifles innovation',
  'This House would prioritize individual liberty over collective security',
  'This House believes that markets should determine resource allocation',
]

function DrillsPage() {
  const navigate = useNavigate()
  const [selectedType, setSelectedType] = useState(null)
  const [customTopic, setCustomTopic] = useState('')
  const [selectedTopic, setSelectedTopic] = useState(() => {
    // Randomly select a topic on initial render
    return SAMPLE_TOPICS[Math.floor(Math.random() * SAMPLE_TOPICS.length)]
  })
  const [position, setPosition] = useState('for')
  const [timerEnabled, setTimerEnabled] = useState(false)
  const [timerMinutes, setTimerMinutes] = useState(3)

  const handleStartDrill = () => {
    const topic = customTopic.trim() || selectedTopic
    if (!topic) {
      alert('Please enter or select a topic')
      return
    }
    if (!selectedType) {
      alert('Please select a drill type')
      return
    }

    let url = `/drill-rebuttal?motion=${encodeURIComponent(topic)}&position=${position}&weakness=${selectedType}`
    if (timerEnabled) {
      url += `&timer=${timerMinutes}`
    }
    navigate(url)
  }

  return (
    <div className="app drills-selection-mode">
      <button
        className="return-to-landing"
        onClick={() => navigate('/')}
        title="Return to home"
      >
        ← Home
      </button>

      <div className="drills-selection-container">
        <div className="drills-selection-header">
          <h1>Practice Drills</h1>
          <p className="drills-subtitle">
            Choose a drill type and topic to practice specific debate skills
          </p>
        </div>

        <div className="drills-selection-content">
          {/* Drill Type Selection */}
          <div className="drills-section">
            <h2>Select Drill Type</h2>
            <div className="drill-type-grid">
              {DRILL_TYPES.map((type) => (
                <button
                  key={type.value}
                  className={`drill-type-card ${selectedType === type.value ? 'active' : ''}`}
                  onClick={() => setSelectedType(type.value)}
                >
                  <h3>{type.label}</h3>
                  <p>{type.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Topic Selection */}
          <div className="drills-section">
            <h2>Select Motion</h2>
            <div className="topic-selection">
              <div className="form-group">
                <label>Choose from sample topics:</label>
                <select
                  className="input-large"
                  value={selectedTopic}
                  onChange={(e) => setSelectedTopic(e.target.value)}
                  disabled={!!customTopic.trim()}
                >
                  {SAMPLE_TOPICS.map((topic) => (
                    <option key={topic} value={topic}>
                      {topic}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="form-group">
                <label>Or enter your own topic:</label>
                <input
                  type="text"
                  className="input-large"
                  placeholder="Enter a debate motion..."
                  value={customTopic}
                  onChange={(e) => setCustomTopic(e.target.value)}
                />
              </div>
            </div>
          </div>

          {/* Position Selection */}
          <div className="drills-section">
            <h2>Your Position</h2>
            <div className="position-buttons">
              <button
                className={`position-btn ${position === 'for' ? 'active' : ''}`}
                onClick={() => setPosition('for')}
              >
                FOR
              </button>
              <button
                className={`position-btn ${position === 'against' ? 'active' : ''}`}
                onClick={() => setPosition('against')}
              >
                AGAINST
              </button>
            </div>
            <p className="position-note">
              You'll practice responding to claims from the opposite side
            </p>
          </div>

          {/* Timer Settings */}
          <div className="drills-section">
            <h2>Timer (Optional)</h2>
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={timerEnabled}
                  onChange={(e) => setTimerEnabled(e.target.checked)}
                />
                Enable Timer (creates time pressure)
              </label>
              {timerEnabled && (
                <select
                  className="input-large"
                  value={timerMinutes}
                  onChange={(e) => setTimerMinutes(parseInt(e.target.value))}
                  style={{ marginTop: '12px' }}
                >
                  <option value={1}>1 minute per response</option>
                  <option value={2}>2 minutes per response</option>
                  <option value={3}>3 minutes per response</option>
                  <option value={5}>5 minutes per response</option>
                  <option value={7}>7 minutes per response</option>
                </select>
              )}
            </div>
          </div>

          {/* Start Button */}
          <div className="drills-action">
            <button
              className="btn-primary btn-large"
              onClick={handleStartDrill}
              disabled={!selectedType}
            >
              Start Drill →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DrillsPage
