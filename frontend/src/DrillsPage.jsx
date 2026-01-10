import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SEO } from './SEO'

const DRILL_TYPES = [
  { value: 'rebuttal', label: 'Rebuttal', description: 'Practice directly refuting opponent arguments' },
  { value: 'structure', label: 'Structure', description: 'Improve organization and clarity of arguments' },
  { value: 'weighing', label: 'Weighing', description: 'Master impact comparison and prioritization' },
  { value: 'evidence', label: 'Evidence', description: 'Strengthen use of examples and data' },
  { value: 'strategy', label: 'Strategy', description: 'Develop strategic thinking and time allocation' },
]

const SAMPLE_TOPICS = [
  'Social media does more harm than good',
  'As a college student, pursuing passions is preferable to selling out',
  'The US dollar\'s global dominance is harmful',
  'Success is primarily determined by luck rather than effort',
  'Free speech should be restricted to combat right-wing populism',
  'Universities are biased against leftist perspectives',
  'Economic sanctions are preferable to military action',
  'AI development in creative industries should be supported',
  'Democracy is a human right',
  'Flawed democracies are preferable to technocratic governance',
  'All land ownership should be nationalized',
  'A nationalized pharmaceutical industry is preferable to a private one',
  'Marxist economic principles are superior',
  'Government regulation stifles innovation',
  'Individual liberty should be prioritized over collective security',
  'Markets should determine resource allocation',
]

function DrillsPage() {
  const navigate = useNavigate()
  const [selectedType, setSelectedType] = useState(null)
  const [customTopic, setCustomTopic] = useState('')
  const [selectedTopic, setSelectedTopic] = useState('Social media does more harm than good')
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

    let url = `/debate-drill-rebuttal?motion=${encodeURIComponent(topic)}&position=${position}&weakness=${selectedType}`
    if (timerEnabled) {
      url += `&timer=${timerMinutes}`
    }
    navigate(url)
  }

  return (
    <>
      <SEO
        title="Debate Practice Drills - DebateLab"
        description="Practice specific debate skills with targeted debate drills. Choose from rebuttal, structure, weighing, evidence, and strategy drills to improve your debate argumentation and debate performance."
        keywords="debate drills, debate practice, debate training, rebuttal practice, debate structure, debate argumentation skills, debate coaching, debate exercises"
        url="https://debatelab.ai/debate-drills"
      />
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
          <h1>Debate Practice Drills</h1>
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
                  maxLength={500}
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
    </>
  )
}

export default DrillsPage
