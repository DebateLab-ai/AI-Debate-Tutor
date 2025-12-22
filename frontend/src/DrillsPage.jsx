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
  'This House believes social media does more harm than good',
  'This House would ban private schools',
  'This House supports universal basic income',
  'This House believes AI should be regulated',
  'This House would legalize all drugs',
  'This House supports open borders',
  'This House would abolish the death penalty',
  'This House believes climate change is the greatest threat to humanity',
  'This House would make voting mandatory',
  'This House supports the right to die',
  'This House would ban animal testing',
  'This House believes in free college education',
  'This House would implement a four-day work week',
  'This House supports the legalization of prostitution',
  'This House would ban all forms of gambling',
  'This House believes in strict gun control',
  'This House would raise the minimum wage significantly',
  'This House supports the abolition of prisons',
  'This House believes in the right to privacy over security',
  'This House would implement a carbon tax',
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

    navigate(`/drill-rebuttal?motion=${encodeURIComponent(topic)}&position=${position}&weakness=${selectedType}`)
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
