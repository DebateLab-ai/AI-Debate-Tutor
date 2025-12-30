import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SEO } from './SEO'

const CONVERTKIT_API_KEY = import.meta.env.VITE_CONVERTKIT_API_KEY
const CONVERTKIT_FORM_ID = '8910819'

// Environment variable is loaded at build time

function MailingListPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [firstName, setFirstName] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!email.trim()) {
      setError('Please enter your email address')
      return
    }

    if (!CONVERTKIT_API_KEY) {
      setError('Mailing list service is not configured. Please contact support.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // ConvertKit API v3 subscribe endpoint
      const requestBody = {
        api_key: CONVERTKIT_API_KEY,
        email: email.trim(),
      }
      
      // Only include first_name if it's not empty
      if (firstName.trim()) {
        requestBody.first_name = firstName.trim()
      }

      const response = await fetch(`https://api.convertkit.com/v3/forms/${CONVERTKIT_FORM_ID}/subscribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      const responseText = await response.text()
      let data
      try {
        data = JSON.parse(responseText)
      } catch (e) {
        // Invalid JSON response
        if (import.meta.env.DEV) {
          console.error('Failed to parse JSON response:', e)
        }
        data = { error: 'Invalid response from server' }
      }

      if (response.ok) {
        // Check if subscription was successful
        if (data.subscription) {
          setSuccess(true)
          setEmail('')
          setFirstName('')
          // Redirect to home after 3 seconds (give user time to see success message)
          setTimeout(() => {
            navigate('/')
          }, 3000)
        } else {
          setError('Subscription may have failed. Please try again or contact support.')
        }
      } else {
        // Log error in development only
        if (import.meta.env.DEV) {
          console.error('ConvertKit API Error:', {
            status: response.status,
            statusText: response.statusText,
            data: data,
          })
        }
        setError(data.message || data.error || 'Something went wrong. Please try again.')
      }
    } catch (err) {
      // Log error in development only
      if (import.meta.env.DEV) {
        console.error('Mailing list signup error:', err)
      }
      setError('Unable to subscribe. Please check your connection and try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <SEO
        title="Join Our Mailing List - DebateLab"
        description="Subscribe to DebateLab's mailing list to get notified about new features, debate tips, updates, and exclusive content."
        keywords="debate newsletter, debate updates, debate tips, debate community"
        url="https://debatelab.ai/mailing-list"
        noindex={true}
      />
      <div className="app mailing-list-mode">
      <button 
        className="return-to-landing" 
        onClick={() => navigate('/')}
        title="Return to home"
      >
        ← Home
      </button>
      
      <div className="mailing-list-container">
        <div className="mailing-list-card">
          <div className="mailing-list-header">
            <img 
              src="/favicon.png" 
              alt="DebateLab" 
              className="mailing-list-favicon"
            />
            <h1 className="mailing-list-title">Join Our Mailing List</h1>
          </div>
          <p className="mailing-list-description">
            Get notified about new features, debate tips, updates, and exclusive content.
          </p>

          {success ? (
            <div className="mailing-list-success">
              <div className="success-icon">✓</div>
              <h2>You're subscribed!</h2>
              <p>Thanks for joining. We'll send you updates soon.</p>
              <p className="redirect-notice">Redirecting to home...</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="mailing-list-form">
              <div className="form-group">
                <label htmlFor="email">Email Address *</label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your.email@example.com"
                  className="input-large"
                  required
                  disabled={loading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="firstName">First Name (Optional)</label>
                <input
                  id="firstName"
                  type="text"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  placeholder="John"
                  className="input-large"
                  disabled={loading}
                />
              </div>

              {error && (
                <div className="form-error">
                  {error}
                </div>
              )}

              <button
                type="submit"
                className="btn-primary btn-large"
                disabled={loading || !email.trim()}
              >
                {loading ? 'Subscribing...' : 'Subscribe'}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
    </>
  )
}

export default MailingListPage

