import React from 'react'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { 
      hasError: false, 
      error: null,
      errorInfo: null
    }
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    // Log error details to console for debugging
    console.error('Error caught by boundary:', error, errorInfo)
    // Update state with error info
    this.setState({
      error,
      errorInfo
    })
    
    // In production, you could send this to an error reporting service:
    // logErrorToService(error, errorInfo)
  }

  handleReset = () => {
    // Reset error state and reload the component tree
    this.setState({ 
      hasError: false, 
      error: null,
      errorInfo: null 
    })
    // Optionally reload the page for a clean state
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI
      return (
        <div style={{
          minHeight: '100vh',
          background: 'var(--bg, #0a0e1a)',
          color: 'var(--text, #e8eaed)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }}>
          <div style={{
            maxWidth: '600px',
            width: '100%',
            background: 'var(--panel, #151b2e)',
            border: '1px solid var(--border, #1f2538)',
            borderRadius: '16px',
            padding: '40px',
            textAlign: 'center',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
          }}>
            <div style={{
              fontSize: '64px',
              marginBottom: '24px'
            }}>
              ⚠️
            </div>
            <h1 style={{
              fontSize: '28px',
              fontWeight: '700',
              margin: '0 0 16px 0',
              color: 'var(--text, #e8eaed)'
            }}>
              Something went wrong
            </h1>
            <p style={{
              fontSize: '16px',
              color: 'var(--muted, #8b92a7)',
              margin: '0 0 32px 0',
              lineHeight: '1.6'
            }}>
              An unexpected error occurred. Don't worry, your data is safe. 
              Try reloading the page or returning to the home page.
            </p>
            
            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'center',
              flexWrap: 'wrap'
            }}>
              <button
                onClick={this.handleReset}
                style={{
                  background: 'var(--accent, #4a90e2)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '12px 24px',
                  fontSize: '16px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                  fontFamily: 'inherit'
                }}
                onMouseOver={(e) => e.target.style.background = 'var(--accent-hover, #5ba0f2)'}
                onMouseOut={(e) => e.target.style.background = 'var(--accent, #4a90e2)'}
              >
                Reload Page
              </button>
              <button
                onClick={() => window.location.href = '/'}
                style={{
                  background: 'transparent',
                  color: 'var(--text, #e8eaed)',
                  border: '1px solid var(--border, #1f2538)',
                  borderRadius: '8px',
                  padding: '12px 24px',
                  fontSize: '16px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  fontFamily: 'inherit'
                }}
                onMouseOver={(e) => {
                  e.target.style.background = 'var(--panel, #151b2e)'
                  e.target.style.borderColor = 'var(--accent, #4a90e2)'
                }}
                onMouseOut={(e) => {
                  e.target.style.background = 'transparent'
                  e.target.style.borderColor = 'var(--border, #1f2538)'
                }}
              >
                Go to Home
              </button>
            </div>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details style={{
                marginTop: '32px',
                textAlign: 'left',
                background: 'var(--input-bg, #0f1524)',
                borderRadius: '8px',
                padding: '16px',
                border: '1px solid var(--border, #1f2538)'
              }}>
                <summary style={{
                  cursor: 'pointer',
                  color: 'var(--muted, #8b92a7)',
                  fontSize: '14px',
                  marginBottom: '12px',
                  userSelect: 'none'
                }}>
                  Error Details (Development Only)
                </summary>
                <pre style={{
                  color: '#ef4444',
                  fontSize: '12px',
                  overflow: 'auto',
                  margin: '0',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {this.state.error.toString()}
                  {this.state.errorInfo?.componentStack && (
                    <>
                      {'\n\n'}
                      {this.state.errorInfo.componentStack}
                    </>
                  )}
                </pre>
              </details>
            )}
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
