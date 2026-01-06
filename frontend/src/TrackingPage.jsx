import { useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'

// Tracking page component that immediately redirects back
// This allows Vercel Analytics to track the page view without custom events
function TrackingPage() {
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    // Get the return URL from query params or default to home
    const params = new URLSearchParams(location.search)
    const returnUrl = params.get('return') || '/'
    
    // Immediately redirect back - Vercel Analytics will have tracked the page view
    navigate(returnUrl, { replace: true })
  }, [navigate, location.search])

  return null
}

export default TrackingPage

