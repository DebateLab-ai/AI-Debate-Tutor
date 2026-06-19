import React, { Fragment } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { HelmetProvider } from 'react-helmet-async'
import { Analytics } from '@vercel/analytics/react'
import LandingPage from './LandingPage'
import App from './App'
import DrillPage from './DrillPage'
import DrillsPage from './DrillsPage'
import MailingListPage from './MailingListPage'
import TrackingPage from './TrackingPage'
import PartnerMockupPage from './PartnerMockupPage'
import ErrorBoundary from './ErrorBoundary'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <HelmetProvider>
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
            <Route path="/" element={<LandingPage />} />
            {/* One App instance for all debate flows — avoids full remount (and lost AI stream) when going /new-debate → /debate/:id */}
            <Route element={<App />}>
              <Route path="/new-debate" element={<Fragment />} />
              <Route path="/debate/:id" element={<Fragment />} />
              <Route path="/debate" element={<Fragment />} />
            </Route>
            <Route path="/debate-drills" element={<DrillsPage />} />
            <Route path="/debate-drill-rebuttal" element={<DrillPage />} />
            <Route path="/mailing-list" element={<MailingListPage />} />
            <Route path="/partner-mockup" element={<PartnerMockupPage />} />
            {/* Tracking routes for analytics (no custom events needed) */}
            <Route path="/track/user-turn" element={<TrackingPage />} />
            <Route path="/track/drill-rebuttal-submit" element={<TrackingPage />} />
            <Route path="/track/drill-submit" element={<TrackingPage />} />
        </Routes>
          <Analytics />
      </BrowserRouter>
    </ErrorBoundary>
    </HelmetProvider>
  </React.StrictMode>,
)

