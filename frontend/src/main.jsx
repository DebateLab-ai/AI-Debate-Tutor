import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { HelmetProvider } from 'react-helmet-async'
import { Analytics } from '@vercel/analytics/react'
import LandingPage from './LandingPage'
import App from './App'
import DrillPage from './DrillPage'
import DrillsPage from './DrillsPage'
import MailingListPage from './MailingListPage'
import ErrorBoundary from './ErrorBoundary'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <HelmetProvider>
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/new-debate" element={<App />} />
            <Route path="/debate/:id" element={<App />} />
            <Route path="/debate" element={<App />} />
            <Route path="/debate-drills" element={<DrillsPage />} />
            <Route path="/debate-drill-rebuttal" element={<DrillPage />} />
            <Route path="/mailing-list" element={<MailingListPage />} />
        </Routes>
          <Analytics />
      </BrowserRouter>
    </ErrorBoundary>
    </HelmetProvider>
  </React.StrictMode>,
)

