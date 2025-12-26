import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Analytics } from '@vercel/analytics/react'
import LandingPage from './LandingPage'
import App from './App'
import DrillPage from './DrillPage'
import DrillsPage from './DrillsPage'
import ErrorBoundary from './ErrorBoundary'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/debate" element={<App />} />
          <Route path="/drills" element={<DrillsPage />} />
          <Route path="/drill-rebuttal" element={<DrillPage />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
    <Analytics />
  </React.StrictMode>,
)

