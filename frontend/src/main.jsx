import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import DrillPage from './DrillPage'
import DrillsPage from './DrillsPage'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/drills" element={<DrillsPage />} />
        <Route path="/drill-rebuttal" element={<DrillPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)

