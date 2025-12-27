import { useState, useEffect } from 'react'
import './Toast.css'

export const ToastContainer = ({ toasts, removeToast }) => {
  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          toast={toast}
          onClose={() => removeToast(toast.id)}
        />
      ))}
    </div>
  )
}

const Toast = ({ toast, onClose }) => {
  useEffect(() => {
    if (toast.duration) {
      const timer = setTimeout(() => {
        onClose()
      }, toast.duration)
      return () => clearTimeout(timer)
    }
  }, [toast.duration, onClose])

  return (
    <div className={`toast toast-${toast.type}`}>
      <div className="toast-content">
        <span className="toast-icon">
          {toast.type === 'success' && '✓'}
          {toast.type === 'error' && '✕'}
          {toast.type === 'info' && 'ℹ'}
          {toast.type === 'warning' && '⚠'}
        </span>
        <span className="toast-message">{toast.message}</span>
      </div>
      <button className="toast-close" onClick={onClose} aria-label="Close notification">
        ×
      </button>
    </div>
  )
}

// Hook for using toasts
export const useToast = () => {
  const [toasts, setToasts] = useState([])

  const addToast = (message, type = 'info', duration = 5000) => {
    const id = Date.now()
    setToasts((prev) => [...prev, { id, message, type, duration }])
  }

  const removeToast = (id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }

  return {
    toasts,
    addToast,
    removeToast,
    success: (message, duration) => addToast(message, 'success', duration),
    error: (message, duration) => addToast(message, 'error', duration),
    info: (message, duration) => addToast(message, 'info', duration),
    warning: (message, duration) => addToast(message, 'warning', duration),
  }
}
