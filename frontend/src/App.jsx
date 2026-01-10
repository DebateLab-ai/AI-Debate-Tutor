import { useState, useEffect, useRef, startTransition } from 'react'
import { Link, useNavigate, useLocation, useParams } from 'react-router-dom'
import { useToast, ToastContainer } from './Toast'
import { SEO, StructuredData, breadcrumbSchema } from './SEO'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Fetch with timeout utility
const fetchWithTimeout = async (url, options = {}, timeoutMs = 60000) => {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Request timed out')
    }
    throw error
  }
}

// Render markdown bold (**text**) as JSX
const renderMarkdown = (text) => {
  if (!text) return null

  const parts = []
  let lastIndex = 0
  const regex = /\*\*(.+?)\*\*/g
  let match

  while ((match = regex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }
    // Add bold text
    parts.push(<strong key={match.index}>{match[1]}</strong>)
    lastIndex = regex.lastIndex
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts.length > 0 ? parts : text
}

const DEBATE_TOPICS = [
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

// Topic categories with 10 prompts each
const TOPIC_CATEGORIES = {
  Politics: [
    'Mandatory voting in democratic elections',
    'Term limits should be imposed on all elected officials',
    'The abolition of the electoral college',
    'Political parties should be publicly funded',
    'Lowering the voting age to 16',
    'Gerrymandering should be illegal',
    'Proportional representation over first-past-the-post',
    'Campaign finance should be strictly limited',
    'The direct election of Supreme Court justices',
    'Political lobbying should be banned',
  ],
  Economics: [
    'A universal basic income',
    'The minimum wage should be tied to inflation',
    'The abolition of private property',
    'Free trade agreements harm developing nations',
    'Progressive taxation over flat taxation',
    'Student loan debt should be forgiven',
    'The nationalization of key industries',
    'Cryptocurrency should replace fiat currency',
    'A maximum wage cap',
    'Economic growth should be prioritized over environmental protection',
  ],
  Social: [
    'Affirmative action in college admissions',
    'Social media platforms should be regulated like public utilities',
    'The decriminalization of all drugs',
    'Healthcare is a human right',
    'Mandatory diversity training in workplaces',
    'Cancel culture is harmful to free speech',
    'The abolition of the death penalty',
    'Gender-neutral bathrooms should be mandatory',
    'Reparations for historical injustices',
    'Social media does more harm than good',
  ],
  Technology: [
    'A ban on facial recognition technology',
    'AI should be regulated by international law',
    'Net neutrality',
    'Social media companies should be broken up',
    'The right to digital privacy over national security',
    'Automation should be taxed to fund retraining programs',
    'Open-source software over proprietary software',
    'Tech companies should be held liable for harmful content',
    'A universal basic income funded by tech taxes',
    'Artificial intelligence will do more harm than good',
  ],
}

function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const params = useParams()
  const toast = useToast()
  const [debateId, setDebateId] = useState(null)
  const [debate, setDebate] = useState(null)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [score, setScore] = useState(null)
  const [scoring, setScoring] = useState(false)
  const [scoreError, setScoreError] = useState(null)

  // Landing/Setup state
  const [topic, setTopic] = useState('Social media does more harm than good')
  const [topicMode, setTopicMode] = useState('custom') // 'custom' or 'category'
  const [selectedCategory, setSelectedCategory] = useState(null)
  const [position, setPosition] = useState('for') // 'for' or 'against'
  const [numRounds, setNumRounds] = useState(2)
  const [mode, setMode] = useState('casual') // 'parliamentary' or 'casual'
  const [setupComplete, setSetupComplete] = useState(false)

  // Handle URL routing - load debate from URL or sync URL to state
  useEffect(() => {
    const urlId = params.id
    const currentPath = location.pathname

    // If we're on /new-debate, ensure state is cleared (user wants a new debate)
    if (currentPath === '/new-debate') {
      if (setupComplete) {
        setSetupComplete(false)
      }
      if (debateId || debate || messages.length > 0) {
        setDebateId(null)
        setDebate(null)
        setMessages([])
        setScore(null)
        setScoreError(null)
      }
      return
    }

    // If URL has an ID, load that debate
    if (urlId) {
      if (urlId !== debateId) {
        setDebateId(urlId)
        setSetupComplete(true)
        fetchDebate(urlId)
      }
      return
    }

    // If on /debate without ID, redirect to new-debate
    if (currentPath === '/debate') {
      navigate('/new-debate', { replace: true })
      return
    }

    // If no ID in URL but we have an active debate, sync URL (safety net)
    if (debateId && setupComplete) {
      navigate(`/debate/${debateId}`, { replace: true })
      return
    }

    // If no ID and no active debate, show form
    if (!debateId && !setupComplete) {
      navigate('/new-debate', { replace: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.id, debateId, setupComplete, location.pathname, navigate])

  // Timer state
  const [timerEnabled, setTimerEnabled] = useState(false)
  const [timerMinutes, setTimerMinutes] = useState(12)
  const [timeRemaining, setTimeRemaining] = useState(null) // in seconds
  const timerIntervalRef = useRef(null)

  // Input state
  const [argument, setArgument] = useState('')
  const [audioFile, setAudioFile] = useState(null)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState(null)
  const inputRef = useRef(null)

  // Offline detection
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)
    
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  const fetchScore = async (targetId = debateId, compute = false) => {
    if (!targetId) return null
    // Prevent multiple simultaneous score fetches
    if (scoring) {
      return null
    }
    if (!isOnline) {
      setScoreError("You are offline. Please check your connection.")
      return null
    }
    setScoring(true)
    try {
      const method = compute ? 'POST' : 'GET'
      const response = await fetchWithTimeout(`${API_BASE}/v1/debates/${targetId}/score`, {
        method,
        headers: { 'Content-Type': 'application/json' },
      }, 60000) // 60 second timeout for scoring

      if (response.ok) {
        const data = await response.json()
        setScore(data)
        setScoreError(null)
        return data
      }

      if (response.status === 404 && !compute) {
        // Score not yet generated; compute it
        return await fetchScore(targetId, true)
      }

      const errorText = await response.text()
      throw new Error(errorText || 'Failed to fetch score')
    } catch (error) {
      console.error('Error fetching score:', error)
      // Never expose raw error messages to users
      if (error.message && error.message.includes('Failed to fetch')) {
        setScoreError("Unable to load score - check your connection")
      } else {
        setScoreError("Unable to load score")
      }
      return null
    } finally {
      setScoring(false)
    }
  }
  // Note: Removed the automatic fetchDebate on debateId change
  // It's now handled by the URL routing effect above to prevent duplicate fetches

  // Timer countdown effect
  useEffect(() => {
    if (timerEnabled && debate?.next_speaker === 'user' && debate?.status === 'active') {
      // Start timer for user's turn
      if (timeRemaining === null) {
        setTimeRemaining(timerMinutes * 60)
      }

      timerIntervalRef.current = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev === null) return null
          if (prev <= 1) {
            clearInterval(timerIntervalRef.current)
            return 0
          }
          return prev - 1
        })
      }, 1000)

      return () => {
        if (timerIntervalRef.current) {
          clearInterval(timerIntervalRef.current)
        }
      }
    } else {
      // Not user's turn or timer disabled - clear timer
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current)
      }
      setTimeRemaining(null)
    }
  }, [timerEnabled, debate?.next_speaker, debate?.status, timeRemaining === null])

  // Auto-submit when timer expires
  useEffect(() => {
    if (timeRemaining === 0 && debate?.next_speaker === 'user' && !submitting) {
      toast.warning("⏱️ Time's up! Your argument was automatically submitted.", 6000)
      submitArgument(true)
    }
  }, [timeRemaining])

  const fetchDebate = async (targetId = debateId) => {
    if (!targetId) return null
    if (!isOnline) {
      console.error('Offline: Cannot fetch debate')
      return null
    }
    try {
      const response = await fetchWithTimeout(`${API_BASE}/v1/debates/${targetId}`, {}, 30000) // 30 second timeout
      if (!response.ok) {
        if (response.status === 404) {
          toast.error('Debate not found. Redirecting to new debate page.')
          // Reset state and redirect
          setDebateId(null)
          setDebate(null)
          setMessages([])
          setSetupComplete(false)
          navigate('/new-debate', { replace: true })
          return null
        }
        throw new Error('Failed to fetch debate')
      }
      const data = await response.json()
      setDebate(data)
      setMessages(data.messages || [])

      if (
        data.status === 'completed' &&
        targetId === debateId &&
        (!score || score.debate_id !== targetId) &&
        !scoring &&
        !scoreError
      ) {
        // Try GET first (score might already exist from a previous check)
        fetchScore(targetId, false)
      }
      return data
    } catch (error) {
      console.error('Error fetching debate:', error)
      return null
    }
  }

  const startDebate = async () => {
    if (topicMode === 'category' && !selectedCategory) {
      toast.error('Please select a category to generate a topic')
      return
    }

    if (!topic.trim()) {
      toast.error('Please enter a debate topic')
      return
    }

    if (!isOnline) {
      toast.error('You are offline. Please check your connection and try again.')
      return
    }
    setLoading(true)
    try {
      // Determine starter based on position
      // If user is "for", user starts; if "against", assistant starts (arguing for)
      const starter = position === 'for' ? 'user' : 'assistant'
      
      const response = await fetchWithTimeout(`${API_BASE}/v1/debates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_rounds: numRounds,
          starter,
          title: `${topic} (User: ${position}, You take the opposite position)`,
          mode: mode,
        }),
      }, 30000) // 30 second timeout

      if (!response.ok) {
        let errorMessage = 'Something went wrong. Please try again.'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.error || errorMessage
        } catch {
          const errorText = await response.text()
          if (errorText) {
            try {
              const parsed = JSON.parse(errorText)
              errorMessage = parsed.detail || parsed.error || errorMessage
            } catch {
              // If it's not JSON, use generic message
            }
          }
        }
        toast.error(errorMessage)
        setLoading(false)
        return
      }
      const data = await response.json()
      
      // Update all state first (React 18+ batches these synchronous updates)
      const debateUrl = `/debate/${data.id}`
      setDebateId(data.id)
      setDebate(data)
      setSetupComplete(true)
      setScore(null)
      setScoreError(null)
      
      // Navigate using replace: false to avoid issues, then generate AI turn if needed
      // Keep loading state true - generateAITurn will manage it
      navigate(debateUrl, { replace: false })
      
      if (starter === 'assistant') {
        // Generate AI turn - it will handle loading state
        // No delay needed as we want smooth transition
        generateAITurn(data.id)
      } else {
        // If user starts, set loading to false
        setLoading(false)
      }
    } catch (error) {
      console.error('Error starting debate:', error)
      // Check if it's a network/connection error or timeout
      if (error.message && error.message.includes('timed out')) {
        toast.error('The request took too long. Please check your connection and try again.')
      } else if (error.message && error.message.includes('Failed to fetch')) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else {
        toast.error('Something went wrong. Please try again.')
      }
      setLoading(false)
    }
    // Note: Don't set loading to false in finally block when assistant starts,
    // as generateAITurn needs to manage loading state
  }

  const transcribeAudio = async (file) => {
    if (!isOnline) {
      toast.error('You are offline. Please check your connection.')
      return null
    }
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetchWithTimeout(`${API_BASE}/v1/transcribe`, {
        method: 'POST',
        body: formData,
      }, 120000) // 120 second timeout for transcription (can be slow)
      if (!response.ok) throw new Error('Transcription failed')
      const data = await response.json()
      return data.text
    } catch (error) {
      console.error('Transcription error:', error)
      if (error.message && error.message.includes('timed out')) {
        toast.error('Transcription took too long. Please try again.')
      } else if (error.message && error.message.includes('Failed to fetch')) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else {
        toast.error('Unable to transcribe audio. Please try again.')
      }
      return null
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const audioChunks = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data)
        }
      }

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
        
        setTranscribing(true)
        const transcribed = await transcribeAudio(audioFile)
        if (transcribed) {
          setArgument(prev => prev ? `${prev} ${transcribed}` : transcribed)
        }
        setTranscribing(false)
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop())
      }

      recorder.start()
      setMediaRecorder(recorder)
      setRecording(true)
    } catch (error) {
      console.error('Recording error:', error)
      toast.error('Unable to access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorder && recording) {
      mediaRecorder.stop()
      setRecording(false)
      setMediaRecorder(null)
    }
  }

  const submitArgument = async (autoSubmit = false) => {
    if (!debateId) {
      if (!autoSubmit) toast.error('Please enter your argument')
      return
    }

    if (!argument.trim() && !autoSubmit) {
      toast.error('Please enter your argument')
      return
    }

    if (!isOnline) {
      toast.error('You are offline. Please check your connection and try again.')
      return
    }

    setSubmitting(true)
    try {
      const finalContent = argument.trim()

      const response = await fetchWithTimeout(`${API_BASE}/v1/debates/${debateId}/turns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speaker: 'user',
          content: finalContent,
        }),
      }, 30000) // 30 second timeout

      if (!response.ok) {
        let errorMessage = 'Something went wrong. Please try again.'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.error || errorMessage
        } catch {
          const errorText = await response.text()
          if (errorText) {
            try {
              const parsed = JSON.parse(errorText)
              errorMessage = parsed.detail || parsed.error || errorMessage
            } catch {
              // If it's not JSON, use generic message
            }
          }
        }
        
        // If debate not found, reset to new debate form
        if (response.status === 404 || errorMessage.toLowerCase().includes('not found')) {
          toast.error('Debate not found. Starting a new debate...')
          setTimeout(() => {
            resetDebate()
          }, 1000)
          setSubmitting(false)
          return
        }
        
        toast.error(errorMessage)
        setSubmitting(false)
        return
      }

      const turnData = await response.json()

      // Create message object from the submission
      // Note: TurnOut doesn't include content, so we use the original argument
      const newMessage = {
        id: turnData.message_id,
        round_no: turnData.round_no,
        speaker: 'user',
        content: finalContent,
        created_at: new Date().toISOString(),
      }
      
      // Update messages state immediately
      setMessages(prev => [...prev, newMessage])
      
      // Update debate state from the response
      setDebate(prev => prev ? {
        ...prev,
        current_round: turnData.current_round,
        next_speaker: turnData.next_speaker,
        status: turnData.status,
      } : null)
      
      setArgument('')
      setAudioFile(null)
      
      // Auto-generate AI response if it's the assistant's turn
      if (turnData.status === 'active' && turnData.next_speaker === 'assistant') {
        // Start AI generation immediately - no delays or tracking interference
        generateAITurn(debateId)
      } else if (turnData.status === 'completed') {
        // Compute score directly (POST) since we know it doesn't exist yet
        await fetchScore(debateId, true)
        
        // Track user turn for completed debates only (when there's no active streaming)
        // Do this after score computation to avoid any interference
        const debateUrl = `/debate/${debateId}`
        if (debateId) {
          setTimeout(() => {
            navigate(`/track/user-turn?return=${encodeURIComponent(debateUrl)}`, { replace: true })
          }, 600)
        }
      }
    } catch (error) {
      console.error('Error submitting argument:', error)
      if (error.message && (error.message.includes('fetch') || error.message.includes('network') || error.message.includes('Failed to fetch'))) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else {
        toast.error('Something went wrong. Please try again.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  const generateAITurn = async (targetId = debateId) => {
    if (!targetId) return
    if (!isOnline) {
      toast.error('You are offline. Please check your connection.')
      return
    }

    // Set loading state - but only if not already loading (prevents flicker)
    if (!loading) {
      setLoading(true)
    }
    
    // Create a temporary message ID for streaming
    const tempMessageId = `temp-${Date.now()}`
    const tempMessage = {
      id: tempMessageId,
      round_no: debate?.current_round || 1,
      speaker: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    }
    
    // Add empty message placeholder for streaming
    setMessages(prev => [...prev, tempMessage])
    setLoading(false) // Turn off loading so we can see the streaming message
    
    let streamingContent = ''
    let finalData = null
    
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}/auto-turn?stream=true`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Failed to generate AI response')
      }

      // Handle streaming response (Server-Sent Events)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.trim() === '') continue
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.chunk) {
                // Append chunk to streaming content
                streamingContent += data.chunk
                // Update the message with accumulated content
                setMessages(prev => prev.map(msg => 
                  msg.id === tempMessageId 
                    ? { ...msg, content: streamingContent }
                    : msg
                ))
              }
              
              if (data.done) {
                // Streaming complete - save final data
                finalData = data
                
                // Replace temporary message with final message
                // Update synchronously - React 18+ will batch these automatically
                const finalMessage = {
                  id: data.message_id,
                  round_no: data.round_no,
                  speaker: 'assistant',
                  content: data.content || streamingContent,
                  created_at: new Date().toISOString(),
                }
                
                // All state updates happen synchronously - React batches them into one render
                setMessages(prev => prev.map(msg => 
                  msg.id === tempMessageId ? finalMessage : msg
                ))
                
                setDebate(prev => prev ? {
                  ...prev,
                  current_round: data.current_round,
                  next_speaker: data.next_speaker,
                  status: data.status,
                } : null)
                
                // If debate completed, compute the score
                if (data.status === 'completed') {
                  setTimeout(() => {
                    fetchScore(targetId, true)
                  }, 400)
                }
              }
            } catch (parseError) {
              // Skip malformed JSON lines
              console.warn('Failed to parse SSE data:', parseError)
            }
          }
        }
      }
      
      // Handle any remaining buffer (edge case where done message is in final buffer)
      if (buffer.trim() && buffer.startsWith('data: ')) {
        try {
          const data = JSON.parse(buffer.slice(6))
          if (data.done && !finalData) {
            finalData = data
            // Replace temporary message with final message
            // Update synchronously - React 18+ will batch these automatically
            const finalMessage = {
              id: data.message_id,
              round_no: data.round_no,
              speaker: 'assistant',
              content: data.content || streamingContent,
              created_at: new Date().toISOString(),
            }
            
            // All state updates happen synchronously - React batches them into one render
            setMessages(prev => prev.map(msg => 
              msg.id === tempMessageId ? finalMessage : msg
            ))
            
            setDebate(prev => prev ? {
              ...prev,
              current_round: data.current_round,
              next_speaker: data.next_speaker,
              status: data.status,
            } : null)
            
            // If debate completed, compute the score
            if (data.status === 'completed') {
              setTimeout(() => {
                fetchScore(targetId, true)
              }, 400)
            }
          }
        } catch (e) {
          console.warn('Failed to parse final buffer:', e)
        }
      }
    } catch (error) {
      console.error('Error generating AI turn:', error)
      
      // Remove temporary message on error
      setMessages(prev => prev.filter(msg => msg.id !== tempMessageId))
      
      // NEVER expose error.message - could contain prompts during network failures
      if (error.message && error.message.includes('timed out')) {
        toast.error('The AI response took too long. Please try again.')
      } else if (error.message && error.message.includes('Failed to fetch')) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else {
        toast.error('Unable to generate AI response. Please try again.')
      }
      setLoading(false)
    }
  }

  const finishDebate = async () => {
    if (!debateId) return
    if (!isOnline) {
      toast.error('You are offline. Please check your connection.')
      return
    }
    setLoading(true)
    try {
      const response = await fetchWithTimeout(`${API_BASE}/v1/debates/${debateId}/finish`, {
        method: 'POST',
      }, 30000) // 30 second timeout
      if (!response.ok) throw new Error('Failed to finish debate')
      await fetchDebate()
      await fetchScore(debateId, true)
    } catch (error) {
      console.error('Error finishing debate:', error)
      if (error.message && (error.message.includes('fetch') || error.message.includes('network') || error.message.includes('Failed to fetch'))) {
        toast.error('Unable to connect. Please check your internet connection and try again.')
      } else {
        toast.error('Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const resetDebate = () => {
    // Stop recording if active
    if (recording && mediaRecorder) {
      stopRecording()
    }
    
    // Clear all state first
    setDebateId(null)
    setDebate(null)
    setMessages([])
    setSetupComplete(false)
    setArgument('')
    setAudioFile(null)
    setScore(null)
    setScoreError(null)
    setRecording(false)
    setTranscribing(false)
    setMediaRecorder(null)
    setMode('casual')
    setTopic('Social media does more harm than good')
    setTopicMode('custom')
    setSelectedCategory(null)
    setNumRounds(2)
    setTimeRemaining(null)
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
    }
    
    // Navigate to new debate form - use replace to avoid back button issues
    navigate('/new-debate', { replace: true })
  }

  const getScoreGrade = (score) => {
    // Score is on a 0-10 scale
    if (score >= 9) return 'Excellent'
    if (score >= 8) return 'Great'
    if (score >= 7) return 'Good'
    if (score >= 6) return 'Fair'
    return 'Needs Improvement'
  }


  // Setup screen
  if (!setupComplete) {
    return (
      <>
        <SEO
          title="Start a Debate - DebateLab"
          description="Start a new debate with an AI opponent. Choose from Politics, Economics, Social, or Technology topics, or create your own. Practice your argumentation skills with instant feedback and personalized coaching."
          keywords="start debate, debate practice, AI debate opponent, debate training, debate topics, argumentation practice"
          url="https://debatelab.ai/new-debate"
          noindex={false}
        />
        <div className="app setup-mode">
        <Link to="/" className="return-to-landing" title="Return to home">
          ← Home
        </Link>
        <div className="setup-container">
          <div className="setup-card">
            <h1>New Debate</h1>
            <p className="subtitle">Practice your debating skills with an AI opponent</p>
            
            <div className="form-group">
              <label>Debate Topic</label>
              <div className="topic-mode-selector" style={{ 
                display: 'flex', 
                gap: '12px', 
                marginBottom: '16px' 
              }}>
                <button
                  type="button"
                  onClick={() => {
                    setTopicMode('custom')
                    setSelectedCategory(null)
                  }}
                  className={topicMode === 'custom' ? 'position-btn active' : 'position-btn'}
                  style={{ flex: 1 }}
                >
                  Custom Topic
                </button>
                <button
                  type="button"
                  onClick={() => setTopicMode('category')}
                  className={topicMode === 'category' ? 'position-btn active' : 'position-btn'}
                  style={{ flex: 1 }}
                >
                  Generate Topic
                </button>
              </div>

              {topicMode === 'custom' ? (
                <>
                  <input
                    type="text"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="e.g., This House opposes the continued use of international military aid"
                    className="input-large"
                    maxLength={500}
                    onKeyPress={(e) => e.key === 'Enter' && startDebate()}
                  />
                  <small style={{ display: 'block', marginTop: '6px', color: '#666', fontSize: '0.875rem' }}>
                    💡 You can edit the sample topic above or type your own
                  </small>
                </>
              ) : (
                <>
                  <div className="category-buttons" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '16px' }}>
                    {Object.keys(TOPIC_CATEGORIES).map((category) => {
                      const categoryColors = {
                        Politics: { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.4)', active: 'rgba(239, 68, 68, 0.9)', text: '#ef4444' },
                        Economics: { bg: 'rgba(234, 179, 8, 0.15)', border: 'rgba(234, 179, 8, 0.4)', active: 'rgba(234, 179, 8, 0.9)', text: '#eab308' },
                        Social: { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgba(34, 197, 94, 0.4)', active: 'rgba(34, 197, 94, 0.9)', text: '#22c55e' },
                        Technology: { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.4)', active: 'rgba(59, 130, 246, 0.9)', text: '#3b82f6' },
                      }
                      const colors = categoryColors[category] || categoryColors.Politics
                      const isActive = selectedCategory === category
                      
                      return (
                        <button
                          key={category}
                          type="button"
                          onClick={() => {
                            setSelectedCategory(category)
                            // Generate random topic from category
                            const topics = TOPIC_CATEGORIES[category]
                            const randomTopic = topics[Math.floor(Math.random() * topics.length)]
                            setTopic(randomTopic)
                          }}
                          style={{
                            padding: '16px',
                            background: isActive ? colors.active : colors.bg,
                            border: `2px solid ${isActive ? colors.active : colors.border}`,
                            borderRadius: '12px',
                            color: isActive ? 'white' : colors.text,
                            fontSize: '16px',
                            fontWeight: 600,
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            textAlign: 'center',
                          }}
                        >
                          {category}
                        </button>
                      )
                    })}
                  </div>
                  {selectedCategory && (
                    <div style={{
                      padding: '16px',
                      background: 'var(--input-bg)',
                      border: '1px solid var(--border)',
                      borderRadius: '12px',
                      marginBottom: '12px'
                    }}>
                      <div style={{ fontSize: '14px', color: '#8b92a7', marginBottom: '8px' }}>
                        Selected: <strong style={{ color: 'var(--accent)' }}>{selectedCategory}</strong>
                      </div>
                      <div style={{ 
                        fontSize: '16px', 
                        color: '#e8eaed',
                        lineHeight: '1.5',
                        marginBottom: '12px',
                        padding: '12px',
                        background: 'rgba(255, 255, 255, 0.03)',
                        borderRadius: '8px',
                        border: '1px solid rgba(255, 255, 255, 0.05)'
                      }}>
                        {topic}
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          const topics = TOPIC_CATEGORIES[selectedCategory]
                          const randomTopic = topics[Math.floor(Math.random() * topics.length)]
                          setTopic(randomTopic)
                        }}
                        className="btn-secondary"
                        style={{ 
                          fontSize: '14px', 
                          padding: '8px 16px',
                          width: '100%'
                        }}
                      >
                        🔄 Generate Another Topic
                      </button>
                    </div>
                  )}
                  <small style={{ display: 'block', marginTop: '8px', color: '#8b92a7', fontSize: '0.875rem' }}>
                    💡 Select a category to generate a random debate topic
                  </small>
                </>
              )}
            </div>

            <div className="form-group">
              <label>Your Position</label>
              <div className="position-buttons">
                <button
                  className={position === 'for' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setPosition('for')}
                >
                  For
                </button>
                <button
                  className={position === 'against' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setPosition('against')}
                >
                  Against
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Debate Mode</label>
              <div className="position-buttons">
                <button
                  className={mode === 'casual' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setMode('casual')}
                >
                  Casual
                  <small>Conversational</small>
                </button>
                <button
                  className={mode === 'parliamentary' ? 'position-btn active' : 'position-btn'}
                  onClick={() => {
                    setMode('parliamentary')
                    if (numRounds > 3) setNumRounds(3) // Cap rounds for parliamentary
                  }}
                >
                  Parliamentary
                  <small>Competition-level debate</small>
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Number of Rounds</label>
              <select
                value={numRounds}
                onChange={(e) => setNumRounds(parseInt(e.target.value))}
                className="input-large"
              >
                {mode === 'parliamentary' ? (
                  <>
                    <option value={1}>1 Round</option>
                    <option value={2}>2 Rounds</option>
                    <option value={3}>3 Rounds</option>
                  </>
                ) : (
                  <>
                    <option value={1}>1 Round</option>
                    <option value={2}>2 Rounds</option>
                    <option value={3}>3 Rounds</option>
                    <option value={4}>4 Rounds</option>
                    <option value={5}>5 Rounds</option>
                    <option value={6}>6 Rounds</option>
                    <option value={7}>7 Rounds</option>
                    <option value={8}>8 Rounds</option>
                    <option value={9}>9 Rounds</option>
                    <option value={10}>10 Rounds</option>
                  </>
                )}
              </select>
            </div>

            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={timerEnabled}
                  onChange={(e) => setTimerEnabled(e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Enable Timer (creates time pressure)
              </label>
              {timerEnabled && (
                <select
                  value={timerMinutes}
                  onChange={(e) => setTimerMinutes(parseInt(e.target.value))}
                  className="input-large"
                  style={{ marginTop: '10px' }}
                >
                  <option value={3}>3 minutes per speech</option>
                  <option value={5}>5 minutes per speech</option>
                  <option value={7}>7 minutes per speech</option>
                  <option value={10}>10 minutes per speech</option>
                  <option value={12}>12 minutes per speech</option>
                  <option value={15}>15 minutes per speech</option>
                </select>
              )}
            </div>

            <button
              className="btn-primary btn-large"
              onClick={startDebate}
              disabled={loading || !topic.trim() || (topicMode === 'category' && !selectedCategory)}
            >
              {loading ? 'Starting...' : 'Start Debate'}
            </button>
          </div>
        </div>
      </div>
      </>
    )
  }

  // Debate screen
  return (
    <>
      <SEO
        title={debate ? `${topic} - DebateLab` : "Debate - DebateLab"}
        description={debate ? `Active debate: ${topic}. Practice your argumentation skills with an AI opponent and get instant feedback on your debate performance.` : "Start a debate with an AI opponent. Practice your argumentation skills and get instant feedback."}
        keywords={`debate, ${topic}, argumentation, debate practice, AI debate opponent`}
        url={`https://debatelab.ai${debateId ? `/debate/${debateId}` : '/new-debate'}`}
        noindex={!!debateId} // Don't index individual debate pages (they're dynamic/private)
      />
      <div className="app debate-mode">
      <ToastContainer toasts={toast.toasts} removeToast={toast.removeToast} />
      <Link to="/" className="return-to-landing" onClick={resetDebate} title="Return to home">
        ← Home
      </Link>
      <header className="debate-header">
        {!isOnline && (
          <div style={{
            background: 'rgba(255, 107, 107, 0.2)',
            border: '1px solid rgba(255, 107, 107, 0.5)',
            color: '#ff6b6b',
            padding: '8px 16px',
            borderRadius: '8px',
            marginBottom: '12px',
            fontSize: '14px',
            textAlign: 'center'
          }}>
            ⚠️ You are offline. Please check your connection.
          </div>
        )}
        <div className="header-content">
          <div>
            <h2>{topic}</h2>
            <p className="header-subtitle">
              Your position: <strong>{position === 'for' ? 'FOR' : 'AGAINST'}</strong>
              {' • '}
              Round {debate?.current_round || 1} of {debate?.num_rounds || 3}
              {' • '}
              <span className={`status ${debate?.status || 'active'}`}>
                {debate?.status === 'active' ? 'Active' : 'Completed'}
              </span>
            </p>
          </div>
          <button className="btn-secondary" onClick={resetDebate}>
            New Debate
          </button>
        </div>
      </header>

      <main className="debate-main">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>
                {debate?.next_speaker === 'assistant'
                  ? debate?.mode === 'parliamentary'
                    ? 'Waiting for AI to generate opening argument... (usually takes 10-20 seconds)'
                    : 'Waiting for AI to generate opening argument...'
                  : 'Make your first argument!'}
              </p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div
              key={`${message.speaker}-${message.round_no}-${index}`}
              className={`message ${message.speaker === 'user' ? 'message-user' : 'message-ai'}`}
            >
              <div className="message-header">
                <span className="message-speaker">
                  {message.speaker === 'user'
                    ? `You (${position === 'for' ? 'FOR' : 'AGAINST'})`
                    : `AI (${position === 'for' ? 'AGAINST' : 'FOR'})`
                  }
                </span>
                <span className="message-round">Round {message.round_no}</span>
              </div>
              <div className="message-content">{renderMarkdown(message.content)}</div>
            </div>
          ))}
          
          {loading && debate?.status === 'active' && (
            <div className="message message-ai">
              <div className="message-header">
                <span className="message-speaker">
                  AI ({position === 'for' ? 'AGAINST' : 'FOR'})
                </span>
              </div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                {debate?.mode === 'parliamentary' && (
                  <div style={{ marginTop: '8px', fontSize: '12px', color: '#8b92a7' }}>
                    Generating response... (usually takes 10-20 seconds)
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {debate?.status === 'active' && debate?.next_speaker === 'user' && (
          <div className="input-container">
            <div className="input-wrapper">
              {timerEnabled && timeRemaining !== null && (
                <div className={`timer-display ${timeRemaining < 60 ? 'timer-warning' : ''}`}>
                  ⏱️ Time Remaining: {Math.floor(timeRemaining / 60)}:{String(timeRemaining % 60).padStart(2, '0')}
                </div>
              )}
              <textarea
                ref={inputRef}
                value={argument}
                onChange={(e) => setArgument(e.target.value)}
                placeholder="Type your argument here..."
                className="argument-input"
                rows={4}
                maxLength={10000}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    submitArgument()
                  }
                }}
                disabled={submitting}
              />
              <div className="input-actions">
                <button
                  className={`speech-btn ${recording ? 'recording' : ''}`}
                  onClick={recording ? stopRecording : startRecording}
                  disabled={transcribing || submitting}
                  title={recording ? 'Stop recording' : 'Start voice recording'}
                >
                  {transcribing ? '⏳ Transcribing...' : recording ? '🔴 Stop' : '🎤 Voice'}
                </button>
                <button
                  className="btn-primary"
                  onClick={submitArgument}
                  disabled={submitting || !argument.trim()}
                >
                  {submitting ? 'Submitting...' : 'Send'}
                </button>
              </div>
              <p className="input-hint">Press Cmd/Ctrl + Enter to send</p>
            </div>
          </div>
        )}

        {debate?.status === 'completed' && (
          <div className="debate-ended">
            <p>Debate completed! Review your score or start a new debate to continue practicing.</p>
            {score ? (
              <div className="score-card">
                <h3>Your Debate Performance</h3>
                <div 
                  className="score-display"
                  data-grade={getScoreGrade(score.overall).toLowerCase().replace(/\s+/g, '-')}
                >
                  <div className="score-number">{Math.round(score.overall)}</div>
                  <div className="score-label">out of 10</div>
                  <div className="score-bar-container">
                    <div 
                      className="score-bar-fill" 
                      style={{ width: `${(score.overall / 10) * 100}%` }}
                    ></div>
                  </div>
                  <div className="score-grade">{getScoreGrade(score.overall)}</div>
                </div>
                <div className="score-metrics">
                  <div>
                    <span>Content & Structure</span>
                    <strong>{score.metrics?.content_structure != null ? score.metrics.content_structure.toFixed(1) : '–'}</strong>
                    <small>{score.content_structure_feedback || 'No feedback available.'}</small>
                  </div>
                  <div>
                    <span>Engagement & Clash</span>
                    <strong>{score.metrics?.engagement != null ? score.metrics.engagement.toFixed(1) : '–'}</strong>
                    <small>{score.engagement_feedback || 'No feedback available.'}</small>
                  </div>
                  <div>
                    <span>Strategy & Execution</span>
                    <strong>{score.metrics?.strategy != null ? score.metrics.strategy.toFixed(1) : '–'}</strong>
                    <small>{score.strategy_feedback || 'No feedback available.'}</small>
                  </div>
                </div>
                <div className="score-feedback">
                  <h4>Judge Feedback</h4>
                  <p>{score.feedback || 'No overall feedback available.'}</p>

                  {/* Show drill recommendation based on identified weakness */}
                  {score.weakness_type && (
                    <div className="drill-recommendation">
                      <p className="drill-rec-text">
                        💡 Focus on improving your <strong>{score.weakness_type}</strong> skills with a targeted drill!
                      </p>
                      <Link
                        to={`/debate-drill-rebuttal?motion=${encodeURIComponent(topic)}&position=${position}&weakness=${encodeURIComponent(score.weakness_type)}`}
                        className="btn-drill"
                      >
                        Practice {score.weakness_type.charAt(0).toUpperCase() + score.weakness_type.slice(1)} Skills →
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="score-card">
                <p className="muted">
                  {scoring
                    ? 'Calculating score... (this usually takes 15-30 seconds)'
                    : scoreError
                      ? `Unable to load score: ${scoreError}`
                      : 'Score not available yet.'}
                </p>
                <button
                  className="btn-secondary"
                  onClick={() => fetchScore(debateId, true)}
                  disabled={scoring}
                >
                  {scoring ? 'Analyzing your performance...' : 'Calculate Score'}
                </button>
              </div>
            )}
            <button className="btn-primary" onClick={resetDebate}>
              New Debate
            </button>
          </div>
        )}
      </main>
    </div>
    </>
  )
}

export default App
