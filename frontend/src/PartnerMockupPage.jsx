import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { SEO } from './SEO'
import {
  createPartnerDebate,
  defaultPartnerBaseUrl,
  downloadPartnerReport,
  finishPartnerDebate,
  getPartnerDebate,
  listPartnerDebates,
  openPartnerDebate,
  pingPartner,
  submitPartnerTurn,
} from './partnerApi'

const ENV_KEY = import.meta.env.VITE_PARTNER_API_KEY || ''

function PartnerMockupPage() {
  const [baseUrl, setBaseUrl] = useState(defaultPartnerBaseUrl())
  const [apiKey, setApiKey] = useState(ENV_KEY)
  const [motion, setMotion] = useState('THW ban single-use plastics globally')
  const [starter, setStarter] = useState('user')
  const [numRounds, setNumRounds] = useState(2)
  const [mode, setMode] = useState('casual')
  const [difficulty, setDifficulty] = useState('intermediate')
  const [externalUserId, setExternalUserId] = useState('superjuniors-student-001')
  const [debaterName, setDebaterName] = useState('Test Student')
  const [turnContent, setTurnContent] = useState(
    'Plastic waste is choking our oceans. The Great Pacific Garbage Patch shows irreversible harm to marine ecosystems and human health through microplastics.',
  )
  const [debate, setDebate] = useState(null)
  const [messages, setMessages] = useState([])
  const [score, setScore] = useState(null)
  const [busy, setBusy] = useState(false)
  const [log, setLog] = useState([])

  const debateId = debate?.id

  const appendLog = (label, data, isError = false) => {
    setLog((prev) => [
      {
        id: `${Date.now()}-${prev.length}`,
        at: new Date().toLocaleTimeString(),
        label,
        data,
        isError,
      },
      ...prev,
    ])
  }

  const run = async (label, fn) => {
    setBusy(true)
    try {
      const result = await fn()
      appendLog(label, result)
      return result
    } catch (err) {
      appendLog(label, err.message, true)
      throw err
    } finally {
      setBusy(false)
    }
  }

  const metadata = useMemo(
    () => (debaterName.trim() ? { Debater: debaterName.trim() } : {}),
    [debaterName],
  )

  const handlePing = () =>
    run('GET /api/v1/ping', () => pingPartner(baseUrl, apiKey))

  const handleCreate = () =>
    run('POST /api/v1/debates', async () => {
      const created = await createPartnerDebate(baseUrl, apiKey, {
        motion,
        starter,
        num_rounds: numRounds,
        mode,
        difficulty,
        external_user_id: externalUserId || undefined,
        metadata: Object.keys(metadata).length ? metadata : undefined,
      })
      setDebate(created)
      setMessages([])
      setScore(null)
      return created
    })

  const handleOpen = () => {
    if (!debateId) return
    return run('POST /api/v1/debates/{id}/open', async () => {
      const opened = await openPartnerDebate(baseUrl, apiKey, debateId, null)
      setDebate((d) => ({
        ...d,
        next_speaker: opened.next_speaker,
        current_round: opened.current_round,
        status: opened.status,
      }))
      setMessages((m) => [...m, opened.assistant_message])
      return opened
    })
  }

  const handleTurn = () => {
    if (!debateId) return
    return run('POST /api/v1/debates/{id}/turns', async () => {
      const turn = await submitPartnerTurn(
        baseUrl,
        apiKey,
        debateId,
        turnContent,
        null,
      )
      setDebate((d) => ({
        ...d,
        next_speaker: turn.next_speaker,
        current_round: turn.current_round,
        status: turn.status,
      }))
      setMessages((m) => {
        const next = [...m, turn.user_message]
        if (turn.assistant_message) next.push(turn.assistant_message)
        return next
      })
      return turn
    })
  }

  const handleFinish = () => {
    if (!debateId) return
    return run('POST /api/v1/debates/{id}/finish', async () => {
      const result = await finishPartnerDebate(baseUrl, apiKey, debateId)
      setScore(result)
      setDebate((d) => ({ ...d, status: 'completed', next_speaker: null }))
      return result
    })
  }

  const handleRefresh = () => {
    if (!debateId) return
    return run('GET /api/v1/debates/{id}', async () => {
      const full = await getPartnerDebate(baseUrl, apiKey, debateId)
      setDebate(full)
      setMessages(full.messages || [])
      setScore(full.score || null)
      return full
    })
  }

  const handleList = () =>
    run('GET /api/v1/debates', () =>
      listPartnerDebates(baseUrl, apiKey, {
        externalUserId: externalUserId || undefined,
        limit: 20,
      }),
    )

  const handlePdf = () => {
    if (!debateId) return
    return run('GET /api/v1/debates/{id}/report.pdf', async () => {
      const blob = await downloadPartnerReport(baseUrl, apiKey, debateId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `debate_report_${debateId.slice(0, 8)}.pdf`
      a.click()
      URL.revokeObjectURL(url)
      return { bytes: blob.size, type: blob.type }
    })
  }

  const needsOpen = debate?.status === 'active' && debate?.next_speaker === 'assistant'
  const canTurn = debate?.status === 'active' && debate?.next_speaker === 'user'
  const canFinish = debate && (debate.status === 'completed' || messages.length > 0)

  return (
    <>
      <SEO
        title="Partner API Mockup - DebateLab"
        description="Internal mockup for testing the DebateLab partner API (/api/v1/*)."
        noindex
      />
      <div className="app setup-mode">
        <div className="setup-container">
          <div className="setup-card partner-mockup-card">
            <Link to="/" className="return-to-landing" title="Return to home">
              ← Home
            </Link>
            <h1>Partner API mockup</h1>
            <p className="setup-subtitle">
              Simulates a third-party integration against <code>/api/v1/*</code>. For testing only —
              do not embed live API keys in a public client.
            </p>

            <section className="partner-mockup-section">
              <h2>Connection</h2>
              <label className="form-label">API base URL</label>
              <input
                className="input-large"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.debatelab.ai"
              />
              <label className="form-label">X-API-Key</label>
              <input
                className="input-large"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk_live_..."
                autoComplete="off"
              />
              <div className="partner-mockup-actions">
                <button type="button" className="btn-secondary" onClick={handlePing} disabled={busy}>
                  Test ping
                </button>
                <button type="button" className="btn-secondary" onClick={handleList} disabled={busy}>
                  List debates
                </button>
              </div>
            </section>

            <section className="partner-mockup-section">
              <h2>Create debate</h2>
              <label className="form-label">Motion</label>
              <input
                className="input-large"
                value={motion}
                onChange={(e) => setMotion(e.target.value)}
              />
              <label className="form-label">external_user_id</label>
              <input
                className="input-large"
                value={externalUserId}
                onChange={(e) => setExternalUserId(e.target.value)}
              />
              <label className="form-label">metadata.Debater</label>
              <input
                className="input-large"
                value={debaterName}
                onChange={(e) => setDebaterName(e.target.value)}
              />
              <div className="partner-mockup-grid">
                <div>
                  <label className="form-label">starter</label>
                  <select
                    className="input-large"
                    value={starter}
                    onChange={(e) => setStarter(e.target.value)}
                  >
                    <option value="user">user (student first)</option>
                    <option value="assistant">assistant (AI first)</option>
                  </select>
                </div>
                <div>
                  <label className="form-label">num_rounds</label>
                  <input
                    className="input-large"
                    type="number"
                    min={1}
                    max={10}
                    value={numRounds}
                    onChange={(e) => setNumRounds(parseInt(e.target.value, 10) || 1)}
                  />
                </div>
                <div>
                  <label className="form-label">mode</label>
                  <select className="input-large" value={mode} onChange={(e) => setMode(e.target.value)}>
                    <option value="casual">casual</option>
                    <option value="wsdc">wsdc</option>
                    <option value="ap">ap</option>
                  </select>
                </div>
                <div>
                  <label className="form-label">difficulty</label>
                  <select
                    className="input-large"
                    value={difficulty}
                    onChange={(e) => setDifficulty(e.target.value)}
                  >
                    <option value="beginner">beginner</option>
                    <option value="intermediate">intermediate</option>
                    <option value="advanced">advanced</option>
                  </select>
                </div>
              </div>
              <button
                type="button"
                className="btn-primary btn-large btn-effects"
                onClick={handleCreate}
                disabled={busy}
              >
                Create debate
              </button>
            </section>

            {debate && (
              <section className="partner-mockup-section">
                <h2>Active debate</h2>
                <div className="partner-mockup-status">
                  <div><strong>id</strong> {debate.id}</div>
                  <div><strong>status</strong> {debate.status}</div>
                  <div><strong>next_speaker</strong> {String(debate.next_speaker)}</div>
                  <div><strong>round</strong> {debate.current_round} / {debate.num_rounds}</div>
                </div>

                {needsOpen && (
                  <button type="button" className="btn-primary" onClick={handleOpen} disabled={busy}>
                    POST /open (AI speaks first)
                  </button>
                )}

                {canTurn && (
                  <>
                    <label className="form-label">Student speech</label>
                    <textarea
                      className="input-large partner-mockup-textarea"
                      rows={5}
                      value={turnContent}
                      onChange={(e) => setTurnContent(e.target.value)}
                    />
                    <button
                      type="button"
                      className="btn-primary btn-large btn-effects"
                      onClick={handleTurn}
                      disabled={busy || !turnContent.trim()}
                    >
                      Submit turn
                    </button>
                  </>
                )}

                <div className="partner-mockup-actions">
                  <button type="button" className="btn-secondary" onClick={handleRefresh} disabled={busy}>
                    Refresh state
                  </button>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handleFinish}
                    disabled={busy || !canFinish}
                  >
                    Finish &amp; score
                  </button>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handlePdf}
                    disabled={busy || !score}
                  >
                    Download PDF
                  </button>
                </div>

                {messages.length > 0 && (
                  <div className="partner-mockup-messages">
                    <h3>Transcript</h3>
                    {messages.map((m) => (
                      <div key={m.id} className={`partner-mockup-msg partner-mockup-msg-${m.speaker}`}>
                        <div className="partner-mockup-msg-meta">
                          Round {m.round_no} · {m.speaker}
                        </div>
                        <p>{m.content}</p>
                      </div>
                    ))}
                  </div>
                )}

                {score && (
                  <div className="partner-mockup-score">
                    <h3>Score — {score.overall}/10</h3>
                    <p>{score.feedback}</p>
                    <ul>
                      <li>Content/structure: {score.metrics?.content_structure}</li>
                      <li>Engagement: {score.metrics?.engagement}</li>
                      <li>Strategy: {score.metrics?.strategy}</li>
                    </ul>
                    <p><strong>weakness_type:</strong> {score.weakness_type}</p>
                  </div>
                )}
              </section>
            )}

            <section className="partner-mockup-section">
              <h2>Request log</h2>
              <div className="partner-mockup-log">
                {log.length === 0 && <p className="partner-mockup-log-empty">No requests yet.</p>}
                {log.map((entry) => (
                  <details key={entry.id} className={entry.isError ? 'partner-log-error' : ''}>
                    <summary>
                      [{entry.at}] {entry.label}
                    </summary>
                    <pre>{typeof entry.data === 'string' ? entry.data : JSON.stringify(entry.data, null, 2)}</pre>
                  </details>
                ))}
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  )
}

export default PartnerMockupPage
