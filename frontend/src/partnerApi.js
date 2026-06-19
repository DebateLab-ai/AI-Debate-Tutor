/**
 * Thin client for the public partner API (/api/v1/*).
 * Intended for server-side or internal testing — do not ship API keys in production SPAs.
 */

const DEFAULT_BASE =
  import.meta.env.VITE_PARTNER_API_BASE_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  'https://api.debatelab.ai'

function buildHeaders(apiKey, extra = {}) {
  if (!apiKey?.trim()) {
    throw new Error('API key is required')
  }
  return {
    'X-API-Key': apiKey.trim(),
    'Content-Type': 'application/json',
    ...extra,
  }
}

export async function partnerFetch(baseUrl, apiKey, path, options = {}, timeoutMs = 90_000) {
  const url = `${baseUrl.replace(/\/$/, '')}${path}`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: buildHeaders(apiKey, options.headers),
    })

    const contentType = response.headers.get('content-type') || ''
    let body
    if (contentType.includes('application/pdf')) {
      body = await response.blob()
    } else if (contentType.includes('application/json')) {
      body = await response.json()
    } else {
      body = await response.text()
    }

    if (!response.ok) {
      const detail =
        typeof body === 'object' && body?.detail
          ? body.detail
          : typeof body === 'string'
            ? body
            : JSON.stringify(body)
      const err = new Error(`${response.status}: ${detail}`)
      err.status = response.status
      err.body = body
      throw err
    }

    return { response, body }
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s`)
    }
    if (err instanceof TypeError) {
      throw new Error(
        'Network error (browser blocked the response — often a server 500 without CORS headers, or no connection)',
      )
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export function defaultPartnerBaseUrl() {
  return DEFAULT_BASE
}

export async function pingPartner(baseUrl, apiKey) {
  const { body } = await partnerFetch(baseUrl, apiKey, '/api/v1/ping', { method: 'GET' }, 15_000)
  return body
}

export async function createPartnerDebate(baseUrl, apiKey, payload) {
  const { body } = await partnerFetch(baseUrl, apiKey, '/api/v1/debates', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  return body
}

export async function openPartnerDebate(baseUrl, apiKey, debateId, idempotencyKey) {
  const { body } = await partnerFetch(baseUrl, apiKey, `/api/v1/debates/${debateId}/open`, {
    method: 'POST',
    headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
  })
  return body
}

export async function submitPartnerTurn(baseUrl, apiKey, debateId, content, idempotencyKey) {
  const { body } = await partnerFetch(baseUrl, apiKey, `/api/v1/debates/${debateId}/turns`, {
    method: 'POST',
    body: JSON.stringify({ content }),
    headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
  }, 120_000)
  return body
}

export async function finishPartnerDebate(baseUrl, apiKey, debateId) {
  const { body } = await partnerFetch(baseUrl, apiKey, `/api/v1/debates/${debateId}/finish`, {
    method: 'POST',
  }, 120_000)
  return body
}

export async function getPartnerDebate(baseUrl, apiKey, debateId) {
  const { body } = await partnerFetch(baseUrl, apiKey, `/api/v1/debates/${debateId}`, {
    method: 'GET',
  })
  return body
}

export async function listPartnerDebates(baseUrl, apiKey, { externalUserId, since, limit } = {}) {
  const params = new URLSearchParams()
  if (externalUserId) params.set('external_user_id', externalUserId)
  if (since) params.set('since', since)
  if (limit) params.set('limit', String(limit))
  const qs = params.toString()
  const { body } = await partnerFetch(
    baseUrl,
    apiKey,
    `/api/v1/debates${qs ? `?${qs}` : ''}`,
    { method: 'GET' },
  )
  return body
}

export async function downloadPartnerReport(baseUrl, apiKey, debateId) {
  const { body } = await partnerFetch(
    baseUrl,
    apiKey,
    `/api/v1/debates/${debateId}/report.pdf`,
    { method: 'GET' },
    60_000,
  )
  return body
}

export async function startPartnerRebuttalDrill(baseUrl, apiKey, payload) {
  const { body } = await partnerFetch(baseUrl, apiKey, '/api/v1/drills/rebuttal/start', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  return body
}

export async function submitPartnerRebuttalDrill(baseUrl, apiKey, payload) {
  const { body } = await partnerFetch(baseUrl, apiKey, '/api/v1/drills/rebuttal/submit', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 90_000)
  return body
}
