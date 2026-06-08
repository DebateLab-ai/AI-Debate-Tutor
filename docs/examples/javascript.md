# Full debate lifecycle — JavaScript

Uses the built-in `fetch` API. Works in Node 18+ and modern browsers. We strongly recommend you call the API from your own server, not from a user's browser — your API key shouldn't ship to clients.

```javascript
const BASE = "https://api.debatelab.ai";
const KEY = process.env.DEBATELAB_KEY; // server-side only

const headers = {
  "X-API-Key": KEY,
  "Content-Type": "application/json",
};

// AI turns take 5-15s, scoring 3-8s. fetch has no default timeout — set one.
async function fetchWithTimeout(url, options = {}, ms = 60_000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(url, { ...options, signal: ctrl.signal });
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
    }
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function createDebate() {
  const res = await fetchWithTimeout(`${BASE}/api/v1/debates`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      motion: "THW ban single-use plastics globally",
      starter: "user",
      num_rounds: 2,
      mode: "casual",
      difficulty: "intermediate",
      external_user_id: "student-123",
      metadata: {
        Debater: "Nguyen An",
        Class: "Advanced Wednesdays",
      },
    }),
  });
  return res.json();
}

async function submitTurn(debateId, content) {
  const res = await fetchWithTimeout(
    `${BASE}/api/v1/debates/${debateId}/turns`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ content }),
    },
  );
  return res.json();
}

async function finishDebate(debateId) {
  const res = await fetchWithTimeout(
    `${BASE}/api/v1/debates/${debateId}/finish`,
    { method: "POST", headers },
  );
  return res.json();
}

async function downloadPdf(debateId, outPath) {
  const res = await fetchWithTimeout(
    `${BASE}/api/v1/debates/${debateId}/report.pdf`,
    { headers },
  );
  const buf = Buffer.from(await res.arrayBuffer());
  const fs = await import("node:fs/promises");
  await fs.writeFile(outPath, buf);
}

async function main() {
  const debate = await createDebate();
  console.log(`Created debate ${debate.id}`);

  let turn = await submitTurn(
    debate.id,
    "Plastic waste is choking our oceans. We see this with the " +
      "Great Pacific Garbage Patch. The harm is irreversible and global.",
  );
  console.log(`Round 1 done. Status: ${turn.status}, next: ${turn.next_speaker}`);

  turn = await submitTurn(
    debate.id,
    "Even granting the proposition framing, they still owe us a mechanism. " +
      "None of their arguments engage with the displacement problem I raised.",
  );
  console.log(`Round 2 done. Status: ${turn.status}`);

  const score = await finishDebate(debate.id);
  console.log(`Overall: ${score.overall} / 10`);
  console.log(`Weakness: ${score.weakness_type}`);

  await downloadPdf(debate.id, "debate_report.pdf");
  console.log("PDF saved to debate_report.pdf");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
```

## Handling errors

Wrap the call you want to recover from and inspect the status:

```javascript
async function submitTurnWithRetry(debateId, content) {
  try {
    return await submitTurn(debateId, content);
  } catch (e) {
    const status = parseInt(e.message.match(/^(\d{3})/)?.[1] || "0", 10);
    if (status === 502) {
      // AI provider failed; safe to retry once.
      return await submitTurn(debateId, content);
    }
    if (status === 429) {
      // Rate limited.
      await new Promise((r) => setTimeout(r, 5000));
      return await submitTurn(debateId, content);
    }
    throw e;
  }
}
```

For production we recommend a proper HTTP client like `undici` or `ky` that gives you typed errors and built-in retry semantics.

## Listing a student's debates

```javascript
async function listDebates({ externalUserId, limit = 50 } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (externalUserId) params.set("external_user_id", externalUserId);
  const res = await fetchWithTimeout(
    `${BASE}/api/v1/debates?${params}`,
    { headers },
  );
  return res.json();
}

const debates = await listDebates({ externalUserId: "student-123" });
for (const d of debates) {
  console.log(`${d.created_at} — ${d.motion} — status=${d.status}`);
}
```
