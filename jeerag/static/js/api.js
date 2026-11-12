/* ==========================================================================
   api.js — typed-ish wrappers for every jeerag endpoint + the SSE chat client
   ========================================================================== */

async function jget(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function jpost(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

const enc = encodeURIComponent;

export const api = {
  stats: () => jget('/api/stats'),
  config: () => jget('/api/config'),
  saveConfig: (updates) => jpost('/api/config', updates),
  cards: () => jget('/api/cards'),
  addCard: (card) => jpost('/api/cards', card),
  grade: (id, score) => jpost('/api/cards', { action: 'grade', id, score }),
  deleteCard: (id) => jpost('/api/cards', { action: 'delete', id }),
  tracker: (days) => jget(`/api/tracker?days=${days}`),
  search: (q) => jget(`/api/search?q=${enc(q)}`),
  diagrams: (q) => jget(`/api/diagrams?q=${enc(q)}`),
  formulas: (q, k = 6) => jget(`/api/formulas?q=${enc(q)}&k=${k}`),
  recent: (limit = 10) => jget(`/api/recent?limit=${limit}`),
  page: (file, page) => jget(`/api/page?file=${enc(file)}&page=${page}`),
  providers: () => jget('/api/providers'),
};

/**
 * POST /api/chat and dispatch Server-Sent Events.
 * Events: context {passages,diagrams,method,loops} · delta {t} ·
 *         done {} · no_llm {message} · error {message}
 * Returns when the stream ends; throws on HTTP/abort errors.
 */
export async function sseChat({ question, history, signal, onEvent }) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const m = chunk.match(/^event: (\w+)\ndata: (.*)$/s);
      if (!m) continue;
      let data;
      try { data = JSON.parse(m[2]); } catch { continue; }
      onEvent(m[1], data);
    }
  }
}
