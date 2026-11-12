/* ==========================================================================
   chat.js — ask/stream/finalize pipeline, persistence, scroll management
   ========================================================================== */

import { sseChat } from './api.js';
import { el, iconEl, copyText } from './ui.js';
import { renderRich, mdToHtml, sanitize, enhance, renderMath } from './markdown.js';
import { toast } from './toasts.js';
import { store } from './store.js';
import { showPage, saveToDeck, showCitations } from './panels.js';

const $ = (id) => document.getElementById(id);
const log = $('log');
const qBox = $('q');

const welcomeTpl = document.getElementById('welcome'); // detached on first ask, reused on "New chat"

let chatState = 'idle';
let controller = null;
let thread = null;
let lastQuestion = '';

/** Completed user/assistant pairs, sent to the LLM as context. */
let history = [];
/** Persisted entries: {role, content, q?, ctx?} */
let msgs = [];

let stick = true; // follow the stream unless the user scrolls up

/* ==========================================================================
   state + shell helpers
   ========================================================================== */

function setState(s) {
  chatState = s;
  const busy = s !== 'idle';
  $('sendLabel').textContent = busy ? 'Stop' : 'Ask';
  $('icSend').hidden = busy;
  $('icStop').hidden = !busy;
  $('sendBtn').classList.toggle('stop', busy);
}

function ensureThread() {
  if (!thread) {
    welcomeTpl?.remove();
    thread = el('div', { class: 'thread' });
    log.append(thread);
  }
  return thread;
}

function showWelcome() {
  thread?.remove();
  thread = null;
  if (welcomeTpl) log.prepend(welcomeTpl);
}

function autoScroll() {
  if (stick) log.scrollTo({ top: log.scrollHeight, behavior: 'instant' });
}

/* ==========================================================================
   message renderers
   ========================================================================== */

function userMsg(text) {
  const bubble = el('div', { class: 'bubble', text });
  return el('article', { class: 'msg user' },
    el('div', { class: 'who' }, el('span', { class: 'tag', text: 'You' }),
      el('span', { class: 'avatar plain', text: 'Y' })),
    bubble,
  );
}

function assistantShell() {
  const bubble = el('div', { class: 'bubble md' });
  const msg = el('article', { class: 'msg assistant' },
    el('div', { class: 'who' },
      el('span', { class: 'avatar', text: 'AI' }),
      el('span', { class: 'tag', text: 'Notes RAG' }),
      el('span', { class: 'who-note small', hidden: true })),
    bubble,
  );
  return { msg, bubble };
}

function skeletonInto(bubble) {
  bubble.replaceChildren(
    el('div', { class: 'retrieve-skel', 'aria-label': 'Retrieving from notes' },
      el('div', { class: 'skel' }), el('div', { class: 'skel' }), el('div', { class: 'skel' })),
  );
}

function thinkingInto(bubble) {
  bubble.replaceChildren(
    el('div', { class: 'thinking' },
      el('i'), el('i'), el('i'), el('span', { text: ' composing answer…' })),
  );
}

/** Sources row: PYQ chip + diagram chips + passage summary chip. */
function sourcesRow(ctx) {
  const row = el('div', { class: 'sources' });
  const n = (ctx.passages || []).length;
  if ((ctx.pyq || []).length) {
    row.append(el('button', { class: 'source-chip pyq', type: 'button',
      onclick: () => showCitations(Object.assign(trimCtx(ctx), {
        passages: ctx.pyq, method: 'pyq', diagrams: [],
      }), { bringToFront: true }) },
      iconEl('sparkles'), `${ctx.pyq.length} PYQ`));
  }
  row.append(el('button', { class: 'source-chip', type: 'button',
    onclick: () => showCitations(trimCtx(ctx), { bringToFront: true }) },
    iconEl('doc'), `${n} passage${n === 1 ? '' : 's'}`));
  for (const d of (ctx.diagrams || []).slice(0, 3)) {
    row.append(el('button', { class: 'source-chip', type: 'button',
      onclick: () => showPage(d.file, d.page) }, iconEl('image'), `p.${d.page}`));
  }
  return row;
}

/** Collapsible retrieval details (rendered once, after the stream). */
function ctxDetails(ctx) {
  const ps = ctx.passages || [];
  const pyq = ctx.pyq || [];
  if (!ps.length && !pyq.length) return null;
  const det = el('details', { class: 'ctx' },
    el('summary', {}, iconEl('chevR'),
      el('span', { text: `${ps.length} passages · ${pyq.length ? pyq.length + ' PYQ · ' : ''}${ctx.method || 'bm25'}` })),
  );
  const inner = el('div', {});
  if (pyq.length) {
    inner.append(el('div', { class: 'p-meta pyq-head', text: 'PAST-YEAR QUESTIONS & OFFICIAL KEYS' }));
    for (const p of pyq.slice(0, 6)) {
      inner.append(el('div', { class: 'p-item pyq-item' },
        el('div', { class: 'p-meta', text: `${p.file} · p.${p.page}` }),
        el('div', { class: 'p-text', text: (p.text || '').slice(0, 400) + ((p.text || '').length > 400 ? '…' : '') }),
      ));
    }
  }
  for (const p of ps.slice(0, 10)) {
    inner.append(el('div', { class: 'p-item' },
      el('div', { class: 'p-meta', text: `${p.file} · p.${p.page} · ${p.topic || ''}` }),
      el('div', { class: 'p-text', text: (p.text || '').slice(0, 400) + ((p.text || '').length > 400 ? '…' : '') }),
    ));
  }
  det.append(inner);
  return det;
}

/* ==========================================================================
   answer actions
   ========================================================================== */

function copyAction(answer) {
  return el('button', { type: 'button', onclick: async (e) => {
    const ok = await copyText(answer);
    const btn = e.currentTarget;
    btn.classList.toggle('done', ok);
    btn.replaceChildren(iconEl(ok ? 'check' : 'copy'), document.createTextNode(ok ? 'Copied' : 'Copy'));
    setTimeout(() => {
      btn.classList.remove('done');
      btn.replaceChildren(iconEl('copy'), document.createTextNode('Copy'));
    }, 1500);
  } }, iconEl('copy'), 'Copy');
}

function deckAction(q, a, ctx) {
  const d = (ctx.diagrams || [])[0];
  return el('button', { type: 'button',
    onclick: (e) => { saveToDeck(q, a, d?.file ?? null, d?.page ?? null, d?.path ?? null, d?.topic ?? null); e.currentTarget.classList.add('done'); } },
    iconEl('deck'), 'Save to Deck');
}

function openPageAction(ctx) {
  const d = (ctx.diagrams || [])[0];
  if (!d) return null;
  return el('button', { type: 'button', onclick: () => showPage(d.file, d.page) },
    iconEl('doc'), `Open p.${d.page}`);
}

function regenAction() {
  return el('button', { type: 'button', onclick: regenerate }, iconEl('refresh'), 'Regenerate');
}

/* ==========================================================================
   the ask / stream pipeline
   ========================================================================== */

let paintQueued = false;
let lastPaint = 0;
let active = null; // {bubble, msg, answer, ctx, question}

function paint() {
  if (!active) return;
  const { bubble, answer } = active;
  if (answer) {
    try {
      bubble.innerHTML = sanitize(mdToHtml(answer));
      enhance(bubble);
      renderMath(bubble);
      bubble.append(el('span', { class: 'caret', 'aria-hidden': 'true' }));
    } catch { /* a bad chunk must never kill the stream */ }
  }
  autoScroll();
}

function queuePaint() {
  if (paintQueued) return;
  paintQueued = true;
  requestAnimationFrame((ts) => {
    paintQueued = false;
    const now = performance.now();
    if (now - lastPaint < 80) { setTimeout(queuePaint, 90); return; }
    lastPaint = now;
    paint();
  });
}

async function runStream(question) {
  const { msg, bubble } = assistantShell();
  skeletonInto(bubble);
  ensureThread().append(msg);
  autoScroll();

  active = { bubble, msg, answer: '', ctx: null, question };
  controller = new AbortController();
  setState('retrieving');
  const whoNote = msg.querySelector('.who-note');

  const finalize = (stopped = false) => {
    const { answer, ctx, question: q, note } = active;
    bubble.replaceChildren();
    if (answer) {
      renderRich(bubble, answer);
    } else if (note) {
      bubble.append(note); // keep the no_llm / error explanation
    } else {
      bubble.append(el('p', { class: 'small',
        text: stopped ? 'Stopped before any tokens arrived.' : 'No answer text returned.' }));
    }
    if (ctx) {
      msg.append(sourcesRow(ctx));
      bubble.append(ctxDetails(ctx) ?? '');
    }
    if (stopped && answer) bubble.append(el('p', { class: 'small', text: '⏹ Stopped by user — partial answer kept.' }));

    if (answer) {
      const acts = el('div', { class: 'acts' },
        copyAction(answer),
        ctx ? deckAction(q, answer, ctx) : null,
        ctx ? openPageAction(ctx) : null,
        regenAction(),
      );
      msg.append(acts);
      history.push({ role: 'user', content: q });
      history.push({ role: 'assistant', content: answer });
    }

    msgs.push({ role: 'assistant', content: answer, q, ctx: ctx ? trimCtx(ctx) : null });
    persist();
    active = null;
    controller = null;
    setState('idle');
    autoScroll();
  };

  try {
    await sseChat({
      question,
      history,
      signal: controller.signal,
      onEvent(ev, data) {
        if (ev === 'context') {
          active.ctx = data;
          setState('thinking');
          thinkingInto(bubble);
          if (whoNote) {
            whoNote.hidden = false;
            whoNote.textContent = ` · ${data.method || 'bm25'}${(data.loops || 1) > 1 ? ` ×${data.loops}` : ''}`;
          }
        } else if (ev === 'delta') {
          setState('streaming');
          active.answer += data.t || '';
          queuePaint();
        } else if (ev === 'no_llm' || ev === 'error') {
          const note = el('div', { class: 'ctx' },
            el('span', { class: 'small', text: data.message || 'Something went wrong.' }));
          if (ev === 'no_llm') {
            note.append(el('button', { class: 'source-chip', type: 'button',
              onclick: () => document.getElementById('btnSettings').click() },
              iconEl('gear'), 'Open settings'));
          }
          bubble.replaceChildren(note);
          active.note = note;
        } else if (ev === 'done') {
          finalize(false);
        }
      },
    });
    if (active) finalize(false); // stream closed without a done event
  } catch (e) {
    if (e.name === 'AbortError') {
      if (active) finalize(true); // keep partial content
    } else if (active) {
      bubble.append(el('p', { class: 'small', text: `Connection error: ${e.message}` }));
      active = null;
      controller = null;
      setState('idle');
    }
  }
}

function trimCtx(ctx) {
  return {
    method: ctx.method,
    loops: ctx.loops,
    passages: (ctx.passages || []).slice(0, 8).map((p) => ({
      file: p.file, page: p.page, topic: p.topic, text: (p.text || '').slice(0, 280),
    })),
    pyq: (ctx.pyq || []).slice(0, 6),
    diagrams: (ctx.diagrams || []).slice(0, 6).map((d) => ({
      url: d.url, path: d.path, file: d.file, page: d.page, caption: (d.caption || '').slice(0, 120),
    })),
  };
}

export function ask(text) {
  const t = (text ?? '').trim();
  if (!t || chatState !== 'idle') return;
  ensureThread().append(userMsg(t));
  msgs.push({ role: 'user', content: t });
  lastQuestion = t;
  persist();
  autoScroll();
  qBox.value = '';
  resizeBox();
  runStream(t);
}

export function regenerate() {
  if (chatState !== 'idle' || !lastQuestion) return;
  // drop the trailing pair + its message bubble
  if (history.length >= 2) history.splice(-2);
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') { msgs.splice(i, 1); break; }
  }
  const lastA = [...ensureThread().children].reverse()
    .find((c) => c.classList.contains('assistant'));
  lastA?.remove();
  persist();
  runStream(lastQuestion);
}

export function stopGeneration() {
  if (controller) controller.abort();
}

export function newChat() {
  if (chatState !== 'idle') return;
  const snapshot = msgs;
  history = [];
  msgs = [];
  lastQuestion = '';
  store.del('draft');
  qBox.value = '';
  showWelcome();
  if (snapshot.length) {
    toast('Started a new chat', { type: 'ok', action: { label: 'Undo', onClick: () => restore(snapshot) } });
  }
}

/* ==========================================================================
   persistence + restore
   ========================================================================== */

function persist() {
  store.set('chat', msgs.slice(-30).map(({ role, content, q, ctx }) => ({ role, content, q, ctx })));
}

function renderRestored(m) {
  ensureThread();
  if (m.role === 'user') {
    thread.append(userMsg(m.content));
    return;
  }
  const { msg, bubble } = assistantShell();
  if (m.content) renderRich(bubble, m.content);
  else bubble.append(el('p', { class: 'small', text: '(no answer text)' }));
  if (m.ctx) {
    msg.append(sourcesRow(m.ctx));
    bubble.append(ctxDetails(m.ctx) ?? '');
    if (m.content) {
      msg.append(el('div', { class: 'acts' },
        copyAction(m.content),
        deckAction(m.q || m.content, m.content, m.ctx),
        openPageAction(m.ctx),
      ));
    }
  }
  thread.append(msg);
}

export function restore(snapshot = store.get('chat', [])) {
  msgs = snapshot;
  history = [];
  for (const m of msgs) if (m.content) history.push({ role: m.role, content: m.content });
  thread?.replaceChildren();
  thread = null;
  showWelcome();
  if (msgs.length) {
    ensureThread();
    for (const m of msgs) renderRestored(m);
    log.scrollTo({ top: log.scrollHeight, behavior: 'instant' });
  }
}

/* ==========================================================================
   composer + scroll wiring
   ========================================================================== */

function resizeBox() {
  qBox.style.height = 'auto';
  qBox.style.height = Math.min(qBox.scrollHeight, 180) + 'px';
}

let draftTimer = null;
function saveDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => store.set('draft', qBox.value), 300);
}

export function initChat() {
  setState('idle');

  $('sendBtn').addEventListener('click', () => {
    if (chatState === 'idle') ask(qBox.value); else stopGeneration();
  });
  qBox.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (chatState === 'idle') ask(qBox.value);
    }
  });
  qBox.addEventListener('input', () => { resizeBox(); saveDraft(); });
  qBox.value = store.get('draft', '') || '';
  resizeBox();

  log.addEventListener('scroll', () => {
    const gap = log.scrollHeight - log.scrollTop - log.clientHeight;
    stick = gap < 90;
    $('scrollDown').hidden = stick;
  }, { passive: true });
  $('scrollDown').addEventListener('click', () => {
    log.scrollTo({ top: log.scrollHeight, behavior: 'smooth' });
  });

  restore();
}
