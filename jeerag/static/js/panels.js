/* ==========================================================================
   panels.js — citations/search drawer, page viewer, Doubt Deck, settings,
   study stats. Each overlay self-registers with the global Esc stack.
   ========================================================================== */

import { api } from './api.js';
import { el, esc, iconEl, overlayStack } from './ui.js';
import { renderRich, renderMath } from './markdown.js';
import { toast } from './toasts.js';
import { ACCENTS, currentAccent, currentTheme, setAccent, setTheme } from './theme.js';

const $ = (id) => document.getElementById(id);

/* ==========================================================================
   Drawer (citations / notes search / figures)
   ========================================================================== */

let drawerMode = null; // 'citations' | 'search'
let drawerOpen = false;

function closeDrawer() {
  if (!drawerOpen) return;
  drawerOpen = false;
  drawerMode = null;
  $('drawer').classList.remove('open');
  $('drawer').setAttribute('aria-hidden', 'true');
  $('scrim').hidden = true;
  overlayStack.remove(closeDrawer);
}

export function drawerIsOpen() { return drawerOpen; }

function openDrawer(mode, title) {
  if (drawerOpen && drawerMode === mode) { // same mode → just refocus
    $('drawerTitle').textContent = title;
    return;
  }
  if (!drawerOpen) {
    drawerOpen = true;
    $('drawer').classList.add('open');
    $('drawer').setAttribute('aria-hidden', 'false');
    $('scrim').hidden = false;
    overlayStack.push(closeDrawer);
  }
  drawerMode = mode;
  $('drawerTitle').textContent = title;
}

export function showCitations(ctx, { bringToFront = false } = {}) {
  const body = $('drawerBody');
  body.replaceChildren();
  openDrawer('citations', `Citations · ${ctx.method || 'bm25'}`);

  // --- passages ---
  const passages = ctx.passages || [];
  const pSec = el('div', { class: 'drawer-sec' },
    el('h3', {}, el('span', { text: 'Passages' }), el('span', { class: 'count', text: String(passages.length) })),
  );
  for (const p of passages) {
    const d = el('details', { class: 'passage' },
      el('summary', {},
        iconEl('chevR'),
        el('span', { class: 'pl', text: `${p.file} · p.${p.page}` }),
        el('span', { class: 'pt', text: p.topic || '' }),
      ),
      el('div', { class: 'p-body' },
        el('div', { text: (p.text || '').slice(0, 600) + ((p.text || '').length > 600 ? '…' : '') }),
        el('button', {
          class: 'btn ghost small-p', type: 'button',
          onclick: () => showPage(p.file, p.page),
        }, iconEl('doc'), `Open p.${p.page}`),
      ),
    );
    pSec.append(d);
  }
  if (!passages.length) pSec.append(el('p', { class: 'small', text: 'No passages.' }));
  body.append(pSec);

  // --- diagrams ---
  const diags = ctx.diagrams || [];
  const dSec = el('div', { class: 'drawer-sec' },
    el('h3', {}, el('span', { text: 'Diagrams' }), el('span', { class: 'count', text: String(diags.length) })),
  );
  if (diags.length) {
    const gal = el('div', { class: 'gal' });
    for (const g of diags) gal.append(figureCard(g));
    dSec.append(gal);
  } else {
    dSec.append(el('p', { class: 'small', text: 'No diagrams on these pages.' }));
  }
  body.append(dSec);

  if (bringToFront) $('drawer').querySelector('.drawer-head h2')?.scrollIntoView();
}

/* ---------- notes / figures search ---------- */

function figureCard(g) {
  const fig = el('figure', {},
    el('img', { src: g.url, alt: `figure · ${g.file} p.${g.page}`, loading: 'lazy',
      onclick: () => showPage(g.file, g.page) }),
    el('figcaption', {},
      el('b', { text: `${g.file} · p.${g.page}`, onclick: () => showPage(g.file, g.page) }),
      document.createTextNode((g.caption || '').slice(0, 110)),
    ),
  );
  return fig;
}

function passageRow(p) {
  return el('details', { class: 'passage' },
    el('summary', {},
      iconEl('chevR'),
      el('span', { class: 'pl', text: `${p.file} · p.${p.page}` }),
      el('span', { class: 'pt', text: p.topic || '' }),
    ),
    el('div', { class: 'p-body' },
      el('div', { text: (p.text || '').slice(0, 500) }),
      el('button', { class: 'btn ghost small-p', type: 'button',
        onclick: () => showPage(p.file, p.page) }, iconEl('doc'), `Open p.${p.page}`),
    ),
  );
}

export function openSearch({ tab = 'notes', initial = '' } = {}) {
  openDrawer('search', 'Search notes');
  const body = $('drawerBody');
  body.replaceChildren();

  const input = el('input', { type: 'search', placeholder: 'Search the index…', value: initial,
    'aria-label': 'Search query' });
  const go = el('button', { class: 'btn primary', type: 'button', text: 'Search' });
  const tabs = el('div', { class: 'seg', role: 'tablist' },
    el('button', { role: 'tab', 'aria-selected': String(tab === 'notes'), text: 'Notes',
      onclick: () => setTab('notes') }),
    el('button', { role: 'tab', 'aria-selected': String(tab === 'figures'), text: 'Figures',
      onclick: () => setTab('figures') }),
  );
  const results = el('div', {});

  let curTab = tab;
  function setTab(t) {
    curTab = t;
    for (const b of tabs.querySelectorAll('[role="tab"]')) {
      b.setAttribute('aria-selected', String(b.textContent.trim() === t));
    }
    if (input.value.trim()) run();
  }

  async function run() {
    const q = input.value.trim();
    if (!q) return;
    results.replaceChildren(el('div', { class: 'retrieve-skel' },
      el('div', { class: 'skel' }), el('div', { class: 'skel' }), el('div', { class: 'skel' })));
    try {
      if (curTab === 'notes') {
        const { passages } = await api.search(q);
        results.replaceChildren();
        if (!passages.length) results.append(el('p', { class: 'small', text: 'No passages matched.' }));
        for (const p of passages.slice(0, 12)) results.append(passageRow(p));
      } else {
        const { diagrams } = await api.diagrams(q);
        results.replaceChildren();
        if (!diagrams.length) {
          results.append(el('p', { class: 'small',
            text: 'No figures matched — run `python -m jeerag.caption` first.' }));
        } else {
          const gal = el('div', { class: 'gal' });
          for (const g of diagrams.slice(0, 24)) gal.append(figureCard(g));
          results.append(gal);
        }
      }
    } catch (e) {
      results.replaceChildren(el('p', { class: 'small', text: `Search failed: ${e.message}` }));
    }
  }

  go.addEventListener('click', run);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); run(); } });

  body.append(
    el('div', { class: 'search-row' }, input, go),
    el('div', { style: 'margin-bottom:12px' }, tabs),
    results,
  );
  if (initial) run();
  input.focus();
}

/** Composer 🖼 button: search figures for the drafted text, else open search. */
export function figuresFromDraft() {
  const draft = $('q').value.trim();
  openSearch({ tab: 'figures', initial: draft });
}

/* ==========================================================================
   Page viewer — renders any indexed PDF page, with prev/next navigation
   ========================================================================== */

let pvFile = null;
let pvPage = 1;
let pvOpen = false;

function closePageview() {
  if (!pvOpen) return;
  pvOpen = false;
  $('pageview').hidden = true;
  overlayStack.remove(closePageview);
}

function pvLoad() {
  const img = $('pvImg');
  $('pvLoad').hidden = false;
  img.style.opacity = '0';
  img.onload = () => {
    $('pvLoad').hidden = true;
    img.style.opacity = '1';
    img.style.animation = 'none';
    requestAnimationFrame(() => { img.style.animation = ''; });
  };
  img.onerror = () => {
    $('pvLoad').hidden = true;
    img.style.opacity = '1';
    toast(`Page ${pvPage} of ${pvFile} is outside the rendered range.`, { type: 'err' });
  };
  $('pvCap').textContent = `${pvFile} · p.${pvPage}`;
  img.src = `/pdfpage/${encodeURIComponent(pvFile)}/${pvPage}.png`;
}

export function showPage(file, page = 1) {
  if (!file) return;
  pvFile = file;
  pvPage = page;
  if (!pvOpen) {
    pvOpen = true;
    $('pageview').hidden = false;
    overlayStack.push(closePageview);
  }
  pvLoad();
}

window.addEventListener('open-page', (e) => showPage(e.detail.file, e.detail.page));

/* ==========================================================================
   Doubt Deck — flashcard flow over the SM-2 queue
   ========================================================================== */

let deck = { queue: [], idx: 0, stats: null, revealed: false };

function closeDeck() {
  $('modal-deck').hidden = true;
  overlayStack.remove(closeDeck);
  deck = { queue: [], idx: 0, stats: null, revealed: false };
}

export async function refreshDeckBadge() {
  try {
    const { stats } = await api.cards();
    const n = stats.due_today || 0;
    const badge = $('duen');
    badge.hidden = n === 0;
    if (n > 0) {
      const prev = badge.textContent;
      badge.textContent = String(n);
      if (prev !== badge.textContent) {
        badge.classList.remove('hot');
        void badge.offsetWidth; // restart animation
        badge.classList.add('hot');
      }
    }
  } catch { /* offline — leave badge as-is */ }
}

function statChip(label, value) {
  return el('span', { class: 'pill' }, el('b', { text: String(value) }), document.createTextNode(` ${label}`));
}

export async function openDeck() {
  $('modal-deck').hidden = false;
  overlayStack.push(closeDeck);
  deck = { queue: [], idx: 0, stats: null, revealed: false };
  const area = $('deckArea');
  const chips = $('deckStats');
  area.replaceChildren(el('div', { class: 'retrieve-skel' },
    el('div', { class: 'skel' }), el('div', { class: 'skel' })));
  try {
    const { due, stats } = await api.cards();
    deck.queue = due || [];
    deck.stats = stats;
    chips.replaceChildren(
      statChip('cards', stats.total),
      statChip('due today', stats.due_today),
      statChip('mature', stats.mature),
      statChip('lapses', stats.lapses),
    );
    renderFlash();
  } catch (e) {
    area.replaceChildren(el('p', { class: 'small', text: `Couldn't load the deck: ${e.message}` }));
  }
}

function renderFlash() {
  const area = $('deckArea');
  area.replaceChildren();

  if (!deck.queue.length) {
    area.append(el('div', { class: 'deck-empty' },
      el('div', { class: 'big' }, iconEl('deck')),
      el('b', { text: 'Nothing due right now' }),
      el('p', { text: 'Save any answer with “Save to Deck” under a reply —' }),
      el('p', { text: 'cards return on an SM-2 schedule.' }),
    ));
    return;
  }

  const card = deck.queue[deck.idx];
  const progress = el('div', { class: 'deck-progress' },
    el('i', { style: `width:${Math.round((deck.idx / deck.queue.length) * 100)}%` }));

  const q = el('div', { class: 'flash-q' });
  q.innerHTML = esc(card.question);
  renderMath(q);

  const meta = el('div', { class: 'flash-meta' });
  if (card.topic) meta.append(el('span', { class: 'pill', text: card.topic }));
  if (card.file) meta.append(el('span', { class: 'pill', text: `${card.file} · p.${card.page}` }));
  meta.append(el('span', { class: 'pill', text: `reps ${card.reps}` }));

  const flash = el('div', { class: 'flash' }, q, meta);
  if (card.diagram) {
    flash.append(el('img', { class: 'thumb', src: `/${card.diagram}`, loading: 'lazy',
      alt: 'card diagram', onclick: () => showPage(card.file, card.page) }));
  }

  const reveal = () => {
    deck.revealed = true;
    const a = el('div', { class: 'flash-a' });
    renderRich(a, card.answer || '—');
    const src = el('div', { class: 'src-line' });
    if (card.file) {
      src.append(el('button', { class: 'source-chip', type: 'button',
        onclick: () => showPage(card.file, card.page) },
        iconEl('doc'), `${card.file} · p.${card.page}`));
    }
    a.append(src);
    flash.append(a);

    const grades = el('div', { class: 'grade-row' });
    for (const [score, label, sub] of [
      [0, 'Again', 'forgot'], [3, 'Hard', '+1d'], [4, 'Good', 'SM-2'], [5, 'Easy', 'boost'],
    ]) {
      grades.append(el('button', { class: `g-${score}`, type: 'button',
        onclick: () => grade(card.id, score) }, label, el('small', { text: sub })));
    }
    flash.append(grades);
    revealBtn.remove();
  };

  const revealBtn = el('button', { class: 'btn primary', type: 'button', style: 'margin:0 16px 16px',
    onclick: reveal }, iconEl('eye'), 'Show answer');

  area.append(progress, flash, revealBtn);
}

async function grade(id, score) {
  try {
    const res = await api.grade(id, score);
    toast(`Next review in ${res.interval} day${res.interval === 1 ? '' : 's'}`, { type: 'ok' });
    deck.idx += 1;
    deck.revealed = false;
    if (deck.idx >= deck.queue.length) {
      refreshDeckBadge();
      $('deckArea').replaceChildren(el('div', { class: 'deck-empty' },
        el('div', { class: 'big' }, iconEl('check')),
        el('b', { text: 'Queue cleared' }),
        el('p', { text: `${deck.queue.length} card${deck.queue.length === 1 ? '' : 's'} reviewed.` }),
      ));
      return;
    }
    renderFlash();
  } catch (e) {
    toast(`Grading failed: ${e.message}`, { type: 'err' });
  }
}

/** Called from chat answer actions. */
export async function saveToDeck(question, answer, file, page, diagram, topic) {
  try {
    await api.addCard({ question, answer, file, page, diagram, topic });
    toast('Saved to Doubt Deck — review it tomorrow.', { type: 'ok' });
    refreshDeckBadge();
  } catch (e) {
    toast(`Couldn't save the card: ${e.message}`, { type: 'err' });
  }
}

/* ==========================================================================
   Settings
   ========================================================================== */

function closeSettings() {
  $('modal-settings').hidden = true;
  overlayStack.remove(closeSettings);
}

function buildAccentSwatches() {
  const wrap = $('accents');
  wrap.replaceChildren();
  for (const a of ACCENTS) {
    wrap.append(el('button', {
      class: 'swatch', type: 'button', role: 'radio',
      'aria-checked': String(currentAccent() === a.id),
      'aria-label': `${a.label} accent`,
      title: a.label,
      style: `background:${a.swatch}`,
      onclick: () => {
        setAccent(a.id);
        for (const b of wrap.querySelectorAll('.swatch')) b.setAttribute('aria-checked', 'false');
        wrap.querySelector(`[aria-label="${a.label} accent"]`).setAttribute('aria-checked', 'true');
      },
    }));
  }
}

function syncThemeSeg() {
  const theme = currentTheme();
  for (const b of $('segTheme').querySelectorAll('[role="radio"]')) {
    b.setAttribute('aria-checked', String(b.dataset.themeSet === theme));
  }
}

export function openSettings() {
  $('modal-settings').hidden = false;
  overlayStack.push(closeSettings);
  syncThemeSeg();
  buildAccentSwatches();

  api.config().then((c) => {
    $('s_base').value = c.llm_base_url || '';
    $('s_model').value = c.llm_model || '';
    $('s_key').value = '';
    $('s_key').placeholder = c.llm_api_key_set ? '(saved — leave blank to keep)' : '(none saved)';
    $('s_topk').value = String(c.top_k || 8);
  }).catch((e) => toast(`Config unavailable: ${e.message}`, { type: 'err' }));

  $('s_base').focus();
}

function wireSettings() {
  $('modal-settings').addEventListener('click', (e) => {
    if (e.target === $('modal-settings')) closeSettings();
  });
  for (const b of $('modal-settings').querySelectorAll('[data-close]')) {
    b.addEventListener('click', closeSettings);
  }
  $('segTheme').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-theme-set]');
    if (btn) setTheme(btn.dataset.themeSet);
  });
  $('s_show').addEventListener('click', () => {
    const key = $('s_key');
    key.type = key.type === 'password' ? 'text' : 'password';
  });
  $('s_save').addEventListener('click', async () => {
    const updates = {
      llm_base_url: $('s_base').value.trim(),
      llm_model: $('s_model').value.trim(),
      top_k: Number($('s_topk').value) || 8,
    };
    const key = $('s_key').value.trim();
    if (key && !key.startsWith('(saved')) updates.llm_api_key = key;
    try {
      await api.saveConfig(updates);
      toast('Settings saved.', { type: 'ok' });
      closeSettings();
    } catch (e) {
      toast(`Save failed: ${e.message}`, { type: 'err' });
    }
  });
}

/* ==========================================================================
   Study stats (tracker summary)
   ========================================================================== */

function closeStats() {
  $('modal-stats').hidden = true;
  overlayStack.remove(closeStats);
}

export async function openStats() {
  $('modal-stats').hidden = false;
  overlayStack.push(closeStats);
  await renderStats(Number($('statsDays').value) || 7);
}

async function renderStats(days) {
  const area = $('statsArea');
  area.replaceChildren(el('div', { class: 'retrieve-skel' },
    el('div', { class: 'skel' }), el('div', { class: 'skel' }), el('div', { class: 'skel' })));
  try {
    const s = await api.tracker(days);
    area.replaceChildren();
    if (!s.queries) {
      area.append(el('div', { class: 'stats-empty' },
        el('b', { text: 'No questions logged yet' }),
        el('p', { class: 'small', text: 'Ask something in the chat — topics will show up here.' })));
      return;
    }
    const max = Math.max(1, ...s.topics.map((t) => t.count));
    area.append(
      el('div', { class: 'stats-hero' },
        el('div', { class: 'stat' }, el('b', { text: String(s.queries) }), el('span', { text: 'questions asked' })),
        el('div', { class: 'stat' }, el('b', { text: String(s.topics.length) }), el('span', { text: 'topics touched' })),
      ),
      el('h3', { text: `Top topics · last ${s.days} days`, style: 'font:700 13px var(--font);color:var(--dim);margin:0 0 12px' }),
    );
    const list = el('div', { class: 'topic-list' });
    for (const t of s.topics.slice(0, 10)) {
      const bar = el('i');
      list.append(el('div', { class: 'topic-row' },
        el('span', { class: 't-name', text: t.topic }),
        el('span', { class: 't-bar' }, bar),
        el('span', { class: 't-n', text: String(t.count) }),
      ));
      requestAnimationFrame(() => { bar.style.width = `${Math.round((t.count / max) * 100)}%`; });
    }
    area.append(list);
  } catch (e) {
    area.replaceChildren(el('p', { class: 'small', text: `Tracker unavailable: ${e.message}` }));
  }
}

/* ==========================================================================
   boot
   ========================================================================== */

export function wirePanels() {
  wireSettings();
  $('drawerClose').addEventListener('click', closeDrawer);
  $('scrim').addEventListener('click', closeDrawer);
  $('pvClose').addEventListener('click', closePageview);
  $('pvPrev').addEventListener('click', () => { if (pvPage > 1) { pvPage -= 1; pvLoad(); } });
  $('pvNext').addEventListener('click', () => { pvPage += 1; pvLoad(); });
  $('pageview').addEventListener('click', (e) => { if (e.target === $('pageview')) closePageview(); });
  $('statsDays').addEventListener('change', (e) => renderStats(Number(e.target.value)));
  window.addEventListener('open-citations', (e) => showCitations(e.detail, { bringToFront: true }));
}
