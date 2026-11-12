/* ==========================================================================
   main.js — boot: theme, icons, header wiring, stats, palette, shortcuts
   ========================================================================== */

import { api } from './api.js';
import { el, iconEl, countUp, overlayStack, hydrateIcons } from './ui.js';
import { initTheme, toggleTheme, currentTheme } from './theme.js';
import { toast } from './toasts.js';
import {
  openSearch, openSettings, openStats, openDeck,
  figuresFromDraft, wirePanels, refreshDeckBadge,
} from './panels.js';
import { initChat, ask, newChat } from './chat.js';
import { initPalette, openPalette, closePalette, wirePalette, paletteOpen } from './palette.js';

const $ = (id) => document.getElementById(id);

/* ---------- boot order: icons → theme → chat → panels ---------- */
hydrateIcons();
initTheme();
const btnTheme = $('btnTheme');
const syncThemeBtn = () => btnTheme.replaceChildren(iconEl(currentTheme() === 'dark' ? 'sun' : 'moon'));
syncThemeBtn();
window.addEventListener('themechange', syncThemeBtn);

/* ---------- chat ---------- */
initChat();
wirePanels();

/* ---------- header actions ---------- */
$('btnPalette').addEventListener('click', openPalette);
$('btnSearch').addEventListener('click', () => openSearch());
$('btnStats').addEventListener('click', openStats);
$('btnDeck').addEventListener('click', openDeck);
$('btnSettings').addEventListener('click', openSettings);
$('btnFigures').addEventListener('click', figuresFromDraft);
btnTheme.addEventListener('click', toggleTheme);
$('statpill').addEventListener('click', openStats);

/* ---------- welcome chips → ask directly ---------- */
const chipActions = [];
for (const chip of document.querySelectorAll('.chip[data-q]')) {
  const q = chip.dataset.q;
  chipActions.push({ id: `ask-${chipActions.length}`, label: q, group: 'Ask your notes', icon: 'sparkles', run: () => ask(q) });
  chip.addEventListener('click', () => ask(q));
}

/* ---------- stats pill (animated count-up) ---------- */
async function loadStats() {
  const pill = $('statpill').querySelector('span');
  try {
    const s = await api.stats();
    pill.replaceChildren();
    for (const [n, label] of [[s.files, 'PDFs'], [s.pages, 'pages'], [s.diagrams, 'diagrams']]) {
      const b = el('b', { text: '0' });
      pill.append(b, document.createTextNode(` ${label} · `));
      countUp(b, Number(n) || 0);
    }
    pill.lastChild.textContent = pill.lastChild.textContent.replace(/ · $/, '');
  } catch {
    pill.textContent = 'offline';
    toast('Index API unreachable — is the server running?', { type: 'err' });
  }
}

/* ---------- command palette ---------- */
function buildPaletteActions() {
  return [
    { id: 'new', label: 'New chat', group: 'Chat', icon: 'plus', keywords: 'clear reset conversation', run: newChat },
    ...chipActions,
    { id: 'search', label: 'Search notes', group: 'Go to', icon: 'search', keywords: 'find bm25 passages', run: () => openSearch() },
    { id: 'figures', label: 'Find figures', group: 'Go to', icon: 'image', keywords: 'diagram caption gallery', run: () => openSearch({ tab: 'figures' }) },
    { id: 'formulas', label: 'Formula sheet', group: 'Go to', icon: 'sparkles', keywords: 'harvest relations revision topic equations', run: () => openSearch({ tab: 'formulas' }) },
    { id: 'deck', label: 'Open Doubt Deck', group: 'Go to', icon: 'deck', keywords: 'review cards spaced repetition sm2', hint: 'due', run: openDeck },
    { id: 'stats', label: 'Study stats', group: 'Go to', icon: 'chart', keywords: 'tracker topics weak', run: openStats },
    { id: 'settings', label: 'Open settings', group: 'Go to', icon: 'gear', keywords: 'llm endpoint model key config', run: openSettings },
    { id: 'theme', label: currentTheme() === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
      group: 'Theme', icon: currentTheme() === 'dark' ? 'sun' : 'moon', keywords: 'dark light mode', run: toggleTheme },
  ];
}
initPalette(buildPaletteActions());
wirePalette();
window.addEventListener('themechange', () => initPalette(buildPaletteActions()));

/* ---------- global keyboard shortcuts ---------- */
document.addEventListener('keydown', (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    paletteOpen() ? closePalette() : openPalette();
  } else if (mod && e.key === '/') {
    e.preventDefault();
    $('q').focus();
  } else if (e.key === 'Escape' && !paletteOpen()) {
    const top = overlayStack.top();
    if (top) {
      e.preventDefault();
      top();
    }
  }
});

/* ---------- boot ---------- */
loadStats();
refreshDeckBadge();
