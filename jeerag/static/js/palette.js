/* ==========================================================================
   palette.js — Ctrl/⌘K command palette: grouped actions, filter, keynav
   ========================================================================== */

import { el, esc, iconEl, overlayStack } from './ui.js';

const dom = {
  root: () => document.getElementById('palette'),
  input: () => document.getElementById('palQ'),
  list: () => document.getElementById('palList'),
};

let actions = [];
let open = false;
let visible = [];
let selected = 0;
let closeFn = null;

export function initPalette(list) {
  actions = list;
}

/** Register extra actions later (e.g. suggested questions). */
export function addPaletteActions(list) {
  actions = [...actions, ...list];
}

export function paletteOpen() { return open; }

export function openPalette() {
  if (open) return;
  open = true;
  closeFn = closePalette;
  dom.root().hidden = false;
  dom.input().value = '';
  render('');
  dom.input().focus();
  overlayStack.push(closeFn);
}

export function closePalette() {
  if (!open) return;
  open = false;
  dom.root().hidden = true;
  overlayStack.remove(closeFn);
  document.getElementById('q')?.focus();
}

function scoreFor(action, q) {
  if (!q) return 1;
  const hay = `${action.label} ${action.keywords || ''}`.toLowerCase();
  const idx = hay.indexOf(q);
  if (idx < 0) return -1;
  return (idx === 0 ? 2 : 1) + (action.label.toLowerCase().startsWith(q) ? 1 : 0);
}

function render(query) {
  const q = query.trim().toLowerCase();
  const matched = actions
    .map((a) => ({ a, s: scoreFor(a, q) }))
    .filter(({ s }) => s >= 0)
    .sort((x, y) => y.s - x.s)
    .map(({ a }) => a);

  visible = matched;
  selected = 0;
  const list = dom.list();
  list.replaceChildren();

  if (!matched.length) {
    list.append(el('div', { class: 'pal-empty', text: `No commands match “${query}”` }));
    return;
  }

  let lastGroup = null;
  matched.forEach((a, i) => {
    if (a.group !== lastGroup) {
      lastGroup = a.group;
      list.append(el('div', { class: 'pal-group', text: a.group }));
    }
    const item = el('button', {
      class: 'pal-item', type: 'button', role: 'option',
      'aria-selected': String(i === selected),
      onclick: () => run(i),
    });
    item.append(iconEl(a.icon));
    item.append(el('span', { html: highlight(a.label, q) }));
    if (a.hint) item.append(el('span', { class: 'hint-side', text: a.hint }));
    item.addEventListener('mousemove', () => select(i));
    list.append(item);
  });
}

function highlight(label, q) {
  const safe = esc(label);
  if (!q) return safe;
  const i = label.toLowerCase().indexOf(q);
  if (i < 0) return safe;
  return (
    esc(label.slice(0, i)) + '<mark>' + esc(label.slice(i, i + q.length)) + '</mark>' +
    esc(label.slice(i + q.length))
  );
}

function select(i) {
  selected = Math.max(0, Math.min(visible.length - 1, i));
  for (const [idx, node] of [...dom.list().querySelectorAll('.pal-item')].entries()) {
    node.setAttribute('aria-selected', String(idx === selected));
  }
  dom.list().querySelector('.pal-item[aria-selected="true"]')
    ?.scrollIntoView({ block: 'nearest' });
}

function run(i) {
  const action = visible[i];
  if (!action) return;
  closePalette();
  action.run();
}

export function wirePalette() {
  const root = dom.root();
  root.addEventListener('mousedown', (e) => { if (e.target === root) closePalette(); });
  dom.input().addEventListener('input', (e) => render(e.target.value));
  dom.input().addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); select(selected + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); select(selected - 1); }
    else if (e.key === 'Enter') { e.preventDefault(); run(selected); }
    else if (e.key === 'Escape') { e.stopPropagation(); closePalette(); }
  });
}
