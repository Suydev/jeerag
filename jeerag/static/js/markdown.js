/* ==========================================================================
   markdown.js — untrusted text → sanitized HTML → KaTeX → UI enhancements
   (markdown, tables, code-copy, citation chips, math)
   ========================================================================== */

import { esc, iconEl } from './ui.js';

const markedAvailable = () => typeof window.marked?.parse === 'function';

export function mdToHtml(text) {
  const t = String(text || '');
  if (markedAvailable()) {
    try {
      window.marked.setOptions({ breaks: true });
      return window.marked.parse(t);
    } catch { /* fall through to plain text */ }
  }
  return `<p>${esc(t).replace(/\n/g, '<br>')}</p>`;
}

export function sanitize(html) {
  if (window.DOMPurify) {
    return window.DOMPurify.sanitize(html, { ADD_ATTR: ['target', 'rel'] });
  }
  // vendored sanitizer missing (offline CDN-less build) — degrade to text
  const d = document.createElement('div');
  d.textContent = html;
  return d.innerHTML;
}

/** Render LaTeX in a subtree, if KaTeX auto-render is loaded. Never throws. */
export function renderMath(node) {
  if (typeof window.renderMathInElement !== 'function' || !window.katex) return;
  try {
    window.renderMathInElement(node, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'option'],
    });
  } catch { /* bad math must not break the message */ }
}

/* ---- enhancement passes over sanitized HTML ---- */

const CITE_RE = /\(([^()\n]{2,64}?),\s*p\.(\d{1,4})\)/g;
const HAS_CITE = /\(([^()\n]{2,64}?),\s*p\.(\d{1,4})\)/; // non-global: safe for .test()

function enhanceCode(root) {
  for (const code of [...root.querySelectorAll('pre > code')]) {
    const pre = code.parentElement;
    if (pre.closest('.codeblock')) continue;
    const wrap = document.createElement('div');
    wrap.className = 'codeblock';
    const head = document.createElement('div');
    head.className = 'code-head';
    const lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || 'text';
    const label = document.createElement('span');
    label.textContent = lang;
    const btn = document.createElement('button');
    btn.className = 'cb-copy';
    btn.type = 'button';
    btn.append(iconEl('copy'), document.createTextNode('Copy'));
    btn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(code.textContent); } catch { /* ignore */ }
      btn.replaceChildren(iconEl('check'), document.createTextNode('Copied'));
      setTimeout(() => btn.replaceChildren(iconEl('copy'), document.createTextNode('Copy')), 1600);
    });
    head.append(label, btn);
    pre.replaceWith(wrap);
    wrap.append(head, pre);
  }
}

function enhanceTables(root) {
  for (const table of [...root.querySelectorAll('table')]) {
    if (table.parentElement.classList.contains('tablewrap')) continue;
    const wrap = document.createElement('div');
    wrap.className = 'tablewrap';
    table.replaceWith(wrap);
    wrap.append(table);
  }
}

function enhanceLinks(root) {
  for (const a of root.querySelectorAll('a[href]')) {
    if (/^https?:/i.test(a.href)) {
      a.target = '_blank';
      a.rel = 'noopener';
    }
  }
}

/** Turn "(chapter-file, p.42)" citations into buttons that open the scan. */
function enhanceCitations(root) {
  const skip = new Set(['PRE', 'CODE', 'A', 'BUTTON', 'OPTION', 'SELECT']);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const p = node.parentElement;
      if (!p || skip.has(p.tagName) || p.closest('.katex, .codeblock, a, button')) {
        return NodeFilter.FILTER_REJECT;
      }
      return HAS_CITE.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const hits = [];
  while (walker.nextNode()) hits.push(walker.currentNode);
  for (const textNode of hits) {
    const frag = document.createDocumentFragment();
    let rest = textNode.nodeValue;
    let m;
    CITE_RE.lastIndex = 0;
    let last = 0;
    while ((m = CITE_RE.exec(rest))) {
      frag.append(document.createTextNode(rest.slice(last, m.index)));
      const file = m[1].trim();
      const page = Number(m[2]);
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'cite-chip';
      chip.textContent = `(${file}, p.${page})`;
      chip.title = 'Open the scanned page';
      chip.addEventListener('click', () =>
        window.dispatchEvent(new CustomEvent('open-page', { detail: { file, page } })));
      frag.append(chip);
      last = m.index + m[0].length;
    }
    frag.append(document.createTextNode(rest.slice(last)));
    textNode.replaceWith(frag);
  }
}

export function enhance(root) {
  enhanceTables(root);
  enhanceCode(root);
  enhanceLinks(root);
  enhanceCitations(root);
}

/** Full pipeline: markdown → sanitize → enhance → KaTeX. */
export function renderRich(node, text) {
  node.innerHTML = sanitize(mdToHtml(text));
  enhance(node);
  renderMath(node);
}
