/* ==========================================================================
   toasts.js — non-blocking notifications with optional action button
   ========================================================================== */

import { el } from './ui.js';

let host = null;

/**
 * toast('Saved to Doubt Deck', { type:'ok', action:{label:'View', onClick} })
 * type: 'info' | 'ok' | 'err' — colours the status dot.
 */
export function toast(message, { type = 'info', action = null, duration = null } = {}) {
  host = host && host.isConnected ? host : document.getElementById('toasts');
  if (!host) return () => {};

  let closed = false;
  const dismiss = () => {
    if (closed) return;
    closed = true;
    node.classList.add('leaving');
    setTimeout(() => node.remove(), 260);
  };

  const node = el('div', { class: `toast ${type}`, role: 'status' },
    el('span', { class: 'dot' }),
    el('span', { class: 't-msg', text: message }),
    action && el('button', {
      class: 't-act', type: 'button', text: action.label,
      onclick: () => { action.onClick?.(); dismiss(); },
    }),
  );

  host.append(node);
  setTimeout(dismiss, duration ?? (action ? 6500 : 3600));
  return dismiss;
}
