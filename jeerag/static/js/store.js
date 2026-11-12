/* ==========================================================================
   store.js — namespaced, crash-proof localStorage
   ========================================================================== */

const NS = 'jee.';

export const store = {
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(NS + key);
      return raw == null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(NS + key, JSON.stringify(value));
    } catch { /* quota or private mode — non-fatal */ }
  },
  del(key) {
    try { localStorage.removeItem(NS + key); } catch { /* ignore */ }
  },
};
