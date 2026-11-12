/* ==========================================================================
   theme.js — dark/light theme + accent colour, persisted
   ========================================================================== */

import { store } from './store.js';

export const ACCENTS = [
  { id: 'teal', label: 'Teal', swatch: '#2dd4bf' },
  { id: 'indigo', label: 'Indigo', swatch: '#818cf8' },
  { id: 'rose', label: 'Rose', swatch: '#fb7185' },
  { id: 'amber', label: 'Amber', swatch: '#fbbf24' },
];

const META = document.querySelector('meta[name="theme-color"]');
const BG = { dark: '#0b0d12', light: '#f4f6f8' };

export function currentTheme() {
  return document.documentElement.dataset.theme || 'dark';
}
export function currentAccent() {
  return document.documentElement.dataset.accent || 'teal';
}

function apply(theme, accent) {
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.dataset.accent = accent;
  META?.setAttribute('content', BG[theme] || BG.dark);
  window.dispatchEvent(new CustomEvent('themechange', { detail: { theme, accent } }));
}

export function setTheme(theme) {
  store.set('theme', theme);
  apply(theme, currentAccent());
}

export function setAccent(accent) {
  store.set('accent', accent);
  apply(currentTheme(), accent);
}

export function toggleTheme() {
  setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
}

/** Restore persisted preferences before first paint of dynamic UI. */
export function initTheme() {
  apply(store.get('theme', 'dark'), store.get('accent', 'teal'));
}
