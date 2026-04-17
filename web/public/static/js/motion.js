// =============================================================================
// Motion One wrappers — Framer-Motion-style API for vanilla HTML pages.
// Loaded as ES module from CDN.  Falls back to CSS if Motion is unavailable.
// =============================================================================

const prefersReducedMotion =
  typeof window !== 'undefined' &&
  window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let _motionPromise = null;
function loadMotion() {
  if (prefersReducedMotion) return Promise.resolve(null);
  if (!_motionPromise) {
    _motionPromise = import('https://cdn.jsdelivr.net/npm/motion@10.18.0/+esm')
      .catch(() => null);
  }
  return _motionPromise;
}
// Kick off prefetch ASAP so the first animation isn't blocked on network.
loadMotion();

/**
 * Fade/slide-up elements matching `selector`, staggered.
 * Each child enters from +14px / opacity 0 to 0 / opacity 1.
 */
export async function enterOnLoad(selector, { stagger = 0.06, delay = 0 } = {}) {
  const els = Array.from(document.querySelectorAll(selector));
  if (!els.length) return;

  if (prefersReducedMotion) {
    els.forEach(el => { el.style.opacity = '1'; el.style.transform = 'none'; });
    return;
  }

  const m = await loadMotion();
  if (!m) {
    els.forEach((el, i) => {
      el.style.animation = `fade-up 420ms cubic-bezier(0.19,1,0.22,1) both`;
      el.style.animationDelay = `${delay + i * stagger}s`;
    });
    return;
  }

  m.animate(
    els,
    { opacity: [0, 1], transform: ['translateY(14px)', 'translateY(0)'] },
    { duration: 0.55, delay: m.stagger(stagger, { start: delay }), easing: [0.19, 1, 0.22, 1] }
  );
}

/** Tween an element from current state to `to` with a fluid spring. */
export async function animate(el, to, opts = {}) {
  if (prefersReducedMotion) { Object.assign(el.style, to); return; }
  const m = await loadMotion();
  if (!m) { Object.assign(el.style, to); return; }
  m.animate(el, to, { duration: 0.45, easing: [0.22, 1, 0.36, 1], ...opts });
}

/** Animate a numeric counter from 0 → value. */
export async function countTo(el, value, { duration = 0.9, decimals = 4 } = {}) {
  if (prefersReducedMotion) {
    el.textContent = Number(value).toFixed(decimals);
    return;
  }
  const m = await loadMotion();
  const start = 0;
  const end = Number(value);
  if (!m) { el.textContent = end.toFixed(decimals); return; }
  const controls = m.animate(
    (progress) => {
      const v = start + (end - start) * progress;
      el.textContent = v.toFixed(decimals);
    },
    { duration, easing: [0.22, 1, 0.36, 1] }
  );
  return controls;
}

/** Animate an SVG ring gauge `<circle>` stroke-dashoffset to reflect `pct` (0..1). */
export async function gaugeTo(circleEl, pct, { duration = 1.1 } = {}) {
  const C = 2 * Math.PI * parseFloat(circleEl.getAttribute('r'));
  circleEl.setAttribute('stroke-dasharray', String(C));
  const to = C * (1 - Math.max(0, Math.min(1, pct)));
  if (prefersReducedMotion) { circleEl.setAttribute('stroke-dashoffset', String(to)); return; }
  const m = await loadMotion();
  if (!m) { circleEl.style.transition = `stroke-dashoffset ${duration}s cubic-bezier(0.22,1,0.36,1)`;
            circleEl.setAttribute('stroke-dashoffset', String(to)); return; }
  m.animate(circleEl,
    { strokeDashoffset: [C, to] },
    { duration, easing: [0.22, 1, 0.36, 1] }
  );
}

/** Sequential bar-grow with stagger for severity rows. */
export async function growBars(selector, { stagger = 0.06 } = {}) {
  const bars = Array.from(document.querySelectorAll(selector));
  if (!bars.length) return;
  bars.forEach(b => { b.style.transformOrigin = 'left center'; });
  if (prefersReducedMotion) { bars.forEach(b => b.style.transform = 'scaleX(1)'); return; }
  const m = await loadMotion();
  if (!m) {
    bars.forEach((b, i) => {
      b.style.animation = `bar-grow 700ms cubic-bezier(0.19,1,0.22,1) both`;
      b.style.animationDelay = `${i * stagger}s`;
    });
    return;
  }
  m.animate(bars,
    { transform: ['scaleX(0)', 'scaleX(1)'] },
    { duration: 0.7, delay: m.stagger(stagger), easing: [0.19, 1, 0.22, 1] });
}

/** Soft spring in on show, slide out on hide. */
export async function showFromHidden(el) {
  el.hidden = false;
  el.style.display = '';
  if (prefersReducedMotion) { el.style.opacity = '1'; return; }
  const m = await loadMotion();
  if (!m) { el.style.animation = 'fade-up 420ms cubic-bezier(0.19,1,0.22,1) both'; return; }
  m.animate(el, { opacity: [0, 1], transform: ['translateY(10px)', 'translateY(0)'] },
            { duration: 0.45, easing: [0.22, 1, 0.36, 1] });
}

/** Cross-page transition using View Transitions API where available. */
export function pageLink(a) {
  a.addEventListener('click', (e) => {
    const href = a.getAttribute('href');
    if (!href || href.startsWith('http') || a.target === '_blank') return;
    if (typeof document.startViewTransition !== 'function') return;
    e.preventDefault();
    document.startViewTransition(() => { window.location.href = href; });
  });
}
