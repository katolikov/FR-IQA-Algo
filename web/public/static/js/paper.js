// =============================================================================
// UPIQAL · Paper page logic
// KaTeX auto-render, scroll-spy for the sticky side-nav, entrance animation.
// =============================================================================

import { enterOnLoad } from '/static/js/motion.js';

// ---------- KaTeX auto-render ------------------------------------------------
// Dynamically inserted <script> tags run asynchronously regardless of `defer`,
// so we load katex core first and only append the auto-render bundle after it
// finishes. This guarantees `renderMathInElement` can actually find `katex`
// when it runs — without the chain, auto-render may fire before katex is
// available and leave raw `$$…$$` in the page.
(function loadKatex() {
  if (window.__katexLoaded) return;
  window.__katexLoaded = true;

  const KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist';

  const css = document.createElement('link');
  css.rel = 'stylesheet';
  css.href = `${KATEX}/katex.min.css`;
  document.head.appendChild(css);

  const render = () => {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true  },
        { left: '\\[', right: '\\]', display: true  },
        { left: '\\(', right: '\\)', display: false },
        { left: '$',  right: '$',   display: false },
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
      throwOnError: false,
      strict: 'ignore',
    });
  };

  const loadScript = (src) => new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = false;   // preserve execution order
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });

  loadScript(`${KATEX}/katex.min.js`)
    .then(() => loadScript(`${KATEX}/contrib/auto-render.min.js`))
    .then(() => {
      // If the DOM is still parsing (unlikely for a module script), wait for it.
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', render, { once: true });
      } else {
        render();
      }
    })
    .catch((err) => console.error('KaTeX failed to load:', err));
})();

// ---------- Scroll-spy for side-nav ------------------------------------------
(function sideNavSpy() {
  const links = document.querySelectorAll('.paper-sidenav a[href^="#"]');
  if (!links.length) return;
  const map = new Map();
  links.forEach(a => {
    const t = document.querySelector(a.getAttribute('href'));
    if (t) map.set(t, a);
  });
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      const link = map.get(e.target);
      if (!link) return;
      if (e.isIntersecting) {
        links.forEach(a => a.classList.remove('is-active'));
        link.classList.add('is-active');
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px', threshold: 0 });
  map.forEach((_, target) => io.observe(target));
})();

// ---------- Theme toggle ------------------------------------------------------
(function initTheme() {
  const saved = localStorage.getItem('upiqal-theme');
  if (saved === 'light' || saved === 'dark') {
    document.documentElement.setAttribute('data-theme', saved);
  }
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const next = cur === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('upiqal-theme', next);
  });
})();

// ---------- Inject SVG sprite inline -----------------------------------------
fetch('/static/icons.svg').then(r => r.text()).then(t => {
  const host = document.getElementById('icon-sprite-host');
  if (host) host.innerHTML = t;
});

// ---------- Entrance anim ----------------------------------------------------
enterOnLoad('[data-enter]', { stagger: 0.08 });
