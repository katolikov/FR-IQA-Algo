// =============================================================================
// UPIQAL · Analyzer app logic
// Drop-zone handling, API call, result rendering, interactive viewer.
// =============================================================================

import { enterOnLoad, animate, gaugeTo, growBars, countTo, showFromHidden } from '/static/js/motion.js';

// Resolve the backend base URL. On Vercel this is '' (same origin → proxy).
// Local dev hits FastAPI directly on same origin. A <meta name="backend-url">
// can override for edge cases.
const BACKEND_BASE = (document.querySelector('meta[name="backend-url"]')?.content || '').replace(/\/$/, '');

const api = (p) => BACKEND_BASE + p;

// -----------------------------------------------------------------------------
// Theme toggle (persisted)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Interactive canvas viewer (zoom / pan / opacity blend)
// -----------------------------------------------------------------------------
class ImageViewer {
  constructor(canvas, container) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.container = container;
    this.baseImg = null;
    this.overlayImg = null;
    this.opacity = 1.0;
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this._dpr = window.devicePixelRatio || 1;
    this.minScale = 0.5;
    this.maxScale = 20;
    this.isDragging = false;
    this._zoomBadgeTimer = null;
    this._interacted = false;

    canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
    canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
    canvas.addEventListener('mouseup', () => this._onMouseUp());
    canvas.addEventListener('mouseleave', () => this._onMouseUp());
    canvas.addEventListener('dblclick', () => this._onDblClick());
    canvas.addEventListener('touchstart', (e) => this._onTouchStart(e), { passive: false });
    canvas.addEventListener('touchmove', (e) => this._onTouchMove(e), { passive: false });
    canvas.addEventListener('touchend', () => this._onTouchEnd());

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(container);
    this._resize();
  }

  setImages(baseSrc, overlaySrc) {
    let loaded = 0;
    const onLoad = () => { if (++loaded === 2) { this.fitToView(); this.render(); } };
    this.baseImg = new Image();    this.baseImg.onload = onLoad;    this.baseImg.src = baseSrc;
    this.overlayImg = new Image(); this.overlayImg.onload = onLoad; this.overlayImg.src = overlaySrc;
  }

  setOpacity(v) { this.opacity = v; this.render(); }

  fitToView() {
    if (!this.baseImg) return;
    const rect = this.container.getBoundingClientRect();
    const iw = this.baseImg.naturalWidth, ih = this.baseImg.naturalHeight;
    if (!iw || !ih) return;
    const pad = 16;
    this.scale = Math.min((rect.width - pad * 2) / iw, (rect.height - pad * 2) / ih);
    this.offsetX = (rect.width - iw * this.scale) / 2;
    this.offsetY = (rect.height - ih * this.scale) / 2;
  }

  render() {
    const ctx = this.ctx, dpr = this._dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = this._cssW || this.container.getBoundingClientRect().width;
    const ch = this._cssH || this.container.getBoundingClientRect().height;
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    ctx.fillStyle = isLight ? '#f1f3f7' : '#08090c';
    ctx.fillRect(0, 0, cw, ch);
    if (!this.baseImg) return;
    const iw = this.baseImg.naturalWidth, ih = this.baseImg.naturalHeight;
    ctx.imageSmoothingEnabled = this.scale < 4;
    ctx.globalAlpha = 1.0;
    ctx.drawImage(this.baseImg, this.offsetX, this.offsetY, iw * this.scale, ih * this.scale);
    if (this.overlayImg) {
      ctx.globalAlpha = this.opacity;
      ctx.drawImage(this.overlayImg, this.offsetX, this.offsetY, iw * this.scale, ih * this.scale);
      ctx.globalAlpha = 1.0;
    }
  }

  _resize() {
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this._dpr = dpr;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this._cssW = rect.width; this._cssH = rect.height;
    if (this.baseImg && this.baseImg.naturalWidth) this.fitToView();
    this.render();
  }

  _onWheel(e) {
    e.preventDefault();
    this._dismissHint();
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = Math.pow(2, -e.deltaY * 0.003);
    const newScale = Math.min(Math.max(this.scale * factor, this.minScale), this.maxScale);
    this.offsetX = mx - (mx - this.offsetX) * (newScale / this.scale);
    this.offsetY = my - (my - this.offsetY) * (newScale / this.scale);
    this.scale = newScale;
    this.render();
    this._showZoomBadge();
  }

  _onMouseDown(e) {
    if (e.button !== 0) return;
    this._dismissHint();
    this.isDragging = true;
    this._dsx = e.clientX; this._dsy = e.clientY;
    this._dox = this.offsetX; this._doy = this.offsetY;
    this.canvas.classList.add('is-grabbing');
  }
  _onMouseMove(e) {
    if (!this.isDragging) return;
    this.offsetX = this._dox + (e.clientX - this._dsx);
    this.offsetY = this._doy + (e.clientY - this._dsy);
    this.render();
  }
  _onMouseUp() {
    if (!this.isDragging) return;
    this.isDragging = false;
    this.canvas.classList.remove('is-grabbing');
  }
  _onDblClick() { this.fitToView(); this.render(); this._showZoomBadge(); }

  _onTouchStart(e) {
    e.preventDefault(); this._dismissHint();
    if (e.touches.length === 1) {
      this.isDragging = true;
      this._dsx = e.touches[0].clientX; this._dsy = e.touches[0].clientY;
      this._dox = this.offsetX; this._doy = this.offsetY;
    } else if (e.touches.length === 2) {
      this._pd = this._tdist(e.touches); this._ps = this.scale;
      this._pm = this._tmid(e.touches); this._pox = this.offsetX; this._poy = this.offsetY;
    }
  }
  _onTouchMove(e) {
    e.preventDefault();
    if (e.touches.length === 1 && this.isDragging) {
      this.offsetX = this._dox + (e.touches[0].clientX - this._dsx);
      this.offsetY = this._doy + (e.touches[0].clientY - this._dsy);
      this.render();
    } else if (e.touches.length === 2) {
      const d = this._tdist(e.touches);
      const r = d / this._pd;
      const ns = Math.min(Math.max(this._ps * r, this.minScale), this.maxScale);
      const rect = this.canvas.getBoundingClientRect();
      const mx = this._pm.x - rect.left, my = this._pm.y - rect.top;
      this.offsetX = mx - (mx - this._pox) * (ns / this._ps);
      this.offsetY = my - (my - this._poy) * (ns / this._ps);
      this.scale = ns; this.render(); this._showZoomBadge();
    }
  }
  _onTouchEnd() { this.isDragging = false; }
  _tdist(t) { const dx = t[0].clientX - t[1].clientX, dy = t[0].clientY - t[1].clientY; return Math.hypot(dx, dy); }
  _tmid(t) { return { x: (t[0].clientX + t[1].clientX) / 2, y: (t[0].clientY + t[1].clientY) / 2 }; }

  _showZoomBadge() {
    const b = document.getElementById('zoom-badge');
    if (!b) return;
    b.textContent = this.scale.toFixed(1) + '×';
    b.classList.add('is-visible');
    clearTimeout(this._zoomBadgeTimer);
    this._zoomBadgeTimer = setTimeout(() => b.classList.remove('is-visible'), 1500);
  }
  _dismissHint() {
    if (this._interacted) return;
    this._interacted = true;
    const h = document.getElementById('viewer-hint');
    if (h) h.style.display = 'none';
  }
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
let refFile = null;
let tgtFile = null;
let lastData = null;
let activeHeatmap = null;

const $ = (id) => document.getElementById(id);

const btnRun = $('btn-run');
const btnLabel = $('btn-label');
const btnSpinner = $('btn-spinner');
const resultsEl = $('results');
const errorEl = $('error-banner');

const viewer = new ImageViewer($('viewer-canvas'), $('viewer-frame'));

// -----------------------------------------------------------------------------
// Drop zones — with image-reattach bug fix
// -----------------------------------------------------------------------------
const RAW_EXTS = ['.raw', '.bin', '.nv21', '.npy'];

function isRaw(file) {
  if (!file) return false;
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  return RAW_EXTS.includes(ext);
}

function setupDropZone(role) {
  const zone    = $(`drop-${role}`);
  const input   = $(`input-${role}`);
  const preview = $(`preview-${role}`);
  const empty   = $(`empty-${role}`);
  const remove  = $(`remove-${role}`);
  const previewImg = preview.querySelector('img');

  // Drag/drop
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('is-dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('is-dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('is-dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  // BUG FIX (Task 3): Reset input.value on every 'change' so selecting the
  // SAME file twice in a row still fires the event. Without this, the user
  // can't reattach after running an analysis.
  input.addEventListener('change', () => {
    if (input.files.length) handleFile(input.files[0]);
    input.value = '';
  });

  // × remove button
  remove.addEventListener('click', (e) => {
    e.stopPropagation();
    e.preventDefault();
    setFile(null);
  });

  function handleFile(file) {
    if (!isRaw(file) && !file.type.startsWith('image/')) return;
    setFile(file);
  }

  function setFile(file) {
    if (role === 'ref') refFile = file; else tgtFile = file;

    if (!file) {
      preview.classList.add('is-hidden');
      empty.classList.remove('is-hidden');
      remove.classList.add('is-hidden');
      previewImg.src = '';
      input.value = '';
    } else if (isRaw(file)) {
      preview.classList.add('is-hidden');
      empty.classList.remove('is-hidden');
      remove.classList.remove('is-hidden');
      // Show the filename in the empty slot so the user knows something IS selected.
      empty.querySelector('[data-slot="filename"]').textContent = `Raw: ${file.name}`;
    } else {
      const reader = new FileReader();
      reader.onload = () => {
        previewImg.src = reader.result;
        preview.classList.remove('is-hidden');
        empty.classList.add('is-hidden');
        remove.classList.remove('is-hidden');
      };
      reader.readAsDataURL(file);
    }
    updateRunButton();
    updateRawOptions();
  }
}

setupDropZone('ref');
setupDropZone('tgt');

function updateRunButton() { btnRun.disabled = !(refFile && tgtFile); }

function updateRawOptions() {
  const panel = $('raw-options');
  if (isRaw(refFile) || isRaw(tgtFile)) panel.classList.remove('is-hidden');
  else panel.classList.add('is-hidden');
}

// -----------------------------------------------------------------------------
// Run analysis
// -----------------------------------------------------------------------------
btnRun.addEventListener('click', async () => {
  if (!refFile || !tgtFile) return;
  errorEl.classList.add('is-hidden');
  resultsEl.classList.add('is-hidden');

  btnRun.disabled = true;
  btnSpinner.classList.remove('is-hidden');
  btnLabel.textContent = 'Analyzing…';

  try {
    const form = new FormData();
    form.append('reference_image', refFile);
    form.append('target_image', tgtFile);
    form.append('width',  $('raw-width').value  || '0');
    form.append('height', $('raw-height').value || '0');
    form.append('pixel_format', $('raw-pixel-format').value);
    form.append('output_format', $('output-format').value);

    const res = await fetch(api('/api/compare'), { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);
    const data = await res.json();
    lastData = data;
    renderResults(data);
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove('is-hidden');
  } finally {
    btnSpinner.classList.add('is-hidden');
    btnLabel.textContent = 'Run UPIQAL Analysis';
    btnRun.disabled = false;
    // BUG FIX (Task 3 continued): clear both file inputs so a fresh analysis
    // with the same image(s) works without a page reload.
    $('input-ref').value = '';
    $('input-tgt').value = '';
  }
});

// -----------------------------------------------------------------------------
// Heatmap metadata
// -----------------------------------------------------------------------------
const HM_META = [
  { key: 'overlay',   label: 'Anomaly Overlay',       isOverlay: true },
  { key: 'anomaly',   label: 'Global Anomaly' },
  { key: 'color',     label: 'Color Degradation' },
  { key: 'structure', label: 'Structural Similarity' },
  { key: 'blocking',  label: 'JPEG Blocking' },
  { key: 'ringing',   label: 'Gibbs Ringing' },
];

const SEVERITY_LABELS = {
  blocking: 'JPEG Blocking',
  ringing: 'Gibbs Ringing',
  noise: 'Noise',
  color_shift: 'Color Shift',
  blur: 'Blur',
};

// -----------------------------------------------------------------------------
// Download helpers
// -----------------------------------------------------------------------------
function downloadBase64Image(b64, name) {
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + b64;
  a.download = name;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
function downloadBinaryMask(name, fmt) {
  const a = document.createElement('a');
  a.href = api('/api/download/' + name + '?fmt=' + fmt);
  a.download = `upiqal-${name}.${fmt}`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
function getOutputFormat() { return $('output-format').value; }

// -----------------------------------------------------------------------------
// Render results
// -----------------------------------------------------------------------------
function renderResults(data) {
  const score = data.score;
  const diag = data.diagnostics;

  // Score ring
  const ring = document.querySelector('.score-ring__bar');
  gaugeTo(ring, score);
  countTo($('score-value'), score, { duration: 1.0, decimals: 4 });

  let label;
  if (score >= 0.9) label = 'Excellent quality';
  else if (score >= 0.7) label = 'Good quality';
  else if (score >= 0.5) label = 'Moderate degradation';
  else if (score >= 0.3) label = 'Poor quality';
  else label = 'Severe degradation';
  $('score-label').textContent = label;

  // Findings
  $('dominant-artifact').textContent = diag.dominant_artifact;
  $('affected-area').innerHTML = `${diag.affected_area}<sup>%</sup>`;

  // Severity
  const sev = $('severity-rows');
  sev.innerHTML = '';
  Object.entries(diag.severity_scores).forEach(([key, val]) => {
    const row = document.createElement('div');
    row.className = 'severity__row';
    row.innerHTML = `
      <div class="severity__head">
        <span class="severity__name">${SEVERITY_LABELS[key] || key}</span>
        <span class="severity__num">${val.toFixed(1)}%</span>
      </div>
      <div class="severity__track">
        <div class="severity__fill" style="width: ${Math.max(3, val)}%"></div>
      </div>`;
    sev.appendChild(row);
  });
  growBars('.severity__fill');

  // Side-by-side
  $('result-ref').src = $('preview-ref').querySelector('img').src || '';
  $('result-tgt').src = 'data:image/png;base64,' + data.target;

  // Gallery
  const gallery = $('hm-gallery');
  gallery.innerHTML = '';
  HM_META.forEach((hm, i) => {
    const src = hm.isOverlay
      ? 'data:image/png;base64,' + data.overlay
      : 'data:image/png;base64,' + data.heatmaps[hm.key];
    const t = document.createElement('div');
    t.className = 'thumb' + (i === 0 ? ' is-active' : '');
    t.dataset.key = hm.key;
    t.innerHTML = `
      <img src="${src}" alt="${hm.label}" loading="lazy" />
      <button class="thumb__dl icon-btn" title="Download ${hm.label}" aria-label="Download ${hm.label}">
        <svg><use href="/static/icons.svg#icon-download"></use></svg>
      </button>
      <p class="thumb__label">${hm.label}</p>`;
    t.addEventListener('click', (e) => {
      if (e.target.closest('.thumb__dl')) return;
      selectHeatmap(hm.key, hm.label, t);
    });
    t.querySelector('.thumb__dl').addEventListener('click', (e) => {
      e.stopPropagation();
      const fmt = getOutputFormat();
      if (fmt === 'png' || hm.isOverlay) {
        const b64 = hm.isOverlay ? data.overlay : data.heatmaps[hm.key];
        downloadBase64Image(b64, `upiqal-${hm.key}.png`);
      } else {
        downloadBinaryMask(hm.key, fmt);
      }
    });
    gallery.appendChild(t);
  });

  selectHeatmap('overlay', 'Anomaly Overlay', gallery.querySelector('.thumb'));
  const slider = $('opacity-slider');
  slider.value = 100; $('opacity-val').textContent = '100%'; slider.style.setProperty('--val', '100%');

  const hint = $('viewer-hint');
  if (hint) { hint.style.display = ''; viewer._interacted = false; }

  showFromHidden(resultsEl);
  resultsEl.classList.remove('is-hidden');
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function selectHeatmap(key, label, thumbEl) {
  if (!lastData) return;
  document.querySelectorAll('.thumb').forEach(el => el.classList.remove('is-active'));
  if (thumbEl) thumbEl.classList.add('is-active');
  activeHeatmap = key;
  $('viewer-title').textContent = label;
  const baseSrc = 'data:image/png;base64,' + lastData.target;
  const overlaySrc = key === 'overlay'
    ? 'data:image/png;base64,' + lastData.overlay
    : 'data:image/png;base64,' + lastData.heatmaps[key];
  viewer.setImages(baseSrc, overlaySrc);
  viewer.setOpacity($('opacity-slider').value / 100);
}

// Opacity
$('opacity-slider').addEventListener('input', function () {
  const v = this.value;
  viewer.setOpacity(v / 100);
  $('opacity-val').textContent = v + '%';
  this.style.setProperty('--val', v + '%');
});

// Save / export
$('btn-save-main').addEventListener('click', () => {
  if (!lastData || !activeHeatmap) return;
  const fmt = getOutputFormat();
  if (fmt === 'png' || activeHeatmap === 'overlay') {
    const b64 = activeHeatmap === 'overlay' ? lastData.overlay : lastData.heatmaps[activeHeatmap];
    downloadBase64Image(b64, `upiqal-${activeHeatmap}.png`);
  } else {
    downloadBinaryMask(activeHeatmap, fmt);
  }
});

$('btn-export').addEventListener('click', () => {
  if (!lastData) return;
  const report = {
    score: lastData.score,
    diagnostics: lastData.diagnostics,
    timestamp: new Date().toISOString(),
    tool: 'UPIQAL FR-IQA v0.3',
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `upiqal-report-${Date.now()}.json`;
  a.click(); URL.revokeObjectURL(url);
});

// -----------------------------------------------------------------------------
// Entrance animations
// -----------------------------------------------------------------------------
enterOnLoad('[data-enter]', { stagger: 0.06 });
