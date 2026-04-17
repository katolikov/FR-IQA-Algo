---
title: UPIQAL Eval
emoji: "🔬"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: "FR-IQA Algorithm backend (FastAPI)"
thumbnail: "https://raw.githubusercontent.com/katolikov/FR-IQA-Algo/main/docs/examples/upiqal_output/anomaly_overlay.png"
---

# UPIQAL — FastAPI inference backend

<p align="center">
  <a href="https://github.com/katolikov/FR-IQA-Algo">
    <img alt="Source on GitHub"
         src="https://img.shields.io/badge/GitHub-katolikov%2FFR--IQA--Algo-181717?logo=github&style=for-the-badge" />
  </a>
  &nbsp;
  <a href="https://github.com/katolikov/FR-IQA-Algo/blob/main/README.md">
    <img alt="Project README"
         src="https://img.shields.io/badge/docs-README-blue?style=for-the-badge" />
  </a>
  &nbsp;
  <a href="https://github.com/katolikov/FR-IQA-Algo/blob/main/LICENSE">
    <img alt="License MIT"
         src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
  </a>
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/katolikov/FR-IQA-Algo/main/docs/examples/upiqal_output/anomaly_overlay.png"
    alt="UPIQAL anomaly overlay on the worked-example cartoon pair"
    width="720"
  />
</p>

**Unified Probabilistic Image Quality and Artifact Locator** — Full-Reference
Image Quality Assessment with spatially precise diagnostics for JPEG
blocking, Gibbs ringing, Gaussian noise, blur, and chromatic transport.

The interactive demo frontend is served at
**[upiqal.qpon](https://upiqal.qpon)** (Vercel, routes `/api/*` here).
Source code, CLI, training pipeline, and algorithm write-up live on
GitHub: **[katolikov/FR-IQA-Algo](https://github.com/katolikov/FR-IQA-Algo)**.

* The static frontend lives on Vercel; its `vercel.json` contains a
  native rewrite that forwards every `/api/*` request straight to this
  Space — no serverless proxy, no environment variables.
* Endpoints: `POST /api/compare`, `GET /healthz`, `GET /api/download/{name}`.

See the
[GitHub README](https://github.com/katolikov/FR-IQA-Algo/blob/main/README.md)
for the full algorithm description, CLI usage, and deployment guide.
