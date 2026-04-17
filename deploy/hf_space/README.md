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
  <img
    src="https://raw.githubusercontent.com/katolikov/FR-IQA-Algo/main/docs/examples/upiqal_output/anomaly_overlay.png"
    alt="UPIQAL anomaly overlay on the worked-example cartoon pair"
    width="720"
  />
</p>

**Unified Probabilistic Image Quality and Artifact Locator** — Full-Reference
Image Quality Assessment with spatially precise diagnostics for JPEG
blocking, Gibbs ringing, Gaussian noise, blur, and chromatic transport.

This Space serves the backend API for
[katolikov/FR-IQA-Algo](https://github.com/katolikov/FR-IQA-Algo).

* The static frontend lives on Vercel; it forwards every `/api/*`
  request to this backend.
* Point Vercel's `BACKEND_URL` environment variable at the public Space
  URL (`https://katolikov-upiqal-eval.hf.space`) and redeploy.
* Endpoints: `POST /api/compare`, `GET /healthz`, `GET /api/download/{name}`.

See the
[GitHub README](https://github.com/katolikov/FR-IQA-Algo/blob/main/README.md)
for the full algorithm description, CLI usage, and deployment guide.
