---
title: UPIQAL Eval
emoji: "🖼️"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: "FR-IQA Algorithm backend (FastAPI)"
---

# UPIQAL — FastAPI inference backend (Hugging Face Space)

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
