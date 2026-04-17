"""Vercel serverless proxy — forwards /api/* calls to a CPU-hosted UPIQAL backend.

PyTorch + VGG16 weights exceed Vercel Python's ~250MB unzipped size limit, so
inference runs on a separate CPU host (Fly.io / Render / Railway). Vercel hosts
the static frontend and this thin proxy forwards requests to ``BACKEND_URL``.

Configure via the ``BACKEND_URL`` environment variable in the Vercel dashboard.
Example: ``BACKEND_URL=https://upiqal.fly.dev``.

The proxy streams ``multipart/form-data`` through without buffering the whole
upload in memory, and preserves status codes and response headers.
"""

from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler

import httpx

BACKEND_URL = (os.environ.get("BACKEND_URL", "") or "").rstrip("/")

# Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1).
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length", "host",
}


def _forward(handler: BaseHTTPRequestHandler, method: str) -> None:
    if not BACKEND_URL:
        handler.send_response(503)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(
            b'{"error":"BACKEND_URL is not configured on this Vercel deployment."}'
        )
        return

    path = handler.path or "/"
    target = f"{BACKEND_URL}{path}"

    # Capture body (bounded — Vercel functions cap at ~4.5MB request; that
    # matches the size of typical image uploads handled by UPIQAL).
    length = int(handler.headers.get("content-length", "0") or "0")
    body = handler.rfile.read(length) if length > 0 else b""

    # Forward meaningful headers.
    fwd_headers = {
        k: v for k, v in handler.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    try:
        with httpx.Client(timeout=60.0, follow_redirects=False) as client:
            resp = client.request(
                method, target, headers=fwd_headers, content=body,
            )
    except httpx.HTTPError as exc:
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        msg = f'{{"error":"upstream request failed","detail":"{type(exc).__name__}"}}'
        handler.wfile.write(msg.encode("utf-8"))
        return

    handler.send_response(resp.status_code)
    for k, v in resp.headers.items():
        if k.lower() in _HOP_BY_HOP:
            continue
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(resp.content)


class handler(BaseHTTPRequestHandler):
    """Vercel Python runtime entry point.

    Vercel looks for a module-level ``handler`` class extending
    ``BaseHTTPRequestHandler``; each request invokes ``do_<METHOD>``.
    """

    # Silence noisy default logging in Vercel build logs.
    def log_message(self, format: str, *args) -> None:  # noqa: D401, A003
        return

    def do_GET(self):     _forward(self, "GET")
    def do_POST(self):    _forward(self, "POST")
    def do_PUT(self):     _forward(self, "PUT")
    def do_DELETE(self):  _forward(self, "DELETE")
    def do_PATCH(self):   _forward(self, "PATCH")
    def do_OPTIONS(self): _forward(self, "OPTIONS")
