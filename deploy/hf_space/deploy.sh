#!/usr/bin/env bash
# Deploy the UPIQAL FastAPI backend to a Hugging Face Space.
#
# What this script does:
#   1. Clone (or pull) the Space git repo into ./build/hf_space.
#   2. Copy the backend sources (upiqal/, web/, weights/, Dockerfile,
#      pyproject.toml, upiqal_cli.py, api/requirements.txt) into it.
#   3. Overwrite the Space README with deploy/hf_space/README.md
#      (which contains the YAML frontmatter HF needs).
#   4. Commit + push to the Space remote; HF builds the Docker image
#      automatically and serves it on port 7860.
#
# Prerequisites:
#   * git + curl
#   * Hugging Face auth token either:
#       - run `hf auth login` once, OR
#       - export HF_TOKEN=hf_xxx (preferred in CI)
#
# Usage:
#   ./deploy/hf_space/deploy.sh [SPACE_ID]
#   SPACE_ID defaults to "katolikov/upiqal-eval".

set -euo pipefail

SPACE_ID="${1:-katolikov/upiqal-eval}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/hf_space"
SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"

# Use HF_TOKEN if set; otherwise rely on ~/.huggingface/token from `hf auth login`.
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_USER="${HF_USER:-$(echo "$SPACE_ID" | cut -d/ -f1)}"
    PUSH_URL="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/$SPACE_ID"
else
    PUSH_URL="$SPACE_URL"
fi

mkdir -p "$REPO_ROOT/build"

# ---- 1. Clone or refresh the Space repo --------------------------------
if [[ -d "$BUILD_DIR/.git" ]]; then
    echo "[deploy] pulling existing clone of $SPACE_ID"
    git -C "$BUILD_DIR" fetch origin main
    git -C "$BUILD_DIR" reset --hard origin/main
else
    echo "[deploy] cloning $SPACE_URL -> $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    git clone "$SPACE_URL" "$BUILD_DIR"
    git -C "$BUILD_DIR" remote set-url origin "$PUSH_URL" 2>/dev/null || true
fi

# ---- 2. Copy backend sources -------------------------------------------
echo "[deploy] syncing backend sources"
# Remove stale inference code but keep .git history.
find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +

copy_clean() {
    # Copy a directory excluding junk Python caches and editor cruft.
    local src="$1" dst="$2"
    rsync -a \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '*.pyo' \
        --exclude '.DS_Store' \
        --exclude '.pytest_cache/' \
        "$src/" "$dst/"
}

copy_clean "$REPO_ROOT/upiqal"       "$BUILD_DIR/upiqal"
copy_clean "$REPO_ROOT/web"          "$BUILD_DIR/web"
copy_clean "$REPO_ROOT/weights"      "$BUILD_DIR/weights"
cp    "$REPO_ROOT/Dockerfile"        "$BUILD_DIR/Dockerfile"
cp    "$REPO_ROOT/pyproject.toml"    "$BUILD_DIR/pyproject.toml"
cp    "$REPO_ROOT/upiqal_cli.py"     "$BUILD_DIR/upiqal_cli.py"
cp    "$REPO_ROOT/api/requirements.txt" "$BUILD_DIR/requirements.txt"

# Drop legacy duplicate directories that the active code no longer reads.
rm -rf "$BUILD_DIR/web/templates" "$BUILD_DIR/web/static"

# Space README with the HF YAML frontmatter.
cp    "$REPO_ROOT/deploy/hf_space/README.md" "$BUILD_DIR/README.md"

# Strip the massive VGG16 checkpoint (downloaded at build time instead).
# Keep the trained Cholesky factor — only 1.2 MB.
rm -f "$BUILD_DIR/weights/vgg16-"*.pth

# Add a bootstrap step so the Space downloads VGG16 weights at startup
# if they aren't already present.
cat > "$BUILD_DIR/weights/download_vgg16.py" <<'PY'
"""Download VGG16 ImageNet pretrained weights if missing."""
import os
import sys
import urllib.request

DEST = os.path.join(os.path.dirname(__file__), "vgg16-397923af.pth")
URL = "https://download.pytorch.org/models/vgg16-397923af.pth"

if os.path.isfile(DEST):
    sys.exit(0)

print(f"[weights] fetching VGG16 from {URL}", flush=True)
urllib.request.urlretrieve(URL, DEST)
print(f"[weights] saved -> {DEST}", flush=True)
PY

# Patch the Space Dockerfile to run the download at build time so the
# first request doesn't time out waiting for the 528 MB fetch.
python3 - <<'PY'
from pathlib import Path
df = Path("build/hf_space/Dockerfile")
text = df.read_text()
marker = "COPY weights/ /app/weights/"
inject = (
    marker
    + "\nRUN python3 /app/weights/download_vgg16.py"
)
if "download_vgg16.py" not in text:
    df.write_text(text.replace(marker, inject))
    print("[deploy] patched Dockerfile to prefetch VGG16 weights")
PY

# ---- 3. Commit + push ---------------------------------------------------
cd "$BUILD_DIR"
git lfs install --local 2>/dev/null || true
git add -A
if git diff --cached --quiet; then
    echo "[deploy] nothing to push (Space already up to date)"
else
    git -c user.name="UPIQAL Deploy Bot" \
        -c user.email="noreply@users.noreply.huggingface.co" \
        commit -m "deploy: sync FastAPI backend from GitHub main"
    git push origin main
    echo "[deploy] pushed to $SPACE_URL"
    echo "[deploy] build progress: $SPACE_URL/logs"
fi

echo ""
echo "=== Next step: point Vercel at the HF Space ==="
echo "  vercel env add BACKEND_URL production"
echo "  # paste: https://${SPACE_ID/\//-}.hf.space"
echo "  vercel --prod"
