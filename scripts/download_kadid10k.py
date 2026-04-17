"""Download and unpack KADID-10k for UPIQAL self-supervised training.

KADID-10k (Konstanz Artificially Distorted Image quality Database)
provides 81 clean reference photographs and ~10 000 distorted variants
at a variety of distortion levels.  For our self-supervised SUSS
training, every image is a valid natural-image sample, so the whole
corpus (~10 200 PNGs) can be fed into ``train_uncertainty.py``.

This script:
    1. Downloads the KADID-10k zip (~2.9 GB) from a Hugging Face mirror
       (``myzhao1999/kadid10k``) via the `hf` CLI.
    2. Unzips it into ``<dest>/kadid10k``.
    3. Strips the AppleDouble ``._*`` resource-fork files that macOS
       leaves inside the archive — otherwise PIL fails to decode them.

Usage
-----
    python3 scripts/download_kadid10k.py            # unpacks to /tmp/kadid
    python3 scripts/download_kadid10k.py --dest ./data/kadid

After the script finishes, point the trainer at the ``images/`` dir:

    python3 train_uncertainty.py \\
        --data-dir <dest>/kadid10k/kadid10k/images \\
        --epochs 15 --steps-per-epoch 500 \\
        --lr 5e-4 --lr-schedule cosine --lr-min 1e-6 \\
        --out weights/L_cholesky_blockdiag.pth

That matches the command we used to produce the bundled checkpoint.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


HF_REPO_ID = "myzhao1999/kadid10k"
ZIP_FILENAME = "kadid10k.zip"


def _run(cmd: list[str]) -> None:
    print(f"[kadid] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _ensure_zip(dest: Path) -> Path:
    zip_path = dest / ZIP_FILENAME
    if zip_path.is_file() and zip_path.stat().st_size > 100_000_000:
        print(f"[kadid] zip already present at {zip_path}")
        return zip_path
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "hf", "download",
            "--repo-type", "dataset",
            HF_REPO_ID,
            "--local-dir", str(dest),
        ]
    )
    if not zip_path.is_file():
        raise SystemExit(
            f"hf download finished but {zip_path} not found; "
            "check the repo layout."
        )
    return zip_path


def _unzip_if_missing(zip_path: Path, dest: Path) -> Path:
    root = dest / "kadid10k"
    if root.is_dir() and any(root.iterdir()):
        print(f"[kadid] already unpacked at {root}")
        return root
    print(f"[kadid] unzipping {zip_path} -> {root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return root


def _strip_applesauce(root: Path) -> int:
    """Delete macOS ``._*`` resource-fork files PIL can't decode."""
    removed = 0
    for p in root.rglob("._*"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[kadid] removed {removed} AppleDouble ._* files")
    return removed


def _find_images_dir(root: Path) -> Path:
    # Known layout: <root>/kadid10k/images or <root>/images
    for candidate in (root / "kadid10k" / "images", root / "images"):
        if candidate.is_dir():
            return candidate
    # Fallback: search for a dir with >80 PNGs (sanity check against the 81 refs)
    for d in root.rglob("images"):
        if d.is_dir():
            png_count = sum(1 for _ in d.glob("*.png"))
            if png_count > 80:
                return d
    raise SystemExit(
        f"Could not locate the KADID images/ directory under {root}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("/tmp/kadid"),
        help="Where to download + unpack KADID-10k (default /tmp/kadid).",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the ~2.9 GB zip file after unpacking (default: delete).",
    )
    args = parser.parse_args()

    # Sanity-check `hf` CLI is available.
    if shutil.which("hf") is None:
        raise SystemExit(
            "The 'hf' CLI is required to download from Hugging Face.\n"
            "Install with: pip install -U 'huggingface_hub[cli]'\n"
            "Then log in (optional for public datasets): hf auth login"
        )

    dest: Path = args.dest
    zip_path = _ensure_zip(dest)
    root = _unzip_if_missing(zip_path, dest)
    _strip_applesauce(root)
    imgs = _find_images_dir(root)
    png_count = sum(1 for _ in imgs.glob("*.png"))
    print(f"[kadid] ready: {png_count} PNGs under {imgs}")

    if not args.keep_zip:
        try:
            zip_path.unlink()
            print(f"[kadid] removed zip {zip_path} (use --keep-zip to keep)")
        except OSError:
            pass

    print()
    print("Suggested training invocation:")
    print(
        "  python3 train_uncertainty.py \\\n"
        f"      --data-dir {imgs} \\\n"
        "      --epochs 15 --steps-per-epoch 500 \\\n"
        "      --lr 5e-4 --lr-schedule cosine --lr-min 1e-6 \\\n"
        "      --out weights/L_cholesky_blockdiag.pth"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
