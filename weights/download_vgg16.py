#!/usr/bin/env python3
"""Download VGG16 ImageNet-pretrained weights to the local weights/ directory.

Run once to fetch the exact checkpoint that torchvision expects:

    python weights/download_vgg16.py

The file is saved as  weights/vgg16-397923af.pth  (~528 MB).
After downloading, commit it via Git LFS so the repository stays self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent
FILENAME = "vgg16-397923af.pth"
DEST = WEIGHTS_DIR / FILENAME

URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


def main() -> None:
    if DEST.exists():
        print(f"Already exists: {DEST}")
        sys.exit(0)

    try:
        import torch.hub
    except ImportError:
        sys.exit("PyTorch is required.  Install it first:  pip install torch")

    print(f"Downloading VGG16 weights to {DEST} ...")
    torch.hub.download_url_to_file(URL, str(DEST), progress=True)
    print("Done.")


if __name__ == "__main__":
    main()
