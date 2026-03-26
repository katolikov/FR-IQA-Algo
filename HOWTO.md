# How to Run UPIQAL

## Prerequisites

- Python 3.9 or later
- Git

## 1. Clone the repository

```bash
git clone <repository-url>
cd FR-IQA-Algo
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

Your prompt should now show `(venv)`.

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Core dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | VGG16 backbone, GPU-accelerated tensor ops |
| `numpy` | Array math and windowing functions |
| `scipy` | Cholesky decomposition, wavelet transforms |
| `scikit-image` | Multi-scale pyramid, Canny edge detector, morphological ops |
| `Pillow` | Image I/O (PNG, JPEG, TIFF …) |
| `opencv-python` | Convolutional filters, morphological dilation |
| `POT` | Sinkhorn–Knopp optimal transport (EMD approximation) |
| `colour-science` | sRGB → Oklab color space conversion |

## 4. Download the VGG16 model

The pipeline requires pre-trained VGG16 weights (~528 MB). Run the download script once:

```bash
python weights/download_vgg16.py
```

The file is saved to `weights/vgg16-397923af.pth`. If the file already exists the script exits immediately.

## 5. Run the CLI script

The CLI currently supports **PNG images only** as input.

```bash
python upiqal_cli.py --reference ref.png --target tgt.png --output-dir results/
```

| Flag | Description |
|---|---|
| `--reference` | Path to the reference (pristine) PNG image |
| `--target` | Path to the target (distorted) PNG image |
| `--output-dir` | Directory to save results (default: auto-generated with timestamp) |
| `--name` | Custom label for the auto-generated output directory name |
| `--max-side` | Maximum pixel dimension for the longer side (default: 512) |

The output directory will contain diagnostic heatmap PNGs and a `report.json` with the overall quality score and artifact breakdown.

## 6. Start the web server

Install `uvicorn` if not already included in your requirements:

```bash
pip install uvicorn
```

Start the server:

```bash
uvicorn main:app --app-dir web --reload --port 8000
```

The API will be available at `http://localhost:8000`.

| Flag | Description |
|---|---|
| `main:app` | Module `web/main.py`, FastAPI instance named `app` |
| `--app-dir web` | Sets `web/` as the working directory for the app |
| `--reload` | Auto-reloads on code changes (development only) |
| `--port 8000` | Port to listen on |

Remove `--reload` in production.

## 7. Deactivate the virtual environment

```bash
deactivate
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
Make sure the virtual environment is active (`source venv/bin/activate`) before running any script.

**`colour` package not found**
Install it explicitly: `pip install colour-science`.
